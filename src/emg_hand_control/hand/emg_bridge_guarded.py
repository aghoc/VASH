"""근전도(grip_rt.py UDP) → ORCA 손. 팀이 실물로 검증한 0 전류 기동·한계 검사·종료 해제 절차(neutral_guarded.main)를 그대로 사용.

orca_core 폴더에서 (팀 README 와 같은 방식):
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/emg_bridge_guarded.py --demo      근전도 없이 자세 순환
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/emg_bridge_guarded.py             근전도 UDP 127.0.0.1:9000 수신
  uv run --frozen --no-dev python ../hand-config/right-no-wrist/emg_bridge_guarded.py --poses     자세 하나씩 천천히 (조정용)
  ... --poses --only tripod --hold 5                                                               자세 하나만 오래 (조정용)
  ... --sweep thumb_cmc thumb_abd --hold 2                                                         관절 하나씩 비율 0→1 훑기 (조정용)

제어 50 Hz: 휴식 → open, 동작 중 미확정 → preshape, 확정 → open + level×(쥐기−open), 수신 0.5 s 끊김 → open
궤적: Smoother (3차 임계감쇠, 관절 동기화, 손가락 시차) — 굽힘 ω 10 (≈0.6 s), 펴기 ω 18 (≈0.35 s), 속도 상한 250/450 °/s
  --raw 로 이전 방식(일정 속도) 비교. 2 s 마다 온도·하드웨어 오류 검사
"""

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time

import yaml

from neutral_guarded import main as run_guarded

HERE = Path(__file__).resolve().parent


def build_poses(hand, path):
    """비율(0~1) 자세 → 관절 각도(°). 약지·소지 mcp·pip 는 중지에서 예측."""
    P = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    taught_path = Path(path).with_name("taught_poses.yaml")      # teach_pose.py 로 손으로 만든 자세가 있으면 우선
    if taught_path.exists():
        T = yaml.safe_load(taught_path.read_text(encoding="utf-8")) or {}
        only = P.get("taught_only") or {}                         # 이 자세들은 손으로 만든 값 중 지정 관절만 사용 (나머지 = 대기 자세 기준)
        for name, q in T.items():
            if name in P and isinstance(q, dict):
                if name in only:
                    P[name] = {**{k: v for k, v in P[name].items() if k.startswith("_")}, **{j: q[j] for j in only[name] if j in q}}
                    print(f"손으로 만든 자세 사용: {name} ({', '.join(only[name])} 만)", flush=True)
                else:                                                  # taught_keep_copy 에 있는 자세는 yaml 의 _copy 를 남김 (10-31: 손 펴기 엄지만 따로 가르치기)
                    keep = name in (P.get("taught_keep_copy") or []) and "_copy" in P[name]
                    P[name] = {**({"_copy": P[name]["_copy"]} if keep else {}), **dict(q)}
                    print(f"손으로 만든 자세 사용: {name}", flush=True)
    for name, ov in (P.get("override") or {}).items():          # 덮어쓰기 (손으로 만든 자세 위에)
        if name not in P: continue
        ov = dict(ov); drop = ov.pop("_coupled", [])
        P[name] = {k: v for k, v in P[name].items() if k not in drop}; P[name].update(ov)
        print(f"덮어쓰기 {name}: {ov}" + (f", 중지에서 예측: {drop}" if drop else ""), flush=True)
    roms = hand.config.joint_roms_dict
    joints = list(hand.config.joint_ids)
    cp = P.get("coupling", {})
    open_f = P["open"]

    def frac_pose(name):
        f = dict(open_f)
        if P[name].get("_base") in P:                 # 예: 손가락 하나 굽히기 = 대기 자세(relax) 기준, 그 손가락만 바꿈
            f.update({k: v for k, v in P[P[name]["_base"]].items() if not k.startswith("_")})
        f.update({k: v for k, v in P[name].items() if not k.startswith("_")})
        for src, js in (P[name].get("_copy") or {}).items():      # 다른 자세의 관절 값 가져오기 (예: 감싸 쥐기 엄지 최종 = 주먹 엄지 위치)
            if src in P:
                f.update({j: P[src][j] for j in js if j in P[src]})
        tuck = cp.get("tuck", {}).get(name, {})
        for finger in ("ring", "pinky"):
            g = cp.get(finger, {}).get("gain", 1.0)
            for jt in ("mcp", "pip"):
                k = f"{finger}_{jt}"
                if k in P[name]:
                    continue
                ref = P[P[name]["_base"]] if P[name].get("_base") in P else open_f    # 대기 기준 자세면 대기 자세에서 예측
                mid, mid0 = f[f"middle_{jt}"], ref.get(f"middle_{jt}", open_f[f"middle_{jt}"])
                base = ref.get(k, open_f.get(k, mid0))
                f[k] = min(1.0, max(0.0, base + g * (mid - mid0) + tuck.get(finger, 0.0)))
        return {j: roms[j][0] + f[j] * (roms[j][1] - roms[j][0]) for j in joints if j in f}

    names = [k for k, v in P.items() if isinstance(v, dict) and any(j in v for j in joints)]
    poses = {k: frac_pose(k) for k in names}
    poses["_roms"] = roms
    if "relax" not in poses:
        poses["relax"] = dict(poses["open"])
    poses["_follow"] = dict(P.get("thumb_follow", {}) or {})
    poses["_paths"] = {k: {g: sorted((float(a), float(b)) for a, b in pts) for g, pts in (v or {}).items()}
                       for k, v in (P.get("paths") or {}).items()}
    poses["_via"] = {k: sorted((float(c), str(n)) for c, n in v) for k, v in (P.get("via") or {}).items()}   # 중간 자세: 대기 → 중간(c) → 자세
    poses["_mode_map"] = dict(P.get("mode_map") or {})        # 모드별 동작 → 자세 (예: 쥐기 모드의 '주먹' = 원통 쥐기)
    poses["_arcs"] = {k: {j: tuple(float(x) for x in a) for j, a in (v or {}).items()} for k, v in (P.get("arcs") or {}).items()}
    missing = [j for j in joints if j not in poses["open"]]
    if missing:
        raise RuntimeError(f"emg_grips.yaml open 에 없는 관절: {missing}")
    return poses, P.get("emg_map", {})


THUMB = ("thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip")


def _clamp(v):
    return max(0.0, min(1.0, v))


def follow_progress(q, key, poses):
    F = poses["_follow"][key]; o, g, d = poses["open"], poses[key], F.get("driver", "middle_mcp")
    full = g[d] - o[d]
    return _clamp((q[d] - o[d]) / full) if abs(full) > 1e-6 else 1.0


def _sched(p, s):
    u = _clamp((p - s) / max(1e-6, 1.0 - s))
    return u * u * (3 - 2 * u)                                                  # S 곡선 (시작·끝 부드럽게)


def follow_thumb(q, key, poses, base=None, p0=0.0):
    """thumb_follow 자세: 엄지 각도를 중지 굽힘 진행도 p 의 함수로 (손가락과 연속으로, 끝마디부터 접힘)
    base/p0: 이 자세로 들어올 때의 엄지 각도·진행도 (예비 자세에서 시작해도 엄지가 open 쪽으로 되돌아가지 않게)"""
    F = poses["_follow"].get(key)
    if not F:
        return q
    o, g = poses["open"], poses[key]
    p = follow_progress(q, key, poses)
    for j, s in F["start"].items():
        b = o[j] if base is None else base[j]
        u0 = _sched(p0, s) if base is not None else 0.0
        u = (_sched(p, s) - u0) / max(1e-6, 1.0 - u0)
        q[j] = b + max(0.0, u) * (g[j] - b)
    return q


def staged_move(hand, poses, cur, key, steps):
    """일반 자세는 한 번에, thumb_follow 자세는 손가락을 나눠 움직이며 매 단계 엄지를 중지 진행도에 맞춤"""
    tgt = poses[key]
    if key in poses["_follow"]:
        for k in range(1, steps + 1):
            a = k / steps
            q = follow_thumb({j: cur[j] + a * (tgt[j] - cur[j]) for j in tgt}, key, poses)
            hand.set_joint_positions(q); time.sleep(0.02)
    else:
        hand.set_joint_positions(tgt, num_steps=steps, step_size=0.02)
    return dict(tgt)


# 손가락 시차 (굽힐 때만, 펼 때는 바로): 검지 → 중지 → 약지 → 소지 순으로 말아 쥠
STAGGER = {"index": 0.0, "middle": 0.03, "ring": 0.06, "pinky": 0.09}
OPEN_DELAY = {"index": 0.12, "middle": 0.12, "ring": 0.14, "pinky": 0.16}   # 펼 때 손가락 늦춤 (s)
LAG = {}          # 관절별 ω 배율 (main 에서 --ring-lag / --pinky-lag 로 채움): 약지·소지가 중지보다 느리게 따라옴


class Smoother:
    """사람처럼 부드러운 관절 궤적 (v3, 10-29)
    · 3차 임계감쇠 필터 (1차 저역통과 → 2차 스프링·댐퍼, 세 극점 모두 ω): 속도·가속도가 0 에서 시작해 종 모양 속도 곡선
      (최소 저크 궤적과 비슷, Flash & Hogan 1985). 95 % 도착 ≈ 6.3/ω s, 목표가 바뀌어도 속도가 끊기지 않음
    · 모든 관절이 같은 ω → 선형 시스템이라 같은 시각에 출발·도착 (관절 동기화).
      속도 상한을 넘으면 모든 관절 속도를 같은 비율로 줄여 동기화 유지
    · 손가락 시차 STAGGER: 굽힘 쪽 목표만 늦게 받음 (펴기는 지연 없음 → 놓기 안전)
    · thumb_follow 자세: 엄지 = 걸러진 중지 진행도의 함수
    """

    def __init__(self, q0, poses):
        self.poses = poses
        self.r, self.x = dict(q0), dict(q0)
        self.v = {j: 0.0 for j in q0}
        self.hist = []                       # (t, 목표) 최근 0.3 s
        self.fkey, self.fbase, self.fp0 = None, None, 0.0
        self.last_grip = None                # 마지막 쥐기 자세 (풀 때 속도를 자세별로: --wrap-rel-*)

    def _delayed(self, j, t, tgt):
        f = j.split("_")[0]; d = STAGGER.get(f, 0.0) if j.endswith(("_mcp", "_pip")) else 0.0
        g = tgt[j]
        if d <= 0 or f == "thumb":
            return g
        past = self.hist[0][1]
        for tt, q in self.hist:
            if tt <= t - d: past = q
            else: break
        o = self.poses["open"][j]
        return past[j] if abs(past[j] - o) < abs(g - o) else g     # 굽힘만 늦춤

    def _delayed_open(self, j, t, g):
        """펼 때 네 손가락은 OPEN_DELAY s 늦게 펴짐 (엄지가 먼저 떠서 비킨 뒤) — 펼 때 엄지·손가락 간섭 방지 (10-31)"""
        f = j.split("_")[0]; d = OPEN_DELAY.get(f, 0.0) if j.endswith(("_mcp", "_pip")) else 0.0
        if d > 0 and self.last_grip == "wrap":      # 감싸 쥐기 풀 때: 네 손가락 늦춤 따로 (검지 기준, 한 손가락마다 +0.02 s)
            d = ARGS.wrap_rel_delay + {"index": 0.0, "middle": 0.0, "ring": 0.02, "pinky": 0.04}.get(f, 0.0)
        if d <= 0 or not self.hist:
            return g
        past = self.hist[0][1]
        for tt, q in self.hist:
            if tt <= t - d: past = q
            else: break
        o = self.poses["open"][j]
        return past[j] if abs(past[j] - o) > abs(g - o) else g     # 펴기만 늦춤

    def step(self, tgt, key, t, dt):
        o = self.poses["open"]
        if self.hist and t < self.hist[-1][0]:     # 시간이 되돌아가면 기록 초기화 (안 하면 시차 손가락이 옛 목표에 멈춤)
            self.hist = []
        self.hist.append((t, dict(tgt)))
        while self.hist and self.hist[0][0] < t - 0.8:
            self.hist.pop(0)
        w0 = ARGS.w_open if key in ("open", "relax") else ARGS.w_flex
        if key not in ("open", "relax", "preshape"):
            self.last_grip = key
        wrap_rel = key in ("open", "relax") and self.last_grip == "wrap"
        follow = key in self.poses["_follow"]
        if key == "preshape":                     # 예비 자세: 일정 속도로 천천히 다가감 (ARGS.t_pre s 에 도착)
            o_ = self.poses["open"]               #   지수 접근이면 확정 직전에 감속 → 확정 후 재가속 = '멈칫' (v3 문제)
        for j in self.x:
            w = w0 * LAG.get(j, 1.0)                                  # 약지 한 단계, 소지 두 단계 느리게 (중지를 따라옴)
            g = self._delayed(j, t, tgt)
            if abs(g - self.poses["open"][j]) < abs(self.x[j] - self.poses["open"][j]):      # 펴는 중
                if j in THUMB:
                    w *= ARGS.wrap_rel_thumb if wrap_rel else ARGS.thumb_open_boost      # 엄지는 먼저 떠서 비킴
                else:
                    if wrap_rel:
                        w *= ARGS.wrap_rel_fingers
                    g = self._delayed_open(j, t, g)                   # 네 손가락은 조금 늦게
            k1 = 1.0 - math.exp(-w * dt)
            if key == "preshape":
                step = abs(tgt[j] - o_[j]) / ARGS.t_pre * dt
                self.r[j] += max(-step, min(step, g - self.r[j]))
            else:
                self.r[j] += (g - self.r[j]) * k1
            a = w * w * (self.r[j] - self.x[j]) - 2.0 * w * self.v[j]
            self.v[j] += a * dt
        s = 1.0                                   # 속도 상한: 모든 관절 같은 비율로 (동기화 유지)
        for j in self.x:
            if follow and j in THUMB:
                continue
            toward_open = (self.v[j] * (o[j] - self.x[j])) > 0
            cap = ARGS.vopen if toward_open else ARGS.vflex
            if abs(self.v[j]) > cap:
                s = min(s, cap / abs(self.v[j]))
        for j in self.x:
            if follow and j in THUMB:
                continue
            self.v[j] *= s; self.x[j] += self.v[j] * dt
        if follow and self.fkey != key:           # 이 자세로 막 들어옴: 지금 엄지·중지 진행도를 출발점으로
            self.fkey, self.fbase, self.fp0 = key, {j: self.x[j] for j in THUMB}, min(0.95, follow_progress(self.x, key, self.poses))
        if not follow:
            self.fkey = None
        if follow:                                # 엄지는 걸러진 중지 위치에서 계산 (이미 부드러움), 2×vflex 상한
            want = follow_thumb(dict(self.x), key, self.poses, self.fbase, self.fp0)
            for j in THUMB:
                nv = max(-2 * ARGS.vflex, min(2 * ARGS.vflex, (want[j] - self.x[j]) / dt))
                self.x[j] += nv * dt; self.v[j] = nv; self.r[j] = self.x[j]
        return self.x

    def settled(self, tgt, tol=1.0):
        return all(abs(self.x[j] - tgt[j]) < tol and abs(self.v[j]) < 5.0 for j in tgt)


def target_of(msg, poses, emg_map):
    """휴식 → 대기 자세(relax, 살짝 오므림). 손 펴기 → relax 에서 세기만큼 open 으로. 쥐기 → relax 에서 세기만큼 쥐기 자세로"""
    o = poses["relax"] if ARGS.rest_pose == "relax" else poses["open"]
    rest_key = ARGS.rest_pose
    if not msg or msg.get("state") != "active" or not msg.get("grip"):
        return o, rest_key
    key = emg_map.get(msg["grip"], "open")
    lv = norm_level(msg.get("level", 0.0))
    if key == "open":
        if not msg.get("committed") and rest_key == "relax":
            return o, rest_key
        g = poses["open"]
        return {j: o[j] + max(lv, 0.6) * (g[j] - o[j]) for j in o}, "open"     # 손 펴기는 약해도 60 % 이상 폄
    if not msg.get("committed"):                 # 예비 자세도 세기에 비례 (약하게 시작하면 조금만 굽힘 → 확정 후 되돌아 펴지는 일 없음)
        g = poses["preshape"]; lp = min(1.0, lv / max(1e-6, ARGS.pre_frac))
        return {j: o[j] + lp * (g[j] - o[j]) for j in o}, "preshape"
    g = poses[key]
    return {j: o[j] + lv * (g[j] - o[j]) for j in o}, key


def norm_level(raw):
    """grip_rt 세기(level) → 쥐는 정도 0~1.
    S02 실측: 가장 약한 동작 ≈ 0.30, RAMP 세게 ≈ 0.65~0.75 로 좁게 몰려 있음 → 이 구간을 0~1 로 펼침
    쥐는 정도 = ((level − lo)/(hi − lo))^gamma × gain   (gamma < 1 이면 약한 힘에서 더 많이 쥠)"""
    x = (float(raw) - ARGS.lv_lo) / max(1e-6, ARGS.lv_hi - ARGS.lv_lo)
    x = max(0.0, min(1.0, x)) ** ARGS.lv_gamma
    return max(0.0, min(1.0, x * ARGS.gain))


class ContactHold:
    """닿음 고정 (10-31): 굽히는 중 관절의 모터 전류가 i_contact 이상이고 실제 위치가 명령보다 err_deg 이상 뒤처지면(= 막힘)
    contact_n 번 연속일 때 '닿음' → 그 관절 목표를 실제 위치 + squeeze_deg 로 고정 (그 이상 밀지 않고 일정한 힘으로만 누름).
    명령이 다시 펴는 쪽으로 돌아오면 해제. 물체가 없으면 관절 끝에서 막혀 같은 방식으로 '완전히 쥔 최대점'이 됨"""

    def __init__(self, hand, poses):
        self.hand, self.poses = hand, poses
        self.dir = {j: (1.0 if poses["fist"][j] >= poses["open"][j] else -1.0) for j in poses["open"]}
        self.cnt = {j: 0 for j in poses["open"]}; self.lock = {}
        self.ok = hasattr(hand, "get_motor_state") and hasattr(hand, "_motor_to_joint_pos")
        m2j = hand.config.motor_to_joint_dict if self.ok else {}
        self.order = [m2j.get(mid) for mid in hand.config.motor_ids] if self.ok else []
        self.last = {}
        # 미끄러짐 반사 (10-31): 고정된 관절이 고정점 쪽으로 더 들어가고(물체가 빠지거나 밀려남) 전류가 허공 수준으로 떨어지면
        #   → 고정 풀고 다시 닫기 (새로 닿으면 다시 고정) + slip_boost s 동안 '꽉' 전류. 사람의 '잡은 물건이 빠지면 반사적으로 더 쥠'
        self.base = {}; self.low = {j: 0 for j in poses["open"]}; self.firm = False; self.t_slip = -1e9; self.n_slip = 0

    def boosted(self):
        return time.monotonic() - self.t_slip < ARGS.slip_boost

    def _sq(self):
        return ARGS.squeeze_deg * (ARGS.firm_sq if (self.firm or self.boosted()) else 1.0)

    def apply(self, cmd, sm=None, tgt=None):
        tgt = tgt or cmd                                        # 해제 판단은 최종 목표로 (부드럽게 따라가는 중간값으로 보면 고정에 다시 붙잡힘)
        if not self.ok or ARGS.no_contact:
            return cmd
        try:
            r = self.hand.get_motor_state()
            jp = self.hand._motor_to_joint_pos(r.position)
        except Exception:
            return cmd
        cur = {j: abs(float(c)) for j, c in zip(self.order, r.current) if j}
        out = dict(cmd)
        for j, q in cmd.items():
            p = jp.get(j)
            if p is None:
                continue
            d = self.dir[j]
            if j in self.lock:
                if (tgt.get(j, q) - self.base[j]) * d < -2.0:  # 목표가 닿은 점보다 펴는 쪽으로 돌아옴 → 해제
                    del self.lock[j]; self.cnt[j] = 0; self.low[j] = 0
                    continue
                slipped = (p - self.base[j]) * d > ARGS.slip_frac * self._sq() and cur.get(j, 0.0) < ARGS.i_slip
                self.low[j] = self.low[j] + 1 if slipped else 0
                if ARGS.slip_n > 0 and self.low[j] >= ARGS.slip_n and (tgt.get(j, q) - p) * d > ARGS.err_deg:
                    del self.lock[j]; self.cnt[j] = 0; self.low[j] = 0          # 미끄러짐 → 다시 닫기 (sm 은 지금 위치에서 목표로 계속)
                    self.t_slip = time.monotonic(); self.n_slip += 1
                    continue
                self.lock[j] = self.base[j] + d * self._sq()                    # 꽉/살살 바뀌면 누르는 깊이도 바뀜
                out[j] = self.lock[j]
                if sm is not None: sm.x[j] = sm.r[j] = self.lock[j]; sm.v[j] = 0.0
                continue
            lag = (q - p) * d                                   # 명령이 실제보다 굽히는 쪽으로 앞선 정도
            if lag > ARGS.err_deg and cur.get(j, 0.0) >= ARGS.i_contact:
                self.cnt[j] += 1
                if self.cnt[j] >= ARGS.contact_n:
                    self.base[j] = p; self.lock[j] = p + d * self._sq()
                    out[j] = self.lock[j]
                    if sm is not None: sm.x[j] = sm.r[j] = self.lock[j]; sm.v[j] = 0.0
            else:
                self.cnt[j] = 0
        self.last = {"locked": sorted(self.lock), "cur": cur}
        return out


def joint_group(j):
    """관절 → 기준점 경로 묶음: thumb / index / middle / ring / pinky (굽힘) / spread (네 손가락 벌림)"""
    f = j.split("_")[0]
    if f == "thumb":
        return "thumb"
    return "spread" if j.endswith("_abd") else f


def path_progress(poses, key, j, c):
    """쥐는 정도 c (0~1) → 이 관절 묶음이 대기 → 자세까지 간 비율 (기준점 사이 직선 보간, Smoother 가 부드럽게 이음).
    경로가 없는 자세·묶음은 c 그대로 (이전 방식)"""
    pp = poses["_paths"].get(key, {})
    pts = pp.get(j) or pp.get(joint_group(j))             # 관절 이름(예: thumb_abd)이 따로 있으면 그것 우선, 없으면 묶음(thumb)
    if not pts:
        return c
    if c <= pts[0][0]:
        return pts[0][1] * (c / pts[0][0]) if pts[0][0] > 0 else pts[0][1]
    for (a0, b0), (a1, b1) in zip(pts, pts[1:]):
        if c <= a1:
            return b0 + (b1 - b0) * (c - a0) / max(1e-6, a1 - a0)
    return pts[-1][1]


def arc_offset(poses, key, j, c):
    """호(arc): 쥐는 정도 c0~c1 사이에서 이 관절을 '펴는 쪽(open)'으로 잠깐 띄웠다 돌아옴 (sin 반주기, 최대 amp 비율)
    예: 주먹 때 엄지 끝이 손바닥 위로 포물선을 그리며 넘어가 손가락 위에 내려앉음 (10-31 영상)"""
    a = poses["_arcs"].get(key, {}).get(j)
    if not a:
        return 0.0
    c0, c1, amp = a
    if not (c0 < c < c1):
        return 0.0
    return amp * math.sin(math.pi * (c - c0) / (c1 - c0)) * (poses["open"][j] - poses["relax"][j])


def pose_at(poses, key, j, c):
    """쥐는 정도 c 에서 이 자세의 관절 j 목표 (대기 → 자세, 기준점 경로 + 호)
    via 가 있으면 대기(0) → 중간 자세(c1) → … → 자세(1) 를 차례로 이음 (예: 감싸 쥐기 = 대기 → 엄지 벌린 준비 자세 → 끝까지 쥐기)"""
    o = poses["relax"]
    via = poses.get("_via", {}).get(key)
    if via:
        keys = [(0.0, "relax")] + [(cc, n) for cc, n in via if n in poses] + [(1.0, key)]
        for (c0, n0), (c1, n1) in zip(keys, keys[1:]):
            if c <= c1:
                u = 0.0 if c1 <= c0 else max(0.0, min(1.0, (c - c0) / (c1 - c0)))
                if n1 == key and poses["_paths"].get(key):                # 마지막 구간(준비 → 자세)은 관절마다 타이밍 (paths 를 이 구간 기준 0~1 로)
                    u = path_progress(poses, key, j, u)                   #   예: 손가락 먼저 감싸고 → 엄지 뿌리 돌아옴 → 엄지 끝마디 마지막에 잠금
                else:
                    u = u * u * (3 - 2 * u)                               # 구간마다 부드럽게 출발·도착
                return poses[n0][j] + u * (poses[n1][j] - poses[n0][j])
        return poses[key][j]
    return o[j] + path_progress(poses, key, j, c) * (poses[key][j] - o[j]) + arc_offset(poses, key, j, c)


def map_key(name, emg_map):
    """grip_rt 동작 이름 → 자세 이름. '엄지 구부리기'·'엄지 굽히기' 도 emg_map 의 '엄지' 로"""
    if name in emg_map:
        return emg_map[name]
    n = name.replace("_", " ").strip()
    for fg in ("엄지", "검지", "중지"):
        key = fg + " 펴기" if "펴" in n else fg                 # '엄지 펴기' → emg_map['엄지 펴기'], '엄지 굽히기' → emg_map['엄지']
        if n.startswith(fg) and "(" not in n and key in emg_map:
            return emg_map[key]
    return None


class Composer:
    """근전도 메시지 → 목표 자세 (v10, 10-30: 사람 근육처럼)
    · 확률 섞기: 동작별 누적 확률(post)로 자세를 가중 평균 → 손가락 자세가 연속으로 바뀜 (확정 전에도 가장 그럴듯한 쪽으로)
    · 스프링 쥐기 (평형점·임피던스 제어): 근전도 세기 → ① 목표 굽힘 (물체보다 squeeze 만큼 더 안쪽)
      ② 모터 전류 상한 = 강성. 물체에 닿으면 손가락이 멈추고 힘 = 강성 × (목표 − 실제) 로 저절로 생김
    · 자세 의도 유지: 같은 동작 동안 굽힘은 세기의 최고값을 천천히(leak/s) 놓음 → 힘을 조금 빼도 자세는 남고 강성(힘)만 줄어듦
    """

    def __init__(self, poses, emg_map):
        self.poses, self.emg_map = poses, emg_map; self.emg_map0 = dict(emg_map)
        self.names = None; self.key = None; self.c = 0.0; self.t_on = 0.0; self.tg = 0.0; self.firm = False; self.dwell = 0.0
        self.now = 0.0; self.t_open = -1e9; self.fist_as = None      # 손 펴기 직후 주먹 = 감싸 쥐기 (10-31)
        self.thumb_keys = {k for k, q in poses.items() if not k.startswith("_") and k not in ("open", "relax", "preshape")
                           and any(abs(q[j] - poses["relax"][j]) > 8 for j in THUMB)}     # 엄지가 실제로 움직이는 동작들

    def _unwind(self, dt, stop_at_via=False):
        """감싸 쥐기 풀기 (10-31): 쥘 때 경로를 거꾸로 — 엄지 끝마디 먼저 펴고 → 엄지 뿌리가 C 자로 돌아나가며 네 손가락 폄
        → 엄지 벌린 준비 자세(wrap_pre)에서 wrap_rel_hold s 머묾 (물체가 엄지·검지 사이에 끼지 않고 빠짐) → 대기.
        다시 쥐면 지금 c 에서 이어서 닫음"""
        vc = min([c for c, n in self.poses["_via"].get("wrap", []) if n in self.poses] or [0.0])
        rate = dt / ARGS.wrap_unwind
        if self.c > vc:
            self.c = max(vc, self.c - rate); self.dwell = 0.0
        elif self.dwell < ARGS.wrap_rel_hold:
            self.dwell += dt
        elif stop_at_via:                       # 손 펴기로 넘어가는 중: 엄지 벌린 준비 자세까지 풀었으면 바로 펴기로 (대기로 좁히지 않음)
            self.c = 0.0
        else:
            self.c = max(0.0, self.c - rate)
        roms = self.poses["_roms"]
        tgt = {j: min(max(pose_at(self.poses, "wrap", j, self.c), roms[j][0]), roms[j][1]) for j in self.poses["relax"]}
        if self.c <= 0:
            self.key = None
        return tgt

    def base(self):
        return (self.poses["relax"], "relax") if ARGS.rest_pose == "relax" else (self.poses["open"], "open")

    def __call__(self, msg, dt):
        self.now += dt
        o, rest_key = self.base()
        if msg and msg.get("names"):
            self.names = msg["names"]
        mm = self.poses["_mode_map"].get((msg or {}).get("mode") or "", {})
        self.emg_map = {**self.emg_map0, **mm} if mm else self.emg_map0       # 모드에 따라 같은 근전도 동작이 다른 자세 (주먹 → 원통 쥐기 / 바위)
        if not msg or msg.get("state") != "active" or not msg.get("grip"):
            self.t_on, self.firm = 0.0, False
            if self.key != "wrap" or self.c <= 0:
                self.fist_as = None
            if self.key == "wrap" and ARGS.wrap_unwind > 0 and self.c > 0:
                self.tg += (0.0 - self.tg) * (1 - math.exp(-dt / ARGS.thumb_fall))
                return self._unwind(dt), rest_key, 0.0
            self.key, self.c = None, 0.0
            self.tg += (0.0 - self.tg) * (1 - math.exp(-dt / ARGS.thumb_fall))
            return o, rest_key, 0.0
        # 대기 안정: 켜진 직후 on_delay s 는 무시, 그 뒤 on_ramp s 동안 서서히 참여 → 짧은 오작동(깜빡임)에 손이 움찔하지 않음
        self.t_on += dt
        eng = min(1.0, max(0.0, (self.t_on - ARGS.on_delay) / max(1e-3, ARGS.on_ramp)))
        if eng <= 0:
            if self.key == "wrap" and ARGS.wrap_unwind > 0 and self.c > 0:     # 풀던 중 다시 켜짐: 확정될 때까지는 계속 풂 (대기로 튀지 않게)
                return self._unwind(dt), rest_key, 0.0
            return o, rest_key, 0.0
        lv = norm_level(msg.get("level", 0.0))
        # 가중치: 섞기 모드면 post, 아니면 확정 동작 1개 (미확정 = preshape)
        wts = {}
        if ARGS.blend and msg.get("post") and self.names and len(self.names) == len(msg["post"]):
            for nm, p in zip(self.names, msg["post"]):
                k = map_key(nm, self.emg_map)
                d = ARGS.dz_thumb if k in ("thumb_flex", "scissors") else ARGS.dz      # 문턱 아래 확률은 버림 (엄지·가위는 더 높게 → 애매할 때 함부로 안 섞임. 10-31: 손 펴기 중 가위 확률이 섞여 V 처럼 보임)
                p = max(0.0, (p - d) / (1.0 - d))
                if k and p > 0:
                    wts[k] = wts.get(k, 0.0) + p
        else:
            k = map_key(msg["grip"], self.emg_map) or "open"
            wts = {k if msg.get("committed") else ("preshape" if k != "open" else rest_key): 1.0}
        # 주먹 판단 (10-31): 손 펴기를 하고 open_window s 안에 주먹 → 감싸 쥐기(wrap, 물체를 잡으려고 손을 벌렸다 쥠 = 실제 손과 같음)
        #   그냥 주먹(대기에서 바로) → 바위(fist). 한 번 정하면 이번 쥐기 동안 유지 (중간에 안 바뀜)
        if "fist" in wts and ARGS.open_window > 0 and "gesture" not in str(msg.get("mode") or ""):   # 동작 모드(바위)에서는 안 바꿈
            if self.fist_as is None:
                self.fist_as = "wrap" if self.now - self.t_open <= ARGS.open_window else "fist"
            if self.fist_as == "wrap":
                wts["wrap"] = wts.get("wrap", 0.0) + wts.pop("fist")
        tot = sum(wts.values()) or 1.0
        wts = {k: v / tot for k, v in wts.items() if k in self.poses}
        if not wts:
            return o, rest_key, 0.0
        key = max(wts, key=wts.get)
        if key == "open" and msg.get("committed"):
            self.t_open = self.now
        # 감싸 쥐기에서 다른 동작(대개 손 펴기 — 놓을 때 펴는 근육)으로 넘어갈 때도 쥘 때 경로를 거꾸로 풂 (10-31 실측: 놓을 때 휴식이 아니라
        #   손 펴기로 판단되면 바로 펴기 자세로 가서 엄지가 검지 쪽으로 좁혀 들어오며 물체가 낌)
        if self.key == "wrap" and key != "wrap" and self.c > 0 and ARGS.wrap_unwind > 0:
            return self._unwind(dt, stop_at_via=(key == "open")), rest_key, 0.0
        # 스스로 쥐기 (10-31): 쥐기 모드 자세(wrap·pinch·tripod)는 확정되면 근전도 세기와 상관없이 auto_close s 동안 끝까지 닫음
        #   → 물체에 닿은 손가락은 ContactHold 가 그 자리에서 멈추고 누름. 힘 = 전류 상한 (살살 i_light / 꽉 i_max). 근육 힘을 빼면(휴식) 놓음 (나)
        auto = key in AUTO_KEYS and ARGS.auto_close > 0 and msg.get("committed")
        if auto:
            self.c = min(1.0, (self.c if key == self.key else 0.0) + dt / ARGS.auto_close)
        else:                                       # 동작 모드: 자세 의도 유지 (같은 동작이면 최고값을 천천히 놓음)
            self.c = max(self.c - ARGS.leak * dt, lv) if key == self.key else lv
        self.key = key
        self.firm = bool(msg.get("firm")) and key in FIRM_KEYS
        if key == "open":                          # 손 펴기도 세기 비례 (v11: 이전 '최소 60 %' 는 쥐었다 놓을 때의 펴는 근육 반동에도 손을 활짝 펴서 대기가 흔들림)
            c = max(self.c, 0.6) if ARGS.open_dz < 0 else max(0.0, (self.c - ARGS.open_dz) / (1.0 - ARGS.open_dz))
        else:
            c = self.c
        c *= eng
        sq = ARGS.squeeze * (1.0 if auto else lv) * eng if key not in ("open", "relax", "preshape") else 0.0
        # 엄지 문: 엄지 쓰는 동작(주먹·집기·세손가락·엄지)의 확률 합이 thumb_gate 를 넘어 '유지'될 때만 엄지가 따라감
        #   천천히 열리고(thumb_rise s) 빨리 닫힘(thumb_fall s) → 순간 오판(중지→세손가락 등)·놓은 뒤 남은 신호에 엄지가 안 접힘
        tw = sum(w for k, w in wts.items() if k in self.thumb_keys)
        want = min(1.0, max(0.0, (tw - ARGS.thumb_gate) / max(1e-3, 1.0 - ARGS.thumb_gate) * 2.0))
        tau = ARGS.thumb_rise if want > self.tg else ARGS.thumb_fall
        self.tg += (want - self.tg) * (1 - math.exp(-dt / tau))
        roms = self.poses["_roms"]; tgt = {}
        for j in o:
            # 기준점 경로: 동작마다 이 관절 묶음의 진행 비율 (예: 주먹은 손가락 먼저·엄지 나중, 집기는 엄지 먼저)
            # 엄지 문은 엄지를 접는 자세에만 (10-31 수정: 이전엔 손 펴기에도 곱해져 엄지가 대기 자리에 머물러 손 펴기 엄지가 안 바뀌었음)
            q = o[j]
            for k, w in wts.items():
                dk = ((pose_at(self.poses, k, j, c) - self.poses["relax"][j]) if k in self.poses["_via"]
                      else (path_progress(self.poses, k, j, c) * (self.poses[k][j] - o[j]) + arc_offset(self.poses, k, j, c)))
                gate = self.tg if (j in THUMB and k in self.thumb_keys) else 1.0
                q += w * (dk + sq * (self.poses[k][j] - o[j])) * gate
            tgt[j] = min(max(q, roms[j][0]), roms[j][1])
        return tgt, key, ((1.0 if auto else lv) if key not in ("open", "relax", "preshape") else 0.0)


class Stiffness:
    """모터 전류 상한 = 손가락 강성. 쥐기 중: i_min + (i_max − i_min)·세기, 휴식: i_relax (사람 손도 쉴 때 말랑함), 펴기: i_max
    오래 쥐면(hold_relief s 이상) 열 관리로 증가분을 80 % 로. 바뀔 때만 (≥ 15 mA, 0.1 s 간격) 써서 버스 부담 줄임"""

    def __init__(self, hand):
        self.hand = hand; self.last = None; self.t_last = 0.0; self.t_grip = None; self.key = None
        self.i_max = ARGS.i_max or float(hand.config.max_current)

    def update(self, key, lv_force, now, firm=False):
        if key in ("open",):
            i = self.i_max
        elif key in ("relax", "preshape") or lv_force <= 0:
            i = ARGS.i_relax
        elif key in AUTO_KEYS and ARGS.auto_close > 0:   # 스스로 쥐기: 두 단계 (10-31 근전도 꽉/살살 GR2 83 %, 놓기 확인 꽉→살살 전환 9/9)
            i = self.i_max if firm else ARGS.i_light
        else:
            if key != self.key: self.t_grip = now
            span = (self.i_max - ARGS.i_min) * (0.8 if now - (self.t_grip or now) > ARGS.hold_relief else 1.0)
            i = ARGS.i_min + span * lv_force
        self.key = key
        i = min(i, self.i_max)
        if self.last is None or (abs(i - self.last) >= 15 and now - self.t_last >= 0.1):
            self.hand.set_max_current(i); self.last = i; self.t_last = now
        return self.last


def run(hand, stopped, read, registers):
    poses, emg_map = build_poses(hand, ARGS.grips)
    for name, q in poses.items():
        if name.startswith("_"): continue
        print(f"{name:9s} " + " ".join(f"{j}={q[j]:.0f}" for j in ("thumb_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")), flush=True)
    hc = {"i": 0, "miss": 0}; fails = {}
    checks = [(mid, kind) for mid in hand.config.motor_ids for kind in ("temp", "err")]

    def health():
        """매 제어 주기마다 읽기 1번만 (모터 하나의 온도 또는 오류) → 32 주기(0.64 s)에 16개 모두 검사.
        이전: 2 s 마다 32번을 한꺼번에 읽어 USB 지연(Windows 기본 16 ms/회)이 쌓이면 손이 순간 멈칫"""
        hc["t"] = hc.get("t", 0) + 1
        if hc["t"] % ARGS.check_every:             # 읽기 1 번이 USB 지연으로 ~16~30 ms → N 주기에 한 번만 (기본 5 → 16 모터 전체 3.2 s)
            return
        mid, kind = checks[hc["i"] % len(checks)]; hc["i"] += 1
        try:
            if kind == "err":
                val = read(mid, registers.ADDR_HARDWARE_ERROR_STATUS, 1)
            else:
                val = read(mid, registers.ADDR_PRESENT_TEMPERATURE, 1)
        except RuntimeError as e:                  # 응답 시간 초과(-3001) 등 일시적 통신 실패: 같은 모터가 연속 3 번이면 중단
            fails[mid] = fails.get(mid, 0) + 1
            hc["miss"] += 1
            if fails[mid] >= 3:
                raise RuntimeError(f"ID {mid}: 상태 읽기 연속 3 번 실패 — 케이블·전원 확인 ({e})")
            return
        fails[mid] = 0
        if kind == "err" and val:
            raise RuntimeError(f"ID {mid}: hardware fault")
        if kind == "temp" and val >= hand.motor_client.max_operating_temp_c:
            raise RuntimeError(f"ID {mid}: 모터 온도 한계 {val}C")

    print("open 으로 천천히 이동", flush=True)
    hand.set_joint_positions(poses["open"], num_steps=50, step_size=0.02)
    cur = dict(poses["open"])

    if ARGS.sweep:                               # 관절 하나씩 비율 0 → 1 로 훑기 (자세 조정용)
        roms = hand.config.joint_roms_dict
        for j in ARGS.sweep:
            lo, hi = roms[j]
            hand.set_joint_positions(poses[ARGS.base], num_steps=40, step_size=0.02)
            print(f"\n== {j} 훑기 ({ARGS.base} 자세에서 이 관절만, 각 {ARGS.hold:g} s) — 원하는 모양일 때의 비율을 기억", flush=True)
            for f in [i / 10 for i in range(0, 11, ARGS.sweep_step)]:
                if stopped.is_set():
                    return 130
                health(); q = dict(poses[ARGS.base]); q[j] = lo + f * (hi - lo)
                print(f"  {j} = {f:.1f}  ({q[j]:.0f}°)", flush=True)
                hand.set_joint_positions(q, num_steps=40, step_size=0.02); stopped.wait(ARGS.hold)
            hand.set_joint_positions(poses["open"], num_steps=40, step_size=0.02)
        return 0

    if ARGS.path:                                # 기준점 경로 보기: 쥐는 정도 0 → 1 → 0 을 단계별로 (각 ARGS.hold s)
        key = ARGS.path; o = poses["relax"]; roms = poses["_roms"]
        sm = Smoother(cur, poses); dt = 1.0 / ARGS.rate; tt = 0.0
        for c in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.6, 0.3, 0.0):
            q = {j: min(max(pose_at(poses, key, j, c), roms[j][0]), roms[j][1]) for j in o}
            print(f"  {key} 쥐는 정도 {c:.1f}:  " + "  ".join(f"{g} {path_progress(poses, key, gj, c):.2f}" for g, gj in
                  (("엄지", "thumb_mcp"), ("검지", "index_mcp"), ("중지", "middle_mcp"), ("약지", "ring_mcp"), ("소지", "pinky_mcp"))), flush=True)
            t = 0.0
            while t < 1.5 + ARGS.hold and not stopped.is_set():
                tick = time.monotonic(); hand.set_joint_positions(sm.step(q, key, tt, dt)); health()
                t += dt; tt += dt; stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
            if stopped.is_set():
                return 130
        return 0

    if ARGS.poses:
        # v15 (10-31): 쥐기 자세는 대기 → 자세를 기준점 경로·중간 자세(via)·호 그대로 따라감 (이전엔 마지막 모양으로 바로 이동해서 순서가 안 보였음)
        if ARGS.only:
            seq = ("relax", ARGS.only, "relax")
        else:
            seq = ("open", "relax", "fist", "relax", "pinch", "relax", "tripod", "relax") + tuple(
                k for k in ("wrap", "scissors", "index_ext", "thumb_ext") if k in poses for k in (k, "relax")) + ("open",)
        roms = poses["_roms"]; dt = 1.0 / ARGS.rate
        contact = ContactHold(hand, poses); sm = None
        for name in seq:
            if stopped.is_set():
                return 130
            health(); print("→", name, flush=True)
            prev = sm.last_grip if sm is not None else None
            sm = Smoother(cur, poses); t = 0.0; sm.last_grip = prev
            if name == "relax" and prev == "wrap" and ARGS.wrap_unwind > 0:     # 감싸 쥐기 풀기 = 쥘 때 경로를 거꾸로 (실제 제어와 같은 계산)
                comp_ = Composer(poses, {}); comp_.key, comp_.c = "wrap", 1.0
                while comp_.key and not stopped.is_set():
                    q = comp_._unwind(dt)
                    tick = time.monotonic(); hand.set_joint_positions(contact.apply(dict(sm.step(q, "relax", t, dt)), sm, q)); health()
                    t += dt; stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
                for _ in range(int(1.0 / dt)):
                    tick = time.monotonic(); hand.set_joint_positions(contact.apply(dict(sm.step(poses["relax"], "relax", t, dt)), sm, poses["relax"])); health()
                    t += dt; stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
            elif name in ("relax", "open", "preshape"):         # 대기·펴기는 바로 (부드럽게)
                while t < 3.0 and not sm.settled(poses[name]) and not stopped.is_set():
                    tick = time.monotonic(); hand.set_joint_positions(contact.apply(dict(sm.step(poses[name], name, t, dt)), sm, poses[name])); health()
                    t += dt; stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
            else:                                              # 쥐기·동작 자세: 쥐는 정도 0 → 1 을 ARGS.close_s 초 동안 (실제 제어와 같은 순서)
                T = ARGS.close_s
                while t < T + 1.0 and not stopped.is_set():
                    c = min(1.0, t / T)
                    q = {j: min(max(pose_at(poses, name, j, c), roms[j][0]), roms[j][1]) for j in poses["relax"]}
                    tick = time.monotonic(); hand.set_joint_positions(contact.apply(dict(sm.step(q, name, t, dt)), sm, q)); health()
                    t += dt; stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
            cur = dict(sm.x)
            if contact.lock:
                print("  닿음 고정: " + ", ".join(contact.lock), flush=True)
            stopped.wait(ARGS.hold)
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host, port = ARGS.udp.split(":"); sock.bind((host, int(port))); sock.setblocking(False)
    demo = [("rest", None), ("active", "주먹"), ("rest", None), ("active", "집기 (중지 펴기)"), ("rest", None), ("active", "세손가락 (중지 굽힘)")]
    dt = 1.0 / ARGS.rate; msg, t_msg = None, 0.0; t0 = time.monotonic(); n = 0
    sm = Smoother(cur, poses); worst = 0.0
    comp = Composer(poses, emg_map); stiff = None if ARGS.no_stiffness else Stiffness(hand)
    contact = ContactHold(hand, poses)
    print(f"섞기 {'켜짐' if ARGS.blend else '꺼짐'}, 스프링 쥐기: 강성 {'고정' if ARGS.no_stiffness else f'{ARGS.i_min:g}~{stiff.i_max:g} mA (휴식 {ARGS.i_relax:g})'}, "
          f"눌러 쥐기 {ARGS.squeeze:g}, 약지·소지 속도 ×{ARGS.ring_lag:g}/×{ARGS.pinky_lag:g}", flush=True)
    print(f"궤적: {'이전 방식 (일정 속도)' if ARGS.raw else f'부드럽게 ω 굽힘 {ARGS.w_flex:g} / 펴기 {ARGS.w_open:g}'}", flush=True)
    print(f"{'데모' if ARGS.demo else '근전도 수신 ' + ARGS.udp}  — Ctrl+C 로 종료 (open 후 토크 해제)", flush=True)
    while not stopped.is_set():
        tick = time.monotonic()
        if ARGS.demo:
            T = 6.0 if ARGS.demo_ramp else 3.0
            ph = (tick - t0) % T; st, g = demo[int((tick - t0) / T) % len(demo)]
            if ARGS.demo_ramp:                    # 세기 0 → 1 → 0 (삼각형, 6 s) — 손이 세기 따라 천천히 쥐었다 펴져야 함
                f = 1.0 - abs(ph / T * 2 - 1.0)
                lvl = ARGS.lv_lo + f * (ARGS.lv_hi - ARGS.lv_lo)
            else:
                lvl = ARGS.lv_hi
            msg = {"state": st, "grip": g, "committed": ph > 0.4, "level": lvl}; t_msg = tick
        else:
            try:
                while True:
                    data, _ = sock.recvfrom(4096); msg = json.loads(data.decode()); t_msg = tick
            except BlockingIOError:
                pass
        if not ARGS.raw:
            tgt, key, lvf = comp(msg if tick - t_msg < 0.5 else None, dt)
            cur = dict(sm.step(tgt, key, tick - t0, dt))
            contact.firm = comp.firm
            cur = contact.apply(cur, sm, tgt)
            health(); hand.set_joint_positions(cur); n += 1
            i_now = stiff.update(key, lvf, tick, firm=comp.firm or contact.boosted()) if stiff else None
            worst = max(worst, (time.monotonic() - tick) * 1000)
            if n % max(1, int(ARGS.rate / 5)) == 0:
                m = msg or {}
                sys.stdout.write(f"\r{m.get('state', '-'):6s} {(m.get('grip') or '—'):16s} → {key:11s} {'꽉' if comp.firm else '  '} 쥠 {comp.c:.2f} "
                                 f"강성 {i_now if i_now is not None else 0:4.0f} mA  닿음 {len(contact.lock):2d} 미끄럼 {contact.n_slip}  검지 {cur['index_mcp']:4.0f}° 중지 {cur['middle_mcp']:4.0f}° "
                                 f"약지 {cur['ring_mcp']:4.0f}° 소지 {cur['pinky_mcp']:4.0f}°  주기 {worst:3.0f} ms  실패 {hc['miss']}   ")
                sys.stdout.flush(); worst = 0.0
            stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
            continue
        follow = key in poses["_follow"]
        for j in cur:                              # 손가락 먼저 (thumb_follow 자세면 엄지는 아래에서 중지에 맞춤)
            if follow and j in THUMB:
                continue
            d = tgt[j] - cur[j]; vmax = (ARGS.vopen if abs(tgt[j] - poses["open"][j]) < abs(cur[j] - poses["open"][j]) else ARGS.vflex) * dt
            cur[j] += max(-vmax, min(vmax, d))
        if follow:                                 # 엄지 = 지금 중지 진행도에서의 값, 너무 빠르지 않게 2×vflex 제한
            want = follow_thumb(dict(cur), key, poses)
            for j in THUMB:
                vmax = 2 * ARGS.vflex * dt
                cur[j] += max(-vmax, min(vmax, want[j] - cur[j]))
        health()
        hand.set_joint_positions(cur)
        n += 1
        if n % max(1, int(ARGS.rate / 5)) == 0:
            m = msg or {}
            sys.stdout.write(f"\r{m.get('state', '-'):6s} {(m.get('grip') or '—'):20s} {'확정' if m.get('committed') else '잠정'} "
                             f"lvl {float(m.get('level', 0)):.2f}  index_mcp {cur['index_mcp']:5.1f}°  middle_mcp {cur['middle_mcp']:5.1f}°   ")
            sys.stdout.flush()
        stopped.wait(max(0.0, dt - (time.monotonic() - tick)))
    print("\n종료 요청 → open", flush=True)
    try:
        hand.set_max_current(float(hand.config.max_current))       # 강성 원래대로 (팀 설정값)
    except Exception:
        pass
    hand.set_joint_positions(poses["open"], num_steps=30, step_size=0.02)
    return 0


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--udp", default="127.0.0.1:9000")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--vflex", type=float, default=250.0, help="굽힘 속도 상한 °/s (안전용)")
    ap.add_argument("--vopen", type=float, default=450.0, help="펴기 속도 상한 °/s")
    ap.add_argument("--w-flex", type=float, default=13.0, help="굽힘 부드러움 ω (클수록 빠름, 95%% 도착 ≈ 6.3/ω s)")
    ap.add_argument("--w-open", type=float, default=18.0, help="펴기 ω (놓기는 빠르게)")
    ap.add_argument("--i-contact", type=float, default=160.0, help="닿음 판단 전류 (mA). 허공에서 굽힐 때 약 100 mA (10-31 측정)")
    ap.add_argument("--err-deg", type=float, default=6.0, help="닿음 판단: 실제 위치가 명령보다 이 각도 이상 뒤처짐")
    ap.add_argument("--contact-n", type=int, default=3, help="닿음 판단: 연속 횟수")
    ap.add_argument("--squeeze-deg", type=float, default=6.0, help="닿은 뒤 그 위치보다 이만큼 더 안쪽을 목표 (누르는 힘)")
    ap.add_argument("--no-contact", action="store_true", help="닿음 고정 끄기")
    ap.add_argument("--auto-close", type=float, default=0.9, help="쥐기 모드 자세(wrap·pinch·tripod)를 확정 뒤 이 시간(s)에 끝까지 닫음 (스스로 쥐기). 0 = 이전 방식 (세기 비례)")
    ap.add_argument("--i-light", type=float, default=200.0, help="스스로 쥐기 '살살' 전류 상한 (mA). '꽉' = --i-max (팀 설정 300)")
    ap.add_argument("--firm-keys", default="wrap,tripod", help="근전도 '꽉'을 받는 자세 (집기는 10-31 검증 5/8 로 불안정 → 항상 살살)")
    ap.add_argument("--firm-sq", type=float, default=1.5, help="'꽉'일 때 닿은 뒤 누르는 깊이 배율 (× --squeeze-deg)")
    ap.add_argument("--i-slip", type=float, default=120.0, help="미끄러짐: 고정된 관절 전류가 이 값(허공 ≈ 100 mA) 아래로")
    ap.add_argument("--slip-frac", type=float, default=0.75, help="미끄러짐: 고정점 쪽으로 누르는 깊이의 이 비율 이상 들어감")
    ap.add_argument("--slip-n", type=int, default=5, help="미끄러짐: 연속 횟수 (50 Hz, 5 = 0.1 s). 0 = 반사 끄기")
    ap.add_argument("--slip-boost", type=float, default=1.5, help="미끄러짐 뒤 이 시간(s) 동안 '꽉' 전류")
    ap.add_argument("--thumb-open-boost", type=float, default=1.0, help="펼 때 엄지 속도 배율 (10-31: 2.5 는 너무 빨라 1.0 으로. 손가락은 OPEN_DELAY 만큼 늦게 펴지므로 엄지가 여전히 먼저 비킴)")
    ap.add_argument("--wrap-rel-fingers", type=float, default=0.5, help="감싸 쥐기 풀 때 네 손가락 속도 배율 (<1 느리게, >1 빠르게)")
    ap.add_argument("--wrap-rel-thumb", type=float, default=0.7, help="감싸 쥐기 풀 때 엄지 속도 배율")
    ap.add_argument("--wrap-rel-delay", type=float, default=0.2, help="감싸 쥐기 풀 때 네 손가락이 엄지보다 늦게 출발하는 시간 (s)")
    ap.add_argument("--wrap-unwind", type=float, default=0.9, help="감싸 쥐기 풀기 = 쥘 때 경로를 거꾸로 (엄지 끝 → 엄지 뿌리가 C 자로 벌어지며 손가락 폄 → 준비 자세 → 대기). 끝→대기 전체 시간 s, 0 = 바로 대기로")
    ap.add_argument("--wrap-rel-hold", type=float, default=0.5, help="풀 때 엄지 벌린 준비 자세(wrap_pre)에서 머무는 시간 s (물체가 빠질 틈)")
    ap.add_argument("--open-window", type=float, default=1.5, help="손 펴기 뒤 이 시간(s) 안에 주먹 → 감싸 쥐기(wrap), 아니면 바위(fist). 0 = 주먹은 emg_grips.yaml 대로")
    ap.add_argument("--close-s", type=float, default=2.5, help="--poses 에서 쥐기 자세를 닫는 시간 (s)")
    ap.add_argument("--path", default=None, help="기준점 경로 보기: 이 자세(fist/pinch/tripod/scissors)를 쥐는 정도 0→1→0 단계별로 (--hold 와 함께)")
    ap.add_argument("--no-blend", dest="blend", action="store_false", help="동작 확률 섞기 끄기 (확정 동작 1개만)")
    ap.add_argument("--thumb-gate", type=float, default=0.6, help="엄지 쓰는 동작 확률 합이 이 값을 넘어야 엄지가 움직임")
    ap.add_argument("--thumb-rise", type=float, default=0.25, help="엄지 문 열리는 시간상수 (s, 클수록 신중)")
    ap.add_argument("--thumb-fall", type=float, default=0.06, help="엄지 문 닫히는 시간상수 (s, 놓을 때 엄지 빨리 복귀)")
    ap.add_argument("--open-dz", type=float, default=0.25, help="손 펴기: 이 세기 아래는 대기 자세 유지 (놓을 때 반동 무시)")
    ap.add_argument("--dz", type=float, default=0.15, help="확률 섞기 문턱 (이 아래 확률은 무시)")
    ap.add_argument("--dz-thumb", type=float, default=0.45, help="엄지만 접기 동작의 확률 문턱 (높을수록 엄지가 확실할 때만 접힘)")
    ap.add_argument("--on-delay", type=float, default=0.10, help="켜진 뒤 이 시간(s)은 대기 유지 (짧은 오작동 무시)")
    ap.add_argument("--on-ramp", type=float, default=0.15, help="그 뒤 이 시간(s) 동안 서서히 움직이기 시작")
    ap.add_argument("--squeeze", type=float, default=0.12, help="쥘 때 목표를 자세보다 이만큼(× 세기) 더 안쪽으로 → 물체에 닿으면 누르는 힘")
    ap.add_argument("--leak", type=float, default=0.05, help="같은 동작 중 굽힘 최고값을 놓는 속도 (/s). 작을수록 자세 유지")
    ap.add_argument("--i-min", type=float, default=150.0, help="쥘 때 최소 강성 (mA)")
    ap.add_argument("--i-max", type=float, default=None, help="쥘 때 최대 강성 (mA, 기본 = 팀 config max_current)")
    ap.add_argument("--i-relax", type=float, default=240.0, help="휴식·대기 때 강성 (mA). 낮을수록 말랑·자연스럽지만 힘줄 마찰로 살짝 흔들림 (200 = 이전), 높을수록 단단")
    ap.add_argument("--hold-relief", type=float, default=15.0, help="이 시간(s) 넘게 쥐면 강성 증가분 80 %% (열 관리)")
    ap.add_argument("--no-stiffness", action="store_true", help="강성(전류) 고정 — 이전 방식")
    ap.add_argument("--ring-lag", type=float, default=0.7, help="약지 속도 배율 (중지 대비, 1 = 같게)")
    ap.add_argument("--pinky-lag", type=float, default=0.5, help="소지 속도 배율")
    ap.add_argument("--rest-pose", choices=["relax", "open"], default="relax", help="휴식 때 자세: relax = 살짝 오므린 대기, open = 쭉 폄 (이전 방식)")
    ap.add_argument("--lv-lo", type=float, default=0.40, help="이 세기 이하 = 쥐지 않음 (S02 가장 약한 동작 ≈ 0.30)")
    ap.add_argument("--lv-hi", type=float, default=0.65, help="이 세기 이상 = 끝까지 쥠 (S02 RAMP 중간~세게 ≈ 0.6~0.75)")
    ap.add_argument("--lv-gamma", type=float, default=1.0, help="세기 곡선 (0.7 = 약한 힘에 민감, 1.5 = 세게 쥐어야 닫힘)")
    ap.add_argument("--pre-frac", type=float, default=0.5, help="예비 자세가 다 되는 세기 (쥐는 정도 비율)")
    ap.add_argument("--demo-ramp", action="store_true", help="--demo 에서 세기를 0→1→0 으로 천천히 바꿈 (비례 확인용)")
    ap.add_argument("--check-every", type=int, default=5, help="상태 읽기 간격 (제어 주기 수). 주기 최대가 크면 늘림")
    ap.add_argument("--t-pre", type=float, default=0.7, help="예비 자세까지 걸리는 시간 s (확정이 보통 0.1~0.4 s 라 그보다 길게 → 중간 감속 없음)")
    ap.add_argument("--raw", action="store_true", help="이전 방식 (일정 속도) — 비교용")
    ap.add_argument("--gain", type=float, default=1.0, help="근전도 세기 배율 (쥐기가 덜 닫히면 1.5~2)")
    ap.add_argument("--grips", default=str(HERE / "emg_grips.yaml"))
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--poses", action="store_true")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--sweep", nargs="+", default=None, help="관절 이름들: open 자세에서 그 관절만 비율 0→1 로 훑기 (예: --sweep thumb_cmc thumb_abd)")
    ap.add_argument("--base", default="open", choices=["open", "relax", "preshape", "fist", "pinch", "tripod", "thumb_flex", "index_flex", "middle_flex", "thumb_ext", "index_ext", "middle_ext", "scissors", "cylinder", "wrap", "wrap_pre"], help="--sweep 의 기준 자세")
    ap.add_argument("--sweep-step", type=int, default=2, help="훑기 간격 (×0.1, 기본 0.2)")
    ap.add_argument("--only", choices=["open", "relax", "preshape", "fist", "pinch", "tripod", "thumb_flex", "index_flex", "middle_flex", "thumb_ext", "index_ext", "middle_ext", "scissors", "cylinder", "wrap", "wrap_pre"], default=None, help="--poses 와 함께: 그 자세 하나만 (open → 자세 → open)")
    ARGS = ap.parse_args()
    FIRM_KEYS.clear(); FIRM_KEYS.update(k.strip() for k in ARGS.firm_keys.split(",") if k.strip())
    for f_, m_ in (("ring", ARGS.ring_lag), ("pinky", ARGS.pinky_lag)):
        for jt in ("mcp", "pip"):
            LAG[f"{f_}_{jt}"] = m_
    return run_guarded(run)


ARGS = None
AUTO_KEYS = {"wrap", "pinch", "tripod", "cylinder"}     # 쥐기 모드 자세 = 스스로 쥐기 (동작 모드 fist=바위·가위·가리키기·엄지 척은 세기 비례 그대로)
FIRM_KEYS = {"wrap", "tripod"}
if __name__ == "__main__":
    raise SystemExit(main())
