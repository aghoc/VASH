#!/usr/bin/env python3
"""
grip_rt.py — 쥐기 동작 실시간 디코더 (사람마다 학습 → 실시간/재생 실행 → UDP 로 서보 쪽에 전달)

학습 (녹화 여러 개 합쳐서, RAMP 녹화가 있으면 세기 비례 보정까지):
  python grip_rt.py train data\\S02\\B1.csv data\\S02\\B2.csv data\\S02\\CH.csv data\\S02\\RAMP.csv --out data\\S02\\grip_model.pkl
실행:
  python grip_rt.py run --model data\\S02\\grip_model.pkl --port COM6 [--udp 127.0.0.1:9000]
  python grip_rt.py run --model data\\S02\\grip_model.pkl --replay data\\S02\\B3.csv [--speed 1]     ← 녹화로 시험

구조 (rt_bench.py 의 D3 와 같은 계산, 50 ms 마다):
  ① 필터 20–450 Hz + 60 Hz 계열 노치 (인과적, 상태 유지)
  ② 200 ms 창 → TD + 리만 특징 → LDA 확률 P (온도 보정)          ← '무엇을'
  ③ 100 ms 창 세기 z = ½·log(채널 평균 (RMS/휴식 RMS)²)           ← '언제'
  ④ 켜짐: 평활 P(휴식) < 0.3 AND z > 직전 휴식 기준 + 0.3·Δ
     꺼짐: z 가 (켜질 때 기준 ↔ 이번 수축 최고) 의 45 % 아래로 150 ms AND P(휴식) ≥ 0.4,  또는 P(휴식) > 0.6
  ⑤ 쥐기: 새는 누적 log 증거 (시정수 0.5 s), 사후확률 ≥ 0.9 & 0.1 s 경과 또는 0.4 s 경과 시 확정,
     확정 뒤에는 누적 증거 차이 > log 20 일 때만 바꿈. 확정 전 = '잠정' (서보는 공통 예비 자세 권장)
  ⑥ 세기 0~1: RAMP 녹화로 쥐기마다 z → 목표 세기 직선 보정 (없으면 학습 동작의 z 5~95 % 구간)
UDP JSON: {"t", "state": "rest"|"active", "grip": 이름|null, "grip_id", "committed": bool, "level", "p": [...]}
"""
import argparse, json, os, pickle, socket, sys, threading, time, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402
import rt_bench as B  # noqa: E402

ACT = list(range(E.NCH))


# ============================ 학습 ============================
def train(paths, out, classes=None):
    if classes:                                          # 녹화마다 동작 목록이 달라도 이름으로 맞춤 (손가락 녹화 + 쥐기 녹화 함께 학습)
        E.CLASSES = [c.strip() for c in classes.split(",")]
    m = B.Model(paths)                                   # TD+리만 LDA (세기 증강) + 세기 문턱
    _, _, _, meta = E.load_csv(paths[0])
    names = ["휴식"] + ([c.split("=", 1)[0] for c in E.CLASSES] if E.CLASSES else meta.get("gestures", "").replace("_", " ").split("|"))
    # 세기 보정: RAMP 녹화(target 열)가 있으면 쥐기마다 z → 목표 세기 최소제곱 직선
    lv = {}
    metas = {p: E.load_csv(p)[3] for p in paths}
    has_bend = any(mt.get("ramp") == "bend" for mt in metas.values())
    acc = {c: ([], []) for c in range(1, m.k)}            # 녹화 여러 개를 모아서 한 번에 직선 맞춤
    for p in paths:
        mt = metas[p]
        if "ramp" not in mt: continue
        if has_bend and mt.get("ramp") != "bend": continue   # 굽힘 녹화가 있으면 세기 = '굽힘 정도' 로 보정 (힘 막대 녹화 제외)
        d = np.loadtxt(p, delimiter=",", skiprows=2, encoding="utf-8"); tgt = d[:, -1]
        y, lab, _, _ = B.filt(p); z = B.zfast(y, int(0.1 * E.FS), m.rms_rest)
        peak = float(mt.get("peak", 0.7))
        for c in range(1, m.k):
            s = (lab == c) & (tgt > 0.05)
            acc[c][0].append(z[s]); acc[c][1].append(tgt[s] / peak)   # 1.0 = 막대 최고 높이 (굽힘 모드는 100 % 굽힘)
    for c, (zz, tt) in acc.items():
        if not zz: continue
        zz, tt = np.concatenate(zz), np.concatenate(tt)
        if len(zz) < 500: continue
        a1, a0 = np.polyfit(zz, tt, 1)
        lv[c] = (float(a1), float(a0), float(np.corrcoef(zz, tt)[0, 1]))
    missing = [c for c in range(1, m.k) if c not in lv]
    if missing:
        lv.update(fallback_levels(m, paths, missing))
    firm = firm_thresholds(m, paths)
    pickle.dump({"model": m, "names": names, "level": lv, "firm": firm, "classes": E.CLASSES, "files": [os.path.basename(p) for p in paths]}, open(out, "wb"))
    print(f"저장 {out}   동작 {names[1:]}   학습 {len(paths)}개 녹화")
    for c, (th, acc, nl, nf) in firm.items():
        print(f"  꽉/살살 문턱 {names[c]:14s}: z {th:.2f}   학습 정답 {acc*100:.0f} %  (살살 {nl}, 꽉 {nf} 번)")
    print(f"세기 문턱 z: 휴식 {m.z_mu:.2f}, 동작 {m.z_act:.2f} → 켜짐 기준 +{0.3*(m.z_act-m.z_mu):.2f}")
    for c, (a1, a0, r) in lv.items():
        print(f"  세기 보정 {names[c]:14s}: level = {a1:.3f}·z {a0:+.3f}   (RAMP 상관 r = {r:.2f})")


def firm_thresholds(m, paths):
    """[살살]·[꽉] 이 붙은 녹화로 동작별 '꽉' 문턱 (10-31): 유지 구간(시작 0.8 s·끝 0.3 s 제외) z 중앙값 → 살살/꽉 을 가장 잘 가르는 z
    → {동작 번호: (문턱, 학습 정답률, 살살 수, 꽉 수)}. 실시간에선 확정 뒤 평활 z 가 문턱 + 0.1 을 넘으면 '꽉', 문턱 − 0.15 아래면 '살살'"""
    keep = E.CLASSES; want = {E.class_key(alt): i + 1 for i, c in enumerate(keep) for alt in (c.split("=", 1)[1] if "=" in c else c).split("|")} if keep else None
    seg = {}
    for p in paths:
        y, lab, _, meta = B.filt(p)
        raw = ["휴식"] + E.file_classes(meta)
        E.CLASSES = None; rl = E.load_csv(p)[1]; E.CLASSES = keep
        z = B.zfast(y, int(0.1 * E.FS), m.rms_rest)
        for s_, t_, l_ in E.segments(rl):
            if l_ <= 0 or t_ - s_ < E.FS: continue
            nm = raw[l_]
            if "[" not in nm or "→" in nm: continue
            c = want.get(E.class_key(nm)) if want else l_
            if not c: continue
            f = 1 if "꽉" in nm else 0 if "살살" in nm else None
            if f is None: continue
            seg.setdefault(c, ([], []))[f].append(float(np.median(z[s_ + int(0.8 * E.FS):t_ - int(0.3 * E.FS)])))
    out = {}
    for c, (L, F) in seg.items():
        if len(L) < 3 or len(F) < 3: continue
        L, F = np.array(L), np.array(F); cand = np.linspace(min(L.min(), F.min()), max(L.max(), F.max()), 200)
        acc = np.array([(np.mean(L < th) + np.mean(F >= th)) / 2 for th in cand])
        best = cand[acc >= acc.max() - 1e-9]; th = float(np.median(best))          # 정답이 같은 문턱들의 가운데 (여유 최대)
        out[c] = (th, float(acc.max()), len(L), len(F))
    return out


def fallback_levels(m, paths, classes):
    """굽힘·RAMP 녹화가 없는 동작의 세기 눈금 (v6, 10-30 영상: 평소 힘으로도 가위·엄지·검지가 30 % 만 쥐어져 '간만 보는' 느낌)
    유지 구간 가운데(시작 0.6 s·끝 0.3 s 제외)의 z 로: 하위 10 % → 0.40 (손 쪽 --lv-lo, 안 쥠), 중앙값 → 0.70 (손 쪽 --lv-hi 0.65 위 = 끝까지)
    → 평소 힘이면 끝까지 쥐고, 약하게 주면 그만큼만"""
    zs = {c: [] for c in classes}
    for p in paths:
        y, lab, _, _ = B.filt(p); z = B.zfast(y, int(0.1 * E.FS), m.rms_rest)
        for s_, t_, l_ in E.segments(lab):
            if l_ in zs and t_ - s_ > E.FS:
                zs[l_].append(z[s_ + int(0.6 * E.FS):t_ - int(0.3 * E.FS)])
    out = {}
    for c, parts in zs.items():
        if not parts: continue
        zc = np.concatenate(parts)
        if len(zc) < 100: continue
        q10, q50 = np.percentile(zc, [10, 50]); sc = (0.70 - 0.40) / max(q50 - q10, 1e-3)
        out[c] = (float(sc), float(0.40 - q10 * sc), np.nan)
    return out


# ============================ 실시간 디코더 ============================
class GripDecoder:
    """rt_bench.dec_hybrid 와 같은 계산을 판단 한 번씩 (상태 유지)"""
    def __init__(self, pack, step=E.STEP, tau_p=0.025, p_on=0.3, p_off=0.6, gate=0.3, rel=0.45, tau_b=1.0, tau_h=0.3,
                 n_low=3, p_agree=0.4, tau_L=0.5, commit_p=0.9, min_s=0.25, commit_s=0.4, switch_llr=np.log(20),
                 hold_after=0.6, rel_hold=0.3, p_off_hold=0.9, rel_lock=-1.0, refractory=0.3, re_gate=0.6,
                 slow_names=("엄지", "가위"), slow_min_s=0.35, slow_switch_s=1.0):   # 10-31: 가위도 느린 동작 (손 펴기 중 가위로 튀는 것 방지, 재생 정답·오작동 변화 없음)
        self.m = pack["model"]; self.names = pack["names"]; self.lv = pack["level"]
        self.firm_th = pack.get("firm", {}); self.firm = False             # 꽉/살살 (10-31): 동작별 문턱, 히스테리시스 +0.1 / −0.15
        from scipy import signal
        self.sos = E.full_sos(); self.zi = None; self._sig = signal
        self.buf = np.zeros((E.WIN, E.NCH)); self.since = 0; self.step = step
        dt = step / E.FS
        self.a = 1 - np.exp(-dt / tau_p); self.ab = 1 - np.exp(-dt / tau_b); self.ah = 1 - np.exp(-dt / tau_h); self.lam = np.exp(-dt / tau_L)
        self.dz = self.m.z_act - self.m.z_mu; self.dt = dt
        self.p_on, self.p_off, self.gate, self.rel, self.n_low, self.p_agree = p_on, p_off, gate, rel, n_low, p_agree
        self.commit_p, self.min_s, self.commit_s, self.switch_llr = commit_p, min_s, commit_s, switch_llr
        # 10-31: min_s 0.1 → 0.25 — 켜진 직후 다른 동작으로 잘못 확정됐다 바뀌는 것 22/53 → 10/53 (OS1·GE_FWD·GR2 재생), 맞게 확정까지 중앙 0.20 → 0.25 s, 정답 52/53 그대로
        # 유지 (사람처럼): 쥐기 확정 후 hold_after s 가 지나면 '적은 노력'으로도 유지 — 꺼짐 기준을 낮춤
        #   꺼짐: 세기가 시작 기준 + rel_hold·(최고−시작) 아래 (기본 rel 0.45 → 0.3), 휴식 확률 p_off_hold (0.6 → 0.75). S02 B3: 풀림 0.74 → 0.93 s, 휴식 오작동 변화 없음
        #   10-31 RELEASE: p_off_hold 0.75 → 0.9 — 살살 유지 중 꺼짐 줄어듦 (9 번 중 '거의 유지' 6 → 7, 끊긴 2 번도 일부 회복), 놓기 지연·휴식 오작동 변화 없음 (GR2 1.9 %, ARM2 1.1 %)
        self.hold_after, self.rel_hold, self.p_off_hold = hold_after, rel_hold, p_off_hold; self.t_commit = 0
        # 풀림 잠금 (10-30 v4): 확정 뒤 세기가 최고의 rel_lock 아래로 내려가면 '힘 푸는 중' → 다른 동작으로 안 바뀜 (풀릴 때의 약한 신호가 엄지로 읽히던 문제)
        #   꺼진 뒤 refractory s 안에는 세기가 re_gate 만큼 확실히 올라야 다시 켜짐 (풀림 끝자락이 새 동작으로 켜지는 것 방지)
        self.rel_lock, self.refractory, self.re_gate = rel_lock, refractory, re_gate
        self.locked = False; self.t_off = -10**9
        # 느린 동작 (기본 '엄지'): 신호가 약해서 힘이 오르는 중·풀리는 중에 잘못 읽히기 쉬움 (10-30 실측: 끼어듦 대부분이 시작 0.1 s 확정 / 다른 동작에서 전환)
        #   → 새로 켜진 뒤 slow_min_s 이상 이어져야 확정, 그 전엔 의수로 안 보냄, 다른 동작에서 이쪽으로 전환 금지
        self.slow = {k for k, n in enumerate(self.names) if k > 0 and E.class_key(n) in {E.class_key(x) for x in slow_names}}
        self.slow_min_s, self.slow_switch_s = slow_min_s, slow_switch_s
        self.pr = None; self.zb = None; self.on = False; self.L = None; self.i = 0; self.t_on = 0; self.cls = 0
        self.zh = self.zpk = self.zb_on = 0.0; self.low = 0; self.level = 0.0

    def push(self, x):
        """x: (n, 8) µV 원신호 → 이번에 나온 판단 목록 [(샘플 위치, 결과 dict)]"""
        if self.zi is None:
            self.zi = self._sig.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = self._sig.sosfilt(self.sos, x, axis=0, zi=self.zi)
        out = []; i = 0
        while i < len(y):
            take = min(self.step - self.since, len(y) - i)
            self.buf = np.roll(self.buf, -take, 0); self.buf[-take:] = y[i:i + take]
            self.since += take; i += take
            if self.since >= self.step:
                self.since = 0; out.append((i - 1, self.decide(self.buf.copy())))
        return out

    def decide(self, w):
        m = self.m
        if np.sqrt(np.mean(w ** 2, 0)).max() > E.ART_UV:          # 전극 들뜸·케이블 잡음 → 손 움직이지 않음 (휴식으로 내보내고 상태 초기화)
            self.on = False; self.cls = 0; self.locked = False; self.t_off = self.i; self.level = 0.0; self.i += 1
            return {"state": "rest", "grip_id": 0, "grip": None, "committed": False, "level": 0.0, "p": None, "z": 0.0,
                    "post": None, "names": None, "artifact": True}
        Fn = (m.m.feat(w[None]) - m.m.fmu) / m.m.fsd
        p = M.softmax_t(Fn @ m.m.W.T + m.m.b, m.m.tau)[0]
        ms = np.mean(w[-int(0.1 * E.FS):] ** 2, 0)
        z = float(0.5 * np.log(np.mean(ms / m.rms_rest ** 2) + 1e-9))
        if self.pr is None: self.pr, self.zb = p.copy(), z
        self.pr += self.a * (p - self.pr); self.i += 1
        committed = False
        if not self.on:
            gate = self.re_gate if (self.i - self.t_off) * self.dt < self.refractory else self.gate
            if self.pr[0] < self.p_on and z > self.zb + gate * self.dz:
                self.on, self.L, self.t_on, self.cls = True, np.zeros(len(p) - 1), self.i, 0
                self.zh = self.zpk = z; self.zb_on = self.zb; self.low = 0; self.locked = False
            else:
                self.zb += self.ab * (z - self.zb)
        if self.on:
            self.zh += self.ah * (z - self.zh); self.zpk = max(self.zpk, self.zh)
            if self.cls != 0:
                frac = (self.zh - self.zb_on) / max(1e-6, self.zpk - self.zb_on)
                if frac < self.rel_lock: self.locked = True
                elif frac > 0.85: self.locked = False                  # 다시 세게 주면 풀림 해제 (새 동작으로 바꿀 수 있음)
            hold = self.cls != 0 and (self.i - self.t_commit) * self.dt >= self.hold_after
            rel, p_off = (self.rel_hold, self.p_off_hold) if hold else (self.rel, self.p_off)
            self.low = self.low + 1 if z < self.zb_on + rel * (self.zpk - self.zb_on) else 0
            if (self.low >= self.n_low and self.pr[0] >= self.p_agree) or self.pr[0] > p_off:
                self.on = False; self.zb = z; self.cls = 0; self.locked = False; self.t_off = self.i
        grip = 0
        if self.on:
            lp = np.log(np.maximum(p[1:], 1e-9)); lp -= lp.max()
            self.L = self.lam * self.L + lp
            post = np.exp(self.L - self.L.max()); post /= post.sum(); b = int(post.argmax()) + 1
            el = (self.i - self.t_on) * self.dt
            if self.cls == 0:
                need = self.slow_min_s if b in self.slow else self.min_s
                if (post.max() >= self.commit_p and el >= need) or el >= max(self.commit_s, need): self.cls = b; self.t_commit = self.i
                grip = b
            else:
                slow_ok = b not in self.slow or el < self.slow_switch_s          # 엄지로 바꾸기는 켜진 뒤 slow_switch_s 안에서만 (시작 때 잘못 잡은 것 바로잡기)
                if not self.locked and b != self.cls and slow_ok and self.L[b - 1] - self.L[self.cls - 1] > self.switch_llr:
                    self.cls = b; self.t_commit = self.i
                grip = self.cls
            committed = self.cls != 0
        if grip:
            a1, a0, _ = self.lv.get(grip, (1.0, 0.0, 0))
            tgt = float(np.clip(a1 * z + a0, 0, 1.2))
        else:
            tgt = 0.0
        self.level += 0.4866 * (tgt - self.level)          # 2.12 Hz 1차 평활 (CTRL_FC 와 같음)
        th = self.firm_th.get(grip) if (grip and self.cls) else None
        if th is None:
            self.firm = False
        elif self.firm:
            self.firm = self.zh >= th[0] - 0.15
        else:
            self.firm = self.zh > th[0] + 0.1 and (self.i - self.t_commit) * self.dt >= 0.2
        post_out = [round(float(v), 3) for v in post] if self.on else None      # 동작별 누적 확률 (손가락 섞기용, names[1:] 순서)
        if self.on and self.locked and self.cls:                                # 풀리는 중: 의수 쪽 섞기도 확정 동작 하나로 고정
            post_out = [1.0 if k == self.cls - 1 else 0.0 for k in range(len(post_out))]
        if self.on and self.slow and self.cls not in self.slow:                  # 느린 동작은 확정 전·다른 동작 중에는 의수 섞기에서 뺌
            for k in self.slow: post_out[k - 1] = 0.0
            tot = sum(post_out) or 1.0; post_out = [round(v / tot, 3) for v in post_out]
        return {"state": "active" if self.on else "rest", "grip_id": grip, "grip": self.names[grip] if grip else None,
                "committed": committed, "level": round(self.level, 3), "firm": bool(self.firm), "p": [round(float(v), 3) for v in p], "z": round(z, 2),
                "post": post_out, "names": self.names[1:] if self.i % 20 == 1 else None}


# ============================ 실행 ============================
def run(a):
    # 모드: --model a.pkl,b.pkl 처럼 여러 개면 모드 전환 (예: 쥐기 모드 ↔ 동작 모드). 모든 모델이 같은 신호를 계속 처리(기준값 유지)하고 현재 모드 결과만 보냄
    #   전환: 터미널에서 빈 Enter, 또는 (--switch open2) '짧게 손 펴기' 두 번
    #   v2 (10-31): 이전 '손 펴기 2 s 안에 두 번' 은 쥐었다 놓을 때의 펴는 근육 반동(GR2: 손 펴기 확정 15 번 중 11 번이 반동, 길이 ≈ 0.5 s)
    #   + 실제 펴기가 겹쳐 쓰는 중에 모드가 바뀜 → 조건 강화:
    #     ① 다른 동작이 확정됐던 뒤 1 s 안의 손 펴기는 안 셈 (반동 제외)  ② 한 번의 펴기는 0.1~1.0 s 짧게 (오래 펴고 있는 건 그냥 펴기)
    #     ③ 첫 펴기 끝 → 둘째 펴기 시작 0.2~1.5 s  ④ 바뀐 뒤 3 s 는 다시 안 바뀜
    paths = a.model.split(",")
    packs = [pickle.load(open(p_, "rb")) for p_ in paths]; decs = [GripDecoder(pk) for pk in packs]
    mode = {"i": 0, "flip": False, "prev_open": False, "t_start": 0.0, "valid": False, "last_end": None, "last_grip": -9.0, "lock": -9.0}
    mode_names = [os.path.splitext(os.path.basename(p_))[0] for p_ in paths]
    pack, dec = packs[0], decs[0]
    names = pack["names"]
    if len(packs) > 1:
        def keys():
            while True:
                try: line = sys.stdin.readline()
                except Exception: return
                if line == "": return                        # 입력 없음(EOF) — 전환 안 함
                if line.strip() == "":                       # 빈 Enter 만 (명령을 붙여넣다 실수로 바뀌지 않게)
                    mode["flip"] = True
        threading.Thread(target=keys, daemon=True).start()
        print(f"모드 {len(packs)}개: " + " / ".join(f"{n} {pk['names'][1:]}" for n, pk in zip(mode_names, packs)) + "  — 전환: 빈 Enter" + (" 또는 짧게(1 s 안) 손 펴기 → 힘 빼기 → 다시 짧게 손 펴기" if a.switch == "open2" else ""))
    udp = None
    if a.udp:
        h, pt = a.udp.split(":"); udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (h, int(pt)))
    q = collections.deque(); lock = threading.Lock(); stop = threading.Event(); truth = None; s = None
    if a.replay:
        x_all, truth, _, _ = E.load_csv(a.replay)
        def feeder():
            t0 = time.perf_counter(); k = 0
            while not stop.is_set() and k < len(x_all):
                due = min(len(x_all), int((time.perf_counter() - t0) * E.FS * a.speed))
                if due > k:
                    with lock: q.append((x_all[k:due], k, time.perf_counter()))
                    k = due
                time.sleep(0.002)
            stop.set()
    else:
        from ads_monitor import open_serial, read_line, FrameParser, auto_port
        port = a.port or auto_port(); s = open_serial(port, a.baud)
        s.write(b"X"); time.sleep(0.3); s.reset_input_buffer(); s.write(b"KFF"); time.sleep(0.2); s.write(b"L")
        hdr = None; t0 = time.time()
        while time.time() - t0 < 6:
            line = read_line(s, 1)
            if line and line.startswith("#LIVE"): hdr = dict(kv.split("=") for kv in line.split(",")[1:]); break
        if not hdr: raise SystemExit("스트리밍 시작 응답 없음 — selftest 먼저")
        uv = float(hdr["vref"]) / float(hdr["gain"]) / 2 ** 23 * 1e6; parser = FrameParser()
        def feeder():
            k = 0
            while not stop.is_set():
                d = s.read(max(4096, s.in_waiting))
                if d:
                    parser.feed(d); c, _ = parser.take()
                    if len(c):
                        with lock: q.append((c * uv, k, time.perf_counter()))
                        k += len(c)
    threading.Thread(target=feeder, daemon=True).start()
    lat, n_ok, n_tot, last_print = [], 0, 0, 0.0
    print(f"모델 {os.path.basename(a.model)} (학습 {pack['files']})   동작 {names[1:]}   Ctrl+C 로 종료")
    try:
        while not (stop.is_set() and not q):
            with lock: chunks = list(q); q.clear()
            if not chunks: time.sleep(0.001); continue
            for x, k0, t_arr in chunks:
                outs = [d_.push(x) for d_ in decs]
                for idx, r in outs[mode["i"]]:
                    if len(packs) > 1:
                        is_open = bool(r["committed"] and r["grip"] and E.class_key(r["grip"]) == "손펴기")
                        tnow = (k0 + idx) / E.FS
                        if a.switch == "open2":
                            if r["committed"] and r["grip"] and not is_open: mode["last_grip"] = tnow
                            if is_open and not mode["prev_open"]:                       # 펴기 시작
                                mode["t_start"] = tnow
                                mode["valid"] = tnow - mode["last_grip"] > 1.0 and tnow > mode["lock"]
                                le = mode["last_end"]
                                mode["second"] = le is not None and 0.2 <= tnow - le <= 1.5
                            if mode["prev_open"] and not is_open:                       # 펴기 끝
                                dur = tnow - mode["t_start"]
                                if mode["valid"] and 0.1 <= dur <= 1.0:
                                    if mode.get("second"):
                                        mode["flip"] = True; mode["last_end"] = None; mode["lock"] = tnow + 3.0
                                    else:
                                        mode["last_end"] = tnow
                                else:
                                    mode["last_end"] = None
                        mode["prev_open"] = is_open
                        if mode["flip"]:
                            mode["flip"] = False; mode["i"] = (mode["i"] + 1) % len(packs); mode["lock"] = tnow + 3.0
                            pack, names = packs[mode["i"]], packs[mode["i"]]["names"]
                            sys.stdout.write(f"\n>>> 모드 전환: {mode_names[mode['i']]}  {names[1:]}\n")
                            r = {"state": "rest", "grip_id": 0, "grip": None, "committed": False, "level": 0.0, "p": None, "z": r.get("z", 0.0),
                                 "post": None, "names": names[1:]}
                        r["mode"] = mode_names[mode["i"]]
                        if r.get("names") is not None or r.get("post") is not None:
                            r["names"] = names[1:]
                    now = time.perf_counter(); lat.append((now - t_arr) * 1000)
                    r["t"] = round((k0 + idx) / E.FS, 3)
                    if udp: udp[0].sendto(json.dumps(r, ensure_ascii=False).encode(), udp[1])
                    if truth is not None:
                        tr = int(truth[min(len(truth) - 1, k0 + idx)]); n_tot += 1; n_ok += int(tr == r["grip_id"])
                    if now - last_print > 0.1:
                        last_print = now
                        g = (r["grip"] or "—") + ("" if r["committed"] or not r["grip"] else " (잠정)") + (" [꽉]" if r.get("firm") else "")
                        bar = "█" * int(20 * min(1, r["level"]))
                        warn = "  ※ 잡음 큼 — 전극·케이블 확인" if r.get("artifact") else ""
                        sys.stdout.write(f"\r{r['t']:7.1f}s  {r['state']:6s}  {g:22s} 세기 {r['level']:.2f} {bar:20s}  z {r['z']:5.2f}{warn}   ")
                        sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); print()
        if s is not None:
            from ads_monitor import stop_and_report
            stop_and_report(s)
        if lat: print(f"처리 지연 (샘플 도착 → 판단) 중앙값 {np.median(lat):.1f} ms, 95 % {np.percentile(lat, 95):.1f} ms")
        if truth is not None and n_tot: print(f"재생 일치율 (지시 라벨 기준, 전환·반응 지연 포함) {100*n_ok/n_tot:.1f} %")


def main():
    ap = argparse.ArgumentParser(description="쥐기 동작 실시간 디코더")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("csv", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--classes", default=None, help='통합 동작 목록 (녹화마다 다를 때). 예: "손 펴기,주먹,집기 (중지 펴기),세손가락 (중지 굽힘),엄지,검지,중지"')
    p = sub.add_parser("run"); p.add_argument("--model", required=True); p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=921600); p.add_argument("--replay", default=None)
    p.add_argument("--speed", type=float, default=1.0); p.add_argument("--udp", default=None)
    p.add_argument("--switch", choices=["open2", "enter"], default="open2", help="모드 전환: open2 = 빈 Enter 또는 짧게 손 펴기 두 번, enter = 빈 Enter 만")
    a = ap.parse_args()
    if a.cmd == "train": train(a.csv, a.out, a.classes)
    else: run(a)


if __name__ == "__main__":
    main()
