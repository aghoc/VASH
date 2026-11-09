"""세손가락 집기: 손끝이 바닥을 쓸며 닫힐 때 — 평면 2 관절(검지 MCP·PIP, ORCA 는 DIP 고정) 모의 계산.
모든 치수·마찰·모터 상수는 추정값. 실제 손 측정 아님.

모델
- 관절 속도 = (모터 토크 − 마찰 − 외력 토크)/b  (준정적, 관성 무시). 단위는 'mA 상당 토크'
- 모터 전류 = clip(kp·(명령−실제), ±cap)  (current-based position 모드 흉내)
- 허공에서 300 °/s 일 때 약 100 mA (10-31 측정 '허공 ≈ 100 mA' 에 맞춤)
- 손끝 = 반지름 7 mm 원, 바닥(y=0)·휴대폰 옆면(x=x_edge, 높이 8 mm) 과 벌칙 접촉 + 쿨롱 마찰
- 명령 = 브리지와 같은 흐름: relax → (c 0.5) tripod_pre(f 0.6) → tripod (+squeeze 0.12), auto_close 0.45 s,
  3 차 임계감쇠 필터 ω 24, 닿음 판단 50 Hz (ContactHold 와 같은 조건)
"""
import math, sys, itertools

DEG = math.pi / 180
# ---- 추정 치수 ----
L1, L2, RT = 46.0, 48.0, 7.0          # mm: 첫 마디, (가운데+끝 마디, DIP 고정), 손끝 반지름
KT = 1.15e-3 * 1.09                    # N·m / mA (관절 기준): XC330-T288 0.92 N·m / 0.80 A (11.1 V) × 관절/모터 비 1.09
R_SPOOL = 5.0                          # mm (추정): 장력 = 모터 토크 / 스풀 반지름
# ---- 모터·관절 (mA 상당) ----
KP = 25.0      # mA/°
FRIC = 40.0    # 줄 마찰 mA
BV = 0.20      # mA/(°/s)
VMAX = 400.0
# ---- 자세 (이 컴퓨터 taught_poses 기준, °) ----
SW_MCP, SW_PIP = 0.0, 35.0     # 쓸기 경로: MCP 는 그대로(바닥을 살살 누름), PIP 를 더 굽혀 손끝을 끌어당김
FLAT_MCP, FLAT_PIP = 45.0, 95.0
RELAX = {"mcp": 12.1, "pip": 27.3}
TRIPOD = {"mcp": 70.3, "pip": 17.9}


def tip(q1, q2, th, H):
    a1 = (th + q1) * DEG; a2 = (th + q1 + q2) * DEG
    x = L1 * math.cos(a1) + L2 * math.cos(a2)
    y = H - L1 * math.sin(a1) - L2 * math.sin(a2)
    J = [[-L1 * math.sin(a1) - L2 * math.sin(a2), -L2 * math.sin(a2)],
         [-L1 * math.cos(a1) - L2 * math.cos(a2), -L2 * math.cos(a2)]]   # d(x,y)/d(q rad)
    return x, y, J


def command(t, path="normal"):
    """auto_close 0.45 s 동안 c 0→1, approach (0.5, 0.6), squeeze 0.12"""
    c = min(1.0, t / 0.45)
    pre = {k: RELAX[k] + 0.6 * (TRIPOD[k] - RELAX[k]) for k in RELAX}
    out = {}
    for k in RELAX:
        if c < 0.5:
            v = RELAX[k] + (c / 0.5) * (pre[k] - RELAX[k])
        else:
            v = pre[k] + (c - 0.5) / 0.5 * (TRIPOD[k] - pre[k])
        out[k] = v + 0.12 * (TRIPOD[k] - RELAX[k]) * (1.0 if c > 0 else 0)
    if path == "flat":    # 평면 물체 전용: MCP 는 조금만, PIP 를 크게 굽혀 손끝(손톱)을 바닥 따라 끌어당김
        A = {"mcp": FLAT_MCP, "pip": FLAT_PIP}
        return {k: RELAX[k] + c * (A[k] - RELAX[k]) for k in RELAX}
    if path == "sweep":   # 쓸기 경로: 바닥 구간에서 PIP 를 더 굽혀 손끝을 손바닥 쪽으로 끌어당김 (MCP 는 덜)
        s = max(0.0, min(1.0, (c - 0.4) / 0.6))
        out["mcp"] -= SW_MCP * s
        out["pip"] += SW_PIP * s
    return out


class Filt:   # 브리지 Smoother 와 같은 3 차 임계감쇠 (ω 24), 속도 상한 350 °/s
    def __init__(self, q):
        self.r = dict(q); self.x = dict(q); self.v = {k: 0.0 for k in q}

    def step(self, g, dt, w=24.0):
        k1 = 1 - math.exp(-w * dt)
        for k in g:
            self.r[k] += (g[k] - self.r[k]) * k1
            a = w * w * (self.r[k] - self.x[k]) - 2 * w * self.v[k]
            self.v[k] += a * dt
        s = min([1.0] + [350.0 / abs(v) for v in self.v.values() if abs(v) > 350])
        for k in g:
            self.v[k] *= s; self.x[k] += self.v[k] * dt
        return self.x


def run(policy, mu=0.6, slide=15.0, th=35.0, cap=200.0, path="normal", T=1.6, verbose=False, D=0.0):
    """policy: base | prog (진행 멈춤 추가) | split (MCP 순응 + PIP 진행 멈춤) | none (닿음 고정 끔)"""
    q = dict(RELAX)
    # 바닥 높이: 이 명령 흐름에서 c≈0.55 일 때 손끝이 바닥에 닿게
    qc55 = command(0.55 * 0.45, path)
    _, y55, _ = tip(qc55["mcp"], qc55["pip"], th, 0.0)
    H = -y55 + RT                            # 손끝 중심 높이 RT 에서 바닥 닿음
    x55, _, _ = tip(qc55["mcp"], qc55["pip"], th, H)
    x_edge = x55 - slide - RT                # 닿은 곳에서 slide mm 손바닥 쪽에 휴대폰 옆면
    f = Filt(q)
    dt = 2e-4; t = 0.0; nxt = 0.0
    lock = {}; base = {}; cnt = {"mcp": 0, "pip": 0}
    caps = {"mcp": cap, "pip": cap}
    if policy == "split":
        caps["mcp"] = 130.0                  # MCP 는 살살 (바닥 누르는 힘 상한), PIP 는 그대로
    vtip = (0.0, 0.0); res = {"floor_t": None, "wall_t": None, "lock_x": {}, "lock_where": {}, "Fwall_max": 0.0,
                             "Nfloor_max": 0.0, "imax": 0.0, "slid": 0.0}
    x0_floor = None; qd = {"mcp": 0.0, "pip": 0.0}; hist_v = []
    cur = {"mcp": 0.0, "pip": 0.0}
    while t < T:
        g = command(t, path)
        x_cmd = f.step(g, dt)
        cmd = {k: (lock[k] if k in lock else x_cmd[k]) for k in q}
        # 접촉 힘
        Ht = H - D * min(1.0, t / 0.6)            # 사람이 팔로 손을 D mm 내려 누름 (0.6 s 동안)
        x, y, J = tip(q["mcp"], q["pip"], th, Ht)
        Fx = Fy = 0.0
        pen = RT - y
        on_floor = pen > 0
        if on_floor:
            N = 4.0 * pen                     # N/mm (손끝 패드 강성, 추정)
            vt = vtip[0]
            Fy += N; Fx += -mu * N * math.tanh(vt / 3.0)
            res["Nfloor_max"] = max(res["Nfloor_max"], N)
            if res["floor_t"] is None: res["floor_t"] = t; x0_floor = x
        # 휴대폰 = 상자 [x_edge-70, x_edge] x [0, 8] mm, 손끝 원과의 최근접점
        bx = min(max(x, x_edge - 70.0), x_edge); by = min(max(y, 0.0), 8.0)
        dx, dy = x - bx, y - by; dist = math.hypot(dx, dy)
        penw = RT - dist
        res["_top"] = res.get("_top", False)
        if penw > 0 and dist > 1e-9:
            nx, ny = dx / dist, dy / dist
            Fw = 4.0 * penw
            Fx += Fw * nx; Fy += Fw * ny
            if ny > 0.7: res["_top"] = True          # 폰 윗면을 누름 (밑으로 못 들어감)
            else:
                res["Fwall_max"] = max(res["Fwall_max"], Fw)
                if res["wall_t"] is None and Fw > 0.2: res["wall_t"] = t
            vt2 = vtip[0] * (-ny) + vtip[1] * nx       # 접선 마찰 (폰 옆면)
            Fx += -mu * Fw * math.tanh(vt2 / 3.0) * (-ny); Fy += -mu * Fw * math.tanh(vt2 / 3.0) * nx
        # 외력 → 관절 토크 (N·mm → N·m → mA)
        tau_ext = {k: (J[0][i] * Fx + J[1][i] * Fy) * 1e-3 / KT for i, k in enumerate(("mcp", "pip"))}
        new = {}
        for k in q:
            i_m = max(-caps[k], min(caps[k], KP * (cmd[k] - q[k])))
            cur[k] = i_m
            net = i_m + tau_ext[k]
            if abs(net) <= FRIC and abs(qd[k]) < 1.0:
                v = 0.0
            else:
                v = (net - FRIC * math.copysign(1, net if abs(qd[k]) < 1.0 else qd[k])) / BV
            v = max(-VMAX, min(VMAX, v))
            new[k] = v
        # 손끝이 바닥 아래로 깊이 못 들어가게 (강성 있는 바닥) — 작은 dt 로 충분
        qd = new
        xo, yo = x, y
        for k in q: q[k] += qd[k] * dt
        x2, y2, _ = tip(q["mcp"], q["pip"], th, Ht)
        vtip = ((x2 - xo) / dt, (y2 - yo) / dt)
        res["imax"] = max(res["imax"], abs(cur["mcp"]), abs(cur["pip"]))
        # 닿음 판단 50 Hz
        if t >= nxt:
            nxt += 0.02
            hist_v.append(dict(qd))
            for k in q:
                if policy == "none" or k in lock:
                    continue
                if policy == "split" and k == "mcp":
                    continue                  # MCP 는 고정 안 함 (전류 상한이 누르는 힘을 정함)
                lag = x_cmd[k] - q[k]
                ok = lag > 6.0 and abs(cur[k]) >= 160.0 * (caps[k] / cap if policy == "split" else 1.0)
                if policy in ("prog", "split"):
                    ok = lag > 6.0 and abs(cur[k]) >= min(160.0, 0.8 * caps[k]) and abs(qd[k]) < 15.0
                cnt[k] = cnt[k] + 1 if ok else 0
                if cnt[k] >= (3 if policy == "base" else 5):
                    base[k] = q[k]; lock[k] = q[k] + 6.0
                    f.x[k] = f.r[k] = lock[k]; f.v[k] = 0.0
                    res["lock_x"][k] = round(t, 3)
                    res["lock_where"][k] = "폰" if penw > -1.0 else ("바닥" if on_floor else "허공")
        if x0_floor is not None:
            res["slid"] = max(res["slid"], x0_floor - x2)
        t += dt
    bx = min(max(x2, x_edge - 70.0), x_edge); by = min(max(y2, 0.0), 8.0)
    d2 = math.hypot(x2 - bx, y2 - by)
    side = (y2 - by) / max(d2, 1e-9) < 0.7
    res["x_left"] = round(d2 - RT, 1)                 # 끝났을 때 폰까지 남은 거리 (≤0 = 닿음)
    res["Fwall_end"] = round(max(0.0, 4.0 * (RT - d2)) if side else 0.0, 2)
    res["top"] = res.get("_top", False)
    res["i_end"] = {k: round(abs(cur[k])) for k in cur}
    res["T_N"] = round(res["imax"] * KT / (R_SPOOL * 1e-3) / 1.09, 1)   # 모터 토크/스풀 반지름 (추정)
    res["Nfloor_end"] = round(max(0.0, 4.0 * (RT - y2)), 2)
    return res


if __name__ == "__main__":
    D_ = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    rows = []
    for policy, path in (("base", "normal"), ("none", "normal"), ("split", "normal"), ("base", "flat"), ("prog", "flat"), ("split", "flat")):
        for mu in (0.3, 0.6, 0.9):
            for slide in (5.0, 15.0, 25.0):
                r = run(policy, mu=mu, slide=slide, path=path, D=D_)
                ok = r["x_left"] <= 0 and r["Fwall_end"] >= 1.0
                rows.append((policy, path, mu, slide, ok, r))
                print(f"{policy:5s} {path:6s} mu {mu:.1f} 쓸기 {slide:4.0f} mm | 성공 {'O' if ok else 'X'} | 남은 {r['x_left']:6.1f} mm "
                      f"옆면 힘 {r['Fwall_end']:5.2f} N 바닥 힘 끝 {r['Nfloor_end']:5.2f} N (최대 {r['Nfloor_max']:.1f}) | 고정 {r['lock_x']} {r['lock_where']} "
                      f"| 쓸린 {r['slid']:.1f} mm | 윗면 {'예' if r['top'] else '-'} | 끝 전류 {r['i_end']} 최대 {r['imax']:.0f} mA ≈ 장력 {r['T_N']} N")
