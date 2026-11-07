#!/usr/bin/env python3
"""
grip_model.py — 7 동작 고정 모델 (11-02). 학습·착용 보정·밴드 위치 검사·합격 시험. 실행은 그대로 grip_rt.py run

  python grip_model.py build --out data\\S01\\grip7_base.pkl          ← 기본 모델 (자주 만들지 않음. 만들기 전에 gate 통과)
  python grip_model.py calib data\\S01\\CAL_오늘.csv --out data\\S01\\grip7_오늘.pkl   ← 밴드 찰 때마다 (위치 검사 포함)
  python grip_model.py gate [--accept]                               ← 합격 시험: 착용 하나씩 빼고 학습 → 보정 → 재생 채점
  python grip_model.py report data\\S01\\TEST.csv data\\S01\\CHAIN.csv --model data\\S01\\grip7_오늘.pkl   ← 동작별·전환별 표

방식: 착용마다 공분산 기준·특징 원점을 맞춘 뒤(정렬) 모은 통계로 LDA. 새 착용은 보정 창을 ×10 무게로 더해서 LDA 를 다시 계산
재생 비교 근거: data\\S01\\분석_재점검_1102.md (학습 원본 없이 1 초 안에 보정되도록 통계만 저장)
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402
import rt_bench as B  # noqa: E402
import grip_rt as G  # noqa: E402

GESTURES = ["손 펴기", "주먹", "집기 (중지 펴기)", "세손가락 (중지 굽힘)", "가위", "검지 펴기", "엄지 펴기"]   # wrap 은 비전 전용 → 분류 안 함
CAL_WEIGHT = 10                 # 보정 창 무게
BIASED = ("집기 (중지 펴기)", "세손가락 (중지 굽힘)")
PT_BIAS = 4.0                   # 이 동작들의 LDA 문턱 올림 (재생 비교: 2 효과 작음, 8 부터 정답 20/24 이하). 11-03 실시간 RUN5 에서 끼어듦 0, 합격 기준을 이 값으로 갱신 (이전 190/144 → 189/141, 오작동 0.93 → 0.44 %)
SNR_MIN_DB = 7.0               # 보정 녹화 채널 신호 대 잡음 하한. 2 채널 이상 미달이면 경고 (보정 4 개로 정한 잠정값)
HUM_UV = 100.0                 # 필터 전 60 Hz 경고 기준 µV (11-02 밤 보정 635 µV 는 필터 뒤에도 휴식 RMS 대부분이 60 Hz)
K = len(GESTURES) + 1
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "S01")
GATE_FILE = os.path.join(DATA, "grip7_gate.json")
# 착용별 녹화. cal = (녹화, 반복) 보정 재료, test = 합격 시험 시도. RPS2·EXT1·HOLD 는 잡음이 커서 제외 (dataset\qc_report.md)
WEARS = {
    "1030b": dict(train=["RPS_practice", "RPS1", "EXT_practice", "EXT2", "CH", "ARM"], test=["RPS1", "EXT2", "CH"],
                  cal=[("RPS_practice", {0}), ("EXT_practice", {0})]),
    "1031": dict(train=["GR_practice", "GR1", "GR_FWD", "GR2", "GE1", "GE_FWD", "ARM2", "OS1"], test=["GR1", "GR2", "GR_FWD", "GE1", "OS1"],
                 cal=[("GR_practice", {0}), ("GE_FWD", {0})]),
    "1102a": dict(train=["CAL_20261103a", "GRtest_20261103a", "GEtest_20261103a", "ARM3", "ARM4"], test=["GRtest_20261103a", "GEtest_20261103a"],
                  cal=[("CAL_20261103a", None)]),
}
REF_WEARS = ("1031", "1102a")   # 밴드 위치 기준 (재생 정답이 높았던 착용). 1030b 는 약 반 칸 돌아간 착용
_cache = {}


def fpath(name):
    return name if os.path.exists(name) else os.path.join(DATA, name + ".csv")


def load(name):
    if name not in _cache:
        x, lab, rep, meta = E.load_csv(fpath(name))
        y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
        _cache[name] = (x.astype(np.float32), y.astype(np.float32), lab.astype(np.int16), rep.astype(np.int16), meta)
    return _cache[name]


def index(name, reps=None):
    """emg_classifier.build_dataset 과 같은 규칙으로 고른 창 시작점 (창을 복사하지 않아 메모리 절약)"""
    _, y, lab, rep, _ = load(name)
    c = np.cumsum(np.r_[np.zeros((1, E.NCH)), np.asarray(y, float) ** 2], 0)
    I, Y, R = [], [], []
    for s, t, l in E.segments(lab):
        if l < 0: continue
        tr = E.TRIM_HOLD if l else E.TRIM_REST
        a0, a1 = s + int(tr[0] * E.FS), t - int(tr[1] * E.FS)
        if a1 - a0 < E.WIN or a0 < E.FS: continue
        for k in range(a0, a1 - E.WIN + 1, E.STEP):
            if reps is not None and int(rep[k]) not in reps: continue
            if np.sqrt((c[k + E.WIN] - c[k]) / E.WIN).max() > E.ART_UV: continue
            I.append(k); Y.append(l); R.append(int(rep[k]))
    return np.array(I, int), np.array(Y, int), np.array(R, int)


def windows(name, idx):
    return np.asarray(load(name)[1][idx[:, None] + np.arange(E.WIN)], float)


# ---- 특징: emg_classifier.features + emg_math.tangent 와 같은 계산을 묶음으로 ----
def td_batch(W, zc):
    rms = np.sqrt(np.mean(W * W, 1)) + 1e-3
    d = np.diff(W, axis=1); wl = np.mean(np.abs(d), 1) + 1e-3
    s0, s1 = W[:, :-1], W[:, 1:]
    zc_ = np.mean((s0 * s1 < 0) & (np.abs(s0 - s1) > zc), 1)
    d0, d1 = d[:, :-1], d[:, 1:]
    ssc = np.mean((d0 * d1 < 0) & ((np.abs(d0) > zc) | (np.abs(d1) > zc)), 1)
    return np.hstack([np.log(rms), np.log(wl), zc_, ssc])


def cov_batch(W, shrink=1e-2):
    C = np.einsum("nij,nik->njk", W, W) / W.shape[1]
    return (1 - shrink) * C + shrink * (np.trace(C, axis1=1, axis2=2) / W.shape[2])[:, None, None] * np.eye(W.shape[2])


def eigfn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(np.maximum(w, 1e-12))[:, None, :]) @ np.swapaxes(V, 1, 2)


def raw_feats(W, zc, Rref):
    w, V = np.linalg.eigh(Rref); P = (V * w ** -0.5) @ V.T
    L = eigfn(P @ cov_batch(W) @ P, np.log)
    iu = np.triu_indices(W.shape[2]); coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return np.hstack([td_batch(W, zc), L[:, iu[0], iu[1]] * coef])


def wear_ref(W, Y, zc):
    """착용 기준: 클래스마다 같은 무게로 공분산 로그-유클리드 평균 + 특징 평균 (동작 비율이 달라도 기준이 안 흔들림)"""
    cs = cov_batch(W); present = np.unique(Y)
    w, V = np.linalg.eigh(np.mean([eigfn(cs[Y == c], np.log).mean(0) for c in present], 0))
    R = (V * np.exp(w)) @ V.T
    F = raw_feats(W, zc, R)
    return R, np.mean([F[Y == c].mean(0) for c in present], 0)


def cal_windows(spec):
    Ws, Ys = [], []
    for n, reps in spec:
        i, y, r = index(n)
        if reps is not None:
            m = np.isin(r, list(reps)) | ((y == 0) & (r <= max(reps)))     # 그 반복까지의 휴식(첫 휴식 포함)
            i, y = i[m], y[m]
        Ws.append(windows(n, i)); Ys.append(y)
    return np.concatenate(Ws), np.concatenate(Ys)


# ---- LDA 통계 ----
def lda_from_stats(st, idx, shrink=None):
    """클래스별 개수·평균·산포 → W, b (emg_classifier.fit_lda 와 같은 식). idx = 이 클래스들로만"""
    shrink = E.SHRINK if shrink is None else shrink
    n = st["n"][idx].sum()
    S = st["scatter"][idx].sum(0) / (n - 1)
    S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    W = st["mu"][idx] @ np.linalg.inv(S)
    return W, -0.5 * np.einsum("ij,ij->i", W, st["mu"][idx])


def add_stats(st, Fn, Y, weight):
    """통계 복사본에 창 Fn 을 weight 배로 더함"""
    out = {k: v.copy() for k, v in st.items()}
    for c in range(K):
        F = Fn[Y == c]
        if not len(F): continue
        n1, mu1 = st["n"][c], st["mu"][c]; n2, mu2 = weight * len(F), F.mean(0)
        d = mu2 - mu1; Z = F - mu2
        out["scatter"][c] = st["scatter"][c] + weight * (Z.T @ Z) + np.outer(d, d) * (n1 * n2 / (n1 + n2))
        out["n"][c] = n1 + n2; out["mu"][c] = (n1 * mu1 + n2 * mu2) / (n1 + n2)
    return out


def expert_idx(names):
    keys = {E.class_key(g) for g in G.EXPERT_DEFAULT}
    return [i for i, n in enumerate(names) if i > 0 and E.class_key(n) in keys]


def apply_stats(pack, st):
    mm = pack["model"].m
    mm.W, mm.b = lda_from_stats(st, np.arange(K))
    ix = expert_idx(pack["names"]); We, be = lda_from_stats(st, np.array(ix))
    # 집기·세손가락은 오판하면 손이 스스로 끝까지 닫아 눈에 띔 → 판단 문턱을 올림 (11-03 재생: 끼어듦 12 → 3, 정답 24 → 22/24, 집기 확정 +112 ms)
    strict = [i for i, n in enumerate(pack["names"]) if n in BIASED]
    mm.b[strict] -= PT_BIAS
    be -= PT_BIAS * np.isin(ix, strict)
    pack["expert"] = {"idx": ix, "W": We, "b": be}


# ---- 학습 ----
def build(wears, with_levels=True):
    E.CLASSES = GESTURES
    rs = []
    for tw in wears:
        for n in WEARS[tw]["train"]:
            i, y, _ = index(n); rs.append(np.asarray(load(n)[1][i[y == 0][::5][:, None] + np.arange(E.WIN)], np.float32).reshape(-1, E.NCH))
    zc = np.concatenate(rs).std(0); del rs
    Fs, Ys, Rs = [], [], []
    rng = np.random.default_rng(0)
    for ti, tw in enumerate(wears):
        R, sh = wear_ref(*cal_windows(WEARS[tw]["cal"]), zc)
        parts = [(n, *index(n)) for n in WEARS[tw]["train"]]
        rest = [(n, i[y == 0]) for n, i, y, _ in parts]
        for fi, (n, i, y, r) in enumerate(parts):
            for a in range(0, len(i), 4000):
                Fs.append((raw_feats(windows(n, i[a:a + 4000]), zc, R) - sh).astype(np.float32)); Ys.append(y[a:a + 4000]); Rs.append(r[a:a + 4000] + 1000 * fi + 100000 * ti)
        for s in E.AUG_SCALES:                 # 세기 증강: s·창 + √(1−s²)·휴식 창 (같은 착용의 휴식)
            for fi, (n, i, y, r) in enumerate(parts):
                act = np.flatnonzero(y > 0)
                for a in range(0, len(act), 4000):
                    A = act[a:a + 4000]; Wa = windows(n, i[A]) * s
                    pick = rng.integers(len(rest), size=len(A)); Wr = np.empty_like(Wa)
                    for f in np.unique(pick):
                        m = pick == f; rn, ri = rest[f]; Wr[m] = windows(rn, ri[rng.integers(len(ri), size=int(m.sum()))])
                    Fs.append((raw_feats(Wa + Wr * np.sqrt(1 - s * s), zc, R) - sh).astype(np.float32)); Ys.append(y[A]); Rs.append(r[A] + 1000 * fi + 100000 * ti)
    F = np.concatenate(Fs).astype(float); Y = np.concatenate(Ys); R_ = np.concatenate(Rs); del Fs
    fmu, fsd = F.mean(0), F.std(0) + 1e-6; Fn = (F - fmu) / fsd; del F
    fo = np.vectorize({r: i % 8 for i, r in enumerate(np.unique(R_))}.get)(R_)      # 온도 보정용 교차검증 (반복 묶음 8 폴드)
    g = np.zeros((len(Y), K))
    for f in range(8):
        te = fo == f
        if te.any():
            W_, b_ = E.fit_lda(Fn[~te], Y[~te], K); g[te] = Fn[te] @ W_.T + b_
    tau = M.fit_temperature(g[::3], Y[::3]); del g
    mu = np.array([Fn[Y == c].mean(0) for c in range(K)]); n = np.bincount(Y, minlength=K).astype(float)
    scatter = np.array([(Fn[Y == c] - mu[c]).T @ (Fn[Y == c] - mu[c]) for c in range(K)])
    st = {"n": n, "mu": mu, "scatter": scatter}
    mm = M.MathModel.__new__(M.MathModel)
    mm.active, mm.use_riemann, mm.alpha, mm.zc_th, mm.shift = list(range(E.NCH)), True, 0.1, zc, 0.0
    mm.Rref, mm.fmu, mm.fsd, mm.tau, mm.k = np.eye(E.NCH), fmu, fsd, tau, K
    m = B.Model.__new__(B.Model); m.m = mm; m.k = K
    names = sum([WEARS[t]["train"] for t in wears], [])
    m.rms_rest, m.z_mu, m.z_sd, m.z_act = force_stats(names)
    pack = {"model": m, "names": ["휴식"] + GESTURES, "level": {}, "firm": {}, "classes": GESTURES, "files": [n + ".csv" for n in names],
            "stats": st, "wears": list(wears), "created": time.strftime("%Y-%m-%d %H:%M")}
    apply_stats(pack, st)
    pack["reference"] = reference({t: [n for n in WEARS[t]["test"]] for t in wears if t in REF_WEARS} or {t: WEARS[t]["test"] for t in wears})
    if with_levels:
        paths = [fpath(n) for n in names]
        pack["level"] = G.fallback_levels(m, paths, list(range(1, K)))
        pack["firm"] = G.firm_thresholds(m, paths)
    return pack


def force_stats(names):
    ys = [load(n) for n in names]
    y = np.concatenate([np.asarray(q[1][2 * E.FS:], float) for q in ys]); lab = np.concatenate([q[2][2 * E.FS:] for q in ys])
    segs = [(s, t, l) for s, t, l in E.segments(lab) if s > E.FS]
    ridx = np.concatenate([np.arange(s + E.FS, t - E.FS // 4) for s, t, l in segs if l == 0 and t - s > 1.5 * E.FS])
    w = int(0.1 * E.FS); c = np.cumsum(np.r_[np.zeros((1, E.NCH)), y ** 2], 0)
    ms = np.zeros_like(y); ms[w:] = (c[w + 1:] - c[1:-w]) / w
    rms_rest = np.sqrt(np.percentile(ms[ridx], 20, 0)); z = B.zfast(y, w, rms_rest)
    zm = np.concatenate([z[s + E.FS // 2:t] for s, t, l in segs if l > 0])
    return rms_rest, float(np.median(z[ridx])), float(z[ridx].std()), float(np.median(zm))


# ---- 밴드 위치 검사 ----
def channel_pattern(names):
    """동작별 채널 세기 (log RMS 중앙값, 동작 7 × 채널 8) + 휴식 RMS"""
    P = {c: [] for c in range(1, K)}; rest = []
    for n in names:
        _, y, lab, _, _ = load(n); y = np.asarray(y, float)
        for s, t, l in E.segments(lab):
            if l > 0 and t - s > E.FS: P[l].append(np.sqrt((y[s + E.FS // 2:t - int(0.15 * E.FS)] ** 2).mean(0)))
            if l == 0 and t - s > 2 * E.FS and s >= E.FS: rest.append(np.sqrt((y[s + int(.75 * E.FS):t - E.FS // 4] ** 2).mean(0)))
    return np.log(np.array([np.median(P[c], 0) if P[c] else np.full(E.NCH, np.nan) for c in range(1, K)])), np.median(rest, 0)


def reference(groups):
    pats = [channel_pattern(v) for v in groups.values()]
    return {"pattern": np.nanmean([p for p, _ in pats], 0), "rest": np.mean([r for _, r in pats], 0)}


def pattern_match(p, ref):
    """채널을 s 칸 돌렸을 때 기준과의 상관 {s: r}"""
    out = {}
    B_ = ref - np.nanmean(ref, 1, keepdims=True)
    for s in range(-2, 3):
        A = np.roll(p, s, axis=1); A = A - np.nanmean(A, 1, keepdims=True)
        m = ~np.isnan(A) & ~np.isnan(B_)
        out[s] = float(np.corrcoef(A[m], B_[m])[0, 1])
    return out


def mains_hum(names):
    """휴식 구간 원신호의 60 Hz 크기 (µV, 채널별 최대)"""
    out = []
    for n in names:
        x, _, lab, _, _ = load(n); x = np.asarray(x, float); m = np.zeros(len(lab), bool)
        for s, t, l in E.segments(lab):
            if l == 0 and s >= 5 * E.FS and t - s > 1.25 * E.FS: m[s + E.FS // 2:t - E.FS // 5] = True
        f, P = signal.welch(x[m] - x[m].mean(0), E.FS, nperseg=E.FS, axis=0)
        out.append(np.sqrt(P[(f >= 58.5) & (f <= 61.5)].sum(0) * (f[1] - f[0])))
    return np.max(out, 0)


def placement_report(names, ref):
    """밴드 위치·접촉 검사 → (일치도, 안내 문장들). 기준값은 착용 3 번으로 정한 잠정값"""
    p, r = channel_pattern(names)
    match = pattern_match(p, ref["pattern"]); r0 = match[0]; best = max(match, key=match.get); msgs = []
    print(f"밴드 위치 일치도 {r0:.2f}  (회전 " + "  ".join(f"{s:+d}:{v:.2f}" for s, v in match.items()) + ")")
    if r0 >= 0.85 and best == 0:
        msgs.append("위치 좋음")
    elif r0 >= 0.8 and best == 0:
        msgs.append("위치 조금 어긋남 — 보정으로 대부분 따라가지만 정확도가 떨어질 수 있음")
    else:
        # roll(+s) 가 잘 맞음 = 지금 CH k 에 기준 CH k+s 자리 무늬 → 밴드가 CH 번호 늘어나는 쪽으로 돌아간 것
        turn = f" (둘레 방향으로 약 {abs(best)} 칸 이내, CH 번호 {'늘어나는' if best > 0 else '줄어드는'} 쪽으로 돌아감)" if best != 0 else ""
        msgs.append(f"위치 많이 어긋남{turn} — 밴드를 다시 차고 보정 다시 권장")
    noisy = [c + 1 for c in range(E.NCH) if r[c] > max(8.0, 3 * ref["rest"][c])]
    hum = mains_hum(names)
    if hum.max() > HUM_UV:                     # 지금까지 정상 착용은 필터 전 60 Hz 가 25 µV 이하
        msgs.append(f"60 Hz 전원 잡음 큼 (필터 전 최대 {hum.max():.0f} µV, 평소 25 µV 이하) — 이 보정은 쓰지 말 것. BIAS 전극 접촉, 노트북 충전기, 의수 전원 확인 뒤 다시 녹화")
    elif noisy:
        msgs.append("휴식 잡음 큼: CH" + ", CH".join(map(str, noisy)) + " — 접촉 확인, 힘 뺀 채로 다시 녹화 (RMS " + ", ".join(f"{r[c - 1]:.0f}" for c in noisy) + " µV)")
    snr = 20 * np.log10(np.exp(np.nanmean(p, 0)) / r)
    print("  채널별 신호 대 잡음 dB: " + " ".join(f"{v:.0f}" for v in snr))
    weak = [c + 1 for c in range(E.NCH) if snr[c] < SNR_MIN_DB]
    if len(weak) >= 2:                         # 11-02 CAL_c (CH6·7·8 약함): 다음 착용 시험에서 18/21, 휴식 오작동 15.5 %. 그 착용의 CAL_d 는 21/21, 0.3 %
        msgs.append("신호 약함: CH" + ", CH".join(map(str, weak)) + " — 이 보정은 쓰지 말 것. 실제로 쓸 힘으로, 전극 접촉 확인 뒤 다시 녹화")
    for m_ in msgs: print(" ", m_)
    return r0, msgs


# ---- 착용 보정 ----
def calibrate(base, spec, fresh=False):
    """fresh=True (11-07): 기본 모델의 동작 통계를 섞지 않고 이 착용 녹화만으로 (특징 계산·판단 규칙은 같음).
    오른팔 첫 착용 20261107a 재생 (CAL 로 학습 → TRANS 채점 49 회): 기본+보정 29/49, 이 착용만 47/49"""
    import copy
    E.CLASSES = GESTURES
    pk = copy.deepcopy({k: v for k, v in base.items()}); mm = pk["model"].m
    W, Y = cal_windows(spec)
    missing = [pk["names"][c] for c in range(K) if (Y == c).sum() < 5]
    if missing:
        raise SystemExit(f"보정 녹화에 동작이 빠졌습니다: {missing}. 7 동작을 다 녹화하세요.")
    mm.Rref, mm.shift = wear_ref(W, Y, mm.zc_th)
    Fn = (raw_feats(W, mm.zc_th, mm.Rref) - mm.shift - mm.fmu) / mm.fsd
    if fresh:
        pk["stats"] = add_stats({k: np.zeros_like(v) for k, v in pk["stats"].items()}, Fn, Y, 1)
        apply_stats(pk, pk["stats"])
    else:
        apply_stats(pk, add_stats(pk["stats"], Fn, Y, CAL_WEIGHT))
    pk["calibrated"] = {"files": [os.path.basename(str(n)) for n, _ in spec], "windows": int(len(Y)), "weight": CAL_WEIGHT,
                        "method": "fresh" if fresh else "calib", "time": time.strftime("%Y-%m-%d %H:%M")}
    return pk


def calibrate_rest(base, y, source="rest", max_uv=E.ART_UV):
    """대기 보정: 필터된 휴식 신호 (샘플 × 채널) 만으로 착용 차이를 옮김. 휴식 평균을 기준 휴식에 맞추고 휴식 통계만 더함.
    재생 비교 (11-05, 3 착용, 정답 합): 동작 보정 93/119, 이 방법 80/119. 밴드 위치 차이는 고치지 못함"""
    import copy
    E.CLASSES = GESTURES
    pk = copy.deepcopy(base); mm = pk["model"].m
    y = np.asarray(y, float); c = np.cumsum(np.r_[np.zeros((1, E.NCH)), y ** 2], 0)
    idx = [k for k in range(0, len(y) - E.WIN + 1, E.STEP) if np.sqrt((c[k + E.WIN] - c[k]) / E.WIN).max() <= max_uv]
    if len(idx) < 50:
        raise ValueError(f"쓸 수 있는 휴식 창이 부족합니다 ({len(idx)} 개)")
    W = y[np.array(idx)[:, None] + np.arange(E.WIN)]
    raw = raw_feats(W, mm.zc_th, mm.Rref)
    mm.shift = raw.mean(0) - pk["stats"]["mu"][0] * mm.fsd - mm.fmu
    Fn = (raw - mm.shift - mm.fmu) / mm.fsd
    apply_stats(pk, add_stats(pk["stats"], Fn, np.zeros(len(W), int), CAL_WEIGHT))
    pk["calibrated"] = {"files": [source], "windows": int(len(W)), "weight": CAL_WEIGHT, "method": "rest", "time": time.strftime("%Y-%m-%d %H:%M")}
    return pk


SVM_TRAIN = os.path.join(DATA, "grip7_svm_train.npz")
SVM_PER_CLASS = 1500                # 클래스별 표본: 재생 비교(11-05, svm_cmp_1105) 와 같은 값


def svm_train_set(base, seed=0):
    """기본 모델과 같은 특징·착용 정렬·세기 증강에서 클래스별로 뽑은 학습 표본 (없을 때 한 번 만듦)"""
    mm = base["model"].m
    if os.path.exists(SVM_TRAIN):
        d = np.load(SVM_TRAIN)
        if str(d["created"]) == base["created"]:
            return d["X"], d["Y"]
    E.CLASSES = GESTURES
    rng = np.random.default_rng(seed); Xs, Ys = [], []
    for tw in WEARS:
        R, sh = wear_ref(*cal_windows(WEARS[tw]["cal"]), mm.zc_th)
        parts = [(n, *index(n)) for n in WEARS[tw]["train"]]
        rest = [(n, i[y == 0]) for n, i, y, _ in parts]
        for n, i, y, _ in parts:
            pick = rng.permutation(len(i))[:1200]
            Xs.append(raw_feats(windows(n, i[pick]), mm.zc_th, R) - sh); Ys.append(y[pick])
            act = np.flatnonzero(y > 0)
            for s in E.AUG_SCALES:
                A = rng.permutation(act)[:300]
                if not len(A): continue
                Wa = windows(n, i[A]) * s; rn, ri = rest[rng.integers(len(rest))]
                Wr = windows(rn, ri[rng.integers(len(ri), size=len(A))])
                Xs.append(raw_feats(Wa + Wr * np.sqrt(1 - s * s), mm.zc_th, R) - sh); Ys.append(y[A])
    X = (np.vstack(Xs) - mm.fmu) / mm.fsd; Y = np.concatenate(Ys)
    keep = np.concatenate([rng.permutation(np.flatnonzero(Y == c))[:SVM_PER_CLASS] for c in range(K)])
    X, Y = X[keep].astype(np.float32), Y[keep]
    np.savez(SVM_TRAIN, X=X, Y=Y, created=base["created"])
    return X, Y


def calibrate_svm(base, spec, fresh=False):
    """LDA 보정과 같은 착용 정렬 + 보정 창 무게로 SVM 학습 (선택 옵션, 재생 비교 svm_cmp_1105). 판단 규칙은 같음"""
    import warnings
    from sklearn.svm import SVC
    pk = calibrate(base, spec, fresh=fresh); mm = pk["model"].m
    X, Y = (np.zeros((0, len(mm.fmu)), np.float32), np.zeros(0, int)) if fresh else svm_train_set(base)   # fresh: 왼팔 등 다른 착용 표본을 안 섞음
    Wc, Yc = cal_windows(spec)
    Xc = (raw_feats(Wc, mm.zc_th, mm.Rref) - mm.shift - mm.fmu) / mm.fsd
    clf = SVC(C=1.09, gamma=1 / X.shape[1], kernel="rbf", probability=True, random_state=0, cache_size=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf.fit(np.vstack([X, Xc]), np.r_[Y, Yc], sample_weight=np.r_[np.ones(len(Y)), np.full(len(Yc), CAL_WEIGHT)])
    pk["svm"] = clf; pk["svm_temp"] = G.SVM_TEMP; pk["expert"] = None; pk["calibrated"]["method"] = "svm_fresh" if fresh else "svm"
    return pk


# ---- 합격 시험 ----
def score(pack, names, dec_kw=None):
    """시도 채점. ok = 시작 0.6 s 뒤~끝 확정 동작 다수결 정답, clean = 처음 확정이 정답이고 시도 내내 안 바뀜, fa = 휴식 안쪽 확정 비율"""
    o = dict(n=0, ok=0, clean=0, wrong_first=0, fa_t=0, fa_n=0, lat=[])
    for name in names:
        x, y, lab, rep, _ = load(name)
        res = G.GripDecoder(pack, **(dec_kw or {})).push(np.asarray(x, float))
        k = np.array([i for i, _ in res]); grip = np.array([r["grip_id"] for _, r in res]); com = np.array([bool(r["committed"]) for _, r in res])
        z5 = B.zfast(np.asarray(y, float), int(0.025 * E.FS), np.sqrt(np.percentile(np.asarray(y[2 * E.FS:], float) ** 2, 20, 0)) + 1e-3)
        tt = B.trial_times(z5, lab)
        for s, e, l in E.segments(lab):
            if l == 0 and e - s >= int(1.5 * E.FS) and s >= E.FS:
                m = (k >= s + E.FS) & (k < e - int(.3 * E.FS)); o["fa_n"] += int(m.sum()); o["fa_t"] += int((m & com & (grip > 0)).sum())
            if l <= 0 or s < E.FS: continue
            g = grip[(k >= s + int(.6 * E.FS)) & (k < e) & com]
            o["n"] += 1; o["ok"] += bool(len(g)) and int(np.bincount(g, minlength=K).argmax()) == l
            g2 = grip[(k >= s - int(.2 * E.FS)) & (k < e + int(.2 * E.FS)) & com & (grip > 0)]
            first = g2[0] if len(g2) else 0
            o["wrong_first"] += first not in (0, l); o["clean"] += bool(first == l and np.all(g2 == l))
            a = [t[0] for t in tt if s - E.FS // 2 <= t[0] < e and t[2] == l]
            if a:
                c = np.flatnonzero((k >= a[0]) & com & (grip == l))
                if len(c): o["lat"].append((k[c[0]] - a[0]) / E.FS * 1000)
    o["lat"] = float(np.median(o["lat"])) if o["lat"] else None
    return o


def segment_results(pack, name, dec_kw=None, reps=None):
    """녹화의 동작 구간별 결과. 연속 동작은 반응·전환이 앞쪽에 있으므로 구간 뒤쪽 절반의 확정 다수결로 채점.
    stray = 이전 동작도 정답도 아닌 동작이 그 구간에서 확정된 횟수"""
    import itertools
    x, _, lab, rep, _ = load(name)
    res = G.GripDecoder(pack, **(dec_kw or {})).push(np.asarray(x, float))
    k = np.array([i for i, _ in res]); g = np.array([r["grip_id"] for _, r in res]); com = np.array([bool(r["committed"]) for _, r in res])
    out = []; prev = 0; fa_t = fa_n = 0
    for s, e, l in E.segments(lab):
        if s < E.FS: continue
        if l == 0 and e - s >= int(1.5 * E.FS):
            m = (k >= s + E.FS) & (k < e - int(.3 * E.FS)); fa_n += int(m.sum()); fa_t += int((m & com & (g > 0)).sum())
        if l <= 0:
            prev = 0; continue
        if reps is None or int(rep[s]) in reps:
            m = (k >= s) & (k < e); gl = g[(k >= s + (e - s) // 2) & (k < e) & com]
            pred = int(np.bincount(gl, minlength=K).argmax()) if len(gl) else 0
            seq = [v for v, _ in itertools.groupby(g[m & com]) if v > 0]
            c = np.flatnonzero(m & com & (g == l))
            out.append(dict(prev=prev, true=l, pred=pred, seq=seq, stray=sum(v not in (l, prev) for v in seq), lat=(k[c[0]] - s) / E.FS * 1000 if len(c) else None))
        prev = l
    return out, fa_t, fa_n


def print_report(pack, names, dec_kw=None):
    """시험 표: 합계·시작/전환·동작별·전환별 7×7 (정답/횟수)"""
    rows, fa_t, fa_n = [], 0, 0
    for n in names:
        r, t, c = segment_results(pack, n, dec_kw); rows += r; fa_t += t; fa_n += c
    N = pack["names"]
    def line(sel):
        ok = sum(r["pred"] == r["true"] for r in sel); lat = [r["lat"] for r in sel if r["lat"] is not None]
        return f"{ok}/{len(sel)} ({100 * ok / max(1, len(sel)):.0f} %), 끼어듦 {sum(r['stray'] for r in sel)}, 지시→정답 확정 {np.median(lat) if lat else 0:.0f} ms"
    print(f"전체 {line(rows)}, 휴식 오작동 {100 * fa_t / max(1, fa_n):.1f} %")
    print(f"  쉬었다 시작: {line([r for r in rows if r['prev'] == 0])}")
    print(f"  바로 전환  : {line([r for r in rows if r['prev'] > 0])}")
    print("동작별 (행=정답, 열=판단)  " + " ".join(f"{n[:3]:>4s}" for n in N))
    for c in range(1, K):
        sel = [r for r in rows if r["true"] == c]
        print(f"  {N[c][:9]:10s}" + f"{sum(r['pred'] == c for r in sel):3d}/{len(sel):<3d}  " + " ".join(f"{sum(r['pred'] == p for r in sel):4d}" for p in range(K)))
    trans = [r for r in rows if r["prev"] > 0]
    if trans:
        seen = {(r["prev"], r["true"]) for r in trans}
        print(f"전환별 정답/횟수 (행=이전, 열=다음). {len(seen)}/{(K - 1) * (K - 2)} 종 시험됨")
        print("  " + " " * 10 + " ".join(f"{N[c][:3]:>5s}" for c in range(1, K)))
        for a_ in range(1, K):
            cells = []
            for b_ in range(1, K):
                sel = [r for r in trans if r["prev"] == a_ and r["true"] == b_]
                cells.append(f"{sum(r['pred'] == b_ for r in sel)}/{len(sel)}" if sel else ("  ·" if a_ != b_ else "  -"))
            print(f"  {N[a_][:9]:10s}" + " ".join(f"{v:>5s}" for v in cells))
    return rows


def summary(rows):
    n = sum(r["n"] for r in rows)
    return dict(n=n, ok=sum(r["ok"] for r in rows), clean=sum(r["clean"] for r in rows), wrong_first=sum(r["wrong_first"] for r in rows),
                fa=round(100 * sum(r["fa_t"] for r in rows) / max(1, sum(r["fa_n"] for r in rows)), 2), lat=round(float(np.mean([r["lat"] for r in rows])), 0))


def gate(accept=False):
    """착용 하나씩 빼고 학습 → 그 착용 보정 → 재생 채점. 저장된 기준보다 나쁘면 불합격"""
    rows = []
    for t in WEARS:
        t0 = time.time(); base = build([x for x in WEARS if x != t], with_levels=False)
        r = score(calibrate(base, WEARS[t]["cal"]), WEARS[t]["test"]); rows.append(r)
        print(f"  {t:6s} 정답 {r['ok']}/{r['n']}  끝까지 오류 없음 {r['clean']}/{r['n']}  첫 확정 틀림 {r['wrong_first']}  휴식 오작동 {100 * r['fa_t'] / max(1, r['fa_n']):.1f} %  확정 {r['lat']:.0f} ms  ({time.time() - t0:.0f} s)", flush=True)
    s = summary(rows); print(f"합계 정답 {s['ok']}/{s['n']}  끝까지 오류 없음 {s['clean']}/{s['n']}  휴식 오작동 {s['fa']} %  확정 {s['lat']:.0f} ms")
    if accept or not os.path.exists(GATE_FILE):
        json.dump(s, open(GATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(f"기준 저장: {GATE_FILE}"); return 0
    ref = json.load(open(GATE_FILE, encoding="utf-8"))
    bad = [f"{k} {s[k]} < 기준 {ref[k]}" for k in ("ok", "clean") if s[k] < ref[k]] + ([f"휴식 오작동 {s['fa']} > 기준 {ref['fa']} + 0.2"] if s["fa"] > ref["fa"] + 0.2 else [])
    print("불합격: " + "; ".join(bad) if bad else f"합격 (기준 정답 {ref['ok']}, 오류 없음 {ref['clean']}, 오작동 {ref['fa']} %)")
    return 1 if bad else 0


def main():
    global CAL_WEIGHT, PT_BIAS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--out", required=True)
    p = sub.add_parser("calib"); p.add_argument("csv", nargs="+"); p.add_argument("--base", default=os.path.join(DATA, "grip7_base.pkl")); p.add_argument("--out", required=True); p.add_argument("--weight", type=int, default=CAL_WEIGHT, help="보정 창 무게 (40 초 보정 = 10, 전환·힘·자세까지 녹화했으면 30)"); p.add_argument("--pt-bias", type=float, default=PT_BIAS, help="집기·세손가락 판단 문턱 올림 (끼어듦이 많을 때 4)"); p.add_argument("--fresh", action="store_true", help="기본 모델 통계를 섞지 않고 이 녹화만으로 (오른팔·처음 차는 위치)"); p.add_argument("--ref", default=None, help="밴드 위치 비교 기준 파일 (오른팔: data/S01/grip7_ref_right.pkl). 없으면 기본 모델의 기준(왼팔)")
    p = sub.add_parser("calib-svm", help="선택 옵션: 같은 보정 녹화로 SVM 모델"); p.add_argument("csv", nargs="+"); p.add_argument("--base", default=os.path.join(DATA, "grip7_base.pkl")); p.add_argument("--out", required=True); p.add_argument("--weight", type=float, default=CAL_WEIGHT); p.add_argument("--pt-bias", type=float, default=PT_BIAS); p.add_argument("--fresh", action="store_true", help="다른 착용 표본을 섞지 않고 이 녹화만으로")
    p = sub.add_parser("gate"); p.add_argument("--accept", action="store_true", help="이번 결과를 새 기준으로 저장")
    p = sub.add_parser("report", help="정답 있는 녹화 채점: 동작별·전환별 표"); p.add_argument("csv", nargs="+"); p.add_argument("--model", required=True); p.add_argument("--min-s", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        pk = build(list(WEARS)); pickle.dump(pk, open(a.out, "wb"))
        print(f"저장 {a.out}  착용 {pk['wears']}  녹화 {len(pk['files'])} 개  동작 {pk['names'][1:]}  τ {pk['model'].m.tau:.2f}")
    elif a.cmd == "calib":
        CAL_WEIGHT, PT_BIAS = a.weight, a.pt_bias
        E.CLASSES = GESTURES
        base = pickle.load(open(a.base, "rb"))
        for c in a.csv:
            d = int(load(c)[4].get("dropped", 0) or 0)
            if d: print(f"※ {os.path.basename(c)} 누락 {d} 샘플")
        ref = pickle.load(open(a.ref, "rb"))["reference"] if a.ref else base["reference"]   # 11-07: 오른팔은 오른팔 착용 기준과 비교
        placement_report(a.csv, ref)
        pk = calibrate(base, [(c, None) for c in a.csv], fresh=a.fresh); pickle.dump(pk, open(a.out, "wb"))
        print(f"보정 저장 {a.out}  (보정 창 {pk['calibrated']['windows']} 개 " + ("· 이 착용만으로 학습)" if a.fresh else f"×{CAL_WEIGHT})"))
    elif a.cmd == "calib-svm":
        CAL_WEIGHT, PT_BIAS = a.weight, a.pt_bias
        E.CLASSES = GESTURES
        t0 = time.time(); pk = calibrate_svm(pickle.load(open(a.base, "rb")), [(c, None) for c in a.csv], fresh=a.fresh)
        tmp = a.out + ".tmp"; pickle.dump(pk, open(tmp, "wb")); os.replace(tmp, a.out)
        print(f"SVM 모델 저장 {a.out}  (보정 창 {pk['calibrated']['windows']} 개 ×{CAL_WEIGHT}, 서포트 벡터 {pk['svm'].n_support_.sum()}, {time.time() - t0:.0f} s)")
    elif a.cmd == "report":
        E.CLASSES = GESTURES
        print_report(pickle.load(open(a.model, "rb")), a.csv, dict(min_s=a.min_s, commit_s=max(0.4, a.min_s + 0.1)) if a.min_s is not None else None)
    else:
        raise SystemExit(gate(a.accept))


if __name__ == "__main__":
    main()
