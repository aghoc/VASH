#!/usr/bin/env python3
"""
emg_math.py — 수학적으로 최적성이 증명된 방법들로 만든 의수 제어 파이프라인 (실험판)

  ① 리만 기하 (SPD 행렬의 아핀 불변 거리)      채널 사이 공분산을 특징으로. 휴식 30 초로 재중심화하면
                                               전극 접촉이 바뀌어 채널 이득이 달라져도 거의 그대로
  ② 온도 보정 (적절 점수 규칙, 1차원 볼록 최적화)  LDA 의 과신한 확률을 실제 맞을 확률로
  ③ 순응 예측 (Conformal, 분포 가정 없는 보장)   "후보 손가락 집합"이 정답을 1−α 이상 포함하도록 보장.
                                               후보가 둘 이상이면 움직이지 않음
  ④ 은닉 마르코프 전방 필터 (베이즈 최적)        다수결 대신 사후확률 재귀. 손가락 유지 시간 사전지식 반영
  ⑤ CUSUM (Page 1954, Lorden·Moustakides 최적성) 정해진 오경보율에서 '힘주기 시작'을 가장 빨리 검출
  ⑥ 최소 저크에 가까운 임계감쇠 3차 평활        (Flash & Hogan 1985 의 최소 저크 궤적은 5차 다항식 점대점 — 여기선 그 모양에 가까운 실시간 추종 필터)

  python emg_math.py eval B2_extensor.csv            두 가지 실험 출력 (분류 특징, 제어 파이프라인)
  python emg_math.py eval B2_extensor.csv --plot m.png
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

DT = E.STEP / E.FS                      # 판단 간격 50 ms


# ============================ ① 리만 기하 ============================
def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(np.maximum(w, 1e-12))) @ V.T


def covs(Xw, shrink=1e-2):
    """창(WIN×n) → 공분산 (수축으로 항상 양의 정부호)"""
    out = []
    for w in Xw:
        C = w.T @ w / len(w)
        out.append((1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C)))
    return np.array(out)


def log_euclid_mean(Cs):
    return _eig_fn(np.mean([_eig_fn(C, np.log) for C in Cs], 0), np.exp)


def tangent(Cs, Rref):
    """재중심화(Rref^-1/2 C Rref^-1/2) 후 행렬 로그 → 위삼각 벡터 (비대각 √2 배: 리만 거리 보존)"""
    P = _eig_fn(Rref, lambda w: w ** -0.5)
    n = Cs.shape[1]; iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return np.array([_eig_fn(P @ C @ P, np.log)[iu] * coef for C in Cs])


# ============================ ② 온도 보정 · ③ 순응 예측 ============================
def lda_logits(F, W, b):
    return F @ W.T + b


def softmax_t(g, tau):
    z = g / tau; z = z - z.max(-1, keepdims=True); p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def fit_temperature(g, y):
    """음의 로그우도 최소 τ (1차원 볼록 → 격자 + 국소 정밀화)"""
    taus = np.exp(np.linspace(np.log(0.5), np.log(200), 120))
    nll = [-np.mean(np.log(softmax_t(g, t)[np.arange(len(y)), y] + 1e-12)) for t in taus]
    return float(taus[int(np.argmin(nll))])


def conformal_q(p_true, alpha):
    """비순응 점수 s = 1 − p(정답). 유한표본 보정 분위수 (Vovk·Lei 등)"""
    s = np.sort(1 - p_true); n = len(s)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(s[k])


# ============================ 모델 (학습 반복만 사용) ============================
class MathModel:
    def __init__(self, Xw, Y, R, active, use_riemann=True, alpha=0.10):
        self.active, self.use_riemann, self.alpha = active, use_riemann, alpha
        rest = np.concatenate([w for w, l in zip(Xw, Y) if l == 0])
        self.zc_th = rest.std(axis=0)
        self.shift = 0.0
        F = self.feat(Xw, fit=True, Y=Y)
        self.rest_fmean = F[Y == 0].mean(0)              # 재보정 기준 (학습 때 휴식 특징 평균)
        self.fmu, self.fsd = F.mean(0), F.std(0) + 1e-6
        Fn = (F - self.fmu) / self.fsd
        # 반복 단위 교차검증 로짓 → 온도·순응 임계값 (학습 데이터 안에서만)
        self.k = k = int(np.max(Y)) + 1                  # 클래스 수 (손가락 4, 쥐기 목록이면 더 많을 수 있음)
        g_cv = np.zeros((len(Y), k))
        for r in sorted(set(R.tolist())):
            te = R == r
            if te.all() or not te.any():
                continue
            W, b = E.fit_lda(Fn[~te], Y[~te], k)
            g_cv[te] = lda_logits(Fn[te], W, b)
        self.tau = fit_temperature(g_cv, Y)
        self.q = conformal_q(softmax_t(g_cv, self.tau)[np.arange(len(Y)), Y], alpha)
        self.W, self.b = E.fit_lda(Fn, Y, k)
        self.prior = np.bincount(Y, minlength=k) / len(Y)
        # 세기·CUSUM 기준
        rms = np.array([np.sqrt(np.mean(w ** 2, 0)) for w in Xw])
        self.rms_rest = np.median(rms[Y == 0], 0)
        self.rms_cls = np.array([np.median(rms[Y == c], 0) for c in range(k)])
        z = np.log(rms / self.rms_rest).mean(1)
        self.mu0, self.sd0 = float(np.median(z[Y == 0])), float(z[Y == 0].std() + 1e-6)
        self.delta = float(np.median(z[Y > 0]) - self.mu0)

    def feat(self, Xw, fit=False, Y=None, rest_ref=None, zc_th=None):
        zc = self.zc_th if zc_th is None else zc_th
        td = np.array([E.features(w, zc) for w in Xw])
        if not self.use_riemann:
            return td - self.shift
        Cs = covs(Xw)
        if fit:
            self.Rref = log_euclid_mean(Cs[Y == 0])
        return np.hstack([td, tangent(Cs, self.Rref if rest_ref is None else rest_ref)]) - self.shift

    def recalibrate(self, rest_windows):
        """재착용 후 휴식 몇 초로 기준만 갱신 (학습은 그대로): 리만 기준 행렬, ZC 문턱, 세기 기준"""
        Xr = np.array(rest_windows)
        self.zc_th = np.concatenate(Xr).std(0)
        if self.use_riemann:
            self.Rref = log_euclid_mean(covs(Xr))
        rms = np.sqrt((Xr ** 2).mean(1))
        old = self.rms_rest.copy(); self.rms_rest = np.median(rms, 0)
        self.rms_cls = self.rms_cls * (self.rms_rest / old)
        # 특징 공간 재중심화: 이득 변화는 log-RMS·log-WL 에 '모든 창에 같은 덧셈'으로 나타나므로 휴식 평균 차이를 빼면 사라짐
        self.shift = 0.0
        self.shift = self.feat(Xr).mean(0) - self.rest_fmean

    def proba(self, Xw, tau=None):
        Fn = (self.feat(Xw) - self.fmu) / self.fsd
        return softmax_t(lda_logits(Fn, self.W, self.b), self.tau if tau is None else tau)


# ============================ ④ HMM · ⑤ CUSUM · ⑥ 3차 평활 궤적 ============================
class MathController:
    def __init__(self, m, hold_s=3.0, rest_s=3.0, switch_p=0.002, arl_s=120.0, omega=18.0,
                 theta_max=90.0, deadzone=0.08, vmax=600.0, off_frac=0.25, start_post=0.9, omega_release=40.0):
        self.m = m
        on, off = DT / rest_s, DT / hold_s
        T = np.full((4, 4), switch_p)
        T[0, 1:] = on / 3; T[1:, 0] = off
        np.fill_diagonal(T, 0); np.fill_diagonal(T, 1 - T.sum(1))
        self.T = T
        self.post = np.array([1.0, 0, 0, 0])
        # CUSUM: 로그우도비 증분 (가우스 평균 변화 μ0 → μ0+δ), 문턱 h ≈ ln(ARL0) (Siegmund 근사, 판단 단위)
        self.k = m.delta / m.sd0 ** 2
        self.h = float(np.log(arl_s / DT))
        self.S_on = self.S_off = 0.0
        self.on = False
        D = np.clip(m.rms_cls / m.rms_rest - 1, 0, None)
        self.D, self.DD = D, np.maximum((D * D).sum(1), 1e-9)
        self.level = np.zeros(4)
        self.active = 0
        self.theta_max, self.dz, self.vmax = theta_max, deadzone, vmax
        self.off_frac, self.start_post = off_frac, start_post
        # 최소 저크에 가까운 추종: x''' = ω³(u − x) − 3ω² x' − 3ω x''  (임계감쇠 3차) → 정확한 이산화 (행렬 지수)
        from scipy.linalg import expm

        def disc(w):
            A = np.array([[0, 1, 0], [0, 0, 1], [-w ** 3, -3 * w ** 2, -3 * w]])
            Bv = np.array([0, 0, w ** 3])
            M = expm(np.block([[A, Bv[:, None]], [np.zeros((1, 4))]]) * DT)
            return M[:3, :3], M[:3, 3]
        self.Ad, self.Bd = disc(omega)                 # 굽힐 때 (부드럽게)
        self.Ad_r, self.Bd_r = disc(omega_release)     # 펼 때 (빠르게 — 놓는 동작은 늦으면 위험)
        self.x = np.zeros((3, 3))           # 손가락 3개 × (각도, 속도, 가속도)

    def update(self, p, rms):
        m = self.m
        # ⑤ CUSUM 양방향
        z = float(np.log(np.maximum(rms, 1e-6) / m.rms_rest).mean())
        l = self.k * (z - m.mu0 - m.delta / 2)
        self.S_on = max(0.0, self.S_on + l); self.S_off = max(0.0, self.S_off - l)
        if not self.on and self.S_on > self.h:
            self.on, self.S_off = True, 0.0
        elif self.on and self.S_off > self.h * self.off_frac:
            self.on, self.S_on = False, 0.0
        # ④ HMM 전방 필터 (방출 = 보정 사후확률 / 사전확률)
        lik = p / np.maximum(m.prior, 1e-6)
        post = (self.post @ self.T) * lik
        self.post = post / post.sum()
        c = int(self.post.argmax())
        # ③ 순응 예측 집합: 1 − p_c ≤ q 인 후보가 하나뿐일 때만 확정
        cand = np.flatnonzero(1 - p <= m.q)
        sure = len(cand) == 1 and cand[0] == c
        # 세기 (휴식 대비 증가분을 동작 패턴에 투영)
        raw = ((rms / m.rms_rest - 1) @ self.D.T) / self.DD
        raw[0] = 0
        self.level += 0.4866 * (np.clip(raw, 0, 1.5) - self.level)
        if not self.on or c == 0:
            self.active = 0
        elif self.active == 0:
            if self.post[c] > self.start_post:
                self.active = c                          # 시작: HMM 사후확률이 충분하면
        elif c != self.active and sure:
            self.active = c                              # 전환: 순응 집합이 그 손가락 하나일 때만
        u = np.zeros(3)
        if self.active:
            a = np.clip((self.level[self.active] - self.dz) / (1 - self.dz), 0, 1)
            u[self.active - 1] = a * self.theta_max
        # ⑥ 3차 평활 추종 + 속도 상한
        for f in range(3):
            Ad, Bd = (self.Ad_r, self.Bd_r) if u[f] < self.x[f, 0] else (self.Ad, self.Bd)
            xn = Ad @ self.x[f] + Bd * u[f]
            if abs(xn[1]) > self.vmax:
                xn[1] = np.sign(xn[1]) * self.vmax
            self.x[f] = xn
        return np.clip(self.x[:, 0], 0, self.theta_max)


# ============================ 평가 ============================
def run_stream(x, active, step_fn, chunk=40):
    """원신호를 실시간처럼 흘리며 50 ms 마다 step_fn(창, 마지막 샘플 번호) 호출"""
    from scipy import signal
    sos = E.full_sos()
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0, active][None, None, :]
    buf = np.zeros((E.WIN, len(active))); since = 0; out = []
    for k0 in range(0, len(x), chunk):
        y, zi = signal.sosfilt(sos, x[k0:k0 + chunk, active], axis=0, zi=zi)
        i = 0
        while i < len(y):
            take = min(E.STEP - since, len(y) - i)
            buf = np.roll(buf, -take, 0); buf[-take:] = y[i:i + take]
            since += take; i += take
            if since >= E.STEP:
                since = 0
                out.append(step_fn(buf.copy(), k0 + i - 1))
    return out


def jerk_rms(ang):
    j = np.diff(ang, n=3, axis=0) / DT ** 3
    return float(np.sqrt(np.mean(j ** 2)))


def cmd_eval(a):
    active = list(range(E.NCH))
    x, lab, rep, _ = E.load_csv(a.csv)
    test = {int(r) for r in a.test_reps.split(",")}
    train = sorted({int(r) for r in rep} - test)
    Xw, Y, R = E.build_dataset([a.csv], active, reps=set(train))
    Xt, Yt, _ = E.build_dataset([a.csv], active, reps=test)
    Xw, Xt = np.array(Xw), np.array(Xt)
    print(f"== {os.path.basename(a.csv)}  학습 반복 {train} / 시험 반복 {sorted(test)}  (학습 창 {len(Y)}, 시험 창 {len(Yt)})")

    # ---------- 실험 A: 특징 + 재착용 모사 (채널 이득 무작위 변화) ----------
    rng = np.random.default_rng(7)
    rest_t = [w for w, l in zip(Xt, Yt) if l == 0]
    print("\n[A] 분류 정확도 (창 단위, 평활 없음).  '이득 변화' = 시험 데이터 채널마다 ×0.6~1.6 (재착용·땀으로 접촉 저항 변화 모사)")
    print(f"    {'방법':34s}{'그대로':>10s}{'이득 변화':>12s}{'+휴식 재보정':>14s}")
    for name, use_r in (("시간영역 32 (현재)", False), ("시간영역 + 리만 공분산 68", True)):
        accs = []
        for gshift in (False, True):
            g = np.exp(rng.uniform(np.log(0.6), np.log(1.6), (20 if gshift else 1, len(active))))
            if not gshift: g[:] = 1
            res = []
            for gi in g:
                mdl = MathModel(Xw, Y, R, active, use_riemann=use_r)
                Xs = Xt * gi
                acc0 = np.mean(mdl.proba(Xs).argmax(1) == Yt)
                mdl.recalibrate([w * gi for w in rest_t])
                acc1 = np.mean(mdl.proba(Xs).argmax(1) == Yt)
                res.append((acc0, acc1))
            accs.append(np.mean(res, 0))
        print(f"    {name:34s}{100*accs[0][0]:9.1f}%{100*accs[1][0]:11.1f}%{100*accs[1][1]:13.1f}%")

    # ---------- 실험 B: 제어 파이프라인 ----------
    mdl = MathModel(Xw, Y, R, active, use_riemann=a.riemann)
    print(f"\n[B] 서보 제어 (재생, 시험 반복만 채점)   온도 τ = {mdl.tau:.1f}, 순응 임계 q = {mdl.q:.3f} (α = {mdl.alpha})")
    eng = E.RealtimeEngine(E.fit_model(np.array([E.features(w, mdl.zc_th) for w in Xw]), Y, active, mdl.zc_th))
    base = E.ServoController(eng)
    mc = MathController(mdl)
    T, A_base, A_math = [], [], []
    for k0 in range(0, len(x), 40):
        for cls, prob, inten, idx, p in eng.push(x[k0:k0 + 40]):
            T.append((k0 + idx) / E.FS)
            A_base.append(base.update(cls, p, eng.last_rms))
            pm = mdl.proba(eng.buf[None])[0]
            A_math.append(mc.update(pm, eng.last_rms))
    T = np.array(T); si = np.clip((T * E.FS).astype(int), 0, len(lab) - 1)
    truth = lab[si]; sel = np.isin(rep[si], list(test))
    segs = [(s0, s1, int(lab[s0])) for s0, s1, _ in E.segments(lab)]
    on = [(s0 / E.FS, l) for s0, s1, l in segs if l and int(rep[s0]) in test]
    off = [s1 / E.FS for s0, s1, l in segs if l and int(rep[s0]) in test]
    res = {}
    for name, A in (("현재 제어기", A_base), ("수학 파이프라인", A_math)):
        A = np.array(A)
        d = E.control_metrics(T[sel], A[sel], truth[sel], on, off)
        d["저크 RMS (°/s³, 낮을수록 매끄러움)"] = jerk_rms(A[sel])
        res[name] = d
    print(f"    {'지표':36s}" + "".join(f"{n:>16s}" for n in res))
    for k in res["현재 제어기"]:
        print(f"    {k:36s}" + "".join(f"{res[n][k]:16.2f}" for n in res))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        ts = T[sel] - T[sel][0]; cols = ["#c62828", "#1565c0", "#6a1b9a"]
        fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
        for f in range(3):
            ax[0].fill_between(ts, 0, (truth[sel] == f + 1) * 1.0, color=cols[f], alpha=0.5, step="mid", label=E.CLASSES[f + 1])
        ax[0].set_yticks([]); ax[0].set_title("정답 (화면 지시)"); ax[0].legend(ncol=3, loc="upper right")
        for axx, (n, A) in zip(ax[1:], (("현재 제어기", A_base), ("수학 파이프라인", A_math))):
            A = np.array(A)[sel]
            for f in range(3): axx.plot(ts, A[:, f], color=cols[f], lw=1.3)
            axx.set_ylim(-5, 100); axx.set_ylabel("각도 (°)"); axx.set_title(n)
        ax[-1].set_xlabel("시간 (s)"); fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"    그림: {a.plot}")
    return res


def main():
    ap = argparse.ArgumentParser(description="수학 기반 의수 제어 파이프라인 (실험판)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval"); p.add_argument("csv"); p.add_argument("--test-reps", default="3,4")
    p.add_argument("--plot", default=None); p.add_argument("--no-riemann", dest="riemann", action="store_false")
    a = ap.parse_args()
    if a.cmd == "eval": cmd_eval(a)


if __name__ == "__main__":
    main()
