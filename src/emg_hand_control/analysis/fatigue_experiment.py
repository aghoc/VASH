#!/usr/bin/env python3
"""
fatigue_experiment.py v2 — 피로 실험(F1) 분석: 대회용 파이프라인 + 랩실용 FC-Riemann 통합 (2026-10-27)

녹화 순서: docs\\피로실험_가이드.md  (밴드 그대로: 피로 전 → 주먹 60 s × 3 → 피로 후 → 10분 → 회복)

  python fatigue_experiment.py --pre data\\F1_pre.csv --post data\\F1_post.csv [--recover data\\F1_recover.csv]
                               [--sustain data\\F1_sustain.csv] [--out data\\fig_fatigue_accuracy.png] [--en]

v1 과 달라진 점 (랩실용 설계서 검토 반영)
  · 모든 적응은 '의사 온라인': 녹화를 시간 순서대로 흘리며 50 ms 마다 판단하고, 그 시점까지 본 데이터로만 적응.
    v1 은 시험 데이터 전체로 적응한 뒤 같은 데이터로 채점해 과대평가였음 (S01 98.0 % → 다시 계산)
  · 확률은 온도 보정 후 값, 후보 판정은 같은 세션 반복 단위로 보정한 순응 예측 임계
  · 피로 지수 φ: 동작별 스펙트럼을 몇 초 누적한 뒤 MDF 를 구해 처음(피로 전) MDF 로 나눔 (창 하나 MDF 의 잡음·편향 회피)
  · 휴식 재보정은 '전극·피부 변화' 보정으로만 해석 (휴식 신호에는 피로가 안 나타남)
  · 다조건 학습(피로 전 + 피로 후 다른 반복)을 강한 비교 대상으로 추가

방법
  M0  정적: 피로 전 녹화로 학습한 LDA (TD 32 + 휴식 재중심화 리만, 세기 증강)
  MR  M0 + 휴식 재보정 (녹화 시작 5 초 휴식, 라벨 불필요)
  M1  M0 + 무제한 자가 적응 (온도 보정 확률 ≥ 0.9, 대회용 v1 방식을 온라인으로)
  M3  M0 + 제한 적응 (순응 후보 1개 창만, 마할라노비스 폭 ≤ b0 + b1(1−φ)) — 랩실용 BoundedAdapter
  M2  피로 조건부 평균 μ_c(φ) = μ_c0 + (1−φ)v_c  (v_c 는 피로 후 '다른 반복' 라벨로 학습 → 반복 단위 교차)
  M2+3  M2 + M3
  MC  다조건 학습: 피로 전 + 피로 후 다른 반복을 섞어 학습 (라벨 필요, 상한 참고)
확인 항목: G0 피로로 5 %p 이상 하락? / G1 같은 동작 MDF 비율 감소, 휴식은 그대로? / φ(라벨 없음) vs φ(라벨) 상관 /
          G2 클래스 평균 이동 중 (1−φ)v_c 가 설명하는 비율 η (반복 단위 교차, 0.5 미만이면 v_c 빼고 M3 만)
"""
import argparse, os, sys
from typing import Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402
import fatigue_poster as FP  # noqa: E402
from fatigue_pipeline import FatigueConditionedLDA, BoundedAdapter  # noqa: E402

ACT = list(range(E.NCH))
NFFT = 256


# ---------------------------------------------------------------- 스펙트럼
def psd(w):
    from scipy import signal
    f, P = signal.welch(w, E.FS, nperseg=NFFT, noverlap=NFFT // 2, axis=0)
    m = (f >= 20) & (f <= 450)
    for h in range(60, 451, 60):
        m &= np.abs(f - h) > 4
    return f[m], P[m]


def mdf_from_psd(f, P):
    c = np.cumsum(P, 0)
    return np.array([np.interp(c[-1, j] / 2, c[:, j], f) for j in range(P.shape[1])])


class SpectralFatigueMonitor:
    """φ = Σ_c w_c Σ_ch E_c,ch · MDF(S_c)_ch / MDF0_c,ch ÷ Σ_c w_c Σ_ch E_c,ch
    S_c: 동작 확률로 가중해 누적(지수 평균)한 동작별 스펙트럼. 휴식 확률 ≤ 0.2 인 창만 동작 쪽 갱신, ≥ 0.8 은 휴식 쪽"""
    def __init__(self, f, ref, rest_ref, lam=0.04):
        self.f, self.ref, self.rest_ref, self.lam = f, ref, rest_ref, lam
        self.S = {c: None for c in ref}; self.wsum = {c: 0.0 for c in ref}
        self.R = None; self.phi = 1.0; self.rest_ratio = 1.0

    def update(self, w, p):
        _, P = psd(w)
        if p[0] >= 0.8:
            self.R = P if self.R is None else (1 - self.lam) * self.R + self.lam * P
            self.rest_ratio = float(np.median(mdf_from_psd(self.f, self.R) / self.rest_ref))
            return self.phi
        if p[0] > 0.2:
            return self.phi
        num = den = 0.0
        for c in self.ref:
            lam = self.lam * p[c]
            if lam <= 0: continue
            self.S[c] = P * p[c] if self.S[c] is None else (1 - lam) * self.S[c] + lam * P
            self.wsum[c] = (1 - self.lam) * self.wsum[c] + self.lam * p[c]
        for c in self.ref:
            Sc = self.S[c]
            if Sc is None or self.wsum[c] < 0.05: continue
            Ech = Sc.sum(0)
            num += self.wsum[c] * np.sum(Ech * mdf_from_psd(self.f, Sc) / self.ref[c])
            den += self.wsum[c] * np.sum(Ech)
        if den > 0:
            self.phi = float(np.clip(num / den, 0.5, 1.1))
        return self.phi


# ---------------------------------------------------------------- 데이터
def stream_windows(path):
    """녹화 전체를 필터 → 50 ms 마다 200 ms 창 (실시간과 같은 순서). 창 끝 시점의 라벨·반복·지시 후 경과 시간"""
    from scipy import signal
    x, lab, rep, _ = E.load_csv(path)
    y = signal.sosfilt(E.full_sos(), x[:, ACT], axis=0)
    ends = np.arange(E.FS + E.WIN, len(x), E.STEP)
    W = np.array([y[e - E.WIN:e] for e in ends])
    L, Rp = lab[ends - 1], rep[ends - 1]
    since = np.zeros(len(ends)); chg = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
    for i, e in enumerate(ends):
        since[i] = (e - 1 - chg[np.searchsorted(chg, e - 1, side="right") - 1]) / E.FS
    score = ((L > 0) & (since >= 0.5)) | ((L == 0) & (since >= 1.0))
    return W, L, Rp, since, score


def per_motion(pred, L, Rp, mask):
    reps = [r for r in sorted(set(Rp[mask].tolist())) if r >= 0]
    return {c: [100 * np.mean(pred[mask & (Rp == r) & (L == c)] == c) for r in reps
                if (mask & (Rp == r) & (L == c)).any()] for c in range(4)}


def mean_acc(d):
    return float(np.mean([np.mean(v) for v in d.values() if len(v)]))


# ---------------------------------------------------------------- 분석
def analyze(pre, post, aug=True) -> dict[str, Any]:
    """post: 파일 하나 또는 목록 (피로 단계 T1, T2, T3 … 순서대로). 목록이면 이어서 하나의 스트림으로 흘리고 단계 번호를 기록"""
    posts = [post] if isinstance(post, str) else list(post)
    pres = [pre] if isinstance(pre, str) else list(pre)
    Xp, Yp, Rp = E.build_dataset(pres, ACT); Xp = np.array(Xp)
    Xa, Ya, Ra = E.augment_force(Xp, Yp, Rp, E.AUG_SCALES if aug else ())
    mm = M.MathModel(Xa, Ya, Ra, ACT, True)                       # 온도 τ, 순응 q 는 같은 세션 반복 단위 교차로
    std = lambda W, m=mm: (m.feat(W) - m.fmu) / m.fsd
    Fa = std(Xa)
    # 피로 전 동작별 기준 스펙트럼 MDF
    f0, _ = psd(Xp[0])
    ref = {c: mdf_from_psd(f0, np.mean([psd(w)[1] for w in Xp[Yp == c][::2]], 0)) for c in (1, 2, 3)}
    rest_ref = mdf_from_psd(f0, np.mean([psd(w)[1] for w in Xp[Yp == 0][::3]], 0))
    # 피로 전: 반복 단위 교차 (기준값)
    pre_acc = {c: [] for c in range(4)}
    for r in sorted(set(Rp.tolist())):
        if r < 0 or r % 1000 == 999: continue
        trm = Ra != r; tem = Rp == r
        mdl = FatigueConditionedLDA().fit(Fa[trm], Ya[trm])
        P = mdl.proba(std(Xp[tem])).argmax(1)
        for c in range(4):
            mc = Yp[tem] == c
            if mc.any(): pre_acc[c].append(100 * np.mean(P[mc] == c))
    # 피로 후 스트림
    parts = [stream_windows(p) for p in posts]
    W = np.concatenate([p_[0] for p_ in parts]); L = np.concatenate([p_[1] for p_ in parts])
    R = np.concatenate([np.where(p_[2] >= 0, p_[2] + 100 * k, -1 - k) for k, p_ in enumerate(parts)])
    since = np.concatenate([p_[3] for p_ in parts]); score = np.concatenate([p_[4] for p_ in parts])
    level = np.concatenate([np.full(len(p_[0]), k) for k, p_ in enumerate(parts)])
    F_base = std(W)
    rest0 = W[(R == -1) & (since >= 1.0)]                          # 첫 시험 녹화 시작 5 초 휴식만
    mr = M.MathModel(Xa, Ya, Ra, ACT, True); mr.recalibrate(list(rest0))
    shift_z = float(np.sqrt(np.mean((np.asarray(mr.shift) / mr.fsd) ** 2)))   # 기준 이동 크기 (표준편차 단위, 평균)
    if shift_z > 1.0:
        # 시험 녹화 시작 5 초가 '진짜 휴식'이 아님 (예: 피로 유발 직후 근육이 덜 풀림) → 재보정하면 전부 어긋남, 건너뜀
        print(f"  휴식 재보정 건너뜀: 시작 5 초 휴식의 특징 이동 {shift_z:.2f} SD (> 1.0) — 근육이 덜 풀린 상태로 판단")
        mr = mm
    F_rest = (mr.feat(W) - mr.fmu) / mr.fsd
    base = FatigueConditionedLDA().fit(Fa, Ya); base.tau = mm.tau
    q = mm.q
    # φ (라벨 없음) — 정적 모델의 온도 보정 확률로 추정
    P0 = base.proba(F_base)
    mon = SpectralFatigueMonitor(f0, ref, rest_ref)
    phi = np.array([mon.update(w, p) for w, p in zip(W, P0)])
    rest_ratio = mon.rest_ratio
    res = {"M0": P0.argmax(1), "MR": base.proba(F_rest).argmax(1)}

    def online(kind, F, model):
        pred = np.zeros(len(F), int)
        ad = BoundedAdapter(model) if kind == "M3" else None
        buf = {c: [] for c in range(4)}
        for i, x in enumerate(F):
            p = model.proba(x, phi[i])[0]; c = int(p.argmax()); pred[i] = c
            if ad is not None:
                single = int(np.sum(p >= 1 - q)) == 1 and p[c] >= 1 - q
                ad.collect(x, c, single)
                if (i + 1) % 40 == 0: ad.step(phi[i])
            else:                                                  # M1: 무제한, 확신 ≥ 0.9
                if p[c] >= 0.9: buf[c].append(x)
                if (i + 1) % 40 == 0:
                    for k in range(4):
                        if len(buf[k]) >= 15:
                            cur = model.means(phi[i])[k]
                            model.delta[k] += 0.5 * (np.mean(buf[k], 0) - cur); buf[k] = []
        return pred

    for kind in ("M1", "M3"):
        mdl = FatigueConditionedLDA().fit(Fa, Ya); mdl.tau = mm.tau
        res[kind] = online(kind, F_base, mdl)
    # 라벨이 필요한 방법: 피로 후 반복 단위 교차 (반복 r 시험, 나머지 반복의 라벨로 v_c 또는 다조건 학습)
    reps = [r for r in sorted(set(R.tolist())) if r >= 0]
    for key in ("M2", "M2+3", "MC"):
        res[key] = np.full(len(W), -1)
    eta_num = eta_den = 0.0
    for r in reps:
        tr = score & (R != r) & (R >= 0); te = R == r
        if tr.any():
            mdl = FatigueConditionedLDA().fit(Fa, Ya); mdl.tau = mm.tau
            mdl.fit_fatigue(F_base[tr], L[tr], phi[tr])
            res["M2"][te] = mdl.proba(F_base[te], float(np.mean(phi[te]))).argmax(1) if te.any() else []
            m23 = FatigueConditionedLDA().fit(Fa, Ya); m23.tau = mm.tau; m23.v = mdl.v.copy()
            res["M2+3"][te] = online("M3", F_base, m23)[te]
            mc = FatigueConditionedLDA().fit(np.vstack([Fa, F_base[tr]]), np.r_[Ya, L[tr]])
            res["MC"][te] = mc.proba(F_base[te]).argmax(1)
            # G2: 반복 r 의 클래스 평균 이동을 다른 반복으로 학습한 (1−φ_r) v_c 가 얼마나 설명하나
            for c in (1, 2, 3):
                mt = score & te & (L == c)
                if mt.sum() < 5: continue
                d = F_base[mt].mean(0) - mdl.mu0[c]
                pred_d = (1 - float(np.mean(phi[mt]))) * mdl.v[c]
                eta_num += np.sum((d - pred_d) ** 2); eta_den += np.sum(d ** 2)
    eta = 1 - eta_num / eta_den if eta_den > 0 else float("nan")
    # 라벨로 계산한 φ (동작 구간별 MDF / 피로 전 같은 동작 MDF, 채널 에너지 가중)
    phi_lab, phi_est = [], []
    for c in (1, 2, 3):
        for r in reps:
            m = score & (L == c) & (R == r)
            if m.sum() < 10: continue
            Pm = np.mean([psd(w)[1] for w in W[m]], 0)
            E_ = Pm.sum(0)
            phi_lab.append(float(np.sum(E_ * mdf_from_psd(f0, Pm) / ref[c]) / np.sum(E_)))
            phi_est.append(float(phi[np.flatnonzero(m)[-1]]))
    # G1: 같은 동작 MDF 비율 (피로 후 전체 / 피로 전)
    g1 = {}
    for c in (1, 2, 3):
        m = score & (L == c)
        Pm = np.mean([psd(w)[1] for w in W[m][::2]], 0); E_ = Pm.sum(0)
        g1[c] = float(np.sum(E_ * mdf_from_psd(f0, Pm) / ref[c]) / np.sum(E_))
    mrest = score & (L == 0)
    g1[0] = float(np.median(mdf_from_psd(f0, np.mean([psd(w)[1] for w in W[mrest][::3]], 0)) / rest_ref))
    acc = {k: per_motion(v, L, R, score & (v >= 0)) for k, v in res.items()}
    levels = {lv: {k: per_motion(v, L, R, score & (v >= 0) & (level == lv)) for k, v in res.items()}
              for lv in range(len(posts))}
    phi_lv = [float(np.mean(phi[(level == lv) & score & (L > 0)])) for lv in range(len(posts))]
    return dict(pre=pre_acc, post=acc, levels=levels, phi_lv=phi_lv, phi=phi, phi_lab=phi_lab, phi_est=phi_est, eta=eta, g1=g1,
                rest_ratio=rest_ratio, tau=mm.tau, q=q, res=res, L=L, R=R, level=level, score=score)


def class_names(path, en=False, override=None):
    """라벨 이름: --names 가 있으면 그것, 아니면 녹화 머리말의 gestures=, 없으면 손가락 기본값"""
    if override:
        return [n.strip() for n in override.split(",")]
    _, _, _, meta = E.load_csv(path)
    g = meta.get("gestures")
    if g:
        return ["Rest" if en else "휴식"] + [n.replace("_", " ") for n in g.split("|")]
    return ["Rest", "Thumb", "Index", "Middle"] if en else ["휴식", "엄지", "검지", "중지"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, nargs="+", help="쉰 상태 기준 녹화 (여러 개 가능)"); ap.add_argument("--post", required=True, nargs="+", help="피로 후 시험 녹화 (여러 개면 T1 T2 T3 순서)")
    ap.add_argument("--recover", default=None); ap.add_argument("--sustain", default=None)
    ap.add_argument("--out", default="data/fig_fatigue_accuracy.png"); ap.add_argument("--en", action="store_true")
    ap.add_argument("--font", default=None); ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--pre-label", default=None, help='기준 녹화 이름 (예: "쉰 상태 (10분 회복 후)")')
    ap.add_argument("--names", default=None, help='라벨 이름 직접 지정 (휴식 포함, 쉼표). 예: "Rest,Open,Fist,Precision"')
    ap.add_argument("--hold-drop", type=float, default=None, help="fatigue_within.py 의 60 초 유지 중 MDF 감소 %% (예: 6.1) — 요약에 표시")
    a = ap.parse_args()
    L_ = (lambda ko, en: en if a.en else ko)
    names = class_names(a.pre[0], a.en, a.names)
    if len(names) != 4:
        sys.exit(f"현재 분석은 휴식 + 동작 3개(라벨 0~3)만 지원합니다. 녹화 동작: {names}")
    order = ["M0", "MR", "M1", "M3", "M2", "M2+3", "MC"]
    label = {"M0": L_("정적", "Static"), "MR": L_("휴식 재보정", "Rest recal."), "M1": L_("무제한 적응", "Unbounded adapt"),
             "M3": L_("제한 적응", "Bounded adapt"), "M2": L_("피로 조건부*", "Fatigue-cond.*"), "M2+3": L_("조건부+제한*", "Cond.+bounded*"),
             "MC": L_("다조건 학습*", "Multi-cond.*")}
    out = analyze(a.pre, a.post, aug=not a.no_aug)
    rec = analyze(a.pre, a.recover, aug=not a.no_aug) if a.recover else None
    pre_m = mean_acc(out["pre"])
    print(f"온도 τ = {out['tau']:.2f}, 순응 임계 q = {out['q']:.3f}")
    print("\n[1] 동작별 정확도 (%)            " + "".join(f"{n:>7s}" for n in names) + "    평균")
    print("    피로 전 (반복 교차)        " + "".join(f"{np.mean(v):7.1f}" for v in out["pre"].values()) + f"  {pre_m:6.1f}")
    for st, o in (("피로 후", out), ("회복 후", rec)):
        if o is None: continue
        for k in order:
            print(f"    {st} {label[k]:14s} " + "".join(f"{np.mean(v):7.1f}" if len(v) else "      -" for v in o["post"][k].values())
                  + f"  {mean_acc(o['post'][k]):6.1f}")
    if len(out["levels"]) > 1:
        print("\n[1b] 피로 단계별 정적 모델 정확도 (%)  " + "".join(f"{n:>7s}" for n in names) + "    평균   φ")
        for lv, d in out["levels"].items():
            print(f"    T{lv+1}                         " + "".join(f"{np.mean(v):7.1f}" if len(v) else "      -" for v in d["M0"].values())
                  + f"  {mean_acc(d['M0']):6.1f}  {out['phi_lv'][lv]:.3f}")
    g0 = pre_m - mean_acc(out["post"]["M0"])
    print(f"\n[G0] 피로로 정확도 하락 {g0:+.1f}%p  → {'통과 (≥ 5 %p)' if g0 >= 5 else '미달 (피로 효과 약함)'}")
    g1 = out["g1"]
    print("[G1] 같은 동작 MDF 비율 (피로 후/전): " + "  ".join(f"{names[c]} {g1[c]:.3f}" for c in (1, 2, 3)) + f"   휴식 {g1[0]:.3f} (1 근처여야 함)")
    if len(out["phi_lab"]) >= 3:
        r = float(np.corrcoef(out["phi_lab"], out["phi_est"])[0, 1])
        print(f"[φ ] 라벨 없는 φ vs 라벨 φ 상관 r = {r:.2f}  (목표 ≥ 0.9),  마지막 φ = {out['phi'][-1]:.3f}")
    print(f"[G2] 피로 방향 모델 설명 비율 η = {out['eta']:.2f}  → {'v_c 유지' if out['eta'] >= 0.5 else 'v_c 빼고 제한 적응만'}")
    print("     * 표시(피로 조건부·조건부+제한·다조건)는 피로 후 다른 반복의 라벨 사용 — 라벨 불필요 방법과 구분해서 볼 것")
    if a.sustain:
        rows = FP.collect([a.sustain])
        print("[피로 유발] 구간별 MDF " + ", ".join(f"{r_['mdf']:.0f}" for r_ in rows))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(10.5, 6.6), dpi=200, facecolor="white")
    nlv = len(out["levels"])
    grays = ["#404040"] if nlv == 1 else [str(0.75 - 0.5 * i / (nlv - 1)) for i in range(nlv)]
    pre_lb = a.pre_label or L_("피로 전", "Rested")
    states = [("pre", pre_lb, "white", out["pre"])] + \
             [(f"T{lv+1}", (L_("피로 후", "Fatigued") if nlv == 1 else f"{L_('피로', 'Fatigue')} T{lv+1}"), grays[lv], out["levels"][lv]["M0"])
              for lv in range(nlv)]
    if rec: states.append(("rec", L_("회복 후", "Recovered"), "#e6e6e6", rec["post"]["M0"]))
    ax = fig.add_axes((0.07, 0.56, 0.60, 0.34)); n = len(states); w = 0.8 / n
    for i, (_, lb, fc, d) in enumerate(states):
        for c in range(4):
            g = d[c]
            if not len(g): continue
            x = c + (i - (n - 1) / 2) * w; m, s = np.mean(g), np.std(g)
            ax.bar(x, m, width=w * 0.95, color=fc, edgecolor="black", lw=0.8, label=lb if c == 0 else None)
            ax.errorbar(x, m, yerr=s, color="black", capsize=2.5, lw=0.8)
            ax.scatter(x + rng.uniform(-w / 5, w / 5, len(g)), g, s=6, color="#777", zorder=3)
            ax.text(x, max(m + s, max(g)) + 2.5, f"{m:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20))
    ax.spines["left"].set_bounds(0, 100); ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(L_("분류 정확도 (%)", "Accuracy (%)")); ax.legend(frameon=False, ncol=n, loc="lower right", bbox_to_anchor=(1, 1))
    ax.text(-0.09, 1.04, "a", transform=ax.transAxes, fontsize=12, weight="bold")
    bx = fig.add_axes((0.76, 0.56, 0.20, 0.34))
    vals = [1.0] + ([np.mean([g1[c] for c in (1, 2, 3)])] if nlv == 1 else list(out["phi_lv"])) + \
           ([np.mean([rec["g1"][c] for c in (1, 2, 3)])] if rec else [])
    for i, (v, (_, lb, fc, _d)) in enumerate(zip(vals, states)):
        bx.bar(i, 100 * v, color=fc, edgecolor="black", lw=0.8, width=0.6); bx.text(i, float(100 * v + 1.5), f"{100*v:.0f}", ha="center", fontsize=8)
    short = [L_("기준", "Base")] + [s[1].split()[-1] for s in states[1:]]   # 긴 이름이 겹치지 않게 짧게
    bx.set_xticks(range(n)); bx.set_xticklabels(short, fontsize=8); bx.set_ylim(0, 115)
    bx.set_ylabel(L_("중앙 주파수 (기준 = 100)", "Median freq. (baseline = 100)")); bx.spines[["right", "top"]].set_visible(False)
    bx.text(-0.35, 1.04, "b", transform=bx.transAxes, fontsize=12, weight="bold")
    cx = fig.add_axes((0.07, 0.10, 0.89, 0.30))
    for i, k in enumerate(order):
        d = out["post"][k]; per_rep = np.mean(np.array([d[c] for c in range(4) if len(d[c])]), 0)
        m, s = per_rep.mean(), per_rep.std()
        cx.bar(i, m, color="white" if k in ("M0", "MC", "M2", "M2+3") else "#404040", edgecolor="black", lw=0.8, width=0.62,
               hatch="///" if k in ("MC", "M2", "M2+3") else None)
        cx.errorbar(i, m, yerr=s, color="black", capsize=3, lw=0.8); cx.text(i, 106, f"{m:.1f}", ha="center", fontsize=8)
    cx.axhline(pre_m, color="black", ls="--", lw=0.8, xmax=(len(order) - 0.5 + 0.6) / (len(order) - 0.5 + 0.55 + 0.6))
    cx.set_xlim(-0.6, len(order) - 0.5 + 0.55)
    cx.text(len(order) - 0.45, pre_m, L_(f"기준\n{pre_m:.0f}%", f"baseline\n{pre_m:.0f}%"), fontsize=7.2, ha="left", va="center")
    cx.set_xticks(range(len(order))); cx.set_xticklabels([label[k].rstrip("*") for k in order], fontsize=8.5)
    cx.set_ylim(0, 118); cx.set_yticks(range(0, 101, 20)); cx.spines["left"].set_bounds(0, 100)
    cx.spines[["right", "top"]].set_visible(False); cx.set_ylabel(L_("피로 후 정확도 (%)", "Accuracy when fatigued (%)"))
    cx.text(-0.05, 1.04, "c", transform=cx.transAxes, fontsize=12, weight="bold")
    from matplotlib.patches import Patch
    cx.legend(handles=[Patch(fc="white", ec="black", lw=0.8, label=L_("보정 없음", "No compensation")),
                       Patch(fc="#404040", ec="black", lw=0.8, label=L_("라벨 없이 보정", "Label-free")),
                       Patch(fc="white", ec="black", lw=0.8, hatch="///", label=L_("라벨 사용 보정", "Label-informed"))],
              frameon=False, ncol=3, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
