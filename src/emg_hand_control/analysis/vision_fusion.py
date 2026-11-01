#!/usr/bin/env python3
"""
vision_fusion.py — 근전도의 한계 조건에서 비전 융합이 '쥐기 선택 오류'를 얼마나 줄이는지 (논문형, 흑백)

  python vision_fusion.py [--out data\\fig_vision_fusion.png] [--vis-acc 0.8] [--en] [--font 경로] [--recompute]

융합 규칙 (의수 3층 설계와 같음, 재학습 없음):
  근전도 → '언제'   : P_emg(휴식) < 0.5 이면 동작 중
  비전   → '무엇을' : 동작 중이면 쥐기 = argmax_c P_emg(c) · q_vis(c)
  근전도만 = 같은 규칙에 q_vis 균일 (공정 비교)
비전 모의 (가정 값이 하나로 고정되지 않도록 정확도를 훑음):
  동작 시도(반복 × 동작)마다 비전이 정확도 a 로 맞힘. 맞으면 q[정답] = a, 틀리면 무작위 다른 쥐기에 a,
  나머지 (1 − a)/(K − 1). 몬테카를로 200 회 평균. a = 1/K 는 정보 없음 = 근전도만.
지표:
  쥐기 선택 오류 = 근전도가 '동작 중'으로 맞게 잡은 창 가운데 다른 동작으로 고른 비율 (비전이 맡는 부분)
  동작 감지 오류 = 휴식 ↔ 동작 착각 창 비율 (근전도가 맡는 부분, 비전 융합으로는 안 바뀜)
근전도 모델: fatigue_experiment 의 M0 와 같음 (TD 32 + 휴식 재중심화 리만, 세기 증강, 수축 LDA, 온도 보정)
"""
import argparse, os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402

ACT = list(range(E.NCH))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NMC = 200


class Model:
    def __init__(self, Xw, Y, R, aug=True):
        Xa, Ya, Ra = E.augment_force(np.asarray(Xw), Y, R, E.AUG_SCALES if aug else ())
        self.m = M.MathModel(Xa, Ya, Ra, ACT, True)

    def proba(self, X):
        m = self.m
        Fn = (m.feat(np.asarray(X)) - m.fmu) / m.fsd
        return M.softmax_t(Fn @ m.W.T + m.b, m.tau)


def load(paths):
    X, Y, R = E.build_dataset(paths, ACT)
    return np.asarray(X), Y, R


def cv_proba(X, Y, R, keep=None):
    P = np.full((len(Y), int(Y.max()) + 1), np.nan)
    for r in sorted(set(R.tolist())):
        if r < 0 or (keep is not None and not keep(r)): continue
        te = R == r
        P[te] = Model(X[~te], Y[~te], R[~te]).proba(X[te])
    m = ~np.isnan(P[:, 0])
    return P[m], Y[m], R[m]


def build_conditions():
    C = {}
    Xb, Yb, Rb = load([f"{D}/F2_B1.csv", f"{D}/F2_B2.csv"])
    C["rest_grip"] = cv_proba(Xb, Yb, Rb)
    mb = Model(Xb, Yb, Rb)
    for t in ("T1", "T2", "T3"):
        X, Y, R = load([f"{D}/F2_{t}.csv"]); C[f"F2_{t}"] = (mb.proba(X), Y, R)
    Xf, Yf, Rf = load([f"{D}/F1_recover.csv", f"{D}/F1_rest2.csv"])
    C["rest_finger"] = cv_proba(Xf, Yf, Rf)
    mf = Model(Xf, Yf, Rf)
    for t in ("T1", "T2", "T3"):
        X, Y, R = load([f"{D}/F1_{t}.csv"]); C[f"F1_{t}"] = (mf.proba(X), Y, R)
    Xg, Yg, Rg = load([f"{D}/20261024_G1.csv"])          # 홀수 반복 = 세게, 짝수 = 약하게 (force=mix_even_weak)
    C["G1_strong"] = cv_proba(Xg, Yg, Rg, keep=lambda r: r % 2 == 1)
    C["G1_weak"] = cv_proba(Xg, Yg, Rg, keep=lambda r: r % 2 == 0)
    return C


# ---------------------------------------------------------------- 융합과 지표
def decide(P, q):
    return np.where(P[:, 0] < 0.5, 1 + np.argmax(P[:, 1:] * q, 1), 0)


def vision_q(Y, R, k, a, rng):
    m = k - 1; q = np.full((len(Y), m), 1.0 / m)
    if a <= 1.0 / m + 1e-9:
        return q
    for r, c in sorted(set(zip(R.tolist(), Y.tolist()))):
        if c == 0: continue
        top = c - 1 if rng.random() < a else rng.choice([j for j in range(m) if j != c - 1])
        row = np.full(m, (1 - a) / (m - 1)); row[top] = a
        q[(R == r) & (Y == c)] = row
    return q


def metrics(P, Y, R, a, seed=0):
    """(쥐기 선택 오류 %, 반복별 SD, 동작 감지 오류 %) — 몬테카를로 평균"""
    k = P.shape[1]; rng = np.random.default_rng(seed)
    n = NMC if 1.0 / (k - 1) + 1e-9 < a < 1 else 1
    reps = sorted(set(R.tolist())); det = (P[:, 0] < 0.5) & (Y > 0)      # 동작 중으로 맞게 잡은 창 (비전과 무관)
    sel, per = 0.0, np.zeros(len(reps))
    for _ in range(n):
        wrong = decide(P, vision_q(Y, R, k, a, rng)) != Y
        sel += 100 * wrong[det].mean()
        per += [100 * wrong[det & (R == r)].mean() if (det & (R == r)).any() else np.nan for r in reps]
    detect = 100 * np.mean((P[:, 0] < 0.5) != (Y > 0))
    return sel / n, float(np.nanstd(per / n)), detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/fig_vision_fusion.png"); ap.add_argument("--en", action="store_true")
    ap.add_argument("--font", default=None); ap.add_argument("--vis-acc", type=float, default=0.8)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--hide-rate", action="store_true", help="범례에서 인식률 숫자를 빼고 b 의 점선에 'a 조건' 표시")
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    cache = f"{D}/vision_fusion_cache.pkl"
    if os.path.exists(cache) and not a.recompute:
        C = pickle.load(open(cache, "rb"))
    else:
        C = build_conditions(); pickle.dump(C, open(cache, "wb"))
    va = a.vis_acc
    show = [("rest_grip", L_("쉰 상태\n(F2 기준)", "Rested\n(F2 base)")), ("F2_T2", L_("피로\n(F2 T2)", "Fatigue\n(F2 T2)")),
            ("G1_strong", L_("비슷한 쥐기\n(G1)", "Similar grips\n(G1)")), ("G1_weak", L_("약한 힘\n(G1)", "Weak force\n(G1)"))]
    print(f"비전 정확도 가정 a = {va:.2f}  (몬테카를로 {NMC}회)")
    print(f"{'조건':16s} {'창':>5s} | 쥐기 선택 오류: 근전도만 → +비전 (감소) | 동작 감지 오류")
    R_ = {}
    for key in C:
        P, Y, R = C[key]
        s0, d0, det = metrics(P, Y, R, 1.0 / (P.shape[1] - 1)); s1, d1, _ = metrics(P, Y, R, va)
        R_[key] = (s0, d0, s1, d1, det)
        print(f"{key:16s} {len(Y):5d} | {s0:5.1f} ± {d0:4.1f} → {s1:5.1f} ± {d1:4.1f}  ({100*(1-s1/s0) if s0 else 0:4.0f}%) | {det:5.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(10.5, 4.0), dpi=200, facecolor="white")

    # (a) 조건별 쥐기 선택 오류
    ax = fig.add_axes((0.065, 0.17, 0.40, 0.70)); w = 0.36
    top = max(R_[k][0] + R_[k][1] for k, _ in show) * 1.3
    for i, (key, _) in enumerate(show):
        s0, d0, s1, d1, _ = R_[key]
        ax.bar(i - w / 2, s0, w, color="white", edgecolor="black", lw=0.8)
        ax.bar(i + w / 2, s1, w, color="#303030", edgecolor="black", lw=0.8)
        ax.errorbar([i - w / 2, i + w / 2], [s0, s1], yerr=[d0, d1], fmt="none", ecolor="black", capsize=2.5, lw=0.8)
        ax.text(i - w / 2, s0 + d0 + top * 0.015, f"{s0:.1f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, s1 + d1 + top * 0.015, f"{s1:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(show))); ax.set_xticklabels([n for _, n in show], fontsize=8.5)
    ax.set_ylim(0, top); ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(L_("쥐기 선택 오류 (%)", "Grasp-selection error (%)"))
    ax.legend(handles=[Patch(fc="white", ec="black", lw=0.8, label=L_("근전도만", "EMG only")),
                       Patch(fc="#303030", ec="black", lw=0.8, label=L_("근전도 + 비전 융합", "EMG + vision fusion") if a.hide_rate else L_(f"근전도 + 비전 융합 (인식률 {va*100:.0f} %)", f"EMG + vision fusion ({va*100:.0f} % recognition)"))],
              frameon=False, fontsize=8, loc="upper left")
    ax.text(-0.13, 1.03, "a", transform=ax.transAxes, fontsize=12, weight="bold")
    ax.set_title(L_("실측 근전도 기반 오프라인 융합 평가", "Offline fusion on recorded EMG"), fontsize=9, loc="left", pad=8)

    # (b) 비전 정확도에 따른 쥐기 선택 오류
    bx = fig.add_axes((0.575, 0.17, 0.40, 0.70))
    sty = [("-", "o", "#9e9e9e"), ("-", "s", "black"), ("--", "^", "black"), (":", "D", "black")]
    for (key, nm), (ls, mk, col) in zip(show, sty):
        P, Y, R = C[key]; k = P.shape[1]
        xs = np.linspace(1.0 / (k - 1), 1.0, 9)
        bx.plot(100 * xs, [metrics(P, Y, R, x)[0] for x in xs], ls=ls, marker=mk, ms=3.5, color=col, lw=1.1, label=nm.replace("\n", " "))
    bx.axvline(100 * va, color="black", lw=0.6, ls=":")
    bx.set_xlabel(L_("비전 쥐기 인식률 (%)", "Vision grasp-recognition rate (%)"))
    bx.set_ylabel(L_("쥐기 선택 오류 (%)", "Grasp-selection error (%)")); bx.set_ylim(0, None)
    bx.spines[["right", "top"]].set_visible(False); bx.legend(frameon=False, fontsize=8)
    if a.hide_rate:
        bx.text(100 * va + 1, bx.get_ylim()[1] * 0.55, L_("← a 의 조건", "← condition in a"), fontsize=7.5, va="center", ha="left")
    bx.text(-0.13, 1.03, "b", transform=bx.transAxes, fontsize=12, weight="bold")
    bx.set_title(L_("비전 인식률에 따른 쥐기 선택 오류", "Grasp-selection error vs. vision recognition rate"), fontsize=9, loc="left", pad=8)
    fig.savefig(a.out, dpi=200, facecolor="white"); print("saved", a.out)


if __name__ == "__main__":
    main()
