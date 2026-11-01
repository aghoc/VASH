#!/usr/bin/env python3
"""
fig_fatigue_parts.py — 포스터용 피로 그림을 따로따로 (논문형, 흑백, T3 제외)

  python fig_fatigue_parts.py [--en] [--font 경로]
출력 (data 폴더):
  fig_fatigue_F2_<동작>.png   F2 쥐기 동작별 정확도: 쉰 상태(반복 교차) → 피로 T1 → T2 (막대 = 평균 ± SD, 점 = 반복)
  fig_fatigue_mdf.png         F2 같은 동작 중앙 주파수 비율 (쉰 상태 = 100), 라벨로 직접 측정 (φ 추정값 아님)
근전도 모델 = fatigue_experiment M0 (정적, 보정 없음). 실험은 T1~T3 세 단계였고 그림에는 T1·T2 만 표시.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fatigue_experiment as FX  # noqa: E402
import emg_classifier as E  # noqa: E402
from fig_fatigue_summary import SETS, D  # noqa: E402



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    ap.add_argument("--levels", type=int, default=2, help="표시할 피로 단계 수 (2 = T1·T2, 3 = T1·T2·T3)")
    a = ap.parse_args()
    NLV = a.levels; suf = "" if NLV == 2 else f"_T{NLV}"
    L_ = (lambda ko, e: e if a.en else ko)
    out = {"F2": FX.analyze(*SETS["F2"])}
    names = FX.class_names(SETS["F2"][0][0], a.en)
    files = ["rest", "fist", "pinch", "tripod"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 10, "axes.linewidth": 0.8})
    fills = ["white", "#d9d9d9", "#8c8c8c", "#303030"]
    stl = [L_("쉰 상태", "Rested")] + [L_(f"피로 T{i}", f"Fatigue T{i}") for i in range(1, NLV + 1)]
    o = out["F2"]; st = [o["pre"]] + [o["levels"][lv]["M0"] for lv in range(NLV)]
    rng = np.random.default_rng(3)
    for c in range(4):
        fig, ax = plt.subplots(figsize=(3.0 + 0.6 * (NLV - 2), 3.0), dpi=200, facecolor="white")
        for i, d in enumerate(st):
            g = d[c]; m, s = np.mean(g), np.std(g)
            ax.bar(i, m, 0.62, color=fills[i], edgecolor="black", lw=0.8)
            ax.errorbar(i, m, yerr=s, color="black", capsize=3, lw=0.8)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=7, color="#888", zorder=3)
            ax.text(i, 104, f"{m:.0f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(st))); ax.set_xticklabels(stl, fontsize=9)
        ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.spines["left"].set_bounds(0, 100)
        ax.spines[["right", "top"]].set_visible(False); ax.set_ylabel(L_("분류 정확도 (%)", "Accuracy (%)"))
        ax.set_title(names[c], fontsize=10.5, pad=12)
        fig.tight_layout(); p = f"{D}/fig_fatigue_F2_{files[c]}{suf}.png"; fig.savefig(p, dpi=200, facecolor="white"); plt.close(fig)
        print(f"{names[c]:14s} " + "  ".join(f"{stl[i]} {np.mean(d[c]):.1f}" for i, d in enumerate(st)) + f"  → {p}")

    # 같은 동작 중앙 주파수 비율 — 라벨로 직접 측정 (동작 구간 스펙트럼 평균의 MDF / 쉰 상태 같은 동작 MDF, 채널 에너지 가중)
    pre, post = SETS["F2"]
    Xp, Yp, _ = E.build_dataset(pre, FX.ACT); Xp = np.array(Xp); f0, _ = FX.psd(Xp[0])
    def spec(X, Y, c):
        return np.mean([FX.psd(w)[1] for w in X[Y == c]], 0)
    ref = {c: FX.mdf_from_psd(f0, spec(Xp, Yp, c)) for c in (1, 2, 3)}
    V = {c: [100.0] for c in (1, 2, 3)}
    for path in post[:NLV]:
        X, Y, _ = E.build_dataset([path], FX.ACT); X = np.array(X)
        for c in (1, 2, 3):
            Pm = spec(X, Y, c); Ech = Pm.sum(0)
            V[c].append(100 * float(np.sum(Ech * FX.mdf_from_psd(f0, Pm) / ref[c]) / np.sum(Ech)))
    fig, ax = plt.subplots(figsize=(3.0 + 0.6 * (NLV - 2), 3.0), dpi=200, facecolor="white")
    for c, mk, ls in ((1, "o", "-"), (2, "s", "--"), (3, "^", ":")):
        ax.plot(range(NLV + 1), V[c], marker=mk, ls=ls, color="black", ms=5, lw=1.1, label=names[c])
        print(f"MDF {names[c]}: " + ", ".join(f"{x:.1f}" for x in V[c]))
    ax.axhline(100, color="#999", lw=0.6)
    ax.set_xticks(range(NLV + 1)); ax.set_xticklabels(stl, fontsize=9)
    ax.set_ylim(85, 105); ax.set_yticks(range(85, 106, 5)); ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(L_("중앙 주파수 (쉰 상태 = 100)", "Median frequency (rested = 100)"))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{D}/fig_fatigue_mdf{suf}.png", dpi=200, facecolor="white"); print("saved", f"{D}/fig_fatigue_mdf{suf}.png")

if __name__ == "__main__":
    main()
