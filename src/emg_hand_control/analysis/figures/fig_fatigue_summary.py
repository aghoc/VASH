#!/usr/bin/env python3
"""
fig_fatigue_summary.py — 피로에 따른 근전도 분류 저하: F1(손가락) + F2(쥐기) 한 그림 (논문형, 흑백)

  python fig_fatigue_summary.py [--out data\\fig_fatigue_summary.png] [--en] [--font 경로]

 (a) F1 손가락: 동작별 정확도 — 쉰 상태(반복 교차) → 피로 T1·T2·T3
 (b) F2 쥐기: 같은 형식
 (c) 같은 동작 중앙 주파수 비율 (쉰 상태 = 100, 라벨 없이 추정한 피로 지수 φ)
근전도 모델은 fatigue_experiment 의 M0 (정적, 보정 없음). 시험 녹화는 시간 순서대로 50 ms 마다 판단.
F1 기준 = 10분 회복 후 녹화 2개 (피로 전 녹화 불안정), F2 기준 = 피로 전 녹화 2개.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fatigue_experiment as FX  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SETS = {
    "F1": ([f"{D}/F1_recover.csv", f"{D}/F1_rest2.csv"], [f"{D}/F1_T1.csv", f"{D}/F1_T2.csv", f"{D}/F1_T3.csv"]),
    "F2": ([f"{D}/F2_B1.csv", f"{D}/F2_B2.csv"], [f"{D}/F2_T1.csv", f"{D}/F2_T2.csv", f"{D}/F2_T3.csv"]),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{D}/fig_fatigue_summary.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    res = {}
    for key, (pre, post) in SETS.items():
        out = FX.analyze(pre, post)
        names = FX.class_names(pre[0], a.en)
        res[key] = (out, names)
        print(f"[{key}] " + "  ".join(names))
        print(f"   쉰 상태 " + " ".join(f"{np.mean(out['pre'][c]):6.1f}" for c in range(4)) + f"   평균 {FX.mean_acc(out['pre']):.1f}")
        for lv in range(3):
            d = out["levels"][lv]["M0"]
            print(f"   T{lv+1}     " + " ".join(f"{np.mean(d[c]):6.1f}" for c in range(4)) + f"   평균 {FX.mean_acc(d):.1f}   φ {out['phi_lv'][lv]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(10.5, 3.9), dpi=200, facecolor="white")
    fills = ["white", "#c8c8c8", "#7a7a7a", "#303030"]
    stl = [L_("쉰 상태", "Rested"), "T1", "T2", "T3"]
    rng = np.random.default_rng(3)

    def panel(ax, out, names, title, tag):
        st = [out["pre"]] + [out["levels"][lv]["M0"] for lv in range(3)]; w = 0.2
        for i, d in enumerate(st):
            for c in range(4):
                g = d[c]; x = c + (i - 1.5) * w; m, s = np.mean(g), np.std(g)
                ax.bar(x, m, w * 0.95, color=fills[i], edgecolor="black", lw=0.7, label=stl[i] if c == 0 else None)
                ax.errorbar(x, m, yerr=s, color="black", capsize=1.8, lw=0.7)
                ax.scatter(x + rng.uniform(-w / 5, w / 5, len(g)), g, s=4, color="#888", zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.spines["left"].set_bounds(0, 100)
        ax.spines[["right", "top"]].set_visible(False); ax.set_ylabel(L_("분류 정확도 (%)", "Accuracy (%)"))
        ax.set_title(title, fontsize=8.5, loc="left")
        fig.text(ax.get_position().x0 - 0.055, 0.93, tag, fontsize=12, weight="bold")

    ax = fig.add_axes((0.06, 0.22, 0.34, 0.62)); panel(ax, *res["F1"], L_("F1 손가락", "F1 fingers"), "a")
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.07), handlelength=1.2, columnspacing=0.8)
    bx = fig.add_axes((0.46, 0.22, 0.34, 0.62)); panel(bx, *res["F2"], L_("F2 쥐기", "F2 grasps"), "b")

    cx = fig.add_axes((0.87, 0.22, 0.12, 0.62))
    for key, mk, ls in (("F1", "o", "-"), ("F2", "s", "--")):
        cx.plot(range(4), [100] + [100 * p for p in res[key][0]["phi_lv"]], marker=mk, ls=ls, color="black", ms=4, lw=1, label=key)
    cx.set_xticks(range(4)); cx.set_xticklabels([L_("쉼", "R"), "T1", "T2", "T3"], fontsize=8)
    cx.set_ylim(80, 103); cx.spines[["right", "top"]].set_visible(False)
    cx.set_ylabel(L_("중앙 주파수 (쉰 상태 = 100)", "Median freq. (rested = 100)"), fontsize=8)
    cx.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.87 - 0.06, 0.93, "c", fontsize=12, weight="bold")
    fig.savefig(a.out, dpi=200, facecolor="white"); print("saved", a.out)


if __name__ == "__main__":
    main()
