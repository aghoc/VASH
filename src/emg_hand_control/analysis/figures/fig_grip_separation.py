#!/usr/bin/env python3
"""
fig_grip_separation.py — 집기 ↔ 세손가락 구분: 중지 지시 없음(G1) vs 중지 펴기/굽힘(G2, F2) (논문형, 흑백)

  python fig_grip_separation.py [--out data\\fig_grip_separation.png] [--en] [--font 경로]

 (a) G1 혼동 행렬 (5종, 세게 반복, 중지 지시 없음)
 (b) F2 기준 혼동 행렬 (중지 펴기/굽힘)
 (c) 녹화별 집기·세손가락 재현율 (G1 → G2 → F2 기준 B1·B2 → 피로 뒤 회복)
모두 반복 단위 교차검증, 창 단위, 평활 없음. 모델은 vision_fusion.Model (fatigue_experiment M0 와 같음).
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_fusion as VF  # noqa: E402

D = VF.D


def conf(P, Y):
    k = P.shape[1]; pred = P.argmax(1); C = np.zeros((k, k))
    for t, p in zip(Y, pred): C[t, p] += 1
    return 100 * C / C.sum(1, keepdims=True)


def recall_reps(P, Y, R, c):
    pred = P.argmax(1)
    return [100 * np.mean(pred[(R == r) & (Y == c)] == c) for r in sorted(set(R.tolist())) if ((R == r) & (Y == c)).any()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{D}/fig_grip_separation.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    Xg, Yg, Rg = VF.load([f"{D}/20261024_G1.csv"])
    G1 = VF.cv_proba(Xg, Yg, Rg, keep=lambda r: r % 2 == 1)                       # 세게 반복만 시험
    recs = {}
    for key, files in (("G2", ["G2.csv"]), ("B1", ["F2_B1.csv"]), ("B2", ["F2_B2.csv"]), ("rec", ["F2_recover.csv"])):
        X, Y, R = VF.load([f"{D}/{f}" for f in files]); recs[key] = VF.cv_proba(X, Y, R)
    Xb, Yb, Rb = VF.load([f"{D}/F2_B1.csv", f"{D}/F2_B2.csv"]); FB = VF.cv_proba(Xb, Yb, Rb)
    # 라벨 번호: G1 1 펴기 2 주먹 3 집기 4 세손가락 5 옆집기 / G2 1 집기 2 세손가락 3 주먹 / F2 1 주먹 2 집기 3 세손가락
    idx = {"G1": (3, 4), "G2": (1, 2), "B1": (2, 3), "B2": (2, 3), "rec": (2, 3)}
    rows = [("G1", G1)] + list(recs.items())
    print("재현율 (%) 평균 ± SD (반복별)        집기         세손가락")
    stats = []
    for key, (P, Y, R) in rows:
        ip, it = idx[key]; rp, rt = recall_reps(P, Y, R, ip), recall_reps(P, Y, R, it)
        stats.append((np.mean(rp), np.std(rp), np.mean(rt), np.std(rt), rp, rt))
        print(f"  {key:4s}  {np.mean(rp):5.1f} ± {np.std(rp):4.1f}   {np.mean(rt):5.1f} ± {np.std(rt):4.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(10.5, 3.9), dpi=200, facecolor="white")

    def mat(ax, C, names, hl, title, tag, rot=40):
        ax.imshow(C, cmap="Greys", vmin=0, vmax=100)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if C[i, j] > 55 else "black")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7.5, rotation=rot, ha="right" if rot else "center"); ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel(L_("예측", "Predicted"), fontsize=8); ax.set_ylabel(L_("실제", "True"), fontsize=8)
        for i in hl:
            ax.add_patch(plt.Rectangle((hl[0] - 0.5, i - 0.5), len(hl), 1, fill=False, ec="black", lw=1.6))
        ax.set_title(title, fontsize=8.5, loc="left")
        x0 = ax.get_position().x0; fig.text(x0 - 0.065, 0.92, tag, fontsize=12, weight="bold")

    n1 = [L_(s, e) for s, e in (("휴식", "Rest"), ("펴기", "Open"), ("주먹", "Fist"), ("집기", "Pinch"), ("세손가락", "Tripod"), ("옆집기", "Lateral"))]
    n2 = [L_(s, e) for s, e in (("휴식", "Rest"), ("주먹", "Fist"), ("집기\n(중지 펴기)", "Pinch\n(middle up)"), ("세손가락\n(중지 굽힘)", "Tripod\n(middle bent)"))]
    mat(fig.add_axes((0.075, 0.24, 0.24, 0.62)), conf(*G1[:2]), n1, (3, 4), L_("중지 지시 없음 (G1)", "No middle-finger cue (G1)"), "a")
    mat(fig.add_axes((0.42, 0.24, 0.22, 0.62)), conf(*FB[:2]), n2, (2, 3), L_("중지 펴기/굽힘 (F2 기준)", "Middle-finger cue (F2 baseline)"), "b", rot=0)

    cx = fig.add_axes((0.735, 0.24, 0.25, 0.62)); w = 0.36
    labs = ["G1", "G2", "B1", "B2", L_("회복", "Rec.")]
    rng = np.random.default_rng(1)
    for i, (mp, sp, mt, st, rp, rt) in enumerate(stats):
        for dx, m, s, v, fc in ((-w / 2, mp, sp, rp, "white"), (w / 2, mt, st, rt, "#303030")):
            cx.bar(i + dx, m, w, color=fc, edgecolor="black", lw=0.8)
            cx.errorbar(i + dx, m, yerr=s, color="black", capsize=2, lw=0.7)
            cx.scatter(i + dx + rng.uniform(-w / 6, w / 6, len(v)), v, s=5, color="#888", zorder=3)
    cx.axvline(0.5, color="black", lw=0.6, ls=":")
    cx.set_xticks(range(len(labs))); cx.set_xticklabels(labs)
    cx.set_ylim(0, 112); cx.set_yticks(range(0, 101, 20)); cx.spines["left"].set_bounds(0, 100)
    cx.spines[["right", "top"]].set_visible(False); cx.set_ylabel(L_("재현율 (%)", "Recall (%)"))
    cx.legend(handles=[Patch(fc="white", ec="black", label=L_("집기", "Pinch")), Patch(fc="#303030", ec="black", label=L_("세손가락", "Tripod"))],
              frameon=False, fontsize=7.5, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.text(0.735 - 0.06, 0.92, "c", fontsize=12, weight="bold")
    fig.savefig(a.out, dpi=200, facecolor="white"); print("saved", a.out)

    # 포스터용: (b) 혼동 행렬만 따로
    fig2 = plt.figure(figsize=(4.2, 3.9), dpi=200, facecolor="white")
    fig = fig2  # mat() 이 fig.text 로 태그를 쓰므로
    ax = fig2.add_axes((0.30, 0.22, 0.66, 0.66))
    C = conf(*FB[:2]); k = len(n2)
    ax.imshow(C, cmap="Greys", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white" if C[i, j] > 55 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(n2, fontsize=8); ax.set_yticklabels(n2, fontsize=8)
    ax.set_xlabel(L_("예측", "Predicted")); ax.set_ylabel(L_("실제", "True"))
    for i in (2, 3):
        ax.add_patch(plt.Rectangle((1.5, i - 0.5), 2, 1, fill=False, ec="black", lw=1.6))
    p = os.path.join(os.path.dirname(a.out), "fig_grip_confusion_F2.png")
    fig2.savefig(p, dpi=200, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
