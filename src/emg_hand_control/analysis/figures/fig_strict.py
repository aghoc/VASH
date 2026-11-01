#!/usr/bin/env python3
"""
fig_strict.py — 엄격한 평가 기준으로 포스터 그림 다시 계산 (F2 쥐기, 논문형 흑백)

  python fig_strict.py [--grace 0.25] [--en] [--font 경로]

평가 기준 (eval_strict.py 와 같음):
  · 녹화 간 시험: 학습에 쓴 녹화는 시험에 절대 안 씀
      쉰 상태 = B1 으로 학습 → B2 시험,  B2 로 학습 → B1 시험 (두 방향 합침)
      피로 T1·T2·T3 = B1+B2 로 학습 → 각 T 녹화 시험
  · 녹화 전체를 시간 순서대로 50 ms 마다 판단 (전환 구간 포함), 지시 변경 후 유예 --grace 초만 제외 (기본 0.25 s < 사람 반응 0.35 s)
  · 막대 = 반복별 정확도 평균 ± SD, 점 = 반복
출력: fig_strict_F2_<동작>.png (T1·T2·T3), fig_strict_confusion_F2.png, 표 (화면 + strict_table.md)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fatigue_experiment as FX  # noqa: E402
import vision_fusion as VF  # noqa: E402
from eval_strict import wilson, kappa  # noqa: E402

D = VF.D


def stream(model, path, grace):
    W, L, Rp, since, _ = FX.stream_windows(path)
    pred = model.proba(W).argmax(1)
    m = since >= grace
    return pred[m], L[m], Rp[m], since[m], L, pred, since


def per_rep(pred, L, R, c):
    return [100 * np.mean(pred[(R == r) & (L == c)] == c) for r in sorted(set(R[R >= 0].tolist())) if ((R == r) & (L == c)).any()]


def trials(L_all, pred_all, since_all, c):
    seg = np.r_[0, np.flatnonzero(np.diff(L_all)) + 1, len(L_all)]; k = n = 0
    for s, t in zip(seg[:-1], seg[1:]):
        if L_all[s] != c: continue
        m = since_all[s:t] >= 0.5
        if not m.any(): continue
        n += 1; k += int(np.bincount(pred_all[s:t][m], minlength=4).argmax() == c)
    return k, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grace", type=float, default=0.25)
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    f = {k: f"{D}/F2_{k}.csv" for k in ("B1", "B2", "T1", "T2", "T3")}
    names = FX.class_names(f["B1"], a.en)

    def train(keys):
        X, Y, R = VF.load([f[k] for k in keys]); return VF.Model(X, Y, R)

    # 쉰 상태: 두 방향
    stages = {}
    acc = {c: [] for c in range(4)}; P_all, L_allc = [], []; tri = {c: [0, 0] for c in range(4)}; kap_in = ([], [])
    for tr_k, te_k in (("B1", "B2"), ("B2", "B1")):
        p, l, r, _, La, Pa, Sa = stream(train([tr_k]), f[te_k], a.grace)
        r = r + (0 if te_k == "B1" else 1000)
        for c in range(4): acc[c] += per_rep(p, l, r, c)
        for c in (1, 2, 3):
            k, n = trials(La, Pa, Sa, c); tri[c][0] += k; tri[c][1] += n
        P_all.append(p); L_allc.append(l); kap_in[0].append(l); kap_in[1].append(p)
    stages["rested"] = (acc, tri, np.concatenate(kap_in[0]), np.concatenate(kap_in[1]))
    conf_rest = (np.concatenate(P_all), np.concatenate(L_allc))
    mB = train(["B1", "B2"])
    for t in ("T1", "T2", "T3"):
        p, l, r, _, La, Pa, Sa = stream(mB, f[t], a.grace)
        acc = {c: per_rep(p, l, r, c) for c in range(4)}
        tri = {c: list(trials(La, Pa, Sa, c)) for c in (1, 2, 3)}
        stages[t] = (acc, tri, l, p)

    keys = ["rested", "T1", "T2", "T3"]; stl = [L_("쉰 상태", "Rested")] + [L_(f"피로 {t}", f"Fatigue {t}") for t in keys[1:]]
    lines = [f"# 엄격한 평가 (F2 쥐기) — 녹화 간 시험, 전환 포함, 유예 {a.grace:g} s", "",
             "| 상태 | " + " | ".join(names) + " | 평균 | κ | 시도 단위 맞음/전체 [95 % CI] |", "|" + "---|" * (len(names) + 4)]
    for k, lb in zip(keys, stl):
        acc, tri, l, p = stages[k]
        m = [np.mean(acc[c]) for c in range(4)]
        kk = sum(tri[c][0] for c in (1, 2, 3)); nn = sum(tri[c][1] for c in (1, 2, 3)); lo, hi = wilson(kk, nn)
        lines.append(f"| {lb} | " + " | ".join(f"{v:.1f}" for v in m) + f" | {np.mean(m):.1f} | {kappa(l, p):.2f} | {kk}/{nn} [{lo:.0f}–{hi:.0f}] |")
    print("\n".join(lines))
    open(f"{D}/strict_table.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 10, "axes.linewidth": 0.8})
    fills = ["white", "#d9d9d9", "#8c8c8c", "#303030"]; files = ["rest", "fist", "pinch", "tripod"]
    rng = np.random.default_rng(3)
    for c in range(4):
        fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=200, facecolor="white")
        for i, k in enumerate(keys):
            g = stages[k][0][c]; m, s = np.mean(g), np.std(g)
            ax.bar(i, m, 0.62, color=fills[i], edgecolor="black", lw=0.8)
            ax.errorbar(i, m, yerr=s, color="black", capsize=3, lw=0.8)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=7, color="#888", zorder=3)
            ax.text(i, 104, f"{m:.0f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(stl, fontsize=9)
        ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.spines["left"].set_bounds(0, 100)
        ax.spines[["right", "top"]].set_visible(False); ax.set_ylabel(L_("분류 정확도 (%)", "Accuracy (%)"))
        ax.set_title(names[c], fontsize=10.5, pad=12)
        fig.tight_layout(); fig.savefig(f"{D}/fig_strict_F2_{files[c]}.png", dpi=200, facecolor="white"); plt.close(fig)

    p, l = conf_rest; k = 4; C = np.zeros((k, k))
    for t_, q in zip(l, p): C[t_, q] += 1
    C = 100 * C / C.sum(1, keepdims=True)
    n2 = [n.replace(" (", "\n(") for n in names]
    fig = plt.figure(figsize=(4.2, 3.9), dpi=200, facecolor="white"); ax = fig.add_axes((0.30, 0.22, 0.66, 0.66))
    ax.imshow(C, cmap="Greys", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white" if C[i, j] > 55 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.set_xticklabels(n2, fontsize=8); ax.set_yticklabels(n2, fontsize=8)
    ax.set_xlabel(L_("예측", "Predicted")); ax.set_ylabel(L_("실제", "True"))
    for i in (2, 3):
        ax.add_patch(plt.Rectangle((1.5, i - 0.5), 2, 1, fill=False, ec="black", lw=1.6))
    fig.savefig(f"{D}/fig_strict_confusion_F2.png", dpi=200, facecolor="white")
    print("혼동표 (쉰 상태, 녹화 간):\n", np.round(C, 1))
    print("saved fig_strict_F2_*.png, fig_strict_confusion_F2.png, strict_table.md")


if __name__ == "__main__":
    main()
