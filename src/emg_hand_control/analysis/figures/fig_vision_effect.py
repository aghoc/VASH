#!/usr/bin/env python3
"""
fig_vision_effect.py — 피로 상태에서 '근전도만' vs '근전도 + 비전 정답 보정(상한)' 비교 (논문형, 흑백)

  python fig_vision_effect.py --pre data\\F1_recover.csv data\\F1_rest2.csv --post data\\F1_T1.csv data\\F1_T2.csv data\\F1_T3.csv
                              --out data\\fig_vision_effect.png  [--en] [--font 경로]

 (a) 피로 단계별 오류율 (동작 평균)
 (b) 가장 피로한 단계의 동작별 오류율
 (c) 1분당 오동작 횟수 (틀린 판단이 MIN_RUN 창 = 150 ms 이상 이어진 횟수 → 서보가 실제로 움직일 수준)
비전 보정 = 피로 후 다른 반복의 정답 라벨로 다조건 학습(MC). 실제 비전은 가끔 틀리므로 '상한'.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import fatigue_experiment as FX  # noqa: E402

MIN_RUN = 3            # 50 ms 창 3개 = 150 ms 이상 이어진 오답만 '오동작'
BASE, VIS = "M0", "MC"


def err_by_motion(d):
    return {c: 100 - float(np.mean(v)) for c, v in d.items() if len(v)}


def false_runs_per_min(pred, L, mask):
    """mask 안에서 pred != L 이 MIN_RUN 창 이상 연속된 구간 수 / 분"""
    wrong = (pred != L) & mask
    n, run = 0, 0
    for i in range(len(wrong)):
        if wrong[i]:
            run += 1
        else:
            if run >= MIN_RUN: n += 1
            run = 0
    if run >= MIN_RUN: n += 1
    minutes = mask.sum() * E.STEP / E.FS / 60
    return n / minutes if minutes > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, nargs="+"); ap.add_argument("--post", required=True, nargs="+")
    ap.add_argument("--out", default="data/fig_vision_effect.png"); ap.add_argument("--en", action="store_true")
    ap.add_argument("--font", default=None); ap.add_argument("--names", default=None, help='예: "Rest,Open,Fist,Precision"')
    a = ap.parse_args()
    L_ = (lambda ko, en: en if a.en else ko)
    names = FX.class_names(a.pre[0], a.en, a.names)
    out = FX.analyze(a.pre, a.post)
    nlv = len(out["levels"])
    res, Lb, lvl, sc = out["res"], out["L"], out["level"], out["score"]

    # (a) 단계별 오류율 (동작 평균, 반복 단위 평균 ± SD)
    def stage_err(k, lv):
        d = out["levels"][lv][k]
        reps = np.array([d[c] for c in range(4) if len(d[c])])
        e = 100 - reps.mean(0)
        return float(e.mean()), float(e.std())
    A = {k: [stage_err(k, lv) for lv in range(nlv)] for k in (BASE, VIS)}
    worst = int(np.argmin(out["phi_lv"]))
    Bm = {k: err_by_motion(out["levels"][worst][k]) for k in (BASE, VIS)}
    C = {k: [false_runs_per_min(res[k], Lb, sc & (res[k] >= 0) & (lvl == lv)) for lv in range(nlv)] for k in (BASE, VIS)}
    tot = {k: false_runs_per_min(res[k], Lb, sc & (res[k] >= 0)) for k in (BASE, VIS)}

    print("[a] 단계별 오류율 (%)")
    for lv in range(nlv):
        print(f"    T{lv+1}: 근전도만 {A[BASE][lv][0]:.1f}  비전 보정 {A[VIS][lv][0]:.1f}")
    print(f"[b] T{worst+1} 동작별 오류율: " + ", ".join(f"{names[c]} {Bm[BASE][c]:.1f}→{Bm[VIS][c]:.1f}" for c in Bm[BASE]))
    print(f"[c] 오동작/분 (≥{MIN_RUN * 50} ms): " + ", ".join(f"T{lv+1} {C[BASE][lv]:.1f}→{C[VIS][lv]:.1f}" for lv in range(nlv))
          + f"  전체 {tot[BASE]:.1f}→{tot[VIS]:.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    sty = {BASE: dict(color="white", edgecolor="black", lw=0.9, label=L_("근전도만", "EMG only")),
           VIS: dict(color="#303030", edgecolor="black", lw=0.9, label=L_("근전도 + 비전 보정", "EMG + vision correction"))}
    fig = plt.figure(figsize=(10.5, 4.2), dpi=200, facecolor="white")
    w = 0.36

    def pair(ax, xs, vb, vv, sb=None, sv=None, fmt="{:.1f}", ytop=None):
        for x, b, v, i in zip(xs, vb, vv, range(len(xs))):
            ax.bar(x - w / 2, b, w, **{**sty[BASE], "label": sty[BASE]["label"] if i == 0 else None})
            ax.bar(x + w / 2, v, w, **{**sty[VIS], "label": sty[VIS]["label"] if i == 0 else None})
            if sb is not None:
                ax.errorbar(x - w / 2, b, yerr=sb[i], color="black", capsize=2.5, lw=0.8)
                ax.errorbar(x + w / 2, v, yerr=sv[i], color="black", capsize=2.5, lw=0.8)
            tb = b + (sb[i] if sb is not None else 0); tv = v + (sv[i] if sv is not None else 0)
            ax.text(x - w / 2, tb + ytop * 0.02, fmt.format(b), ha="center", fontsize=7.5)
            ax.text(x + w / 2, tv + ytop * 0.02, fmt.format(v), ha="center", fontsize=7.5)
            if b > 0 and b - v > 0.5:
                ax.text(x, max(tb, tv) + ytop * 0.10, f"−{100 * (1 - v / b):.0f}%", ha="center", fontsize=8.5, weight="bold")
        ax.set_ylim(0, ytop); ax.spines[["right", "top"]].set_visible(False)

    stg = [f"T{lv+1}" for lv in range(nlv)]
    ax = fig.add_axes((0.065, 0.17, 0.27, 0.62))
    top = max(m + s for m, s in A[BASE]) * 1.35
    pair(ax, np.arange(nlv), [m for m, _ in A[BASE]], [m for m, _ in A[VIS]], [s for _, s in A[BASE]], [s for _, s in A[VIS]], ytop=top)
    ax.set_xticks(range(nlv)); ax.set_xticklabels([L_(f"피로 {s}", f"Fatigue {s}") for s in stg])
    ax.set_ylabel(L_("오류율 (%)", "Error rate (%)"))
    ax.text(-0.2, 1.06, "a", transform=ax.transAxes, fontsize=12, weight="bold")

    bx = fig.add_axes((0.40, 0.17, 0.27, 0.62))
    cs = list(Bm[BASE]); top = max(Bm[BASE].values()) * 1.3
    pair(bx, np.arange(len(cs)), [Bm[BASE][c] for c in cs], [Bm[VIS][c] for c in cs], ytop=top)
    bx.set_xticks(range(len(cs))); bx.set_xticklabels([names[c].replace(" (", "\n(") for c in cs])
    bx.set_ylabel(L_(f"오류율 (%), 가장 피로한 T{worst+1}", f"Error rate (%), most fatigued T{worst+1}"))
    bx.text(-0.2, 1.06, "b", transform=bx.transAxes, fontsize=12, weight="bold")

    cx = fig.add_axes((0.74, 0.17, 0.24, 0.62))
    top = max(C[BASE]) * 1.35
    pair(cx, np.arange(nlv), C[BASE], C[VIS], ytop=top)
    cx.set_xticks(range(nlv)); cx.set_xticklabels([L_(f"피로 {s}", f"Fatigue {s}") for s in stg])
    cx.set_ylabel(L_("오동작 (회/분)", "False actions (per min)"))
    cx.text(-0.22, 1.06, "c", transform=cx.transAxes, fontsize=12, weight="bold")

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=9.5)
    fig.savefig(a.out, dpi=200, facecolor="white"); print("saved", a.out)


if __name__ == "__main__":
    main()
