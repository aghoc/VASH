#!/usr/bin/env python3
"""
fatigue_within.py — 60 초 유지 수축 '안에서' 생기는 피로 (논문형 그림, 흑백)

  python fatigue_within.py data\\F1_sus1.csv data\\F1_sus2.csv data\\F1_sus3.csv --out data\\fig_fatigue_within.png  [--en]

 (a) 같은 60 초 유지의 처음 3 s 와 마지막 3 s 실제 파형 (가장 강한 채널, 같은 눈금)
 (b) 두 구간 스펙트럼과 중앙 주파수
 (c) 모든 유지 수축의 중앙 주파수 추이 (4 s 창, 시작값 = 100 %로 정규화, 평균 ± 표준편차)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import fatigue_poster as FP  # noqa: E402


def holds(files):
    from scipy import signal
    out = []
    for fn in files:
        x, lab, rep, _ = E.load_csv(fn)
        y = signal.sosfilt(E.full_sos(), x, axis=0)
        for s0, s1, l in E.segments(lab):
            if l > 0 and s1 - s0 > 30 * E.FS:
                out.append((os.path.basename(fn), int(rep[s0]), y[s0:s1]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--out", default="data/fig_fatigue_within.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    H = holds(a.csv)
    seg0 = H[0][2]
    top = np.argsort(np.sqrt((seg0[2 * E.FS:] ** 2).mean(0)))[-3:]
    ch = int(top[-1])
    tgrid = np.arange(2, int(len(seg0) / E.FS) - 4, 4)
    curves = []
    for name, r, seg in H:
        md = [FP.mdf_psd(seg[t * E.FS:(t + 4) * E.FS][:, top])[0].mean() for t in tgrid if (t + 4) * E.FS <= len(seg)]
        curves.append(100 * np.array(md) / md[0])
        print(f"{name} 반복{r}: MDF {md[0]:.1f} → {md[-1]:.1f} Hz ({100*(md[-1]/md[0]-1):+.1f} %)")
    n = min(len(c) for c in curves); C = np.array([c[:n] for c in curves]); tg = tgrid[:n] + 2
    k = np.polyfit(np.repeat(tg[None], len(C), 0).ravel(), C.ravel(), 1)[0]
    from scipy import stats
    rho, pv = stats.spearmanr(np.repeat(tg[None], len(C), 0).ravel(), C.ravel())
    print(f"유지 {len(C)}회 평균: 마지막 {C[:, -1].mean():.1f} % (시작 100 %), 기울기 {k:+.3f} %/s, ρ {rho:+.2f}, p {pv:.1e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    L = (lambda ko, en: en if a.en else ko)
    early = seg0[2 * E.FS:5 * E.FS, ch]; late = seg0[-5 * E.FS:-2 * E.FS, ch]
    lim = np.ceil(np.percentile(np.abs(np.r_[early, late]), 99.7) / 25) * 25
    fig = plt.figure(figsize=(10.5, 6.4), dpi=200, facecolor="white")
    T0 = len(seg0) / E.FS
    for i, (sg, lb) in enumerate(((early, L("처음 (2–5 s)", "Start (2–5 s)")), (late, L(f"마지막 ({T0-5:.0f}–{T0-2:.0f} s)", f"End ({T0-5:.0f}–{T0-2:.0f} s)")))):
        ax = fig.add_axes((0.07 + i * 0.46, 0.64, 0.42, 0.26))
        t = np.arange(len(sg)) / E.FS
        ax.plot(t, sg, color="black", lw=0.45); ax.set_xlim(0, 3); ax.set_ylim(-lim, lim)
        ax.set_yticks([-lim, 0, lim]); ax.set_ylabel("µV" if i == 0 else ""); ax.set_xlabel(L("시간 (s)", "time (s)"))
        ax.spines[["right", "top"]].set_visible(False); ax.set_title(lb, fontsize=9.5, loc="left")
        if i == 0: ax.text(-0.13, 1.08, "a", transform=ax.transAxes, fontsize=12, weight="bold")
    bx = fig.add_axes((0.07, 0.10, 0.36, 0.38))
    mds = []
    for sg, col, ls, lb in ((early, "#9e9e9e", "-", L("처음", "start")), (late, "black", "-", L("마지막", "end"))):
        md, f, P = FP.mdf_psd(sg[:, None]); p = P[:, 0] / P[:, 0].max(); mds.append(md[0])
        bx.plot(f, p, color=col, lw=1.4, ls=ls, label=f"{lb}  MDF {md[0]:.0f} Hz"); bx.axvline(md[0], color=col, ls="--", lw=0.9)
    bx.set_xlim(20, 300); bx.set_ylim(0, 1.15); bx.set_yticks([]); bx.set_xlabel(L("주파수 (Hz)", "frequency (Hz)"))
    bx.spines[["right", "top", "left"]].set_visible(False); bx.legend(frameon=False, fontsize=8.5)
    bx.text(-0.08, 1.04, "b", transform=bx.transAxes, fontsize=12, weight="bold")
    cx = fig.add_axes((0.52, 0.10, 0.40, 0.38))
    for c in C: cx.plot(tg, c, color="#bdbdbd", lw=0.8)
    m, s = C.mean(0), C.std(0)
    cx.fill_between(tg, m - s, m + s, color="#e0e0e0"); cx.plot(tg, m, color="black", lw=1.8)
    cx.axhline(100, color="black", lw=0.6, ls=":")
    cx.set_xlabel(L("유지 시간 (s)", "hold time (s)")); cx.set_ylabel(L("중앙 주파수 (시작 = 100 %)", "Median frequency (start = 100 %)"))
    cx.spines[["right", "top"]].set_visible(False)
    cx.text(0.98, 0.95, f"{C[:, -1].mean() - 100:+.1f} %  ({L('유지', 'holds')} n = {len(C)}, p = {pv:.0e})", transform=cx.transAxes,
            ha="right", va="top", fontsize=9, weight="bold")
    cx.text(-0.13, 1.04, "c", transform=cx.transAxes, fontsize=12, weight="bold")
    fig.text(0.07, 0.955, L(f"Fig. 5. 60 초 유지 주먹 쥐기 중 근육 피로 (CH{ch+1}, 실제 파형, 편집 없음)",
                            f"Fig. 5. Muscle fatigue during 60 s sustained fist (CH{ch+1}, raw EMG, unedited)"), fontsize=10.5, weight="bold")
    fig.savefig(a.out, dpi=200, facecolor="white"); print("saved", a.out)


if __name__ == "__main__":
    main()
