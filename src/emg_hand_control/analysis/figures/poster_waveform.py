#!/usr/bin/env python3
"""
poster_waveform.py — 포스터용 동작별 근전도 그림 (8채널 파형 + 패턴 지문 막대)

  python poster_waveform.py data\\20261024_poster.csv --out data\\poster_emg.png
  녹화: finger_test.py record --gestures "주먹,손 펴기,V" --reps 3
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

NAVY, GRAY, LIGHT = "#1e4e79", "#9aa5b1", "#e8eef5"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="poster_emg.png")
    ap.add_argument("--font", default=None, help="글꼴 파일(.ttc) 경로 — 없으면 맑은 고딕")
    ap.add_argument("--en", action="store_true", help="영어 표기")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(0, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams["font.family"] = fam; plt.rcParams["axes.unicode_minus"] = False

    from scipy import signal
    x, lab, rep, meta = E.load_csv(a.csv)
    names = meta["gestures"].replace("_", " ").split("|")
    en = {"주먹": "Fist", "손 펴기": "Open hand", "V": "V sign"}
    y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
    segs = [(s0, s1, int(lab[s0])) for s0, s1, _ in E.segments(lab)]
    rest_rms = np.median(np.array([np.sqrt((y[s0 + E.FS:s1 - 600] ** 2).mean(0)) for s0, s1, l in segs
                                   if l == 0 and s1 - s0 > 2 * E.FS and s0 > E.FS]), 0)
    # 동작마다: 각 반복의 유지 구간 RMS → 중앙값 / 대표 반복 = 전체 세기가 중앙값에 가장 가까운 것
    info = []
    for g in range(1, len(names) + 1):
        hs = [(s0, s1) for s0, s1, l in segs if l == g]
        rms = np.array([np.sqrt((y[s0 + 1000:s1 - 600] ** 2).mean(0)) for s0, s1 in hs])
        med = np.median(rms, 0)
        k = int(np.argmin(np.abs(np.log(rms.mean(1) / med.mean()))))
        info.append((hs[k], med / rest_rms))
    lim = max(np.percentile(np.abs(y[s0 - E.FS:s1]), 99.5) for (s0, s1), _ in info)

    G = len(names)
    fig = plt.figure(figsize=(4.6 * G, 8.6), dpi=150)
    ymax = max(r.max() for _, r in info) * 1.3
    for gi, ((s0, s1), ratio) in enumerate(info):
        ax = fig.add_axes((0.05 + gi * 0.95 / G, 0.36, 0.95 / G - 0.03, 0.56))
        t0 = s0 - E.FS; seg = y[t0:s1 + E.FS // 2]
        t = (np.arange(len(seg)) - E.FS) / E.FS
        for c in range(8):
            off = (7 - c) * 2.2 * lim
            ax.plot(t, seg[:, c] + off, color=NAVY, lw=0.45)
            if gi == 0:
                ax.text(t[0] - 0.08, off, f"CH{c+1}", ha="right", va="center", fontsize=9, color="#333")
        ax.axvspan(0, (s1 - s0) / E.FS, color=LIGHT, zorder=0)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-1.3 * lim, 7 * 2.2 * lim + 1.3 * lim)
        ax.set_yticks([]); ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_xlabel("time (s)" if a.en else "시간 (s)", fontsize=9)
        ax.set_title(en.get(names[gi], names[gi]) if a.en else names[gi], fontsize=16, weight="bold", color=NAVY, pad=10)
        ax.text(0.02, 0.995, "hold" if a.en else "동작 유지", transform=ax.transAxes, fontsize=8, color="#5b7a99", va="top")
        if gi == 0:
            ax.plot([t[0] + 0.1, t[0] + 0.1], [-1.1 * lim, -1.1 * lim + 100], color="#333", lw=1.5)
            ax.text(t[0] + 0.15, -1.1 * lim + 50, "100 µV", fontsize=8, va="center")
        bx = fig.add_axes((0.05 + gi * 0.95 / G, 0.06, 0.95 / G - 0.03, 0.19))
        bx.bar(np.arange(1, 9), ratio, color=[NAVY if v >= np.sort(ratio)[-3] else GRAY for v in ratio], width=0.7)
        bx.set_yscale("log"); bx.axhline(1, color="#888", lw=0.8); bx.set_ylim(0.8, ymax)
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        tk = [v for v in (1, 2, 5, 10, 20, 50, 100) if v <= ymax]
        bx.yaxis.set_major_locator(FixedLocator(tk)); bx.yaxis.set_major_formatter(FixedFormatter([str(v) for v in tk]))
        bx.yaxis.set_minor_locator(NullLocator())
        bx.set_xticks(range(1, 9)); bx.set_xticklabels([f"{i}" for i in range(1, 9)], fontsize=9)
        bx.set_xlabel("channel" if a.en else "채널", fontsize=9)
        if gi == 0:
            bx.set_ylabel("RMS / rest (×)" if a.en else "휴식 대비 세기 (배)", fontsize=9)
        bx.spines[["right", "top"]].set_visible(False)
    fig.text(0.5, 0.272, "Pattern fingerprint — top 3 channels highlighted" if a.en else
             "패턴 지문 — 동작마다 강하게 반응하는 채널(진한 색 3개)이 다르다", ha="center", fontsize=11, color="#333")
    fig.savefig(a.out, dpi=150, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
