#!/usr/bin/env python3
"""
poster_scope.py — 실제 녹화 파일 한 구간을 측정 장비 화면처럼 그대로 보여 주는 그림 (포스터용)

  python poster_scope.py data\\20261024_poster.csv --out data\\poster_scope.png            (검은 화면, 오실로스코프 느낌)
  python poster_scope.py data\\20261024_poster.csv --light --out data\\poster_scope_light.png  (흰 모눈종이, 근전도 기록지 느낌)

편집·평균·보정 없이 녹화된 한 구간(첫 반복)을 시간 순서 그대로 이어서 그림.
필터: 20–450 Hz 대역통과 + 60 Hz 계열 노치 (근전도 장비 기본 표시와 같은 조건)
"""
import argparse, os, sys, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="poster_scope.png")
    ap.add_argument("--light", action="store_true"); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--uvdiv", type=float, default=100.0, help="세로 한 칸 µV")
    ap.add_argument("--font", default=None); ap.add_argument("--en", action="store_true")
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
    en = {"주먹": "FIST", "손 펴기": "OPEN", "V": "V-SIGN"}
    y = signal.sosfilt(E.full_sos(), x, axis=0)
    # 구간: 지정 반복의 첫 동작 1 s 전 ~ 그 반복 마지막 휴식 끝
    idx = np.flatnonzero(rep == a.rep)
    first_act = idx[np.argmax(lab[idx] > 0)]
    s0 = max(E.FS, first_act - E.FS); s1 = idx[-1]
    seg = y[s0:s1]; L = lab[s0:s1]; t = np.arange(len(seg)) / E.FS

    if a.light:
        bg, grid_minor, grid_major, trace, txt, mark = "white", "#f6d5d5", "#eaa3a3", "#111111", "#222", "#b71c1c"
    else:
        bg, grid_minor, grid_major, trace, txt, mark = "#0b0f0c", "#132018", "#22402d", "#39ff7a", "#cfe9d6", "#ffd54f"
    nch, div = 8, a.uvdiv
    gap = 4 * div                                         # 채널 간격 4칸
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=bg)
    ax = fig.add_axes((0.07, 0.10, 0.90, 0.78), facecolor=bg)
    T = t[-1]
    ylo, yhi = -2.5 * div, (nch - 1) * gap + 2.5 * div
    # 모눈: 가로 0.2 s 작은 칸 / 1 s 큰 칸, 세로 div
    for xx in np.arange(0, T + 1e-9, 0.2):
        ax.axvline(xx, color=grid_major if abs(xx - round(xx)) < 1e-6 else grid_minor, lw=0.8 if abs(xx - round(xx)) < 1e-6 else 0.4, zorder=0)
    for yy in np.arange(ylo, yhi + 1e-9, div):
        ax.axhline(yy, color=grid_minor, lw=0.4, zorder=0)
    for c in range(nch):
        off = (nch - 1 - c) * gap
        ax.axhline(off, color=grid_major, lw=0.6, zorder=0)
        ax.plot(t, seg[:, c] + off, color=trace, lw=0.5, zorder=2)
        ax.text(-0.012 * T, off, f"CH{c+1}", ha="right", va="center", color=txt, fontsize=11, family="monospace", weight="bold")
    # 지시 표시 (녹화 때 화면에 뜬 지시 그대로)
    for k0, k1, l in E.segments(L):
        if l == 0:
            continue
        ax.axvspan(k0 / E.FS, k1 / E.FS, ymin=0.985, ymax=1.0, color=mark, zorder=3)
        ax.text((k0 + k1) / 2 / E.FS, yhi + 0.6 * div, en.get(names[l - 1], names[l - 1]) if a.en else names[l - 1],
                ha="center", va="bottom", color=mark, fontsize=14, weight="bold")
        for kk in (k0, k1):
            ax.axvline(kk / E.FS, color=mark, lw=0.8, ls=(0, (4, 4)), zorder=1, alpha=0.7)
    ax.set_xlim(0, T); ax.set_ylim(ylo, yhi + 2.2 * div)
    ax.set_xticks(np.arange(0, T + 1e-9, 2)); ax.set_yticks([])
    ax.tick_params(colors=txt, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(grid_major)
    ax.set_xlabel("time (s)" if a.en else "시간 (s)", color=txt, fontsize=11)
    stamp = datetime.datetime.fromtimestamp(os.stat(a.csv).st_mtime - len(x) / E.FS + s0 / E.FS).strftime("%Y-%m-%d %H:%M:%S")
    head = (f"ADS1299 · 8 ch · 2000 SPS · 24-bit · gain 24   |   {div:.0f} µV/div · 0.2 s/div   |   "
            f"20–450 Hz band-pass + 60 Hz notch   |   {stamp}")
    fig.text(0.07, 0.945, head, color=txt, fontsize=10.5, family="monospace")
    fig.text(0.07, 0.915, ("Raw recording, single take — no averaging or editing (forearm band, 40 % length)" if a.en else
                           "실제 녹화 한 구간 그대로 — 평균·편집 없음 (아래팔 밴드 전극, 팔 길이 40 % 지점)"),
             color=txt, fontsize=10, alpha=0.8)
    fig.savefig(a.out, dpi=150, facecolor=bg)
    print("saved", a.out, f"({T:.1f} s, 반복 {a.rep})")


if __name__ == "__main__":
    main()
