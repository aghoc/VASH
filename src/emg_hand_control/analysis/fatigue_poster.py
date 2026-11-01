#!/usr/bin/env python3
"""
fatigue_poster.py — 한 세션 동안 연속 녹화한 파일들로 근육 피로 추이 그림 (포스터용)

  python fatigue_poster.py data\\20261024_S01_R1.csv data\\20261024_S01_R2.csv ... --out data\\fatigue.png

각 동작 유지 구간(시작 0.5 s·끝 0.3 s 제외)마다
  · 중앙 주파수 MDF: 20–450 Hz, 60 Hz 배수 ±4 Hz 제외, Welch 스펙트럼의 면적 절반 지점
  · 가장 강하게 반응한 3채널 평균 (동작마다 쓰는 근육이 달라도 '일하는 근육'을 봄)
시각: 파일 저장 시각 − 녹화 길이 = 녹화 시작 → 구간 위치를 더해 세션 경과 시간(분)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

NAVY, GRAY, LIGHT = "#1e4e79", "#9aa5b1", "#e8eef5"


def mdf_psd(seg):
    from scipy import signal
    f, P = signal.welch(seg, fs=E.FS, nperseg=512, axis=0)
    m = (f >= 20) & (f <= 450)
    for h in range(60, 451, 60):
        m &= np.abs(f - h) > 4
    f, P = f[m], P[m]
    c = np.cumsum(P, axis=0)
    md = np.array([np.interp(c[-1, k] / 2, c[:, k], f) for k in range(P.shape[1])])   # 칸 사이 보간 (3.9 Hz 계단 제거)
    return md, f, P


def collect(files):
    from scipy import signal
    rows = []
    for fn in files:
        x, lab, rep, meta = E.load_csv(fn)
        t0 = os.stat(fn).st_mtime - len(x) / E.FS
        y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
        names = meta.get("gestures", "엄지|검지|중지").replace("_", " ").split("|")
        tag = os.path.basename(fn).split("_", 1)[-1].rsplit(".", 1)[0]
        for s0, s1, l in E.segments(lab):
            if l == 0:
                continue
            seg = y[s0 + E.FS // 2:s1 - 600]
            if len(seg) < E.FS:
                continue
            md, f, P = mdf_psd(seg)
            rms = np.sqrt((seg ** 2).mean(0))
            top = np.argsort(rms)[-3:]
            Pn = P[:, top] / P[:, top].sum(0)
            rows.append(dict(t=t0 + s0 / E.FS, tag=tag, name=names[l - 1], rep=int(rep[s0]),
                             mdf=float(md[top].mean()), rms=float(rms[top].mean()), f=f, psd=Pn.mean(1)))
    rows.sort(key=lambda r: r["t"])
    t00 = rows[0]["t"]
    for r in rows:
        r["min"] = (r["t"] - t00) / 60
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--out", default="fatigue.png")
    ap.add_argument("--font", default=None); ap.add_argument("--en", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="처음/마지막 비교 구간 수 (0 = 전체의 1/5)")
    ap.add_argument("--mark-rewear", nargs="*", default=[], help="재착용 직후 파일 태그 (예: S01_R3)")
    a = ap.parse_args()
    rows = collect(a.csv)
    tm = np.array([r["min"] for r in rows]); md = np.array([r["mdf"] for r in rows]); _ = None
    k, b = np.polyfit(tm, md, 1)
    n = a.n or max(5, len(rows) // 5)
    early, late = rows[:n], rows[-n:]
    md0, md1 = np.mean([r["mdf"] for r in early]), np.mean([r["mdf"] for r in late])
    from scipy import stats
    rho, pval = stats.spearmanr(tm, md)
    print(f"동작 구간 {len(rows)}개, 세션 {tm[-1]:.1f} 분")
    print(f"MDF 처음 {n}개 평균 {md0:.1f} Hz → 마지막 {n}개 {md1:.1f} Hz ({100*(md1/md0-1):+.1f} %), 기울기 {k:+.2f} Hz/분, "
          f"스피어만 ρ = {rho:+.2f} (p = {pval:.3g})")
    for tag in dict.fromkeys(r["tag"] for r in rows):
        s = [r for r in rows if r["tag"] == tag]
        print(f"  {tag:10s} {s[0]['min']:5.1f}–{s[-1]['min']:5.1f} 분  MDF 평균 {np.mean([r['mdf'] for r in s]):6.1f} Hz  "
              f"RMS(상위3) {np.mean([r['rms'] for r in s]):6.1f} µV  ({len(s)}구간)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(0, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams["font.family"] = fam; plt.rcParams["axes.unicode_minus"] = False
    T = (lambda ko, en: en if a.en else ko)

    fig = plt.figure(figsize=(14, 5.2), dpi=150)
    # ① MDF 추이
    ax = fig.add_axes((0.05, 0.14, 0.52, 0.74))
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    for i, tag in enumerate(tags):
        s = [r for r in rows if r["tag"] == tag]
        ax.axvspan(s[0]["min"] - 0.05, s[-1]["min"] + 0.05, color=LIGHT if i % 2 == 0 else "#f4f6f9", zorder=0)
        ax.text((s[0]["min"] + s[-1]["min"]) / 2, 1.01, tag.split("_")[-1], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color="#5b7a99")
    pal = ["#c62828", "#1565c0", "#6a1b9a", "#ef6c00", "#00838f", "#5d4037"]
    fixed = {"엄지": "#c62828", "검지": "#1565c0", "중지": "#6a1b9a"}
    nms = sorted(dict.fromkeys(r["name"] for r in rows), key=lambda n: list(fixed).index(n) if n in fixed else 9)
    for j, nm in enumerate(nms):
        idx = [i for i, r in enumerate(rows) if r["name"] == nm]
        ax.scatter(tm[idx], md[idx], s=22, color=fixed.get(nm, pal[j % len(pal)]), alpha=0.85, zorder=3, label=nm if len(nms) <= 6 else None)
    if len(nms) <= 6:
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88), ncol=len(nms), fontsize=9.5, handletextpad=0.2)
    for i in range(1, len(tags)):
        prev = [r for r in rows if r["tag"] == tags[i - 1]][-1]["min"]; cur = [r for r in rows if r["tag"] == tags[i]][0]["min"]
        if a.mark_rewear and tags[i] in a.mark_rewear:
            ax.text((prev + cur) / 2, 0.02, T("재착용", "re-worn"), transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#888")
    xs = np.linspace(0, tm[-1], 50)
    ax.plot(xs, k * xs + b, color="#c62828", lw=2, zorder=4)
    ax.set_xlabel(T("세션 경과 시간 (분)", "Session time (min)")); ax.set_ylabel(T("중앙 주파수 MDF (Hz)", "Median frequency (Hz)"))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.98, 0.95, T(f"{k:+.1f} Hz/분   (ρ = {rho:+.2f}, p = {pval:.1g})", f"{k:+.1f} Hz/min   (ρ = {rho:+.2f}, p = {pval:.1g})"),
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color="#c62828", weight="bold")
    ax.set_title(T("① 동작을 반복할수록 근전도 주파수가 낮아짐", "① EMG frequency drops as contractions repeat"),
                 loc="left", fontsize=12, weight="bold", color=NAVY, pad=18)
    # ② 처음 vs 마지막 스펙트럼
    bx = fig.add_axes((0.64, 0.14, 0.34, 0.74))
    f = rows[0]["f"]
    for grp, col, lb in ((early, GRAY, T(f"처음 {n}회", f"First {n}")), (late, NAVY, T(f"마지막 {n}회", f"Last {n}"))):
        p = np.mean([r["psd"] for r in grp], 0)
        bx.plot(f, p / p.max(), color=col, lw=2, label=lb)
        m = np.mean([r["mdf"] for r in grp])
        bx.axvline(m, color=col, ls="--", lw=1.2)
    bx.annotate("", xy=(md1, 1.05), xytext=(md0, 1.05), arrowprops=dict(arrowstyle="->", color="#c62828", lw=2))
    bx.text((md0 + md1) / 2, 1.09, f"{md0:.0f} → {md1:.0f} Hz ({100*(md1/md0-1):+.0f} %)", ha="center",
            fontsize=11, color="#c62828", weight="bold")
    bx.set_xlim(20, 300); bx.set_ylim(0, 1.18)
    bx.set_xlabel(T("주파수 (Hz)", "Frequency (Hz)")); bx.set_yticks([])
    bx.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.92))
    bx.spines[["right", "top", "left"]].set_visible(False)
    bx.set_title(T("② 스펙트럼이 낮은 주파수 쪽으로 이동", "② Spectrum shifts to lower frequency"),
                 loc="left", fontsize=12, weight="bold", color=NAVY, pad=18)
    fig.savefig(a.out, dpi=150, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
