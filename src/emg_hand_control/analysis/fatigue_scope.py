#!/usr/bin/env python3
"""
fatigue_scope.py — 근육 피로를 '실제 근전도 파형'으로 보여 주는 포스터 그림 (측정 장비 화면 스타일)

  python fatigue_scope.py data\\20261024_S01_R1.csv data\\20261024_S01_R2.csv data\\20261024_S01_R3.csv --out data\\fatigue_scope.png
  옵션: --light (흰 기록지), --en (영어), --finger 중지 (손가락 직접 지정, 기본 = 감소가 가장 큰 손가락)

같은 손가락 동작의 '처음 반복'과 '마지막 반복'을 같은 채널·같은 눈금으로 나란히:
  ① 3 초 유지 구간 전체   ② 200 ms 확대 (파형이 느려짐)   ③ 주파수 스펙트럼 + 중앙 주파수
  ④ 모든 반복의 중앙 주파수 추이 (세션 6 분)
편집·평균 없음. 필터 20–450 Hz + 60 Hz 계열 노치.
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
        x, lab, rep, meta = E.load_csv(fn)
        t0 = os.stat(fn).st_mtime - len(x) / E.FS
        y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
        names = meta.get("gestures", "엄지|검지|중지").replace("_", " ").split("|")
        tag = os.path.basename(fn).split("_", 1)[-1].rsplit(".", 1)[0]
        for s0, s1, l in E.segments(lab):
            if l == 0 or s1 - s0 < 2 * E.FS:
                continue
            out.append(dict(t=t0 + s0 / E.FS, tag=tag, name=names[l - 1], sig=y[s0:s1 + E.FS // 2], hold=s1 - s0))
    out.sort(key=lambda h: h["t"])
    for h in out:
        h["min"] = (h["t"] - out[0]["t"]) / 60
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--out", default="fatigue_scope.png")
    ap.add_argument("--light", action="store_true"); ap.add_argument("--en", action="store_true")
    ap.add_argument("--finger", default=None); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    H = holds(a.csv)
    # 손가락마다: 처음 반복의 가장 강한 채널 → 그 채널로 처음 vs 마지막 MDF 비교, 감소가 가장 큰 손가락 선택
    best = None
    for nm in dict.fromkeys(h["name"] for h in H):
        hs = [h for h in H if h["name"] == nm]
        first, last = hs[0], hs[-1]
        seg0 = first["sig"][E.FS // 2:first["hold"] - 600]
        ch = int(np.argmax(np.sqrt((seg0 ** 2).mean(0))))
        m0 = FP.mdf_psd(seg0[:, [ch]])[0][0]
        m1 = FP.mdf_psd(last["sig"][E.FS // 2:last["hold"] - 600][:, [ch]])[0][0]
        print(f"  {nm}: CH{ch+1}  처음 {m0:.1f} Hz → 마지막 {m1:.1f} Hz ({100*(m1/m0-1):+.1f} %)")
        if (a.finger and nm == a.finger) or (not a.finger and (best is None or m1 / m0 < best[4] / best[3])):
            best = (nm, first, last, m0, m1, ch)
    nm, first, last, m0, m1, ch = best
    print(f"선택: {nm} CH{ch+1}  {first['tag']} {first['min']:.1f}분 vs {last['tag']} {last['min']:.1f}분")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(0, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams["font.family"] = fam; plt.rcParams["axes.unicode_minus"] = False
    T = (lambda ko, en: en if a.en else ko)
    if a.light:
        bg, gmin, gmaj, c0, c1, txt, acc = "white", "#f6d5d5", "#eaa3a3", "#1565c0", "#111111", "#222", "#b71c1c"
    else:
        bg, gmin, gmaj, c0, c1, txt, acc = "#0b0f0c", "#132018", "#22402d", "#8ab4f8", "#39ff7a", "#cfe9d6", "#ffd54f"

    def scope(ax, xlim, ylim, xdiv, ydiv):
        ax.set_facecolor(bg)
        for xx in np.arange(xlim[0], xlim[1] + 1e-9, xdiv / 5):
            major = abs((xx - xlim[0]) / xdiv - round((xx - xlim[0]) / xdiv)) < 1e-6
            ax.axvline(xx, color=gmaj if major else gmin, lw=0.7 if major else 0.35, zorder=0)
        for yy in np.arange(ylim[0], ylim[1] + 1e-9, ydiv):
            ax.axhline(yy, color=gmaj if abs(yy) < 1e-9 else gmin, lw=0.4, zorder=0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_yticks([])
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values(): sp.set_color(gmaj)

    s0 = first["sig"][:, ch]; s1 = last["sig"][:, ch]
    lim = np.ceil(max(np.percentile(np.abs(s0), 99.7), np.percentile(np.abs(s1), 99.7)) / 50) * 50
    fig = plt.figure(figsize=(16, 9.5), dpi=150, facecolor=bg)
    fig.text(0.05, 0.955, T(f"근육 피로: 같은 동작({nm})의 처음과 마지막 — CH{ch+1} 실제 파형",
                            f"Muscle fatigue: same motion ({nm}): first vs last contraction — CH{ch+1} raw EMG"),
             color=txt, fontsize=15, weight="bold")
    fig.text(0.05, 0.925, f"ADS1299 · 2000 SPS · 24-bit · 20–450 Hz band-pass + 60 Hz notch · {lim/2:.0f} µV/div · "
             + T("같은 채널·같은 눈금, 편집·평균 없음", "same channel & scale, no editing or averaging"),
             color=txt, fontsize=10, alpha=0.85)
    for i, (h, sg, col, lab) in enumerate(((first, s0, c0, T(f"처음  ({h0['tag'].split('_')[-1]}, {h0['min']:.1f}분)", f"Start ({h0['tag'].split('_')[-1]}, {h0['min']:.1f} min)") if (h0 := first) else ""),
                                            (last, s1, c1, T(f"{last['min']:.0f}분 후 ({last['tag'].split('_')[-1]}, {last['min']:.1f}분)", f"After {last['min']:.0f} min ({last['tag'].split('_')[-1]}, {last['min']:.1f} min)")))):
        t = np.arange(len(sg)) / E.FS
        ax = fig.add_axes((0.05 + i * 0.47, 0.56, 0.43, 0.30))
        scope(ax, (0, t[-1]), (-lim, lim), 1.0, lim / 2)
        ax.plot(t, sg, color=col, lw=0.55, zorder=2)
        ax.axvspan(0, h["hold"] / E.FS, ymin=0.97, ymax=1, color=acc)
        ax.set_title(lab, color=txt, fontsize=13, loc="left", weight="bold")
        ax.set_xlabel(T("시간 (s)", "time (s)"), color=txt, fontsize=9)
        # 200 ms 확대: 유지 구간 가운데
        z0 = int(1.5 * E.FS); z = sg[z0:z0 + int(0.2 * E.FS)]; tz = np.arange(len(z)) / E.FS * 1000
        ax.axvspan(z0 / E.FS, z0 / E.FS + 0.2, color=acc, alpha=0.18, zorder=1)
        bx = fig.add_axes((0.05 + i * 0.47, 0.30, 0.43, 0.20))
        scope(bx, (0, 200), (-lim, lim), 20, lim / 2)
        bx.plot(tz, z, color=col, lw=1.0, zorder=2)
        bx.set_xlabel(T("확대 200 ms (ms)", "zoom 200 ms (ms)"), color=txt, fontsize=9)
    # ③ 스펙트럼
    cx = fig.add_axes((0.05, 0.06, 0.40, 0.18)); cx.set_facecolor(bg)
    for h, col, mm, lb in ((first, c0, m0, T("처음", "start")), (last, c1, m1, T(f"{last['min']:.0f}분 후", f"after {last['min']:.0f} min"))):
        md, f, P = FP.mdf_psd(h["sig"][E.FS // 2:h["hold"] - 600][:, [ch]])
        p = P[:, 0] / P[:, 0].max()
        cx.plot(f, p, color=col, lw=1.6, label=f"{lb}  MDF {mm:.0f} Hz")
        cx.axvline(mm, color=col, ls="--", lw=1)
    cx.annotate("", xy=(m1, 1.08), xytext=(m0, 1.08), arrowprops=dict(arrowstyle="->", color=acc, lw=2))
    cx.text((m0 + m1) / 2, 1.13, f"{m0:.0f} → {m1:.0f} Hz ({100*(m1/m0-1):+.0f} %)", ha="center", color=acc, fontsize=11, weight="bold")
    cx.set_xlim(20, 300); cx.set_ylim(0, 1.3); cx.set_yticks([])
    cx.tick_params(colors=txt, labelsize=9); [sp.set_color(gmaj) for sp in cx.spines.values()]
    cx.set_xlabel(T("주파수 (Hz)", "frequency (Hz)"), color=txt, fontsize=9)
    leg = cx.legend(frameon=False, fontsize=9, loc="upper right"); [t_.set_color(txt) for t_ in leg.get_texts()]
    # ④ 전체 추이
    rows = FP.collect(a.csv)
    tm = np.array([r["min"] for r in rows]); md = np.array([r["mdf"] for r in rows])
    k, b = np.polyfit(tm, md, 1)
    from scipy import stats
    rho, pv = stats.spearmanr(tm, md)
    dx = fig.add_axes((0.55, 0.06, 0.40, 0.18)); dx.set_facecolor(bg)
    dx.scatter(tm, md, s=14, color=c1, alpha=0.8)
    dx.plot(tm, k * tm + b, color=acc, lw=2)
    dx.tick_params(colors=txt, labelsize=9); [sp.set_color(gmaj) for sp in dx.spines.values()]
    dx.set_xlabel(T("세션 경과 (분) — 동작 45회 전체", "session time (min) — all 45 contractions"), color=txt, fontsize=9)
    dx.set_ylabel("MDF (Hz)", color=txt, fontsize=9)
    dx.text(0.98, 0.92, T(f"{k:+.1f} Hz/분  (ρ {rho:+.2f}, p {pv:.0e})", f"{k:+.1f} Hz/min  (ρ {rho:+.2f}, p {pv:.0e})"),
            transform=dx.transAxes, ha="right", va="top", color=acc, fontsize=10.5, weight="bold")
    fig.savefig(a.out, dpi=150, facecolor=bg)
    print("saved", a.out)


if __name__ == "__main__":
    main()
