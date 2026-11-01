#!/usr/bin/env python3
"""
fig_fatigue_dramatic.py — 60 초 유지 주먹의 '처음 vs 끝' (같은 수축·같은 채널) — 피로 변화가 가장 큰 예시 (논문형 흑백)

  python fig_fatigue_dramatic.py [--files data\\F2_sus1.csv data\\F2_sus2.csv data\\F2_sus3.csv] [--en] [--font 경로]
출력: data\\fig_fatigue_dramatic.png
 · 모든 유지 수축 × 세기 상위 3채널 가운데 중앙 주파수 감소가 가장 큰 (수축, 채널)을 자동 선택 → '가장 극적인 예시'임을 캡션에 밝힐 것
 · 처음 = 시작 2~6 s, 끝 = 종료 5~1 s 전 (같은 4 s 길이), 편집·평균 없음
 (a) 4 s 원파형 (같은 눈금)  (b) 200 ms 확대  (c) 스펙트럼 + 중앙 주파수
 화면에 전체 유지 수축의 평균 변화도 같이 출력 (캡션용)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import fatigue_experiment as FX  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[f"{D}/F2_sus{i}.csv" for i in (1, 2, 3)])
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    from scipy import signal
    cands, allchg = [], []
    for fn in a.files:
        x, lab, rep, _ = E.load_csv(fn)
        y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
        for s, t, l in E.segments(lab):
            if l == 0 or t - s < 30 * E.FS: continue
            A = y[s + 2 * E.FS:s + 6 * E.FS]; B = y[t - 5 * E.FS:t - E.FS]
            f, PA = FX.psd(A); _, PB = FX.psd(B)
            mA, mB = FX.mdf_from_psd(f, PA), FX.mdf_from_psd(f, PB)
            rA = np.sqrt((A ** 2).mean(0))
            top = np.argsort(rA)[-3:]
            allchg.append(np.average(100 * (mB / mA - 1), weights=PA.sum(0)))
            for ch in top:
                cands.append((100 * (mB[ch] / mA[ch] - 1), os.path.basename(fn), int(rep[s]), int(ch), A[:, ch], B[:, ch], f, PA[:, ch], PB[:, ch], mA[ch], mB[ch]))
    cands.sort(key=lambda c: c[0])
    chg, fn, r, ch, A, B, f, PA, PB, mA, mB = cands[0]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    print(f"선택: {fn} 반복 {r} CH{ch+1}  MDF {mA:.1f} → {mB:.1f} Hz ({chg:+.1f} %),  RMS {rms(A):.1f} → {rms(B):.1f} µV")
    print(f"참고(캡션용): 유지 {len(allchg)}회 전체 채널 에너지 가중 평균 MDF 변화 {np.mean(allchg):+.1f} ± {np.std(allchg):.1f} %")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(10.5, 3.6), dpi=200, facecolor="white")
    lim = float(np.ceil(np.percentile(np.abs(np.r_[A, B]), 99.8) / 25) * 25)
    t = np.arange(len(A)) / E.FS
    lbA, lbB = L_("처음 (2–6 s)", "Start (2–6 s)"), L_("끝 (마지막 5–1 s)", "End (last 5–1 s)")
    for i, (v, lb) in enumerate(((A, lbA), (B, lbB))):
        ax = fig.add_axes((0.06, 0.58 - i * 0.42, 0.30, 0.30))
        ax.plot(t, v, color="black", lw=0.35); ax.set_xlim(0, 4); ax.set_ylim(-lim, lim); ax.set_yticks([-lim, 0, lim])
        ax.spines[["right", "top"]].set_visible(False); ax.set_title(lb, fontsize=9, loc="left")
        ax.axvspan(1.9, 2.1, color="#d0d0d0", zorder=0)
        if i == 1: ax.set_xlabel(L_("시간 (s)", "Time (s)"))
        else: ax.set_xticklabels([])
        ax.set_ylabel("µV", fontsize=8)
    fig.text(0.015, 0.93, "a", fontsize=12, weight="bold")
    z0, z1 = int(1.9 * E.FS), int(2.1 * E.FS); tz = np.arange(z1 - z0) / E.FS * 1000
    for i, (v, lb) in enumerate(((A, lbA), (B, lbB))):
        bx = fig.add_axes((0.42, 0.58 - i * 0.42, 0.22, 0.30))
        bx.plot(tz, v[z0:z1], color="black", lw=0.9); bx.set_xlim(0, 200); bx.set_ylim(-lim, lim); bx.set_yticks([])
        bx.spines[["right", "top", "left"]].set_visible(False); bx.set_title(lb, fontsize=9, loc="left")
        if i == 1: bx.set_xlabel(L_("확대 200 ms (ms)", "Zoom 200 ms (ms)"))
        else: bx.set_xticklabels([])
    fig.text(0.39, 0.93, "b", fontsize=12, weight="bold")
    cx = fig.add_axes((0.72, 0.16, 0.26, 0.72))
    n = max(PA.max(), PB.max())
    cx.plot(f, PA / n, color="#9e9e9e", lw=1.5, label=f"{L_('처음', 'Start')}  {mA:.0f} Hz")
    cx.plot(f, PB / n, color="black", lw=1.5, label=f"{L_('끝', 'End')}  {mB:.0f} Hz")
    cx.axvline(mA, color="#9e9e9e", ls="--", lw=1); cx.axvline(mB, color="black", ls="--", lw=1)
    cx.annotate("", xy=(mB, 1.05), xytext=(mA, 1.05), arrowprops=dict(arrowstyle="->", lw=1.4))
    cx.text((mA + mB) / 2, 1.09, f"{chg:+.0f} %", ha="center", fontsize=10, weight="bold")
    cx.set_xlim(20, 300); cx.set_ylim(0, 1.2); cx.set_yticks([])
    cx.set_xlabel(L_("주파수 (Hz)", "Frequency (Hz)")); cx.set_ylabel(L_("파워 (정규화)", "Power (norm.)"))
    cx.spines[["right", "top"]].set_visible(False); cx.legend(frameon=False, fontsize=8, title=L_("중앙 주파수", "Median freq."), title_fontsize=8)
    fig.text(0.685, 0.93, "c", fontsize=12, weight="bold")
    fig.text(0.06, 0.95, L_(f"60 초 주먹 유지 (CH{ch+1})", f"60 s sustained fist (CH{ch+1})"), fontsize=9.5)
    p = f"{D}/fig_fatigue_dramatic.png"; fig.savefig(p, dpi=200, facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
