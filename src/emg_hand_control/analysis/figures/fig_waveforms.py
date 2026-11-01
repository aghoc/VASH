#!/usr/bin/env python3
"""
fig_waveforms.py — 동작별 8채널 실제 근전도 파형 (동작마다 그림 1장, 논문형 흑백)

  python fig_waveforms.py [--csv data\\F2_B2.csv] [--rep -1] [--en] [--font 경로]
출력: data\\fig_wave_<rest|fist|pinch|tripod>.png
 · 같은 녹화·같은 반복, 20–450 Hz 대역통과 + 60 Hz 계열 노치, 편집·평균 없음
 · 모든 그림이 같은 세로 눈금 (채널 간격·µV 축척 동일) → 동작끼리 크기 비교 가능
 · 회색 띠 = 동작 유지 구간 (지시 3 s), 앞뒤 1 s 포함. 반복은 지시 전 1 s 가 가장 조용한 것을 자동 선택 (--rep 로 고정 가능)
 · 휴식 그림 = 휴식 구간 중 가장 조용한 3 s
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=f"{D}/F2_B2.csv"); ap.add_argument("--rep", type=int, default=-1, help="-1 = 동작마다 지시 전이 가장 조용한 반복 자동 선택")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    from scipy import signal
    x, lab, rep, meta = E.load_csv(a.csv)
    y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
    g = meta.get("gestures", "").replace("_", " ").split("|")
    names = [L_("휴식", "Rest")] + g
    files = ["rest"] + ["fist", "pinch", "tripod"][:len(g)]
    segs = E.segments(lab)
    clips = {}
    def rms(a_, b_): return float(np.sqrt(np.mean(y[a_:b_] ** 2)))
    for c in range(1, len(names)):
        cand = [(s, t) for s, t, l in segs if l == c and (a.rep < 0 or rep[s] == a.rep)]
        if a.rep < 0:   # 대표 반복 자동 선택: 지시 전 1 s 가 가장 조용한(휴식이 잘 된) 반복
            cand.sort(key=lambda st: rms(st[0] - E.FS, st[0]) / rms(st[0], st[1]))
        s0, s1 = cand[0]
        clips[c] = (s0 - E.FS, s1 + E.FS, s0, s1)
        print(f"{names[c]}: 반복 {rep[s0]}  (지시 전 1 s RMS / 유지 RMS = {rms(s0 - E.FS, s0) / rms(s0, s1):.2f})")
    # 휴식: 녹화 시작 휴식 (꼭 가만히 있도록 지시한 구간), 필터 안정 후 5 s
    best = None
    for s_, t_, l_ in segs:      # 휴식 구간 안(앞뒤 0.5 s 제외)에서 가장 조용한 3 s
        if l_ != 0: continue
        a0, a1 = max(s_ + E.FS // 2, int(1.5 * E.FS)), t_ - E.FS // 2
        for k in range(a0, a1 - 3 * E.FS + 1, E.FS // 10):
            v = rms(k, k + 3 * E.FS)
            if best is None or v < best[0]: best = (v, k)
    clips[0] = (best[1], best[1] + 3 * E.FS, None, None)
    print(f"휴식: 가장 조용한 3 s (RMS {best[0]:.1f} µV)")
    # 공통 채널 간격: 모든 동작 구간의 99.5 백분위
    allv = np.concatenate([np.abs(y[c0:c1]).ravel() for c0, c1, _, _ in clips.values()])
    sp = float(np.ceil(np.percentile(allv, 99.5) * 2 / 50) * 50)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    for c in range(len(names)):
        c0, c1, h0, h1 = clips[c]
        t = (np.arange(c0, c1) - c0) / E.FS
        fig, ax = plt.subplots(figsize=(3.4, 4.2), dpi=200, facecolor="white")
        if h0 is not None:
            ax.axvspan((h0 - c0) / E.FS, (h1 - c0) / E.FS, color="#e6e6e6", zorder=0)
        for ch in range(E.NCH):
            off = -ch * sp
            ax.plot(t, y[c0:c1, ch] + off, color="black", lw=0.35)
        ax.set_yticks([-ch * sp for ch in range(E.NCH)]); ax.set_yticklabels([f"CH{ch+1}" for ch in range(E.NCH)])
        ax.set_ylim(-(E.NCH - 0.4) * sp, 0.6 * sp); ax.set_xlim(0, t[-1])
        ax.set_xlabel(L_("시간 (s)", "Time (s)")); ax.spines[["right", "top", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # 축척 막대
        ax.set_xlim(0, t[-1] * 1.12); ax.spines["bottom"].set_bounds(0, t[-1]); ax.set_xticks(range(int(t[-1]) + 1))
        xb = t[-1] * 1.06; yb = -(E.NCH - 1) * sp - sp / 4
        ax.plot([xb, xb], [yb, yb + sp / 2], color="black", lw=1.2)
        ax.text(xb, yb - sp * 0.12, f"{sp/2:.0f} µV", ha="center", va="top", fontsize=7)
        ax.set_title(names[c], fontsize=10.5)
        fig.tight_layout(); p = f"{D}/fig_wave_{files[c]}.png"; fig.savefig(p, dpi=200, facecolor="white"); plt.close(fig)
        print("saved", p)
    print(f"채널 간격 {sp:.0f} µV (모든 그림 동일), 녹화 {os.path.basename(a.csv)}")


if __name__ == "__main__":
    main()
