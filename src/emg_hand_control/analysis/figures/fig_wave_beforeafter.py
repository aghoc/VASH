#!/usr/bin/env python3
"""
fig_wave_beforeafter.py — 동작별 8채널 파형: 쉰 상태(피로 전) vs 피로 후, 나란히 (동작마다 그림 1장, 논문형 흑백)

  python fig_wave_beforeafter.py [--pre data\\F2_B2.csv] [--post data\\F2_T3.csv] [--en] [--font 경로]
출력: data\\fig_wave_ba_<rest|fist|pinch|tripod>[_T2].png  (T3 가 아니면 파일명 끝에 단계 표시)
 · 같은 착용, 같은 필터(20–450 Hz + 60 Hz 계열 노치), 편집·평균 없음
 · 네 그림 × 두 칸 모두 같은 세로 눈금 → 크기 직접 비교 가능
 · 각 녹화에서 지시 전 1 s 가 가장 조용한 반복을 대표로 자동 선택 (fig_waveforms.py 와 같은 규칙)
 · 칸 아래 숫자 = 그 반복 유지 구간의 채널 에너지 가중 중앙 주파수 (MDF)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import fatigue_experiment as FX  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def clips_of(path):
    from scipy import signal
    x, lab, rep, meta = E.load_csv(path)
    y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
    segs = E.segments(lab); k = int(lab.max()) + 1
    def rms(a, b): return float(np.sqrt(np.mean(y[a:b] ** 2)))
    out = {}
    for c in range(1, k):
        cand = sorted([(s, t) for s, t, l in segs if l == c and s > E.FS], key=lambda st: rms(st[0] - E.FS, st[0]) / rms(*st))
        s0, s1 = cand[0]; out[c] = (s0 - E.FS, s1 + E.FS, s0, s1, int(rep[s0]))
    best = None
    for s, t, l in segs:
        if l != 0: continue
        for a in range(max(s + E.FS // 2, int(1.5 * E.FS)), t - E.FS // 2 - 3 * E.FS + 1, E.FS // 10):
            v = rms(a, a + 3 * E.FS)
            if best is None or v < best[0]: best = (v, a)
    out[0] = (best[1], best[1] + 3 * E.FS, None, None, -1)
    return y, out, meta


def mdf(seg):
    f, P = FX.psd(seg); E_ = P.sum(0)
    return float(np.sum(E_ * FX.mdf_from_psd(f, P)) / np.sum(E_))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pre", default=f"{D}/F2_B2.csv"); ap.add_argument("--post", default=f"{D}/F2_T3.csv")
    ap.add_argument("--post-label", default=None); ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    L_ = (lambda ko, e: e if a.en else ko)
    yA, cA, meta = clips_of(a.pre); yB, cB, _ = clips_of(a.post)
    g = meta.get("gestures", "").replace("_", " ").split("|")
    names = [L_("휴식", "Rest")] + g; files = ["rest", "fist", "pinch", "tripod"]
    ptag = os.path.basename(a.post).split("_")[-1].split(".")[0]; suf = "" if ptag == "T3" else f"_{ptag}"
    tag = a.post_label or L_("피로 후", "Fatigued") + f" ({os.path.basename(a.post).split('_')[-1].split('.')[0]})"
    allv = np.concatenate([np.abs(y[c[0]:c[1]]).ravel() for y, cc in ((yA, cA), (yB, cB)) for c in cc.values()])
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
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 4.4), dpi=200, facecolor="white", sharey=True)
        info = []
        for j, (ax, y, cc, lb) in enumerate(zip(axs, (yA, yB), (cA, cB), (L_("쉰 상태", "Rested"), tag))):
            c0, c1, h0, h1, r = cc[c]; t = (np.arange(c0, c1) - c0) / E.FS
            if h0 is not None:
                ax.axvspan((h0 - c0) / E.FS, (h1 - c0) / E.FS, color="#e6e6e6", zorder=0)
            for ch in range(E.NCH):
                ax.plot(t, y[c0:c1, ch] - ch * sp, color="black", lw=0.35)
            seg = y[h0:h1] if h0 is not None else y[c0:c1]
            if h0 is not None: seg = seg[E.FS // 2:]
            m = mdf(seg); info.append(m)
            ax.set_xlim(0, t[-1] * (1.12 if j == 1 else 1.0)); ax.spines["bottom"].set_bounds(0, t[-1]); ax.set_xticks(range(int(t[-1]) + 1))
            ax.set_xlabel(L_("시간 (s)", "Time (s)")); ax.spines[["right", "top", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{lb}\nMDF {m:.0f} Hz", fontsize=9)
        axs[0].set_yticks([-ch * sp for ch in range(E.NCH)]); axs[0].set_yticklabels([f"CH{ch+1}" for ch in range(E.NCH)])
        axs[0].set_ylim(-(E.NCH - 0.4) * sp, 0.6 * sp)
        t1 = axs[1].get_xlim()[1]; xb = t1 * 0.95; yb = -(E.NCH - 1) * sp - sp / 4
        axs[1].plot([xb, xb], [yb, yb + sp / 2], color="black", lw=1.2)
        axs[1].text(xb, yb - sp * 0.12, f"{sp/2:.0f} µV", ha="center", va="top", fontsize=7)
        fig.suptitle(names[c], fontsize=10.5)
        fig.tight_layout(); p = f"{D}/fig_wave_ba_{files[c]}{suf}.png"; fig.savefig(p, dpi=200, facecolor="white"); plt.close(fig)
        print(f"{names[c]:14s} 반복 {cA[c][4]} → {cB[c][4]}   MDF {info[0]:.1f} → {info[1]:.1f} Hz ({100*(info[1]/info[0]-1):+.1f} %)   {p}")
    print(f"채널 간격 {sp:.0f} µV (모든 칸 동일)")


if __name__ == "__main__":
    main()
