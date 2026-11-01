#!/usr/bin/env python3
"""
paper_figure3.py — 포스터용 논문 형식 그림: 동작별 분류 정확도 개선 (기존 vs 제안)

  python paper_figure3.py --out data\\fig_per_motion.png        (영어: --en)

S01 R2(안정 후)로 학습 → 밴드를 다시 찬 R3 를 실시간 재생(50 ms 판단). 반복 5회.
판정: 지시 후 0.5 s 지난 구간에서, 서보 각도가 10° 넘은 손가락(가장 큰 것) = 예측 동작, 없으면 휴식.
  기존 = 일반 학습 + 다수결·히스테리시스 제어
  제안 = 세기 증강 학습 + CUSUM · HMM · 순응 예측 · 3차 평활 궤적
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402
from paper_figure import ds, feats, ACT, D  # noqa: E402


def run():
    Xw, Y, R = ds([f"{D}/20261024_S01_R2.csv"]); zc = np.concatenate(Xw[Y == 0]).std(0)
    eng = E.RealtimeEngine(E.fit_model(feats(Xw, zc), Y, ACT, zc)); base = E.ServoController(eng)
    Xa, Ya, Ra = E.augment_force(Xw, Y, R, (0.6, 0.8))
    mdl = M.MathModel(Xa, Ya, Ra, ACT, True); prop = M.MathController(mdl)
    x, lab, rep, _ = E.load_csv(f"{D}/20261024_S01_R3.csv")
    T, A, B = [], [], []
    for k0 in range(0, len(x), 40):
        for cls, prob, inten, idx, p in eng.push(x[k0:k0 + 40]):
            T.append((k0 + idx) / E.FS); A.append(base.update(cls, p, eng.last_rms))
            B.append(prop.update(mdl.proba(eng.buf[None])[0], eng.last_rms))
    T = np.array(T); si = np.clip((T * E.FS).astype(int), 0, len(lab) - 1)
    truth, trep = lab[si], rep[si]
    since = np.zeros(len(T)); last = -1e9; prev = None
    for i, (tt, l) in enumerate(zip(T, truth)):
        if l != prev: last, prev = tt, l
        since[i] = tt - last
    sel = (T > 2) & (since >= 0.5) & (trep >= 0)
    out = {}
    for key, AA in (("base", np.array(A)), ("prop", np.array(B))):
        pred = np.where(AA.max(1) > 10, AA.argmax(1) + 1, 0)
        conf = np.zeros((4, 4))
        for t, p in zip(truth[sel], pred[sel]): conf[t, p] += 1
        per_rep = {c: [100 * np.mean(pred[sel & (trep == r) & (truth == c)] == c)
                       for r in sorted(set(int(v) for v in trep[sel]))] for c in range(4)}
        out[key] = dict(conf=conf / conf.sum(1, keepdims=True) * 100, rep=per_rep,
                        overall=100 * np.mean(pred[sel] == truth[sel]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{D}/fig_per_motion.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    res = run()
    L = (lambda ko, en: en if a.en else ko)
    names = [L("휴식", "Rest"), L("엄지", "Thumb"), L("검지", "Index"), L("중지", "Middle")]
    for k in ("base", "prop"):
        print(k, f"전체 {res[k]['overall']:.1f}%", {names[c]: f"{np.mean(v):.1f}±{np.std(v):.1f}" for c, v in res[k]["rep"].items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(10.5, 6.6), dpi=200, facecolor="white")

    # (a) 동작별 정확도
    ax = fig.add_axes((0.07, 0.56, 0.86, 0.34))
    xs = np.arange(5)
    for i, (key, fc, dot) in enumerate((("base", "white", "black"), ("prop", "#404040", "#9e9e9e"))):
        vals = [res[key]["rep"][c] for c in range(4)] + [None]
        for j in range(4):
            g = vals[j]; m, s = np.mean(g), np.std(g); x = xs[j] + (i - 0.5) * 0.36
            ax.bar(x, m, width=0.34, color=fc, edgecolor="black", linewidth=0.8,
                   label=(L("기존", "Baseline") if i == 0 else L("제안", "Proposed")) if j == 0 else None)
            ax.errorbar(x, m, yerr=s, color="black", capsize=3, lw=0.9)
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(g)), g, s=8, color=dot, zorder=3)
            ax.text(x, max(m + s, max(g)) + 2.5, f"{m:.1f}", ha="center", fontsize=8.2)
        o = res[key]["overall"]; x = xs[4] + (i - 0.5) * 0.36
        ax.bar(x, o, width=0.34, color=fc, edgecolor="black", linewidth=0.8, hatch="///" if i == 0 else None)
        ax.text(x, o + 2.5, f"{o:.1f}", ha="center", fontsize=8.2, weight="bold")
    for j in range(4):
        d = np.mean(res["prop"]["rep"][j]) - np.mean(res["base"]["rep"][j])
        ax.text(xs[j], 119, f"{d:+.1f}%p", ha="center", fontsize=9.5, weight="bold", color="black" if d > 0 else "#777")
    d = res["prop"]["overall"] - res["base"]["overall"]
    ax.text(xs[4], 119, f"{d:+.1f}%p", ha="center", fontsize=9.5, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names + [L("전체", "Overall")], fontsize=10)
    ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.spines["left"].set_bounds(0, 100)
    ax.spines[["right", "top"]].set_visible(False); ax.set_ylabel(L("분류 정확도 (%)", "Accuracy (%)"))
    ax.axvline(3.5, color="#999", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=2)
    ax.text(-0.06, 1.04, "a", transform=ax.transAxes, fontsize=12, weight="bold")
    ax.set_title(L("동작별 분류 정확도", "Per-motion accuracy"), fontsize=10, pad=14, loc="left", x=0.0)

    # (b) 혼동 행렬
    for i, (key, ttl) in enumerate((("base", L("기존", "Baseline")), ("prop", L("제안", "Proposed")))):
        bx = fig.add_axes((0.07 + i * 0.30, 0.07, 0.22, 0.34))
        C = res[key]["conf"]
        bx.imshow(C, cmap="Greys", vmin=0, vmax=100)
        for r in range(4):
            for c in range(4):
                bx.text(c, r, f"{C[r, c]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if C[r, c] > 55 else "black", weight="bold" if r == c else None)
        bx.set_xticks(range(4)); bx.set_xticklabels(names, fontsize=8); bx.set_yticks(range(4)); bx.set_yticklabels(names, fontsize=8)
        bx.set_xlabel(L("예측", "Predicted"), fontsize=8.5)
        if i == 0:
            bx.set_ylabel(L("실제", "True"), fontsize=8.5)
            bx.text(-0.30, 1.08, "b", transform=bx.transAxes, fontsize=12, weight="bold")
        bx.set_title(f"{ttl}  ({res[key]['overall']:.1f} %)", fontsize=9.5)
    cap = L("Fig. 3. 동작별 분류 성능 (평균 ± 표준편차, 점 = 반복별 값, n = 5).\n"
            "안정화 후 녹화(R2)로 학습하고 밴드를 다시 찬 녹화(R3)를\n실시간 재생(50 ms 판단). 지시 0.5 s 이후 구간,\n"
            "각도 10° 넘은 손가락을 예측 동작으로 판정.\n"
            "(a) 동작별·전체 정확도와 향상 폭.\n(b) 혼동 행렬 (행 = 실제, 값 = 행 기준 %).\n"
            "기존 = 일반 학습 + 다수결·히스테리시스 제어.\n"
            "제안 = 세기 증강 학습 + CUSUM 시작 검출 · HMM ·\n순응 예측 · 3차 평활 궤적.\n"
            "참가자 1명(비절단자), ADS1299 8채널, 2 kSPS.",
            "Fig. 3. Per-motion classification (mean ± SD; dots = repetitions, n = 5).\n"
            "Trained on settled session R2, real-time replay of\nre-worn session R3 (50 ms decisions). Scored 0.5 s after cue;\n"
            "predicted motion = finger whose angle exceeds 10°.\n"
            "(a) Per-motion and overall accuracy with gains.\n(b) Confusion matrices (rows = true, row-normalized %).\n"
            "Baseline = standard training + majority vote / hysteresis.\n"
            "Proposed = force-augmented training + CUSUM onset, HMM,\nconformal prediction, 3rd-order smoothing.\n"
            "One able-bodied participant, ADS1299 8 ch, 2 kSPS.")
    fig.text(0.66, 0.07, cap, fontsize=7.6, va="bottom", linespacing=1.55)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
