#!/usr/bin/env python3
"""
paper_figure2.py — 포스터용 논문 형식 그림: 동작 분류·제어 개선 (전극 배치 제외)

  python paper_figure2.py --out data\\fig_improvement.png        (영어: --en)

모든 값은 녹화 파일에서 다시 계산 (S01 R2 학습 → R3 재생, 반복 5회 = 점 5개).
  기존 = 일반 학습 + 기존 제어기 (다수결·히스테리시스)
  제안 = 세기 증강 학습 + 수학 파이프라인 (CUSUM · HMM · 순응 예측 · 3차 평활 궤적)
 (a) 손가락별 분류 정확도   (b) 오류율 (휴식 오작동, 다른 손가락)   (c) 재착용 오류율   (d) 쥐기 오류율
오류율로 나타내고 상대 감소율(%)을 표시 — 평균값은 그대로, 축은 0 부터.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402
import emg_math as M  # noqa: E402
from paper_figure import ds, feats, fold_acc, ACT, D  # noqa: E402


def hold_metrics(T, A, truth, trep, segs, rep, settle=0.5, thr=10.0):
    """반복마다: 손가락별 정답률, 휴식 오작동, 다른 손가락 움직임"""
    since = np.zeros(len(T)); last = -1e9; prev = None
    for i, (tt, l) in enumerate(zip(T, truth)):
        if l != prev: last, prev = tt, l
        since[i] = tt - last
    out = []
    for r in sorted(set(int(v) for v in rep if v >= 0)):
        sel = (trep == r) & (T > 2) & (since >= settle)
        on = A > thr
        d = {}
        for f in (1, 2, 3):
            m = sel & (truth == f)
            ok = on[m, f - 1] & ~np.delete(on[m], f - 1, axis=1).any(1)
            d[f] = 100 * ok.mean() if m.any() else np.nan
        mr = sel & (truth == 0); d["rest"] = 100 * on[mr].any(1).mean()
        mh = sel & (truth > 0)
        idx = np.flatnonzero(mh)
        d["wrong"] = 100 * np.mean([np.delete(on[i], truth[i] - 1).any() for i in idx])
        out.append(d)
    return out


def compute():
    res = {}
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
    segs = None
    res["base"] = hold_metrics(T, np.array(A), lab[si], rep[si], segs, rep)
    res["prop"] = hold_metrics(T, np.array(B), lab[si], rep[si], segs, rep)
    # 재착용 (창 단위 손가락 오류율)
    X3, Y3, R3 = ds([f"{D}/20261024_S01_R3.csv"])
    for key, f in (("rewear_R1", "R1"), ("rewear_R2", "R2")):
        X, Yk, _ = ds([f"{D}/20261024_S01_{f}.csv"]); z = np.concatenate(X[Yk == 0]).std(0)
        F = feats(X, z); mu, sd = F.mean(0), F.std(0) + 1e-6
        W, b = E.fit_lda((F - mu) / sd, Yk, 4)
        P = E.lda_proba((feats(X3, z) - mu) / sd, W, b).argmax(1)
        res[key] = [100 - 100 * np.mean(P[(R3 == r) & (Y3 > 0)] == Y3[(R3 == r) & (Y3 > 0)])
                    for r in sorted(set(R3.tolist())) if r >= 0]
    # 쥐기 (세게) 오류율
    X, Yg, Rg = ds([f"{D}/20261024_G1.csv"]); z = np.concatenate(X[Yg == 0]).std(0); F = feats(X, z)
    firm = (Rg % 2 == 1) & (Yg > 0)
    res["grip5"] = [100 - v for v in fold_acc(F, Yg, Rg, 6, keep=firm)]
    res["grip4"] = [100 - v for v in fold_acc(F, Yg, Rg, 5, keep=firm, mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4})]
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{D}/fig_improvement.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    res = compute()
    g = lambda key, k: [d[k] for d in res[key]]
    for k in (1, 2, 3, "rest", "wrong"):
        b, p = g("base", k), g("prop", k)
        print(f"{k}: 기존 {np.nanmean(b):.1f}±{np.nanstd(b):.1f}  제안 {np.nanmean(p):.1f}±{np.nanstd(p):.1f}")
    for k in ("rewear_R1", "rewear_R2", "grip5", "grip4"):
        print(k, f"{np.mean(res[k]):.1f}±{np.std(res[k]):.1f}", [round(v, 1) for v in res[k]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9, "axes.linewidth": 0.8})
    L = (lambda ko, en: en if a.en else ko)
    rng = np.random.default_rng(3)
    W0, W1 = "white", "#404040"

    def bar(ax, x, vals, fc, w=0.36, dot="black"):
        m, s = np.nanmean(vals), np.nanstd(vals)
        ax.bar(x, m, width=w, color=fc, edgecolor="black", linewidth=0.8)
        ax.errorbar(x, m, yerr=s, color="black", capsize=3.5, lw=0.9)
        ax.scatter(x + rng.uniform(-w / 4, w / 4, len(vals)), vals, s=8, color=dot, zorder=3)
        ax.text(x, max(m + s, np.nanmax(vals)) + 2.5, f"{m:.1f}", ha="center", fontsize=8.3)
        return m, s

    def bracket(ax, x0, x1, y, txt):
        ax.plot([x0, x0, x1, x1], [y - 2, y, y, y - 2], color="black", lw=0.8)
        ax.text((x0 + x1) / 2, y + 1.2, txt, ha="center", fontsize=8.8, weight="bold")

    def style(ax, ylabel, letter, title, top=100, lx=-0.2, head=1.22):
        ax.set_ylim(0, top * head); ax.set_yticks(range(0, top + 1, 20 if top == 100 else 10))
        ax.spines["left"].set_bounds(0, top); ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9.5, pad=6)
        ax.text(lx, 1.05, letter, transform=ax.transAxes, fontsize=12, weight="bold")

    fig = plt.figure(figsize=(10.5, 6.8), dpi=200, facecolor="white")
    # (a) 손가락별 정확도
    ax = fig.add_axes((0.07, 0.56, 0.40, 0.33))
    names = [L("엄지", "Thumb"), L("검지", "Index"), L("중지", "Middle")]
    for j, f in enumerate((1, 2, 3)):
        bar(ax, j - 0.2, g("base", f), W0); bar(ax, j + 0.2, g("prop", f), W1, dot="#9e9e9e")
    mb = np.nanmean([np.nanmean(g("base", f)) for f in (1, 2, 3)]); mp = np.nanmean([np.nanmean(g("prop", f)) for f in (1, 2, 3)])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(names)
    style(ax, L("분류 정확도 (%)", "Accuracy (%)"), "a", lx=-0.12, title= L(f"손가락별 분류 정확도 (평균 {mb:.1f} → {mp:.1f} %)",
                                                         f"Per-finger accuracy (mean {mb:.1f} → {mp:.1f} %)"))
    ax.bar(0, 0, color=W0, edgecolor="black", label=L("기존", "Baseline")); ax.bar(0, 0, color=W1, edgecolor="black", label=L("제안", "Proposed"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2)
    # (b) 오류율
    ax = fig.add_axes((0.58, 0.56, 0.38, 0.33))
    for j, (k, lb) in enumerate((("rest", L("휴식 중 오작동", "False motion at rest")), ("wrong", L("다른 손가락 움직임", "Wrong finger moves")))):
        mb_, sb = bar(ax, j - 0.2, g("base", k), W0); mp_, sp = bar(ax, j + 0.2, g("prop", k), W1, dot="#9e9e9e")
        top = max(mb_ + sb, np.nanmax(g("base", k))) + 10
        bracket(ax, j - 0.2, j + 0.2, top, f"−{100 * (1 - mp_ / mb_):.0f} %")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L("휴식 중 오작동", "False motion\nat rest"), L("다른 손가락 움직임", "Wrong finger\nmoves")])
    style(ax, L("오류율 (%)", "Error rate (%)"), "b", L("오작동 (낮을수록 좋음)", "Errors (lower is better)"), top=60, head=1.42)
    # (c) 재착용 오류율
    ax = fig.add_axes((0.07, 0.10, 0.18, 0.30))
    m0, s0 = bar(ax, 0, res["rewear_R1"], W0, w=0.6); m1, s1 = bar(ax, 1, res["rewear_R2"], W1, w=0.6, dot="#9e9e9e")
    bracket(ax, 0, 1, max(m0 + s0, max(res["rewear_R1"])) + 15, f"×{m0 / max(m1, 0.1):.0f} ↓")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L("착용 직후\n학습", "Trained at\ndonning"), L("안정화 후\n학습", "Trained after\nsettling")], fontsize=8.3)
    style(ax, L("오류율 (%)", "Error rate (%)"), "c", L("재착용 시험", "Re-wear test"), head=1.32)
    # (d) 쥐기 오류율
    ax = fig.add_axes((0.36, 0.10, 0.18, 0.30))
    m0, s0 = bar(ax, 0, res["grip5"], W0, w=0.6); m1, s1 = bar(ax, 1, res["grip4"], W1, w=0.6, dot="#9e9e9e")
    bracket(ax, 0, 1, max(m0 + s0, max(res["grip5"])) + 7, f"−{100 * (1 - m1 / m0):.0f} %")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L("5종", "5 grips"), L("4종", "4 grips")], fontsize=8.3)
    style(ax, L("오류율 (%)", "Error rate (%)"), "d", L("쥐기 분류 (세게)", "Grip classification (firm)"), top=40, head=1.3)
    cap = L("Fig. 3. 동작 분류·제어 개선 (평균 ± 표준편차, 점 = 반복별 값, n = 5; d 는 n = 3).\n"
            "(a, b) 안정화 후 녹화(R2)로 학습하고 밴드를 다시 찬 녹화(R3)를 실시간 재생(50 ms 판단).\n"
            "    기존 = 일반 학습 + 다수결·히스테리시스 제어,\n"
            "    제안 = 세기 증강 학습 + CUSUM 시작 검출 · HMM · 순응 예측 · 3차 평활 궤적.\n"
            "(c) 착용 직후(R1) 또는 안정화 후(R2) 녹화로 학습 → R3 손가락 오류율 (창 단위).\n"
            "(d) 쥐기 녹화의 세게(60~70 %) 반복, 4종 = 집기·세 손가락 집기 통합.\n"
            "참가자 1명(비절단자), 하루 녹화. ADS1299 8채널, 2 kSPS, 20–450 Hz.",
            "Fig. 3. Classification and control improvements (mean ± SD; dots = repetitions, n = 5; d: n = 3).\n"
            "(a, b) Trained on settled session R2, real-time replay of re-worn session R3 (50 ms decisions).\n"
            "    Baseline = standard training + majority vote / hysteresis;\n"
            "    Proposed = force-augmented training + CUSUM onset, HMM, conformal prediction, 3rd-order smoothing.\n"
            "(c) Finger error on R3 when trained on R1 (just donned) vs R2 (settled), window level.\n"
            "(d) Firm repetitions (60–70 %) of grip session; 4 grips = pinch and tripod merged.\n"
            "One able-bodied participant, single day. ADS1299, 8 ch, 2 kSPS, 20–450 Hz.")
    fig.text(0.60, 0.10, cap, fontsize=7.6, va="bottom", linespacing=1.6)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
