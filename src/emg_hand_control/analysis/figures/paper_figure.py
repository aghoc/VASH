#!/usr/bin/env python3
"""
paper_figure.py — 논문 형식 성능 그림 (흑백, 평균 ± 표준편차 + 반복별 점, 그림 설명 포함)

  python paper_figure.py --out data\\fig_performance.png          (영어: --en)

모든 값을 녹화 파일에서 다시 계산함 (그림과 숫자가 항상 일치).
 (a) 전극 배치: 손가락 4동작, 반복 단위 교차검증 — 반복(fold)마다 정확도
 (b) 재착용: R1 또는 R2 로 학습 → R3(재착용) 시험, 반복마다 손가락 정확도
 (c) 쥐기 종류: G1 세게 반복, 5종 vs 4종(집기 통합), 반복마다 동작 정확도
 (d) 서보 제어: R2 학습 → R3 재생, 기존 vs 수학 파이프라인 (휴식 오작동·다른 손가락·맞는 손가락)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402

ACT = list(range(E.NCH))
D = "data"


def feats(X, zc):
    return np.array([E.features(w, zc) for w in X])


def fold_acc(F, Y, R, k, keep=None, mapping=None):
    """반복(rep) 하나씩 빼고 학습 → 그 반복 정확도 목록. keep: 시험에 쓸 창 마스크, mapping: 라벨 합치기"""
    if mapping is not None:
        Y = np.array([mapping[v] for v in Y]); k = max(mapping.values()) + 1
    out = []
    for r in sorted(set(R.tolist())):
        if r < 0 or r % 1000 == 999:
            continue
        tr, te = R != r, R == r
        if keep is not None: te &= keep
        if not te.any(): continue
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        W, b = E.fit_lda((F[tr] - mu) / sd, Y[tr], k)
        out.append(float(np.mean(E.lda_proba((F[te] - mu) / sd, W, b).argmax(1) == Y[te])) * 100)
    return out


def ds(files, reps=None):
    X, Y, R = E.build_dataset(files, ACT, reps)
    return np.array(X), np.array(Y), np.array(R)


def compute():
    res = {}
    # (a)
    for key, files in (("palm", [f"{D}/../A1_flexor.csv"]), ("back", [f"{D}/../B2_extensor.csv"]),
                       ("band", [f"{D}/20261024_S01_R2.csv", f"{D}/20261024_S01_R3.csv"])):
        X, Y, R = ds(files); zc = np.concatenate(X[Y == 0]).std(0)
        res[key] = fold_acc(feats(X, zc), Y, R, 4)
    # (b) R1 또는 R2 학습 → R3 반복별 손가락 정확도
    X3, Y3, R3 = ds([f"{D}/20261024_S01_R3.csv"])
    for key, f in (("rewear_R1", "R1"), ("rewear_R2", "R2")):
        X, Y, R = ds([f"{D}/20261024_S01_{f}.csv"]); zc = np.concatenate(X[Y == 0]).std(0)
        F = feats(X, zc); mu, sd = F.mean(0), F.std(0) + 1e-6
        W, b = E.fit_lda((F - mu) / sd, Y, 4)
        P = E.lda_proba((feats(X3, zc) - mu) / sd, W, b).argmax(1)
        res[key] = [float(np.mean(P[(R3 == r) & (Y3 > 0)] == Y3[(R3 == r) & (Y3 > 0)])) * 100
                    for r in sorted(set(R3.tolist())) if r >= 0]
    # (c)
    X, Y, R = ds([f"{D}/20261024_G1.csv"]); zc = np.concatenate(X[Y == 0]).std(0); F = feats(X, zc)
    firm = (R % 2 == 1) & (Y > 0)
    res["grip5"] = fold_acc(F, Y, R, 6, keep=firm)
    res["grip4"] = fold_acc(F, Y, R, 5, keep=firm, mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4})
    # (d)
    import emg_math as M
    Xw, Y, R = ds([f"{D}/20261024_S01_R2.csv"]); zc = np.concatenate(Xw[Y == 0]).std(0)
    eng = E.RealtimeEngine(E.fit_model(feats(Xw, zc), Y, ACT, zc)); cur = E.ServoController(eng)
    mdl = M.MathModel(Xw, Y, R, ACT, True); mc = M.MathController(mdl)
    x, lab, rep, _ = E.load_csv(f"{D}/20261024_S01_R3.csv")
    T, A, B = [], [], []
    for k0 in range(0, len(x), 40):
        for cls, prob, inten, idx, p in eng.push(x[k0:k0 + 40]):
            T.append((k0 + idx) / E.FS); A.append(cur.update(cls, p, eng.last_rms))
            B.append(mc.update(mdl.proba(eng.buf[None])[0], eng.last_rms))
    T = np.array(T); si = np.clip((T * E.FS).astype(int), 0, len(lab) - 1); tr = lab[si]; trep = rep[si]
    segs = [(s0, s1, int(lab[s0])) for s0, s1, _ in E.segments(lab)]
    for key, AA in (("ctrl_base", np.array(A)), ("ctrl_math", np.array(B))):
        per = {"rest": [], "wrong": [], "right": []}
        for r in sorted(set(int(v) for v in rep if v >= 0)):
            sel = (trep == r) & (T > 2)
            on = [(s0 / E.FS, l) for s0, s1, l in segs if l and int(rep[s0]) == r]
            off = [s1 / E.FS for s0, s1, l in segs if l and int(rep[s0]) == r]
            m = E.control_metrics(T[sel], AA[sel], tr[sel], on, off)
            per["rest"].append(m["휴식 중 오작동 %"]); per["wrong"].append(m["힘줄 때 다른 손가락 움직임 %"])
            per["right"].append(m["힘줄 때 맞는 손가락 %"])
        res[key] = per
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{D}/fig_performance.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    res = compute()
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {kk: f"{np.mean(vv):.1f}±{np.std(vv):.1f}" for kk, vv in v.items()})
        else:
            print(k, f"{np.mean(v):.1f} ± {np.std(v):.1f}  (n={len(v)})", [round(x, 1) for x in v])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Arial", "Malgun Gothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(1, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9,
                         "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    L = (lambda ko, en: en if a.en else ko)
    rng = np.random.default_rng(3)

    def bars(ax, groups, labels, fills, ylabel, ylim=(0, 100), letter="", title=""):
        for i, (g, fc) in enumerate(zip(groups, fills)):
            m, s = np.mean(g), np.std(g)
            ax.bar(i, m, width=0.6, color=fc, edgecolor="black", linewidth=0.8)
            ax.errorbar(i, m, yerr=s, color="black", capsize=4, lw=0.9)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=9, color="black", zorder=3, alpha=0.75)
            ax.text(i, max(m + s, max(g)) + 2.5, f"{m:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel(ylabel)
        ax.spines["left"].set_bounds(0, 100)
        ax.spines[["right", "top"]].set_visible(False)
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=12, weight="bold")
        ax.set_title(title, fontsize=9.5, pad=6)

    fig = plt.figure(figsize=(10, 6.6), dpi=200, facecolor="white")
    ax = fig.add_axes((0.08, 0.58, 0.25, 0.33))
    bars(ax, [res["palm"], res["back"], res["band"]],
         [L("손바닥 쪽", "Palmar"), L("손등 쪽", "Dorsal"), L("둘레 밴드", "Band")],
         ["white", "#bdbdbd", "#404040"], L("분류 정확도 (%)", "Accuracy (%)"), letter="a",
         title=L("전극 배치", "Electrode placement"))
    ax = fig.add_axes((0.41, 0.58, 0.22, 0.33))
    bars(ax, [res["rewear_R1"], res["rewear_R2"]], [L("착용 직후\n학습", "Trained\nat donning"), L("안정 후\n학습", "Trained\nafter settling")],
         ["white", "#404040"], L("재착용 정확도 (%)", "Re-wear accuracy (%)"), letter="b", title=L("재착용 시험", "Re-wearing test"))
    ax = fig.add_axes((0.72, 0.58, 0.22, 0.33))
    bars(ax, [res["grip5"], res["grip4"]], [L("5종", "5 grips"), L("4종", "4 grips")],
         ["white", "#404040"], L("쥐기 정확도 (%)", "Grip accuracy (%)"), letter="c", title=L("쥐기 종류 (세게)", "Grip set (firm)"))
    ax = fig.add_axes((0.08, 0.13, 0.55, 0.30))
    keys = [("rest", L("휴식 중\n오작동 ↓", "False motion\nat rest ↓")), ("wrong", L("다른 손가락\n움직임 ↓", "Wrong finger\nmoves ↓")),
            ("right", L("맞는 손가락 ↑", "Correct\nfinger ↑"))]
    for j, (kk, lb) in enumerate(keys):
        for i, (src, fc) in enumerate((("ctrl_base", "white"), ("ctrl_math", "#404040"))):
            g = res[src][kk]; x = j * 2.6 + i * 0.9; m, s = np.mean(g), np.std(g)
            ax.bar(x, m, width=0.8, color=fc, edgecolor="black", linewidth=0.8,
                   label=(L("기존 제어", "Baseline") if i == 0 else L("제안 (수학 파이프라인)", "Proposed")) if j == 0 else None)
            ax.errorbar(x, m, yerr=s, color="black", capsize=4, lw=0.9)
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(g)), g, s=9, color="black" if i == 0 else "#9e9e9e", zorder=3)
            ax.text(x, max(m + s, max(g)) + 2.5, f"{m:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks([j * 2.6 + 0.45 for j in range(3)]); ax.set_xticklabels([k[1] for k in keys], fontsize=8.5)
    ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.spines["left"].set_bounds(0, 100)
    ax.set_ylabel(L("비율 (%)", "Rate (%)")); ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2)
    ax.text(-0.10, 1.06, "d", transform=ax.transAxes, fontsize=12, weight="bold")
    cap = L("Fig. 3. 녹화 데이터로 측정한 성능 (평균 ± 표준편차, 점 = 반복별 값).\n"
            "(a) 손가락 4동작 반복 단위 교차검증: 초기 손바닥 쪽·손등 쪽 배치(2026-10-09)와 둘레 밴드 8채널(S01 R2+R3).\n"
            "(b) 착용 직후(R1) 또는 안정 후(R2) 녹화로 학습하고 밴드를 다시 찬 R3 에서 시험한 손가락 정확도.\n"
            "(c) 쥐기 녹화 G1 의 세게(60~70 %) 반복, 동작 클래스 정확도 (4종 = 집기·세 손가락 집기 통합).\n"
            "(d) R2 학습 → R3 실시간 재생(50 ms 판단). 제안 = CUSUM 시작 검출 · HMM · 순응 예측 · 3차 평활 궤적.\n"
            "참가자 1명(비절단자), 하루 녹화. 필터 20–450 Hz + 60 Hz 계열 노치, 200 ms 창, LDA.",
            "Fig. 3. Performance measured on own recordings (mean ± SD; dots = individual repetitions).\n"
            "(a) Leave-one-repetition-out accuracy, 4 finger classes: initial palmar/dorsal placements vs 8-ch circumferential band.\n"
            "(b) Finger accuracy on re-worn session R3, trained on R1 (just donned) or R2 (settled).\n"
            "(c) Grip accuracy, firm repetitions (60–70 %) of session G1; 4 grips = pinch and tripod merged.\n"
            "(d) Real-time replay (R2 → R3, 50 ms decisions). Proposed = CUSUM onset, HMM, conformal prediction, 3rd-order smoothing.\n"
            "One able-bodied participant, single day. 20–450 Hz band-pass + 60 Hz notches, 200 ms windows, LDA.")
    fig.text(0.68, 0.13, cap, fontsize=7.6, va="bottom", linespacing=1.55, wrap=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
