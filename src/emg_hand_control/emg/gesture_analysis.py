#!/usr/bin/env python3
"""
gesture_analysis.py — 동작 목록 녹화(finger_test record --gestures) 분석

  python gesture_analysis.py data\\20261024_G1.csv
  출력: 품질, 동작별 채널 패턴(지문), 반복 단위 교차검증(전체 / 약하게 / 세게),
        약하게↔세게 교차 시험, '시작 순간 판단 후 고정' 정확도, 그림 *_gesture.png
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emg_classifier as E  # noqa: E402


def load(path):
    x, lab, rep, meta = E.load_csv(path)
    names = meta.get("gestures", "").replace("_", " ").split("|") if meta.get("gestures") else ["엄지", "검지", "중지"]
    return x, lab, rep, meta, ["휴식"] + names


def windows(path, active, reps=None):
    X, Y, R = E.build_dataset([path], active, reps)
    return np.array(X), np.array(Y), np.array(R)


def lda(Ftr, Ytr, k):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    W, b = E.fit_lda((Ftr - mu) / sd, Ytr, k)
    return lambda F: E.lda_proba((F - mu) / sd, W, b)


def cv(F, Y, R, k, test_mask=None, train_mask=None):
    conf = np.zeros((k, k), int)
    for r in sorted(set(R.tolist())):
        if r < 0:
            continue
        tr = R != r; te = R == r
        if train_mask is not None: tr &= train_mask | (Y == 0)
        if test_mask is not None: te &= test_mask | (Y == 0)
        if not te.any(): continue
        P = lda(F[tr], Y[tr], k)(F[te]).argmax(1)
        for t, p in zip(Y[te], P): conf[t, p] += 1
    return np.trace(conf) / max(1, conf.sum()), conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    active = list(range(E.NCH))
    x, lab, rep, meta, names = load(a.csv)
    k = len(names)
    mix = "mix" in meta.get("force", "")
    X, Y, R = windows(a.csv, active)
    rest = np.concatenate(X[Y == 0]); zc = rest.std(0)
    F = np.array([E.features(w, zc) for w in X])
    rms = np.sqrt((X ** 2).mean(1))
    strong = (R % 2 == 1) if mix else np.ones(len(R), bool)
    print(f"== {os.path.basename(a.csv)}  동작 {names[1:]}  반복 {meta.get('reps')}  누락 {meta.get('dropped')}  "
          f"창 {len(Y)}개{'  (짝수 반복 = 약하게, 홀수 = 세게)' if mix else ''}")
    print("\n[1] 채널 RMS 중앙값 (µV)   CH1 … CH8")
    rr = {}
    for c in range(k):
        for tag, m in ((("약", ~strong), ("강", strong)) if mix and c else (("", np.ones(len(Y), bool)),)):
            sel = (Y == c) & m
            if sel.any():
                rr[(c, tag)] = np.median(rms[sel], 0)
                print(f"   {names[c] + (' ' + tag if tag else ''):10s} " + " ".join(f"{v:6.1f}" for v in rr[(c, tag)]))
    acc, conf = cv(F, Y, R, k)
    print(f"\n[2] 반복 단위 교차검증 ({k}클래스, 창 단위, 평활 없음): {100*acc:.1f}%")
    print("    혼동표 행=실제 열=예측: " + " ".join(f"{n[:4]:>5s}" for n in names))
    for i in range(k):
        print(f"    {names[i][:6]:>6s}  " + " ".join(f"{v:5d}" for v in conf[i]) + f"   재현율 {100*conf[i,i]/max(1,conf[i].sum()):5.1f}%")
    if mix:
        for nm, trm, tem in (("세게 학습 → 약하게 시험", strong, ~strong), ("약하게 학습 → 세게 시험", ~strong, strong),
                             ("둘 다 학습 → 약하게 시험", None, ~strong), ("둘 다 학습 → 세게 시험", None, strong)):
            ac, cf = cv(F, Y, R, k, test_mask=tem, train_mask=trm)
            fing = np.trace(cf[1:, 1:]) / max(1, cf[1:].sum())
            print(f"    {nm:24s} 전체 {100*ac:5.1f}%   동작만 {100*fing:5.1f}%")
    # [3] 시작 순간 판단 후 고정: 동작 구간마다 에너지 상승 시점부터 0.25 s 창들의 확률 합으로 결정
    print("\n[3] 시작 순간(에너지 상승 후 0.25 s) 판단 → 고정 방식 (반복 단위로 학습/시험 분리)")
    from scipy import signal
    y = signal.sosfilt(E.full_sos(), x, axis=0); y[:E.FS] = 0
    rest_rms = np.median(rms[Y == 0], 0)
    segs = [(s0, s1, int(lab[s0])) for s0, s1, _ in E.segments(lab) if lab[s0] > 0]
    ok = tot = 0; ok_w = tot_w = 0; lat = []
    per = np.zeros((k, k), int)
    for r in sorted(set(int(v) for v in rep if v >= 0)):
        tr = R != r
        f = lda(F[tr], Y[tr], k)
        for s0, s1, l in segs:
            if int(rep[s0]) != r: continue
            # 에너지 상승 검출: 지시 후 1.5 s 안에서 log(RMS/휴식) 평균이 0.7 넘는 첫 시점
            t_on = None
            for t in range(s0, min(s1, s0 + int(1.5 * E.FS)), E.STEP):
                w = y[t:t + E.WIN]
                z = np.log(np.sqrt((w ** 2).mean(0)) / rest_rms + 1e-9).mean()
                if z > 0.7: t_on = t; break
            if t_on is None:
                tot += 1; per[l, 0] += 1; continue
            ws = [y[t:t + E.WIN] for t in range(t_on, t_on + int(0.25 * E.FS) + 1, E.STEP)]
            P = f(np.array([E.features(w, zc) for w in ws]))
            P[:, 0] = 0
            c = int(np.log(P + 1e-9).sum(0).argmax())
            per[l, c] += 1; tot += 1; ok += int(c == l); lat.append((t_on - s0) / E.FS + 0.25 + E.WIN / E.FS / 2)
            hw = [y[t:t + E.WIN] for t in range(s0 + E.FS // 2, s1 - E.WIN, E.STEP)]
            Pw = f(np.array([E.features(w, zc) for w in hw])).argmax(1)
            ok_w += int(np.sum(Pw == l)); tot_w += len(Pw)
    print(f"    동작 {tot}번 중 맞음 {ok}번 = {100*ok/max(1,tot):.1f}%   (같은 구간 창마다 판단: {100*ok_w/max(1,tot_w):.1f}%)")
    print(f"    결정까지 지시 후 중앙값 {np.median(lat):.2f} s (사람 반응시간 포함)")
    print("    동작별: " + "  ".join(f"{names[i]} {per[i,i]}/{per[i].sum()}" for i in range(1, k)))
    wrong = [(names[i], names[j], per[i, j]) for i in range(1, k) for j in range(k) if i != j and per[i, j]]
    if wrong: print("    틀린 경우: " + ", ".join(f"{a_}→{b_} {n}" for a_, b_, n in wrong))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, k - 1, figsize=(3.2 * (k - 1), 3.4), sharey=True)
        for i, ax in enumerate(np.atleast_1d(axes)):
            c = i + 1
            for tag, col in ((("약", "#90a4ae"), ("강", "#1e4e79")) if mix else (("", "#1e4e79"),)):
                if (c, tag) in rr:
                    v = rr[(c, tag)] / rr[(0, "")]
                    ax.bar(np.arange(1, 9) + (-0.2 if tag == "약" else 0.2 if tag == "강" else 0), v,
                           width=0.4 if mix else 0.7, color=col, label=tag or None)
            ax.set_yscale("log"); ax.set_title(names[c]); ax.set_xticks(range(1, 9))
            ax.axhline(1, color="#999", lw=0.8)
        np.atleast_1d(axes)[0].set_ylabel("휴식 대비 RMS (배)")
        if mix: np.atleast_1d(axes)[0].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=120); print(f"\n그림: {a.plot}")


if __name__ == "__main__":
    main()
