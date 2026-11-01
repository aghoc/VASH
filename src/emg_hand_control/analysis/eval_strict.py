#!/usr/bin/env python3
"""
eval_strict.py — '100 % = 과적합?' 의심에 답하기 위한 엄격한 평가 (F2 쥐기)

  python eval_strict.py

 1) 녹화 간 시험: 한 녹화로만 학습 → 다른 녹화 전체를 시간 순서대로 (B1→B2, B2→B1, G2→B1·B2)
 2) 전환 구간 포함: 지시 변경 후 유예 0 / 0.25 / 0.5 s 별 정확도 (0 s = 지시 순간부터 모든 창 채점)
 3) 시도 단위 정확도 + 95 % 신뢰구간 (Wilson): 동작 시도 1회(3 s 유지) = 창 다수결 1표
    창 단위 %는 창끼리 상관이 커서 표본 수가 부풀려짐 → 시도 단위가 정직한 표본 수
 4) 균형 정확도 · Cohen κ (우연 일치 보정)
모델은 vision_fusion.Model (= fatigue_experiment M0: TD + 리만, 세기 증강, LDA). 학습에 쓴 녹화는 시험에 절대 안 씀.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fatigue_experiment as FX  # noqa: E402
import vision_fusion as VF  # noqa: E402

D = VF.D
NAMES = ["휴식", "주먹", "집기", "세손가락"]
G2MAP = np.array([0, 2, 3, 1])     # G2 라벨(1 집기, 2 세손가락, 3 주먹) → F2 라벨(1 주먹, 2 집기, 3 세손가락)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def kappa(y, p, k=4):
    C = np.zeros((k, k))
    for a, b in zip(y, p): C[a, b] += 1
    n = C.sum(); po = np.trace(C) / n; pe = np.sum(C.sum(0) * C.sum(1)) / n ** 2
    return (po - pe) / (1 - pe)


def evaluate(model, path):
    W, L, Rp, since, _ = FX.stream_windows(path)
    pred = model.proba(W).argmax(1)
    out = {}
    for g in (0.0, 0.25, 0.5):
        m = since >= g
        bal = np.mean([np.mean(pred[m & (L == c)] == c) for c in range(4) if (m & (L == c)).any()])
        out[g] = (100 * np.mean(pred[m] == L[m]), 100 * bal, kappa(L[m], pred[m]))
    # 시도 단위: 동작 구간(라벨 > 0)마다 유예 0.5 s 이후 창 다수결
    trials = {c: [0, 0] for c in (1, 2, 3)}
    seg = np.r_[0, np.flatnonzero(np.diff(L)) + 1, len(L)]
    for s, t in zip(seg[:-1], seg[1:]):
        c = int(L[s])
        if c == 0: continue
        m = since[s:t] >= 0.5
        if not m.any(): continue
        vote = np.bincount(pred[s:t][m], minlength=4).argmax()
        trials[c][0] += int(vote == c); trials[c][1] += 1
    return out, trials


def main():
    files = {k: f"{D}/{v}" for k, v in (("B1", "F2_B1.csv"), ("B2", "F2_B2.csv"), ("G2", "G2.csv"),
                                         ("T1", "F2_T1.csv"), ("T2", "F2_T2.csv"), ("T3", "F2_T3.csv"), ("rec", "F2_recover.csv"))}

    def train(keys, remap=None):
        X, Y, R = VF.load([files[k] for k in keys])
        if remap is not None: Y = remap[Y]
        return VF.Model(X, Y, R)

    class Remapped:     # G2 모델의 출력 순서를 F2 라벨 순서로
        def __init__(self, m): self.m = m
        def proba(self, X):
            P = self.m.proba(X); Q = np.zeros_like(P); Q[:, G2MAP] = P; return Q

    cases = [("B1 → B2", train(["B1"]), "B2"), ("B2 → B1", train(["B2"]), "B1"),
             ("G2 → B1", train(["G2"], G2MAP), "B1"), ("G2 → B2", train(["G2"], G2MAP), "B2")]
    mB = train(["B1", "B2"])
    cases += [(f"B1+B2 → {t}", mB, t) for t in ("T1", "T2", "T3", "rec")]
    print("창 단위 정확도 % (균형 정확도, κ) — 유예 0 s / 0.25 s / 0.5 s   |   시도 단위 (다수결) 맞음/전체 [95 % CI]")
    tot = {c: [0, 0] for c in (1, 2, 3)}
    for name, m, t in cases:
        o, tr = evaluate(m, files[t])
        s = "  ".join(f"{o[g][0]:5.1f} ({o[g][1]:5.1f}, κ {o[g][2]:.2f})" for g in (0.0, 0.25, 0.5))
        k = sum(v[0] for v in tr.values()); n = sum(v[1] for v in tr.values()); lo, hi = wilson(k, n)
        print(f"  {name:12s} {s}  |  {k}/{n} [{lo:.0f}–{hi:.0f}]  " + " ".join(f"{NAMES[c]} {tr[c][0]}/{tr[c][1]}" for c in (1, 2, 3)))
        if not name.startswith("G2"):
            for c in (1, 2, 3): tot[c][0] += tr[c][0]; tot[c][1] += tr[c][1]


if __name__ == "__main__":
    main()
