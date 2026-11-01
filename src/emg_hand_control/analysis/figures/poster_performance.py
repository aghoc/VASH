#!/usr/bin/env python3
"""
poster_performance.py — 지금까지 녹화 데이터로 측정한 분류·제어 성능 변화 막대 그래프 (포스터용)

  python poster_performance.py --out data\\performance.png        (영어: --en)

숫자는 모두 실제 녹화로 계산한 값 (출처는 아래 표의 '근거' 칸, 분석 기록: data\\분석_S01.md, 분석_G1.md, 최신_코딩작업.md)
"""
import argparse

# (이름_ko, 이름_en, 값 %, 강조)
PANELS = [
    dict(ko="① 전극 배치 개선", en="① Electrode placement",
         note_ko="손가락 4동작 · 반복 단위 교차검증", note_en="4 finger classes · leave-one-rep-out CV",
         bars=[("손바닥 쪽\n(초기 배치)", "Palm side\n(initial)", 77.5, 0),     # A1_flexor 10-09
               ("손등 쪽\n(초기 배치)", "Back side\n(initial)", 86.5, 0),       # B2_extensor 10-09
               ("둘레 밴드\n8채널 (v1)", "Circumferential\nband (v1)", 95.0, 1)]),  # S01 R2+R3 10-24
    dict(ko="② 다시 착용해도 통할까", en="② Re-wearing the band",
         note_ko="재착용 녹화(R3)에서 손가락 정확도", note_en="finger accuracy on re-worn session (R3)",
         bars=[("착용 직후\n데이터로 학습", "Trained right\nafter donning", 20.4, 0),   # R1 → R3
               ("안정 후\n데이터로 학습", "Trained after\nsettling", 96.7, 1)]),        # R2 → R3
    dict(ko="③ 약하게 쥘 때 (중지)", en="③ Weak grip (middle finger)",
         note_ko="녹화 대비 50 % 힘 모사 · 중지 정답률", note_en="50 % force (simulated) · middle finger",
         bars=[("기존 학습", "Standard\ntraining", 24.0, 0),
               ("세기 증강\n학습", "Force-augmented\ntraining", 89.9, 1)]),
    dict(ko="④ 쥐기 종류 (세게)", en="④ Grip types (firm)",
         note_ko="반복 단위 교차검증 · 동작만", note_en="leave-one-rep-out CV · grips only",
         bars=[("5종", "5 grips", 78.8, 0),
               ("4종\n(집기 통합)", "4 grips\n(pinch merged)", 95.5, 1)]),
]
# ⑤ 제어 단계: R2 학습 → R3 재생 (낮을수록 좋은 지표 포함)
CTRL = dict(ko="⑤ 서보 제어 단계 (재착용 데이터 재생)", en="⑤ Servo control stage (replay of re-worn session)",
            rows=[("휴식 중 오작동", "False motion at rest", 33.4, 9.6, "low"),
                  ("다른 손가락 움직임", "Wrong finger moves", 10.0, 3.0, "low"),
                  ("맞는 손가락", "Correct finger", 89.7, 93.9, "high")],
            legend=(("기존 제어", "Baseline control"), ("수학 파이프라인", "Math pipeline")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="performance.png")
    ap.add_argument("--en", action="store_true"); ap.add_argument("--font", default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    if a.font:
        font_manager.fontManager.addfont(a.font); fam.insert(0, font_manager.FontProperties(fname=a.font).get_name())
    plt.rcParams["font.family"] = fam; plt.rcParams["axes.unicode_minus"] = False
    L = (lambda ko, en: en if a.en else ko)
    NAVY, GRAY, RED, TXT = "#1e4e79", "#b8c2cc", "#c62828", "#222"

    fig = plt.figure(figsize=(17, 9), dpi=150, facecolor="white")
    fig.text(0.03, 0.955, L("녹화 데이터로 측정한 성능 변화", "Measured performance improvements (own recordings)"),
             fontsize=19, weight="bold", color=TXT)
    W = 0.215
    for i, p in enumerate(PANELS):
        ax = fig.add_axes((0.04 + i * (W + 0.03), 0.50, W, 0.36))
        labs = [L(b[0], b[1]) for b in p["bars"]]; vals = [b[2] for b in p["bars"]]
        cols = [NAVY if b[3] else GRAY for b in p["bars"]]
        xs = range(len(vals))
        ax.bar(xs, vals, color=cols, width=0.62)
        for x, v in zip(xs, vals):
            ax.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=12, weight="bold", color=NAVY if cols[x] == NAVY else "#556")
        ax.set_xticks(list(xs)); ax.set_xticklabels(labs, fontsize=9.5)
        ax.set_ylim(0, 120); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=9)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(L(p["ko"], p["en"]), loc="left", fontsize=13, weight="bold", color=NAVY, pad=22)
        ax.text(0, 1.04, L(p["note_ko"], p["note_en"]), transform=ax.transAxes, fontsize=8.8, color="#666")
        d = vals[-1] - vals[0]
        ax.annotate("", xy=(len(vals) - 1, 106), xytext=(0, 106),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
        ax.text((len(vals) - 1) / 2, 109, f"+{d:.1f}%p", ha="center", va="bottom", color=RED, fontsize=11.5, weight="bold")
        if i == 0:
            ax.set_ylabel(L("정확도 (%)", "Accuracy (%)"), fontsize=10)
    # ⑤ 제어
    ax = fig.add_axes((0.13, 0.08, 0.50, 0.30))
    import numpy as np
    rows = CTRL["rows"]; y = np.arange(len(rows))
    b0 = [r[2] for r in rows]; b1 = [r[3] for r in rows]
    ax.barh(y + 0.19, b0, height=0.36, color=GRAY, label=L(*CTRL["legend"][0]))
    ax.barh(y - 0.19, b1, height=0.36, color=NAVY, label=L(*CTRL["legend"][1]))
    for yy, v0, v1, r in zip(y, b0, b1, rows):
        ax.text(v0 + 1, yy + 0.19, f"{v0:.1f}%", va="center", fontsize=10, color="#556")
        ax.text(v1 + 1, yy - 0.19, f"{v1:.1f}%", va="center", fontsize=10.5, color=NAVY, weight="bold")
        better = (v1 < v0) if r[4] == "low" else (v1 > v0)
        tag = L("낮을수록 좋음", "lower is better") if r[4] == "low" else L("높을수록 좋음", "higher is better")
        ax.text(106, yy, ("✔ " if better else "") + tag, va="center", fontsize=9, color=RED if better else "#888")
    ax.set_yticks(y); ax.set_yticklabels([L(r[0], r[1]) for r in rows], fontsize=11)
    ax.invert_yaxis(); ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(L(CTRL["ko"], CTRL["en"]), loc="left", fontsize=13, weight="bold", color=NAVY, pad=10)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)
    # 설명
    notes = L("· ① A1·B2: 2026-10-09 초기 배치 녹화 / 둘레 밴드: 2026-10-24 S01 R2+R3\n"
              "· ② R1(착용 직후) 또는 R2(안정 후)로 학습 → 밴드를 풀고 다시 찬 R3 에서 시험\n"
              "· ③ R2+R3 동작 신호를 50 % 로 줄이고 실제 휴식 잡음을 더해 모사 (실측 아님)\n"
              "· ④ 쥐기 5종 녹화 G1 의 세게(60~70 %) 반복만 시험. 약하게(20~30 %) 하면 62 %\n"
              "· ⑤ 50 ms 마다 판단하는 실시간 재생. 수학 파이프라인 = CUSUM·HMM·순응 예측·3차 평활 궤적\n"
              "· 모든 수치는 한 사람(비절단자)의 하루 녹화 기준",
              "· ① A1/B2: initial placements (2026-10-09) / band: 2026-10-24 session S01 R2+R3\n"
              "· ② trained on R1 (just donned) or R2 (settled) → tested on R3 after re-wearing\n"
              "· ③ simulated: R2+R3 contractions scaled to 50 % with real rest noise added\n"
              "· ④ firm repetitions (60–70 %) of grip session G1; weak (20–30 %) gives 62 %\n"
              "· ⑤ real-time replay, decision every 50 ms; math pipeline = CUSUM, HMM, conformal, min-jerk\n"
              "· single able-bodied participant, one day of recordings")
    fig.text(0.68, 0.08, notes, fontsize=9.2, color="#444", va="bottom", linespacing=1.7)
    fig.savefig(a.out, dpi=150, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
