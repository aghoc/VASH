import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import sys
from matplotlib import font_manager
FONT = sys.argv[1] if len(sys.argv) > 1 else "Malgun Gothic"
if FONT.endswith((".ttc", ".ttf", ".otf")):
    font_manager.fontManager.addfont(FONT); FONT = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = [FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(16, 9.6), dpi=110)
ax = fig.add_axes((0.065, 0.08, 0.535, 0.84))
ax.set_xlim(0, 100); ax.set_ylim(100, -6)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["척측\n(뼈 능선)", "손등 가운데", "요측\n(엄지 쪽)", "손바닥 가운데", "척측\n(뼈 능선)"], fontsize=9)
ax.set_yticks(range(0, 101, 10)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], fontsize=9)
ax.set_ylabel("팔꿈치(주두, 0%) → 손목(척골 경상돌기, 100%)", fontsize=10)
ax.axvline(50, color="#555", lw=1, ls=":")
ax.text(25, -3.5, "손등 쪽 (신근)", ha="center", fontsize=12, weight="bold")
ax.text(75, -3.5, "손바닥 쪽 (굴근)", ha="center", fontsize=12, weight="bold")

def mus(x0, x1, y0, y1, color, name, deep=False, fs=8.5):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=2",
                       fc=color if not deep else "none", ec=color, lw=1.6 if deep else 0.8,
                       ls="--" if deep else "-", alpha=0.35 if not deep else 0.95)
    ax.add_patch(p)
    ly = (y0 + y1) / 2 if deep else min(y0 + 9, 24)
    ax.text((x0 + x1) / 2, ly, name, ha="center", va="center", fontsize=fs,
            color="#222" if not deep else color, weight="bold" if deep else "normal")

# 손등 쪽 (표층)
mus(2, 10, 8, 92, "#8d6e63", "척측\n수근\n신근\n손목", fs=7.5)
mus(10, 16, 12, 85, "#a1887f", "새끼\n폄근", fs=7.5)
mus(16, 35, 12, 78, "#ef6c00", "")
ax.text(25.5, 11, "손가락폄근 (EDC)", ha="center", fontsize=9.5, weight="bold", color="#bf360c")
mus(35, 42, 8, 62, "#8d6e63", "짧은요측\n수근신근\n손목", fs=7.5)
mus(42, 50, 0, 58, "#bcaaa4", "긴요측수근신근\n·위팔노근\n손목·팔꿈치", fs=7.2)
# 손등 쪽 (심층, 점선)
mus(12, 26, 38, 72, "#1565c0", "긴엄지폄근\n(EPL, 깊음)\n엄지", deep=True, fs=8)
mus(28, 44, 34, 80, "#6a1b9a", "긴엄지벌림근·\n짧은엄지폄근\n(APL/EPB, 깊음)\n엄지", deep=True, fs=7.5)
mus(13, 21, 62, 90, "#00838f", "집게폄근\n(EIP)\n검지", deep=True, fs=7.5)

# 손바닥 쪽 (표층)
mus(51, 56, 0, 50, "#bcaaa4", "위팔\n노근", fs=7.2)
mus(56, 64, 4, 58, "#8d6e63", "요측\n수근굴근\n손목", fs=7.5)
mus(64, 86, 12, 70, "#43a047", "")
ax.text(73, 11, "얕은손가락굽힘근 (FDS)", ha="center", fontsize=9.5, weight="bold", color="#1b5e20")
mus(88, 98, 4, 86, "#8d6e63", "척측\n수근\n굴근\n손목", fs=7.5)
# 손바닥 쪽 (심층)
mus(52, 64, 30, 80, "#c62828", "긴엄지굽힘근\n(FPL, 깊음)\n엄지", deep=True, fs=8)
mus(76, 96, 14, 80, "#2e7d32", "", deep=True)
ax.text(86, 74, "깊은손가락굽힘근\n(FDP, 깊음)\n검지 부분은 요측", ha="center", fontsize=7.8, color="#2e7d32", weight="bold")

# 사용 불가 영역
ax.add_patch(Rectangle((0, 50), 100, 50, fc="#9e9e9e", alpha=0.28, ec="none"))
ax.axhline(50, color="#424242", lw=2)
ax.text(99, 52.5, "50% — 절단 환자 사용 한계 (이 아래 사용 불가)", ha="right", va="top", fontsize=10, color="#212121", weight="bold")

# Hu et al. 2015 EDC 손가락별 활성 중심
for y, lab, col in [(27, "중지 중심 ≈30%", "#d50000"), (45, "약지·새끼 40~50%", "#6d4c41"), (60, "검지 중심 ≈60% (범위 밖)", "#00838f")]:
    ax.plot(25.5, y, marker="*", ms=17, color=col, mec="k", mew=0.6, zorder=9)
    ax.text(27.5, y + 0.4, lab, fontsize=8.6, va="center", color=col, weight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
ax.text(1, 98, "★ Hu et al. 2015: 손가락폄근(EDC) 손가락별 활성 중심 (주두→척골 경상돌기 %)", fontsize=8.5, color="#333")

# 제안 패드: 34~46% 띠, 8열 × 2행 (P 행 / N 행)
y0, y1 = 33, 47
ax.add_patch(FancyBboxPatch((4, y0), 92, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5",
                            fc="#fff59d", ec="#f9a825", lw=2.5, alpha=0.55, zorder=4))
cols = [7, 20, 29, 38, 47, 60, 75, 90]
roles = ["엄지(EPL)·손목", "약지/새끼·엄지", "중지", "검지·엄지(APL)", "엄지(APL/EPB)·손목", "엄지(FPL)·검지", "중지·검지(FDS)", "검지·약지(FDP)"]
for i, x in enumerate(cols):
    ax.add_patch(Circle((x, 36), 1.5, fc="#d32f2f", ec="k", lw=0.8, zorder=7))
    ax.add_patch(Circle((x, 44), 1.5, fc="#1976d2", ec="k", lw=0.8, zorder=7))
    ax.plot([x, x], [37.5, 42.5], color="k", lw=0.8, zorder=6)
    ax.text(x, 40, f"CH{i+1}", ha="center", va="center", fontsize=8.5, weight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.text(50, 49.0, "제안 패드: 34~46% 띠를 팔 둘레로 감싸는 8열×2행 (윗줄 ● P, 아랫줄 ● N, 간격 약 2 cm)",
        ha="center", fontsize=10, weight="bold", color="#e65100", zorder=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#f9a825", alpha=0.9))
ax.plot(46, -1, marker="o", ms=12, color="#455a64", zorder=7, clip_on=False)
ax.text(44.5, -1, "BIAS(외측상과)", ha="right", va="center", fontsize=8.6, color="#263238", zorder=8, weight="bold")
ax.set_title("왼팔 전완을 펼쳐 놓은 지도 (손등 쪽에서 본 방향, 개념도 — 실제 위치는 촉진·측정으로 보정)",
             fontsize=11.5, pad=26)

# 오른쪽: 채널 표
tx = fig.add_axes((0.63, 0.08, 0.35, 0.84)); tx.axis("off")
rows = [("CH", "둘레 위치", "주 대상 근육 (깊이)", "손가락"),
        ("1", "손등 척측", "ECU · 깊은 EPL", "엄지(펴기)"),
        ("2", "손등 척측-가운데", "EDC 약지/새끼 · EPL", "엄지·약지"),
        ("3", "손등 가운데", "EDC 중지 구획", "중지(펴기)"),
        ("4", "손등 요측-가운데", "EDC/ECRB · 깊은 APL", "검지·엄지"),
        ("5", "요측 가장자리", "APL/EPB 노출부 · ECRL", "엄지(벌림)"),
        ("6", "손바닥 요측", "FCR · 깊은 FPL · FDS 검지", "엄지(굽힘)·검지"),
        ("7", "손바닥 가운데", "FDS (중지·약지·검지)", "중지·검지"),
        ("8", "손바닥 척측", "FDP · FCU", "검지·약지")]
y = 0.97
for k, r in enumerate(rows):
    w = "bold" if k == 0 else "normal"
    for xx, t in zip((0.0, 0.07, 0.36, 0.78), r):
        tx.text(xx, y, t, fontsize=10 if k else 10.5, weight=w, va="top", transform=tx.transAxes)
    y -= 0.052
tx.plot([0, 1], [0.925, 0.925], color="k", lw=0.8, transform=tx.transAxes)
notes = [
    "근거",
    "· Hu 2015 (7×9 HD-EMG): 손가락폄근의 손가락별 활성 중심",
    "   중지 ≈30%, 약지·새끼 40~50%, 검지 ≈60% (주두→척골 기준)",
    "   → 50% 이내에서는 손등만으로 검지를 따로 잡기 어려움",
    "· Dai & Hu 2019: 손바닥 쪽 굴근도 손가락별 패턴이 겹치지만",
    "   고밀도 배열 + 패턴 인식으로 94~100% 분류",
    "· 교수님 패드 (Kim 2023): 전완 8채널 패치로 10동작 96%",
    "· 인접 전극 간 공통신호 손등 ~50%, 손바닥 >60% (교차 간섭)",
    "   → 한 손가락 = 한 채널 대신 둘레 전체 패턴으로 분류",
    "",
    "우리 측정 (B2, 손등)",
    "· 손등 척측(현 CH2) → 엄지 14.6배  ⇒ 깊은 EPL 추정, 문헌과 일치",
    "· 손등 가운데 30~45%(현 CH6) → 중지 19.1배  ⇒ Hu 2015 중지 30%와 일치",
    "· 검지는 손등 50% 이내 어디서도 거의 안 잡힘  ⇒ 손바닥 쪽(FDS·FDP) 필요",
    "",
    "패드 규격 (제안)",
    "· 띠 위치 34~46% (팔 길이 25 cm면 팔꿈치에서 8.5~11.5 cm)",
    "· 전극 지름 10 mm, P–N 간격 20 mm (근섬유 방향)",
    "· 둘레 간격 ≈ 둘레/8 (둘레 25 cm면 약 3 cm)",
    "· 척골 능선(뼈 바로 위)은 비워 둠",
]
yy = 0.45
for n in notes:
    tx.text(0, yy, n, fontsize=9.3, va="top", transform=tx.transAxes,
            weight="bold" if n in ("근거", "우리 측정 (B2, 손등)", "패드 규격 (제안)") else "normal")
    yy -= 0.0225
fig.savefig(sys.argv[2], dpi=110)
print("saved", sys.argv[2])
