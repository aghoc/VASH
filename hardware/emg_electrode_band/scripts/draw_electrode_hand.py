"""
draw_electrode_hand.py — 측정 기준 v1 전극 배치를 손·아래팔 그림으로 (왼팔)
실행: python draw_electrode_hand.py [글꼴이름 또는 .ttc 경로] [출력.png]
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Ellipse, Rectangle

FONT = sys.argv[1] if len(sys.argv) > 1 else "Malgun Gothic"
if FONT.endswith((".ttc", ".ttf", ".otf")):
    font_manager.fontManager.addfont(FONT); FONT = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = [FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = sys.argv[2] if len(sys.argv) > 2 else "electrode_hand_guide.png"

SKIN, SKIN_D, LINE = "#f2d2bd", "#d9a98b", "#8a5a44"
BLK, P_C, N_C, BAND = "#2b2f3a", "#d32f2f", "#1976d2", "#6b7280"
L = 250.0                                  # 팔 길이 (주두 → 척골 경상돌기, mm, 예시)
Y_C = 0.40 * L                             # 블록 중심선 40 %
PITCH, BLK_W, BLK_L, BAND_W = 20.0, 25.7, 60.0, 25.0
U = [7, 20, 29, 38, 47, 60, 75, 90]        # 배치도의 둘레 위치 (0 = 척골 능선, 25 손등 가운데, 50 요측, 75 손바닥 가운데)
ROLE = ["엄지·손목", "약지/새끼·엄지", "중지", "검지·엄지", "엄지(벌림)", "엄지·검지", "중지·검지", "검지·약지"]


def width(y):                               # 아래팔 보이는 폭 (mm)
    return 56 + 26 * (1 - y / L) + 9 * math.exp(-((y - 70) / 60) ** 2)


def arm(ax):
    ys = np.linspace(-18, L, 80)
    left = [(-width(max(y, 0)) / 2 - (3 if y < 0 else 0), y) for y in ys]
    right = [(width(max(y, 0)) / 2 + (3 if y < 0 else 0), y) for y in ys[::-1]]
    ax.add_patch(Polygon(left + right, closed=True, fc=SKIN, ec=LINE, lw=1.6, zorder=1))


def finger(ax, base, ang, length, w, bend=0.0, hl=None):
    """손가락: 마디 3개를 둥근 굵은 선으로. bend>0 이면 손바닥 쪽으로 접힌 모양(짧게 + 끝 어둡게)"""
    x, y = base; a = math.radians(ang)
    segs = [0.45, 0.30, 0.25]
    pts = [(x, y)]
    for i, f in enumerate(segs):
        ln = length * f * (1 - 0.55 * bend * (i > 0))
        x += ln * math.sin(a); y += ln * math.cos(a)
        pts.append((x, y))
    px, py = zip(*pts)
    lw = w * 2.05
    ax.plot(px, py, color=LINE, lw=lw + 2.2, solid_capstyle="round", zorder=3)
    ax.plot(px, py, color=hl or SKIN, lw=lw, solid_capstyle="round", zorder=4)
    if bend:
        ax.add_patch(Circle(pts[-1], w * 0.42, color=SKIN_D, zorder=5))
    for p in pts[1:-1]:
        ax.plot([p[0] - w * 0.38 * math.cos(a), p[0] + w * 0.38 * math.cos(a)],
                [p[1] + w * 0.38 * math.sin(a), p[1] - w * 0.38 * math.sin(a)], color=SKIN_D, lw=0.8, zorder=5)


def hand(ax, y0, thumb_side, palm=False, flex=None, scale=1.0, x0=0.0):
    """왼손. thumb_side = +1(오른쪽) / −1(왼쪽). flex = 1 엄지 / 2 검지 / 3 중지 (접은 손가락)"""
    s = scale
    pw, ph = 78 * s, 92 * s
    ax.add_patch(FancyBboxPatch((x0 - pw / 2, y0), pw, ph, boxstyle=f"round,pad=0,rounding_size={22*s}",
                                fc=SKIN, ec=LINE, lw=1.6, zorder=2))
    ax.add_patch(Polygon([(x0 - width(L) / 2 * s, y0 - 2), (x0 + width(L) / 2 * s, y0 - 2),
                          (x0 + pw / 2, y0 + 25 * s), (x0 - pw / 2, y0 + 25 * s)], fc=SKIN, ec="none", zorder=2))
    if palm:
        ax.plot([x0 - 0.30 * pw, x0 + 0.38 * pw], [y0 + 0.70 * ph, y0 + 0.78 * ph], color=SKIN_D, lw=1.2, zorder=3)
        ax.plot([x0 - 0.38 * pw, x0 + 0.20 * pw], [y0 + 0.55 * ph, y0 + 0.62 * ph], color=SKIN_D, lw=1.2, zorder=3)
    # 검지·중지·약지·새끼 (엄지 쪽부터)
    names = [2, 3, 4, 5]
    xs = [0.31, 0.10, -0.12, -0.33]
    lens = [72, 80, 74, 58]
    angs = [4, 0, -4, -9]
    for n, fx, ln, ag in zip(names, xs, lens, angs):
        fx *= thumb_side
        hl = "#ffcf33" if flex == n - 0 and n in (2, 3) and flex in (2, 3) else None
        finger(ax, (x0 + fx * pw, y0 + ph - 6 * s), ag * thumb_side, ln * s, 8.5 * s,
               bend=1.0 if (flex == n and n in (2, 3)) else 0.0, hl=hl)
    # 엄지
    hl = "#ffcf33" if flex == 1 else None
    finger(ax, (x0 + thumb_side * 0.40 * pw, y0 + 0.22 * ph), thumb_side * (42 if flex != 1 else 15), 62 * s, 9.5 * s,
           bend=1.0 if flex == 1 else 0.0, hl=hl)


def electrode_view(ax, center_u, title, thumb_side, palm):
    arm(ax)
    hand(ax, L, thumb_side, palm=palm)
    # 띠 범위 34–46 %, 50 % 한계
    for f, st, c, lab in ((0.34, "--", "#f9a825", "34 %"), (0.46, "--", "#f9a825", "46 %"), (0.50, "-", "#424242", "50 % 한계")):
        ax.plot([-60, 60], [f * L] * 2, ls=st, color=c, lw=1.3, zorder=6)
        ax.text(62, f * L, lab, va="center", fontsize=9, color=c if f != 0.5 else "#212121", weight="bold")
    ax.add_patch(Rectangle((-58, 0.50 * L), 116, L * 0.5, fc="#9e9e9e", alpha=0.18, zorder=6))
    ax.text(0, 0.75 * L, "절단 환자 사용 불가\n(50 % 위쪽)", ha="center", va="center", fontsize=9, color="#616161", zorder=7)
    # 밴드
    yb = Y_C
    ax.add_patch(Rectangle((-width(yb) / 2, yb - BAND_W / 2), width(yb), BAND_W, fc=BAND, alpha=0.55, zorder=7))
    for i, u in enumerate(U):
        th = 2 * math.pi * (u - center_u) / 100
        if abs(th) >= math.pi / 2 * 0.98:
            continue
        x = width(yb) / 2 * math.sin(th)
        w = max(BLK_W * math.cos(th) * 0.85, 5)
        ax.add_patch(FancyBboxPatch((x - w / 2, yb - BLK_L / 2), w, BLK_L, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc=BLK, ec="black", lw=0.8, zorder=8))
        for dy, c, t in ((-PITCH / 2, P_C, "P"), (PITCH / 2, N_C, "N")):
            ax.add_patch(Ellipse((x, yb + dy), max(9 * math.cos(th), 3), 9, fc=c, ec="white", lw=0.8, zorder=9))
        k = sum(1 for uu in U[:i] if abs(2 * math.pi * (uu - center_u) / 100) < math.pi / 2 * 0.98)
        ax.text(x, yb + BLK_L / 2 + 6 + (14 if k % 2 else 0), f"CH{i+1}", ha="center", fontsize=10, weight="bold", color="#111", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # 팔꿈치 표시, BIAS
    ax.text(0, -30, "팔꿈치 (주두 0 %)", ha="center", fontsize=10, weight="bold")
    ax.text(0, L + 4, "손목 (100 %)", ha="center", fontsize=8.5, color="#5d4037", zorder=9)
    bias_x = thumb_side * (width(0) / 2 + 2)        # 외측상과 = 엄지 쪽 팔꿈치
    ax.add_patch(Circle((bias_x, -6), 6, fc="#455a64", ec="white", lw=1, zorder=9))
    ax.text(bias_x + thumb_side * 9, -6, "BIAS\n(외측상과)", ha="left" if thumb_side > 0 else "right",
            va="center", fontsize=8.5, color="#263238", weight="bold")
    side_l, side_r = ("척측\n(새끼 쪽)", "요측\n(엄지 쪽)") if thumb_side > 0 else ("요측\n(엄지 쪽)", "척측\n(새끼 쪽)")
    ax.text(-width(Y_C) / 2 - 6, Y_C - 45, side_l, ha="right", fontsize=8.5, color="#555")
    ax.text(width(Y_C) / 2 + 6, Y_C - 45, side_r, ha="left", fontsize=8.5, color="#555")
    ax.set_title(title, fontsize=13, weight="bold", pad=8)
    ax.set_xlim(-95, 95); ax.set_ylim(-45, 440); ax.set_aspect("equal"); ax.axis("off")


fig = plt.figure(figsize=(17, 11.5), dpi=120)
fig.suptitle("측정 기준 v1 — 왼팔 전극 배치 (블록 8개 · 밴드 25 mm · 블록 중심선 = 팔 길이 40 %)", fontsize=16, weight="bold", y=0.99)
ax1 = fig.add_axes((0.01, 0.20, 0.27, 0.74)); electrode_view(ax1, 25, "① 손등 쪽에서 본 모습", +1, False)
ax2 = fig.add_axes((0.29, 0.20, 0.27, 0.74)); electrode_view(ax2, 75, "② 손바닥 쪽에서 본 모습", -1, True)

# ③ 단면
ax3 = fig.add_axes((0.585, 0.50, 0.40, 0.40)); ax3.set_aspect("equal"); ax3.axis("off")
A, B = 46, 36
t = np.linspace(0, 2 * np.pi, 200)
ax3.fill(A * np.cos(t), B * np.sin(t), color=SKIN, ec=LINE, lw=1.6)
ax3.add_patch(Ellipse((-30, 8), 13, 11, fc="#f5f5f0", ec="#9e9e9e", lw=1.2)); ax3.text(-30, 8, "척골", ha="center", va="center", fontsize=8)
ax3.add_patch(Ellipse((25, -2), 16, 11, fc="#f5f5f0", ec="#9e9e9e", lw=1.2)); ax3.text(25, -2, "요골", ha="center", va="center", fontsize=8)
for i, u in enumerate(U):
    ph = math.pi / 2 - 2 * math.pi * (u - 25) / 100       # 위 = 손등 가운데(u=25), 오른쪽 = 요측(u=50)
    x, y = A * math.cos(ph), B * math.sin(ph)
    nx, ny = x / A ** 2, y / B ** 2; nn = math.hypot(nx, ny); nx, ny = nx / nn, ny / nn
    bx, by = x + nx * 6, y + ny * 6
    ang = math.degrees(math.atan2(ny, nx)) - 90
    blk = Rectangle((bx - 9, by - 4.5), 18, 9, angle=ang, rotation_point="center", fc=BLK, ec="black", lw=0.8)
    ax3.add_patch(blk)
    ax3.text(x + nx * 19, y + ny * 19, f"CH{i+1}", ha="center", va="center", fontsize=10.5, weight="bold")
    ax3.text(x + nx * 33 + (8 if i == 4 else 0), y + ny * 33 - (7 if i == 4 else 0), ROLE[i], ha="center", va="center", fontsize=8, color="#555")
ax3.annotate("척골 능선\n(비워 둠)", xy=(-A - 1, 0), xytext=(-A - 30, -26), fontsize=9, color="#b71c1c", weight="bold",
             ha="center", arrowprops=dict(arrowstyle="->", color="#b71c1c"))
ax3.text(0, B + 44, "손등", ha="center", fontsize=11, weight="bold"); ax3.text(0, -B - 46, "손바닥", ha="center", fontsize=11, weight="bold")
ax3.text(A + 62, 18, "요측\n(엄지)", ha="center", va="center", fontsize=10, weight="bold")
ax3.set_xlim(-100, 105); ax3.set_ylim(-88, 88)
ax3.set_title("③ 단면 (팔꿈치 쪽에서 손 방향으로, 손등이 위)\nCH1 → CH8 순서로 감기", fontsize=12, weight="bold")

# ④ 블록 한 개 확대
ax4 = fig.add_axes((0.59, 0.17, 0.15, 0.28)); ax4.set_aspect("equal"); ax4.axis("off")
ax4.add_patch(FancyBboxPatch((-12.85, -30), 25.7, 60, boxstyle="round,pad=0,rounding_size=3", fc=BLK, ec="black"))
ax4.add_patch(Rectangle((-16, -12.5), 32, 25, fc=BAND, alpha=0.6))
for dy, c, tt in ((-10, P_C, "P"), (10, N_C, "N")):
    ax4.add_patch(Circle((0, dy), 4.5, fc=c, ec="white")); ax4.text(0, dy, tt, color="white", ha="center", va="center", fontsize=9, weight="bold")
ax4.annotate("", xy=(17, -10), xytext=(17, 10), arrowprops=dict(arrowstyle="<->"))
ax4.text(19, 0, "20 mm", va="center", fontsize=9)
ax4.text(0, -36, "↓ 팔꿈치 쪽 (P, 전선 나감)", ha="center", fontsize=8.5); ax4.text(0, 33, "↑ 손목 쪽 (N)", ha="center", fontsize=8.5)
ax4.set_xlim(-30, 45); ax4.set_ylim(-42, 40)
ax4.set_title("④ 블록 1개 (25.7 × 60 mm)", fontsize=11, weight="bold")

# ⑤ 동작 4개
axs = [fig.add_axes((0.755 + 0.06 * k, 0.15, 0.058, 0.27)) for k in range(4)]
labels = ["휴식", "엄지", "검지", "중지"]
for k, ax in enumerate(axs):
    hand(ax, 0, -1, palm=True, flex=(None, 1, 2, 3)[k], scale=0.55)
    ax.add_patch(Polygon([(-width(L) / 2 * 0.55, 0), (width(L) / 2 * 0.55, 0), (width(L) / 2 * 0.62, -25), (-width(L) / 2 * 0.62, -25)],
                         fc=SKIN, ec=LINE, lw=1.2, zorder=1))
    ax.set_xlim(-45, 45); ax.set_ylim(-25, 110); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(labels[k], fontsize=10.5, weight="bold", color=["#757575", "#c62828", "#1565c0", "#6a1b9a"][k])
fig.text(0.87, 0.445, "⑤ 녹화 동작 (손바닥 쪽, 노란 손가락을 구부려 유지)", ha="center", fontsize=11, weight="bold")

notes = ("붙이는 순서  ① 팔 길이 L = 팔꿈치 뾰족한 뼈(주두) → 손목 새끼 쪽 뼈(척골 경상돌기) 측정   ② 0.4 × L 지점에 펜으로 둘레 선   "
         "③ 척골 능선(팔꿈치→손목 새끼 쪽으로 만져지는 뼈 줄) 표시\n"
         "④ CH1 블록을 능선 바로 옆 손등 쪽에, P(빨강)가 팔꿈치 쪽을 향하게   ⑤ 손등 → 엄지 쪽 → 손바닥 순서로 CH2 … CH8   "
         "⑥ 능선 위는 비워 둠   ⑦ BIAS 전극은 팔꿈치 바깥 뼈(외측상과)\n"
         "그림 치수는 L = 25 cm 예시 (P 중심 36 %, N 중심 44 %). 근육 위치는 문헌 기반 개념도 — 실제로는 손가락을 움직이며 실시간 파형으로 확인")
fig.text(0.015, 0.035, notes, fontsize=10, va="bottom", linespacing=1.6,
         bbox=dict(boxstyle="round,pad=0.6", fc="#fffde7", ec="#f9a825"))
fig.savefig(OUT, dpi=120)
print("saved", OUT)
