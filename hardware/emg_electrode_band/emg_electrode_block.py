"""
emg_electrode_block.py — 스냅 케이블 2개(P/N)를 넣는 밴드형 EMG 블록 v3 (CadQuery 파라메트릭)

스냅 케이블 단자 모양(평면): 원형 머리 Ø15 → 목이 좁아지며(테이퍼) → 주름 부싱(Ø5.1, 10.3 mm) → 전선
  실측: 머리 Ø15.0 × 높이 6.75, 머리 중심(반원 기준선)→부싱 시작 B = 20 mm, 부싱 길이 C = 10.3, 부싱 지름 D = 5.1

좌표: X = 밴드 방향(팔 둘레), Y = 근섬유 방향(−Y 팔꿈치 / +Y 손목), Z = 위. Z=0 이 피부 쪽 바닥.
배치: P 머리 y=−PITCH/2 꼬리 → 팔꿈치(−Y), N 머리 y=+PITCH/2 꼬리 → 손목(+Y).
      (밴드 방향으로 꼬리를 빼면 이웃 블록과 부딪히므로 근섬유 방향으로 뺌)
블록은 머리 + 목 전체를 열쇠구멍 모양 포켓으로 잡고, 부싱·전선은 블록 끝에서 빠져나감.
v5 (2026-10-14): 전선 홈 위에 딸깍 뚜껑(별도 출력 emg_electrode_lid.stl), 홈 폭 2.4 복귀, 높이 +1.2.
v4 (2026-10-14): 포켓 천장 평평(밀어내기 구멍 삭제), 밴드 방향 폭 +6 mm(벽 5.2 → 25.7), 전선 홈 폭 1.4, 밴드 25 mm.
v3 (한 방향 배출): N 전선은 손목 쪽 끝에서 부싱이 나온 뒤 180° 꺾어 끝면 세로 홈 → 윗면 홈을 따라
     팔꿈치 쪽으로 되돌아옴. 결과적으로 P·N 전선이 모두 팔꿈치(−Y) 쪽으로 나감. 홈은 모두 열린 홈(눌러 끼움).

실행:  python emg_electrode_block.py   →  emg_electrode_block.step / .stl
"""
import math
import cadquery as cq

# ---------------- 치수 (mm) ----------------
HEAD_D      = 15.0    # 머리 지름 (실측)
HEAD_H      = 6.75    # 머리 높이 (실측)
NECK_L      = 20.0    # 머리 중심(반원 기준선) → 부싱 시작 (실측 B)
BOOT_D      = 5.1     # 부싱 지름 (실측 D)
BOOT_L      = 10.3    # 부싱 길이 (실측 C) — 블록 밖으로 나옴
FIT         = 0.30    # 포켓 여유 (0.2~0.4)
PROTRUDE    = 0.30    # 머리 바닥이 블록 바닥보다 튀어나오는 양
PITCH       = 20.0    # P–N 머리 중심 간격 (측정 기준 v1: 20 mm). 전극 바깥지름 ≤ 19 mm 이어야 겹치지 않음
WALL        = 5.2     # 머리 포켓 옆 벽 (v4: 2.2 → 5.2, 밴드 방향 양쪽 +3 mm → 폭 25.7, 팔 둘레 21 cm 이상)
ROOF        = 1.6     # 포켓 위 천장
STRAP_W     = 25.0    # 밴드 폭 (v4: 25 mm 웨빙/벨크로)
STRAP_T     = 2.0     # 밴드 두께
STRAP_CLR   = 0.6
TOP         = 4.8     # 밴드 홈 위 두께 = 바닥 1.2 + 전선 홈 2.4 + 뚜껑 1.2 (v5: 3.6 → 4.8)
WIRE_D      = 2.2     # 전선 외경 (실측 후 수정)
GROOVE_W    = WIRE_D + 0.2   # 되돌림 전선 홈 폭 (v5: 뚜껑이 전선을 잡으므로 원래 2.4 로 복귀)
# ---- 딸깍 뚜껑 (v5) ----
LID_T       = 1.2     # 뚜껑 두께 = 홈 위 오목부 깊이
LID_W       = 7.0     # 뚜껑 폭
LID_CLR     = 0.10    # 뚜껑 옆 틈 (한쪽)
SNAP_R      = 0.45    # 걸림 돌기/홈 반지름
SNAP_P      = 0.30    # 돌기가 뚜껑 옆면 밖으로 나온 양 → 끼울 때 간섭 = SNAP_P − LID_CLR = 0.2 mm
SNAP_L      = 6.0     # 돌기 길이
SNAP_Y      = (-22.0, 0.0, 22.0)   # 돌기 위치 (3쌍, v5b: ±18 → ±22, 끝 들뜸 12 → 8 mm)
FLEX_L      = 14.0    # 돌기 뒤 휘는 보 길이 (v5b: 16 → 14, 변형률 ≈ 12·0.2·1.0/14² ≈ 1.2 %, 힘 ≈ 1.5배)
FLEX_W      = 0.8     # 휘는 보 뒤 틈 폭
FLEX_T      = 1.0     # 휘는 보 두께
RET_X       = 5.2     # 되돌림 전선 홈의 X 위치 (머리 중심선에서 옆으로)
CORNER_R    = 3.0
LABEL       = True

# ---------------- 파생 ----------------
R       = (HEAD_D + FIT) / 2
w_end   = (BOOT_D + FIT) / 2
pocket_h = HEAD_H - PROTRUDE
size_x  = 2 * R + 2 * WALL
half_y  = PITCH / 2 + NECK_L            # 블록 끝 = 목 끝(부싱 시작)
size_y  = 2 * half_y
slot_w  = STRAP_W + STRAP_CLR
slot_t  = STRAP_T + STRAP_CLR
z_roof  = pocket_h + ROOF
height  = z_roof + slot_t + TOP
yP, yN  = -PITCH / 2, +PITCH / 2
assert slot_w < size_y - 2 * (BOOT_D + 4), "밴드 홈이 블록 길이에 비해 너무 넓음"


def keyhole(cy, direction, extra=1.0):
    """머리 원 + 원에 접하며 좁아지는 목(테이퍼) + 블록 밖까지 이어지는 부싱 홈. direction=+1(+Y) / -1(-Y)"""
    L = NECK_L + extra                     # 블록 밖으로 살짝 더 파서 출구를 깔끔하게
    # 원(반지름 R)에 접하고 거리 NECK_L 지점에서 폭 w_end 인 직선: 접점 각도 계산
    # 끝점 (±w_end, NECK_L) 에서 원에 그은 외접선 — 끝 폭이 R 보다 작으므로 접점은 원 위쪽 반원 측면
    d = NECK_L
    # 끝점과 원 중심 사이 거리
    dist = math.hypot(w_end, d)
    alpha = math.atan2(d, w_end)           # 중심→끝점 각
    beta = math.acos(R / dist)             # 접점 보정각
    th = alpha - beta                      # 오른쪽(+x) 접점 각도
    tx, ty = R * math.cos(th), R * math.sin(th)
    pts = [(tx, ty), (w_end, d), (w_end, L), (-w_end, L), (-w_end, d), (-tx, ty)]
    s = 1 if direction > 0 else -1
    pts = [(x, cy + s * y) for x, y in pts]
    neck = cq.Workplane("XY").polyline(pts).close().extrude(pocket_h)
    head = cq.Workplane("XY").center(0, cy).circle(R).extrude(pocket_h)
    return head.union(neck)


body = (cq.Workplane("XY").box(size_x, size_y, height, centered=(True, True, False))
        .edges("|Z").fillet(CORNER_R)
        .faces(">Z").edges().fillet(1.0))

body = body.cut(keyhole(yP, -1)).cut(keyhole(yN, +1))     # 포켓 천장은 평평 (v4: 밀어내기 구멍 삭제)

slot = cq.Workplane("XY").workplane(offset=z_roof).box(size_x + 2, slot_w, slot_t, centered=(True, True, False))
body = body.cut(slot)

# 되돌림 전선 홈 (N 전선: 손목 쪽 끝면 세로 홈 → 윗면 홈 → 팔꿈치 쪽 끝)
gw = GROOVE_W                          # 홈 폭 (전선보다 좁게 → 눌러 끼움)
gd = WIRE_D + 0.2                      # 홈 깊이
z_lid = height - LID_T                 # 뚜껑 오목부 바닥 = 전선 홈 윗면
top_groove = (cq.Workplane("XY").workplane(offset=z_lid - gd)
              .center(RET_X, 0).rect(gw, size_y + 2).extrude(gd + 0.01))
body = body.cut(top_groove)
# 뚜껑 오목부 (양 끝 개방) + 옆벽 걸림 홈 + 손톱 홈
rec_w = LID_W + 2 * LID_CLR
body = body.cut(cq.Workplane("XY").workplane(offset=z_lid).center(RET_X, 0).rect(rec_w, size_y + 2).extrude(LID_T + 1))
for sx in (-1, 1):
    for y in SNAP_Y:
        body = body.cut(cq.Workplane("XZ").workplane(offset=-(y + SNAP_L / 2))
                        .center(RET_X + sx * rec_w / 2, z_lid + LID_T / 2).circle(SNAP_R).extrude(SNAP_L))
body = body.cut(cq.Workplane("XY").workplane(offset=z_lid).center(RET_X + rec_w / 2 + 0.75, half_y - 2.0)
                .rect(1.5, 4.0).extrude(LID_T + 1))
end_groove = (cq.Workplane("XY").center(RET_X, half_y - gd / 2)
              .rect(gw, gd + 0.01).extrude(height + 1))
body = body.cut(end_groove)

# ---------------- 뚜껑 (별도 출력, 바닥면 = 아래) ----------------
lid = cq.Workplane("XY").box(LID_W, size_y, LID_T, centered=(True, True, False)).edges("|Z").fillet(0.5)
for sx in (-1, 1):
    for y in SNAP_Y:
        bump = (cq.Workplane("XZ").workplane(offset=-(y + SNAP_L / 2))
                .center(sx * (LID_W / 2 + SNAP_P - SNAP_R), LID_T / 2).circle(SNAP_R).extrude(SNAP_L))
        lid = lid.union(bump.intersect(cq.Workplane("XY").box(LID_W + 2, size_y, LID_T, centered=(True, True, False))))
        slot_x = sx * (LID_W / 2 - FLEX_T - FLEX_W / 2)
        lid = lid.cut(cq.Workplane("XY").center(slot_x, y).rect(FLEX_W, FLEX_L).extrude(LID_T))
lid_placed = lid.translate((RET_X, 0, z_lid))      # 조립 위치 (미리보기용)

LABEL_X = -(size_x / 2 + (RET_X - rec_w / 2)) / 2 - 0.5     # 각인은 뚜껑 반대편
if LABEL:
    for y, t in ((yP, "P"), (yN, "N")):
        body = body.cut(cq.Workplane("XY").workplane(offset=height - 0.5).center(LABEL_X, y)
                        .text(t, 7, 0.6, kind="bold", halign="center", valign="center"))

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cq.exporters.export(body, os.path.join(here, "emg_electrode_block.step"))
    cq.exporters.export(body, os.path.join(here, "emg_electrode_block.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(lid, os.path.join(here, "emg_electrode_lid.step"))
    cq.exporters.export(lid, os.path.join(here, "emg_electrode_lid.stl"), tolerance=0.01, angularTolerance=0.1)
    bb = body.val().BoundingBox()
    print(f"외형 {bb.xlen:.1f} × {bb.ylen:.1f} × {bb.zlen:.1f} mm (X 밴드 × Y 근섬유 × Z)")
    print(f"머리 포켓 Ø{2*R:.2f}, 목 끝 폭 {2*w_end:.2f} @ {NECK_L} mm, 포켓 깊이 {pocket_h:.2f}, 밴드 홈 {slot_w:.1f}×{slot_t:.1f}")
    print(f"P 부싱은 팔꿈치 쪽 끝, N 부싱은 손목 쪽 끝에서 {BOOT_L} mm 나온 뒤 N 전선은 180° 꺾여 윗면 홈으로 팔꿈치 쪽 배출")
    print(f"윗면 홈 {gw:.1f}×{gd:.1f} @X{RET_X}, 홈 아래 남는 두께 {TOP - LID_T - gd:.1f} mm (밴드 홈 위)")
    lb = lid.val().BoundingBox()
    print(f"뚜껑 {lb.xlen:.2f} × {lb.ylen:.1f} × {lb.zlen:.1f} mm, 오목부 폭 {rec_w:.1f}, 걸림 간섭 {SNAP_P - LID_CLR:.2f} mm × {len(SNAP_Y)}쌍")
