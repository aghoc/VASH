"""
export_worn_assembly.py — 착용 형태(팔 둘레 타원)로 조립한 밴드를 재질별 PLY 로 내보낸다.
Blender 사실적 렌더(render_blender.py)의 입력. 팔은 포함하지 않는다.
실행: python export_worn_assembly.py [출력폴더]
"""
import os, sys
import numpy as np
import pyvista as pv
import render_band as R

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R.HERE, "assembly_worn")
os.makedirs(OUT, exist_ok=True)

COLOR2MAT = {R.COL["block"]: "block", R.COL["band"]: "band", R.COL["boot"]: "boot",
             R.COL["wireN"]: "wireN", R.COL["wireP"]: "wireP", R.COL["harness"]: "harness"}


class Collector:
    def __init__(self):
        self.parts = {}

    def add_mesh(self, mesh, color=None, **_):
        self.parts.setdefault(COLOR2MAT[color], []).append(mesh.extract_surface().triangulate())


arm = R.ellipse_path(R.ARM_A, R.ARM_B)
s_blocks = (np.arange(R.N_BLOCKS) + 0.5) / R.N_BLOCKS * arm.L
col = Collector()
R.build(col, arm, s_blocks, (0, arm.L))

# 스냅 단자 금속 머리 (블록 바닥으로 0.3 mm 돌출) — 안쪽에서 보일 때용
for s in s_blocks:
    for yc in (R.B.yP, R.B.yN):
        head = pv.Cylinder(center=(0, yc, R.B.HEAD_H / 2 - R.B.PROTRUDE), direction=(0, 0, 1),
                           radius=R.B.HEAD_D / 2 - 0.05, height=R.B.HEAD_H, resolution=64).triangulate()
        col.parts.setdefault("snap", []).append(R.placed(head, arm, s).triangulate())

for name, meshes in col.parts.items():
    m = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
    m = m.extract_surface().triangulate().clean()
    m.points = m.points / 1000.0          # mm → m (Blender 단위)
    m.save(os.path.join(OUT, f"{name}.ply"), binary=True)
    print(name, m.n_points, m.n_cells)
