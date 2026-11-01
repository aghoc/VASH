"""
render_band.py — 전극 블록 v3 × 8 + 밴드 + 스냅 단자 전선 조립 렌더링 (PyVista/VTK, 오프스크린)

실행:  python render_band.py
결과:  emg_band_flat.png  (밴드를 펼쳐 책상 위에 놓은 모습, 참고 사진 구도)
       emg_band_worn.png  (팔뚝 둘레에 착용한 모습)
"""
import os, sys, json, subprocess, types, tempfile
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))

# CadQuery(OCP) 와 VTK 렌더러를 한 프로세스에서 같이 쓰면 화면이 깨지므로
# 블록 메쉬·치수는 별도 프로세스에서 받는다.
_TMP_STL = os.path.join(tempfile.gettempdir(), "emg_block_tmp.stl")
_PARAMS = ["HEAD_D", "HEAD_H", "PROTRUDE", "BOOT_D", "BOOT_L", "STRAP_W", "STRAP_T", "WIRE_D", "RET_X",
           "yP", "yN", "half_y", "z_roof", "slot_t", "height", "size_x"]
_code = (f"import sys,json; sys.path.insert(0,{HERE!r}); import emg_electrode_block as B; "
         f"B.body.val().exportStl({_TMP_STL!r}, 0.03, 0.15); "
         f"print(json.dumps({{k: float(getattr(B, k)) for k in {_PARAMS!r}}}))")
_out = subprocess.run([sys.executable, "-c", _code], capture_output=True, text=True, check=True).stdout
B = types.SimpleNamespace(**json.loads(_out.strip().splitlines()[-1]))
BLK = pv.read(_TMP_STL).clean()

N_BLOCKS = 8
ARM_A, ARM_B = 45.0, 36.0               # 팔 단면 타원 반축 (둘레 ≈ 255 mm, 50% L 부근 가정)
D_BAND = B.z_roof + B.slot_t / 2         # 피부(z=0) → 밴드 중심 높이
Y_HARN = -(B.half_y + B.BOOT_L + 9)      # 팔꿈치 쪽 전선 묶음 위치

COL = dict(block="#1d2028", band="#4b505c", boot="#c62828", wireN="#151515", wireP="#b71c1c",
           harness="#9aa3ae", table="#d9dde3", skin="#e6c3a5")


# ---------------- 경로(밴드 중심선) ----------------
class Path2D:
    """XZ 평면의 닫힌/열린 곡선. frame(s) → 위치 p, 접선 tg, 바깥 법선 n (모두 2D)"""
    def __init__(self, pts, closed):
        self.pts, self.closed = pts, closed
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.s = np.r_[0, np.cumsum(d)]
        self.L = self.s[-1]

    def frame(self, s):
        x = np.interp(s, self.s, self.pts[:, 0]); z = np.interp(s, self.s, self.pts[:, 1])
        h = 0.5
        x2 = np.interp(min(s + h, self.L), self.s, self.pts[:, 0]); z2 = np.interp(min(s + h, self.L), self.s, self.pts[:, 1])
        x1 = np.interp(max(s - h, 0), self.s, self.pts[:, 0]); z1 = np.interp(max(s - h, 0), self.s, self.pts[:, 1])
        tg = np.array([x2 - x1, z2 - z1]); tg /= np.linalg.norm(tg)
        n = np.array([-tg[1], tg[0]])         # 접선을 반시계 90° → (평면: +Z, 시계 방향 타원: 바깥)
        return np.array([x, z]), tg, n


def ellipse_path(a, b):
    t = np.linspace(2 * np.pi, 0, 4001)          # 시계 방향 → 법선이 바깥쪽
    return Path2D(np.c_[a * np.cos(t), b * np.sin(t)], True)


def to_world(V, path, s):
    """블록 로컬(X=밴드 방향, Y=근섬유/팔 축, Z=피부→바깥) → 월드(X,Z=단면, Y=팔 축)"""
    p, tg, n = path.frame(s)
    ex = np.array([tg[0], 0, tg[1]]); ez = np.array([n[0], 0, n[1]])
    V = np.asarray(V)
    return np.array([p[0], 0, p[1]]) + V[:, :1] * ex + V[:, 1:2] * np.array([0, 1, 0]) + V[:, 2:3] * ez


def placed(mesh, path, s):
    return pv.PolyData(to_world(mesh.points, path, s), mesh.faces)


def ribbon(path, offset, y0, width, thick, s0=None, s1=None, n=600):
    """경로를 법선 방향 offset 만큼 띄운 띠 (단면 width(Y) × thick)"""
    s0 = 0 if s0 is None else s0; s1 = path.L if s1 is None else s1
    ss = np.linspace(s0, s1, n)
    rows = []
    for s in ss:
        for dz, dy in ((-thick / 2, -width / 2), (thick / 2, -width / 2), (thick / 2, width / 2), (-thick / 2, width / 2)):
            rows.append([0.0, y0 + dy, offset + dz])
    loc = np.array(rows)
    pts = np.vstack([to_world(loc[4 * i:4 * i + 4], path, s) for i, s in enumerate(ss)])
    faces = []
    for i in range(n - 1):
        for k in range(4):
            a, b = 4 * i + k, 4 * i + (k + 1) % 4
            faces += [4, a, b, 4 * (i + 1) + (k + 1) % 4, 4 * (i + 1) + k]
    faces += [4, 0, 1, 2, 3, 4, 4 * (n - 1) + 3, 4 * (n - 1) + 2, 4 * (n - 1) + 1, 4 * (n - 1)]
    return pv.PolyData(pts, np.array(faces)).triangulate().compute_normals(auto_orient_normals=True)


def build(pl, path, s_blocks, harness_span):
    kw_blk = dict(color=COL["block"], smooth_shading=True, split_sharp_edges=True,
                  specular=0.25, specular_power=18, ambient=0.12, diffuse=0.85)
    # 밴드 (블록 밴드 홈을 지나감)
    pl.add_mesh(ribbon(path, D_BAND, 0.0, B.STRAP_W, B.STRAP_T), color=COL["band"],
                smooth_shading=True, specular=0.1, ambient=0.15)
    # 팔꿈치 쪽 납작 전선 묶음 (참고 사진의 평판 케이블 역할)
    pl.add_mesh(ribbon(path, 1.2, Y_HARN, 9.0, 1.6, *harness_span), color=COL["harness"],
                smooth_shading=True, specular=0.3, ambient=0.2)

    zb = B.BOOT_D / 2 + 0.2
    yE = B.half_y + B.BOOT_L
    ztop = B.height - (B.WIRE_D + 0.2) / 2
    w = B.WIRE_D / 2
    for s in s_blocks:
        pl.add_mesh(placed(BLK, path, s).compute_normals(split_vertices=True, feature_angle=35), **kw_blk)
        for sgn in (-1, +1):        # 부싱: P = 팔꿈치(−Y), N = 손목(+Y)
            boot = pv.Cylinder(center=(0, sgn * (B.half_y + B.BOOT_L / 2), zb), direction=(0, 1, 0),
                               radius=B.BOOT_D / 2, height=B.BOOT_L, resolution=32)
            pl.add_mesh(placed(boot, path, s), color=COL["boot"], smooth_shading=True, specular=0.3)
        # N 전선: 손목 쪽 부싱 → U턴 → 윗면 되돌림 홈 → 팔꿈치 쪽 묶음
        n_path = np.array([[0, yE, zb], [0, yE + 4, zb + 1.5], [B.RET_X * 0.5, yE + 5.5, ztop * 0.6],
                           [B.RET_X, yE + 2, ztop + 0.5], [B.RET_X, B.half_y - 1, ztop], [B.RET_X, 0, ztop],
                           [B.RET_X, -B.half_y + 1, ztop], [B.RET_X, -B.half_y - 5, ztop - 2],
                           [B.RET_X * 0.6, Y_HARN + 6, 4.0], [B.RET_X * 0.4, Y_HARN + 3, 2.2]])
        p_path = np.array([[0, -yE, zb], [0, -yE - 2, zb], [-1.0, Y_HARN + 4, 2.6], [-1.2, Y_HARN + 2.5, 2.2]])
        for pts, col in ((n_path, COL["wireN"]), (p_path, COL["wireP"])):
            pl.add_mesh(pv.Spline(to_world(pts, path, s), 150).tube(radius=w, n_sides=16),
                        color=col, smooth_shading=True, specular=0.4)


def save(pl, fn, cam, view_angle=30):
    pl.camera_position = cam
    pl.camera.view_angle = view_angle
    pl.render()
    pl.screenshot(os.path.join(HERE, fn))
    pl.close()
    print("saved", fn)


def new_plotter():
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1150))
    pl.set_background("#f3f5f8")
    pl.enable_lightkit()
    return pl


if __name__ == "__main__":
    # ===== 1) 펼친 밴드 (참고 사진 구도) =====
    pitch = 2 * np.pi * np.sqrt((ARM_A ** 2 + ARM_B ** 2) / 2) / N_BLOCKS    # ≈ 착용 시 블록 간격
    L_flat = pitch * N_BLOCKS + 60
    xs = np.linspace(-L_flat / 2, L_flat / 2, 400)
    flat = Path2D(np.c_[xs, np.zeros_like(xs)], False)
    s_blocks = 30 + pitch * (np.arange(N_BLOCKS) + 0.5)
    pl = new_plotter()
    build(pl, flat, s_blocks, (s_blocks[0] - 8, flat.L))
    pl.add_mesh(pv.Plane(center=(0, -10, -0.4), direction=(0, 0, 1), i_size=L_flat + 120, j_size=200),
                color=COL["table"], ambient=0.35, diffuse=0.6)
    save(pl, "emg_band_flat.png", [(-175, -300, 215), (12, -12, 0), (0, 0, 1)], 30)

    # ===== 2) 팔뚝 착용 =====
    arm = ellipse_path(ARM_A, ARM_B)
    s_blocks = (np.arange(N_BLOCKS) + 0.5) / N_BLOCKS * arm.L
    pl = new_plotter()
    build(pl, arm, s_blocks, (0, arm.L))
    # 팔뚝(단순 타원 기둥, 젤 두께만큼 안쪽)
    cyl = pv.Cylinder(center=(0, -10, 0), direction=(0, 1, 0), radius=1.0, height=190, resolution=160, capping=True)
    cyl = cyl.triangulate().subdivide(2, "linear")
    P = cyl.points.copy(); P[:, 0] *= ARM_A - 0.8; P[:, 2] *= ARM_B - 0.8; cyl.points = P
    cyl = cyl.compute_normals(split_vertices=True, feature_angle=60)
    pl.add_mesh(cyl, color=COL["skin"], smooth_shading=True, ambient=0.25, specular=0.05)
    save(pl, "emg_band_worn.png", [(170, -250, 230), (0, -12, 0), (0, 0, 1)], 30)
