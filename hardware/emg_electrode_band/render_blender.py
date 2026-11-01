"""
render_blender.py — 착용 형태 밴드 조립품(팔 없음) 사실적 렌더 (Blender Cycles)

1) python export_worn_assembly.py <PLY폴더>
2) blender -b -P render_blender.py -- <PLY폴더> <출력.png> [샘플수] [가로] [세로] [카메라: front|top|inner]
"""
import sys, os, math
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = argv[0], argv[1]
SAMPLES = int(argv[2]) if len(argv) > 2 else 128
W = int(argv[3]) if len(argv) > 3 else 1600
H = int(argv[4]) if len(argv) > 4 else 1150
VIEW = argv[5] if len(argv) > 5 else "front"
# 선택: 세로 띠 나눠 렌더 (번호 k / 총 n) — 시간 제한이 있을 때 띠별로 렌더 후 합치기
STRIP = tuple(map(int, argv[6].split("/"))) if len(argv) > 6 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def hexrgb(h, a=1.0):
    h = h.lstrip("#"); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [a]   # sRGB → linear


def material(name, color, rough, metal=0.0, spec=0.5, bump=0.0, bump_scale=800.0, bevel=0.0, coat=0.0, sss=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = hexrgb(color)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular"].default_value = spec
    b.inputs["Clearcoat"].default_value = coat
    if sss:
        b.inputs["Subsurface"].default_value = sss
        b.inputs["Subsurface Color"].default_value = hexrgb(color)
    normal_in = None
    if bevel:
        bv = nt.nodes.new("ShaderNodeBevel"); bv.inputs["Radius"].default_value = bevel; bv.samples = 8
        normal_in = bv.outputs["Normal"]
    if bump:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = bump_scale
        nz.inputs["Detail"].default_value = 6
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.0002
        nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
        if normal_in is not None:
            nt.links.new(normal_in, bp.inputs["Normal"])
        normal_in = bp.outputs["Normal"]
    if normal_in is not None:
        nt.links.new(normal_in, b.inputs["Normal"])
    return m


def strap_material():
    """직물 밴드: 가는 줄무늬(Wave) + 노이즈 범프"""
    m = material("band", "#33363d", 0.85, spec=0.3)
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "Y"
    wv.inputs["Scale"].default_value = 900; wv.inputs["Distortion"].default_value = 0.5
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 3000
    mix = nt.nodes.new("ShaderNodeMath"); mix.operation = "ADD"
    nt.links.new(tc.outputs["Object"], wv.inputs["Vector"]); nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], mix.inputs[0]); nt.links.new(nz.outputs["Fac"], mix.inputs[1])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.45; bp.inputs["Distance"].default_value = 0.0003
    nt.links.new(mix.outputs["Value"], bp.inputs["Height"]); nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    # 색 약간 얼룩
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = hexrgb("#2a2c31"); ramp.color_ramp.elements[1].color = hexrgb("#3d4048")
    nt.links.new(wv.outputs["Fac"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


MATS = {
    "block":   material("block", "#141518", 0.55, spec=0.45, bump=0.12, bump_scale=2500, bevel=0.0004),  # 무광 검정 수지
    "band":    strap_material(),
    "boot":    material("boot", "#9e0a0d", 0.42, spec=0.5, sss=0.05),                 # 빨강 고무 부싱
    "wireN":   material("wireN", "#0d0d0e", 0.32, spec=0.5, coat=0.2),                 # 검정 PVC
    "wireP":   material("wireP", "#8f0b10", 0.32, spec=0.5, coat=0.2),                 # 빨강 PVC
    "harness": material("harness", "#8a919b", 0.38, spec=0.5, coat=0.1),               # 회색 묶음 케이블
    "snap":    material("snap", "#d8dadc", 0.22, metal=1.0),                           # 니켈 도금 스냅
}

zmin = 1e9
for name, mat in MATS.items():
    p = os.path.join(SRC, f"{name}.ply")
    if not os.path.exists(p):
        continue
    bpy.ops.import_mesh.ply(filepath=p)
    ob = bpy.context.selected_objects[0]; ob.name = name
    ob.data.materials.clear(); ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    ob.data.use_auto_smooth = True
    ob.data.auto_smooth_angle = math.radians(35 if name == "block" else 70)
    zmin = min(zmin, min((ob.matrix_world @ v.co).z for v in ob.data.vertices))

# ---- 바닥: 밝은 무광 스튜디오 판 (휘어진 배경) ----
bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, zmin - 0.0002))
floor = bpy.context.object; floor.name = "floor"
fm = material("floor", "#e9e7e3", 0.6, spec=0.3); floor.data.materials.append(fm)

# ---- 조명 ----
def area(name, loc, target, size, power, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    L = bpy.context.object; L.name = name; L.data.size = size; L.data.energy = power; L.data.color = color
    d = Vector(target) - Vector(loc); L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return L

area("key",  (-0.25, -0.30, 0.45), (0, 0, 0), 0.35, 4.5, (1.0, 0.97, 0.93))
area("fill", (0.40, -0.10, 0.20), (0, 0, 0), 0.50, 1.6, (0.92, 0.96, 1.0))
area("rim",  (0.05, 0.40, 0.30), (0, 0, 0), 0.25, 3.5)
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = hexrgb("#f2f3f5")
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18

# ---- 카메라 (이전 렌더와 같은 구도, mm → m) ----
CAMS = {"front": ((0.170, -0.250, 0.230), (0, -0.012, -0.005), 30),
        "top":   ((0.020, -0.120, 0.330), (0, -0.010, 0.000), 30),
        "inner": ((0.060, 0.300, 0.060), (0, -0.010, 0.000), 30)}
cpos, ctgt, fov = CAMS[VIEW]
bpy.ops.object.camera_add(location=cpos)
cam = bpy.context.object; sc.camera = cam
cam.rotation_euler = (Vector(ctgt) - Vector(cpos)).to_track_quat("-Z", "Y").to_euler()
cam.data.sensor_fit = "VERTICAL"; cam.data.angle = math.radians(fov)
cam.data.dof.use_dof = True; cam.data.dof.focus_distance = (Vector(ctgt) - Vector(cpos)).length
cam.data.dof.aperture_fstop = 5.6

# ---- 렌더 설정 ----
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
sc.cycles.samples = SAMPLES; sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.02
sc.cycles.use_denoising = True
try:
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
except Exception:
    pass
sc.cycles.max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.resolution_percentage = 100
sc.view_settings.view_transform = "Filmic"; sc.view_settings.look = "Medium High Contrast"
if STRIP:
    k, n = STRIP
    sc.render.use_border = True; sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_max_x = 0.0, 1.0
    sc.render.border_min_y, sc.render.border_max_y = 1 - (k + 1) / n, 1 - k / n   # k=0 이 맨 위
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("saved", OUT)
