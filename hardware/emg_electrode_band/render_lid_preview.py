"""render_lid_preview.py — v5 딸깍 뚜껑 미리보기 (분해도 · 뚜껑 · 단면)"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import emg_electrode_block as B, cadquery as cq, cairosvg
from PIL import Image, ImageDraw, ImageFont
R = lambda d: (d[2], d[1], -d[0])
rot = lambda s: s.rotate((0, 0, 0), (0, 1, 0), 90)
fn = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
comp = lambda *ws: cq.Workplane("XY").add(cq.Compound.makeCompound([w.val() for w in ws]))
explode = comp(B.body, B.lid_placed.translate((0, 0, 9)))
asm = comp(B.body, B.lid_placed)
lid_flip = B.lid.rotate((0, 0, 0), (0, 1, 0), 180)
# 단면: 돌기 위치(y = 0) 얇게 잘라 Y 방향으로 투영, 뚜껑 주변만
cut = cq.Workplane("XY").box(16, 0.2, 9, centered=(True, True, False)).translate((B.RET_X, 0, B.height - 8.5))
sec = comp(B.body.intersect(cut), B.lid_placed.intersect(cut))
views = [(explode, (1.2, -1.4, 1.0), "분해도: 뚜껑을 위에서 눌러 끼움", 640, 440),
         (lid_flip, (1.2, -1.4, 1.0), "뚜껑 아랫면: 양옆 돌기 3쌍 + 휘는 보(돌기 뒤 틈)", 640, 440),
         (asm, (1.2, -1.4, 1.0), "조립 후 (전선은 홈 안, 뚜껑이 덮음)", 640, 440),
         (sec, (0, -1, 0), "단면 (돌기 위치): 전선 홈 2.4×2.4 + 뚜껑 1.2, 돌기가 벽 홈에 걸림", 640, 440)]
ims = []
for i, (shape, d, title, w, h) in enumerate(views):
    opt = {"width": w, "height": h, "marginLeft": 40, "marginTop": 40, "showAxes": False, "projectionDir": R(d),
           "strokeWidth": 0.4 if i < 3 else 0.15, "strokeColor": (25, 25, 25), "hiddenColor": (180, 180, 190), "showHidden": i < 3}
    cq.exporters.export(rot(shape), f"/tmp/l_{i}.svg", opt=opt)
    cairosvg.svg2png(url=f"/tmp/l_{i}.svg", write_to=f"/tmp/l_{i}.png", background_color="white")
    ims.append((Image.open(f"/tmp/l_{i}.png").convert("RGB"), title))
W, H = ims[0][0].size
out = Image.new("RGB", (W * 2, H * 2 + 70), "white"); dr = ImageDraw.Draw(out)
lb = B.lid.val().BoundingBox()
dr.text((20, 18), f"v5 전선 홈 딸깍 뚜껑  {lb.xlen - 2*(B.SNAP_P):.1f}(돌기 제외) × {lb.ylen:.0f} × {B.LID_T} mm  |  걸림 간섭 {B.SNAP_P - B.LID_CLR:.1f} mm × 3쌍  |  블록 높이 {B.height:.1f} mm",
        font=ImageFont.truetype(fn, 20), fill=(0, 0, 0))
for i, (im, t) in enumerate(ims):
    x = (i % 2) * W; y = 70 + (i // 2) * H; out.paste(im, (x, y))
    dr.text((x + 15, y + 2), t, font=ImageFont.truetype(fn, 17), fill=(200, 60, 0))
out.save(os.path.join(HERE, "emg_electrode_lid_preview.png"))
print("ok")
