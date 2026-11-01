import sys
sys.path.insert(0, "/sessions/epic-admiring-lamport/mnt/ads_bringup/cad")
import emg_electrode_block as B, cadquery as cq, cairosvg
from PIL import Image, ImageDraw, ImageFont
body = B.body.rotate((0,0,0),(0,1,0),90)
R = lambda d: (d[2], d[1], -d[0])
views = [("top_iso", (1.2, -1.4, 1.0), "윗면 쪽 (되돌림 전선 홈 · 밴드 홈 · P/N 각인)"),
         ("bottom_iso", (1.2, -1.4, -1.1), "바닥면 쪽 (피부 쪽: 열쇠구멍 포켓 2개, 꼬리는 양 끝으로)"),
         ("side_x", (1, 0, 0), "전선 출구 면 (+X, 밴드 방향)"),
         ("front_y", (0, -1, 0), "측면 (Y 방향, 피부 쪽 아래)")]
fn = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ims = []
for name, d, title in views:
    opt = {"width": 640, "height": 440, "marginLeft": 40, "marginTop": 40, "showAxes": False,
           "projectionDir": R(d), "strokeWidth": 0.4, "strokeColor": (25, 25, 25), "hiddenColor": (180, 180, 190), "showHidden": True}
    cq.exporters.export(body, f"/tmp/x_{name}.svg", opt=opt)
    cairosvg.svg2png(url=f"/tmp/x_{name}.svg", write_to=f"/tmp/x_{name}.png", background_color="white")
    ims.append((Image.open(f"/tmp/x_{name}.png").convert("RGB"), title))
W, H = ims[0][0].size
out = Image.new("RGB", (W * 2, H * 2 + 70), "white"); dr = ImageDraw.Draw(out)
bb = B.body.val().BoundingBox()
dr.text((20, 18), f"EMG 전극 블록 v5 (한 방향 배출 + 딸깍 뚜껑)  {bb.xlen:.1f}(밴드) × {bb.ylen:.1f}(근섬유) × {bb.zlen:.1f}(높이) mm  |  P–N {B.PITCH:.0f} mm  |  N 전선 180° 되돌림 → 윗면 홈 → 팔꿈치 쪽",
        font=ImageFont.truetype(fn, 20), fill=(0, 0, 0))
for i, (im, t) in enumerate(ims):
    x = (i % 2) * W; y = 70 + (i // 2) * H; out.paste(im, (x, y))
    dr.text((x + 15, y + 2), t, font=ImageFont.truetype(fn, 17), fill=(200, 60, 0))
out.save("/sessions/epic-admiring-lamport/mnt/ads_bringup/cad/emg_electrode_block_preview.png")
out.save("/sessions/epic-admiring-lamport/mnt/outputs/blk.png")
print("ok")
