"""VASH 앱 아이콘: 진한 남색 둥근 사각형 + 파란 근전도 파형 (vash.ico = 바로가기, vash.png = 웹 탭 아이콘)"""
import math
from pathlib import Path
from PIL import Image, ImageDraw

N = 1024
img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((40, 40, N - 40, N - 40), radius=220, fill=(18, 26, 38, 255))

# 근전도 파형: 양쪽은 잔잔, 가운데 근육 수축 구간에서 커짐
x0, x1, yc = 170, N - 170, N // 2
amp = [0, 0.08, -0.08, 0.25, -0.55, 1.0, -0.85, 0.5, -0.25, 0.08, -0.08, 0]   # 꼭짓점 12 개
pts = [(x0 + (x1 - x0) * i / (len(amp) - 1), yc - 300 * a) for i, a in enumerate(amp)]
d.line(pts, fill=(76, 141, 255, 255), width=56, joint="curve")
for p in (pts[0], pts[-1]):
    d.ellipse((p[0] - 28, p[1] - 28, p[0] + 28, p[1] + 28), fill=(76, 141, 255, 255))

out = Path(__file__).parent / "assets"
img.save(out / "vash.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.resize((256, 256), Image.LANCZOS).save(out / "vash.png")
print("saved", out / "vash.ico")
