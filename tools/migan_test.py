"""MI-GAN (plain migan.onnx, 512x512) on a crop of a photo with a rectangular hole: does it paint plausible background?
usage: python migan_test.py <model.onnx> <photo.jpg> <out.png> x0 y0 x1 y1 (hole in photo pixels)"""
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

model, photo, out = sys.argv[1], sys.argv[2], sys.argv[3]
x0, y0, x1, y1 = map(int, sys.argv[4:8])
im = Image.open(photo).convert("RGB")
# A square crop around the hole, 2x its size, scaled to 512.
cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
half = max(x1 - x0, y1 - y0)
box = (max(0, cx - half), max(0, cy - half), min(im.width, cx + half), min(im.height, cy + half))
crop = im.crop(box).resize((512, 512), Image.LANCZOS)
sx, sy = 512 / (box[2] - box[0]), 512 / (box[3] - box[1])
mask = Image.new("L", (512, 512), 255)
ImageDraw.Draw(mask).rectangle([(x0 - box[0]) * sx, (y0 - box[1]) * sy, (x1 - box[0]) * sx, (y1 - box[1]) * sy], fill=0)
img = np.asarray(crop).transpose(2, 0, 1)[None].astype(np.uint8)
msk = np.asarray(mask)[None, None].astype(np.uint8)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
res = sess.run(None, {"image": img, "mask": msk})[0][0].transpose(1, 2, 0)
holed = np.asarray(crop).copy()
holed[np.asarray(mask) == 0] = 0
sheet = Image.new("RGB", (1024, 512))
sheet.paste(Image.fromarray(holed), (0, 0))
sheet.paste(Image.fromarray(res.astype(np.uint8)), (512, 0))
sheet.save(out)
print(out, res.shape, res.dtype)
