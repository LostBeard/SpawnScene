"""MI-GAN (ours) vs LaMa (big-lama, Apache-2.0) on the same holes, CPU onnxruntime, 512 crops.
Each case: a square crop around a rectangular hole (the kind of hole a depth edge opens: a foreground object removed),
sheet = holed input | MI-GAN | LaMa. Also prints the mean abs difference to the ORIGINAL pixels inside the hole - not
'correct' (the original shows the object, not what is behind it), only a check both produce sane values.
usage: inpaint_compare.py <migan.onnx> <lama_fp32.onnx> <photo> <out.jpg> x0,y0,x1,y1 [x0,y0,x1,y1 ...]"""
import sys, time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

migan_p, lama_p, photo, out = sys.argv[1:5]
boxes = [tuple(map(int, b.split(','))) for b in sys.argv[5:]]
im = Image.open(photo).convert("RGB")
opt = ort.SessionOptions(); opt.intra_op_num_threads = 8
mig = ort.InferenceSession(migan_p, opt, providers=["CPUExecutionProvider"])
lam = ort.InferenceSession(lama_p, opt, providers=["CPUExecutionProvider"])
rows = []
for (x0, y0, x1, y1) in boxes:
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(x1 - x0, y1 - y0)
    box = (max(0, cx - half), max(0, cy - half), min(im.width, cx + half), min(im.height, cy + half))
    crop = im.crop(box).resize((512, 512), Image.LANCZOS)
    sx, sy = 512 / (box[2] - box[0]), 512 / (box[3] - box[1])
    known = Image.new("L", (512, 512), 255)
    ImageDraw.Draw(known).rectangle([(x0 - box[0]) * sx, (y0 - box[1]) * sy, (x1 - box[0]) * sx, (y1 - box[1]) * sy], fill=0)
    k = np.asarray(known)
    img8 = np.asarray(crop)
    t = time.time()
    m = mig.run(None, {"image": img8.transpose(2, 0, 1)[None].astype(np.uint8), "mask": k[None, None].astype(np.uint8)})[0][0].transpose(1, 2, 0)
    tm = time.time() - t
    t = time.time()
    lo = lam.run(None, {"image": (img8.astype(np.float32) / 255).transpose(2, 0, 1)[None],
                        "mask": (k == 0).astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
    tl = time.time() - t
    if lo.max() <= 1.5: lo = lo * 255   # some exports return 0..1
    hole = k == 0
    holed = img8.copy(); holed[hole] = 0
    print(f"box {x0},{y0},{x1},{y1}: MI-GAN {tm:.2f}s, LaMa {tl:.2f}s (LaMa out range {lo.min():.1f}..{lo.max():.1f})")
    row = Image.new("RGB", (1536, 512))
    for j, a in enumerate([holed, m.astype(np.uint8), np.clip(lo, 0, 255).astype(np.uint8)]):
        p = Image.fromarray(a); ImageDraw.Draw(p).text((8, 8), ["hole", "MI-GAN", "LaMa"][j], fill=(255, 255, 0)); row.paste(p, (512 * j, 0))
    rows.append(row)
sheet = Image.new("RGB", (1536, 512 * len(rows)))
for i, r in enumerate(rows): sheet.paste(r, (0, 512 * i))
sheet.resize((1152, 384 * len(rows))).save(out, quality=85)
print(out)
