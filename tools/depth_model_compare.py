"""Single photo: DAv3 Small (our default) vs MoGe-2 ViT-S, both on CPU onnxruntime. Each depth map is unprojected
with ONE shared focal (MoGe's estimate) and point-rendered (z-buffer, 1 px points) from the photo camera turned about
the scene's median depth - the moved view where tearing shows: a ramp at an object edge becomes a smeared wall.
usage: python depth_compare.py <dav3 model.onnx> <moge model.onnx> <photo.jpg> <out.jpg> [long_side=1008]"""
import sys, time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

dav3_path, moge_path, photo, out = sys.argv[1:5]
long_side = int(sys.argv[5]) if len(sys.argv) > 5 else 1008
im = Image.open(photo).convert("RGB")
s = long_side / max(im.size)
W, H = (round(im.width * s / 14) * 14, round(im.height * s / 14) * 14)
im = im.resize((W, H), Image.LANCZOS)
rgb = np.asarray(im).astype(np.float32) / 255.0
opt = ort.SessionOptions(); opt.intra_op_num_threads = 8

# MoGe-2: [0,1] image, num_tokens sets the internal resolution (2500 = its finest level).
t = time.time()
moge = ort.InferenceSession(moge_path, opt, providers=["CPUExecutionProvider"])
o = dict(zip([x.name for x in moge.get_outputs()],
             moge.run(None, {"image": rgb.transpose(2, 0, 1)[None], "num_tokens": np.array(2500, dtype=np.int64)})))
pts = o["points"][0]                      # H,W,3 camera space (metric after * scale? scale applied below)
mask = o["mask"][0] > 0.5
sk = next(k for k in o if "scale" in k)
print("moge", f"{time.time() - t:.1f}s", pts.shape, "scale", o[sk])
zm = pts[..., 2] * float(np.ravel(o[sk])[0])
# MoGe's focal (pixels) from its own points: u - cx = f * X / Z.
u = np.arange(W)[None, :].repeat(H, 0) + 0.5 - W / 2
v = np.arange(H)[:, None].repeat(W, 1) + 0.5 - H / 2
ok = mask & (pts[..., 2] > 0)
xz, yz = pts[..., 0][ok] / pts[..., 2][ok], pts[..., 1][ok] / pts[..., 2][ok]
f = float((np.sum(u[ok] * xz) + np.sum(v[ok] * yz)) / (np.sum(xz * xz) + np.sum(yz * yz)))
print("focal px", f"{f:.1f}", "hfov", f"{np.degrees(2 * np.arctan(W / 2 / f)):.1f}")

# DAv3 Small: ImageNet-normalised, [1,1,3,H,W].
t = time.time()
dav3 = ort.InferenceSession(dav3_path, opt, providers=["CPUExecutionProvider"])
x = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
d = dav3.run(["predicted_depth"], {"pixel_values": x.transpose(2, 0, 1)[None, None]})[0][0, 0]
print("dav3", f"{time.time() - t:.1f}s", d.shape)
zd = d * np.median(zm[mask]) / np.median(d[mask])   # same median depth as MoGe, so the two views match


def moved(z, valid, deg, label):
    """Point-render the depth map from the photo camera orbited `deg` about the median depth."""
    X, Y = u * z / f, v * z / f
    c = np.median(z[valid])
    a = np.radians(deg)
    Xr = np.cos(a) * X + np.sin(a) * (z - c)
    Zr = -np.sin(a) * X + np.cos(a) * (z - c) + c
    keep = valid & (Zr > 0.05 * c)
    px = np.round(f * Xr[keep] / Zr[keep] + W / 2 - 0.5).astype(int)
    py = np.round(f * Y[keep] / Zr[keep] + H / 2 - 0.5).astype(int)
    zz, col = Zr[keep], rgb[keep]
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, zz, col = px[inb], py[inb], zz[inb], col[inb]
    order = np.argsort(-zz)                  # far first, near overwrites
    img = np.zeros((H, W, 3), np.float32) + np.array([1.0, 0.0, 1.0], np.float32)   # magenta = no point
    img[py[order], px[order]] = col[order]
    pim = Image.fromarray((img * 255).astype(np.uint8))
    ImageDraw.Draw(pim).text((8, 8), label, fill=(255, 255, 0))
    return pim


def depth_vis(z, valid, label):
    lo, hi = np.percentile(z[valid], [2, 98])
    g = np.clip((z - lo) / (hi - lo), 0, 1)
    pim = Image.fromarray((np.stack([1 - g, 1 - np.abs(g - 0.5) * 2, g], -1) * 255).astype(np.uint8))
    ImageDraw.Draw(pim).text((8, 8), label, fill=(255, 255, 255))
    return pim


all_valid = np.ones_like(mask)
rows = [
    [im, depth_vis(zd, all_valid, "DAv3 Small depth"), depth_vis(zm, mask, "MoGe-2 ViT-S depth")],
    [Image.new("RGB", (W, H)), moved(zd, all_valid, 20, "DAv3 Small, orbit 20"), moved(zm, mask, 20, "MoGe-2, orbit 20")],
    [Image.new("RGB", (W, H)), moved(zd, all_valid, 35, "DAv3 Small, orbit 35"), moved(zm, mask, 35, "MoGe-2, orbit 35")],
]
sheet = Image.new("RGB", (3 * W, 3 * H))
for r, row in enumerate(rows):
    for c, p in enumerate(row):
        sheet.paste(p, (c * W, r * H))
sheet.save(out, quality=88)
np.savez_compressed(out + ".npz", zd=zd, zm=zm, mask=mask, f=f)
print(out)
