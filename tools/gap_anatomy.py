"""Where a held-out PSNR gap between two trainers comes from (parity, 2026-10-09). For every held-out view, PSNR of each
tool's render against the photo:
  raw       as rendered
  affine    after the best per-image 3x4 colour fit to the photo (what is left is NOT exposure / white balance)
  blur s    both render and photo blurred by a Gaussian of sigma s (what is left is NOT fine detail)
A gap that shrinks after the affine fit is colour; one that shrinks after blurring is detail.

  python tools/gap_anatomy.py <gsplat renders dir> <shots dir> <scene tag prefix> <images dir>

The gsplat renders (val_stepN_XXXX.png, photo | render) are in sorted held-out order, the same order as the sorted photo
names every 8th (llffhold=8) - checked by the photo halves matching.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageFilter


def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2) + 1e-12)


def affine(src, dst):
    x = np.concatenate([src.reshape(-1, 3), np.ones((src.shape[0] * src.shape[1], 1))], 1)
    m, *_ = np.linalg.lstsq(x, dst.reshape(-1, 3), rcond=None)
    return np.clip((x @ m).reshape(src.shape), 0, 1)


def blur(a, s):
    return np.asarray(Image.fromarray((a * 255).round().astype(np.uint8)).filter(ImageFilter.GaussianBlur(s))) / 255.0


def main():
    gdir, shots, prefix, images = sys.argv[1:5]
    gs = sorted(glob.glob(os.path.join(gdir, "val_step*_*.png")))
    names = sorted(os.listdir(images))[::8]
    if len(gs) != len(names):
        sys.exit(f"{len(gs)} gsplat renders vs {len(names)} held-out photos")
    rows = []
    for f, name in zip(gs, names):
        stem = os.path.splitext(name)[0]
        ours_f = os.path.join(shots, f"{prefix}__held-{stem}-trainer.png")
        if not os.path.exists(ours_f):
            print("no render of ours for", stem); continue
        a = np.asarray(Image.open(f).convert("RGB")) / 255.0
        w = a.shape[1] // 2
        photo, g = a[:, :w], a[:, w:]
        o = np.asarray(Image.open(ours_f).convert("RGB")) / 255.0
        # A run that trained at another size (SpawnScene before 8cc8f05 rounded to even: 1558x1039 -> 1040) trained
        # against the photo RESIZED to it, so its render is scored against that (as tools/rescore.py does).
        po = photo
        if o.shape[:2] != photo.shape[:2]:
            po = np.asarray(Image.fromarray((photo * 255).round().astype(np.uint8))
                            .resize((o.shape[1], o.shape[0]), Image.BICUBIC)) / 255.0
        r = [psnr(g, photo), psnr(o, po), psnr(affine(g, photo), photo), psnr(affine(o, po), po)]
        for s in (2, 6):
            r += [psnr(blur(g, s), blur(photo, s)), psnr(blur(o, s), blur(po, s))]
        rows.append(r)
        print(f"{stem}: raw g {r[0]:.2f} o {r[1]:.2f} ({r[1]-r[0]:+.2f}) | affine {r[3]-r[2]:+.2f} | blur2 {r[5]-r[4]:+.2f} | blur6 {r[7]-r[6]:+.2f}")
    m = np.mean(rows, 0)
    print(f"MEAN over {len(rows)}: raw gsplat {m[0]:.2f} ours {m[1]:.2f} gap {m[1]-m[0]:+.2f}; after affine {m[3]-m[2]:+.2f}"
          f" ({m[2]:.2f} / {m[3]:.2f}); blur 2 {m[5]-m[4]:+.2f}; blur 6 {m[7]-m[6]:+.2f}")


if __name__ == "__main__":
    main()
