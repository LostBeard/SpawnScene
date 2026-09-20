"""Score captured novel views against the TempleRing ground-truth photographs.

    python tools/score_novel_view.py <RUN_TAG> [<BASELINE_RUN_TAG>]

Reports PSNR and SSIM per view. With a baseline run tag, also reports the delta, which is the
number that says whether a change helped.

Only held-out views are scored by default: a view the reconstruction was built from is not
evidence about novel-view quality. Training views are printed separately, marked, because they
are a useful upper bound (if we cannot reproduce a view we trained on, nothing downstream
matters).

SSIM is implemented here rather than pulled from skimage to avoid adding a dependency; it is
the standard Wang et al. formulation with an 11x11 Gaussian window, sigma 1.5, on the luma
channel.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# Repo root, derived from this file so the tool works from any clone.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "SpawnScene", "wwwroot", "datasets", "TempleRing")


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def _filter2(img, k):
    """'valid' 2-D correlation. Small images, so a strided view is plenty fast."""
    kh, kw = k.shape
    h, w = img.shape
    if h < kh or w < kw:
        raise ValueError("image smaller than the SSIM window")
    shape = (h - kh + 1, w - kw + 1, kh, kw)
    strides = img.strides * 2
    windows = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return np.einsum("ijkl,kl->ij", windows, k)


def ssim(a, b):
    """Mean SSIM on luma. a, b are HxW float in [0,1]."""
    k = _gaussian_kernel()
    c1, c2 = (0.01 ** 2), (0.03 ** 2)

    mu_a, mu_b = _filter2(a, k), _filter2(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sa = _filter2(a * a, k) - mu_a2
    sb = _filter2(b * b, k) - mu_b2
    sab = _filter2(a * b, k) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float(np.mean(num / den))


def score_run(tag):
    out = os.path.join(ROOT, "_shots", "novelview", tag)
    meta_path = os.path.join(out, "poses.json")
    if not os.path.isdir(out):
        sys.exit(f"no such run: {out}")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    training = set(meta.get("trainingViews") or [])

    rows = []
    for name in sorted(os.listdir(out)):
        if not name.lower().endswith(".png"):
            continue
        gt_path = os.path.join(DATASET, name)
        if not os.path.exists(gt_path):
            continue
        rendered = load_rgb(os.path.join(out, name))
        gt = load_rgb(gt_path, size=(rendered.shape[1], rendered.shape[0]))
        rows.append({
            "view": name,
            "held_out": name not in training,
            "psnr": psnr(rendered, gt),
            "ssim": ssim(luma(rendered), luma(gt)),
        })
    return rows, meta


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    rows, meta = score_run(tag)
    if not rows:
        sys.exit("no scored views - was anything captured?")

    base = {}
    if len(sys.argv) > 2:
        brows, _ = score_run(sys.argv[2])
        base = {r["view"]: r for r in brows}

    print(f"\nrun: {tag}")
    if meta.get("trainingViews"):
        print(f"training views (excluded from the headline): {', '.join(meta['trainingViews'])}")
    print()
    hdr = f"{'view':<20}{'kind':<10}{'PSNR dB':>10}{'SSIM':>9}"
    if base:
        hdr += f"{'dPSNR':>9}{'dSSIM':>9}"
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        kind = "held-out" if r["held_out"] else "TRAINING"
        line = f"{r['view']:<20}{kind:<10}{r['psnr']:>10.2f}{r['ssim']:>9.4f}"
        if base and r["view"] in base:
            line += f"{r['psnr'] - base[r['view']]['psnr']:>+9.2f}{r['ssim'] - base[r['view']]['ssim']:>+9.4f}"
        print(line)

    held = [r for r in rows if r["held_out"]]
    if held:
        mp = float(np.mean([r["psnr"] for r in held]))
        ms = float(np.mean([r["ssim"] for r in held]))
        print("-" * len(hdr))
        line = f"{'MEAN (held-out)':<30}{mp:>10.2f}{ms:>9.4f}"
        if base:
            bheld = [base[r["view"]] for r in held if r["view"] in base]
            if bheld:
                bp = float(np.mean([r["psnr"] for r in bheld]))
                bs = float(np.mean([r["ssim"] for r in bheld]))
                line += f"{mp - bp:>+9.2f}{ms - bs:>+9.4f}"
        print(line)
    print()


if __name__ == "__main__":
    main()
