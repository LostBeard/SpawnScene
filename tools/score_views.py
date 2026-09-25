"""Per-view scores of view-* captures against their photos: PSNR, SSIM, and SHARPNESS.

    python tools/score_views.py <Dataset> <RUN_TAG> [<RUN_TAG> ...]

Sharpness = mean |Laplacian| of the render / the same of the photo, on luma at the render's size. A soft
render reads well below 1.0 while its PSNR can still look fine - TJ judges sharpness by eye, this puts a
number next to it. Views are paired by PHOTO across runs, as compose_views.py does.
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SHOTS = os.path.join(ROOT, '_shots', 'dataset')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _photos import load_photo  # noqa: E402


def luma(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def laplacian_energy(y):
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(np.mean(np.abs(lap)))


def ssim(a, b):
    # Wang et al., 11x11 Gaussian sigma 1.5, on luma, as score_novel_view.py.
    from numpy.lib.stride_tricks import sliding_window_view as swv
    g = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5 ** 2)); g /= g.sum()
    k = np.outer(g, g)
    def filt(x):
        w = swv(x, (11, 11))
        return np.einsum('ijkl,kl->ij', w, k)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = filt(a), filt(b)
    saa = filt(a * a) - mu_a ** 2; sbb = filt(b * b) - mu_b ** 2; sab = filt(a * b) - mu_a * mu_b
    return float(np.mean(((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name, tags = sys.argv[1], sys.argv[2:]
    for tag in tags:
        side = json.load(open(os.path.join(SHOTS, f'{name}__{tag}__views.json')))
        rows = []
        for k, v in sorted(side['views'].items(), key=lambda kv: (kv[0].split('-')[0] != 'held', kv[1]['photo'])):
            shot = Image.open(os.path.join(SHOTS, f'{name}__{tag}__view-{k}.png')).convert('RGB')
            photo = load_photo(side, name, tag, k).resize(shot.size, Image.LANCZOS)
            a = np.asarray(shot, np.float64); b = np.asarray(photo, np.float64)
            psnr = 10 * np.log10(255 ** 2 / max(np.mean((a - b) ** 2), 1e-12))
            ya, yb = luma(a), luma(b)
            rows.append((k, os.path.basename(v['photo']), psnr, ssim(ya, yb), laplacian_energy(ya) / laplacian_energy(yb)))
        print(f'== {tag}')
        for k, ph, p, s, sh in rows:
            print(f'   {k:9s} {ph:12s} PSNR {p:5.2f}  SSIM {s:.3f}  sharpness {sh:.2f}')
        held = [r for r in rows if r[0].startswith('held')]
        if held:
            print(f'   held-out mean: PSNR {np.mean([r[2] for r in held]):.2f}  SSIM {np.mean([r[3] for r in held]):.3f}  '
                  f'sharpness {np.mean([r[4] for r in held]):.2f}')


if __name__ == '__main__':
    main()
