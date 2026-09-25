"""Trainer render vs viewer capture of the SAME pose, from the dataset harness dumps alone (no photos, no server).

    python tools/trainer_viewer_shift.py Truck <tag> [<tag> ...] [--min-agreement DB]

With --min-agreement, exits 1 if any run's mean agreement is below DB: the viewer must show what the trainer
trained. Since the viewer renders the trainer's own footprint (2026-09-25) agreement is ~55.7 dB (8-bit rounding);
the eigen-quad viewer before it measured 26.7 dB on TruckFull 30K. 45 separates the two with room.

Per view: PSNR of viewer against trainer, mean signed (viewer - trainer) per channel in 0..255 levels, and the
signed difference binned by the trainer's value. A colour-pipeline transfer function shows as the SAME binned
curve on every view; a scene-dependent renderer difference does not (see the 2026-09-25 viewer-gap notes).
"""
import os
import sys

import numpy as np
from PIL import Image

SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shots', 'dataset')
BINS = [0, 16, 32, 64, 96, 128, 160, 192, 224, 256]


def views_of(name, tag):
    """Every view with both a trainer dump and a viewer capture, discovered from the files (datasets differ)."""
    import glob
    prefix = os.path.join(SHOTS, f'{name}__{tag}__view-')
    keys = [p[len(prefix):-len('-trainer.png')] for p in glob.glob(prefix + '*-trainer.png')]
    return sorted(k for k in keys if os.path.exists(prefix + k + '.png'))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = sys.argv[1:]
    min_agree = None
    if '--min-agreement' in args:
        i = args.index('--min-agreement')
        min_agree = float(args[i + 1])
        del args[i:i + 2]
    name, tags = args[0], args[1:]
    failed = []
    for tag in tags:
        print(f'== {tag}')
        psnrs = []
        for k in views_of(name, tag):
            t = os.path.join(SHOTS, f'{name}__{tag}__view-{k}-trainer.png')
            v = os.path.join(SHOTS, f'{name}__{tag}__view-{k}.png')
            if not (os.path.exists(t) and os.path.exists(v)):
                print(f'   {k:9s} missing')
                continue
            a = np.asarray(Image.open(t).convert('RGB'), np.float64)
            b = np.asarray(Image.open(v).convert('RGB').resize((a.shape[1], a.shape[0]), Image.LANCZOS), np.float64)
            d = b - a
            psnr = 10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12))
            psnrs.append(psnr)
            flat_a, flat_d = a.ravel(), d.ravel()
            binned = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = (flat_a >= lo) & (flat_a < hi)
                binned.append(f'{flat_d[m].mean():+5.1f}' if m.sum() > 500 else '    .')
            print(f'   {k:9s} {psnr:5.2f} dB  signed {d[..., 0].mean():+5.2f} {d[..., 1].mean():+5.2f} '
                  f'{d[..., 2].mean():+5.2f}  by trainer value: {" ".join(binned)}')
        if psnrs:
            print(f'   mean agreement {np.mean(psnrs):.2f} dB   (bins start {" ".join(str(x) for x in BINS[:-1])})')
        if min_agree is not None and (not psnrs or np.mean(psnrs) < min_agree):
            failed.append(tag)
    if failed:
        print(f'FAIL: trainer/viewer agreement below {min_agree} dB: {", ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
