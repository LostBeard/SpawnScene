"""Trainer render vs viewer capture vs photo, per view.

    python tools/compare_trainer_viewer.py <Dataset> <RUN_TAG>

Needs a run whose harness saved view-<k>-trainer.png (the trainer's rasteriser on the scene the viewer shows,
Studio.StashTrainerRenderAsync) next to view-<k>.png (the viewer's capture of the same pose). Prints PSNR of
each against the photo and against each other, and writes <tag>__trainer-vs-viewer.png: trainer | viewer |
|difference| x4, one row per view. The two renderers should agree; where they do not is the viewer's loss.
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SHOTS = os.path.join(ROOT, '_shots', 'dataset')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _photos import load_photo  # noqa: E402


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / max(np.mean((a - b) ** 2), 1e-12))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name, tag = sys.argv[1], sys.argv[2]
    side = json.load(open(os.path.join(SHOTS, f'{name}__{tag}__views.json')))
    rows = []
    for k, v in sorted(side['views'].items(), key=lambda kv: (kv[0].split('-')[0] != 'held', kv[1]['photo'])):
        tr_path = os.path.join(SHOTS, f'{name}__{tag}__view-{k}-trainer.png')
        if not os.path.exists(tr_path):
            print(f'   {k}: no trainer render')
            continue
        view = Image.open(os.path.join(SHOTS, f'{name}__{tag}__view-{k}.png')).convert('RGB')
        tr = Image.open(tr_path).convert('RGB')
        photo = load_photo(side, name, tag, k)
        if tr.size != view.size:
            tr = tr.resize(view.size, Image.LANCZOS)
        if photo.size != view.size:
            photo = photo.resize(view.size, Image.LANCZOS)
        a, b, p = (np.asarray(x, np.float64) for x in (tr, view, photo))
        diff = np.abs(a - b)
        print(f'   {k:9s} trainer/photo {psnr(a, p):5.2f}  viewer/photo {psnr(b, p):5.2f}  '
              f'trainer/viewer {psnr(a, b):5.2f}  mean |d| {diff.mean():5.2f}  mean signed (viewer-trainer) '
              f'{", ".join(f"{m:+.2f}" for m in (b - a).reshape(-1, 3).mean(0))}')
        rows.append((k, tr, view, Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8))))
    if not rows:
        return
    w, h = rows[0][1].size
    sheet = Image.new('RGB', (w * 3, h * len(rows)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for r, (k, tr, view, df) in enumerate(rows):
        for c, (im, label) in enumerate(((tr, 'TRAINER'), (view, 'VIEWER'), (df, '|DIFF| x4'))):
            sheet.paste(im, (c * w, r * h))
            d.rectangle((c * w, r * h, c * w + 170, r * h + 18), fill=(0, 0, 0))
            d.text((c * w + 4, r * h + 3), f'{label} {k}', fill=(255, 255, 255))
    out = os.path.join(SHOTS, f'{name}__{tag}__trainer-vs-viewer.png')
    sheet.save(out)
    print(out)


if __name__ == '__main__':
    main()
