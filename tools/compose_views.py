"""Render | real photo, side by side, for every view-* capture of a dataset run.

    python tools/compose_views.py <Dataset> <RUN_TAG> [<RUN_TAG> ...]

Reads _shots/dataset/<Dataset>__<TAG>__views.json (written by _cdp_dataset.js) and the matching
<Dataset>__<TAG>__view-<kind>-<i>.png captures, fetches each photo from the app server named in the
sidecar, and writes _shots/dataset/<Dataset>__<TAG>__compare.png. With several tags the renders go in
columns, one per tag, then the photo - the same view across runs, for a by-eye A/B.

Numbers do not decide quality on this project; TJ's eyes do, and these are for him.
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SHOTS = os.path.join(ROOT, '_shots', 'dataset')
CELL_W = 640


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB')


def fit(img, w):
    return img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def label(img, text):
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), text, font=font)
    d.rectangle((0, 0, box[2] + 12, box[3] + 10), fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return img


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name, tags = sys.argv[1], sys.argv[2:]
    sides = []
    for tag in tags:
        p = os.path.join(SHOTS, f'{name}__{tag}__views.json')
        if not os.path.exists(p):
            sys.exit(f'missing {p} - was the run made with the view-* capture harness?')
        sides.append(json.load(open(p)))
    # Pair runs by PHOTO, not by view number: a run that drops views renumbers the rest, and "view 62" then
    # names a different photograph in each run.
    by_photo = []
    for sd in sides:
        by_photo.append({v['photo']: k for k, v in sd['views'].items()})
    photos = [ph for ph in by_photo[0] if all(ph in bp for bp in by_photo)]
    photos.sort(key=lambda ph: (by_photo[0][ph].split('-')[0] != 'held', ph))
    if not photos:
        sys.exit('no photo was captured by every run - pass the same &capture= list to each')

    rows = []
    for ph in photos:
        cells = []
        photo = fetch(sides[0]['app'].rstrip('/') + '/' + ph.lstrip('/'))
        for ti, tag in enumerate(tags):
            k = by_photo[ti][ph]
            shot = os.path.join(SHOTS, f'{name}__{tag}__view-{k}.png')
            img = Image.open(shot).convert('RGB') if os.path.exists(shot) else Image.new('RGB', (CELL_W, 360))
            kind = 'HELD-OUT (never trained on)' if k.startswith('held') else 'supervised'
            # The VIEWER's render scored against the photo (the trainer logs its own per view).
            ref = np.asarray(photo.resize(img.size, Image.LANCZOS), dtype=np.float64)
            mse = np.mean((np.asarray(img, dtype=np.float64) - ref) ** 2)
            psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
            cells.append(label(fit(img, CELL_W), f'{tag}  {kind}  view {k.split("-")[1]}  viewer {psnr:.1f} dB'))
        cells.append(label(fit(photo, CELL_W), f'PHOTO  {os.path.basename(ph)}'))
        h = max(c.height for c in cells)
        row = Image.new('RGB', (CELL_W * len(cells) + 8 * (len(cells) - 1), h), (40, 40, 40))
        for i, c in enumerate(cells):
            row.paste(c, (i * (CELL_W + 8), 0))
        rows.append(row)

    out = Image.new('RGB', (rows[0].width, sum(r.height for r in rows) + 8 * (len(rows) - 1)), (40, 40, 40))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 8
    dst = os.path.join(SHOTS, f'{name}__{"_vs_".join(tags)}__compare.png')
    out.save(dst)
    print(dst)


if __name__ == '__main__':
    main()
