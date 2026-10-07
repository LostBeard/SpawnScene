"""Wander views (Studio.Wander: poses no photo was taken from) side by side, one row per view.

    python tools/compose_wander.py <Dataset> <TAG>[,<TAG2>...] [<gsplat render dir>] [--only in,up]

Columns: each tag's capture of view-wander<kind>-<i> (_shots/dataset/<Dataset>__<TAG>__view-wander...png), then the
reference trainer's render of the same pose when a gsplat dir is given (gsplat_wander<kind>-<i>.png, written by
render_turns.py from the run's TURN-POSE lines). Writes _shots/dataset/<Dataset>__<TAG...>__wander.png.
"""
import os
import re
import sys

from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SHOTS = os.path.join(ROOT, '_shots', 'dataset')
CELL_W = 512

args = [a for a in sys.argv[1:] if not a.startswith('--')]
only = None
if '--only' in sys.argv:
    only = sys.argv[sys.argv.index('--only') + 1].split(',')
    args = [a for a in args if a != sys.argv[sys.argv.index('--only') + 1]]
dataset, tags = args[0], args[1].split(',')
gdir = args[2] if len(args) > 2 else None

pat = re.compile(rf'^{re.escape(dataset)}__{re.escape(tags[0])}__view-(wander(\w+)-(\d+))\.png$')
views = sorted((m.group(1), m.group(2), int(m.group(3))) for f in os.listdir(SHOTS) if (m := pat.match(f)))
if only:
    views = [v for v in views if v[1] in only]
if not views:
    sys.exit(f'no wander captures for {dataset} {tags[0]}')


def cell(path, label):
    if path and os.path.exists(path):
        im = Image.open(path).convert('RGB')
        im = im.resize((CELL_W, round(im.height * CELL_W / im.width)))
    else:
        im = Image.new('RGB', (CELL_W, CELL_W * 2 // 3), (60, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(label) + 8, 16], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 0))
    return im


rows = []
for name, kind, i in views:
    cells = [cell(os.path.join(SHOTS, f'{dataset}__{t}__view-{name}.png'), f'{t} {name}') for t in tags]
    if gdir:
        cells.append(cell(os.path.join(gdir, f'gsplat_{name}.png'), f'gsplat {name}'))
    h = max(c.height for c in cells)
    row = Image.new('RGB', (CELL_W * len(cells), h))
    for k, c in enumerate(cells):
        row.paste(c, (k * CELL_W, 0))
    rows.append(row)

out = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)))
y = 0
for r in rows:
    out.paste(r, (0, y))
    y += r.height
suffix = '_'.join(tags) + ('_' + '-'.join(only) if only else '')
path = os.path.join(SHOTS, f'{dataset}__{suffix}__wander.png')
out.save(path)
print(path, out.size)
