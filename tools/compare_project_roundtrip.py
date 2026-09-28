"""Compare the project-autotest captures (Studio.ProjectAutotest.cs).

    python tools/compare_project_roundtrip.py TruckFull tuvok-b25-project

project_live     - the trained scene still in memory
project_reloaded - the same scene read back from OPFS (must match live)
project_misread_rgb, project_no_sh - red checks: the same saved scene loaded the ways a broken load would
(SH DC drawn as RGB; SH bands missing). They must differ from live by much more than reloaded does, or the
comparison could not have caught a broken round trip.

Mean absolute difference per channel (0..255) over the whole frame; the HUD is hidden during captures.
"""
import sys, os
from PIL import Image, ImageChops, ImageStat

name, tag = sys.argv[1], sys.argv[2]
d = os.path.join(os.path.dirname(__file__), '..', '_shots', 'dataset')
def load(kind):
    p = os.path.join(d, f'{name}__{tag}__free-project_{kind}.png')
    return Image.open(p).convert('RGB') if os.path.exists(p) else None

live = load('live')
if live is None:
    sys.exit('no live capture')
def mad(img):
    return sum(ImageStat.Stat(ImageChops.difference(live, img)).mean) / 3.0

rows = []
for kind in ('reloaded', 'misread_rgb', 'no_sh'):
    img = load(kind)
    rows.append((kind, None if img is None else mad(img)))
for kind, v in rows:
    print(f'{kind:12s} mean |diff| vs live: ' + ('missing' if v is None else f'{v:.3f} / 255'))

reloaded = dict(rows)['reloaded']
reds = [v for k, v in rows if k != 'reloaded' and v is not None]
ok = reloaded is not None and reloaded < 1.0 and all(r > 5 * max(reloaded, 0.2) for r in reds)
print('ROUND TRIP ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
