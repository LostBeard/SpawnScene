"""numpy port of DepthEdgeSnap v2 (Services/DepthEdgeSnap.cs) + the flying-pixel metric: does our snap remove what no
depth model avoids? usage: snap_eval.py <photos dir> <name>... (reads <name>_moge.jpg.npz from depth_compare.py)"""
import sys, numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

def snap(z, rgb, radius=3, min_rel=0.03, sigma=0.08, fisher=0.0, maxmid=1.0):
    H, W = z.shape
    step = max(1, round(max(W, H) / 518))
    pad = radius * step
    zp = np.pad(z, pad, mode="edge"); cp = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    offs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    sh = lambda a, dy, dx: a[pad + dy * step: pad + dy * step + H, pad + dx * step: pad + dx * step + W]
    lo, hi = z.copy(), z.copy()
    for dy, dx in offs:
        s = sh(zp, dy, dx); lo = np.minimum(lo, s); hi = np.maximum(hi, s)
    mid = 0.5 * (lo + hi)
    q1, q3 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    inmid = np.zeros((H, W))
    for dy, dx in offs:
        sv = sh(zp, dy, dx); inmid += (sv > q1) & (sv < q3)
    inmid /= len(offs)
    fall = 1 / (2 * (radius * 0.75) ** 2)
    n = np.zeros((H, W, 3)); nw = np.zeros((H, W)); f = np.zeros((H, W, 3)); fw = np.zeros((H, W))
    n2 = np.zeros((H, W)); f2 = np.zeros((H, W))
    for dy, dx in offs:
        w = np.exp(-(dy * dy + dx * dx) * fall)
        near = sh(zp, dy, dx) < mid
        c = sh(cp, dy, dx)
        n += (w * near)[..., None] * c; nw += w * near; n2 += w * near * (c * c).sum(-1)
        f += (w * ~near)[..., None] * c; fw += w * ~near; f2 += w * ~near * (c * c).sum(-1)
    act = (z > 0) & (hi - lo > min_rel * z) & (nw > 0) & (fw > 0) & (inmid <= maxmid)
    n /= np.maximum(nw, 1e-9)[..., None]; f /= np.maximum(fw, 1e-9)[..., None]
    sep = ((n - f) ** 2).sum(-1)
    act &= sep >= sigma * sigma
    if fisher > 0:
        # within-side colour spread (sum of channel variances) of each side
        vn = n2 / np.maximum(nw, 1e-9) - (n * n).sum(-1); vf = f2 / np.maximum(fw, 1e-9) - (f * f).sum(-1)
        act &= sep >= fisher * (np.maximum(vn, 0) + np.maximum(vf, 0))
    to_near = ((rgb - n) ** 2).sum(-1); to_far = ((rgb - f) ** 2).sum(-1)
    return np.where(act, np.where(to_near <= to_far, lo, hi), z)

def flying(z, m):
    hi, lo = maximum_filter(z, 7), minimum_filter(z, 7)
    step = ((hi - lo) / lo > 0.25) & m
    t = (z - lo) / np.maximum(hi - lo, 1e-9)
    return (step & (t > 0.2) & (t < 0.8)).mean() * 100

pdir = sys.argv[1]
for name in sys.argv[2:]:
    d = np.load(f"{name}_moge.jpg.npz"); zd = d["zd"]; m = d["mask"]
    H, W = zd.shape
    rgb = np.asarray(Image.open(f"{pdir}/{name}.jpg").convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
    row = [f"raw {flying(zd, m):.2f}", f"live {flying(snap(zd, rgb), m):.2f}"]
    for mm in (0.4, 0.3, 0.2, 0.15):
        row.append(f"mid<={mm} {flying(snap(zd, rgb, maxmid=mm), m):.2f}")
    print(name.ljust(12), "  ".join(row))
