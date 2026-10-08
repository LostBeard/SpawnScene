"""Camera poses for the viewer benchmark, from a 3DGS .ply: the home view SpawnScene seats an import at (20-80% box of the
scene turned y-up: centre + (0, 0.35 r, -1.6 r)) plus N seeded pseudo-random views around it (azimuth any, elevation
5-35 deg, distance 1.2-2.2 r, target jittered 0.15 r). Written twice: 'ours' in SpawnScene's upright frame, 'ply' in the
file's own frame (y, z negated; up = -Y) for viewers that draw the PLY as stored.
usage: python make_poses.py <scene.ply> <out.json> [count=6] [seed=7]"""
import json, sys
import numpy as np

path, out = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 6
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
f = open(path, 'rb').read()
e = f.index(b'end_header\n') + len(b'end_header\n')
props = [l.split()[-1].decode() for l in f[:e].split(b'\n') if l.startswith(b'property')]
n = int([l for l in f[:e].split(b'\n') if l.startswith(b'element vertex')][0].split()[-1])
a = np.frombuffer(f[e:e + n * 4 * len(props)], dtype='<f4').reshape(n, len(props))
P = {p: a[:, i] for i, p in enumerate(props)}
ours = np.stack([P['x'], -P['y'], -P['z']], 1)
lo, hi = np.percentile(ours, 20, axis=0), np.percentile(ours, 80, axis=0)
c = (lo + hi) / 2
r = 0.5 * float(np.linalg.norm(hi - lo))
rng = np.random.default_rng(seed)
poses = [(c + np.array([0, 0.35 * r, -1.6 * r]), c)]
for _ in range(count):
    az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(5, 35))
    d = rng.uniform(1.2, 2.2) * r
    dirv = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    poses.append((c + dirv * d, c + rng.uniform(-0.15, 0.15, 3) * r))
flip = np.array([1, -1, -1])
res = {"centre": c.tolist(), "radius": r, "ours": [], "ply": []}
for pos, tgt in poses:
    res["ours"].append({"pos": pos.round(5).tolist(), "target": tgt.round(5).tolist(), "up": [0, 1, 0]})
    res["ply"].append({"pos": (pos * flip).round(5).tolist(), "target": (tgt * flip).round(5).tolist(), "up": [0, -1, 0]})
json.dump(res, open(out, 'w'), indent=1)
print(f"{len(poses)} poses, centre {c.round(3)}, radius {r:.3f} -> {out}")
