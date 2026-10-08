"""Statistics of a 3DGS PLY, the same numbers SpawnScene's &splatstats=1 logs (Studio.SplatStats): opacity, size (largest
axis) and distance to the nearest training camera in units of the cameras' spread, and the dark opaque splats.

    python tools/splat_stats.py <ply> <cameras.npy>

<cameras.npy>: the training camera centres (N x 3) IN THE PLY'S FRAME. For gsplat (normalize=True) write them with its
own Parser, e.g. from gsplat-src/examples:
    from datasets.colmap import Parser; import numpy as np
    p = Parser(data_dir=..., factor=2, normalize=True); np.save("cams.npy", p.camtoworlds[:, :3, 3])
"""
import sys

import numpy as np


def read_ply(path):
    with open(path, "rb") as fh:
        names, n = [], 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"): n = int(line.split()[2])
            elif line.startswith("property"): names.append(line.split()[2])
            elif line == "end_header": break
        a = np.fromfile(fh, dtype="<f4", count=n * len(names)).reshape(n, len(names))
    return {nm: a[:, i] for i, nm in enumerate(names)}


def main():
    v = read_ply(sys.argv[1])
    cams = np.load(sys.argv[2])
    spread = np.sqrt(((cams - cams.mean(0)) ** 2).sum(1).mean())
    pos = np.stack([v["x"], v["y"], v["z"]], 1)
    opac = 1 / (1 + np.exp(-v["opacity"]))
    size = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).max(1) / spread
    rgb = 0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    dist = np.full(len(pos), np.inf)
    for c in cams:
        dist = np.minimum(dist, np.linalg.norm(pos - c, axis=1))
    dist /= spread
    n = len(pos)
    pc = lambda a, qs: " ".join(f"p{q * 100:.0f}={np.quantile(a, q):.3g}" for q in qs)
    print(f"[Stats] {n:,} splats, camera spread {spread:.4g} (sizes and distances below are in spreads)")
    print(f"[Stats] opacity {pc(opac, (0.1, 0.5, 0.9))}; > 0.5: {(opac > 0.5).mean():.1%}")
    print(f"[Stats] size (largest axis) {pc(size, (0.5, 0.9, 0.99, 0.999))}")
    print(f"[Stats] distance to nearest camera {pc(dist, (0.01, 0.05, 0.5))}; < 0.25: {(dist < 0.25).mean():.2%}, < 0.5: {(dist < 0.5).mean():.2%}")
    dark = (lum < 0.2) & (opac > 0.5)
    print(f"[Stats] dark opaque (luma < 0.2, opacity > 0.5): {dark.mean():.2%}, of them within 0.5 spreads of a camera: {(dark & (dist < 0.5)).sum():,}")


if __name__ == "__main__":
    main()
