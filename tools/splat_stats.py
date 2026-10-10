"""Statistics of a 3DGS PLY, the same numbers SpawnScene's &splatstats=1 logs (Studio.SplatStats): opacity, size (largest
axis) and distance to the nearest training camera in units of the cameras' spread, and the dark opaque splats.

    python tools/splat_stats.py <ply> <cameras.npy>
    python tools/splat_stats.py <ply> --colmap <sparse dir with images.bin>   (a PLY in the COLMAP frame, e.g. Brush's)

<cameras.npy>: the training camera centres (N x 3) IN THE PLY'S FRAME. For gsplat (normalize=True) write them with its
own Parser, e.g. from gsplat-src/examples:
    from datasets.colmap import Parser; import numpy as np
    p = Parser(data_dir=..., factor=2, normalize=True); np.save("cams.npy", p.camtoworlds[:, :3, 3])
"""
import struct
import sys

import numpy as np


def colmap_centres(sparse):
    """Camera centres C = -R^T t from a COLMAP images.bin."""
    out = []
    with open(sparse + "/images.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            fh.read(4)
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", fh.read(56))
            fh.read(4)
            while fh.read(1) != b"\0":
                pass
            (k,) = struct.unpack("<Q", fh.read(8))
            fh.read(24 * k)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
            out.append(-R.T @ np.array([tx, ty, tz]))
    return np.array(out)


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
    cams = colmap_centres(sys.argv[3]) if sys.argv[2] == "--colmap" else np.load(sys.argv[2])
    spread = np.sqrt(((cams - cams.mean(0)) ** 2).sum(1).mean())
    pos = np.stack([v["x"], v["y"], v["z"]], 1)
    opac = 1 / (1 + np.exp(-v["opacity"]))
    axes = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    size = axes.max(1) / spread
    aniso = axes.max(1) / np.maximum(axes.min(1), 1e-12)
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
    print(f"[Stats] anisotropy (largest / smallest axis) {pc(aniso, (0.5, 0.9, 0.99))}; > 10: {(aniso > 10).mean():.1%}")
    print(f"[Stats] distance to nearest camera {pc(dist, (0.01, 0.05, 0.5))}; < 0.25: {(dist < 0.25).mean():.2%}, < 0.5: {(dist < 0.5).mean():.2%}")
    dark = (lum < 0.2) & (opac > 0.5)
    print(f"[Stats] dark opaque (luma < 0.2, opacity > 0.5): {dark.mean():.2%}, of them within 0.5 spreads of a camera: {(dark & (dist < 0.5)).sum():,}")
    # Radial: distance from the camera centroid in spreads (parity 2026-10-10: Train's held-out sky blobs are bright, far,
    # thinly-seen splats). Per shell: share of splats, median opacity, bright (luma > 0.9) and bright opaque (also > 0.3).
    r = np.linalg.norm(pos - cams.mean(0), axis=1) / spread
    edges = [0, 1, 2, 4, 8, 16, np.inf]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        k = int(m.sum())
        if k == 0:
            parts.append(f"{lo:g}-{hi:g}: 0"); continue
        parts.append(f"{lo:g}-{hi:g}: {k / n:.2%} op50 {np.median(opac[m]):.2f} bright {(lum[m] > 0.9).mean():.1%} "
                     f"bright-opaque {((lum[m] > 0.9) & (opac[m] > 0.3)).sum():,}")
    print("[Stats] radial (spreads from the camera centroid) " + " | ".join(parts))


if __name__ == "__main__":
    main()
