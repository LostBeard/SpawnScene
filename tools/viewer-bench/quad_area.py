"""Fragments the sorted viewer rasterises per pose: its axis-aligned quad (vs_trainer: alpha box ∩ 3-sigma tile box)
against a quad aligned with the ellipse's own axes (the alpha contour's oriented bounding rectangle).

    python tools/viewer-bench/quad_area.py <scene.ply> <poses.json> [W=1600] [H=900] [fov=50]

Both quads cover every pixel the fragment test keeps, so the ratio is overdraw only. Screen clipping is approximated by
clipping each quad's bounding box to the viewport.
"""
import json
import sys

import numpy as np


def read_ply(path):
    # Binary little-endian 3DGS PLY, every property a float (the reference trainer's layout).
    with open(path, "rb") as fh:
        names = []
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"): n = int(line.split()[2])
            elif line.startswith("property"):
                assert line.split()[1] == "float", line
                names.append(line.split()[2])
            elif line == "end_header": break
        a = np.fromfile(fh, dtype="<f4", count=n * len(names)).reshape(n, len(names))
    return {nm: a[:, i] for i, nm in enumerate(names)}


def main():
    ply, poses_path = sys.argv[1], sys.argv[2]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 900
    fov = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
    f = (H / 2) / np.tan(np.radians(fov) / 2)
    v = read_ply(ply)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)
    M = R * sc[:, None, :]
    sig = M @ M.transpose(0, 2, 1)
    poses = json.load(open(poses_path))["ply"]
    for k, p in enumerate(poses):
        cpos, tgt, up = map(np.array, (p["pos"], p["target"], p["up"]))
        fwd = tgt - cpos; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        upv = np.cross(right, fwd)
        A = np.stack([right, upv, fwd])
        rel = pos - cpos
        cam = rel @ A.T
        keep = (cam[:, 2] > 0.2) & (op > 1 / 255)
        c = cam[keep]; s = sig[keep]; o = op[keep]
        scam = A @ s @ A.T
        z_ = c[:, 2]
        lim = 1.3 * 0.5 * np.array([W, H]) / f
        cx = np.clip(c[:, 0] / z_, -lim[0], lim[0]) * z_
        cy = np.clip(c[:, 1] / z_, -lim[1], lim[1]) * z_
        j00 = f / z_; j02 = -f * cx / z_ ** 2; j11 = f / z_; j12 = -f * cy / z_ ** 2
        s00, s01, s02 = scam[:, 0, 0], scam[:, 0, 1], scam[:, 0, 2]
        s11, s12, s22 = scam[:, 1, 1], scam[:, 1, 2], scam[:, 2, 2]
        a0 = j00 * s00 + j02 * s02; a1 = j00 * s01 + j02 * s12; a2 = j00 * s02 + j02 * s22
        b1 = j11 * s11 + j12 * s12; b2 = j11 * s12 + j12 * s22
        ca = a0 * j00 + a2 * j02 + 0.3; cb = a1 * j11 + a2 * j12; cc = b1 * j11 + b2 * j12 + 0.3
        det = ca * cc - cb * cb
        mid = 0.5 * (ca + cc)
        l1 = mid + np.sqrt(np.maximum(mid * mid - det, 0)); l2 = np.maximum(mid - np.sqrt(np.maximum(mid * mid - det, 0)), 1e-12)
        k2 = 2 * np.log(255 * o)
        ok = (det > 1e-20) & (k2 > 0)
        px = W / 2 + f * c[:, 0] / z_; py = H / 2 - f * c[:, 1] / z_
        r = 3 * np.sqrt(l1)
        hx = np.sqrt(np.maximum(k2 * ca, 0)) + 1; hy = np.sqrt(np.maximum(k2 * cc, 0)) + 1
        # axis-aligned: alpha box ∩ 3-sigma tile box (tile snapping ignored: it only grows the current quad)
        x0 = np.maximum(px - np.minimum(hx, r), 0); x1 = np.minimum(px + np.minimum(hx, r), W)
        y0 = np.maximum(py - np.minimum(hy, r), 0); y1 = np.minimum(py + np.minimum(hy, r), H)
        aabb = np.where(ok & (x1 > x0) & (y1 > y0), (x1 - x0) * (y1 - y0), 0)
        # oriented: alpha contour's rectangle along the eigenvectors, each half-axis capped at the 3-sigma radius
        ha = np.minimum(np.sqrt(np.maximum(k2 * l1, 0)), r) + 1; hb = np.minimum(np.sqrt(np.maximum(k2 * l2, 0)), r) + 1
        orient_full = 4 * ha * hb
        # screen clip, approximately: scale by the fraction of the AABB that is on screen
        full_aabb = (2 * np.minimum(hx, r)) * (2 * np.minimum(hy, r))
        frac = np.where(full_aabb > 0, aabb / np.maximum(full_aabb, 1e-9), 0)
        orient = np.where(ok, np.minimum(orient_full, full_aabb) * frac, 0)
        # GaussianSplats3D's quad: ellipse-aligned, half-axes sqrt(8 lambda) (2.83 sigma, every opacity), no 1/255 rule
        g = np.where(ok, np.minimum(4 * np.sqrt(8 * l1) * np.sqrt(8 * l2), full_aabb * 4) * frac, 0)
        print(f"pose {k}: {keep.sum():,} in front, axis-aligned {aabb.sum() / 1e6:6.1f} M px, "
              f"ellipse-aligned {orient.sum() / 1e6:6.1f} M px (x{aabb.sum() / max(orient.sum(), 1):.2f}), "
              f"GS3D-style {g.sum() / 1e6:6.1f} M px (x{aabb.sum() / max(g.sum(), 1):.2f})")


if __name__ == "__main__":
    main()
