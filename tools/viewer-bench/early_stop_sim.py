"""How many blended fragments would front-to-back drawing with early termination skip?

    python tools/viewer-bench/early_stop_sim.py <scene.ply> <poses.json> [scale=0.25] [batches=4,8,16,32]

Simulates the sorted viewer at a reduced resolution (1600x900 x scale, focal scaled with it): splats in global
centre-depth order, front to back, each covering the pixels where alpha >= 1/255 (the viewer's and trainer's rule,
3-sigma radius cap). The trainer and the reference rasteriser stop a pixel once its transmittance T < 1e-4.

Reports, per pose: fragments the viewer blends today (all of them); fragments left if every pixel stopped exactly at
T < 1e-4 (ideal); and fragments left when drawing in K equal batches with opaque pixels marked only BETWEEN batches
(what a hardware rasteriser with an early depth test can do). Splats are processed in chunks of 1024, with T taken at
the chunk's start: a slight over-count of the fragments that are still needed.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from quad_area import read_ply  # noqa: E402


def main():
    ply, poses_path = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    batch_list = [int(b) for b in (sys.argv[4] if len(sys.argv) > 4 else "4,8,16,32").split(",")]
    W, H = int(1600 * scale), int(900 * scale)
    f = (900 / 2) / np.tan(np.radians(50) / 2) * scale
    v = read_ply(ply)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
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
        cam = (pos - cpos) @ A.T
        keep = (cam[:, 2] > 0.2) & (op > 1 / 255)
        c, s, o = cam[keep], A @ sig[keep] @ A.T, op[keep]
        zc = c[:, 2]
        lim = 1.3 * 0.5 * np.array([W, H]) / f
        cxc = np.clip(c[:, 0] / zc, -lim[0], lim[0]) * zc
        cyc = np.clip(c[:, 1] / zc, -lim[1], lim[1]) * zc
        j00 = f / zc; j02 = -f * cxc / zc ** 2; j11 = f / zc; j12 = -f * cyc / zc ** 2
        a0 = j00 * s[:, 0, 0] + j02 * s[:, 0, 2]; a1 = j00 * s[:, 0, 1] + j02 * s[:, 1, 2]
        a2 = j00 * s[:, 0, 2] + j02 * s[:, 2, 2]
        b1 = j11 * s[:, 1, 1] + j12 * s[:, 1, 2]; b2 = j11 * s[:, 1, 2] + j12 * s[:, 2, 2]
        ca = a0 * j00 + a2 * j02 + 0.3; cb = a1 * j11 + a2 * j12; cc = b1 * j11 + b2 * j12 + 0.3
        det = ca * cc - cb * cb
        k2 = 2 * np.log(np.maximum(255 * o, 1e-30))
        mid = 0.5 * (ca + cc); l1 = mid + np.sqrt(np.maximum(mid * mid - det, 0))
        px = W / 2 + f * c[:, 0] / zc; py = H / 2 - f * c[:, 1] / zc
        r = 3 * np.sqrt(l1)
        hx = np.minimum(np.sqrt(np.maximum(k2 * ca, 0)), r); hy = np.minimum(np.sqrt(np.maximum(k2 * cc, 0)), r)
        x0 = np.clip(np.floor(px - hx), 0, W).astype(np.int64); x1 = np.clip(np.ceil(px + hx), 0, W).astype(np.int64)
        y0 = np.clip(np.floor(py - hy), 0, H).astype(np.int64); y1 = np.clip(np.ceil(py + hy), 0, H).astype(np.int64)
        ok = (det > 1e-20) & (k2 > 0) & (x1 > x0) & (y1 > y0)
        order = np.argsort(zc[ok], kind="stable")  # front to back by centre depth, as the viewer's sort
        idx = np.nonzero(ok)[0][order]
        conA, conB, conC = cc[idx] / det[idx], cb[idx] / det[idx], ca[idx] / det[idx]
        n = len(idx)
        T = np.ones(W * H)
        bounds = {K: [int(round(n * i / K)) for i in range(1, K)] for K in batch_list}
        nxt = {K: 0 for K in batch_list}
        Tsnap = {K: np.ones(W * H) for K in batch_list}
        total = needed = 0
        batched = {K: 0 for K in batch_list}
        CH = 1024
        for s0 in range(0, n, CH):
            for K in batch_list:
                # Opaque pixels are marked between batches: refresh at the first chunk at or past each boundary.
                if nxt[K] < len(bounds[K]) and s0 >= bounds[K][nxt[K]]:
                    Tsnap[K] = T.copy()
                    while nxt[K] < len(bounds[K]) and s0 >= bounds[K][nxt[K]]:
                        nxt[K] += 1
            sl = slice(s0, min(s0 + CH, n)); ii = idx[sl]
            ww = (x1[ii] - x0[ii]); hh = (y1[ii] - y0[ii]); cnt = ww * hh
            rep = np.repeat(np.arange(len(ii)), cnt)
            start = np.repeat(np.cumsum(cnt) - cnt, cnt)
            local = np.arange(cnt.sum()) - start
            gx = x0[ii][rep] + local % ww[rep]; gy = y0[ii][rep] + local // ww[rep]
            dx = gx + 0.5 - px[ii][rep]; dy = gy + 0.5 - py[ii][rep]
            power = -0.5 * (conA[sl][rep] * dx * dx + conC[sl][rep] * dy * dy) + conB[sl][rep] * dx * dy
            alpha = np.minimum(0.99, o[ii][rep] * np.exp(np.minimum(power, 0)))
            live = (power <= 0) & (alpha >= 1 / 255)
            pix = (gy * W + gx)[live]; al = alpha[live]
            total += len(pix)
            needed += int((T[pix] >= 1e-4).sum())
            for K in batch_list:
                batched[K] += int((Tsnap[K][pix] >= 1e-4).sum())
            np.multiply.at(T, pix, 1 - al)
        line = f"pose {k}: {total / 1e6:6.2f} M blended today; ideal stop {needed / total:5.1%}"
        line += "".join(f", K={K} {batched[K] / total:5.1%}" for K in batch_list)
        print(line, flush=True)


if __name__ == "__main__":
    main()
