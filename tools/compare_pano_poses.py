"""Compare SpawnScene's recovered poses for pinhole views of 360 frames with a COLMAP spherical reconstruction (Big Sur, CC BY 4.0).

    python tools/compare_pano_poses.py <our_colmap_dir> <rig.json> <reference images.bin> [--report out.json]

<our_colmap_dir> holds images.txt from the harness's &exportcolmap=1 (world -> camera; names <frame>_<view>.jpg); <rig.json> is written by
tools/equirect_to_views.py (view_from_pano per frame and view); the reference is COLMAP's binary images.bin (its poses are cam_from_world of the
EQUIRECTANGULAR camera, the same frame convention as EquirectProjection).

Per placed view: pano_from_world = view_from_pano^T * view_from_world. Per frame the views are averaged (chordal mean) and their disagreement
is reported (the rig check: views of one frame share a pose). Then, for every pair of frames, the relative rotation R_i R_j^T (independent of
the arbitrary world frame) is compared with the reference's, and the camera centres are compared after a similarity fit (Umeyama).
"""
import argparse
import json
import math
import os
import struct

import numpy as np


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def chordal_mean(Rs):
    M = sum(Rs)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def read_images_txt(path):
    out = {}
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    for l in lines:  # one line per image (points2D lines are empty and filtered out)
        p = l.split()
        if len(p) < 10:
            continue
        q = list(map(float, p[1:5])); t = np.array(list(map(float, p[5:8])))
        R = quat_to_R(*q)
        out[p[9]] = (R, t, -R.T @ t)
    return out


def read_reference(path, frames):
    buf = open(path, "rb").read()
    n = struct.unpack_from("<Q", buf, 0)[0]; o = 8; ref = {}
    for _ in range(n):
        o += 4
        q = struct.unpack_from("<4d", buf, o); o += 32
        t = struct.unpack_from("<3d", buf, o); o += 24
        o += 4
        e = buf.index(b"\x00", o); name = buf[o:e].decode(); o = e + 1
        npts = struct.unpack_from("<Q", buf, o)[0]; o += 8 + 24 * npts
        stem = os.path.splitext(name)[0]
        if stem in frames:
            R = quat_to_R(*q); t = np.array(t)
            ref[stem] = (R, -R.T @ t)
    return ref


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ E @ Vt
    scale = (sig * np.diag(E)).sum() / (S ** 2).sum() * len(src)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours"); ap.add_argument("rig"); ap.add_argument("reference")
    ap.add_argument("--report")
    a = ap.parse_args()
    rig = json.load(open(a.rig))
    ours = read_images_txt(os.path.join(a.ours, "images.txt"))
    frames = list(rig["frames"].keys())
    ref = read_reference(a.reference, set(frames))
    per_frame, spread, placed_views = {}, {}, {}
    for f in frames:
        Rs, Cs = [], []
        for view, vfp in rig["frames"][f].items():
            key = f"{f}_{view}.jpg"
            if key not in ours:
                continue
            Rv, tv, Cv = ours[key]
            Rpw = np.array(vfp).reshape(3, 3).T @ Rv           # pano_from_world
            Rs.append(Rpw); Cs.append(Cv)
        placed_views[f] = len(Rs)
        if len(Rs) < 2:
            continue
        Rm = chordal_mean(Rs)
        spread[f] = max(angle_deg(R @ Rm.T) for R in Rs)
        per_frame[f] = (Rm, np.mean(Cs, axis=0))
    print(f"frames with >= 2 placed views: {len(per_frame)} of {len(frames)}; views placed per frame: {sorted(placed_views.values())}")
    print("rig consistency (max angle of one view's pano pose from its frame's mean, deg): "
          + " ".join(f"{spread[f]:.2f}" for f in per_frame))
    common = [f for f in per_frame if f in ref]
    errs = []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            fi, fj = common[i], common[j]
            ours_rel = per_frame[fi][0] @ per_frame[fj][0].T
            ref_rel = ref[fi][0] @ ref[fj][0].T
            errs.append(angle_deg(ours_rel @ ref_rel.T))
    errs = np.array(errs)
    print(f"relative rotation vs COLMAP over {len(errs)} frame pairs: median {np.median(errs):.2f} deg, mean {errs.mean():.2f}, max {errs.max():.2f}")
    # also try the mirrored hypothesis: a convention slip would show up as a large systematic error here, not a small one
    src = np.array([per_frame[f][1] for f in common]); dst = np.array([ref[f][1] for f in common])
    scale, R, t = umeyama(src, dst)
    fit = (scale * (R @ src.T).T + t) - dst
    path = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    print(f"camera centres after a similarity fit: RMS {np.sqrt((fit ** 2).sum(1).mean()):.3f} (reference path length {path:.2f}, extent {np.ptp(dst, axis=0).round(2)}), scale {scale:.4g}")
    if a.report:
        json.dump({"frames": len(per_frame), "pairs": len(errs), "median_deg": float(np.median(errs)), "max_deg": float(errs.max()),
                   "spread_deg": spread, "views_per_frame": placed_views}, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
