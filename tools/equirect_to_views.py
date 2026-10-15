"""Equirectangular (360) frames -> pinhole views, with the SAME convention as SpawnScene.Services.EquirectProjection (COLMAP's
EQUIRECTANGULAR model: panorama frame x right, y DOWN, z forward; lon = atan2(x, z), lat = asin(y/|d|); u = W (0.5 + lon/2pi),
v = H (0.5 + lat/pi), pixel centres at +0.5; view_from_pano rows = the view's right / down / forward axes).

    python tools/equirect_to_views.py <equirect_dir> <out_dir> [--views ring10|cube6] [--size 1024] [--fov 90] [--every N] [--limit N]

Writes <out_dir>/<frame>_<view>.jpg and <out_dir>/rig.json: {"fov": .., "size": .., "frames": {frame: {view: [9 floats view_from_pano]}}}.
Used to test the pipeline on 360 footage offline (Big Sur, CC BY 4.0) before the in-browser conversion exists; Research/360-video-2026-10-15.md.
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

RING10 = [(f"h{k * 45:03d}", k * 45.0, 0.0) for k in range(8)] + [("up", 0.0, 90.0), ("down", 0.0, -90.0)]
CUBE6 = [("h000", 0.0, 0.0), ("h090", 90.0, 0.0), ("h180", 180.0, 0.0), ("h270", 270.0, 0.0), ("up", 0.0, 90.0), ("down", 0.0, -90.0)]


def view_rotation(heading_deg, pitch_deg):
    """view_from_pano, row-major 3x3 (matches EquirectProjection.ViewRotation)."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.sin(h), -math.sin(p), math.cos(p) * math.cos(h)])
    right = np.array([math.cos(h), 0.0, -math.sin(h)])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def view_maps(R, size, fov_deg, w, h, pad):
    """cv2.remap maps (x, y in the PADDED panorama's pixel coordinates, centres at integers)."""
    f = 0.5 * size / math.tan(0.5 * math.radians(fov_deg))
    i = (np.arange(size) + 0.5 - 0.5 * size) / f
    rx, ry = np.meshgrid(i, i)
    ray = np.stack([rx, ry, np.ones_like(rx)], axis=-1)            # (size, size, 3), view frame
    d = ray @ R                                                    # pano_dir = R^T * ray  (row-vector form)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1] / np.linalg.norm(d, axis=-1), -1.0, 1.0))
    u = w * (0.5 + lon / (2 * math.pi))
    v = h * (0.5 + lat / math.pi)
    return (u - 0.5 + pad).astype(np.float32), (v - 0.5).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("equirect_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--views", default="ring10", choices=["ring10", "cube6"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    views = RING10 if a.views == "ring10" else CUBE6
    files = sorted(glob.glob(os.path.join(a.equirect_dir, "*.jpg")))[:: a.every]
    if a.limit:
        files = files[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    rig = {"fov": a.fov, "size": a.size, "views": a.views, "frames": {}}
    rots = {name: view_rotation(hd, pt) for name, hd, pt in views}
    maps = None
    for n, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        h, w = img.shape[:2]
        pad = 2
        padded = np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)   # wrap the longitude seam; the poles clamp
        if maps is None or maps[0] != (w, h):
            maps = ((w, h), {name: view_maps(R, a.size, a.fov, w, h, pad) for name, R in rots.items()})
        stem = os.path.splitext(os.path.basename(path))[0]
        rig["frames"][stem] = {}
        for name, R in rots.items():
            mx, my = maps[1][name]
            view = cv2.remap(padded, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(a.out_dir, f"{stem}_{name}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rig["frames"][stem][name] = [float(x) for x in R.reshape(-1)]
        print(f"{n + 1}/{len(files)} {stem}", flush=True)
    json.dump(rig, open(os.path.join(a.out_dir, "rig.json"), "w"))


if __name__ == "__main__":
    main()
