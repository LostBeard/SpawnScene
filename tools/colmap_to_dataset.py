"""Convert a COLMAP sparse reconstruction into a SpawnScene dataset.

    python tools/colmap_to_dataset.py <colmap_scene_dir> <DatasetName> [--every N] [--max N]

Writes SpawnScene/wwwroot/datasets/<DatasetName>/:
    manifest.json   image list, image directory, and where the poses are
    poses.par       Middlebury-format camera parameters

The par format is not a coincidence. WorldSpaceGeometry.ParseMiddleburyParams already reads
it for TempleRing, and COLMAP stores exactly the same quantities in the same convention
(world-to-camera R and t, OpenCV axes with row0 right, row1 down, row2 forward), so reusing
it means no new pose-parsing code in C# and one parser to be wrong in rather than two.

The images are NOT copied. They are 168 MB for drjohnson alone, and anything under wwwroot
is recopied on every publish - this repo has already had a publish balloon to 13 GB from
output landing inside the project. The manifest records where they live and the static
server mounts that directory; see the command this prints when it finishes.

WHY THIS DATASET MATTERS: every Bathroom measurement conflates two error sources, because
Bathroom has no ground-truth poses - "held out 12.41 dB" mixes our poses being wrong with
our optimiser being wrong, and there is no way to separate them. Deep Blending's drjohnson
and playroom are real indoor rooms WITH COLMAP poses, so the optimiser can be measured
against known-good poses, and our pose pipeline can be measured against a room rather than
against TempleRing, which is an object on a turntable.
"""
import argparse
import json
import os
import struct
import sys

# COLMAP camera models: id -> (name, param count). Only the undistorted ones are usable
# directly; anything with distortion would need undistorting first and is refused rather
# than silently treated as a pinhole.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
UNDISTORTED = {"SIMPLE_PINHOLE", "PINHOLE"}


def _read(f, n, fmt):
    return struct.unpack("<" + fmt, f.read(n))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, 8, "Q")[0]):
            cid, model, w, h = _read(f, 24, "iiQQ")
            name, nparams = CAMERA_MODELS[model]
            params = _read(f, 8 * nparams, "d" * nparams)
            cams[cid] = dict(model=name, width=w, height=h, params=params)
    return cams


def read_images(path):
    out = []
    with open(path, "rb") as f:
        for _ in range(_read(f, 8, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, 64, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = _read(f, 8, "Q")[0]
            obs = []
            for _ in range(npts):
                x, y, p3d = _read(f, 24, "ddq")
                if p3d != -1:
                    obs.append((x, y, p3d))
            out.append(dict(id=iid, q=(qw, qx, qy, qz), t=(tx, ty, tz),
                            cam=cid, name=name.decode(), obs=obs))
    return out


def read_points3d(path):
    pts = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, 8, "Q")[0]):
            pid, x, y, z, r, g, b, err = _read(f, 43, "QdddBBBd")
            ntrack = _read(f, 8, "Q")[0]
            f.read(8 * ntrack)
            pts[pid] = (x, y, z)
    return pts


def quat_to_R(q):
    """COLMAP stores (w, x, y, z) for the WORLD-TO-CAMERA rotation."""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def intrinsics(cam):
    p = cam["params"]
    if cam["model"] == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    return p[0], p[0], p[1], p[2]          # SIMPLE_PINHOLE: f, cx, cy


def reprojection_error(images, cams, points):
    """Does our reading of R, t and K actually put the 3D points on their 2D observations?

    This is the whole correctness argument for the conversion. A transposed rotation, a
    quaternion in the wrong order or a camera-to-world mix-up all produce a file that LOOKS
    fine and puts every camera somewhere wrong - and COLMAP already ships the data that ties
    poses to pixels, so there is no reason to trust the convention instead of checking it.
    """
    total, n = 0.0, 0
    for im in images[: min(len(images), 40)]:
        fx, fy, cx, cy = intrinsics(cams[im["cam"]])
        R, t = quat_to_R(im["q"]), im["t"]
        for (u, v, pid) in im["obs"][:200]:
            if pid not in points:
                continue
            X = points[pid]
            xc = R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + t[0]
            yc = R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + t[1]
            zc = R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2]
            if zc <= 1e-6:
                continue
            du = fx * xc / zc + cx - u
            dv = fy * yc / zc + cy - v
            total += (du * du + dv * dv) ** 0.5
            n += 1
    return (total / n if n else float("inf")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("name")
    ap.add_argument("--every", type=int, default=1, help="take every Nth image")
    ap.add_argument("--max", type=int, default=0, help="cap the image count (0 = no cap)")
    args = ap.parse_args()

    sparse = os.path.join(args.scene, "sparse", "0")
    if not os.path.isdir(sparse):
        sys.exit(f"no sparse/0 under {args.scene}")

    cams = read_cameras(os.path.join(sparse, "cameras.bin"))
    images = read_images(os.path.join(sparse, "images.bin"))
    points = read_points3d(os.path.join(sparse, "points3D.bin"))
    images.sort(key=lambda im: im["name"])

    bad = {c["model"] for c in cams.values()} - UNDISTORTED
    if bad:
        sys.exit(f"camera model(s) {sorted(bad)} carry distortion; undistort the scene first "
                 f"rather than treating them as a pinhole")

    err, n = reprojection_error(images, cams, points)
    print(f"reprojection check: {err:.3f} px mean over {n} observations")
    if err > 2.0:
        sys.exit(f"mean reprojection error {err:.2f} px is too high - the pose convention is "
                 f"being read wrongly, and every camera would be placed wrong")

    picked = images[:: max(1, args.every)]
    if args.max:
        picked = picked[: args.max]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "SpawnScene", "wwwroot",
                           "datasets", args.name)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "poses.par"), "w") as f:
        f.write(f"{len(picked)}\n")
        for im in picked:
            fx, fy, cx, cy = intrinsics(cams[im["cam"]])
            R = quat_to_R(im["q"])
            k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            flat = [v for row in R for v in row]
            vals = " ".join(f"{v:.9g}" for v in k + flat + list(im["t"]))
            f.write(f"{im['name']} {vals}\n")

    first = cams[picked[0]["cam"]]
    manifest = dict(
        name=args.name,
        source=os.path.abspath(args.scene),
        imageDir="images",
        images=[im["name"] for im in picked],
        width=first["width"],
        height=first["height"],
        poses="poses.par",
        note="Converted from COLMAP by tools/colmap_to_dataset.py. Images are mounted, not "
             "copied - see MOUNTS in tools/_spa_server.js.",
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  {len(picked)} of {len(images)} images, {first['width']}x{first['height']}")
    print()
    print("Serve the images without copying them into wwwroot:")
    print(f'  MOUNTS="/datasets/{args.name}/images={os.path.abspath(args.scene)}/images" \\')
    print(f"    node tools/_spa_server.js SpawnScene/bin/PublishRelease/wwwroot 8080")


if __name__ == "__main__":
    main()
