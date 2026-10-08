"""COLMAP text model -> COLMAP binary model (cameras.bin, images.bin, points3D.bin), standard layout.

    python tools/colmap_txt_to_bin.py <txt_dir> <out_dir>

For SpawnScene's &exportcolmap=1 export (Studio.ColmapExport): gsplat's pycolmap fork reads .bin correctly, while its
text readers are Python 2 code (np.array(map(...))) that also stop at the first empty line - every image's empty
points line. PINHOLE cameras only; images carry no 2D points; points carry no tracks.
"""
import os
import struct
import sys

MODELS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = lambda name: [l.split() for l in open(os.path.join(src, name), encoding="utf-8")
                         if l.strip() and not l.startswith("#")]
    cams = rows("cameras.txt")
    with open(os.path.join(out, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            cid, model, w, h, params = int(c[0]), MODELS[c[1]], int(c[2]), int(c[3]), [float(v) for v in c[4:]]
            f.write(struct.pack("<iiQQ", cid, model, w, h))
            f.write(struct.pack("<" + "d" * len(params), *params))
    imgs = rows("images.txt")
    with open(os.path.join(out, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for r in imgs:
            iid = int(r[0]); q = [float(v) for v in r[1:5]]; t = [float(v) for v in r[5:8]]; cid = int(r[8]); name = r[9]
            f.write(struct.pack("<I4d3dI", iid, *q, *t, cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    pts = rows("points3D.txt")
    with open(os.path.join(out, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for p in pts:
            f.write(struct.pack("<Q3d3Bd", int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                                int(p[4]), int(p[5]), int(p[6]), float(p[7])))
            f.write(struct.pack("<Q", 0))
    print(f"{len(cams)} cameras, {len(imgs)} images, {len(pts)} points -> {out}")


if __name__ == "__main__":
    main()
