# Export a dataset's views (grayscale, optionally resized) + GT-overlapping pairs with their TRUE F, for the C#
# matching workbench (SpawnScene.Tests *MatchingTests). Usage: python tools/gt_matching_export.py <src> <manifest> <outdir> <maxdim|0> [minShared]
import json, os, struct, sys
import numpy as np, cv2

def read_cameras(p):
    cams = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        np_by_model = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            k = np_by_model[model]
            prm = struct.unpack("<" + "d" * k, f.read(8 * k))
            cams[cid] = (model, w, h, prm)
    return cams

def read_images(p):
    imgs = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32)); t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\0": break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            data = np.frombuffer(f.read(24 * n2), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")]))
            imgs[name.decode()] = (q, t, cid, data)
    return imgs

def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


src, man, out, maxdim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
min_shared = int(sys.argv[5]) if len(sys.argv) > 5 else 30
names = json.load(open(man))["images"]
cams = read_cameras(os.path.join(src, "sparse", "0", "cameras.bin"))
imgs = read_images(os.path.join(src, "sparse", "0", "images.bin"))
def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
os.makedirs(out, exist_ok=True)
K, R, T, P3 = {}, {}, {}, {}
for i, nm in enumerate(names):
    q, t, cid, data = imgs[nm]
    model, w, h, prm = cams[cid]
    fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model in (0, 2) else (prm[0], prm[1], prm[2], prm[3])
    g = cv2.imread(os.path.join(src, "images", nm), cv2.IMREAD_GRAYSCALE)
    if maxdim and max(g.shape) > maxdim:
        r = maxdim / max(g.shape)
        g = cv2.resize(g, (round(g.shape[1] * r), round(g.shape[0] * r)), interpolation=cv2.INTER_AREA)
    # K is for the COLMAP camera's w x h; the file (and any resize) can differ - tandt's images/ are HALF of it.
    sx, sy = g.shape[1] / w, g.shape[0] / h
    if i == 0: print("camera", w, "x", h, "-> image", g.shape[1], "x", g.shape[0], f"(scale {sx:.4f}, {sy:.4f})")
    # pixel-centre convention: x' = (x + 0.5) s - 0.5
    K[nm] = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1.0]])
    R[nm] = qrot(q); T[nm] = np.array(t)
    P3[nm] = set(int(v) for v in data["p"] if v >= 0)
    hh, ww = g.shape
    with open(os.path.join(out, f"{i:03d}.gray"), "wb") as f:
        f.write(struct.pack("<ii", ww, hh)); f.write(g.tobytes())

def F_of(a, b):
    Rr = R[b] @ R[a].T; tr = T[b] - Rr @ T[a]
    return np.linalg.inv(K[b]).T @ skew(tr) @ Rr @ np.linalg.inv(K[a])

n = 0
with open(os.path.join(out, "pairs.txt"), "w") as f:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = len(P3[names[i]] & P3[names[j]])
            if s >= min_shared:
                F = F_of(names[i], names[j])
                f.write(f"{i} {j} {s} " + " ".join(f"{v:.17g}" for v in F.ravel()) + "\n"); n += 1
print(len(names), "images,", n, "GT pairs ->", out, "size", g.shape[::-1])
