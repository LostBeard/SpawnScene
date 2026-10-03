# Can a photo's OWN pixels tell a pose from its symmetric twin? (2026-10-03, DrJohnson image 41 / IMG_6576.)
# Heinly et al. ECCV 2014 ("conflicting observations"): project the rest of the model into the camera and check the photo
# shows it. Here: COLMAP's coloured points3D projected into image 41 (z-buffered), colour agreement with the photo, for
#   - the pose PnP gives from its normal LightGlue+ matches (dj_rot_aug.py: 180 deg off COLMAP), and
#   - the pose from the matches of the image turned 180 deg (0.6 deg off),
# plus every camera at its COLMAP pose, as the distribution a correct pose lands in.
import os, sys, struct
import numpy as np, cv2, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
KP = 3072
FW, FH, IW, IH = 1024, 673, 1024, 672
SX = 1332 / FW   # feature frame -> COLMAP image frame

def read_points3d(p):
    xyz, rgb = [], []
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        ids = {}
        for i in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            x = struct.unpack("<3d", f.read(24)); c = struct.unpack("<3B", f.read(3)); f.read(8)
            tl = struct.unpack("<Q", f.read(8))[0]; f.read(8 * tl)
            ids[pid] = len(xyz); xyz.append(x); rgb.append(c)
    return ids, np.array(xyz), np.array(rgb, np.float32)

pid_index, P, C = read_points3d(SRC + r"\sparse\0\points3D.bin")
print(f"{len(P)} COLMAP points")

def Kof(nm):  # intrinsics in the 1024x673 feature frame
    k = K[nm].copy(); k[:2] /= SX; return k

def photo(nm):
    im = cv2.imread(os.path.join(SRC, "images", nm))
    return cv2.GaussianBlur(cv2.resize(im, (FW, FH), interpolation=cv2.INTER_AREA), (5, 5), 0)[:, :, ::-1].astype(np.float32)

def score(nm, Rw, tw, cell=6, tol=40.0):
    """Fraction of visible (z-buffered) model points whose colour agrees with the photo; and how many were visible."""
    k = Kof(nm); img = photo(nm)
    Xc = P @ Rw.T + tw
    front = Xc[:, 2] > 0.05
    uv = (Xc[front] @ k.T); uv = uv[:, :2] / uv[:, 2:3]; z = Xc[front, 2]; col = C[front]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < FW - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < FH - 1)
    uv, z, col = uv[inside], z[inside], col[inside]
    # z-buffer on a coarse grid: only the nearest point of each cell counts (the others are hidden behind it)
    cx = (uv[:, 0] // cell).astype(int); cy = (uv[:, 1] // cell).astype(int)
    key = cy * (FW // cell + 1) + cx
    order = np.lexsort((z, key)); first = np.ones(len(order), bool); first[1:] = key[order][1:] != key[order][:-1]
    vis = order[first]
    if len(vis) == 0: return 0.0, 0
    px = img[uv[vis, 1].astype(int), uv[vis, 0].astype(int)]
    d = np.abs(px - col[vis]).mean(1)
    return float((d < tol).mean()), len(vis)

def pose_of(nm): return R[nm], T[nm]

# --- correct poses: the distribution -------------------------------------------------------------------------------
gt_scores = {nm: score(nm, *pose_of(nm)) for nm in names}
vals = np.array([s for s, _ in gt_scores.values()])
print(f"COLMAP poses, all 44: agreement median {np.median(vals):.3f}, min {vals.min():.3f} "
      f"({min(gt_scores, key=lambda n: gt_scores[n][0])}), p10 {np.percentile(vals, 10):.3f}")
print(f"  image 41 IMG_6576 at its COLMAP pose: {gt_scores['IMG_6576.jpg']}")

# --- image 41: the two resected candidates -------------------------------------------------------------------------
ext = ort.InferenceSession(os.path.join(KDIR, f"raco_aliked_extractor_k{KP}.onnx"), providers=["CPUExecutionProvider"])
mat = ort.InferenceSession(os.path.join(KDIR, f"lightglue_matcher_k{KP}.onnx"), providers=["CPUExecutionProvider"])
def net_image(nm):
    im = cv2.imread(os.path.join(SRC, "images", nm)); im = cv2.resize(im, (FW, FH), interpolation=cv2.INTER_AREA)
    return cv2.resize(im, (IW, IH), interpolation=cv2.INTER_LINEAR)
def extract(im, rot):
    r = np.rot90(im, k=rot).copy()
    x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    k, nk, d = ext.run(["keypoints", "normalized_keypoints", "descriptors"], {"images": x})
    k = k[0].astype(np.float64); h, w = r.shape[:2]
    for _ in range(rot % 4):
        h0, w0 = w, h; k = np.stack([w0 - 1 - k[:, 1], k[:, 0]], 1); h, w = h0, w0
    return (k + 0.5) * [FW / IW, FH / IH] - 0.5, nk[0], d[0]

def obs3d(nm):
    """COLMAP's 2D observations of image nm (feature frame) with a 3D point."""
    q, t, cid, data = imgs[nm]
    m = data["p"] >= 0
    xy = np.stack([data["x"][m], data["y"][m]], 1) / SX
    return xy, np.array([pid_index[int(p)] for p in data["p"][m]])

def resect(target, partners, rot):
    ka, na, da = extract(net_image(target), rot)
    w, px = [], []
    for p in partners:
        kb, nb, db = extract(net_image(p), 0)
        mm, _ = mat.run(["matches0", "mscores0"], {"normalized_keypoints": np.stack([na, nb])[:, None],
                                                  "descriptors": np.stack([da, db])[:, None]})
        mm = mm[0]; ok = np.nonzero(mm >= 0)[0]
        oxy, opid = obs3d(p)
        for i in ok:
            d = np.linalg.norm(oxy - kb[mm[i]], axis=1); j = int(np.argmin(d))
            if d[j] < 2.0: w.append(P[opid[j]]); px.append(ka[i])
    w = np.array(w, np.float64); px = np.array(px, np.float64)
    ok, rv, tv, inl = cv2.solvePnPRansac(w, px, Kof(target), None, reprojectionError=4.0, iterationsCount=2000)
    Rr = cv2.Rodrigues(rv)[0]
    def rot_err(A, B): return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
    return Rr, tv.ravel(), len(w), 0 if inl is None else len(inl), rot_err(Rr, R[target])

target, partners = "IMG_6576.jpg", ["IMG_6380.jpg", "IMG_6452.jpg"]
for rot in (0, 2):
    Rr, tr, n, ninl, err = resect(target, partners, rot)
    s, nv = score(target, Rr, tr)
    print(f"image 41 from the {rot*90:3d}-deg matches: PnP {ninl}/{n} inliers, {err:6.1f} deg off COLMAP -> "
          f"agreement {s:.3f} over {nv} visible points")

# --- refine each candidate within its own basin: local search on the agreement itself ---------------------------
from scipy.spatial.transform import Rotation as Rot
def refine(nm, Rr, tr, rounds=4):
    best = score(nm, Rr, tr)[0]
    step_r, step_t = np.radians(3.0), 0.05 * np.linalg.norm(np.mean(P, 0) - (-Rr.T @ tr))
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for axis in range(6):
                for sgn in (+1, -1):
                    if axis < 3:
                        dv = np.zeros(3); dv[axis] = sgn * step_r
                        R2 = Rot.from_rotvec(dv).as_matrix() @ Rr; t2 = tr
                    else:
                        c = -Rr.T @ tr; dc = np.zeros(3); dc[axis - 3] = sgn * step_t
                        R2 = Rr; t2 = -Rr @ (c + dc)
                    s = score(nm, R2, t2)[0]
                    if s > best + 1e-4: best, Rr, tr, improved = s, R2, t2, True
        step_r /= 2; step_t /= 2
    return Rr, tr, best

def rot_err(A, B): return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
for rot in (0, 2):
    Rr, tr, n, ninl, err = resect(target, partners, rot)
    R2, t2, s2 = refine(target, Rr, tr)
    c_gt = -R[target].T @ T[target]; c2 = -R2.T @ t2
    print(f"image 41 from the {rot*90:3d}-deg matches, REFINED on agreement: {rot_err(R2, R[target]):6.1f} deg off, "
          f"centre {np.linalg.norm(c2 - c_gt):.3f} from COLMAP's -> agreement {s2:.3f}")
