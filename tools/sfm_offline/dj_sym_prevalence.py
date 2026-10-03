# How often does a CORRECT pair also match strongly with one image turned 180 deg? (2026-10-03.)
# The detection proposed for DrJohnson image 41 (dj_rot_aug.py): a pair is "symmetry-ambiguous" when the match with
# image A turned 180 deg keeps >= RATIO of the normal match's E inliers at a relative rotation > ANGLE deg away from it.
# Before it can drop pairs in the app, measure on every candidate pair (>= 50 cached k1024 matches) with COLMAP truth:
# flagged pairs whose normal rotation is RIGHT are false alarms; flagged pairs whose normal rotation is WRONG are hits.
import os, sys, time
import numpy as np, cv2, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
KP = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
FW, FH, IW, IH = 1024, 673, 1024, 672
ext = ort.InferenceSession(os.path.join(KDIR, f"raco_aliked_extractor_k{KP}.onnx"), providers=["CPUExecutionProvider"])
mat = ort.InferenceSession(os.path.join(KDIR, f"lightglue_matcher_k{KP}.onnx"), providers=["CPUExecutionProvider"])
f = 1035.5 * 1024 / 1332
Kc = np.array([[f, 0, 666 * 1024 / 1332], [0, f, 438 * 1024 / 1332], [0, 0, 1.0]])

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
def rot_err(A, B): return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
def relpose(fa, fb):
    ka, na, da = fa; kb, nb, db = fb
    m, _ = mat.run(["matches0", "mscores0"], {"normalized_keypoints": np.stack([na, nb])[:, None],
                                             "descriptors": np.stack([da, db])[:, None]})
    m = m[0]; ok = np.nonzero(m >= 0)[0]
    if len(ok) < 15: return None, 0
    E, inl = cv2.findEssentialMat(ka[ok], kb[m[ok]], Kc, cv2.RANSAC, 0.999, 2.0)
    if E is None: return None, 0
    _, Rr, _, _ = cv2.recoverPose(E[:3], ka[ok], kb[m[ok]], Kc, mask=inl)
    return Rr, int(inl.sum())

z = np.load(os.path.join(HERE, "dj_k1024.npz"), allow_pickle=True)
pairs = [(int(a), int(b)) for a, b, ia in zip(z["a"], z["b"], z["ia"]) if len(ia) >= 50]
print(f"{len(pairs)} candidate pairs (>= 50 cached k1024 matches), K={KP}", flush=True)
t0 = time.time()
f0 = [extract(net_image(nm), 0) for nm in names]
f2 = [extract(net_image(nm), 2) for nm in names]
print(f"extracted 2 x {len(names)} in {time.time() - t0:.0f}s", flush=True)
rows = []
for q, (a, b) in enumerate(pairs):
    R0, i0 = relpose(f0[a], f0[b]); R2, i2 = relpose(f2[a], f0[b])
    Rgt = R[names[b]] @ R[names[a]].T
    e0 = rot_err(R0, Rgt) if R0 is not None else np.nan
    e2 = rot_err(R2, Rgt) if R2 is not None else np.nan
    d02 = rot_err(R0, R2) if R0 is not None and R2 is not None else np.nan
    rows.append((a, b, i0, i2, e0, e2, d02))
    if q % 20 == 0: print(f"  {q}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
np.save(os.path.join(HERE, f"dj_sym_k{KP}.npy"), np.array(rows, float))
rows = np.array(rows, float)
for RATIO in (0.3, 0.5, 0.7):
    for ANGLE in (30,):
        flag = (rows[:, 3] >= RATIO * rows[:, 2]) & (rows[:, 6] > ANGLE)
        right = rows[:, 4] < 10; wrong = rows[:, 4] >= 30
        print(f"RATIO {RATIO} ANGLE {ANGLE}: flagged {int(flag.sum())} of {len(rows)}; "
              f"false alarms (normal rotation RIGHT, < 10 deg) {int((flag & right).sum())} of {int(right.sum())}; "
              f"hits (normal rotation WRONG, >= 30 deg) {int((flag & wrong).sum())} of {int(wrong.sum())}")
print("normal-wrong pairs:", [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), round(r[4], 1), round(r[5], 1)) for r in rows if r[4] >= 30])
EOF_SENTINEL = None
