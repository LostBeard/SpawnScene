# Does DrJohnson image 41 (IMG_6576, a top-down view of the patterned rug) match its partners at a ROTATED
# correspondence because the learned front end is not rotation-invariant? (2026-10-03, b134-b144: SpawnScene places it
# ~128% of the spread off COLMAP, forward ~95 deg, with every verified pair agreeing.)
# Extract IMG_6576 at 0/90/180/270 deg, map its keypoints back to the unrotated frame, match with LightGlue+ against
# IMG_6380 (image 13) and IMG_6452 (image 23), 5-point E-RANSAC, and score the relative rotation against COLMAP.
import os, sys
import numpy as np, cv2, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
KP = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
FW, FH, IW, IH = 1024, 673, 1024, 672
ext = ort.InferenceSession(os.path.join(KDIR, f"raco_aliked_extractor_k{KP}.onnx"), providers=["CPUExecutionProvider"])
mat = ort.InferenceSession(os.path.join(KDIR, f"lightglue_matcher_k{KP}.onnx"), providers=["CPUExecutionProvider"])

def net_image(nm):
    im = cv2.imread(os.path.join(SRC, "images", nm))
    im = cv2.resize(im, (FW, FH), interpolation=cv2.INTER_AREA)
    return cv2.resize(im, (IW, IH), interpolation=cv2.INTER_LINEAR)

def extract(im, rot):
    """Keypoints (in the UNROTATED 1024x673 feature frame), normalized keypoints (as the matcher saw them), descriptors."""
    r = np.rot90(im, k=rot).copy()          # rot * 90 deg counter-clockwise
    x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    k, nk, d = ext.run(["keypoints", "normalized_keypoints", "descriptors"], {"images": x})
    k = k[0].astype(np.float64)
    h, w = r.shape[:2]
    # Undo rot90 (counter-clockwise): a point (x, y) of the rotated image came from the original at...
    for _ in range(rot % 4):
        # one CCW turn mapped original (x0, y0) of an (H0 x W0) image to (y0, W0 - 1 - x0); invert it
        h0, w0 = w, h   # size before this turn
        x0 = w0 - 1 - k[:, 1]; y0 = k[:, 0]
        k = np.stack([x0, y0], 1); h, w = h0, w0
    k = (k + 0.5) * [FW / IW, FH / IH] - 0.5
    return k, nk[0], d[0]

f = 1035.5 * 1024 / 1332
Kc = np.array([[f, 0, 666 * 1024 / 1332], [0, f, 438 * 1024 / 1332], [0, 0, 1.0]])

def rot_err(A, B): return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))

target = "IMG_6576.jpg"
partners = ["IMG_6380.jpg", "IMG_6452.jpg"]
tim = net_image(target)
pfeat = {p: extract(net_image(p), 0) for p in partners}
for rot in range(4):
    ka, na, da = extract(tim, rot)
    for p in partners:
        kb, nb, db = pfeat[p]
        m, s = mat.run(["matches0", "mscores0"], {"normalized_keypoints": np.stack([na, nb])[:, None],
                                                 "descriptors": np.stack([da, db])[:, None]})
        m = m[0]; ok = np.nonzero(m >= 0)[0]
        pa = ka[ok]; pb = kb[m[ok]]
        if len(ok) < 8:
            print(f"rot {rot*90:3d}  {target} -> {p}: {len(ok)} matches"); continue
        E, inl = cv2.findEssentialMat(pa, pb, Kc, cv2.RANSAC, 0.999, 2.0)
        if E is None:
            print(f"rot {rot*90:3d}  {target} -> {p}: {len(ok)} matches, no E"); continue
        n_in, Rr, t, _ = cv2.recoverPose(E[:3], pa, pb, Kc, mask=inl)
        Rgt = R[p] @ R[target].T
        print(f"rot {rot*90:3d}  {target} -> {p}: {len(ok)} matches, {int(inl.sum())} E inliers, "
              f"rotation error vs COLMAP {rot_err(Rr, Rgt):6.1f} deg")
