# DrJohnson wide-baseline matcher workbench (Tuvok 2026-09-30).
# Ground truth: COLMAP sparse/0 (poses, intrinsics, per-image 2D->3D observations) for the 44 images SpawnScene uses.
# For every GT-overlapping pair, count matches that satisfy the TRUE epipolar geometry (Sampson < 2 px).
import json, struct, sys, time
import numpy as np, cv2

SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
MAN = r"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\DrJohnson\manifest.json"
names = json.load(open(MAN))["images"]

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

cams = read_cameras(SRC + r"\sparse\0\cameras.bin")
imgs = read_images(SRC + r"\sparse\0\images.bin")
K = {}; R = {}; T = {}; P3 = {}
for nm in names:
    q, t, cid, data = imgs[nm]
    model, w, h, prm = cams[cid]
    fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model in (0, 2) else (prm[0], prm[1], prm[2], prm[3])
    K[nm] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    R[nm] = qrot(q); T[nm] = np.array(t)
    P3[nm] = set(int(v) for v in data["p"] if v >= 0)

def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def F_of(a, b):  # x_b^T F x_a = 0
    Rr = R[b] @ R[a].T; tr = T[b] - Rr @ T[a]
    return np.linalg.inv(K[b]).T @ skew(tr) @ Rr @ np.linalg.inv(K[a])

pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        s = len(P3[names[i]] & P3[names[j]])
        if s >= 30: pairs.append((names[i], names[j], s))
print(f"{len(names)} images, {len(pairs)} GT-overlapping pairs (>=30 shared COLMAP points) of {len(names)*(len(names)-1)//2}")

gray = {nm: cv2.imread(SRC + r"\images" + "\\" + nm, cv2.IMREAD_GRAYSCALE) for nm in names}
print("image size", gray[names[0]].shape[::-1])

def sampson(F, xa, xb):
    xa1 = np.c_[xa, np.ones(len(xa))]; xb1 = np.c_[xb, np.ones(len(xb))]
    Fx = xa1 @ F.T; Ftx = xb1 @ F
    num = np.sum(xb1 * Fx, axis=1) ** 2
    den = Fx[:, 0]**2 + Fx[:, 1]**2 + Ftx[:, 0]**2 + Ftx[:, 1]**2
    return np.sqrt(num / den)

def evaluate(label, detect, norm, ratio=0.8, cross=False):
    t0 = time.time()
    feats = {nm: detect(gray[nm]) for nm in names}
    bf = cv2.BFMatcher(norm, crossCheck=False)
    good_pairs = 0; tot_correct = 0; tot_matches = 0; per = []
    for a, b, s in pairs:
        ka, da = feats[a]; kb, db = feats[b]
        if da is None or db is None or len(ka) < 2 or len(kb) < 2: per.append(0); continue
        m = bf.knnMatch(da, db, k=2)
        m = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
        if not m: per.append(0); continue
        xa = np.array([ka[x.queryIdx].pt for x in m]); xb = np.array([kb[x.trainIdx].pt for x in m])
        c = int(np.sum(sampson(F_of(a, b), xa, xb) < 2.0))
        per.append(c); tot_correct += c; tot_matches += len(m)
        if c >= 15: good_pairs += 1
    per = np.array(per)
    print(f"{label:34s} verifiable pairs (>=15 correct) {good_pairs:3d}/{len(pairs)}  median correct {int(np.median(per)):4d}  "
          f"precision {tot_correct / max(tot_matches,1):.2f}  ({time.time()-t0:.0f}s)")

class FastBrief:  # SpawnScene's baseline shape: FAST-9 t=25, top 2000, BRIEF-256 on a blurred image, no orientation/scale
    def __init__(self): self.fast = cv2.FastFeatureDetector_create(25, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    def __call__(self, g):
        k = sorted(self.fast.detect(g), key=lambda p: -p.response)[:2000]
        try:
            ext = cv2.xfeatures2d.BriefDescriptorExtractor_create(32)
        except Exception:
            ext = None
        if ext is None:  # fallback: ORB descriptor with angle forced to 0 = steered-less rBRIEF at one scale
            for p in k: p.angle = 0; p.octave = 0; p.size = 31
            return cv2.ORB_create(2000, 1.2, 1, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 25).compute(g, k)
        return ext.compute(cv2.GaussianBlur(g, (5, 5), 2), k)

evaluate("FAST+BRIEF (baseline, 1 scale)", FastBrief(), cv2.NORM_HAMMING)
evaluate("ORB 2000 (oriented, 8 levels)", lambda g: cv2.ORB_create(2000).detectAndCompute(g, None), cv2.NORM_HAMMING)
evaluate("ORB 5000", lambda g: cv2.ORB_create(5000).detectAndCompute(g, None), cv2.NORM_HAMMING)
evaluate("SIFT 2000", lambda g: cv2.SIFT_create(2000).detectAndCompute(g, None), cv2.NORM_L2)
evaluate("SIFT 8000", lambda g: cv2.SIFT_create(8000).detectAndCompute(g, None), cv2.NORM_L2)
try:
    evaluate("AKAZE", lambda g: cv2.AKAZE_create().detectAndCompute(g, None), cv2.NORM_HAMMING)
except Exception as e:
    print("AKAZE failed", e)
