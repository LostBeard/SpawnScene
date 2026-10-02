# Cache SpawnScene's learned front end on DrJohnson, as the app runs it (2026-10-01): Kornia RaCo-ALIKED k1024 extractor
# at 1024x672 (the app decodes to 1024x673 and stretches to multiples of 32), keypoints mapped back to the 1024x673
# feature frame with the app's half-pixel convention, LightGlue+ k1024 matcher on all 946 pairs (1 pair per run, as b83).
# Output: dj_k1024.npz - kp [44,K,2] (1024x673 pixels), and per pair a, b, ia, ib (matched keypoint indices), score.
import json, os, time
import numpy as np, cv2, onnxruntime as ort

SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
MAN = r"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\DrJohnson\manifest.json"
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
HERE = os.path.dirname(os.path.abspath(__file__))
K = 1024
names = json.load(open(MAN))["images"]
n = len(names)
FW, FH, IW, IH = 1024, 673, 1024, 672

ext = ort.InferenceSession(os.path.join(KDIR, f"raco_aliked_extractor_k{K}.onnx"), providers=["CPUExecutionProvider"])
mat = ort.InferenceSession(os.path.join(KDIR, f"lightglue_matcher_k{K}.onnx"), providers=["CPUExecutionProvider"])
kp = np.zeros((n, K, 2), np.float32); nkp = np.zeros((n, K, 2), np.float32); desc = np.zeros((n, K, 128), np.float32)
t0 = time.time()
for i, nm in enumerate(names):
    im = cv2.imread(os.path.join(SRC, "images", nm))
    im = cv2.resize(im, (FW, FH), interpolation=cv2.INTER_AREA)
    im = cv2.resize(im, (IW, IH), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    k, nk, d = ext.run(["keypoints", "normalized_keypoints", "descriptors"], {"images": x})
    kp[i] = (k[0] + 0.5) * [FW / IW, FH / IH] - 0.5
    nkp[i] = nk[0]; desc[i] = d[0]
print(f"extracted {n} images in {time.time() - t0:.1f}s", flush=True)
A, B, IA, IB, SC = [], [], [], [], []
t0 = time.time()
for a in range(n):
    for b in range(a + 1, n):
        m, s = mat.run(["matches0", "mscores0"], {"normalized_keypoints": np.stack([nkp[a], nkp[b]])[:, None],
                                                 "descriptors": np.stack([desc[a], desc[b]])[:, None]})
        m = m[0]; s = s[0]
        ok = np.nonzero(m >= 0)[0]
        A.append(a); B.append(b); IA.append(ok.astype(np.int32)); IB.append(m[ok].astype(np.int32)); SC.append(s[ok].astype(np.float32))
    print(f"  pairs of {a}: {time.time() - t0:.0f}s", flush=True)
np.savez(os.path.join(HERE, f"dj_k{K}.npz"), kp=kp, names=np.array(names), a=np.array(A), b=np.array(B),
         ia=np.array(IA, dtype=object), ib=np.array(IB, dtype=object), sc=np.array(SC, dtype=object))
print("saved", flush=True)
