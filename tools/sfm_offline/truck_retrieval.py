# Pair retrieval at TruckFull scale (2026-10-01): 251 images, 31,375 pairs - LightGlue on all of them is ~5 h in the
# browser. Score every pair by ALIKED mutual nearest neighbours passing a 0.9 ratio test (the best variant on DrJohnson),
# keep each image's top-k partners, measure recall of COLMAP-true pairs (>= 30 shared points) and of strong ones (>= 200).
import os, json, time, struct
import numpy as np, cv2, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\TJ\Downloads\tandt_db\tandt\truck"
MAN = r"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\TruckFull\manifest.json"
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
code = open(os.path.join(HERE, "dj_matchers.py")).read()
exec(code[code.index("def read_cameras"):code.index("cams = read_cameras")])
names = json.load(open(MAN))["images"]
n = len(names)
imgs = read_images(os.path.join(SRC, "sparse", "0", "images.bin"))
P3 = {nm: set(int(v) for v in imgs[nm][3]["p"] if v >= 0) for nm in names}
shared = np.zeros((n, n), int)
for a in range(n):
    for b in range(a + 1, n):
        shared[a, b] = shared[b, a] = len(P3[names[a]] & P3[names[b]])
cache = os.path.join(HERE, "truck_k1024_desc.npy")
if not os.path.exists(cache):
    ext = ort.InferenceSession(os.path.join(KDIR, "raco_aliked_extractor_k1024.onnx"), providers=["CPUExecutionProvider"])
    D = np.zeros((n, 1024, 128), np.float32)
    t0 = time.time()
    for i, nm in enumerate(names):
        im = cv2.imread(os.path.join(SRC, "images", nm))
        h, w = im.shape[:2]; s = 1024 / max(w, h); fw, fh = round(w * s), round(h * s)
        iw, ih = max(32, round(fw / 32) * 32), max(32, round(fh / 32) * 32)
        im = cv2.resize(cv2.resize(im, (fw, fh), interpolation=cv2.INTER_AREA), (iw, ih), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        D[i] = ext.run(["descriptors"], {"images": x})[0][0]
    np.save(cache, D)
    print(f"extracted {n} in {time.time() - t0:.0f}s", flush=True)
D = np.load(cache)
D /= np.linalg.norm(D, axis=2, keepdims=True)
t0 = time.time()
score = np.zeros((n, n))
for a in range(n):
    for b in range(a + 1, n):
        S = D[a] @ D[b].T
        top2 = -np.partition(-S, 1, axis=1)[:, :2]
        nn_ab = S.argmax(1); nn_ba = S.argmax(0)
        mutual = nn_ba[nn_ab] == np.arange(S.shape[0])
        d1 = np.sqrt(np.maximum(0, 2 - 2 * top2[:, 0])); d2 = np.sqrt(np.maximum(0, 2 - 2 * top2[:, 1]))
        score[a, b] = score[b, a] = np.sum(mutual & (d1 < 0.9 * d2))
print(f"scored {n * (n - 1) // 2} pairs in {time.time() - t0:.0f}s", flush=True)
true30 = {(a, b) for a in range(n) for b in range(a + 1, n) if shared[a, b] >= 30}
true200 = {(a, b) for a in range(n) for b in range(a + 1, n) if shared[a, b] >= 200}
print(f"{len(true30)} pairs share >= 30 COLMAP points, {len(true200)} share >= 200")
for k in [10, 15, 20, 30, 40]:
    sel = set()
    for a in range(n):
        for b in np.argsort(-score[a])[:k + 1]:
            if b != a: sel.add((min(a, b), max(a, b)))
    print(f"top-{k:2d} per image: {len(sel)} pairs ({len(sel) / (n * (n - 1) / 2):.1%} of all) - recall >=30 {len(true30 & sel) / len(true30):.0%}, "
          f">=200 {len(true200 & sel) / len(true200):.0%}", flush=True)
