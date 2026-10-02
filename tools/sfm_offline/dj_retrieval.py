# Pair retrieval for the learned front end (2026-10-01): LightGlue costs ~0.6 s/pair in the browser, so all 31k TruckFull
# pairs are out of reach. Score every pair by mutual nearest neighbours of the ALIKED descriptors (a 1024x1024 dot product,
# cheap on the GPU), keep each image's top-k partners, and measure what fraction of the pairs that matter survive:
# COLMAP-true pairs (>= 30 shared points) and the pairs LightGlue + verification actually used (rotation-consistent).
import os, sys, json
import numpy as np, cv2, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
KDIR = r"D:\users\tj\Projects\SpawnScene\SpawnScene\_scratch\kornia"
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
n = len(names)
truepairs = {(a, b) for a in range(n) for b in range(a + 1, n) if len(P3[names[a]] & P3[names[b]]) >= 30}
cache = os.path.join(HERE, "dj_k1024_desc.npy")
if not os.path.exists(cache):
    ext = ort.InferenceSession(os.path.join(KDIR, "raco_aliked_extractor_k1024.onnx"), providers=["CPUExecutionProvider"])
    D = np.zeros((n, 1024, 128), np.float32)
    for i, nm in enumerate(names):
        im = cv2.imread(os.path.join(SRC, "images", nm))
        im = cv2.resize(cv2.resize(im, (1024, 673), interpolation=cv2.INTER_AREA), (1024, 672), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        D[i] = ext.run(["descriptors"], {"images": x})[0][0]
    np.save(cache, D)
D = np.load(cache)
# LightGlue's verified pairs (dj_k1024.npz matches with >= 15) as the "used" set.
z = np.load(os.path.join(HERE, "dj_k1024.npz"), allow_pickle=True)
lg = {(int(a), int(b)) for a, b, ia in zip(z["a"], z["b"], z["ia"]) if len(ia) >= 50}
score = np.zeros((n, n))
for thr in [0.0]:
    for a in range(n):
        for b in range(a + 1, n):
            S = D[a] @ D[b].T
            nn_ab = S.argmax(1); nn_ba = S.argmax(0)
            mutual = nn_ba[nn_ab] == np.arange(len(nn_ab))
            best = S[np.arange(len(nn_ab)), nn_ab]
            score[a, b] = score[b, a] = np.sum(mutual & (best > 0.75))
print(f"{len(truepairs)} COLMAP-true pairs, {len(lg)} LightGlue pairs with >= 50 matches")
for k in [5, 8, 10, 15, 20]:
    sel = set()
    for a in range(n):
        for b in np.argsort(-score[a])[:k]:
            if b != a: sel.add((min(a, b), max(a, b)))
    rt = len(truepairs & sel) / len(truepairs); rl = len(lg & sel) / len(lg)
    print(f"top-{k:2d} per image: {len(sel)} pairs ({len(sel) / (n * (n - 1) / 2):.0%} of all) - recall COLMAP-true {rt:.0%}, LightGlue>=50 {rl:.0%}")
