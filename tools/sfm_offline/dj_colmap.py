# COLMAP (pycolmap 4.2.1) on SpawnScene's EXACT DrJohnson features and matches (2026-10-01): can a proven pipeline -
# GLOMAP global mapping, and COLMAP incremental mapping - reconstruct DrJohnson from them? Scored against COLMAP's
# own published reconstruction (sparse/0) by similarity alignment of camera centres.
import os, sys, json, shutil
import numpy as np, cv2, pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
n = len(names)
Cgt = {nm: -R[nm].T @ T[nm] for nm in names}
z = np.load(os.path.join(HERE, "dj_k1024.npz"), allow_pickle=True)
kp = z["kp"]
W, H = 1024, 673
work = os.path.join(HERE, "colmap_ws")
img_dir = os.path.join(work, "images")
if os.path.exists(work): shutil.rmtree(work)
os.makedirs(img_dir)
for nm in names:
    im = cv2.imread(os.path.join(SRC, "images", nm))
    cv2.imwrite(os.path.join(img_dir, nm), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
db_path = os.path.join(work, "db.db")
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
f = 1035.5 * W / 1332
reader = pycolmap.ImageReaderOptions()
reader.camera_model = "SIMPLE_PINHOLE"
reader.camera_params = f"{f},{W / 2},{H / 2}"
pycolmap.Database.open(db_path).close()
pycolmap.import_images(db_path, img_dir, pycolmap.CameraMode.SINGLE, options=reader)
db = pycolmap.Database.open(db_path)
ids = {im.name: im.image_id for im in db.read_all_images()}
for i, nm in enumerate(names):
    db.write_keypoints(ids[nm], (kp[i] + 0.5).astype(np.float32))   # COLMAP: pixel centres at +0.5
for a, b, ia, ib in zip(z["a"], z["b"], z["ia"], z["ib"]):
    if len(ia) == 0: continue
    db.write_matches(ids[names[a]], ids[names[b]], np.stack([ia, ib], 1).astype(np.uint32))
db.close()
pairs_path = os.path.join(work, "pairs.txt")
open(pairs_path, "w").write("\n".join(f"{names[a]} {names[b]}" for a, b in zip(z["a"], z["b"])))
pycolmap.verify_matches(db_path, pairs_path)

def score(rec, label):
    got = {im.name: im for im in rec.images.values()}
    common = [nm for nm in names if nm in got]
    X = np.array([got[nm].projection_center() for nm in common]); G = np.array([Cgt[nm] for nm in common])
    mx, mg = X.mean(0), G.mean(0); Xc, Gc = X - mx, G - mg
    U, S, Vt = np.linalg.svd(Gc.T @ Xc); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum()
    Y = s * Xc @ Rr.T + mg
    spread = np.median(np.linalg.norm(G - np.median(G, 0), axis=1))
    e = np.linalg.norm(Y - G, axis=1) / spread
    print(f"{label}: {len(common)}/{n} images, {rec.num_points3D()} points, position error median {np.median(e):.2%} p90 {np.percentile(e, 90):.1%} of spread", flush=True)

if mode in ("both", "global"):
    out = os.path.join(work, "global"); os.makedirs(out, exist_ok=True)
    recs = pycolmap.global_mapping(db_path, img_dir, out)
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        score(rec, f"GLOMAP global, model {k}")
if mode in ("both", "incremental"):
    out = os.path.join(work, "incremental"); os.makedirs(out, exist_ok=True)
    recs = pycolmap.incremental_mapping(db_path, img_dir, out)
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        score(rec, f"COLMAP incremental, model {k}")
