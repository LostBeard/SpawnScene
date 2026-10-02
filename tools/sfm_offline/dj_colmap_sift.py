# Validation for dj_colmap.py's scorer and the dataset: (1) COLMAP's published sparse/0 scored against itself, (2) COLMAP's
# own SIFT pipeline (extract + exhaustive match + incremental and global mapping) on the same resized images.
import os, sys, shutil
import numpy as np, cv2, pycolmap
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
n = len(names)
Cgt = {nm: -R[nm].T @ T[nm] for nm in names}
src = open(os.path.join(HERE, "dj_colmap.py")).read()
exec(src[src.index("def score"):src.index("if mode in")])
gt = pycolmap.Reconstruction(os.path.join(SRC, "sparse", "0"))
score(gt, "published sparse/0 vs itself")
W, H = 1024, 673
work = os.path.join(HERE, "colmap_sift"); img_dir = os.path.join(work, "images")
if os.path.exists(work): shutil.rmtree(work)
os.makedirs(img_dir)
for nm in names:
    im = cv2.imread(os.path.join(SRC, "images", nm))
    cv2.imwrite(os.path.join(img_dir, nm), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
db_path = os.path.join(work, "db.db")
pycolmap.Database.open(db_path).close()
f = 1035.5 * W / 1332
reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_PINHOLE"; reader.camera_params = f"{f},{W / 2},{H / 2}"
pycolmap.extract_features(db_path, img_dir, reader_options=reader, camera_mode=pycolmap.CameraMode.SINGLE)
pycolmap.match_exhaustive(db_path)
for label, fn in [("SIFT + COLMAP incremental", pycolmap.incremental_mapping), ("SIFT + GLOMAP global", pycolmap.global_mapping)]:
    out = os.path.join(work, label.split()[-1]); os.makedirs(out, exist_ok=True)
    recs = fn(db_path, img_dir, out)
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        score(rec, f"{label}, model {k}")
