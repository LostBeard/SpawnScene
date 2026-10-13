"""COLMAP (pycolmap) on a sample photo set, undistorted to pinhole - a reference pose set for the user path's SfM.

    python tools/colmap_sample.py <image_dir> <out_dir> [--model SIMPLE_RADIAL]

TJ 2026-10-13: the pine cone sample trains to 20 dB on its OWN photos through the user path (our SfM + pinhole), where the
COLMAP benchmark scenes reach 27-37 dB with the same trainer. This gives the same photos COLMAP's poses and a lens model,
undistorted to pinhole (<out_dir>/undistorted/images + sparse), for tools/colmap_to_dataset.py. Prints the camera COLMAP
fitted (focal, radial k) and the reprojection error, the two things the user path cannot model or reports worse.
"""
import argparse
import os
import shutil

import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("image_dir")
ap.add_argument("out_dir")
ap.add_argument("--model", default="SIMPLE_RADIAL")
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
db = os.path.join(a.out_dir, "db.db")
if os.path.exists(db):
    os.remove(db)
reader = pycolmap.ImageReaderOptions()
reader.camera_model = a.model
pycolmap.extract_features(db, a.image_dir, reader_options=reader, camera_mode=pycolmap.CameraMode.SINGLE)
pycolmap.match_exhaustive(db)
sparse = os.path.join(a.out_dir, "sparse")
if os.path.exists(sparse):
    shutil.rmtree(sparse)
os.makedirs(sparse)
recs = pycolmap.incremental_mapping(db, a.image_dir, sparse)
best_k, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
print(f"models: {[(k, r.num_reg_images()) for k, r in recs.items()]}; best {best_k}: {best.num_reg_images()} images, "
      f"{best.num_points3D()} points, mean reprojection error {best.compute_mean_reprojection_error():.3f} px")
for cid, cam in best.cameras.items():
    print(f"camera {cid}: {cam.model} {cam.width}x{cam.height} params {list(cam.params)}")
und = os.path.join(a.out_dir, "undistorted")
if os.path.exists(und):
    shutil.rmtree(und)
pycolmap.undistort_images(und, os.path.join(sparse, str(best_k)), a.image_dir)
print(f"undistorted to {und}")
