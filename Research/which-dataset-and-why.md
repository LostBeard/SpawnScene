# Which dataset to work with, and why

Written 2026-09-21 for whoever picks this up next. New file, nothing else touched.

## The one rule that matters

**Use a scene that appears in the 3D Gaussian Splatting paper's evaluation, and that ships a
COLMAP `sparse/0` folder containing `points3D.bin`.**

Two reasons, both load-bearing:

1. **A published number to measure against.** 3DGS reports roughly 29 dB PSNR on drjohnson. Our
   runs sit around 13-15 dB. Without a published target you cannot tell "15 dB" from "working",
   and a whole day can go into celebrating a +1 dB delta on a reconstruction that is nowhere near
   usable. That happened.
2. **`points3D.bin` is the initialisation.** The reference does not unproject depth maps. It
   initialises from the sparse SfM cloud (`GaussianModel.create_from_pcd`), where every point is
   triangulated from at least two images by construction. A dataset without that cloud cannot be
   run the way the reference runs, and cannot be compared to its numbers.

A corollary that cost this project real time: **Bathroom, TempleRing and any unposed phone capture
fail both tests.** They have no ground-truth poses and no sparse cloud, so every measurement made
on them mixes pose error with optimiser error and has no reference value to be judged against.
Do not tune on them.

## What is already on this machine

`C:\Users\TJ\Downloads\tandt_db\` holds the standard 3DGS evaluation download:

| Path | Scene | Type | Notes |
|---|---|---|---|
| `db/drjohnson` | Deep Blending | indoor room | 263 images, 1332x876, 80,861 sparse points |
| `db/playroom` | Deep Blending | indoor room | same family, generally a bit easier |
| `tandt/truck` | Tanks and Temples | outdoor object | usually the easiest to get right first |
| `tandt/train` | Tanks and Temples | outdoor | harder than truck |

All four have `sparse/0` with `cameras.bin`, `images.bin`, `points3D.bin`.

## What I was actually using

`db/drjohnson`. One PINHOLE camera: fx 1035.5, fy 1034.97, cx 666, cy 438, 1332x876.

Converted into two subsampled datasets under `SpawnScene/wwwroot/datasets/`:

```
python tools/colmap_to_dataset.py "C:/Users/TJ/Downloads/tandt_db/db/drjohnson" DrJohnson      --every 6   #  44 views
python tools/colmap_to_dataset.py "C:/Users/TJ/Downloads/tandt_db/db/drjohnson" DrJohnsonDense --every 2   # 132 views
```

Both produce `manifest.json`, `poses.par` (Middlebury format) and `points3d.bin`
(i32 count, then 3x f32 xyz + 4x u8 RGBA per point; 79,922 points after filtering to track
length >= 2 and reprojection error <= 2 px).

## What I would pick instead, starting over

**`tandt/truck` or `db/playroom`.** drjohnson is a good scene and is exactly what the reference
uses, but an indoor room is the hardest case: heavy occlusion, large untextured wall and floor
regions where the sparse cloud is thin, and cameras that see very different subsets of the scene.

If the goal is "get ONE scene looking correct end to end", start on `truck`, confirm you land near
the published number, and only then move indoors. Getting a room right while the pipeline itself is
unproven confounds "this scene is hard" with "this code is wrong".

## Practical notes about the converter and the harness

- **Images are mounted, not copied.** `tools/_spa_server.js` reads each dataset's `manifest.json`,
  finds its `source` field, and mounts `/datasets/<name>/images` at the original folder. drjohnson
  is 168 MB of JPEG and anything under `wwwroot` is recopied on every publish - this repo has
  already had a publish balloon to 13 GB from output landing inside the project.
- **The server builds its mount table at startup.** Adding a new dataset requires restarting it, or
  the images 404.
- **The converter proves its own conversion.** It reprojects the sparse points through the parsed
  poses and refuses to write if the mean error exceeds 2 px. drjohnson comes out at 0.586 px over
  7,896 observations, which is what makes the pose convention trustworthy rather than assumed.
- **It refuses distorted camera models.** Only PINHOLE and SIMPLE_PINHOLE; anything carrying
  distortion has to be undistorted first rather than silently treated as a pinhole.
- **`--every N` picks every Nth image after sorting by filename.** Subsampling is not free: 33
  supervised views gave 2.52 supervising views per splat, 99 gave 8.49. More views is the single
  biggest quality lever measured here.

## One more thing worth having

The INRIA reference implementation publishes **pre-trained `.ply` models** for these exact scenes.
Loading one through SpawnScene's display path would separate renderer bugs from generator bugs in a
single step - if a known-good splat file renders as a mess, the problem is the viewer, not the
reconstruction. That check has never been run here, and it is the cheapest one available.

## Known-good targets, approximate

3DGS at 30,000 iterations reports roughly: drjohnson ~29 dB, playroom ~30 dB, truck ~25 dB,
train ~22 dB. Treat these as the right order of magnitude and check the paper for exact figures
before quoting them anywhere. The point is the gap: if a run reports 15 dB, it is not a tuning
problem.
