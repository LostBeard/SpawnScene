# DrJohnson 44-view positioning - what limits it, and what fixed what (2026-10-01)

Follows `sfm-front-end-drjohnson-2026-09-30.md` (RaCo-ALIKED + LightGlue+ chosen). That front end now runs in the app
(`&features=learned`, Kornia k1024 / k3072 through the hub). This note records the positioning work that followed.

## Headline

**The 44-view DrJohnson subset is beyond any SfM pipeline at this resolution.** COLMAP 4.2.1 (pycolmap) on the same 44
images at 1024x673:

| Pipeline | Registered | Position error vs the published reconstruction (median of camera spread) |
|---|---|---|
| COLMAP SIFT + incremental | 8 / 44 | 1.21% |
| COLMAP SIFT + GLOMAP global | 43 / 44 | 96.8% |
| COLMAP incremental on SpawnScene's k1024 matches | 39 / 44 | 92.8% |
| GLOMAP global on SpawnScene's k1024 matches | 44 / 44 | 98.3% |
| **SpawnScene b84** (k1024, five-point poses, BA) | 37 placed | **29 at 0.71%**, 8 wrong |

The published reconstruction (sparse/0) uses all 263 full-resolution images. Neighbours in the 44 subset are 66-88 deg
apart. The scorer (Umeyama similarity on camera centres, error / median spread) scores sparse/0 against itself at 0.00%.

## Fixes, in the order they mattered (SpawnScene commits)

| Commit | Fix | DrJohnson k1024 |
|---|---|---|
| `39b1094` | Global positioners mapped their solution onto placeholder poses (cascade posed 6/44, 22 of 23 connected cameras shared one copied pose): spread 0, scale 0, every camera on one point | held-out 8.04 -> 15.41 dB |
| `633fb35` | GLOMAP order: rotations, relative-pose filter (10 deg), THEN tracks - tracks from every verified pair carried repeated-structure pairs (25.5 deg median off) | 16.08 dB |
| `4b5a86b` | Relative pose refined on inliers (Sampson LM, 5 DoF) | loop-consistent 49 -> 79, cams 23 -> 33 |
| `1b2771f` | Re-registration through verified pairs (tracks had none for unplaced cameras) | 5 more registered |
| `afe677b` | Calibrated five-point E-RANSAC (Nister) instead of decomposing F | loop-consistent 101, BA 29/37 at 0.71% |
| `9aaace5` | Keep the strongly connected camera core | (b85 pending) |

Relative-pose estimators on DrJohnson's real k1024 matches, true pairs (>= 30 shared COLMAP points):

| Estimator | Median | p75 | < 5 deg |
|---|---|---|---|
| F-RANSAC -> E = K^T F K | 8.25 | 27.9 | 45% |
| F + Sampson LM | 1.30 | 21.8 | 67% |
| 5-point E-RANSAC | 1.94 | 6.62 | 75% |
| 5-point E + LM (shipped) | **0.94** | **3.57** | **78%** |
| 8-point projected onto E | 8.07 | 35.6 | 42% |

F-RANSAC lets planar-degenerate fits win (the scene is walls); the five-point minimal set is what fixes it. An EXACTLY
planar synthetic scene is ambiguous for any two-view method; a wall with 20% off-plane structure is not.

## Why positions still fail (desktop reproduction: `SpawnScene.Tests/DrJohnsonGlobalInitTests`)

Data: `_scratch/djmatch/k1024.bin` (the app's front end, cached through onnxruntime). The test replays the app's global
init in seconds; its diagnosis cases separate the causes:

| Positioning input | Median position error |
|---|---|
| App as is (estimated rotations, real tracks) | 69% |
| COLMAP rotations, real tracks | 64-77% |
| COLMAP rotations, exact (reprojected) observations | **0.06%** |
| COLMAP rotations, exact + 1 px noise | 0.15% |
| COLMAP rotations, real observations minus those > 4 px from COLMAP (6%) | **0.29%** |

So the solver is sound and the rotations are good; ~6% of observations - wrong matches that slide along the epipolar
line (repetitive walls) and therefore pass every two-view check - wreck it. More LM iterations, other Huber scales,
E inliers instead of F inliers, conflict-free tracks: none of these help (conflicts are already 0). Bundle adjustment
from that init converges to a SELF-CONSISTENT wrong layout: RMS 1.1 px, clusters right internally and misplaced against
each other through weak links. Linked by >= 60 shared points, the core is 0.46%; joined through every link, 76%.

Hence `9aaace5`: keep the largest strongly connected core and report the rest as not placed (COLMAP's answer is the
same - separate models). Translation averaging from pairwise directions (2.0 deg median accurate) was tried offline and
did not help with a simple robust BATA either (72%; 3.1% with exact directions) - not pursued.

## Tooling

- Offline harness (scratchpad `djsfm/`): `dj_cache_k1024.py` (front end via onnxruntime), `dj_analyze.py` (pair filters vs
  truth), `dj_relpose.py` (estimators), `dj_colmap.py` / `dj_colmap_sift.py` (pycolmap reference runs).
- pycolmap 4.2.1 installed with `pip install --user` (COLMAP 4 with GLOMAP's global pipeline).
