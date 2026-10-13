# Floaters vs the photos' depth maps (2026-10-13)

TJ, live site, pine cone sample (Standard, then Max): "soooo many blurry floaties all over the room" the moment the camera
moves; the old single-photo DA2 path (14M splats from one 5K photo, every splat ON the depth surface) had none. Question:
are the depth maps used at all?

## What the code did (before today)

- Multi-photo: depth seeds the start (DepthFusionInit, `depthinit=4`); training is photometric only.
- `&depthloss` (L1 on the blended MEAN inverse depth, decaying 1 -> 0.01) existed but was off: "neutral" on Bathroom held
  out - photos that sit beside training photos, where a floater lines up with what is behind it.
- The floater carve measured each splat against the render's OWN surface (where T crosses 0.5): an opaque floater IS that
  surface and reads ~0.5 (TrainerGate: 0.4922), so the 0.9 carve kept it.

## Built (3e53cce, + depth maps at 672 px)

- Census against each photo's depth map (`&depthcheck=1` measure, `&depthcarve=1` carve) + agreement bins.
- Free-space loss `&freespace=X`: per pixel the opacity in front of (1 - 0.15) x the photo's depth, 1 - T_f.
- Depth maps: `&depthtargetpx` (default 384; 672 = the multi-view prediction size measured the same, below).
- TrainerGate PASS: opaque floater 1.0 vs photo depth (0.49 vs the render's surface); free-space gradient vs finite
  differences within 2.5%; red checks.

## Measured - pine cone, 7K, 1024 px

**User path (dA-dD, no held-out photos):** 39% of all blending weight in front of the photos' depth (>15%); 60% of pixels
render a surface nearer than the photo's depth. Depth carve 39% -> 37% (weak: few splats are in front in >= 90% of their
weight). Free space: pixels nearer than half the photo's depth 15.8% -> 0.0%.

**Supervised PSNR rewards floaters.** Same dataset path, poses only swapped: our SfM 21.3 sup / 15.71 held out; COLMAP
poses (depth-unprojected start) 31.4 sup / **12.4** held out. Judge held out (and off the photo path), never supervised.

**Not the cause:** our focal (BA ends at 803 vs COLMAP 784; 1007 was the start value), lens distortion (COLMAP
SIMPLE_RADIAL k1 -0.089 vs SIMPLE_PINHOLE: 23.21 vs 23.62 sup, 15.73 vs 15.65 held out), our poses (held out 15.71 ours
vs 15.65 COLMAP). Brush v0.3.0 on COLMAP's undistorted photos: 16.23 / 0.697 held out (12 views) - the same radial smear.
The capture itself: close-ups with the room out of focus, a mirror-glossy table, a person in one shot.

**Held out, own SfM, dataset path, same 10 photos, shared scorer (tools/rescore.py):**

| run | flags | PSNR | SSIM | LPIPS-alex | weight in front of photo depth | train |
|---|---|---|---|---|---|---|
| sA | depthcheck | 15.64 | 0.657 | 0.559 | 39.3% | 152 s |
| sDL | + depthloss=1 | 16.26 | 0.680 | 0.520 | - | 169 s |
| sF1 | + freespace=1 | 16.31 | 0.686 | 0.519 | 0.39% | 123 s |
| **sF03** | **+ freespace=0.3** | **16.49** | **0.694** | **0.501** | - | 128 s |

Off the photo path (Wander pan views = turning around in the room): `img/pinecone-freespace-pan-2026-10-13.jpg` (left
sA, right sF03) - walls flat, frames / door / armchairs / floor readable, the fog mostly gone.

**Round 2 (same 10 held-out photos):** weight 0.3 at 384 px 16.54 / 0.7107, at 672 px 16.55 / 0.7109 (100 vs 33 MB: kept
size is not the limit); weight 0.1 16.47 / 0.7062 (5.5% in front); 0.3 + depthloss=1 16.49 / 0.7093 (adds nothing);
cover mode (`&freespacecover`, push scaled by the opacity behind) 16.51 / 0.7098.

## Validation for a default (llffhold=8 held out, `&freespace=0.3` vs off)

| scene | path | base | free space 0.3 | note |
|---|---|---|---|---|
| Pine cone | own SfM, dataset | 15.64 / 0.657 | **16.49 / 0.694** | LPIPS 0.559 -> 0.501; room readable off-path |
| Pine cone | user path | 14.53 / 0.646 | **14.68 / 0.677** | right half 14.88 -> 15.30 |
| Kitchen360 | own SfM, dataset | 31.71 / 0.929 | 31.53 / 0.930 | 2.6x faster (353 -> 137 s); off-path fog gone (`img/kitchen-freespace-out-2026-10-13.jpg`), one see-through gap |
| Bathroom | user path | 19.21 / 0.859 | 18.64 / 0.855 | ALL on held-out photo 30 (HDR): 16.45 -> 13.81 (reproduced 16.51 -> 14.01); the other three up; cover mode 14.14 |
| Bicycle360 | own SfM, dataset | (baseline run hung at boot) | 24.63 / 0.728 | 2.8M splats, 484 s |

Not a default yet: the Bathroom photo-30 loss is unexplained (cover mode did not fix it, so not the see-through hole
alone). Next batch (tuvok-y.sh): held-out renders on the user path (`&dumpheld`, 0a01889), per-photo held-out pose
refinement in the log (5135841), and the HELD-OUT photo depth measure (53c9a28) - the held-out photos' own depth maps,
which never reach training (TJ's idea; the supervised measure partly grades its own homework once free space trains
against those maps).
