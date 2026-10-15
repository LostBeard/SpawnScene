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

## Built (3e53cce .. 5135841)

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

## Evening (2026-10-13): depth start and depth model, on today's defaults (free space 0.3 cover)

**Depth seeds, pine cone own SfM held out** (same 10 photos): agreement mode (default) 16.56 / 0.711 (639K seeds);
uncontradicted `&depthinitmode=free` 16.65 / 0.718 (289K seeds - FEWER: it also drops agreed samples another photo sees
through); stride 2 16.46 / 0.695 (2.55M seeds: denser hurts); free + stride 2 16.57 / 0.709.
Uncontradicted across scenes vs the default: pine cone user path 14.68 -> 14.73 (SSIM +0.008, 130 -> 108 s), Bathroom
18.59 -> 18.61, Kitchen 31.70 -> 31.65, **Bicycle 24.65 -> 24.43** (thin foliage: other photos rightly see through gaps).
Stays opt-in.

**Depth model:** the multi-photo path always ran DAv3 Small - the Depth model setting reached only single photos (fixed
098aca6; the first "Base" run measured Small twice, same 639,154 seeds). DAv3 Base for real (439 MB model storage):
pine cone 16.30 / 0.698 vs Small 16.56 / 0.711 - worse (it also moves the poses). With 384 vs 672 px maps the same, the
depth maps' quality is not the lever on this capture; DA3METRIC-LARGE (a second, mono model for the maps only) parked.

**High preset** (pine cone user path, 15K iterations at 1600 px, `&preset=High`, held out): free space off 13.90 / 0.653
(SSIM ENDS BELOW ITS START 0.664; 62.6% of the weight in front of the held-out photos' depth; 1.32M splats, 772 s) - on
(the default) 14.24 / 0.681, 5.5% in front, 1.03M splats, 603 s. Higher presets grow more floaters on this capture;
free space matters more there. `img/pinecone-high-preset-freespace-pan-2026-10-13.jpg` (left off, right on).

## 2026-10-14: Bathroom photo 30 - the dump and the logged score disagree (CPU-only look, no GPU)

Photo 30 (the HDR window photo) is the whole Bathroom loss for free space (16.56 -> 13.92 logged, hB0 vs hB3). The
`&dumpheld` renders of both arms (`_shots/dataset/Bathroom__tuvok-hB0/hB3__held-IMG_20260223_133612910_HDR-trainer.png`)
scored against the photo's JPEG (resized 3120x4160 -> 768x1024, bilinear; numpy, no GPU):

| arm | logged `final view 30` | dump vs JPEG | window box MSE | window pixels > 0.95 |
|---|---|---|---|---|
| off (hB0) | 16.56 | 16.65 | 0.0494 | 5% |
| on (hB3) | 13.92 | **16.62** | 0.0479 | 26% (photo: 29%) |

The OFF arm's dump agrees with its logged score; the ON arm's does not (2.7 dB). So the logged 13.92 is NOT what the
dumped render looks like against the photo: either the dump (StashTrainerRenderAsync: viewer's packed buffer after the
end carve, `hv.Camera`, SH degree capped to the viewer's) and the score (EvaluateAsync: the trainer's `packed`, `views[i].Camera`,
`ScoreAgainstAsync` vs the training target) see different scenes/cameras for this view, or the target differs from the JPEG.
Checked in code: targets carry no EXIF exposure scaling; both paths call `RenderForwardAsync`. Not yet known which side is
wrong, so no claim about floaters or the window from either number. Visually (dump): off = hazy grey veil over the
window's left part; on = the blue pattern and a white band much nearer the photo's contrast (window brightness 0.815
vs photo 0.726, off 0.773). The "free space costs Bathroom 0.6 dB" row in the table above rests on the logged score.

Next (one 2.5-minute Bathroom 7K run when the GPU is free, not an hour): in the same process log the score of view 30 AND
write the trainer's render, the training target (ReadTargetAsync) and the dump's render for it, then compare the three
pixel for pixel. That settles whether the logged held-out number or the dump is the faithful one before more is built on it.

### Resolved (same day, one Bathroom 7K run, `&dumpheld=1`, free space ON): the in-app PSNR does not clamp the render

New `HELD-EVAL` log (Studio.TrainerRenderDump.DumpHeldEvalFramesAsync): for each held-out view, the trainer's score and a CPU
PSNR of THE SAME render against THE SAME target (frames written as `held-<name>-evalrender.png` / `-target.png`):

| view | trainer score | CPU PSNR, same frames (clamped) |
|---|---|---|
| 0 | 17.04 | 17.04 |
| 8 | 22.10 | 22.10 |
| 23 | 21.69 | 21.83 |
| 30 (HDR window) | **13.88** | **16.52** |

`eval_sse` (SplatTrainerShaders) is `rendered - target` with NO clamp on the render, and the render is the raw SH colour; a
picture, the viewer, and the PNG scorers (gsplat/Brush/`rescore.py`, all 8-bit) clamp it. Only the view whose window blows out
differs (2.64 dB), and free space's brighter window made the ON arm read as the loss. So **the Bathroom -0.6 dB row was
overshoot the picture never shows; view 30 on a clamped score is 16.52, level with off (16.56 logged, no overshoot).** Every
in-app held-out PSNR before this fix is slightly low where a render exceeds 1 (largest on HDR / window captures); the
parity numbers vs Brush/gsplat were scored from PNGs by `rescore.py` and are unaffected. SSIM still scores the unclamped
render (shared with the training loss); not changed.

Correction (10-15): an earlier version of this note said the viewer-side `&dumpheld` render was DC-only (viewer SH degree 0). WRONG: the `viewer SH degree 0` in the HELD-EVAL lines is logged BEFORE the viewer receives the trained bands (`[Train] viewer SH degree 3` comes after). Compared pixel for pixel, the old viewer dump and the new `-evalrender` frame are the same image (mean difference 0.004/255 on all four views), and both score exactly what a clamped CPU PSNR says (16.52 for view 30). So the viewer dump was faithful and the clamp explains the whole 13.88 vs 16.62 gap.
Fix (65cd9d7; gate green: gpu 14.4992 dB = clamped oracle, unclamped 12.3490; red-check with the clamp removed FAILS by name): `eval_sse` clamps; TrainerGate PsnrGateAsync (no gate scored PSNR before: the SSIM
case discards it) writes a hostile render (x1.8 and below-0 stripes) into the output buffer and compares with a clamping
oracle, plus a vacuity check that clamping moves the score > 0.5 dB.
