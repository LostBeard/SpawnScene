# Parity matrix (TJ, 2026-10-08: "match or beat them in quality and speed in every area possible")

The tracker. One row per scene and iteration count; one cell per tool. A number goes in only with its run ID and a kept
artifact (log / stats JSON); "-" = not run yet. Protocol: every 8th photo held out (sorted names; gsplat --test_every 8 =
SpawnScene &llffhold=8), same images and resolution (ONE clean image folder per scene; gsplat otherwise makes its own
bicubic images_N_png and Brush caps at 1920 px), COLMAP poses unless the column says own SfM. Two 7K protocols, never mixed
in one row: "7K/30K" = step 7000 of a 30K run (how the 3DGS paper and gsplat publish 7K) and "7K-end" = a run whose
schedule ends at 7K (gsplat --max_steps 7000, SpawnScene 7000 iterations). Every tool's saved renders rescored by one
script (LPIPS/SSIM implementations differ). Held-out PSNR / SSIM / LPIPS; splats; wall time on the RTX 4070.

## Training quality

| Scene (px) | iters | SpawnScene (COLMAP poses) | SpawnScene (own SfM) | gsplat 1.5.3 default | Brush | 3DGS paper |
|---|---|---|---|---|---|---|
| Truck (979) | 30K | 24.95 / 0.887, 0.89M, 25 min (c49) | - | 25.13 / 0.877 / 0.095, 3.79M (val_step29999.json) | - | see refs |
| Truck (979) | 7K | 23.97 / 0.859 (t0, 10-08 defaults) | - | 23.83 / 0.848 / 0.144, 2.51M (truck val_step6999 - step 7K of the 30K run: NOT a 7K run) | - | see refs |
| Bicycle (1237) | 7K-end | 25.06 / 0.766 (k1, 10-07 defaults) | 24.97 / 0.768 (m0, 10-08 defaults) | 21.29 / 0.552 (old bicycle7k on gsplat's own images_4_png: superseded by the clean-folder 23.76 below) | - | see refs |
| Bicycle (1237, clean folder) | 7K-end | **24.34 / 0.739** (par7k, 10-08 defaults; fair 24.89) | - | 23.76 / 0.641 / 0.368, 3.24M (parity7k) | - | see refs |
| Garden (images_4) | 7K-end | 26.57 / 0.855 (par7k) | - | 26.00 / 0.809 / 0.149, 3.62M | - | see refs |
| Stump (images_4) | 7K-end | 26.73 / 0.788 (par7k) | - | 25.03 / 0.682 / 0.304, 3.39M | - | see refs |
| Room (images_2) | 7K-end | 29.97 / 0.918 (par7kfull, 1556x1038) - gsplat +0.13 dB, ours SSIM +0.015 | - | 30.10 / 0.903 / 0.207, 1.03M | - | see refs |
| Counter (images_2) | 7K-end | 27.05 / 0.890 (par7k) - **gsplat +0.57 dB PSNR**, SSIM equal | - | 27.62 / 0.889 / 0.192, 0.88M | - | see refs |
| Kitchen (images_2) | 7K-end | 27.34 / 0.909 (par7kfull, 1558x1040) - **gsplat +2.07 dB** | - | 29.41 / 0.914 / 0.118, 1.20M | - | see refs |
| Bonsai (images_2) | 7K-end | 29.65 / 0.942 (par7kfull, 1560x1038) - gsplat +0.53 dB, ours SSIM +0.014 | - | 30.18 / 0.928 / 0.156, 1.23M | - | see refs |
| Truck (979, clean folder) | 7K-end | 24.04 / 0.860 (par7k) | - | 23.87 / 0.853 / 0.134, 2.06M | - | see refs |
| Train (979) | 7K-end | 18.98 / 0.731 (par7k) - **gsplat +1.43 dB** | - | 20.41 / 0.771 / 0.231, 0.93M | - | see refs |
| DrJohnson | 7K-end | 26.99 / 0.873 (par7k) - **gsplat +1.30 dB** | - | 28.29 / 0.890 / 0.235, 2.10M | - | see refs |
| Playroom | 7K-end | 29.73 / 0.919 (par7k) | - | 29.43 / 0.899 / 0.203, 1.31M | - | see refs |

Data: one clean folder per scene (junctions, nothing copied) at the gsplat scratch `gs/parity/<scene>/` = `sparse/0` +
`images/` -> F:/Downloads/mipnerf360/<scene>/images_4 (bicycle, garden, stump) or images_2 (room, counter, kitchen,
bonsai), C:/Users/TJ/Downloads/tandt_db/... images (truck, train, drjohnson, playroom) - the same files SpawnScene's
manifests name. Flowers / treehill are not on disk. Queued 10-08 16:19: gsplat 7K-end on all 11 (after the Hamamni
ablation chain), then SpawnScene GTPOSES on the same folders.

**Photo streaming verified (2026-10-09, 3a96380):** Truck 7K with a 256 MiB budget kept 125 of 251 views on the GPU and
streamed the rest from browser memory: held out 24.04 / 0.8601 at 29.9 it/s, vs all resident 24.04 / 0.8600 at 28.1 it/s
- identical quality, no slowdown. A photo budget no longer shrinks the training resolution.

**Resolution (2026-10-08):** the par7k chain passed &targetmb=1536 (the Truck protocol's photo budget); Kitchen, Bonsai and
Room did not fit it and trained at 89-94% of the benchmark size - not like for like with gsplat. Counter (full size) is.
Re-runs with a budget that fits queued after the add-back chain (tuvok-par7k-indoor.sh).

**Scoring (2026-10-08):** each tool's numbers above are scored by its own code. One scorer for all: tools/rescore.py
(torchmetrics PSNR / SSIM / LPIPS-alex / LPIPS-vgg, run in the gsplat env, on the CPU while the GPU trains). Checked on
Hamamni: gsplat own 21.41 / 0.732 -> rescored 21.37 / 0.731 / alex 0.467 / vgg 0.541; SpawnScene oc2 own 20.19 / 0.703
-> rescored 20.15 / 0.699 / alex 0.528 / vgg 0.551. Our scorer agrees with torchmetrics to 0.04 dB / 0.004 SSIM, so the
self-scored rows are comparable; LPIPS (new) also favours gsplat on Hamamni. The dataset path now dumps every held-out
render with &dumpheld=1 (it saved only 3 before), for LPIPS on every scene.

Reference numbers with sources: [parity-references-2026-10-08.md](parity-references-2026-10-08.md) (being written).

### First pass, SpawnScene (10-08 defaults, COLMAP poses, 7K-end) vs gsplat 1.5.3 default - PSNR delta (ours - gsplat)

Bicycle +0.58, Garden +0.57, Stump +1.70, Truck +0.17, Playroom +0.30 | Room -0.13, Bonsai -0.53, Counter -0.57 (SSIM
equal or higher on all three), **Train -1.43, DrJohnson -1.30, Kitchen -2.07** (Room / Kitchen / Bonsai re-run at full
size: resolution was not the cause - Kitchen 27.63 at 94% size, 27.34 at full). SSIM higher than gsplat on 7 of 8 like-for-like scenes. We trail on the indoor Mip-NeRF scenes and on Train /
DrJohnson; the Hamamni finding (our splats far more translucent; gsplat never resets opacity) is the lead to test on
them.

### No opacity cap does not transfer (nc, 2026-10-09)

Train: defaults 18.98 / nc1 (cap off) 18.78 / nc2 (cap off + extras off) 19.13; DrJohnson: 26.99 / 26.29 / 25.93 (gsplat
20.41 / 28.29); Counter 27.05 / 27.01 / 27.35 (gsplat 27.62); Bicycle 24.34 / 24.27 / 24.03 (SSIM 0.739 -> 0.699 with the extras off: they earn their place there). Hamamni's best
setting does not close the benchmark losses - moves within ~0.3 dB either way, and the extras help Bicycle.

### Splat counts: we end with 38-71% of gsplat's on every scene

Kitchen 0.73M vs 1.20M (61%), Train 0.60M vs 0.93M (65%), DrJohnson 1.44M vs 2.10M (69%), Counter 0.61M vs 0.88M (69%),
Bicycle 2.30M vs 3.24M (71%), Truck 0.78M vs 2.06M (38%). AbsGS at 8e-4 (+ the carve) was tuned on Bicycle / Truck, where
fewer splats cost nothing; the losses are the detailed indoor / Train / DrJohnson scenes. Queued (tuvok-dg.sh): dg1 the
reference signal (&absgrad=0), dg2 AbsGS at 4e-4, on Kitchen, Train, DrJohnson.

### Densify threshold (dg, 2026-10-09; one scorer = tools/rescore.py on the held-out dumps)

| Scene | run | splats | ours (in-app) | rescore PSNR / SSIM / LPIPS-alex / vgg | gsplat rescore |
|---|---|---|---|---|---|
| Kitchen | default (AbsGS 8e-4) | 0.73M | 27.34 / 0.909 | - (no dumps) | 29.37 / 0.913 / 0.117 / 0.184, 1.20M |
| Kitchen | dg1 &absgrad=0 (2e-4) | 1.33M | 27.89 / 0.919 | 27.76 / 0.906 / 0.128 / 0.194 | |
| Kitchen | dg2 &densifygrad=4e-4 (AbsGS) | 2.05M | 27.64 / 0.910 | 27.54 / 0.897 / 0.137 / 0.196 | |
| Train | default | 0.60M | 18.98 | - | 20.42 / 0.771 / 0.231 / 0.310, 0.93M |
| Train | dg1 &absgrad=0 | 0.86M | 19.01 / 0.728 | 19.04 / 0.719 / 0.305 / 0.360 | |

| Train | dg2 &densifygrad=4e-4 | 1.31M | 18.48 / 0.721 | - | |
| DrJohnson | default | 1.44M | 26.99 | - | 28.28 / 0.889 / 0.230 / 0.363, 2.10M |
| DrJohnson | dg1 &absgrad=0 | 1.91M | 26.61 / 0.868 | 26.61 / 0.861 / 0.283 / 0.399 | |
| DrJohnson | dg2 &densifygrad=4e-4 | 3.02M | 24.60 / 0.841 | - | |

More splats do not close the gap: Kitchen dg2 has 1.7x gsplat's count and scores lower than dg1; Train dg1 +0.03 dB,
dg2 -0.50; DrJohnson dg1 -0.38, dg2 (3.02M) -2.39. On DrJohnson held out falls steadily as splats grow (1.44M 26.99,
1.91M 26.61, 3.02M 24.60) while gsplat reaches 28.28 with 2.10M: our extra splats fit the training photos without
generalising - placement (floaters / wrong depth), not too few splats.

### Kitchen at the exact photo size; near-camera blobs (ex0, 2026-10-09)

ex0 (dg1 settings, exact 1558x1039, same rescore with no resize): 1.41M splats, **27.11 / 0.896 / LPIPS 0.137** vs dg1's
27.76 (scored against a resized photo). Resampling is NOT the 0.65 dB: resizing gsplat's render and photo alike moves it
only 29.37 -> 29.46 (a one-row MISALIGNMENT costs 1.6 dB). Per view ex0 matches dg1 within ~0.3 dB except a few
catastrophic views: **DSCF0720 -15.9, DSCF0680 -12.3, DSCF0688 -7.0 vs gsplat** (dg1: 0688 -9.6, 0680 -3.8, 0720 -4.3)
= large dark smooth blobs right in front of the held-out camera, over the table. They move between runs and cost Kitchen
~0.9 dB of mean alone; gsplat has none. Splats no training view removes: inside the 0.2 near plane of the training
cameras next to them, or outside their frustums. **The biggest single lever found so far.** fl chain (after bm): Kitchen,
two seeds (&shuffleseed) at near plane 0.2 and 0.05, with &splatstats now on the dataset path counting opaque splats
within 0.1 spreads of held-out vs supervised cameras (b-commit after fa71d15).

### Near plane 0.05 vs 0.2 (np, 2026-10-09; dg1 settings, same build)

Train: np0 (0.2) 18.78 / 0.724, np1 (0.05) 19.08 / 0.728 (rescore 18.81 -> 19.10). The worst views do NOT move (00073
handrail -6.35 -> -6.62, 00001 -5.00 -> -5.09, 00049 -4.16 -> -4.25): the +0.3 is spread over ordinary views, at the
noise of one run (np0 vs dg1, same settings, other build: 18.78 vs 19.01). **The near plane is not Train's near-camera
failure.** DrJohnson: np0 26.66 / 0.868, np1 26.86 / 0.869 (rescore gap to gsplat -1.62 -> -1.42); here the near views
DO move: IMG_6392 (ceiling) -4.93 -> -3.14, IMG_6313 -3.20 -> -2.16, 6292 -4.95 -> -4.55. Same-settings DrJohnson runs agree
within 0.05 (dg1 26.61, np0 26.66), so +0.2 is above its noise. Verdict: a small consistent help (+0.3 Train, +0.2
DrJohnson), not the main gap. Candidate default for TJ alongside the exposure question; scale-relative (gsplat's is in a
normalised frame) before it could be one.

### Per-photo exposure gains cost Kitchen 1.4 dB (ex1, 2026-10-09)

| Kitchen (exact size, &absgrad=0, seed 1) | splats | in-app | rescore PSNR / SSIM / LPIPS-alex / vgg | gap to gsplat (raw / after colour fit) |
|---|---|---|---|---|
| ex0 gains ON (default) | 1.41M | 27.10 / 0.905 | 27.11 / 0.896 / 0.137 / 0.201 | -2.26 / -1.39 |
| ex1 &exposure=0 | 1.41M | 28.51 / 0.912 | **28.52 / 0.904 / 0.128 / 0.193** | **-0.85 / -0.85** |
| gsplat | 1.20M | | 29.37 / 0.913 / 0.117 / 0.184 | |

Gains off: +1.41 dB and the colour part of the gap is GONE (the colour fit no longer helps). The worst blob views shrink
too (0720 -15.9 -> -4.9, 0680 -12.3 -> -4.9; one seed each, the blobs move between runs). Gains became a default on 10-08
from the Bathroom phone capture (auto exposure: fair score +5.4 dB there); Mip-NeRF 360 is fixed exposure, and held-out
views get the folded MEAN gain. A defaults question for TJ, not changed: xp chain (after np) = defaults vs &exposure=0
on Train, DrJohnson, Counter, Bonsai, Kitchen (seed 2), Room, Bicycle, Truck, same build. bm/fl dropped (bm conflated
three extras; fl's near-camera stats ride along in xp).

xp results (defaults x0 vs &exposure=0 x1, same build f2, seed 1 unless noted):

| Scene | x0 gains on | x1 gains off | delta | notes |
|---|---|---|---|---|
| Train | 18.93 / 0.730, 0.60M | 18.90 / 0.725, 0.59M | -0.03 | no effect; stats gsplat-like (opacity median 0.20, anisotropy median 6.2, 32% > 10); 34 opaque splats near held-out cameras only |
 **The densify threshold is not the lever; the default (AbsGS 8e-4) stays.**

**Gap anatomy, DrJohnson dg1 vs gsplat:** raw -1.67, after the colour fit -1.53, blur 2 -1.72, blur 6 -1.64: not colour,
large-area. IMG_6392 -5.6 (a barely-seen ceiling: both trainers smear it, ours leaves a black hole where T stays open);
IMG_6292 -4.8: the whole view is softer, the radiator's fins wavy streaks where gsplat's are clean. Not intrinsics (fx/fy
read separately, Studio.Projects K[0]/K[4]; 795.1 / 796.1 at import size); no pose refinement on the dataset path; posLr
matches gsplat's (1.6e-4 x extent 7.196). gsplat's own splats are needles too (DrJohnson 7K: anisotropy median 7.3, p90
32, 40% > 10; opacity median 0.185). &splatstats now logs anisotropy (np runs report ours).
Densify mechanics match gsplat's: split above 0.01 x scene extent (SplatDensityControl.PercentDense; gsplat grow_scale3d
0.01 x scene_scale), children / 1.6; clones dominate (~35K clones to ~1K splits a step past 1K iterations); each apply
costs ~0.5 dB supervised at once (DrJohnson apply probe) and recovers. SH schedule matches (degree +1 per 1000 to 3).

dg1 Kitchen: +0.55 dB from more splats (1.33M, now more than gsplat's 1.20M), still -1.61 dB behind gsplat with MORE
splats - the count is not the whole gap. The rescore resized the photos 1558x1039 -> 1558x1040: training rounded to even
sizes (fixed 8cc8f05: exact photo size, runs from here on), so the rescore is slightly against us (in-app 27.89 scores
against its own 1040-row target).

**Gap anatomy, Kitchen dg1 vs gsplat** (tools/gap_anatomy.py: per held-out view, PSNR raw / after a per-image 3x4 colour
fit / after blurring both): raw -1.61, **after the colour fit -0.80**, blur sigma 2 -2.14, sigma 6 -2.54. So half the gap is
per-image colour (DSCF0864-0896 almost only colour: ours brighter and warmer, gsplat on the photo) and the rest is
LARGE-AREA error, not fine detail (blurring widens it). DSCF0688 alone is -9.6 dB (27.76 vs 18.20): a dark blob in front
of the camera, bottom right - a near-camera floater no training view rejects. We win 0704, 0712, 0808.
Queued (tuvok-ex.sh, after dg): ex0 = dg1 at the exact photo size, ex1 = + &exposure=0 (per-photo gains: gsplat has none),
both with &splatstats=1 (dark opaque splats near cameras).

**Gap anatomy, Train dg1 vs gsplat:** raw -1.37, after the colour fit -1.08, blur 2 -1.31, blur 6 -1.18 - mostly NOT
colour, concentrated in a few views: 00073 -6.4 (a near-camera handrail doubled and smeared; gsplat's crisp), 00001
-5.0, 00049 -4.2, 00065 -3.8. **Near plane:** our trainer culls at 0.2 scene units (3DGS's rasteriser); gsplat at 0.01 in
its normalised frame (~0.05 COLMAP units on these scenes, camera spread ~3.7). Share of the points in front of a camera
nearer than 0.2: DrJohnson mean 4.9%, max 30%; Train 1.0% / 8.5%; Truck 0.5%; Bicycle 0.8%; Kitchen 0.04%, Counter 0.3%
(so not Kitchen's cause). `&nearplane=X` added (4062502); np chain (after ex): Train and DrJohnson, np0 = dg1 settings,
np1 = + nearplane 0.05.

## Phone / Commons captures (no ground-truth poses: own SfM only)

| Capture | SpawnScene | gsplat on SpawnScene's poses | notes |
|---|---|---|---|
| Bathroom (35 phone photos) | held out 19.00 / 0.850, fair 24.82, see-through 5.6% (w0) | - | room |
| Hamamni Baths (59, Samsung S21 FE; 473x1024; 7 held out: 009..057) | hb2: 19.18 / 0.703, 1.54M, ~9 min (7K-end) | **21.41 / 0.732 / LPIPS 0.471, 1.88M, 3 min** (gsplat default, 7K-end, our exported cameras) | gsplat wins 6 of 7 views (033: 27.25 vs 22.16; 009: 23.27 vs 18.77); ours wins 057. TJ live: blobs + missing walls. Depth for 19/57 views fixed (b33eb0a) - not the cause. Ablation of our defaults next |

## Viewer

Done 2026-10-08: Docs/benchmarks.md#viewer (SpawnScene 0.55-0.61x GaussianSplats3D uncapped; 60 fps for all at 60 Hz).

## Video

Audit: [video-path-audit-2026-10-08.md](video-path-audit-2026-10-08.md). The code works through the DATASET harness only
(TruckVideo 2026-09-25: 126/126 frames posed, focal 582.3 vs GT 581.9, held out 20.64 / 0.764 at 7K, a 2-week-old
build); the USER path (add a video to a project -> Generate) has never run; `&videoframes` is documented but read
nowhere (always 120 frames); TruckVideo is a slideshow of the Truck photos (no blur / rolling shutter). Three small
changes enable a user-path test (see the audit). Queued after parity (TJ).

### Hamamni ablation (2026-10-08; 7K, llffhold=8, same 7 held-out photos; one SpawnScene default off per run)

| Run | off | held out PSNR / SSIM | fair |
|---|---|---|---|
| hb2 | - (baseline) | 19.18 / 0.703 | 21.62 |
| a1 | depthinit | 18.20 / 0.675 | 20.36 |
| a2 | projectrefine | 19.31 / 0.711 | 22.01 |
| a3 | absgrad | 19.54 / 0.712 | 22.16 |
| a4 | carve | 19.36 / 0.707 | 21.93 |
| a5 | randombg | 18.13 / 0.698 | 20.83 |
| a6 | exposure | 19.22 / 0.691 | 19.13 |
| gsplat | (its default, our cameras) | 21.41 / 0.732 | - |

| a0 | - (baseline repeat: noise ~0.2 dB) | 19.38 / 0.710 | 21.88 |
| a7 | all six off (nearest gsplat's setup) | 19.53 / 0.701, 1.91M splats | 18.86 |

a7 settles it: with every SpawnScene-only default off, still 1.9 dB under gsplat (21.41) at a similar splat count (1.91M vs
1.88M) - the gap is in the CORE trainer. Per photo it is concentrated: 033 (a close look at the upper wall) 19.31 vs
gsplat 27.25; the others within 0-2 dB. Not the near plane (033's surfaces are 1.8-2.3 units away, the cull is 0.2).
a8 (size cap 10x, i.e. none): 19.21 / 0.706, 033 21.78; a9 (0.5x): 18.52 / 0.700, 033 17.32 - the cap is not it. Photo 033
swings 17.3-22.2 dB across our runs (gsplat 27.25): a region our trainer fits inconsistently. Next: a10 (= a7 + the
trainer's held-out renders) to see it next to gsplat's.

a10 (= a7, all six off, run again): 19.91 / 0.712, 033 23.33 (a7: 19.31 - one photo swinging 4 dB between identical
runs). GT | gsplat | ours (`_shots/hamamni_gt_gsplat_ours.png`): on 033 ours has DARK floaters in front of the upper wall
(a grey-brown smudge mid-left, a brown blob bottom-right) where gsplat shows faint white haze; on the pool (009) the two
match. TJ's "floating blur blobs" = these: dark splats in empty space in front of walls few photos see. Next: why our
densify / prune leaves them and gsplat's does not (splat statistics: scale, opacity, distance to the cameras).

**Splat statistics** (&splatstats=1 / tools/splat_stats.py; sizes and distances in camera spreads):

| | splats | opacity p50 | > 0.5 | size p50 / p99 | within 0.5 spreads of a camera | held out |
|---|---|---|---|---|---|---|
| gsplat | 1.88M | **0.65** | **53%** | 0.016 / 0.064 | 3.6% | 21.41 |
| ours, defaults (a11) | 1.56M | **0.12** | **12%** | 0.014 / 0.066 | 4.9% | 19.27 |
| ours, extras off (a12) | 1.96M | 0.14 | 15% | 0.014 / 0.049 | 5.1% | 19.54 |

Sizes and placement match; OPACITY does not: our scene is mostly translucent splats - haze, blur, see-through, the
"blur blobs". Suspect: the opacity learning rate (on the logit) is 0.025 for us, 0.05 in gsplat; both reset opacity at
3000 / 6000, so at 7K each has 1000 steps to regrow - at half the rate for us.

o1 (defaults + &opacitylr=0.05): opacity median 0.12 -> 0.20, > 0.5 12% -> 31%, but held out WORSE: 18.37 / 0.695 (ours
19.2-19.4). The rate is not the fix, and the opacity is still far from gsplat's.

**Found (Research/opacity-vs-gsplat-2026-10-08.md, verified):** gsplat 1.5.3's DefaultStrategy NEVER resets opacity -
`if step % self.reset_every == 0 & step > 0:` parses as the chained comparison `(step % r == (0 & step)) and
((0 & step) > 0)`, always false (checked in the installed package too). We reset once in a 7K run, at 3000, capping
every splat at 0.01; 48-73% of our final splats are clones made after it from capped parents - a translucent scene.
(So "both reset at 3000/6000" above is wrong for both: gsplat never, we once.) Test: `&opacitycap=0` keeps the schedule
(the size prunes still switch on at 3000) but caps nothing - oc1 (defaults) / oc2 (extras off) running.

oc1 (defaults, no cap): opacity median 0.17, > 0.5 23% (from 0.12 / 12%) but held out WORSE, 18.06 / 0.688 (033: 15.42).
The cap is not the whole story: without it our opacity stays far from gsplat's 0.65, and the reset was also clearing
floaters for us.

**oc2 (all six extras off + no cap): 20.19 / 0.703, photo 033 25.44 - our best Hamamni**, opacity median 0.35, 41% > 0.5
(gsplat 0.65 / 53%). The gap to gsplat is now 1.2 dB (was 2.2). Removing the cap helps without our extras and hurts with
them (oc1 18.06): one of them interacts with an uncapped opacity (and also holds opacity down: oc1 0.17 vs oc2 0.35).
Queued after the 11-scene chain: ab1-ab6 = oc2 with ONE extra back on each, ab7 = oc2 + opacity lr 0.05.

**Add-back (no opacity cap; one extra back on each; base oc2 20.19 / 0.703):**

| run | back on | held out PSNR / SSIM | fair | opacity p50 | splats |
|---|---|---|---|---|---|
| ab1 | depthinit | 19.46 / 0.707 | 19.61 | 0.26 | 2.54M |
| ab2 | projectrefine | 18.43 / 0.651 | 19.97 | 0.31 | 1.80M |
| ab3 | absgrad | 18.57 / 0.675 | 18.15 | 0.17 | 2.13M |
| ab4 | carve | 19.92 / 0.697 | 19.47 | 0.29 | 1.02M |
| ab5 | randombg | 18.52 / 0.687 | 18.66 | 0.39 | 1.94M |
| ab6 | exposure | 18.77 / 0.713 | 21.74 | 0.35 | 2.24M |
| ab7 | opacity lr 0.05 | 19.75 / 0.700 | 19.41 | 0.42 | 2.09M |

On Hamamni with the cap off, no extra beats the plain trainer on held-out PSNR (identical runs differ ~0.4 dB here, so
carve and the opacity rate are within noise; exposure trades raw PSNR for the fair score, as designed). One capture,
though, and these defaults won elsewhere: nc1 / nc2 (cap off alone / cap off + extras off) on Train, DrJohnson, Counter
and Bicycle are queued (_runs/tuvok-nc.sh).

No single default explains the 2.2 dB: depth init and random background are worth ~1 dB each here; the others move it
by +0.2-0.4. a0 (baseline repeat: noise) and a7 (all six off, nearest gsplat's setup) pending.

## Open gaps (biggest first)
1. Hamamni Baths: -2.2 dB held out against gsplat ON THE SAME CAMERAS (trainer, not capture). Ablating our defaults.
2. No fair 7K reference on any scene yet; only Truck 30K is like for like (-0.18 dB PSNR, +0.010 SSIM, 4.3x fewer splats).
