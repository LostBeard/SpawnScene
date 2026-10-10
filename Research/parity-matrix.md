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
| Bicycle (1237, clean folder) | 7K-end | 24.34 / 0.739 (par7k, 10-08 defaults; RESAMPLED to 1236 - flattered, see xp below; exact size 23.59 / 0.716) | - | 23.76 / 0.641 / 0.368, 3.24M (parity7k) | - | see refs |
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

### We UNDERFIT, not overfit (2026-10-09)

Held-out gap vs distance to the nearest training camera (tools/gap_anatomy.py per view + COLMAP centres): the gap is
LARGER for views next to a training camera - Train nearer half -1.49 / farther -1.15, DrJohnson -1.61 / -1.24, Kitchen
(ex1) -1.16 / -0.56. And our score on the TRAINING views is at or below gsplat's on held-out ones: Kitchen supervised 29.35
(gsplat held out 29.37), Train 20.09 (20.42), DrJohnson 29.03 (28.28). The optimiser fits the photos less well at a similar
splat count. Checked equal to gsplat: Adam eps 1e-15 (all three Adam passes), betas, scale / rotation / colour / SH lrs, means
lr x extent. Different: opacity lr 0.025 (gsplat 0.05; tested on Hamamni only) and **the position-lr decay: ours is measured
against a fixed 30,000 (a 7K run ends at 0.34x), gsplat's 7K-end run decays over its own 7,000 to 0.01x** - our centres are
still moving at a third of the start rate when the run ends. `&poslrsteps=N` added; pl chain (after xp): Train, DrJohnson,
Kitchen (+ with &exposure=0), Counter at poslrsteps=7000 vs xp's x0.

### Position-lr decay over the run's own length: +0.6 to +1.1 dB (pl, 2026-10-09)

&poslrsteps=7000 (positions decay 100x over the 7K run, as gsplat's 7K-end does) vs x0 (defaults: decay against a fixed
30K, a 7K run ends at 0.34x), same seeds; build f3 vs f2 (only the option and the up fix differ). Shared scorer
PSNR / SSIM / LPIPS-alex:

| Scene | x0 defaults | pl1 poslrsteps=7000 | delta | gsplat | gap now |
|---|---|---|---|---|---|
| Train | 18.97 / 0.721 / 0.308, 0.60M | **19.54 / 0.769 / 0.249**, 0.47M | +0.57 / +0.047 / -0.059 | 20.42 / 0.771 / 0.231 | -0.88 (was -1.45) |
| DrJohnson | 26.81 / 0.865 / 0.286, 1.44M | **27.92 / 0.893 / 0.236**, 1.00M | +1.11 / +0.028 / -0.050 | 28.28 / 0.889 / 0.230 | -0.36, SSIM ahead |
| Kitchen (seed 2) | 27.10 / 0.887 / 0.146, 0.80M | **28.20 / 0.911 / 0.124**, 0.60M | +1.10 / +0.024 / -0.022 | 29.37 / 0.913 / 0.117 | -1.17 (gains still on) |

The training views fit better too (Train supervised 20.09 -> 20.78, DrJohnson 29.03 -> 31.52): it was the underfitting.
Fewer splats (positions settle, fewer cross the densify bar). The comment on PositionLrMaxSteps ("tying it to the run length
... held-out PSNR fell 1.5 dB", an old 8K measurement) does not hold on today's trainer. **Defaults question for TJ** (with
the exposure gains): decay over the run's own iterations.

| Run | rescore PSNR / SSIM / LPIPS-alex | gsplat | gap |
|---|---|---|---|
| Kitchen seed 2, poslrsteps + gains off (pl2) | **28.66 / 0.908 / 0.124**, 0.60M | 29.37 / 0.913 / 0.117 | -0.71 (defaults: -2.27) |
| Counter, poslrsteps (pl1) | 27.21 / 0.887 / 0.191, 0.48M (x0 27.05 / 0.881 / 0.197) | 27.60 / 0.888 / 0.189 | -0.39; SSIM / LPIPS level |

The two stack on Kitchen (+1.10 then +0.46 = +1.56 over defaults). **cand chain** (after gs): Bathroom (phone, own SfM,
fair score) b0 defaults / b1 poslrsteps / b2 both - the capture the gains were made default for - then both changes on
Playroom (+ its baseline), Train, DrJohnson, Counter, Bonsai, Room, Bicycle, Truck, Garden, Stump: the full table for TJ.

### Brush v0.3.0 (2026-10-09)

Official Windows release (sha256 b68e3e9c...), `brush_app.exe <parity folder> --total-steps 7000 --eval-split-every 8
--eval-every 7000 --eval-save-to-disk` - the same pixels (parity/<scene>/images = the benchmark JPGs), the same split (file
name order, every 8th), position lr decaying over the 7K run (= gsplat's 7K-end). Renders at the photo's exact size, scored by
tools/rescore.py `renders` mode. Same GPU, nothing else running.

| Counter 7K | PSNR / SSIM / LPIPS-alex / vgg | splats | time |
|---|---|---|---|
| Brush v0.3.0 | 27.61 / 0.889 / **0.184** / 0.293 | 0.39M | **172 s wall** (incl. load) |
| gsplat 1.5.3 | 27.60 / 0.888 / 0.189 / 0.295 | 0.88M | 266 s training |
| SpawnScene c1 (both candidate changes) | 27.62 / 0.888 / 0.191 / 0.296 | 0.48M | 462 s training (15.2 it/s) |

Quality: three-way parity on Counter. **Speed: we are the slowest - 2.7x Brush, 1.7x gsplat** (browser WebGPU vs Brush's native
wgpu).

**Three-way, 11 scenes, 7K** (shared scorer, PSNR / SSIM / LPIPS-alex; ours = c1, both candidate changes - NOT today's
defaults; Kitchen ours = seed 1 rb r0). Brush wall time includes loading.

| Scene | Brush v0.3.0 | gsplat 1.5.3 | SpawnScene c1 | best PSNR | Brush splats / wall |
|---|---|---|---|---|---|
| Bicycle | 24.00 / 0.677 / 0.323 | 23.75 / 0.640 / 0.367 | **24.43 / 0.716 / 0.256** | ours | 0.65M / 107 s |
| Garden | 26.14 / 0.804 / 0.166 | 25.99 / 0.809 / 0.149 | **26.36 / 0.827 / 0.131** | ours | 0.97M / 133 s |
| Stump | 25.30 / 0.694 / 0.300 | 25.00 / 0.682 / 0.304 | **25.88 / 0.736 / 0.225** | ours | 0.45M / 96 s |
| Playroom | 29.51 / 0.900 / 0.206 | 29.47 / 0.897 / 0.199 | **29.97 / 0.906 / 0.192** | ours | 0.46M / 103 s |
| Counter | 27.61 / 0.889 / 0.184 | 27.60 / 0.888 / 0.189 | 27.62 / 0.888 / 0.191 | tie | 0.39M / 172 s |
| DrJohnson | 28.22 / 0.896 / 0.226 | **28.28** / 0.889 / 0.230 | 28.12 / 0.894 / 0.234 | gsplat (ours -0.16) | 0.97M / 125 s |
| Room | **30.25** / 0.907 / 0.190 | 30.07 / 0.899 / 0.203 | 30.02 / 0.907 / 0.194 | Brush (ours -0.23) | 0.50M / 158 s |
| Truck | **24.71** / 0.860 / 0.132 | 23.88 / 0.852 / 0.134 | 24.31 / 0.859 / 0.136 | Brush (ours -0.40) | 0.59M / 86 s |
| Kitchen | 29.18 / 0.907 / 0.123 | **29.37** / 0.913 / 0.117 | 28.62 / 0.909 / 0.124 | gsplat (ours -0.75) | 0.40M / 178 s |
| Bonsai | **30.47** / 0.929 / 0.147 | 30.15 / 0.926 / 0.151 | 29.76 / 0.925 / 0.158 | Brush (ours -0.71) | 0.59M / 173 s |
| Train | **21.07** / 0.792 / 0.200 | 20.42 / 0.771 / 0.231 | 19.52 / 0.764 / 0.256 | Brush (ours -1.55) | 0.84M / 118 s |
| **mean PSNR** | **26.95** | 26.73 | 26.78 | | |

Brush is the stronger reference: best mean PSNR, best on 4 scenes, and 1.5-3 min per scene. We lead Bicycle, Garden, Stump
and Playroom on all three metrics. The losses concentrate on Train (-1.55 vs Brush), Kitchen, Bonsai. Brush v0.3.0 also
decays the SCALE lr over the run and grows "MCMC-like"; a source-level comparison is being written
(Research/brush-vs-spawnscene-2026-10-09.md).

### Brush's position lr does not transfer (lr, 2026-10-09)

Research/brush-vs-spawnscene-2026-10-09.md: our position lr starts 17x Brush's on Train / Kitchen. On top of c1:

| Run | Train | Kitchen (seed 1) | Bicycle (control) |
|---|---|---|---|
| c1 (our rate, decay over the run to 1%) | 19.49 / 0.773 | 28.61 / 0.917 | 24.42 / 0.722 |
| Brush's ratio, to 5% (&poslr=0.058 / 0.35 Bicycle, &poslrdecay=0.05) | 19.34 / 0.760 | 28.60 / 0.910 | 24.21 / 0.706 |
| bracket &poslr=0.25 &poslrdecay=0.05 | 19.57 / 0.771 | - | - |

Neutral to worse everywhere: once the decay runs over the run's own length, a 4-17x lower start does not matter. Brush's
Train lead is elsewhere - its growth (no clones, no opacity reset, image-preserving 10% weighted splits) is next.

### No opacity reset (nr, 2026-10-09)

Brush never resets opacity. c1 + subgroups with OPACITYRESET=0 vs sg4 (reset at 3000), same build: Train 19.39 / 0.771 vs
19.50 / 0.773, Kitchen 28.47 / 0.916 vs 28.76 / 0.918, Bicycle 24.34 / 0.720 vs 24.42 / 0.721 - slightly worse on all three.
The reset stays; Brush's no-reset works inside its whole growth scheme (image-preserving splits, prune-and-replace), not as
a piece on its own. Ruled out so far for Brush's Train lead: position lr, near plane, opacity lr, opacity reset.

### Revised opacity on growth (ro, 2026-10-09)

`&revisedopacity=1`: clone (and parent) / split children get opacity 1 - sqrt(1 - a), so a growth step does not brighten the
image (gsplat revised_opacity; Brush grows image-preservingly). On top of c1 + subgroups vs sg4:

| Scene | before | revised opacity | delta |
|---|---|---|---|
| Train | 19.50 / 0.773, 0.46M | **19.82 / 0.779**, 0.42M | **+0.32** (Train's same-settings runs agree within 0.03) |
| Kitchen (seed 1) | 28.76 / 0.918, 0.59M | 28.69 / 0.918, 0.53M | -0.07 |
| Bicycle | 24.42 / 0.721, 1.95M | 24.49 / 0.724, 1.90M | +0.07 |

The first lever that moves Train toward Brush (21.07). Fewer splats. ro2, the other 8 scenes (vs cand c1): Counter 27.67 (+0.04),
Bonsai 29.90 (+0.13), **DrJohnson 28.80 (+0.69)**, Room 30.21 (+0.18), Truck 24.52 (+0.22), Garden 26.41 (+0.06), Stump 26.03
(+0.15), Playroom 30.22 (+0.24) - **better on 10 of 11** (Kitchen -0.07, noise).

**Three-way with all four candidate changes** (&poslrsteps=7000 &exposure=0 &subgroups=1 &revisedopacity=1; shared scorer,
PSNR / SSIM / LPIPS-alex; Brush / gsplat rows from the three-way table above):

| Scene | Brush v0.3.0 | gsplat 1.5.3 | SpawnScene (4 changes) | best PSNR |
|---|---|---|---|---|
| Bicycle | 24.00 / 0.677 / 0.323 | 23.75 / 0.640 / 0.367 | **24.50 / 0.717 / 0.261** | ours |
| Garden | 26.14 / 0.804 / 0.166 | 25.99 / 0.809 / 0.149 | **26.42 / 0.826 / 0.136** | ours |
| Stump | 25.30 / 0.694 / 0.300 | 25.00 / 0.682 / 0.304 | **26.04 / 0.740 / 0.227** | ours |
| Playroom | 29.51 / 0.900 / 0.206 | 29.47 / 0.897 / 0.199 | **30.22 / 0.909 / 0.188** | ours |
| DrJohnson | 28.22 / 0.896 / 0.226 | 28.28 / 0.889 / 0.230 | **28.80 / 0.901 / 0.219** | ours |
| Counter | 27.61 / 0.889 / 0.184 | 27.60 / 0.888 / 0.189 | **27.67** / 0.888 / 0.192 | ours (tie) |
| Room | **30.25** / 0.907 / 0.190 | 30.07 / 0.899 / 0.203 | 30.21 / 0.908 / 0.196 | Brush (-0.04) |
| Truck | **24.71** / 0.860 / 0.132 | 23.88 / 0.852 / 0.134 | 24.53 / 0.862 / 0.134 | Brush (-0.18) |
| Bonsai | **30.47** / 0.929 / 0.147 | 30.15 / 0.926 / 0.151 | 29.90 / 0.926 / 0.158 | Brush (-0.57) |
| Kitchen | 29.18 / 0.907 / 0.123 | **29.37** / 0.913 / 0.117 | 28.69 / 0.910 / 0.124 | gsplat (-0.68) |
| Train | **21.07** / 0.792 / 0.200 | 20.42 / 0.771 / 0.231 | 19.85 / 0.770 / 0.251 | Brush (-1.22) |
| **mean** | 26.95 / 0.841 / 0.200 | 26.73 / 0.833 / 0.207 | **26.98 / 0.851 / 0.190** | |

**Level with Brush on mean PSNR (+0.03), ahead of both on mean SSIM and LPIPS, best PSNR on 6 of 11.** Training time 2.5-4.5 min
a scene (subgroups) vs Brush 1.5-3 min wall. Still behind: Train (-1.22 vs Brush), Kitchen (-0.68 vs gsplat), Bonsai (-0.57).
Bathroom (phone, own SfM) with all four (`&exposure=auto` -> gains ON, 6.13 stops): held out 18.82 / 0.849, fair 24.97 (today's
defaults 18.89 / 24.85) - safe on the capture the gains exist for.

**Where the remaining gaps are (gap_anatomy, all four changes):** Kitchen -0.68 is mostly ONE view - DSCF0824 -11.26 dB, a dark
blob in front of the held-out camera (0.32 of the mean gap; the blob moves between seeds: ex0 had 0720 / 0680). Train -0.57
vs gsplat: 00001 -5.2 (a colour fit recovers most) and 00073 -4.7 (the handrail). Bonsai -0.25 vs gsplat, spread evenly.
`&camerabubble=X` (removes splats within X camera-spreads of a supervised camera at each carve - a camera stands in free
space) added. cb (on top of all four): Kitchen 0.05 -> 29.13 / 0.919 (no catastrophic view, worst -1.49), 0.1 -> 28.81 with a
blob on DSCF0688 (-12.96) although it removed MORE splats (up to 892 a carve); Room 0.1 30.27 (ro 30.21), Bicycle 0.1 24.49
(ro 24.49). The bubble is harmless but does NOT target the blob (the clean 0.05 run = seed luck): the blob is not at a
camera centre. `&blobpick=1` (diagnostic) picks the splats painting held-out views 4+ dB under the median. bp (all four
changes, blobpick): Kitchen seed 1 **28.95**, seed 2 **29.18** (gsplat 29.37) - no catastrophic view in either; the flagged views
are hard for every tool (DSCF0656 23-24, gsplat 24.33) and DSCF0688 (23.7 vs gsplat 27.8) is distant background through the
windows (picked splats 10-23 units deep, in frame of only 32-52 of 244 photos), not a blob. Kitchen with all four changes so
far: 28.69 (blob run), 28.95, 29.18, 29.13 (bubble 0.05) - typically -0.2..-0.4 vs gsplat, with an occasional blob. Seeds 3-5
with blobpick: 29.14 / 28.97 / 29.14 (five seeds: 28.95-29.18, mean 29.08 = -0.29 vs gsplat).

**The blob, identified (bp seed 4, DSCF0688 21.6 dB vs gsplat 27.8):** two dark splats (#516251, #523275; SH DC ~-1.4 = ~0.1
grey) 0.22-0.26 units in front of the held-out camera (just past the 0.2 near plane), 0.07-0.09 spreads from the nearest
supervised camera, **in frame of only 2 of the 244 supervised photos**. Under-constrained: two photos used them to darken a
frame corner and no other view ever checked them. `&minviews=N` (removes splats in frame of fewer than N supervised photos at
every carve) added. mv (minviews=3 on top of all four): Kitchen seed 1 28.88 (28.95 without), seed 4 28.99 (28.97; the DSCF0688
blob GONE, but DSCF0720 fell to 23.6 instead), Train 19.83 (19.82), Bicycle 24.43 (24.49), **Room 29.84 (30.21: -0.37)** - it
removes up to ~5,700 splats a carve, and in a room real surfaces near the cameras are in frame of few photos too. Not a default;
stays opt-in. Removing the blob's splats alone does not lift Kitchen's mean - the trainer then fails another view. The blob is a
symptom of under-constrained regions near the camera path; a fix has to constrain them, not delete them.

### Speed: where our step goes (Counter c1, &trainprofile=1, 2026-10-09)

Final cycle (1 sync per phase, so slower than a real step: 641 s vs 462 s unprofiled): **108 ms/step = backward 47.1 +
scatter 13.0** (56%), prev 6.4, ranges+raster 6.8, emit+count 6.3, loss+ssim 5.3, sort 5.2, clear 4.2, sh 4.0, adam 3.7,
geometry 3.2, loss read 1.1. Brush's whole 7K run averages under 24.6 ms a step. The backward pass is the speed gap: we do a
256-thread tree reduction per key (~9 barriers) plus a separate scatter pass; Brush does one subgroup add per splat and
atomics (Research/brush-vs-spawnscene-2026-10-09.md). Needs the WebGPU `subgroups` feature: **available** on this box
(Chrome 151, NVIDIA Lovelace, subgroup size 32; probed from a 127.0.0.1 page - also timestamp-query, shader-f16). Devices
without it (or with other sizes: Intel iGPU, Quest) keep the current path. Then tighter tile boxes and dropping the
per-step key-count readback.

**Subgroup reduction alone does not help (sg, 2026-10-09, fb19719, &subgroups=1, TrainerGate PASS):** Counter c1 468.6 s
(default 462 s), 27.62 / 0.896 (same); profiled backward 51.1 ms (tree 47.1). The barriers were not the cost. What sits
on every key's critical path in both versions is thread 0's write-out - 9 scattered global writes and two AbsGS
compare-exchange loops - with 255 invocations waiting at the next barrier. sg2: thread 0 only parks each key's totals,
the tile writes a 64-key batch out in parallel after it (one extra barrier per batch): gate PASS, **455.3 s** (vs 462),
27.63 / 0.896, backward 48.7 ms - noise-level. So neither the barriers nor the write-out: the per-key reduction itself
(~55 shuffles per thread per key) for every key on every subgroup. sg3: skip it on subgroups no pixel of which the splat
touches (`subgroupAny(hit)`, gsplat skips warps the same way).

**sg3 = the speed fix: Counter c1 + &subgroups=1 trains in 271.3 s (25.8 it/s) vs 462 s - 1.70x faster**, 27.59 / 0.896
(profiled run 27.63 / 0.896; default 27.63 / 0.896). Profiled step: backward 48.7 -> **19.7 ms**, scatter 13.2 -> 6.0.
TrainerGate PASS (subgroup path x5). Now at gsplat's speed on Counter (266 s training); Brush 172 s wall. The cost was
reducing every key over every subgroup; most subgroups of a 16x16 tile never see a given splat. sg4 (Train, Kitchen, Bicycle)
checks quality and time on more scenes:

| 7K, c1 settings | default backward | &subgroups=1 | speed-up | quality default -> subgroups | gsplat training | Brush wall (incl. load) |
|---|---|---|---|---|---|---|
| Counter | 462 s | **271 s** | 1.70x | 27.63 / 0.896 -> 27.59 / 0.896 | 266 s | 172 s |
| Train | 268.5 s | **155.9 s** | 1.72x | 19.49 / 0.773 -> 19.50 / 0.773 | 146 s | 118 s |
| Kitchen (seed 1) | 441.0 s | **274.0 s** | 1.61x | 28.61 / 0.917 -> 28.76 / 0.918 | 307 s | 178 s |
| Bicycle | 475.7 s | **266.0 s** | 1.79x | 24.42 / 0.722 -> 24.42 / 0.721 | 318 s | 107 s |

Quality unchanged (within noise) on all four; 1.6-1.8x faster. **We now match or beat gsplat's training time** (faster on
Kitchen and Bicycle, within 10 s on Counter / Train) with fewer splats; Brush is still 1.3-2.5x faster on wall time. A third
candidate default for TJ (opt-in until then; devices without 'subgroups' keep the tree path).
Counter: ~13.5 keys per splat (5.67M keys / 419K splats).

### Brush's schedule pieces on top of the new defaults (bs, 2026-10-10)

| Run | Train | Kitchen (seed 1) | Bicycle |
|---|---|---|---|
| new defaults (dv) | 19.81 / 0.779 | 29.06 / 0.918 | 24.51 / 0.724 |
| s1 &shramp=0 (all SH bands from step 0) | 19.93 / 0.781 (+0.12) | 28.96 / 0.918 (-0.10) | 24.57 / 0.725 (+0.06) |
| s2 &scalelr=0.01&scalelrend=0.006 | **20.07 / 0.786 (+0.26)** | **29.59 / 0.921 (+0.53)** | **24.68 / 0.735 (+0.17)** |

The log-scale lr at Brush's 0.01 decaying to 0.006 (ours: a flat 0.005) wins all three - Kitchen now above gsplat (29.37)
and Brush (29.18). SH-from-0 is mixed: dropped. sl (s2 on the rest, vs ro2 = the same settings): Counter 27.80 (+0.13),
Bonsai 29.93 (+0.03), DrJohnson 29.14 (+0.34), Room **30.69** (+0.48), Truck 24.68 (+0.16), Garden 26.63 (+0.22), Stump 26.13
(+0.10), Playroom 30.50 (+0.28); Bathroom held out 19.22 / 0.859 (18.89), fair 25.15 (24.89). **Better on all 12 - made the
DEFAULT 2026-10-10.** In-app mean over the 11 benchmark scenes with all five defaults: **27.26** (Brush 26.95, gsplat 26.73,
shared scorer); best PSNR on 9 of 11 (behind Brush on Train 20.07 vs 21.07 and Bonsai 29.93 vs 30.47; Truck 24.68 vs 24.71).

### Brush's opacity lr 0.01 (ol, 2026-10-10)

On top of the five defaults: Train 20.17 / 0.781 (20.11 / 0.787: neutral), Bonsai **29.02** (29.93: **-0.91**), Bicycle:
`Pack-at-upload complete: 1,743,570 splats PSNR 14.39 -> 24.42 dB, SSIM 0.2749 -> 0.7156` (24.68). Ours (0.025) stays - 0.05 (tr2) and 0.01 both worse.

### Smaller initial splats (is, 2026-10-10)

`&initscale=X` (SparsePointCloudInit.ScaleMultiplier; Brush's start ~0.41x ours), on top of the five defaults:

| Scene | defaults | x0.41 | x0.7 |
|---|---|---|---|
| Train | 20.11 / 0.787 | 20.14 / 0.792 | 20.16 / 0.789 |
| Bonsai | 29.93 / 0.934 | 30.05 / 0.935 | 29.96 / 0.935 |
| Kitchen (seed 1) | 29.59 / 0.921 | 29.46 / 0.922 | 29.56 / 0.921 |

Small and mixed (+0.03..+0.12, Kitchen -0.13 at 0.41): not a default, not Brush's Train edge. Ruled out for Train so far:
position lr (rate and end), near plane, opacity lr (0.01 and 0.05), opacity reset, SH from step 0, initial size. Left:
Brush's growth scheme as a whole (10% weighted-random image-preserving splits, prune-and-replace) - a GpuDensify change.

### Brush growth v1 (&growth=brush, 2026-10-10; Research/brush-growth-port-plan-2026-10-10.md)

Train, on top of the five defaults (20.11 / 0.787, 0.42M): g1 refine every 200, 10% selected, our bar: 19.84 / 0.770, 0.22M;
g2 + bar 2e-4: 19.92 / 0.788, 0.50M; g3 every 100, 20%: 20.03 / 0.780, 0.30M. Not better yet; growth amount matters more than
the selection (Brush ends Train at 0.84M). bg2: g4 every 100, 20%, bar 2e-4: **20.27 / 0.809**, 1.16M, 300 s; g5 40%: 20.35 /
0.813, 1.51M, 404 s; g6 30%, bar 1e-4: 20.37 / 0.820, 2.92M, 683 s. SSIM passes Brush's (0.792) but PSNR saturates near 20.4 -
Brush gets 21.07 with 0.84M in 118 s: its Train lead is NOT how much it grows. D-SSIM weight is the same (0.2). Left: relocation
of pruned splats and low-opacity mean noise (MCMC-like exploration); &mcmc=1 (gsplat MCMC) on the new defaults next.

### Speed after the subgroup default (Counter, new defaults, &trainprofile=1, 2026-10-10)

Final cycle (profiled, one sync per phase): backward 19.3, ranges+raster 6.8, prev 6.3, emit+count 6.3, scatter 5.8, sort 5.4,
loss+ssim 5.0, clear 4.1, sh 4.0, geometry 3.6, adam 3.2, loss read 0.7 = ~70 ms; unprofiled the step is ~39 ms (271 s / 7K),
Brush ~25. Now spread over many passes. Roadmap: (1) a GPU timestamp-query profiler (the `timestamp-query` feature is on
this Chrome) - per-pass GPU time without the syncs that inflate these numbers; (2) drop the per-step key-count readback
(emit+count waits on 4 bytes, a pipeline bubble every step) - indirect dispatch for the sort; (3) clear only the used
prefix of keys / values (whole capacity memset every step); (4) "prev" = between-step CPU work (view pick, target set).

### Train gap follow-ups on top of both changes (tr, 2026-10-09)

| Run | Train | DrJohnson |
|---|---|---|
| c1 (poslrsteps + gains off) | 19.49 / 0.773 | 28.11 / 0.902 |
| tr1 = c1 + &nearplane=0.05 | 19.52 / 0.774 (+0.03, noise) | 28.11 / 0.902 (0) |
| tr2 = c1 + &opacitylr=0.05 | 19.40 / 0.767 (-0.09) | 27.86 / 0.898 (-0.25) |

Neither helps once both changes are in. Next (ab2): the two extras gsplat lacks and that were never ablated on the
benchmark separately - random background (&randombg=0) and floater carve (&carve=0) - on Train and Kitchen (seed 2).

| ab2 (on top of c1) | Train | Kitchen (seed 2) |
|---|---|---|
| c1 | 19.49 / 0.773, 0.46M | 28.65 / 0.916, 0.60M |
| a1 &randombg=0 | 19.58 / 0.774 (+0.09) | **28.98 / 0.919 (+0.33)**, 0.62M |
| a2 &carve=0 | 19.66 / 0.777 (+0.17) | 28.34 / 0.912 (-0.31), 0.63M |

Train's deltas are inside its run noise (~0.2). Kitchen: the carve earns its place (off = -0.31); the random background
costs it 0.33 (above its seed noise, x0 seeds 27.10 / 27.08). The random background is my default (Tuvok proposed and built
it, TJ approved 10-08) for Bathroom's see-through (12.0% -> 5.4% of off-path pixels, held out neutral there); a held-out
benchmark cannot see see-through, so dropping it would be a trade, not a fix. Option for TJ, not a recommendation.

**rb (2026-10-09, TJ: "if you now don't think it is a good default, that is worth knowing"):** random background off vs on,
on top of c1: Counter 27.63 vs 27.63 (0.00), Bonsai 29.81 vs 29.77 (+0.04), Train 19.58 vs 19.49 (+0.09), Kitchen seed 1
28.50 vs 28.61 (**-0.11**), Kitchen seed 2 28.98 vs 28.65 (+0.33). All within noise and the two Kitchen seeds disagree in
sign (off: 28.50 / 28.98 = ~0.5 dB seed spread on Kitchen at this level). **No benchmark cost; it stays a good default**
(Bathroom see-through halved, Hamamni -1.05 dB without it). The ab2 +0.33 was one noisy seed.

### Candidate defaults on the phone capture (cand, 2026-10-09)

| Bathroom (35 phone photos, own SfM) | splats | held out PSNR / SSIM | fair (gains fitted on left half, right half scored) |
|---|---|---|---|
| b0 defaults | 0.76M | 18.89 / 0.848 | 24.85 (w0 last night 24.82) |
| b1 &poslrsteps=7000 | 0.72M | 18.98 / 0.846 | **24.94** |
| b2 &poslrsteps=7000 &exposure=0 | 0.79M | **17.03 / 0.797** | **18.99** |

Position decay over the run: safe here too (+0.09). **Gains off: -5.9 dB fair** - the photos' EXIF exposure spans 6.13 stops
(1/60..1/30 s, ISO 61..~700, HDR merges). So gains cannot simply go; they must follow the capture. `&exposure=auto`
(6ddaeb5): gains only when the photos' EXIF exposure spread >= 1/3 stop (no EXIF = off; the benchmark JPGs have none).
EXIF spread measured: Bathroom 6.13 stops, SouthBuilding 3.06. **au1 (Bathroom, poslrsteps + exposure=auto, build f4): the
gate logged "35 of 35 photos carry EXIF exposure, spread 6.13 stops -> per-photo gains ON"; held out 19.02 / 0.846, fair
24.63** - b1's configuration (18.98 / 24.94), difference = run noise (fair over 4 views).

Playroom (cand c0/c1): defaults 29.84 / 0.920, 0.84M; poslrsteps + gains off **29.98 / 0.923**, 0.58M (gsplat 29.43).

**cand c1 = both changes (&poslrsteps=7000 &exposure=0), in-app held out, vs xp x0 defaults (same seeds, exact sizes):**

| Scene | x0 defaults | c1 both | delta | gsplat (7K-end) | c1 vs gsplat |
|---|---|---|---|---|---|
| Train | 18.93 / 0.730 | 19.49 / 0.773, 0.46M | +0.56 | 20.42 / 0.771 | -0.93 (SSIM level) |
| DrJohnson | 26.80 / 0.872 | 28.11 / 0.902, 1.00M | +1.31 | 28.28 / 0.889 | -0.17 (SSIM +0.013) |
| Counter | 27.05 / 0.890 | 27.63 / 0.896, 0.48M | +0.58 | 27.62 / 0.889 | **+0.01** (SSIM +0.007) |
| Kitchen (seed 2, pl2) | 27.08 / 0.895 | 28.65 / 0.916, 0.60M | +1.57 | 29.41 / 0.914 | -0.76 (SSIM +0.002) |
| Playroom | 29.84 / 0.920 | 29.98 / 0.923, 0.58M | +0.14 | 29.43 | +0.55 |
| Bonsai | 29.23 / 0.931 | 29.77 / 0.932, 0.59M | +0.54 (gains off alone 29.87) | 30.18 | -0.41 |
| Room | 29.77 / 0.911 | 30.03 / 0.915, 0.62M | +0.26 (gains off alone 29.99) | 30.10 | -0.07 |
| Bicycle | 23.59 / 0.716 | 24.42 / 0.722, 1.95M | +0.83 | 23.75 / 0.640 | **+0.67** (SSIM +0.082) |
| Truck | 24.06 / 0.860 | 24.30 / 0.868, 0.70M | +0.24 | 23.87 | **+0.43** |
| Garden | 26.01 / 0.842 | 26.35 / 0.836, 1.50M | +0.34 (gains off alone 26.41 / 0.842) | 26.00 / 0.809 | **+0.35** |


### DrJohnson shown upside down (TJ, 2026-10-09) - fixed a510f48

The cameras' mean up agreed 69.4% (pitched at ceilings and floors), under the 80% gravity gate, so DrJohnson (and Playroom,
55.9%) stayed in COLMAP's frame. Now 50-80% is accepted when the right axes' plane normal agrees within 15 deg and the rig
is not a rolled orbit (TempleRing). Camera-set measurements per scene in the commit. Training is a rigid turn of cameras and
splats together, so scores are unaffected up to float noise; pl's DrJohnson runs (build f3) are the first in the new frame.

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
| DrJohnson | 26.80 / 0.872, 1.44M | 27.07 / 0.873, 1.44M | **+0.27** | above its run noise (0.05); gains fitted 0.93..1.06 there |
| Counter | 27.05 / 0.890, 0.61M | 27.50 / 0.890, 0.61M | **+0.45** | gsplat 27.62: gap -0.57 -> -0.12 |
| Bonsai | 29.23 / 0.931, 0.61M | 29.87 / 0.932, 0.61M | **+0.64** | gsplat 30.18: gap -0.95 -> -0.31 |
| Kitchen (seed 2, defaults) | 27.08 / 0.895, 0.80M | 27.87 / 0.894, 0.79M | **+0.79** | seed 1 (&absgrad=0, ex0/ex1): +1.41; gsplat 29.41 |
| Room | 29.77 / 0.911, 0.83M | 29.99 / 0.911, 0.83M | +0.22 | gsplat 30.10 |
| Bicycle | 23.59 / 0.716, 2.46M | 23.75 / 0.714, 2.42M | +0.16 | rescore 23.59 / 0.710 / 0.251 and 23.76 / 0.708 / 0.253; gsplat rescore 23.75 / 0.640 / 0.367 |

| Truck | 24.06 / 0.860, 0.78M | 24.12 / 0.860, 0.78M | +0.06 | gsplat 23.87; 979 wide = always trained at its exact (odd) size: par7k 24.04, same |

**xp verdict (gains off, same build, exact sizes):** better on 7 of 8 scenes, never worse beyond noise - Kitchen +1.41 (seed 1)
/ +0.79 (seed 2), Bonsai +0.64, Counter +0.45, DrJohnson +0.27, Room +0.22, Bicycle +0.16, Truck +0.06, Train -0.03.
For TJ (defaults): gains were made a default on 10-08 from the Bathroom phone capture (auto exposure, +5.4 dB fair there).
Options: (a) off by default, on for captures that need it; (b) on only when the photos' EXIF exposure / ISO / aperture
vary (the benchmark JPGs carry no EXIF; the Bathroom photo does); (c) gains regularised toward 1. Not changed.

**🔴 The par7k numbers on Bicycle, Garden, Stump, Room, Bonsai and Kitchen were measured on RESAMPLED photos.** Those
photos have an odd side (Bicycle 1237x822, Garden 1297x840, Stump 1245x825, Room 1557x1038, Bonsai 1559x1039, Kitchen
1558x1039); before 8cc8f05 training rounded to even, so the target AND the in-app scoring photo were a smoothed resample,
while gsplat scored against the originals. Resampling alone lifts a score: gsplat's own Bicycle renders 23.75 -> 23.99
(+0.24) when render and photo are both resized to 1236; Kitchen +0.09. So the par7k leads on Bicycle (+0.58), Garden
(+0.57) and Stump (+1.70) are flattered by an unknown part. Native scenes (Counter 1558x1038, Truck, Train, DrJohnson,
Playroom) are unaffected - Counter's baseline is identical (27.05) before and after. At exact size: Bicycle 23.59 (gains
on) / 23.75 (off) vs gsplat 23.75 = level on PSNR, ahead on SSIM (0.710 vs 0.640) and LPIPS (0.251 vs 0.367).
Garden and Stump re-run at exact size queued (gs chain, after od).

**od result (Bicycle, 2026-10-09):** even MAXDIM=1236 (trains 1236x820, resampled) vs exact 1237x822, two seeds each -
seeds agree within 0.06, so the difference is systematic. In-app (each scored against its own training-size photo) even 25.00
/ 24.96 vs exact 23.59 / 23.53; shared scorer resizing the PHOTO to the render 24.40 / 24.35 vs 23.59 / 23.53. Scored the way
gsplat is - our render resized UP to the native photo - even **23.89 / 0.696** vs exact **23.59 / 0.710** (gsplat 23.75 /
0.640): the resampled run is smoother (+0.30 PSNR on Bicycle's grass, -0.014 SSIM), not better. No odd-size bug; exact size
stays (it is gsplat's protocol). The old "+0.58 Bicycle lead" was mostly scoring against a smoothed photo.

**gs (exact size):** Garden x0 (gains on) 26.01 / 0.842, 1.90M; x1 (gains off) **26.41 / 0.842** (+0.40); gsplat 26.00 /
0.809. par7k's 26.57 (resampled to 1296) was flattered: at exact size Garden is level on PSNR with defaults, +0.41 with
gains off, SSIM +0.033 ahead either way. Stump x0 **25.97 / 0.760**, 1.76M; x1 26.16 / 0.761 (+0.19); gsplat 25.03 / 0.682:
a real lead of +0.94 (par7k claimed +1.70 on resampled photos). Gains off: better on 8 of 9 scenes (Train level).

**Bicycle's baseline fell:** par7k defaults 24.34 / 0.739 (1236x822, resampled from 1237) -> x0 23.59 / 0.716 (exact
1237x822 since 8cc8f05), same settings otherwise (targetmb 1536 vs 2560 - both resident). The shared scorer agrees (23.59),
so not a scoring bug; no shader indexes pixels in pairs. Odd size or seed noise: od chain (after pl) = Bicycle at
MAXDIM=1236 (the old even fit) seeds 1 and 2, exact seed 2. If even wins on both seeds, the exact-size change is reverted
(or the odd-size path fixed) before anything else builds on it.
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
