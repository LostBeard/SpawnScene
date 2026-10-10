# Overnight 2026-10-09: parity vs gsplat and Brush - four default changes for TJ

Protocol (Research/parity-matrix.md): COLMAP poses, every 8th photo held out, 7K iterations with the schedule ending at 7K
for both tools, photos at their exact size, in-app held-out PSNR / SSIM (the shared scorer, tools/rescore.py, agrees within
0.04 dB / 0.004 - checked on every scene where it was run). gsplat 1.5.3 default strategy, same folders.

## DEPLOYED 10-10 10:54 (336762b, run 38060769917): six new defaults live on spawnscene.com

TJ 10-10: "defaults are your decision as you are the one running the tests ... quality and correctness is more important
than speed but speed is important". Shipped: position-lr decay over the run, exposure=auto, subgroup backward, revised
opacity, Brush's scale-lr schedule, tight tile footprints. Verified with no flags locally (Counter 3.03M keys, 239 s,
27.80) and LIVE (Hamamni sample through the hub, Generate 7000: subgroup backward, scaleLr 0.01, gains on - the sample's
photos carry no EXIF - 157.4 s vs 353 s on 10-08, 1.41M splats, DONE).

## Status (10-10 00:35): the four are DEFAULTS in code (13cd973), verified, NOT deployed

TJ (10-09): "the current defaults are the current defaults because you set them. if you now think the defaults should change
and you are sure, then change them." Changed: position lr decay over the run, `&exposure=auto` (no-EXIF photos keep the
gains; dataset path: video sets on, still benchmark sets off), subgroup backward, revised opacity. Verified with NO flags
(build f11): TrainerGate PASS (subgroup path); Kitchen 29.06 / 0.918 (262 s), Train 19.81 / 0.779 (156 s), Bicycle 24.51 /
0.724 (271 s) - gains off by the auto rule; Bathroom 18.89 / 0.849, fair 24.89 - gains on (EXIF 6.13 stops). Every log shows
the subgroup backward. Deploying to spawnscene.com is TJ's call.

**10-10 03:25: a fifth default** - the log-scale lr 0.01 decaying to 0.006 (Brush's; was a flat 0.005): better on all 11
benchmark scenes (+0.03..+0.53) and Bathroom (fair 25.15). In-app mean PSNR with all five: **27.26** vs Brush 26.95 / gsplat
26.73; behind only on Train (-1.0 vs Brush), Bonsai (-0.54) and Truck (-0.03). Verified with no flags (build f13): Train logs scaleLr 0.01 + subgroup backward, 20.11 / 0.787. Cost: ~18% more training time (Counter 266 -> 312 s, Train 156 -> 188 s; larger splats, more tiles).

Not made defaults: &minviews (Room -0.37), &camerabubble (does not hit the blob), &nearplane (+0.2-0.3, not scale-relative
yet), no opacity reset (worse), Brush's position lr (neutral to worse), opacity lr 0.05 (worse).

## The decision (as proposed)

Four changes (three quality, one speed), measured on the benchmark scenes and TJ's Bathroom phone capture. **None is a default yet.**

1. **Position learning rate decays over the run's own length** (`&poslrsteps=<iterations>`, today a fixed 30,000: a 7K run
   ends at 0.34x its starting rate, gsplat's at 0.01x). We were UNDERFITTING: our score on the training photos was at or
   below gsplat's on photos it never saw, and our gap was largest next to a training camera.
2. **Per-photo exposure gains only when the photos' EXIF exposure varies** (`&exposure=auto`, 6ddaeb5; today gains are always
   on). On fixed-exposure captures gains only add freedom that costs held-out quality; on a phone's auto exposure they are
   essential (Bathroom: 6.1 stops of EXIF spread, gains worth 5.9 dB).

3. **Subgroup backward** (`&subgroups=1`, added 18:40): the backward pass reduces each key's gradients with WebGPU subgroup
   operations and skips subgroups the splat does not touch. **1.6-1.8x faster training with unchanged quality** on Counter
   (462 -> 271 s), Train (268 -> 156 s), Kitchen (441 -> 274 s), Bicycle (476 -> 266 s) - now as fast as gsplat (CUDA) or
   faster; Brush still 1.3-2.5x faster. TrainerGate PASS. Devices without the `subgroups` feature keep today's path.

4. **Revised opacity on growth** (`&revisedopacity=1`, added 21:00): a clone and its parent, and both split children, get
   opacity 1 - sqrt(1 - a), so a growth step does not brighten the image (gsplat's revised_opacity). Better on 10 of 11
   scenes (DrJohnson +0.69, Train +0.32, Playroom +0.24, Truck +0.22, ...; Kitchen -0.07), fewer splats.

**With all four: mean PSNR 26.98 vs Brush 26.95 and gsplat 26.73; SSIM 0.851 vs 0.841 / 0.833; LPIPS 0.190 vs 0.200 / 0.207;
best PSNR on 6 of 11 scenes** (table in Research/parity-matrix.md). Still behind on Train, Kitchen, Bonsai.

## Results (held out PSNR / SSIM; x0 = today's defaults, c1 = both changes with gains off = what auto does on these JPGs)

| Scene | gsplat | today (x0) | gains off | **both (c1)** | c1 vs gsplat |
|---|---|---|---|---|---|
| Bicycle | 23.75 / 0.640 | 23.59 / 0.716 | 23.75 / 0.714 | **24.42 / 0.722** | **+0.67** |
| Garden | 26.00 / 0.809 | 26.01 / 0.842 | 26.41 / 0.842 | 26.35 / 0.836 | **+0.35** |
| Stump | 25.03 / 0.682 | 25.97 / 0.760 | 26.16 / 0.761 | 25.88 / 0.744 | **+0.85** |
| Room | 30.10 / 0.903 | 29.77 / 0.911 | 29.99 / 0.911 | 30.03 / 0.915 | -0.07 |
| Counter | 27.62 / 0.889 | 27.05 / 0.890 | 27.50 / 0.890 | 27.63 / 0.896 | **+0.01** |
| Kitchen (seed 2) | 29.41 / 0.914 | 27.08 / 0.895 | 27.87 / 0.894 | 28.65 / 0.916 | -0.76 |
| Bonsai | 30.18 | 29.23 / 0.931 | 29.87 / 0.932 | 29.77 / 0.932 | -0.41 |
| Truck | 23.87 | 24.06 / 0.860 | 24.12 / 0.860 | 24.30 / 0.868 | **+0.43** |
| Train | 20.41 / 0.771 | 18.93 / 0.730 | 18.90 / 0.725 | 19.49 / 0.773 | -0.92 |
| DrJohnson | 28.29 / 0.889 | 26.80 / 0.872 | 27.07 / 0.873 | 28.11 / 0.902 | -0.18 |
| Playroom | 29.43 | 29.84 / 0.920 | - | 29.98 / 0.923 | **+0.55** |

Today: ahead of gsplat on PSNR on 3 of 11 (Stump, Truck, Playroom), level on Garden. With both changes: **6 ahead or level on PSNR**
(Bicycle, Garden, Stump, Counter, Truck, Playroom), Room and DrJohnson within 0.2, SSIM ahead or level on every scene
measured with SSIM. Changes vs today: +0.14 to +1.57 dB on every scene except Stump (-0.09 / SSIM -0.016).
Splat counts drop 3% (Bonsai) to 31% (DrJohnson, Playroom).

The position decay's own effect (c1 minus gains off): DrJohnson +1.04, Kitchen +0.78, Bicycle +0.67, Train +0.59, Truck
+0.18, Counter +0.13, Room +0.04, Garden -0.06, Bonsai -0.10, **Stump -0.28**.

Bathroom (phone, own SfM; fair = gains fitted on the left half of each held-out photo, right half scored):

| Bathroom | held out | fair |
|---|---|---|
| today | 18.89 / 0.848 | 24.85 |
| position decay | 18.98 / 0.846 | 24.94 |
| position decay + gains off | 17.03 / 0.797 | **18.99** |
| position decay + &exposure=auto | **19.02 / 0.846** | **24.63** - gains ON ("35 of 35 photos carry EXIF exposure, spread 6.13 stops"); the same configuration as the row above, so the difference is run noise (fair = 4 views) |
| **all four changes** (+ subgroups + revised opacity) | **18.82 / 0.849** | **24.97** - gains ON from EXIF; within noise of today's defaults (24.85): safe on the phone capture |

## Brush v0.3.0 (added 15:30)

The third reference, same pixels / split / scorer, 11 scenes at 7K: **mean PSNR Brush 26.95, SpawnScene (both changes)
26.78, gsplat 26.73.** We lead Bicycle, Garden, Stump, Playroom on every metric; tie Counter; trail Brush most on Train
(-1.55), Bonsai (-0.71), Kitchen (-0.46; gsplat leads Kitchen). Brush takes 1.5-3 min a scene; we take 2.7x Brush on Counter
(462 s training vs 172 s wall). Table in Research/parity-matrix.md.

## Corrections made overnight (factual stats)

- **The par7k leads on Bicycle (+0.58), Garden (+0.57) and Stump (+1.70) were inflated.** Photos with an odd side were
  trained and scored resampled to an even size while gsplat was scored on the originals; resampling alone lifts gsplat's own
  Bicycle renders +0.24. Fixed 8cc8f05 (exact photo size). Measured at exact size, today's defaults are level with gsplat on
  Bicycle and Garden and +0.94 on Stump.
- **DrJohnson and Playroom were shown upside down** (TJ): the gravity gate needed 80% camera-up agreement (they have 69%
  / 56%). Fixed a510f48, tests on the real cameras; TempleRing stays refused.

## Ruled out (measured)

Densify threshold / signal (more splats generalised WORSE on DrJohnson: 1.44M 26.99, 3.02M 24.60); densify mechanics,
SH schedule, Adam epsilon / betas, scale / rotation / colour learning rates (all match gsplat); intrinsics (fx/fy kept
separately); odd image sizes (Bicycle even vs exact = smoother, not better); near plane 0.05 (+0.2 to +0.3, not the main gap).

## Still open

- Train (-0.92) and Kitchen (-0.76). Tried on top of both changes: near plane 0.05 (Train +0.03, DrJohnson 0), gsplat's
  opacity lr 0.05 (-0.09 / -0.25), carve off (Train +0.17 noise, Kitchen -0.31), random background off (Train +0.09 noise,
  Kitchen seed 2 +0.33). The random background is my default (I proposed it, TJ approved): checked on 5 runs (rb) - off
  vs on 0.00 / +0.04 / +0.09 / -0.11 / +0.33, within noise, Kitchen's two seeds disagree in sign. No benchmark cost; it
  stays (Bathroom see-through halved, Hamamni -1.05 dB without it).
- Kitchen's held-out views lose 7-16 dB to dark blobs in front of the camera on some seeds (smaller with gains off).
- Stump: the position decay costs 0.28 there.
- 30K comparisons, Brush, the video path (after parity).
