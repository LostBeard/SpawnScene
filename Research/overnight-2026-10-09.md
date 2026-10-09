# Overnight 2026-10-09: parity vs gsplat - two default changes for TJ

Protocol (Research/parity-matrix.md): COLMAP poses, every 8th photo held out, 7K iterations with the schedule ending at 7K
for both tools, photos at their exact size, in-app held-out PSNR / SSIM (the shared scorer, tools/rescore.py, agrees within
0.04 dB / 0.004 - checked on every scene where it was run). gsplat 1.5.3 default strategy, same folders.

## The decision

Two changes, measured on 10 benchmark scenes and TJ's Bathroom phone capture. **Neither is a default yet.**

1. **Position learning rate decays over the run's own length** (`&poslrsteps=<iterations>`, today a fixed 30,000: a 7K run
   ends at 0.34x its starting rate, gsplat's at 0.01x). We were UNDERFITTING: our score on the training photos was at or
   below gsplat's on photos it never saw, and our gap was largest next to a training camera.
2. **Per-photo exposure gains only when the photos' EXIF exposure varies** (`&exposure=auto`, 6ddaeb5; today gains are always
   on). On fixed-exposure captures gains only add freedom that costs held-out quality; on a phone's auto exposure they are
   essential (Bathroom: 6.1 stops of EXIF spread, gains worth 5.9 dB).

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
