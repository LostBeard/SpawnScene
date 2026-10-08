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
| Bicycle (1237) | 7K | 25.06 / 0.766 (k1, 10-07 defaults) | 24.97 / 0.768 (m0, 10-08 defaults) | 21.29 / 0.552 (bicycle7k, a --max_steps 7000 run; low, unexplained) | - | see refs |
| Garden, Stump, Room, Counter, Kitchen, Bonsai | 7K / 30K | - | - | - | - | see refs |
| Train (979) | 7K / 30K | - | - | - | - | see refs |
| DrJohnson, Playroom | 7K / 30K | - | - | - | - | see refs |

Data: one clean folder per scene (junctions, nothing copied) at the gsplat scratch `gs/parity/<scene>/` = `sparse/0` +
`images/` -> F:/Downloads/mipnerf360/<scene>/images_4 (bicycle, garden, stump) or images_2 (room, counter, kitchen,
bonsai), C:/Users/TJ/Downloads/tandt_db/... images (truck, train, drjohnson, playroom) - the same files SpawnScene's
manifests name. Flowers / treehill are not on disk. Queued 10-08 16:19: gsplat 7K-end on all 11 (after the Hamamni
ablation chain), then SpawnScene GTPOSES on the same folders.

Reference numbers with sources: [parity-references-2026-10-08.md](parity-references-2026-10-08.md) (being written).

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

No single default explains the 2.2 dB: depth init and random background are worth ~1 dB each here; the others move it
by +0.2-0.4. a0 (baseline repeat: noise) and a7 (all six off, nearest gsplat's setup) pending.

## Open gaps (biggest first)
1. Hamamni Baths: -2.2 dB held out against gsplat ON THE SAME CAMERAS (trainer, not capture). Ablating our defaults.
2. No fair 7K reference on any scene yet; only Truck 30K is like for like (-0.18 dB PSNR, +0.010 SSIM, 4.3x fewer splats).
