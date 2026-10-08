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

Reference numbers with sources: [parity-references-2026-10-08.md](parity-references-2026-10-08.md) (being written).

## Phone / Commons captures (no ground-truth poses: own SfM only)

| Capture | SpawnScene | gsplat on SpawnScene's poses | notes |
|---|---|---|---|
| Bathroom (35 phone photos) | held out 19.00 / 0.850, fair 24.82, see-through 5.6% (w0) | - | room |
| Hamamni Baths (59, Samsung S21 FE) | live0: supervised 23.74, "TONS of floating blur blobs ... missing walls" (TJ) | - | depth for 19/57 views; fix b33eb0a, hb1 measuring |

## Viewer

Done 2026-10-08: Docs/benchmarks.md#viewer (SpawnScene 0.55-0.61x GaussianSplats3D uncapped; 60 fps for all at 60 Hz).

## Video

Audit: [video-path-audit-2026-10-08.md](video-path-audit-2026-10-08.md). The code works through the DATASET harness only
(TruckVideo 2026-09-25: 126/126 frames posed, focal 582.3 vs GT 581.9, held out 20.64 / 0.764 at 7K, a 2-week-old
build); the USER path (add a video to a project -> Generate) has never run; `&videoframes` is documented but read
nowhere (always 120 frames); TruckVideo is a slideshow of the Truck photos (no blur / rolling shutter). Three small
changes enable a user-path test (see the audit). Queued after parity (TJ).

## Open gaps (biggest first)
1. Hamamni Baths: missing walls / blobs (depth coverage) - in progress.
2. No fair 7K reference on any scene yet; only Truck 30K is like for like (-0.18 dB PSNR, +0.010 SSIM, 4.3x fewer splats).
