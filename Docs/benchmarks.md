# SpawnScene benchmarks

How SpawnScene's scenes and viewer compare with other Gaussian splatting tools, measured, with the protocol stated so
anyone can repeat it. Every number here comes from a run whose log is kept; where a comparison is still being measured
it says so.

## Protocol

- **Held-out photos.** Every 8th photo (sorted by name) is never trained on; PSNR and SSIM are measured on those -
  the standard 3DGS / Mip-NeRF 360 split (`--test_every 8` in gsplat, `llffhold=8` in SpawnScene).
- **Same inputs.** Same photos at the same resolution and, unless stated, the same COLMAP camera poses for every tool -
  a comparison of trainers, not of pose estimators.
- **Same iteration count** (7K or 30K, the reference's two checkpoints).
- **Fair exposure score** (SpawnScene logs it for every run): phone photos are exposed one by one, so we also fit each
  held-out photo's per-channel gains on its LEFT half and score its RIGHT half - the reference 3DGS's
  `--train_test_exp` protocol. Without it a scene that learned the photos' average exposure is scored against photos it
  was never meant to match.
- Hardware: NVIDIA RTX 4070 (12 GB), Chrome 151, Windows 11. gsplat on the same machine (CUDA 12.4, PyTorch 2.4.1).

## Training quality

| Scene | Iterations | Metric | SpawnScene (WebGPU, in a browser tab) | gsplat 1.5.3 `default` (CUDA) |
|---|---|---|---|---|
| Tanks and Temples *Truck* (all 251 photos, 979 px) | 30K | PSNR / SSIM | 24.95 dB / **0.887** | **25.13** dB / 0.877 |
| | | splats | **0.89M** | 3.79M |
| | | time | 25 min | - |
| Mip-NeRF 360 *Bicycle* (194 photos, 1237 px) | 7K | PSNR / SSIM | **25.07 dB / 0.766** | 23.14 dB / 0.666 |

Provenance: Truck - SpawnScene run c49 (2026-10-06), gsplat run of the same date and split. Bicycle - SpawnScene run k1
(2026-10-07, COLMAP poses), gsplat 7K from 2026-10-06 at the same resolution and split. **A fresh paired re-run of both
tools with the logs published here is in progress.**

### Without COLMAP

SpawnScene places the cameras itself (learned matching + GPU structure from motion). On Bicycle the held-out score with
its own poses is 25.17 dB / 0.765 (run b0r, 2026-10-08) against 25.07 / 0.766 with COLMAP's - posing in the browser
costs nothing measurable there.

### Phone captures

The benchmarks have no auto exposure and no thin coverage; phone photos of a room have both. A 35-photo capture of a
bathroom (held out every 8th photo):

| SpawnScene version | held out PSNR / SSIM | fair score |
|---|---|---|
| before 2026-10-07 (sparse SfM init, no exposure model) | 15.58 dB / 0.709 | - |
| + seeds from the photos' depth | 16.88 / 0.796 | 18.88 |
| + per-photo exposure gains (current default) | **18.57 / 0.844** | **24.32** |

## Viewer

**In progress:** frame time of SpawnScene's renderer against Spark, PlayCanvas (SuperSplat's viewer), antimatter15/splat
and GaussianSplats3D on the same scene (Inria's *Train*, 741,883 splats, SH degree 3) at the same window size.

### Formats

SpawnScene opens 3DGS `.ply`, SuperSplat compressed `.ply`, PlayCanvas `.sog`, Niantic `.spz` (v2/v3) and `.splat`,
decoding all of them on the GPU. Each decoder is tested against the format owner's reference decoder (ported) and
verified on the owner's published sample files - see [Research/scene-formats-2026-10-08.md](../Research/scene-formats-2026-10-08.md).

## Training in a browser: who else does it

[Brush](https://github.com/ArthurBrussee/brush) (Rust, wgpu) also trains 3DGS in a browser tab. **A paired comparison on
the scenes above is in progress.** Desktop trainers (gsplat, the reference 3DGS) are the quality bar.
