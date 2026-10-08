# Viewer speed against other web viewers (2026-10-08)

TJ asked for factual side-by-side comparisons with other generators and viewers. Docs/benchmarks.md#viewer has the
public table; this is the working record behind it.

## Setup

Inria *Train* 7K (741,883 splats, SH 3), 1600x900, DPR 1, vertical FOV 50°, seven seeded poses
(`tools/viewer-bench/make_poses.py`), RTX 4070, Chrome 151, `SPAWNSCENE_CHROME_UNCAPPED=1`, 4 s warm-up, 8 s measured.
Spark 2.3.1, PlayCanvas 2.23.1, GaussianSplats3D 0.4.7 (SH degree set to 3), SpawnScene at 3709283+.
Driver: `tools/_cdp_viewer_bench.js` (`BENCH_TAG`, `BENCH_EXTRA` for SpawnScene URL options). Shots and JSON:
`_shots/viewer_bench/{r1,r2,c1,aot,aot2}` (not committed).

## Results

- At 60 Hz all four hold 60 fps at every pose.
- Uncapped, SpawnScene 221-310 fps; GaussianSplats3D 392-559; SpawnScene is 0.55-0.61x of GaussianSplats3D at every
  pose, so the gap scales with the work drawn (not a fixed overhead - an earlier draft of the docs said fixed; corrected
  in bfcee15).
- SpawnScene and GaussianSplats3D repeat within 1%. Spark and PlayCanvas swing up to 2.8x between rounds at an
  unchanged camera. Uncapped, Spark's worker sort falls behind its draw loop: at poses 4-6 in some rounds it drew with
  a stale order (far side of the train over the near side, mirrored lettering). At 60 Hz the same poses draw correctly
  (mean abs diff to GaussianSplats3D 3.9-6.7 vs 46-52 when stale). Gliding the camera between poses did not change it.
- The interpreted (`SpawnSceneAot=false`) and AOT builds time the same within 1%: GPU-bound.

## Where SpawnScene's frame time goes (harness-only `&viewexp=`, AOT build)

| Variant | vs shipped | vs GaussianSplats3D | image vs shipped |
|---|---|---|---|
| shipped (rgba16float target, depth attachment, axis-aligned quads) | 1.00 | 0.55-0.61 | - |
| `nodepth` (no depth attachment) | 1.00-1.01 | | identical |
| `obb` (ellipse-aligned quads, tile box tested in the fragment shader) | 0.99-1.01 | | identical |
| `10bit` (rgb10a2unorm target) | 1.01-1.04 | | 24.4-30.4 dB |
| `obb,10bit` | 1.08-1.12 | 0.61-0.66 | 24.4-30.4 dB |
| `8bit` (rgba8unorm target) | 1.37-1.42 | | 24.2-30.3 dB |
| `obb,8bit` | 1.29-1.32 | 0.73-0.78 | 24.2-30.3 dB |

- `tools/viewer-bench/quad_area.py` estimated that axis-aligned quads rasterise 1.20-1.26x the pixels of
  ellipse-aligned ones, and 1.14-1.21x GaussianSplats3D's (2.83-sigma ellipse quads). The obb variant is
  pixel-identical but no faster: pixel count is not the limit.
- **The blend target is.** An 8-bit target is 1.37-1.42x faster, but it changes the picture by up to 120/255 in places
  (24-30 dB) - far more than rounding, consistent with unorm clamping to [0,1] after every blend while the trainer
  composites unclamped in f32. 2026-09-24 measured that 8-bit blending cost the viewer 0.7-3.6 dB against the trainer's
  own render (Truck), which is why the target is rgba16float. 10-bit gets the same picture change for almost no speed.
- Which viewers agree (mean PSNR over the 7 poses, SpawnScene with CAS off - CAS moves these by < 0.05 dB):
  SpawnScene vs PlayCanvas 31.2 dB; SpawnScene `8bit` vs Spark 31.7, vs GaussianSplats3D 30.2; GaussianSplats3D vs
  Spark 30.1; SpawnScene vs GaussianSplats3D 23.8, vs Spark 24.4. Why PlayCanvas lands with the fp16 render is not
  established (it draws to the canvas; its colour path was not examined).

## Decision

Defaults unchanged: the fp16 target is the fidelity-to-the-trainer choice and stays. The `&viewexp` variants stay as
harness flags. Open: a faster fp16 path (the blend is bandwidth: 8 bytes a pixel read+write per fragment; options are
fewer fragments where it matters - the obb result says the GPU is not fragment-count-bound here - or a compute
rasteriser like the trainer's tile renderer), and a check of GaussianSplats3D/Spark against the reference renderer at
these poses (needs the gsplat environment back).

## Follow-up: a faster path that keeps the fp16 picture (TJ: "if it would be an improvement it is worth looking into.
## visual fidelity is very important. fast means nothing if it's ugly.")

- **The trainer's tile rasteriser as the viewer:** no. Its forward pass (Truck 549K, 979x546, 09-24 profile) is
  ~23 ms a frame - emit 6.5, sort 10.5 (tile x depth keys, millions of them), raster 6.0 - against ~4 ms for the
  current viewer at 2.7x the pixels. A compute viewer would first need a sort several times faster than that one.
- **Early termination in the hardware path:** draw front to back in K batches, marking opaque pixels (T < 1e-4, the
  trainer's and the reference's stop) in the depth buffer between batches so later fragments die at the early depth
  test. It would match the trainer MORE closely (the viewer now blends splats behind opaque pixels; the trainer does
  not). `tools/viewer-bench/early_stop_sim.py` (1/4 resolution, the viewer's footprint rule) on the seven poses:
  fragments still blended - ideal per-pixel stop 61-69%, K=8 72-80%, K=16 68-77%, K=32 65-74%. Train 7K is hazy (few
  pixels reach T < 1e-4 early), so the ceiling is ~1.3-1.5x on blending, less after K extra full-screen passes.
- **Where speed matters is not the RTX 4070** (already 3.7-5x over 60 Hz at 1600x900). It is the Quest browser and
  laptop GPUs - and on a tile-based mobile GPU each pass break stores and reloads the tile memory, so K batches could
  cost more than they save there. Next step: measure the headset (and an integrated GPU) at these poses before
  building either.
