# Viewer GPU memory per splat, and how to shrink it (2026-10-11)

TJ: "should we be supporting live rendering of the format ... PlayCanvas supports rendering the format which allows
rendering larger scenes because the data is smaller". The gain is not the file format (zstd / gzip are CPU work either
way) but keeping splats QUANTIZED in VRAM and dequantizing in the shader. This is where our viewer stands.

## Per-splat buffers today (read from the code; a scene without an LOD tree)

| Buffer | Bytes / splat | Where |
|---|---|---|
| Source splats, 14 x f32 (pos, colour, scale, opacity, quat) | 56 | GpuGaussianRenderer packed buffer, owned by GpuSplatSorter; Edit tools write it |
| Render vertex buffer (pos f32x3, colour/alpha f16x4, scale f32x3, quat f32x4) | 48 | GpuGaussianRenderer `_splatBuffer` (SplatFormat.PackedBytes) |
| Sort indices + distances (i32 each) | 8 | GpuSplatSorter |
| Radix sort scratch (alternate keys + values, histogram) | ~9 | GpuRadixSort.EnsureCapacity |
| SH bands 1-3: 45 x f32 in 3 parts | 180 (0 at SH 0) | GpuGaussianRenderer `_shRest` |
| **Total** | **~301 at SH 3, ~121 at SH 0** | |

**Measured (AOT build, Chrome, nvidia-smi total with the scene open, nothing else on the GPU; twice, identical):**
Brush Counter 392,017 splats (SH 3) 945 MiB, Brush Garden 969,954 (SH 3) 1,110 MiB, idle 696 MiB -> 165 MiB for 577,937
splats = **299 B / splat**, matching the ~301 read from the code. (The first try had loaded nothing: the importer refused
Brush PLYs - fixed, 9e78783.) Fixed cost of the open viewer ~250 MiB over idle.

## Targets (SOG / SPZ-style: ~65 B/splat)

1. **SH bands as 8-bit** (per-scene scale per band; SPZ stores 8-bit SH with 5 / 4 significant bits): 180 -> 45 B.
   Total 301 -> ~166 B (-45%). The largest single item and contained: the shader dequantizes where it reads SH.
2. **Compact source splats**: pos f32x3 (12), colour + alpha u8x4 (4), scale f16x3 (6), rotation smallest-three (4) =
   26-28 B instead of 56. Edit tools then write the compact form. Total ~138 B (~2.2x less).
3. Render buffer f16 scale / quat (48 -> ~32 B).

## Rules for doing it

- **Fidelity first** (TJ: "fast means nothing if it's ugly"): each step is measured against today's f32 render at fixed
  poses (the viewer bench, tools/_cdp_viewer_bench.js) - PSNR of the quantized render vs the f32 render, and an
  off-path look. 8-bit BLEND targets changed the picture by 24-30 dB (viewer-speed-2026-10-08); quantized STORED
  attributes are a different thing and must be measured, not assumed.
- The trainer keeps f32 (gradients need it); quantization is viewer-side, on upload / load.
- SPZ (8-bit SH, 24-bit positions) could then upload without expanding to f32.
- LOD streaming (pages in a fixed pool) is the other route to huge scenes; compact splats make every page hold more.

## Step 1 measured: 8-bit SH changes the picture by 57 dB (2026-10-11)

`&shq8sim=1` (GpuGaussianRenderer.ShQuantizeSimulate; SphericalHarmonics.WgslPartAccessQ8Sim) rounds every SH band
coefficient to 8 bits (fixed scale 1/128, clamp [-1, 127/128]) where the pack shader reads it - the render the real 8-bit
storage would give, without changing the storage yet. Viewer bench, Inria Train 7K (741,883 splats, SH 3), the 7 training
poses, 1600x900, sorted, CAS off; q8 render vs the f32 render:

| Pose | 0 | 1 | 2 | 3 | 4 | 5 | 6 | Mean |
|---|---|---|---|---|---|---|---|---|
| PSNR (dB) | 57.23 | 56.85 | 56.93 | 56.78 | 57.02 | 57.18 | 57.33 | **57.04** |

Max |diff| 2/255 at every pose, mean 0.12-0.14/255; 31-35% of pixels move by 1-2 levels. For scale: 8-bit blend targets
were 24-30 dB (viewer-speed-2026-10-08), and a 7K-vs-30K training difference is a few dB at ~25 dB - this is far below
anything visible.

Coefficient range on this scene: max |v| 0.40 / 0.32 / 0.29 for bands 1 / 2 / 3 (99.9th percentile 0.18 / 0.15 / 0.14);
nothing near the +-1 clamp. So the fixed 1/128 scale used ~40% of the codes: the real storage uses a **per-band scale =
the band's max |v|** (3 floats per scene, in the renderer's uniforms), ~2.5x finer steps than measured here - 57 dB is the
floor. Next: the real buffer (u8x4-packed, 45 B/splat instead of 180, a draw-only copy - save / export / Edit keep f32),
then the memory measurement (expect ~299 -> ~165 B/splat at SH 3).

## Step 1 built: 8-bit SH codes in VRAM, `&shq8=1` (2026-10-11)

GpuGaussianRenderer.ShCompact (GpuGaussianRenderer.ShCompact.cs): at the first pack after a scene's SH bands arrive,
the f32 parts become 8-bit codes (4 to a u32, 15 a row a part) with one scale per band = the band's largest |v|, found
on the GPU (workgroup + global atomicMax on the bits; never read back); the f32 is freed. The pack pass reads codes
through SphericalHarmonics.WgslPartAccessQ8 (code * m / 127). Every f32 reader (ShRestBuffers, CopyShPartToIlgpu, the
readbacks for save, LOD install) expands to f32 first; the next draw compacts again; readbacks hold compaction off
across their awaits. LOD trees / the streamed pool stay f32.

Train 7K (741,883 splats, SH 3), same harness, nothing else on the GPU:

| | f32 bands | 8-bit codes |
|---|---|---|
| nvidia-smi with the scene open (two runs each, identical) | 1091 MiB | 996 MiB |
| Saved | | **95 MiB = 135 B / splat** (the code's 180 -> 45) |
| Render vs f32, 7 training poses 1600x900 | - | **60.08 dB mean** (59.84-60.48), max 1/255, 16-19% of pixels 1 level |

So the viewer is ~164 B / splat at SH 3 instead of ~299 (1.8x more splats in the same memory). Per-band scale beat
the fixed-step simulation (57.0 dB, max 2/255) as predicted.

**Round trip, exact (sh_step):** the step s = m / 127 is kept to 16 mantissa bits, so code * s is exact in f32 and a
decoded band's largest |v| gives back the same s even through the GPU's inexact division. Checked:
- `&shq8cycles=3` (three extra expand + compact cycles): render pixel-identical to one compaction at all 7 poses; VRAM
  995 vs 996 MiB (no leak). Fidelity unchanged by the exact step (60.08 dB).
- The user path (`AUTOTEST=project`, Bathroom 12 photos, 1500 iterations, 326,379 splats, `&shq8=1`): train ->
  compact at the handoff -> save (the trainer's f32) -> reload (compacts) -> the legacy SH load check (expand, write as
  row-major, reload, compact, expand): **PASS bit for bit**, 14,687,055 floats. Also 3000 iterations / 667K splats:
  saved and reloaded fine (the bit check skips scenes over 500K).

Status: opt-in (`&shq8=1`), not deployed as a default - for TJ: render 60 dB from f32 (max 1/255), viewer -45% memory
at SH 3; the cost is that a LOADED scene that is edited, saved or trained from while compact carries 8-bit SH from then
on (freshly trained scenes save the trainer's f32).

## What is left per splat, and the order (2026-10-11, after 8-bit SH)

At SH 3 with `&shq8=1`: SH 45 + source 56 + render vertex buffer 48 + sort 8 + radix scratch ~9 = ~166 B (measured
~164). The two f32 row buffers are now the bulk.

1. **Render vertex buffer 48 -> 36 B** (pos f32x3, colour/alpha f16x4, scale f16x4, quat f16x4 - instanced vertex
   attributes, so only the pack shader's writes and the two vertex layouts change). History: it WAS f16 until 9943cce
   (2026-10-04, the conic-sign fix), which went to f32 with "measured no change on the reference scene, kept for
   needle-heavy imports" - a precaution, not a measured loss. So the measurement that decides it: f16 vs f32 render
   on a needle-heavy import (gsplat's Truck, the scene that exposed the conic sign) plus Train 7K, viewer bench PSNR /
   max diff, and the near tyre at gsplat's exact camera (&park). Tiny splats: f16's smallest normal is 6.1e-5 (scene
   units) - check the scale distribution of real scenes for values below it (subnormals keep ~1e-7 at reduced
   precision; the EWA filter floors the screen footprint anyway). -12 B/splat.
2. **Source rows 56 -> ~36 B** (pos f32x3, DC + opacity f16x4, scale f16x3 + pad, quat snorm16x4 or f16x4). 36 readers
   in 17 files take `PackedSplatBuffer` as f32 rows and the trainer writes through it, so it would follow the SH pattern:
   compact for display, expand on any access, compact again at the next draw. The per-frame readers that must read the
   compact rows directly: the pack shader and GpuSplatSorter's three cull kernels (ILGPU; positions, plus scale /
   opacity for the cull). -20 B/splat.
3. Dropping the render vertex buffer entirely (draw straight from compact source rows through the sort indices, the SH
   colour evaluated per frame in the vertex stage instead of baked at pack) is the large one (-36..48 B) but changes the
   stochastic path's pack-once design and the cost per frame: measure fps before deciding.

Both 1 and 2 end at ~134 B/splat (2.2x the f32 viewer's 299); 3 at ~100. SOG/SPZ-class ~65 B needs quantized positions
too (SOG: 16-bit per axis + per-scene bounds) - later, with LOD pages as the other route to huge scenes.

## Step 2 (render vertex buffer 48 -> 36 B) measured and built (2026-10-11)

Simulated first (`&vbqsim=1`, pack-shader rounding, since removed): scale through f16, quaternion through SNORM16
(not f16 - 9943cce's needles). Scale check on real scenes: Brush Truck 7K has 14% needles (axis ratio > 100) and 1.0%
of axes below f16's normal range (6.1e-5; subnormal, up to 13% relative - all far below a pixel); our Train 7K 0.03%.
Then built: SplatFormat.PackedBytes 36 (pos f32x3, colour/alpha f16x4, scale f16x4, quat snorm16x4 normalized at pack).

| vs the f32 vertex, viewer bench 1600x900 | mean PSNR | min pose | max diff |
|---|---|---|---|
| Train 7K, 7 poses, 50 deg | 59.68 dB | 59.24 | 2/255 |
| Brush Truck 7K (needles), 8 poses, 50 deg | 60.50 dB | 59.27 | 2/255 |
| Brush Truck 7K, same poses at 15 deg (3.3x closer look) | 60.40 dB | 59.21 | 3/255 (0.0003% of px > 2) |

Real vs simulated: the same PSNR to two decimals (98-263 pixels of ~11M differ by one level). VRAM, Train 7K over idle:
352 -> 344 MiB f32 SH, 257 -> 247 MiB with `&shq8=1` (12 B x 742K = 8.5 MiB). This one is a format change, not an
option (built 2026-10-11, NOT deployed - TJ). Per splat at SH 3 now: ~287 B default, ~152 B with `&shq8=1`.
