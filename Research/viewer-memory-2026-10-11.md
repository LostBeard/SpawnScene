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
