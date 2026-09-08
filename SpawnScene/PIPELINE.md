# SpawnScene Pipeline

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DepthSplat Page                                 │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ ① Load   │    │ ② Upload     │    │ ③ Estimate   │    │ ④ Generate│ │
│  │   Model   │───▶│   Images     │───▶│   Depth      │───▶│  Gaussians│ │
│  └──────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘ │
│                                                                 │       │
└─────────────────────────────────────────────────────────────────┼───────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ ⑤ Viewer     │
                                                          │   (60 FPS)   │
                                                          └──────────────┘
```

## Detailed Pipeline

### ① Load Depth Model  (100% native — zero ORT)
```
Depth Anything V3 Small   (onnx-community/depth-anything-v3-small, ~50 MB)

DepthEstimationService.LoadModelAsync()
    └─▶ DepthEstimationPipeline.CreateFromHubAsync(acc, HubModelStream, repoId)
        SpawnDev.ILGPU.ML native engine — NOT ONNX Runtime.
        Pipeline owns everything:
          • model download via WebTorrent hub + OPFS cache
          • zero-copy stream-to-GPU weight load (never enters the .NET/WASM heap)
          • DAv3 inference on the shared ILGPU WebGPU device (bit-exact vs ORT)
```

### ② Upload Images
```
User uploads photo(s) or loads sample
    └─▶ Decode via createImageBitmap → OffscreenCanvas → RGBA byte[]
        └─▶ ImportedImage { FileName, Width, Height, RgbaPixels }
```

### ③ Estimate Depth  (native DepthEstimationPipeline)

> Super-resolution (upscale-before-depth) was an ONNX Runtime feature and was **retired
> 2026-07-01** in the zero-ORT migration. `SuperResolutionService` is now a parked static
> kernel holder (RGBA↔NCHW) for a future *native* SR pass; `ProjectSettings.UseSuperResolution`
> is parked with it. No SR runs today.

```
ImportedImage
    │
    ▼
DepthEstimationService.EstimateDepthAsync()
    └─▶ DepthEstimationPipeline.EstimateGpuRawAsync(rgba, w, h, w, h)
    ┌────────────────────────────────────────────────────────────┐
    │  All GPU-resident, inside the SpawnDev.ILGPU.ML pipeline:   │
    │    RGBA ──CPU──▶ GPU (once)                                 │
    │    → ImageNet preprocess + resize to model input (518×518) │
    │    → DAv3 inference on the shared ILGPU WebGPU device       │
    │      (native engine, bit-exact vs ONNX Runtime)            │
    │    → GPU-resident predicted depth (inverse depth/disparity) │
    │    │                                                       │
    │    ▼                                                       │
    │  DepthResult {                                             │
    │    RawDepthGpu: MemoryBuffer1D<float>  (GPU-resident)      │
    │    Width, Height, MinDepth, MaxDepth                       │
    │  }                                                         │
    └────────────────────────────────────────────────────────────┘
```

### Depth Preview (side panel)
```
DepthResult.RawDepthGpu
    └─▶ GpuDepthColorizer.ColorizePresentAsync()     [GPU]
        ILGPU Turbo colormap kernel → WebGPUCanvasRenderer
        (no CPU readback — GPU direct to canvas)
```

### ④ Generate Gaussians
```
DepthResult + source image (GpuImage or ImportedImage)
    │
    ▼
DepthToGaussianKernel.GeneratePackedGpuBufferAsync()     [GPU]
    ┌────────────────────────────────────────────────────────────┐
    │  RGBA bytes ──CPU──▶ GPU (packed int[W*H])                │
    │    │                                                       │
    │    ▼                                                       │
    │  ILGPU UnprojectAndPackKernel                              │
    │    GPU depth + GPU RGBA → 10 floats per splat              │
    │    (pos3 + color3 + scale3 + opacity1)                     │
    │    Applies subsample (1/2/4/8) and edge sharpness          │
    │    Atomic compaction — only valid splats in output          │
    │    │                                                       │
    │    ▼                                                       │
    │  GPU-resident packed float buffer                          │
    │  (100K+ splats, never touches CPU)                         │
    │  └─▶ 1 int (valid count) ──GPU──▶ CPU          [4 bytes]  │
    └────────────────────────────────────────────────────────────┘
    │
    ▼
GpuGaussianRenderer.UploadSceneFromGpuBuffer()
    └─▶ Transfers buffer ownership to renderer pipeline
```

### ⑤ Viewer (45-60 FPS render loop)

Two render modes available via `GpuGaussianRenderer.RenderMode`:

#### Stochastic Mode (default) — Sort-Free

```
requestAnimationFrame
    │
    ▼
CameraController.Tick(dt)                    [CPU, per-frame]
    Pointer-lock mouse look + WASD/QE movement
    │
    ▼
RenderService.RenderFrame()
    │
    ▼
GpuGaussianRenderer.RenderStochastic()       [GPU]
    ┌────────────────────────────────────────────────────────────┐
    │  Velocity tracking (no sort, no cull kernel)               │
    │                                                            │
    │  Velocity-adaptive parameters:                             │
    │    • Dilation: scale *= 1 + sqrt(velocity)*5 (max +5%)    │
    │    • Min alpha floor: 0.15 when moving, 0 when still      │
    │    • SPP: 2 moving, 3 convergence burst, 1 converged      │
    │    • Accumulation: reset each frame when moving            │
    │                                                            │
    │  For each sub-sample (1–3 per frame):                      │
    │    ┌──────────────────────────────────────────────────┐    │
    │    │  Pass 1: Stochastic splat render                 │    │
    │    │    → _stochasticTexture (clear each sub-sample)  │    │
    │    │    Billboard quads + EWA anti-alias               │    │
    │    │    Fragment: stochastic discard (u >= alpha)      │    │
    │    │    DepthWriteEnabled=true, no alpha blending      │    │
    │    │    Depth test selects closest surviving sample    │    │
    │    │                                                   │    │
    │    │  Pass 2: Accumulate blend                         │    │
    │    │    → _accumTexture (LoadOp=load, EMA blend)      │    │
    │    │    weight = 1/frameCount (running average)        │    │
    │    └──────────────────────────────────────────────────┘    │
    │                                                            │
    │  Pass 3: CAS display                                       │
    │    → Canvas (sharpening + present)                         │
    │                                                            │
    │  Adaptive resolution:                                      │
    │    Fast camera movement → half-res canvas                  │
    │    Slow/still → full-res canvas                            │
    └────────────────────────────────────────────────────────────┘
```

#### Sorted Mode (legacy) — Traditional Alpha Blending

```
requestAnimationFrame → CameraController.Tick(dt) → RenderService.RenderFrame()
    │
    ▼
GpuSplatSorter.Sort()                       [GPU, async]
    ┌────────────────────────────────────────────────────────────┐
    │  Polls _syncTask.IsCompleted (non-blocking)                │
    │  CullAndDistanceKernel → RadixSort (16-bit or 32-bit)     │
    │  Rate gated: 50ms minimum between submissions              │
    └────────────────────────────────────────────────────────────┘
    │
    ▼
GpuGaussianRenderer.RenderSorted()           [GPU]
    ┌────────────────────────────────────────────────────────────┐
    │  If sortRan: PackComputeShader (float32 → packed vertex)   │
    │  SplatPipeline: billboard quads + EWA + alpha blending     │
    │  Optional CAS post-process                                 │
    │  → Canvas                                                  │
    └────────────────────────────────────────────────────────────┘
```

## Quality Presets

| Preset   | Subsample | Edge Sharpness | Super Resolution   | Sort Mode |
|----------|-----------|----------------|--------------------|-----------|
| Fast     | 4         | Off            | —                  | 16-bit    |
| Standard | 2         | Medium (0.3)   | —                  | 16-bit    |
| High     | 1         | Medium (0.3)   | — (SR retired)     | 32-bit    |

> Super Resolution retired 2026-07-01 (was ORT-only); parked for a future native pass.

## Image Preprocessing Quality

| Stage | Algorithm | Notes |
|-------|-----------|-------|
| RGBA → 518×518 NCHW | Bicubic (Catmull-Rom) | 4×4 sample neighborhood, anti-aliased downscale |
| 518×518 depth → original res | Joint Bilateral Upsampling | 5×5 window, edge-guided by source color image |

## Alternative Entry: PLY/SPLAT File Loading

```
User drops .ply or .splat file on Viewer
    │
    ├─▶ PlyParser.Parse(bytes)   → GaussianScene (CPU Gaussian3D[])
    └─▶ SplatParser.Parse(bytes) → GaussianScene (CPU Gaussian3D[])
        │
        ▼
    SceneManager.ActiveScene = scene
        └─▶ RenderService uploads CPU Gaussians to GPU
            └─▶ Normal render loop (sort + render)
```

## CPU ↔ GPU Transfer Summary

Only these transfers cross the CPU/GPU boundary:

| Transfer | Direction | Size | Reason |
|----------|-----------|------|--------|
| Image RGBA pixels | CPU → GPU | W×H×4 bytes | File I/O (unavoidable) |
| Depth min/max | GPU → CPU | 8 bytes | Scalar metadata for UI |
| Splat count | GPU → CPU | 4 bytes | Compaction counter |
| PLY/SPLAT scene data | CPU → GPU | N×14 floats | File I/O (unavoidable) |

Depth model weights stream JS-side straight to the GPU (WebTorrent hub → OPFS → GPU) and never
enter the .NET/WASM managed heap. Scene save/load is likewise JS-side (OPFS `Uint8Array`/`BlobStream`).
