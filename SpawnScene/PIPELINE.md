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

### ① Load Depth Model
```
User selects model
    │
    ├─▶ DistillAnyDepth Small (default)     ~99 MB ONNX
    └─▶ DepthAnythingV2 Small               ~99 MB ONNX

OnnxRuntime.Init()
    └─▶ CreateInferenceSession (WebGPU EP, shared GPUDevice with ILGPU)
```

### ② Upload Images
```
User uploads photo(s) or loads sample
    └─▶ Decode via createImageBitmap → OffscreenCanvas → RGBA byte[]
        └─▶ ImportedImage { FileName, Width, Height, RgbaPixels }
```

### ③ Estimate Depth

Super-resolution is optionally applied **before** depth estimation to increase the input image resolution. It is only active when:
- Quality preset = "High"
- SR model has been manually loaded by the user

```
ImportedImage
    │
    ├─▶ [If High preset + SR model loaded]
    │       SuperResolutionService.UpscaleAsync()
    │       ┌────────────────────────────────────────────────────┐
    │       │  RGBA bytes ──CPU──▶ GPU (packed int[W*H])        │
    │       │    └─▶ ILGPU RgbaToNchwKernel  → float[1,3,H,W]  │
    │       │    └─▶ ORT SR inference (WebGPU, zero-copy)       │
    │       │    └─▶ Output float[1,3,H*2,W*2]                  │
    │       │    └─▶ ILGPU NchwToRgbaKernel  → GpuImage         │
    │       │         (packed RGBA stays GPU-resident)           │
    │       └────────────────────────────────────────────────────┘
    │       Models: sr_x2.onnx (~33 MB) or sr_x4.onnx (~33 MB)
    │       GpuImage.PackedRgba passed directly to depth + Gaussian stages
    │
    └─▶ [Otherwise] original ImportedImage uploaded to GPU once
            │
            ▼
    DepthEstimationService.EstimateDepthAsync()
    ┌────────────────────────────────────────────────────────────┐
    │  [CPU path] RGBA bytes ──CPU──▶ GPU (packed int[W*H])     │
    │  [SR path]  GpuImage.PackedRgba already on GPU (no upload)│
    │    │                                                       │
    │    ▼                                                       │
    │  ILGPU PreprocessRgbaKernel                    [GPU]       │
    │    Bicubic (Catmull-Rom) resize to 518×518                 │
    │    RGBA → NCHW float32                                     │
    │    ImageNet normalize (mean/std per channel)               │
    │    │                                                       │
    │    ▼                                                       │
    │  ORT inference (zero-copy GPU input)           [GPU]       │
    │    DistillAnyDepth or DepthAnythingV2                      │
    │    Output: float[1,518,518] depth map                      │
    │    │                                                       │
    │    ▼                                                       │
    │  ILGPU GuidedDepthUpsampleKernel               [GPU]       │
    │    Joint bilateral upsampling: 518×518 → original W×H      │
    │    5×5 window weighted by spatial distance + color          │
    │    similarity using source RGBA as edge guide              │
    │    (depth edges align with color edges)                    │
    │    │                                                       │
    │    ▼                                                       │
    │  ILGPU MinMaxKernel (atomic reduce)            [GPU]       │
    │    └─▶ 2 floats (min, max) ──GPU──▶ CPU       [8 bytes]   │
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
    │  [CPU path] RGBA bytes ──CPU──▶ GPU (packed int[W*H])     │
    │  [SR path]  GpuImage.PackedRgba already on GPU (no upload)│
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

### ⑤ Viewer (60 FPS render loop)
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
GpuSplatSorter.Sort()                       [GPU, async]
    ┌────────────────────────────────────────────────────────────┐
    │  Polls _syncTask.IsCompleted (non-blocking)                │
    │                                                            │
    │  If previous sort still in-flight:                         │
    │    → skip, render with stale vertex buffer                 │
    │                                                            │
    │  If sort completed (or first frame):                       │
    │    → FrustumCull + DistanceKernel (depth quantization)     │
    │    → ILGPU RadixSort (16-bit: 4 passes, 32-bit: 8 passes) │
    │    → Back-to-front ordering for alpha blending             │
    │    → Submit new sort, capture _syncTask                    │
    │    → sortRan = true                                        │
    │                                                            │
    │  Rate gated: 50ms minimum between submissions              │
    └────────────────────────────────────────────────────────────┘
    │
    ▼
GpuGaussianRenderer.Render()                 [GPU]
    ┌────────────────────────────────────────────────────────────┐
    │  If sortRan:                                               │
    │    WebGPU PackComputeShader                                │
    │      float32 → float16/uint8 vertex packing                │
    │                                                            │
    │  WebGPU SplatPipeline (render pass)                        │
    │    Billboard quads + EWA anti-alias filter                 │
    │    Packed vertex: pos(f32) + color_alpha(u8x4) + scale(f16)│
    │                                                            │
    │  If CAS enabled:                                           │
    │    WebGPU CAS post-process pass (sharpening 0.0–1.0)       │
    │                                                            │
    │  Adaptive resolution:                                      │
    │    Fast camera movement → half-res canvas                  │
    │    Slow/still → full-res canvas                            │
    │                                                            │
    │  → Canvas (GPU direct present)                             │
    └────────────────────────────────────────────────────────────┘
```

## Quality Presets

| Preset   | Subsample | Edge Sharpness | Super Resolution | Sort Mode |
|----------|-----------|----------------|------------------|-----------|
| Fast     | 4         | Off            | No               | 16-bit    |
| Standard | 2         | Medium (0.3)   | No               | 16-bit    |
| High     | 1         | Medium (0.3)   | 2× (if loaded)   | 32-bit    |

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

SR upscaled images stay GPU-resident via `GpuImage` — no CPU readback.
