# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cd SpawnScene
dotnet run
# Opens at https://localhost:5001
```

**Publish (release):**
```bash
dotnet publish ./SpawnScene/ --nologo -c:Release --output publish
```

There are no tests or linting tools configured.

## Project Overview

SpawnScene is a fully client-side Blazor WebAssembly Gaussian Splatting application. It generates 3D scenes from a single photo using monocular depth estimation (DepthAnything V2), with the entire pipeline running on the GPU via WebGPU and SpawnDev.ILGPU. No server backend.

**Stack:** .NET 10 / C# 13, Blazor WASM, SpawnDev.ILGPU 4.0.0 (WebGPU compute), ONNX Runtime Web 1.25 (WebGPU EP), native WebGPU (WGSL shaders), SpawnDev.BlazorJS (JS interop).

**Browser requirement:** WebGPU-capable (Chrome 113+, Edge 113+, Safari 18+). No fallbacks exist.

## GPU-First Pipeline Rule

**This is the most important architectural constraint.** Data must never leave the GPU unless unavoidable. Before any CPU readback, ask: can an ILGPU kernel or WebGPU shader do this instead?

Acceptable CPU transfers (must have `// CPU transfer: <reason>` comment):
- File I/O (images, PLY, SPLAT)
- Scalar metadata only (e.g. 2 floats min/max for UI)

Anti-patterns to avoid:
- `await outputTensor.GetDataAsync<Float32Array>()` — copies GPU→CPU; use `ExternalWebGPUMemoryBuffer` instead
- CPU packing loops + upload — use ILGPU kernel instead
- CPU colorization → ImageData — use ILGPU kernel + `WebGPUCanvasRenderer.PresentAsync()`
- Backend selection/fallback logic — WebGPU is always available, cast directly to `WebGPUAccelerator`
- `DepthResult` must hold only GPU-resident `MemoryBuffer1D<float>`, never `float[]` arrays

## Architecture

### GPU Pipeline (data flow)

```
Photo (CPU read, unavoidable)
  → Upload RGBA once → GPU
  → ILGPU PreprocessKernel (RGBA → NCHW 518x518)
  → ONNX WebGPU inference (DepthAnything V2 Small)
  → ILGPU ResizeKernel (518x518 → original res)
  → ILGPU MinMaxReduce (2 floats → CPU, UI metadata only)
  → ILGPU UnprojectAndPackKernel (depth + RGBA → 10 floats/splat)
  → ILGPU RadixSort (back-to-front ordering)
  → WebGPU pack compute (Float32 → Float16/UNorm8 vertex format)
  → WebGPU splat render (billboard quads + EWA filter + CAS sharpening)
  → Canvas
```

### Render Loop

RAF → `RenderService.RenderFrame()` → `GpuGaussianRenderer.Render()` (hot path, synchronous).

- **Sort frame:** `GpuSplatSorter.Sort()` polls `_syncTask.IsCompleted` (non-blocking). If sort completed, pack compute shader runs, then render with new vertex buffer.
- **Non-sort frame:** render with stale vertex buffer from last pack (no GPU submission for sort).
- Sort is self-throttling: natural rate = GPU sort duration, 50ms minimum floor between submissions.

### Adaptive Resolution

Canvas pixel dimensions halve during fast camera movement, restore when slow. Thresholds in `GpuGaussianRenderer`:
- `LowResEnterVelocity = 0.0002f`
- `LowResExitVelocity = 0.00005f`

### Key Services

| Service | Role |
|---|---|
| `GpuService` | ILGPU WebGPU accelerator lifecycle; device sharing with ORT via `GpuShareService` |
| `DepthEstimationService` | ONNX depth inference + GPU pre/post-processing kernels |
| `DepthToGaussianKernel` | ILGPU kernel: depth + RGBA → packed Gaussian buffer |
| `GpuSplatSorter` | Async non-blocking ILGPU radix sort with frustum culling |
| `GpuGaussianRenderer` | WebGPU splat renderer: pack compute + render pass + CAS post-processing |
| `RenderService` | RAF render loop orchestration + scene upload coordination |
| `SceneManager` | Active scene + camera state, fires `OnSceneChanged`/`OnCameraChanged` events |
| `CameraController` | FPS-style camera (WASD + mouse look + scroll zoom) |

### Pages

- **Home** (`/`) — Landing page
- **DepthSplat** (`/depth-splat`) — Load model → upload photo → estimate depth → generate Gaussians
- **Viewer** (`/viewer`) — Interactive 3D splat viewer, also loads `.ply`/`.splat` files

### Build Constraints (csproj)

- `PublishTrimmed = false` — ILGPU kernel methods are invoked via reflection
- `RunAOTCompilation = false` — ILGPU needs IL at runtime
- `CompressionEnabled = false`
- `TrimmerRootAssembly` entries for ILGPU, ILGPU.Algorithms, SpawnDev.ILGPU

### Deployment

GitHub Actions workflow (`.github/workflows/deploy-to-github-pages.yml`, manual trigger) publishes to `gh-pages` branch. It rewrites the base tag in `index.html` to `/SpawnScene/` and copies `index.html` to `404.html` for SPA routing.
