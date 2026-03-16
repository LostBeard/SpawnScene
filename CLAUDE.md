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

**Stack:** .NET 10 / C# 13, Blazor WASM, SpawnDev.ILGPU 4.0.0 (WebGPU compute), ONNX Runtime Web 1.25 (WebGPU EP, DistillAnyDepth + DepthAnything V2), native WebGPU (WGSL shaders), SpawnDev.BlazorJS (JS interop).

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

### Render Modes (`SplatRenderMode`)

The renderer supports two modes, switchable via `GpuGaussianRenderer.RenderMode`:

- **Stochastic** (default) — Sort-free stochastic rasterization with temporal accumulation. No per-frame radix sort. ~45-60 FPS.
- **Sorted** — Traditional sorted alpha blending (cull → radix sort → pack → render). Legacy mode for A/B comparison.

### GPU Pipeline (data flow)

```
Photo (CPU read, unavoidable)
  → Upload RGBA once → GPU
  → ILGPU PreprocessKernel (RGBA → NCHW 518x518)
  → ONNX WebGPU inference (DistillAnyDepth Small, default)
  → ILGPU ResizeKernel (518x518 → original res)
  → ILGPU MinMaxReduce (2 floats → CPU, UI metadata only)
  → ILGPU UnprojectAndPackKernel (depth + RGBA → 10 floats/splat)
  → ILGPU IdentityFill + WebGPU pack compute (one-time at upload)
  → Per frame (stochastic mode):
      → WebGPU stochastic splat render (billboard quads + stochastic discard + depth test)
      → WebGPU accumulation blend (temporal EMA into persistent texture)
      → WebGPU CAS display (sharpening → canvas)
  → Canvas
```

### Stochastic Render Loop

RAF → `RenderService.RenderFrame()` → `GpuGaussianRenderer.Render()` → `RenderStochastic()`.

Per frame, SPP × (stochastic render + accumulate) + 1 display pass:
1. **Stochastic splat render** → `_stochasticTexture`: billboard quads with EWA, fragment does stochastic discard (`u >= effective_alpha`), opaque writes with depth test.
2. **Accumulate blend** → `_accumTexture`: fullscreen EMA blend with weight `1/frameCount`.
3. **CAS display** → canvas: contrast-adaptive sharpening on accumulated result.

Key behaviors:
- **Moving camera:** `_accumFrameCount` resets to 0 each frame → no inter-frame ghosting. SPP sub-samples averaged within the frame only.
- **Still camera:** `_accumFrameCount` grows to 1024 → deep progressive convergence.
- **Velocity-adaptive SPP:** movement=2, convergence burst=3, converged=1.
- **Min alpha floor:** during movement, low-alpha edge fragments boosted to 0.15 survival → fills holes.
- **Velocity-adaptive dilation:** subtle splat fattening via uniform (max +5%) bridges sub-pixel gaps.

### Sorted Render Loop (legacy)

Same as before: `Sort()` polls `_syncTask.IsCompleted` (non-blocking). If sort completed, pack compute runs, then render with new vertex buffer. Sort is self-throttling: 50ms minimum floor.

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
| `GpuSplatSorter` | ILGPU radix sort (sorted mode) + velocity tracking + identity fill (stochastic mode) |
| `GpuGaussianRenderer` | WebGPU renderer: stochastic + sorted pipelines, accumulation, CAS post-processing |
| `RenderService` | RAF render loop orchestration + scene upload coordination |
| `SceneManager` | Active scene + camera state, fires `OnSceneChanged`/`OnCameraChanged` events |
| `CameraController` | FPS-style camera (WASD + mouse look + scroll zoom) |

### Pages

- **Home** (`/`) — Landing page (standard Blazor HTML)
- **Studio** (`/studio`) — Unified tool page: project management + scene generation + 3D viewer. Entire UI rendered via WebGPU (no HTML elements). Uses `StudioLayout` (full-viewport, no sidebar).
- **DepthSplat** (`/depth-splat`) — Legacy: standalone depth estimation + generation UI
- **Viewer** (`/viewer`) — Legacy: standalone 3D splat viewer, loads `.ply`/`.splat` files

### WebGPU UI Framework (`UI/`)

Custom immediate-mode-style UI rendered entirely via WebGPU for VR compatibility:
- `FontAtlas` — runtime bitmap font atlas (OffscreenCanvas → GPUTexture, 4 sizes)
- `UIRenderer` — batched quad renderer (up to 4096 quads, single draw call overlay)
- `InputManager` — polling-based input (mouse/keyboard/gamepad, pending buffer pattern)
- `UIElement` — retained-mode tree with hit testing
- Components: `UILabel`, `UIButton`, `UIPanel`, `UISlider`

### Project System

- `ProjectService` — OPFS-backed CRUD for projects (source images, generated scenes, settings)
- `Project` / `ProjectSettings` / `ProjectSource` / `ProjectScene` — data models
- OPFS structure: `/spawnscene/projects.json` (index) + `/spawnscene/projects/{id}/` (files)

### Build Constraints (csproj)

- `PublishTrimmed = false` — ILGPU kernel methods are invoked via reflection
- `RunAOTCompilation = false` — ILGPU needs IL at runtime
- `CompressionEnabled = false`
- `TrimmerRootAssembly` entries for ILGPU, ILGPU.Algorithms, SpawnDev.ILGPU

### Deployment

GitHub Actions workflow (`.github/workflows/deploy-to-github-pages.yml`, manual trigger) publishes to `gh-pages` branch. It rewrites the base tag in `index.html` to `/SpawnScene/` and copies `index.html` to `404.html` for SPA routing.
