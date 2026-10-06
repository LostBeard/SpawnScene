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

**Tests:** `dotnet test SpawnScene.Tests -c Release` (NUnit; links app sources, CPU). GPU checks run in the browser through
`tools/_cdp_*.js` against a served publish: `_cdp_trainer_gate.js` (trainer shaders vs CPU oracles), `_cdp_dataset.js`
(datasets, `AUTOTEST=project` for the user path, `AUTOTEST=textlab` for text rendering), `_cdp_page.js` (screenshot any
page; from Git Bash pass `MSYS_NO_PATHCONV=1`). Score captured views with `tools/score_views.py`.

## Project Overview

SpawnScene is a fully client-side Blazor WebAssembly Gaussian Splatting application. It generates 3D scenes from a single photo using monocular depth estimation (DepthAnything V2), with the entire pipeline running on the GPU via WebGPU and SpawnDev.ILGPU. No server backend.

**Stack (2026-10-02):** .NET 10 / C# 13, Blazor WASM (AOT by default, IL kept), SpawnDev.SpawnJS.Blazor 3.x (browser
interop), SpawnDev.ILGPU 5.3.x (WebGPU compute), SpawnDev.ILGPU.ML 5.3.x (GPU inference: Depth Anything V3, RaCo-ALIKED +
LightGlue+ learned matching), SpawnDev.GameUI (the WebGPU UI), native WebGPU (WGSL training and rendering shaders). Exact
versions: SpawnScene/SpawnScene.csproj. Multi-photo scenes: learned matching -> GPU SfM (five-point, rotation averaging,
GPU bundle adjustment) -> Gaussian splat training against the photos; one photo: monocular depth -> splats.

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

- AOT by default (`SpawnSceneAot`, 2026-10-02): `RunAOTCompilation = true` with `WasmStripILAfterAOT = false` (ILGPU
  compiles kernels from IL at runtime) and `PublishTrimmed = true` (WASM AOT requires it; the SpawnScene assembly is
  rooted). The AOT publish takes about an hour; `-p:SpawnSceneAot=false` for a fast interpreted dev publish
- `CompressionEnabled = false`
- `TrimmerRootAssembly` entries for ILGPU, ILGPU.Algorithms, SpawnDev.ILGPU

### Training defaults worth knowing

- Densification uses AbsGS's signal (sum of per-pixel |dL/dmean2D|, bar 8e-4) since 2026-10-06; `&absgrad=0` restores
  the reference signed sum (2e-4). Bicycle 7K held out +0.43 dB / SSIM +0.038 with fewer splats; Truck equal with 57%
  fewer. TrainerGate's `absgrad` case guards it.
- Every per-splat / per-pixel WGSL pass dispatches through `SplatTrainerGpu.DispatchLinear` (wraps past 65535
  workgroups; the shader rebuilds the flat index from `num_workgroups`). An X-only dispatch dies past 4,194,240 items.
- `&gpumem=N` sets a run's training budget (a fresh harness profile is otherwise Auto = 4 GB, cap ~3.3M splats).

### Massive scenes: LOD tree, streaming, partitioned training (Plans/lod-streaming.md)

- **LOD tree** (`LodTree` CPU oracle, `GpuLodTree` GPU build, `LodMerge`): Tiny-LoD grid merges, monotone metric so the
  parallel cut is exact; `GpuSplatSorter` draws only the cut (compacted, prefix-sorted, drawn indirect), steered to a
  splat budget (`&lodbudget`, floored at the scene's tau). XR measures the cut at the eye's focal and gets a device
  budget (600K Quest browser / 1.5M tethered).
- **`.spawnscene` v3** (`LodChunkFile`): the tree breadth-first (`LodLayout`/`GpuLodLayout`, leaves = LOD size 0),
  16K-node run-aligned chunks gzipped one by one, each with its parent chunks (`Needs`). Written by `LodStreamWriter`
  one block at a time under a top (Export streaming = one block).
- **Streaming** (`GpuLodPager`, `&lodpool=N`): a fixed pool of pages; the paged cut stands a node in for missing
  children and asks for their chunk (priority = stand-in px); loads closed under `Needs`, evicts only unused pages
  (never one loaded since the previous readback). Sources: a File by slices (Open scene file), memory, or HTTP Range.
- **Partitioned training** (`&blocks=CxR`, Studio.Partition): coarse model + blocks with outside frozen; a result past
  one run's splat cap (or `&streamed=1`) is never merged - it is saved as a streamed project scene
  (`ProjectScene.Format = lod`, `scenes/{id}.spawnscene`) and always opened streamed.
- Tests: LodLayoutTests / LodPagerSimTests / GpuLodLayoutTests (CPU accelerator); `LodLayoutRealSceneTests` is Explicit.

### Disk hygiene (TJ, 2026-10-05)

One publish folder per purpose (`_pub_tuvok_est` 8104, `_pub_tuvok_gate` 8105), overwritten; every harness run deletes
its `spawnscene-harness-<port>` profile; an export lives in one place (`_pub_tuvok_est/wwwroot/samples`); screenshots
only for numbers being reported.

### Deployment

GitHub Actions workflow (`.github/workflows/deploy-to-github-pages.yml`, manual trigger) publishes to `gh-pages` branch. It rewrites the base tag in `index.html` to `/SpawnScene/` and copies `index.html` to `404.html` for SPA routing.
