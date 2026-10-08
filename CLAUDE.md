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

**Other tools' scenes:** "Open scene file" and `?import=<url>` also take a 3DGS `.ply` (GaussianPly + GaussianPlyImport),
PlayCanvas's compressed `.ply` (SuperSplat's export; GaussianPly.ParseCompressed) and `.sog` (zip of WebP textures; SogMeta,
SogImport - WebP decoded by the browser into a WebGPU texture, never a 2D canvas, which premultiplies), Niantic's `.spz` v2/v3 (SpzImport) and
antimatter15's `.splat` (SplatFileImport), converted on the GPU into a new project. All are turned y-up by default (3DGS
PLYs are in their SfM frame, y down, and the other formats in circulation carry the same frame - Spark's examples turn
SPZ too); `&sceneup=keep` leaves them. Imports seat on the scene's dense core (20-80% box). Tests: GaussianPlyImportTests,
CompressedPlyImportTests (vs a port of PlayCanvas's decoder), SpzImportTests (Niantic's packing ported),
SplatFileImportTests, SogImportTests (vs a port of PlayCanvas's SOG iterator); the y-up turn is checked against SH physics. Verified 2026-10-08 on Inria's Train (7K PLY),
antimatter15's train.splat, Spark's butterfly/penguin .spz, PlayCanvas's biker/guitar compressed PLY and skull.sog (v2, SH 3; also unbundled: `?import=.../meta.json` fetches the textures beside it, pixel-identical). SPZ v4 (zstd
streams, already in PlayCanvas's examples) is refused with a reason: Chrome's DecompressionStream has no zstd. A kernel
with no SH bands must bind three DISTINCT stand-in buffers (WebGPU refuses aliased read_write bindings; CPU tests cannot
see it). `_cdp_page.js` takes `PAGE_LOG=<regex>` to print the app's own console lines.

**Samples:** `wwwroot/samples/catalog.json` (SampleCatalog) - openly licensed Commons sets and photos in the
LostBeard/spawnscene-samples HF dataset, fetched through the hub's `/src` proxy (never huggingface.co directly).
`autotest=samples&name=<folder>` loads one into a project. Tools: commons_fetch.py, package_sample.py,
make_sample_catalog.py. Research/demo-samples-2026-10-07.md.

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

- **Sorted** (default since 2026-10-07) - sorted alpha blending (cull, radix sort, pack, render), at full resolution
  (`AdaptiveResMode.ForceFull`). The defaults live in the renderer so no path can forget them: the Generate button used to
  leave a fresh scene in Stochastic, and TJ saw floaters there that a reopen (sorted) did not show.
- **Stochastic** - sort-free stochastic rasterization with temporal accumulation, a Settings choice. 1-2 samples a pixel
  and a 0.15 alpha floor while moving: faint splats become visible floaters.

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
- Versus the reference (gsplat, same photos held out, llffhold=8, COLMAP poses), TruckFull 30K: ours 24.95 dB / SSIM
  0.887 with 0.89M splats in 25 min, gsplat 25.13 / 0.877 with 3.79M (c49, 2026-10-06). `&mcmc=1` (gsplat MCMCStrategy)
  exists but lost its 7K A/B, so it is opt-in.
- **Floater carve** (default since 2026-10-07, `&carve=0` off): a GPU census (SplatTrainerGpu.Floaters) measures each
  splat's share of blending weight in front of the photos' surfaces; >= 0.9 is removed every 1000 iterations while
  densifying and at the end, plus splats under 1 px of weight in every photo. Bicycle 7K held out 25.01 -> 24.97 dB,
  SSIM 0.7630 -> 0.7643, sky drips and near-grass haze gone off the photo path; TruckFull 7K 23.74 -> 23.94 dB.
- **Census: a pixel with no surface has nothing in front of it** (fixed 2026-10-07, 37c1fd2). Pixels that never turned
  opaque kept a 3e38 sentinel and every splat on them counted as a floater: the carve deleted the still-thin walls of
  TJ's Bathroom ("TONS of holes"). TrainerGate's thin-splat-in-a-hole case guards it. Bicycle unchanged by the fix
  (k1 25.06 / 0.7662 vs 25.07 / 0.7663). The end carve's splats are compacted before the save (GpuDensify PruneOnly).
- **Depth init + per-photo exposure gains are DEFAULTS (2026-10-08, TJ):** `DepthFusionInitStride` 4 (seeds from the
  photos' DAv3 depth where two views agree; `&depthinit=0` off) and exposure gains only (`&exposure=0` off, `=1`/`affine`
  the full 3x4). Bathroom held out 15.58 -> 18.57 dB with both, fair score (gains fitted on the left half, right half
  scored - logged as "held out RIGHT HALF") 18.88 -> 24.32 from the gains; Bicycle fair +0.16; Truck neutral.
- **Opt-ins under evaluation** (also on ANY Studio URL, e.g. `spawnscene.com/studio?depthinit=4&exposure=1`):
  `exposure=1` per-photo 3x4 affine exposure, the photos' mean folded into the scene at the end (TrainerGate exposure
  case); `depthinit=N` seeds from the photos' DAv3 depth where two views agree, coloured from the device decode
  (DepthFusionInit, DepthFusionInitTests). Bathroom both: held out 15.58 -> 18.27 dB, SSIM 0.709 -> 0.836 (g4).
  `edgesnap=1` single-photo depth edges snapped to colour (DepthEdgeSnap, DepthEdgeSnapTests); `inpaint=1` MI-GAN paints
  the hidden behind-edge and past-the-frame layers (HiddenLayerInpaint, LostBeard/spawnscene-models via the hub;
  autotest=inpaint-parity = onnxruntime to 3 decimals); `depthmodel=depth-anything-v3-base`;
  `depthloss=X` (with depthinit) depth supervision - L1 on the rendered inverse depth against each photo's scaled DAv3
  depth, X -> X/100 (SplatTrainerGpu.Depth; the `//DEPTH:` shader lines build the supervised pipelines, the defaults stay
  byte-identical; TrainerGate depth stage = finite differences). Edge snap's slope guard `snapmid` (0.3). Harness-only:
  `&carveunseenpx=W`, `&inpaintreach=X`, `&cellstretch=X`. Research/single-photo-tearing-2026-10-07.md, PLANS.md.
- **Capture feedback:** ProjectScene.PhotosPlaced/Total/NotPlaced; the scene card counts them, the Photos tab badges
  the photos SfM could not place. A photo held sideways enters SfM with transposed intrinsics and trains with one
  quarter turn; the focal calibration uses the majority intrinsics' pairs.
- **Splat size cap 0.1 x rig radius** (was 0.05, a leftover from i32 fixed-point gradients): far background at the cap
  shattered into tiny depth-drifting splats = dark specks in the sky off-path. Bicycle 7K held out 24.98 -> 25.07 dB,
  TruckFull 23.94 -> 24.10; no cap at all is worse on Truck (23.85) and brings sky drips back on Bicycle.
- Judge quality OFF the photo path too: Studio.Wander's views (in/up/low/out/mid, pan-0..7 = turning around at the rig
  centre with the photo count per heading, over-0/1 = outside the rig) in the project and dataset
  autotests, `tools/compose_wander.py <Dataset> <TAG>[,<TAG2>] [gsplat dir]`. Every held-out view sits beside a photo.
- Every per-splat / per-pixel WGSL pass dispatches through `SplatTrainerGpu.DispatchLinear` (wraps past 65535
  workgroups; the shader rebuilds the flat index from `num_workgroups`). An X-only dispatch dies past 4,194,240 items.
- `&gpumem=N` sets a run's training budget (a fresh harness profile is otherwise Auto = 4 GB, cap ~3.3M splats).

### Single photo: what a moved camera sees (2026-10-06)

- Focal length from EXIF, else DAv3's camera estimate (the old 1.2x-the-long-side guess made rooms 1.5-2x too deep).
- Each depth-grid splat spans its cell (`SplatCovariance.SurfaceDiskFromNeighbors`), so receding surfaces stay solid.
- `OcclusionFill` adds two hidden layers on the GPU: background behind every depth edge (max/min depth filters + a
  far-priority push-pull pyramid) and the photo continued 35% past its frame. `&occfill=0` turns it off. Room sample,
  empty share of a view moved 10-30% of the scene depth: 15-39% before, 0-3% after (13.7% on a 30% orbit).
- The viewer's Settings panel has **Scene depth** for photo scenes (`SplatRows.Relief`): splats slide along their own
  pixel rays, so the photo's view is unchanged and only the depth scales. `&scenedepth=k` sets it in a harness run.
- Measure with `autotest=generate-room&holeviews=1` and `SPAWNSCENE_EDIT_SHOT_ON='\[Holes\] VIEW (\w+) READY'`.

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
