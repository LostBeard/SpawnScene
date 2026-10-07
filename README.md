# SpawnScene

> Photos and video in, explorable 3D Gaussian Splat scenes out - reconstructed, trained, edited and viewed entirely in your browser.

**Try it: [spawnscene.com](https://spawnscene.com)**

**SpawnScene** is a fully client-side Gaussian Splatting studio built with Blazor WebAssembly. Feature matching,
structure from motion, bundle adjustment, Gaussian splat training, depth estimation and rendering all run on your GPU
through WebGPU, using [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) and
[SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML). There is no server: nothing you load leaves your
machine.

## Screenshots

**Trained from photos** - Mip-NeRF 360 *Bicycle* and Tanks and Temples *Truck*, reconstructed and trained in the browser:

[![Trained Bicycle](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/trained-bicycle.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/trained-bicycle.jpg)
[![Trained Truck](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/trained-truck.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/trained-truck.jpg)

**One photo** - the Room sample as a scene, then the camera orbited away from where the photo was taken:

[![Single photo](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/single-photo-room.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/single-photo-room.jpg)
[![Single photo, camera moved](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/single-photo-room-orbit.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/single-photo-room-orbit.jpg)

**Streaming a multi-room scene** - Deep Blending *DrJohnson* as a level-of-detail tree, loading only the chunks the view needs:

[![Streaming DrJohnson](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/streaming-drjohnson.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/streaming-drjohnson.jpg)

**Editing** - an add / subtract selection (both wheels, minus the rear hub) ready to delete, move or copy:

[![Editor](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/editor-selection.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/editor-selection.jpg)

**Projects** - each project keeps its photos and scenes in browser storage, with quality presets for reconstruction:

[![Project page](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/project-page.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/project-page.jpg)

## Features

### From many photos or a video

- **Learned feature matching** - RaCo-ALIKED keypoints with LightGlue+ matching, on the GPU.
- **Structure from motion on the GPU** - five-point relative poses, rotation averaging, global positioning and GPU
  bundle adjustment place the cameras; no COLMAP needed. A video is split into frames first.
- **Gaussian splat training in the browser** - a WebGPU trainer with SSIM + L1 loss, spherical harmonics up to degree 3,
  and AbsGS density control. On Tanks and Temples *Truck* (30K iterations, every 8th photo held out) it scores
  24.95 dB PSNR / 0.887 SSIM with 0.89M splats in 25 minutes, against the gsplat reference's 25.13 / 0.877 with 3.79M.
- **Quality presets** for photo resolution, keypoints, iterations and scene size, sized to your GPU's memory.

### From a single photo

- **Depth Anything V3** depth, with the camera's focal length from EXIF or estimated by the model.
- **Super-resolution** (ESPCN x3) for small photos before they become splats - finer colour and geometry.
- **No holes when you move**: every splat spans its surface cell, a hidden background layer fills in behind each depth
  edge, and the photo continues softly past its frame.
- **Scene depth slider** to correct a depth estimate that came out too deep or too flat, without changing the photo's
  own view.

### Big scenes

- **Level-of-detail rendering** - a splat tree drawn to a budget, so huge scenes stay smooth.
- **Streaming `.spawnscene` files** - chunked, gzipped LOD trees that open immediately and stream the rest from a file
  or over HTTP range requests, under a fixed GPU memory pool.
- **Partitioned training** - a scene larger than one training run is trained in blocks and saved as a streamed scene.

### Editing

- Select by rectangle or **brush**; **add** (Shift), **subtract** (Ctrl), invert, or select everything.
- **Filters**: only the faint splats (haze, floaters), or the largest N% (blobs).
- Delete, keep only, move, copy / cut / paste, insert another scene, undo.
- Save as a new scene, or export a `.spawnscene` file (flat or streaming).

### Viewing

- WebGPU renderer with sorted and stochastic (sort-free) modes, EWA anti-aliasing and contrast-adaptive sharpening.
- **WebXR**: view scenes in VR or AR (Quest browser and tethered headsets), with a device-sized splat budget and
  in-headset box selection.
- The whole interface is drawn with WebGPU ([SpawnDev.GameUI](https://github.com/LostBeard/SpawnDev.GameUI)), so it
  works the same in a headset.

## Tech stack

| Component | Technology |
|---|---|
| App | .NET 10 Blazor WebAssembly (AOT), C# 13 |
| JS interop | [SpawnDev.SpawnJS](https://github.com/LostBeard/SpawnDev.SpawnJS) |
| GPU compute | [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) (WebGPU backend) |
| Machine learning | [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML) - Depth Anything V3, RaCo-ALIKED, LightGlue+, ESPCN |
| User interface | [SpawnDev.GameUI](https://github.com/LostBeard/SpawnDev.GameUI) (WebGPU) |
| Rendering and training | Native WebGPU, WGSL shaders |
| Storage | Origin Private File System (projects, photos, scenes) |

## Requirements

- A **WebGPU** browser: Chrome or Edge 113+, or Safari 18+. Training large scenes wants a discrete GPU.
- Nothing to install.

## Getting started

### Run locally

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).

```bash
cd SpawnScene
dotnet run
```

Then open the URL shown in the terminal. A Release publish compiles AOT by default (`-p:SpawnSceneAot=false` for a
quick interpreted build).

### Use it

1. **Create a project** and add photos or a video.
2. **Generate a scene**: one photo builds a scene from depth; several photos are matched, posed and trained.
3. **Explore** with the mouse and WASD, open **Edit** to clean the scene up, or enter **VR / AR**.
4. **Export** a `.spawnscene` file, or **Open scene file** to view one.

## Project layout

```
SpawnScene/
├── Pages/      Studio.*.cs - the studio, split by area (projects, training, editing, LOD, XR, ...)
├── Services/   GPU services - trainer, renderer, SfM, matching, depth, LOD tree and pager, editor kernels
├── Models/     projects, cameras, scenes
└── wwwroot/    samples, datasets for the built-in tests, screenshots
SpawnScene.Tests/   NUnit tests (CPU accelerator)
tools/              browser test harnesses (Chrome DevTools Protocol)
Plans/              design notes (e.g. lod-streaming.md)
```

## License

MIT License - see [LICENSE](LICENSE.txt) for details.

Models downloaded at run time carry their own licences (RaCo, ALIKED BSD-3-Clause, LightGlue): see
[THIRD-PARTY-NOTICES](SpawnScene/wwwroot/licenses/THIRD-PARTY-NOTICES.md).

## Author

**Todd Tanner** ([@LostBeard](https://github.com/LostBeard))
