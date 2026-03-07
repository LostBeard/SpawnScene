# SpawnScene

> Create interactive 3D Gaussian Splat scenes from a single photo — entirely in your browser.

![Blazor WebAssembly](https://img.shields.io/badge/Blazor-WebAssembly-512BD4?style=flat-square)
![.NET 10](https://img.shields.io/badge/.NET-10-512BD4?style=flat-square)
![WebGPU](https://img.shields.io/badge/WebGPU-required-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**SpawnScene** is a fully client-side Gaussian Splatting application built with Blazor WebAssembly. It uses monocular depth estimation (DepthAnything V2) to generate 3D scenes from a single photograph, with the entire pipeline running on the GPU via WebGPU and SpawnDev.ILGPU.

## ✨ What It Does

- **🔮 Depth Estimation** — DepthAnything V2 (Small) via ONNX Runtime Web on the WebGPU execution provider
- **⚡ GPU Gaussian Generation** — ILGPU compute kernel unprojects depth + color into 100K+ packed Gaussian splats
- **🔀 Radix Sort** — ILGPU O(n) radix sort depth-sorts splats every frame for correct alpha blending
- **👁 Real-Time Viewer** — WebGPU splat renderer at 60 FPS with EWA anti-aliasing and CAS post-processing sharpening
- **📦 PLY / SPLAT Loading** — Load pre-built scenes from `.ply` or `.splat` files
- **🔒 Fully Client-Side** — No server, no uploads. All GPU compute runs in your browser.


### Screenshots
[![Garden Original](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/samples/garden_hd.png)](https://lostbeard.github.io/SpawnScene/)  
[![Garden Splat 1](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/garden-splat-1.jpg)](https://lostbeard.github.io/SpawnScene/)  
[![Garden Splat 2](https://raw.githubusercontent.com/LostBeard/SpawnScene/master/SpawnScene/wwwroot/screenshots/garden-splat-2.jpg)](https://lostbeard.github.io/SpawnScene/)  



## 🚀 Pipeline

```
Single photo
  → GPU (one-time upload)
  → ILGPU PreprocessKernel     RGBA → NCHW float[1,3,518,518]
  → ONNX WebGPU inference      DepthAnything V2 Small
  → ILGPU ResizeKernel         518×518 → original resolution
  → ILGPU MinMaxReduce         2 floats to CPU (UI metadata only)
  → ILGPU UnprojectAndPackKernel  depth + RGBA → 10 floats/splat
  → ILGPU RadixSort            back-to-front ordering
  → WebGPU pack compute        Float32 → Float16/UNorm8 vertex format
  → WebGPU splat render        billboard quads + EWA filter + CAS
  → Canvas
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| App framework | Blazor WebAssembly (.NET 10) |
| JS interop | [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) |
| GPU compute | [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) 3.5.0 (WebGPU backend) |
| Depth estimation | ONNX Runtime Web 1.25 (WebGPU EP) |
| Depth model | DepthAnything V2 Small |
| Rendering | Native WebGPU (WGSL shaders) |
| Language | C# 13 |

## 📋 Requirements

- **WebGPU-capable browser**: Chrome 113+, Edge 113+, or Safari 18+
- No installation required — runs entirely client-side

## 📁 Project Structure

```
SpawnScene/
├── Models/
│   ├── Gaussian3D.cs              # 3D Gaussian splat struct
│   ├── GaussianScene.cs           # Scene container (CPU or GPU-resident)
│   ├── CameraParams.cs            # Camera intrinsics/extrinsics + view matrix
│   └── DepthResult.cs             # GPU-resident depth map + metadata
├── Services/
│   ├── GpuShareService.cs         # WebGPU device sharing (monkey-patches navigator.gpu)
│   ├── GpuService.cs              # ILGPU WebGPU accelerator lifecycle
│   ├── DepthEstimationService.cs  # ONNX depth inference + GPU pre/post-processing
│   ├── DepthToGaussianKernel.cs   # ILGPU kernel: depth → packed Gaussian buffer
│   ├── GpuSplatSorter.cs          # ILGPU radix sort for back-to-front ordering
│   ├── GpuGaussianRenderer.cs     # WebGPU splat renderer (pack compute + render passes)
│   ├── GpuDepthColorizer.cs       # GPU Turbo colormap for depth preview
│   ├── RenderService.cs           # Render loop + scene upload coordination
│   ├── SceneManager.cs            # Active scene + camera state
│   └── CameraController.cs       # FPS-style camera (WASD + mouse look)
├── Pages/
│   ├── Home.razor                 # Landing page
│   ├── DepthSplat.razor           # Depth estimation + scene generation UI
│   └── Viewer.razor               # Interactive 3D splat viewer
└── Formats/
    ├── PlyParser.cs               # PLY file parser
    └── SplatParser.cs             # SPLAT file parser
```

## 🏃 Getting Started

### Prerequisites
- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- A WebGPU-capable browser (Chrome 113+ recommended)

### Run Locally
```bash
cd SpawnScene
dotnet run
```

Navigate to `https://localhost:5001` (or the URL shown in the terminal).

### Usage
1. **Depth Splat** — Load the DepthAnything V2 model, upload a photo, estimate depth, generate Gaussians
2. **View in Splat Viewer** — Navigate to the 3D viewer to explore the generated scene
3. **Viewer** — Can also load `.ply` or `.splat` files directly

## 🗺️ Roadmap

- 🔲 Full zero-copy depth pipeline — keep ONNX output on GPU (blocked by [ORT bug #26107](https://github.com/microsoft/onnxruntime/issues/26107))
- 🔲 Multi-image scenes — merge depth splat clouds from multiple photos
- 🔲 Live camera input — real-time scene generation from device camera
- 🔲 WebXR / VR headset viewing
- 🔲 Scene export (PLY / SPLAT)
- 🔲 Improved Gaussian quality — learned opacity, covariance, spherical harmonics
- 🔲 Peer-to-peer cooperative scanning via WebRTC
- 🔲 Video input support

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Todd Tanner** ([@LostBeard](https://github.com/LostBeard))
