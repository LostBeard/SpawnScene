# How SpawnScene works

Everything below runs in the browser on the user's GPU (WebGPU), written in C# and compiled to WGSL compute shaders by
[SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) or hand-written WGSL. Neural networks run on
[SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML), our own ONNX graph engine on the same GPU - no ONNX
Runtime, no server. Measurements behind each choice are in [benchmarks.md](benchmarks.md) and `Research/`.

## Many photos: from pixels to a trained scene

### 1. Matching
RaCo-ALIKED keypoints (1024 a photo by default, 3072 on the High preset) and LightGlue+ matching, both as GPU networks.
Up to 60 photos every pair is matched; above that, each photo's 30 best partners by a GPU overlap score (mutual nearest
neighbour descriptors passing a ratio test - on Truck its top 20 partners are ~98% true pairs, at a tenth of all-pairs
cost).

### 2. Structure from motion (no COLMAP)
- Relative poses per pair: five-point essential-matrix RANSAC on the GPU.
- Global rotation averaging, then global positioning, then GPU bundle adjustment: Levenberg-Marquardt with Huber
  weights, solved matrix-free - the Schur complement is never built; its products are formed per point and per camera
  from the observation Jacobians, and a block-Jacobi preconditioned CG over the cameras runs on the device.
- Cameras the global step could not place are re-registered against the triangulated points in passes; a camera whose
  observations disagree with the solution, or that shares too few points with the rest, is dropped - and the project
  page says which, and why.
- Photos held sideways enter with transposed intrinsics; the focal length is calibrated from the majority orientation.
- The result is levelled (the cameras' mean up becomes +Y).

### 3. Initial splats
- The bundle-adjusted sparse points (as in the reference 3DGS), **plus seeds from the photos' depth**: each photo's
  Depth Anything V3 depth is scaled to the SfM points it sees (median ratio), and a depth sample becomes a seed only
  where another photo's depth agrees within 3% - walls with few matched features start covered. Default since
  2026-10-08 (phone room +1.3 dB held out, Bicycle neutral).

### 4. Training (Kerbl et al. 2023, with these changes)
- Loss 0.8 L1 + 0.2 D-SSIM; spherical harmonics up to degree 3 (one band per 1000 iterations); Adam with the
  reference's learning rates; camera pose refinement.
- **Density control: AbsGS** (Ye et al. 2024) - the densify signal is the sum of per-pixel |dL/dmean2D| instead of
  the signed sum. Bicycle 7K +0.43 dB with fewer splats; Truck the same quality with 57% fewer.
- **Floater census**: a GPU pass measures each splat's share of blending weight in front of the photos' surfaces
  (the depth at which transmittance falls through 0.5); splats that are mostly in front of what every photo saw are
  removed while densifying and at the end.
- **Per-photo exposure** (gains only): each photo gets three per-channel gains, learned with the scene and folded into
  it at the end (the reference's appearance model is a full 3x4 affine; its offsets cannot fold into a semi-transparent
  scene and cost fixed-exposure captures 0.2-0.6 dB, so we keep the gains). Default since 2026-10-08.
- Optional: depth supervision (`&depthloss`, an L1 on rendered inverse depth, gradient checked against finite
  differences), Mip-Splatting's 3D filter (`&mipfilter`), MCMC densification (`&mcmc`).
- WebGPU has no float atomics: per-splat gradients are reduced per tile in the backward pass and summed with an atomic
  compare-and-swap on f32 bit patterns.
- Every trainer shader is checked against a CPU oracle in a GPU test gate (`autotest=trainer-gate`).

## One photo
- Depth Anything V3 (monocular), focal length from EXIF or the model's own camera estimate.
- Each pixel becomes a splat sized to its surface cell, so receding surfaces stay solid.
- **Occlusion fill**: a hidden background layer behind every depth edge (max/min filters + a far-priority push-pull
  pyramid) and the photo continued past its frame, so a moved camera does not see holes.
- Options: depth edges snapped to colour edges (two-plateau snap with a slope guard: on four test photos the
  "flying pixels" at depth steps fall 20-45% and textured ground is left alone); the hidden layer painted by MI-GAN or
  big-LaMa.

## Viewer
- Sorted alpha blending at full resolution (GPU radix sort), EWA filter, optional stochastic sort-free mode.
- Level-of-detail tree with a parallel cut steered to a splat budget; `.spawnscene` v3 files stream chunks over HTTP
  range requests into a fixed GPU memory pool.
- WebXR VR / AR, with the interface itself drawn in WebGPU (SpawnDev.GameUI).
- Opens other tools' formats (3DGS PLY, compressed PLY, SOG, SPZ, SPLAT), decoded on the GPU.

## References
- B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
  SIGGRAPH 2023.
- Z. Ye et al. *AbsGS: Recovering Fine Details for 3D Gaussian Splatting.* ACM MM 2024.
- Z. Yu et al. *Mip-Splatting: Alias-free 3D Gaussian Splatting.* CVPR 2024.
- S. Kheradmand et al. *3D Gaussian Splatting as Markov Chain Monte Carlo.* NeurIPS 2024.
- V. Ye et al. *gsplat: An Open-Source Library for Gaussian Splatting.* JMLR 2025.
- L. Yang et al. *Depth Anything V3.*; P. Lindenberger et al. *LightGlue.* ICCV 2023; ALIKED (Zhao et al. 2023).
- A. Sargsyan et al. *MI-GAN.* ICCV 2023; R. Suvorov et al. *LaMa.* WACV 2022.
