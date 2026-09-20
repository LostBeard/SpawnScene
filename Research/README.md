# Research: 3D Gaussian Splatting

Sources gathered 2026-09-20 while working out why SpawnScene's reconstruction was stuck at
~13 dB PSNR. Organised by what each is useful FOR, with the load-bearing numbers inline so you
do not have to re-fetch a paper to remember why it is here.

**The one-line conclusion:** depth fusion is only an *initialisation*. 3DGS quality comes from
the per-scene optimiser. No surveyed method reports competitive quality without one.

---

## 1. Canonical 3DGS — the method we are implementing

- **Paper** — Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time
  Radiance Field Rendering*, SIGGRAPH 2023 (Best Paper).
  https://arxiv.org/abs/2308.04079 · full text https://ar5iv.labs.arxiv.org/html/2308.04079
  · project page https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Reference implementation** (training loop, adaptive density control):
  https://github.com/graphdeco-inria/gaussian-splatting
- **Differentiable rasteriser** (the CUDA forward/backward kernels we are porting the maths of):
  https://github.com/graphdeco-inria/diff-gaussian-rasterization
  Backward pass in particular: `cuda_rasterizer/backward.cu`

### Numbers worth remembering

| Benchmark | PSNR @7K | PSNR @30K | train time @30K (A6000) |
|---|---|---|---|
| Mip-NeRF360 | 25.60 | 27.21 | 41m33s |
| Tanks & Temples | 21.20 | 23.14 | 26m54s |
| Deep Blending | 27.78 | 29.41 | 36m02s |

Those are **100-300 input views**. Do not quote them as a target for a 4-16 view capture.

### Hyperparameters (reference defaults)

```
sh_degree            = 3          # progressive: +1 every 1000 iters from 0
iterations           = 30000      # checkpoints at 7000 / 30000
position_lr          = 1.6e-4 -> 1.6e-6, exponential, x scene_extent
feature_lr           = 0.0025     # SH rest band gets feature_lr / 20
opacity_lr           = 0.025      # original 2023 release used 0.05
scaling_lr           = 0.005      # original 2023 release shipped 0.001 (a bug)
rotation_lr          = 0.001
lambda_dssim         = 0.2        # L = 0.8*L1 + 0.2*(1 - SSIM)
percent_dense        = 0.01
densify_from/until   = 500 / 15000, every 100 iters
densify_grad_thresh  = 0.0002     # L2 norm of SCREEN-SPACE position gradient
opacity_reset        = every 3000 iters, to min(opacity, 0.01)
prune                = opacity < 0.005, screen radius > 20px, world size > 0.1*extent
split                = 2 children, sampled from parent PDF, scale / 1.6, parent removed
```

Parameterisation (get this wrong and the learning rates are meaningless):
scale stored in **log** space, opacity as a **logit** (sigmoid on use), rotation **normalised**
quaternion, covariance never optimised directly but rebuilt as `Sigma = R S S^T R^T`.

Initial values: opacity `0.1`, rotation identity, scale `log(sqrt(nearest-neighbour dist^2))`,
SH DC from point colour and all higher bands **zero**.

### Initialisation sensitivity — why our messy init is acceptable

Paper's own ablation (Table 3): **random init 20.42 dB vs SfM init 26.05 dB** at 30K.
But structured-yet-approximate inits land within ~0.2-0.7 dB of SfM:
- https://arxiv.org/html/2404.12547 — *Evaluating Alternatives to SfM Point Cloud Initialization*
- https://arxiv.org/html/2403.09413 — *Relaxing Accurate Initialization Constraint for 3DGS*
- https://openaccess.thecvf.com/content/ICCV2025/papers/Pan_Liberated-GS_3D_Gaussian_Splatting_Independent_from_SfM_Point_Clouds_ICCV_2025_paper.pdf

**Ours is approximate, not random.** We do not need a perfect merge before training.

---

## 2. The maths we need to transcribe

- **gsplat math supplement** — closed-form backward derivations. The single most useful document
  for writing our WGSL backward pass. https://arxiv.org/pdf/2312.02121
- **gsplat library** (modular CUDA rasteriser; good structural template — separate projection /
  tile-intersect / rasterise-fwd / rasterise-bwd / Adam kernels):
  https://github.com/nerfstudio-project/gsplat · paper http://jmlr.org/papers/volume26/24-1476/24-1476.pdf
- **joeyan/gaussian_splatting MATH.md** — a from-scratch written derivation of every gradient:
  https://github.com/joeyan/gaussian_splatting/blob/main/MATH.md
- **taichi-splatting** — cleanly decomposed differentiable rasteriser, useful as a modularisation
  template: https://github.com/uc-vision/taichi-splatting

### Why backward walks back-to-front

Forward composites front-to-back accumulating transmittance `T`. Only `T_final` and the
last-contributing index are stored per pixel (storing all `T_i` would be huge). Backward
therefore starts at the last contributor and recovers `T_i` from `T_{i+1}` by dividing out each
alpha. That reverse recurrence is inherent to differentiating an alpha composite.

---

## 3. Browser / WebGPU — feasibility and the hard constraint

- **Brush** — Rust + Burn + CubeCL → wgpu → WGSL. **Actually trains 3DGS in a browser tab.**
  MCMC densification. The existence proof and closest reference to our problem.
  https://github.com/ArthurBrussee/brush · demo https://arthurbrussee.github.io/brush-demo
  · write-up https://radiancefields.com/gaussian-splatting-in-browser-brush
- **Efficient Differentiable Hardware Rasterization for 3DGS** — quad/subgroup gradient reduction
  before atomics; >10x faster backward than naive per-pixel atomics. Most transferable
  optimisation idea for us. https://arxiv.org/abs/2505.18764
- **WebSplatter** — wait-free hierarchical radix sort designed because WebGPU gives no guarantees
  on workgroup scheduling order or fine-grained atomic behaviour. https://arxiv.org/abs/2602.03207
- **VkSplat** — Vulkan/Slang compute 3DGS training, closest measured non-CUDA baseline.
  Mip-NeRF360 7 scenes: 412s vs gsplat 1384s; VRAM 3.01 GiB vs 4.56 GiB. WebGPU is a stated
  future backend. https://arxiv.org/abs/2605.00219 · numbers https://harry7557558.github.io/vksplat/

### THE constraint: WebGPU has no f32 atomics

`atomic<T>` is `i32`/`u32` only (https://www.w3.org/TR/WGSL/). The native wgpu float-atomic
feature is **not** exposed to browsers. Per-Gaussian gradient accumulation across every pixel a
splat touches is therefore the central design problem. Options, best first:

1. **Gaussian-major (gather) backward** — one thread per Gaussian gathering from the pixels it
   touched. Needs no cross-thread atomics at all. Costs one extra indirection buffer, which our
   forward tile-binning already produces.
2. **Workgroup-local reduction** then a sparse atomic.
3. **Fixed-point**: scale by ~32768, `atomicAdd` into `i32`, dequantise.
   Pattern: https://toji.dev/webgpu-best-practices/compute-vertex-data.html

### Other WebGPU gotchas

- **TDR / device loss**: a dispatch that runs too long kills the context (~2s on Windows). Submit
  a few iterations per frame and yield. Handle `device.lost`.
  https://toji.dev/webgpu-best-practices/device-loss.html
- **256 invocations/workgroup** default — exactly a 16x16 tile, no headroom. 16 KB workgroup
  storage. https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits
- **Subgroups and f16 are optional features** — request and fall back.
- Browser viewers (rendering only, useful as renderer references, none train):
  antimatter15/splat, mkkellogg/GaussianSplats3D, playcanvas/supersplat,
  https://github.com/cvlab-epfl/gaussian-splatting-web,
  https://github.com/Scthe/gaussian-splatting-webgpu

---

## 4. Sparse views — what to expect at OUR view count

This is the calibration that matters. Vanilla 3DGS **with full training**:

| views | PSNR | dataset |
|---|---|---|
| 2 | 13.89 | Tanks & Temples |
| 3 | 16.94 | LLFF |
| 3 | 14.74 | DTU |
| 6 | 17.70 | Tanks & Temples |
| 100-300 | 27-29 | Mip-NeRF360 |

Depth-regularised sparse-view methods reach **~19-20 dB at 3 views**:

- **FSGS** — depth used as a **Pearson correlation** regulariser (matches shape, ignores absolute
  scale) + proximity-guided densification. LLFF 3-view **19.88 dB** vs vanilla 3DGS 16.94.
  https://arxiv.org/abs/2312.00451 · https://github.com/VITA-Group/FSGS
- **DNGaussian** — global-local depth normalisation; patch-local z-score handles per-view scale
  ambiguity directly. LLFF 3-view 19.12, DTU 3-view 18.91 vs vanilla 14.74.
  https://arxiv.org/abs/2403.06912 · https://github.com/Fictionarry/DNGaussian
- **SparseGS** — depth priors + diffusion SDS + floater pruning. https://arxiv.org/abs/2312.00206
- **CoR-GS** — two fields co-trained; disagreement marks bad geometry.
  https://arxiv.org/abs/2405.12110
- **CoherentGS** — LLFF 3-view 20.33. https://arxiv.org/pdf/2403.19495
- **InstantSplat** — DUSt3R init + short joint optimisation. **Ablation: init-only 26.82 dB vs
  optimised 28.58 dB (+1.76 avg, up to +2.59)** — the cleanest evidence that optimisation matters
  even on top of an excellent init. https://arxiv.org/abs/2403.20309 ·
  https://github.com/NVlabs/InstantSplat
- Survey: https://arxiv.org/pdf/2507.16406

**These two (FSGS / DNGaussian) are the templates for using our monocular depth correctly** — as
a soft shape regulariser during optimisation, never as frozen absolute geometry.

---

## 5. Video and room-scale capture (our actual target)

Dense temporal sampling is materially easier than a few wide-baseline photos: adjacent frames
have near-total overlap, so pose estimation stops being the failure mode.

- **KeyGS** — keyframe-centric: select co-visibility-diverse keyframes, pose those, interpolate
  the rest, then jointly refine. https://arxiv.org/html/2412.20767
- Practical video→3DGS pipeline writeup: https://www.wirelog.net/posts/2025-04-26-video-to-3dgs/
- SLAM-style incremental (relevant to a live stream): **RTG-SLAM**
  https://arxiv.org/abs/2404.19706 · **MonoGS** · **Photo-SLAM** · **SplaTAM**
  https://spla-tam.github.io/ · **MGSO** https://arxiv.org/html/2409.13055v3
  Caveat found: PhotoSLAM/MonoGS are reported weak on large or dynamic scenes.

---

## 6. Feed-forward alternatives (evaluated, not chosen)

Predict Gaussians in one pass, no per-scene optimisation. **None have ONNX exports or browser
deployments**; all assume PyTorch + CUDA. The posed ones still need a pose source.

| method | input | pose needed | size | link |
|---|---|---|---|---|
| MVSplat | sparse multi-view | yes | **12M** (smallest) | https://arxiv.org/abs/2403.14627 |
| DepthSplat | up to 12 views | yes | — | https://arxiv.org/abs/2410.13862 |
| pixelSplat | image pairs | yes | 125M | https://arxiv.org/abs/2312.12337 |
| Splatt3R | stereo pair | **no** | — | https://arxiv.org/abs/2408.13912 |
| AnySplat | uncalibrated multi-view | **no** | — | https://arxiv.org/abs/2505.23716 |
| Splatter Image | single image | no | — | https://arxiv.org/abs/2312.13150 |

Geometry foundation models (would make excellent init, all too large for the browser today):
**DUSt3R** https://arxiv.org/abs/2312.14132 · **MASt3R** https://arxiv.org/abs/2406.09756 ·
**VGGT** (CVPR 2025 best paper, 1B params, 200M/500M variants planned)
https://arxiv.org/abs/2503.11651

MVSplat at 12M is the only plausible browser port, and it needs known poses. Revisit if
`SpawnDev.ILGPU.ML` makes that cheap.

---

## 7. Training-speed work (for later, once it works at all)

- **FastGS** — full T&T scene in ~100s, multi-view-consistency densification.
  https://arxiv.org/abs/2511.04283
- **DashGaussian** — 200s. https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_DashGaussian_Optimizing_3D_Gaussian_Splatting_in_200_Seconds_CVPR_2025_paper.pdf
- **LiteGS** — 13.4x faster. https://arxiv.org/abs/2503.01199
- **Gaussians on a Diet** — 80% lower peak memory, runs on Jetson. https://arxiv.org/abs/2604.20046
- **PocketGS** — on-device mobile training. https://arxiv.org/abs/2601.17354
