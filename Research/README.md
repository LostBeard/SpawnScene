# Research: 3D Gaussian Splatting

Sources gathered 2026-09-20 while working out why SpawnScene's reconstruction was stuck at
~13 dB PSNR. Organised by what each is useful FOR, with the load-bearing numbers inline so you
do not have to re-fetch a paper to remember why it is here.

**The one-line conclusion:** depth fusion is only an *initialisation*. 3DGS quality comes from
the per-scene optimiser. No surveyed method reports competitive quality without one.

---

## 0. Other documents here

- **`validation-strategy.md`** — how to measure anything when no link in the chain is proven yet.
  Read this before trusting a number: two wrong components can agree with each other, and on
  2026-09-21 they did, for a whole session.
- **`datasets.md`** — what data we have, and what question each one can actually answer.

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

---

## 8. Measured on this project (keep this current)

Held-out novel views on TempleRing, scored by `tools/score_novel_view.py` against the real
photographs, rendered by the display renderer at the ground-truth poses.

| change | PSNR dB | SSIM | MVS depth agreement |
|---|---|---|---|
| depth fusion only, no optimiser | 13.06 | 0.4600 | 0.8% |
| + colour and opacity optimised, 1600 iters | 16.08 | 0.5368 | 0.8% |
| + source photographs turned upright before depth | 18.31 | 0.5943 | 2.2% |

Reference points for what to expect, from section 2: vanilla 3DGS **with full training** gets
~16.9 dB at 3 views and ~17.7 at 6; depth-regularised sparse-view methods reach ~20 dB at 3
views; 27-29 dB needs 100-300 views. TempleRing gives us 16.

### Bathroom — an unposed room, 2026-09-21

Held-out PSNR **at initialisation**, before any optimisation. Init is quoted rather than the final
number because it reproduces to 0.09 dB between runs while the final number varies by 1.40 dB.

| change | held-out init, dB |
|---|---|
| SfM poses, 22 of 35 views posed | 6.54 |
| chunked DAv3 poses, spread anchors, 14 posed | 7.28 |
| + anchors chosen by measured view OVERLAP, 34 posed | 8.11 |
| + keep splats the screening reference cannot SEE | **12.41** |

12.41 dB before any optimisation is higher than any FINISHED reconstruction this capture had
produced (previous best 9.70). The remaining problem moved from the geometry to the optimiser.

⚠ Bathroom has no ground-truth poses, so none of these numbers attribute to poses or optimiser
separately. See `datasets.md`.

### DAv3 multi-view poses, measured against ground truth

`?autotest=dav3-pose` fits recovered cameras to TempleRing's calibration. On an RTX 40-series:

| views per forward | poses returned | error vs ground truth | same cameras across two batches |
|---|---|---|---|
| 6 | 6/6 | 5.4% / 4.5% of camera spread | **0.4%** |
| 8 | 8/8 | 5.2% / 9.6% | 0.9% |
| 10 | none — GPU memory | — | — |

So the poses are good and **not batch-dependent**, which is what makes chunked inference viable:
run the model repeatedly with shared anchor views and fold each pass into the first one's frame.

The cap was a `const 6` with no measurement behind it. The ceiling is device-dependent, so the
answer is to PROVE it on the device at run time and back off, not to pick a better constant.

### Things this project measured that the papers do not discuss

- **Feed a monocular depth model an upright picture.** Every TempleRing photo is a quarter turn
  off level. Correcting it nearly tripled the fraction of pixels where independent per-view
  depths agree in 3D (0.8% -> 2.2%). Nothing in the pipeline was wrong - the calibration is
  self-consistent with the rotated pixels - so it produced no error, only worse depth. Any
  capture pipeline taking video from a handheld device has this problem and will not be told
  about it.

- **Check the orientation PER VIEW, and do not generalise from a sample.** TempleRing is not
  uniformly rotated: 31 of its 47 entries need one counter-clockwise quarter turn and 16 need
  three, because the camera ring flips partway round. Sampling seven entries and concluding
  "all 47" was wrong, and an implementation built on that conclusion would turn a third of the
  dataset upside down.

- **Fixed-point quantisation is a real constraint on gradient precision, not just on range.**
  WebGPU has no float atomics, so per-splat gradients cross an i32 atomic scaled by 2^20. With
  an L1 loss averaged over the image, dL/d(pixel) is 1/(3*W*H) - about 1e-6 at 640x480 - so a
  small splat's gradient is only a few quanta. The GPU gate therefore reports agreement in
  QUANTA as well as relatively; at 128x96 the whole 2D gradient set agrees to 0.38 quanta mean,
  which reads as a 3% relative error and is entirely rounding.

- **The depth sort key needs the scene's actual depth range.** Quantising `depth * 1024` into an
  18-bit field spent ~400 of 262143 levels on a scene 0.4 deep, so distinct splats collapsed onto
  one key and composited in whatever order the atomic allocator handed out.

- **PSNR does not see a sparse reconstruction melting.** Across one training run held-out PSNR
  moved 12.50 -> 12.21 dB, essentially flat, while the render went from a recognisable room to
  fog. PSNR over a sparse scene is dominated by large smooth regions, so smoothing structure away
  barely moves it; SSIM fell 0.6264 -> 0.5782 over the same run. Any sparse-view work reporting
  PSNR alone can be improving the number while destroying the reconstruction.

- **A consistency screen written for an object throws away a room.** Screening each view's splats
  against a reference view's depth and dropping whatever falls outside that view's frustum is
  correct for a turntable capture, where out-of-frustum means the far side of the object. For a
  room, out-of-frustum IS the other walls - the entire reason the extra views were added. Keeping
  them took the kept fraction from 3% to 62% and the held-out initialisation from 8.11 to 12.41 dB.
  A splat the reference cannot see is UNVERIFIED, not wrong.

- **Anchor views for multi-pass inference should be chosen by OVERLAP, not by spread.** Spreading
  anchors evenly across a capture is right for an orbit, where every frame sees the subject, and
  close to the worst possible choice for someone walking through a room, where the frames furthest
  apart in time are the least likely to have seen the same wall. Spread anchors: 0 of 10 passes
  could be folded into a common frame. Overlap-chosen anchors: 10 of 10, at 1.5-3.4% residual.

- **A reconstruction has no gravity in it, and every viewer assumes one.** DAv3 and COLMAP both
  recover geometry up to an arbitrary rotation. Bathroom's reconstruction came out with its up
  vector at essentially -Y, and the display camera controller rebuilds its up as world +Y on every
  frame - so the room rendered on its side and tumbled when the camera moved, while every number
  stayed good, because the TRAINER renders from the real camera basis. Estimating up as the mean of
  the cameras' own up vectors works for anything a person carries (Bathroom agreement 0.913) and
  must be refused for a rig that rolls the camera (TempleRing 0.491, 90 degrees off).

- **Adam steps splats that have no gradient.** With batch size 1 over a round robin of views, a
  splat visible in one view of 26 takes ~25 steps per cycle on a gradient of exactly zero, dragged
  by momentum decaying at 0.9. Measured: **89.8%** of splats take such a step each iteration, and a
  splat seen once drifts 0.046 in summed colour+opacity over the next 10. The geometry Adam pass in
  this codebase guards against it; the colour/opacity one did not, on the stated grounds that it is
  "harmless for colour". Whether that is true at batch size 1 is being measured, not assumed.
