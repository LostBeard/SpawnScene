# Quality roadmap: what the best generators do, what we do, what is next (2026-10-07)

TJ, 2026-10-07: "keep working on quality. it is extremely important ... make this the best generator and viewer on the
web." This file is the survey behind PLANS.md's quality section: each technique with what it fixes, the evidence, and
our status. MEASURED rows are ours; the rest are the authors' claims until we measure them.

## How we judge (do not skip)

- **Off the photo path.** Every held-out photo sits next to a training photo, so held-out PSNR cannot see floaters
  that only show when the camera moves (TJ, Bicycle 10-07: "fine at the default spot ... floaters EVERYWHERE" once
  moved). Studio.Wander's views (in/up/low/out/mid around the subject; since 10-07 also pan-0..7 = standing in the rig
  centre and turning around, and over-0/1 = outside the rig looking back) are captured in every project/dataset
  autotest. `tools/compose_wander.py`.
- **Against the reference.** gsplat on the same photos and poses, rendered at the same wander poses (render_turns.py from
  the TURN-POSE lines). Research/README.md + memory ref-gsplat-reference-trainer-windows.
- **Per-photo, not only the mean.** The Bathroom census bug (below) showed as a 13 dB drop on ONE photo; the mean moved
  less. `[Train] final view N ... sup PSNR` lines per photo; diff two runs per photo.
- **On TJ's captures**, not only the benchmark sets: phone photos have auto exposure, HDR merges, mixed orientation and
  thin coverage. The benchmarks have none of that.

## Fixed today (MEASURED)

| What | Effect |
|---|---|
| Floater census counted every splat on a not-yet-opaque pixel as "in front" (3e38 sentinel) | The carve deleted still-thin walls: Bathroom "TONS of holes"; supervised 29.65 vs 33.47 dB with &carve=0, one photo -13 dB. Gate case added (red 0.9960 -> 0.0000). Re-measure of Bicycle / Truck on the fix: k1/k2. |
| End carve left its splats in the file at opacity 0 | Compacted (Bathroom 707K -> 557K splats, 158 -> 125 MB) |

## MEASURED 2026-10-07 late: Bathroom (TJ's 35 phone photos, user path, 7K, llffhold=8, fixed SfM 33/35)

| Run | Options | Held out PSNR | SSIM |
|---|---|---|---|
| g0 | defaults | 15.58 | 0.709 |
| g1 | &depthinit=4 | 16.91 | 0.795 |
| g2 | &depthinit=4 (repeat; edge snap had no colours) | 16.97 | 0.798 |
| g3 | &exposure=1 (mean folded) | 17.43 | 0.759 |
| g4 | &depthinit=4 &exposure=1 | **18.27** | **0.836** |

g1/g2 put the run-to-run noise at ~0.06 dB. Off the photo path (`img/bathroom-g0-vs-g4-depthinit-exposure-2026-10-07.jpg`,
turning around at the rig centre): window, shower corner, curtain and caddy, mirror crisp, walls one colour. Both are
candidates for defaults pending Bicycle (b0/b1, e1) and Truck (e2).

Bicycle360 through the user path (own SfM, 194 photos, llffhold=8, 7K): **b1 `&depthinit=4` held out 25.16 dB / SSIM
0.7661**, 2.46M splats, 1.17M fusion seeds from 193 views in 2.6 s, 192/194 cameras placed. For scale, k1 (same 7K, COLMAP
poses, no depth init) was 25.06 / 0.7662 - no loss on a large outdoor capture. b0 (the baseline) was INVALID: a dropped
fetch of DAv3's weights sent generate down the legacy 2D path (0 training views); fixed in b623d5c, rerun as b0r.

## The candidates

### 1. Photometric: exposure and colour per photo - HIGH for phone captures

Phones auto-expose, tone-map and white-balance every shot independently. TJ's Bathroom: shutter x ISO spans 0.16 to 10.9
(typical 0.8-4.3, ~6x), and 4 of 35 are HDR merges. A scene that must match every photo exactly explains brightness
differences with geometry: floaters hugging the camera that darken or lighten one view. The literature calls these
floaters by that cause.

- **Reference 3DGS (Inria, 2024 update): per-image affine exposure** (3x4 matrix on the rendered RGB, lr 0.01 ->
  0.001, `--train_test_exp` fits test images' exposure on their left half). Cheapest option. We have none.
- **Bilateral grid** (Wang et al. 2024, "Bilateral Guided Radiance Field Processing", in gsplat as
  `--use_bilateral_grid`): a per-view locally-affine colour transform; models local tone mapping too. "Largely free from
  artifacts like floaters" on phone captures.
- **PPISP** (NVIDIA, Apache 2.0, going into gsplat and 3DGRUT): exposure, vignetting, white balance and a camera
  response curve as separate physical modules, plus a controller that predicts them for novel views.

Our plan: per-image affine first (12 parameters a photo, one extra pass between render and loss, gradients summed per
view, the inverse applied to dL/dpixel). Bathroom is the test case; a held-out photo's exposure is fitted on half of it
as the reference does, or scored after a per-image affine fit.

### 2. Geometry priors from monocular depth - HIGH for rooms, thin coverage

We already run Depth Anything V3 on every photo for poses and initialisation, then throw its depth away during
training. Rooms are where photometric loss alone fails: few photos per wall, plain surfaces.

- **Reference 3DGS `-d`**: L1 between the rendered INVERSE depth and the mono inverse depth (scale/offset fitted per image
  to the SfM points, `utils/make_depth_scale.py`), weight 1.0 decaying to 0.01 over training. README: big gains on
  Deep Blending (indoor), small or negative elsewhere. That pattern is exactly ours: indoor = Bathroom, DrJohnson,
  Playroom.
- DN-Splatter (WACV 2025): depth and normal priors with an image-gradient-aware weight; meshes too.
- IndoorGS (CVPR 2025): planes and lines as geometric cues for indoor rooms.
- Dense init from mono depth (several 2025 papers): back-project aligned mono depth, voxel-downsample, use as the initial
  splats where SfM is sparse. Bathroom starts from 7,793 SfM splats (h1).

Our plan: (a) dense init from DAv3 depth aligned to the SfM points where the SfM cloud is thin; (b) the inverse-depth
L1 with the reference's decaying weight, `&depthreg=1` first, default if indoor sets gain and outdoor sets do not lose.

### 3. Density control - MEDIUM (we are at parity)

MEASURED: AbsGS signal default (Bicycle +0.43 dB, fewer splats); MCMC lost our A/B; Pixel-GS opt-in, no gain measured.
gsplat's own table on Mip-NeRF 360 (A100): default 29.00 dB / 3.2M, absgrad 29.11 / 2.5M, MCMC at 3M 29.65 / 3.0M.
MCMC's win in gsplat is at an EQUAL or larger budget from a full start; ours started from 49.6K SfM points with 5%
growth and could not reach the budget by 7K. Worth a second look at 30K only.

- Revising Densification (Bulo et al., ECCV 2024): densify where the per-pixel ERROR is, a primitive budget, and a
  correction for the opacity bias that cloning introduces. Error-driven growth is the principled fix for "few splats
  where the photo is wrong", which is what thin coverage looks like.
- Taming 3DGS (2024): score-based, budgeted, purely constructive growth; 4-5x smaller and faster at equal quality.
  Its accelerated optimiser is now in the reference repo (1.6x; sparse Adam 2.7x).

### 4. Floaters off the photo path - MEDIUM (have a census carve; keep validating)

Ours: the census carve (front share >= 0.9 of the blending weight in front of the photos' surfaces) + size cap 0.1 x rig
radius + far background left at the cap. Remaining class: the far background's depth ambiguity from a low, look-up pose
(Bicycle). Candidates: a far-depth prior (sky / background sphere: initialise and keep far content on a shell), opacity
regularisation (gsplat MCMC's 0.01 opacity and scale regularisers), and exposure compensation (1) for floaters whose
cause is photometric.

### 5. Anti-aliasing and zoom - MEDIUM for the viewer

- Mip-Splatting (CVPR 2024 best student paper): a 3D smoothing filter capped at each splat's highest sampling rate in
  the training views + a 2D Mip filter replacing the dilation. Fixes the "needles and holes when zooming in / shimmer
  when zooming out" that a user sees immediately in a viewer. The reference repo has the EWA (2D) part as
  `--antialiasing`. We have a `MipScaleFloor` pass (`&mipfilter`) - default off; measure on the wander views at in/out.
- gsplat table: antialiased 29.03 vs 29.00 on the benchmark (the benchmark cannot show it: same resolution and
  distance as training). The gain is off-path, our kind of measurement.

### 6. The viewer itself - MEDIUM

- **Popping**: one depth per splat makes the global sort flip as the camera turns. StopThePop (SIGGRAPH 2024):
  hierarchical per-pixel sort, 4% slower, and consistent enough to halve the splat count. Needs a per-tile sort in the
  rasteriser - the trainer's tile rasteriser already has the structure.
- **Streaming and LOD**: done (LOD tree, .spawnscene v3, paged cut). Spark 2.0 (World Labs) is the comparison: RAD
  format, LoD splat tree, shared LRU pager, ExtSplats; formats PLY/SPZ/SPLAT/KSPLAT/SOG. We read PLY and SPLAT - add SPZ
  and SOG import so people can open what others publish.
- Brush (Arthur Brussee, Burn + wgpu): trains in the browser too (Chrome 134+ Windows/macOS), "faster than gsplat".
  The closest competitor to SpawnScene's generator; no published quality numbers to compare.

### 7. Capture guidance - HIGH value, cheap

A room needs photos looking at every wall from at least two places. Bathroom: 24 of 34 cameras placed by the global
init, 10 re-registered, 2 dropped, a landscape photo skipped by the multi-view depth pass; the pan views show which
headings have photos at all (`[Wander] pan-k: N/32 photos look within 30 deg`). Show the user the same thing: a
coverage map after SfM, and the dropped photos by name with why.

## Order (what moves TJ's scenes most per day of work)

1. Re-measure the carve on the fix (Bathroom h6-h9, Bicycle k1, Truck k2); retune the unseen bar.
2. Per-image exposure (affine) - Bathroom first.
3. DAv3 depth: dense init where SfM is thin, then the decaying inverse-depth L1.
4. Mip-Splatting 3D filter default-on if the wander in/out views gain.
5. The landscape-photo skip in the multi-view depth pass; dropped-photo report in the UI.
6. StopThePop-style per-tile sort in the viewer.

## Sources

- Reference 3DGS repo (depth regularisation, exposure, antialiasing, Taming optimiser): https://github.com/graphdeco-inria/gaussian-splatting
- gsplat feature ablation (Mip-NeRF 360): https://docs.gsplat.studio/main/tests/eval.html
- Bilateral Guided Radiance Field Processing: https://arxiv.org/abs/2406.00448
- PPISP: https://radiancefields.com/nvidia-announces-ppisp-for-radiance-fields
- Robust Gaussian Splatting (blur, poses, colour for phone captures): https://arxiv.org/abs/2404.04211
- Revising Densification: https://arxiv.org/abs/2404.06109
- Taming 3DGS: https://arxiv.org/abs/2406.15643
- StopThePop: https://arxiv.org/abs/2402.00525
- DN-Splatter: https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf
- IndoorGS: https://openaccess.thecvf.com/content/CVPR2025/papers/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.pdf
- Spark 2.0: https://sparkjs.dev/docs/new-features-2.0/
- Brush: https://github.com/ArthurBrussee/brush
