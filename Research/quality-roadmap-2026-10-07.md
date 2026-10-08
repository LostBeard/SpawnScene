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

Bathroom c0 (`&depthinit=4`, seeds now COLOURED from the device decode) held out 16.64 / 0.792, c1 (+ `&edgesnap=1` on the
fusion depth) 16.57 / 0.7915, vs grey-seeded g1/g2 16.91-16.97 / 0.795-0.798; the Wander views of g2 and c0 are
indistinguishable. Seed colour is neutral (training repaints it within the first cycle); the edge snap does nothing for
the multi-photo seeds (agreement between two views already rejects ramp samples).

**b0r (Bicycle user path, no depth init, rerun of the invalid b0):** held out 25.17 / 0.7649, 2.14M splats - vs b1
`&depthinit=4` 25.16 / 0.7661, 2.46M. Depth init is neutral outdoors and +1.3 dB on the Bathroom (g0 -> g1): a default
candidate for TJ.

**Exposure on fixed-exposure captures (2026-10-08):** e1 Bicycle `&depthinit=4&exposure=1` held out 24.55 / 0.7663 vs b1
(depthinit only) 25.16 / 0.7661; e2 TruckFull `&exposure=1` 23.87 / 0.8587 vs k2 24.07 / 0.8592. SSIM unchanged, PSNR
down: a colour offset. Every photo learned a small positive offset (Truck mean +0.009..+0.017) and an offset folds into
the scene only where it is opaque. `&exposure=gains` (per-channel gains only, gate-checked) queued as h0-h2 on all three.
Not a default until it keeps Bathroom's gain without costing the benchmarks.
**h0 Bathroom `&depthinit=4&exposure=gains`: held out 18.64 / 0.844** vs d0 (same build, full affine) 18.23 / 0.840 -
gains-only is BETTER on the phone capture too (+0.4 dB). **h1 Bicycle `&depthinit=4&exposure=gains`: 24.94 / 0.7682**
vs b1 (no exposure) 25.16 / 0.7661 and e1 (full affine) 24.55 / 0.7663 - most of the affine's loss recovered, best SSIM,
still -0.22 dB PSNR. Part may be the scoring: held-out photos are rendered at the photos' MEAN exposure, while the
reference fits each held-out photo's exposure (on half the image) before scoring. Fair scoring is now logged for every run ("held out RIGHT HALF", d7e394f); the j chain reruns the Bicycle and
Bathroom pairs with it. **h2 TruckFull `&exposure=gains`: 24.00 / 0.8598** vs k2 (none) 24.07 / 0.8592 and e2 (affine)
23.87 / 0.8587 - neutral within noise; the folded offsets are exactly 0, as designed. Summary so far: gains-only wins
the phone room (+0.4 dB over the affine, the best Bathroom number), costs nothing on Truck, 0.2 dB PSNR (SSIM up) on
Bicycle under the mean-exposure score.

**Fair score (j chain, 2026-10-08; right half of each held-out photo, per-photo gains fitted on its left half - the
reference's train_test_exp protocol, every run):**

| run | options | held out (mean exposure) | right half as rendered | right half, gains fitted |
|---|---|---|---|---|
| j0 Bicycle | depthinit=4 | 25.14 / 0.7666 | 25.135 | 25.416 |
| j1 Bicycle | depthinit=4 exposure=gains | 24.99 / 0.7683 | 24.959 | **25.575** |
| j2 Bathroom | depthinit=4 | 16.88 / 0.7958 | 17.097 | 18.877 |
| j3 Bathroom | depthinit=4 exposure=gains | **18.57 / 0.8439** | **19.418** | **24.316** |

Bicycle's photos vary in exposure too (fitted gains 0.91-1.13), so the mean-exposure score penalised the run that
modelled it: under the fair score gains-only WINS Bicycle (+0.16 dB, SSIM +0.0017).
On the phone room gains-only wins by +1.7 dB held out and +5.4 dB under the fair score: without it each photo's
exposure is baked into the scene (fitted gains 0.89-1.30 for the same four photos).

**PROPOSAL for TJ (not applied): make `depthinit=4` + `exposure=gains` the defaults.** Evidence: Bathroom (phone room)
+1.3 dB from depth init (g0 -> g1) and +1.7 dB / +5.4 fair from gains (j2 -> j3); Bicycle depth init neutral (b0r 25.17
vs b1 25.16), gains +0.16 fair (j0 -> j1); Truck gains neutral (h2 24.00 vs k2 24.07); off-path Wander views equal or
better in every pair looked at. Cost: depth fusion 2.6 s on Bicycle and ~15% more splats. The full affine
(`exposure=1`) stays an option; it loses on fixed-exposure captures (offsets do not fold).

**Mip-Splatting 3D filter on the proposed defaults (k0/k1, same build as j1/j3):** Bicycle `&mipfilter=0.2` 24.97 /
0.7682, fair 25.569 (j1 24.99 / 0.7683, 25.575); Bathroom 18.62 / 0.8439, fair 24.157 (j3 18.57 / 0.8439, 24.316).
Neutral on held-out photos, as expected, and the Wander "in" views (half way to the subject, ~2x) look the same - not
close enough for the filter's case (needles when zooming far past the photos). Stays opt-in; a fair test needs a
close-up view set (4-8x) or a render at another resolution, which the Wander set does not have yet.
**m0/m1 (fresh build, Bicycle, proposed defaults +/- `mipfilter=0.2`):** m0 24.97 / 0.7683, fair 25.572 (reproduces
j1's 25.575 - same-config noise ~0.003); m1 25.02 / 0.7684, fair 25.638: +0.07 dB, at the level of the different-run
spread seen before (g1/g2 0.06). The new close-{q} Wander views (85% toward the rig's subject point) did not test
magnification: on Bicycle they end up beside the bike looking past it at the hedges, at normal distance, and look the
same in both runs. A real close-up set must aim at a surface point and stop at a fixed fraction of the photos' distance
to IT. Verdict: mipfilter is harmless and possibly slightly positive; not enough to make it a default yet.

### Depth supervision (PLANS 3b), built 2026-10-08, opt-in `&depthloss=X` (with `&depthinit`)

L1 between the rendered inverse depth sum(w/z) and each photo's DAv3 depth scaled by DepthFusionInit (kept per view,
resampled to <= 384 px; Bicycle 193 views ~75 MB), weight X -> X/100 over training as the reference does. Inverse depth
blends as a fourth channel in the depth-supervised builds of the raster / scatter / geometry passes (`//DEPTH:` lines,
SplatTrainerShaders.DepthVariant): the default pipelines are byte-identical (TrainerShaderValidationTests runs naga on
all 44 shader builds). TrainerGate depth stage: rendered inverse depth inside the scene's 1/z range; analytic
dL/d(position) along the view axis vs central finite differences of the step's loss, 5 splats within 3%, cos 1.000;
150 position-only steps halve the depth error (0.155 -> 0.084). Red check: the new -g/z^2 term with its sign flipped
fails the gate (3/5 within 25%). A/B queued: d0 = g4's options, d1 = + `&depthloss=1`.

**Measured (Bathroom, 7K, same AOT build):** d0 `&depthinit=4&exposure=1` held out 18.23 / 0.840 (= g4, 18.27 / 0.836,
reproduced); d1 + `&depthloss=1` 18.20 / 0.8375, 0.89M splats vs 0.76M. Off the photo path (Wander) the two are alike, d1
a little fuller in the up views and with one black gap in pan-2 where d0 has grey smear. Neutral at 7K with depth init
already seeding the walls: it stays opt-in. Open: the 1500-iteration smoke run reached 18.65 / 0.836 - a short schedule
with depth may match the 7K one; a d2 at 1500/3000 without depth would say whether that is the loss or just fewer steps.
**Answered (i0/i1, Bathroom 1,500 its, depthinit + exposure):** i0 without depth loss 18.48 / 0.831, i1 with 18.58 / 0.834
(d0 at 7K: 18.23 / 0.840). It is the schedule, not the loss: a thin 33-photo room's held-out PSNR saturates by 1,500
iterations (133 s vs 300 s), 7K adds only SSIM. Depth loss +0.1 dB at 1,500 - at the noise floor. Stays opt-in. A
capture-size-aware schedule (stop when held-out / a validation view stops improving) is a candidate for thin captures.

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

## Carve re-baseline h6-h9 (Bathroom, 7K, llffhold=8, 2026-10-07 evening - OLD defaults: no depth init, no exposure)

| Run | Setting | held out PSNR / SSIM | supervised | unseen splats carved |
|---|---|---|---|---|
| h6 | carve on (unseen bar 1 px) | 15.57 / 0.7116 | 32.44 | 159,874 |
| h7 | `&carve=0` | 15.42 / 0.7135 | 36.53 | - |
| h8 | `&carveunseen=0` (floaters only) | 15.23 / 0.6999 | 33.51 | 0 |
| h9 | `&carveunseenpx=0.05` | 15.19 / 0.7075 | 33.49 | 22,719 |

All within 0.4 dB held out (4-5 held-out photos: noise level); the carve costs 4 dB supervised for nothing measurable
held out. Recorded 10-08; repeated as n0-n3 on the 10-08 defaults with the fair score before deciding (PLANS item 1).

## Carve re-baseline n0-n3 (Bathroom, 7K, llffhold=8, 2026-10-08 - the 10-08 defaults: depthinit=4 + exposure gains)

| Run | Setting | held out PSNR / SSIM | fair (right half, gains on left) | splats |
|---|---|---|---|---|
| n0 | carve on, unseen bar 1 px (default) | 18.69 / 0.8458 | 24.75 | 769K (209,246 unseen carved) |
| n1 | `&carve=0` | 18.68 / 0.8416 | 24.11 | 969K |
| n2 | `&carveunseen=0` (floaters only) | 18.67 / 0.8460 | 24.90 | 970K |
| n3 | `&carveunseenpx=0.05` | 18.70 / 0.8471 | 24.87 | 934K (37,269 unseen carved) |

- The floater carve earns its keep on the room: +0.64-0.79 dB fair over no carve, SSIM +0.004.
- The unseen carve (1 px) removes 21% of the splats at no measurable cost (fair 24.75 vs 24.90, noise level on 4 views).
- Off the photo path (Wander pan-0..7, `_shots/dataset/Bathroom__tuvok-n0_tuvok-n1_tuvok-n2_pan__wander.png`): the
  well-covered headings are identical; the headings 2/33 photos face (pan-1..3) show the same dark smears in all three -
  a coverage problem, not the carve's.
- **Decision: defaults stay (carve on, unseen bar 1 px).** PLANS item 1 closed.

## The black off the photo path is HOLES - random training background fixes most of it (2026-10-08)

Bathroom, 10-08 defaults, 7K, llffhold=8, captured over MAGENTA (`&bg=1,0,1`, harness):
- p0 (black training background, as shipped): the dark smears at the headings few photos face turn magenta - they are
  pixels the splats do not cover, the black background showing through. Even well-photographed surfaces are partly
  see-through: 0.1-34% magenta per Wander view, mean 12.0% over 34 views.
- Why: training composites over black, so a half-transparent wall matches the photos as well as a solid one.
- q0 (`&randombg=1`: each training step composites over a random colour, the reference's --random_background; gate:
  pure-L1 control and the background term vs finite differences both cos 1.000, mutant cos -0.39):
  mean 4.1% magenta; 27 of 34 views at <= 1% (shelves, curtain, mirror, towel, walls solid). What remains (the "-3"
  views and pan-2, 11-31%) faces where hardly any photo looked - real coverage gaps.

| Run | training background | held out PSNR / SSIM | fair | magenta (holes), mean of 34 views |
|---|---|---|---|---|
| n0 | black | 18.69 / 0.8458 | 24.75 | - |
| p0 | black (same settings as n0: run-to-run noise ~0.25 dB, 0.4 fair) | 18.45 / 0.8439 | 24.37 | 12.0% |
| q0 | random | **18.83 / 0.8496** | **25.17** | **4.1%** |

Side by side: `_shots/dataset/Bathroom__tuvok-p0_tuvok-q0_pan__wander.png`. Bicycle (r0 vs m0) and Truck (t0/t1)
running before proposing it as a default (TJ's call).

### Bicycle with the [0, 1] random background: a LOSS (r0, 2026-10-08)

| Run | held out PSNR / SSIM | fair |
|---|---|---|
| m0 (black, today's defaults) | 24.97 / 0.7683 | 25.572 |
| r0 (`&randombg=1`, [0, 1]) | 24.60 / 0.7519 | 25.302 |

-0.37 dB held out, SSIM -0.016, fair -0.27 (Bicycle's protocol noise is ~0.003 fair). Off-path views look alike; r0 is a
shade darker and its exposure gains are ~2% higher on every photo. Why: a [0, 1] background has mean 0.5, so wherever
transmittance stays above 0 (sky, thin foliage edges) training darkens the colours by T x 0.5 to match on average -
and the scene is scored and shown over black. Rooms do not pay it because their walls go opaque. Next: `&randombg=2`,
a zero-mean background ([-0.5, 0.5]: same variance, so transparency costs the same; expected composite = the render
over black) - u0 Bicycle, u1 Bathroom (holes over magenta), t2 Truck.

Truck (benchmark protocol, COLMAP poses, 7K): t0 black 23.97 / 0.8594, fair 24.197; t1 `&randombg=1` 23.96 / 0.8588,
fair 24.189 - neutral.

### Zero-mean random background (`&randombg=2`, [-0.5, 0.5]) - u0 / u1 / t2

| Scene | black | `&randombg=1` [0, 1] | `&randombg=2` zero-mean |
|---|---|---|---|
| Bathroom held out / fair | 18.45-18.69 / 24.37-24.75 (p0, n0) | 18.83 / 25.17 (q0) | 18.86 / 25.15 (u1) |
| Bathroom holes (magenta, mean of 34 views; views <= 1%) | 12.0%; 1/34 | 4.1%; 27/34 | 4.0%; 24/34 |
| Truck held out / fair | 23.97 / 24.197 (t0) | 23.96 / 24.189 (t1) | 23.99 / 24.209 (t2) |
| Bicycle held out / fair | 24.97 / 25.572 (m0) | 24.60 / 25.302 (r0) | 24.78 / 25.445 (u0) |

Zero mean removes the colour bias (u0's exposure gains match m0's: 1.130 1.049 1.135 vs 1.127 1.048 1.125) and keeps the
whole room gain; Truck neutral; Bicycle still -0.19 dB held out / -0.13 fair - the push to opacity itself, where the
sky cannot be opaque. Next: v0/v1 at half width (`&randombgamp=0.5`).

### Half width (`&randombg=2&randombgamp=0.5`, [-0.25, 0.25]) - v0 / v1

| Scene | black | zero-mean full | zero-mean half |
|---|---|---|---|
| Bicycle held out / fair | 24.97 / 0.7683 / 25.572 (m0) | 24.78 / 0.7610 / 25.445 (u0) | 24.89 / 0.7656 / 25.535 (v0) |
| Bathroom held out / fair | 18.45-18.69 / 24.37-24.75 | 18.86 / 25.15 (u1) | 18.82 / 0.8499 / 24.76 (v1) |
| Bathroom holes (mean; views <= 1%) | 12.0%; 1/34 | 4.0%; 24/34 | 5.4%; 23/34 |

Half the push keeps most of the room fix and costs Bicycle -0.08 dB held out / -0.04 fair. Truck t3 running.

