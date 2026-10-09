# Brush v0.3.0 vs SpawnScene: training differences, source level (2026-10-09)

Question: on the 7K-end benchmark (same JPGs, every 8th held out, one scorer) Brush v0.3.0 beats us on Train (21.07 vs
19.52), Bonsai (30.47 vs 29.76), Truck (24.71 vs 24.31) and Kitchen (29.18 vs 28.62), and trains about 2.7x faster
(Counter: 172 s wall vs our 462 s of training). Ours = c1 (`&poslrsteps=7000 &exposure=0`). Which concrete differences
could explain the quality and the speed?

Method: read-only. No build, no GPU work, no training run. Brush source = the v0.3.0 release `source.tar.gz` (`gh release
download v0.3.0 -R ArthurBrussee/brush`), extracted to `<scratch>/brush-src/x/lpips-convert-0.3.0/` (the tarball's top
folder has that name but holds the whole workspace). Brush paths below are relative to its `crates/`. Ours are relative to
`SpawnScene/SpawnScene/`. One CPU-only Python script read the COLMAP binaries to compare learning-rate scales (section 2).
Speed numbers come from Tuvok's `&trainprofile=1` Counter run, which was still running while this was written
(`_runs/tuvok-prof-Counter360.log`). I did not run it. The `[Densify] apply probe` lines come from the finished c1 logs.

This extends [parity-references-2026-10-08.md](parity-references-2026-10-08.md) and
[opacity-vs-gsplat-2026-10-08.md](opacity-vs-gsplat-2026-10-08.md). Rows that only repeat those documents are left out.

How Brush was run (parity-matrix.md): `brush_app.exe <folder> --total-steps 7000 --eval-split-every 8 --eval-every 7000
--eval-save-to-disk`, with every training flag at its default. So every Brush schedule below decays over **7000** steps.

## 0. Corrections to earlier notes

- **Brush v0.3.0 trains on a black background, not a noisy one.** `brush-train/src/train.rs:116-118`: `let background =
  Vec3::ZERO;` ("Could generate a random background color, but so far results just seem worse"). parity-references
  gotcha 8 (`background_noise_strength` 0.1) describes `main`, not the release we benchmarked.
- **Brush v0.3.0's growth fraction is 0.1, not 0.2** (`brush-train/src/config.rs:61-62`). Brush also draws the growers
  at random, weighted by their refine weight (`train.rs:415-426`, `multinomial.rs`). It does not keep the top fraction.
  The `&densifyfrac` comment at `Pages/Studio.razor.cs:687` ("Brush: 0.2") and the top-fraction selection in
  `Services/GpuDensify.cs:170` therefore do not reproduce Brush.
- "Brush grows MCMC-like" (parity-matrix.md) is only half right. It does **relocate** dead splats onto live ones,
  sampled by opacity (`train.rs:385-396`). Its position noise, however, is about 1e-4 of MCMC's (section 1, row 14).
  It also has no MCMC opacity/scale regularisers at MCMC strength (its weights are 1e-9 / 1e-8).

## 1. Differences table

"Ours" means the c1 parity run: dataset path, `GEOM=1 GTPOSES=1 INIT=points DENSIFY=100 OPACITYRESET=3000 DENSIFYFRAC=1`,
`&poslrsteps=7000&exposure=0` (`_runs/tuvok-cand.sh`).

### Optimiser and schedules

| # | Parameter | Brush v0.3.0 | SpawnScene (c1) | Expected effect |
|---|---|---|---|---|
| 1 | **Position lr** | `2e-5 x median_size` -> `1e-6 x median_size` (x0.05), exponential over total_steps. `median_size` = middle axis (full width) of the 75th-percentile box of the current splats, recomputed every refine. config.rs:15-20; train.rs:70, :78-79, :230, :434; render bounding_box.rs:23-27 | `1.6e-4 x extent` -> x0.01 over `poslrsteps` (7000). `extent` = 1.1 x max train-camera distance from the centroid. Studio.Training.cs:540, :733-739, :263; TrainingSchedule.cs:55-64 | **Ours is 17.3x Brush's at the start and 3.5x at the end on Train and Kitchen.** It is 13.8x / 2.8x on Counter, 7.5x / 1.5x on Truck, 6.1x / 1.2x on Bonsai, and only 2.9x / 0.6x on Bicycle, where we win (section 2). Smaller Adam steps let centres settle. Our one measured big win on Train was making the late position lr smaller (+0.57 dB, pl1). **High.** |
| 2 | Scale lr | 1e-2 -> 6e-3, exponential over total_steps. config.rs:39-44; train.rs:81-82, :234 | 5e-3, constant. Studio.Training.cs:547 | Brush starts at 2x ours and ends at 1.2x. Brush scales can fit faster early. Medium. |
| 3 | Opacity lr | 1e-2 (on the logit) | 0.025. SplatOptimizer.cs:24 | Ours is 2.5x. tr2 (0.05) cost Train -0.09, so the direction toward 0.01 is untested. Low-medium. |
| 4 | SH lrs | DC 2e-3, rest DC/20 = 1e-4. config.rs:27-32; train.rs:244-257 | DC 2.5e-3, rest /20 = 1.25e-4. SplatOptimizer.cs:21; SplatTrainerGpu.cs:2211 | Ours is 1.25x. Low. |
| 5 | **SH degree schedule** | **Full degree 3 from step 0**: `with_sh_degree(3)` once, no ramp anywhere. brush-process/src/train_stream.rs:105; brush-dataset/src/config.rs:6-7 | +1 every 1000 steps (0 until 999, 3 from 3000). SphericalHarmonics.cs:216-217; Studio.Training.cs:726 | Brush fits view dependence for all 7000 steps, we fit it for 4000 at full degree. Train has strong view-dependent sky/metal. Medium-low (unverified). |
| 6 | Rotation lr | 1e-3 | 1e-3. Studio.Training.cs:548 | Same. |
| 7 | Adam | betas 0.9/0.999, eps 1e-15, dense, per-parameter bias correction; new rows get zero moments. train.rs:66-68; adam_scaled.rs:121-164 | Same values, dense. SplatTrainerShaders.cs:1364-1374 | Same. |
| 8 | Loss | `0.8 L1 - 0.2 SSIM` (equals 0.8 L1 + 0.2 D-SSIM). SSIM is 11-tap, sigma 1.5, **zero-padded "same"** convolution over all 3 channels. train.rs:154-158; ssim.rs:28-58 (file lines) | 0.8 L1 + 0.2 D-SSIM, **valid** windows only, per channel. SplatTrainerGpu.cs:2061-2123 | Brush's border pixels get an SSIM gradient, ours do not. Low. |
| 9 | Aux losses | Opacity loss `1e-9 x sum(raw_logit x (visible + 1e-3))`. Scale loss `1e-8 / median_size x sum(scale x (visible + 1e-3))`. Both weighted `clamp(0.9 - t, 0, 1)`, so they stop at step 6300. train.rs:142-144, :187-205 | None (MCMC only). Studio.Training.cs:462-463 | With Adam and eps 1e-15, a splat with no photometric gradient still steps at the full lr. Brush splats that are never visible slide toward transparency and get pruned at 2/255. Low-medium (cleans up unseen splats). |
| 10 | Background | Black. train.rs:116-118 | Zero-mean random [-0.25, 0.25] each step. SplatTrainerGpu.cs:1998-2009, :2041-2043 | rb measured: noise on Train (+0.09 off). Low. |

### Density control

| # | Parameter | Brush v0.3.0 | SpawnScene (c1) | Expected effect |
|---|---|---|---|---|
| 11 | Refine cadence and window | Every **200** steps from step 200 to 15000, so the whole 7K run. config.rs:52-53, :65-66; train.rs:337 | Every **100** from 599 to 6900. Studio.Training.cs:854-864 | Brush: half as many applies (35 vs 65), and growth starts 400 steps earlier. Medium-low. |
| 12 | Growth signal | Per step, per splat: sum over pixels of `abs(dL/dx * W) + abs(dL/dy * H)` (an L1 of per-pixel gradients, only where alpha is unclamped). Across steps it keeps the **max**, then divides by the number of steps the splat was visible. Threshold 4e-5. rasterize_backwards.wgsl:174-192; stats.rs:23-26, :36-37; config.rs:56-57 | AbsGS: per-pixel abs of dCentre summed over the tile, then the mean over visible steps. Threshold 8e-4. SplatTrainerShaders.cs:1050, :1095-1096; SplatDensityControl.cs:48 | Different statistic (max/count vs mean) and different units. The bars cannot be compared without a run. Result: Brush ends Train with 0.84M splats vs our 0.47M, Counter 0.39M vs 0.48M. Medium (unverified direction). |
| 13 | **What grows, and how** | `round(0.1 x count over threshold) - pruned_count` splats, drawn at random weighted by the refine weight, capped at 10M. Each one is **split in place**: both copies get `log_scale - ln sqrt2` (scale / 1.414), opacity `1 - sqrt(1 - o)` each, and are offset `+-R (N(0, 0.5) x scale)`. Nothing is a clone. train.rs:398-428, :483-536 | Every candidate grows (frac 1). If max scale <= 0.01 x extent: **clone** (exact copy, same position, same opacity). Else **split**: 2 children at scale / 1.6, each with the parent's opacity. SplatDensityControl.cs:64, :70, :73; GpuDensify.cs:349-350, :356-381 | Brush's split keeps the image: two half-alpha copies composite back to about `o`. Our clone composites to `1 - (1-o)^2`, and our split children keep full `o`. **Measured on the c1 logs** (`[Densify] apply probe`, supervised PSNR of 3 views just before and after each apply), median change per apply: Train -0.18, Bonsai -0.21, Truck -0.12, Bicycle -0.11, Counter -0.08 dB. The p10 values are -0.72 / -1.15 / -0.86 / -0.56 / -0.42. The probe does not show how fast the drop recovers. Medium. |
| 14 | Dead-splat handling | Every refine: prune `opacity < 2/255`, any `log_scale < -15`, or a centre more than 10 x median_size from the bounds centre. The same number are **re-added** by splitting live splats drawn by opacity. train.rs:42, :356-396 | Prune `opacity < 0.005`. World-size prune (> 0.1 x extent) only after the reset. No replacement. GpuDensify.cs:266; SplatDensityControl.cs:76, :86 | Brush keeps capacity on the surface it already has. Low-medium. |
| 15 | **Opacity reset** | **None** (no reset code in brush-train). | Capped to 0.01 at 3000. Studio.Training.cs:869-871; SplatDensityControl.cs:79 | The c1 apply probe at the reset step: Train -11.6, Counter -16.4, Bonsai -16.2, Truck -18.2, Bicycle -12.4 dB supervised, to be re-earned in 4000 steps. nc (cap off, older schedule) cost Train -0.2, so this is not proven harmful here. Retest on top of #1. Medium-low. |
| 16 | Position noise | `noise = R (N(0,1) x scale) x (1-o)^100 x lr_mean x 40`, clamped to +-0.25 median_size, every step, all splats (near-transparent ones in effect). train.rs:290-315 | None (MCMC only). Studio.Training.cs:464 | At Train's lr_mean 6.9e-5 the multiplier is 2.8e-3 x scale, so the noise is negligible. Low. |
| 17 | Scale ceiling | None per step. The only bound is the out-of-bounds prune. | log-scale clamped to 0.1 x extent every Adam step. Studio.Training.cs:550 | a8 measured the cap is not the gap. Low. |

### Initialisation

| # | Parameter | Brush v0.3.0 | SpawnScene (c1) | Expected effect |
|---|---|---|---|---|
| 18 | **Init scale** | `0.5 x mean(dist to 2 NN)`, clamped to [1e-3, 0.1 x median_size]. render gaussian_splats.rs:132-135 | RMS of the distances to 3 NN, capped at 10x median. SparsePointCloudInit.cs:29, :72, :133 | **Ours starts 2.2-2.8x larger** (Train 2.43x, Kitchen 2.79x, Bonsai 2.68x; section 2). Coarser blobs at the start mean more overlap and more early clone/split. Medium-low. |
| 19 | Init opacity / rotation | Uniform in [logit 0.1, logit 0.25], random unit quaternions. gaussian_splats.rs:148-156, :172-178 | 0.1, identity. SparsePointCloudInit.cs:26, :124-127 | Low. |

### Rasteriser (quality side)

| # | Parameter | Brush v0.3.0 | SpawnScene (c1) | Expected effect |
|---|---|---|---|---|
| 20 | Alpha clamp | 0.999. rasterize.wgsl:83; rasterize_backwards.wgsl:152, :174 | 0.99. SplatTrainerShaders.cs:131 | Already in the opacity doc. Low. |
| 21 | Splat footprint | Per-axis extent `sqrt(2 ln(255 o) Sigma_ii)` (to the alpha = 1/255 contour), plus a per-tile ellipse test (StopThePop). project_forward.wgsl:68; map_gaussian_to_intersects.wgsl:35-38, :61; helpers.wgsl:234-268 | Square box at 3 sigma of the largest eigenvalue. SplatTrainerShaders.cs:243-249 | Quality: for o > about 0.37, the alpha = 1/255 contour lies **outside** 3 sigma (3.33 sigma at o = 1), so our opaque splats are truncated at 3 sigma where alpha is still up to 0.011. Low for quality, high for speed (#S2). |
| 22 | Projection cull | Near 0.01 (scene units), skip `opacity < 1/255` entirely. project_forward.wgsl:34, :64 | Near 0.2 (`&nearplane`), no opacity cull. SplatTrainerGpu.cs:556 | np / tr1 measured: +0.03 to +0.3 on Train. Low. |

### Data and eval (checked, not a cause)

Both tools see the same pixels: Brush caps at 1920 px and the Train JPGs are 980 px (`brush-dataset/src/config.rs:16-17`).
Both use the same split: name order, `i % 8 == 0` (`colmap.rs:150-155`). Brush shuffles each epoch per loader thread
(`scene_loader.rs:92-98`), which is close to our per-epoch shuffle. Brush scores on black with an 8-bit round trip
(`eval.rs:37-54`), and our rescore uses one scorer for all tools. COV_BLUR / EWA 0.3 with no compensation is the same
(`helpers.wgsl:188-202`; SplatTrainerShaders.cs:110).

### Speed (per-step cost)

Profile excerpt from our Counter run (`&trainprofile=1`, 1558x1038, 210 views). Each phase syncs the GPU, so the absolute
times are inflated, but the shares hold. At cycle 16 (13.5 it/s): **85.2 ms/step = backward 38.8, scatter 8.9**,
prev 4.5, clear 3.4, emit+count 5.0, sort 4.1, ranges+raster 4.7, loss+ssim 4.6, adam 3.3, sh 3.6, geometry 3.3,
loss read 0.9. Backward plus scatter is 56% of the step and grows with the splat count (cycle 6: 25.2 ms, cycle 16:
47.7 ms). Brush's whole Counter run is 172 s for 7000 steps including load and eval, so it averages **under 24.6 ms/step**.

| # | Item | Brush v0.3.0 | SpawnScene | Expected effect |
|---|---|---|---|---|
| S1 | **Backward reduction** | 64 threads x 4 pixels each. Per splat: `subgroupAdd` of 10 values, then one lane does the atomics (CAS on WebGPU, hardware float atomics natively). No per-key buffers. rasterize_backwards.wgsl:56-57, :145-231 | One 256-thread workgroup per tile. **Per key**, an 8-round shared-memory tree reduction of 11 floats with a barrier each round (about 9 barriers a key), whether or not any pixel was hit. Then it writes 9 floats a key to grad_a/b/c, and a **separate scatter pass** CAS-adds 9 floats a key into the splats. SplatTrainerShaders.cs:1051-1110, :1158-1200; SplatTrainerGpu.cs:2148, :2173 | Backward + scatter = 47.7 of 85.2 ms. Subgroup reduction removes the barriers and the per-key round trip. **Highest.** Needs the WebGPU `subgroups` feature (check `adapter.features` on TJ's Chrome; availability not verified here). |
| S2 | Keys per splat | Opacity-aware per-axis extent, tile-ellipse test, opacity < 1/255 culled before binning (#21, #22). | 3-sigma square of the major axis, no tile test, no opacity cull. | Fewer keys shrink emit, sort, raster, backward and scatter together. Counter c1 measured 13.5-17 keys a splat (`[Densify] keysPerSplat` lines). Brush's count is unmeasured, so the size of the saving is unverified. High. |
| S3 | Sort | Depth sort of the **visible splats** (32 bits), then a stable sort of intersections on **tile bits only** (about 12 bits = 2 x 8-bit passes). render.rs:166-169, :259-269 | One sort of all keys on 18 depth bits + tile bits (about 30 bits = 4 passes). SplatTrainerGpu.cs:915-916 | Half the passes over the large array. Medium. |
| S4 | Per-step CPU sync | Counts stay on the GPU. Dispatches are sized indirectly (`create_dispatch_buffer`). render.rs:182, :212, :300 | `await _counter.CopyToHostAsync` (key count) **mid-step, every step**, before the sort is encoded. SplatTrainerGpu.cs:887 | The GPU idles during the map round trip, and "prev" + "clear" add 8 ms/step of host and clear overhead. Clearing all of keys and values each step (SplatTrainerGpu.cs:848-849) is also avoidable. Medium. |
| S5 | SSIM | One separable 3-channel conv2d (burn). ssim.rs | 3 channels x 4 dispatches, each its own submission. SplatTrainerGpu.cs:2084-2123 | 4.6 ms/step. A fused 3-channel pass would save a few ms. Low-medium. |
| S6 | Densify / carve overhead | Refine every 200. CPU multinomial on a readback of N weights. | Every 100: GPU plan, renderer upload, Resize (about 30 ms, the "[Trainer] sized" stage times), bounds readback. Floater census every 1000 (Train 1.8-2.6 s each, about 16 s of the 268 s run). Studio.Training.cs:875-895 | A few % of wall time. Low. |

## 2. Learning-rate and init-scale scales, per scene (CPU, COLMAP binaries)

Script: `<scratch>/scripts/lr_scale.py`. It reads `points3D.bin` and `images.bin` and computes Brush's `median_size`
(75th-percentile box of the SfM points, as at Brush's step 0) and our `extent` (1.1 x max distance of the train cameras,
every 8th by name held out). Our Train log prints extent 7.452, and the script gets 7.4518. Init scale is the median over
100 random points.

| Scene | Brush median_size | ours extent | pos lr start: ours / Brush | pos lr end | init scale ours / Brush | 7K PSNR, ours - Brush |
|---|---|---|---|---|---|---|
| Train | 3.442 | 7.452 | 1.19e-3 / 6.88e-5 = **17.3x** | 3.5x | 2.43x | **-1.55** |
| Kitchen | 2.288 | 4.954 | 7.93e-4 / 4.58e-5 = **17.3x** | 3.5x | 2.79x | -0.56 |
| Counter | 2.935 | 5.062 | 13.8x | 2.8x | 2.18x | +0.01 |
| Truck | 6.208 | 5.848 | 7.5x | 1.5x | 2.38x | -0.40 |
| Bonsai | 6.749 | 5.159 | 6.1x | 1.2x | 2.68x | -0.71 |
| Bicycle | 13.765 | 4.971 | 2.9x | 0.6x | 2.46x | **+0.43** |

The correlation is suggestive, not clean: Counter ties at 13.8x and Bonsai loses at 6.1x. Note that gsplat uses the same
position lr as ours and Brush still beats gsplat on Train by 0.65 dB. Brush's median_size is recomputed from the splats
every 200 steps, so its later values are not in this table.

## 3. What to A/B on Train first (ranked)

Each item runs on top of c1 (`&poslrsteps=7000&exposure=0`), on Train first. Add Kitchen and **Bicycle** as the control,
since we lead on Bicycle and it should not get worse. Same seed (`&shuffleseed=1`). Report held out on the shared scorer.

1. **Position lr at Brush's scale.** Knobs exist: `&poslr=X` (PositionLrScale, parsed only on the dataset path,
   Studio.razor.cs:670-671) and `&poslrdecay=Y` (end / start, Studio.razor.cs:705-706). Brush-equivalent runs:
   Train `&poslr=0.058&poslrdecay=0.05`, Kitchen `&poslr=0.058&poslrdecay=0.05`, Bicycle `&poslr=0.35&poslrdecay=0.05`.
   The poslr value is Brush start / ours start from section 2, and Brush end / start = 1e-6 / 2e-5 = 0.05. Add a bracket
   run at `&poslr=0.25` on Train. Why first: it is the largest numeric difference in the table (17x on Train). It is
   also the only lever that has already paid on Train (+0.57 from a smaller late lr). If it fails, a code change would
   use Brush's own reference length, the 75%-box median of the splats (#1).
2. **Brush-style split, and no reset.** Two runs:
   - (a) `&opacitycap=0` alone. The knob exists (Studio.razor.cs:276). It keeps the reset's schedule and the size-prune
     unlock, and the reset currently costs 11-18 dB supervised at step 3000.
   - (b) A code change in `GpuDensify` (CopyRow at :340/:350 and the split at :356-381). Grow by splitting in place with
     Brush's rule: both copies get scale / sqrt2, opacity `1 - sqrt(1 - o)`, and offsets `+-R (N(0, 0.5) x scale)`.
     No exact-copy clones.

   `&densify=200&densifyfrac=0.1` alone does **not** reproduce Brush (#12-#13: different signal, top fraction instead of
   weighted sampling, no relocation). It would mostly cut our growth, and Train already has 0.47M splats vs Brush's 0.84M.
3. **Scale lr and init scale.** Code change:
   - Add `&scalelr=` / `&scalelrend=` at Studio.Training.cs:547, decayed per iteration like PositionLr (:733-739).
     Brush is 0.01 -> 0.006.
   - Add an init-scale multiplier in SparsePointCloudInit. About 0.41 reproduces Brush's 0.5 x mean(2 NN) on these
     scenes.

   Test them together (#2, #18), then apart if they help.
4. **SH degree 3 from step 0.** Code change: an option at Studio.Training.cs:726 that sets
   `ActiveShDegree = SphericalHarmonics.MaxDegree`. The `&shdeg` knob only caps the viewer. Brush never ramps (#5).
5. **Opacity lr 0.01.** `&opacitylr=0.01` exists (Studio.razor.cs:272-274). It is cheap, but 0.05 already measured -0.09,
   so expect a small effect. Run it last or fold it into #1's run.

For speed, the order is: S1 (subgroup backward reduction plus direct per-splat atomics, which also removes the scatter
pass), S2 (opacity-aware per-axis tile binning with the tile-ellipse test), S4 (no per-step key-count readback; size the
sort and raster dispatches indirectly), S3 (two-stage sort), S5 (fused SSIM). Re-profile after each, with
`&trainprofile=1` on Counter, nothing else running.

## Verified / not verified

**Verified** (read in source on both sides, the cites above):
- Every Brush default in the tables and Brush's growth, prune and split arithmetic.
- That v0.3.0 trains on black and has no opacity reset and no SH ramp.
- Our values and schedules for the c1 configuration.
- The lr and init-scale ratios (computed from the COLMAP files; our extent matches the Train log to 4 decimals).
- The apply-probe medians (from the c1 logs).
- Our profile shares (from the live profile log, cycles 1-16; the run was still going).

**Not verified:**
- Any quality effect of any row. Nothing was trained for this note.
- Brush's keys per splat and per-step phase times. Brush was not profiled, so the S2 and S3 savings are estimates.
- Whether TJ's Chrome exposes WebGPU `subgroups` (S1).
- How quickly the per-apply probe drop recovers.
- The comparability of Brush's 4e-5 growth bar with our 8e-4.
