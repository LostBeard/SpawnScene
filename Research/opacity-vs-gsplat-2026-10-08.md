# Opacity and density control: SpawnScene vs gsplat 1.5.3 DefaultStrategy (2026-10-08)

Question: on Hamamni Baths (7K-end, llffhold=8, our cameras) gsplat ends with opacity median 0.65 (53% > 0.5) and we
end with 0.12-0.14 (12-15% > 0.5) at the same splat sizes and placement. Why are our splats translucent?

Method: read-only, line by line. No build, no GPU, no training run. The only thing executed was a Python one-liner
that evaluates gsplat's reset condition (pure expression, no gsplat code). Run logs read: `_runs/tuvok-a11-hamamni.log`
(defaults), `_runs/tuvok-a12-hamamni.log` (all six extras off), `_runs/tuvok-o1-hamamni.log` (opacity lr 0.05), and
gsplat's own `out/hamamni7k.log`.

Paths: `<gs>` = `C:\Users\TJ\AppData\Local\Temp\claude\D--users-tj-Projects\feb93008-c1ed-4d6b-ba7d-c6a5775d5095\scratchpad\gs`.
gsplat files are under `<gs>/gsplat-src/` (`default.py` = `gsplat/strategy/default.py`, `ops.py` =
`gsplat/strategy/ops.py`, `trainer` = `examples/simple_trainer.py`). The installed package that ran
(`<gs>/env/Lib/site-packages/gsplat`, version `1.5.3+pt24cu124`) is byte-identical to the source for `default.py`
and `ops.py` (`diff -q`: no output). The run was `simple_trainer.py default --data_factor 2 --max_steps 7000`
(`<gs>/run_hamamni.sh`), so every DefaultStrategy field is at its default (no `--steps_scaler`).
Our files are under `SpawnScene/SpawnScene/SpawnScene/`.

## Headline finding: gsplat 1.5.3 NEVER resets opacity. We reset once, at 3000.

`default.py:195`:

```python
if step % self.reset_every == 0 & step > 0:
```

In Python `&` binds tighter than `==` and `>`, so this parses as `(step % reset_every) == (0 & step) > 0`, a chained
comparison: `(step % reset_every == 0) and (0 > 0)`. `0 > 0` is always False, so `reset_opa` is dead code. Proved two
ways:

- `ast.dump` of the expression: `Compare(left=BinOp(step % R), ops=[Eq(), Gt()], comparators=[BinOp(0 & step), 0])`;
  enumerating `step` over 0..30000 with R=3000 gives an empty list of firing steps.
- gsplat's own verbose log (`out/hamamni7k.log`): at step 3100 it pruned **1,624** splats (and its size prune went live
  at step > 3000). Our first densify after OUR reset pruned **31,112 faint + 336 bloated** (a12 log line 1559). A reset
  to 0.01 followed by a 0.005 prune cannot leave gsplat's count untouched. Steps 6000/6100 are also unremarkable
  (1,330 / 1,311 pruned).

So the belief in `parity-matrix.md` ("both reset opacity at 3000 / 6000") and `parity-references-2026-10-08.md`
lines 30-31 ("resets opacity at 3000 and 6000") is wrong on BOTH sides:

- gsplat 1.5.3 DefaultStrategy: no reset ever (bug). Its splats climb from the 0.1 init for all 7000 steps.
- SpawnScene: one reset, at 3000. 6000 is skipped by `Studio.Training.cs:816-818` (`scheduleTotal - (g+1) >= 3000`);
  every log has exactly one "opacity reset to at most 0.01" line.

And most of our final scene is descended from the post-reset population: a12 had 536,779 splats after the reset's
prune and 1.96M at the end, so about **73% of the final splats were born after the reset** as clones / split children
of parents whose opacity had just been capped to 0.01 (a11: 1.29M at the reset, 2.46M before the end carve, about 48%).
A clone and a split child copy the parent's CURRENT opacity (gsplat `ops.py:110`, `:164`; ours
`GpuDensify.cs:350`, `:372`), so in gsplat a late clone inherits a mature opacity and in ours it inherits a recovering one.

## Settings table

"Same" = identical behaviour. Ours = the project / sample path (`TrainProjectSceneAsync`), which is what a11/a12 ran.

| # | Setting / behaviour | gsplat 1.5.3 (as run) | gsplat cite | SpawnScene (as run) | Ours cite | Same? |
|---|---|---|---|---|---|---|
| 1 | Initial opacity | `logit(0.1)` for every SfM point | trainer:103, :255 | 0.1 (linear; logit seeded on GPU) for SfM and depth-init splats | SparsePointCloudInit.cs:26, :121; DepthFusionInit.cs:231, :329; SplatTrainerShaders.cs:2250-2251 | Same |
| 2 | Initial scale | `log(sqrt(mean of squared dist to 3 NN) * init_scale=1.0)`, isotropic, no cap | trainer:105, :244-246 | RMS of 3 NN distances, isotropic, capped at 10x median (or maxScale) | SparsePointCloudInit.cs:29, :88-119 | Same (plus outlier cap) |
| 3 | Initial rotation | `torch.rand(N,4)` (random, normalised in raster) | trainer:254 | identity | SparsePointCloudInit.cs:124-127 | Differs, irrelevant for isotropic init |
| 4 | Opacity activation | `sigmoid(logit)` | trainer:498 | `sigmoid(logit)`; logit kept in its own buffer, re-seeded from the packed opacity (clamped to [1e-6, 1-1e-6]) after every densify | SplatTrainerShaders.cs:1413-1424, :2250-2251; Studio.Training.cs:1593 | Same (logit capped at +-13.8) |
| 5 | Opacity optimiser | torch Adam, betas 0.9/0.999, eps 1e-15, dense (every splat every step; `visible_adam=False`) | trainer:290-297, :845-848 | Adam 0.9/0.999, eps 1e-15, dense (`SkipZeroGradientSteps` default false) | SplatTrainerShaders.cs:1364-1373, :1393; SplatTrainerGpu.cs:1620, :2162-2170; Studio.Training.cs:63 | Same |
| 6 | Opacity lr | **0.05**, constant | trainer:135, :262 | **0.025**, constant (`&opacitylr=` honoured in every mode) | SplatOptimizer.cs:24; Studio.Training.cs:137, :713; Studio.razor.cs:271-274 | **Differs (half)** |
| 7 | Adam step count / bias correction | per-tensor `step`, kept across densify (`key != "step"`) | ops.py:84-89 | global `_adamStepCount`, kept across densify | SplatTrainerGpu.cs:2162; Studio.Training.cs:1581 | Same |
| 8 | Moments of new splats | zero for duplicates and split children; survivors keep theirs | ops.py:112-113, :169-171, :202-203 | zero for children (`adamSources = -1`), survivors keep theirs | SplatDensityControl.cs:358-372; Studio.Training.cs:1379-1381 | Same |
| 9 | **Opacity reset: when** | `reset_every=3000`, but the condition is **always False** (precedence bug) -> **never** | default.py:88, **:195** | every 3000 while densifying, skipped if less than 3000 iterations remain -> **once, at 3000** in a 7K run | Studio.Projects.cs:875; Studio.Training.cs:816-818 | **Differs** |
| 10 | Opacity reset: to what, clamp or set | (dead) clamp-to-max `logit(prune_opa*2 = 0.01)`, opacity moments zeroed | default.py:196-201; ops.py:228-236 | clamp-to-max 0.01 on every kept row, clone and split child; opacity moments zeroed (slot 3) | SplatDensityControl.cs:79, :411; GpuDensify.cs:325, :338-340, :350, :372; Studio.Training.cs:1581 | Same operation, but only ours runs |
| 11 | Faint prune | `sigmoid < 0.005`, at every refine step | default.py:79, :319 | `opacity < 0.005`, at every densify step | SplatDensityControl.cs:76, :236; GpuDensify.cs:266 | Same |
| 12 | World-size prune | `max scale > 0.1 * scene_scale`, when `step > reset_every` (3000), regardless of whether a reset ran | default.py:83, :320-324 | `max scale > 0.1 * rigRadius`, only after a reset has RUN (`_hadOpacityReset`) | SplatDensityControl.cs:86, :242-244; GpuDensify.cs:267; Studio.Training.cs:1301, :1361-1363 | Same timing in a7K run with reset on; **turning our reset off also turns this off** |
| 13 | Screen-size prune | `radii > 0.15` only if `step < refine_scale2d_stop_iter` (0) -> off | default.py:84-85, :330-331 | `MaxScreenRadiusPx = +inf` -> off | SplatDensityControl.cs:104 | Same (off) |
| 14 | Scale ceiling during optimisation | none (only the prune of #12) | - | log-scale clamped to `0.1 * rigRadius` every Adam step | Studio.Training.cs:51, :492-497; SplatTrainerShaders.cs:2488-2489 | Differs (a8 measured the cap is not the gap) |
| 15 | Scene extent | `1.1 * max camera distance from centre` | trainer:347 | `1.1 x` the same over the train rig | Studio.Training.cs:476 | Same |
| 16 | Grow gradient | mean over visible steps of `||dL/dmean2d||`, x `W/2`, `H/2` (x n_cameras = 1) | default.py:220-226, :250-253, :271 | same quantity; signed sum (`&absgrad=0`) or per-pixel abs sum (AbsGS, default) | SplatTrainerShaders.cs:1653-1700 (`:1680`); SplatTrainerGpu.cs:1382 | Same with `&absgrad=0` |
| 17 | Grow denominator | steps with `radii > 0` (in frustum, opacity >= 1/255) | default.py:246-252; ProjectionEWA3DGSFused.cu:171-175 | steps with `screen_radius > 0` (emitted keys; no opacity cull) | SplatTrainerShaders.cs:1674-1676 | Nearly same (ours also counts opacity < 1/255 splats) |
| 18 | Grow threshold | 2e-4 (absgrad off) | default.py:80 | 8e-4 with AbsGS (default), 2e-4 with `&absgrad=0` (a12) | SplatDensityControl.cs:48; Studio.razor.cs:241-242 | Same in a12 |
| 19 | Clone vs split | `max scale <= 0.01 * scene_scale` -> duplicate, else split; duplicates first and never split | default.py:274-298 | `max scale > 0.01 * extent` -> split, else clone | SplatDensityControl.cs:70, :251; GpuDensify.cs:273 | Same |
| 20 | Fraction of candidates grown | all | default.py:279-286 | all (`GrowthSelectFraction = 1`), capped by the GPU budget (3.28M on a12, not reached) | Studio.Projects.cs:876-880 | Same |
| 21 | Split child | two samples `mean + R (s * N(0,1))`, scale / 1.6, opacity copied (`revised_opacity=False`) | ops.py:144-164 | same draw, scale / 1.6, opacity copied | SplatDensityControl.cs:73, :300-337; GpuDensify.cs:356-381 | Same |
| 22 | Clone | exact copy, same position | ops.py:109-110 | exact copy, same position | SplatDensityControl.cs:284-291; GpuDensify.cs:349-350 | Same |
| 23 | Densify every / start | every 100 steps, `step > 500` (first at step 600) | default.py:86, :89, :167-171 | every 100 iterations, `g >= 500`, on `(g+1) % 100 == 0` (first at g=599) | Studio.Training.cs:220, :801-811; Studio.Projects.cs:874 | Same (one step earlier) |
| 24 | Densify stop vs max_steps | `refine_stop_iter = 15000` absolute; `--max_steps 7000` does not change it -> refines to step 6900 | default.py:87, :162-163; trainer:190-206 (only via `--steps_scaler`) | `DensifyUntilIter = 15000` absolute -> densifies to 6900 | Studio.Training.cs:295, :806 | Same |
| 25 | Stats reset | after every refine | default.py:189-190 | after every densify (also when the plan is empty) | Studio.Training.cs:1356, :1595 | Same |
| 26 | Opacity / scale regularisers | `opacity_reg = scale_reg = 0` (default strategy) | trainer:144-146, :712-715 | 0 unless `&mcmc=1` | Studio.Training.cs:420-421; SplatTrainerShaders.cs:1416-1417 | Same (off) |
| 27 | Loss | `0.8 L1 + 0.2 (1 - SSIM)`, fused SSIM `padding="valid"`, per channel | trainer:107, :685-687 | `0.8 L1 + 0.2 D-SSIM`, valid windows, per RGB channel | SplatTrainerGpu.cs:2049-2105, :1405; ImageQuality.cs:224 | Same |
| 28 | Background | black (`random_bkgd=False`) | trainer:128, :670-672 | zero-mean random `[-0.25, 0.25]` per step (default); black with `&randombg=0` (a12) | SplatTrainerGpu.cs:1976-1987, :2018-2021 | Same in a12. Random bg pushes opacity UP, so not the cause |
| 29 | dL/d(opacity) in the rasteriser | `alpha = min(0.999, o*G)`; skip `alpha < 1/255`; grad zero when `o*G > 0.999`; T stop 1e-4 | RasterizeToPixels3DGSFwd.cu:148-154; RasterizeToPixels3DGSBwd.cu:177-178, :222; Common.h:54 | `alpha = min(0.99, o*G)`; skip `< 1/255`; grad zero when `o*G >= 0.99`; T stop 1e-4; bg term as the reference | SplatTrainerShaders.cs:32, :130-132, :1004-1031 | **Differs: clamp 0.99 vs 0.999** |
| 30 | 2D dilation / opacity compensation | `eps2d = 0.3`, classic mode: no compensation | rendering.py:46; ProjectionEWA3DGSFused.cu:153-170 | `EWA_FILTER_PX2 = 0.3`, no compensation; Mip 3D floor off (`MipFilter = 0`) | SplatTrainerShaders.cs:110, :217-219; SplatTrainerGpu.cs:179 | Same |
| 31 | Projection cull | near 0.01; splats with `opacity < 1/255` get radius 0 (no keys, no grad, not counted) | trainer:110; ProjectionEWA3DGSFused.cu:171-178 | near 0.2; no opacity cull (alpha test per pixel) | SplatTrainerShaders.cs:142, :250 | Differs (not an opacity driver) |
| 32 | Position lr schedule | `1.6e-4 * scene_scale`, decays 100x over `max_steps` = **7000** | trainer:131, :259, :558-563 | `1.6e-4 * extent`, decays 100x over a fixed **30000** -> ends at 0.34x | Studio.Training.cs:238, :487, :683-686 | **Differs** |
| 33 | Scale / rotation / SH lrs | 5e-3 / 1e-3 / sh0 2.5e-3, shN /20 | trainer:133-141 | 5e-3 / 1e-3 / 2.5e-3, rest /20 | Studio.Training.cs:494-495; SplatOptimizer.cs:21; SplatTrainerGpu.cs:2189 | Same |
| 34 | SH degree schedule | +1 every 1000 steps to 3 | trainer:101, :636 | +1 every 1000 iterations to 3 | SphericalHarmonics.cs:216-217 | Same |
| 35 | Extras that write opacity (defaults) | none | - | carve every 1000 (opacity -> 0, then faint-pruned), end carve, prune-unconstrained after cycle 1; all off or no-op in a12 | Studio.Training.cs:822-824, :1061-1067, :786-795 | Off in a12 (median still 0.14) |

## Ranked differences most likely to leave our splats translucent

### 1. Our opacity reset at 3000 (gsplat 1.5.3 never resets) - highest

Why: it is the only difference in the table that acts on opacity directly and at scale, and it survives the a12
ablation (all six extras off, median still 0.14). At 3000 every splat is capped to 0.01 (logit -4.6). gsplat's
splats instead have 7000 uninterrupted steps from 0.1. Worse, growth does not stop: about 73% of a12's final splats
were cloned or split AFTER the cap from parents that were themselves at or near 0.01, and a clone / split child
inherits that opacity. The splats that do climb back need to cover 4.6 logit units at Adam lr 0.025 while being seen
in a fraction of the views, and the late children (densify runs to 6900) get only a few hundred steps. That is a
population that ends mostly translucent, which is what we measure. It also fits o1: doubling the opacity lr lifted the
median only to 0.20 because the reset, not the rate, sets where everyone starts. Our own Truck note at
Studio.Training.cs:812-815 (a reset at 6000 cost 4 dB held out at 7K) is the same mechanism at its extreme.

Minimal test (keeps gsplat's size-prune timing, so only the cap changes):

1. `Pages/Studio.Training.cs`, next to `OpacityResetEveryIters` (line 287), add
   `public static bool OpacityResetCaps { get; set; } = true;` with a comment citing gsplat `default.py:195`.
2. `Pages/Studio.Training.cs:1301`: in `new GpuDensify.Options(sceneExtent, _hadOpacityReset, budget, resetOpacity, ...)`
   pass `resetOpacity && OpacityResetCaps` as the 4th argument (it drives the clamp in `GpuDensify.CopyRow`).
3. `Pages/Studio.Training.cs:1379`: pass `resetOpacity && OpacityResetCaps` to `InstallGrownSetAsync` (it zeroes the
   opacity moments, slot 3, at line 1581).
4. Leave lines 1361-1363 alone: `_hadOpacityReset = true` still unlocks the world-size prune after 3000, exactly
   gsplat's `step > reset_every` (`default.py:320`).
5. `Pages/Studio.razor.cs`, in the every-mode block next to `opacitylr` (line 271):
   `if (query.TryGetValue("opacitycap", out var ocq)) OpacityResetCaps = ocq is not ("0" or "false");`
   (A URL `&opacityreset=0` does NOT work on this path: `TrainProjectSceneAsync` overwrites it with 3000 at
   `Studio.Projects.cs:875`.)

Run a12's URL plus `&opacitycap=0`, then a11's (defaults) plus `&opacitycap=0`; compare the `[Stats] opacity` line
and held out. Cruder one-line alternative: `Studio.Projects.cs:875` `OpacityResetEveryIters = 0;`, but that also
disables the world-size prune for the whole run (row 12), so it is not a gsplat parity run.

### 2. Opacity lr 0.025 vs gsplat 0.05 - only meaningful together with #1

Why: gsplat's 0.05 was measured (o1) with OUR reset still on, so it measured "how fast do splats recover from 0.01",
not parity. Without the reset, 0.05 is what gsplat actually ran; our 0.025 is the current Inria value, which Inria
pairs with resets and a 30K schedule. Test: #1 plus `&opacitylr=0.05` (already parsed in every mode,
Studio.razor.cs:271-274). If it is adopted, change `SplatOptimizer.cs:24` `DefaultOpacityLr`.

### 3. Position lr decays over 30000 steps, gsplat's over 7000 - medium-low for opacity

Why: with `--max_steps 7000` gsplat's means lr is 1% of its start by the end (`trainer:561-562`); ours ends at
`0.01^(7000/30000) = 0.34x` (`Studio.Training.cs:238`, `:683-686`). Positions still moving at a third of their rate
late in training keep the screen-space gradients high: our last densify cloned 40,117 (a12 log line 2450) where gsplat's
step 6900 duplicated 23,178. Every clone is a co-located exact copy; two copies of opacity `a` render as
`1 - (1-a)^2`, and the loss then pulls both down, so a late clone wave is a translucent-pair generator. Test (parity
schedule): `Studio.Training.cs:685` replace `PositionLrMaxSteps` with `scheduleTotal` (in scope since line 670). Note
the comment at lines 226-237 measured this as worse on an older build; re-measure it on top of #1, not alone.

### 4. Alpha clamp 0.99 vs 0.999 - low

Why: our `MAX_ALPHA` zeroes the opacity gradient wherever `o*G >= 0.99` (SplatTrainerShaders.cs:1030-1031), gsplat at
0.999 (`RasterizeToPixels3DGSBwd.cu:222`). This only bites splats that are already nearly opaque, so it can trim the
top of the distribution (our p90 0.57-0.62 vs gsplat's p50 0.65) but cannot produce a 0.14 median. Test: 0.99 -> 0.999
in `SplatTrainerShaders.cs:32` and `:131`, and keep the viewer and CPU twins identical: `SplatRasterizer.cs:25`,
`GpuGaussianRenderer.cs:2690`, `GaussianRenderer.cs:189`, `GaussianTrainer.cs:335`.

### 5. Per-step scale ceiling at 0.1 x extent - low

Why: gsplat never clamps scale, it prunes above the same bar after 3000. A splat pinned at the ceiling cannot grow to
cover a large flat region, so the optimiser may cover it with several overlapping, lower-opacity splats. a8 (cap x10,
effectively none) did not close the PSNR gap, and splat sizes already match gsplat's, so this is unlikely to be the
opacity driver. Side note: because the clamp and the world-size prune use the same bar (`0.1 * rigRadius`), the
"bloated" prunes only catch splats whose `exp(log(cap))` rounds a hair above the cap. Test: `&maxscale=10` together
with #1.

### Ruled out by the table

Initial opacity / scale, activation, Adam (eps, betas, dense stepping, bias correction, moment handling), faint prune,
screen-size prune, grow criterion and normalisation (with `&absgrad=0`), clone / split rules and split opacity, densify
window, regularisers, loss, EWA dilation, SH schedule: all match. Random background, exposure, depth init, pose
refinement, carve and prune-unconstrained were off in a12 with the same opacity picture. The projection near plane
(0.2 vs 0.01) and the `opacity < 1/255` projection cull differ but do not act on opacity values.

## Verified / not verified

Verified: gsplat's reset never fires (AST, enumeration, and gsplat's own prune counts around 3000 / 6000); our project
path resets once at 3000 (code and the a11 / a12 / o1 logs); the installed gsplat equals the source read; every row's
cite was read on both sides. NOT verified: that removing our reset raises the opacity median and held-out PSNR. That
is the #1 test above; nothing was run for this note.
