# Validating a pipeline where nothing is proven yet

TJ, 2026-09-21, after finding a room rendering on its side while every number said it was fine:

> *"we have not fully tested the renderer and we are trying to create a generator that is also
> unproven using the unproven renderer."*

That is the central measurement problem on this project, and it is worth stating plainly because
it keeps producing wasted days.

## The problem

A reconstruction pipeline is a chain, and right now **not one link is independently proven**:

```
photos -> depth -> poses -> splats -> RENDERER -> metric -> a number
           ^        ^         ^          ^          ^
           |________|_________|__________|__________|
                    all unproven, and only the NUMBER is observed
```

When the number is bad, it could be any of them. Worse - and this is what actually happened - when
the number is GOOD it can still be any of them, because two wrong components can agree with each
other.

**Concrete instance, 2026-09-21.** Bathroom scored 12.41 dB held out, the best the capture had ever
produced, while the room rendered rotated 90 degrees and tumbled when the camera moved. Both were
true. There are TWO rendering paths - `SplatTrainerGpu`'s rasteriser, which every PSNR and SSIM
number comes from, and the display path, which is what a person looks at - and only the first one
was ever scored. No amount of work on the metric could have found it, and I spent that session
improving the metric.

⚠ **Be precise about WHICH component.** The bug was in `CameraController.UpdateCamera`, which
replaces the camera basis with world +Y on every frame. The same day, a blank frame came from
`CameraController.FitToScene` aiming at a hardcoded TempleRing point. Both are the CAMERA layer.
`GpuGaussianRenderer` was handed a wrong camera on both occasions and drew exactly what it was
asked to. It has rendered a 14,000,000-splat scene and the DAv2-generated scenes correctly, which
is real evidence that the rasterisation, sorting and blending work.

The first draft of this document labelled the renderer "unproven" and let that read as "suspect",
which is wrong twice over: it conflated two components, and it discarded the positive evidence
that already existed. **Unproven means "not gated by an automated check", not "probably broken".**
Those need different words, because the second one sends the next person hunting in the wrong
file.

## The technique: isolate one link by making every other link known-good

For each component, find an input where everything else is already trusted.

| To isolate | Feed it | Where that comes from |
|---|---|---|
| **the metric** | two images with a known score | analytic fixture pinned to `tools/score_novel_view.py`, itself pinned to the published SSIM definition |
| **the renderer** | a known-good splat scene | reference 3DGS `.ply` for drjohnson / playroom / train / truck (see `datasets.md`) |
| **our splats** | a known-good renderer | load OUR export in a third-party viewer, e.g. superspl.at |
| **the optimiser** | known-good poses | COLMAP poses from Deep Blending / Tanks and Temples |
| **our poses** | a known-good answer | fit recovered cameras to COLMAP or Middlebury poses and report the residual |
| **a GPU kernel** | a CPU oracle | the pattern this repo already uses in `Studio.TrainerGate.cs` |

The rule that falls out of it:

> **Never let two unvalidated components be tested only against each other.**

A GPU kernel checked against a CPU oracle is fine. A GPU kernel checked against another GPU kernel
is not. A generator scored by a renderer nobody has validated is not.

## Corollaries that cost real time before they were written down

**A metric can be adequate and still blind.** PSNR on Bathroom moved 12.50 -> 12.21 dB across a
training run - essentially flat - while the render melted from a recognisable room into fog. PSNR
over a sparse reconstruction is dominated by large smooth regions, so smoothing structure away
barely moves it. SSIM saw it (0.6264 -> 0.5782). **Always render the picture next to the number**,
and if the two disagree, believe the picture and go find out why the number did not.

**Score from somewhere nobody stood.** A render from a capture pose is close to a re-projection of
the photo it came from, so it flatters any reconstruction - including one that is only a stack of
per-view shells. Held-out views help but sit near the capture path. The harness now takes five
free-view captures per run (left, right, back, up, turned) because a room that only holds together
from its own capture poses is not a room.

**Know the noise floor before comparing runs.** Two runs of the SAME configuration on Bathroom:
initialisation reproduced to **0.09 dB**, final differed by **1.40 dB**, and held-out swung
**2.74 dB within a single run**. Several conclusions drawn that day compared final numbers
differing by less than that. Measure the repeat before believing a difference.

**Compare runs on a mean, never on the last sample.** Held-out PSNR swings 1.88 to 2.74 dB
*within* a single run, and two runs of the same configuration ended 1.52 dB apart. The final
number is one sample of an oscillation, so an A/B read off endpoints cannot resolve anything
smaller than the swing - and beating that by averaging whole extra runs costs GPU hours. The
per-cycle curve is already collected; its mean has roughly half the standard error of any one
sample and is free. The run log prints it on a line labelled `COMPARE:` for exactly this reason.

**Count occurrences, not appearances.** One `KEY OVERFLOW` line looked like a persistent condition
and produced a wrong diagnosis, a shipped fix and a lost GPU device. `grep -c` said 1, and the very
next line said the trainer had already re-sized itself with headroom. A system that recovers
usually says so, and it says so on the next line.

**A statistic a single outlier can swing is not a verdict.** `(max-min)/mean` over pairwise ratios
labelled TempleRing's tightly clustered values a "25.7% shape difference" on data whose fold was
exact to 0.4%. Use a median and a median-absolute-deviation, and print the numbers rather than a
conclusion.

**An unmeasured constant is a bug with a delay.** Every one that got checked this project was
wrong: `MaxMultiViewImages = 6` (real ceiling 8 on this card, and device-dependent), the
consistency screen's out-of-frustum rule (right for an object, throws away the room), the splat
budget (should be derived from the trainer's binding limit), the scene-up threshold (1e-3 would
have tipped the regression fixture over). If a number has no measurement behind it, it is a
hypothesis.

## Where this project currently stands on the chain

| link | status |
|---|---|
| metric (PSNR, SSIM) | **proven** - CPU oracle pinned to the Python scorer at 1e-6; GPU agrees to 2.7e-8 |
| trainer rasteriser | **proven** - `Studio.TrainerGate.cs` compares forward, gradients and the geometry chain against CPU oracles |
| gradient accumulation | **proven** - reduction gated against a CPU pass over the same buffer |
| display renderer | ungated, but substantial positive evidence: 14M splats and DAv2 scenes render correctly |
| camera layer (`CameraController`) | unit-tested since 2026-09-21 (`CameraControllerTests`) - pose survives an input event, `FitToScene` aims at ray convergence. Still no end-to-end viewer check |
| depth -> splats | partly - unit-tested unproject, but no end-to-end check against a reference reconstruction |
| poses | measured on TempleRing (5% of camera spread) and on a posed ROOM (in progress) |
| optimiser | tested against known-good poses and it overfits - but 2026-09-21 showed the CAUSE is upstream: 91.9% of splats are constrained by at most one view, so there is nothing for an optimiser to generalise from |
| initialisation | 🔴 **this is the broken link.** Per-view depth shells; mean 0.59 supervising views per splat |
| splat sampling schedule | proven - `TrainingScheduleTests` asserts probes do not alias with the round robin |
| view support counter | proven - CPU-gated in `Studio.TrainerGate.cs` |

## The hypothesis this chain has never tested

Written down **before** the measurement, so it cannot be rationalised afterwards.

Every optimiser experiment here - fixed-point scales, the zero-gradient Adam guard, shuffled
view order, an SSIM loss term - assumes training *could* generalise and is being held back by a
step rule. On drjohnson, with COLMAP **ground-truth** poses, supervised PSNR rises 13.13 -> 14.34
while held-out falls 12.58 -> 11.84. That is textbook overfitting, and the poses were not the
cause, because they were correct by construction.

There is a structural reason it might be unfixable by any optimiser knob. Initialisation
unprojects a monocular depth map **per view**: 44 views give 44 private depth shells stacked in
one world, 25,344 splats each. If a splat only ever receives a gradient from the single view it
came from, then training is not a reconstruction - it is 44 independent per-view fits sharing a
buffer. Each can lower its own view's loss while saying nothing about a view nobody trained on,
and nothing can contradict it.

**Prediction if true:** most splats will be constrained by at most one view, and the
single-view fraction will be far above what a genuinely shared scene would produce.

**Prediction if false:** a substantial fraction of splats will be moved by two or more views,
the overfitting has an ordinary cause, and the optimiser experiments are worth finishing.

`[Train] view support over N views` reports this after the first full cycle. It is gated against
a CPU pass in `Studio.TrainerGate.cs`, accumulated twice over the same gradients so the check
covers the counting and not just the threshold.

### Measured, 2026-09-21, drjohnson with COLMAP ground-truth poses

```
[Train] view support over 33 views (1,115,136 splats):
        never 49.5%, 1 view 42.4%, 2 7.4%, 3 0.7%, 4+ 0.0%  (mean 0.59 views/splat)
```

**91.9% of splats are constrained by at most one view.** The prediction was confirmed, by a
margin no noise floor on this project comes close to.

Read as geometry, with 25,344 splats per view and 44 views:

| | splats | what it is |
|---|---|---|
| never constrained | 551,992 (49.5%) | invisible from every supervised view |
| exactly 1 view | 472,818 (42.4%) | fits its own photo, nothing can contradict it |
| 2 or more | 90,326 (**8.1%**) | the only part actually being reconstructed |

The 11 held-out shells account for only 278,784 of the never-constrained splats. The other
**273,208 come from supervised views and are occluded even in their own view** - 32.7% of every
supervised shell is buried inside some other shell. The shells do not merely fail to reinforce
each other, they bury each other.

That is the whole result. In the same run supervised PSNR rose 13.21 -> 16.10 dB, the largest
supervised gain this project has recorded, while held-out fell 12.58 -> 11.39. The optimiser is
working exactly as designed: it is efficiently fitting 42% of the parameters to one photo each.

**What it rules out.** The zero-gradient Adam guard, shuffled view order and an SSIM loss term
were all queued on the premise that training *could* generalise. None of them can change a
parameterisation in which 92% of the parameters are unconstrained or single-constrained. The
A/B on the guard had already come back inside the noise; this says why, and says not to run the
other two.

**What it points at**, in order of cost:

1. **Prune the never-constrained half.** 551,992 splats receive no gradient from any supervised
   view over a full cycle. They cost memory and raster time, they cannot improve, and they are
   free to be wrong in exactly the held-out views we score on. `SplatDensityControl` exists and
   needs GPU compaction.
2. **Fuse the shells**, so a surface three views can see is one set of splats rather than three
   stacked at slightly different depths. `MvsGeometricFusion` exists. This is what would move
   the 8.1% upward, and it is the only thing that can.
3. Only then revisit the optimiser.

### Fixed, same day: initialise from the sparse SfM cloud

Done, and it is the reference algorithm rather than anything invented here -
`GaussianModel.create_from_pcd` in Kerbl et al. Isotropic scale from the RMS distance to the
three nearest neighbours, opacity 0.1, identity rotation, colour from the point. The COLMAP
cloud was already being read by `tools/colmap_to_dataset.py` and used only for a reprojection
check.

| | depth shells | sparse cloud |
|---|---|---|
| splats | 1,115,136 | 79,922 |
| mean views per splat | 0.59 | **2.52** |
| constrained by 2+ views | 8.1% | **73.6%** |
| never constrained | 49.5% | **7.4%** |
| held-out init PSNR | 12.58 | **14.38** |
| held-out final SSIM | 0.4456 | **0.5360** |

Held-out SSIM ended **above** where it started, which no configuration had managed before.

⚠️ **One perf trap worth remembering.** The nearest-neighbour grid sized its cells from the
bounding box, and an SfM cloud has a handful of points triangulated out into the sky. The box
was enormous, the cell was enormous, every real point landed in one cell, and the grid
degenerated into the all-pairs search it exists to avoid: **349 seconds for 79,922 points**.
Percentile bounds fixed it - 18 ms for 20,012 points with outliers, and the real cloud gives
the same median scale to four decimals. A spatial index sized from extremes is sized by its
outliers.

The render is still blurry, and that is expected: 79,922 splats is where the reference STARTS,
and it finishes between one and three million through adaptive density control. That step is
now wired (`densify_accum` plus `DensifyAsync`); the decision layer had been sitting there
tested and unused for want of a GPU signal.

## Where to look first

The gap that matters most is the CAMERA layer: it is ungated, and it is where both wrong
conclusions actually came from. It is also the cheaper thing to gate - a unit test that sets a
pose and asserts the camera basis survives an input event needs no GPU at all, and would have
caught the up-vector bug outright.

Loading a reference `.ply` through the display path is still worth doing, but as confirmation
rather than suspicion: it would turn the existing informal evidence - 14M splats, DAv2 scenes -
into something automated.
