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
true. There are TWO renderers in this codebase - `SplatTrainerGpu`'s rasteriser, which every PSNR
and SSIM number comes from, and `GpuGaussianRenderer`, which is what a person looks at. The bug was
in the second one only (`CameraController.UpdateCamera` forces `Vector3.UnitY` every frame). No
amount of work on the metric could have found it, and I spent that session improving the metric.

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
| display renderer | 🔴 **UNPROVEN** - had the up-vector bug; never checked against a known-good splat |
| depth -> splats | partly - unit-tested unproject, but no end-to-end check against a reference reconstruction |
| poses | measured on TempleRing (5% of camera spread) and on a posed ROOM (in progress) |
| optimiser | 🔴 unproven against known-good poses - that is what Deep Blending is for |

The display renderer is the gap that matters most, because it is what anyone judging this project
looks at, and it is the one that produced a wrong conclusion. Loading a reference `.ply` through it
is the cheapest decisive test available.
