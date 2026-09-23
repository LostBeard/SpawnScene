# Handoff: Pose cascade isolation on DrJohnson (2026-09-23)

Next agent: read `AGENTS.md`, Truck handoff
`Research/handoff-photo-quality-truck-2026-09-23.md`, and this file. Claim **SpawnScene**
before editing. **Fable is warranted** for the cascade coverage / fold problem below;
Auto already locked the defaults the measurements support.

## Decision context

Truck GT optimiser is proven (sup 25.46 / held 19.83). This session isolated whether
multi-image failure is still the optimiser or the pose cascade, using DrJohnson (COLMAP
oracle available). Bathroom was smoke-only, not a tune loop.

## Measured pair (same 2K recipe: GEOM densify/100, opacity reset/3000, maxdensify 450k)

| Run | Tag | Pose | Init | Supervised | Held / COMPARE | Notes |
|---|---|---|---|---|---|---|
| GT oracle | `dj2k-gt-pose` | colmap 44/44 | points 80k | **31.96** SSIM 0.907 | last ~19.8 / COMPARE **18.55** | Optimiser floor healthy indoors |
| DAv3 cascade | `dj2k-dav3-pose` | dav3-chunked **20/44** (8 chunks rejected) | depth | ~**22.0** @cycle 106 | ~**12.95** (stuck ~12.2-13) | **FAIL** mid-train: `mapAsync` / external Instance gone |
| SfM cascade | `dj2k-sfm-pose` | sfm 43/44 cams, **5/6** depth views merged | depth+SfM scale | **14.88** SSIM 0.472 | **12.63** / COMPARE **11.97** | Held-out **cross-match wrong target** on every sampled view |

Logs: `_runs/dj2k-gt-pose.log`, `_runs/dj2k-dav3-pose.log`, `_runs/dj2k-sfm-pose.log`.
Shots: `_shots/dataset/DrJohnson__dj2k-*.png`.

### What the delta means

1. **Optimiser is not the multi-image blocker.** With GT poses + points init, DrJohnson 2K
   hits ~32 dB supervised. Same train knobs cannot save a bad cascade.
2. **DAv3-chunked beats SfM on this room** for supervised fit, but **half the views never
   enter one frame** (8/14 chunks rejected on Umeyama residual >15% of anchor spread). Held
   out stays near the untrained floor (~13 dB).
3. **SfM is actively wrong here** when paired with DAv3 depths: cameras and geometry disagree
   enough that held-out images match the wrong target. Do not prefer SfM under `auto` for
   the joint-depth product path.
4. **Free-camera shots look like blobs even on the GT run.** Capture pose 0 looks nearly
   nadir (`fwd.y ≈ -1`); the FPS controller also drops roll (`up·worldUp` warnings). High
   supervised PSNR does not make free-* CDP shots readable - judge mid training cameras /
   gtpose-* seats, not only free-*.

## Defaults locked from measurement (pushed in working tree; commit when TJ asks)

| Knob | Was | Now | Why |
|---|---|---|---|
| `MultiViewGenerationService.PosePreference` | `"auto"` (SfM first) | `"dav3"` | SfM 14.9 + wrong cross-match vs dav3 ~22 |
| Dataset / studio query default `poses` | `"auto"` | `"dav3"` | Autotest was overriding the service default |
| `KeepOutsideReferenceView` | `false` | `true` | DrJohnson fuse: 0/39928 kept, 39915 "outside ref view" |
| Thin-pose warning in `RecordTrainingViews` | none | warn if posed*2 < images | Surfaces 20/44 without refusing train |

Object captures that need the old frustum screen: `?outside=0`. SfM A/B: `?poses=sfm`.

## Bathroom smoke (`bath2k-dav3-outside`)

With the new defaults (dav3 + outsideKept=True):

- **Cascade coverage: 34/35 posed, 0 chunks rejected** - much healthier than DrJohnson.
- Consistency fuse: 0 outside rejects; some views still die on "behind" / depth disagree.
- Init produced **~773k splats**; unconstrained prune then **device loss**
  (`A valid external Instance reference no longer exists` inside `ApplySplatPlanAsync` /
  `PruneUnconstrainedAsync`). Same failure class as growhost ~780k / DAv3 DrJohnson mid-run.
- Main CDP shot is **white** (train never started). Free shots captured after FAIL are not
  a quality signal.

So Bathroom is no longer "cascade cannot pose the room" under these defaults; it is
**host Apply / binding pressure at high splat count** plus whatever depth-disagree / behind
rejects remain. Raising `maxdensify` or GPU-side Apply is parked engineering; the DrJohnson
**chunk reject / fold** problem is the Fable-shaped cascade cliff.

## Fable brief (hand off when quota allows)

All three Fable-trigger conditions from the plan hold for **DrJohnson**:

1. GTPOSES=1 is healthy (~32 / ~19).
2. GTPOSES=0 with best Auto defaults (dav3, outside on) is far below that floor (held ~13).
3. Logs show cascade-internal failure: Umeyama chunk rejects (residual 17-30% of spread,
   limit 15%), not a missing URL knob.

Ask Fable to:

1. Root-cause why DrJohnson walk-through rejects 8/14 chunks while Bathroom accepts 14/14
   style coverage (anchor triangle scale swings 0.5-1.4× on pair 3-37).
2. Improve fold acceptance without placing wrong clouds (do not just raise the 15% limit).
3. Keep Bathroom's good coverage; do not regress outsideKept / dav3 defaults.
4. Optionally harden `ApplySplatPlanAsync` / prune so ~800k init does not kill the device
   (or cap init budget before prune on rooms) - secondary to (1)-(2).

## Fable findings (2026-09-23, Trip/Fable) - DONE in working tree, uncommitted

**Root cause (from `_runs/dj2k-dav3-pose.log` lines 1723-1796, not a theory).** Anchors 3/12/37.
Pair 3-12 distance ratio held 0.88-1.12 across all 13 passes (that is per-pass scale, absorbed by
the fold). Every pair with 37 swung 0.52-1.43. The model places 37 differently depending on which
new views share its pass. `TryFitChunkToReference` fitted **positions only**: 3 points
over-determine a similarity by 2, so nothing can name the liar; Umeyama smears 37's error over 3
and 12 and RMS crosses 15% -> 8 rejects. Bathroom's anchors 31/32/33 are three consecutive
frames the model relates trivially, hence 0 rejects there (its 27% worst pair on chunk 5 was on
the 0.018-long 31-32 edge, tiny in absolute terms).

**Fix (`MultiViewChunkPlan.TryFitChunkToReference`, new 6-out overload).** A camera is a pose,
not a point. Per anchor, `R_i = basis_chunk^T * basis_ref` is a direct frame-rotation estimate.
Every anchor PAIR whose R_i agree (<= `MaxAnchorRotationRadians`) proposes (mean R, distance-ratio
scale, midpoint translation); scored by anchors agreeing in position (15% of spread) AND
orientation; winner refitted on inliers (mean rotation, LS scale/translation). Accept needs
`MinFoldAnchors`=2 AND a strict majority of recovered anchors (2/3, 3/4, 4/6). Caller logs
`(anchor N excluded)` and a new `anchor rotation disagreement` line per chunk.
15% limit unchanged. Planning still carries 3 anchors (the third is what makes a bad one nameable).

Tests: `SpawnScene.Tests/MultiViewChunkPlanTests.cs` 22/22, full suite 199/199.
`ThreePoseAnchors_OneMisplaced_FoldsOnTheOtherTwoAndNamesIt` is the DrJohnson case with a red
check that point-only Umeyama rejects the same data. Old tests that built anchors with
identity orientation but rotated positions (data no extrinsics matrix produces) now use `Seen()`.

### Next agent (Auto is enough) - SUPERSEDED by Trip findings below

1. Publish + rerun `_runs/dj2k-dav3-pose.cmd`. Expect chunks 2,3,5,7,8,9,11,12 to fold with
   `(anchor 37 excluded)`; posed should go 20/44 -> ~44/44. Read the new
   `anchor rotation disagreement` lines: 3-12 should be small, x-37 large.
2. **Calibrate `MaxAnchorRotationRadians` (10 deg initial, NOT measured)** from those lines and
   Bathroom's. It must sit above 3-12's noise and below 37's disagreement. Record both numbers in
   the constant's doc comment.
3. Rerun `_runs/bath2k-dav3-outside.cmd`: must stay 0 chunks rejected. Bathroom's device loss at
   ~773k splats is the separate parked item (handoff item 4).
4. Held-out PSNR on DrJohnson vs the 12.95 floor; commit when TJ asks (Auto defaults + this).

## Trip findings (2026-09-23) - Fable fold verified; device loss fixed in ML

### 1. Fold: 44/44 posed, chunk 10 needed a second fix

Fable's pose-aware fold landed 44/44 / 0 rejected on first publish. Chunk 10 still excluded
anchor **12** instead of 37. Pair-score logging showed why: member position residual is
`(baseline/2)*sin(half direction error)`, so scoring it as a distance rewards short baselines.
Both 3-12 and 3-37 had a 5.3 deg direction misfit; 1.47 vs 0.63 was only the 1.04 vs 0.45
baseline. Fix: score pair members as an angle against `MaxAnchorRotationRadians`. Test
`ThreePoseAnchors_SharedDirectionMisfit_ShortBaselineDoesNotWin` (red-checked). After that,
chunk 10 reports `(anchor 37 excluded)` like the others. Suite 200/200.

### 2. `MaxAnchorRotationRadians` calibrated (DrJohnson)

10 deg kept. Honest 3-12: 1.0-3.4 deg. Liar 37: 52-54 down to 1.4 depending on chunk;
orientation alone names it on 7/12 bad passes, position vote catches the rest. Numbers are in
the constant's doc comment. Bathroom 2000 anchors 31/32/33: every disagreement 0.1-0.5 deg
(same gate, never rejects an honest Bathroom anchor).

### 3. Device loss ROOT CAUSE (was parked item 4) - FIXED in ML 5.2.20-local.8

Not Resize, not carry peak, not a 128 MiB binding. Windows GPU-process counter: dedicated VRAM
climbed to **7.4 GB** during the 14 joint passes; trainer working set is ~1.2 GB. Chrome
dropped the GPU process with no crash dump; page only saw `device.lost reason=unknown`.

`InferenceSession.Run` never returned graph OUTPUTS to the BufferPool (rented+pinned; decode
loop is the only path that recycled them). Plus free intermediate buckets are never reclaimed
on WebGPU (`createBuffer` never throws, so under-pressure reclaim never fires). One
`ReleaseWorkingMemory` after the cascade freed **4322 MB**. Then the 913k densify Resize
completed and training ran to DONE.

Pushed: SpawnDev.ILGPU.ML `11fe9dd` / 5.2.20-local.8. SpawnScene on that package calls
`_depthService.ReleaseWorkingMemory()` at cascade end and train start. Log to trust:
`[Depth] released NNNN MB of depth working memory`.

### 4. Held-out vs 12.95 floor (run7, `_runs/dj2k-dav3-pose.run7-fixed-20260923.log`)

| | |
|---|---|
| Posed | **44/44**, 0 rejected |
| Init | 920,688 → prune to 913,498 |
| Supervised @cycle 60 | **25.10** SSIM 0.8527 |
| Held-out peak | **13.83** @cycle 2 |
| Held-out last | **12.65** @cycle 60 |
| Prior stuck floor | ~12.95 |

Held-out still drifts down while supervised climbs (overfit to training views / pose noise).
It is no longer "stuck because train died." Commit of the SpawnScene tree still waits on TJ.

### Still open

- ~~Bathroom smoke (step 3)~~: **34/35 posed, 0 rejected**, released 4322 MB, train DONE past
  the old 773k device-loss point (`_runs/bath2k-dav3-outside.run-fixed-20260923.log`).
- **Held-out quality is pose shape, measured.** DrJohnson dav3-chunked vs COLMAP after best
  Umeyama (`_runs/dj-pose-vs-gt.log`): position RMS **109.7% of spread**, median forward error
  **61.5 deg** (p90 108.6). Held-out cross-match picks the wrong target on every sampled view;
  GT with the same bookkeeping picks its own. Coverage and device loss are no longer the
  blocker - the cameras do not describe the same room COLMAP does.
- Harness: `SPAWNSCENE_CHROME_LOG` + VRAM sampler are how the next silent GPU death gets named.
- Next: ~~TempleRing `?autotest=dav3-pose`~~ DONE (`_runs/dav3-pose-temple.log`): single-pass
  N=6 vs GT residual **5.4% / 4.5% of spread**, batch dependence **0.4%**. Extrinsics decode is
  correct. The DrJohnson 110% RMS is the **chunked cascade on a walk-through**, not `[R|t]` unpack.
  Next measurement: same single-pass gate on a DrJohnson 6-view subset vs COLMAP.

## Harness notes

- Publish: wipe `_pub_trip_pose` first, then
  `dotnet publish ./SpawnScene/ -c:Release --output _pub_trip_pose -p:BaseIntermediateOutputPath=obj/pub-harness/`
- Serve: `node tools/_spa_server.js _pub_trip_pose/wwwroot 8080`
- CDP: `SPAWNSCENE_CDP_PORT=9225`
- Chrome GPU log: `SPAWNSCENE_CHROME_LOG=_runs/....log` (wired in `_chrome_harness.js`)
- Wasm string check: search UTF-16LE at BOTH byte alignments (strings sit at odd offsets)
- Recipes under `_runs/dj2k-*.cmd`, `_runs/bath2k-dav3-outside.cmd`
