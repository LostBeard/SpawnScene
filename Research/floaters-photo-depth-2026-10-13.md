# Floaters vs the photos' depth maps (2026-10-13)

TJ, live site, pine cone sample (Standard, then Max): "soooo many blurry floaties all over the room" the moment the camera
moves; the old single-photo DA2 path (14M splats from one 5K photo, every splat ON the depth surface) had none. Question:
are the depth maps used at all?

## What the code did (before today)

- Multi-photo: depth seeds the start (DepthFusionInit, `depthinit=4`); training is photometric only.
- `&depthloss` (L1 on the blended MEAN inverse depth, decaying 1 -> 0.01) existed but was off: "neutral" on Bathroom held
  out - photos that sit beside training photos, where a floater lines up with what is behind it.
- The floater carve measured each splat against the render's OWN surface (where T crosses 0.5): an opaque floater IS that
  surface and reads ~0.5 (TrainerGate: 0.4922), so the 0.9 carve kept it.

## Built (3e53cce .. 5135841)

- Census against each photo's depth map (`&depthcheck=1` measure, `&depthcarve=1` carve) + agreement bins.
- Free-space loss `&freespace=X`: per pixel the opacity in front of (1 - 0.15) x the photo's depth, 1 - T_f.
- Depth maps: `&depthtargetpx` (default 384; 672 = the multi-view prediction size measured the same, below).
- TrainerGate PASS: opaque floater 1.0 vs photo depth (0.49 vs the render's surface); free-space gradient vs finite
  differences within 2.5%; red checks.

## Measured - pine cone, 7K, 1024 px

**User path (dA-dD, no held-out photos):** 39% of all blending weight in front of the photos' depth (>15%); 60% of pixels
render a surface nearer than the photo's depth. Depth carve 39% -> 37% (weak: few splats are in front in >= 90% of their
weight). Free space: pixels nearer than half the photo's depth 15.8% -> 0.0%.

**Supervised PSNR rewards floaters.** Same dataset path, poses only swapped: our SfM 21.3 sup / 15.71 held out; COLMAP
poses (depth-unprojected start) 31.4 sup / **12.4** held out. Judge held out (and off the photo path), never supervised.

**Not the cause:** our focal (BA ends at 803 vs COLMAP 784; 1007 was the start value), lens distortion (COLMAP
SIMPLE_RADIAL k1 -0.089 vs SIMPLE_PINHOLE: 23.21 vs 23.62 sup, 15.73 vs 15.65 held out), our poses (held out 15.71 ours
vs 15.65 COLMAP). Brush v0.3.0 on COLMAP's undistorted photos: 16.23 / 0.697 held out (12 views) - the same radial smear.
The capture itself: close-ups with the room out of focus, a mirror-glossy table, a person in one shot.

**Held out, own SfM, dataset path, same 10 photos, shared scorer (tools/rescore.py):**

| run | flags | PSNR | SSIM | LPIPS-alex | weight in front of photo depth | train |
|---|---|---|---|---|---|---|
| sA | depthcheck | 15.64 | 0.657 | 0.559 | 39.3% | 152 s |
| sDL | + depthloss=1 | 16.26 | 0.680 | 0.520 | - | 169 s |
| sF1 | + freespace=1 | 16.31 | 0.686 | 0.519 | 0.39% | 123 s |
| **sF03** | **+ freespace=0.3** | **16.49** | **0.694** | **0.501** | - | 128 s |

Off the photo path (Wander pan views = turning around in the room): `img/pinecone-freespace-pan-2026-10-13.jpg` (left
sA, right sF03) - walls flat, frames / door / armchairs / floor readable, the fog mostly gone.

**Round 2 (same 10 held-out photos):** weight 0.3 at 384 px 16.54 / 0.7107, at 672 px 16.55 / 0.7109 (100 vs 33 MB: kept
size is not the limit); weight 0.1 16.47 / 0.7062 (5.5% in front); 0.3 + depthloss=1 16.49 / 0.7093 (adds nothing);
cover mode (`&freespacecover`, push scaled by the opacity behind) 16.51 / 0.7098.

## Validation for a default (llffhold=8 held out, `&freespace=0.3` vs off)

| scene | path | base | free space 0.3 | note |
|---|---|---|---|---|
| Pine cone | own SfM, dataset | 15.64 / 0.657 | **16.49 / 0.694** | LPIPS 0.559 -> 0.501; room readable off-path |
| Pine cone | user path | 14.53 / 0.646 | **14.68 / 0.677** | right half 14.88 -> 15.30 |
| Kitchen360 | own SfM, dataset | 31.71 / 0.929 | 31.53 / 0.930 | 2.6x faster (353 -> 137 s); off-path fog gone (`img/kitchen-freespace-out-2026-10-13.jpg`), one see-through gap |
| Bathroom | user path | 19.21 / 0.859 | 18.64 / 0.855 | ALL on held-out photo 30 (HDR): 16.45 -> 13.81 (reproduced 16.51 -> 14.01); the other three up; cover mode 14.14 |
| Bicycle360 | own SfM, dataset | (baseline run hung at boot) | 24.63 / 0.728 | 2.8M splats, 484 s |

Not a default yet: the Bathroom photo-30 loss is unexplained (cover mode did not fix it, so not the see-through hole
alone). Next batch (tuvok-y.sh): held-out renders on the user path (`&dumpheld`, 0a01889), per-photo held-out pose
refinement in the log (5135841), and the HELD-OUT photo depth measure (53c9a28) - the held-out photos' own depth maps,
which never reach training (TJ's idea; the supervised measure partly grades its own homework once free space trains
against those maps).

## Evening (2026-10-13): depth start and depth model, on today's defaults (free space 0.3 cover)

**Depth seeds, pine cone own SfM held out** (same 10 photos): agreement mode (default) 16.56 / 0.711 (639K seeds);
uncontradicted `&depthinitmode=free` 16.65 / 0.718 (289K seeds - FEWER: it also drops agreed samples another photo sees
through); stride 2 16.46 / 0.695 (2.55M seeds: denser hurts); free + stride 2 16.57 / 0.709.
Uncontradicted across scenes vs the default: pine cone user path 14.68 -> 14.73 (SSIM +0.008, 130 -> 108 s), Bathroom
18.59 -> 18.61, Kitchen 31.70 -> 31.65, **Bicycle 24.65 -> 24.43** (thin foliage: other photos rightly see through gaps).
Stays opt-in.

**Depth model:** the multi-photo path always ran DAv3 Small - the Depth model setting reached only single photos (fixed
098aca6; the first "Base" run measured Small twice, same 639,154 seeds). DAv3 Base for real (439 MB model storage):
pine cone 16.30 / 0.698 vs Small 16.56 / 0.711 - worse (it also moves the poses). With 384 vs 672 px maps the same, the
depth maps' quality is not the lever on this capture; DA3METRIC-LARGE (a second, mono model for the maps only) parked.

**High preset** (pine cone user path, 15K iterations at 1600 px, `&preset=High`, held out): free space off 13.90 / 0.653
(SSIM ENDS BELOW ITS START 0.664; 62.6% of the weight in front of the held-out photos' depth; 1.32M splats, 772 s) - on
(the default) 14.24 / 0.681, 5.5% in front, 1.03M splats, 603 s. Higher presets grow more floaters on this capture;
free space matters more there. `img/pinecone-high-preset-freespace-pan-2026-10-13.jpg` (left off, right on).

## 2026-10-14: Bathroom photo 30 - the dump and the logged score disagree (CPU-only look, no GPU)

Photo 30 (the HDR window photo) is the whole Bathroom loss for free space (16.56 -> 13.92 logged, hB0 vs hB3). The
`&dumpheld` renders of both arms (`_shots/dataset/Bathroom__tuvok-hB0/hB3__held-IMG_20260223_133612910_HDR-trainer.png`)
scored against the photo's JPEG (resized 3120x4160 -> 768x1024, bilinear; numpy, no GPU):

| arm | logged `final view 30` | dump vs JPEG | window box MSE | window pixels > 0.95 |
|---|---|---|---|---|
| off (hB0) | 16.56 | 16.65 | 0.0494 | 5% |
| on (hB3) | 13.92 | **16.62** | 0.0479 | 26% (photo: 29%) |

The OFF arm's dump agrees with its logged score; the ON arm's does not (2.7 dB). So the logged 13.92 is NOT what the
dumped render looks like against the photo: either the dump (StashTrainerRenderAsync: viewer's packed buffer after the
end carve, `hv.Camera`, SH degree capped to the viewer's) and the score (EvaluateAsync: the trainer's `packed`, `views[i].Camera`,
`ScoreAgainstAsync` vs the training target) see different scenes/cameras for this view, or the target differs from the JPEG.
Checked in code: targets carry no EXIF exposure scaling; both paths call `RenderForwardAsync`. Not yet known which side is
wrong, so no claim about floaters or the window from either number. Visually (dump): off = hazy grey veil over the
window's left part; on = the blue pattern and a white band much nearer the photo's contrast (window brightness 0.815
vs photo 0.726, off 0.773). The "free space costs Bathroom 0.6 dB" row in the table above rests on the logged score.

Next (one 2.5-minute Bathroom 7K run when the GPU is free, not an hour): in the same process log the score of view 30 AND
write the trainer's render, the training target (ReadTargetAsync) and the dump's render for it, then compare the three
pixel for pixel. That settles whether the logged held-out number or the dump is the faithful one before more is built on it.

### Resolved (same day, one Bathroom 7K run, `&dumpheld=1`, free space ON): the in-app PSNR does not clamp the render

New `HELD-EVAL` log (Studio.TrainerRenderDump.DumpHeldEvalFramesAsync): for each held-out view, the trainer's score and a CPU
PSNR of THE SAME render against THE SAME target (frames written as `held-<name>-evalrender.png` / `-target.png`):

| view | trainer score | CPU PSNR, same frames (clamped) |
|---|---|---|
| 0 | 17.04 | 17.04 |
| 8 | 22.10 | 22.10 |
| 23 | 21.69 | 21.83 |
| 30 (HDR window) | **13.88** | **16.52** |

`eval_sse` (SplatTrainerShaders) is `rendered - target` with NO clamp on the render, and the render is the raw SH colour; a
picture, the viewer, and the PNG scorers (gsplat/Brush/`rescore.py`, all 8-bit) clamp it. Only the view whose window blows out
differs (2.64 dB), and free space's brighter window made the ON arm read as the loss. So **the Bathroom -0.6 dB row was
overshoot the picture never shows; view 30 on a clamped score is 16.52, level with off (16.56 logged, no overshoot).** Every
in-app held-out PSNR before this fix is slightly low where a render exceeds 1 (largest on HDR / window captures); the
parity numbers vs Brush/gsplat were scored from PNGs by `rescore.py` and are unaffected. SSIM still scores the unclamped
render (shared with the training loss); not changed.

Correction (10-15): an earlier version of this note said the viewer-side `&dumpheld` render was DC-only (viewer SH degree 0). WRONG: the `viewer SH degree 0` in the HELD-EVAL lines is logged BEFORE the viewer receives the trained bands (`[Train] viewer SH degree 3` comes after). Compared pixel for pixel, the old viewer dump and the new `-evalrender` frame are the same image (mean difference 0.004/255 on all four views), and both score exactly what a clamped CPU PSNR says (16.52 for view 30). So the viewer dump was faithful and the clamp explains the whole 13.88 vs 16.62 gap.
Fix (65cd9d7; gate green: gpu 14.4992 dB = clamped oracle, unclamped 12.3490; red-check with the clamp removed FAILS by name): `eval_sse` clamps; TrainerGate PsnrGateAsync (no gate scored PSNR before: the SSIM
case discards it) writes a hostile render (x1.8 and below-0 stripes) into the output buffer and compares with a clamping
oracle, plus a vacuity check that clamping moves the score > 0.5 dB.

## 2026-10-15: the window holes (Bathroom, default Floater cleanup), measured over a magenta background

The Wander views of the default Bathroom run showed large BLACK blobs beside the window (pan-2, pan-3). Captured over
`&bg=1,0,1` (the viewer's clear colour; holes turn magenta, dark splats stay black) they are **see-through holes**, not
dark splats. Share of magenta pixels per Wander view, same Bathroom 7K user path, one variable changed (`&freespace=0`):

| view | free space off | default A | default B (repeat) |
|---|---|---|---|
| pan-2 (toward the window) | 0.03% | 7.29% | **2.91%** |
| pan-3 | 0.00% | 1.20% | 0.73% |
| close-3 / in-3 / low-3 / out-3 (the side no photo faces) | 22.1 / 19.1 / 17.3 / 14.4% | 27.4 / 24.2 / 21.2 / 17.6% | 27.8 / 24.7 / 21.8 / 18.1% |
| mean over 28 views | 2.64% | 3.56% | 3.44% |

Held out (clamped scorer, 4 photos): off 19.29 / 0.8585, default 19.32 / 0.8533 and 19.29 / 0.8526. So free space is level
on held-out PSNR here (the earlier -0.6 dB was the unclamped scorer) and a little worse on SSIM (-0.005).

What is solid: with free space on, the side no photo faces has ~3-5 points more see-through (both repeats agree, 27.4 and
27.8 vs 22.1), and pan-2 / pan-3 have window holes the off arm does not (off has a white veil there instead: the floaters
it removes were covering the gap). The room that photos did see (walls, shelf, curtain, mirror) is clean in all arms.

**Run-to-run noise is large**: pan-2 of the SAME default flags was 7.29% then 2.91%. A single run per arm cannot rank
anything near that size. (Held-out PSNR repeats within 0.03 dB; SSIM within 0.001; the see-through shares do not.)

Tried and dropped: `&freespacesat=0.9` (no free-space push where the photo's smallest channel is above 0.9, on the idea
that blown-out window pixels carry no depth evidence): pan-2 3.09%, pan-3 0.22%, mean 3.24% - inside the noise above; held out
19.26 / 0.8538. Not kept (the code was reverted, never committed). Open: a fill for directions no photo saw is a separate
feature (parity first, fill later); free space only makes the honest emptiness visible where the floaters used to hide it.

## 2026-10-15: pine cone (user path) at the MAX preset, free space + cover on (the default), over magenta

One run (`tuvok-mx1`, AOT build 65cd9d7 + clamp): 92 photos at 2048x1365 (this sample's photos are 2048 px on the long side; the Auto 4 GB
budget allows photos up to 1024 MB, so nothing was shrunk), 30,000 iterations in 1,755 s (17.1 it/s), 1.08M splats at the end.

| preset (pine cone, user path, held out llffhold=8) | PSNR / SSIM | weight in front of held-out depth | train |
|---|---|---|---|
| Standard 7K, free space on (10-13) | 14.68 / 0.677 | 4.8% | 130 s |
| High 15K at 1600 px, free space off (10-13) | 13.90 / 0.653 | 62.6% | 772 s |
| High 15K at 1600 px, free space on (10-13) | 14.24 / 0.681 | 5.5% | 603 s |
| **Max 30K at 2048 px, free space on (10-15)** | **14.09 / 0.667** | **5.70%** | 1,755 s |

(The 10-13 rows used the scorer before the clamp fix; the pine cone has little over-range colour, but it was not re-measured.)
Max does not beat High or Standard on held-out PSNR on this capture, and the floater measure holds (5.7% in front vs 62.6% for
High with free space off). The end carve removed 15,162 floaters + 331,498 unseen splats (held out 14.127 -> 14.092).

Wander views (turning around in the room, over magenta): the pine cone is sharp (diamond pattern, petals), the room is
readable (chairs, frames, window) and there is no fog. What is left: SEE-THROUGH VOIDS in directions the capture does not face -
magenta share pan-2 **50.85%**, pan-3 18.45%, pan-1 5.76%, low-0 15.46%, low-3 14.92%, out-0 7.51%, out-3 7.83%, over-0 5.54%;
mean over 30 views 4.22% (Bathroom default: 3.4-3.6%). Everything the photos saw is solid. The project page already tells the
user which directions no photo faces (capture feedback). A fill for those directions is the open feature; it is separate from
floaters and from parity.

## 2026-10-15: fill unseen directions (the 10-08 prototype) as a user setting, and what was wrong with it

TJ 10-15: "yes, please do fill ... make sure it is an option the user can change". The prototype (`&fillunseen=1`, UnseenFill, big-LaMa
paints the uncovered pixels of views rendered from the capture centre, painted cells become splats) existed since 10-08 (see
`quality-roadmap-2026-10-07.md`). Now `ProjectSettings.FillUnseen` + a Quality row "Fill unseen directions" Off | On (Off by default:
invented content; On = big-LaMa, 208 MB once); `&fillunseen=1` still wins from the URL. Tests: ProjectSettingsFillUnseenTests (red-checked).

Pine cone user path, Standard 7K, over magenta, one variable per step (fl0 none; fl1 the 10-08 schedule):

| arm | mean see-through, 30 Wander views | held out | note |
|---|---|---|---|
| fl0 no fill | 2.07% | 14.73 / 0.6777 | low-3 16.95%, out-3 8.24%, over-0 7.39% |
| fl1 the 10-08 fill (x3) | 0.35% | 14.70 / 0.6773 | 264,791 splats, 36.9 s; the up view 89.6% hole -> painted BLACK |
| fl2 + pitch +-65 rings | 0.32% | 14.72 / 0.6781 | up view 1.6% hole; but the rings' known context was still black |
| **fl4 + trainer resize** | **0.36%** | 14.72 / 0.6775 | context is real ceiling colour; black dome gone (over-0 0.00%) |
| fl3 far-priority depth pyramid (tried) | 0.53% | 14.72 / 0.6780 | no effect on the streaks, over-0 6.97%: dropped |
| fl5 depth-discontinuity guard (tried) | 0.59% | 14.72 / 0.6772 | no effect on the streaks, over-0 6.74%: dropped |

Root causes found from the fill's own strips (`&filldump=1`: input | painted | depth | mask): (1) the up view was 89.6% hole, so LaMa had no
context and painted black - fixed by +-65 pitch rings, painted inward; (2) the splats a view pastes rendered as opaque BLACK in the next
views, because the trainer's per-splat buffers (colour, radius, ...) are sized for the TRAINED splat count and a render of more splats
cannot store their colours - the fill now resizes the trainer first (`SplatTrainerGpu.SizedSplatCount`). (2) was invisible in the
see-through measure (black counts as covered), only the strips showed it. Not found: the thin radial streaks seen from far off-centre
(over / out views) and a few uncovered gaps (low-0: 4.4%): two guesses (far-priority depth, a depth-discontinuity guard) were A/B'd and
changed nothing. Inside views (pan-*) are clean in every arm. The setting path was checked end to end (`&fillsetting=1`, no URL
override): `[Inpaint] LaMa loaded`, `[Fill] unseen fill (lama)`. Held out is unchanged by the fill (it paints directions no photo faces).

### Fill seams (same day): disk overlap and the "known" threshold - mixed, defaults unchanged

Thin magenta seams remain where the fill meets trained content (Bathroom ceiling contour). Two harness knobs (defaults unchanged: overlap 1.0,
known 0.7): `&filloverlap=X` (each fill disk's scale) and `&fillknown=X` (coverage 1 - T at or above which a pixel counts as the scene's).
Measured with the strong-magenta share and a continuous see-through measure (mean of ((R-G)+(B-G))/2 over the 30 Wander views, over magenta):

| scene | arm | strong | continuous | held out |
|---|---|---|---|---|
| Bathroom | no fill | 3.323% | 5.348% | 19.32 / 19.29 |
| Bathroom | fill 1.0 / 0.7 (default) | 0.051% | 1.985% | 19.26 |
| Bathroom | fill 1.3 / 0.7 | 0.011% | 1.875% | 19.29 |
| Bathroom | fill 1.3 / 0.9 | 0.030% | 1.677% | 19.32 |
| Pine cone | no fill | 2.065% | 5.038% | 14.73 |
| Pine cone | fill 1.0 / 0.7 (default) | 0.321% | 2.722% | 14.72 |
| Pine cone | fill 1.3 / 0.9 | 0.178% | 1.901% | 14.73 |

Bathroom with 1.3 / 0.9: pan-2's seams are gone and the ceilings are cleaner. Pine cone with 1.3 / 0.9: the metric is better and the PICTURE IS
WORSE - the radial streak fans around the pine cone's tip and silhouette are much larger (the fringe pixels next to a near object are painted
at depths that bridge from the object to the far wall; viewed from the side they are sheets). A see-through metric rewards coverage and cannot
see this; only the Wander images do. So the threshold is not adopted as a default. Open: the bridge surfaces are the cause of the streaks (a
hole next to a near object should take the BACKGROUND depth, not an interpolation between object and wall); the far-priority pyramid tried
earlier did not remove them at known 0.7, so the next test is a far-priority depth restricted to the fringe pixels, judged on the images.

**Far-priority redone on the corrected pipeline (rings + trainer resize; the first test ran while later rings still had black context):** pine cone, setting-driven fill, only the pyramid changed: strong magenta 0.321% -> 0.576%, continuous 2.72% -> 3.24%, held out 14.72 / 0.6776 vs 14.72 / 0.6770, and the streak fan over the pine cone is unchanged (the outside view over-0 regains large holes). So the streaks are NOT bridge surfaces between a near object and the wall. They converge on the capture centre (the pine cone) and look like the painted ceiling shell, stretched along the ray at grazing incidence, seen from a viewpoint off the centre; inside views (pan-*) are clean. Not pursued further: an invented shell is only right from the centre, and the setting says so in its hint.
