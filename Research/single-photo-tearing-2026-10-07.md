# Single-photo tearing: what causes it, what can fix it (2026-10-07)

TJ, 2026-10-07: "single photos scenes have a lot of tearing. any chance some depthmap and source image preprocessing
could help... the depth map edges could be better aligned with the color image" - then: "that same depth map
preprocessing may also help with the multi image pipeline", "ml models that can remove objects from a scene... where we
need to see behind things that we cannot really see behind?", "maybe other/larger depth models are helpful/worth trying?"

## What the moved views show (kitchen sample, 3840 px, CC0; generate-room &sampleurl=kitchen&holeviews=1&holebg=0)

Three different things read as "tearing":

1. **Ramps at depth edges.** The model sees ~518 px; its depth is resized to the photo smoothly, so an object edge is a
   ramp several photo pixels wide. The unproject kernel refuses to tilt a splat toward a neighbour >5% deeper, but on a
   ramp each step is <5%: splats orient along it and stretch (MaxCellStretch 8 footprints) - a sheet from foreground to
   background.
2. **Surfaces the photo sees edge-on.** The cereal box's side, chair backs: the photo has almost no pixels for them, so
   a few cells stretch to cover them (MaxCellStretch) - colour smears and zebra stripes from an orbit.
3. **The hidden background layer showing.** Behind every depth edge OcclusionFill puts a background layer coloured by a
   push-pull blur: from a moved camera it reads as grey streaky sheets. With &occfill=0 those areas are black holes -
   it is hole filling, working as designed, but it looks like tearing.

## Tried tonight

- **DepthEdgeSnap** (`&edgesnap=1`, opt-in): each pixel whose neighbourhood spans >3% of its depth takes the
  colour-weighted median of depths on a 5x5 grid spaced by the upsampling factor (~7 px at 3840). Effect on (1):
  small - edges a little cleaner; the box and chair smear (2) unchanged. Wider windows (&snapr=3/4, &snapstep=1.5/2)
  break surfaces into speckle and staircase edges. Not a default.
- **MaxCellStretch** (`&cellstretch=X`): 3 and 2 look the same as 8 from the moved views. Not the cause.

**Conclusion:** the smear is the DEPTH. DAv3 Small makes the cereal box a slanted wall receding into the counter and the
splats draw exactly that; no post-processing of that depth can recover a box. The lever is a sharper/larger depth model
(below), then inpainting for what was behind.

## Candidates

### Better depth (attacks 1 and 2 at the source)

| Model | Licence (weights) | Size | Notes |
|---|---|---|---|
| DAv3 Small (ours) | Apache-2.0 | ~50 MB fp16 | 518 px input; soft edges |
| **DA3MONO-LARGE** | **Apache-2.0** | ~350M params, ~700 MB fp16 | DAv3's monocular-specialised large model - the obvious first try |
| DA3METRIC-LARGE | Apache-2.0 | same | metric depth (single-photo scale!) |
| DA3-BASE | Apache-2.0 | ~120M params | multi-view, for the pose/fusion path |
| DA3-LARGE (multi-view) | CC-BY-NC-4.0 | - | allowed (SpawnScene is non-commercial, TJ 10-07), flagged NC; prefer open |
| **MoGe-2 ViT-S / ViT-B** (Microsoft) | MIT | ~35M / ~100M params | "sharp details" + metric scale + normals; boundary F1 17.9 on iBims-1 vs Depth Pro 14.3 |
| Depth Pro (Apple) | Apple sample-code licence (use and redistribute with notice) | ~1.9 GB | best boundary F1 on Sintel (0.409 vs DAv2 0.228); too heavy for a browser default |

Delivery: through hub.spawndev.com (never huggingface.co directly), OPFS-cached; a large model is a one-time download.
Runtime: SpawnDev.ILGPU.ML (Data's area) - an ONNX export of each candidate must run there first.

### Seeing behind things (attacks 3) - TJ's object-removal idea

This is "3D photo inpainting" (Shih et al., CVPR 2020: context-aware layered depth inpainting): inpaint colour AND
depth in the band behind each depth edge, so the revealed background is plausible scene, not a blur.

| Model | Licence | Size | Notes |
|---|---|---|---|
| **MI-GAN** (Picsart, ICCV 2023) | MIT | ~7x smaller than LaMa | already runs in browsers (ONNX Runtime Web + WebGPU: inpaint-web). The author's HF repo `andraniksargsyan/migan` has `migan.onnx` (512x512 net) and `migan_pipeline_v2.onnx` (pre/post-processing for any image size inside the graph) |
| LaMa (big-lama) | Apache-2.0 | ~200 MB | large masks (FFC), the standard object remover |

**MEASURED 2026-10-07 (onnxruntime CPU, tools/migan_test.py):** the plain `migan.onnx` (opset 12; uint8 `image`
[B,3,H,W] + uint8 `mask` [B,1,H,W], 255 = known, 0 = fill; ops Conv/LeakyRelu/Clip/Resize/Pad/Cast/Reshape/Transpose -
no data-dependent shapes) on a 512 crop of the kitchen sample with a 256 px hole over the cabinets and a pendant lamp:
the cabinet door, the range hood's edge and the tile line continue, the lamp shade is completed - see
`img/migan-kitchen-2026-10-07.png` (left: hole, right: MI-GAN). The pipeline export (`migan_pipeline_v2.onnx`) adds
NonZero / GatherND / ScatterND (data-dependent shapes): use the plain net and crop around each hole ourselves.

**MEASURED in the browser (autotest=inpaint-parity, 2026-10-07):** the float-I/O variant (tools/migan_float_io.py; hosted
as LostBeard/spawnscene-models/migan_float.onnx with the MIT notice, fetched through the hub) runs on SpawnDev.ILGPU.ML
(WebGPU) and matches onnxruntime to 3 decimals on 0..255 (hole centre 227.314 vs 227.314; red check: the hole differs
from the input by 19.8 mean). Loads in 1.3 s, first 512x512 run 2.7 s including kernel compiles.

**Integrated (opt-in `&inpaint=1`, 2026-10-07):** OcclusionFill masks the cells that get a hidden behind-edge splat,
letterboxes photo + mask into MI-GAN's 512 square, and those splats take the painted colour (HiddenLayerInpaint; the
push-pull blur stays the fallback). Kitchen, with &edgesnap=1, moved views (`img/inpaint-hidden-layer-kitchen-2026-10-07.jpg`,
top: blur, bottom: MI-GAN): the yellow/orange blobs of foreground colour behind the cereal boxes are gone - the
revealed area reads as countertop and backsplash. Soft (512 px painting of a 3840 px photo): tiles of 512 around each
masked region would sharpen it. Load 1.2 s, paint 518 ms.

**Tried and reverted: 512-cell tiles at grid resolution** for the behind-edge layer (20 tiles, ~0.15 s each). Worse:
at full resolution the masked band sits against the foreground object's UNMASKED interior (only cells within the
fill radius of an edge are masked), and MI-GAN continues the box's colours into it - the yellow blobs came back. The
single 512 pass sees the band surrounded mostly by background. The fix is the mask, not the resolution: mask the whole
near layer of an object (depth-layer segmentation, or a dilation of the near side until the depth jump), then paint.

**&inpaintreach=X** (mask the near side out to X fill radii): 2 and 3 look the same as 1 in the single 512 pass on the
kitchen - at that scale the band is already surrounded by background. Kept as a knob for a tiles + wide-mask retry.

**Past the frame too (same &inpaint=1):** the padded grid letterboxed into 512 with the margin masked; the border layer
takes MI-GAN's continuation. Garden path from moved views (`img/outpaint-garden-2026-10-07.jpg`, top: push-pull, bottom:
MI-GAN): the olive smeared tunnel around the photo becomes sky, trees, grass and the path running on. Soft at 512 px;
paint 0.25-0.45 s per layer.

Plan: OcclusionFill already knows WHERE the hidden layer goes (the far side of each depth edge, the band past the
frame). Build the mask from it, inpaint the photo there (MI-GAN first), and colour the hidden layer from the inpainted
image instead of the push-pull blur; its depth stays the far-side depth.

## Measured: DepthEdgeSnap v2 (two plateaus by colour), kitchen

The first snap (colour-weighted median) failed its own unit test: ramp samples share the pixel's colour, so the median
picked the ramp. v2 splits the window at the midpoint depth into near and far sides and gives the pixel the plateau
(lo/hi) of the side whose mean colour it matches; same-colour sides = a slope, untouched (DepthEdgeSnapTests: ramp ->
plateaus, slope unchanged). From the moved views: the cereal box is compact instead of a sheet smeared over the counter -
the geometry tearing is gone at that edge. What opens behind it now shows OcclusionFill's push-pull BLUR (soft colour
blobs) - the case for inpainting the hidden layer (MI-GAN). Still opt-in (&edgesnap=1) until TJ has looked.

## Measured: DAv3 Base vs Small (kitchen, 2026-10-07)

`&depthmodel=depth-anything-v3-base` (onnx-community, Apache-2.0, ~500 MB, now selectable in project settings): from the
orbit view the cereal box stands more upright with a shorter smear, the island edge is cleaner; thin objects (the
hanging lamps) smear in both. A real but modest gain for 5x the download - an option, not the default yet. MoGe-2
ViT-S/B (author's ONNX exports: Ruicheng/moge-2-vits-normal-onnx, -vitb-) is the next candidate; it outputs a point
map + normals, so it needs its own pipeline in ILGPU.ML (Data). DA3MONO-LARGE has no ONNX export yet.

Licences (TJ 2026-10-07): SpawnScene is non-commercial, so NC weights are allowed; prefer open ones, flag NC.

## Measured: MoGe-2 vs DAv3 Small, and what the snap really fixes (2026-10-07 late)

Numbers, not impressions. `tools/depth_model_compare.py` runs both models on CPU onnxruntime (author's ONNX exports via
the hub: Ruicheng/moge-2-vits-normal-onnx 141 MB, -vitb- 419 MB; DAv3 Small onnx-community 105 MB) at 1008 px, unprojects
each with MoGe's own focal and point-renders orbits of 20/35 degrees (`img/moge2-vs-dav3-kitchen-2026-10-07.jpg`).
`tools/flying_pixels.py`: inside a 7x7 window with a >25% depth step, the share of pixels stranded in the middle 20-80%
of the step - the points that become rubber sheets in a moved view.

| photo | DAv3 Small | MoGe-2 ViT-S | MoGe-2 ViT-B |
|---|---|---|---|
| kitchen | 0.39% | 0.39% | 0.40% |
| living room | 0.31% | 0.22% | 0.30% |
| castle room | 0.59% | 0.80% | 0.81% |
| garden path | 2.30% | 3.53% | 4.29% |

**MoGe-2 does not have fewer flying pixels.** Its depth maps LOOK crisper (island, cereal box outlines) and its moved
views are slightly cleaner on the kitchen, but at the steps it is a tie indoors and worse on foliage, at 1.3x (S) / 4x (B)
the download. Regression depth networks all blur steps; swapping models does not fix tearing. What MoGe-2 does offer:
metric scale, a focal estimate and a validity mask (sky) in one pass. Its graph's ops (41 types incl. If, ReduceL2,
ConvTranspose) are all in ILGPU.ML's registry. Parked as an option, not the fix.

**The snap is the lever - with a slope guard.** `tools/snap_flying_eval.py` (numpy port of DepthEdgeSnap):

| photo | raw | snap (colour test only) | snap + bimodal guard 0.3 |
|---|---|---|---|
| kitchen | 0.39% | 0.28% | **0.25%** |
| living room | 0.31% | 0.25% | **0.22%** |
| castle room | 0.59% | **0.32%** | **0.32%** |
| garden path | 2.30% | 3.71% (worse) | 2.49% |

The colour-only snap terraced the textured garden path (gravel + brick pass the two-sides-differ test by chance;
`img/edgesnap-bimodal-guard-2026-10-07.jpg`, top middle). An EDGE's depths bunch at two plateaus; a SLOPE's spread evenly.
The guard: snap only when at most 30% of the window lies in the middle half of [lo, hi]. It keeps every indoor gain and
removes most of the garden harm (DepthEdgeSnapTests.ATexturedSlopeIsLeftAlone, red before: 291 px snapped). A colour
coherence (Fisher) guard was tried first: it traded the indoor gains away with the garden harm. Snapping twice is worse
everywhere (never iterate it). MoGe-2 + snap is the best indoor pair measured (kitchen 0.19%, living room 0.11%) - its
edges sit closer to the colour edges.

## Measured: inpainting mask reach (kitchen, 2026-10-08)

`&inpaintreach=1/2/3` (the behind-edge mask widened to 2-3 fill radii into the near side, so MI-GAN sees less of the
foreground's interior as context): moved views orbit30 / xneg20 / dolly30 in `img/inpaint-reach-kitchen-2026-10-08.jpg`.
Wider is WORSE: a dark stripe down the fridge's edge (xneg20) and a blurred ghost of the high chair (orbit30) at reach 2
and 3; reach 1 is the cleanest. The default stays 1. With tiles at full resolution (also worse, earlier) both ways of
"more context control" are closed; the remaining lever for the hidden layer is a better inpainter (LaMa, Apache-2.0, 200 MB)
or painting depth too (3D-photo inpainting's layered depth).

## Measured: LaMa vs MI-GAN for the hidden layer (2026-10-08)

`tools/inpaint_compare.py` (CPU onnxruntime, the same 512 crops and holes, `img/migan-vs-lama-kitchen-2026-10-08.jpg`):
three kitchen objects removed - the high chair, a pendant lamp, the cereal boxes. **LaMa (big-lama, Apache-2.0,
Carve/LaMa-ONNX lama_fp32.onnx, 208 MB) is clearly better at "what is behind it"**: it continues the cabinet and floor
where MI-GAN paints another chair, and the cabinet face where MI-GAN leaves the lamp's pole as a dark streak. Both
struggle with the cereal boxes (LaMa continues the backsplash tiles but keeps a blue smear). MI-GAN is 0.6 s, LaMa 1.6 s on
CPU; 30 MB vs 208 MB.

LaMa did not run on SpawnDev.ILGPU.ML: two library bugs, both fixed in the ML repo (a4a5c011) - N-D broadcasting in
compile-time constant folding (LaMa's DFT matrices [64,1] * [33] folded to [64,1]) and ConvTranspose output_padding
(64 -> 127 -> 253 -> 505). After the fixes LaMa on OpenCL matches onnxruntime to 0.003 (0..255). SpawnScene has
`&inpaintmodel=lama` wired (HiddenLayerInpaint; served through the hub's /hf, cached), but it needs an ML package
with those fixes: SpawnScene is on ML 5.3.2 from nuget.org, so until a release (TJ's go) the option logs the compile
failure and keeps the blur.

## Order

1. ~~MaxCellStretch A/B~~ (no effect).
2. ~~A larger / sharper depth model~~: measured, MoGe-2 S/B do not reduce flying pixels (above). DA3MONO-LARGE has no
   ONNX export to test. Edge snap + bimodal guard is the fix; judge it in the app (&edgesnap=1) next.
3. MI-GAN for the hidden layer's colour.
4. Then the same depth refinement for DepthFusionInit (TJ: the multi-photo path seeds from the same depth maps).
