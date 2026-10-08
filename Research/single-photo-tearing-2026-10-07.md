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
| DA3-LARGE (multi-view) | CC-BY-NC-4.0 | - | NOT usable (commercial) |
| **MoGe-2 ViT-S / ViT-B** (Microsoft) | MIT | ~35M / ~100M params | "sharp details" + metric scale + normals; boundary F1 17.9 on iBims-1 vs Depth Pro 14.3 |
| Depth Pro (Apple) | Apple sample-code licence (use and redistribute with notice) | ~1.9 GB | best boundary F1 on Sintel (0.409 vs DAv2 0.228); too heavy for a browser default |

Delivery: through hub.spawndev.com (never huggingface.co directly), OPFS-cached; a large model is a one-time download.
Runtime: SpawnDev.ILGPU.ML (Data's area) - an ONNX export of each candidate must run there first.

### Seeing behind things (attacks 3) - TJ's object-removal idea

This is "3D photo inpainting" (Shih et al., CVPR 2020: context-aware layered depth inpainting): inpaint colour AND
depth in the band behind each depth edge, so the revealed background is plausible scene, not a blur.

| Model | Licence | Size | Notes |
|---|---|---|---|
| **MI-GAN** (Picsart, ICCV 2023) | MIT | ~7x smaller than LaMa | already runs in browsers (ONNX Runtime Web + WebGPU: inpaint-web) |
| LaMa (big-lama) | Apache-2.0 | ~200 MB | large masks (FFC), the standard object remover |

Plan: OcclusionFill already knows WHERE the hidden layer goes (the far side of each depth edge, the band past the
frame). Build the mask from it, inpaint the photo there (MI-GAN first), and colour the hidden layer from the inpainted
image instead of the push-pull blur; its depth stays the far-side depth.

## Order

1. ~~MaxCellStretch A/B~~ (no effect).
2. A larger / sharper depth model behind a setting: DA3MONO-LARGE and MoGe-2 ViT-S through ILGPU.ML (ask Data).
3. MI-GAN for the hidden layer's colour.
4. Then the same depth refinement for DepthFusionInit (TJ: the multi-photo path seeds from the same depth maps).
