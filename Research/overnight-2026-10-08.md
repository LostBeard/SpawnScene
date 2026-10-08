# Overnight 2026-10-07/08 (Tuvok had the conn) - what changed, what needs TJ

Live is still **d2b6d44**. Nothing deployed, no defaults changed, no user content published. Everything below is pushed.

## Decisions for TJ

1. **Make `depthinit=4` + `exposure=gains` the defaults** (my recommendation). Held out, same build per pair:

   | capture | change | held out | fair score (gains fitted on the left half, right half scored) |
   |---|---|---|---|
   | Bathroom (phone room) | depth init (g0 -> g1) | 15.58 -> 16.91 dB, SSIM 0.709 -> 0.795 | - |
   | Bathroom | exposure=gains (j2 -> j3) | 16.88 -> 18.57 dB, SSIM 0.796 -> 0.844 | 18.88 -> **24.32** |
   | Bicycle | depth init (b0r -> b1) | 25.17 -> 25.16 (neutral) | - |
   | Bicycle | exposure=gains (j0 -> j1) | 25.14 -> 24.99, SSIM up | 25.42 -> **25.58** |
   | Truck | exposure=gains (k2 -> h2) | 24.07 -> 24.00 (noise) | - |

   Off the photo path (Wander views) equal or better in every pair. Cost: ~3 s of depth fusion, ~15% more splats.
   The full affine `exposure=1` should stay an option only: its offsets do not fold into the scene (-0.2..-0.6 dB on
   fixed-exposure captures). Detail: quality-roadmap-2026-10-07.md.
2. **An ML release** (5.3.4 awaits your go) would carry two library fixes I made in SpawnDev.ILGPU.ML (a4a5c011:
   N-D broadcast in constant folding, ConvTranspose output_padding). They make big-LaMa run (= onnxruntime to 0.003);
   SpawnScene's `&inpaintmodel=lama` lights up with it. PMT model lanes were NOT run on the fold change - Data has a
   note asking for it in the next release gate.
3. **SPZ v4** (zstd; already in PlayCanvas's examples) needs a decoder dependency: ZstdSharp (MIT, pure C#) or a small
   JS one. Your call; until then such files are refused with that reason.
4. **Deploy** the pushed work below when you have looked (it changes no defaults).

## Shipped to main (not deployed)

- **Open other tools' scenes**: 3DGS .ply, SuperSplat compressed .ply, .sog (bundled or meta.json URL), .spz v2/v3,
  .splat - all decoded on the GPU, turned y-up, seated on the dense core. Verified on Inria Train, antimatter15
  train.splat, Spark butterfly/penguin, PlayCanvas biker/guitar/skull. scene-formats-2026-10-08.md.
- **Capture feedback**: the scene card now actually shows "N of M photos placed" (it was hidden under the buttons) and
  which directions no photo faces; the Photos tab says WHY each photo was left out (SfM's own reason).
- **Fair held-out score** logged for every run ("held out RIGHT HALF").
- **Depth supervision** `&depthloss=X` (gate-verified) - measured neutral, opt-in.
- **Exposure gains-only** `&exposure=gains`.
- **Edge snap slope guard** (single photo): kitchen flying pixels 0.39 -> 0.25%, garden no longer terraced.
- **Depth model load retries**; a failed load no longer saves an untrained collage silently.
- `&inpaintmodel=lama` (needs the ML release).

## Measured and parked

- MoGe-2 ViT-S/B: no fewer flying pixels than DAv3 Small - a bigger depth model is not the tearing fix.
- Mip 3D filter: neutral to +0.07 dB; the close-up Wander views need a surface target to test it properly.
- Inpaint mask reach > 1: worse. LaMa vs MI-GAN: LaMa better behind objects, MI-GAN sharper on foliage.
- Depth loss at 7K and 1.5K: within noise. A thin room's held-out saturates by ~1,500 iterations.
