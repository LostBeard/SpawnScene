# Project settings: presets, overrides, device limits (design, 2026-10-02)

TJ, 2026-10-02: "these source images and my hardware are just possible resources... some people will have far better
hardware and very different source media.... we should allow access to settings that promote the best results but
also allow tweaking based on the hardware, and desired quality/time investment by the user."

Design rules:
- Defaults aim for the best result the device can afford.
- Every knob is MEASURED for what it buys before it gets a place in the panel.
- Limits come from the device's real WebGPU limits, not constants.
- Nothing caps below the source silently: the run logs the size it used and why.

## What the measurements say each knob buys

All scored with `tools/score_views.py` at a fixed viewport against the original photos.

| Knob | Evidence | Verdict |
|---|---|---|
| Front end: learned (RaCo-ALIKED + LightGlue+) vs FAST/BRIEF | Bathroom 32/34 vs 8/34 cameras; DrJohnson core within 1.2% of COLMAP; TruckFull 19.63 vs 19.78 dB at 3K | Learned by default. FAST/BRIEF stays as a fallback (no model download, any device). |
| Training iterations | TruckFull 7K: 23.1-23.2 dB, sharpness 0.98 (3K: 21.6 trainer held-out). Bathroom 2K -> 7K: no held-out change (14.6-15.8 -> 14.5-14.7) | Real for well-covered scenes; the preset's main time/quality dial. |
| Training resolution (above the import size since f5559f1) | Bathroom 720/1024/1600: no measurable difference at 2K or 7K. Truck photos are 979 px, so nothing to gain there | Keep, cap at the photo's size, default 1024. Not a headline quality lever on the data we have. |
| Max splats | GPU memory bound (TruckFull reached 1.7M under 3M) | Device-derived default, override in Advanced. |
| Dense init FAST threshold (&densefast) | Bathroom: 10 doubles dense points (2,268 -> 4,243), held-out within noise | Advanced only, default 25 until a dataset shows a gain. |
| Learned keypoints 1024 / 3072 | Models exist; not yet measured end to end | Measure before exposing. |
| More / better-placed photos | Bathroom held-out views between 24 supervised photos stay ~15 dB whatever the trainer does | Say so in the UI: the single biggest lever for a small room is coverage. |

## Iterations measured (TruckFull, 2026-10-02, b108 / b121 / b122)

| Iterations | Trainer held-out | Captured held-out @1600x892 | SSIM | Sharpness | Splats |
|---|---|---|---|---|---|
| 7K (Standard) | 23.20 dB | 23.10 dB | 0.831 | 0.98 | ~1.6M |
| 15K (High) | 24.11 dB | 23.38 dB | 0.845 | 1.09 | 1.74M |
| 30K (Max) | 24.51 dB | 23.44 dB | 0.845 | 1.24 | 1.73M |

- Captured views are rendered at 1600 px from a 979 px training size, which caps that metric. The trainer's own held-out
  number tracks the iterations better.
- 15K and 30K ran beside another GPU job (16 it/s against 24), so time is quoted only as ratios.
- 1.74M splats measured 2.98 GB live, inside Auto's 4 GB budget.

## Presets (multi-photo)

| Preset | Iterations | Resolution | Max splats | Front end | Use |
|---|---|---|---|---|---|
| Draft | 3K | 720 | 500K | learned | a quick look, weak GPU |
| Standard (default) | 7K | 1024 | device max | learned | most captures |
| High | 15K | 1600 (capped at photo) | device max | learned | well-covered scenes |
| Max | 30K | photo size | device max | learned | best result, long run |

Splat caps (2026-10-02, TJ: "3 million splats is very low... the 5k living room image with single image scene generation
would generate over 14,000,000 splats and render at 60fps. VERY large scenes are a part of the goal"):
- Every preset but Draft now takes **device max**: as many as the GPU memory setting fits, and never more than one
  binding of the widest per-splat row holds (45-float SH rest, 180 B: 11.9M at 2047 MiB). Max splats offers 500K / 1M /
  3M / 6M / 10M / Device. GPU memory offers up to 48 GB. Projects saved under a preset with the old 3M cap upgrade on
  load.
- Training cost MEASURED by group (b126, TruckFull 7K, 1.58M splats): SH bank 720 B/splat, tile keys 352 (a floor of 8
  keys a splat), other per-splat 120, Adam 112. The key floor is GONE (b127): the key buffers grow when a frame
  overflows and re-emit it (trainer gate "key growth" stage), so capacity follows measured demand (~2-3 keys a splat
  late in training). Keys 532 -> 133 MB, live 2,779 -> 2,378 MB, held-out 23.25 -> 23.23 dB (noise), one growth in
  the run. ~1,050 B/splat now; a 12 GB setting trains ~10M splats.
- Of the ~1,050 B/splat, the degree-3 SH bank (value, gradient, two Adam moments, 45 floats each) is 720.
  That is the next lever for big scenes on a given GPU (fp16 moments / fp16 SH storage), and past 11.9M the SH banks
  must split across bindings. Rendering is far cheaper per splat (14M at 60 fps).

Changing any row below the preset switches the preset label to "Custom" (the values stay where the user put them).

## Device-derived limits ("GPU memory" setting)

WebGPU does not expose VRAM. What it does expose, and what SpawnScene already reads:
- `maxStorageBufferBindingSize`: 2047 MiB on the RTX 4070 / Chrome.
- `maxBufferSize`.

Today two budgets are constants:
- the training target stack, 640 MB in projects and 256 MB in the dataset path;
- the max-splat cap.

Proposal:
- **GPU memory budget** setting: Auto / 2 GB / 4 GB / 8 GB / 16 GB.
  - Auto = min(device binding limit, a conservative tier from adapter info).
  - The target stack and the splat cap are derived from it: target stack = 30% of the budget, capped at the binding limit.
- The trainer already measures key demand per window (the keysPerSplat probe), so keys need no setting.
- When a budget forces a lower training size, show it in the status line ("training at 768x1024 to fit the 2 GB budget"), not only in the console.

## Panel layout (implemented d2efc66, to extend)

Sidebar sections, top to bottom:
1. **Quality**: preset segmented control (Draft / Standard / High / Max / Custom) and an estimated time line once a
   run has measured it/s on this device.
2. **Reconstruction**: training resolution, iterations, max splats (the current rows).
3. **Device**: GPU memory budget (Auto default), with the detected limit shown.
4. **Advanced** (collapsed): front end, learned keypoints, dense FAST threshold.
5. **Single photo**: depth quality and depth model (the current rows).
6. Generate, pinned at the foot.

## Open before implementing
- Estimated time: DONE (TrainingTimeEstimate). Each run of 1K+ iterations stores this device's seconds per
  iteration-megapixel in localStorage. The Training iterations hint shows "Training takes about N min on this computer"
  for the project's photos at the size the trainer would pick. Splat count is not in the model (TruckFull 1.7M splats
  ran 16-24 it/s at 979x546), so it says "about". The old hint's "70 minutes at 30K" was stale (b122: 31 min).
- The budget fit now picks the LARGEST size that fits (square root), not 3/4 steps from the setting
  (TruckFull at 1600 px in 256 MB: 692 px, was 676).
- Learned k3072 vs k1024 end-to-end measurement.
- Settings are per project today (ProjectSettings). Device settings (GPU memory budget) belong to the device, so they
  go in app-level settings (localStorage is fine, a per-viewer convenience).
