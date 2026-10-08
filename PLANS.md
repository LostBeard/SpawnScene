# SpawnScene Plans

Goal (TJ, 2026-10-07): **the best Gaussian splat generator and viewer on the web.** Quality first: a scene has to hold up
when the camera leaves the photo path, on the photos people actually take (phones, rooms, thin coverage), not only on
the benchmark captures.

Detail lives elsewhere; this file is the map and the order:
- Research/quality-roadmap-2026-10-07.md - the survey (what the best generators do, evidence, our status).
- Research/demo-samples-2026-10-07.md - demo sources, licenses, hosting.
- Plans/lod-streaming.md - massive scenes (done: LOD tree, .spawnscene v3, paged streaming, partitioned training).
- Research/README.md, validation-strategy.md - the 3DGS reference numbers and how to measure without fooling ourselves.
- CLAUDE.md "Training defaults worth knowing" - every default with the measurement that set it.

## Where we are (2026-10-07)

**Multi-photo (the main product):** learned matching (RaCo-ALIKED + LightGlue) -> GPU SfM (five-point, rotation
averaging, GPU bundle adjustment, dense FAST/BRIEF init) -> WebGPU 3DGS training (AbsGS densification, SH 3, D-SSIM,
pose refinement, floater census carve, size cap 0.1 x rig radius) -> sorted full-resolution viewer, LOD streaming,
WebXR. Versus gsplat on the same photos and COLMAP poses (TruckFull 30K): 24.95 dB / SSIM 0.887 with 0.89M splats vs
25.13 / 0.877 with 3.79M. At parity on the benchmark.

**Single photo:** DAv3 depth (EXIF or estimated focal) -> splats sized to their cells -> occlusion fill (background
behind depth edges, the photo continued past its frame) -> Scene depth control.

**What still goes wrong (measured or seen by TJ):**
1. Rooms from phone photos (TJ's Bathroom): smears and empty areas off the photo path, black where no photo looked.
   Init is 7.8K SfM points for a whole room; walls fill from those seeds or not at all. Exposure varies 6x.
2. Far background from extreme poses (Bicycle, low look-up pose): depth-ambiguous specks.
3. Photos dropped by the pipeline: a landscape photo among portraits is skipped by the multi-view depth pass; 2 of 35
   Bathroom photos unplaced. The user is not told which or why.
4. Popping when turning (one depth per splat in the global sort).

**Fixed 2026-10-07:** the floater census counted every splat on a not-yet-opaque pixel as a floater (37c1fd2) - the carve
was deleting the thin walls of exactly the captures that need them (TJ's "TONS of holes"). The stochastic renderer was
the Generate default (now sorted, full resolution). AOT trainer crash (ILGPU 5.3.5). End-carve splats left in the
file (compacted).

## Next, in order

Each item: measured on Bathroom (phone room), Bicycle (outdoor 360) and TruckFull (object) - held-out photos AND the
Studio.Wander views (in/up/low/out/mid, pan-0..7, over-0/1), side by side with the run before. A change that wins the
held-out number and looks worse off-path does not ship.

1. **Re-baseline the carve on the census fix** (k1 Bicycle, k2 Truck, h6-h9 Bathroom, running). Decide the unseen bar
   (1 px costs Bathroom 0.77 dB supervised; try 0.05) and whether the in-training carve earns its keep on rooms
   (Bathroom: supervised 37.4 dB without it, 33.5 with; held-out equal).
0. **Single-photo tearing, measured 10-08:** MoGe-2 ViT-S/B (MIT) do not have fewer flying pixels than DAv3 Small (tie
   indoors, worse on foliage) - a bigger model is not the fix. The edge snap is, with its slope guard (bimodal depth
   window, `&snapmid`): kitchen 0.39 -> 0.25% flying pixels, castle 0.59 -> 0.32, garden no longer terraced.
   Research/single-photo-tearing-2026-10-07.md.
0. **MEASURED 10-07 late:** Bathroom `&depthinit=4 &exposure=1` held out 15.58 -> 18.27 dB, SSIM 0.709 -> 0.836 (g4) -
   make both the defaults once Bicycle (b0/b1, e1) and Truck (e2) show no loss. Single photo: `&edgesnap=1` +
   `&inpaint=1` (MI-GAN) make the hidden layers plausible - TJ to judge on the live site (URL flags work there).
2. **Per-photo exposure** (opt-in, gate-verified). MEASURED 10-08: the full 3x4 affine costs fixed-exposure captures
   0.2-0.6 dB (its offsets do not fold into the scene); `&exposure=gains` (per-channel gains only) wins the phone room
   (h0 Bathroom 18.64 dB vs affine 18.23), is neutral on Truck (24.00 vs 24.07 none), -0.22 dB / SSIM up on Bicycle under
   the mean-exposure score. Fair score (gains fitted on the left half, right half scored, every run) added; j0-j3 decide
   the default with it.
3. **Depth from DAv3 in training** (we compute it for posing and throw it away):
   a. Dense init: each photo's DAv3 depth aligned (scale/shift) to the SfM points it sees, back-projected on a grid,
      voxel-thinned - seeds on every surface a photo saw, not only where features matched.
   b. Inverse-depth L1 against the aligned depth, weight 1.0 -> 0.01 over training (the reference's `-d`). Indoor first.
      **Built 2026-10-08** (`&depthloss=1` with `&depthinit`; gate-verified against finite differences). MEASURED: neutral
      on Bathroom at 7K (d1 18.20 vs d0 18.23) - stays opt-in; i0/i1 test short schedules (a 1500-it run matched 7K).
   a. is measured: Bathroom +1.3 dB, Bicycle neutral (b0r 25.17 vs b1 25.16) - default candidate for TJ.
4. **Capture feedback:** after SfM, show which photos were dropped and why, and a coverage ring (headings with photos,
   as the pan views log). Fix the landscape-photo skip in the multi-view depth pass.
5. **Anti-aliasing for the viewer:** Mip-Splatting 3D filter (`&mipfilter`) default-on if the in/out wander views
   gain; 2D Mip filter in the viewer.
6. **Far background:** a background shell / far-depth prior for sky and distant scenery (the Bicycle low-pose specks).
7. **Density control at 30K:** revisit MCMC at equal budget, error-driven densification (Bulo et al.), Taming-style
   budget for the device.
8. **Viewer:** per-tile sort against popping (StopThePop). Import: 3DGS .ply, compressed .ply, .sog, .spz v2/v3 and .splat
   DONE 2026-10-08 (GPU conversion, y-up turn); next SPZ v4 (zstd: needs a decoder).

## Demo samples (done 2026-10-07, pending TJ)

samples/catalog.json -> huggingface.co/datasets/LostBeard/spawnscene-samples: 3 sets (Hamamni Baths interior, ceramic
pine cone, Korno rock; CC BY-SA 4.0 / CC0) + 5 CC0 photos at 3840 px. Each set gets judged off the photo path before it
stays (s1-s3 runs). Wanted from TJ: OK to add his Bathroom set; 2-3 phone captures of his own (an object on a table, a
room walked around, an outdoor scene) - the only multi-photo sources that are both clean and ours.

## Standing measurement rules

- Off the photo path or it did not happen (Wander views in every project/dataset autotest; compose_wander.py).
- Per-photo numbers, not only means: one photo losing 13 dB was the census bug's signature.
- Against gsplat at the same poses (render_turns.py) when a claim is about parity.
- Phone captures (Bathroom) next to the benchmark sets; the benchmarks have no auto exposure and no thin rooms.
- No timing comparisons while a peer's heavy job runs; check the board first.
