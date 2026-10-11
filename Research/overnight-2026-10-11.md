# While TJ was away, 2026-10-10 afternoon to 2026-10-11 (Tuvok)

Everything below is measured; the numbers and runs are in Research/parity-matrix.md. Defaults were changed under TJ's
standing call ("defaults are your decision ... quality and correctness is more important than speed but speed is
important"); each one was checked held out, on the user path (Bathroom) and off the photo path (Wander views).

## Where SpawnScene stands now (one scorer for every tool, 11 benchmark scenes, 7K)

| | PSNR | SSIM | LPIPS-alex |
|---|---|---|---|
| **SpawnScene (today's defaults)** | **27.44** | **0.862** | **0.166** |
| Brush v0.3.0 | 26.95 | 0.841 | 0.200 |
| gsplat 1.5.3 | 26.73 | 0.833 | 0.207 |

Best PSNR on 9 of 11. Behind Brush on Train (-0.85: the photos' own exposure; with exposure gains our content is ahead of
Brush's after a colour fit) and Bonsai (-0.29; the High preset gets 31.27 vs Brush's 30.47). Training 130-230 s a scene
(Counter 216 s) vs Brush 1.5-3 min wall incl. load (Counter 172 s).

## Shipped (deployed to spawnscene.com, verified)

1. **Deferred key count** (5da9983): no per-step GPU->CPU wait for the sort's key count - indirect sort on the GPU.
   Bit-identical render (TrainerGate); Counter 239 -> 200 s, Train 148 -> 127 s.
2. **No densify apply probe** (aa50054): a log-only check that cost ~1.2 ms a step with the GPU idle. Counter -3.5%.
3. **Densify bar 6e-4** (7b90e28; was 8e-4): all 11 scenes up, mean +0.18 dB, ~+10% training time; viewer (Bicycle, user
   path) 183 -> 155 fps uncapped, scene 91 -> 120 MB; off-path unchanged. The 10-06 test that kept 8e-4 ran on older
   defaults.
4. **High / Max presets densify at 4e-4** (0f71eb5): at 15K, Train 21.05 / Bonsai 31.27 / Counter 28.82 vs 20.96 / 30.95 /
   28.66 at 6e-4, ~+27% time. Standard / Draft / Custom keep 6e-4. Off-path checked (Bathroom).
5. `&densifygrad` now applies on the user path too (f9d5056; it was dataset-only).
6. **UI fixes**: project card Delete opened the project and Confirm did nothing (fixed); card hover turned solid cyan.
   Both fixed in SpawnDev.GameUI **1.0.0 on nuget.org** (TJ-approved): only the topmost element gets a click, and the
   hover ring is a ring.

## Measured and NOT shipped (opt-in flags, for the record)

- Train's gap to Brush is bright blobs in the held-out SKY: an opaque far splat shell (6.7% of splats past 16 camera
  spreads; Brush has none) grown only by densify. Tried: SH cap, far prune, far-growth stop, Brush's whole growth scheme
  (selection, max-over-views signal, relocation), exposure affine - none closes it at Standard.
- **Learned sky map** (`&skymap=1`, PLANS item 6): Train +0.26, Bathroom +0.17 held out - but off the photo path the room
  leans on the map (Wander see-through 1.5% -> 2.2%, worst view 7.8 -> 14.4%) and the viewer does not draw it.
  **Decision for TJ if wanted:** an outdoor-only sky with the viewer and file format carrying it.
- Batching the per-step passes: not worth it - the step is GPU-bound now (encode 4 ms of ~33).
- Densify itself: ~5% of a step. Parked.

## Waiting on TJ

- Single-photo hidden layers: `&edgesnap=1` + `&inpaint=1` (MI-GAN) - your judgement on the live site (PLANS item 0).
- A real phone video end to end (the TruckVideo user-path test passed: 22.41 vs 22.93 from the same photos).
- Viewer speed on the Quest / a laptop GPU (desktop holds 60 fps everywhere; we are 0.55-0.61x GaussianSplats3D uncapped).
- Sky map product scope (above).

## Shared GPU

Data (Vertex) released SpawnDev.ILGPU.ML 5.3.5 + 5.3.6, SpawnDev.AI 1.1.0, AI.Server 1.1.0 + 1.1.1 overnight; the GPU
was traded through the board and messages without overlapping runs.
