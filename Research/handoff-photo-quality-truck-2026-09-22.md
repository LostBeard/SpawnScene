# Handoff: Photo-quality Truck splat (2026-09-22)

Next agent: read this, then `AGENTS.md` + memories under
`C:\Users\TJ\.claude\projects\D--users-tj-Projects\memory\feedback_conic*` and
`feedback_end_idx*`, `feedback_dead_view*`. Claim **SpawnScene** on the board before editing.

## Goal (unchanged)

Multi-metric win only if **all** hold: sharp shots (rails/tire/foliage), conic live tens of %,
scale median shrink vs init ~0.017, supervised PSNR ≥22 @7K, held-out near paper ~21 dB.
COMPARE never overrides a soft shot. Plan file (do not edit):
`C:\Users\TJ\.cursor\plans\Photo quality Truck-c87442ce.plan.md`.

## Status

Phase 1 geometry learning is **unblocked**. Full DoD is **not** met (still soft, ~17 dB supervised).

| Metric | Best measured | Target |
|---|---|---|
| Conic live (2K gate) | 50% → 35% → 22% | ≥~20% mid |
| Conic live (7K final late) | 10–23% (diluted by densify) | tens of % |
| Scale median | 0.017 → **0.0135** (final 7K) | clearly below init |
| Supervised @7K | **17.4 dB** (peak sample ~17.3) | ≥22 |
| Held / COMPARE @7K | **15.4** / **13.7** | ~21 |
| Shots | anisotropic streaks, still soft dreamscape | hard edges |

## What landed (uncommitted unless you commit)

- `FixedPointGradScale`: Fit exp [-16,40], `PercentileAbsNonzero` (peak×1e-6 floor), `BlendToward`
- `SplatTrainerGpu`: **MinConicScale / DefaultConicScale = 2^26** (match CentreScale); p50 EMA Fit + clamp sample; retarget every 32 steps; scatter addend **±1e5**; clear colour/T/end/keys/values each forward; scale hist sample API
- `Studio.Training`: scale hist log; **do not DropDeadViews** (keep ~4–7 zero-grad views)
- Densify knobs used: `densifygrad=1e-6`, `densifyfrac=1`, `maxdensify=450000`, hybrid Adam/SH remap kept
- Tests: `FixedPointScaleFitTests` (6 passing)

**Do not** force `my_end = range.y` in `raster_backward`. That zeroed dead views (0/95) but cut supervised ~17→14 (`truck7k-endidfix`). Keep end_idx clamp.

## Key runs

| Tag | Notes |
|---|---|
| `truck2k-conic2e26-20260922-183229` | Phase1 pass: conic ≥20%, scale starts shrinking |
| `truck7k-conic2e26-20260922-185101` | First 7K w/ 2^26; densifygrad **1.5e-6** → only ~273k splats; sup 17.4 / held 15.3 |
| `truck7k-endidfix-20260922-195921` | end_idx bypass; **0 dead**; PSNR regress; killed mid-run |
| `truck7k-final-20260922-203058` | **Canonical latest**: end_idx restored, densify 1e-6 → **~450k** cap; sup 17.4 / held 15.4 / COMPARE 13.7 |

Logs: `_runs/<tag>.log` (+ `.pids`, `.cmd`). Shots: `_shots/dataset/Truck__<tag>*.png`.
Publish slot used: `_pub_trip_conic2\` (nuke obj when republishing). CDP Trip often **9225**; kill only PIDs in that run’s `.pids`.

## Why soft remains (proven mechanisms)

1. **Conic was dead from undersized ConicScale (~2^12 vs centre 2^26)** → fixed by MinConicScale 2^26 + clamp.
2. **Densify at 1.5e-6 starved growth**; 1e-6 hits maxdensify 450k (use that).
3. **~6/95 views** still `RASTER_BACKWARD` (keys+rgb live, mean T~0.01, gradPerKey=0) with correct end_idx. Root cause open. Full-tile walk is the wrong fix.
4. Scale shrinks but not enough for photo edges in 7K; Adam sees clamp-attenuated conic magnitudes (uniform atten. should be OK for Adam—verify if mixed clamp/no-clamp is the issue).

## Next session — highest leverage order

1. **Dead-view root cause** on recurring indices (9, 34, 54, 74, 100…): sample `end_idx` vs tile `range`, NaNs in `grad_a`, depth range for those cams. Instrument one failing view; do **not** bypass end_idx globally.
2. **Sharper geometry**: if conic stays live, probe higher `LogScaleLr` (today 0.005) or scatter clamp/Fit so typical keys are not always at ±1e5. Re-gate 2K: conic ≥20% + median scale clearly <0.01 + mid shot less soft than `Truck__truck7k-conicfix-20260922-150147.png`.
3. **Then** one 7K with densifygrad=1e-6, maxdensify=450k, opacityreset=3000, hybrid remap. Judge shots first; PSNR second.
4. Only if 720 is structurally sharp but soft at pixel scale: try `maxdim` ~979.

## Out of scope until above

30K schedule, new datasets, stochastic display path, chasing dB with densify floods past 450k.

## Board / repo

- Repo: `D:\users\tj\Projects\SpawnScene\SpawnScene` (git nested one folder in).
- Many related changes still **uncommitted** (trainer/shaders/tests). Commit only if TJ asks.
- Trip should claim SpawnScene on `_DevComms/board.md` when picking up.
