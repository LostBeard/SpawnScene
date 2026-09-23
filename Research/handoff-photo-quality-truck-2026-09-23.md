# Handoff: Photo-quality Truck splat (2026-09-23, supersedes 2026-09-22)

Next agent: read `AGENTS.md`, then memories `feedback_fixed_point_grad_accum_was_root_cause.md`
and `feedback_near_plane_not_behind_eye_test.md`. The six `feedback_conic_*` memories are
SUPERSEDED (they tuned a bug). Claim **SpawnScene** on the board before editing.

## Status: two root causes found and fixed, quality is now photo-like

| Metric | 2026-09-22 best | Now (2K, densify on) | Now (7K, densify OFF) | Target |
|---|---|---|---|---|
| Supervised PSNR | 17.4 @7K | **22.58 @2K** | 19.51 | >=22 @7K |
| Held-out PSNR | 11-15 (baseline was 10.6) | **18.50** (COMPARE 17.33) | 10.72 (pre near-plane fix) | ~21 |
| Dead views | 5-9 of 95 | **0 of 95** | 5-9 | 0 |
| Shots | soft dreamscape | wood grain, rust, hub, mud flap | photo-like | hard edges |

Shots: `_shots/dataset/Truck__truck2k-np-densify-probe*.png`,
`_shots/dataset/Truck__truck7k-f32-nodensify-20260922-225358*.png` (free-left has readable
"Sanford Square Market" lettering).

## Root cause 1 (commit 9476714): i32 fixed-point gradient accumulator

Scale fitting + per-key +-1e5 saturation destroyed conic a:b:c ratios. Replaced with f32 CAS
atomic add (`atomicCompareExchangeWeak` on u32 bit patterns). TrainerGate 2D grad error
7.7e-4 rel -> 0.13 ppm. Densify criterion restored to Kerbl (mean over views of
||dL/dmean2D|| in NDC, 2e-4).

## Root cause 2 (commit b94d160): near-plane cull was a behind-the-eye test

`cz <= 1e-6` let splats 3-25 m to the SIDE of a camera but within 1e-4 of its plane through.
They project to a frame-covering footprint, the f32 conic is garbage, one splat clamps to
MAX_ALPHA at 100% of pixels (zero derivative) -> loss but no gradient (the "dead view"
signature: mean T == 0.01, no NaNs). The same splats made every HELD-OUT view render flat,
which is why held-out PSNR sat at the untrained baseline while free cameras looked right.
Fix: `NEAR_PLANE = 0.2` (graphdeco `in_frustum`) in project(), raster_backward, GeometryAdam,
`SplatGeometryGradients.MinDepth`, TrainerGate oracle, viewer shader. Unit test
`Project_CullsAtTheReferenceNearPlane_NotJustBehindTheEye`, red-checked.

## Densify

The 7K densify run BEFORE the near-plane fix stalled at ~15 dB; with the fix, 2K densify
reaches 22.58/18.50. The `[Densify] apply probe` (PSNR of 3 supervised views before/after
each apply) drops ~1.5 dB on the first applies and ~0.05 dB later. `?densifynoop=1` runs the
apply path on an empty plan; run `truck2k-np-densify-noop` answers whether the apply path
itself costs anything (expect before == after). Read `_runs/truck2k-np-densify-noop.log`.

## Runs in flight / queued (CDP 9225, SPA on 8080 serving `_pub_trip_np/wwwroot`)

- `_runs/truck2k-np-densify-noop.log` - apply-path bisection (chained after run A, cmd 21372)
- `_runs/truck7k-np-densify.log` - headline 7K with densify + opacity reset at 3000 (cmd 23044,
  waits for 21372). Compare to paper Truck 7K ~23 test PSNR.

## Harness

`tools/_cdp_dataset.js Truck <iters>` with env GEOM=1 GTPOSES=1 DENSIFY=100 OPACITYRESET=3000
DENSIFYFRAC=1 MAXDENSIFY=450000 HELDEVERY=2 INIT=points RUN_TAG=<tag> SPAWNSCENE_CDP_PORT=9225
EXTRA="&densifynoop=1". Launch detached via `Invoke-CimMethod Win32_Process Create` on a
`_runs/<tag>.cmd`. Publish first: `dotnet publish ./SpawnScene/ -c:Release --output _pub_trip_np`,
then `node tools/_cdp_trainer_gate.js` (compiles every shader; a missing WGSL const only
shows up here or in a run).

## Next

1. Read noop + 7K results. If 7K supervised >= 22 and held-out ~20+, Truck DoD is met on PSNR;
   judge shots visually (rails, tire, foliage).
2. If the noop probe shows a drop, the apply path (host round trip, Resize, InitOptimizerState,
   hybrid Adam/SH remap) is the bug; otherwise the early 1.5 dB apply drops are plan semantics
   (clone alpha compounding, split opacity) and recover within the interval.
3. Opacity reset fires at 3000 in the 7K run; watch the held-out dip and recovery.
4. Then: real captures (Bathroom, room) without GT poses - the pose cascade is the next
   variable, now that the optimiser is proven on GT poses.
