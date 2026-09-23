# Differentiable rasteriser audit — 2026-09-21 (Sonnet 5, fresh-context read)

Requested by TJ as a second pair of eyes on the open question: supervised PSNR sits at 15-18 dB
fitting views directly (should reach 30+), and the newest finding (commit `0fa9b73`) is that
view 0 produces zero gradient across all 79,922 splats. Audit only — no code changed.

Every claim below is labelled by how it was established. Where I did not execute something,
I say so; do not read "I traced it by hand" as "I proved it runs correctly."

---

## VERIFIED BY EXECUTION

**`dotnet test SpawnScene.Tests -c Release --filter "SplatRasterizer|SplatTileRasterizer|SplatGeometryGradients|SplatCovariance"` — 39/39 passed**, including the central-finite-difference
gradient checks (`Gradients_MatchFiniteDifference_*`, `ColourGradient_MatchesCentralFiniteDifference`,
`ConicGradient_MatchesCentralFiniteDifference`, `Mean2DGradient_MatchesCentralFiniteDifference`,
`OpacityGradient_MatchesCentralFiniteDifference`) and the CPU tile-vs-simple-rasteriser agreement
checks (`TiledForward_MatchesTheSimpleRasteriser`, `TiledBackward_MatchesTheSimpleRasteriser`,
`TiledBackward_MatchesWhenPixelsSaturateEarly`).

This is real, run-today evidence that the **CPU oracle** (`SplatRasterizer`,
`SplatTileRasterizer`, `SplatGeometryGradients`) is currently correct on this build. It does
**not** by itself prove the WGSL/GPU kernels agree with that oracle — see below.

---

## READ FROM SOURCE, NOT EXECUTED — high confidence, but flagging the epistemic gap

**1. WGSL forward/backward math (`SplatTrainerShaders.cs`) traces consistently against the
CPU-verified formulas.** I hand-derived the projection (quaternion→rotation→covariance,
world→camera transform, perspective Jacobian, 2×2 conic inversion), the forward alpha
compositing, the backward transmittance-recovery/suffix-colour recursion, and the full
~190-line geometry chain rule (conic → Σ2D → Σ_world → rotation/scale/quaternion/position) term
by term. Every sign and factor matched standard EWA-splatting / 3DGS backward math and the
comments' own claims about what they mirror.

**I did not run the actual GPU-vs-CPU gate.** `tools/_cdp_trainer_gate.js` exists specifically
to compare the WGSL kernels' output against this CPU oracle on real WebGPU hardware
(`/studio?autotest=trainer-gate`), and I did not launch it — TJ asked me to stop after the CPU
test run, and this needs a build+publish+served app+Chrome CDP session, which is a materially
bigger and more disruptive action than a filtered `dotnet test`. **Do not treat "the shader math
looks right on paper" as equivalent to "the gate passes."** Running that gate is the single
highest-value next step to convert this section from "traced by hand" to "verified."

**2. A real bug in the census's own reliability, in `SplatTrainerGpu.TrainStepAsync`
(`SpawnScene/Services/SplatTrainerGpu.cs`, roughly lines 1206-1235):**

```
await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
if (LastKeyCount == 0) return 0f;
...
_lossFixed!.MemSetToZero();
...
_gradFixed!.MemSetToZero();
```

`_gradFixed` is only cleared *after* the `LastKeyCount == 0` early return. So a view that emits
zero tile-overlap keys skips the clear and the census (`ReadGradientStatsAsync`, called
unconditionally per supervised view in `Studio.Training.cs`) reads whatever `_gradFixed` held
from the last view that *did* complete a step — not zero. This can only produce a false **live**
reading (masking a genuinely dead view), never a false **dead** one, so it cannot explain the
view-0 finding, but it means the "half the views are dead" census figures already in the commit
history could be undercounts. This is a straightforward reading of the control flow as written;
I did not attach a debugger or reproduce it at runtime.

**3. The training loss backpropagated is L1 only.** `TrainStepAsync` dispatches `_lossL1`, then
`_rasterBackward`, `_scatterGrad`, `_adamStep`, `_adamGeometry` — no `_ssimRowsPipe`/
`_ssimReducePipe` call anywhere in that function (confirmed by reading the full method and by
grep: those two pipelines are only ever dispatched from the evaluation path, for the reported
SSIM metric, not from training). The reference 3DGS trains on `0.8·L1 + 0.2·(1-D-SSIM)`; here
it's pure L1. This gap is already partially acknowledged in the **uncommitted** diff to
`SplatDensityControl.cs`: the densification gradient threshold was dropped from the reference's
`2e-4` to `2e-5`, with a comment attributing the change to "our L1-only loss produces weaker
position gradients than the reference's 0.8 L1 + 0.2 SSIM," citing a measured p90 NDC gradient
of `4.8e-5` on Truck. That measurement is theirs, not mine, but the code fact (no SSIM in the
backward path) is directly confirmed. This is a plausible contributor to weak geometry/position
gradients project-wide; I have **no measurement tying it specifically to the view-0 zero-gradient
finding**, and it should not be read as an explanation for that until tested.

---

## NOT VERIFIED — explicitly flagged as unresolved, not conclusions

I have **no runtime evidence** for why view 0 specifically produces zero gradient. Two
mechanisms are consistent with everything I read but neither is confirmed:

- View 0 emits zero tile-overlap keys (`LastKeyCount == 0`) — a camera-pose/depth-range issue
  specific to that view, not a shader bug. `TrainStepAsync` would short-circuit before loss,
  backward, or scatter ever run, so the census's "0.0% live" reading would be a correct symptom
  of a different cause than a broken backward pass.
- View 0 emits keys, but every overlapping splat's `alpha = opacity * weight` rounds under
  `MIN_ALPHA` (1/255) for that view's specific depth/scale combination, so the forward
  early-outs before contributing and the backward never walks anything.

The uncommitted diff already adds the exact instrumentation needed to distinguish these
(`keysPerView[it]` logged alongside the dead-view report, plus the `STAGE PROBE` that reports
`mean|rgb|`, `mean T`, `max|dL/dpix|`, `max|gradPerKey|` and buckets the failure into FORWARD /
LOSS / RASTER_BACKWARD / SCATTER). **That instrumentation has not been run yet with this build.**
Running it is the direct way to answer this, not further code reading.

TJ separately observed that on-screen renders during this session looked like noise, not a
scene. I did not launch the app myself this session (only `dotnet test` on the CPU test
project). That observation is first-hand evidence from TJ/Trip's session, not something I
produced or verified — noted here because it's consistent with (not proof of) the numeric
findings above: this looks like a real rendering/reconstruction failure, not a metrics-only
artifact.

---

## Summary for Trip

- CPU oracle: verified correct today by execution (39/39, finite-difference gated).
- WGSL kernels: traced by hand against that oracle, no discrepancy found, **but the automated
  GPU gate (`tools/_cdp_trainer_gate.js`) was not run this session** — run it before ruling out
  the shader path.
- Found and can hand you directly: a census-staleness bug (dead views can be masked as live) and
  a confirmed L1-only training loss (SSIM computed but never backpropagated) — both are facts
  from reading the code, not speculation, but neither is confirmed as *the* cause of the view-0
  zero.
- The view-0 zero-gradient cause itself is still open. Best next step is running the
  already-written stage probe on a real training session, not more static reading.
