using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Gate for the WGSL tile rasteriser: does it reproduce the CPU model?
///
/// The comparison is deliberately NOT against the display renderer. That would conflate two
/// differences at once - a different algorithm (hardware blending vs compute compositing) and
/// a possible shader bug. Instead this runs the GPU kernels and
/// <see cref="SplatTileRasterizer"/> over byte-identical splat data and compares pixels. The
/// CPU model is already verified against the finite-difference-checked
/// <see cref="SplatRasterizer"/>, so any disagreement here is the shader.
///
/// A small synthetic scene is used on purpose: few enough splats to read back and compare
/// exactly, but with overlap, anisotropy and rotation so the covariance path is exercised.
///
/// Entry: <c>/studio?autotest=trainer-gate</c>
/// Logs:  <c>[TrainerGate] PASS|FAIL ...</c>
/// </summary>
public partial class Studio
{
    const int GateWidth = 128;
    const int GateHeight = 96;
    const int GateSplats = 240;

    async Task RunTrainerGateAsync()
    {
        Console.WriteLine("[TrainerGate] starting");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();
            var accel = _gpuService.WebGPUAccelerator;

            // ── A synthetic scene with known parameters ──
            var rng = new Random(20260920);
            int n = GateSplats;
            var packed = new float[n * SplatFormat.Floats];
            var centre = new Vector3(0f, 0f, 0f);
            for (int i = 0; i < n; i++)
            {
                int o = i * SplatFormat.Floats;
                // A loose cloud in front of the camera.
                packed[o + 0] = (float)(rng.NextDouble() - 0.5) * 0.6f;
                packed[o + 1] = (float)(rng.NextDouble() - 0.5) * 0.4f;
                packed[o + 2] = (float)(rng.NextDouble() - 0.5) * 0.6f;
                packed[o + 3] = 0.15f + (float)rng.NextDouble() * 0.8f;
                packed[o + 4] = 0.15f + (float)rng.NextDouble() * 0.8f;
                packed[o + 5] = 0.15f + (float)rng.NextDouble() * 0.8f;
                // Anisotropic, so the covariance path matters rather than cancelling out.
                packed[o + 6] = 0.010f + (float)rng.NextDouble() * 0.020f;
                packed[o + 7] = 0.010f + (float)rng.NextDouble() * 0.020f;
                packed[o + 8] = 0.004f + (float)rng.NextDouble() * 0.008f;
                packed[o + 9] = 0.25f + (float)rng.NextDouble() * 0.6f;
                // A real rotation, normalised.
                var q = Quaternion.Normalize(new Quaternion(
                    (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f,
                    (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f));
                packed[o + 10] = q.X; packed[o + 11] = q.Y; packed[o + 12] = q.Z; packed[o + 13] = q.W;
            }

            using var splatBuf = accel.Allocate1D<float>(packed.Length);
            splatBuf.CopyFromCPU(packed);
            await accel.SynchronizeAsync();

            var cam = new CameraParams
            {
                Width = GateWidth,
                Height = GateHeight,
                FocalX = 140f,
                FocalY = 140f,
                CenterX = GateWidth / 2f,
                CenterY = GateHeight / 2f,
                Near = 0.01f,
                Far = 100f,
                Position = new Vector3(0.35f, 0.25f, 1.1f),
            };
            cam.Forward = Vector3.Normalize(centre - cam.Position);
            cam.Up = Vector3.UnitY;

            // Depth range for the sort key. Conservative bounds around the cloud.
            float dist = Vector3.Distance(cam.Position, centre);
            float depthNear = MathF.Max(dist - 0.8f, 0.01f);
            float depthFar = dist + 0.8f;

            // ── GPU ──
            using var trainer = new SplatTrainerGpu(_gpuService);
            trainer.Initialize();
            trainer.Resize(GateWidth, GateHeight, n);
            var gpuColour = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar);

            if (trainer.LastOverflowed)
            {
                Console.WriteLine("[TrainerGate] FAIL: key buffer overflowed");
                return;
            }
            Console.WriteLine($"[TrainerGate] gpu keys={trainer.LastKeyCount:N0}");

            // ── CPU model over the same data ──
            var cpuSplats = ProjectForCpu(packed, n, cam);
            var bin = SplatTileRasterizer.Bin(cpuSplats, GateWidth, GateHeight);
            var (cpuColour, _, _) = SplatTileRasterizer.Forward(cpuSplats, bin);
            Console.WriteLine($"[TrainerGate] cpu keys={bin.Keys.Length:N0}");

            // ── Compare ──
            double sumAbs = 0;
            float maxAbs = 0;
            int lit = 0;
            for (int i = 0; i < gpuColour.Length; i++)
            {
                float d = MathF.Abs(gpuColour[i] - cpuColour[i]);
                sumAbs += d;
                if (d > maxAbs) maxAbs = d;
                if (cpuColour[i] > 0.01f) lit++;
            }
            float meanAbs = (float)(sumAbs / gpuColour.Length);

            float gpuMean = gpuColour.Average();
            float cpuMean = cpuColour.Average();
            Console.WriteLine(
                $"[TrainerGate] lit={lit} gpuMean={gpuMean:F5} cpuMean={cpuMean:F5} " +
                $"meanAbs={meanAbs:F6} maxAbs={maxAbs:F6}");

            // A blank GPU image would trivially "agree" with a blank CPU image, so require the
            // comparison to have actually had content before believing a small difference.
            bool hasContent = lit > gpuColour.Length / 50 && cpuMean > 0.01f;
            bool close = meanAbs < 2e-3f && maxAbs < 5e-2f;

            if (!hasContent)
                Console.WriteLine("[TrainerGate] FAIL: nothing rendered - comparison is vacuous");
            else if (!close)
                Console.WriteLine("[TrainerGate] FAIL: GPU and CPU disagree");
            else
                Console.WriteLine("[TrainerGate] forward PASS");

            if (!hasContent || !close) return;

            if (!await SsimGateAsync(trainer, accel, gpuColour)) return;

            // ── Training: fit colour+opacity back to a target rendered from KNOWN parameters ──
            // The target is this same scene; the splats are then perturbed. So the optimiser
            // has a reachable answer and we can check it walks toward it, rather than only
            // that some number went down - which a loop that merely dims everything also does.
            trainer.SetTarget(gpuColour);

            var perturbed = (float[])packed.Clone();
            for (int i = 0; i < n; i++)
            {
                int o = i * SplatFormat.Floats;
                perturbed[o + 3] = 0.5f; perturbed[o + 4] = 0.5f; perturbed[o + 5] = 0.5f;
                perturbed[o + 9] = 0.5f;
            }
            using var trainBuf = accel.Allocate1D<float>(perturbed.Length);
            trainBuf.CopyFromCPU(perturbed);
            await accel.SynchronizeAsync();

            trainer.InitOptimizerState(trainBuf, n);

            float first = 0f, last = 0f;
            const int iterations = 300;
            for (int it = 0; it < iterations; it++)
            {
                float loss = await trainer.TrainStepAsync(trainBuf, n, cam, depthNear, depthFar);
                if (it == 0) first = loss;
                last = loss;
                if (it % 50 == 0) Console.WriteLine($"[TrainerGate] iter {it,4} loss {loss:F6}");
            }

            // Did the parameters move toward the truth, or just the loss downward?
            float[] fitted = await trainBuf.CopyToHostAsync<float>(0, perturbed.Length);
            float errBefore = 0f, errAfter = 0f;
            for (int i = 0; i < n; i++)
            {
                int o = i * SplatFormat.Floats;
                for (int c = 0; c < 3; c++)
                {
                    errBefore += MathF.Abs(0.5f - packed[o + 3 + c]);
                    errAfter += MathF.Abs(fitted[o + 3 + c] - packed[o + 3 + c]);
                }
            }
            errBefore /= n * 3; errAfter /= n * 3;

            Console.WriteLine($"[TrainerGate] loss {first:F6} -> {last:F6}");
            Console.WriteLine($"[TrainerGate] mean |colour - truth| {errBefore:F4} -> {errAfter:F4}");

            bool lossFell = last < first * 0.5f;
            bool recovered = errAfter < errBefore * 0.6f;

            if (!lossFell) { Console.WriteLine("[TrainerGate] FAIL: loss did not fall"); return; }
            if (!recovered) { Console.WriteLine("[TrainerGate] FAIL: colours did not move toward truth"); return; }
            Console.WriteLine("[TrainerGate] colour/opacity PASS");

            // -- Gradients: do the shaders compute what the verified CPU oracles compute? --
            if (!await GradientGateAsync(trainer, splatBuf, packed, n, cam, depthNear, depthFar)) return;

            Console.WriteLine("[TrainerGate] PASS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TrainerGate] FAIL: {ex}");
        }
    }

    /// <summary>
    /// Compare the GPU gradients against the CPU oracles, on the same data, for one step.
    ///
    /// Two separate comparisons, because two separate things can be wrong:
    ///   the 2D gradients  -> SplatTileRasterizer.Backward, itself checked against the
    ///                        finite-difference-verified SplatRasterizer.Backward
    ///   the geometry chain -> SplatGeometryGradients.Backward, finite-difference verified
    ///
    /// Comparing only the second would pass with a broken rasteriser feeding it, and comparing
    /// only the first would pass with the whole covariance chain transposed.
    /// </summary>
    async Task<bool> GradientGateAsync(
        SplatTrainerGpu trainer,
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf,
        float[] packed, int n, CameraParams cam, float depthNear, float depthFar)
    {
        // A fresh render against a target the splats do NOT already match, so the gradients
        // are large enough to compare. Fitting to itself would compare two piles of noise.
        var rng = new Random(31337);
        var target = new float[GateWidth * GateHeight * 3];
        for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
        trainer.SetTarget(target);

        // Read back the splats as the optimiser last left them - the gradients must be
        // evaluated at the state the GPU actually rendered.
        float[] state = await splatBuf.CopyToHostAsync<float>(0, packed.Length);

        var geoStep = new SplatTrainerGpu.GeometryStep(
            PositionLr: 0f, LogScaleLr: 0f, RotationLr: 0f, MinScale: 1e-7f, MaxScale: 1f);
        await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar,
            colourLr: 0f, opacityLr: 0f, geometry: geoStep);

        float[] gpu2d = await trainer.ReadGradientsAsync(n);
        float[] gpuGeo = await trainer.ReadGeometryGradientsAsync(n);

        // -- CPU, from the same state --
        var cpuSplats = ProjectForCpu(state, n, cam);
        var bin = SplatTileRasterizer.Bin(cpuSplats, GateWidth, GateHeight);
        var (colour, finalT, endIdx) = SplatTileRasterizer.Forward(cpuSplats, bin);
        var dPix = SplatRasterizer.L1Gradient(colour, target);
        var cpu2d = SplatTileRasterizer.Backward(cpuSplats, bin, finalT, endIdx, dPix);

        // ProjectForCpu drops splats behind the camera, so CPU index != splat index. Rebuild
        // the mapping rather than assuming they line up - they did not, and a silent
        // misalignment would compare every splat against its neighbour.
        var cpuIndexOf = MapProjectedIndices(state, n, cam);

        // The gradients cross a 2^20 fixed-point atomic, once per KEY, so the error floor is
        // quantisation and not algebra. Measuring this in QUANTA rather than as a ratio is what
        // separates "the shader is wrong" from "the number is small": at this gate's 128x96,
        // dL/d(pixel) is 1/(3*128*96) and a per-splat gradient can be only a few hundred quanta
        // to begin with, so a 3% relative error on one of those is a rounding artefact. A
        // relative bound is still applied, but only to the values large enough for it to mean
        // something.
        const double RelevantMagnitude = 1e-4;

        double sumAbs = 0, maxAbs = 0;
        double sumRelBig = 0, maxRelBig = 0;
        int compared = 0, comparedBig = 0;
        for (int i = 0; i < n; i++)
        {
            if (!cpuIndexOf.TryGetValue(i, out int ci)) continue;
            var c = cpu2d[ci];
            float[] want = { c.R, c.G, c.B, c.Opacity, c.Px, c.Py, c.ConicA, c.ConicB, c.ConicC };
            for (int k = 0; k < want.Length; k++)
            {
                float got = gpu2d[i * SplatTrainerGpu.GradsPerSplat + k];
                double err = Math.Abs(got - want[k]);

                // In QUANTA of this slot's own scale - they differ by 64x between the conic
                // and everything else, so one absolute bound would mean two different things.
                // The RELATIVE check must use the unscaled error; multiplying the quantised
                // one by 2^20 turned a 0.17% disagreement into a reported 1800x.
                double quanta = err * trainer.FixedScaleFor(k);
                sumAbs += quanta;
                if (quanta > maxAbs) maxAbs = quanta;
                compared++;

                if (Math.Abs(want[k]) > RelevantMagnitude)
                {
                    double rel = err / Math.Abs(want[k]);
                    sumRelBig += rel;
                    if (rel > maxRelBig) maxRelBig = rel;
                    comparedBig++;
                }
            }
        }
        double meanAbsQ = compared > 0 ? sumAbs / compared : 1e9;
        double maxAbsQ = maxAbs;
        double meanRelBig = comparedBig > 0 ? sumRelBig / comparedBig : 1.0;
        Console.WriteLine(
            $"[TrainerGate] 2D gradients: {compared} values, mean err {meanAbsQ:F2} quanta, " +
            $"max {maxAbsQ:F1} quanta; of the {comparedBig} above {RelevantMagnitude:G2}, " +
            $"mean rel {meanRelBig:F5}, max rel {maxRelBig:F5}");

        if (compared < n * 5)
        {
            Console.WriteLine($"[TrainerGate] FAIL: only {compared} gradient values compared");
            return false;
        }
        if (comparedBig < n)
        {
            Console.WriteLine(
                $"[TrainerGate] FAIL: only {comparedBig} gradients are large enough to compare " +
                "relatively - the fixture is not exercising the backward hard enough");
            return false;
        }
        // A few quanta of disagreement is the rounding; a wrong shader is off by orders.
        if (!(meanAbsQ < 4.0) || !(maxAbsQ < 400.0))
        {
            Console.WriteLine("[TrainerGate] FAIL: GPU and CPU 2D gradients disagree beyond quantisation");
            return false;
        }
        if (!(meanRelBig < 0.02) || !(maxRelBig < 0.2))
        {
            Console.WriteLine("[TrainerGate] FAIL: GPU and CPU 2D gradients disagree on the large values");
            return false;
        }

        // -- Geometry chain, fed with the GPU's OWN 2D gradients --
        // Feeding the CPU chain the CPU 2D gradients would let a small disagreement above
        // reappear here magnified, and report a chain bug that is not there.
        var view = ViewFor(cam);
        double sumGeo = 0, maxGeo = 0;
        int comparedGeo = 0;
        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            int gb = i * SplatTrainerGpu.GradsPerSplat;

            var up = new SplatGeometryGradients.UpstreamGrad
            {
                ScreenX = gpu2d[gb + 4], ScreenY = gpu2d[gb + 5],
                ConicA = gpu2d[gb + 6], ConicB = gpu2d[gb + 7], ConicC = gpu2d[gb + 8],
            };
            // The shader skips a splat with no gradient at all; so must this.
            if (up.ScreenX == 0f && up.ScreenY == 0f &&
                up.ConicA == 0f && up.ConicB == 0f && up.ConicC == 0f) continue;

            var geom = new SplatGeometryGradients.Geometry
            {
                PosX = state[o + 0], PosY = state[o + 1], PosZ = state[o + 2],
                ScaleX = state[o + 6], ScaleY = state[o + 7], ScaleZ = state[o + 8],
                QuatX = state[o + 10], QuatY = state[o + 11],
                QuatZ = state[o + 12], QuatW = state[o + 13],
            };
            var want = SplatGeometryGradients.Backward(geom, view, up);
            float[] w = { want.PosX, want.PosY, want.PosZ,
                          want.ScaleX, want.ScaleY, want.ScaleZ,
                          want.QuatX, want.QuatY, want.QuatZ, want.QuatW };

            for (int k = 0; k < w.Length; k++)
            {
                float got = gpuGeo[i * SplatTrainerGpu.GeomGradsPerSplat + k];
                float scale = MathF.Max(MathF.Abs(w[k]), 1e-6f);
                float rel = MathF.Abs(got - w[k]) / scale;
                sumGeo += rel;
                if (rel > maxGeo) maxGeo = rel;
                comparedGeo++;
            }
        }
        double meanGeo = comparedGeo > 0 ? sumGeo / comparedGeo : 1.0;
        Console.WriteLine(
            $"[TrainerGate] geometry chain: {comparedGeo} values, mean rel {meanGeo:F6}, max rel {maxGeo:F6}");

        if (comparedGeo < n * 5)
        {
            Console.WriteLine($"[TrainerGate] FAIL: only {comparedGeo} geometry values compared");
            return false;
        }
        // Same arithmetic on both sides here, in the same precision, so this one should be
        // tight. A loose bound would accept a transposed matrix.
        if (!(meanGeo < 1e-3) || !(maxGeo < 0.05))
        {
            Console.WriteLine("[TrainerGate] FAIL: WGSL geometry chain disagrees with the CPU oracle");
            return false;
        }

        // The gradient-health reduction, against a CPU pass over the SAME buffer. This is the
        // strongest form available: gpu2d is already the full accumulator, so the two are not
        // merely similar computations on similar data, they are the same numbers.
        //
        // It is here because the probe this replaces went blind for a whole session - it read a
        // prefix of a view-major buffer, which is not a sample - and nothing caught it.
        var stats = await trainer.ReadGradientStatsAsync(n);
        long cColour = 0, cCentre = 0, cConic = 0;
        double cSumCentre = 0, cMaxCentre = 0, cMaxConic = 0;
        for (int i = 0; i < n; i++)
        {
            int b = i * SplatTrainerGpu.GradsPerSplat;
            if (gpu2d[b] != 0f || gpu2d[b + 1] != 0f || gpu2d[b + 2] != 0f) cColour++;

            // gpu2d is already divided by the fixed-point scales; the reduction reports QUANTA,
            // so scale back to compare like with like.
            double cen = Math.Max(Math.Abs(gpu2d[b + 4]), Math.Abs(gpu2d[b + 5]))
                         * trainer.FixedScaleFor(4);
            if (cen > 0) { cCentre++; cSumCentre += cen; }
            cMaxCentre = Math.Max(cMaxCentre, cen);

            double con = Math.Max(Math.Abs(gpu2d[b + 6]),
                         Math.Max(Math.Abs(gpu2d[b + 7]), Math.Abs(gpu2d[b + 8])))
                         * trainer.FixedScaleFor(6);
            if (con > 0) cConic++;
            cMaxConic = Math.Max(cMaxConic, con);
        }
        double cMeanCentre = cCentre > 0 ? cSumCentre / cCentre : 0;

        Console.WriteLine(
            $"[TrainerGate] grad stats: colour {stats.ColourLive}/{cColour}, " +
            $"centre {stats.CentreLive}/{cCentre}, conic {stats.ConicLive}/{cConic}, " +
            $"meanCentre {stats.MeanCentreQuanta:F1}/{cMeanCentre:F1} quanta, " +
            $"stale {stats.StaleColourFraction:P1}");

        if (cColour == 0 && cCentre == 0)
        {
            Console.WriteLine("[TrainerGate] FAIL: no gradients at all - the stats check is vacuous");
            return false;
        }
        if (stats.ColourLive != cColour || stats.CentreLive != cCentre || stats.ConicLive != cConic)
        {
            Console.WriteLine("[TrainerGate] FAIL: grad_stats counts disagree with the CPU pass");
            return false;
        }
        if (cMeanCentre > 0 &&
            Math.Abs(stats.MeanCentreQuanta - cMeanCentre) / cMeanCentre > 1e-3)
        {
            Console.WriteLine("[TrainerGate] FAIL: grad_stats mean magnitude disagrees");
            return false;
        }
        Console.WriteLine("[TrainerGate] grad stats PASS");

        // The view-support counter, against the same CPU pass. Its report is load-bearing - it
        // is the line that says whether optimiser experiments are worth running at all - so it
        // gets a CPU oracle like everything else here rather than being believed.
        //
        // Accumulated TWICE over the same gradients: once only proves the threshold, and would
        // pass equally if the shader assigned 1 instead of incrementing. Twice proves the
        // counting, which is the part the report depends on.
        long cAny = 0;
        for (int i = 0; i < n; i++)
        {
            int b = i * SplatTrainerGpu.GradsPerSplat;
            bool any = false;
            for (int c = 0; c < SplatTrainerGpu.GradsPerSplat; c++)
                if (gpu2d[b + c] != 0f) { any = true; break; }
            if (any) cAny++;
        }

        trainer.ResetViewSupport(n);
        trainer.AccumulateViewSupport(n);
        trainer.AccumulateViewSupport(n);
        var support = await trainer.ReadViewSupportAsync(n);

        Console.WriteLine(
            $"[TrainerGate] view support: 2-view {support.TwoViews}/{cAny}, " +
            $"never {support.Unconstrained}/{n - cAny}, mean {support.MeanViews:F3}");

        if (cAny == 0)
        {
            Console.WriteLine("[TrainerGate] FAIL: no splat has any gradient - support check is vacuous");
            return false;
        }
        if (support.TwoViews != cAny || support.Unconstrained != n - cAny ||
            support.OneView != 0 || support.ThreeViews != 0 || support.FourOrMore != 0)
        {
            Console.WriteLine(
                "[TrainerGate] FAIL: view support buckets disagree with the CPU pass - " +
                $"expected {cAny} at exactly 2 and {n - cAny} at 0, got " +
                $"0:{support.Unconstrained} 1:{support.OneView} 2:{support.TwoViews} " +
                $"3:{support.ThreeViews} 4+:{support.FourOrMore}");
            return false;
        }
        if (Math.Abs(support.MeanViews - 2.0 * cAny / n) > 1e-4)
        {
            Console.WriteLine("[TrainerGate] FAIL: view support mean disagrees with its own buckets");
            return false;
        }
        Console.WriteLine("[TrainerGate] view support PASS");

        Console.WriteLine("[TrainerGate] gradients PASS");
        return true;
    }

    /// <summary>Which entry of <see cref="ProjectForCpu"/> each splat index became.</summary>
    static Dictionary<int, int> MapProjectedIndices(float[] packed, int n, CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);
        var map = new Dictionary<int, int>();
        int next = 0;
        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var rel = new Vector3(packed[o + 0], packed[o + 1], packed[o + 2]) - pos;
            if (Vector3.Dot(fwd, rel) <= 1e-6f) continue;

            var q = new SplatCovariance.Quat
            {
                X = packed[o + 10], Y = packed[o + 11], Z = packed[o + 12], W = packed[o + 13],
            };
            float len = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
            q = new SplatCovariance.Quat { X = q.X / len, Y = q.Y / len, Z = q.Z / len, W = q.W / len };
            var cov3 = SplatCovariance.Cov3DFromScaleQuat(
                MathF.Max(packed[o + 6], 1e-9f), MathF.Max(packed[o + 7], 1e-9f),
                MathF.Max(packed[o + 8], 1e-9f), q);
            var camCov = SplatCovariance.RotateToCamera(cov3,
                right.X, right.Y, right.Z, up.X, up.Y, up.Z, fwd.X, fwd.Y, fwd.Z);
            var cov2 = SplatCovariance.ProjectCov2D(camCov,
                Vector3.Dot(right, rel), Vector3.Dot(up, rel), Vector3.Dot(fwd, rel),
                cam.FocalX, cam.FocalY);
            if (!(cov2.A * cov2.C - cov2.B * cov2.B > 1e-20f)) continue;

            map[i] = next++;
        }
        return map;
    }

    static SplatGeometryGradients.View ViewFor(CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var r, out var u, out var f, out var p);
        return new SplatGeometryGradients.View
        {
            EyeX = p.X, EyeY = p.Y, EyeZ = p.Z,
            Rx = r.X, Ry = r.Y, Rz = r.Z,
            Ux = u.X, Uy = u.Y, Uz = u.Z,
            Fx3 = f.X, Fy3 = f.Y, Fz3 = f.Z,
            FocalX = cam.FocalX, FocalY = cam.FocalY,
            CenterX = cam.CenterX, CenterY = cam.CenterY,
        };
    }

    /// <summary>
    /// Project packed splats to screen space the way the shader does, so the CPU model sees
    /// the same inputs. Mirrors <c>project()</c> in SplatTrainerShaders.Common.
    /// </summary>
    static List<SplatRasterizer.Splat2D> ProjectForCpu(float[] packed, int n, CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);
        var list = new List<SplatRasterizer.Splat2D>(n);

        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var p = new Vector3(packed[o + 0], packed[o + 1], packed[o + 2]);
            var rel = p - pos;
            float cx = Vector3.Dot(right, rel);
            float cy = Vector3.Dot(up, rel);
            float cz = Vector3.Dot(fwd, rel);
            if (cz <= 1e-6f) continue;

            var q = new SplatCovariance.Quat
            {
                X = packed[o + 10], Y = packed[o + 11], Z = packed[o + 12], W = packed[o + 13],
            };
            float len = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
            q = new SplatCovariance.Quat { X = q.X / len, Y = q.Y / len, Z = q.Z / len, W = q.W / len };

            var cov3 = SplatCovariance.Cov3DFromScaleQuat(
                MathF.Max(packed[o + 6], 1e-9f), MathF.Max(packed[o + 7], 1e-9f), MathF.Max(packed[o + 8], 1e-9f), q);
            var camCov = SplatCovariance.RotateToCamera(cov3,
                right.X, right.Y, right.Z, up.X, up.Y, up.Z, fwd.X, fwd.Y, fwd.Z);
            var cov2 = SplatCovariance.ProjectCov2D(camCov, cx, cy, cz, cam.FocalX, cam.FocalY);

            float det = cov2.A * cov2.C - cov2.B * cov2.B;
            if (!(det > 1e-20f)) continue;
            float invDet = 1f / det;

            list.Add(new SplatRasterizer.Splat2D
            {
                // Screen y grows DOWN, camera y grows UP.
                Px = cam.FocalX * cx / cz + cam.CenterX,
                Py = cam.CenterY - cam.FocalY * cy / cz,
                ConicA = cov2.C * invDet,
                ConicB = -cov2.B * invDet,
                ConicC = cov2.A * invDet,
                R = packed[o + 3], G = packed[o + 4], B = packed[o + 5],
                Opacity = packed[o + 9],
                Depth = cz,
            });
        }
        return list;
    }

    /// <summary>
    /// Does the GPU SSIM compute the same thing as <see cref="ImageQuality"/>?
    ///
    /// That matters because SSIM is about to become the number the optimiser work is judged by:
    /// PSNR reported a Bathroom run as flat (12.50 -> 12.21 dB) while the render melted from a
    /// recognisable room into fog.
    ///
    /// The chain this closes is three-deep. ImageQualityTests pins the C# oracle against
    /// tools/score_novel_view.py, the scorer that produces the TempleRing numbers; this gate
    /// pins the WGSL against that same C#. So a disagreement anywhere shows up, without the
    /// shader needing to score the Python's own fixture.
    ///
    /// Scored on a REAL render rather than an analytic fixture on purpose. A smooth symmetric
    /// fixture survives a transposed row stride, a wrong target offset and a flipped axis; a
    /// real render against a degraded copy of itself, placed at a NON-ZERO index in the target
    /// stack, does not.
    /// </summary>
    async Task<bool> SsimGateAsync(
        SplatTrainerGpu trainer, SpawnDev.ILGPU.WebGPU.WebGPUAccelerator accel, float[] rendered)
    {
        int frameFloats = GateWidth * GateHeight * 3;

        // Index 0 is a degraded copy, index 1 is the render itself. Using index 1 for the
        // identical case is what proves the target offset is applied: a shader that ignored it
        // would score both entries against slot 0 and the identical case would fail.
        var degraded = Degrade(rendered, GateWidth, GateHeight);
        var stack = new float[frameFloats * 2];
        System.Array.Copy(degraded, 0, stack, 0, frameFloats);
        System.Array.Copy(rendered, 0, stack, frameFloats, frameFloats);

        using var stackBuf = accel.Allocate1D<float>(stack.Length);
        stackBuf.CopyFromCPU(stack);
        await accel.SynchronizeAsync();

        var (_, gpuDegraded) = await trainer.ScoreAgainstAsync(stackBuf, 0);
        var (_, gpuSelf) = await trainer.ScoreAgainstAsync(stackBuf, 1);
        double cpuDegraded = ImageQuality.MeanSsim(rendered, degraded, GateWidth, GateHeight);

        Console.WriteLine(
            $"[TrainerGate] SSIM: degraded gpu {gpuDegraded:F6} cpu {cpuDegraded:F6} " +
            $"(d {Math.Abs(gpuDegraded - cpuDegraded):E2}); self {gpuSelf:F6}");

        // Two near-identical images agree trivially, so a match only means something when the
        // score sits away from both ends.
        if (!(cpuDegraded > 0.05 && cpuDegraded < 0.98))
        {
            Console.WriteLine(
                $"[TrainerGate] FAIL: SSIM fixture is vacuous at {cpuDegraded:F4} - the degraded " +
                "image is too close to the render, or too far, for agreement to prove anything");
            return false;
        }
        if (Math.Abs(gpuDegraded - cpuDegraded) > 1e-4)
        {
            Console.WriteLine("[TrainerGate] FAIL: GPU and CPU SSIM disagree");
            return false;
        }
        if (Math.Abs(gpuSelf - 1.0) > 1e-5)
        {
            Console.WriteLine(
                $"[TrainerGate] FAIL: SSIM of an image against itself is {gpuSelf:F6}, not 1.0 " +
                "- the target offset is likely being ignored");
            return false;
        }

        Console.WriteLine("[TrainerGate] SSIM PASS");
        return true;
    }

    /// <summary>
    /// A blurred, dimmed copy: structurally similar but not identical, which is what puts SSIM
    /// in the middle of its range where a comparison is informative.
    /// </summary>
    static float[] Degrade(float[] rgb, int width, int height)
    {
        var outp = new float[rgb.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    float sum = 0;
                    int taps = 0;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int sy = y + dy;
                        if (sy < 0 || sy >= height) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int sx = x + dx;
                            if (sx < 0 || sx >= width) continue;
                            sum += rgb[(sy * width + sx) * 3 + c];
                            taps++;
                        }
                    }
                    outp[(y * width + x) * 3 + c] =
                        Math.Clamp(0.9f * (sum / taps) + 0.03f, 0f, 1f);
                }
            }
        }
        return outp;
    }
}
