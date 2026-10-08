using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Depth supervision gate (SplatTrainerGpu.Depth, the <c>//DEPTH:</c> builds of the raster / scatter / geometry passes).
/// <list type="number">
/// <item>The rendered inverse depth is a blend of the splats' 1/z: every pixel lies in [0, 1/z_near of the scene].</item>
/// <item>The gradient: against a constant target depth (1.2 x the scene's median), the analytic dL/d(position) along the
/// camera axis of the five splats it moves most must match central finite differences of the step's own loss. The colour
/// target is the scene's own render, so the colour terms sit at their minimum and cancel to first order; what is left is
/// the depth loss through every path (the blend weights via dL/d(alpha), the centre and footprint via the projection, and
/// 1/z itself via the new -g/z^2 term). Dropping any of them shows here.</item>
/// <item>It trains: with only the depth loss moving positions, the depth loss falls.</item>
/// </list>
/// Snapshot and restore around it: the zero-rate steps still rewrite scale and opacity from the trainer's own buffers.
/// </summary>
public partial class Studio
{
    async Task<bool> DepthSupervisionGateAsync(SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        var accel = _gpuService.WebGPUAccelerator;
        float[] snapshot = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
        const int MapW = 8, MapH = 6;
        using var map = accel.Allocate1D<float>(MapW * MapH);
        try
        {
            // The scene's depths along the camera axis (CPU: the gate's small scene, already on the host).
            var fwd = Vector3.Normalize(cam.Forward);
            var z = new List<float>();
            for (int i = 0; i < n; i++)
            {
                int o = i * SplatFormat.Floats;
                float d = Vector3.Dot(new Vector3(snapshot[o], snapshot[o + 1], snapshot[o + 2]) - cam.Position, fwd);
                if (d > 0.2f) z.Add(d);
            }
            if (z.Count < 10) { Console.WriteLine("[TrainerGate] FAIL depth: the gate scene has too few splats in front"); return false; }
            z.Sort();
            float zMin = z[0], zMed = z[z.Count / 2];

            map.CopyFromCPU(Enumerable.Repeat(1.2f * zMed, MapW * MapH).ToArray());
            trainer.SetDepthTarget(map, 0, MapW, MapH, 1f);
            trainer.DepthLossWeight = 1f;
            var frozen = new SplatTrainerGpu.GeometryStep(0f, 0f, 0f, 1e-9f, 1e9f);

            // Warm-up: the first zero-rate step rewrites scale / opacity from the trainer's buffers; the colour target is
            // the render after it.
            await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            var img = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar, readback: true);
            trainer.SetTarget(img);
            float baseLoss = await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            var inv = await trainer.ReadRenderedInverseDepthAsync();
            float invMax = inv.Length > 0 ? inv.Max() : 0f, invMin = inv.Length > 0 ? inv.Min() : 0f;
            float covered = inv.Count(v => v > 0f) / (float)Math.Max(1, inv.Length);
            Console.WriteLine($"[TrainerGate] depth forward: rendered inverse depth in [{invMin:G3}, {invMax:G3}], scene 1/z in " +
                $"[{1f / z[^1]:G3}, {1f / zMin:G3}], {covered:P0} of pixels covered; step loss {baseLoss:G4}");
            if (!(invMin >= 0f) || invMax > 1.0001f / zMin || covered < 0.05f)
            {
                Console.WriteLine("[TrainerGate] FAIL depth: the rendered inverse depth is not a blend of the splats' 1/z");
                return false;
            }

            var geom = await trainer.ReadGeometryGradientsAsync(n);
            var pick = Enumerable.Range(0, n)
                .Select(i => (i, g: Vector3.Dot(new Vector3(geom[i * 10], geom[i * 10 + 1], geom[i * 10 + 2]), fwd)))
                .OrderByDescending(t => MathF.Abs(t.g)).Take(5).ToList();
            var analytic = new float[pick.Count];
            var fd = new float[pick.Count];
            float[] work = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
            for (int k = 0; k < pick.Count; k++)
            {
                int i = pick[k].i, o = i * SplatFormat.Floats;
                analytic[k] = pick[k].g;
                var p0 = new Vector3(work[o], work[o + 1], work[o + 2]);
                float eps = 0.002f * MathF.Max(Vector3.Dot(p0 - cam.Position, fwd), 0.2f);
                async Task<float> LossAt(Vector3 p)
                {
                    splatBuf.View.SubView(o, 3).CopyFromCPU(new[] { p.X, p.Y, p.Z });
                    return await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
                }
                float lp = await LossAt(p0 + fwd * eps), lm = await LossAt(p0 - fwd * eps);
                splatBuf.View.SubView(o, 3).CopyFromCPU(new[] { p0.X, p0.Y, p0.Z });
                fd[k] = (lp - lm) / (2f * eps);
            }
            double ab = 0, aa = 0, bb = 0;
            int close = 0;
            for (int k = 0; k < pick.Count; k++)
            {
                ab += analytic[k] * fd[k]; aa += analytic[k] * analytic[k]; bb += fd[k] * fd[k];
                if (MathF.Abs(analytic[k] - fd[k]) <= 0.25f * MathF.Max(MathF.Abs(analytic[k]), MathF.Abs(fd[k]))) close++;
            }
            float cos = aa > 0 && bb > 0 ? (float)(ab / Math.Sqrt(aa * bb)) : 0f;
            Console.WriteLine($"[TrainerGate] depth gradient along the view axis, analytic vs finite difference: " +
                string.Join("  ", pick.Select((t, k) => $"#{t.i} {analytic[k]:G3}/{fd[k]:G3}")) + $"; cos {cos:F3}, {close}/{pick.Count} within 25%");
            if (cos < 0.995f || close < pick.Count - 1)
            {
                Console.WriteLine("[TrainerGate] FAIL depth: the analytic depth gradient does not match the loss");
                return false;
            }

            // It trains: positions only (colour and opacity frozen), the depth error on the pixels covered before and
            // after must fall (the uncovered ones are a floor positions cannot move; the colour loss pulls the other way).
            var move = new SplatTrainerGpu.GeometryStep(1e-3f * zMed, 0f, 0f, 1e-9f, 1e9f);
            for (int s = 0; s < 150; s++)
                await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, move, readLoss: s == 149);
            var invAfter = await trainer.ReadRenderedInverseDepthAsync();
            float want = 1f / (1.2f * zMed);
            double eBefore = 0, eAfter = 0; int both = 0;
            for (int i = 0; i < Math.Min(inv.Length, invAfter.Length); i++)
                if (inv[i] > 0.5f * want && invAfter[i] > 0.5f * want) { eBefore += Math.Abs(inv[i] - want); eAfter += Math.Abs(invAfter[i] - want); both++; }
            eBefore /= Math.Max(1, both); eAfter /= Math.Max(1, both);
            Console.WriteLine($"[TrainerGate] depth training, positions only, 150 steps: mean |inverse depth error| on {both} covered pixels " +
                $"{eBefore:G4} -> {eAfter:G4}");
            if (both < 100 || !(eAfter < 0.7 * eBefore))
            {
                Console.WriteLine("[TrainerGate] FAIL depth: the depth loss did not fall");
                return false;
            }
            Console.WriteLine("[TrainerGate] depth PASS");
            return true;
        }
        finally
        {
            trainer.DepthLossWeight = 0f;
            trainer.SetDepthTarget(null);
            splatBuf.View.SubView(0, (long)n * SplatFormat.Floats).CopyFromCPU(snapshot);
            await accel.SynchronizeAsync();
            trainer.ResetPeakKeyDemand();
        }
    }
}
