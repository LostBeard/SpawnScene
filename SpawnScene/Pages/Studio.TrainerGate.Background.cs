using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Background gate (SplatTrainerGpu.RandomBackground, the reference's --random_background): each training step's
/// render is sum(T a c) + T_final * background, and the backward carries the background's share into dL/d(alpha)
/// (-T_final / (1 - a) * background . dL/dpixel, the reference backward.cu's bg_dot_dpixel term).
/// <list type="number">
/// <item>The target is the scene's own render over BLACK and the step composites over a fixed colour: the only error
/// left is the background showing through, so the loss and its gradient run through the new term. Pure L1, after a
/// control through the colour path alone: the loss a step RETURNS is the L1 part only (LossReduce), while the gradient
/// also carries the D-SSIM share - so with D-SSIM on, finite differences of the returned loss cannot match (they did
/// not, by 2.5-6x, 2026-10-08; the D-SSIM gradient itself is FD-gated via ImageQuality). The analytic
/// dL/d(position) along the camera axis of the five splats it moves most must match central finite differences of the
/// step's own loss. Without the term the analytic gradient misses the background entirely.</item>
/// <item>It descends: opacity only, over the same background the loss falls (a little: uncovered pixels are a floor).</item>
/// </list>
/// Snapshot and restore around it, as the depth gate.
/// </summary>
public partial class Studio
{
    async Task<bool> BackgroundGateAsync(SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        var accel = _gpuService.WebGPUAccelerator;
        float[] snapshot = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
        try
        {
            var fwd = Vector3.Normalize(cam.Forward);
            var frozen = new SplatTrainerGpu.GeometryStep(0f, 0f, 0f, 1e-9f, 1e9f);
            // The analytic dL/d(position) along the view axis of the five splats it moves most vs central finite
            // differences of the step's own loss, at the trainer's current target and background.
            async Task<(float cos, int close, int count)> CompareAsync(string label)
            {
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
                Console.WriteLine($"[TrainerGate] {label} gradient along the view axis, analytic vs finite difference: " +
                    string.Join("  ", pick.Select((t, k) => $"#{t.i} {analytic[k]:G3}/{fd[k]:G3}")) + $"; cos {cos:F3}, {close}/{pick.Count} within 25%");
                return (cos, close, pick.Count);
            }
            // Warm-up over black (the first zero-rate step rewrites scale / opacity from the trainer's buffers); the
            // target is the render over black after it.
            await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            var img = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar, readback: true);
            trainer.SetTarget(img);
            // Pure L1: the returned step loss is the L1 part only, so finite differences of it match the gradient only
            // when the D-SSIM share is 0 (with it on they were 2.5-6x apart - the loss readout, not the gradient).
            trainer.DssimWeight = 0f;
            // Control: no background, a target 0.7x the render - a colour error of similar size, through the colour path
            // alone. Must match, or the setup cannot judge the background term.
            trainer.SetTarget(img.Select(v => 0.7f * v).ToArray());
            await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            var ctl = await CompareAsync("control (no background, target 0.7x, L1)");
            if (ctl.cos < 0.995f || ctl.close < ctl.count - 1)
            {
                Console.WriteLine("[TrainerGate] FAIL background: the control (colour path, L1) does not match finite differences");
                return false;
            }
            trainer.SetTarget(img);
            float blackLoss = await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            trainer.FixedBackground = new Vector3(0.9f, 0.2f, 0.6f);
            float bgLoss = await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f, 0f, frozen, readLoss: true);
            Console.WriteLine($"[TrainerGate] background: step loss over black {blackLoss:G4}, over (0.9,0.2,0.6) {bgLoss:G4}");
            if (!(bgLoss > blackLoss + 1e-4f))
            {
                Console.WriteLine("[TrainerGate] FAIL background: compositing a background did not change the step's loss");
                return false;
            }

            var (cos, close, count) = await CompareAsync("background");
            if (cos < 0.995f || close < count - 1)
            {
                Console.WriteLine("[TrainerGate] FAIL background: the analytic gradient does not match the loss");
                return false;
            }

            // It descends: opacity only, the loss must fall. Only a little (MEASURED 0.4454 -> 0.438): most of the gate
            // scene's pixels have no splat at all, so they show the background whatever the opacities do - a floor
            // opacity cannot move. The finite-difference check above is the verification; this checks the sign.
            float first = 0f, last = 0f;
            for (int s = 0; s < 150; s++)
            {
                float l = await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, 0f,
                    SplatOptimizer.DefaultOpacityLr, frozen, readLoss: true);
                if (s == 0) first = l;
                if (s == 149) last = l;
            }
            Console.WriteLine($"[TrainerGate] background training, opacity only, 150 steps: loss {first:G4} -> {last:G4}");
            if (!(last < first))
            {
                Console.WriteLine("[TrainerGate] FAIL background: the loss did not fall");
                return false;
            }
            Console.WriteLine("[TrainerGate] background PASS");
            return true;
        }
        finally
        {
            trainer.FixedBackground = null;
            trainer.DssimWeight = ImageQuality.LambdaDssim;
            splatBuf.View.SubView(0, (long)n * SplatFormat.Floats).CopyFromCPU(snapshot);
            await accel.SynchronizeAsync();
            trainer.ResetPeakKeyDemand();
        }
    }
}
