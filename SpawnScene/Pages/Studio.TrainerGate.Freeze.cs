using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Frozen-context gate (SplatTrainerGpu.TrainableVolume, Studio.Partition): with a volume that excludes half the
/// splats, real training steps must leave every excluded splat where it was - position, colour, scale, rotation and
/// opacity - while the included ones still learn. Starts from a fresh optimizer state, as a partitioned block does
/// (InitOptimizerState), so no momentum from earlier gates can move anything.
/// </summary>
public partial class Studio
{
    async Task<bool> FreezeGateAsync(
        SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        const int F = SplatFormat.Floats;
        var geo = new SplatTrainerGpu.GeometryStep(
            PositionLr: 1e-3f, LogScaleLr: 5e-3f, RotationLr: 1e-3f, MinScale: 1e-7f, MaxScale: 1f);
        var p0 = await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
        // A rotated, offset frame like a partitioned block's (Studio.PlaneVolume): NOT the identity, which is its own
        // transpose and would hide a matrix laid out the wrong way round between the WGSL pass and ILGPU's kernels.
        var m = System.Numerics.Matrix4x4.CreateRotationY(0.6f) * System.Numerics.Matrix4x4.CreateRotationX(0.35f)
            * System.Numerics.Matrix4x4.CreateTranslation(0.3f, -0.2f, 0.7f);
        var us = Enumerable.Range(0, n).Select(i => System.Numerics.Vector3.Transform(
            new System.Numerics.Vector3(p0[i * F], p0[i * F + 1], p0[i * F + 2]), m).X).OrderBy(x => x).ToArray();
        float split = us[n / 2];
        // Trainable: u <= split in that frame. The other half is frozen context.
        var volume = SplatEditor.Volume.From(m, -1e30f, split, -1e30f, 1e30f, -1e30f, 1e30f);
        var frozen = Enumerable.Range(0, n).Where(i => !SplatEditor.Inside(volume, p0[i * F], p0[i * F + 1], p0[i * F + 2])).ToArray();

        var saved = trainer.TrainableVolume;
        try
        {
            trainer.InitOptimizerState(splatBuf, n);
            trainer.TrainableVolume = volume;
            for (int k = 0; k < 3; k++)
                await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 2.5e-3f, opacityLr: 5e-2f, geometry: geo);
        }
        finally { trainer.TrainableVolume = saved; }

        var p1 = await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
        int movedFrozen = 0, worstSplat = -1, worstSlot = -1;
        float worst = 0f;
        foreach (int i in frozen)
        {
            bool moved = false;
            for (int k = 0; k < F; k++)
            {
                float a = p0[i * F + k], b = p1[i * F + k];
                // Scale, opacity and rotation are rewritten each step (from log scale, the opacity logit, and the
                // quaternion renormalised): allow those round trips' ulps. MEASURED: 1.2e-7 on a quaternion component.
                float tol = k >= 6 ? 1e-5f * MathF.Max(1f, MathF.Abs(a)) : 0f;
                float d = MathF.Abs(a - b);
                if (d > tol) { moved = true; if (d > worst) { worst = d; worstSplat = i; worstSlot = k; } }
            }
            if (moved) movedFrozen++;
        }
        int learned = 0;
        for (int i = 0; i < n; i++)
        {
            if (!SplatEditor.Inside(volume, p0[i * F], p0[i * F + 1], p0[i * F + 2])) continue;
            for (int k = 0; k < F; k++) if (p0[i * F + k] != p1[i * F + k]) { learned++; break; }
        }
        Console.WriteLine($"[TrainerGate] frozen context: {frozen.Length} splats outside the volume, {movedFrozen} changed " +
            (worstSplat >= 0 ? $"(worst splat {worstSplat} float {worstSlot} by {worst:G4})" : "") +
            $"; {learned} of {n - frozen.Length} inside learned");
        if (movedFrozen > 0 || learned == 0)
        {
            Console.WriteLine($"[TrainerGate] FAIL frozen context: {(movedFrozen > 0 ? "frozen splats moved" : "nothing inside learned")}");
            return false;
        }
        Console.WriteLine("[TrainerGate] frozen context PASS");
        return await FrozenDensifyGateAsync(splatBuf, n, volume, frozen);
    }

    /// <summary>
    /// GpuDensify on THIS accelerator (the CPU test covers the logic; this covers the WebGPU build of the kernels and
    /// their parameters): every splat a densify candidate, half of them faint, every one past the screen-size bar,
    /// the opacity reset on - and the frozen ones must come out exactly as they went in, the others pruned or grown.
    /// </summary>
    async Task<bool> FrozenDensifyGateAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        SplatEditor.Volume volume, int[] frozen)
    {
        const int F = SplatFormat.Floats;
        var a = _gpuService.WebGPUAccelerator;
        var p = await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
        for (int i = 0; i < n; i += 2) p[i * F + 9] = 0.001f;   // faint: below SplatDensityControl.MinOpacity
        var stats = new float[n * 2];
        for (int i = 0; i < n; i++) { stats[i * 2] = 1f; stats[i * 2 + 1] = 1f; }   // average 1: far over the bar
        var radius = Enumerable.Repeat(1e4f, n).ToArray();                          // past any screen-size bar
        using var pb = a.Allocate1D(p);
        using var sb = a.Allocate1D(stats);
        using var rb = a.Allocate1D(radius);
        var d = new GpuDensify(a);
        var r = await d.RunAsync(pb.View, n, sb.View, rb.View,
            new GpuDensify.Options(SceneExtent: 1f, AfterFirstOpacityReset: true, MaxSplats: int.MaxValue,
                ResetOpacity: true, Seed: 5, Trainable: volume));
        try
        {
            var outRows = await r.Packed.CopyToHostAsync<float>(0, (long)r.Count * F);
            var feat = await r.FeatureSources.CopyToHostAsync<int>(0, r.Count);
            var isFrozen = new HashSet<int>(frozen);
            int frozenOut = 0, changed = 0;
            for (int j = 0; j < r.Count; j++)
            {
                if (feat[j] < 0 || !isFrozen.Contains(feat[j])) continue;
                frozenOut++;
                for (int k = 0; k < F; k++) if (outRows[j * F + k] != p[feat[j] * F + k]) { changed++; break; }
            }
            Console.WriteLine($"[TrainerGate] frozen densify: {frozen.Length} frozen in, {frozenOut} out, {changed} changed; " +
                $"inside: {r}");
            if (frozenOut != frozen.Length || changed > 0 || r.PrunedFaint + r.PrunedBig == 0)
            {
                Console.WriteLine("[TrainerGate] FAIL frozen densify: " +
                    (frozenOut != frozen.Length ? $"{frozen.Length - frozenOut} frozen splats removed or {frozenOut - frozen.Length} added" :
                     changed > 0 ? "frozen splats changed" : "nothing inside was pruned - the gate is not exercising the prunes"));
                return false;
            }
            Console.WriteLine("[TrainerGate] frozen densify PASS");
            return true;
        }
        finally { r.Packed.Dispose(); r.AdamSources.Dispose(); r.FeatureSources.Dispose(); }
    }
}
