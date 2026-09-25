using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Densify radius gate: after one step, does every contributing splat report the 3-sigma screen radius the
/// CPU projection gives it, and every other splat 0? Feeds the reference max_screen_size prune
/// (SplatDensityControl.MaxScreenRadiusPx), which nothing used to fill - so it never fired.
/// </summary>
public partial class Studio
{
    async Task<bool> DensifyRadiusGateAsync(
        SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        trainer.ResetDensifyStats();
        // Zero learning rates and no geometry step: the splats do not move, so the CPU projects the same ones.
        await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0f);
        trainer.AccumulateDensifyStats(n);
        var stats = await trainer.ReadDensifyStatsAsync(n);
        var packed = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);

        int visible = 0, bad = 0, first = -1;
        float worst = 0f;
        for (int i = 0; i < n; i++)
        {
            float got = stats[i].MaxScreenRadiusPx;
            if (stats[i].VisibleCount == 0)
            {
                if (got != 0f) { bad++; if (first < 0) first = i; }
                continue;
            }
            visible++;
            var one = packed.AsSpan(i * SplatFormat.Floats, SplatFormat.Floats).ToArray();
            var proj = ProjectForCpu(one, 1, cam);
            float want = proj.Count == 1 ? SplatTileRasterizer.Extent(proj[0]) : 0f;
            float rel = MathF.Abs(got - want) / MathF.Max(want, 1e-6f);
            worst = MathF.Max(worst, rel);
            if (!(rel < 1e-3f)) { bad++; if (first < 0) first = i; }
        }
        if (visible == 0 || bad > 0)
        {
            Console.WriteLine($"[TrainerGate] FAIL: densify radius: {bad} wrong of {n} ({visible} visible)" +
                (first >= 0 ? $", first splat {first}: got {stats[first].MaxScreenRadiusPx}, visible {stats[first].VisibleCount}" : ""));
            return false;
        }
        Console.WriteLine($"[TrainerGate] densify radius PASS: {visible}/{n} contributing splats, max rel err {worst:G3}");
        return await DensifyFrustumDenominatorGateAsync(trainer, splatBuf, n, cam, depthNear, depthFar, packed, stats);
    }

    /// <summary>
    /// Frustum denominator (SplatTrainerGpu.DensifyDenominatorFrustum, the reference's radii &gt; 0): after one
    /// step every splat whose CPU projection lands on at least one tile counts 1 visible step, every other splat
    /// 0, and a splat that emitted but received no gradient adds 0 to the gradient sum.
    ///
    /// The gate scene alone cannot tell the two denominators apart: every on-screen splat in it gets a gradient
    /// (MEASURED: 240/240). So this runs on a private copy in which every 16th splat has opacity 1e-4 - below
    /// the 1/255 raster cutoff, so it still projects and emits keys but can never be credited a gradient. The
    /// contribution-mode baseline is re-measured on the same copy, and the stage demands such splats exist.
    /// </summary>
    async Task<bool> DensifyFrustumDenominatorGateAsync(
        SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar,
        float[] scenePacked, SplatDensityControl.Accumulator[] sceneContrib)
    {
        // Gate scale (a few hundred splats): the host copy is the test's input, not a production path.
        var packed = (float[])scenePacked.Clone();
        for (int i = 0; i < n; i += 16) packed[i * SplatFormat.Floats + 9] = 1e-4f;
        using var silentBuf = splatBuf.Accelerator.Allocate1D<float>(packed.Length);
        silentBuf.CopyFromCPU(packed);

        async Task<SplatDensityControl.Accumulator[]> StepAsync(bool frustum)
        {
            bool was = SplatTrainerGpu.DensifyDenominatorFrustum;
            SplatTrainerGpu.DensifyDenominatorFrustum = frustum;
            try
            {
                trainer.ResetDensifyStats();
                await trainer.TrainStepAsync(silentBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0f);
                trainer.AccumulateDensifyStats(n);
                return await trainer.ReadDensifyStatsAsync(n);
            }
            finally { SplatTrainerGpu.DensifyDenominatorFrustum = was; }
        }
        // The opacity step writes splats[9] from the trainer's own logit buffer, so the logits must come from
        // the copy too, or the first step restores every silent splat's original opacity (MEASURED: it did).
        trainer.SeedLogits(silentBuf, n);
        SplatDensityControl.Accumulator[] contrib, stats;
        try
        {
            contrib = await StepAsync(false);
            stats = await StepAsync(true);
        }
        finally { trainer.SeedLogits(splatBuf, n); }

        int tilesX = (cam.Width + SplatTileRasterizer.TileSize - 1) / SplatTileRasterizer.TileSize;
        int tilesY = (cam.Height + SplatTileRasterizer.TileSize - 1) / SplatTileRasterizer.TileSize;
        int onScreen = 0, emittedNoGrad = 0, bad = 0, first = -1;
        for (int i = 0; i < n; i++)
        {
            var proj = ProjectForCpu(packed.AsSpan(i * SplatFormat.Floats, SplatFormat.Floats).ToArray(), 1, cam);
            bool emits = proj.Count == 1
                && SplatTileRasterizer.Span(proj[0], SplatTileRasterizer.Extent(proj[0]), tilesX, tilesY).Count > 0;
            int want = emits ? 1 : 0;
            if (emits) onScreen++;
            bool noGrad = emits && contrib[i].VisibleCount == 0;
            if (noGrad) emittedNoGrad++;
            // Same step, same splats: a contributing splat's gradient sum is unchanged; a silent one adds 0.
            float wantSum = contrib[i].VisibleCount > 0 ? contrib[i].GradientSum : 0f;
            bool sumOk = MathF.Abs(stats[i].GradientSum - wantSum) <= 1e-5f * MathF.Max(1f, MathF.Abs(wantSum));
            if (stats[i].VisibleCount != want || !sumOk) { bad++; if (first < 0) first = i; }
        }
        if (bad > 0 || emittedNoGrad == 0)
        {
            Console.WriteLine($"[TrainerGate] FAIL: densify frustum denominator: {bad} wrong of {n} " +
                $"({onScreen} on screen, {emittedNoGrad} on screen with no gradient)" +
                (first >= 0 ? $", first splat {first}: visible {stats[first].VisibleCount}, sum {stats[first].GradientSum:G6} " +
                    $"(contrib mode: visible {contrib[first].VisibleCount}, sum {contrib[first].GradientSum:G6})" : "") +
                (emittedNoGrad == 0 ? " - no on-screen splat without a gradient, so the modes are indistinguishable" : ""));
            return false;
        }
        Console.WriteLine($"[TrainerGate] densify frustum denominator PASS: {onScreen}/{n} on screen, " +
            $"{emittedNoGrad} of them with no gradient counted visible");
        return true;
    }
}
