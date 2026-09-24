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
        return true;
    }
}
