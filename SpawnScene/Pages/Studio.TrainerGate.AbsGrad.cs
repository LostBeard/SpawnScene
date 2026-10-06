using ILGPU;
using ILGPU.Runtime;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// AbsGS gate (SplatTrainerGpu.AbsGrad): from ONE backward pass, the densify statistic in both modes. Per splat and axis
/// the sum of per-pixel magnitudes is at least the magnitude of the signed sum (|sum v| &lt;= sum |v|), so every
/// splat's AbsGS value must be at least its reference value - and above it for splats whose pixels pull opposite
/// ways, which a real scene has plenty of. A max-reduce, a lost compare-exchange add or a wrong binding breaks one or
/// the other.
/// </summary>
public partial class Studio
{
    async Task<bool> AbsGradGateAsync(SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        bool was = trainer.AbsGrad;
        try
        {
            trainer.AbsGrad = false;
            trainer.ResetDensifyStats();
            await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0f);
            trainer.AccumulateDensifyStats(n);
            var signed = await trainer.ReadDensifyStatsAsync(n);
            trainer.AbsGrad = true;
            trainer.ResetDensifyStats();
            trainer.AccumulateDensifyStats(n);   // the same backward's data
            var abs = await trainer.ReadDensifyStatsAsync(n);
            int counted = 0, below = 0, larger = 0, first = -1;
            for (int i = 0; i < n; i++)
            {
                if (signed[i].VisibleCount == 0) continue;
                counted++;
                float s = signed[i].GradientSum, a = abs[i].GradientSum;
                if (a < s * (1f - 1e-4f) - 1e-12f) { below++; if (first < 0) first = i; }
                if (a > s * 1.01f) larger++;
            }
            Console.WriteLine($"[TrainerGate] absgrad: {counted} splats with a gradient, {larger} with |grad| sums > 1% over the " +
                $"signed sum, {below} under it" + (first >= 0 ? $" (first {first}: abs {abs[first].GradientSum:G4} < signed {signed[first].GradientSum:G4})" : ""));
            if (counted == 0 || below > 0 || larger == 0)
            {
                Console.WriteLine("[TrainerGate] FAIL absgrad: " + (counted == 0 ? "no splat got a gradient" :
                    below > 0 ? "a sum of magnitudes came out under its signed sum" : "no splat's magnitudes exceeded its signed sum"));
                return false;
            }
            Console.WriteLine("[TrainerGate] absgrad PASS");
            return true;
        }
        finally
        {
            trainer.AbsGrad = was;
            trainer.ResetDensifyStats();
        }
    }
}
