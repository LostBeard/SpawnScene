using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// &amp;camerabubble=X: at every floater carve, remove splats within X camera-spreads of a supervised camera's centre
    /// (0 = off, the default). A camera stands in free space - it took a photo from there - so nothing opaque belongs at it.
    /// Parity 2026-10-09: Kitchen's held-out views lose 7-16 dB to a dark blob in front of the camera on some seeds (ro:
    /// DSCF0824 -11.3 dB = 0.32 of the scene's 0.68 dB gap to gsplat); 250-370 opaque splats sit within 0.1 spreads of a
    /// held-out camera and 700-800 of a supervised one (xp splat stats). The radius is relative to the rig, like the stats.
    /// </summary>
    public static float CameraBubbleSpread { get; set; }

    static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, float>? _bubbleKernel;
    static Accelerator? _bubbleFor;

    /// <summary>Opacity 0 for splats within <see cref="CameraBubbleSpread"/> camera-spreads of a supervised camera.
    /// Returns how many (CPU transfer: one int, for the log).</summary>
    private async Task<int> CameraBubbleAsync(MemoryBuffer1D<float, Stride1D.Dense> packed, int n,
        IReadOnlyList<TrainingView> views, IReadOnlyList<int> supervised, string when)
    {
        if (CameraBubbleSpread <= 0f || n <= 0 || supervised.Count == 0) return 0;
        if (_trainer!.TrainableVolume != null) return 0;   // frozen context: see FloaterCensusAsync
        var cams = supervised.Select(vi => views[vi].Camera.Position).ToArray();
        var mean = cams.Aggregate(Vector3.Zero, (a, b) => a + b) / cams.Length;
        float spread = MathF.Sqrt(cams.Average(c => (c - mean).LengthSquared()));
        float r = CameraBubbleSpread * spread;
        var a = _gpuService.WebGPUAccelerator;
        if (!ReferenceEquals(_bubbleFor, a)) { _bubbleKernel = null; _bubbleFor = a; }
        _bubbleKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, float>(BubbleKernel);
        var flat = new float[cams.Length * 3];
        for (int i = 0; i < cams.Length; i++) { flat[i * 3] = cams[i].X; flat[i * 3 + 1] = cams[i].Y; flat[i * 3 + 2] = cams[i].Z; }
        using var camBuf = a.Allocate1D(flat);
        using var count = a.Allocate1D<int>(1);
        count.MemSetToZero();
        _bubbleKernel((Index1D)n, packed.View, camBuf.View, count.View, n, cams.Length, r * r);
        var removed = (await count.CopyToHostAsync<int>(0, 1))[0];
        Console.WriteLine($"[Floaters] camera bubble ({when}): {removed:N0} splats within {CameraBubbleSpread:G3} spreads ({r:G3} units) " +
            $"of the {cams.Length} supervised cameras removed");
        return removed;
    }

    static void BubbleKernel(Index1D i, ArrayView<float> packed, ArrayView<float> cams, ArrayView<int> count, int n, int camCount, float r2)
    {
        if (i >= n) return;
        long o = (long)i.X * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        float x = packed[o], y = packed[o + 1], z = packed[o + 2];
        for (int c = 0; c < camCount; c++)
        {
            float dx = x - cams[c * 3], dy = y - cams[c * 3 + 1], dz = z - cams[c * 3 + 2];
            if (dx * dx + dy * dy + dz * dz < r2)
            {
                packed[o + SplatFormat.OffOpacity] = 0f;
                Atomic.Add(ref count[0], 1);
                return;
            }
        }
    }
}
