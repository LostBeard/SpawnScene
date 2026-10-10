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

    /// <summary>
    /// &amp;minviews=N: at every floater carve, remove splats whose centre lies inside fewer than N supervised photos (in
    /// frame and past the near plane; 0 = off, the default). Parity 2026-10-09 (&amp;blobpick): Kitchen's held-out blob on
    /// DSCF0688 (seed 4, 21.6 dB vs gsplat 27.8) was two dark splats 0.22-0.26 units in front of the camera, 0.07-0.09
    /// spreads from the nearest supervised camera, IN FRAME OF ONLY 2 OF 244 PHOTOS - under-constrained splats two photos
    /// used to darken a corner, which no other view ever checked. Real surfaces are seen by many (Kitchen's distant window
    /// background: 32-52 photos).
    /// </summary>
    public static int MinViewsSupport { get; set; }

    static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, float, int>? _supportKernel;
    static Accelerator? _supportFor;

    /// <summary>Opacity 0 for splats in frame of fewer than <see cref="MinViewsSupport"/> supervised photos. Returns how many.</summary>
    private async Task<int> ViewSupportCarveAsync(MemoryBuffer1D<float, Stride1D.Dense> packed, int n,
        IReadOnlyList<TrainingView> views, IReadOnlyList<int> supervised, string when)
    {
        if (MinViewsSupport <= 0 || n <= 0 || supervised.Count == 0) return 0;
        if (_trainer!.TrainableVolume != null) return 0;   // frozen context: see FloaterCensusAsync
        // 18 floats a camera: position, forward, up, right, fx, fy, cx, cy, w, h (the BlobPick / Wander projection).
        const int CamFloats = 18;
        var flat = new float[supervised.Count * CamFloats];
        for (int k = 0; k < supervised.Count; k++)
        {
            var c = views[supervised[k]].Camera;
            var f = Vector3.Normalize(c.Forward); var u = Vector3.Normalize(c.Up); var r = Vector3.Cross(f, u);
            int o = k * CamFloats;
            flat[o] = c.Position.X; flat[o + 1] = c.Position.Y; flat[o + 2] = c.Position.Z;
            flat[o + 3] = f.X; flat[o + 4] = f.Y; flat[o + 5] = f.Z;
            flat[o + 6] = u.X; flat[o + 7] = u.Y; flat[o + 8] = u.Z;
            flat[o + 9] = r.X; flat[o + 10] = r.Y; flat[o + 11] = r.Z;
            flat[o + 12] = c.FocalX; flat[o + 13] = c.FocalY; flat[o + 14] = c.CenterX; flat[o + 15] = c.CenterY;
            flat[o + 16] = c.Width; flat[o + 17] = c.Height;
        }
        var a = _gpuService.WebGPUAccelerator;
        if (!ReferenceEquals(_supportFor, a)) { _supportKernel = null; _supportFor = a; }
        _supportKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, float, int>(SupportKernel);
        using var camBuf = a.Allocate1D(flat);
        using var count = a.Allocate1D<int>(1);
        count.MemSetToZero();
        _supportKernel((Index1D)n, packed.View, camBuf.View, count.View, n, supervised.Count, SplatTrainerGpu.NearPlane, MinViewsSupport);
        var removed = (await count.CopyToHostAsync<int>(0, 1))[0];
        Console.WriteLine($"[Floaters] view support ({when}): {removed:N0} splats in frame of fewer than {MinViewsSupport} of the " +
            $"{supervised.Count} supervised photos removed");
        return removed;
    }

    static void SupportKernel(Index1D i, ArrayView<float> packed, ArrayView<float> cams, ArrayView<int> count, int n, int camCount,
        float near, int minViews)
    {
        if (i >= n) return;
        long o = (long)i.X * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        float x = packed[o], y = packed[o + 1], z0 = packed[o + 2];
        int seen = 0;
        for (int c = 0; c < camCount && seen < minViews; c++)
        {
            int b = c * 18;
            float rx = x - cams[b], ry = y - cams[b + 1], rz = z0 - cams[b + 2];
            float z = rx * cams[b + 3] + ry * cams[b + 4] + rz * cams[b + 5];
            if (z <= near) continue;
            float px = cams[b + 12] * (rx * cams[b + 9] + ry * cams[b + 10] + rz * cams[b + 11]) / z + cams[b + 14];
            float py = cams[b + 15] - cams[b + 13] * (rx * cams[b + 6] + ry * cams[b + 7] + rz * cams[b + 8]) / z;
            if (px >= 0f && py >= 0f && px < cams[b + 16] && py < cams[b + 17]) seen++;
        }
        if (seen < minViews)
        {
            packed[o + SplatFormat.OffOpacity] = 0f;
            Atomic.Add(ref count[0], 1);
        }
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
