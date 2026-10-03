using ILGPU;
using System.Numerics;
using SpawnScene.Models;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnScene.Services;

/// <summary>
/// World-space axis-aligned bounds of a packed splat buffer, computed on the GPU.
///
/// Training needs a depth range per view to quantise its sort keys into an 18-bit field. Getting
/// that from the scene's actual extent (rather than a constant tuned to one dataset) is what lets
/// the same code path handle a room as well as an object on a turntable.
///
/// Only the six bounds come back to the host - scalar metadata, not bulk data.
///
/// Atomic min/max over floats is done on the integer bit pattern through a monotonic mapping,
/// because the GPU only has integer atomics. Positive floats already compare correctly as signed
/// ints; negatives compare backwards, so their magnitude bits are flipped. The mapping is exact
/// and order-preserving across the whole range including -0.0.
/// </summary>
public static class SplatBounds
{
    public readonly record struct Aabb(
        float MinX, float MinY, float MinZ,
        float MaxX, float MaxY, float MaxZ)
    {
        public float CentreX => 0.5f * (MinX + MaxX);
        public float CentreY => 0.5f * (MinY + MaxY);
        public float CentreZ => 0.5f * (MinZ + MaxZ);
        public float Diagonal => MathF.Sqrt(
            (MaxX - MinX) * (MaxX - MinX) +
            (MaxY - MinY) * (MaxY - MinY) +
            (MaxZ - MinZ) * (MaxZ - MinZ));
    }

    /// <summary>float -> signed int, order preserving. Public so it can be tested directly.</summary>
    public static int Ordered(float f)
    {
        int i = BitConverter.SingleToInt32Bits(f);
        return i ^ ((i >> 31) & 0x7FFFFFFF);
    }

    /// <summary>Inverse of <see cref="Ordered"/>.</summary>
    public static float Unordered(int i)
    {
        return BitConverter.Int32BitsToSingle(i ^ ((i >> 31) & 0x7FFFFFFF));
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _kernel;
    static Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int>? _seed;

    static void BoundsKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> bounds,   // 6: minX,minY,minZ,maxX,maxY,maxZ (ordered ints)
        int splatCount)
    {
        int i = index;
        if (i >= splatCount) return;
        int o = i * SplatFormat.Floats;

        // A splat with no opacity contributes nothing to any view, so it must not stretch the
        // depth range that every other splat's sort key is quantised against.
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;

        for (int a = 0; a < 3; a++)
        {
            int v = Ordered(packed[o + a]);
            Atomic.Min(ref bounds[a], v);
            Atomic.Max(ref bounds[3 + a], v);
        }
    }

    static void SeedKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> bounds,
        int minSeed,
        int maxSeed)
    {
        int i = index;
        if (i >= 6) return;
        bounds[i] = i < 3 ? minSeed : maxSeed;
    }

    /// <summary>
    /// Compute the bounds of <paramref name="splatCount"/> splats in a packed
    /// (<see cref="SplatFormat.Floats"/> per splat) buffer. Returns null if nothing has opacity.
    /// </summary>
    public static async Task<Aabb?> ComputeAsync(
        Accelerator accel,
        MemoryBuffer1D<float, Stride1D.Dense> packed,
        int splatCount)
    {
        if (splatCount <= 0) return null;

        using var bounds = accel.Allocate1D<int>(6);
        _seed ??= accel.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<int, Stride1D.Dense>, int, int>(SeedKernel);
        _seed(6, bounds.View, int.MaxValue, int.MinValue);

        _kernel ??= accel.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(BoundsKernel);
        _kernel(splatCount, packed.View, bounds.View, splatCount);
        await accel.SynchronizeAsync();

        // CPU transfer: 6 scalars of scene metadata.
        int[] b = await bounds.CopyToHostAsync<int>(0, 6);

        // Untouched seeds mean every splat was transparent - no usable bounds.
        if (b[0] == int.MaxValue) return null;

        return new Aabb(
            Unordered(b[0]), Unordered(b[1]), Unordered(b[2]),
            Unordered(b[3]), Unordered(b[4]), Unordered(b[5]));
    }

    // ── Robust bounds: per-axis percentiles, so a few stray floaters do not set the scene's size ──────────────────
    /// <summary>Histogram bins per axis (plus one "below" and one "above" counter at each end).</summary>
    public const int RobustBins = 1024;
    const int AxisSlots = RobustBins + 2;

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int,
        float, float, float, float, float, float>? _histKernel;

    static void HistogramKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<int, Stride1D.Dense> hist,   // 3 axes x AxisSlots: [below, bins..., above]
        int splatCount,
        float minX, float minY, float minZ,
        float invX, float invY, float invZ)      // bins per scene unit
    {
        int i = index;
        if (i >= splatCount) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        for (int a = 0; a < 3; a++)
        {
            float t = a == 0 ? (packed[o] - minX) * invX : a == 1 ? (packed[o + 1] - minY) * invY : (packed[o + 2] - minZ) * invZ;
            int slot = t < 0f ? 0 : t >= RobustBins ? RobustBins + 1 : 1 + (int)t;
            Atomic.Add(ref hist[a * AxisSlots + slot], 1);
        }
    }

    /// <summary>
    /// The <paramref name="tail"/> and 1 - <paramref name="tail"/> quantiles of one axis's histogram (layout
    /// [below, RobustBins bins, above] over [min, min + RobustBins / inv)), as the start of the low bin and the end of
    /// the high bin - so the range always contains the quantiles. Quantiles inside the below/above counters clamp to
    /// the histogram's range. Public so it can be tested directly.
    /// </summary>
    public static (float Lo, float Hi) QuantileRange(ReadOnlySpan<int> axis, float min, float inv, double tail)
    {
        long total = 0;
        foreach (var c in axis) total += c;
        if (total == 0) return (min, min + RobustBins / inv);
        double cut = tail * total;
        long run = axis[0];
        int lo = 0;
        while (lo < RobustBins - 1 && run + axis[1 + lo] <= cut) run += axis[1 + lo++];
        run = axis[RobustBins + 1];
        int hi = RobustBins - 1;
        while (hi > lo && run + axis[1 + hi] <= cut) run += axis[1 + hi--];
        return (min + lo / inv, min + (hi + 1) / inv);
    }

    /// <summary>
    /// Bounds that hold all but the <paramref name="tail"/> fraction of splats at each end of each axis (default 1%),
    /// on the GPU: the exact box, then a histogram over it, then a second histogram over the first one's quantile range
    /// (a floater 1000 scene units out leaves 1024 bins too coarse for a 3-unit room on its own). The host reads back
    /// 6 ints and two 12 KB histograms. Returns null if nothing has opacity. The XR AR placement sizes a scene with it.
    /// </summary>
    public static async Task<Aabb?> ComputeRobustAsync(
        Accelerator accel,
        MemoryBuffer1D<float, Stride1D.Dense> packed,
        int splatCount,
        double tail = 0.01)
    {
        var exact = await ComputeAsync(accel, packed, splatCount);
        if (exact is not { } box) return null;
        _histKernel ??= accel.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int,
            float, float, float, float, float, float>(HistogramKernel);
        using var hist = accel.Allocate1D<int>(3 * AxisSlots);

        float[] boxMin = { box.MinX, box.MinY, box.MinZ }, boxMax = { box.MaxX, box.MaxY, box.MaxZ };
        var min = (float[])boxMin.Clone();
        var max = (float[])boxMax.Clone();
        var lo = new float[3];
        var hi = new float[3];
        for (int pass = 0; pass < 2; pass++)
        {
            var inv = new float[3];
            for (int a = 0; a < 3; a++) inv[a] = RobustBins / MathF.Max(max[a] - min[a], 1e-6f);
            hist.MemSetToZero();
            _histKernel(splatCount, packed.View, hist.View, splatCount, min[0], min[1], min[2], inv[0], inv[1], inv[2]);
            await accel.SynchronizeAsync();
            // CPU transfer: 3 x 1026 bin counts of scene metadata (the quantile search is a few thousand adds).
            int[] h = await hist.CopyToHostAsync<int>(0, 3 * AxisSlots);
            for (int a = 0; a < 3; a++)
            {
                (lo[a], hi[a]) = QuantileRange(h.AsSpan(a * AxisSlots, AxisSlots), min[a], inv[a], tail);
                // The next pass looks one bin wider each side, so a quantile on a bin edge is not lost to rounding.
                float w = 1f / inv[a];
                min[a] = MathF.Max(lo[a] - w, boxMin[a]);
                max[a] = MathF.Min(hi[a] + w, boxMax[a]);
            }
        }
        return new Aabb(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    }

    /// <summary>
    /// Near/far planes that bracket this box from one camera. Used to quantise the tile sort
    /// key's depth field, so it must cover every splat or distinct depths collapse to one value
    /// and the front-to-back order is lost.
    /// </summary>
    public static (float Near, float Far) DepthRangeFor(Aabb box, CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out _, out _, out var fwd, out var pos);
        float lo = float.MaxValue, hi = float.MinValue;
        for (int c = 0; c < 8; c++)
        {
            var corner = new Vector3(
                (c & 1) == 0 ? box.MinX : box.MaxX,
                (c & 2) == 0 ? box.MinY : box.MaxY,
                (c & 4) == 0 ? box.MinZ : box.MaxZ);
            float d = Vector3.Dot(fwd, corner - pos);
            if (d < lo) lo = d;
            if (d > hi) hi = d;
        }
        // Corners behind the camera are legitimate; the range just has to stay positive and
        // non-degenerate.
        float near = MathF.Max(lo, 1e-4f);
        float far = MathF.Max(hi, near * 1.001f + 1e-4f);
        return (near, far);
    }
}
