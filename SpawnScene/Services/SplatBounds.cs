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
