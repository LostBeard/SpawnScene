using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Row kernels for scene editing that change the splat count (<see cref="SplatClipboard"/>): select the indices of the
/// splats in a volume, gather rows by index, append rows (moved) after a scene's. Rows are any fixed width - packed
/// splats (SplatFormat.Floats) or SH parts (PartFloatsPerSplat) - so a splat's SH rows travel with it.
/// </summary>
public static class SplatRows
{
    // ── Kernels ─────────────────────────────────────────────────────────────────────────────────────────────

    static void SelectKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, SplatEditor.Volume v,
        ArrayView1D<int, Stride1D.Dense> indices, ArrayView1D<int, Stride1D.Dense> count, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        if (!SplatEditor.Inside(v, packed[o], packed[o + 1], packed[o + 2])) return;
        int slot = Atomic.Add(ref count[0], 1);
        indices[slot] = i;
    }

    static void GatherKernel(Index1D j, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> dst, int rowFloats, int k)
    {
        if (j >= k) return;
        int s = indices[j] * rowFloats, d = j * rowFloats;
        for (int f = 0; f < rowFloats; f++) dst[d + f] = src[s + f];
    }

    static void AppendKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> scene, ArrayView1D<float, Stride1D.Dense> clip,
        ArrayView1D<float, Stride1D.Dense> dst, int n, int k, int rowFloats, float dx, float dy, float dz, int movePosition)
    {
        if (i >= n + k) return;
        int d = i * rowFloats;
        if (i < n)
        {
            for (int f = 0; f < rowFloats; f++) dst[d + f] = scene[d + f];
            return;
        }
        int s = (i - n) * rowFloats;
        for (int f = 0; f < rowFloats; f++) dst[d + f] = clip[s + f];
        if (movePosition != 0)
        {
            dst[d + SplatFormat.OffPos] += dx;
            dst[d + SplatFormat.OffPos + 1] += dy;
            dst[d + SplatFormat.OffPos + 2] += dz;
        }
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, SplatEditor.Volume, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _select;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _gather;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, float, int>? _append;
    static Accelerator? _loadedFor;

    static void Load(Accelerator a)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _select = null; _gather = null; _append = null; _loadedFor = a; }
        _select ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, SplatEditor.Volume, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(SelectKernel);
        _gather ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(GatherKernel);
        _append ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, float, int>(AppendKernel);
    }

    /// <summary>The indices of the visible splats inside the volume (any order), on the GPU, with their count
    /// (<paramref name="count"/> is the CountAsync result: the buffer is sized by it).</summary>
    public static async Task<MemoryBuffer1D<int, Stride1D.Dense>> SelectIndicesAsync(Accelerator a,
        MemoryBuffer1D<float, Stride1D.Dense> packed, int n, SplatEditor.Volume v, int count)
    {
        Load(a);
        var indices = a.Allocate1D<int>(Math.Max(1, count));
        using var counter = a.Allocate1D<int>(1);
        counter.MemSetToZero();
        _select!(n, packed.View, v, indices.View, counter.View, n);
        await a.SynchronizeAsync();
        return indices;
    }

    /// <summary>Rows <paramref name="indices"/>[0..k) of <paramref name="src"/> (rowFloats a row), packed together.</summary>
    public static MemoryBuffer1D<float, Stride1D.Dense> GatherRows(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> src,
        MemoryBuffer1D<int, Stride1D.Dense> indices, int k, int rowFloats)
    {
        Load(a);
        var dst = a.Allocate1D<float>((long)Math.Max(1, k) * rowFloats);
        if (k > 0) _gather!(k, src.View, indices.View, dst.View, rowFloats, k);
        return dst;
    }

    /// <summary>A new buffer: <paramref name="n"/> rows of <paramref name="scene"/>, then <paramref name="k"/> rows of
    /// <paramref name="clip"/> - their positions moved by <paramref name="offset"/> when <paramref name="moveRows"/>
    /// (packed splat rows; SH rows have no position).</summary>
    public static MemoryBuffer1D<float, Stride1D.Dense> AppendMoved(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> scene, int n,
        MemoryBuffer1D<float, Stride1D.Dense> clip, int k, int rowFloats, Vector3 offset, bool moveRows)
    {
        Load(a);
        var dst = a.Allocate1D<float>((long)(n + k) * rowFloats);
        _append!(n + k, scene.View, clip.View, dst.View, n, k, rowFloats, offset.X, offset.Y, offset.Z, moveRows ? 1 : 0);
        return dst;
    }
}
