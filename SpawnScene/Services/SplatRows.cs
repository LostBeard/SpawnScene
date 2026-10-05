using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
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
        if (!SplatEditor.Selected(v, i, packed[o], packed[o + 1], packed[o + 2])) return;
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

    // Display colour = C0 * dc + 0.5 (SphericalHarmonics.WgslViewRgb). A trained scene stores dc, others RGB 0..1.
    const float ShC0 = 0.28209479177387814f;

    static void ConvertColoursKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, int n, int toShDc)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats + SplatFormat.OffColor;
        for (int c = 0; c < 3; c++)
        {
            float v = packed[o + c];
            packed[o + c] = toShDc != 0 ? (v - 0.5f) / ShC0 : XMath.Clamp(ShC0 * v + 0.5f, 0f, 1f);
        }
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>? _convert;

    /// <summary>
    /// One packed row rotated by the unit quaternion (rx, ry, rz, rw) about the origin: the position turned, and the
    /// splat's orientation composed AFTER its own (q' = r * q, Hamilton, x y z w - the convention GpuDensify's split and
    /// System.Numerics' Vector3.Transform use). Scalar, so the kernel and the CPU tests run the same code.
    /// </summary>
    public static void RotateRow(ArrayView1D<float, Stride1D.Dense> packed, long o, float rx, float ry, float rz, float rw)
    {
        float px = packed[o], py = packed[o + 1], pz = packed[o + 2];
        // v' = v + 2w (r x v) + 2 r x (r x v)
        float tx = 2f * (ry * pz - rz * py), ty = 2f * (rz * px - rx * pz), tz = 2f * (rx * py - ry * px);
        packed[o] = px + rw * tx + (ry * tz - rz * ty);
        packed[o + 1] = py + rw * ty + (rz * tx - rx * tz);
        packed[o + 2] = pz + rw * tz + (rx * ty - ry * tx);
        long q = o + 10;
        float qx = packed[q], qy = packed[q + 1], qz = packed[q + 2], qw = packed[q + 3];
        packed[q] = rw * qx + rx * qw + ry * qz - rz * qy;
        packed[q + 1] = rw * qy - rx * qz + ry * qw + rz * qx;
        packed[q + 2] = rw * qz + rx * qy - ry * qx + rz * qw;
        packed[q + 3] = rw * qw - rx * qx - ry * qy - rz * qz;
    }

    static void RotateKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, int n, float rx, float ry, float rz, float rw)
    {
        if (i >= n) return;
        RotateRow(packed, (long)i.X * SplatFormat.Floats, rx, ry, rz, rw);
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, float, float, float, float>? _rotate;

    /// <summary>Rotate <paramref name="n"/> packed splats about the origin by unit quaternion <paramref name="r"/>
    /// (positions and orientations), in place on the GPU.</summary>
    public static void Rotate(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, System.Numerics.Quaternion r)
    {
        Load(a);
        if (n > 0) _rotate!(n, packed.View, n, r.X, r.Y, r.Z, r.W);
    }

    /// <summary>
    /// The rotation that turns <paramref name="meanUp"/> to +Y (the shortest arc), or null when it already points
    /// within ~3 degrees of +Y - a scene that is upright stays bit-identical.
    /// </summary>
    public static System.Numerics.Quaternion? UprightRotation(System.Numerics.Vector3 meanUp)
    {
        if (meanUp.LengthSquared() < 1e-12f) return null;
        var u = System.Numerics.Vector3.Normalize(meanUp);
        float c = u.Y;   // dot(u, +Y)
        if (c > 0.9986f) return null;
        var axis = System.Numerics.Vector3.Cross(u, System.Numerics.Vector3.UnitY);
        if (axis.LengthSquared() < 1e-12f) axis = System.Numerics.Vector3.UnitX;   // exactly upside down: any horizontal axis
        return System.Numerics.Quaternion.CreateFromAxisAngle(System.Numerics.Vector3.Normalize(axis), MathF.Acos(Math.Clamp(c, -1f, 1f)));
    }

    /// <summary>Rewrite <paramref name="n"/> packed splats' colours as SH DC coefficients (from RGB) or as RGB (from
    /// SH DC - the view-dependent bands are dropped by the caller): so splats from a trained and an untrained scene
    /// can live in one scene.</summary>
    public static void ConvertColours(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, bool toShDc)
    {
        Load(a);
        if (n > 0) _convert!(n, packed.View, n, toShDc ? 1 : 0);
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, SplatEditor.Volume, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _select;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _gather;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, float, int>? _append;
    static Accelerator? _loadedFor;

    static void Load(Accelerator a)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _select = null; _gather = null; _append = null; _convert = null; _rotate = null; _loadedFor = a; }
        _rotate ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, float, float, float, float>(RotateKernel);
        _convert ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(ConvertColoursKernel);
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
