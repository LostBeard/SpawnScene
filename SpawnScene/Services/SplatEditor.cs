using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Scene editing on the GPU: select splats by a volume, then delete them, keep only them (isolate), and undo.
///
/// One selection test serves every tool: a splat is inside when its position, transformed by a 4x4 matrix (row
/// vectors) and divided by w (w &gt; 0), lands in a box in that space. A screen rectangle on the desktop is the camera's
/// view * projection with the rectangle's NDC range and depth 0..1 (everything under the rectangle, near to far); a box
/// drawn with a controller in XR is the inverse of the box's transform with -1..1 on each axis.
///
/// Edits change opacity only (0 = gone; every renderer and the bounds pass skip it), so they are reversible: each edit
/// first copies the opacity column aside on the GPU, and Undo copies it back. Nothing but a selected count (one int)
/// comes to the host.
/// </summary>
public sealed class SplatEditor : IDisposable
{
    /// <summary>A selection volume (see the class summary).</summary>
    public struct Volume
    {
        public float M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44;
        public float X0, X1, Y0, Y1, Z0, Z1;
        /// <summary>When RowTo &gt; RowFrom the selection is the rows RowFrom..RowTo instead of a region - what a
        /// paste or an inserted scene just added, so it can be moved without catching the scene around it.</summary>
        public int RowFrom, RowTo;
        /// <summary>1: the selection is everything the region (or row range) does NOT take.</summary>
        public int Invert;
        /// <summary>When &gt; 0, only splats fainter than this (opacity) are selected - floaters and haze.</summary>
        public float OpacityBelow;
        /// <summary>When &gt; 0, only splats whose largest axis is longer than this (scene units) - blobs and needles.</summary>
        public float SizeAbove;

        /// <summary>This selection with the same invert and filters as <paramref name="from"/>.</summary>
        public Volume WithFiltersOf(Volume from)
        {
            var v = this;
            v.Invert = from.Invert; v.OpacityBelow = from.OpacityBelow; v.SizeAbove = from.SizeAbove;
            return v;
        }

        /// <summary>Whether this selection filters by anything besides its region.</summary>
        public readonly bool Filtered => Invert != 0 || OpacityBelow > 0f || SizeAbove > 0f;

        /// <summary>The same selection after its splats moved by <paramref name="offset"/>: a region moves with
        /// them (so a second move takes the same splats); a row range is unchanged.</summary>
        public Volume MovedBy(Vector3 offset)
        {
            if (RowTo > RowFrom) return this;
            var m = new Matrix4x4(M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44);
            var moved = From(Matrix4x4.CreateTranslation(-offset) * m, X0, X1, Y0, Y1, Z0, Z1);
            return moved.WithFiltersOf(this);
        }

        /// <summary>Rows <paramref name="from"/>..<paramref name="to"/> (exclusive).</summary>
        public static Volume Rows(int from, int to) => new() { RowFrom = from, RowTo = to, M44 = 1 };

        public static Volume From(Matrix4x4 m, float x0, float x1, float y0, float y1, float z0, float z1) => new()
        {
            M11 = m.M11, M12 = m.M12, M13 = m.M13, M14 = m.M14,
            M21 = m.M21, M22 = m.M22, M23 = m.M23, M24 = m.M24,
            M31 = m.M31, M32 = m.M32, M33 = m.M33, M34 = m.M34,
            M41 = m.M41, M42 = m.M42, M43 = m.M43, M44 = m.M44,
            X0 = x0, X1 = x1, Y0 = y0, Y1 = y1, Z0 = z0, Z1 = z1,
        };

        /// <summary>Everything under a screen rectangle (NDC x0..x1, y0..y1, y up), from the near plane to the far one,
        /// for a camera whose view * projection is <paramref name="viewProjection"/> (WebGPU depth 0..1).</summary>
        public static Volume ScreenRect(Matrix4x4 viewProjection, float ndcX0, float ndcX1, float ndcY0, float ndcY1)
            => From(viewProjection, MathF.Min(ndcX0, ndcX1), MathF.Max(ndcX0, ndcX1),
                MathF.Min(ndcY0, ndcY1), MathF.Max(ndcY0, ndcY1), 0f, 1f);

        /// <summary>The whole scene (a box no splat can leave): the region for "select all" and for filters alone.</summary>
        public static Volume All() => From(Matrix4x4.Identity, -3e38f, 3e38f, -3e38f, 3e38f, -3e38f, 3e38f);

        /// <summary>Inside a box whose transform maps the cube -1..1 into the scene.</summary>
        public static Volume Box(Matrix4x4 boxToScene)
        {
            Matrix4x4.Invert(boxToScene, out var sceneToBox);
            return From(sceneToBox, -1, 1, -1, 1, -1, 1);
        }
    }

    public enum Mode { DeleteInside = 0, KeepInside = 1 }

    /// <summary>
    /// Whether splat <paramref name="i"/> at (x, y, z) with this opacity and largest scale is selected: the filters
    /// first (fainter than, larger than), then the row range or region, inverted when asked.
    /// </summary>
    public static bool Selected(Volume v, int i, float x, float y, float z, float opacity, float maxScale)
    {
        if (v.OpacityBelow > 0f && opacity >= v.OpacityBelow) return false;
        if (v.SizeAbove > 0f && maxScale <= v.SizeAbove) return false;
        bool region = v.RowTo > v.RowFrom ? i >= v.RowFrom && i < v.RowTo : Inside(v, x, y, z);
        return region != (v.Invert != 0);
    }

    /// <summary><see cref="Selected(Volume, int, float, float, float, float, float)"/> for row <paramref name="i"/> of a packed buffer.</summary>
    public static bool SelectedRow(Volume v, ArrayView1D<float, Stride1D.Dense> packed, int i)
    {
        int o = i * SplatFormat.Floats;
        float s = packed[o + 6];
        if (packed[o + 7] > s) s = packed[o + 7];
        if (packed[o + 8] > s) s = packed[o + 8];
        return Selected(v, i, packed[o], packed[o + 1], packed[o + 2], packed[o + SplatFormat.OffOpacity], s);
    }

    /// <summary>The selection test, shared by the kernels and the tests.</summary>
    public static bool Inside(Volume v, float x, float y, float z)
    {
        float cx = x * v.M11 + y * v.M21 + z * v.M31 + v.M41;
        float cy = x * v.M12 + y * v.M22 + z * v.M32 + v.M42;
        float cz = x * v.M13 + y * v.M23 + z * v.M33 + v.M43;
        float cw = x * v.M14 + y * v.M24 + z * v.M34 + v.M44;
        if (cw <= 1e-7f) return false;
        float nx = cx / cw, ny = cy / cw, nz = cz / cw;
        return nx >= v.X0 && nx <= v.X1 && ny >= v.Y0 && ny <= v.Y1 && nz >= v.Z0 && nz <= v.Z1;
    }

    static void CountKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, Volume v, ArrayView1D<int, Stride1D.Dense> count, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        if (SelectedRow(v, packed, i)) Atomic.Add(ref count[0], 1);
    }

    static void ApplyKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, Volume v, int keepInside, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        bool inside = SelectedRow(v, packed, i);
        if (inside != (keepInside != 0)) packed[o + SplatFormat.OffOpacity] = 0f;
    }

    static void SaveOpacityKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> saved, int n)
    {
        if (i >= n) return;
        saved[i] = packed[i * SplatFormat.Floats + SplatFormat.OffOpacity];
    }

    static void RestoreOpacityKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> saved, int n)
    {
        if (i >= n) return;
        packed[i * SplatFormat.Floats + SplatFormat.OffOpacity] = saved[i];
    }

    static void MoveKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, Volume v, float dx, float dy, float dz, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        if (!SelectedRow(v, packed, i)) return;
        packed[o] += dx; packed[o + 1] += dy; packed[o + 2] += dz;
    }

    static void SavePositionsKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> saved, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        saved[i * 3] = packed[o]; saved[i * 3 + 1] = packed[o + 1]; saved[i * 3 + 2] = packed[o + 2];
    }

    static void RestorePositionsKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> saved, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        packed[o] = saved[i * 3]; packed[o + 1] = saved[i * 3 + 1]; packed[o + 2] = saved[i * 3 + 2];
    }

    static void ZeroFromKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> saved, int from, int n)
    {
        if (i >= n || i < from) return;
        saved[i] = 0f;
    }

    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>? _zeroFrom;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, ArrayView1D<int, Stride1D.Dense>, int>? _count;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>? _sizeHist;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, int, int>? _apply;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _save, _restore, _savePos, _restorePos;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, float, float, float, int>? _move;

    /// <summary>One undo step: what it restores (the opacity column, or positions) and the saved values.</summary>
    readonly record struct Snapshot(bool Positions, MemoryBuffer1D<float, Stride1D.Dense> Values)
    {
        public void Dispose() => Values.Dispose();
    }

    /// <summary>Undo snapshots (newest last). Held under <see cref="UndoBudgetBytes"/>; the oldest goes first, but the
    /// newest edit can always be undone.</summary>
    readonly List<Snapshot> _undo = new();
    int _undoSplats = -1;

    /// <summary>GPU memory the undo history may hold (14M splats: an opacity step is 56 MB, a move step 168 MB).</summary>
    public long UndoBudgetBytes { get; set; } = 256L * 1024 * 1024;

    public int UndoDepth => _undo.Count;

    void Load(Accelerator a)
    {
        _count ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, ArrayView1D<int, Stride1D.Dense>, int>(CountKernel);
        _sizeHist ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(SizeHistogramKernel);
        _apply ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, int, int>(ApplyKernel);
        _save ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(SaveOpacityKernel);
        _restore ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(RestoreOpacityKernel);
        _zeroFrom ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(ZeroFromKernel);
        _savePos ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(SavePositionsKernel);
        _restorePos ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(RestorePositionsKernel);
        _move ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, float, float, float, int>(MoveKernel);
    }

    /// <summary>Visible splats inside the volume.</summary>
    public async Task<int> CountAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, Volume v)
    {
        if (n <= 0) return 0;
        Load(a);
        using var count = a.Allocate1D<int>(1);
        count.MemSetToZero();
        _count!(n, packed.View, v, count.View, n);
        await a.SynchronizeAsync();
        // CPU transfer: one int, the selected count for the UI.
        var c = await count.CopyToHostAsync<int>(0, 1);
        return c[0];
    }

    // Size histogram: log2 of a splat's largest axis, SizeBinsPerOctave bins an octave from 2^SizeLog2Min.
    public const int SizeBins = 2048;
    const float SizeLog2Min = -24f, SizeBinsPerOctave = 64f;

    /// <summary>
    /// The largest-axis length above which the largest <paramref name="fraction"/> of the visible splats lie (the
    /// "Largest N%" filter): a 2048-bin log2 histogram on the device, walked from the top on the host. Bins are 1/64
    /// octave (1.1%) wide, so the result is the lower edge of the bin the quantile falls in. 0 when nothing is visible.
    /// </summary>
    public async Task<float> SizeQuantileAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, double fraction)
    {
        if (n <= 0 || fraction <= 0) return 0f;
        Load(a);
        using var hist = a.Allocate1D<int>(SizeBins);
        hist.MemSetToZero();
        _sizeHist!(n, packed.View, hist.View, n);
        await a.SynchronizeAsync();
        // CPU transfer: the 8 KiB histogram.
        var h = await hist.CopyToHostAsync<int>(0, SizeBins);
        return SizeQuantile(h, fraction);
    }

    /// <summary>The host half of <see cref="SizeQuantileAsync"/>, shared with the tests.</summary>
    public static float SizeQuantile(int[] hist, double fraction)
    {
        long total = 0;
        foreach (int c in hist) total += c;
        if (total == 0) return 0f;
        long want = Math.Max(1, (long)Math.Ceiling(total * fraction)), run = 0;
        for (int b = hist.Length - 1; b >= 0; b--)
        {
            run += hist[b];
            if (run >= want) return MathF.Pow(2f, SizeLog2Min + b / SizeBinsPerOctave);
        }
        return MathF.Pow(2f, SizeLog2Min);
    }

    static void SizeHistogramKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<int, Stride1D.Dense> hist, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        float s = XMath.Max(packed[o + 6], XMath.Max(packed[o + 7], packed[o + 8]));
        if (!(s > 0f)) return;
        int b = (int)((XMath.Log2(s) - SizeLog2Min) * SizeBinsPerOctave);
        b = b < 0 ? 0 : b >= SizeBins ? SizeBins - 1 : b;
        Atomic.Add(ref hist[b], 1);
    }

    /// <summary>Delete the volume's splats, or keep only them; undoable.</summary>
    public async Task ApplyAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, Volume v, Mode mode)
    {
        if (n <= 0) return;
        Load(a);
        PushSnapshot(a, packed, n, positions: false);
        _apply!(n, packed.View, v, mode == Mode.KeepInside ? 1 : 0, n);
        await a.SynchronizeAsync();
    }

    /// <summary>Move the selected splats by <paramref name="offset"/> (scene units); undoable.</summary>
    /// <remarks>A drag (the headset's grip) pushes one undo step when it starts and moves without one each frame
    /// (<paramref name="pushUndo"/> false), so Undo takes the whole drag back.</remarks>
    public async Task MoveAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, Volume v, Vector3 offset,
        bool pushUndo = true)
    {
        if (n <= 0) return;
        Load(a);
        if (pushUndo) PushSnapshot(a, packed, n, positions: true);
        _move!(n, packed.View, v, offset.X, offset.Y, offset.Z, n);
        await a.SynchronizeAsync();
    }

    void PushSnapshot(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, bool positions)
    {
        if (n != _undoSplats) ClearUndo();   // a different scene: old snapshots do not fit it
        _undoSplats = n;
        var values = a.Allocate1D<float>((long)n * (positions ? 3 : 1));
        (positions ? _savePos! : _save!)(n, packed.View, values.View, n);
        _undo.Add(new Snapshot(positions, values));
        while (_undo.Count > 1 && _undo.Sum(s => s.Values.LengthInBytes) > UndoBudgetBytes)
        {
            _undo[0].Dispose();
            _undo.RemoveAt(0);
        }
    }

    /// <summary>Undo the last edit. False when there is nothing to undo.</summary>
    public async Task<bool> UndoAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n)
    {
        if (_undo.Count == 0 || n != _undoSplats) return false;
        Load(a);
        var snap = _undo[^1];
        _undo.RemoveAt(_undo.Count - 1);
        (snap.Positions ? _restorePos! : _restore!)(n, packed.View, snap.Values.View, n);
        await a.SynchronizeAsync();
        snap.Dispose();
        return true;
    }

    /// <summary>
    /// After a paste grew the scene from <paramref name="pastedFrom"/> to <paramref name="n"/> splats: the old history
    /// no longer fits it, so it starts again with one step whose Undo hides the pasted splats (rows pastedFrom..n).
    /// </summary>
    public async Task ResetUndoForPasteAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, int pastedFrom)
    {
        ClearUndo();
        if (n <= 0) return;
        Load(a);
        _undoSplats = n;
        var snap = a.Allocate1D<float>(n);
        _save!(n, packed.View, snap.View, n);
        _zeroFrom!(n, snap.View, pastedFrom, n);
        _undo.Add(new Snapshot(false, snap));
        await a.SynchronizeAsync();
    }

    public void ClearUndo()
    {
        foreach (var s in _undo) s.Dispose();
        _undo.Clear();
        _undoSplats = -1;
    }

    public void Dispose() => ClearUndo();
}
