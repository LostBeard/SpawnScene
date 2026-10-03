using ILGPU;
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

        /// <summary>Inside a box whose transform maps the cube -1..1 into the scene.</summary>
        public static Volume Box(Matrix4x4 boxToScene)
        {
            Matrix4x4.Invert(boxToScene, out var sceneToBox);
            return From(sceneToBox, -1, 1, -1, 1, -1, 1);
        }
    }

    public enum Mode { DeleteInside = 0, KeepInside = 1 }

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
        if (Inside(v, packed[o], packed[o + 1], packed[o + 2])) Atomic.Add(ref count[0], 1);
    }

    static void ApplyKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> packed, Volume v, int keepInside, int n)
    {
        if (i >= n) return;
        int o = i * SplatFormat.Floats;
        if (packed[o + SplatFormat.OffOpacity] <= 0f) return;
        bool inside = Inside(v, packed[o], packed[o + 1], packed[o + 2]);
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

    static void ZeroFromKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> saved, int from, int n)
    {
        if (i >= n || i < from) return;
        saved[i] = 0f;
    }

    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>? _zeroFrom;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, ArrayView1D<int, Stride1D.Dense>, int>? _count;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, int, int>? _apply;
    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _save, _restore;

    /// <summary>Undo snapshots (opacity columns, newest last). Held under <see cref="UndoBudgetBytes"/>; the oldest
    /// goes first, but the newest edit can always be undone.</summary>
    readonly List<MemoryBuffer1D<float, Stride1D.Dense>> _undo = new();
    int _undoSplats = -1;

    /// <summary>GPU memory the undo history may hold (a 14M-splat scene: 56 MB a step, so 4 steps).</summary>
    public long UndoBudgetBytes { get; set; } = 256L * 1024 * 1024;

    public int UndoDepth => _undo.Count;

    void Load(Accelerator a)
    {
        _count ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, ArrayView1D<int, Stride1D.Dense>, int>(CountKernel);
        _apply ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, Volume, int, int>(ApplyKernel);
        _save ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(SaveOpacityKernel);
        _restore ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(RestoreOpacityKernel);
        _zeroFrom ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(ZeroFromKernel);
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

    /// <summary>Delete the volume's splats, or keep only them; undoable.</summary>
    public async Task ApplyAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n, Volume v, Mode mode)
    {
        if (n <= 0) return;
        Load(a);
        if (n != _undoSplats) ClearUndo();   // a different scene: old snapshots do not fit it
        _undoSplats = n;
        var snap = a.Allocate1D<float>(n);
        _save!(n, packed.View, snap.View, n);
        _undo.Add(snap);
        while (_undo.Count > 1 && (long)_undo.Count * n * sizeof(float) > UndoBudgetBytes)
        {
            _undo[0].Dispose();
            _undo.RemoveAt(0);
        }
        _apply!(n, packed.View, v, mode == Mode.KeepInside ? 1 : 0, n);
        await a.SynchronizeAsync();
    }

    /// <summary>Undo the last edit. False when there is nothing to undo.</summary>
    public async Task<bool> UndoAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int n)
    {
        if (_undo.Count == 0 || n != _undoSplats) return false;
        Load(a);
        var snap = _undo[^1];
        _undo.RemoveAt(_undo.Count - 1);
        _restore!(n, packed.View, snap.View, n);
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
        _undo.Add(snap);
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
