using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Copy and paste of scene parts, on the GPU. Copy gathers the splats inside a selection volume
/// (<see cref="SplatEditor.Volume"/>) - their packed rows and, for a trained scene, their SH colour rows - into a
/// clipboard; Paste builds a new, larger scene: the current splats followed by the clipboard's, moved by an offset.
/// Cut is Copy then Delete. Translation only for now: turning a copy would also have to rotate its SH bands.
///
/// The row kernels live in <see cref="SplatRows"/> (any accelerator, tested on the CPU one); this class is the glue
/// that moves the renderer's buffers (packed splats, raw WebGPU SH parts) through them.
/// </summary>
public sealed class SplatClipboard : IDisposable
{
    /// <summary>The copied splats (SplatFormat.Floats a splat).</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Packed { get; }
    /// <summary>Their SH rows, one buffer per part (PartFloatsPerSplat a splat), or null (no SH).</summary>
    public MemoryBuffer1D<float, Stride1D.Dense>[]? Sh { get; }
    public int Count { get; }
    public int ShDegree { get; }
    public bool ColoursAreShDc { get; private set; }
    /// <summary>The copy's bounds (exact) - where it came from, and how big it is.</summary>
    public SplatBounds.Aabb Bounds { get; }

    SplatClipboard(MemoryBuffer1D<float, Stride1D.Dense> packed, MemoryBuffer1D<float, Stride1D.Dense>[]? sh, int count,
        int shDegree, bool shDc, SplatBounds.Aabb bounds)
    {
        Packed = packed; Sh = sh; Count = count; ShDegree = shDegree; ColoursAreShDc = shDc; Bounds = bounds;
    }

    /// <summary>A whole scene as a clipboard (for Insert scene): takes ownership of the buffers.</summary>
    public static async Task<SplatClipboard> FromSceneAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, int count,
        MemoryBuffer1D<float, Stride1D.Dense>[]? sh, int shDegree, bool shDc)
    {
        var bounds = await SplatBounds.ComputeRobustAsync(a, packed, count) ?? default;
        return new SplatClipboard(packed, sh, count, sh != null ? shDegree : 0, shDc, bounds);
    }

    /// <summary>Store the copy's colours the way the target scene does (trained SH DC vs RGB). Going to RGB drops the
    /// SH bands (base colour only).</summary>
    public async Task MatchColoursAsync(Accelerator a, bool toShDc)
    {
        if (toShDc == ColoursAreShDc) return;
        SplatRows.ConvertColours(a, Packed, Count, toShDc);
        await a.SynchronizeAsync();
        ColoursAreShDc = toShDc;
    }

    // ── Copy / Paste against the live renderer ──────────────────────────────────────────────────────────────

    /// <summary>Copy the visible splats in the volume (and their SH rows) to a new clipboard; null if none.</summary>
    public static async Task<SplatClipboard?> CopyAsync(Accelerator a, GpuGaussianRenderer renderer, SplatEditor editor,
        SplatEditor.Volume v)
    {
        var packed = renderer.PackedSplatBuffer;
        int n = renderer.SplatCount;
        if (packed == null || n <= 0) return null;
        int k = await editor.CountAsync(a, packed, n, v);
        if (k <= 0) return null;
        using var indices = await SplatRows.SelectIndicesAsync(a, packed, n, v, k);
        var clipPacked = SplatRows.GatherRows(a, packed, indices, k, SplatFormat.Floats);
        MemoryBuffer1D<float, Stride1D.Dense>[]? clipSh = null;
        if (renderer.ShDegree > 0 && renderer.ShRestBuffers is { } parts)
        {
            clipSh = new MemoryBuffer1D<float, Stride1D.Dense>[parts.Length];
            for (int p = 0; p < parts.Length; p++)
            {
                using var whole = renderer.CopyShPartToIlgpu(a, p, n);
                clipSh[p] = SplatRows.GatherRows(a, whole, indices, k, SphericalHarmonics.PartFloatsPerSplat);
                await a.SynchronizeAsync();
            }
        }
        await a.SynchronizeAsync();
        // Robust (1-99%) bounds: a few far splats caught by a screen rectangle would otherwise set its size.
        var bounds = await SplatBounds.ComputeRobustAsync(a, clipPacked, k) ?? default;
        return new SplatClipboard(clipPacked, clipSh, k, renderer.ShDegree, renderer.ColoursAreShDc, bounds);
    }

    /// <summary>
    /// Paste: the scene becomes its splats followed by the clipboard's, moved by <paramref name="offset"/>. The scene's
    /// SH bands grow with it (zero rows for a copy without SH). Returns the new splat count. Only within one colour
    /// representation (RGB vs SH DC) - pasting across them would need converting the colours.
    /// </summary>
    public async Task<int> PasteAsync(Accelerator a, GpuGaussianRenderer renderer, Vector3 offset)
    {
        var packed = renderer.PackedSplatBuffer;
        int n = renderer.SplatCount;
        if (packed == null || n <= 0) return n;
        if (ColoursAreShDc != renderer.ColoursAreShDc)
            throw new InvalidOperationException("The copy and this scene store colour differently (trained vs not).");
        var newPacked = SplatRows.AppendMoved(a, packed, n, Packed, Count, SplatFormat.Floats, offset, moveRows: true);
        await a.SynchronizeAsync();
        if (renderer.ShDegree > 0 && renderer.ShRestBuffers is { } parts)
        {
            var newParts = new SpawnDev.SpawnJS.JSObjects.GPUBuffer[parts.Length];
            for (int p = 0; p < parts.Length; p++)
            {
                using var whole = renderer.CopyShPartToIlgpu(a, p, n);
                // A copy made without SH (from an untrained scene) brings zero bands: base colour only.
                MemoryBuffer1D<float, Stride1D.Dense>? zeros = null;
                if (Sh == null)
                {
                    zeros = a.Allocate1D<float>((long)Count * SphericalHarmonics.PartFloatsPerSplat);
                    zeros.MemSetToZero();
                }
                using var grown = SplatRows.AppendMoved(a, whole, n, Sh?[p] ?? zeros!, Count, SphericalHarmonics.PartFloatsPerSplat, Vector3.Zero, moveRows: false);
                zeros?.Dispose();
                await a.SynchronizeAsync();
                newParts[p] = renderer.NewShPartFrom(grown, n + Count);
            }
            renderer.SetShRest(newParts, renderer.ShDegree);
        }
        await renderer.UploadSceneFromGpuBuffer(newPacked, n + Count);   // takes ownership of newPacked
        return n + Count;
    }

    public void Dispose()
    {
        Packed.Dispose();
        if (Sh != null) foreach (var s in Sh) s.Dispose();
    }
}
