using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Level-of-detail viewing (Plans/lod-streaming.md, phase A): a loaded scene is turned into an LOD tree on the GPU
/// (GpuLodTree) and drawn through its cut (GpuSplatSorter's LOD cull), so the splats drawn each frame depend on the
/// view, not on the scene's size. URL: &amp;lodtau=1.5 (pixels; 0 = off), &amp;lodbudget=N (splats a frame; tau follows it).
/// </summary>
public partial class Studio
{
    /// <summary>LOD cut threshold in pixels for a loaded scene; 0 draws the scene as it is.</summary>
    public static float LodTauOption { get; set; }

    /// <summary>Splats a frame the LOD cut is steered to (0 = the fixed <see cref="LodTauOption"/>).</summary>
    public static int LodBudgetOption { get; set; }

    GpuRadixSort? _lodSort;

    /// <summary>Build the LOD tree of the scene on screen and draw it through the cut at <see cref="LodTauOption"/>.</summary>
    async Task InstallLodTreeAsync()
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (packed == null || n < 2 || LodTauOption <= 0f) return;
        var a = _gpuService.WebGPUAccelerator;
        var t0 = DateTime.UtcNow;

        // The first level's cell: the median splat size, from a strided sample (CPU transfer: <= 10K x 3 floats).
        int k = Math.Min(n, 10_000);
        var picks = new int[k];
        for (int i = 0; i < k; i++) picks[i] = (int)((long)i * n / k);
        float baseStep;
        using (var idx = a.Allocate1D(picks))
        using (var rows = SplatRows.GatherRows(a, packed, idx, k, SplatFormat.Floats))
        {
            await a.SynchronizeAsync();
            var f = await rows.CopyToHostAsync<float>(0, (long)k * SplatFormat.Floats);
            var sizes = new float[k];
            for (int i = 0; i < k; i++) sizes[i] = LodTree.SizeOf(f.AsSpan(i * SplatFormat.Floats, SplatFormat.Floats));
            Array.Sort(sizes);
            baseStep = Math.Max(sizes[k / 2], 1e-6f);
        }

        var device = a.NativeAccelerator.NativeDevice!;
        var queue = a.NativeAccelerator.Queue!;
        _lodSort ??= new GpuRadixSort(device, queue, a);
        var tree = await GpuLodTree.BuildAsync(a, packed, n, baseStep, (keys, values, count) =>
        {
            _lodSort.EnsureCapacity(count);
            _lodSort.Sort(keys.GetGPUBuffer()!, values.GetGPUBuffer()!, count, 32);
        });
        Console.WriteLine($"[LOD] tree built in {(DateTime.UtcNow - t0).TotalSeconds:F1}s: {n:N0} leaves -> {tree.NodeCount:N0} nodes, " +
            $"{tree.Levels} levels, base cell {baseStep:G3}");
        _gpuRenderer.LodBudget = LodBudgetOption;
        await _gpuRenderer.InstallLodAsync(tree, LodTauOption);
        tree.Dispose();   // what the renderer did not take (child lists, counters)
    }
}
