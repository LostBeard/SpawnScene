using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Partitioned training (ScenePartition): a scene bigger than one training run's splat budget is trained as blocks -
/// each block from the seed splats in its training box, supervised by its own views - and each block keeps only the
/// splats inside its own cell, parked in OPFS until every block is done, then merged into one scene.
/// </summary>
public partial class Studio
{
    /// <summary>Training blocks as columns x rows on the ground plane; 1 x 1 trains the scene in one run.
    /// URL: &amp;blocks=2x2.</summary>
    public static (int Columns, int Rows) TrainingBlocks { get; set; } = (1, 1);

    /// <summary>Prefix for the training HUD line ("Block 2 / 4 · "), empty for a single run.</summary>
    string _trainHudPrefix = "";

    /// <summary>Seed splats sampled for the plan: positions only, a CPU transfer of at most this many rows.</summary>
    const int PlanSampleSplats = 50_000;

    /// <summary>
    /// Train the viewed scene as <see cref="TrainingBlocks"/> blocks with <see cref="TrainProjectSceneAsync"/>'s settings,
    /// then install the merged scene in the viewer as one trained scene. Returns the iterations each block ran (0 = did
    /// not train).
    /// </summary>
    private async Task<int> TrainPartitionedAsync(int iterations, int maxSplats, int maxDimension)
    {
        var scene = _sceneManager.ActiveScene;
        var project = _activeProject;
        var a = _gpuService.WebGPUAccelerator;
        if (scene == null || project == null || _gpuRenderer.PackedSplatBuffer == null || _gpuRenderer.SplatCount <= 0)
            return await TrainProjectSceneAsync(iterations, maxSplats, maxDimension);

        var allViews = scene.TrainingViews.ToList();
        int seedN = _gpuRenderer.SplatCount;
        bool seedDc = _gpuRenderer.ColoursAreShDc;
        // A private copy of the seed: uploading a block hands the renderer's buffer to the sorter, which frees it.
        using var seed = a.Allocate1D<float>((long)seedN * SplatFormat.Floats);
        seed.View.CopyFrom(_gpuRenderer.PackedSplatBuffer.View.SubView(0, (long)seedN * SplatFormat.Floats));

        var plan = ScenePartition.Make(allViews.Select(v => v.Camera).ToList(), await SampleSeedPositionsAsync(seed, seedN),
            new ScenePartition.Options(TrainingBlocks.Columns, TrainingBlocks.Rows));
        Console.WriteLine($"[Partition] {plan.Blocks.Length} blocks ({TrainingBlocks.Columns} x {TrainingBlocks.Rows}) over " +
            $"{allViews.Count} views and {seedN:N0} seed splats; up {Fmt(plan.Up)}, long axis {Fmt(plan.AxisU)}");
        foreach (var b in plan.Blocks)
            Console.WriteLine($"[Partition] block {b.Index}: {b.Views.Length} views ({b.OwnViews} inside its cell), " +
                $"{b.Points:N0} of the sampled seed points in its training box");

        await _projectService.ClearWorkFilesAsync(project.Id);
        var parked = new List<(int Block, int Count, int ShParts)>();
        int shDegree = 0, ranIters = 0;
        try
        {
            foreach (var block in plan.Blocks)
            {
                if (_trainStopRequested) break;
                _trainHudPrefix = $"Block {block.Index + 1} / {plan.Blocks.Length} · ";
                // CPU transfer: one count for the block's seed.
                var trainBox = PlaneVolume(plan, block.TrainMin, block.TrainMax);
                int m = await _splatEditor.CountAsync(a, seed, seedN, trainBox);
                int supervised = block.Views.Count(i => allViews[i].UsedForSupervision);
                if (m == 0 || supervised < 2)
                {
                    Console.WriteLine($"[Partition] block {block.Index}: skipped ({m:N0} seed splats, {supervised} supervised views)");
                    continue;
                }
                using (var idx = await SplatRows.SelectIndicesAsync(a, seed, seedN, trainBox, m))
                {
                    var blockSeed = SplatRows.GatherRows(a, seed, idx, m, SplatFormat.Floats);
                    await a.SynchronizeAsync();
                    _gpuRenderer.SetShRest(null, 0);
                    _gpuRenderer.ColoursAreShDc = seedDc;
                    await _gpuRenderer.UploadSceneFromGpuBuffer(blockSeed, m);   // the sorter owns blockSeed now
                }
                scene.TrainingViews = block.Views.Select(i => allViews[i]).ToList();
                Console.WriteLine($"[Partition] block {block.Index}: training {m:N0} seed splats against {scene.TrainingViews.Count} views");

                int it = await TrainProjectSceneAsync(iterations, maxSplats, maxDimension);
                if (it == 0) { Console.WriteLine($"[Partition] block {block.Index}: FAIL - training did not run"); return 0; }
                ranIters = Math.Max(ranIters, it);
                shDegree = Math.Max(shDegree, _gpuRenderer.ShDegree);

                var (kept, shParts) = await ParkBlockCoreAsync(project.Id, plan, block);
                parked.Add((block.Index, kept, shParts));
            }
        }
        finally
        {
            scene.TrainingViews = allViews;
            _trainHudPrefix = "";
        }

        if (parked.Count == 0) { Console.WriteLine("[Partition] FAIL: no block trained"); return 0; }
        await MergeParkedBlocksAsync(project.Id, parked, shDegree);
        await _projectService.ClearWorkFilesAsync(project.Id);

        // A measurement run (&llffhold) scores the merged scene against every view in one pass - the blocks' own
        // held-out lines score parts of it. 0 iterations: the same setup and test-time pose alignment as a trained
        // run's final score, so "[Train] trainer HELD OUT" compares with a single run's.
        if (allViews.Any(v => !v.UsedForSupervision))
        {
            Console.WriteLine("[Partition] scoring the merged scene against every view");
            _trainHudPrefix = "Scoring · ";
            try { await TrainProjectSceneAsync(0, maxSplats, maxDimension); }
            finally { _trainHudPrefix = ""; }
        }
        return ranIters;
    }

    /// <summary>
    /// The trained block on screen: keep the visible splats inside the block's own cell, read them (and their SH parts)
    /// to JS memory and park them in the project's work files. Returns the rows kept and the SH parts written.
    /// </summary>
    async Task<(int Kept, int ShParts)> ParkBlockCoreAsync(string projectId, ScenePartition.Plan plan, ScenePartition.Block block)
    {
        var a = _gpuService.WebGPUAccelerator;
        var packed = _gpuRenderer.PackedSplatBuffer!;
        int n = _gpuRenderer.SplatCount;
        var core = PlaneVolume(plan, block.CoreMin, block.CoreMax);
        int k = await _splatEditor.CountAsync(a, packed, n, core);
        Console.WriteLine($"[Partition] block {block.Index}: {n:N0} splats trained, {k:N0} inside its cell kept");
        if (k <= 0) return (0, 0);

        using var idx = await SplatRows.SelectIndicesAsync(a, packed, n, core, k);
        using (var rows = SplatRows.GatherRows(a, packed, idx, k, SplatFormat.Floats))
        {
            await a.SynchronizeAsync();
            // CPU transfer: file I/O - the block's rows to OPFS through JS memory, never the .NET heap.
            using var u8 = await rows.CopyToHostUint8ArrayAsync(0, (long)k * SplatFormat.Floats * sizeof(float));
            await _projectService.WriteWorkFileAsync(projectId, $"block{block.Index}.packed.bin", u8);
        }
        int parts = 0;
        if (_gpuRenderer.ShDegree > 0 && _gpuRenderer.ShRestBuffers is { } sh)
        {
            for (int p = 0; p < sh.Length; p++)
            {
                using var whole = _gpuRenderer.CopyShPartToIlgpu(a, p, n);
                using var rows = SplatRows.GatherRows(a, whole, idx, k, SphericalHarmonics.PartFloatsPerSplat);
                await a.SynchronizeAsync();
                // CPU transfer: file I/O, as above.
                using var u8 = await rows.CopyToHostUint8ArrayAsync(0, (long)k * SphericalHarmonics.PartFloatsPerSplat * sizeof(float));
                await _projectService.WriteWorkFileAsync(projectId, $"block{block.Index}.sh{p}.bin", u8);
            }
            parts = sh.Length;
        }
        return (k, parts);
    }

    /// <summary>
    /// Read the parked blocks back and install them in the viewer as one trained scene: packed rows written one block
    /// after another into one buffer, SH parts likewise (blocks without SH, or with fewer bands, get zero bands).
    /// </summary>
    async Task MergeParkedBlocksAsync(string projectId, List<(int Block, int Count, int ShParts)> parked, int shDegree)
    {
        var a = _gpuService.WebGPUAccelerator;
        long total = parked.Sum(p => (long)p.Count);
        if (total > int.MaxValue) throw new InvalidOperationException($"{total:N0} splats is more than one scene can index");
        MemoryBuffer1D<float, Stride1D.Dense>? merged = a.Allocate1D<float>(Math.Max(1L, total * SplatFormat.Floats));
        bool withSh = shDegree > 0 && parked.All(p => p.ShParts == SphericalHarmonics.Parts);
        var shBlocks = new List<ArrayBuffer[]>();
        try
        {
            long row = 0;
            foreach (var (block, count, _) in parked)
            {
                if (count <= 0) continue;
                using (var bytes = await _projectService.ReadWorkFileAsync(projectId, $"block{block}.packed.bin")
                    ?? throw new InvalidOperationException($"block {block}'s parked rows are missing"))
                    _gpuRenderer.WriteIlgpuBytes(merged, row * SplatFormat.Floats * sizeof(float), bytes);
                if (withSh)
                {
                    var parts = new ArrayBuffer[SphericalHarmonics.Parts];
                    for (int p = 0; p < parts.Length; p++)
                        parts[p] = await _projectService.ReadWorkFileAsync(projectId, $"block{block}.sh{p}.bin")
                            ?? throw new InvalidOperationException($"block {block}'s SH part {p} is missing");
                    shBlocks.Add(parts);
                }
                row += count;
            }
            await a.SynchronizeAsync();
            _gpuRenderer.ColoursAreShDc = true;   // every block was trained
            await _gpuRenderer.UploadSceneFromGpuBuffer(merged, (int)total);   // the sorter owns merged now
            merged = null;
            if (withSh) _gpuRenderer.LoadShRestBlocks(shBlocks, shDegree);
            else _gpuRenderer.SetShRest(null, 0);
            if (ViewerShDegreeCap is int cap) _gpuRenderer.CapShDegree(cap);
            _gpuRenderer.RepackForDisplay();
            Console.WriteLine($"[Partition] merged {parked.Count} blocks: {total:N0} splats, SH degree {_gpuRenderer.ShDegree}");
        }
        finally
        {
            merged?.Dispose();
            foreach (var parts in shBlocks) foreach (var p in parts) p.Dispose();
        }
    }

    /// <summary>
    /// Up to <see cref="PlanSampleSplats"/> seed positions, strided over the rows (gathered on the GPU first, so the
    /// readback is the sample, not the scene).
    /// </summary>
    async Task<Vector3[]> SampleSeedPositionsAsync(MemoryBuffer1D<float, Stride1D.Dense> seed, int n)
    {
        var a = _gpuService.WebGPUAccelerator;
        int k = Math.Min(n, PlanSampleSplats);
        var picks = new int[k];
        for (int i = 0; i < k; i++) picks[i] = (int)((long)i * n / k);
        using var idx = a.Allocate1D(picks);
        using var rows = SplatRows.GatherRows(a, seed, idx, k, SplatFormat.Floats);
        await a.SynchronizeAsync();
        // CPU transfer: the plan's sample, k x 14 floats (2.8 MB at most).
        var f = await rows.CopyToHostAsync<float>(0, (long)k * SplatFormat.Floats);
        var pts = new Vector3[k];
        for (int i = 0; i < k; i++) pts[i] = new Vector3(f[i * SplatFormat.Floats], f[i * SplatFormat.Floats + 1], f[i * SplatFormat.Floats + 2]);
        return pts;
    }

    /// <summary>
    /// A SplatEditor volume for a plane rectangle [min, max] (ScenePartition plane coordinates), all heights: world to
    /// (u, v, height) in the row-vector form SplatEditor.Inside uses. Open edges become +-1e30 (finite, so no infinity
    /// reaches a shader). Inside is inclusive at both ends where Block.Owns is half-open, so a splat exactly on a seam
    /// is kept by both neighbours: a duplicate, never a hole.
    /// </summary>
    static SplatEditor.Volume PlaneVolume(ScenePartition.Plan plan, Vector2 min, Vector2 max)
    {
        static float Clamp(float v) => float.IsNegativeInfinity(v) ? -1e30f : float.IsPositiveInfinity(v) ? 1e30f : v;
        var u = plan.AxisU; var v = plan.AxisV; var h = plan.Up; var o = plan.Origin;
        var m = new Matrix4x4(
            u.X, v.X, h.X, 0f,
            u.Y, v.Y, h.Y, 0f,
            u.Z, v.Z, h.Z, 0f,
            -Vector3.Dot(o, u), -Vector3.Dot(o, v), -Vector3.Dot(o, h), 1f);
        return SplatEditor.Volume.From(m, Clamp(min.X), Clamp(max.X), Clamp(min.Y), Clamp(max.Y), -1e30f, 1e30f);
    }

    static string Fmt(Vector3 v) => $"({v.X:F2}, {v.Y:F2}, {v.Z:F2})";
}
