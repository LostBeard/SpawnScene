using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Partitioned training (ScenePartition): a scene bigger than one training run's splat budget is trained as blocks,
/// each supervised by its own views, and each block keeps only the splats inside its own cell, parked in OPFS until
/// every block is done, then merged into one scene.
/// <para>
/// Blocks start from ONE coarse model of the whole scene (CityGaussian's coarse-then-refine): the coarse run trains
/// the seed at <see cref="PartitionCoarseFraction"/> of the iterations and half the splat budget, then each block
/// refines it with every splat outside its training box frozen (SplatTrainerGpu.TrainableVolume) - the rest of the
/// scene still explains its own pixels, and the block's budget grows only inside its box. Trained from the seed
/// alone, each block re-solved the whole scene its own way and the merged quadrants disagreed at the seams: TruckFull
/// 7K 2x2 23.34 dB held out against 23.73 for one run, while every block alone scored 23.72 (2026-10-05).
/// </para>
/// </summary>
public partial class Studio
{
    /// <summary>Training blocks as columns x rows on the ground plane; 1 x 1 trains the scene in one run.
    /// URL: &amp;blocks=2x2.</summary>
    public static (int Columns, int Rows) TrainingBlocks { get; set; } = (1, 1);

    /// <summary>Share of the iterations the coarse whole-scene model gets before the blocks refine it; 0 = no coarse
    /// stage (each block trains from the seed, the first version). URL: &amp;coarse=0.5.</summary>
    public static float PartitionCoarseFraction { get; set; } = 0.5f;

    /// <summary>Diagnosis (&amp;frozendiag=1): the first densify step that drops frozen context reads both sides back and
    /// logs which splats went and why.</summary>
    public static bool DiagnoseFrozenDensify { get; set; }

    /// <summary>The training box of the block being refined: TrainOnTrainingViewsAsync hands it to the trainer
    /// (SplatTrainerGpu.TrainableVolume), so everything outside it is frozen context. Null outside a block.</summary>
    SplatEditor.Volume? _frozenOutside;

    /// <summary>The cell of the block being refined: densification grows only inside it (SplatTrainerGpu.GrowOnlyInside).
    /// Null outside a block.</summary>
    SplatEditor.Volume? _growOnlyInside;

    /// <summary>
    /// The partitioned run's global clock: iterations already trained before this stage (a block refining the coarse
    /// model) and the whole run's length. TrainOnTrainingViewsAsync runs every iteration schedule - SH degree, position
    /// learning rate, densify window, opacity reset - on offset + iteration out of the total, so coarse + blocks follow
    /// exactly a single run's schedule. Restarted per stage, the coarse model (3,500 of 7,000) never got the reset a
    /// single run gets at 3,000 and each block restarted the position rate at its initial value. 0 = a normal run.
    /// </summary>
    int _scheduleOffset, _scheduleTotal;

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
        int coarseIters = (int)(iterations * Math.Clamp(PartitionCoarseFraction, 0f, 0.9f));
        int blockIters = iterations - coarseIters;
        // The coarse model: packed rows and SH parts, kept on the GPU while the blocks refine copies of it.
        MemoryBuffer1D<float, Stride1D.Dense>? coarse = null;
        MemoryBuffer1D<float, Stride1D.Dense>[]? coarseSh = null;
        int coarseN = 0, coarseDegree = 0, coarseVisible = 0, frozenBefore = -1;
        try
        {
            if (coarseIters > 0)
            {
                _trainHudPrefix = "Coarse model · ";
                Console.WriteLine($"[Partition] coarse model: {coarseIters:N0} iterations over every view, up to {maxSplats / 2:N0} splats; " +
                    $"then {blockIters:N0} per block");
                int ci;
                _scheduleOffset = 0; _scheduleTotal = iterations;
                try { ci = await TrainProjectSceneAsync(coarseIters, maxSplats / 2, maxDimension); }
                finally { _scheduleTotal = 0; }
                if (ci == 0) { Console.WriteLine("[Partition] FAIL - the coarse model did not train"); return 0; }
                coarseN = _gpuRenderer.SplatCount;
                coarseDegree = _gpuRenderer.ShDegree;
                coarse = a.Allocate1D<float>((long)coarseN * SplatFormat.Floats);
                coarse.View.CopyFrom(_gpuRenderer.PackedSplatBuffer!.View.SubView(0, (long)coarseN * SplatFormat.Floats));
                if (coarseDegree > 0 && _gpuRenderer.ShRestBuffers != null)
                    coarseSh = Enumerable.Range(0, SphericalHarmonics.Parts).Select(part => _gpuRenderer.CopyShPartToIlgpu(a, part, coarseN)).ToArray();
                await a.SynchronizeAsync();
                coarseVisible = await _splatEditor.CountAsync(a, coarse, coarseN, SplatEditor.Volume.Rows(0, coarseN));
                Console.WriteLine($"[Partition] coarse model: {coarseN:N0} splats ({coarseVisible:N0} visible), SH degree {coarseDegree}");
            }

            foreach (var block in plan.Blocks)
            {
                if (_trainStopRequested) break;
                _trainHudPrefix = $"Block {block.Index + 1} / {plan.Blocks.Length} · ";
                var trainBox = PlaneVolume(plan, block.TrainMin, block.TrainMax);
                int supervised = block.Views.Count(i => allViews[i].UsedForSupervision);
                if (coarse != null)
                {
                    // CPU transfer: one count, for the log and the skip test.
                    int inBox = await _splatEditor.CountAsync(a, coarse, coarseN, trainBox);
                    frozenBefore = coarseVisible - inBox;
                    if (inBox == 0 || supervised < 2)
                    {
                        Console.WriteLine($"[Partition] block {block.Index}: skipped ({inBox:N0} coarse splats in its box, {supervised} supervised views)");
                        continue;
                    }
                    var copy = a.Allocate1D<float>((long)coarseN * SplatFormat.Floats);
                    copy.View.CopyFrom(coarse.View);
                    await a.SynchronizeAsync();
                    _gpuRenderer.ColoursAreShDc = true;
                    await _gpuRenderer.UploadSceneFromGpuBuffer(copy, coarseN);   // the sorter owns copy now
                    if (coarseSh != null)
                        _gpuRenderer.SetShRest(coarseSh.Select(part => _gpuRenderer.NewShPartFrom(part, coarseN)).ToArray(), coarseDegree);
                    else _gpuRenderer.SetShRest(null, 0);
                    Console.WriteLine($"[Partition] block {block.Index}: refining {inBox:N0} of the coarse model's {coarseN:N0} splats " +
                        $"(the rest frozen) against {block.Views.Length} views");
                }
                else
                {
                    // CPU transfer: one count for the block's seed.
                    int m = await _splatEditor.CountAsync(a, seed, seedN, trainBox);
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
                    Console.WriteLine($"[Partition] block {block.Index}: training {m:N0} seed splats against {block.Views.Length} views");
                }
                scene.TrainingViews = block.Views.Select(i => allViews[i]).ToList();

                int it;
                _frozenOutside = coarse != null ? trainBox : null;
                _growOnlyInside = coarse != null ? PlaneVolume(plan, block.CoreMin, block.CoreMax) : null;
                _scheduleOffset = coarse != null ? coarseIters : 0;
                _scheduleTotal = coarse != null ? iterations : 0;
                try { it = await TrainProjectSceneAsync(blockIters, maxSplats, maxDimension); }
                finally { _frozenOutside = null; _growOnlyInside = null; _scheduleOffset = 0; _scheduleTotal = 0; }
                if (it == 0) { Console.WriteLine($"[Partition] block {block.Index}: FAIL - training did not run"); return 0; }
                ranIters = Math.Max(ranIters, coarseIters + it);
                if (coarse != null)
                {
                    // The frozen context must come out as it went in: nothing outside the box pruned (only clones that
                    // drifted out of it can add). CPU transfer: two counts.
                    var trained = _gpuRenderer.PackedSplatBuffer!;
                    int n2 = _gpuRenderer.SplatCount;
                    int visible2 = await _splatEditor.CountAsync(a, trained, n2, SplatEditor.Volume.Rows(0, n2));
                    int frozenAfter = visible2 - await _splatEditor.CountAsync(a, trained, n2, trainBox);
                    Console.WriteLine($"[Partition] block {block.Index}: frozen context {frozenBefore:N0} visible splats before, " +
                        $"{frozenAfter:N0} after{(frozenAfter < frozenBefore ? " - LOST CONTEXT" : "")}");
                }
                shDegree = Math.Max(shDegree, _gpuRenderer.ShDegree);

                var (kept, shParts) = await ParkBlockCoreAsync(project.Id, plan, block);
                parked.Add((block.Index, kept, shParts));
            }
        }
        finally
        {
            scene.TrainingViews = allViews;
            _trainHudPrefix = "";
            coarse?.Dispose();
            if (coarseSh != null) foreach (var part in coarseSh) part.Dispose();
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
