using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// The scene as its LOD tree in a file (.spawnscene v3, <see cref="LodChunkFile"/>; Plans/lod-streaming.md phase B):
/// export builds the tree on the GPU, lays it out breadth-first and writes it in gzipped 64K-node chunks; open reads
/// them back into the GPU and draws through the cut. Opening loads every chunk for now; paging comes next.
/// </summary>
public partial class Studio
{
    /// <summary>Node-ordered SH of the open v3 file: the renderer copied them, kept until the next open.</summary>
    MemoryBuffer1D<float, Stride1D.Dense>[]? _lodFileSh;

    /// <summary>Write the scene on screen as a .spawnscene v3 (LOD tree, chunked) download.</summary>
    async Task ExportLodSceneFileAsync()
    {
        var a = _gpuService.WebGPUAccelerator;
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (packed == null || n < 2) { _editNote = "Nothing to export"; return; }
        var owned = new List<IDisposable>();
        var js = new List<IDisposable>();
        try
        {
            var t0 = DateTime.UtcNow;
            float baseStep = await LodBaseStepAsync(a, packed, n);
            GpuLodTree laid;
            MemoryBuffer1D<int, Stride1D.Dense>? order = null;
            using (var tree = await GpuLodTree.BuildAsync(a, packed, n, baseStep, LodSortPairs()))
                laid = await GpuLodLayout.BuildAsync(a, tree, o => order = o);
            owned.Add(laid); owned.Add(order!);
            int nodes = laid.NodeCount;

            // SH for every node, in the new order: the leaves' own, zero for merges.
            bool withSh = _gpuRenderer.ShDegree > 0 && _gpuRenderer.ShRestBuffers != null;
            MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
            if (withSh)
            {
                sh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
                for (int p = 0; p < sh.Length; p++)
                {
                    using var whole = _gpuRenderer.CopyShPartToIlgpu(a, p, n);
                    sh[p] = GpuLodLayout.GatherLeafRows(a, whole, order!, n, nodes, SphericalHarmonics.PartFloatsPerSplat);
                    owned.Add(sh[p]);
                    await a.SynchronizeAsync();   // the gather has read `whole` before it goes
                }
            }

            const int Chunk = LodChunkFile.DefaultChunkNodes;
            var chunks = new List<LodChunkFile.Chunk>();
            var zipped = new List<Uint8Array>();
            long offset = 0, rawTotal = 0;
            int roots = 0;
            for (int first = 0; first < nodes; first += Chunk)
            {
                int count = Math.Min(Chunk, nodes - first);
                var part = new List<IDisposable>();
                try
                {
                    // The chunk's rows (and SH) on their own, for a codec frame of their own.
                    var rows = a.Allocate1D<float>((long)count * SplatFormat.Floats); part.Add(rows);
                    rows.View.CopyFrom(laid.Nodes.View.SubView((long)first * SplatFormat.Floats, (long)count * SplatFormat.Floats));
                    MemoryBuffer1D<float, Stride1D.Dense>[]? shRows = null;
                    if (sh != null)
                    {
                        shRows = new MemoryBuffer1D<float, Stride1D.Dense>[sh.Length];
                        for (int p = 0; p < sh.Length; p++)
                        {
                            shRows[p] = a.Allocate1D<float>((long)count * SphericalHarmonics.PartFloatsPerSplat); part.Add(shRows[p]);
                            shRows[p].View.CopyFrom(sh[p].View.SubView((long)first * SphericalHarmonics.PartFloatsPerSplat,
                                (long)count * SphericalHarmonics.PartFloatsPerSplat));
                        }
                    }
                    var frame = await ChunkFrameAsync(a, rows, count);
                    var (geo, app, shq) = SceneCodec.Encode(a, rows, shRows, frame);
                    part.Add(geo); part.Add(app); part.Add(shq);
                    await a.SynchronizeAsync();

                    // CPU transfer: file I/O - the chunk's streams, then gzipped by the browser.
                    var raw = new List<Uint8Array>
                    {
                        await geo.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.GeoWords * 4),
                        await app.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.AppWords * 4),
                    };
                    if (shRows != null) raw.Add(await shq.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.ShWords * 4));
                    raw.Add(await laid.Parent.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                    raw.Add(await laid.ChildCount.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                    raw.Add(await laid.Bounds.CopyToHostUint8ArrayAsync((long)first * 16, (long)count * 16));
                    raw.Add(await laid.LodSize.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                    js.AddRange(raw);
                    if (first == 0)
                    {
                        // Breadth-first: the roots are the first nodes, the ones with no parent.
                        using var parents = new Int32Array(raw[shRows != null ? 3 : 2].Buffer, raw[shRows != null ? 3 : 2].ByteOffset, count);
                        var p0 = parents.ToArray();
                        while (roots < count && p0[roots] < 0) roots++;
                    }
                    rawTotal += raw.Sum(r => (long)r.ByteLength);
                    using var blobIn = new Blob(raw, new BlobOptions { Type = "application/octet-stream" });
                    var z = await GzipAsync(blobIn, decompress: false);
                    js.Add(z);
                    var zu = new Uint8Array(z); js.Add(zu);
                    zipped.Add(zu);
                    chunks.Add(new LodChunkFile.Chunk(first, count, offset, z.ByteLength,
                        new[] { frame.MinX - frame.TailLoX, frame.MinY - frame.TailLoY, frame.MinZ - frame.TailLoZ,
                                frame.MinX + frame.SizeX + frame.TailHiX, frame.MinY + frame.SizeY + frame.TailHiY, frame.MinZ + frame.SizeZ + frame.TailHiZ },
                        frame.InnerArray()));
                    offset += z.ByteLength;
                }
                finally { foreach (var d in part) d.Dispose(); }
            }

            string name = _activeProject?.Name ?? "SpawnScene scene";
            var c = _sceneManager.Camera;
            var header = new LodChunkFile.Header3(name, nodes, n, roots, _gpuRenderer.ColoursAreShDc, withSh ? _gpuRenderer.ShDegree : 0,
                _viewedProjectScene?.TrainedIterations ?? 0, DateTime.UtcNow,
                new[] { c.Position.X, c.Position.Y, c.Position.Z, c.Forward.X, c.Forward.Y, c.Forward.Z, c.Up.X, c.Up.Y, c.Up.Z },
                Chunk, chunks.ToArray());
            var prefix = new Uint8Array(LodChunkFile.Prefix3(header)); js.Add(prefix);
            var fileParts = new List<Uint8Array> { prefix };
            fileParts.AddRange(zipped);
            using var blob = new Blob(fileParts, new BlobOptions { Type = "application/octet-stream" });
            DownloadSceneBlob(blob, name);
            _editNote = $"Exported {n:N0} splats as an LOD tree ({blob.Size / (1024 * 1024)} MB)";
            Console.WriteLine($"[Edit] exported '{name}' as an LOD tree in {(DateTime.UtcNow - t0).TotalSeconds:F1}s: {n:N0} splats, " +
                $"{nodes:N0} nodes ({roots} roots), {chunks.Count} chunks, {blob.Size / (1024 * 1024)} MB " +
                $"(v3; {rawTotal / (1024 * 1024)} MB before gzip)");
        }
        catch (Exception ex) { _editNote = "Export failed"; Console.WriteLine($"[Edit] LOD export failed: {ex.Message}"); }
        finally
        {
            foreach (var d in js) d.Dispose();
            foreach (var d in owned) d.Dispose();
        }
    }

    /// <summary>
    /// A chunk's codec frame, as the v2 export frames a scene: linear over the box holding all but 0.5% at each end
    /// of each axis, padded by a quarter, log tails out to the full bounds.
    /// </summary>
    static async Task<SceneCodec.Frame> ChunkFrameAsync(SpawnDev.ILGPU.WebGPU.WebGPUAccelerator a, MemoryBuffer1D<float, Stride1D.Dense> rows, int count)
    {
        var box = await SplatBounds.ComputeAsync(a, rows, count) ?? new SplatBounds.Aabb(0, 0, 0, 1, 1, 1);
        var robust = await SplatBounds.ComputeRobustAsync(a, rows, count, 0.005) ?? box;
        float px = 0.25f * (robust.MaxX - robust.MinX), py = 0.25f * (robust.MaxY - robust.MinY), pz = 0.25f * (robust.MaxZ - robust.MinZ);
        var inner = new SplatBounds.Aabb(robust.MinX - px, robust.MinY - py, robust.MinZ - pz, robust.MaxX + px, robust.MaxY + py, robust.MaxZ + pz);
        return SceneCodec.Frame.From(box, inner, count);
    }

    /// <summary>
    /// Open a .spawnscene v3: every chunk gunzipped, decoded on the GPU into its place in the tree's arrays, and the
    /// tree drawn through its cut (at &amp;lodtau / &amp;lodbudget, else 1.5 px). <paramref name="dataStart"/> is where
    /// the chunks begin in <paramref name="bytes"/>.
    /// </summary>
    async Task OpenLodFileAsync(ArrayBuffer bytes, LodChunkFile.Header3 h, long dataStart)
    {
        var a = _gpuService.WebGPUAccelerator;
        var t0 = DateTime.UtcNow;
        int nodes = h.NodeCount;
        bool withSh = h.ShDegree > 0;
        if (_lodFileSh != null) { foreach (var b in _lodFileSh) b.Dispose(); _lodFileSh = null; }
        var laid = new GpuLodTree { LeafCount = h.LeafCount, NodeCount = nodes };
        laid.Nodes = a.Allocate1D<float>((long)nodes * SplatFormat.Floats);
        laid.Parent = a.Allocate1D<int>(nodes);
        laid.Bounds = a.Allocate1D<float>((long)nodes * 4);
        laid.LodSize = a.Allocate1D<float>(nodes);
        laid.ChildCount = a.Allocate1D<int>(nodes);
        laid.FirstChild = a.Allocate1D<int>(1);
        laid.ChildList = a.Allocate1D<int>(1);
        var sh = withSh
            ? Enumerable.Range(0, SphericalHarmonics.Parts).Select(_ => a.Allocate1D<float>((long)nodes * SphericalHarmonics.PartFloatsPerSplat)).ToArray()
            : null;
        try
        {
            foreach (var c in h.Chunks)
            {
                using var zipped = new Uint8Array(bytes, dataStart + c.Offset, c.Bytes);
                using var blobIn = new Blob(new[] { zipped }, new BlobOptions { Type = "application/octet-stream" });
                using var raw = await GzipAsync(blobIn, decompress: true);
                var L = LodChunkFile.Layout.Of(c.Count, withSh);
                if (raw.ByteLength != L.End) throw new InvalidDataException($"chunk at node {c.First}: {raw.ByteLength} bytes, expected {L.End}");
                using var geo = a.Allocate1D<uint>((long)c.Count * SceneCodec.GeoWords);
                using var app = a.Allocate1D<uint>((long)c.Count * SceneCodec.AppWords);
                using var shw = withSh ? a.Allocate1D<uint>((long)c.Count * SceneCodec.ShWords) : null;
                _gpuRenderer.WriteIlgpu(geo, 0, raw, (int)L.Geo, L.App - L.Geo);
                _gpuRenderer.WriteIlgpu(app, 0, raw, (int)L.App, L.Sh - L.App);
                if (shw != null) _gpuRenderer.WriteIlgpu(shw, 0, raw, (int)L.Sh, L.Parent - L.Sh);
                _gpuRenderer.WriteIlgpu(laid.Parent, (long)c.First * 4, raw, (int)L.Parent, L.ChildCount - L.Parent);
                _gpuRenderer.WriteIlgpu(laid.ChildCount, (long)c.First * 4, raw, (int)L.ChildCount, L.Bounds - L.ChildCount);
                _gpuRenderer.WriteIlgpu(laid.Bounds, (long)c.First * 16, raw, (int)L.Bounds, L.LodSize - L.Bounds);
                _gpuRenderer.WriteIlgpu(laid.LodSize, (long)c.First * 4, raw, (int)L.LodSize, L.End - L.LodSize);
                var (rows, rowsSh) = SceneCodec.Decode(a, geo, app, shw, LodChunkFile.FrameOf(c));
                try
                {
                    laid.Nodes.View.SubView((long)c.First * SplatFormat.Floats, (long)c.Count * SplatFormat.Floats).CopyFrom(rows.View);
                    if (sh != null && rowsSh != null)
                        for (int p = 0; p < sh.Length; p++)
                            sh[p].View.SubView((long)c.First * SphericalHarmonics.PartFloatsPerSplat, (long)c.Count * SphericalHarmonics.PartFloatsPerSplat)
                                .CopyFrom(rowsSh[p].View);
                    await a.SynchronizeAsync();
                }
                finally
                {
                    rows.Dispose();
                    if (rowsSh != null) foreach (var p in rowsSh) p.Dispose();
                }
            }
            Console.WriteLine($"[Import] '{h.Name}' (v3): {h.LeafCount:N0} splats, {nodes:N0} LOD nodes in {h.Chunks.Length} chunks, " +
                $"decoded in {(DateTime.UtcNow - t0).TotalSeconds:F1}s - opening");

            _gpuRenderer.UseRgbColours();
            _gpuRenderer.ColoursAreShDc = h.ColoursAreShDc;
            _gpuRenderer.LodBudget = LodBudgetOption;
            await _gpuRenderer.InstallLaidLodAsync(laid, sh, h.ShDegree, LodTauOption > 0f ? LodTauOption : 1.5f);
            _lodFileSh = sh;
            sh = null;

            var gaussianScene = new GaussianScene { GpuSplatCount = h.LeafCount, SourceName = "lod-file" };
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
            _renderService.SetActiveSceneGpuLoaded(gaussianScene);
            _sceneManager.ActiveScene = gaussianScene;
            _statusMessage = null;
            _state = StudioState.SceneViewer;
            _cameraController?.FitToScene();
            BuildViewerHudUI();
            await SeatAtHomeViewAsync(new ProjectScene { HomeView = h.HomeView, TrainedIterations = h.TrainedIterations, SplatCount = h.LeafCount });
        }
        finally
        {
            laid.Dispose();   // what the renderer did not take (child counts, stand-ins)
            if (sh != null) foreach (var b in sh) b.Dispose();
        }
    }
}
