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

    /// <summary>&amp;lodpool=N: stream a v3 file through a pool of about N node slots (0 = load every chunk).</summary>
    public static int LodPoolOption { get; set; }

    GpuLodPager? _lodPager;
    Blob? _lodFileBlob;

    /// <summary>
    /// Open a .spawnscene v3 STREAMED (GpuLodPager): chunk 0 at once, then the chunks the cut asks for, through a pool
    /// of <see cref="LodPoolOption"/> slots. The chunks come from a Blob of the file's data (here the fetched bytes; a
    /// slice of it is what an HTTP Range read will return).
    /// </summary>
    async Task OpenLodStreamAsync(ArrayBuffer bytes, LodChunkFile.Header3 h, long dataStart)
    {
        _lodFileBlob?.Dispose();
        using (var data = new Uint8Array(bytes, dataStart, bytes.ByteLength - dataStart))
            _lodFileBlob = new Blob(new[] { data }, new BlobOptions { Type = "application/octet-stream" });
        var blob = _lodFileBlob;
        async Task<ArrayBuffer> ChunkBytes(LodChunkFile.Chunk c)
        {
            using var slice = blob.Slice(c.Offset, c.Offset + c.Bytes);
            return await GzipAsync(slice, decompress: true);
        }
        await OpenLodStreamAsync(h, ChunkBytes, "in memory");
    }

    /// <summary>
    /// ?import=&lt;url&gt;&amp;lodpool=N on a .spawnscene v3: read only its header, then each chunk by an HTTP Range request
    /// when the cut asks for it - nothing else of the file is downloaded. False when the server does not answer Range
    /// requests or the file is not v3 (the import then downloads it whole).
    /// </summary>
    async Task<bool> TryOpenLodUrlStreamAsync(string url)
    {
        async Task<ArrayBuffer?> RangeAsync(long start, long count)
        {
            using var window = _js.Get<Window>("window");
            using var response = await window.Fetch(url, new SpawnDev.SpawnJS.FetchOptions
            {
                Headers = new Dictionary<string, string> { ["Range"] = $"bytes={start}-{start + count - 1}" },
            });
            if (response.Status != 206) return null;   // a whole-file answer: no Range support
            return await response.ArrayBuffer();
        }
        using var prefix = await RangeAsync(0, 12);
        if (prefix == null) return false;
        byte[] first;
        using (var u = new Uint8Array(prefix)) first = u.ReadBytes();
        if (SceneFile.Version(first) != 3) return false;
        int headerLen = SceneFile.HeaderLength(first);
        using var headerBytes = await RangeAsync(12, headerLen);
        if (headerBytes == null) return false;
        LodChunkFile.Header3 h;
        using (var u = new Uint8Array(headerBytes)) h = LodChunkFile.ParseHeader3(u.ReadBytes());
        long dataStart = SceneFile.DataOffset(headerLen);
        long fetched = 0;
        async Task<ArrayBuffer> ChunkBytes(LodChunkFile.Chunk c)
        {
            using var zipped = await RangeAsync(dataStart + c.Offset, c.Bytes)
                ?? throw new InvalidDataException("the server stopped answering Range requests");
            fetched += c.Bytes;
            using var blob = new Blob(new[] { zipped }, new BlobOptions { Type = "application/octet-stream" });
            return await GzipAsync(blob, decompress: true);
        }
        await OpenLodStreamAsync(h, ChunkBytes, "HTTP Range");
        Console.WriteLine($"[Import] streamed by Range: header {12 + headerLen:N0} bytes, chunk 0 {h.Chunks[0].Bytes / 1024:N0} KB " +
            $"of a {(dataStart + h.Chunks.Sum(c => c.Bytes)) / (1024 * 1024):N0} MB file");
        return true;
    }

    /// <summary>
    /// Open a v3 tree streamed from <paramref name="source"/> through a pool of <paramref name="poolNodes"/> slots (0 =
    /// <see cref="LodPoolOption"/>).
    /// </summary>
    async Task OpenLodStreamAsync(LodChunkFile.Header3 h, GpuLodPager.ChunkBytes source, string from, int poolNodes = 0)
    {
        var a = _gpuService.WebGPUAccelerator;
        var t0 = DateTime.UtcNow;
        _lodPager?.Dispose(); _lodPager = null;
        _gpuRenderer.UseRgbColours();
        _gpuRenderer.ColoursAreShDc = h.ColoursAreShDc;
        _gpuRenderer.LodBudget = LodBudgetOption;
        _lodPager = await GpuLodPager.CreateAsync(a, _gpuRenderer, h, source, poolNodes > 0 ? poolNodes : LodPoolOption,
            LodTauOption > 0f ? LodTauOption : 1.5f);
        Console.WriteLine($"[Import] '{h.Name}' (v3, streamed {from}): {h.LeafCount:N0} splats, {h.NodeCount:N0} LOD nodes in " +
            $"{h.Chunks.Length} chunks, chunk 0 on screen in {(DateTime.UtcNow - t0).TotalSeconds:F1}s");
        ShowLodScene(h);
        await SeatAtHomeViewAsync(new ProjectScene { HomeView = h.HomeView, TrainedIterations = h.TrainedIterations, SplatCount = h.LeafCount });
    }

    /// <summary>The viewer state for an open v3 file (no project behind it).</summary>
    void ShowLodScene(LodChunkFile.Header3 h)
    {
        var gaussianScene = new GaussianScene { GpuSplatCount = h.LeafCount, SourceName = "lod-file" };
        _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
        _renderService.SetActiveSceneGpuLoaded(gaussianScene);
        _sceneManager.ActiveScene = gaussianScene;
        _statusMessage = null;
        _state = StudioState.SceneViewer;
        _cameraController?.FitToScene();
        BuildViewerHudUI();
    }

    /// <summary>Write the scene on screen as a .spawnscene v3 (LOD tree, chunked) download.</summary>
    async Task ExportLodSceneFileAsync()
    {
        if (_editBusy) return;
        var a = _gpuService.WebGPUAccelerator;
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (packed == null || n < 2) { _editNote = "Nothing to export"; return; }
        if (_gpuRenderer.LodActive) { _editNote = "Already an LOD scene"; RefreshEditStatus(); return; }
        _editBusy = true; _editNote = "Building the LOD tree..."; RefreshEditStatus();
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

            // SH for every node, in the new order: the leaves' own, and each merge the weighted mean of its children's
            // (GpuLodLayout.MergeSh) - a distant merged region keeps its view-dependent colour.
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
                    GpuLodLayout.MergeSh(a, laid, sh[p], SphericalHarmonics.PartFloatsPerSplat);
                    await a.SynchronizeAsync();   // the gather has read `whole` before it goes
                }
            }

            // Chunk boundaries (whole sibling runs) and what each chunk needs, from the topology the file carries anyway.
            // CPU transfer: file I/O - parent and child count of every node (they are written into the chunks).
            const int Chunk = LodChunkFile.DefaultChunkNodes;
            await a.SynchronizeAsync();
            var topo = new LodTree
            {
                NodeCount = nodes, LeafCount = n,
                Parent = (await laid.Parent.CopyToHostAsync<int>(0, nodes)).ToArray(),
                ChildCount = (await laid.ChildCount.CopyToHostAsync<int>(0, nodes)).ToArray(),
            };
            int roots = 0;
            while (roots < nodes && topo.Parent[roots] < 0) roots++;
            topo.FirstChild = new int[nodes];
            for (int i = 0, next = roots; i < nodes; i++) { topo.FirstChild[i] = topo.ChildCount[i] > 0 ? next : -1; next += topo.ChildCount[i]; }
            var starts = LodLayout.ChunkStarts(topo, Chunk);
            var chunks = new List<LodChunkFile.Chunk>();
            var zipped = new List<Uint8Array>();
            long offset = 0, rawTotal = 0;
            for (int ci = 0; ci + 1 < starts.Length; ci++)
            {
                int first = starts[ci], count = starts[ci + 1] - first;
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
                    raw.Add(await laid.FirstChild.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                    raw.Add(await laid.Bounds.CopyToHostUint8ArrayAsync((long)first * 16, (long)count * 16));
                    raw.Add(await laid.LodSize.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                    js.AddRange(raw);
                    rawTotal += raw.Sum(r => (long)r.ByteLength);
                    using var blobIn = new Blob(raw, new BlobOptions { Type = "application/octet-stream" });
                    var z = await GzipAsync(blobIn, decompress: false);
                    js.Add(z);
                    var zu = new Uint8Array(z); js.Add(zu);
                    zipped.Add(zu);
                    chunks.Add(new LodChunkFile.Chunk(first, count, offset, z.ByteLength,
                        new[] { frame.MinX - frame.TailLoX, frame.MinY - frame.TailLoY, frame.MinZ - frame.TailLoZ,
                                frame.MinX + frame.SizeX + frame.TailHiX, frame.MinY + frame.SizeY + frame.TailHiY, frame.MinZ + frame.SizeZ + frame.TailHiZ },
                        frame.InnerArray(), LodLayout.ParentChunks(topo, starts, ci)));
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
            _editBusy = false;
            RefreshEditStatus();
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
        if (LodPoolOption > 0) { await OpenLodStreamAsync(bytes, h, dataStart); return; }
        _lodPager?.Dispose(); _lodPager = null;
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
        laid.FirstChild = a.Allocate1D<int>(nodes);
        laid.ChildCount = a.Allocate1D<int>(1);
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
                _gpuRenderer.WriteIlgpu(laid.Parent, (long)c.First * 4, raw, (int)L.Parent, L.FirstChild - L.Parent);
                _gpuRenderer.WriteIlgpu(laid.FirstChild, (long)c.First * 4, raw, (int)L.FirstChild, L.Bounds - L.FirstChild);
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

            ShowLodScene(h);
            await SeatAtHomeViewAsync(new ProjectScene { HomeView = h.HomeView, TrainedIterations = h.TrainedIterations, SplatCount = h.LeafCount });
        }
        finally
        {
            laid.Dispose();   // what the renderer did not take (child counts, stand-ins)
            if (sh != null) foreach (var b in sh) b.Dispose();
        }
    }
}
