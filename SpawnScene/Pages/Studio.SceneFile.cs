using SpawnDev.ILGPU;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// .spawnscene export / import: carry a complete scene (packed splats + SH bands + metadata) between browsers - a scene
// trained on the desktop is not in the headset's storage. Export downloads the scene on screen; ?import=<url> fetches
// one into a new project and opens it (the Quest test link).
public partial class Studio
{
    /// <summary>
    /// Download the scene on screen (visible splats only, SH bands included) as a .spawnscene v2 file: quantized on the
    /// GPU (SceneCodec, 76 bytes a splat instead of 236) and each stream gzipped by the browser. The raw v1 export of a
    /// 3.28M-splat bicycle was 737 MB, and Chrome cancelled the download.
    /// </summary>
    async Task ExportSceneFileAsync()
    {
        if (_editBusy) return;
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null) return;
        _editBusy = true; _editNote = null; RefreshEditStatus();
        var owned = new List<IDisposable>();
        var js = new List<IDisposable>();
        try
        {
            var a = _gpuService.WebGPUAccelerator;
            int n = _gpuRenderer.SplatCount;
            // The visible rows, on the GPU: Delete / Keep only leave opacity-0 rows behind.
            var all = SplatEditor.Volume.Rows(0, n);
            int visible = await _splatEditor.CountAsync(a, packed, n, all);
            if (visible <= 0) { _editNote = "Nothing to export"; return; }
            bool withSh = _gpuRenderer.ShDegree > 0 && _gpuRenderer.ShRestBuffers != null;
            var keptPacked = packed;
            ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense>[]? keptSh = null;
            int count = n;
            if (visible < n)
            {
                var indices = await SplatRows.SelectIndicesAsync(a, packed, n, all, visible); owned.Add(indices);
                keptPacked = SplatRows.GatherRows(a, packed, indices, visible, SplatFormat.Floats); owned.Add(keptPacked);
                if (withSh)
                {
                    keptSh = new ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense>[3];
                    for (int part = 0; part < 3; part++)
                    {
                        var whole = _gpuRenderer.CopyShPartToIlgpu(a, part, n); owned.Add(whole);
                        keptSh[part] = SplatRows.GatherRows(a, whole, indices, visible, SphericalHarmonics.PartFloatsPerSplat);
                        owned.Add(keptSh[part]);
                    }
                }
                count = visible;
            }
            else if (withSh)
            {
                keptSh = Enumerable.Range(0, 3).Select(part => _gpuRenderer.CopyShPartToIlgpu(a, part, n)).ToArray();
                owned.AddRange(keptSh);
            }

            var box = await SplatBounds.ComputeAsync(a, keptPacked, count) ?? new SplatBounds.Aabb(0, 0, 0, 1, 1, 1);
            // Linear codes over the box holding all but 0.5% at each end of each axis, padded by a quarter of its size;
            // the floaters past it get log-spaced codes (SceneCodec.QuantPosP).
            var robust = await SplatBounds.ComputeRobustAsync(a, keptPacked, count, 0.005) ?? box;
            float px = 0.25f * (robust.MaxX - robust.MinX), py = 0.25f * (robust.MaxY - robust.MinY), pz = 0.25f * (robust.MaxZ - robust.MinZ);
            var inner = new SplatBounds.Aabb(robust.MinX - px, robust.MinY - py, robust.MinZ - pz, robust.MaxX + px, robust.MaxY + py, robust.MaxZ + pz);
            var frame = SceneCodec.Frame.From(box, inner, count);
            var (geo, app, shq) = SceneCodec.Encode(a, keptPacked, keptSh, frame);
            owned.Add(geo); owned.Add(app); owned.Add(shq);
            await a.SynchronizeAsync();

            // CPU transfer: file I/O - the encoded streams, a third of the raw scene, then gzipped by the browser.
            var raw = new List<Uint8Array>
            {
                await geo.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.GeoWords * sizeof(uint)),
                await app.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.AppWords * sizeof(uint)),
            };
            if (withSh) raw.Add(await shq.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.ShWords * sizeof(uint)));
            js.AddRange(raw);
            var lens = new long[3];
            var zipped = new List<Uint8Array>();
            long rawBytes = 0;
            for (int k = 0; k < raw.Count; k++)
            {
                rawBytes += raw[k].ByteLength;
                using var blobIn = new Blob(new[] { raw[k] }, new BlobOptions { Type = "application/octet-stream" });
                var z = await GzipAsync(blobIn, decompress: false);
                js.Add(z);
                var zu = new Uint8Array(z); js.Add(zu);
                zipped.Add(zu);
                lens[k] = z.ByteLength;
            }

            string name = _activeProject?.Name ?? "SpawnScene scene";
            var c = _sceneManager.Camera;   // the file opens at the view it was exported from
            var header = new SceneFile.Header2(name, count, _gpuRenderer.ColoursAreShDc, withSh ? _gpuRenderer.ShDegree : 0,
                _viewedProjectScene?.TrainedIterations ?? 0, DateTime.UtcNow,
                new[] { c.Position.X, c.Position.Y, c.Position.Z, c.Forward.X, c.Forward.Y, c.Forward.Z, c.Up.X, c.Up.Y, c.Up.Z },
                new[] { box.MinX, box.MinY, box.MinZ, box.MaxX, box.MaxY, box.MaxZ }, lens, Inner: frame.InnerArray());
            var prefix = new Uint8Array(SceneFile.Prefix2(header)); js.Add(prefix);
            var fileParts = new List<Uint8Array> { prefix };
            fileParts.AddRange(zipped);
            using var blob = new Blob(fileParts, new BlobOptions { Type = "application/octet-stream" });
            DownloadSceneBlob(blob, name);
            long rawV1 = (long)count * (SplatFormat.Floats + (withSh ? 45 : 0)) * sizeof(float);
            _editNote = $"Exported {count:N0} splats ({blob.Size / (1024 * 1024)} MB)";
            Console.WriteLine($"[Edit] exported '{name}': {count:N0} splats" + (withSh ? $", SH degree {header.ShDegree}" : "") +
                $", {blob.Size / (1024 * 1024)} MB (v2; raw v1 would be {rawV1 / (1024 * 1024)} MB, quantized {rawBytes / (1024 * 1024)} MB before gzip)");
        }
        catch (Exception ex) { _editNote = "Export failed"; Console.WriteLine($"[Edit] export failed: {ex.Message}"); }
        finally
        {
            foreach (var d in js) d.Dispose();
            foreach (var d in owned) d.Dispose();
            _editBusy = false;
            RefreshEditStatus();
        }
    }

    /// <summary>Hand a finished .spawnscene to the browser as a download named after the scene.</summary>
    void DownloadSceneBlob(Blob blob, string name)
    {
        string url = blob.ToObjectURL();
        using var document = _js.Get<Document>("document");
        using var anchor = document.CreateElement<HTMLAnchorElement>("a");
        anchor.Href = url;
        anchor.Download = string.Concat(name.Select(ch => char.IsLetterOrDigit(ch) || ch is '-' or '_' ? ch : '_')) + SceneFile.Extension;
        anchor.Click();
        _ = Task.Delay(120_000).ContinueWith(_ => URL.RevokeObjectURL(url));
    }

    /// <summary>gzip (or gunzip) a blob through the browser's CompressionStream - streamed, no second full copy in .NET.</summary>
    static async Task<ArrayBuffer> GzipAsync(Blob input, bool decompress)
    {
        using var src = input.Stream();
        if (decompress)
        {
            using var gunzip = new DecompressionStream("gzip");
            using var piped = src.PipeThrough(gunzip);
            using var resp = new Response(piped, (ResponseOptions?)null);
            return await resp.ArrayBuffer();
        }
        using var gzip = new CompressionStream("gzip");
        using var zipped = src.PipeThrough(gzip);
        using var zresp = new Response(zipped, (ResponseOptions?)null);
        return await zresp.ArrayBuffer();
    }

    /// <summary>
    /// ?export=latest (harness): open the newest scene of the most recently modified project and download it as a
    /// .spawnscene - the same export the Edit toolbar runs, without driving the UI by screen position.
    /// </summary>
    async Task RunExportIfRequestedAsync()
    {
        var uri = new Uri(_nav.Uri);
        if (!uri.Query.Contains("export=latest", StringComparison.OrdinalIgnoreCase)) return;
        try
        {
            _projects = await _projectService.ListProjectsAsync();
            var project = _projects.Where(p => p.Scenes.Count > 0).OrderByDescending(p => p.ModifiedAt).FirstOrDefault();
            if (project == null) { Console.WriteLine("[Export] FAIL: no saved scene"); return; }
            var scene = project.Scenes.OrderByDescending(s => s.CreatedAt).First();
            Console.WriteLine($"[Export] '{project.Name}' scene {scene.Id}: {scene.SplatCount:N0} splats");
            OnOpenProject(project);
            await LoadProjectSceneAsync(scene);
            // &lodfile=1: the scene as its LOD tree (.spawnscene v3, Studio.LodFile) instead of the flat v2 file.
            if (uri.Query.Contains("lodfile=1", StringComparison.OrdinalIgnoreCase)) await ExportLodSceneFileAsync();
            else await ExportSceneFileAsync();
            Console.WriteLine($"[Export] DONE: {_editNote}");
        }
        catch (Exception ex) { Console.WriteLine($"[Export] FAIL: {ex.Message}"); }
    }

    /// <summary>
    /// ?import=&lt;url&gt;: fetch a .spawnscene file, save it as a scene of a new project, and open it in the viewer.
    /// </summary>
    async Task RunImportIfRequestedAsync()
    {
        var uri = new Uri(_nav.Uri);
        var query = uri.Query.TrimStart('?').Split('&', StringSplitOptions.RemoveEmptyEntries)
            .Select(p => p.Split('=', 2)).Where(p => p.Length == 2)
            .ToDictionary(p => Uri.UnescapeDataString(p[0]), p => Uri.UnescapeDataString(p[1]), StringComparer.OrdinalIgnoreCase);
        if (!query.TryGetValue("import", out var importUrl) || string.IsNullOrWhiteSpace(importUrl)) return;
        try
        {
            Console.WriteLine($"[Import] fetching {importUrl}");
            using var window = _js.Get<Window>("window");
            using var response = await window.Fetch(importUrl);
            if (!response.Ok) { Console.WriteLine($"[Import] FAIL: HTTP {response.Status}"); return; }
            using var bytes = await response.ArrayBuffer();
            using var first = new Uint8Array(bytes, 0, 12);
            var firstBytes = first.ReadBytes();
            int version = SceneFile.Version(firstBytes);
            int headerLen = SceneFile.HeaderLength(firstBytes);
            using var headerView = new Uint8Array(bytes, 12, headerLen);
            long offset = SceneFile.DataOffset(headerLen);
            if (version == 3)
            {
                // An LOD tree (.spawnscene v3): a viewer file, opened as it is - not copied into a project.
                ApplyImportViewerOptions(query);
                await OpenLodFileAsync(bytes, LodChunkFile.ParseHeader3(headerView.ReadBytes()), offset);
                await FinishImportAsync(query);
                return;
            }
            Uint8Array packedU8;
            var sh = new List<Uint8Array>();
            ProjectScene scene;
            string sceneName;
            if (version == 2)
            {
                // v2: gunzip each stream, decode on the GPU (SceneCodec), then read the raw rows back for the project store.
                var h2 = SceneFile.ParseHeader2(headerView.ReadBytes());
                var a = _gpuService.WebGPUAccelerator;
                var words = new ILGPU.Runtime.MemoryBuffer1D<uint, ILGPU.Stride1D.Dense>?[3];
                try
                {
                    for (int k = 0; k < 3; k++)
                    {
                        long len = h2.StreamBytes.Length > k ? h2.StreamBytes[k] : 0;
                        if (len <= 0) continue;
                        using var zipped = new Uint8Array(bytes, offset, len);
                        using var blobIn = new Blob(new[] { zipped }, new BlobOptions { Type = "application/octet-stream" });
                        using var rawStream = await GzipAsync(blobIn, decompress: true);
                        words[k] = _gpuRenderer.IlgpuWordsFromArrayBuffer(a, rawStream);
                        offset += len;
                    }
                    var b = h2.Bounds;
                    var outer = new SplatBounds.Aabb(b[0], b[1], b[2], b[3], b[4], b[5]);
                    var frame = h2.Inner is { Length: 6 } r
                        ? SceneCodec.Frame.From(outer, new SplatBounds.Aabb(r[0], r[1], r[2], r[3], r[4], r[5]), h2.SplatCount)
                        : SceneCodec.Frame.From(outer, h2.SplatCount);
                    var (dp, dsh) = SceneCodec.Decode(a, words[0]!, words[1]!, words[2], frame);
                    try
                    {
                        await a.SynchronizeAsync();
                        // CPU transfer: file I/O - the decoded rows go to the project store as any saved scene does.
                        packedU8 = await dp.CopyToHostUint8ArrayAsync(0, (long)h2.SplatCount * SplatFormat.Floats * sizeof(float));
                        if (dsh != null)
                            foreach (var part in dsh)
                                sh.Add(await part.CopyToHostUint8ArrayAsync(0, (long)h2.SplatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
                    }
                    finally
                    {
                        dp.Dispose();
                        if (dsh != null) foreach (var part in dsh) part.Dispose();
                    }
                }
                finally { foreach (var w in words) w?.Dispose(); }
                scene = new ProjectScene
                {
                    SplatCount = h2.SplatCount, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = h2.ColoursAreShDc,
                    ShDegree = sh.Count > 0 ? h2.ShDegree : 0, TrainedIterations = h2.TrainedIterations, HomeView = h2.HomeView,
                };
                sceneName = h2.Name;
            }
            else
            {
                var h = SceneFile.ParseHeader(headerView.ReadBytes());
                long packedBytes = (long)h.SplatCount * h.FloatsPerSplat * sizeof(float);
                packedU8 = new Uint8Array(bytes, offset, packedBytes);
                offset += packedBytes;
                for (int p = 0; p < h.ShParts; p++)
                {
                    long partBytes = (long)h.SplatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
                    sh.Add(new Uint8Array(bytes, offset, partBytes));
                    offset += partBytes;
                }
                scene = new ProjectScene
                {
                    SplatCount = h.SplatCount, FloatsPerSplat = h.FloatsPerSplat, ColoursAreShDc = h.ColoursAreShDc,
                    ShDegree = h.ShParts > 0 ? h.ShDegree : 0, TrainedIterations = h.TrainedIterations,
                    HomeView = h.HomeView,
                };
                sceneName = h.Name;
            }

            var project = await _projectService.CreateProjectAsync(sceneName);
            await _projectService.SaveSceneAsync(project.Id, scene, packedU8);
            if (sh.Count > 0) await _projectService.SaveSceneShRestAsync(project.Id, scene, sh.ToArray());
            packedU8.Dispose();
            foreach (var s in sh) s.Dispose();
            Console.WriteLine($"[Import] '{sceneName}' (v{version}): {scene.SplatCount:N0} splats" + (scene.ShDegree > 0 ? $", SH degree {scene.ShDegree}" : "") + " - opening");

            _projects = await _projectService.ListProjectsAsync();
            var opened = _projects.First(p => p.Id == project.Id);
            OnOpenProject(opened);
            ApplyImportViewerOptions(query);
            await LoadProjectSceneAsync(opened.Scenes.First(s => s.Id == scene.Id));
            await FinishImportAsync(query);
        }
        catch (Exception ex) { Console.WriteLine($"[Import] FAIL: {ex.Message}"); }
    }

    /// <summary>Viewer options an import is opened with (the autotest branch that parses them never runs for an import).</summary>
    void ApplyImportViewerOptions(Dictionary<string, string> query)
    {
        // &lodtau=N draws the scene through its LOD tree (Studio.Lod); &lodpx=N is the sub-pixel cull (0 = none).
        if (query.TryGetValue("lodtau", out var ltq) && float.TryParse(ltq, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var ltv))
            LodTauOption = Math.Max(0f, ltv);
        if (query.TryGetValue("lodbudget", out var lbq) && int.TryParse(lbq, out var lbv))
            LodBudgetOption = Math.Max(0, lbv);
        // &fpslog=1 (harness): log the frame rate each second; with SPAWNSCENE_CHROME_UNCAPPED=1 it is render cost.
        if (query.TryGetValue("fpslog", out var fpq) && fpq is "1" or "true")
            _renderService.OnFpsUpdated += fps => Console.WriteLine($"[FPS] {fps:F1}" +
                (_gpuRenderer.LodDrawn >= 0 ? $" ({_gpuRenderer.LodDrawn:N0} drawn by the LOD cut)" : ""));
        if (query.TryGetValue("lodpx", out var lpq) && float.TryParse(lpq, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var lpv))
            _gpuRenderer.LodCullPixels = Math.Max(0f, lpv);
    }

    /// <summary>What every import does once its scene is on screen: the harness's exact camera, XR hook, render mode.</summary>
    async Task FinishImportAsync(Dictionary<string, string> query)
    {
        // &park=w,h,fx,fy,cx,cy,px,py,pz,fwdx,fwdy,fwdz,upx,upy,upz (harness): seat the viewer at an EXACT camera,
        // intrinsics and roll included, so a reference renderer can draw the identical view for a side by side. The
        // home view goes through yaw/pitch (roll dropped), which is right for a person and wrong for an A/B.
        if (query.TryGetValue("park", out var parkQ))
        {
            var v = parkQ.Split(',').Select(t => float.Parse(t, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            if (v.Length == 15)
            {
                var cam = new CameraParams
                {
                    Width = (int)v[0], Height = (int)v[1], FocalX = v[2], FocalY = v[3], CenterX = v[4], CenterY = v[5],
                    Position = new(v[6], v[7], v[8]), Forward = new(v[9], v[10], v[11]), Up = new(v[12], v[13], v[14]),
                };
                // Deterministic: the stochastic renderer needs many still frames to converge, and a harness tab ran at
                // 0.2 fps - its captures were a few noisy samples (pastel speckle, broken needles), not the scene.
                if (!query.ContainsKey("render")) _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
                await ParkOnGroundTruthPoseAsync("park", cam);
                Console.WriteLine("[Import] PARKED");
            }
        }
        // &xrhook=1 (harness): the XR entry hook, as the Room autotest offers it (tools/_cdp_xr.js).
        if (query.ContainsKey("xrhook"))
        {
            _xrHook ??= new SpawnDev.SpawnJS.ActionCallback<string>(m => _ = EnterXRAsync(m));
            _js.Set("__spawnsceneEnterXR", _xrHook);
            Console.WriteLine("[Autotest] XR hook ready");
        }
        if (query.TryGetValue("render", out var rmode))
            _gpuRenderer.RenderMode = rmode == "sorted" ? SplatRenderMode.Sorted : SplatRenderMode.Stochastic;
        Console.WriteLine($"[Import] DONE (render mode {_gpuRenderer.RenderMode})");
    }
}
