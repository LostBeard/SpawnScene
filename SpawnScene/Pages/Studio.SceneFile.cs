using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// .spawnscene export / import: carry a complete scene (packed splats + SH bands + metadata) between browsers - a scene
// trained on the desktop is not in the headset's storage. Export downloads the scene on screen; ?import=<url> fetches
// one into a new project and opens it (the Quest test link).
public partial class Studio
{
    /// <summary>Download the scene on screen (visible splats only, SH bands included) as a .spawnscene file.</summary>
    async Task ExportSceneFileAsync()
    {
        if (_editBusy) return;
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null) return;
        _editBusy = true; _editNote = null; RefreshEditStatus();
        var parts = new List<Uint8Array>();
        try
        {
            int n = _gpuRenderer.SplatCount;
            var (count, packedU8, sh) = await ReadVisibleRowsAsync(packed, n);
            packedU8 ??= await _gpuRenderer.ReadPackedUint8ArrayAsync(count);
            if (sh == null && _gpuRenderer.ShDegree > 0) sh = await _gpuRenderer.ReadShRestUint8ArraysAsync();
            if (packedU8 == null) { _editNote = "Nothing to export"; return; }
            string name = _activeProject?.Name ?? "SpawnScene scene";
            var c = _sceneManager.Camera;   // the file opens at the view it was exported from
            var header = new SceneFile.Header(name, count, SplatFormat.Floats, _gpuRenderer.ColoursAreShDc,
                sh != null ? _gpuRenderer.ShDegree : 0, sh?.Length ?? 0, _viewedProjectScene?.TrainedIterations ?? 0, DateTime.UtcNow,
                new[] { c.Position.X, c.Position.Y, c.Position.Z, c.Forward.X, c.Forward.Y, c.Forward.Z, c.Up.X, c.Up.Y, c.Up.Z });
            parts.Add(new Uint8Array(SceneFile.Prefix(header)));
            parts.Add(packedU8);
            if (sh != null) parts.AddRange(sh);
            using var blob = new Blob(parts, new BlobOptions { Type = "application/octet-stream" });
            string url = blob.ToObjectURL();
            using var document = _js.Get<Document>("document");
            using var a = document.CreateElement<HTMLAnchorElement>("a");
            a.Href = url;
            a.Download = string.Concat(name.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_')) + SceneFile.Extension;
            a.Click();
            _ = Task.Delay(60_000).ContinueWith(_ => URL.RevokeObjectURL(url));
            _editNote = $"Exported {count:N0} splats";
            Console.WriteLine($"[Edit] exported '{name}': {count:N0} splats" + (sh != null ? $", SH degree {header.ShDegree}" : "") +
                $", {(blob.Size / (1024 * 1024))} MB");
        }
        catch (Exception ex) { _editNote = "Export failed"; Console.WriteLine($"[Edit] export failed: {ex.Message}"); }
        finally
        {
            foreach (var p in parts) p.Dispose();
            _editBusy = false;
            RefreshEditStatus();
        }
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
            await ExportSceneFileAsync();
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
            int headerLen = SceneFile.HeaderLength(first.ReadBytes());
            using var headerView = new Uint8Array(bytes, 12, headerLen);
            var h = SceneFile.ParseHeader(headerView.ReadBytes());
            long offset = SceneFile.DataOffset(headerLen);
            long packedBytes = (long)h.SplatCount * h.FloatsPerSplat * sizeof(float);
            using var packedU8 = new Uint8Array(bytes, offset, packedBytes);
            offset += packedBytes;
            var sh = new List<Uint8Array>();
            for (int p = 0; p < h.ShParts; p++)
            {
                long partBytes = (long)h.SplatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
                sh.Add(new Uint8Array(bytes, offset, partBytes));
                offset += partBytes;
            }

            var project = await _projectService.CreateProjectAsync(h.Name);
            var scene = new ProjectScene
            {
                SplatCount = h.SplatCount, FloatsPerSplat = h.FloatsPerSplat, ColoursAreShDc = h.ColoursAreShDc,
                ShDegree = h.ShParts > 0 ? h.ShDegree : 0, TrainedIterations = h.TrainedIterations,
                HomeView = h.HomeView,
            };
            await _projectService.SaveSceneAsync(project.Id, scene, packedU8);
            if (sh.Count > 0) await _projectService.SaveSceneShRestAsync(project.Id, scene, sh.ToArray());
            foreach (var s in sh) s.Dispose();
            Console.WriteLine($"[Import] '{h.Name}': {h.SplatCount:N0} splats" + (h.ShParts > 0 ? $", SH degree {h.ShDegree}" : "") + " - opening");

            _projects = await _projectService.ListProjectsAsync();
            var opened = _projects.First(p => p.Id == project.Id);
            OnOpenProject(opened);
            await LoadProjectSceneAsync(opened.Scenes.First(s => s.Id == scene.Id));
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
        catch (Exception ex) { Console.WriteLine($"[Import] FAIL: {ex.Message}"); }
    }
}
