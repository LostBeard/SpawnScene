using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Formats;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// "Open scene file" for scenes other tools made, converted on the GPU into a new project as a .spawnscene v1/v2 import
/// is: a 3DGS .ply (the reference trainer, gsplat, nerfstudio, Postshot, Polycam...; GaussianPlyImport), turned y-up by
/// default (a 3DGS .ply is in its SfM frame, y down), or Niantic's .spz (SpzImport; in practice also y down);
/// <c>&amp;sceneup=keep</c> leaves either as it is.
/// </summary>
public partial class Studio
{
    /// <summary>Another tool's scene is turned y-up unless <c>&amp;sceneup=keep</c> (or the older <c>&amp;plyup=keep</c>).</summary>
    static bool TurnImportYUp(Dictionary<string, string> query) =>
        !((query.TryGetValue("sceneup", out var s) && s == "keep") || (query.TryGetValue("plyup", out var p) && p == "keep"));

    /// <summary>A .ply or .spz: imported (or refused with a status) and true; false when it is neither.</summary>
    async Task<bool> ImportForeignBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        if (await ImportPlyBlobAsync(file, name, query)) return true;
        if (await ImportSpzBlobAsync(file, name, query)) return true;
        return await ImportSplatBlobAsync(file, name, query);
    }

    /// <summary>Import antimatter15's .splat - no magic, so by name and a whole number of 32-byte splats.</summary>
    async Task<bool> ImportSplatBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        if (!name.EndsWith(".splat", StringComparison.OrdinalIgnoreCase) || file.Size == 0 || file.Size % SplatFileImport.BytesPerSplat != 0)
            return false;
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        bool flip = TurnImportYUp(query);
        // CPU transfer: file I/O. The bytes stay JS-side; the GPU converts them.
        using var whole = await file.ArrayBuffer();
        int n = (int)(whole.ByteLength / SplatFileImport.BytesPerSplat);
        var packed = await SplatFileImport.ConvertAsync(_gpuService.WebGPUAccelerator, whole, flip);
        Uint8Array packedU8;
        try { packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)n * SplatFormat.Floats * sizeof(float)); }
        finally { packed.Dispose(); }
        var scene = new ProjectScene { SplatCount = n, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true, ImportedFrom = name };
        Console.WriteLine($"[Import] {name}: .splat, {n:N0} splats, converted in {t0.Elapsed.TotalSeconds:F1}s" + (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, new List<Uint8Array>(), query, "splat");
        return true;
    }

    /// <summary>Import a .spz (gzip around the NGSP stream); false when the file is not gzip or not SPZ inside.</summary>
    async Task<bool> ImportSpzBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        byte[] magic;
        using (var slice = file.Slice(0, Math.Min(2L, file.Size)))
        using (var mb = await slice.ArrayBuffer())
        using (var u = new Uint8Array(mb))
            magic = u.ReadBytes();
        if (magic.Length < 2 || magic[0] != 0x1f || magic[1] != 0x8b) return false;
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        // CPU transfer: none - gunzipped by the browser, JS-side.
        using var raw = await GzipAsync(file, decompress: true);
        byte[] head;
        using (var hu = new Uint8Array(raw, 0, Math.Min(16L, raw.ByteLength)))
            head = hu.ReadBytes();
        if (head.Length < 4 || BitConverter.ToUInt32(head) != SpzImport.Magic) return false;
        SpzImport.Header h;
        try { h = SpzImport.ParseHeader(head); }
        catch (FormatException ex)
        {
            _statusMessage = $"Cannot open {name}: {ex.Message}";
            Console.WriteLine($"[Import] {name}: {ex.Message}");
            return true;
        }
        bool flip = TurnImportYUp(query);
        var (packed, sh) = await SpzImport.ConvertAsync(_gpuService.WebGPUAccelerator, raw, h, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the decoded rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)h.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, (long)h.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = h.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true,
            ShDegree = sh != null ? h.KeptShDegree : 0, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: SPZ v{h.Version}, {h.Count:N0} splats, SH degree {h.ShDegree}" +
            (h.ShDegree > 3 ? " (band 4 dropped)" : "") + (h.Antialiased ? ", antialiased" : "") + $", decoded in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, shU8, query, "spz");
        return true;
    }

    /// <summary>Import a picked or fetched PLY; false (and a status) when it is not one we can read.</summary>
    async Task<bool> ImportPlyBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        byte[] head;
        using (var slice = file.Slice(0, Math.Min(65536L, file.Size)))
        using (var hb = await slice.ArrayBuffer())
        using (var u = new Uint8Array(hb))
            head = u.ReadBytes();
        if (!GaussianPly.IsPly(head)) return false;
        GaussianPly.Layout L;
        try { L = GaussianPly.Parse(head); }
        catch (FormatException ex)
        {
            _statusMessage = $"Cannot open {name}: {ex.Message}";
            Console.WriteLine($"[Import] {name}: {ex.Message}");
            return true;
        }
        bool flip = TurnImportYUp(query);
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        // CPU transfer: file I/O. The bytes stay JS-side; the GPU converts them a chunk at a time.
        using var whole = await file.ArrayBuffer();
        var a = _gpuService.WebGPUAccelerator;
        var (packed, sh) = await GaussianPlyImport.ConvertAsync(a, whole, L, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the converted rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, L.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, L.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = (int)L.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true,
            ShDegree = sh != null ? L.ShDegree : 0, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: 3DGS PLY, {L.Count:N0} splats, SH degree {L.ShDegree}, converted in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, shU8, query, "ply");
        return true;
    }

    /// <summary>Store an imported scene's rows (and SH parts) in a new project and open it. Disposes the arrays.</summary>
    async Task SaveAndOpenImportedSceneAsync(string sceneName, ProjectScene scene, Uint8Array packedU8, List<Uint8Array> sh,
        Dictionary<string, string> query, string kind)
    {
        var project = await _projectService.CreateProjectAsync(sceneName);
        await _projectService.SaveSceneAsync(project.Id, scene, packedU8);
        if (sh.Count > 0) await _projectService.SaveSceneShRestAsync(project.Id, scene, sh.ToArray());
        packedU8.Dispose();
        foreach (var s in sh) s.Dispose();
        Console.WriteLine($"[Import] '{sceneName}' ({kind}): {scene.SplatCount:N0} splats" + (scene.ShDegree > 0 ? $", SH degree {scene.ShDegree}" : "") + " - opening");

        _projects = await _projectService.ListProjectsAsync();
        var opened = _projects.First(p => p.Id == project.Id);
        OnOpenProject(opened);
        ApplyImportViewerOptions(query);
        await LoadProjectSceneAsync(opened.Scenes.First(s => s.Id == scene.Id));
        await FinishImportAsync(query);
    }
}
