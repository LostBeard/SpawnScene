using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Formats;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// "Open scene file" for scenes other tools made: a 3DGS .ply (the reference trainer, gsplat, nerfstudio, Postshot,
/// Polycam...) converted on the GPU (GaussianPlyImport) into a new project, as a .spawnscene v1/v2 import is. The file is
/// turned y-up by default (a 3DGS .ply is in its SfM frame, y down); <c>&amp;plyup=keep</c> leaves it as it is.
/// </summary>
public partial class Studio
{
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
        bool flip = !(query.TryGetValue("plyup", out var up) && up == "keep");
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
