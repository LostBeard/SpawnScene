using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;
using SpawnScene.Models;
using System.Text;
using System.Text.Json;

namespace SpawnScene.Services;

/// <summary>
/// OPFS-backed project storage. Projects are stored as JSON metadata + binary scene data.
///
/// OPFS structure:
///   /spawnscene/
///     projects.json              (project index — list of Project metadata)
///     projects/{id}/
///       sources/{filename}       (original image files)
///       scenes/{sceneId}.bin     (packed Gaussian float[] data)
/// </summary>
public class ProjectService
{
    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false,
    };

    private List<Project>? _projects;

    /// <summary>List all projects (cached after first load).</summary>
    public async Task<List<Project>> ListProjectsAsync()
    {
        if (_projects != null) return _projects;

        try
        {
            var root = await GetRootDirAsync();
            var data = await ReadTextAsync(root, "projects.json");
            _projects = data != null ? JsonSerializer.Deserialize<List<Project>>(data, _jsonOpts) ?? new() : new();
            // Projects saved under a preset before its current values (3M cap, 1024 keypoints) take them.
            foreach (var project in _projects) ReconstructionPresets.UpgradeLegacyPresetValues(project.Settings);
        }
        catch
        {
            _projects = new();
        }
        return _projects;
    }

    /// <summary>Create a new project and persist it.</summary>
    public async Task<Project> CreateProjectAsync(string name, ProjectSettings? settings = null)
    {
        var projects = await ListProjectsAsync();
        var project = new Project
        {
            Name = name,
            Settings = settings ?? new ProjectSettings(),
        };
        projects.Add(project);

        // Create project directory in OPFS
        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, project.Id);
        await projDir.GetDirectoryHandle("sources", create: true);
        await projDir.GetDirectoryHandle("scenes", create: true);

        await SaveIndexAsync();
        Console.WriteLine($"[ProjectService] Created project '{name}' ({project.Id})");
        return project;
    }

    /// <summary>Delete a project and all its files.</summary>
    public async Task DeleteProjectAsync(string projectId)
    {
        var projects = await ListProjectsAsync();
        projects.RemoveAll(p => p.Id == projectId);

        try
        {
            var root = await GetRootDirAsync();
            using var projectsDir = await root.GetDirectoryHandle("projects");
            await projectsDir.RemoveEntry(projectId, recursive: true);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ProjectService] Error deleting project dir: {ex.Message}");
        }

        await SaveIndexAsync();
        Console.WriteLine($"[ProjectService] Deleted project {projectId}");
    }

    /// <summary>Save a source image file to the project.</summary>
    public async Task AddSourceAsync(string projectId, string fileName, byte[] data, int width, int height)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;

        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, projectId);
        using var sourcesDir = await projDir.GetDirectoryHandle("sources", create: true);

        await WriteBinaryAsync(sourcesDir, fileName, data);

        project.Sources.Add(new ProjectSource
        {
            FileName = fileName,
            SizeBytes = data.Length,
            Width = width,
            Height = height,
        });
        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    /// <summary>Remove a source image from the project.</summary>
    public async Task RemoveSourceAsync(string projectId, string fileName)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;

        project.Sources.RemoveAll(s => s.FileName == fileName);

        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var sourcesDir = await projDir.GetDirectoryHandle("sources");
            await sourcesDir.RemoveEntry(fileName);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ProjectService] Error removing source file: {ex.Message}");
        }

        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    /// <summary>Read a source image file from the project.</summary>
    /// <summary>
    /// A source photo as a browser <see cref="File"/> (a Blob) - the bytes stay in JS. Decode it with
    /// <c>MediaInterop.DecodeToDeviceAsync</c> and it goes OPFS -> GPU without touching the .NET heap. Caller disposes.
    /// </summary>
    public async Task<SpawnDev.SpawnJS.JSObjects.File?> GetSourceFileAsync(string projectId, string fileName)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var sourcesDir = await projDir.GetDirectoryHandle("sources");
            using var fileHandle = await sourcesDir.GetFileHandle(fileName);
            return await fileHandle.GetFile();
        }
        catch
        {
            return null;
        }
    }

    public async Task<byte[]?> GetSourceAsync(string projectId, string fileName)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var sourcesDir = await projDir.GetDirectoryHandle("sources");
            return await ReadBinaryAsync(sourcesDir, fileName);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Save a generated scene's packed float data.</summary>
    public async Task SaveSceneAsync(string projectId, ProjectScene scene, byte[] packedData)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;

        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, projectId);
        using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);

        await WriteBinaryAsync(scenesDir, $"{scene.Id}.bin", packedData);

        scene.SizeBytes = packedData.Length;
        project.Scenes.Add(scene);
        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    /// <summary>
    /// Save a scene whose packed splat data is a JS <see cref="Uint8Array"/> (straight off the GPU).
    /// The bytes stream GPU → JS → OPFS without ever touching the .NET/WASM managed heap — the only
    /// scalable path for large scenes (a 5K image is ~14.7M splats ≈ 588 MB, which OOMs the managed
    /// heap if marshalled into a byte[]). Caller still owns/disposes the Uint8Array.
    /// </summary>
    public async Task SaveSceneAsync(string projectId, ProjectScene scene, Uint8Array packedData)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;

        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, projectId);
        using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);

        await WriteBinaryAsync(scenesDir, $"{scene.Id}.bin", packedData);

        scene.SizeBytes = packedData.Length;
        project.Scenes.Add(scene);
        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    /// <summary>
    /// Save a trained scene's SH rest parts next to its packed data (scenes/{id}.sh{p}.bin, one per
    /// SphericalHarmonics part), JS memory straight to OPFS. Call after
    /// <see cref="SaveSceneAsync(string, ProjectScene, Uint8Array)"/>; adds to the scene's size and records the layout.
    /// </summary>
    public async Task SaveSceneShRestAsync(string projectId, ProjectScene scene, Uint8Array[] parts)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;
        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, projectId);
        using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);
        for (int part = 0; part < parts.Length; part++)
            await WriteBinaryAsync(scenesDir, $"{scene.Id}.sh{part}.bin", parts[part]);
        scene.ShParts = parts.Length;
        var stored = project.Scenes.FirstOrDefault(s => s.Id == scene.Id);
        if (stored != null)
        {
            stored.SizeBytes += parts.Sum(p => (long)p.Length);
            stored.ShParts = parts.Length;
        }
        await SaveIndexAsync();
    }

    /// <summary>
    /// Autotest only: rewrite a scene's SH as the pre-split layout (one row-major scenes/{id}.sh.bin, ShParts 0) so the
    /// legacy load path (GpuGaussianRenderer.LoadShRestRows) can be checked against the part path on the same data.
    /// </summary>
    public async Task RewriteSceneShAsLegacyRowsAsync(string projectId, string sceneId, Uint8Array rows)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        var stored = project?.Scenes.FirstOrDefault(s => s.Id == sceneId);
        if (stored == null) return;
        var root = await GetRootDirAsync();
        var projDir = await GetProjectDirAsync(root, projectId);
        using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);
        await WriteBinaryAsync(scenesDir, $"{sceneId}.sh.bin", rows);
        for (int part = 0; part < SphericalHarmonics.Parts; part++)
            try { await scenesDir.RemoveEntry($"{sceneId}.sh{part}.bin"); } catch { }
        stored.ShParts = 0;
        await SaveIndexAsync();
    }

    /// <summary>A trained scene's SH rest parts as JS ArrayBuffers (never the .NET heap), or null if any is missing.</summary>
    public async Task<ArrayBuffer[]?> ReadSceneShRestPartsAsync(string projectId, string sceneId, int parts)
    {
        var buffers = new List<ArrayBuffer>();
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            for (int part = 0; part < parts; part++)
            {
                using var fileHandle = await scenesDir.GetFileHandle($"{sceneId}.sh{part}.bin");
                using var file = await fileHandle.GetFile();
                buffers.Add(await file.ArrayBuffer());
            }
            return buffers.ToArray();
        }
        catch
        {
            foreach (var b in buffers) b.Dispose();
            return null;
        }
    }

    /// <summary>
    /// A scene saved before the SH part split: its SH rest bands as one row-major JS ArrayBuffer (never the .NET
    /// heap), or null when the scene has none. Caller disposes.
    /// </summary>
    public async Task<ArrayBuffer?> ReadSceneShRestAsync(string projectId, string sceneId)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            using var fileHandle = await scenesDir.GetFileHandle($"{sceneId}.sh.bin");
            using var file = await fileHandle.GetFile();
            return await file.ArrayBuffer();
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Open a streaming <see cref="Stream"/> over a scene's packed data in OPFS, WITHOUT reading it into
    /// memory. Returns a <see cref="BlobStream"/> (an <c>IJSReadStream</c>, <c>CanReadSync=false</c>) so the
    /// caller can stream it straight to the GPU via <c>ArrayView.CopyFromStreamAsync</c> — the bytes flow
    /// OPFS → JS → GPU chunk-by-chunk and never enter the .NET/WASM managed heap. The returned stream owns
    /// the underlying File/Blob; dispose it when done. Returns null if the scene file is missing.
    /// </summary>
    public async Task<Stream?> OpenSceneStreamAsync(string projectId, string sceneId)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            using var fileHandle = await scenesDir.GetFileHandle($"{sceneId}.bin");
            var file = await fileHandle.GetFile(); // File : Blob — owned+disposed by the BlobStream
            return new BlobStream(file);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Read a scene's packed float data.</summary>
    public async Task<byte[]?> GetSceneDataAsync(string projectId, string sceneId)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            return await ReadBinaryAsync(scenesDir, $"{sceneId}.bin");
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Save a scene thumbnail image (RGBA bytes already on the .NET heap — prefer the Uint8Array overload).</summary>
    public async Task SaveSceneThumbnailAsync(string projectId, string sceneId, byte[] pngData)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);
            await WriteBinaryAsync(scenesDir, $"{sceneId}.thumb", pngData);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ProjectService] Error saving scene thumbnail: {ex.Message}");
        }
    }

    /// <summary>
    /// Save a scene thumbnail from a JS TypedArray (e.g. ImageData.Data) without
    /// pulling the pixels into the .NET/WASM managed heap.
    /// </summary>
    public async Task SaveSceneThumbnailAsync(string projectId, string sceneId, TypedArray rgbaData)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes", create: true);
            // Writable.write accepts BufferSource; TypedArray is the JS-side path.
            using var fileHandle = await scenesDir.GetFileHandle($"{sceneId}.thumb", create: true);
            using var writable = await fileHandle.CreateWritable();
            await writable.Write(rgbaData);
            await writable.Close();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ProjectService] Error saving scene thumbnail: {ex.Message}");
        }
    }

    /// <summary>Load a scene thumbnail image.</summary>
    public async Task<byte[]?> GetSceneThumbnailAsync(string projectId, string sceneId)
    {
        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            return await ReadBinaryAsync(scenesDir, $"{sceneId}.thumb");
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Delete a scene from a project.</summary>
    public async Task DeleteSceneAsync(string projectId, string sceneId)
    {
        var project = (await ListProjectsAsync()).FirstOrDefault(p => p.Id == projectId);
        if (project == null) return;

        project.Scenes.RemoveAll(s => s.Id == sceneId);

        try
        {
            var root = await GetRootDirAsync();
            var projDir = await GetProjectDirAsync(root, projectId);
            using var scenesDir = await projDir.GetDirectoryHandle("scenes");
            await scenesDir.RemoveEntry($"{sceneId}.bin");
            try { await scenesDir.RemoveEntry($"{sceneId}.sh.bin"); } catch { /* untrained scenes have none */ }
            for (int part = 0; part < SphericalHarmonics.Parts; part++)
                try { await scenesDir.RemoveEntry($"{sceneId}.sh{part}.bin"); } catch { /* legacy or untrained */ }
        }
        catch { }

        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    /// <summary>Calculate total size of a project on disk.</summary>
    public long GetProjectSize(Project project)
    {
        long size = 0;
        foreach (var src in project.Sources) size += src.SizeBytes;
        foreach (var scene in project.Scenes) size += scene.SizeBytes;
        return size;
    }

    /// <summary>Update project metadata and save.</summary>
    public async Task UpdateProjectAsync(Project project)
    {
        project.ModifiedAt = DateTime.UtcNow;
        await SaveIndexAsync();
    }

    // ─── Work files: intermediate data a long job parks in OPFS (projects/{id}/work/), not part of any scene ───

    /// <summary>Write JS bytes to projects/{id}/work/{name} (a trained block of a partitioned run waiting to be merged).</summary>
    public async Task WriteWorkFileAsync(string projectId, string name, Uint8Array data)
    {
        using var root = await GetRootDirAsync();
        using var projDir = await GetProjectDirAsync(root, projectId);
        using var workDir = await projDir.GetDirectoryHandle("work", create: true);
        await WriteBinaryAsync(workDir, name, data);
    }

    /// <summary>A work file as a JS ArrayBuffer (never the .NET heap), or null when it is missing. Caller disposes.</summary>
    public async Task<ArrayBuffer?> ReadWorkFileAsync(string projectId, string name)
    {
        try
        {
            using var root = await GetRootDirAsync();
            using var projDir = await GetProjectDirAsync(root, projectId);
            using var workDir = await projDir.GetDirectoryHandle("work");
            using var fileHandle = await workDir.GetFileHandle(name);
            using var file = await fileHandle.GetFile();
            return await file.ArrayBuffer();
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Delete the project's work files (after a merge, or before a new partitioned run).</summary>
    public async Task ClearWorkFilesAsync(string projectId)
    {
        try
        {
            using var root = await GetRootDirAsync();
            using var projDir = await GetProjectDirAsync(root, projectId);
            await projDir.RemoveEntry("work", recursive: true);
        }
        catch
        {
            // Nothing parked.
        }
    }

    // ─── OPFS Helpers ───

    private async Task<FileSystemDirectoryHandle> GetRootDirAsync()
    {
        using var navigator = SpawnJSRuntime.Instance.Get<Navigator>("navigator");
        using var storage = navigator.Storage;
        using var opfsRoot = await storage.GetDirectory();
        return await opfsRoot.GetDirectoryHandle("spawnscene", create: true);
    }

    private async Task<FileSystemDirectoryHandle> GetProjectDirAsync(FileSystemDirectoryHandle root, string projectId)
    {
        using var projectsDir = await root.GetDirectoryHandle("projects", create: true);
        return await projectsDir.GetDirectoryHandle(projectId, create: true);
    }

    private async Task SaveIndexAsync()
    {
        if (_projects == null) return;
        var root = await GetRootDirAsync();
        var json = JsonSerializer.Serialize(_projects, _jsonOpts);
        await WriteTextAsync(root, "projects.json", json);
    }

    private static async Task WriteTextAsync(FileSystemDirectoryHandle dir, string name, string text)
    {
        using var fileHandle = await dir.GetFileHandle(name, create: true);
        using var writable = await fileHandle.CreateWritable();
        var bytes = Encoding.UTF8.GetBytes(text);
        using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "application/json" });
        await writable.Write(blob);
        await writable.Close();
    }

    private static async Task WriteBinaryAsync(FileSystemDirectoryHandle dir, string name, byte[] data)
    {
        using var fileHandle = await dir.GetFileHandle(name, create: true);
        using var writable = await fileHandle.CreateWritable();
        using var blob = new Blob(new byte[][] { data }, new BlobOptions { Type = "application/octet-stream" });
        await writable.Write(blob);
        await writable.Close();
    }

    /// <summary>
    /// Write a JS <see cref="Uint8Array"/> straight to OPFS. The Uint8Array is passed by reference
    /// to <c>FileSystemWritableFileStream.write</c> — the bytes never cross into the .NET managed
    /// heap. This is the scalable path for GPU-sized payloads.
    /// </summary>
    private static async Task WriteBinaryAsync(FileSystemDirectoryHandle dir, string name, Uint8Array data)
    {
        using var fileHandle = await dir.GetFileHandle(name, create: true);
        using var writable = await fileHandle.CreateWritable();
        await writable.Write(data);
        await writable.Close();
    }

    private static async Task<string?> ReadTextAsync(FileSystemDirectoryHandle dir, string name)
    {
        try
        {
            using var fileHandle = await dir.GetFileHandle(name);
            using var file = await fileHandle.GetFile();
            return await file.Text();
        }
        catch
        {
            return null;
        }
    }

    private static async Task<byte[]?> ReadBinaryAsync(FileSystemDirectoryHandle dir, string name)
    {
        try
        {
            using var fileHandle = await dir.GetFileHandle(name);
            using var file = await fileHandle.GetFile();
            using var arrayBuffer = await file.ArrayBuffer();
            using var uint8 = new Uint8Array(arrayBuffer);
            return uint8.ReadBytes();
        }
        catch
        {
            return null;
        }
    }
}
