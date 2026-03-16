using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
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
            if (data != null)
                _projects = JsonSerializer.Deserialize<List<Project>>(data, _jsonOpts) ?? new();
            else
                _projects = new();
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

    /// <summary>Save a scene thumbnail image.</summary>
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

    // ─── OPFS Helpers ───

    private async Task<FileSystemDirectoryHandle> GetRootDirAsync()
    {
        using var navigator = BlazorJSRuntime.JS.Get<Navigator>("navigator");
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
