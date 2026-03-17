using SpawnScene.Services;
using System.Text.Json.Serialization;

namespace SpawnScene.Models;

/// <summary>
/// A Gaussian Splat project containing source media, generated scenes, and settings.
/// Persisted to OPFS via ProjectService.
/// </summary>
public class Project
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..12];
    public string Name { get; set; } = "Untitled";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime ModifiedAt { get; set; } = DateTime.UtcNow;
    public ProjectSettings Settings { get; set; } = new();
    public List<ProjectSource> Sources { get; set; } = new();
    public List<ProjectScene> Scenes { get; set; } = new();
}

/// <summary>Source image or video in a project.</summary>
public class ProjectSource
{
    public string FileName { get; set; } = "";
    public long SizeBytes { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
}

/// <summary>A generated Gaussian splat scene within a project.</summary>
public class ProjectScene
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public long SizeBytes { get; set; }
    public int SplatCount { get; set; }
    public string QualityPreset { get; set; } = "Standard";
}

/// <summary>Per-project generation and render settings.</summary>
public class ProjectSettings
{
    public string DepthModel { get; set; } = "depth-anything-v3-small";
    public string QualityPreset { get; set; } = "Standard";
    public int Subsample { get; set; } = 2;
    public float EdgeSharpness { get; set; } = 0.3f;
    public bool UseSuperResolution { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SplatRenderMode RenderMode { get; set; } = SplatRenderMode.Stochastic;
    public float SharpeningStrength { get; set; } = 0.5f;
}
