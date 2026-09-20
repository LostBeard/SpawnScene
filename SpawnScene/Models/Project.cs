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

    /// <summary>
    /// Floats per splat in this scene's .bin. Absent in records written before splats carried a
    /// rotation, so it deserializes to 0 and <see cref="EffectiveFloatsPerSplat"/> treats it as
    /// the old 10-float layout rather than silently reading the file at the wrong stride.
    /// </summary>
    public int FloatsPerSplat { get; set; }

    /// <summary>Stride to read this scene's .bin at. 10 = pre-rotation layout, needs widening.</summary>
    [JsonIgnore]
    public int EffectiveFloatsPerSplat => FloatsPerSplat > 0 ? FloatsPerSplat : LegacyFloatsPerSplat;

    /// <summary>The packed layout before a per-splat quaternion existed: pos3 color3 scale3 opacity1.</summary>
    public const int LegacyFloatsPerSplat = 10;
}

/// <summary>Per-project generation and render settings.</summary>
public class ProjectSettings
{
    public string DepthModel { get; set; } = "depth-anything-v3-small";
    public string QualityPreset { get; set; } = "Standard";
    public int Subsample { get; set; } = 2;
    public float EdgeSharpness { get; set; } = 0.3f;
    // Parked for a future NATIVE super-resolution pass (ORT SR retired 2026-07-01). See SuperResolutionService.cs.
    public bool UseSuperResolution { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SplatRenderMode RenderMode { get; set; } = SplatRenderMode.Stochastic;
    public float SharpeningStrength { get; set; } = 0.5f;
}
