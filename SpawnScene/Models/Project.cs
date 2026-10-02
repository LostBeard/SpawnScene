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
    /// <summary>
    /// True for a TRAINED scene: the packed colour slots hold SH DC coefficients, not RGB, and the viewer must
    /// convert (GpuGaussianRenderer.ColoursAreShDc). False for every generated-only scene.
    /// </summary>
    public bool ColoursAreShDc { get; set; }
    /// <summary>SH degree of the saved rest bands (0 = none). The bands are in scenes/{Id}.sh.bin.</summary>
    public int ShDegree { get; set; }
    /// <summary>Training iterations this scene received (0 = not trained).</summary>
    public int TrainedIterations { get; set; }

    /// <summary>Stride to read this scene's .bin at. 10 = pre-rotation layout, needs widening.</summary>
    [JsonIgnore]
    public int EffectiveFloatsPerSplat => FloatsPerSplat > 0 ? FloatsPerSplat : LegacyFloatsPerSplat;

    /// <summary>The packed layout before a per-splat quaternion existed: pos3 color3 scale3 opacity1.</summary>
    public const int LegacyFloatsPerSplat = 10;
}

/// <summary>
/// Multi-photo quality presets: one choice sets iterations, training resolution and the splat cap together
/// (Research/project-settings-presets-2026-10-02.md). Iterations are the main time/quality dial (TruckFull 7K: 23.1 dB,
/// sharpness 0.98); resolution is capped at the photos' own size, so <see cref="PhotoSize"/> means "as large as the
/// photos are".
/// </summary>
public static class ReconstructionPresets
{
    /// <summary>A training resolution larger than any photo: TrainingSize caps it at the photos' own size.</summary>
    public const int PhotoSize = 16384;

    public static readonly (string Name, int Iterations, int MaxDimension, int MaxSplats, string Hint)[] All =
    {
        ("Draft", 3000, 720, 500_000, "A quick look: about a third of Standard's training."),
        ("Standard", 7000, 1024, 3_000_000, "The reference's first checkpoint. Right for most captures."),
        ("High", 15000, 1600, 3_000_000, "Longer training at up to 1600 px. Pays off when the photos cover the scene densely."),
        ("Max", 30000, PhotoSize, 3_000_000, "The reference's full run at the photos' own size. Slowest."),
    };

    /// <summary>Apply preset <paramref name="name"/> to <paramref name="s"/>; false if there is no such preset.</summary>
    public static bool Apply(ProjectSettings s, string name)
    {
        foreach (var p in All)
        {
            if (p.Name != name) continue;
            s.TrainIterations = p.Iterations;
            s.TrainMaxDimension = p.MaxDimension;
            s.TrainMaxSplats = p.MaxSplats;
            s.ReconstructionPreset = p.Name;
            return true;
        }
        return false;
    }

    /// <summary>The preset whose values the settings hold, or "Custom".</summary>
    public static string Match(ProjectSettings s)
    {
        foreach (var p in All)
            if (s.TrainIterations == p.Iterations && s.TrainMaxDimension == p.MaxDimension && s.TrainMaxSplats == p.MaxSplats)
                return p.Name;
        return "Custom";
    }
}

/// <summary>Per-project generation and render settings.</summary>
public class ProjectSettings
{
    public string DepthModel { get; set; } = "depth-anything-v3-small";
    public string QualityPreset { get; set; } = "Standard";
    public int Subsample { get; set; } = 2;
    public float EdgeSharpness { get; set; } = 0.3f;
    /// <summary>
    /// Optimiser iterations after a multi-image generation (0 = do not train). 7,000 is the reference's
    /// first checkpoint; 30,000 is its full run (TruckFull 251 photos: ~70 min training in the browser).
    /// </summary>
    public int TrainIterations { get; set; } = 7000;
    /// <summary>
    /// Ceiling on the splat count while training grows the scene (densification). Bounds GPU memory: TruckFull
    /// reached 1.7M under a 3M cap (b24). Lower it on a smaller GPU.
    /// </summary>
    public int TrainMaxSplats { get; set; } = 3_000_000;
    /// <summary>
    /// Longest side, in pixels, the photos are trained at (the trainer also shrinks further to fit its target
    /// memory budget). Higher = sharper detail, more GPU memory and time per iteration.
    /// </summary>
    public int TrainMaxDimension { get; set; } = 1024;

    /// <summary>
    /// The multi-photo quality preset these training settings came from (<see cref="ReconstructionPresets"/>), or
    /// "Custom" once any of them was changed by hand.
    /// </summary>
    public string ReconstructionPreset { get; set; } = "Standard";
    // Parked for a future NATIVE super-resolution pass (ORT SR retired 2026-07-01). See SuperResolutionService.cs.
    public bool UseSuperResolution { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SplatRenderMode RenderMode { get; set; } = SplatRenderMode.Stochastic;
    public float SharpeningStrength { get; set; } = 0.5f;
}
