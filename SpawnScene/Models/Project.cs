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
    /// <summary>Attribution for photos that came from a sample (SampleCatalog): "Name - credit, license, source".
    /// Shown on the project page; null for the user's own photos.</summary>
    public string? Credit { get; set; }
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
    /// <summary>The scene this one was edited from (viewer Edit tools, then Save), or null.</summary>
    public string? EditedFrom { get; set; }
    /// <summary>Where the viewer (and so the headset) starts: position, forward, up (9 floats), or null. A saved scene
    /// keeps no training cameras, and the single-photo default (origin, looking +Z) means nothing for an SfM
    /// reconstruction - a reopened Truck started inside the truck.</summary>
    public float[]? HomeView { get; set; }

    /// <summary>A single-photo scene's tan of half field of view (x, y) - the viewer's lens at home (GaussianScene.PhotoHalfTan).</summary>
    public float[]? PhotoHalfTan { get; set; }

    /// <summary>
    /// How the scene is stored: null = packed rows (scenes/{id}.bin, SH beside it), opened whole; <see cref="FormatLod"/>
    /// = its LOD tree as a .spawnscene v3 (scenes/{id}.spawnscene), opened STREAMED - a scene larger than one GPU holds
    /// (a partitioned run, Studio.Partition) is kept and viewed only this way.
    /// </summary>
    public string? Format { get; set; }

    public const string FormatLod = "lod";
    /// <summary>
    /// How the SH bands are stored: SphericalHarmonics.Parts files (scenes/{id}.sh{p}.bin, PartFloatsPerSplat floats a
    /// splat each), or 0 for a scene saved before the split (one row-major scenes/{id}.sh.bin, 45 floats a splat).
    /// </summary>
    public int ShParts { get; set; }

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
/// photos are". MEASURED 2026-10-02, TruckFull trainer held-out: 7K 23.20, 15K 24.11, 30K 24.51 dB (captured at 1600x892:
/// 23.10 / 23.38 / 23.44, SSIM .831 / .845 / .845).
/// </summary>
public static class ReconstructionPresets
{
    /// <summary>A training resolution larger than any photo: TrainingSize caps it at the photos' own size.</summary>
    public const int PhotoSize = 16384;

    /// <summary>
    /// A splat cap larger than any device holds: GpuMemoryBudget.Derive caps it at what this device's GPU memory setting
    /// and binding limit fit. Training stops growing a scene on its own once the photos are covered (TruckFull levelled
    /// at 1.74M), so a high cap costs memory only on scenes that need it. Very large scenes are a goal (TJ, 2026-10-02:
    /// a 5K single-photo scene renders 14M splats at 60 fps); the old fixed 3M cap was not.
    /// </summary>
    public const int DeviceMaxSplats = int.MaxValue;

    /// <summary>The cap every preset but Draft had before <see cref="DeviceMaxSplats"/> (2026-10-02).</summary>
    const int LegacyPresetMaxSplats = 3_000_000;

    /// <summary>
    /// Learned keypoints per photo (LearnedFeatureMatcher.KeypointBudget; Kornia publishes 1024 and 3072). MEASURED
    /// 2026-10-03: DrJohnson (wide baselines) 3072 vs 1024 placed 30 vs 27 cameras and held out 18.74 vs 15.43 dB (b134
    /// vs b92); TruckFull (well covered) 23.31 vs 23.24 dB, sparse cloud 19.5K vs 8.5K points (b140 vs b138), for ~6x the
    /// matching time (251 photos: LightGlue 20 min vs 3.5 min; retrieval stays ~1 min at either, b141).
    /// </summary>
    public const int StandardKeypoints = 1024, HighKeypoints = 3072;

    public static readonly (string Name, int Iterations, int MaxDimension, int MaxSplats, int Keypoints, string Hint)[] All =
    {
        ("Draft", 3000, 720, 500_000, StandardKeypoints, "A quick look: about a third of Standard's training."),
        ("Standard", 7000, 1024, DeviceMaxSplats, StandardKeypoints, "The reference's first checkpoint. Right for most captures."),
        ("High", 15000, 1600, DeviceMaxSplats, HighKeypoints, "About twice Standard's training, and 3072 keypoints a photo: more cameras placed on wide-baseline captures (DrJohnson +3.3 dB), ~0.9 dB sharper on a well-covered one (TruckFull)."),
        ("Max", 30000, PhotoSize, DeviceMaxSplats, HighKeypoints, "The full reference run at the photos' own size with 3072 keypoints: about twice High's training time, for a little more (+0.4 dB)."),
    };

    /// <summary>
    /// A project saved under a preset before that preset's current values moves to them: the old fixed 3M splat cap, and
    /// the 1024 keypoints every preset had before High and Max took 3072. Anything set by hand stays. Returns true when
    /// it changed <paramref name="s"/>.
    /// </summary>
    public static bool UpgradeLegacyPresetValues(ProjectSettings s)
    {
        foreach (var p in All)
        {
            if (p.Name != s.ReconstructionPreset) continue;
            if (s.TrainIterations != p.Iterations || s.TrainMaxDimension != p.MaxDimension) return false;
            bool capOk = s.TrainMaxSplats == p.MaxSplats || s.TrainMaxSplats == LegacyPresetMaxSplats;
            bool keypointsOk = s.LearnedKeypoints == p.Keypoints || s.LearnedKeypoints == StandardKeypoints;
            if (!capOk || !keypointsOk) return false;
            bool changed = s.TrainMaxSplats != p.MaxSplats || s.LearnedKeypoints != p.Keypoints;
            s.TrainMaxSplats = p.MaxSplats;
            s.LearnedKeypoints = p.Keypoints;
            return changed;
        }
        return false;
    }

    /// <summary>Apply preset <paramref name="name"/> to <paramref name="s"/>; false if there is no such preset.</summary>
    public static bool Apply(ProjectSettings s, string name)
    {
        foreach (var p in All)
        {
            if (p.Name != name) continue;
            s.TrainIterations = p.Iterations;
            s.TrainMaxDimension = p.MaxDimension;
            s.TrainMaxSplats = p.MaxSplats;
            s.LearnedKeypoints = p.Keypoints;
            s.ReconstructionPreset = p.Name;
            return true;
        }
        return false;
    }

    /// <summary>The preset whose values the settings hold, or "Custom".</summary>
    public static string Match(ProjectSettings s)
    {
        foreach (var p in All)
            if (s.TrainIterations == p.Iterations && s.TrainMaxDimension == p.MaxDimension && s.TrainMaxSplats == p.MaxSplats
                && s.LearnedKeypoints == p.Keypoints)
                return p.Name;
        return "Custom";
    }
}

/// <summary>Per-project generation and render settings.</summary>
public enum SuperResolutionMode { Off, Auto, On }

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
    /// Ceiling on the splat count while training grows the scene (densification). The device's GPU memory setting caps
    /// it further (GpuMemoryBudget); <see cref="ReconstructionPresets.DeviceMaxSplats"/> = as many as the device fits.
    /// </summary>
    public int TrainMaxSplats { get; set; } = ReconstructionPresets.DeviceMaxSplats;
    /// <summary>
    /// Longest side, in pixels, the photos are trained at (the trainer also shrinks further to fit its target
    /// memory budget). Higher = sharper detail, more GPU memory and time per iteration.
    /// </summary>
    public int TrainMaxDimension { get; set; } = 1024;

    /// <summary>Learned keypoints per photo for matching (ReconstructionPresets.StandardKeypoints / HighKeypoints).</summary>
    public int LearnedKeypoints { get; set; } = ReconstructionPresets.StandardKeypoints;

    /// <summary>
    /// The multi-photo quality preset these training settings came from (<see cref="ReconstructionPresets"/>), or
    /// "Custom" once any of them was changed by hand.
    /// </summary>
    public string ReconstructionPreset { get; set; } = "Standard";
    /// <summary>Was the parked flag of the retired ORT super-resolution; superseded by <see cref="SuperResolution"/>.</summary>
    public bool UseSuperResolution { get; set; }

    /// <summary>Super-resolution of a single photo before it becomes splats (Studio.SuperRes): Auto = photos under 800 px.</summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SuperResolutionMode SuperResolution { get; set; } = SuperResolutionMode.Auto;

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SplatRenderMode RenderMode { get; set; } = SplatRenderMode.Sorted;
    public float SharpeningStrength { get; set; } = 0.5f;
}
