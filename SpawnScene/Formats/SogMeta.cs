using System.Text.Json;

namespace SpawnScene.Formats;

/// <summary>
/// The meta.json of PlayCanvas's SOG ("spatially ordered gaussians": a zip of lossless WebP textures; splat-transform
/// writes it, SuperSplat exports it). playcanvas/engine gsplat-sog-data.js reads it; version 2 quantises scales, SH DC and
/// SH rest through 256-entry codebooks, version 1 (no "version", a "shape") through min/max ranges. Parsed with
/// JsonDocument (no reflection: trim/AOT safe).
/// </summary>
public sealed class SogMeta
{
    public int Version { get; init; }
    public int Count { get; init; }
    public float[] MeansMin { get; init; } = new float[3];
    public float[] MeansMax { get; init; } = new float[3];
    public string[] MeansFiles { get; init; } = Array.Empty<string>();
    public float[]? ScalesCodebook { get; init; }
    public float[] ScalesMin { get; init; } = new float[3];
    public float[] ScalesMax { get; init; } = new float[3];
    public string[] ScalesFiles { get; init; } = Array.Empty<string>();
    public string[] QuatsFiles { get; init; } = Array.Empty<string>();
    public float[]? Sh0Codebook { get; init; }
    public float[] Sh0Min { get; init; } = new float[4];
    public float[] Sh0Max { get; init; } = new float[4];
    public string[] Sh0Files { get; init; } = Array.Empty<string>();
    public float[]? ShNCodebook { get; init; }
    public float ShNMin { get; init; }
    public float ShNMax { get; init; }
    /// <summary>[centroids, labels], or empty when the scene has no SH rest bands.</summary>
    public string[] ShNFiles { get; init; } = Array.Empty<string>();

    static float[] Floats(JsonElement e) => e.EnumerateArray().Select(x => x.GetSingle()).ToArray();
    static string[] Files(JsonElement o) => o.TryGetProperty("files", out var f) ? f.EnumerateArray().Select(x => x.GetString() ?? "").ToArray() : Array.Empty<string>();
    static float[]? Codebook(JsonElement o) => o.TryGetProperty("codebook", out var c) ? Floats(c) : null;
    static float[] Arr(JsonElement o, string name, int n) => o.TryGetProperty(name, out var a) && a.ValueKind == JsonValueKind.Array ? Floats(a) : new float[n];

    public static SogMeta Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        int version = r.TryGetProperty("version", out var v) ? v.GetInt32() : 1;
        if (version is < 1 or > 2) throw new FormatException($"SOG version {version} is not read yet (1 and 2 are)");
        var means = r.GetProperty("means");
        int count = r.TryGetProperty("count", out var c) ? c.GetInt32()
            : means.TryGetProperty("shape", out var shape) ? shape[0].GetInt32() : throw new FormatException("SOG meta.json has no count");
        var scales = r.GetProperty("scales");
        var sh0 = r.GetProperty("sh0");
        bool hasShN = r.TryGetProperty("shN", out var shN);
        var meta = new SogMeta
        {
            Version = version, Count = count,
            MeansMin = Arr(means, "mins", 3), MeansMax = Arr(means, "maxs", 3), MeansFiles = Files(means),
            ScalesCodebook = Codebook(scales), ScalesMin = Arr(scales, "mins", 3), ScalesMax = Arr(scales, "maxs", 3), ScalesFiles = Files(scales),
            QuatsFiles = Files(r.GetProperty("quats")),
            Sh0Codebook = Codebook(sh0), Sh0Min = Arr(sh0, "mins", 4), Sh0Max = Arr(sh0, "maxs", 4), Sh0Files = Files(sh0),
            ShNCodebook = hasShN ? Codebook(shN) : null,
            ShNMin = hasShN && shN.TryGetProperty("mins", out var nmin) && nmin.ValueKind == JsonValueKind.Number ? nmin.GetSingle() : 0f,
            ShNMax = hasShN && shN.TryGetProperty("maxs", out var nmax) && nmax.ValueKind == JsonValueKind.Number ? nmax.GetSingle() : 0f,
            ShNFiles = hasShN ? Files(shN) : Array.Empty<string>(),
        };
        if (meta.MeansFiles.Length != 2 || meta.QuatsFiles.Length != 1 || meta.ScalesFiles.Length != 1 || meta.Sh0Files.Length != 1)
            throw new FormatException("SOG meta.json does not list the textures it should");
        if (version == 2 && (meta.ScalesCodebook?.Length != 256 || meta.Sh0Codebook?.Length != 256 || (meta.ShNFiles.Length > 0 && meta.ShNCodebook?.Length != 256)))
            throw new FormatException("SOG v2 meta.json is missing a 256-entry codebook");
        return meta;
    }

    /// <summary>SH rest bands from the centroid texture's width (64 palette entries a row): 192 -> 1, 512 -> 2, 960 -> 3.</summary>
    public static int BandsForCentroidWidth(int width) => width switch { 192 => 1, 512 => 2, 960 => 3, _ => 0 };
}
