namespace SpawnScene.Models;

/// <summary>
/// The project page's "Or try a sample" list: <c>wwwroot/samples/catalog.json</c>. The photos themselves are hosted
/// elsewhere (<see cref="Base"/>), so the app deploy stays small and a sample can be added without a rebuild of the
/// list's code. Every entry carries its credit and license; loading one stores the credit on the project.
/// </summary>
public sealed class SampleCatalog
{
    /// <summary>Absolute URL every sample's <see cref="SampleEntry.Folder"/> is relative to (ends in '/').</summary>
    public string Base { get; set; } = "";
    public List<SampleEntry> Samples { get; set; } = new();
}

public sealed class SampleEntry
{
    public string Name { get; set; } = "";
    /// <summary>"photo" (one image: a depth scene) or "set" (several photos of one scene: a reconstruction).</summary>
    public string Kind { get; set; } = "set";
    /// <summary>Folder under <see cref="SampleCatalog.Base"/> holding <see cref="Images"/>.</summary>
    public string Folder { get; set; } = "";
    public List<string> Images { get; set; } = new();
    /// <summary>Total download size, for the button ("59 photos, 38 MB").</summary>
    public long Bytes { get; set; }
    /// <summary>Who made the photos, as the license asks to be credited.</summary>
    public string Credit { get; set; } = "";
    public string License { get; set; } = "";
    public string LicenseUrl { get; set; } = "";
    /// <summary>Where the photos came from (the source page), for the credit line.</summary>
    public string Source { get; set; } = "";

    public bool IsSet => Kind == "set";
}
