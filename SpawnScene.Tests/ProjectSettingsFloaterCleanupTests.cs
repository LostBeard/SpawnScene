using System.Text.Json;
using NUnit.Framework;
using SpawnScene.Models;

namespace SpawnScene.Tests;

/// <summary>
/// ProjectSettings.FloaterCleanup (2026-10-13, the free-space default): a project saved before the setting existed has no
/// "floaterCleanup" in its JSON and must load with it ON (the measured default), not off; an explicit Off stays Off.
/// Same options as ProjectService (camelCase).
/// </summary>
public class ProjectSettingsFloaterCleanupTests
{
    static readonly JsonSerializerOptions Opts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    [Test]
    public void OldProjectWithoutTheSettingLoadsWithCleanupOn()
    {
        var s = JsonSerializer.Deserialize<ProjectSettings>("{\"depthModel\":\"depth-anything-v3-small\",\"trainIterations\":7000}", Opts)!;
        Assert.That(s.FloaterCleanup, Is.True);
    }

    [Test]
    public void ExplicitOffStaysOffThroughARoundTrip()
    {
        var s = JsonSerializer.Deserialize<ProjectSettings>("{\"floaterCleanup\":false}", Opts)!;
        Assert.That(s.FloaterCleanup, Is.False);
        var again = JsonSerializer.Deserialize<ProjectSettings>(JsonSerializer.Serialize(s, Opts), Opts)!;
        Assert.That(again.FloaterCleanup, Is.False);
        Assert.That(JsonSerializer.Serialize(new ProjectSettings(), Opts), Does.Contain("\"floaterCleanup\":true"));
    }
}
