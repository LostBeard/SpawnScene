using System.Text.Json;
using NUnit.Framework;
using SpawnScene.Models;

namespace SpawnScene.Tests;

/// <summary>
/// ProjectSettings.FillUnseen (2026-10-15, the unseen-direction fill as a user setting): invented content, so a project saved
/// before the setting existed has no "fillUnseen" in its JSON and must load with it OFF; an explicit On stays On.
/// Same options as ProjectService (camelCase).
/// </summary>
public class ProjectSettingsFillUnseenTests
{
    static readonly JsonSerializerOptions Opts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    [Test]
    public void OldProjectWithoutTheSettingLoadsWithFillOff()
    {
        var s = JsonSerializer.Deserialize<ProjectSettings>("{\"depthModel\":\"depth-anything-v3-small\",\"trainIterations\":7000}", Opts)!;
        Assert.That(s.FillUnseen, Is.False);
    }

    [Test]
    public void ExplicitOnStaysOnThroughARoundTrip()
    {
        var s = JsonSerializer.Deserialize<ProjectSettings>("{\"fillUnseen\":true}", Opts)!;
        Assert.That(s.FillUnseen, Is.True);
        var again = JsonSerializer.Deserialize<ProjectSettings>(JsonSerializer.Serialize(s, Opts), Opts)!;
        Assert.That(again.FillUnseen, Is.True);
        Assert.That(JsonSerializer.Serialize(new ProjectSettings(), Opts), Does.Contain("\"fillUnseen\":false"));
    }
}
