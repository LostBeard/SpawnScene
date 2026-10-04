using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>The .spawnscene container header (SceneFile, 2026-10-03).</summary>
public class SceneFileTests
{
    [Test]
    public void Prefix_RoundTrips()
    {
        var h = new SceneFile.Header("Truck", 508_807, 14, true, 2, 3, 2100, new DateTime(2026, 10, 3, 12, 0, 0, DateTimeKind.Utc));
        var prefix = SceneFile.Prefix(h);
        int len = SceneFile.HeaderLength(prefix.AsSpan(0, 12));
        Assert.That(SceneFile.DataOffset(len), Is.EqualTo(prefix.Length));
        Assert.That(SceneFile.ParseHeader(prefix.AsSpan(12, len)), Is.EqualTo(h));
    }

    [Test]
    public void NotASceneFile_Throws()
    {
        var png = new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0 };
        Assert.Throws<InvalidDataException>(() => SceneFile.HeaderLength(png));
    }
}
