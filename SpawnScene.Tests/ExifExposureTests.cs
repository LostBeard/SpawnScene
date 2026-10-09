using NUnit.Framework;
using SpawnScene.Models;

namespace SpawnScene.Tests;

/// <summary>
/// ExifReader.ExtractExposure on real photos: TJ's Bathroom (phone, auto exposure - 6.1 stops of spread, where per-photo
/// gains are worth 5.9 dB fair) and a benchmark JPG (no EXIF). &amp;exposure=auto turns gains on from this spread
/// (Studio.Training.ExposureAutoOption; parity 2026-10-09).
/// </summary>
public class ExifExposureTests
{
    static string DatasetDir(string dataset) => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
        "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", dataset));

    [Test]
    public void Bathroom_ExposureVariesByStops()
    {
        var dir = DatasetDir("Bathroom");
        if (!Directory.Exists(dir)) Assert.Ignore("Bathroom is not in this checkout");
        var stops = Directory.GetFiles(dir, "*.jpg").Select(f => ExifReader.ExtractExposure(File.ReadAllBytes(f))?.Stops)
            .ToList();
        Assert.That(stops.All(s => s.HasValue), "every phone photo carries exposure time, f-number and ISO");
        float spread = stops.Max()!.Value - stops.Min()!.Value;
        // Measured with PIL 2026-10-09: 6.13 stops over the 35 photos.
        Assert.That(spread, Is.EqualTo(6.13f).Within(0.02f));
        Assert.That(spread, Is.GreaterThanOrEqualTo(1f / 3f), "Studio.ExposureAutoMinStops");
    }

    [Test]
    public void FirstBathroomPhoto_Values()
    {
        var f = Path.Combine(DatasetDir("Bathroom"), "IMG_20260223_133436884.jpg");
        if (!File.Exists(f)) Assert.Ignore("Bathroom is not in this checkout");
        var e = ExifReader.ExtractExposure(File.ReadAllBytes(f));
        Assert.That(e, Is.Not.Null);
        Assert.That(e!.ExposureTime, Is.EqualTo(1f / 30f).Within(1e-6f));
        Assert.That(e.Iso, Is.EqualTo(92f));
        Assert.That(e.FNumber, Is.EqualTo(2f).Within(1e-6f));
    }

    [Test]
    public void NotAJpeg_IsNull() => Assert.That(ExifReader.ExtractExposure(new byte[64]), Is.Null);
}
