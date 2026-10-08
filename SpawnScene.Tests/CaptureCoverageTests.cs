using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;

namespace SpawnScene.Tests;

/// <summary>CaptureCoverage: photos' horizontal headings in 8 sectors relative to the first photo, clockwise from above
/// (+Y up), and the card's wording.</summary>
public class CaptureCoverageTests
{
    static CameraParams Facing(float degreesClockwise, float pitch = 0f)
    {
        // First photo looks down -Z; clockwise from above (+Y up) turns -Z toward +X.
        float a = degreesClockwise * MathF.PI / 180f;
        var f = Vector3.Normalize(new Vector3(MathF.Sin(a), pitch, -MathF.Cos(a)));
        return new CameraParams { Width = 100, Height = 100, FocalX = 100, FocalY = 100, CenterX = 50, CenterY = 50, Position = Vector3.Zero, Forward = f, Up = Vector3.UnitY };
    }

    [Test]
    public void SectorsRunClockwiseFromTheFirstPhoto()
    {
        var c = CaptureCoverage.FacingCounts(new[] { Facing(0), Facing(90), Facing(95), Facing(180), Facing(-90), Facing(20, 0.3f) })!;
        Assert.That(c, Is.EqualTo(new[] { 2, 0, 2, 0, 1, 0, 1, 0 }));   // ahead x2 (0, 20), right x2, back, left
        Assert.That(CaptureCoverage.Describe(c), Is.EqualTo("No photos face ahead-right, back-right, back-left, ahead-left of the first photo"));
    }

    [Test]
    public void AllRoundAndFrontOnlyCaptures()
    {
        var round = Enumerable.Range(0, 16).Select(k => Facing(k * 22.5f)).ToList();
        Assert.That(CaptureCoverage.Describe(CaptureCoverage.FacingCounts(round)!), Is.EqualTo("Photos face every direction"));
        var front = new[] { Facing(0), Facing(10), Facing(-15), Facing(30) };
        Assert.That(CaptureCoverage.Describe(CaptureCoverage.FacingCounts(front)!), Is.Null, "a front-facing capture's gaps are not news");
        Assert.That(CaptureCoverage.FacingCounts(new[] { Facing(0) }), Is.Null);
    }
}
