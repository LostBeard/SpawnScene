using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>SplatBounds.QuantileRange (robust bounds for XR AR placement, 2026-10-03).</summary>
public class SplatBoundsQuantileTests
{
    const int B = SplatBounds.RobustBins;

    static int[] Axis() => new int[B + 2];

    [Test]
    public void Uniform_TrimsOnePercentEachEnd()
    {
        var h = Axis();
        for (int i = 0; i < B; i++) h[1 + i] = 100;
        float inv = B / 10f;   // [0, 10)
        var (lo, hi) = SplatBounds.QuantileRange(h, 0f, inv, 0.01);
        Assert.That(lo, Is.EqualTo(0.1f).Within(0.02f));
        Assert.That(hi, Is.EqualTo(9.9f).Within(0.02f));
    }

    [Test]
    public void Floaters_DoNotStretchTheRange()
    {
        // 10,000 splats in bins 100..199, 20 floaters in the last bin and 30 past the range.
        var h = Axis();
        for (int i = 100; i < 200; i++) h[1 + i] = 100;
        h[B] = 20;
        h[B + 1] = 30;
        float inv = 1f;   // a bin per unit, from -50
        var (lo, hi) = SplatBounds.QuantileRange(h, -50f, inv, 0.01);
        // The 1% cut is 100.5 of 10,050: past the first populated bin's 100, so in the second.
        Assert.That(lo, Is.EqualTo(-50f + 101).Within(1e-4f), "start of the bin holding the 1% quantile");
        Assert.That(hi, Is.EqualTo(-50f + 200).Within(1e-4f), "end of the last populated bin, floaters cut");
    }

    [Test]
    public void Range_AlwaysContainsTheQuantiles()
    {
        // One bin holding everything: low edge and high edge of that same bin.
        var h = Axis();
        h[1 + 500] = 1000;
        var (lo, hi) = SplatBounds.QuantileRange(h, 0f, 1f, 0.01);
        Assert.That(lo, Is.EqualTo(500f));
        Assert.That(hi, Is.EqualTo(501f));
    }

    [Test]
    public void Empty_ReturnsTheHistogramRange()
    {
        var (lo, hi) = SplatBounds.QuantileRange(Axis(), 2f, 0.5f, 0.01);
        Assert.That(lo, Is.EqualTo(2f));
        Assert.That(hi, Is.EqualTo(2f + B / 0.5f));
    }

    [Test]
    public void MedianOfHistogram_FindsTheMiddleDistance()
    {
        var h = new int[100];
        h[10] = 1; h[20] = 5; h[80] = 2;            // 8 samples: the 4th is in bin 20
        Assert.That(SplatBounds.MedianOfHistogram(h, 10f), Is.EqualTo(2.05f).Within(1e-5f));
        Assert.That(SplatBounds.MedianOfHistogram(new int[10], 1f), Is.Null);
    }

    [Test]
    public void ViewDistanceScale_PutsTheSubjectThreeMetresAway()
    {
        // The Truck start view: its subject 1.242 units away -> 0.414 units per metre (it was 1.72 from the bounds).
        Assert.That(XRSceneAlignment.ComfortScale(1.242f), Is.EqualTo(1.242f / 3f).Within(1e-5f));
        Assert.That(XRSceneAlignment.ComfortScale(0f), Is.EqualTo(1f), "nothing in view: scale 1");
    }
}
