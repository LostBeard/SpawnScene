using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// A pair's true matches obey one epipolar geometry; chance matches do not. MEASURED on Truck: ~34 chance
/// matches on every pair (adjacent frames 216) poisoned bundle adjustment's tracks.
/// </summary>
public class EpipolarRansacTests
{
    static CameraParams Cam(Vector3 pos, Vector3 target) => new()
    {
        Width = 979, Height = 546, FocalX = 800, FocalY = 800, CenterX = 489.5f, CenterY = 273f,
        Position = pos, Forward = Vector3.Normalize(target - pos),
        Up = Vector3.UnitY,
    };

    static (float[] a, float[] b, bool[] truth) Pair(int inliers, int outliers, int seed)
    {
        var rng = new Random(seed);
        var c1 = Cam(new Vector3(0, 1, 8), new Vector3(0, 0.5f, 0));
        var c2 = Cam(new Vector3(1.4f, 1.1f, 7.8f), new Vector3(0.1f, 0.5f, 0));   // ~10 deg apart, like Truck neighbours
        var a = new List<float>(); var b = new List<float>(); var t = new List<bool>();
        while (t.Count(x => x) < inliers)
        {
            var p = new Vector3((float)(rng.NextDouble() * 6 - 3), (float)(rng.NextDouble() * 3 - 1), (float)(rng.NextDouble() * 3 - 1.5));
            if (!WorldSpaceGeometry.Project(c1, p, out var u1, out var v1, out _) || !WorldSpaceGeometry.Project(c2, p, out var u2, out var v2, out _)) continue;
            if (u1 < 0 || v1 < 0 || u1 >= 979 || v1 >= 546 || u2 < 0 || v2 < 0 || u2 >= 979 || v2 >= 546) continue;
            a.Add(u1 + (float)(rng.NextDouble() - 0.5)); a.Add(v1 + (float)(rng.NextDouble() - 0.5));
            b.Add(u2 + (float)(rng.NextDouble() - 0.5)); b.Add(v2 + (float)(rng.NextDouble() - 0.5));
            t.Add(true);
        }
        for (int i = 0; i < outliers; i++)
        {
            a.Add((float)(rng.NextDouble() * 979)); a.Add((float)(rng.NextDouble() * 546));
            b.Add((float)(rng.NextDouble() * 979)); b.Add((float)(rng.NextDouble() * 546));
            t.Add(false);
        }
        return (a.ToArray(), b.ToArray(), t.ToArray());
    }

    [Test]
    public void RecoversTheTrueMatches_AmongForty_PercentOutliers()
    {
        var (a, b, truth) = Pair(inliers: 120, outliers: 80, seed: 5);
        var r = EpipolarRansac.Estimate(a, b, thresholdPx: 2.0);
        Assert.That(r, Is.Not.Null);
        int tp = 0, fp = 0;
        for (int i = 0; i < truth.Length; i++) { if (r!.Inliers[i] && truth[i]) tp++; if (r.Inliers[i] && !truth[i]) fp++; }
        TestContext.Out.WriteLine($"inliers {r!.InlierCount} (true {tp}, false {fp}) in {r.Iterations} iterations");
        Assert.That(tp, Is.GreaterThanOrEqualTo(114), "recall >= 95%");
        Assert.That(fp, Is.LessThanOrEqualTo(6), "chance matches passing a 2 px epipolar band");
    }

    [Test]
    public void PureNoisePair_IsRejected()
    {
        // The Truck noise floor: ~34 chance matches between frames that share nothing.
        var (a, b, _) = Pair(inliers: 0, outliers: 34, seed: 9);
        var r = EpipolarRansac.Estimate(a, b, thresholdPx: 2.0, minInliers: 15);
        TestContext.Out.WriteLine(r == null ? "rejected" : $"accepted with {r.InlierCount}");
        Assert.That(r, Is.Null, "a pair of chance matches must not verify");
    }
}
