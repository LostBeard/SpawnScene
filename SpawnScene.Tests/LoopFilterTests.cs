using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// GlobalSfmInit.FilterByLoops (2026-09-30): DrJohnson's verified pairs included 31% repeated-structure pairs, wrong by
/// 90-180 deg yet geometrically consistent in two views. Triplet consistency must remove them and keep the true pairs.
/// </summary>
public class LoopFilterTests
{
    static double[] Rot(Random rng, double maxDeg, double minDeg = 0)
    {
        double ax = rng.NextDouble() * 2 - 1, ay = rng.NextDouble() * 2 - 1, az = rng.NextDouble() * 2 - 1;
        double len = Math.Sqrt(ax * ax + ay * ay + az * az); ax /= len; ay /= len; az /= len;
        double t = (minDeg + rng.NextDouble() * (maxDeg - minDeg)) * Math.PI / 180, c = Math.Cos(t), s = Math.Sin(t), C = 1 - c;
        return new[] { c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s,
                       ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s,
                       az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C };
    }

    /// <summary>DrJohnson b67 (2026-09-30): 3 verified pairs, none in a consistent triangle -> no observations for the
    /// positioning; the GPU positioner was built, returned at once and was disposed with a PENDING clear, which broke the
    /// next unrelated GPU submit. With nothing to solve, positioning must be skipped (and say so), not attempted.</summary>
    [Test]
    public void Apply_NoObservations_SkipsPositioning()
    {
        var cams = Enumerable.Range(0, 4).Select(i => new SpawnScene.Models.CameraParams
        {
            Width = 640, Height = 480, FocalX = 500, FocalY = 500, CenterX = 320, CenterY = 240,
            Position = new System.Numerics.Vector3(i, 0, 0),
        }).ToList();
        var rng = new Random(1);
        var edges = new List<GlobalSfmInit.RelativePose>
        {
            new(0, 1, Rot(rng, 5), new double[] { 1, 0, 0 }, 40),
            new(2, 3, Rot(rng, 5), new double[] { 1, 0, 0 }, 40),
        };
        string summary = GlobalSfmInit.Apply(cams, edges, new List<BundleAdjuster.Observation>(), 0, 500);
        TestContext.Out.WriteLine(summary);
        Assert.That(summary, Does.Contain("positioning skipped"));
    }

    [Test]
    public void RepeatedStructurePairs_AreRemoved_TruePairsKept()
    {
        var rng = new Random(3);
        int n = 60;
        var abs = Enumerable.Range(0, n).Select(_ => Rot(rng, 180)).ToArray();
        var edges = new List<GlobalSfmInit.RelativePose>();
        var corrupted = new HashSet<(int, int)>();
        for (int a = 0; a < n; a++)
            for (int d = 1; d <= 3; d++)
            {
                int b = (a + d) % n;
                var truth = GlobalSfmInit.Mul(abs[b], GlobalSfmInit.Transpose(abs[a]));
                bool bad = rng.NextDouble() < 0.25;
                var r = bad ? GlobalSfmInit.Mul(Rot(rng, 180, 60), truth) : GlobalSfmInit.Mul(Rot(rng, 0.3), truth);
                if (bad) corrupted.Add((a, b));
                edges.Add(new GlobalSfmInit.RelativePose(a, b, r, new double[] { 1, 0, 0 }, 100));
            }
        var kept = GlobalSfmInit.FilterByLoops(edges);
        int keptBad = kept.Count(e => corrupted.Contains((e.A, e.B)));
        int keptGood = kept.Count - keptBad;
        int good = edges.Count - corrupted.Count;
        TestContext.Out.WriteLine($"{edges.Count} edges, {corrupted.Count} corrupted: kept {keptGood} of {good} true, {keptBad} corrupted");
        Assert.That(keptBad, Is.EqualTo(0), "no repeated-structure pair survives");
        Assert.That(keptGood, Is.GreaterThan(0.6 * good), "most true pairs are in a consistent triangle");
    }
}
