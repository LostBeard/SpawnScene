using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Dense triangulation with fixed cameras (2026-10-01). Truck poses (cameras 0-24), 4,000 points in front of them,
/// features = their projections + 0.5 px noise; matches on neighbouring pairs (gap 1-3), with 20% of each pair's matches
/// replaced by WRONG ones (to a random other feature). Most true points must come back accurately (< 0.05; the scene's
/// sigma is 1.2) and at most 1% may be garbage (> 0.25 from any true point).
/// </summary>
public class DenseTriangulationTests
{
    [Test]
    public void Triangulate_FixedCameras_KeepsTruePointsRejectsWrongMatches()
    {
        var all = GlobalSfmInitTests.TruckCameras();
        if (all == null) Assert.Ignore("Truck poses not present");
        var cams = all.Take(25).ToList();
        var rng = new Random(7);
        double G() => Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position + Vector3.Normalize(c.Forward) * 4f) / cams.Count;
        var truth = Enumerable.Range(0, 4000).Select(_ => centre + new Vector3((float)G(), (float)G(), (float)G()) * 1.2f).ToList();
        // features[c]: (point id, feature)
        var feats = cams.Select(_ => new List<ImageFeature>()).ToList();
        var featOf = cams.Select(_ => new Dictionary<int, int>()).ToList();
        for (int c = 0; c < cams.Count; c++)
            for (int p = 0; p < truth.Count; p++)
            {
                if (!WorldSpaceGeometry.Project(cams[c], truth[p], out var u, out var v, out var zc) || zc <= 0) continue;
                if (u < 0 || v < 0 || u >= cams[c].Width || v >= cams[c].Height) continue;
                featOf[c][p] = feats[c].Count;
                feats[c].Add(new ImageFeature { X = u + (float)(G() * 0.5), Y = v + (float)(G() * 0.5), PackedColor = p + 1 });
            }
        var pairs = new List<(int, int, IReadOnlyList<FeatureMatch>)>();
        int wrongTotal = 0;
        for (int a = 0; a < cams.Count; a++)
            for (int gap = 1; gap <= 3 && a + gap < cams.Count; gap++)
            {
                int b = a + gap;
                var list = new List<FeatureMatch>();
                foreach (var (p, ia) in featOf[a])
                {
                    if (!featOf[b].TryGetValue(p, out int ib)) continue;
                    if (rng.NextDouble() < 0.2) { ib = rng.Next(feats[b].Count); wrongTotal++; }
                    list.Add(new FeatureMatch { IndexA = ia, IndexB = ib });
                }
                pairs.Add((a, b, list));
            }
        var r = DenseTriangulation.Triangulate(cams, c => feats[c], pairs);
        TestContext.Out.WriteLine(r.Summary + $"; {wrongTotal} wrong matches planted");
        // Accuracy: each point against its NEAREST true point (a track whose first observation was a wrong match still
        // lands on a real point when its other observations agree).
        int good = 0, bad = 0;
        var badByViews = new Dictionary<int, int>(); var goodByViews = new Dictionary<int, int>(); var badDist = new List<float>();
        for (int i = 0; i < r.Points.Count; i++)
        {
            float best = float.MaxValue;
            foreach (var t in truth) best = Math.Min(best, Vector3.DistanceSquared(r.Points[i], t));
            int views = r.Tracks[i].Count;
            // Within 0.05 (4% of the scene's sigma) is accurate; past 0.25 is GARBAGE (a wrong match). Between is depth
            // noise of short baselines (adjacent Truck views) - harmless for splat initialisation, which training moves.
            if (MathF.Sqrt(best) < 0.05f) { good++; goodByViews[views] = goodByViews.GetValueOrDefault(views) + 1; }
            else if (MathF.Sqrt(best) > 0.25f) { bad++; badByViews[views] = badByViews.GetValueOrDefault(views) + 1; badDist.Add(MathF.Sqrt(best)); }
        }
        badDist.Sort();
        if (badDist.Count > 0) TestContext.Out.WriteLine($"bad points' distance to the nearest true point: median {badDist[badDist.Count / 2]:F3}, p90 {badDist[badDist.Count * 9 / 10]:F3}, max {badDist[^1]:F3} (scene sigma 1.2)");
        TestContext.Out.WriteLine("good by views: " + string.Join(" ", goodByViews.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}:{kv.Value}")) +
            "; bad by views: " + string.Join(" ", badByViews.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}:{kv.Value}")));
        int seenTwice = Enumerable.Range(0, truth.Count).Count(p => featOf.Count(f => f.ContainsKey(p)) >= 2);
        TestContext.Out.WriteLine($"{good} accurate points (< 0.05), {bad} garbage (> 0.25) of {r.Points.Count}; {seenTwice} points seen by 2+ cameras");
        Assert.That(good, Is.GreaterThan(seenTwice * 7 / 10), "most true points come back");
        Assert.That(bad, Is.LessThanOrEqualTo(r.Points.Count / 100), "wrong matches must not make points");
    }
}
