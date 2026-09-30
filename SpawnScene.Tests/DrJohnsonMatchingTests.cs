using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Wide-baseline matching against COLMAP ground truth (2026-09-30). DrJohnson's 44 dataset views: the pipeline verified 3 of
/// 49 candidate pairs (15 of 315 with every pair allowed), so the global init placed 2-6 of 44 cameras. COLMAP's own model
/// gives, for every pair that truly overlaps, the TRUE fundamental matrix; a match is correct when its Sampson distance
/// under it is below 2 px. Measured with OpenCV on the same data: FAST+BRIEF 7 of 119 pairs verifiable (15+ correct
/// matches), ORB 5000 70, SIFT 8000 79 - orientation alone 19, scale alone 36.
/// Data: _scratch/djgt (grayscale views + pairs.txt: a b shared F[9]), exported from the dataset's COLMAP sparse model;
/// the test is skipped without it.
/// </summary>
public class DrJohnsonMatchingTests
{
    static string? DataDir(string name = "djgt")
    {
        var d = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
        while (d != null && !Directory.Exists(Path.Combine(d.FullName, "_scratch", name))) d = d.Parent;
        return d == null ? null : Path.Combine(d.FullName, "_scratch", name);
    }

    /// <summary>TruckFull (video, small baselines) must not lose what the pyramid gains on DrJohnson: 251 views at the
    /// pipeline's 979 px, 8,794 pairs sharing 200+ COLMAP points (_scratch/truckgt, gt_export.py).</summary>
    // MEASURED 2026-09-30 at (2000, 0.75): the replaced FAST+BRIEF 4,888 of 8,794; this detector 4,920 (1 level: 3,842).
    // The floor holds it at the old detector's level - the pyramid must not buy DrJohnson at Truck's expense.
    [TestCase(8, 4850)]
    public void TruckFull_TrueNeighbours_AreMatchable(int levels, int floor)
    {
        var dir = DataDir("truckgt");
        if (dir == null) Assert.Ignore("_scratch/truckgt not present");
        var (ok, n, med) = Score(dir, new FeatureDetector(2000, 25, levels), new FeatureMatcher(0.75f, 64));
        TestContext.Out.WriteLine($"TruckFull, {levels} level(s): {ok} of {n} true pairs verifiable, median correct {med}");
        Assert.That(ok, Is.GreaterThanOrEqualTo(floor));
    }

    internal static (int VerifiablePairs, int Pairs, double MedianCorrect) Score(string dir, FeatureDetector detector,
        FeatureMatcher matcher)
    {
        var files = Directory.GetFiles(dir, "*.gray").OrderBy(f => f).ToArray();
        var feats = new List<ImageFeature>[files.Length];
        Parallel.For(0, files.Length, i =>
        {
            var b = File.ReadAllBytes(files[i]);
            int w = BitConverter.ToInt32(b, 0), h = BitConverter.ToInt32(b, 4);
            feats[i] = detector.Detect(b[8..], w, h);
        });
        var lines = File.ReadAllLines(Path.Combine(dir, "pairs.txt"));
        var correct = new int[lines.Length];
        Parallel.For(0, lines.Length, li =>
        {
            var t = lines[li].Split(' ');
            int a = int.Parse(t[0]), bIdx = int.Parse(t[1]);
            var F = t.Skip(3).Take(9).Select(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            int c = 0;
            foreach (var m in matcher.Match(feats[a], feats[bIdx]))
            {
                var pa = feats[a][m.IndexA]; var pb = feats[bIdx][m.IndexB];
                if (Sampson(F, pa.X, pa.Y, pb.X, pb.Y) < 2.0) c++;
            }
            correct[li] = c;
        });
        var sorted = correct.OrderBy(x => x).ToArray();
        return (correct.Count(c => c >= 15), correct.Length, sorted[sorted.Length / 2]);
    }

    /// <summary>Sampson distance of (xa, xb) under F, where xb^T F xa = 0.</summary>
    static double Sampson(double[] F, double ax, double ay, double bx, double by)
    {
        double fx0 = F[0] * ax + F[1] * ay + F[2], fx1 = F[3] * ax + F[4] * ay + F[5], fx2 = F[6] * ax + F[7] * ay + F[8];
        double ftx0 = F[0] * bx + F[3] * by + F[6], ftx1 = F[1] * bx + F[4] * by + F[7];
        double num = bx * fx0 + by * fx1 + fx2;
        return Math.Abs(num) / Math.Sqrt(fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1);
    }

    // MEASURED 2026-09-30 (ORB-style detector): 40 / 66 / 53 / 77 of 119; the unoriented single-scale FAST+BRIEF it
    // replaced: 7 at (2000, 0.75) and 11 at (2000, 0.9). Floors ~10% under the measurement.
    [TestCase(2000, 0.75f, 36)]
    [TestCase(2000, 0.9f, 60)]
    [TestCase(4000, 0.75f, 48)]
    [TestCase(4000, 0.9f, 70)]
    public void DrJohnson_TrueNeighbours_AreMatchable(int features, float ratio, int floor)
    {
        var dir = DataDir();
        if (dir == null) Assert.Ignore("_scratch/djgt not present (export from the DrJohnson COLMAP model)");
        var (ok, n, med) = Score(dir, new FeatureDetector(features), new FeatureMatcher(ratio, 64));
        TestContext.Out.WriteLine($"{features} features, ratio {ratio}: {ok} of {n} true pairs verifiable (15+ correct matches), median correct {med}");
        Assert.That(ok, Is.GreaterThanOrEqualTo(floor), "true neighbours must stay matchable");
    }
}
