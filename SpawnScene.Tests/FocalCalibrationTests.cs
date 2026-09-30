using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// GlobalSfmInit.CalibrateFocal - GLOMAP's view-graph focal calibration (Fetzer cost, Cauchy 1e-2), 2026-09-30.
/// </summary>
public class FocalCalibrationTests
{
    static double[] Rot(Random rng, double maxDeg)
    {
        double ax = rng.NextDouble() * 2 - 1, ay = rng.NextDouble() * 2 - 1, az = rng.NextDouble() * 2 - 1;
        double len = Math.Sqrt(ax * ax + ay * ay + az * az); ax /= len; ay /= len; az /= len;
        double t = rng.NextDouble() * maxDeg * Math.PI / 180, c = Math.Cos(t), s = Math.Sin(t), C = 1 - c;
        return new[] { c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s,
                       ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s,
                       az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C };
    }

    /// <summary>F = K^-T [t]x R K^-1 from known poses at focal 800: the calibration must return 800.</summary>
    [Test]
    public void Synthetic_ExactFundamentals_RecoverTheFocal()
    {
        var rng = new Random(4);
        const double f = 800, cx = 512, cy = 384;
        var kinv = new double[] { 1 / f, 0, -cx / f, 0, 1 / f, -cy / f, 0, 0, 1 };
        var fs = new List<double[]>();
        for (int i = 0; i < 30; i++)
        {
            var r = Rot(rng, 40);
            double tx = rng.NextDouble() - 0.5, ty = rng.NextDouble() - 0.5, tz = rng.NextDouble() - 0.5;
            var tx_ = new double[] { 0, -tz, ty, tz, 0, -tx, -ty, tx, 0 };
            fs.Add(GlobalSfmInit.Mul(GlobalSfmInit.Mul(GlobalSfmInit.Transpose(kinv), GlobalSfmInit.Mul(tx_, r)), kinv));
        }
        var cal = GlobalSfmInit.CalibrateFocal(fs, cx, cy, prior: 620);   // prior 22.5% low, like DAv3 on DrJohnson
        Assert.That(cal, Is.Not.Null);
        TestContext.Out.WriteLine($"calibrated {cal!.Value.Focal:F2} from {cal.Value.Supporting}/{cal.Value.Pairs}");
        Assert.That(cal.Value.Focal, Is.EqualTo(f).Within(0.002 * f));
    }

    /// <summary>With SpawnScene's own features and F-RANSAC (the pipeline's inputs), on real data.</summary>
    /// <remarks>MEASURED 2026-09-30: TruckFull 597.1 (+2.6%, 763 of 932 near pairs); DrJohnson 1109.0 (+7.1%) - only 9 true
    /// pairs verify with the default FAST+BRIEF (SIFT F-RANSAC pairs gave 1046.9, +1.1%). Still a third of DAv3's -22%.
    /// </remarks>
    [TestCase("djgt", 666.0, 438.0, 812.0, 1035.5, 0.10)]
    [TestCase("truckgt", 489.25, 272.75, 609.7, 581.9, 0.03)]
    public void RealPairs_OurFeatures_CalibrateNearColmap(string name, double cx, double cy, double prior, double colmap, double tol)
    {
        var d = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
        while (d != null && !Directory.Exists(Path.Combine(d.FullName, "_scratch", name))) d = d.Parent;
        if (d == null) Assert.Ignore($"_scratch/{name} not present");
        var dir = Path.Combine(d.FullName, "_scratch", name);
        var files = Directory.GetFiles(dir, "*.gray").OrderBy(f => f).ToArray();
        var lines = File.ReadAllLines(Path.Combine(dir, "pairs.txt")).Select(l => l.Split(' '))
            .Where(t => name != "truckgt" || int.Parse(t[1]) - int.Parse(t[0]) <= 4).ToArray();   // Truck: near pairs (as candidates are)
        var need = lines.SelectMany(t => new[] { int.Parse(t[0]), int.Parse(t[1]) }).Distinct().ToArray();
        var feats = new List<ImageFeature>[files.Length];
        var det = new FeatureDetector();
        Parallel.ForEach(need, i => { var b = File.ReadAllBytes(files[i]); feats[i] = det.Detect(b[8..], BitConverter.ToInt32(b, 0), BitConverter.ToInt32(b, 4)); });
        var m = new FeatureMatcher(0.75f, 64);
        var fs = new System.Collections.Concurrent.ConcurrentBag<double[]>();
        Parallel.ForEach(lines, t =>
        {
            int a = int.Parse(t[0]), b = int.Parse(t[1]);
            var mm = m.Match(feats[a], feats[b]);
            if (mm.Count < 15) return;
            var xa = new float[mm.Count * 2]; var xb = new float[mm.Count * 2];
            for (int i = 0; i < mm.Count; i++)
            {
                xa[i * 2] = feats[a][mm[i].IndexA].X; xa[i * 2 + 1] = feats[a][mm[i].IndexA].Y;
                xb[i * 2] = feats[b][mm[i].IndexB].X; xb[i * 2 + 1] = feats[b][mm[i].IndexB].Y;
            }
            var r = EpipolarRansac.Estimate(xa, xb, 2.0, 15, seed: a * 1000 + b);
            if (r != null) fs.Add(r.F);
        });
        var cal = GlobalSfmInit.CalibrateFocal(fs.ToList(), cx, cy, prior);
        Assert.That(cal, Is.Not.Null, $"no estimate from {fs.Count} pairs");
        double err = (cal!.Value.Focal - colmap) / colmap;
        TestContext.Out.WriteLine($"{name}: {cal.Value.Focal:F1} from {cal.Value.Supporting}/{cal.Value.Pairs} pairs (prior {prior}, COLMAP {colmap}, {err:+0.0%;-0.0%})");
        Assert.That(Math.Abs(err), Is.LessThan(tol));
    }
}
