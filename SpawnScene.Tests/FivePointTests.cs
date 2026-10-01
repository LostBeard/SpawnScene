using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The calibrated relative pose (2026-10-01): Nistér's five-point solver and E-RANSAC. See <see cref="FivePoint"/> for the
/// DrJohnson measurement that motivated it (true-pair rotations 8.25 deg median from F, 0.94 from 5-point E + LM).
/// </summary>
public class FivePointTests
{
    static double Gauss(Random rng) =>
        Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble());

    static double[] RandomRotation(Random rng, double maxDeg)
    {
        var axis = Vector3.Normalize(new Vector3((float)Gauss(rng), (float)Gauss(rng), (float)Gauss(rng)));
        double ang = (rng.NextDouble() * 2 - 1) * maxDeg * Math.PI / 180;
        var r = new double[9];
        BundleAdjuster.Rodrigues(axis.X * ang, axis.Y * ang, axis.Z * ang, r);
        return r;
    }

    static double[] EssentialOf(double[] r, double[] t)
    {
        var tx = new[] { 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0 };
        var e = GlobalSfmInit.Mul(tx, r);
        double n = Math.Sqrt(e.Sum(v => v * v));
        return e.Select(v => v / n).ToArray();
    }

    static (double X, double Y) Project(double[] r, double[] t, double[] p)
    {
        double x = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
        double y = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
        double z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
        return (x / z, y / z);
    }

    /// <summary>Exact correspondences over 200 random poses: one of the returned essential matrices must be the true one.</summary>
    [Test]
    public void Solve_ExactCorrespondences_ContainsTheTrueEssential()
    {
        var rng = new Random(5);
        int found = 0, trials = 200, maxSols = 0;
        double worst = 0;
        for (int trial = 0; trial < trials; trial++)
        {
            var r = RandomRotation(rng, 40);
            var t = new[] { Gauss(rng), Gauss(rng), Gauss(rng) };
            double tn = Math.Sqrt(t.Sum(v => v * v)); t = t.Select(v => v / tn).ToArray();
            var ra = new (double X, double Y)[5]; var rb = new (double X, double Y)[5];
            for (int i = 0; i < 5; i++)
            {
                var p = new[] { Gauss(rng), Gauss(rng), 4 + Gauss(rng) };
                ra[i] = (p[0] / p[2], p[1] / p[2]);
                rb[i] = Project(r, t, p);
            }
            var truth = EssentialOf(r, t);
            var sols = FivePoint.Solve(ra, rb);
            maxSols = Math.Max(maxSols, sols.Count);
            double best = double.MaxValue;
            foreach (var e in sols)
            {
                double dp = 0, dm = 0;
                for (int k = 0; k < 9; k++) { dp += (e[k] - truth[k]) * (e[k] - truth[k]); dm += (e[k] + truth[k]) * (e[k] + truth[k]); }
                best = Math.Min(best, Math.Sqrt(Math.Min(dp, dm)));
            }
            if (best < 1e-6) found++;
            else worst = Math.Max(worst, best);
        }
        TestContext.Out.WriteLine($"true E among the solutions in {found}/{trials} trials (at most {maxSols} solutions; worst miss {worst:G3})");
        Assert.That(maxSols, Is.LessThanOrEqualTo(10));
        Assert.That(found, Is.GreaterThanOrEqualTo(trials * 99 / 100));
    }

    [Test]
    public void RealRoots_FindsAllRealRoots()
    {
        // (z - 1)(z + 2)(z - 0.5)(z^2 + 1): real roots 1, -2, 0.5
        double[] Mul(double[] a, double[] b) { var r = new double[a.Length + b.Length - 1]; for (int i = 0; i < a.Length; i++) for (int j = 0; j < b.Length; j++) r[i + j] += a[i] * b[j]; return r; }
        var p = Mul(Mul(Mul(new[] { -1.0, 1 }, new[] { 2.0, 1 }), new[] { -0.5, 1 }), new[] { 1.0, 0, 1 });
        var roots = FivePoint.RealRoots(p).OrderBy(x => x).ToArray();
        Assert.That(roots.Length, Is.EqualTo(3));
        Assert.That(roots[0], Is.EqualTo(-2).Within(1e-9));
        Assert.That(roots[1], Is.EqualTo(0.5).Within(1e-9));
        Assert.That(roots[2], Is.EqualTo(1).Within(1e-9));
    }

    /// <summary>
    /// A WALL scene (DrJohnson is mostly walls): 80% of the points on one plane, 20% off it, 0.7 px noise, 25% outliers.
    /// F-RANSAC locks onto the plane (a plane-induced F has 2 free DoF of degeneracy); the calibrated five-point E does not.
    /// (An EXACTLY planar scene is ambiguous for any two-view method - the five-point problem then has two valid solutions
    /// - so the off-plane 20% is what a real wall with doors, frames and furniture provides.)
    /// </summary>
    [Test]
    public void FromMatchesCalibrated_WallScene_BeatsTheFundamentalPath()
    {
        var rng = new Random(13);
        const double f = 800, cx = 512, cy = 336;
        var errE = new List<double>(); var errF = new List<double>();
        for (int trial = 0; trial < 30; trial++)
        {
            var r = RandomRotation(rng, 25);
            var t = new[] { Gauss(rng), 0.3 * Gauss(rng), 0.3 * Gauss(rng) };
            double tn = Math.Sqrt(t.Sum(v => v * v)); t = t.Select(v => v / tn * 0.6).ToArray();
            var xa = new List<float>(); var xb = new List<float>();
            // A plane 4 units ahead, slightly tilted; 20% of the points off it (2-7 units deep).
            for (int i = 0; i < 300; i++)
            {
                double u = rng.NextDouble() * 4 - 2, v = rng.NextDouble() * 3 - 1.5;
                var p = new[] { u, v, rng.NextDouble() < 0.2 ? 2 + 5 * rng.NextDouble() : 4 + 0.3 * u };
                double bz = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
                if (bz <= 0) continue;
                var (ax, ay) = (p[0] / p[2], p[1] / p[2]);
                var (bx, by) = Project(r, t, p);
                double pax = ax * f + cx, pay = ay * f + cy, pbx = bx * f + cx, pby = by * f + cy;
                if (pbx < 0 || pby < 0 || pbx >= 1024 || pby >= 672 || pax < 0 || pay < 0 || pax >= 1024 || pay >= 672) continue;
                if (rng.NextDouble() < 0.25) { pbx = rng.NextDouble() * 1024; pby = rng.NextDouble() * 672; }
                xa.Add((float)(pax + Gauss(rng) * 0.7)); xa.Add((float)(pay + Gauss(rng) * 0.7));
                xb.Add((float)(pbx + Gauss(rng) * 0.7)); xb.Add((float)(pby + Gauss(rng) * 0.7));
            }
            if (xa.Count / 2 < 60) continue;
            var pe = GlobalSfmInit.FromMatchesCalibrated(0, 1, xa.ToArray(), xb.ToArray(), f, cx, cy, cx, cy, seed: trial + 1);
            var ransac = EpipolarRansac.Estimate(xa.ToArray(), xb.ToArray(), thresholdPx: 2.0, minInliers: 15, seed: trial + 1);
            var pf = ransac == null ? null : GlobalSfmInit.FromFundamental(0, 1, ransac.F, xa.ToArray(), xb.ToArray(), ransac.Inliers, f, cx, cy, cx, cy);
            errE.Add(pe == null ? 180 : GlobalSfmInit.AngleDeg(pe.R, r));
            errF.Add(pf == null ? 180 : GlobalSfmInit.AngleDeg(pf.R, r));
        }
        errE.Sort(); errF.Sort();
        TestContext.Out.WriteLine($"{errE.Count} planar pairs: E path median {errE[errE.Count / 2]:F3} p90 {errE[errE.Count * 9 / 10]:F3} deg; " +
            $"F path median {errF[errF.Count / 2]:F3} p90 {errF[errF.Count * 9 / 10]:F3} deg");
        Assert.That(errE.Count, Is.GreaterThan(20));
        Assert.That(errE[errE.Count * 9 / 10], Is.LessThan(1.0), "the calibrated path must hold on a wall scene (p90)");
        Assert.That(errE[errE.Count / 2], Is.LessThan(errF[errF.Count / 2]), "and beat the F path");
    }
}

public class FivePointCost
{
    [Test, Explicit("timing")]
    public void Cost_PerPair()
    {
        var rng = new Random(3);
        double G() => Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble());
        const double f = 800, cx = 512, cy = 336;
        var r = new double[9]; BundleAdjuster.Rodrigues(0.1, 0.2, 0.05, r);
        var t = new[] { 0.6, 0.05, 0.1 };
        var xa = new List<float>(); var xb = new List<float>();
        for (int i = 0; i < 200; i++)
        {
            var p = new[] { G(), G(), 4 + G() };
            double bx = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0], by = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1], bz = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
            bool outl = rng.NextDouble() < 0.3;
            xa.Add((float)(p[0] / p[2] * f + cx + G() * 0.7)); xa.Add((float)(p[1] / p[2] * f + cy + G() * 0.7));
            xb.Add((float)(outl ? rng.NextDouble() * 1024 : bx / bz * f + cx + G() * 0.7)); xb.Add((float)(outl ? rng.NextDouble() * 672 : by / bz * f + cy + G() * 0.7));
        }
        var a = xa.ToArray(); var b = xb.ToArray();
        GlobalSfmInit.FromMatchesCalibrated(0, 1, a, b, f, cx, cy, cx, cy);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int k = 0; k < 20; k++) GlobalSfmInit.FromMatchesCalibrated(0, 1, a, b, f, cx, cy, cx, cy, seed: k + 1);
        double pair = sw.Elapsed.TotalMilliseconds / 20;
        var ra = new (double, double)[5]; var rb2 = new (double, double)[5];
        for (int i = 0; i < 5; i++) { ra[i] = ((a[i * 2] - cx) / f, (a[i * 2 + 1] - cy) / f); rb2[i] = ((b[i * 2] - cx) / f, (b[i * 2 + 1] - cy) / f); }
        sw.Restart();
        for (int k = 0; k < 2000; k++) FivePoint.Solve(ra, rb2);
        double solve = sw.Elapsed.TotalMilliseconds / 2000;
        TestContext.Out.WriteLine($"native: {pair:F2} ms per pair (200 matches, 30% outliers), {solve * 1000:F1} us per five-point solve");
    }
}
