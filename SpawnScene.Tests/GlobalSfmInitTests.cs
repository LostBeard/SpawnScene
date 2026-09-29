using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// GlobalSfmInit on synthetic data built from Truck's COLMAP cameras (126 views): relative poses from exact fundamental
/// matrices, rotation averaging under noise and garbage pairs, translation averaging from a perturbed start, and the
/// whole initialisation from a smoothly BENT start - the failure it exists for (2026-09-28: bundle adjustment from the
/// bent depth cascade ended 1.9-2.5% of spread off COLMAP, from COLMAP's poses 0.1%).
/// </summary>
public class GlobalSfmInitTests
{
    const int W = 979, H = 546;

    static List<CameraParams>? TruckCameras()
    {
        string path = Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..",
            "SpawnScene", "wwwroot", "datasets", "Truck", "poses.par"));
        if (!File.Exists(path)) return null;
        return WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(path), 1957, 1091)
            .OrderBy(e => e.filename, StringComparer.Ordinal).Select(e => e.camera.ScaledTo(W, H)).ToList();
    }

    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    static double Gauss(Random rng) =>
        Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble());

    static double[] SmallRotation(Random rng, double sigmaDeg)
    {
        double s = sigmaDeg * Math.PI / 180;
        double wx = Gauss(rng) * s, wy = Gauss(rng) * s, wz = Gauss(rng) * s;
        var r = new double[9];
        BundleAdjuster.Rodrigues(wx, wy, wz, r);
        return r;
    }

    /// <summary>Exact relative pose (R_ab, unit t_ab) between two cameras.</summary>
    static (double[] R, double[] T) Relative(CameraParams a, CameraParams b)
    {
        var ra = GlobalSfmInit.RotationOf(a); var rb = GlobalSfmInit.RotationOf(b);
        var r = GlobalSfmInit.Mul(rb, GlobalSfmInit.Transpose(ra));
        var d = a.Position - b.Position;
        double tx = rb[0] * d.X + rb[1] * d.Y + rb[2] * d.Z;
        double ty = rb[3] * d.X + rb[4] * d.Y + rb[5] * d.Z;
        double tz = rb[6] * d.X + rb[7] * d.Y + rb[8] * d.Z;
        double n = Math.Sqrt(tx * tx + ty * ty + tz * tz);
        return (r, new[] { tx / n, ty / n, tz / n });
    }

    /// <summary>Pairs a truck capture would verify: neighbours within 6 in capture order.</summary>
    static IEnumerable<(int A, int B)> NeighbourPairs(int n)
    {
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b <= Math.Min(n - 1, a + 6); b++)
                yield return (a, b);
    }

    [Test]
    public void FromFundamental_RecoversTheRelativePose()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(3);
        // Scene points in front of the truck cameras: around the centroid of what camera 0 looks at.
        var c0 = cams[0];
        var centre = c0.Position + Vector3.Normalize(c0.Forward) * 4f;
        var pts = Enumerable.Range(0, 400).Select(_ => centre + new Vector3((float)Gauss(rng), (float)Gauss(rng), (float)Gauss(rng)) * 1.2f).ToList();
        int checkedPairs = 0;
        foreach (var (ia, ib) in new[] { (0, 1), (0, 3), (5, 9), (10, 12) })
        {
            var a = cams[ia]; var b = cams[ib];
            var xa = new List<float>(); var xb = new List<float>();
            foreach (var p in pts)
            {
                if (!WorldSpaceGeometry.Project(a, p, out var uax, out var uay, out _) || !WorldSpaceGeometry.Project(b, p, out var ubx, out var uby, out _)) continue;
                xa.Add(uax); xa.Add(uay); xb.Add(ubx); xb.Add(uby);
            }
            Assume.That(xa.Count / 2, Is.GreaterThan(50));
            // F = K_b^-T [t]x R K_a^-1 (square pixels at the mean focal: the test's own model).
            var (r, t) = Relative(a, b);
            double f = 0.5 * (a.FocalX + a.FocalY);
            var tx = new[] { 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0 };
            var e = GlobalSfmInit.Mul(tx, r);
            var kaInv = new[] { 1 / f, 0, -a.CenterX / f, 0, 1 / f, -a.CenterY / f, 0, 0, 1 };
            var kbInv = new[] { 1 / f, 0, -b.CenterX / f, 0, 1 / f, -b.CenterY / f, 0, 0, 1 };
            var fm = GlobalSfmInit.Mul(GlobalSfmInit.Mul(GlobalSfmInit.Transpose(kbInv), e), kaInv);
            // Reproject with square pixels too (the truth has fx != fy; keep the test exact).
            var sa = new CameraParams { Width = W, Height = H, FocalX = (float)f, FocalY = (float)f, CenterX = a.CenterX, CenterY = a.CenterY, Position = a.Position, Forward = a.Forward, Up = a.Up };
            var sb = new CameraParams { Width = W, Height = H, FocalX = (float)f, FocalY = (float)f, CenterX = b.CenterX, CenterY = b.CenterY, Position = b.Position, Forward = b.Forward, Up = b.Up };
            xa.Clear(); xb.Clear();
            foreach (var p in pts)
            {
                if (!WorldSpaceGeometry.Project(sa, p, out var uax, out var uay, out _) || !WorldSpaceGeometry.Project(sb, p, out var ubx, out var uby, out _)) continue;
                xa.Add(uax); xa.Add(uay); xb.Add(ubx); xb.Add(uby);
            }
            var inl = Enumerable.Repeat(true, xa.Count / 2).ToArray();
            var rel = GlobalSfmInit.FromFundamental(ia, ib, fm, xa.ToArray(), xb.ToArray(), inl, f, a.CenterX, a.CenterY, b.CenterX, b.CenterY);
            Assert.That(rel, Is.Not.Null, $"pair {ia}-{ib}");
            double angR = GlobalSfmInit.AngleDeg(rel!.R, r);
            double dot = rel.T[0] * t[0] + rel.T[1] * t[1] + rel.T[2] * t[2];
            TestContext.Out.WriteLine($"pair {ia}-{ib}: rotation error {angR:F4} deg, direction dot {dot:F6}");
            Assert.That(angR, Is.LessThan(0.01));
            Assert.That(dot, Is.GreaterThan(0.9999));
            checkedPairs++;
        }
        Assert.That(checkedPairs, Is.EqualTo(4));
    }

    [Test]
    public void AverageRotations_SurvivesNoiseAndGarbagePairs()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(11);
        int n = cams.Count;
        var truth = cams.Select(GlobalSfmInit.RotationOf).ToArray();
        var edges = new List<GlobalSfmInit.RelativePose>();
        int garbage = 0;
        foreach (var (a, b) in NeighbourPairs(n))
        {
            var (r, t) = Relative(cams[a], cams[b]);
            if (rng.NextDouble() < 0.10) { r = SmallRotation(rng, 60); garbage++; }  // a wrong pair
            else r = GlobalSfmInit.Mul(SmallRotation(rng, 0.5), r);                // 0.5 deg noise
            edges.Add(new GlobalSfmInit.RelativePose(a, b, r, t, 100 + rng.Next(200)));
        }
        // Fallback / frame reference: the truth itself (only the frame alignment uses it here).
        var rot = GlobalSfmInit.AverageRotations(n, edges, truth, out var connected);
        var q = GlobalSfmInit.AlignFrame(rot, truth, connected);
        var errs = Enumerable.Range(0, n).Select(i => GlobalSfmInit.AngleDeg(GlobalSfmInit.Mul(rot[i], q), truth[i])).OrderBy(x => x).ToList();
        TestContext.Out.WriteLine($"{edges.Count} pairs ({garbage} garbage): rotation error median {errs[n / 2]:F3} deg, p90 {errs[n * 9 / 10]:F3}, max {errs[^1]:F3}");
        Assert.That(connected.All(c => c));
        // 0.5 deg of noise per pair on a chain-like graph (each view paired with its 12 neighbours) floors the median near
        // 0.36 deg whatever the round count (MEASURED); the garbage pairs must not add to it or flip a camera.
        Assert.That(errs[n / 2], Is.LessThan(0.5));
        Assert.That(errs[^1], Is.LessThan(1.5));
    }

    /// <summary>
    /// Synthetic tracks for Truck's cameras: points around the scene centre the cameras look at (least-squares closest
    /// point to all optical axes), each seen only by cameras within +-12 views of a random home view (video tracks are
    /// local), projected with square pixels at the mean focal, plus Gaussian pixel noise.
    /// </summary>
    // TruckFull's measured views-per-track mix (2, 3, 4, 5, 6+ views), cumulative.
    static readonly double[] RealTrackMix = { 0.562, 0.751, 0.844, 0.897, 1.0 };

    static (List<BundleAdjuster.Observation> Obs, int Points, double Focal) SyntheticTracks(List<CameraParams> cams, Random rng,
        int points, double noisePx, bool realLengths = false, double outlierFraction = 0)
    {
        // Closest point to all optical axes: sum (I - f f^T) (X - C) = 0.
        double[] m = new double[9], b = new double[3];
        foreach (var c in cams)
        {
            var f = Vector3.Normalize(c.Forward);
            double[] fv = { f.X, f.Y, f.Z }, cv = { c.Position.X, c.Position.Y, c.Position.Z };
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    double pij = (i == j ? 1 : 0) - fv[i] * fv[j];
                    m[i * 3 + j] += pij;
                    b[i] += pij * cv[j];
                }
        }
        var inv = Invert(m);
        var centre = new Vector3((float)(inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2]),
            (float)(inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2]), (float)(inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2]));
        float radius = cams.Average(c => Vector3.Distance(c.Position, centre)) * 0.25f;
        double focal = cams.Average(c => 0.5 * (c.FocalX + c.FocalY));
        var obs = new List<BundleAdjuster.Observation>();
        int id = 0;
        for (int p = 0; p < points; p++)
        {
            var x = centre + new Vector3((float)Gauss(rng), (float)Gauss(rng), (float)Gauss(rng)) * radius;
            int home = rng.Next(cams.Count);
            int lo = Math.Max(0, home - 12), hi = Math.Min(cams.Count - 1, home + 12);
            if (realLengths)
            {
                // A run of consecutive views with TruckFull's length mix (6+ drawn as 6-10).
                double u0 = rng.NextDouble();
                int len = u0 < RealTrackMix[0] ? 2 : u0 < RealTrackMix[1] ? 3 : u0 < RealTrackMix[2] ? 4 : u0 < RealTrackMix[3] ? 5 : 6 + rng.Next(5);
                lo = Math.Max(0, Math.Min(home, cams.Count - len)); hi = Math.Min(cams.Count - 1, lo + len - 1);
            }
            var seen = new List<BundleAdjuster.Observation>();
            for (int i = lo; i <= hi; i++)
            {
                var c = cams[i];
                var sq = new CameraParams { Width = W, Height = H, FocalX = (float)focal, FocalY = (float)focal, CenterX = c.CenterX, CenterY = c.CenterY, Position = c.Position, Forward = c.Forward, Up = c.Up };
                if (!WorldSpaceGeometry.Project(sq, x, out var u, out var v, out var zc) || zc <= 0) continue;
                if (u < 0 || v < 0 || u >= W || v >= H) continue;
                if (rng.NextDouble() < outlierFraction)
                    seen.Add(new BundleAdjuster.Observation(i, id, (float)(rng.NextDouble() * W), (float)(rng.NextDouble() * H)));
                else
                    seen.Add(new BundleAdjuster.Observation(i, id, u + (float)(Gauss(rng) * noisePx), v + (float)(Gauss(rng) * noisePx)));
            }
            if (seen.Count < 2) continue;
            obs.AddRange(seen);
            id++;
        }
        return (obs, id, focal);
    }

    static double[] Invert(double[] m)
    {
        double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
        double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
        double id = 1 / (a * A + b * B + c * C);
        return new[] { A * id, -(b * i - c * h) * id, (b * f - c * e) * id, B * id, (a * i - c * g) * id, -(a * f - c * d) * id,
            C * id, -(a * h - b * g) * id, (a * e - b * d) * id };
    }

    [Test]
    public void GlobalPositioning_RecoversCentresFromAPerturbedStart()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(5);
        int n = cams.Count;
        var (obs, points, focal) = SyntheticTracks(cams, rng, 3000, 0.5);
        var truthR = cams.Select(GlobalSfmInit.RotationOf).ToArray();
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position) / n;
        float spread = MathF.Sqrt(cams.Average(c => (c.Position - centre).LengthSquared()));
        var start = cams.Select(Copy).ToList();
        foreach (var c in start) c.Position += new Vector3((float)Gauss(rng), (float)Gauss(rng), (float)Gauss(rng)) * (0.2f * spread);
        var connected = Enumerable.Repeat(true, n).ToArray();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (centres, _, _) = GlobalSfmInit.GlobalPositioningRobust(start, truthR, connected, obs, points, focal);
        var est = cams.Select((c, i) => { var k = Copy(c); k.Position = centres[i]; return (CameraParams?)k; }).ToList();
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, cams.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        TestContext.Out.WriteLine($"{points} points, {obs.Count} observations: centres {acc.PositionRms / acc.Spread:P3} of spread " +
            $"(start 20% noise), {sw.Elapsed.TotalSeconds:F2}s");
        Assert.That(acc.PositionRms / acc.Spread, Is.LessThan(0.01));
    }

    /// <summary>A smooth bend, the shape the depth cascade produces: each camera and its position rotated progressively about
    /// the capture centroid (up to ~15 deg across the capture) and stretched, plus a few grossly misplaced views.</summary>
    static List<CameraParams> Bent(List<CameraParams> cams)
    {
        int n = cams.Count;
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position) / n;
        var start = cams.Select(Copy).ToList();
        for (int i = 0; i < n; i++)
        {
            float ang = 0.26f * (i / (float)n - 0.5f);
            var q = Quaternion.CreateFromAxisAngle(Vector3.UnitY, ang);
            start[i].Position = centre + Vector3.Transform(start[i].Position - centre, q) * (1 + 0.1f * (i / (float)n));
            start[i].Forward = Vector3.Transform(start[i].Forward, q);
            start[i].Up = Vector3.Transform(start[i].Up, q);
        }
        foreach (int i in new[] { 30, 31, 32, 70 }) start[i].Position += new Vector3(2, 0, 1);
        return start;
    }

    /// <summary>
    /// GLOMAP-style robust positioning (random start, Huber, per-observation scales) under the same real-data conditions.
    /// The linear formulation reached 0.34% median on clean real-length tracks but 40% with 5% mismatches.
    /// </summary>
    [TestCase(false, 0.0, TestName = "GlobalPositioningRobust_LongTracks_Clean")]
    [TestCase(true, 0.0, TestName = "GlobalPositioningRobust_RealLengths_Clean")]
    [TestCase(true, 0.05, TestName = "GlobalPositioningRobust_RealLengths_Outliers5")]
    [TestCase(true, 0.10, TestName = "GlobalPositioningRobust_RealLengths_Outliers10")]
    public void GlobalPositioningRobust(bool realLengths, double outliers)
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(21);
        int n = cams.Count;
        var (obs, points, focal) = SyntheticTracks(cams, rng, realLengths ? 12000 : 3000, 0.5, realLengths, outliers);
        var truthR = cams.Select(GlobalSfmInit.RotationOf).ToArray();
        var connected = Enumerable.Repeat(true, n).ToArray();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (centres, _, summary) = GlobalSfmInit.GlobalPositioningRobust(cams, truthR, connected, obs, points, focal);
        var est = cams.Select((c, i) => { var k = Copy(c); k.Position = centres[i]; return (CameraParams?)k; }).ToList();
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, cams.Cast<CameraParams?>().ToList(), out var acc, out var perView, out _), Is.True);
        var perCam = new int[n];
        foreach (var o in obs) perCam[o.Camera]++;
        var worst = Enumerable.Range(0, n).OrderByDescending(i => perView[i]).Take(8)
            .Select(i => $"#{i} {perView[i]:P1} ({perCam[i]} obs)");
        TestContext.Out.WriteLine($"  worst: {string.Join(", ", worst)}; over 2%: {perView.Count(f => f > 0.02f)}");
        TestContext.Out.WriteLine($"[{TestContext.CurrentContext.Test.Name}] {sw.Elapsed.TotalSeconds:F1}s {summary}: " +
            $"centres {acc.PositionRms / acc.Spread:P3} of spread, median {acc.MedianPosFrac:P3}");
        Assert.That(acc.MedianPosFrac, Is.LessThan(0.01));
    }

    [Test]
    public void Apply_UnbendsABentStart()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(9);
        int n = cams.Count;
        var edges = new List<GlobalSfmInit.RelativePose>();
        foreach (var (a, b) in NeighbourPairs(n))
        {
            var (r, t) = Relative(cams[a], cams[b]);
            r = GlobalSfmInit.Mul(SmallRotation(rng, 0.3), r);
            edges.Add(new GlobalSfmInit.RelativePose(a, b, r, t, 200));
        }
        var (obs, points, focal) = SyntheticTracks(cams, rng, 3000, 0.5);
        // A smooth bend, the shape the depth cascade produces: rotate each camera and its position progressively about
        // the capture centroid (up to ~15 deg across the capture), stretch it, plus a few grossly misplaced views.
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position) / n;
        var start = cams.Select(Copy).ToList();
        for (int i = 0; i < n; i++)
        {
            float ang = 0.26f * (i / (float)n - 0.5f);
            var q = Quaternion.CreateFromAxisAngle(Vector3.UnitY, ang);
            start[i].Position = centre + Vector3.Transform(start[i].Position - centre, q) * (1 + 0.1f * (i / (float)n));
            start[i].Forward = Vector3.Transform(start[i].Forward, q);
            start[i].Up = Vector3.Transform(start[i].Up, q);
        }
        foreach (int i in new[] { 30, 31, 32, 70 }) start[i].Position += new Vector3(2, 0, 1);
        var truthN = cams.Cast<CameraParams?>().ToList();
        WorldSpaceGeometry.TryMeasureCameraSetAccuracy(start.Cast<CameraParams?>().ToList(), truthN, out var acc0, out _, out _);
        string summary = GlobalSfmInit.Apply(start, edges, obs, points, focal);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(start.Cast<CameraParams?>().ToList(), truthN, out var acc, out _, out _), Is.True);
        TestContext.Out.WriteLine($"{summary}; start {acc0.PositionRms / acc0.Spread:P2} fwd {acc0.MedianForwardDeg:F2} deg -> " +
            $"{acc.PositionRms / acc.Spread:P3} fwd {acc.MedianForwardDeg:F3} deg");
        Assert.That(acc0.PositionRms / acc0.Spread, Is.GreaterThan(0.03), "the start must be bent to mean anything");
        Assert.That(acc.PositionRms / acc.Spread, Is.LessThan(0.01));
        Assert.That(acc.MedianForwardDeg, Is.LessThan(0.3));
    }
}
