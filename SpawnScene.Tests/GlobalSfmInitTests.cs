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

    internal static List<CameraParams>? TruckCameras()
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

    internal static double[] SmallRotation(Random rng, double sigmaDeg)
    {
        double s = sigmaDeg * Math.PI / 180;
        double wx = Gauss(rng) * s, wy = Gauss(rng) * s, wz = Gauss(rng) * s;
        var r = new double[9];
        BundleAdjuster.Rodrigues(wx, wy, wz, r);
        return r;
    }

    /// <summary>Exact relative pose (R_ab, unit t_ab) between two cameras.</summary>
    internal static (double[] R, double[] T) Relative(CameraParams a, CameraParams b)
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
    internal static IEnumerable<(int A, int B)> NeighbourPairs(int n)
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

    internal static (List<BundleAdjuster.Observation> Obs, int Points, double Focal) SyntheticTracks(List<CameraParams> cams, Random rng,
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
        var centres = GlobalSfmInit.GlobalPositioningRobust(start, truthR, connected, obs, points, focal).Centres;
        var est = cams.Select((c, i) => { var k = Copy(c); k.Position = centres[i]; return (CameraParams?)k; }).ToList();
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, cams.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        TestContext.Out.WriteLine($"{points} points, {obs.Count} observations: centres {acc.PositionRms / acc.Spread:P3} of spread " +
            $"(start 20% noise), {sw.Elapsed.TotalSeconds:F2}s");
        Assert.That(acc.PositionRms / acc.Spread, Is.LessThan(0.01));
    }

    /// <summary>A smooth bend, the shape the depth cascade produces: each camera and its position rotated progressively about
    /// the capture centroid (up to ~15 deg across the capture) and stretched, plus a few grossly misplaced views.</summary>
    static List<CameraParams> Bent(List<CameraParams> cams, float strength = 1)
    {
        int n = cams.Count;
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position) / n;
        var start = cams.Select(Copy).ToList();
        for (int i = 0; i < n; i++)
        {
            float ang = strength * 0.26f * (i / (float)n - 0.5f);
            var q = Quaternion.CreateFromAxisAngle(Vector3.UnitY, ang);
            start[i].Position = centre + Vector3.Transform(start[i].Position - centre, q) * (1 + strength * 0.1f * (i / (float)n));
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
        var gp = GlobalSfmInit.GlobalPositioningRobust(cams, truthR, connected, obs, points, focal);
        var (centres, summary) = (gp.Centres, gp.Summary);
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
        // With exact rotations the last round must CONVERGE, not stop at the cap: a solver that crawls (the linearised
        // scale step did: 600 iterations per round, never converged) is too slow for the browser at any accuracy.
        Assert.That(gp.Converged, Is.True, "the last round stopped at the iteration cap");
        // The inlier selection must give back what an unconverged early round wrongly rejected: 2-view tracks lose both
        // observations to one mismatch, so the selection may leave out up to ~2x the mismatched fraction, never more.
        // MEASURED 2026-09-29: clean 3 of 30,000 left out (permanent rejection: 6,353), 10% mismatches 19.3%.
        Assert.That(gp.Outliers, Is.LessThanOrEqualTo(gp.Observations * (2.2 * outliers + 0.01)), "good observations left out");
    }

    /// <summary>
    /// Robust positioning fed ESTIMATED rotations, as the pipeline does: AverageRotations over noisy pairs with 10% garbage
    /// pairs, frame-aligned to the truth. Real TruckFull (2026-09-28): pair rotations 0.86 deg median off COLMAP (p90 17 deg),
    /// averaged ~1.16 deg. Every observation of a camera then shares its rotation error, so this is a bias, not noise.
    /// </summary>
    [TestCase(0.5, TestName = "GlobalPositioningRobust_EstimatedRotations_Pair05")]
    [TestCase(1.5, TestName = "GlobalPositioningRobust_EstimatedRotations_Pair15")]
    public void GlobalPositioningRobust_EstimatedRotations(double pairNoiseDeg)
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(31);
        int n = cams.Count;
        var truthR = cams.Select(GlobalSfmInit.RotationOf).ToArray();
        var edges = new List<GlobalSfmInit.RelativePose>();
        foreach (var (a, b) in NeighbourPairs(n))
        {
            var (r, t) = Relative(cams[a], cams[b]);
            r = rng.NextDouble() < 0.10 ? SmallRotation(rng, 60) : GlobalSfmInit.Mul(SmallRotation(rng, pairNoiseDeg), r);
            edges.Add(new GlobalSfmInit.RelativePose(a, b, r, t, 100 + rng.Next(200)));
        }
        var rot = GlobalSfmInit.AverageRotations(n, edges, truthR, out var connected);
        var q = GlobalSfmInit.AlignFrame(rot, truthR, connected);
        for (int i = 0; i < n; i++) rot[i] = GlobalSfmInit.Mul(rot[i], q);
        var rotErr = Enumerable.Range(0, n).Select(i => GlobalSfmInit.AngleDeg(rot[i], truthR[i])).OrderBy(x => x).ToList();
        var (obs, points, focal) = SyntheticTracks(cams, rng, 12000, 0.5, true, 0.10);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var gp = GlobalSfmInit.GlobalPositioningRobust(cams, rot, connected, obs, points, focal);
        var (centres, summary) = (gp.Centres, gp.Summary);
        var est = cams.Select((c, i) => { var k = Copy(c); k.Position = centres[i]; return (CameraParams?)k; }).ToList();
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, cams.Cast<CameraParams?>().ToList(), out var acc, out var perView, out _), Is.True);
        var worst = Enumerable.Range(0, n).OrderByDescending(i => perView[i]).Take(8).Select(i => $"#{i} {perView[i]:P1} (rot {GlobalSfmInit.AngleDeg(rot[i], truthR[i]):F2} deg)");
        TestContext.Out.WriteLine($"  rotations: median {rotErr[n / 2]:F3} deg, p90 {rotErr[n * 9 / 10]:F3}, max {rotErr[^1]:F3}");
        TestContext.Out.WriteLine($"  worst: {string.Join(", ", worst)}; over 2%: {perView.Count(f => f > 0.02f)}");
        TestContext.Out.WriteLine($"[{TestContext.CurrentContext.Test.Name}] {sw.Elapsed.TotalSeconds:F1}s {summary}: " +
            $"centres {acc.PositionRms / acc.Spread:P3} of spread, median {acc.MedianPosFrac:P3}");
        // A camera's rotation error tilts every one of its bearings alike, so no positioning can undo it: the error is
        // linear in it. MEASURED 2026-09-29: 0.334 deg -> 0.463% median, 1.116 deg -> 1.573% (1.39 and 1.41 % per degree).
        // BA refines it away afterwards (Apply_ThenBundleAdjust_EstimatedRotations); the bar catches a solver that stops
        // being limited by the rotations alone (e.g. without rejection rounds 5% mismatches left 7%).
        Assert.That(acc.MedianPosFrac, Is.LessThan(0.02 * rotErr[n / 2] + 0.002));
    }

    /// <summary>Production's next step after the init: triangulate every track from the given poses, then bundle-adjust
    /// with production's settings. Returns the median centre error as a fraction of spread.</summary>
    static (double Start, double Final) TriangulateAndAdjust(List<CameraParams> start, List<CameraParams> truth, List<BundleAdjuster.Observation> obs, string label)
    {
        var byPoint = obs.GroupBy(o => o.Point).OrderBy(g => g.Key);
        var points = new List<Vector3>();
        var baObs = new List<BundleAdjuster.Observation>();
        foreach (var g in byPoint)
        {
            var track = g.Select(o => (o.Camera, o.U, o.V)).ToList();
            if (!BundleAdjuster.Triangulate(start, track, out var x)) continue;
            int id = points.Count;
            points.Add(x);
            foreach (var (c, u, v) in track) baObs.Add(new BundleAdjuster.Observation(c, id, u, v));
        }
        var init = start.Select(Copy).ToList();
        var (ba, result) = BundleAdjusterTruckScaleTests.SolveLikeProduction(init, points, baObs);
        for (int i = 0; i < init.Count; i++) ba.WriteCamera(i, init[i]);
        var truthN = truth.Cast<CameraParams?>().ToList();
        WorldSpaceGeometry.TryMeasureCameraSetAccuracy(start.Cast<CameraParams?>().ToList(), truthN, out var acc0, out _, out _);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(init.Cast<CameraParams?>().ToList(), truthN, out var acc, out _, out _), Is.True);
        TestContext.Out.WriteLine($"[{label}] start median {acc0.MedianPosFrac:P3} fwd {acc0.MedianForwardDeg:F2} deg -> BA ({points.Count} points, " +
            $"{result.Iterations} it, {result.Seconds:F0}s, RMS {result.InitialRmsPixels:F1} -> {result.FinalRmsPixels:F2} px, focal {ba.SharedFocal:F1}): " +
            $"median {acc.MedianPosFrac:P3}, rms {acc.PositionRms / acc.Spread:P3}, fwd {acc.MedianForwardDeg:F3} deg");
        return (acc0.MedianPosFrac, acc.MedianPosFrac);
    }

    /// <summary>
    /// The whole no-COLMAP path at real-data rotation quality: a bent start, AverageRotations over pairs at 1.5 deg/axis with
    /// 10% garbage (~1.1 deg averaged, as TruckFull measured), then production's triangulate + bundle adjustment. Tracks are
    /// BundleAdjusterTruckScaleTests' (Truck's real COLMAP points, its measured track mix, 3% outliers), NOT SyntheticTracks:
    /// on SyntheticTracks' shallow blob with 10% unfiltered random-pixel observations, BA started from the TRUE poses drifted
    /// to 1.04% median at 0.72 px (a worse fit than a bent start's 0.57 px) - no sharp minimum, so no start can be judged there.
    /// Floor: the same BA from the true poses. Control: from the bent start without the global init.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-29: floor 0.054%; control from a 7.1% bend 0.056% and from an 18% bend (strength 2.5, used here)
    /// 0.055%; global init start 3.35% whatever the bend -> 0.057%. So on synthetic tracks BA alone recovers any bend - the
    /// real TruckFull failure (bent cascade -> 1.9-2.5%, COLMAP start -> 0.1%) needs something these tracks lack, and only a
    /// real run can show the global init fixes it. This test holds the init itself to its start and the chain to the floor.
    /// </remarks>
    // focalScale 1.045: production has no COLMAP focal - the global init gets the DAv3 per-view median (TruckFull 609.7
    // vs the 583.9 BA converges to, 4.5% high); every real run so far passed &globalfocal=583.7 (2026-09-30).
    [TestCase(1.0)]
    [TestCase(1.045)]
    public void Apply_ThenBundleAdjust_EstimatedRotations(double focalScale)
    {
        const float bend = 2.5f;
        var problem = BundleAdjusterTruckScaleTests.BuildTruckProblem(seed: 5);
        if (problem == null) Assert.Ignore("Truck dataset not in this checkout");
        var (cams, _, _, obs) = problem.Value;
        int points = obs.Max(o => o.Point) + 1;
        double focal = cams.Select(c => 0.5 * (c.FocalX + c.FocalY)).OrderBy(x => x).ElementAt(cams.Count / 2);
        var rng = new Random(41);
        int n = cams.Count;
        var edges = new List<GlobalSfmInit.RelativePose>();
        foreach (var (a, b) in NeighbourPairs(n))
        {
            var (r, t) = Relative(cams[a], cams[b]);
            r = rng.NextDouble() < 0.10 ? SmallRotation(rng, 60) : GlobalSfmInit.Mul(SmallRotation(rng, 1.5), r);
            edges.Add(new GlobalSfmInit.RelativePose(a, b, r, t, 100 + rng.Next(200)));
        }
        var (_, floor) = TriangulateAndAdjust(cams.Select(Copy).ToList(), cams, obs, "floor: BA from the true poses");
        var bent = Bent(cams, bend);
        TriangulateAndAdjust(bent, cams, obs, "control: BA from the bent start");
        var start = bent.Select(Copy).ToList();
        double focalIn = focal * focalScale;
        foreach (var c in start) { c.FocalX = (float)focalIn; c.FocalY = (float)focalIn; }
        string summary = GlobalSfmInit.Apply(start, edges, obs, points, focalIn);
        TestContext.Out.WriteLine(summary);
        var (initStart, global) = TriangulateAndAdjust(start, cams, obs, $"global init (focal x{focalScale}) + BA");
        Assert.That(floor, Is.LessThan(0.005), "the data must pin the truth down, or no start can be judged");
        // ~1.1 deg rotations at ~1.4% per degree (GlobalPositioningRobust_EstimatedRotations) plus the 3% outliers: 3.35%.
        Assert.That(initStart, Is.LessThan(0.05), "the global init itself");
        Assert.That(global, Is.LessThan(2 * floor + 0.0005), "BA from the global init must reach the floor");
    }

    /// <summary>
    /// TruckFull b63 (2026-09-30): the track sampling left some connected cameras with NO positioning observation; their
    /// block of the camera system was zero, every damped step failed "not positive definite", and no camera moved from the
    /// random start (99.6% off COLMAP). A starved camera must be left out and reported, and everyone else placed.
    /// </summary>
    [Test]
    public void Apply_ACameraWithoutObservations_DoesNotStallTheRest()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(9);
        int n = cams.Count;
        var edges = new List<GlobalSfmInit.RelativePose>();
        foreach (var (a, b) in NeighbourPairs(n))
        {
            var (r, t) = Relative(cams[a], cams[b]);
            edges.Add(new GlobalSfmInit.RelativePose(a, b, GlobalSfmInit.Mul(SmallRotation(rng, 0.3), r), t, 200));
        }
        var (obs, points, focal) = SyntheticTracks(cams, rng, 3000, 0.5);
        const int starved = 50;
        obs = obs.Where(o => o.Camera != starved).ToList();   // camera 50 keeps its pairs, loses every track observation
        var start = cams.Select(Copy).ToList();
        foreach (var c in start) c.Position += new Vector3((float)rng.NextDouble() - 0.5f, 0, (float)rng.NextDouble() - 0.5f) * 0.3f;
        string summary = GlobalSfmInit.Apply(start, edges, obs, points, focal);
        TestContext.Out.WriteLine(summary);
        Assert.That(summary, Does.Contain("1 connected camera(s) without 2 positioning observations"));
        var est = start.Cast<CameraParams?>().ToList();
        est[starved] = null;
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, cams.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        Assert.That(acc.MedianPosFrac, Is.LessThan(0.01), "every other camera placed");
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

    /// <summary>
    /// Leave-one-out reprojection skips cameras with no pose (2026-10-01, DrJohnson b73/b75/b76): 21 of 44 cameras were
    /// placeholders the global init could not place. Evaluated, they missed by 1e6 px (behind the camera), the median
    /// "typical" miss became 1e6 px and the misplaced limit 4e6 px - every real misplacement passed. Truck poses,
    /// synthetic tracks; cameras 0-11 get a placeholder pose (camera 40's, turned around).
    /// </summary>
    [Test]
    public void LeaveOneOut_SkipsUnplacedCameras()
    {
        var cams = TruckCameras();
        if (cams == null) Assert.Ignore("Truck poses not present");
        var rng = new Random(3);
        var (obs, points, _) = SyntheticTracks(cams, rng, 4000, 0.5, realLengths: true);
        var unplaced = new HashSet<int>(Enumerable.Range(0, 12));
        var posed = cams.Select(Copy).ToList();
        foreach (int i in unplaced)
        {
            var p = Copy(cams[40]);
            p.Forward = -p.Forward;
            posed[i] = p;
        }
        var tracks = obs.GroupBy(o => o.Point)
            .Select(g => (IReadOnlyList<(int Camera, float U, float V)>)g.Select(o => (o.Camera, o.U, o.V)).ToList()).ToList();
        var (medians, counts) = GlobalSfmInit.LeaveOneOutMedians(posed, tracks, unplaced);
        foreach (int i in unplaced)
            Assert.That(double.IsNaN(medians[i]), $"camera {i} has no pose and must not be evaluated (median {medians[i]})");
        var placed = Enumerable.Range(0, cams.Count).Where(i => !unplaced.Contains(i) && counts[i] > 0).Select(i => medians[i]).OrderBy(m => m).ToList();
        Assert.That(placed.Count, Is.GreaterThan(cams.Count / 2));
        Assert.That(placed[placed.Count / 2], Is.LessThan(3.0), "typical placed camera misses by ~noise");
    }
}
