using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Bundle adjustment at Truck's scale on Truck's geometry: the 126 real COLMAP cameras (poses.par) and real COLMAP
/// points (points3d.bin), with tracks drawn to Truck's measured track-length mix, 0.5 px noise, 3% outlier
/// observations and DAv3-grade initial poses and focal. MEASURED 2026-09-24 in the browser on the real no-COLMAP
/// path: one BA solve of this size took 348 LM iterations / 503 attempts, 78 s (build 15.3 s, Schur+CG 61 s).
/// This reproduces the problem natively so the solver can be profiled and a faster one proven equal.
/// </summary>
public class BundleAdjusterTruckScaleTests
{
    const int W = 979, H = 546;

    static string DatasetFile(string name) => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
        "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", "Truck", name));

    static float Gauss(Random rng) =>
        (float)(Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble()));

    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    /// <summary>The Truck problem, deterministic for a seed. Null when the dataset is not in this checkout.</summary>
    internal static (List<CameraParams> truth, List<CameraParams> init, List<Vector3> points,
        List<BundleAdjuster.Observation> obs)? BuildTruckProblem(int seed, int tracksWanted = 21_000)
    {
        string posesPath = DatasetFile("poses.par"), pointsPath = DatasetFile("points3d.bin");
        if (!File.Exists(posesPath) || !File.Exists(pointsPath)) return null;
        var truth = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(posesPath), 1957, 1091)
            .OrderBy(e => e.filename, StringComparer.Ordinal)
            .Select(e => e.camera.ScaledTo(W, H)).ToList();
        var cloud = SparsePointCloudInit.Parse(File.ReadAllBytes(pointsPath)).Positions;

        var rng = new Random(seed);
        // Truck's measured views-per-track mix (tuvok-truckphoto-7k): 2:14186 3:4078 4:1744 5:891 6+:1440.
        double[] cumulative = { 0.635, 0.817, 0.895, 0.935, 1.0 };
        var order = Enumerable.Range(0, cloud.Length).OrderBy(_ => rng.Next()).ToList();
        var tracks = new List<List<(int Camera, float U, float V)>>();
        var visible = new List<(int c, float u, float v)>();
        foreach (int p in order)
        {
            if (tracks.Count >= tracksWanted) break;
            visible.Clear();
            for (int c = 0; c < truth.Count; c++)
                if (WorldSpaceGeometry.Project(truth[c], cloud[p], out var u, out var v, out var z)
                    && z > 0.1f && u >= 0 && v >= 0 && u < W && v < H)
                    visible.Add((c, u, v));
            if (visible.Count < 2) continue;
            double r = rng.NextDouble();
            int len = r < cumulative[0] ? 2 : r < cumulative[1] ? 3 : r < cumulative[2] ? 4 : r < cumulative[3] ? 5 : 6 + rng.Next(10);
            len = Math.Min(len, visible.Count);
            // Consecutive visible views: a feature is tracked along the capture path.
            int start = rng.Next(visible.Count - len + 1);
            var track = new List<(int, float, float)>(len);
            for (int k = start; k < start + len; k++)
            {
                var (c, u, v) = visible[k];
                u += 0.5f * Gauss(rng); v += 0.5f * Gauss(rng);
                if (rng.NextDouble() < 0.03) { u = (float)(rng.NextDouble() * W); v = (float)(rng.NextDouble() * H); }
                track.Add((c, u, v));
            }
            tracks.Add(track);
        }

        // DAv3-grade start: centres off by ~5% of the rig spread, ~2 deg of rotation, focal 5% high. Camera 0 is
        // the gauge, as in production (fixedCamera = first participating view).
        var centroid = Vector3.Zero;
        foreach (var c in truth) centroid += c.Position;
        centroid /= truth.Count;
        float spread = truth.Max(c => Vector3.Distance(c.Position, centroid));
        var init = new List<CameraParams>();
        for (int c = 0; c < truth.Count; c++)
        {
            var p = Copy(truth[c]);
            p.FocalX *= 1.05f; p.FocalY *= 1.05f;
            if (c > 0)
            {
                p.Position += new Vector3(Gauss(rng), Gauss(rng), Gauss(rng)) * (0.05f * spread / 1.7f);
                var axis = Vector3.Normalize(new Vector3(Gauss(rng), Gauss(rng), Gauss(rng)));
                var q = Quaternion.CreateFromAxisAngle(axis, 2f * MathF.PI / 180f * Gauss(rng));
                p.Forward = Vector3.Normalize(Vector3.Transform(p.Forward, q));
                p.Up = Vector3.Normalize(Vector3.Transform(p.Up, q));
            }
            init.Add(p);
        }

        var points = new List<Vector3>();
        var obs = new List<BundleAdjuster.Observation>();
        foreach (var track in tracks)
        {
            if (!BundleAdjuster.Triangulate(init, track, out var x)) continue;
            int id = points.Count;
            points.Add(x);
            foreach (var (c, u, v) in track) obs.Add(new BundleAdjuster.Observation(c, id, u, v));
        }
        return (truth, init, points, obs);
    }

    /// <summary>Production settings (MultiViewGenerationService.BundleAdjust): 150 iterations, 3 rounds, shared focal.</summary>
    internal static (BundleAdjuster ba, BundleAdjuster.Result result) SolveLikeProduction(
        List<CameraParams> init, List<Vector3> points, List<BundleAdjuster.Observation> obs)
    {
        var ba = new BundleAdjuster(init, points, obs, fixedCamera: 0, sharedFocal: true);
        var result = ba.Solve(new BundleAdjuster.Options
        {
            MaxIterations = 150,
            Rounds = 3,
            IterationLog = Environment.GetEnvironmentVariable("BA_ITERLOG") == "1"
                ? (it, cost, rel, mc, mp) => TestContext.Out.WriteLine($"    it {it,3} cost {cost:G6} rel {rel:E2} max|dCam| {mc:E2} max|dPt| {mp:E2}")
                : null,
            RoundLog = (round, iters, rms, kept) =>
                TestContext.Out.WriteLine($"  round {round}: {iters} iterations, RMS {rms:F2} px, {kept} obs kept"),
        });
        return (ba, result);
    }

    [Test]
    public void TruckScale_RefinesToSfmGrade_AndReportsWhereTheTimeGoes()
    {
        var problem = BuildTruckProblem(seed: 5);
        if (problem == null) Assert.Ignore("Truck dataset not generated in this checkout");
        var (truth, init, points, obs) = problem.Value;

        var (ba, result) = SolveLikeProduction(init, points, obs);
        var refined = init.Select(Copy).ToList();
        for (int c = 0; c < refined.Count; c++) ba.WriteCamera(c, refined[c]);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            refined.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            init.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var acc0, out _, out _), Is.True);

        TestContext.Out.WriteLine(
            $"Truck-scale BA: {truth.Count} cams, {points.Count} points, {result.Observations} obs ({result.ObservationsKept} kept), " +
            $"{result.Iterations} iters, RMS {result.InitialRmsPixels:F1} -> {result.FinalRmsPixels:F3} px, {result.Seconds:F1}s");
        TestContext.Out.WriteLine($"  timing: {ba.TimingSummary()}");
        TestContext.Out.WriteLine(
            $"  pose vs truth: {acc0.PositionRms / acc0.Spread:P2} -> {acc.PositionRms / acc.Spread:P3} of spread, " +
            $"fwd {acc0.MedianForwardDeg:F2} -> {acc.MedianForwardDeg:F3} deg");

        if (Environment.GetEnvironmentVariable("BA_ITERLOG") == "1")
        {
            var cen = Vector3.Zero; foreach (var c in truth) cen += c.Position; cen /= truth.Count;
            float spr = truth.Max(c => Vector3.Distance(c.Position, cen));
            var kept = ba.KeptObservationsPerPoint();
            var far = Enumerable.Range(0, points.Count).Select(p => (p, d: Vector3.Distance(ba.PointAt(p), cen) / spr))
                .OrderByDescending(t => t.d).Take(8);
            foreach (var (p, d) in far)
                TestContext.Out.WriteLine($"  far point {p}: {d:F1} spreads out (start {Vector3.Distance(points[p], cen) / spr:F1}), kept obs {kept[p]}");
            int farCount = Enumerable.Range(0, points.Count).Count(p => Vector3.Distance(ba.PointAt(p), cen) > 20 * spr);
            TestContext.Out.WriteLine($"  points beyond 20 spreads: {farCount}; with 0 kept obs: {kept.Count(k => k == 0)}");
        }

        // Fingerprint of the solution, for proving a faster solver computes the same one.
        double fp = 0;
        for (int c = 0; c < refined.Count; c++)
            fp += refined[c].Position.X * (c + 1) + refined[c].Position.Y * 0.5 * (c + 1) + refined[c].Forward.Z * 7;
        TestContext.Out.WriteLine($"  solution fingerprint {fp:R}, focal {refined[1].FocalX:R}");

        Assert.That(acc0.PositionRms / acc0.Spread, Is.GreaterThan(0.02f), "the start must be DAv3-grade to mean anything");
        Assert.That(acc.PositionRms / acc.Spread, Is.LessThan(0.005f), "BA must reach SfM grade (< 0.5% of spread)");
        Assert.That(result.FinalRmsPixels, Is.LessThan(1.0));
    }
}
