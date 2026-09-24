using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Bundle adjustment must turn DAv3-grade poses into SfM-grade ones. MEASURED 2026-09-24 on Truck: the
/// cascade's poses (median 7.8% of camera spread, p90 27%) shred a splat scene that COLMAP poses render
/// cleanly. The rig here reproduces that error level on known geometry.
/// </summary>
public class BundleAdjusterTests
{
    const int W = 980, H = 546;

    static CameraParams Cam(Vector3 pos, Vector3 target, float f = 900)
    {
        var fwd = Vector3.Normalize(target - pos);
        var right = Vector3.Normalize(Vector3.Cross(fwd, Vector3.UnitY));
        var up = Vector3.Normalize(Vector3.Cross(right, fwd));
        return new CameraParams
        {
            Width = W, Height = H, FocalX = f, FocalY = f, CenterX = W / 2f, CenterY = H / 2f,
            Position = pos, Forward = fwd, Up = up,
        };
    }

    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    /// <summary>A walk past a street scene: cameras on a 120-degree arc around a truck-sized point cloud.</summary>
    static (List<CameraParams> truth, List<Vector3> points) Rig(int cams, int points, int seed)
    {
        var rng = new Random(seed);
        var pts = new List<Vector3>();
        for (int i = 0; i < points; i++)
            pts.Add(new Vector3((float)(rng.NextDouble() * 8 - 4), (float)(rng.NextDouble() * 3 - 0.5), (float)(rng.NextDouble() * 4 - 2)));
        var truth = new List<CameraParams>();
        for (int i = 0; i < cams; i++)
        {
            double a = -Math.PI / 3 + 2 * Math.PI / 3 * i / (cams - 1);
            var pos = new Vector3((float)(9 * Math.Sin(a)), 1.2f + 0.3f * (float)Math.Sin(i * 0.7), (float)(9 * Math.Cos(a)));
            truth.Add(Cam(pos, new Vector3(0, 0.5f, 0)));
        }
        return (truth, pts);
    }

    static float Gauss(Random rng) =>
        (float)(Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble()));

    static CameraParams Perturb(CameraParams c, Random rng, float posSigma, float rotDeg)
    {
        var p = Copy(c);
        p.Position += new Vector3(Gauss(rng), Gauss(rng), Gauss(rng)) * posSigma;
        var axis = Vector3.Normalize(new Vector3(Gauss(rng), Gauss(rng), Gauss(rng)));
        var q = Quaternion.CreateFromAxisAngle(axis, rotDeg * MathF.PI / 180f * Gauss(rng));
        p.Forward = Vector3.Normalize(Vector3.Transform(c.Forward, q));
        p.Up = Vector3.Normalize(Vector3.Transform(c.Up, q));
        return p;
    }

    [Test]
    public void DAv3GradePoses_RefineToSfmGrade()
    {
        var rng = new Random(11);
        var (truth, pts) = Rig(cams: 40, points: 3000, seed: 3);

        // Observations: project with the TRUE cameras, 0.5 px noise, 10% replaced by a random pixel.
        var obs = new List<BundleAdjuster.Observation>();
        var perPoint = new List<(int Camera, float U, float V)>[pts.Count];
        for (int p = 0; p < pts.Count; p++)
        {
            perPoint[p] = new();
            for (int c = 0; c < truth.Count; c++)
            {
                if (!WorldSpaceGeometry.Project(truth[c], pts[p], out var u, out var v, out _)) continue;
                if (u < 0 || v < 0 || u >= W || v >= H) continue;
                if (rng.NextDouble() < 0.6) continue; // not every feature is detected in every view
                u += 0.5f * Gauss(rng); v += 0.5f * Gauss(rng);
                if (rng.NextDouble() < 0.10) { u = (float)(rng.NextDouble() * W); v = (float)(rng.NextDouble() * H); }
                perPoint[p].Add((c, u, v));
            }
        }

        // Start where DAv3 leaves us: camera centres off by ~8% of the spread, a few degrees of rotation.
        float spread = 9f;
        var init = new List<CameraParams> { Copy(truth[0]) }; // gauge: camera 0 is the reference
        for (int c = 1; c < truth.Count; c++) init.Add(Perturb(truth[c], rng, 0.08f * spread / 1.7f, 3f));

        var points = new List<Vector3>();
        foreach (var track in perPoint)
        {
            if (track.Count < 2 || !BundleAdjuster.Triangulate(init, track, out var x)) continue;
            int id = points.Count;
            points.Add(x);
            foreach (var (c, u, v) in track) obs.Add(new BundleAdjuster.Observation(c, id, u, v));
        }

        var before = Accuracy(init, truth);
        var ba = new BundleAdjuster(init, points, obs);
        var result = ba.Solve();
        var refined = init.Select(Copy).ToList();
        for (int c = 0; c < refined.Count; c++) ba.WriteCamera(c, refined[c]);
        var after = Accuracy(refined, truth);

        TestContext.Out.WriteLine(
            $"points {points.Count}, obs {result.Observations} kept {result.ObservationsKept}, iters {result.Iterations}, " +
            $"rms {result.InitialRmsPixels:F2} -> {result.FinalRmsPixels:F3} px, {result.Seconds:F2}s");
        TestContext.Out.WriteLine($"pose vs truth: before {before.pos:P2} / {before.fwd:F2} deg, after {after.pos:P3} / {after.fwd:F3} deg");
        TestContext.Out.WriteLine($"timing: {ba.TimingSummary()}");

        Assert.That(before.pos, Is.GreaterThan(0.03f), "the perturbation must be DAv3-sized for the test to mean anything");
        Assert.That(after.pos, Is.LessThan(0.005f), $"position error {after.pos:P2} of spread after BA");
        Assert.That(after.fwd, Is.LessThan(0.2f), $"forward error {after.fwd:F2} deg after BA");
        Assert.That(result.FinalRmsPixels, Is.LessThan(1.0), "inlier reprojection RMS should reach the 0.5 px noise level");
    }

    /// <summary>
    /// One camera, many frames, and DAv3 guessing the focal per frame (here +10% bias, +-8% scatter). Held
    /// fixed, the wrong focals bend the geometry; solved as ONE shared focal, BA recovers it and the poses.
    /// </summary>
    [Test]
    public void SharedFocal_RecoversTheCameraAndThePoses()
    {
        var rng = new Random(21);
        var (truth, pts) = Rig(cams: 40, points: 3000, seed: 4);
        const float trueF = 900;
        var perPoint = new List<(int Camera, float U, float V)>[pts.Count];
        for (int p = 0; p < pts.Count; p++)
        {
            perPoint[p] = new();
            for (int c = 0; c < truth.Count; c++)
            {
                if (!WorldSpaceGeometry.Project(truth[c], pts[p], out var u, out var v, out _)) continue;
                if (u < 0 || v < 0 || u >= W || v >= H || rng.NextDouble() < 0.6) continue;
                perPoint[p].Add((c, u + 0.5f * Gauss(rng), v + 0.5f * Gauss(rng)));
            }
        }
        List<CameraParams> Start(Random r)
        {
            var init = new List<CameraParams> { Copy(truth[0]) };
            for (int c = 1; c < truth.Count; c++) init.Add(Perturb(truth[c], r, 0.08f * 9f / 1.7f, 3f));
            foreach (var cam in init)
            {
                float f = trueF * 1.10f * (1 + 0.08f * Gauss(r));
                cam.FocalX = f; cam.FocalY = f;
            }
            return init;
        }

        (float pos, float fwd, double f) Run(bool shared)
        {
            var init = Start(new Random(77));
            var points = new List<Vector3>();
            var obs = new List<BundleAdjuster.Observation>();
            foreach (var track in perPoint)
            {
                if (track.Count < 2 || !BundleAdjuster.Triangulate(init, track, out var x)) continue;
                int id = points.Count;
                points.Add(x);
                foreach (var (c, u, v) in track) obs.Add(new BundleAdjuster.Observation(c, id, u, v));
            }
            var ba = new BundleAdjuster(init, points, obs, sharedFocal: shared);
            ba.Solve();
            for (int c = 0; c < init.Count; c++) ba.WriteCamera(c, init[c]);
            var (pos, fwd) = Accuracy(init, truth);
            return (pos, fwd, ba.SharedFocal);
        }

        var fixedF = Run(shared: false);
        var sharedF = Run(shared: true);
        TestContext.Out.WriteLine($"per-view fixed focal: {fixedF.pos:P3} / {fixedF.fwd:F3} deg");
        TestContext.Out.WriteLine($"shared focal: {sharedF.pos:P3} / {sharedF.fwd:F3} deg, f {sharedF.f:F1} (true {trueF})");
        Assert.That(Math.Abs(sharedF.f - trueF) / trueF, Is.LessThan(0.005), "shared focal within 0.5%");
        Assert.That(sharedF.pos, Is.LessThan(0.005f), "poses SfM-grade with the focal solved");
        Assert.That(fixedF.pos, Is.GreaterThan(2 * sharedF.pos), "wrong fixed focals must visibly bend the solution (negative control)");
    }

    static (float pos, float fwd) Accuracy(IReadOnlyList<CameraParams> est, IReadOnlyList<CameraParams> truth)
    {
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            est.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        return (acc.PositionRms / acc.Spread, acc.MedianForwardDeg);
    }

    [Test]
    public void Tracks_DropAnyTrackThatClaimsTwoFeaturesInOneImage()
    {
        var tracks = BundleAdjuster.BuildTracks(new[]
        {
            (0, 1, 1, 5), (1, 5, 2, 9),      // clean 3-view track
            (0, 2, 1, 6), (1, 6, 0, 3),      // image 0 twice (features 2 and 3): inconsistent
        });
        Assert.That(tracks.Count, Is.EqualTo(1));
        Assert.That(tracks[0], Is.EqualTo(new List<(int, int)> { (0, 1), (1, 5), (2, 9) }));
    }
}
