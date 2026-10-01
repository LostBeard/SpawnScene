using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// BA's scale gauge (2026-09-30): from the global init, TruckFull BA shrank the scene ~400x while fitting at 0.88 px, and
/// training on that scale collapsed (held-out PSNR 4.7). ScaleGaugedSolution must restore the input centroid and spread
/// and keep every camera-to-point direction.
/// </summary>
public class ScaleGaugedSolutionTests
{
    private sealed class Drifted : IBundleSolution
    {
        public Vector3[] C = [], X = [];
        public double SharedFocal => 500;
        public void WriteCamera(int i, CameraParams cam) { cam.Position = C[i]; cam.Forward = Vector3.UnitZ; cam.Up = Vector3.UnitY; }
        public Vector3 PointAt(int p) => X[p];
        public (int Total, int Kept, double MedianError)[] CameraStats() => [];
        public int[] KeptObservationsPerPoint() => [];
        public string TimingSummary() => "";
    }

    [Test]
    public void RestoresInputScaleAndKeepsBearings()
    {
        var rng = new Random(3);
        Vector3 R() => new((float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2);
        int n = 30, np = 200;
        var input = new List<CameraParams>();
        for (int i = 0; i < n; i++) input.Add(new CameraParams { Position = R() });
        var pts = Enumerable.Range(0, np).Select(_ => R() * 3).ToArray();
        // The solve: same shape, shrunk 400x and moved (what BA did from the global init).
        var off = new Vector3(0.15f, 0f, -0.9f);
        var sol = new Drifted
        {
            C = input.Select(c => c.Position / 400f + off).ToArray(),
            X = pts.Select(p => p / 400f + off).ToArray(),
        };
        var exclude = new HashSet<int> { 7 };
        sol.C[7] = new Vector3(50, 50, 50);   // an excluded camera must not count

        var g = ScaleGaugedSolution.Create(sol, input, exclude, out double drift);
        Assert.That(drift, Is.EqualTo(1 / 400.0).Within(1e-4 / 400));
        var cam = new CameraParams();
        for (int i = 0; i < n; i++)
        {
            if (exclude.Contains(i)) continue;
            g.WriteCamera(i, cam);
            Assert.That(Vector3.Distance(cam.Position, input[i].Position), Is.LessThan(1e-3f), $"camera {i}");
        }
        for (int p = 0; p < np; p++)
            Assert.That(Vector3.Distance(g.PointAt(p), pts[p]), Is.LessThan(3e-3f), $"point {p}");
    }

    /// <summary>ToCurrentFrame's trusted mask (2026-09-30): views the depth cascade could not pose enter SfM with a
    /// placeholder pose (a copy of one posed view's). Even when they are the MAJORITY they must not set the frame; the
    /// solution must land on the trusted cameras' own centre and scale.</summary>
    [Test]
    public void ToCurrentFrame_PlaceholdersDoNotSetTheFrame()
    {
        var rng = new Random(7);
        Vector3 R() => new((float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2);
        int n = 70;
        var truth = Enumerable.Range(0, n).Select(_ => R()).ToArray();
        var cams = new List<CameraParams>();
        var trusted = new bool[n];
        for (int i = 0; i < n; i++)
        {
            trusted[i] = i < 30;
            cams.Add(new CameraParams { Position = trusted[i] ? truth[i] : truth[0] });   // placeholders: one copied pose
        }
        var connected = Enumerable.Repeat(true, n).ToArray();
        var sol = truth.Select(p => p * 0.01f + new Vector3(3, 1, -4)).ToArray();
        var centres = GlobalSfmInit.ToCurrentFrame(cams, connected, i => sol[i], trusted);
        var errs = Enumerable.Range(0, n).Select(i => Vector3.Distance(centres[i], truth[i])).OrderBy(e => e).ToArray();
        Assert.That(errs[n / 2], Is.LessThan(0.1f), "median camera error (spread ~2)");
        Assert.That(errs[^1], Is.LessThan(0.25f), "worst camera, placeholders included, lands on its true place");
    }

    /// <summary>ToCurrentFrame with ONE camera both connected and trusted (2026-10-01, DrJohnson b73/b75: the depth cascade
    /// posed 6 of 44 views, the global init connected 23, and they overlapped in one). The frame match used that single
    /// camera's spread - zero - so the scale was 0 and every connected camera landed on one point (leave-one-out 1e6 px,
    /// BA collapsed). Too few shared cameras must fall back to the trusted cameras' centre and spread.</summary>
    [Test]
    public void ToCurrentFrame_OneSharedCamera_DoesNotCollapse()
    {
        var rng = new Random(11);
        Vector3 R() => new((float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2);
        int n = 44;
        var truth = Enumerable.Range(0, n).Select(_ => R()).ToArray();
        var cams = new List<CameraParams>();
        var trusted = new bool[n];
        var connected = new bool[n];
        for (int i = 0; i < n; i++)
        {
            trusted[i] = i < 6;
            connected[i] = i >= 5 && i < 28;   // 23 connected; only camera 5 is also trusted
            cams.Add(new CameraParams { Position = trusted[i] ? truth[i] : truth[0] });
        }
        var sol = truth.Select(p => p * 0.01f + new Vector3(3, 1, -4)).ToArray();
        var centres = GlobalSfmInit.ToCurrentFrame(cams, connected, i => sol[i], trusted);
        var placed = Enumerable.Range(0, n).Where(i => connected[i]).Select(i => centres[i]).ToArray();
        var mid = placed.Aggregate(Vector3.Zero, (a, b) => a + b) / placed.Length;
        var spread = placed.Select(p => (p - mid).Length()).OrderBy(d => d).ToArray()[placed.Length / 2];
        // The truth's own median spread is ~1.6; mapped onto the 6 trusted cameras' spread it stays the same order.
        Assert.That(spread, Is.GreaterThan(0.3f), "connected cameras must keep a spread (scale 0 collapsed them to a point)");
        Assert.That(centres.All(c => float.IsFinite(c.X) && float.IsFinite(c.Y) && float.IsFinite(c.Z)), "finite centres");
    }

    /// <summary>GlobalSfmInit.ToCurrentFrame (2026-09-30): the positioning left 3 of 251 TruckFull cameras wildly off; the
    /// mean/RMS frame match let them carry the spread and shrank the real cluster ~390x. A robust match must place the
    /// inliers at the current frame's scale.</summary>
    [Test]
    public void ToCurrentFrame_IgnoresFarOutliers()
    {
        var rng = new Random(5);
        Vector3 R() => new((float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2, (float)rng.NextDouble() * 4 - 2);
        int n = 251;
        var cams = new List<CameraParams>();
        for (int i = 0; i < n; i++) cams.Add(new CameraParams { Position = R() });
        var connected = Enumerable.Repeat(true, n).ToArray();
        // The solution: the same shape at 1/1000 scale, moved - except 3 cameras thrown far away.
        var sol = cams.Select(c => c.Position / 1000f + new Vector3(7, -3, 2)).ToArray();
        sol[65] = new Vector3(900, 0, 0); sol[93] = new Vector3(0, -800, 0); sol[213] = new Vector3(0, 0, 1200);
        var centres = GlobalSfmInit.ToCurrentFrame(cams, connected, i => sol[i]);
        var errs = new List<float>();
        for (int i = 0; i < n; i++)
            if (i is not (65 or 93 or 213)) errs.Add(Vector3.Distance(centres[i], cams[i].Position));
        errs.Sort();
        // The median centre of a random blob is not its exact centroid, so allow a small offset; scale must be ~1.
        Assert.That(errs[errs.Count / 2], Is.LessThan(0.05f), "median inlier camera error (spread ~2)");
        Assert.That(errs[^1], Is.LessThan(0.1f), "worst inlier camera error");
    }
}
