using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Re-register a camera from 2D-3D correspondences. MEASURED on Truck: BA cannot rescue cameras the
/// DAv3 cascade placed 60-156 degrees wrong; this places them from the points the others agree on.
/// </summary>
public class CameraResectionTests
{
    [TestCase(0f)]
    [TestCase(150f)]   // what the cascade handed us for Truck view 38 - resection must not care
    public void RecoversThePose_With30PercentOutliers(float cascadeErrorDeg)
    {
        var rng = new Random(3);
        // Up must be orthogonal to Forward: WorldSpaceGeometry does not re-orthogonalise it, so a tilted
        // forward with Up = UnitY is not a rigid camera and no pose can fit its projections.
        var fwdT = Vector3.Normalize(new Vector3(-0.25f, -0.12f, -1f));
        var upT = Vector3.Normalize(Vector3.Cross(Vector3.Cross(fwdT, Vector3.UnitY), fwdT));
        var truth = new CameraParams
        {
            Width = 979, Height = 546, FocalX = 582, FocalY = 582, CenterX = 489.5f, CenterY = 273f,
            Position = new Vector3(2.1f, 1.3f, 7.4f),
            Forward = fwdT,
            Up = upT,
        };
        var world = new List<Vector3>();
        var px = new List<Vector2>();
        while (world.Count < 300)
        {
            var p = new Vector3((float)(rng.NextDouble() * 8 - 4), (float)(rng.NextDouble() * 3 - 0.5), (float)(rng.NextDouble() * 4 - 2));
            if (!WorldSpaceGeometry.Project(truth, p, out var u, out var v, out _) || u < 0 || v < 0 || u >= 979 || v >= 546) continue;
            world.Add(p);
            px.Add(rng.NextDouble() < 0.3
                ? new Vector2((float)(rng.NextDouble() * 979), (float)(rng.NextDouble() * 546))
                : new Vector2(u + (float)(rng.NextDouble() - 0.5), v + (float)(rng.NextDouble() - 0.5)));
        }
        // Only the INTRINSICS come from the prior; the pose is ignored entirely, however wrong it was.
        var prior = new CameraParams
        {
            Width = 979, Height = 546, FocalX = 582, FocalY = 582, CenterX = 489.5f, CenterY = 273f,
            Forward = Vector3.Transform(truth.Forward, Quaternion.CreateFromAxisAngle(Vector3.UnitY, cascadeErrorDeg * MathF.PI / 180f)),
            Up = Vector3.UnitY,
        };
        Assert.That(CameraResection.ResectRansac(world, px, prior, out var cam, out int inl, thresholdPx: 3), Is.True);
        float posErr = Vector3.Distance(cam.Position, truth.Position) / truth.Position.Length();
        float fwdErr = MathF.Acos(Math.Clamp(Vector3.Dot(Vector3.Normalize(cam.Forward), truth.Forward), -1f, 1f)) * 180f / MathF.PI;
        TestContext.Out.WriteLine($"inliers {inl}/{world.Count}, position error {posErr:P3}, forward {fwdErr:F3} deg");
        Assert.That(inl, Is.GreaterThanOrEqualTo(190));
        Assert.That(posErr, Is.LessThan(0.01f));
        Assert.That(fwdErr, Is.LessThan(0.3f));
    }

    /// <summary>
    /// OpenCV axes must be orthonormal even when Up is not perpendicular to Forward (world-up on a tilted
    /// camera). They used to use down = -up, a skewed basis that no rigid pose reproduces.
    /// </summary>
    [Test]
    public void OpenCvAxes_AreOrthonormal_ForATiltedCameraWithWorldUp()
    {
        var cam = new CameraParams { Forward = Vector3.Normalize(new Vector3(-0.25f, -0.4f, -1f)), Up = Vector3.UnitY };
        WorldSpaceGeometry.GetOpenCvAxes(cam, out var r, out var d, out var f);
        Assert.That(MathF.Abs(Vector3.Dot(r, d)), Is.LessThan(1e-6f));
        Assert.That(MathF.Abs(Vector3.Dot(d, f)), Is.LessThan(1e-6f), "down must be perpendicular to forward");
        Assert.That(MathF.Abs(Vector3.Dot(r, f)), Is.LessThan(1e-6f));
        Assert.That(d.Length(), Is.EqualTo(1f).Within(1e-6f));
        Assert.That(Vector3.Distance(Vector3.Cross(r, d), f), Is.LessThan(1e-6f), "right-handed: right x down = forward");
        Assert.That(Vector3.Dot(d, Vector3.UnitY), Is.LessThan(0f), "down still points down");
    }

    /// <summary>
    /// Real Truck data (views 79 and 34, dumped by &amp;badump=1): OpenCV's solvePnPRansac places them with
    /// 80/87 and 27/51 correspondences agreeing at 8 px; this resection returned nothing, and the pipeline
    /// dropped the views. The points are tightly clustered relative to their distance (near-affine).
    /// </summary>
    [TestCase("truck_resection_view79.txt", 75)]
    [TestCase("truck_resection_view34.txt", 24)]
    public void RealTruckViews_AreRegistered_LikeOpenCv(string file, int minInliers)
    {
        var world = new List<Vector3>();
        var px = new List<Vector2>();
        foreach (var line in File.ReadAllLines(Path.Combine(TestContext.CurrentContext.TestDirectory, "Data", file)))
        {
            var v = line.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(t => float.Parse(t, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            world.Add(new Vector3(v[0], v[1], v[2]));
            px.Add(new Vector2(v[3], v[4]));
        }
        var k = new CameraParams { Width = 979, Height = 546, FocalX = 582.6043f, FocalY = 582.6043f, CenterX = 489.5f, CenterY = 273f };
        bool ok = CameraResection.ResectRansac(world, px, k, out var cam, out int inl, thresholdPx: 8, iterations: 500, seed: 3);
        TestContext.Out.WriteLine($"{file}: ok {ok}, inliers {inl}/{world.Count}");
        Assert.That(ok, Is.True);
        Assert.That(inl, Is.GreaterThanOrEqualTo(minInliers));
    }
}
