using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The pose-vs-COLMAP metric (2026-09-30): a plain least-squares alignment let TruckFull's 3 misplaced global-init
/// cameras drag the fit - "median 100.66%" for an init that BA turned into 0.10%. The alignment must be robust, and the
/// misplaced cameras must still be counted and reported as the worst.
/// </summary>
public class CameraSetAccuracyRobustTests
{
    [Test]
    public void FarOutliersDoNotDragTheAlignment()
    {
        var rng = new Random(11);
        int n = 251;
        var gt = new CameraParams?[n];
        var est = new CameraParams?[n];
        var rot = Matrix4x4.CreateFromYawPitchRoll(0.7f, -0.3f, 1.1f);
        for (int i = 0; i < n; i++)
        {
            float a = i * 0.05f;
            var p = new Vector3(MathF.Cos(a) * 3, 0.2f * MathF.Sin(3 * a), MathF.Sin(a) * 3);
            var f = Vector3.Normalize(-p + new Vector3(0, 0.1f, 0));
            gt[i] = new CameraParams { Position = p, Forward = f, Up = Vector3.UnitY };
            // The estimate: the same poses under a similarity (scale 0.01, rotated, moved).
            est[i] = new CameraParams
            {
                Position = 0.01f * Vector3.Transform(p, rot) + new Vector3(5, -2, 7),
                Forward = Vector3.TransformNormal(f, rot), Up = Vector3.TransformNormal(Vector3.UnitY, rot),
            };
        }
        int[] far = [65, 93, 213];
        foreach (int i in far) est[i]!.Position = new Vector3(rng.Next(200, 900), rng.Next(-900, -200), rng.Next(300, 800));

        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, gt, out var acc, out var posFrac, out _), Is.True);
        Assert.That(acc.MedianPosFrac, Is.LessThan(1e-3f), "median position error of the 248 exact cameras");
        Assert.That(acc.MedianForwardDeg, Is.LessThan(0.1f), "median forward error");
        Assert.That(acc.AlignedOn, Is.EqualTo(n - far.Length));
        Assert.That(acc.Scale, Is.EqualTo(100f).Within(0.5f));
        // The misplaced cameras are still measured - and are the three worst.
        var worst3 = Enumerable.Range(0, posFrac.Length).OrderByDescending(i => posFrac[i]).Take(3).OrderBy(i => i).ToArray();
        Assert.That(worst3, Is.EqualTo(far));
    }
}
