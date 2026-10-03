using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Homography RANSAC (Homography, 2026-10-03): planar pairs fit one, general 3D pairs do not.</summary>
public class HomographyTests
{
    static Vector2 Project(Vector3 x, Matrix4x4 view, float f) { var c = Vector3.Transform(x, view); return new(f * c.X / c.Z + 320, f * c.Y / c.Z + 240); }

    static (List<Vector2> A, List<Vector2> B) Views(Func<Random, Vector3> point, int n)
    {
        var rng = new Random(7);
        var va = Matrix4x4.CreateLookAt(new Vector3(0, 0, -5), Vector3.Zero, Vector3.UnitY) * Matrix4x4.CreateScale(1, 1, -1);
        var vb = Matrix4x4.CreateLookAt(new Vector3(2, 0.5f, -4.5f), Vector3.Zero, Vector3.UnitY) * Matrix4x4.CreateScale(1, 1, -1);
        var a = new List<Vector2>(); var b = new List<Vector2>();
        for (int i = 0; i < n; i++) { var x = point(rng); a.Add(Project(x, va, 500)); b.Add(Project(x, vb, 500)); }
        return (a, b);
    }

    [Test]
    public void PlanarPoints_AllFitOneHomography()
    {
        var (a, b) = Views(r => new Vector3((float)r.NextDouble() * 4 - 2, (float)r.NextDouble() * 4 - 2, 0), 200);
        Assert.That(Homography.InlierCount(a, b, 1f), Is.EqualTo(200));
    }

    [Test]
    public void DepthVaryingPoints_DoNot()
    {
        var (a, b) = Views(r => new Vector3((float)r.NextDouble() * 4 - 2, (float)r.NextDouble() * 4 - 2, (float)r.NextDouble() * 4 - 2), 200);
        int inl = Homography.InlierCount(a, b, 1f);
        TestContext.Out.WriteLine($"non-planar: {inl}/200 on the best homography");
        Assert.That(inl, Is.LessThan(100));
    }

    [Test]
    public void Solve4_ReproducesItsPoints()
    {
        var a = new List<Vector2> { new(0, 0), new(100, 0), new(100, 100), new(0, 100) };
        var b = new List<Vector2> { new(10, 5), new(120, 0), new(115, 110), new(0, 95) };
        Assert.That(Homography.Solve4(a, b, [0, 1, 2, 3], out var h), Is.True);
        for (int i = 0; i < 4; i++) Assert.That(Homography.TransferSq(h, a[i], b[i]), Is.LessThan(1e-6f));
    }
}
