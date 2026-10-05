using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The block plan for partitioned training (ScenePartition): every point kept by exactly one block, cells with equal
/// camera counts along the capture's long axis, and each block supervised by its own cameras plus the ones that see it.
/// </summary>
public class ScenePartitionTests
{
    static CameraParams Cam(Vector3 pos, Vector3 forward, int w = 800, int h = 600, float f = 600f) => new()
    {
        Position = pos, Forward = Vector3.Normalize(forward), Up = Vector3.UnitY,
        Width = w, Height = h, FocalX = f, FocalY = f, CenterX = w / 2f, CenterY = h / 2f,
    };

    /// <summary>A street: 40 cameras walking along x at z = 0 looking at a facade at z = 10, points on the facade.</summary>
    static (List<CameraParams> Cams, Vector3[] Points) Street(int cams = 40, float length = 80f)
    {
        var list = new List<CameraParams>();
        for (int i = 0; i < cams; i++) list.Add(Cam(new Vector3(i * length / (cams - 1), 1.6f, 0f), Vector3.UnitZ));
        var rng = new Random(3);
        var pts = new Vector3[4000];
        for (int i = 0; i < pts.Length; i++)
            pts[i] = new Vector3((float)rng.NextDouble() * (length + 10f) - 5f, (float)rng.NextDouble() * 8f, 10f + (float)rng.NextDouble());
        return (list, pts);
    }

    [Test]
    public void EveryPoint_HasExactlyOneOwner()
    {
        var (cams, pts) = Street();
        var plan = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 3, Rows: 2));
        Assert.That(plan.Blocks, Has.Length.EqualTo(6));
        var rng = new Random(9);
        for (int k = 0; k < 5000; k++)
        {
            // Anywhere, including far outside the capture.
            var p = new Vector3((float)(rng.NextDouble() * 400 - 200), (float)(rng.NextDouble() * 50 - 25), (float)(rng.NextDouble() * 400 - 200));
            var q = plan.ToPlane(p);
            Assert.That(plan.Blocks.Count(b => b.Owns(q)), Is.EqualTo(1), $"{p}");
        }
    }

    [Test]
    public void Columns_FollowTheStreet_WithEqualCameraCounts()
    {
        var (cams, pts) = Street();
        var plan = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 4, Rows: 1));
        Assert.That(MathF.Abs(plan.AxisU.X), Is.GreaterThan(0.99f), $"long axis {plan.AxisU}");
        Assert.That(plan.Up.Y, Is.GreaterThan(0.99f));
        foreach (var b in plan.Blocks) Assert.That(b.OwnViews, Is.EqualTo(10), $"block {b.Index}");
        // Each block's points: the quarter of the facade in front of its cameras, not the whole street.
        foreach (var b in plan.Blocks) Assert.That(b.Points, Is.LessThan(pts.Length / 2), $"block {b.Index} points");
    }

    [Test]
    public void Views_IncludeOwnCameras_AndOnesThatSeeTheCell()
    {
        var (cams, pts) = Street();
        // A camera in the second cell, past the first one's margin, turned back toward the first cell's facade
        // (about a fifth of its image).
        cams.Add(Cam(new Vector3(30f, 1.6f, 0f), new Vector3(-15f, 0f, 10f)));
        int looker = cams.Count - 1;
        var plan = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 4, Rows: 1, Overlap: 0.2f, Visibility: 0.1f));
        var first = plan.Blocks.Single(b => b.Owns(plan.ToPlane(new Vector3(2f, 0f, 10f))));
        for (int i = 0; i < 40; i++)
            if (first.Owns(plan.ToPlane(cams[i].Position))) Assert.That(first.Views, Does.Contain(i));
        Assert.That(first.Owns(plan.ToPlane(cams[looker].Position)), Is.False);
        Assert.That(first.Views, Does.Contain(looker), "a camera seeing the cell from outside it should train it");
        // A camera at the other end looking at its own facade does not see the first cell.
        Assert.That(first.Views, Does.Not.Contain(39));
    }

    [Test]
    public void TrainingBox_ContainsTheCore_AndOverlapsTheNeighbour()
    {
        var (cams, pts) = Street();
        var plan = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 2, Rows: 1, Overlap: 0.2f));
        var (a, b) = (plan.Blocks[0], plan.Blocks[1]);
        // Order the two along u so 'lo' is the one with the open lower edge.
        var lo = float.IsNegativeInfinity(a.CoreMin.X) ? a : b;
        var hi = lo == a ? b : a;
        Assert.That(lo.TrainMax.X, Is.GreaterThan(lo.CoreMax.X));
        Assert.That(hi.TrainMin.X, Is.LessThan(hi.CoreMin.X));
        Assert.That(lo.TrainMax.X, Is.GreaterThan(hi.CoreMin.X), "the training boxes overlap across the seam");
    }

    [Test]
    public void Coverage_IsTheImageShareOfTheVisiblePoints()
    {
        var cam = Cam(Vector3.Zero, Vector3.UnitZ, 800, 600, 400f);
        // A square at z = 10 spanning x, y in [-5, 5]: projects to 400 x 400 px, centred.
        var square = new List<Vector3>();
        for (int i = 0; i <= 10; i++) for (int j = 0; j <= 10; j++) square.Add(new Vector3(i - 5f, j - 5f, 10f));
        Assert.That(ScenePartition.CoverageOf(cam, square), Is.EqualTo(400f * 400f / (800f * 600f)).Within(1e-3f));
        // Behind the camera: nothing.
        var behind = square.Select(p => p with { Z = -10f }).ToList();
        Assert.That(ScenePartition.CoverageOf(cam, behind), Is.EqualTo(0f));
        // Up in the image is +y in the world: points above the axis land in the top half (smaller v).
        var above = new List<Vector3> { new(-1f, 1f, 10f), new(1f, 3f, 10f) };
        Assert.That(ScenePartition.CoverageOf(cam, above), Is.EqualTo(80f * 80f / (800f * 600f)).Within(1e-3f));
    }

    [Test]
    public void Floaters_DoNotWidenTheTrainingBoxes()
    {
        var (cams, pts) = Street();
        // A few SfM floaters kilometres out along the street.
        var withFloaters = pts.Concat(new[] { new Vector3(-5000f, 0f, 10f), new Vector3(9000f, 3f, 12f), new Vector3(40f, 0f, 8000f) }).ToArray();
        var clean = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 4, Rows: 1));
        var noisy = ScenePartition.Make(cams, withFloaters, new ScenePartition.Options(Columns: 4, Rows: 1));
        for (int b = 0; b < 4; b++)
        {
            Assert.That(noisy.Blocks[b].Points, Is.LessThan(withFloaters.Length / 2), $"block {b} trains most of the scene");
            Assert.That(MathF.Abs(noisy.Blocks[b].TrainMax.X - clean.Blocks[b].TrainMax.X), Is.LessThan(2f).Or.NaN, $"block {b} upper margin");
        }
    }

    [Test]
    public void OneBlock_TrainsEverything()
    {
        var (cams, pts) = Street(12, 20f);
        var plan = ScenePartition.Make(cams, pts, new ScenePartition.Options(Columns: 1, Rows: 1));
        var only = plan.Blocks.Single();
        Assert.That(only.Views, Is.EqualTo(Enumerable.Range(0, 12).ToArray()));
        Assert.That(only.Points, Is.EqualTo(pts.Length));
    }
}
