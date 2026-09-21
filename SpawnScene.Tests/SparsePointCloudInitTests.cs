using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The grid nearest-neighbour search is an optimisation of an exact answer, so it is tested
/// against the exact answer - brute force - rather than against a stored number. That is the
/// one check that can fail if a ring is skipped, a boundary is off by one, or the early exit
/// fires too soon.
/// </summary>
[TestFixture]
public class SparsePointCloudInitTests
{
    static float[] BruteForceSpacing(Vector3[] pts)
    {
        var result = new float[pts.Length];
        for (int i = 0; i < pts.Length; i++)
        {
            var d = new List<float>();
            for (int j = 0; j < pts.Length; j++)
                if (j != i) d.Add(Vector3.DistanceSquared(pts[i], pts[j]));
            d.Sort();
            int k = Math.Min(SparsePointCloudInit.SpacingNeighbours, d.Count);
            float sum = 0;
            for (int m = 0; m < k; m++) sum += d[m];
            result[i] = MathF.Sqrt(MathF.Max(sum / k, SparsePointCloudInit.MinSpacingSq));
        }
        return result;
    }

    static Vector3[] RandomCloud(int n, int seed, float spread = 10f)
    {
        var rng = new Random(seed);
        var pts = new Vector3[n];
        for (int i = 0; i < n; i++)
            pts[i] = new Vector3(
                (float)(rng.NextDouble() - 0.5) * spread,
                (float)(rng.NextDouble() - 0.5) * spread,
                (float)(rng.NextDouble() - 0.5) * spread);
        return pts;
    }

    [Test]
    public void SpacingMatchesBruteForce()
    {
        foreach (int n in new[] { 8, 50, 400, 1500 })
        foreach (int seed in new[] { 1, 7 })
        {
            var pts = RandomCloud(n, seed);
            var fast = SparsePointCloudInit.LocalSpacing(pts);
            var slow = BruteForceSpacing(pts);

            for (int i = 0; i < n; i++)
                Assert.That(fast[i], Is.EqualTo(slow[i]).Within(1e-5f),
                    $"n={n} seed={seed} point {i}: grid search disagrees with brute force");
        }
    }

    [Test]
    public void SpacingMatchesBruteForceOnAClusteredCloud()
    {
        // A uniform cloud can hide a broken ring expansion, because the answer is always in the
        // first block. A real SfM cloud is clumpy: dense on texture, empty across a room.
        var rng = new Random(3);
        var pts = new List<Vector3>();
        for (int c = 0; c < 6; c++)
        {
            var centre = new Vector3(c * 40f, (c % 2) * 25f, 0);
            for (int i = 0; i < 60; i++)
                pts.Add(centre + new Vector3(
                    (float)rng.NextDouble() * 0.3f,
                    (float)rng.NextDouble() * 0.3f,
                    (float)rng.NextDouble() * 0.3f));
        }
        // One point alone in the void, whose neighbours are a whole cluster away.
        pts.Add(new Vector3(500, 500, 500));

        var arr = pts.ToArray();
        var fast = SparsePointCloudInit.LocalSpacing(arr);
        var slow = BruteForceSpacing(arr);
        for (int i = 0; i < arr.Length; i++)
            Assert.That(fast[i], Is.EqualTo(slow[i]).Within(1e-4f),
                $"clustered point {i} at {arr[i]}");
    }

    [Test]
    public void SpacingIsScaleProportional()
    {
        // A splat should be as big as the gap it covers, so doubling the point spacing must
        // double the scale. This is the property the depth path got wrong with a constant.
        var pts = RandomCloud(300, 11);
        var doubled = pts.Select(p => p * 2f).ToArray();

        var a = SparsePointCloudInit.LocalSpacing(pts);
        var b = SparsePointCloudInit.LocalSpacing(doubled);
        for (int i = 0; i < pts.Length; i++)
            Assert.That(b[i], Is.EqualTo(a[i] * 2f).Within(1e-4f), $"point {i}");
    }

    [Test]
    public void DuplicatePointsDoNotProduceZeroScale()
    {
        // A zero-scale Gaussian has no gradient and can never recover, so the reference clamps.
        var pts = new[]
        {
            new Vector3(1, 1, 1), new Vector3(1, 1, 1), new Vector3(1, 1, 1),
            new Vector3(1, 1, 1), new Vector3(1, 1, 1), new Vector3(5, 5, 5),
        };
        foreach (var s in SparsePointCloudInit.LocalSpacing(pts))
            Assert.That(s, Is.GreaterThan(0f), "a duplicated point got a zero scale");
    }

    [Test]
    public void PackedLayoutMatchesSplatFormat()
    {
        var cloud = new PointCloud
        {
            Positions = [new Vector3(1, 2, 3), new Vector3(4, 5, 6), new Vector3(1, 2, 4)],
            Colors = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)],
        };
        var packed = SparsePointCloudInit.BuildPacked(cloud);

        Assert.That(packed, Has.Length.EqualTo(3 * SplatFormat.Floats));
        Assert.That(packed[SplatFormat.OffPos], Is.EqualTo(1f));
        Assert.That(packed[SplatFormat.OffPos + 1], Is.EqualTo(2f));
        Assert.That(packed[SplatFormat.OffColor], Is.EqualTo(1f));
        Assert.That(packed[SplatFormat.OffOpacity], Is.EqualTo(SparsePointCloudInit.InitialOpacity));

        // Identity quaternion, xyzw - a w in the wrong lane rotates every splat and is invisible
        // on an isotropic one, so assert the lane rather than the norm.
        Assert.That(packed[SplatFormat.OffQuat], Is.EqualTo(0f));
        Assert.That(packed[SplatFormat.OffQuat + 3], Is.EqualTo(1f));

        for (int i = 0; i < 3; i++)
        {
            int o = i * SplatFormat.Floats;
            Assert.That(packed[o + SplatFormat.OffScale], Is.GreaterThan(0f), $"splat {i} scale");
            Assert.That(packed[o + SplatFormat.OffScale],
                Is.EqualTo(packed[o + SplatFormat.OffScale + 2]).Within(1e-6f),
                $"splat {i} should be isotropic");
        }
    }

    [Test]
    public void ParseRoundTripsTheConverterFormat()
    {
        // Byte-for-byte the layout tools/colmap_to_dataset.py writes: i32 count, then per point
        // 3x f32 and 4x u8. Built here by hand so the test fails if either side drifts.
        var bytes = new List<byte>();
        bytes.AddRange(BitConverter.GetBytes(2));
        foreach (var (p, c) in new[]
                 {
                     (new Vector3(1.5f, -2.5f, 3.5f), new byte[] { 255, 128, 0, 255 }),
                     (new Vector3(-9f, 0f, 0.25f), new byte[] { 0, 0, 255, 255 }),
                 })
        {
            bytes.AddRange(BitConverter.GetBytes(p.X));
            bytes.AddRange(BitConverter.GetBytes(p.Y));
            bytes.AddRange(BitConverter.GetBytes(p.Z));
            bytes.AddRange(c);
        }

        var cloud = SparsePointCloudInit.Parse(bytes.ToArray());
        Assert.That(cloud.Count, Is.EqualTo(2));
        Assert.That(cloud.Positions[0], Is.EqualTo(new Vector3(1.5f, -2.5f, 3.5f)));
        Assert.That(cloud.Positions[1].X, Is.EqualTo(-9f));
        Assert.That(cloud.Colors[0].X, Is.EqualTo(1f).Within(1e-6f));
        Assert.That(cloud.Colors[0].Y, Is.EqualTo(128f / 255f).Within(1e-6f));
        Assert.That(cloud.Colors[1].Z, Is.EqualTo(1f).Within(1e-6f));
    }

    [Test]
    public void ParseRefusesATruncatedFile()
    {
        // A short read gives a header promising more points than are there. Silently returning
        // the ones that fit would hand the trainer a partial scene that looks fine.
        var bytes = new List<byte>();
        bytes.AddRange(BitConverter.GetBytes(1000));
        bytes.AddRange(new byte[16 * 3]);
        Assert.Throws<ArgumentException>(() => SparsePointCloudInit.Parse(bytes.ToArray()));
    }

    [Test]
    public void ParsesTheRealDrJohnsonCloudIfPresent()
    {
        string path = Path.Combine(TestContext.CurrentContext.TestDirectory,
            "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", "DrJohnson",
            "points3d.bin");
        if (!File.Exists(Path.GetFullPath(path)))
            Assert.Ignore("drjohnson cloud not generated in this checkout");

        var cloud = SparsePointCloudInit.Parse(File.ReadAllBytes(Path.GetFullPath(path)));
        Assert.That(cloud.Count, Is.GreaterThan(10_000), "sparse cloud is implausibly small");

        var spacing = SparsePointCloudInit.LocalSpacing(cloud.Positions);
        Assert.That(spacing.All(s => s > 0 && float.IsFinite(s)), Is.True,
            "every splat needs a finite positive scale");

        // Sanity on the order of magnitude: a room is metres across, so splats should be
        // centimetres, not millimetres or metres. A number far outside this means the cloud is
        // in different units than the cameras and nothing would line up.
        var sorted = spacing.Order().ToArray();
        float median = sorted[sorted.Length / 2];
        Assert.That(median, Is.InRange(0.001f, 1.0f),
            $"median splat scale {median} is not room-sized");
        TestContext.Out.WriteLine(
            $"{cloud.Count:N0} points, median scale {median:F4}, " +
            $"p10 {sorted[sorted.Length / 10]:F4}, p90 {sorted[sorted.Length * 9 / 10]:F4}");
    }
}

/// <summary>
/// The grid is only fast if its cell size reflects where the points actually are. An SfM cloud
/// has far outliers, and sizing the grid from the bounding box made drjohnson take 349 seconds.
/// </summary>
[TestFixture]
public class SparsePointCloudOutlierTests
{
    [Test]
    public void OutliersDoNotDegradeTheGridToAllPairs()
    {
        // 20,000 points in a room-sized box, plus a few triangulated out into the sky - the
        // exact shape that broke it.
        var rng = new Random(42);
        var pts = new List<System.Numerics.Vector3>();
        for (int i = 0; i < 20_000; i++)
            pts.Add(new System.Numerics.Vector3(
                (float)rng.NextDouble() * 12f,
                (float)rng.NextDouble() * 4f,
                (float)rng.NextDouble() * 12f));
        for (int i = 0; i < 12; i++)
            pts.Add(new System.Numerics.Vector3(
                (float)(rng.NextDouble() - 0.5) * 8000f,
                (float)(rng.NextDouble() - 0.5) * 8000f,
                (float)(rng.NextDouble() - 0.5) * 8000f));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var spacing = SparsePointCloudInit.LocalSpacing(pts.ToArray());
        sw.Stop();

        TestContext.Out.WriteLine($"{pts.Count:N0} points with outliers in {sw.ElapsedMilliseconds} ms");
        Assert.That(spacing.All(v => v > 0 && float.IsFinite(v)), Is.True);

        // Generous, because a CI machine is not this one. The bug made this case quadratic:
        // 20k points all-pairs is 400M distance tests and took tens of seconds.
        Assert.That(sw.ElapsedMilliseconds, Is.LessThan(5_000),
            "grid degenerated to all-pairs - outliers are setting the cell size again");
    }

    [Test]
    public void OutlierCloudStillMatchesBruteForce()
    {
        // Speed is worthless if the answer changed. Smaller so brute force is affordable.
        var rng = new Random(9);
        var pts = new List<System.Numerics.Vector3>();
        for (int i = 0; i < 600; i++)
            pts.Add(new System.Numerics.Vector3(
                (float)rng.NextDouble() * 12f, (float)rng.NextDouble() * 4f,
                (float)rng.NextDouble() * 12f));
        pts.Add(new System.Numerics.Vector3(5000, -3000, 900));
        pts.Add(new System.Numerics.Vector3(-4000, 7000, -200));
        var arr = pts.ToArray();

        var fast = SparsePointCloudInit.LocalSpacing(arr);
        for (int i = 0; i < arr.Length; i++)
        {
            var d = new List<float>();
            for (int j = 0; j < arr.Length; j++)
                if (j != i) d.Add(System.Numerics.Vector3.DistanceSquared(arr[i], arr[j]));
            d.Sort();
            float sum = d[0] + d[1] + d[2];
            float slow = MathF.Sqrt(MathF.Max(sum / 3f, SparsePointCloudInit.MinSpacingSq));
            Assert.That(fast[i], Is.EqualTo(slow).Within(1e-3f), $"point {i} at {arr[i]}");
        }
    }
}
