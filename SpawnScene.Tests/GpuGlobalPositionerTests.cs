using System.Numerics;
using ILGPU;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// <see cref="GpuGlobalPositioner"/> against the managed <see cref="GlobalSfmInit.GlobalPositioningRobust"/> on the ILGPU
/// CPU accelerator: the same inputs (Truck's cameras, real track lengths, mismatched observations, true or estimated
/// rotations) must give the same centres - both solvers share the start, the angle measure and the histogram median, so
/// they differ only in summation order and the device's PCG against the host's Cholesky - and the same accuracy.
/// </summary>
public class GpuGlobalPositionerTests
{
    static Context PoliteCpuContext() => Context.Create(b => b.CPU(new CPUDevice(4, 4, 2)));

    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    // The data must pin the solution down, or two correct solvers drift apart along a flat valley and the comparison means
    // nothing: MEASURED 2026-09-29, 4,000 real-length points (~80 observations per camera) left the MANAGED solver 101% and
    // 72% of spread off the truth, and the two solvers 3.6% apart. Each case asserts the managed solver's accuracy first
    // (a well-posedness bar, not an accuracy target: with ~1.1 deg estimated rotations the managed result is 3.04%, bounded
    // by the rotations - GlobalSfmInitTests.GlobalPositioningRobust_EstimatedRotations).
    [TestCase(0.0, 0.0, 3000, false, 0.005, TestName = "GpuGlobalPositioner_EqualsManaged_LongTracksClean")]
    [TestCase(0.10, 1.5, 3000, false, 0.05, TestName = "GpuGlobalPositioner_EqualsManaged_LongTracksOutliers10EstimatedRotations")]
    [TestCase(0.10, 0.0, 12000, true, 0.01, TestName = "GpuGlobalPositioner_EqualsManaged_RealLengthsOutliers10")]
    public async Task EqualsManaged(double outliers, double pairNoiseDeg, int pointsWanted, bool realLengths, double managedBar)
    {
        GpuGlobalPositioner.Trace = m => TestContext.Progress.WriteLine("  " + m);
        var cams = GlobalSfmInitTests.TruckCameras();
        if (cams == null) Assert.Ignore("Truck dataset not in this checkout");
        var rng = new Random(21);
        int n = cams.Count;
        var (obs, points, focal) = GlobalSfmInitTests.SyntheticTracks(cams, rng, pointsWanted, 0.5, realLengths, outliers);
        var truthR = cams.Select(GlobalSfmInit.RotationOf).ToArray();
        var rot = truthR;
        var connected = Enumerable.Repeat(true, n).ToArray();
        if (pairNoiseDeg > 0)
        {
            var edges = new List<GlobalSfmInit.RelativePose>();
            foreach (var (a, b) in GlobalSfmInitTests.NeighbourPairs(n))
            {
                var (r, t) = GlobalSfmInitTests.Relative(cams[a], cams[b]);
                r = rng.NextDouble() < 0.10 ? GlobalSfmInitTests.SmallRotation(rng, 60) : GlobalSfmInit.Mul(GlobalSfmInitTests.SmallRotation(rng, pairNoiseDeg), r);
                edges.Add(new GlobalSfmInit.RelativePose(a, b, r, t, 100 + rng.Next(200)));
            }
            rot = GlobalSfmInit.AverageRotations(n, edges, truthR, out connected);
            var q = GlobalSfmInit.AlignFrame(rot, truthR, connected);
            for (int i = 0; i < n; i++) rot[i] = GlobalSfmInit.Mul(rot[i], q);
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var managed = GlobalSfmInit.GlobalPositioningRobust(cams, rot, connected, obs, points, focal);
        double tManaged = sw.Elapsed.TotalSeconds;

        using var context = PoliteCpuContext();
        using var accel = context.CreateCPUAccelerator(0);
        sw.Restart();
        using var gp = new GpuGlobalPositioner(accel, cams, rot, connected, obs, points, focal);
        var gpu = await gp.SolveAsync();
        double tGpu = sw.Elapsed.TotalSeconds;

        var truthN = cams.Cast<CameraParams?>().ToList();
        double Median(GlobalSfmInit.PositioningResult r)
        {
            var est = cams.Select((c, i) => { var k = Copy(c); k.Position = r.Centres[i]; return (CameraParams?)k; }).ToList();
            Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, truthN, out var acc, out _, out _), Is.True);
            return acc.MedianPosFrac;
        }
        double mMed = Median(managed), gMed = Median(gpu);
        var centre = cams.Aggregate(Vector3.Zero, (s, c) => s + c.Position) / n;
        double spread = Math.Sqrt(cams.Average(c => (double)(c.Position - centre).LengthSquared()));
        double diff = Math.Sqrt(Enumerable.Range(0, n).Average(i => (double)Vector3.DistanceSquared(managed.Centres[i], gpu.Centres[i]))) / spread;
        TestContext.Out.WriteLine($"managed {tManaged:F1}s: {managed.Summary}: median {mMed:P3}");
        TestContext.Out.WriteLine($"GPU (CPU accelerator) {tGpu:F1}s: {gpu.Summary}: median {gMed:P3}");
        TestContext.Out.WriteLine($"GPU vs managed centres: RMS {diff:P4} of spread; outliers {gpu.Outliers} vs {managed.Outliers}");
        Assert.That(mMed, Is.LessThan(managedBar), "the data must pin the solution down, or no comparison means anything");
        Assert.That(diff, Is.LessThan(1e-3), "the GPU solver must compute the managed solver's solution");
        Assert.That(gMed, Is.LessThan(mMed * 1.1 + 1e-4));
    }
}
