using System.Numerics;
using ILGPU;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// <see cref="GpuBundleAdjuster"/> against the same gate as the managed solver (BundleAdjusterTruckScaleTests): a
/// Truck-shaped problem seeded with DAv3-grade poses must refine to SfM grade (camera RMS &lt; 0.5% of spread, &lt; 1 px),
/// on the ILGPU CPU accelerator. Written 2026-09-28 when BA moved onto the GPU: the managed solver took 1,358.8 s of
/// TruckFull in the browser (Mono interpreter). The kernels are float, the managed solver double, so the two agree on
/// the answer's QUALITY, not bit for bit; the managed solution is printed next to it for comparison.
/// </summary>
public class GpuBundleAdjusterTests
{
    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    /// <summary>
    /// The ILGPU CPU accelerator with 2 parallel groups of at most 16 threads. The default device spreads the grid over
    /// every core: MEASURED 2026-09-28, the per-camera workgroup kernels held 78% of 12 logical cores for minutes (one
    /// run burned 11,317 CPU-seconds) - too heavy for a unit test on a machine someone is using.
    /// </summary>
    static Context PoliteCpuContext() => Context.Create(b => b.CPU(new ILGPU.Runtime.CPU.CPUDevice(4, 4, 2)));

    static async Task<(GpuBundleAdjuster ba, BundleAdjuster.Result result)> SolveLikeProductionAsync(
        ILGPU.Runtime.Accelerator accel, List<CameraParams> init, List<Vector3> points, List<BundleAdjuster.Observation> obs)
    {
        var ba = new GpuBundleAdjuster(accel, init, points, obs, fixedCamera: 0, sharedFocal: true);
        var result = await ba.SolveAsync(new BundleAdjuster.Options
        {
            MaxIterations = 150,
            Rounds = 3,
            RoundLog = (round, iters, rms, kept) =>
                TestContext.Out.WriteLine($"  round {round}: {iters} iterations, RMS {rms:F2} px, {kept} obs kept"),
        });
        return (ba, result);
    }

    static async Task RefinesToSfmGradeAsync(
        (List<CameraParams> truth, List<CameraParams> init, List<Vector3> points, List<BundleAdjuster.Observation> obs)? problem,
        string name, bool compareManaged)
    {
        if (problem == null) Assert.Ignore($"{name} dataset not generated in this checkout");
        var (truth, init, points, obs) = problem.Value;

        using var context = PoliteCpuContext();
        using var accel = context.CreateCPUAccelerator(0);
        var (ba, result) = await SolveLikeProductionAsync(accel, init, points, obs);
        using var baOwner = ba;
        var refined = init.Select(Copy).ToList();
        for (int c = 0; c < refined.Count; c++) ba.WriteCamera(c, refined[c]);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            refined.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var acc, out _, out _), Is.True);
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            init.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var acc0, out _, out _), Is.True);

        TestContext.Out.WriteLine(
            $"{name}-scale GPU BA: {truth.Count} cams, {points.Count} points, {result.Observations} obs ({result.ObservationsKept} kept), " +
            $"{result.Iterations} iters, RMS {result.InitialRmsPixels:F1} -> {result.FinalRmsPixels:F3} px, {result.Seconds:F1}s");
        TestContext.Out.WriteLine($"  timing: {ba.TimingSummary()}");
        TestContext.Out.WriteLine(
            $"  pose vs truth: {acc0.PositionRms / acc0.Spread:P2} -> {acc.PositionRms / acc.Spread:P3} of spread, " +
            $"fwd {acc0.MedianForwardDeg:F2} -> {acc.MedianForwardDeg:F3} deg, focal {ba.SharedFocal:F2}");

        if (compareManaged)
        {
            var (mba, mres) = BundleAdjusterTruckScaleTests.SolveLikeProduction(init, points, obs);
            var managed = init.Select(Copy).ToList();
            for (int c = 0; c < managed.Count; c++) mba.WriteCamera(c, managed[c]);
            WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
                managed.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var accM, out _, out _);
            double diff = Math.Sqrt(Enumerable.Range(0, managed.Count)
                .Average(c => (double)Vector3.DistanceSquared(managed[c].Position, refined[c].Position)));
            TestContext.Out.WriteLine(
                $"  managed: {mres.Iterations} iters, RMS {mres.FinalRmsPixels:F3} px, {mres.ObservationsKept} kept, {mres.Seconds:F1}s, " +
                $"{accM.PositionRms / accM.Spread:P3} of spread, focal {mba.SharedFocal:F2}; GPU vs managed camera RMS " +
                $"{diff / acc.Spread:P3} of spread");
        }

        Assert.That(acc0.PositionRms / acc0.Spread, Is.GreaterThan(0.02f), "the start must be DAv3-grade to mean anything");
        Assert.That(acc.PositionRms / acc.Spread, Is.LessThan(0.005f), "GPU BA must reach SfM grade (< 0.5% of spread)");
        Assert.That(result.FinalRmsPixels, Is.LessThan(1.0));
    }

    /// <summary>
    /// Equivalence gate: a small Truck problem, GPU solver vs the managed one on the SAME input with the SAME options. In
    /// double they are the same algorithm, so the cameras must agree far below the SfM grade - MEASURED on the full
    /// Truck-scale problem: 0.001% of spread. Kept small (1,000 tracks, 30 LM iterations, 2 rounds, 16 CG iterations) because the ILGPU CPU
    /// accelerator runs the workgroup kernels' barriers with real threads: 3,000 tracks at production settings ran past
    /// 30 minutes (2026-09-28).
    /// </summary>
    [Test]
    public async Task SmallTruck_GpuSolver_MatchesManagedSolver()
    {
        var problem = BundleAdjusterTruckScaleTests.BuildTruckProblem(seed: 7, tracksWanted: 1_000);
        if (problem == null) Assert.Ignore("Truck dataset not generated in this checkout");
        var (truth, init, points, obs) = problem.Value;
        // MaxCgIterations 16: MEASURED 2026-09-28 at the default 200 this took 18 m 19 s on the capped CPU device (11,405 CG
        // iterations at ~94 ms) and PASSED - 60 iterations each, RMS 0.6625 px, 2,390/2,486 kept, cameras 0.0043% of spread.
        // Both solvers get the same cap, so the comparison is unchanged.
        BundleAdjuster.Options Options() => new() { MaxIterations = 30, Rounds = 2, MaxCgIterations = 16 };

        using var context = PoliteCpuContext();
        using var accel = context.CreateCPUAccelerator(0);
        using var ba = new GpuBundleAdjuster(accel, init, points, obs, fixedCamera: 0, sharedFocal: true);
        var result = await ba.SolveAsync(Options());
        var mba = new BundleAdjuster(init, points, obs, fixedCamera: 0, sharedFocal: true);
        var mres = mba.Solve(Options());
        var gpu = init.Select(Copy).ToList();
        var managed = init.Select(Copy).ToList();
        for (int c = 0; c < gpu.Count; c++) { ba.WriteCamera(c, gpu[c]); mba.WriteCamera(c, managed[c]); }
        Assert.That(WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
            managed.Cast<CameraParams?>().ToList(), truth.Cast<CameraParams?>().ToList(), out var accM, out _, out _), Is.True);
        double diff = Math.Sqrt(Enumerable.Range(0, gpu.Count)
            .Average(c => (double)Vector3.DistanceSquared(gpu[c].Position, managed[c].Position)));
        TestContext.Out.WriteLine(
            $"GPU: {result.Iterations} iters, RMS {result.FinalRmsPixels:F4} px, {result.ObservationsKept}/{result.Observations} kept, " +
            $"{result.Seconds:F1}s ({ba.TimingSummary()}); managed: {mres.Iterations} iters, RMS {mres.FinalRmsPixels:F4} px, " +
            $"{mres.ObservationsKept} kept, {mres.Seconds:F1}s; GPU vs managed cameras {diff / accM.Spread:P4} of spread");

        Assert.That(diff / accM.Spread, Is.LessThan(1e-4), "GPU and managed solutions must agree (same algorithm, both double)");
        Assert.That(Math.Abs(result.FinalRmsPixels - mres.FinalRmsPixels), Is.LessThan(1e-3));
        Assert.That(result.ObservationsKept, Is.EqualTo(mres.ObservationsKept));
    }

    [Test, Explicit("~30+ minutes on the ILGPU CPU accelerator")]
    public Task TruckScale_GpuSolver_RefinesToSfmGrade() =>
        RefinesToSfmGradeAsync(BundleAdjusterTruckScaleTests.BuildTruckProblem(seed: 5), "Truck", compareManaged: true);

    [Test, Explicit("benchmark: TruckFull scale")]
    public Task TruckFullScale_GpuSolver_RefinesToSfmGrade() =>
        RefinesToSfmGradeAsync(BundleAdjusterTruckScaleTests.BuildTruckProblem(seed: 5, tracksWanted: 44_000, dataset: "TruckFull",
            trackMix: new[] { 0.562, 0.751, 0.844, 0.897, 1.0 }), "TruckFull", compareManaged: false);
}
