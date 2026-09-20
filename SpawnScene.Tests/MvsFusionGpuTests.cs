using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Equivalence gate: <see cref="MvsFusionGpu"/> must agree with the <see cref="MvsGeometricFusion"/>
/// CPU oracle on the same synthetic-sphere + TempleRing GT fixture the oracle itself is gated on.
///
/// Runs on the ILGPU CPU accelerator so it needs no GPU. The kernels are backend-agnostic; what this
/// pins is the arithmetic, not the device.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter MvsFusionGpu</c>
/// </summary>
public class MvsFusionGpuTests
{
    const int W = 640, H = 480;

    static string ParPath
    {
        get
        {
            var candidates = new[]
            {
                Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
                    "..", "..", "..", "..", "SpawnScene", "SpawnScene", "wwwroot", "datasets", "TempleRing", "templeR_par.txt")),
                Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
                    "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", "TempleRing", "templeR_par.txt")),
                @"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\TempleRing\templeR_par.txt",
            };
            foreach (var c in candidates)
                if (File.Exists(c)) return c;
            Assert.Fail("templeR_par.txt not found");
            return "";
        }
    }

    static List<CameraParams> LoadFourFarthest()
    {
        var all = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), W, H);
        var cams = all.Select(c => c.camera).ToList();
        var idx = WorldSpaceGeometry.PickFarthestCameras(cams, 4);
        return idx.Select(i => cams[i]).ToList();
    }

    /// <summary>The same sphere the oracle's Hausdorff test uses, so both gates share a fixture.</summary>
    static (List<CameraParams> cams, List<float[]> depths) BuildFixture()
    {
        var cams = LoadFourFarthest();
        var center = new Vector3(0.028f, 0.042f, -0.054f);
        var cloud = MvsGeometricFusion.SampleSphereCloud(center, 0.08f, 2000);
        var depths = cams.Select(c => MvsGeometricFusion.RenderDepthMap(c, cloud, W, H)).ToList();
        return (cams, depths);
    }

    static Context NewContext() => Context.Create(b => b.CPU().EnableAlgorithms());

    /// <summary>Flatten per-view maps the way <see cref="MvsFusionGpu.PackViews"/> does on device.</summary>
    static float[] Flatten(IReadOnlyList<float[]> maps)
    {
        var flat = new float[maps.Count * W * H];
        for (int i = 0; i < maps.Count; i++)
            Array.Copy(maps[i], 0, flat, i * W * H, W * H);
        return flat;
    }

    [Test]
    public async Task ForwardBack_StatsMatchCpuOracle()
    {
        var (cams, depths) = BuildFixture();

        MvsGeometricFusion.FuseDepthMaps(
            cams, depths, W, H, out var cpu,
            subsample: 2,
            maxDepthError: MvsGeometricFusion.DefaultMaxDepthError,
            maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
            minViews: MvsGeometricFusion.DefaultMinViews);

        using var context = NewContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        using var fusion = new MvsFusionGpu(accelerator);

        using var metric = accelerator.Allocate1D(Flatten(depths));
        using var camBuf = fusion.UploadCameras(cams);
        // Separate dummy for the unused conf slot: WebGPU forbids aliasing two storage bindings,
        // so the test must not pass `metric` twice even though the CPU backend tolerates it.
        using var noConf = accelerator.Allocate1D<float>(1);

        var gpu = await fusion.ForwardBackAsync(
            metric.View, noConf.View, camBuf.View,
            W, H, cams.Count,
            subsample: 2,
            maxDepthError: MvsGeometricFusion.DefaultMaxDepthError,
            maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
            minViews: MvsGeometricFusion.DefaultMinViews,
            hasConf: false, confMin: 0f);

        TestContext.Out.WriteLine(
            $"cpu  input={cpu.Input} kept={cpu.Kept} minview_rej={cpu.MinViewReject}");
        TestContext.Out.WriteLine(
            $"gpu  input={gpu.Input} kept={gpu.Kept} minview_rej={gpu.MinViewReject}");

        // Guard the guard: a fixture with nothing kept would pass any comparison.
        Assert.That(cpu.Kept, Is.GreaterThan(200), "fixture kept too little to be a real gate");

        Assert.That(gpu.Input, Is.EqualTo(cpu.Input), "input pixel count diverged");
        Assert.That(gpu.Kept, Is.EqualTo(cpu.Kept), "kept count diverged from the oracle");
        Assert.That(gpu.MinViewReject, Is.EqualTo(cpu.MinViewReject), "minView rejects diverged");
    }

    [Test]
    public async Task CleanedDepth_MatchesCpuOracle()
    {
        var (cams, depths) = BuildFixture();

        var cpuClean = MvsGeometricFusion.CleanDepthMaps(
            cams, depths, W, H, out _,
            maxDepthError: MvsGeometricFusion.DefaultMaxDepthError,
            maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
            minViews: MvsGeometricFusion.DefaultMinViews,
            densifyRadius: 0);

        using var context = NewContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        using var fusion = new MvsFusionGpu(accelerator);

        using var metric = accelerator.Allocate1D(Flatten(depths));
        using var camBuf = fusion.UploadCameras(cams);
        using var clean = accelerator.Allocate1D<float>((long)cams.Count * W * H);
        clean.MemSetToZero();
        using var noConf = accelerator.Allocate1D<float>(1);

        await fusion.ForwardBackAsync(
            metric.View, noConf.View, camBuf.View,
            W, H, cams.Count,
            subsample: 1,
            maxDepthError: MvsGeometricFusion.DefaultMaxDepthError,
            maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
            minViews: MvsGeometricFusion.DefaultMinViews,
            hasConf: false, confMin: 0f,
            outClean: clean.View);

        // Pass 3 of the oracle: cross-view warp fill (nearest wins, both sides).
        int warped = await fusion.WarpFillAsync(
            clean.View, metric.View, camBuf.View, W, H, cams.Count, agreeDepthError: 0.05f);
        TestContext.Out.WriteLine($"gpu warp filled={warped}");

        var gpuClean = clean.GetAsArray1D();

        int cpuFilled = 0, gpuFilled = 0, mismatch = 0;
        float worst = 0f;
        for (int v = 0; v < cams.Count; v++)
        for (int p = 0; p < W * H; p++)
        {
            float c = cpuClean[v][p];
            float g = gpuClean[v * W * H + p];
            if (c > 1e-6f) cpuFilled++;
            if (g > 1e-6f) gpuFilled++;
            if ((c > 1e-6f) != (g > 1e-6f)) { mismatch++; continue; }
            if (c > 1e-6f)
            {
                float d = MathF.Abs(c - g);
                if (d > worst) worst = d;
            }
        }

        TestContext.Out.WriteLine(
            $"cpuFilled={cpuFilled} gpuFilled={gpuFilled} keepMismatch={mismatch} worstDelta={worst:E3}");

        Assert.That(cpuFilled, Is.GreaterThan(1000), "fixture too sparse to gate the clean pass");
        Assert.That(mismatch, Is.EqualTo(0), "GPU and CPU disagree on which pixels survive");
        Assert.That(worst, Is.LessThan(1e-5f), "cleaned depth values diverged from the oracle");
    }

    [Test]
    public async Task ScaleProbe_RecoversAnInjectedScale()
    {
        var (cams, depths) = BuildFixture();

        // Knock view 0 off by a known factor; the probe should search back toward 1/1.06.
        const float injected = 1.06f;
        var skewed = depths.Select(d => (float[])d.Clone()).ToList();
        for (int p = 0; p < skewed[0].Length; p++)
            if (skewed[0][p] > 1e-6f) skewed[0][p] *= injected;

        using var context = NewContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        using var fusion = new MvsFusionGpu(accelerator);

        using var metric = accelerator.Allocate1D(Flatten(skewed));
        using var camBuf = fusion.UploadCameras(cams);

        float f = await fusion.OptimizeScaleFactorAsync(
            metric.View, camBuf.View, view: 0, W, H, cams.Count,
            maxDepthError: 0.05f, maxReprojPx: 2f, probeSubsample: 8);

        float corrected = injected * f;
        TestContext.Out.WriteLine($"injected={injected:F3} probe={f:F3} corrected={corrected:F3}");

        // The grid is 0.85..1.15 in 0.02 steps, so it cannot land exactly on 1/1.06 = 0.943.
        // It must land within one step of it.
        Assert.That(corrected, Is.EqualTo(1f).Within(0.03f),
            $"probe chose x{f:F3}, which leaves the view {corrected:F3} off true scale");
    }

    [Test]
    public void PackedCameras_RoundTripThroughDeviceGeometry()
    {
        // Guard the guard: if PackCameras and WorldSpaceGeometry disagree, every other assertion
        // here is comparing two wrong things.
        var cams = LoadFourFarthest();
        var packed = MvsFusionGpu.PackCameras(cams);

        for (int i = 0; i < cams.Count; i++)
        {
            WorldSpaceGeometry.GetOpenCvAxes(cams[i], out var right, out var down, out var fwd);
            int o = i * MvsFusionGpu.CamStride;
            Assert.That(packed[o + 3], Is.EqualTo(right.X).Within(1e-6f), $"cam{i} right.X");
            Assert.That(packed[o + 6], Is.EqualTo(down.X).Within(1e-6f), $"cam{i} down.X");
            Assert.That(packed[o + 9], Is.EqualTo(fwd.X).Within(1e-6f), $"cam{i} fwd.X");
            Assert.That(packed[o + 12], Is.EqualTo(cams[i].FocalX).Within(1e-6f), $"cam{i} fx");
        }
    }
}
