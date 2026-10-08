using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// DepthFusionInit on a scene with a known answer: two cameras 0.5 apart looking at the plane z = 5, each depth map in its
/// own raw unit (view 0 stores depth / 2.5, view 1 depth / 5 - the per-chunk scale the pose pass leaves), SfM points on
/// the plane. Every seed must land on the plane; a seed exists only where both views see the plane (one agreeing view);
/// each overlap sample is emitted once, by view 0. Then view 1's right half is corrupted (depth 30% off): view 0's samples
/// that land there lose their only agreement and must disappear - the private-shell rejection.
/// </summary>
public class DepthFusionInitTests
{
    const int W = 64, H = 48;
    const float PlaneZ = 5f;

    static CameraParams Cam(float x) => new()
    {
        Width = W, Height = H, FocalX = 60, FocalY = 60, CenterX = 32, CenterY = 24,
        Position = new Vector3(x, 0, 0), Forward = Vector3.UnitZ, Up = Vector3.UnitY, Near = 0.1f, Far = 100,
    };

    static MemoryBuffer1D<float, Stride1D.Dense> Depth(Accelerator a, float rawUnit, Func<int, int, bool>? corrupt = null)
    {
        var d = new float[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                d[y * W + x] = PlaneZ / rawUnit * (corrupt != null && corrupt(x, y) ? 1.3f : 1f);
        return a.Allocate1D(d);
    }

    static List<Vector3> PlanePoints()
    {
        var pts = new List<Vector3>();
        for (int i = -10; i <= 10; i++) for (int j = -8; j <= 8; j++) pts.Add(new Vector3(i * 0.25f, j * 0.25f, PlaneZ));
        return pts;
    }

    static async Task<(float[] packed, int count)> Fuse(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> d0,
        MemoryBuffer1D<float, Stride1D.Dense> d1, int stride)
    {
        var cams = new CameraParams?[] { Cam(0f), Cam(0.5f) };
        var depths = new DepthFusionInit.DepthMap?[] { new(d0, W, H), new(d1, W, H) };
        var r = await DepthFusionInit.FuseAsync(a, cams, depths, new[] { 0, 1 }, PlanePoints(), null, stride, 0.03f, 1f);
        Assert.That(r, Is.Not.Null, "both views should scale to the SfM points");
        var (buf, n, report) = r!.Value;
        TestContext.WriteLine($"scales {report.Scales}; {n} seeds from {report.Candidates} samples");
        var host = n > 0 ? buf.GetAsArray1D() : Array.Empty<float>();
        buf.Dispose();
        return (host, n);
    }

    /// <summary>Samples of view 0 (stride grid, as the kernel takes them) whose world point is inside view 1's frame.</summary>
    static int ExpectedOverlap(int stride, Func<int, int, bool>? corrupt1 = null)
    {
        int count = 0;
        var c0 = Cam(0f); var c1 = Cam(0.5f);
        for (int sy = 0; sy < (H + stride - 1) / stride; sy++)
            for (int sx = 0; sx < (W + stride - 1) / stride; sx++)
            {
                int px = Math.Min(sx * stride + stride / 2, W - 1), py = Math.Min(sy * stride + stride / 2, H - 1);
                var world = WorldSpaceGeometry.UnprojectPixel(c0, px + 0.5f, py + 0.5f, PlaneZ);
                if (!WorldSpaceGeometry.Project(c1, world, out float u, out float v, out _)) continue;
                if (u < 0 || v < 0 || u >= W || v >= H) continue;
                if (corrupt1 != null && corrupt1((int)u, (int)v)) continue;
                count++;
            }
        return count;
    }

    [Test]
    public async Task SeedsLandOnTheSurfaceOncePerAgreedSample()
    {
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        using var d0 = Depth(a, 2.5f);
        using var d1 = Depth(a, 5f);
        const int stride = 4;
        var (p, n) = await Fuse(a, d0, d1, stride);

        int expected = ExpectedOverlap(stride);
        Assert.That(expected, Is.GreaterThan(50).And.LessThan(W * H / (stride * stride)), "the views overlap in part");
        Assert.That(n, Is.EqualTo(expected), "one seed per view-0 sample view 1 also sees; none from view 1 (view 0 is first)");
        for (int i = 0; i < n; i++)
        {
            Assert.That(p[i * SplatFormat.Floats + 2], Is.EqualTo(PlaneZ).Within(1e-3f), $"seed {i} z");
            Assert.That(p[i * SplatFormat.Floats + SplatFormat.OffOpacity], Is.EqualTo(SparsePointCloudInit.InitialOpacity));
            // Spacing at depth 5, stride 4, focal 60.
            Assert.That(p[i * SplatFormat.Floats + SplatFormat.OffScale], Is.EqualTo(PlaneZ * stride / 60f).Within(1e-4f));
        }
    }

    [Test]
    public async Task ASampleNoOtherViewAgreesWithIsDropped()
    {
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        // A minority of view 1 (its right third): its scale still fits the SfM points (the median follows the majority).
        Func<int, int, bool> rightHalf = (x, y) => x >= 2 * W / 3;
        using var d0 = Depth(a, 2.5f);
        using var d1 = Depth(a, 5f, rightHalf);
        const int stride = 4;
        var (_, n) = await Fuse(a, d0, d1, stride);
        int expected = ExpectedOverlap(stride, rightHalf);
        Assert.That(expected, Is.LessThan(ExpectedOverlap(stride)), "the corruption must cover part of the overlap");
        // View 1's own scale still fits (most of it is right), and its corrupted samples agree with nobody; view 0's
        // samples landing on the corrupted third lose their only agreement.
        Assert.That(n, Is.EqualTo(expected));
    }
}
