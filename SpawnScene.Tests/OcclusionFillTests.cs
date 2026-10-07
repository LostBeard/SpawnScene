using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The hidden layers of a single-photo scene (OcclusionFill) on the ILGPU CPU accelerator, on a photo whose answer is
/// known: a red square at depth 1 in front of a blue wall at depth 2.
/// </summary>
public class OcclusionFillTests
{
    const int F = SplatFormat.Floats;

    [Test]
    public void Fill_PutsTheWallBehindTheSquaresEdges_AndContinuesPastTheFrame()
    {
        const int W = 64, H = 64, sub = 1, sq0 = 24, sq1 = 40, radius = 6, margin = 8;
        var depth = new float[W * H];
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                bool inSquare = x >= sq0 && x < sq1 && y >= sq0 && y < sq1;
                depth[y * W + x] = inSquare ? 1f : 2f;
                rgba[y * W + x] = inSquare ? 0x000000FF : 0x00FF0000;   // r in the low byte: red square, blue wall
            }
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var d = accel.Allocate1D(depth);
        using var c = accel.Allocate1D(rgba);
        int capacity = OcclusionFill.ExtraCapacity(W, H, margin, 1);
        using var outPacked = accel.Allocate1D<float>((long)capacity * F);
        using var counter = accel.Allocate1D<int>(1);
        counter.MemSetToZero();
        var scratch = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        OcclusionFill.Append(accel, d.View, c.View, outPacked.View, counter.View, new OcclusionFill.Params
        {
            Width = W, Height = H, Subsample = sub, GridW = W, GridH = H, Radius = radius, Margin = margin, Capacity = capacity,
            FocalX = 64, FocalY = 64, CenterX = 32, CenterY = 32, DepthScale = 1f, Tau = 0.08f, BackgroundBand = 0.92f,
            SizeFactor = 1.5f, Opacity = 0.95f, BorderStride = 1,
        }, scratch);
        accel.Synchronize();
        foreach (var b in scratch) b.Dispose();
        int n = counter.GetAsArray1D()[0];
        var rows = outPacked.GetAsArray1D();

        int behind = 0, border = 0, wrongBehind = 0, deepInside = 0;
        for (int i = 0; i < n; i++)
        {
            int o = i * F;
            float z = rows[o + 2];
            // Back to pixel coordinates: x = -(u - cx) z / f.
            float u = 32 - rows[o] * 64 / z, v = 32 - rows[o + 1] * 64 / z;
            bool inFrame = u > -0.5f && u < W - 0.5f && v > -0.5f && v < H - 0.5f;
            if (!inFrame) { border++; continue; }
            behind++;
            bool insideSquare = u >= sq0 - 0.5f && u < sq1 - 0.5f && v >= sq0 - 0.5f && v < sq1 - 0.5f;
            bool blueWall = MathF.Abs(z - 2f) < 0.05f && rows[o + 5] > 0.9f && rows[o + 3] < 0.1f;
            if (!insideSquare || !blueWall) wrongBehind++;
            // Beyond the max filter's reach of the wall, the square's centre has nothing behind it.
            if (u >= sq0 + radius + 1 && u < sq1 - radius - 1 && v >= sq0 + radius + 1 && v < sq1 - radius - 1) deepInside++;
        }
        Assert.That(behind, Is.GreaterThan(0), "the band behind the square's edges was filled");
        Assert.That(wrongBehind, Is.EqualTo(0), "every in-frame fill splat is blue wall at depth 2, behind the square");
        Assert.That(deepInside, Is.EqualTo(0), "no fill where no wall is within reach");
        int padded = (W + 2 * margin) * (H + 2 * margin) - W * H;
        Assert.That(border, Is.EqualTo(padded), "one splat per margin cell (the wall continues; no second layer behind it)");
    }
}
