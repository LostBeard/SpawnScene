using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// DepthEdgeSnap on a known edge: a red foreground (depth 2) left of a blue wall (depth 5), with the depth resized
/// smoothly across 4 pixels = 2 model pixels (the ramp a resize of a soft model edge leaves) while the COLOUR edge is sharp. After the snap no pixel may hold an
/// in-between depth, and each takes the side its colour belongs to. A flat region is left exactly as it was.
/// </summary>
public class DepthEdgeSnapTests
{
    const int W = 1036, H = 40;   // 2x the model's 518: one grid step = 2 px
    const int Edge = 500;

    [Test]
    public void ARampLandsOnTheSideItsColourBelongsTo()
    {
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        var depth = new float[W * H];
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                float t = Math.Clamp((x - (Edge - 2)) / 4f, 0f, 1f);   // ramp over [Edge-2, Edge+2]
                depth[y * W + x] = 2f + 3f * t;
                rgba[y * W + x] = x < Edge ? 0x0000FF : 0xFF0000;     // r in the low byte: red | blue
            }
        using var d = a.Allocate1D(depth);
        using var c = a.Allocate1D(rgba);
        using var o = a.Allocate1D<float>(depth.Length);
        DepthEdgeSnap.Run(a, d.View, c.View, o.View, W, H);
        var outp = o.GetAsArray1D();

        int y0 = H / 2;
        for (int x = Edge - 12; x < Edge + 12; x++)
        {
            float v = outp[y0 * W + x];
            Assert.That(v == 2f || v == 5f, $"x {x}: {v} is an in-between depth");
            Assert.That(v, Is.EqualTo(x < Edge ? 2f : 5f), $"x {x} took the other side");
        }
        // Far from the edge: untouched.
        Assert.That(outp[y0 * W + 100], Is.EqualTo(depth[y0 * W + 100]));
        Assert.That(outp[y0 * W + 900], Is.EqualTo(depth[y0 * W + 900]));
    }

    /// <summary>A floor receding in depth with one colour is a slope, not an edge: no staircase.</summary>
    [Test]
    public void ASlopeOfOneColourIsLeftAlone()
    {
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        var depth = new float[W * H];
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                depth[y * W + x] = 1f + x * 0.01f;            // 1% a pixel: far over the 3% range bar in any window
                rgba[y * W + x] = 0x808080 + (x % 3);         // one colour (a hint of texture)
            }
        using var d = a.Allocate1D(depth);
        using var c = a.Allocate1D(rgba);
        using var o = a.Allocate1D<float>(depth.Length);
        DepthEdgeSnap.Run(a, d.View, c.View, o.View, W, H);
        var outp = o.GetAsArray1D();
        for (int x = 10; x < W - 10; x += 37)
            Assert.That(outp[(H / 2) * W + x], Is.EqualTo(depth[(H / 2) * W + x]), $"x {x} was snapped on a slope");
    }
}
