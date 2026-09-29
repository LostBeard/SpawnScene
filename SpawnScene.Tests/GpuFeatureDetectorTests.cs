using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Equivalence gate: <see cref="GpuFeatureDetector"/> (FAST, NMS, BRIEF on the device) must return EXACTLY what
/// <see cref="FeatureDetector.Detect"/> returns - same features, same order, same scores, same descriptors - on the
/// ILGPU CPU accelerator, where kernel float math is .NET's own. Written 2026-09-28 when the detector moved onto the
/// GPU so project photos never have to come back to the .NET heap.
/// </summary>
public class GpuFeatureDetectorTests
{
    /// <summary>Deterministic photo-like frame: overlapping shapes of varied brightness, gradients, noise.</summary>
    static byte[] Frame(int w, int h, int seed, int shapes, int noise)
    {
        var rng = new Random(seed);
        var g = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                g[y * w + x] = (byte)(60 + (x * 80 / w) + (y * 40 / h));
        for (int s = 0; s < shapes; s++)
        {
            int cx = rng.Next(w), cy = rng.Next(h), rx = rng.Next(4, 60), ry = rng.Next(4, 60);
            byte v = (byte)rng.Next(256);
            bool rect = rng.Next(2) == 0;
            for (int y = Math.Max(0, cy - ry); y < Math.Min(h, cy + ry); y++)
                for (int x = Math.Max(0, cx - rx); x < Math.Min(w, cx + rx); x++)
                {
                    float dx = (x - cx) / (float)rx, dy = (y - cy) / (float)ry;
                    if (rect || dx * dx + dy * dy <= 1f) g[y * w + x] = v;
                }
        }
        for (int i = 0; i < g.Length; i++)
            g[i] = (byte)Math.Clamp(g[i] + rng.Next(-noise, noise + 1), 0, 255);
        return g;
    }

    static async Task AssertSameAsync(byte[] gray, int w, int h)
    {
        var expected = new FeatureDetector().Detect(gray, w, h);

        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        var grayInts = new int[gray.Length];
        for (int i = 0; i < gray.Length; i++) grayInts[i] = gray[i];
        using var grayBuf = accel.Allocate1D(grayInts);
        var actual = await new GpuFeatureDetector().DetectAsync(accel, grayBuf.View, w, h);

        Assert.That(actual.Count, Is.EqualTo(expected.Count), "feature count");
        for (int f = 0; f < expected.Count; f++)
        {
            var e = expected[f]; var a = actual[f];
            Assert.That((a.X, a.Y, a.Score), Is.EqualTo((e.X, e.Y, e.Score)), $"feature {f} position/score");
            Assert.That(a.Descriptor, Is.EqualTo(e.Descriptor), $"feature {f} ({e.X},{e.Y}) descriptor");
        }
        TestContext.Out.WriteLine($"{w}x{h}: {expected.Count} features identical");
    }

    [Test]
    public async Task GpuDetector_MatchesCpuDetector_ManyFeatures()
    {
        // Enough texture that far more than 2,000 cells hold a corner: exercises the top-N sort and its ties.
        await AssertSameAsync(Frame(1024, 768, 7, 900, 18), 1024, 768);
    }

    [Test]
    public async Task GpuDetector_MatchesCpuDetector_FewFeatures()
    {
        await AssertSameAsync(Frame(320, 240, 11, 12, 2), 320, 240);
    }

    [Test]
    public async Task GpuDetector_Portrait_NonMultipleOf8()
    {
        // 8x8 cells that run off the edge, and a portrait frame (phone photos).
        await AssertSameAsync(Frame(203, 317, 3, 80, 10), 203, 317);
    }
}
