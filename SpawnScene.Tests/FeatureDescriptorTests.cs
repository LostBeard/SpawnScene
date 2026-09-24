using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The feature descriptor must MATCH the same point across two views. On DrJohnson the import's pair
/// match counts carried no overlap signal at all (adjacent frames median 19, frames >=10 apart median 19),
/// so "anchors chosen by overlap" picked from noise and chose the one view DAv3 misplaces by 200% of the
/// camera spread. These tests measure matching directly on a synthetic view pair with a KNOWN warp.
/// </summary>
public class FeatureDescriptorTests
{
    const int W = 640, H = 480;

    /// <summary>Deterministic textured scene: multi-octave value noise plus hard-edged rectangles (corners).</summary>
    static float[] Scene(int seed, int w, int h)
    {
        var rng = new Random(seed);
        var img = new float[w * h];
        foreach (var (cell, amp) in new[] { (64, 60f), (16, 35f), (4, 18f) })
        {
            int gw = w / cell + 2, gh = h / cell + 2;
            var g = new float[gw * gh];
            for (int i = 0; i < g.Length; i++) g[i] = (float)rng.NextDouble() * amp;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float fx = (float)x / cell, fy = (float)y / cell;
                    int x0 = (int)fx, y0 = (int)fy;
                    float tx = fx - x0, ty = fy - y0;
                    float a = g[y0 * gw + x0], b = g[y0 * gw + x0 + 1];
                    float c = g[(y0 + 1) * gw + x0], d = g[(y0 + 1) * gw + x0 + 1];
                    img[y * w + x] += (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
                }
        }
        for (int r = 0; r < 120; r++)
        {
            int rx = rng.Next(w), ry = rng.Next(h), rw = 6 + rng.Next(40), rh = 6 + rng.Next(40);
            float v = (float)rng.NextDouble() * 120f - 60f;
            for (int y = ry; y < Math.Min(h, ry + rh); y++)
                for (int x = rx; x < Math.Min(w, rx + rw); x++)
                    img[y * w + x] += v;
        }
        return img;
    }

    /// <summary>Sample <paramref name="src"/> through p_src = (p_dst - t) / s (bilinear), add gain, bias and noise.</summary>
    static byte[] Render(float[] src, float scale, float tx, float ty, float gain, float bias, float noise, int seed)
    {
        var rng = new Random(seed);
        var dst = new byte[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                float sx = (x - tx) / scale, sy = (y - ty) / scale;
                float v = 0;
                if (sx >= 0 && sy >= 0 && sx < W - 1 && sy < H - 1)
                {
                    int x0 = (int)sx, y0 = (int)sy;
                    float fx = sx - x0, fy = sy - y0;
                    v = (src[y0 * W + x0] * (1 - fx) + src[y0 * W + x0 + 1] * fx) * (1 - fy)
                      + (src[(y0 + 1) * W + x0] * (1 - fx) + src[(y0 + 1) * W + x0 + 1] * fx) * fy;
                }
                // Box-Muller sensor noise.
                double u1 = Math.Max(rng.NextDouble(), 1e-12), u2 = rng.NextDouble();
                float n = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)) * noise;
                dst[y * W + x] = (byte)Math.Clamp(MathF.Round(v * gain + bias + 64 + n), 0, 255);
            }
        return dst;
    }

    static (int total, int correct) MatchAgainstWarp(byte[] a, byte[] b, float scale, float tx, float ty)
    {
        var det = new FeatureDetector();
        var fa = det.Detect(a, W, H);
        var fb = det.Detect(b, W, H);
        var matches = new FeatureMatcher().Match(fa, fb);
        // Diagnostic: for features whose true correspondent was also detected (within 2 px), how far apart
        // are their descriptors? This separates "detector does not repeat" from "descriptor does not match".
        int repeat = 0; long dsum = 0;
        foreach (var f in fa)
        {
            float px = f.X * scale + tx, py = f.Y * scale + ty;
            var g = fb.FirstOrDefault(q => (q.X - px) * (q.X - px) + (q.Y - py) * (q.Y - py) <= 4f);
            if (g == null) continue;
            repeat++;
            int d = 0; for (int k = 0; k < 32; k++) d += System.Numerics.BitOperations.PopCount((uint)(f.Descriptor[k] ^ g.Descriptor[k]));
            dsum += d;
        }
        TestContext.Out.WriteLine($"features A={fa.Count} B={fb.Count}, repeated {repeat}, mean true-pair Hamming {(repeat > 0 ? (double)dsum / repeat : double.NaN):F1}/256");
        int correct = 0;
        foreach (var m in matches)
        {
            float px = fa[m.IndexA].X * scale + tx, py = fa[m.IndexA].Y * scale + ty;
            float dx = fb[m.IndexB].X - px, dy = fb[m.IndexB].Y - py;
            if (dx * dx + dy * dy <= 3f * 3f) correct++;
        }
        return (matches.Count, correct);
    }

    /// <summary>
    /// A neighbouring frame: shifted, 3% zoom, exposure change, sensor noise sigma 4. Most matches must
    /// land on the true corresponding point, and there must be many of them.
    /// </summary>
    [Test]
    public void NeighbouringView_MatchesLandOnTheTrueCorrespondence()
    {
        var scene = Scene(7, W, H);
        var a = Render(scene, 1f, 0, 0, 1f, 0, 4f, 1);
        const float s = 1.03f, tx = 17.3f, ty = -9.6f;
        var b = Render(scene, s, tx, ty, 0.9f, 10f, 4f, 2);
        var (total, correct) = MatchAgainstWarp(a, b, s, tx, ty);
        float precision = total == 0 ? 0 : (float)correct / total;
        TestContext.Out.WriteLine($"neighbour: {total} matches, {correct} correct ({precision:P1})");
        // Gate from the measured defects: FAST-12 quick reject + unsmoothed 2.4 px BRIEF gave 1 match / 0 correct;
        // the FAST fix alone 8 / 4 (50%); both fixes 138 / 137 (99.3%).
        Assert.That(correct, Is.GreaterThanOrEqualTo(100), $"only {correct} correct matches of {total}");
        Assert.That(precision, Is.GreaterThanOrEqualTo(0.9f), $"precision {precision:P1}");
    }

    /// <summary>
    /// The overlap signal: a neighbouring view must produce several times more matches than an unrelated
    /// scene. This is what the anchor picker reads.
    /// </summary>
    [Test]
    public void OverlapSignal_NeighbourFarExceedsUnrelated()
    {
        var scene = Scene(7, W, H);
        var other = Scene(99, W, H);
        var a = Render(scene, 1f, 0, 0, 1f, 0, 4f, 1);
        var b = Render(scene, 1.03f, 17.3f, -9.6f, 0.9f, 10f, 4f, 2);
        var c = Render(other, 1f, 0, 0, 1f, 0, 4f, 3);
        var det = new FeatureDetector();
        var fa = det.Detect(a, W, H);
        int near = new FeatureMatcher().Match(fa, det.Detect(b, W, H)).Count;
        int far = new FeatureMatcher().Match(fa, det.Detect(c, W, H)).Count;
        TestContext.Out.WriteLine($"overlap signal: neighbour {near} vs unrelated {far}");
        Assert.That(near, Is.GreaterThanOrEqualTo(5 * Math.Max(far, 1)), $"neighbour {near} vs unrelated {far}");
    }
}
