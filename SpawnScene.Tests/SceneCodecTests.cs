using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The .spawnscene v2 quantization (SceneCodec), on the same scalar functions its GPU kernels run: every field must come
/// back within its quantization step, so a saved scene renders as it did.
/// </summary>
public class SceneCodecTests
{
    [Test]
    public void Half_RoundTrips_WithinATenthOfAPercent()
    {
        foreach (float v in new[] { 1e-4f, 3.3e-3f, 0.0123f, 0.5f, 1f, -1f, 2.71828f, -17.25f, 123.4f, 6000f, -0.3f })
        {
            float back = SceneCodec.HalfToFloat(SceneCodec.FloatToHalf(v));
            Assert.That(MathF.Abs(back - v), Is.LessThanOrEqualTo(MathF.Abs(v) * 1e-3f), $"{v} -> {back}");
        }
        Assert.That(SceneCodec.HalfToFloat(SceneCodec.FloatToHalf(0f)), Is.EqualTo(0f));
    }

    [Test]
    public void Position_RoundTrips_WithinOneStepOf24Bits()
    {
        float min = -137.5f, size = 401.25f, step = size / ((1 << 24) - 1);
        var rng = new Random(7);
        for (int k = 0; k < 1000; k++)
        {
            float v = min + (float)rng.NextDouble() * size;
            float back = SceneCodec.DequantPos(SceneCodec.QuantPos(v, min, size), min, size);
            Assert.That(MathF.Abs(back - v), Is.LessThanOrEqualTo(step + MathF.Abs(v) * 1e-6f));
        }
        // Outside the frame clamps to its edges rather than wrapping.
        Assert.That(SceneCodec.DequantPos(SceneCodec.QuantPos(min - 50f, min, size), min, size), Is.EqualTo(min).Within(1e-3f));
        Assert.That(SceneCodec.DequantPos(SceneCodec.QuantPos(min + size + 50f, min, size), min, size), Is.EqualTo(min + size).Within(1e-3f));
    }

    /// <summary>
    /// The Truck's case: 98% of it within 25 units, floaters out to 13,737. Inside the inner box the step must be set by
    /// the box, not the floaters; past it, relative precision; and codes must stay in order across the seams.
    /// </summary>
    [Test]
    public void PiecewisePosition_FloatersDoNotCoarsenTheInnerBox()
    {
        var outer = new SplatBounds.Aabb(-416.9f, -812.2f, -6810.5f, 3365f, 1174.5f, 13736.8f);
        var inner = new SplatBounds.Aabb(-21f, -2f, -17f, 14f, 6.5f, 10f);
        var f = SceneCodec.Frame.From(outer, inner, 1);
        float min = f.MinZ, size = f.SizeZ, lo = f.TailLoZ, hi = f.TailHiZ;
        Assert.That(size, Is.EqualTo(27f).Within(1e-4f));

        float innerStep = size / ((1 << 24) - 1 - 2 * (1 << 20));
        var rng = new Random(5);
        for (int k = 0; k < 2000; k++)
        {
            float v = min + (float)rng.NextDouble() * size;
            float back = SceneCodec.DequantPosP(SceneCodec.QuantPosP(v, min, size, lo, hi), min, size, lo, hi);
            // One step, plus float32 rounding of min + t * size (an ulp at |z| = 17 is 1.9e-6, about the step).
            Assert.That(MathF.Abs(back - v), Is.LessThanOrEqualTo(innerStep + (MathF.Abs(min) + size) * 2.4e-7f), $"inner {v} -> {back}");
        }
        // Linear over the whole bounds, the same z steps 1.2e-3: a thousand times coarser.
        Assert.That(innerStep, Is.LessThan((outer.MaxZ - outer.MinZ) / ((1 << 24) - 1) / 500f));

        // Tails: relative error about 1e-5 at worst, both ends, near the box and far out.
        foreach (float e in new[] { 1e-3f, 0.05f, 1f, 37f, 900f, 6700f })
        {
            float below = min - e, above = min + size + e;
            if (e <= lo)
            {
                float b = SceneCodec.DequantPosP(SceneCodec.QuantPosP(below, min, size, lo, hi), min, size, lo, hi);
                Assert.That(MathF.Abs(b - below), Is.LessThanOrEqualTo(2e-5f * (e + size / 16f) + MathF.Abs(below) * 1e-6f), $"below {below} -> {b}");
            }
            float a = SceneCodec.DequantPosP(SceneCodec.QuantPosP(above, min, size, lo, hi), min, size, lo, hi);
            Assert.That(MathF.Abs(a - above), Is.LessThanOrEqualTo(2e-5f * (e + size / 16f) + MathF.Abs(above) * 1e-6f), $"above {above} -> {a}");
        }
        // The extremes land on the bounds; beyond them clamps.
        Assert.That(SceneCodec.DequantPosP(SceneCodec.QuantPosP(outer.MaxZ, min, size, lo, hi), min, size, lo, hi), Is.EqualTo(outer.MaxZ).Within(0.2f));
        Assert.That(SceneCodec.DequantPosP(SceneCodec.QuantPosP(outer.MinZ, min, size, lo, hi), min, size, lo, hi), Is.EqualTo(outer.MinZ).Within(0.1f));
        Assert.That(SceneCodec.QuantPosP(1e9f, min, size, lo, hi), Is.EqualTo((1u << 24) - 1u));
        Assert.That(SceneCodec.QuantPosP(-1e9f, min, size, lo, hi), Is.EqualTo(0u));

        // Monotonic across the seams: order is kept, so nothing folds back over the box.
        uint prev = 0;
        for (float v = outer.MinZ; v <= outer.MaxZ; v += v < min - 1 || v > min + size + 1 ? 13.7f : 0.0137f)
        {
            uint q = SceneCodec.QuantPosP(v, min, size, lo, hi);
            Assert.That(q, Is.GreaterThanOrEqualTo(prev), $"code went backwards at {v}");
            prev = q;
        }
    }

    [Test]
    public void PiecewisePosition_NoTails_IsLinearOverTheBox()
    {
        // Inner covering the bounds: no tails, every code inside, still within one step.
        var b = new SplatBounds.Aabb(-3f, -1f, 0f, 5f, 2f, 9f);
        var f = SceneCodec.Frame.From(b, b, 1);
        Assert.That(f.TailLoX + f.TailHiX + f.TailLoY + f.TailHiY + f.TailLoZ + f.TailHiZ, Is.EqualTo(0f));
        float step = f.SizeX / ((1 << 24) - 1 - 2 * (1 << 20));
        for (float v = -3f; v <= 5f; v += 0.0731f)
        {
            float back = SceneCodec.DequantPosP(SceneCodec.QuantPosP(v, f.MinX, f.SizeX, 0f, 0f), f.MinX, f.SizeX, 0f, 0f);
            Assert.That(MathF.Abs(back - v), Is.LessThanOrEqualTo(step + 1e-6f), $"{v} -> {back}");
        }
        // The legacy frame (files without Header2.Inner) is unchanged.
        var legacy = SceneCodec.Frame.From(b, 1);
        Assert.That(legacy.Piecewise, Is.EqualTo(0));
        Assert.That(legacy.SizeX, Is.EqualTo(8f));
    }

    [Test]
    public void Sh_RoundTrips_WithinHalfAStep_AndClampsBeyondTheRange()
    {
        for (float v = -1f; v <= 1f; v += 0.0137f)
        {
            float back = SceneCodec.DequantSh(SceneCodec.QuantSh(v));
            Assert.That(MathF.Abs(back - v), Is.LessThanOrEqualTo(0.5f / 127f + 1e-6f), $"{v} -> {back}");
        }
        Assert.That(SceneCodec.DequantSh(SceneCodec.QuantSh(3f)), Is.EqualTo(1f).Within(1e-6f));
        Assert.That(SceneCodec.DequantSh(SceneCodec.QuantSh(0f)), Is.EqualTo(0f).Within(1e-6f));
    }

    [Test]
    public void Rotation_SmallestThree_RecoversTheRotation()
    {
        var rng = new Random(11);
        double worstDeg = 0;
        for (int k = 0; k < 5000; k++)
        {
            float x = (float)(rng.NextDouble() * 2 - 1), y = (float)(rng.NextDouble() * 2 - 1),
                  z = (float)(rng.NextDouble() * 2 - 1), w = (float)(rng.NextDouble() * 2 - 1);
            if (k < 4) { x = k == 0 ? 1 : 0; y = k == 1 ? 1 : 0; z = k == 2 ? 1 : 0; w = k == 3 ? -1 : 0; }   // axis-aligned + sign
            float len = MathF.Sqrt(x * x + y * y + z * z + w * w);
            x /= len; y /= len; z /= len; w /= len;
            SceneCodec.UnpackRotation(SceneCodec.PackRotation(x, y, z, w), out float bx, out float by, out float bz, out float bw);
            // q and -q are the same rotation: compare the angle between them.
            double dot = Math.Min(1.0, Math.Abs((double)x * bx + (double)y * by + (double)z * bz + (double)w * bw));
            worstDeg = Math.Max(worstDeg, 2 * Math.Acos(dot) * 180 / Math.PI);
        }
        TestContext.Out.WriteLine($"worst rotation error {worstDeg:F3} deg");
        Assert.That(worstDeg, Is.LessThan(0.25));
    }

    [Test]
    public void Frame_DegenerateAxis_StaysFinite()
    {
        var f = SceneCodec.Frame.From(new SplatBounds.Aabb(1, 2, 3, 1, 5, 3), 10);
        Assert.That(f.SizeX, Is.EqualTo(1f));
        Assert.That(f.SizeY, Is.EqualTo(3f));
        Assert.That(SceneCodec.DequantPos(SceneCodec.QuantPos(1f, f.MinX, f.SizeX), f.MinX, f.SizeX), Is.EqualTo(1f).Within(1e-6f));
    }
}
