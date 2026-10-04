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
