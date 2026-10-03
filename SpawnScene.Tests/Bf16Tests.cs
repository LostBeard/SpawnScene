using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>bfloat16 packing for the trainer's SH Adam moments (Bf16, 2026-10-02).</summary>
public class Bf16Tests
{
    [Test]
    public void RoundsToNearestEven()
    {
        // 1 + 2^-8 sits exactly halfway between bf16 1.0 and 1 + 2^-7: ties go to the even mantissa (1.0).
        Assert.That(Bf16.Round(1f + MathF.Pow(2, -8)), Is.EqualTo(1f));
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 (odd) and 1 + 2^-6 (even): up.
        Assert.That(Bf16.Round(1f + 3 * MathF.Pow(2, -8)), Is.EqualTo(1f + MathF.Pow(2, -6)));
        // Just above a tie rounds up.
        Assert.That(Bf16.Round(1f + MathF.Pow(2, -8) + MathF.Pow(2, -20)), Is.EqualTo(1f + MathF.Pow(2, -7)));
        Assert.That(Bf16.Round(-2.5f), Is.EqualTo(-2.5f));
    }

    [Test]
    public void KeepsF32Range_WhereHalfWouldUnderflow()
    {
        // A second moment of a small SH gradient: 1e-12 is far below IEEE half's smallest subnormal (6e-8).
        float v = 1e-12f;
        Assert.That(Bf16.Round(v), Is.EqualTo(v).Within(v * 0.01f));
        Assert.That((float)(Half)v, Is.EqualTo(0f), "IEEE half flushes it to zero; bf16 must not");
        Assert.That(float.IsNaN(Bf16.Round(float.NaN)), Is.True);
        Assert.That(Bf16.Round(float.PositiveInfinity), Is.EqualTo(float.PositiveInfinity));
    }

    [Test]
    public void PackUnpack_OddRowWidth_RoundTripsTheRoundedValues()
    {
        const int width = 45; // SH rest floats per splat: 23 words, the last half-empty
        Assert.That(Bf16.WordsPerRow(width), Is.EqualTo(23));
        var values = new float[3 * width];
        for (int i = 0; i < values.Length; i++) values[i] = (i - 60) * 0.0137f + MathF.Sin(i);
        var packed = Bf16.Pack(values, width);
        Assert.That(packed.Length, Is.EqualTo(3 * 23));
        var back = Bf16.Unpack(packed, width);
        for (int i = 0; i < values.Length; i++)
            Assert.That(back[i], Is.EqualTo(Bf16.Round(values[i])), $"element {i}");
        // Rows do not bleed: the empty half of each row's last word stays zero.
        for (int r = 0; r < 3; r++) Assert.That(packed[r * 23 + 22] >> 16, Is.EqualTo(0u), $"row {r}");
    }
}
