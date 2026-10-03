using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// SH rest bands in three part buffers (SphericalHarmonics.Parts, 2026-10-02): one 45-float buffer capped training at
/// 11.9M splats per storage binding.
/// </summary>
public class ShPartsTests
{
    [Test]
    public void ThreePartsOfFiveBands()
    {
        Assert.That(SphericalHarmonics.Parts * SphericalHarmonics.PartFloatsPerSplat, Is.EqualTo(SphericalHarmonics.RestFloatsPerSplat));
        Assert.That(SphericalHarmonics.PartFloatsPerSplat, Is.EqualTo(15));
        // The WGSL accessor's constant must match the C# layout.
        Assert.That(SphericalHarmonics.WgslPartAccess, Does.Contain($"SH_PART_FLOATS : u32 = {SphericalHarmonics.PartFloatsPerSplat}u;"));
        // A part binding of 2047 MiB holds 35.8M splats, three times the single buffer's 11.9M.
        long binding = 2047L * 1024 * 1024;
        Assert.That(binding / (SphericalHarmonics.PartFloatsPerSplat * 4), Is.GreaterThan(35_000_000));
    }

    [Test]
    public void SplitJoin_RoundTrips_AndPutsEachBandInItsPart()
    {
        const int n = 4;
        var rows = new float[n * SphericalHarmonics.RestFloatsPerSplat];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < SphericalHarmonics.RestFloatsPerSplat; k++)
                rows[i * 45 + k] = i * 1000 + k; // splat i, coefficient k
        var parts = SphericalHarmonics.SplitParts(rows);
        Assert.That(parts.Length, Is.EqualTo(3));
        Assert.That(parts.All(p => p.Length == n * 15), Is.True);
        // Coefficient k of splat i is part k/15, index i*15 + k%15.
        Assert.That(parts[0][2 * 15 + 0], Is.EqualTo(2000 + 0));
        Assert.That(parts[1][3 * 15 + 4], Is.EqualTo(3000 + 19));
        Assert.That(parts[2][1 * 15 + 14], Is.EqualTo(1000 + 44));
        Assert.That(SphericalHarmonics.JoinParts(parts), Is.EqualTo(rows));
    }
}
