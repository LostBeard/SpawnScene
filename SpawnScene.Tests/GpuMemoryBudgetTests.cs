using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>The "GPU memory" device setting turned into training limits (GpuMemoryBudget, 2026-10-02).</summary>
public class GpuMemoryBudgetTests
{
    const long MiB = 1024 * 1024;

    [Test]
    public void Auto_IsFourGigabytes_QuarterForPhotos_RestForSplats()
    {
        var (targets, splats) = GpuMemoryBudget.Derive(0, 2047 * MiB, int.MaxValue);
        Assert.That(targets, Is.EqualTo(1024 * MiB), "a quarter of 4 GB, under the 2047 MiB binding limit");
        long expect = (4096 - 1024 - 256) * MiB / GpuMemoryBudget.BytesPerSplat;
        Assert.That(splats, Is.EqualTo((int)expect), "the rest at the measured bytes per splat");
        // At ~1,350 B/splat Auto held 2.2M; keys on demand and bf16 SH moments (b127/b128) made it ~3.3M.
        Assert.That(splats, Is.GreaterThan(3_000_000));
    }

    [Test]
    public void PhotoStack_NeverExceedsTheDeviceBindingLimit()
    {
        Assert.That(GpuMemoryBudget.Derive(16, 2047 * MiB, 3_000_000).TargetStackBytes, Is.EqualTo(2047 * MiB));
        Assert.That(GpuMemoryBudget.Derive(2, 128 * MiB, 3_000_000).TargetStackBytes, Is.EqualTo(128 * MiB));
    }

    [Test]
    public void DeviceMax_IsTheBudgetUpToTheBindingLimit()
    {
        const int deviceMax = int.MaxValue;
        // 12 GB (an RTX 4070): the budget fits ~11.6M splats, well under one binding's 35.8M SH part rows.
        long budget12 = (12L * 1024 - 2047 - 256) * MiB / GpuMemoryBudget.BytesPerSplat;
        Assert.That(GpuMemoryBudget.Derive(12, 2047 * MiB, deviceMax).MaxSplats, Is.EqualTo((int)budget12));
        Assert.That(budget12, Is.GreaterThan(11_000_000).And.LessThan(35_773_917));
        // 48 GB: the budget would fit ~53M, but one 2047 MiB binding of 15-float SH part rows holds 35.8M (it was
        // 11.9M with all 45 floats in one buffer).
        long binding = 2047 * MiB / GpuMemoryBudget.WidestSplatRowBytes;
        Assert.That(GpuMemoryBudget.Derive(48, 2047 * MiB, deviceMax).MaxSplats, Is.EqualTo((int)binding));
        Assert.That(binding, Is.EqualTo(35_773_917));
    }

    [Test]
    public void KeyCap_FollowsTheBudget_WithinTheGradientBinding()
    {
        // Auto (4 GB): a quarter at 52 B a key, 20.6M - the fixed 40M (2 GB) it replaced was half the budget.
        Assert.That(GpuMemoryBudget.MaxTotalKeys(0, 2047 * MiB), Is.EqualTo(4096 * MiB / 4 / GpuMemoryBudget.BytesPerKey));
        // 12 GB: 61.9M, room for ~3 keys a splat at its ~11.6M splats.
        Assert.That(GpuMemoryBudget.MaxTotalKeys(12, 2047 * MiB), Is.GreaterThan(3L * 11_600_000));
        // 48 GB: capped by one 2047 MiB binding of 12-byte key gradients (178.9M).
        Assert.That(GpuMemoryBudget.MaxTotalKeys(48, 2047 * MiB), Is.EqualTo(2047 * MiB / 12));
        // A tiny device still gets the floor.
        Assert.That(GpuMemoryBudget.MaxTotalKeys(2, 16 * MiB), Is.EqualTo(4_000_000));
    }

    [Test]
    public void SplatCap_NeverExceedsThePreset_NorDropsBelowTheFloor()
    {
        Assert.That(GpuMemoryBudget.Derive(16, 2047 * MiB, 500_000).MaxSplats, Is.EqualTo(500_000), "the preset wins when smaller");
        Assert.That(GpuMemoryBudget.Derive(2, 128 * MiB, 3_000_000).MaxSplats, Is.GreaterThanOrEqualTo(100_000));
    }
}
