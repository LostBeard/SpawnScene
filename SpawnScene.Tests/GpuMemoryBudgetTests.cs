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
        var (targets, splats) = GpuMemoryBudget.Derive(0, 2047 * MiB, 3_000_000);
        Assert.That(targets, Is.EqualTo(1024 * MiB), "a quarter of 4 GB, under the 2047 MiB binding limit");
        long expect = (4096 - 1024 - 256) * MiB / GpuMemoryBudget.BytesPerSplat;
        Assert.That(splats, Is.EqualTo((int)expect), "the rest at the measured bytes per splat");
        Assert.That(splats, Is.LessThan(3_000_000), "4 GB cannot hold the preset's 3M");
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
        // 12 GB (an RTX 4070): the budget fits ~10M splats, under the binding's 11.9M.
        long budget12 = (12L * 1024 - 2047 - 256) * MiB / GpuMemoryBudget.BytesPerSplat;
        Assert.That(GpuMemoryBudget.Derive(12, 2047 * MiB, deviceMax).MaxSplats, Is.EqualTo((int)budget12));
        Assert.That(budget12, Is.GreaterThan(9_500_000).And.LessThan(11_924_639));
        // 48 GB: the budget would fit ~36M, but one 2047 MiB binding of 45-float SH rows holds 11.9M.
        long binding = 2047 * MiB / GpuMemoryBudget.WidestSplatRowBytes;
        Assert.That(GpuMemoryBudget.Derive(48, 2047 * MiB, deviceMax).MaxSplats, Is.EqualTo((int)binding));
        Assert.That(binding, Is.EqualTo(11_924_639));
    }

    [Test]
    public void SplatCap_NeverExceedsThePreset_NorDropsBelowTheFloor()
    {
        Assert.That(GpuMemoryBudget.Derive(16, 2047 * MiB, 500_000).MaxSplats, Is.EqualTo(500_000), "the preset wins when smaller");
        Assert.That(GpuMemoryBudget.Derive(2, 128 * MiB, 3_000_000).MaxSplats, Is.GreaterThanOrEqualTo(100_000));
    }
}
