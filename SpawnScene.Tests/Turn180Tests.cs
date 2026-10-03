using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// GpuImageOps.Turn180 (2026-10-03): the symmetric-pair check extracts features from a photo turned 180 degrees and maps
/// them back as (W - 1 - x, H - 1 - y). That mapping is only right if the turned pixel (x, y) is the source's
/// (W - 1 - x, H - 1 - y).
/// </summary>
public class Turn180Tests
{
    [Test]
    public void TurnedPixel_IsTheSourceMirroredInBothAxes()
    {
        const int w = 7, h = 5;
        var src = new int[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) src[y * w + x] = y * 1000 + x;
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        using var buf = accel.Allocate1D(src);
        using var turned = GpuImageOps.Turn180(accel, buf);
        accel.Synchronize();
        var t = turned.GetAsArray1D();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                Assert.That(t[y * w + x], Is.EqualTo(src[(h - 1 - y) * w + (w - 1 - x)]), $"({x}, {y})");
    }
}
