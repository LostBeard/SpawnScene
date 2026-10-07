using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Copy / paste row kernels (SplatRows, 2026-10-03), on ILGPU's CPU accelerator.</summary>
public class SplatRowsTests
{
    [Test]
    public async Task CopyThenPaste_AppendsTheSelectedSplatsMoved()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        const int F = SplatFormat.Floats;
        // Five splats at x = 0..4; a colour channel tagged with the index so rows can be told apart.
        var data = new float[5 * F];
        for (int i = 0; i < 5; i++)
        {
            data[i * F + SplatFormat.OffPos] = i;
            data[i * F + SplatFormat.OffColor] = 0.1f * i;
            data[i * F + SplatFormat.OffOpacity] = 0.9f;
        }
        data[2 * F + SplatFormat.OffOpacity] = 0f;   // splat 2 was deleted: never copied
        using var packed = accel.Allocate1D(data);

        // Select x in 1..3 (splats 1, 3 visible; 2 deleted).
        var v = SplatEditor.Volume.Box(Matrix4x4.CreateScale(1.1f, 1, 1) * Matrix4x4.CreateTranslation(2, 0, 0));
        using var indices = await SplatRows.SelectIndicesAsync(accel, packed, 5, v, 2);
        var idx = indices.GetAsArray1D().OrderBy(x => x).ToArray();
        Assert.That(idx, Is.EqualTo(new[] { 1, 3 }));

        using var clip = SplatRows.GatherRows(accel, packed, indices, 2, F);
        accel.Synchronize();
        var clipRows = clip.GetAsArray1D();
        var clipColours = new[] { clipRows[SplatFormat.OffColor], clipRows[F + SplatFormat.OffColor] }.OrderBy(x => x).ToArray();
        Assert.That(clipColours, Is.EqualTo(new[] { 0.1f, 0.3f }).Within(1e-6f), "the selected rows, whole");

        using var grown = SplatRows.AppendMoved(accel, packed, 5, clip, 2, F, new Vector3(10, 0, 0), moveRows: true);
        accel.Synchronize();
        var g = grown.GetAsArray1D();
        Assert.That(g.Length, Is.EqualTo(7 * F));
        Assert.That(g.Take(5 * F), Is.EqualTo(data), "the scene's own rows are untouched");
        var pastedX = new[] { g[5 * F], g[6 * F] }.OrderBy(x => x).ToArray();
        Assert.That(pastedX, Is.EqualTo(new[] { 11f, 13f }), "the copies, moved by the offset");
    }

    [Test]
    public void AppendWithoutMoving_ForShRows()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        using var scene = accel.Allocate1D(new float[] { 1, 2, 3, 4 });   // 2 rows of 2
        using var clip = accel.Allocate1D(new float[] { 5, 6 });           // 1 row
        using var grown = SplatRows.AppendMoved(accel, scene, 2, clip, 1, 2, new Vector3(100, 100, 100), moveRows: false);
        accel.Synchronize();
        Assert.That(grown.GetAsArray1D(), Is.EqualTo(new float[] { 1, 2, 3, 4, 5, 6 }));
    }

    [Test]
    public void Relief_KeepsEveryPixel_ScalesDepthAboutTheCentre_AndComposes()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        const int F = SplatFormat.Floats;
        // Three splats on different rays at depths 1, 2, 3; scene-depth centre 2.
        var data = new float[3 * F];
        for (int i = 0; i < 3; i++)
        {
            float z = i + 1;
            data[i * F] = 0.3f * z; data[i * F + 1] = -0.2f * z; data[i * F + 2] = z;
            data[i * F + 6] = data[i * F + 7] = 0.01f * z; data[i * F + 8] = 0.002f * z;
        }
        using var packed = accel.Allocate1D(data);
        SplatRows.Relief(accel, packed, 3, 2f, 0.5f);
        accel.Synchronize();
        var a = packed.GetAsArray1D();
        for (int i = 0; i < 3; i++)
        {
            float z = a[i * F + 2], want = 2f + (i + 1 - 2f) * 0.5f;
            Assert.That(z, Is.EqualTo(want).Within(1e-5f), $"depth {i}");
            Assert.That(a[i * F] / z, Is.EqualTo(0.3f).Within(1e-5f), "same pixel x");
            Assert.That(a[i * F + 1] / z, Is.EqualTo(-0.2f).Within(1e-5f), "same pixel y");
            Assert.That(a[i * F + 6] / z, Is.EqualTo(0.01f).Within(1e-6f), "the footprint follows the depth");
        }
        SplatRows.Relief(accel, packed, 3, 2f, 3f);   // 0.5 then 3 = 1.5 overall
        accel.Synchronize();
        a = packed.GetAsArray1D();
        Assert.That(a[2], Is.EqualTo(2f + (1f - 2f) * 1.5f).Within(1e-5f), "composes: x0.5 then x3 is x1.5");
        Assert.That(a[2 * F + 2], Is.EqualTo(2f + (3f - 2f) * 1.5f).Within(1e-5f));
    }
}
