using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Scene editing (SplatEditor, 2026-10-03): the selection test and the delete / keep / undo kernels.</summary>
public class SplatEditorTests
{
    [Test]
    public void Box_SelectsWhatIsInsideIt()
    {
        // A box 4 x 2 x 2 centred at (5, 0, 0).
        var v = SplatEditor.Volume.Box(Matrix4x4.CreateScale(2, 1, 1) * Matrix4x4.CreateTranslation(5, 0, 0));
        Assert.That(SplatEditor.Inside(v, 5, 0, 0), Is.True);
        Assert.That(SplatEditor.Inside(v, 6.9f, 0.9f, -0.9f), Is.True);
        Assert.That(SplatEditor.Inside(v, 7.1f, 0, 0), Is.False, "past the long side");
        Assert.That(SplatEditor.Inside(v, 5, 1.1f, 0), Is.False, "above it");
    }

    [Test]
    public void ScreenRect_SelectsWhatIsUnderIt_InFrontOnly()
    {
        // A camera at the origin looking down +Z (the single-photo convention), 90 degree view.
        var cam = new SpawnScene.Models.CameraParams
        {
            Width = 1000, Height = 1000, FocalX = 500, FocalY = 500, CenterX = 500, CenterY = 500,
            Near = 0.01f, Far = 100f, Position = Vector3.Zero, Forward = Vector3.UnitZ, Up = Vector3.UnitY,
        };
        var proj = SpawnScene.Models.CameraParams.CreateWebGpuProjection(cam.FocalX, cam.FocalY, cam.CenterX, cam.CenterY,
            cam.Width, cam.Height, cam.Near, cam.Far);
        var vp = cam.ViewMatrix * proj;
        // The right half of the screen.
        var v = SplatEditor.Volume.ScreenRect(vp, 0f, 1f, -1f, 1f);
        var right = Vector3.Transform(new Vector3(1, 0, 0), Matrix4x4.CreateFromQuaternion(Quaternion.Identity));
        // Which world direction is screen-right for this camera? Take the projection's word for it.
        var probe = Vector4.Transform(new Vector4(0.5f, 0, 3, 1), vp);
        float sx = MathF.Sign(probe.X / probe.W);
        Assert.That(SplatEditor.Inside(v, 0.5f * sx, 0, 3), Is.True, "on screen-right, in front");
        Assert.That(SplatEditor.Inside(v, -0.5f * sx, 0, 3), Is.False, "on screen-left");
        Assert.That(SplatEditor.Inside(v, 0.5f * sx, 0, -3), Is.False, "behind the camera");
    }

    [Test]
    public async Task DeleteKeepUndo_OnTheGpuKernels()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        // Four splats at x = 0, 1, 2, 3, all opaque.
        var data = new float[4 * SplatFormat.Floats];
        for (int i = 0; i < 4; i++) { data[i * SplatFormat.Floats] = i; data[i * SplatFormat.Floats + SplatFormat.OffOpacity] = 0.8f; }
        using var packed = accel.Allocate1D(data);
        float[] Opacity() => packed.GetAsArray1D().Where((_, k) => k % SplatFormat.Floats == SplatFormat.OffOpacity).ToArray();

        var editor = new SplatEditor();
        // A box around x = 1..2.
        var v = SplatEditor.Volume.Box(Matrix4x4.CreateScale(0.6f, 1, 1) * Matrix4x4.CreateTranslation(1.5f, 0, 0));
        Assert.That(await editor.CountAsync(accel, packed, 4, v), Is.EqualTo(2));

        await editor.ApplyAsync(accel, packed, 4, v, SplatEditor.Mode.DeleteInside);
        Assert.That(Opacity(), Is.EqualTo(new[] { 0.8f, 0f, 0f, 0.8f }));
        Assert.That(await editor.CountAsync(accel, packed, 4, v), Is.EqualTo(0), "deleted splats are not selected again");

        Assert.That(await editor.UndoAsync(accel, packed, 4), Is.True);
        Assert.That(Opacity(), Is.EqualTo(new[] { 0.8f, 0.8f, 0.8f, 0.8f }));

        await editor.ApplyAsync(accel, packed, 4, v, SplatEditor.Mode.KeepInside);
        Assert.That(Opacity(), Is.EqualTo(new[] { 0f, 0.8f, 0.8f, 0f }));
        Assert.That(await editor.UndoAsync(accel, packed, 4), Is.True);
        Assert.That(await editor.UndoAsync(accel, packed, 4), Is.False, "nothing left to undo");
        Assert.That(Opacity(), Is.EqualTo(new[] { 0.8f, 0.8f, 0.8f, 0.8f }));
        editor.Dispose();
    }

    [Test]
    public async Task Filters_AndInvert_SelectByOpacityAndSize_OnTheGpuKernels()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        // Six splats at x = 0..5: opacity 0.05 / 0.9 alternating, largest axis 0.01 for x < 3 and 0.5 from x = 3.
        const int F = SplatFormat.Floats;
        var data = new float[6 * F];
        for (int i = 0; i < 6; i++)
        {
            data[i * F] = i;
            data[i * F + SplatFormat.OffOpacity] = i % 2 == 0 ? 0.05f : 0.9f;
            float s = i < 3 ? 0.01f : 0.5f;
            data[i * F + 6] = s * 0.2f; data[i * F + 7] = s; data[i * F + 8] = s * 0.5f;   // the largest is not always x
        }
        using var packed = accel.Allocate1D(data);
        var editor = new SplatEditor();
        var all = SplatEditor.Volume.All();
        Assert.That(await editor.CountAsync(accel, packed, 6, all), Is.EqualTo(6));

        var faint = all; faint.OpacityBelow = 0.1f;
        Assert.That(await editor.CountAsync(accel, packed, 6, faint), Is.EqualTo(3), "x = 0, 2, 4");
        var large = all; large.SizeAbove = 0.1f;
        Assert.That(await editor.CountAsync(accel, packed, 6, large), Is.EqualTo(3), "x = 3, 4, 5 by their y axis");
        var both = faint; both.SizeAbove = 0.1f;
        Assert.That(await editor.CountAsync(accel, packed, 6, both), Is.EqualTo(1), "x = 4 only");

        // A box around x = 1..2, inverted: x = 0, 3, 4, 5; with the faint filter: 0 and 4.
        var box = SplatEditor.Volume.Box(Matrix4x4.CreateScale(0.6f, 1, 1) * Matrix4x4.CreateTranslation(1.5f, 0, 0));
        var outside = box; outside.Invert = 1;
        Assert.That(await editor.CountAsync(accel, packed, 6, outside), Is.EqualTo(4));
        var outsideFaint = outside; outsideFaint.OpacityBelow = 0.1f;
        await editor.ApplyAsync(accel, packed, 6, outsideFaint, SplatEditor.Mode.DeleteInside);
        float[] op = packed.GetAsArray1D().Where((_, k) => k % F == SplatFormat.OffOpacity).ToArray();
        Assert.That(op, Is.EqualTo(new[] { 0f, 0.9f, 0.05f, 0.9f, 0f, 0.9f }));
        Assert.That(await editor.UndoAsync(accel, packed, 6), Is.True);

        // A moved region keeps its filters.
        var moved = outsideFaint.MovedBy(new Vector3(1, 0, 0));
        Assert.That(moved.Invert, Is.EqualTo(1));
        Assert.That(moved.OpacityBelow, Is.EqualTo(0.1f));
        Assert.That(await editor.CountAsync(accel, packed, 6, moved), Is.EqualTo(2), "box now x = 2..3: outside and faint = 0, 4");
        editor.Dispose();
    }

    [Test]
    public async Task LargestPercent_SelectsThatShareOfTheVisibleSplats()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        const int F = SplatFormat.Floats, n = 10_000;
        var rng = new Random(4);
        var data = new float[n * F];
        for (int i = 0; i < n; i++)
        {
            data[i * F] = i;
            data[i * F + SplatFormat.OffOpacity] = i < 1000 ? 0f : 0.7f;              // 1000 deleted: not counted
            float s = MathF.Exp((float)(rng.NextDouble() * 8 - 7));                     // log-uniform, ~1e-3 .. 2.7
            data[i * F + 6] = s * 0.3f; data[i * F + 7] = s * 0.1f; data[i * F + 8] = s;
        }
        using var packed = accel.Allocate1D(data);
        var editor = new SplatEditor();
        foreach (double f in new[] { 0.01, 0.05, 0.1 })
        {
            var v = SplatEditor.Volume.All();
            v.SizeAbove = await editor.SizeQuantileAsync(accel, packed, n, f);
            int got = await editor.CountAsync(accel, packed, n, v);
            // The bin edge under the quantile: at least the share, and at most one 1/64-octave bin more.
            Assert.That(got, Is.InRange((int)(9000 * f), (int)(9000 * f) + 60), $"largest {f:P0}");
        }
        editor.Dispose();
    }

    [Test]
    public void ScreenCircle_IsRoundOnScreen_NotItsBox()
    {
        // The identity as view * projection: NDC = world x, y; a 1000 x 500 screen, a 100 px dab at its centre covers
        // NDC x within 0.2 and y within 0.4.
        var v = SplatEditor.Volume.ScreenCircle(Matrix4x4.Identity, new Vector2(500, 250), 100, 1000, 500);
        Assert.That(SplatEditor.Inside(v, 0f, 0f, 0.5f), Is.True, "centre");
        Assert.That(SplatEditor.Inside(v, 0.19f, 0f, 0.5f), Is.True, "inside on x");
        Assert.That(SplatEditor.Inside(v, 0f, 0.39f, 0.5f), Is.True, "inside on y");
        Assert.That(SplatEditor.Inside(v, 0.15f, 0.3f, 0.5f), Is.False, "the box's corner, outside the circle");
        Assert.That(SplatEditor.Inside(v, 0.21f, 0f, 0.5f), Is.False, "past the radius");
        Assert.That(SplatEditor.Inside(v, 0f, 0f, 1.5f), Is.False, "past far");
    }

    [Test]
    public async Task MaskSelection_ReplaceAddSubtract_FiltersOnTop_DrivesTheEdits()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        const int F = SplatFormat.Floats, n = 10;
        // Ten splats at x = 0..9; odd ones faint (0.05), even ones opaque (0.9).
        var data = new float[n * F];
        for (int i = 0; i < n; i++)
        {
            data[i * F] = i;
            data[i * F + SplatFormat.OffOpacity] = i % 2 == 1 ? 0.05f : 0.9f;
            data[i * F + 6] = data[i * F + 7] = data[i * F + 8] = 0.01f;
        }
        using var packed = accel.Allocate1D(data);
        var editor = new SplatEditor();
        SplatEditor.Volume Box(float from, float to) =>
            SplatEditor.Volume.Box(Matrix4x4.CreateScale((to - from) / 2f, 1, 1) * Matrix4x4.CreateTranslation((from + to) / 2f, 0, 0));
        var mask = SplatEditor.Volume.Masked();

        await editor.CombineAsync(accel, packed, n, Box(-0.5f, 2.5f), SplatEditor.Combine.Replace);     // 0 1 2
        Assert.That(await editor.CountAsync(accel, packed, n, mask), Is.EqualTo(3));
        await editor.CombineAsync(accel, packed, n, Box(5.5f, 8.5f), SplatEditor.Combine.Add);          // + 6 7 8
        Assert.That(await editor.CountAsync(accel, packed, n, mask), Is.EqualTo(6));
        await editor.CombineAsync(accel, packed, n, Box(1.5f, 6.5f), SplatEditor.Combine.Subtract);     // - 2 6
        Assert.That(await editor.CountAsync(accel, packed, n, mask), Is.EqualTo(4), "0 1 7 8");

        var faint = mask; faint.OpacityBelow = 0.1f;
        Assert.That(await editor.CountAsync(accel, packed, n, faint), Is.EqualTo(2), "the faint of 0 1 7 8: 1 7");
        var outside = mask; outside.Invert = 1;
        Assert.That(await editor.CountAsync(accel, packed, n, outside), Is.EqualTo(6), "2 3 4 5 6 9");
        Assert.That(mask.MovedBy(new Vector3(5, 0, 0)).UseMask, Is.EqualTo(1), "a mask follows its splats");

        // The rows the clipboard would copy, then Delete through the mask.
        using (var idx = await SplatRows.SelectIndicesAsync(accel, packed, n, mask, 4, editor))
            Assert.That(idx.GetAsArray1D().OrderBy(i => i), Is.EqualTo(new[] { 0, 1, 7, 8 }));
        await editor.ApplyAsync(accel, packed, n, mask, SplatEditor.Mode.DeleteInside);
        float[] op = packed.GetAsArray1D().Where((_, k) => k % F == SplatFormat.OffOpacity).ToArray();
        Assert.That(op.Select((o, i) => o == 0f ? i : -1).Where(i => i >= 0), Is.EqualTo(new[] { 0, 1, 7, 8 }));

        // A new scene size starts an empty mask: replacing into it does not inherit the old bits.
        await editor.CombineAsync(accel, packed, n - 1, Box(2.5f, 3.5f), SplatEditor.Combine.Add);
        Assert.That(await editor.CountAsync(accel, packed, n - 1, mask), Is.EqualTo(1), "only 3");
        editor.Dispose();
    }

    [Test]
    public async Task MoveRows_ThenDelete_UndoesInOrder()
    {
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        const int F = SplatFormat.Floats;
        var data = new float[4 * F];
        for (int i = 0; i < 4; i++) { data[i * F] = i; data[i * F + SplatFormat.OffOpacity] = 0.8f; }
        using var packed = accel.Allocate1D(data);
        float[] Xs() => Enumerable.Range(0, 4).Select(i => packed.GetAsArray1D()[i * F]).ToArray();
        float[] Opacity() => Enumerable.Range(0, 4).Select(i => packed.GetAsArray1D()[i * F + SplatFormat.OffOpacity]).ToArray();

        var editor = new SplatEditor();
        var rows = SplatEditor.Volume.Rows(2, 4);   // what a paste just added
        Assert.That(await editor.CountAsync(accel, packed, 4, rows), Is.EqualTo(2));

        await editor.MoveAsync(accel, packed, 4, rows, new Vector3(10, 0, 0));
        Assert.That(Xs(), Is.EqualTo(new[] { 0f, 1f, 12f, 13f }), "only the rows moved");

        await editor.ApplyAsync(accel, packed, 4, SplatEditor.Volume.Rows(0, 1), SplatEditor.Mode.DeleteInside);
        Assert.That(Opacity(), Is.EqualTo(new[] { 0f, 0.8f, 0.8f, 0.8f }));

        Assert.That(await editor.UndoAsync(accel, packed, 4), Is.True);   // the delete
        Assert.That(Opacity(), Is.EqualTo(new[] { 0.8f, 0.8f, 0.8f, 0.8f }));
        Assert.That(Xs(), Is.EqualTo(new[] { 0f, 1f, 12f, 13f }), "positions untouched by the opacity undo");

        Assert.That(await editor.UndoAsync(accel, packed, 4), Is.True);   // the move
        Assert.That(Xs(), Is.EqualTo(new[] { 0f, 1f, 2f, 3f }));
        editor.Dispose();
    }

    [Test]
    public void MovedRegion_StillSelectsTheSplatsThatMoved()
    {
        var box = SplatEditor.Volume.Box(Matrix4x4.CreateTranslation(5, 0, 0));
        var offset = new Vector3(0, 3, -2);
        var moved = box.MovedBy(offset);
        Assert.That(SplatEditor.Inside(moved, 5, 3, -2), Is.True, "the box centre, moved");
        Assert.That(SplatEditor.Inside(moved, 5, 0, 0), Is.False, "where it was");
        var rows = SplatEditor.Volume.Rows(3, 9);
        Assert.That(rows.MovedBy(offset).RowFrom, Is.EqualTo(3), "a row range is unchanged");
    }
}
