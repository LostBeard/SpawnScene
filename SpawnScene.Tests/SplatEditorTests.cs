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
