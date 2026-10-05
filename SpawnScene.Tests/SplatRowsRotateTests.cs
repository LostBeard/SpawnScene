using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Turning a reconstruction upright (MultiViewGenerationService.GenerateAsync): SplatRows.Rotate must move every
/// splat exactly as System.Numerics moves the cameras - position and orientation - so the scene and its photos stay in
/// register; UprightRotation must take the cameras' mean up to +Y and leave an upright scene alone.
/// </summary>
public class SplatRowsRotateTests
{
    const int F = SplatFormat.Floats;

    static Matrix4x4 RotationOf(float x, float y, float z, float w) => Matrix4x4.CreateFromQuaternion(new Quaternion(x, y, z, w));

    [Test]
    public void Rotate_MovesPositionsAndOrientationsLikeTheCameras()
    {
        const int n = 200;
        var rng = new Random(5);
        var packed = new float[n * F];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < F; k++) packed[i * F + k] = (float)(rng.NextDouble() * 4 - 2);
            var q = Quaternion.Normalize(new Quaternion(packed[i * F + 10], packed[i * F + 11], packed[i * F + 12], packed[i * F + 13]));
            packed[i * F + 10] = q.X; packed[i * F + 11] = q.Y; packed[i * F + 12] = q.Z; packed[i * F + 13] = q.W;
        }
        var r = Quaternion.CreateFromAxisAngle(Vector3.Normalize(new Vector3(0.3f, 0.2f, 0.9f)), 2.4f);

        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        using var buf = accel.Allocate1D(packed);
        SplatRows.Rotate(accel, buf, n, r);
        accel.Synchronize();
        var got = buf.GetAsArray1D();

        for (int i = 0; i < n; i++)
        {
            int o = i * F;
            var p = Vector3.Transform(new Vector3(packed[o], packed[o + 1], packed[o + 2]), r);
            Assert.That(got[o], Is.EqualTo(p.X).Within(1e-5f), $"splat {i} x");
            Assert.That(got[o + 1], Is.EqualTo(p.Y).Within(1e-5f), $"splat {i} y");
            Assert.That(got[o + 2], Is.EqualTo(p.Z).Within(1e-5f), $"splat {i} z");
            // Orientation: the new quaternion turns any axis as the old one did, followed by r.
            var before = RotationOf(packed[o + 10], packed[o + 11], packed[o + 12], packed[o + 13]);
            var after = RotationOf(got[o + 10], got[o + 11], got[o + 12], got[o + 13]);
            foreach (var axis in new[] { Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ })
            {
                var expected = Vector3.Transform(Vector3.Transform(axis, before), r);
                var actual = Vector3.Transform(axis, after);
                Assert.That(Vector3.Distance(expected, actual), Is.LessThan(1e-5f), $"splat {i} orientation of {axis}");
            }
            // Everything else in the row is untouched.
            for (int k = 3; k < 10; k++) Assert.That(got[o + k], Is.EqualTo(packed[o + k]), $"splat {i} float {k}");
        }
    }

    [Test]
    public void UprightRotation_TurnsTheMeanUpToPlusY()
    {
        // DrJohnson's own-SfM cameras (2026-10-05).
        var drj = SplatRows.UprightRotation(new Vector3(-0.04f, -1.00f, 0.00f));
        Assert.That(drj, Is.Not.Null);
        var up = Vector3.Transform(Vector3.Normalize(new Vector3(-0.04f, -1.00f, 0.00f)), drj!.Value);
        Assert.That(up.Y, Is.EqualTo(1f).Within(1e-5f));

        // Exactly upside down: still turned upright.
        var flipped = SplatRows.UprightRotation(new Vector3(0f, -3f, 0f));
        Assert.That(Vector3.Transform(-Vector3.UnitY, flipped!.Value).Y, Is.EqualTo(1f).Within(1e-5f));

        // Tilted 45 degrees: turned.
        var tilted = SplatRows.UprightRotation(new Vector3(1f, 1f, 0f));
        Assert.That(Vector3.Transform(Vector3.Normalize(new Vector3(1f, 1f, 0f)), tilted!.Value).Y, Is.EqualTo(1f).Within(1e-5f));

        // Upright (TruckFull: (0, 1, 0)) and nearly upright: left alone, so those scenes stay bit-identical.
        Assert.That(SplatRows.UprightRotation(new Vector3(0f, 1f, 0f)), Is.Null);
        Assert.That(SplatRows.UprightRotation(new Vector3(0.02f, 1f, -0.03f)), Is.Null);
        Assert.That(SplatRows.UprightRotation(Vector3.Zero), Is.Null);
    }
}
