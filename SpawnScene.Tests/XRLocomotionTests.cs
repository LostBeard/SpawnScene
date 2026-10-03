using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>VR thumbstick locomotion (XRLocomotion, 2026-10-03): move along the head's facing, snap-turn about the head.</summary>
public class XRLocomotionTests
{
    static Quaternion Yaw(float deg) => Quaternion.CreateFromAxisAngle(Vector3.UnitY, deg * MathF.PI / 180f);
    static readonly Vector3 Head = new(0.2f, 1.6f, -0.3f);

    // The room placed in a scene with the head at (5, 2, 3) facing +X (a real alignment, not identity).
    static Matrix4x4 Placed(Quaternion headRot) => XRSceneAlignment.SceneFromRoom(Head, headRot, new Vector3(5, 2, 3), Vector3.UnitX);

    static Vector3 HeadForwardInScene(Matrix4x4 m, Quaternion headRot) =>
        Vector3.Normalize(Vector3.TransformNormal(Vector3.Transform(-Vector3.UnitZ, headRot), m));

    [TestCase(0f)]
    [TestCase(70f)]
    public void LeftStickForward_MovesAlongTheHeadFacing(float headYaw)
    {
        var rot = Yaw(headYaw);
        var m0 = Placed(Yaw(0));   // placed facing one way, then the head turns: movement follows where it looks NOW
        var loco = new XRLocomotion();
        var m1 = loco.Step(m0, Head, rot, new Vector2(0, -1), Vector2.Zero, 0.5f, 2f);
        var moved = Vector3.Transform(Head, m1) - Vector3.Transform(Head, m0);
        var fwd = HeadForwardInScene(m0, rot);
        Assert.That(moved.Length(), Is.EqualTo(1f).Within(1e-4f), "full deflection: speed * dt");
        Assert.That(Vector3.Dot(Vector3.Normalize(moved), fwd), Is.GreaterThan(0.9999f), "along the facing");
        Assert.That(moved.Y, Is.EqualTo(0f).Within(1e-5f), "level");
    }

    [Test]
    public void LeftStickRight_StrafesRight()
    {
        var m0 = Placed(Quaternion.Identity);
        var m1 = new XRLocomotion().Step(m0, Head, Quaternion.Identity, new Vector2(1, 0), Vector2.Zero, 1f, 1f);
        var moved = Vector3.Transform(Head, m1) - Vector3.Transform(Head, m0);
        // Facing +X with +Y up: right is +Z.
        Assert.That(Vector3.Dot(Vector3.Normalize(moved), Vector3.UnitZ), Is.GreaterThan(0.9999f));
    }

    [Test]
    public void RightStickForward_Rises()
    {
        var m0 = Placed(Quaternion.Identity);
        var m1 = new XRLocomotion().Step(m0, Head, Quaternion.Identity, Vector2.Zero, new Vector2(0, -1), 1f, 1f);
        var moved = Vector3.Transform(Head, m1) - Vector3.Transform(Head, m0);
        Assert.That(moved.Y, Is.EqualTo(1f).Within(1e-4f));
        Assert.That(new Vector2(moved.X, moved.Z).Length(), Is.LessThan(1e-5f));
    }

    [Test]
    public void Deadzone_RestingStickDoesNothing()
    {
        var m0 = Placed(Quaternion.Identity);
        var m1 = new XRLocomotion().Step(m0, Head, Quaternion.Identity, new Vector2(0.1f, -0.1f), new Vector2(0.1f, 0.05f), 1f, 1f);
        Assert.That(m1, Is.EqualTo(m0));
    }

    [Test]
    public void SnapTurnRight_TurnsRightAboutTheHead_OncePerPush()
    {
        var m0 = Placed(Quaternion.Identity);
        var loco = new XRLocomotion();
        var fwd0 = HeadForwardInScene(m0, Quaternion.Identity);
        var right0 = Vector3.Cross(fwd0, Vector3.UnitY);
        var m1 = loco.Step(m0, Head, Quaternion.Identity, Vector2.Zero, new Vector2(1, 0), 1f / 72, 1f);
        var fwd1 = HeadForwardInScene(m1, Quaternion.Identity);
        Assert.That(Vector3.Dot(fwd1, fwd0), Is.EqualTo(MathF.Cos(MathF.PI / 6)).Within(1e-4f), "30 degrees");
        Assert.That(Vector3.Dot(fwd1, right0), Is.GreaterThan(0.49f), "toward the right");
        Assert.That(Vector3.Distance(Vector3.Transform(Head, m1), Vector3.Transform(Head, m0)), Is.LessThan(1e-4f), "about the head");
        Assert.That(Vector3.Dot(Vector3.TransformNormal(Vector3.UnitY, m1), Vector3.UnitY), Is.GreaterThan(0.9999f), "up stays up");

        // Held: no second turn. Released past the re-arm point, pushed again: one more.
        var m2 = loco.Step(m1, Head, Quaternion.Identity, Vector2.Zero, new Vector2(1, 0), 1f / 72, 1f);
        Assert.That(m2, Is.EqualTo(m1), "held stick turns once");
        var m3 = loco.Step(m2, Head, Quaternion.Identity, Vector2.Zero, Vector2.Zero, 1f / 72, 1f);
        var m4 = loco.Step(m3, Head, Quaternion.Identity, Vector2.Zero, new Vector2(-1, 0), 1f / 72, 1f);
        Assert.That(Vector3.Dot(HeadForwardInScene(m4, Quaternion.Identity), fwd0), Is.GreaterThan(0.9999f), "left undoes right");
    }
}
