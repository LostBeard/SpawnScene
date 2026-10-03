using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Grip world grab (XRWorldGrab, 2026-10-03): what is under a hand stays under it.</summary>
public class XRWorldGrabTests
{
    static Matrix4x4 Placed() => XRSceneAlignment.SceneFromRoom(new Vector3(0, 1.6f, 0), Quaternion.Identity, new Vector3(5, 2, 3), Vector3.UnitX);

    static void Near(Vector3 a, Vector3 b, string what) => Assert.That(Vector3.Distance(a, b), Is.LessThan(1e-4f), $"{what}: {a} vs {b}");

    [Test]
    public void OneHand_DragsTheScene()
    {
        var grab = new XRWorldGrab();
        var m0 = Placed();
        var h0 = new Vector3(0.3f, 1.2f, -0.4f);
        var held = Vector3.Transform(h0, m0);
        var m = grab.Step(m0, false, default, true, h0);
        Assert.That(m, Is.EqualTo(m0), "grabbing does not move anything");
        var h1 = new Vector3(-0.1f, 1.5f, 0.2f);
        m = grab.Step(m, false, default, true, h1);
        Near(Vector3.Transform(h1, m), held, "held point under the hand");
        Assert.That(XRWorldGrab.Scale(m), Is.EqualTo(1f).Within(1e-5f));
    }

    [Test]
    public void TwoHands_ScaleAndTurn_KeepBothPointsUnderTheHands()
    {
        var grab = new XRWorldGrab();
        var m0 = Placed();
        Vector3 l0 = new(-0.3f, 1.2f, -0.4f), r0 = new(0.3f, 1.2f, -0.4f);
        var heldL = Vector3.Transform(l0, m0);
        var heldR = Vector3.Transform(r0, m0);
        var m = grab.Step(m0, true, l0, true, r0);
        // Hands twice as far apart, the line between them turned 40 degrees, and moved.
        float t = 40f * MathF.PI / 180f;
        var dir = new Vector3(MathF.Cos(t), 0, MathF.Sin(t));
        var mid = new Vector3(0.1f, 1.2f, -0.2f);
        Vector3 l1 = mid - dir * 0.6f, r1 = mid + dir * 0.6f;
        m = grab.Step(m, true, l1, true, r1);
        Near(Vector3.Transform(l1, m), heldL, "left");
        Near(Vector3.Transform(r1, m), heldR, "right");
        Assert.That(XRWorldGrab.Scale(m), Is.EqualTo(0.5f).Within(1e-4f), "hands apart: the scene grows (fewer scene units per metre)");
        Assert.That(Vector3.Dot(Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitY, m)), Vector3.UnitY), Is.GreaterThan(0.9999f), "up stays up");
    }

    [Test]
    public void SecondHandJoining_DoesNotJump()
    {
        var grab = new XRWorldGrab();
        var m = Placed();
        Vector3 l = new(-0.3f, 1.2f, -0.4f), r = new(0.3f, 1.2f, -0.4f);
        m = grab.Step(m, true, l, false, default);
        m = grab.Step(m, true, l + new Vector3(0.1f, 0, 0), false, default);
        var before = m;
        m = grab.Step(m, true, l + new Vector3(0.1f, 0, 0), true, r);
        Assert.That(m, Is.EqualTo(before));
    }

    [Test]
    public void Scale_IsClamped()
    {
        var grab = new XRWorldGrab();
        var m = Placed();
        m = grab.Step(m, true, new Vector3(-1f, 1, 0), true, new Vector3(1f, 1, 0));
        m = grab.Step(m, true, new Vector3(-0.0008f, 1, 0), true, new Vector3(0.0008f, 1, 0));   // 1250x, above the 1 mm guard
        Assert.That(XRWorldGrab.Scale(m), Is.EqualTo(XRWorldGrab.MaxScale).Within(1f));
    }
}
