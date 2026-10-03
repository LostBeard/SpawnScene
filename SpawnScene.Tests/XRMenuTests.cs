using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>In-headset menu geometry (XRMenu, 2026-10-03): placement and ray -> panel pixel.</summary>
public class XRMenuTests
{
    static Quaternion Yaw(float deg) => Quaternion.CreateFromAxisAngle(Vector3.UnitY, deg * MathF.PI / 180f);

    [TestCase(0f)]
    [TestCase(-60f)]
    public void LookingStraightAtIt_HitsTheMiddle(float yaw)
    {
        var head = new Vector3(0.3f, 1.6f, -0.2f);
        var model = XRMenuGeometry.PlaceInFront(head, Yaw(yaw));
        var centre = Vector3.Transform(Vector3.Zero, model);
        var px = XRMenuGeometry.RayToPanelPixel(model, head, Vector3.Normalize(centre - head));
        Assert.That(px, Is.Not.Null);
        Assert.That(px!.Value.X, Is.EqualTo(XRMenuGeometry.PanelWidth / 2).Within(0.01f));
        Assert.That(px.Value.Y, Is.EqualTo(XRMenuGeometry.PanelHeight / 2).Within(0.01f));
    }

    [Test]
    public void PixelAxes_RightIsRight_DownIsDown()
    {
        var head = new Vector3(0, 1.6f, 0);
        var model = XRMenuGeometry.PlaceInFront(head, Quaternion.Identity);
        var centre = Vector3.Transform(Vector3.Zero, model);
        // A point 0.1 m to the viewer's right (+X) and 0.05 m down on the panel.
        var target = centre + new Vector3(0.1f, -0.05f, 0);
        var px = XRMenuGeometry.RayToPanelPixel(model, head, Vector3.Normalize(target - head))!.Value;
        Assert.That(px.X, Is.EqualTo(XRMenuGeometry.PanelWidth / 2 + 0.1f / XRMenuGeometry.WorldScale).Within(0.5f));
        Assert.That(px.Y, Is.EqualTo(XRMenuGeometry.PanelHeight / 2 + 0.05f / XRMenuGeometry.WorldScale).Within(0.5f));
    }

    [Test]
    public void Misses_ReturnNull()
    {
        var head = new Vector3(0, 1.6f, 0);
        var model = XRMenuGeometry.PlaceInFront(head, Quaternion.Identity);
        Assert.That(XRMenuGeometry.RayToPanelPixel(model, head, Vector3.UnitZ), Is.Null, "pointing away");
        Assert.That(XRMenuGeometry.RayToPanelPixel(model, head, Vector3.Normalize(new Vector3(1, 0, -0.2f))), Is.Null, "beside it");
    }
}
