using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>In-headset selection box (XRBoxTool, 2026-10-03).</summary>
public class XRBoxToolTests
{
    [Test]
    public void DraggedBox_SelectsWhatIsBetweenTheCorners_InTheScene()
    {
        // A real placement (head at a camera, scene at 0.29 units per metre, turned) so room and scene differ.
        var m = XRSceneAlignment.SceneFromRoom(new Vector3(0, 1.6f, 0), Quaternion.CreateFromAxisAngle(Vector3.UnitY, 0.4f),
            new Vector3(1, 2, 3), Vector3.UnitX, 0.29f);
        var tool = new XRBoxTool { Active = true };
        Vector3 a = new(-0.2f, 1.0f, -0.8f), b = new(0.3f, 1.4f, -0.5f);
        Assert.That(tool.Step(a, true, m), Is.False);
        Assert.That(tool.Step((a + b) / 2, true, m), Is.False);
        Assert.That(tool.Step(b, false, m), Is.True, "release finishes the box");

        var v = SplatEditor.Volume.Box(tool.BoxToScene!.Value);
        Vector3 Scene(Vector3 room) => Vector3.Transform(room, m);
        var inside = Scene(new Vector3(0.0f, 1.2f, -0.6f));
        var outside = Scene(new Vector3(0.4f, 1.2f, -0.6f));
        Assert.That(SplatEditor.Inside(v, inside.X, inside.Y, inside.Z), Is.True);
        Assert.That(SplatEditor.Inside(v, outside.X, outside.Y, outside.Z), Is.False);
    }

    [Test]
    public void FinishedBox_StaysOnTheScene_WhenTheViewerMoves()
    {
        var m0 = XRSceneAlignment.SceneFromRoom(new Vector3(0, 1.6f, 0), Quaternion.Identity, Vector3.Zero, -Vector3.UnitZ, 0.5f);
        var tool = new XRBoxTool { Active = true };
        tool.Step(new Vector3(0, 1, -1), true, m0);
        tool.Step(new Vector3(0.2f, 1.2f, -0.8f), false, m0);
        // The viewer then walks 1 m (the scene slides in the room): the box's room corners move with the scene.
        var m1 = Matrix4x4.CreateTranslation(0, 0, 1) * m0;
        var c0 = XRBoxTool.Corners(tool.BoxToRoom(m0)!.Value);
        var c1 = XRBoxTool.Corners(tool.BoxToRoom(m1)!.Value);
        Assert.That(Vector3.Distance(c1[0] - c0[0], new Vector3(0, 0, -1)), Is.LessThan(1e-4f));
    }

    [Test]
    public void Corners_AndEdges_DescribeTheBox()
    {
        var c = XRBoxTool.Corners(XRBoxTool.RoomBox(new Vector3(0, 0, 0), new Vector3(2, 4, 6)));
        Assert.That(c[0], Is.EqualTo(new Vector3(0, 0, 0)));
        Assert.That(c[7], Is.EqualTo(new Vector3(2, 4, 6)));
        foreach (var (a, b) in XRBoxTool.Edges)
            Assert.That(System.Numerics.BitOperations.PopCount((uint)(a ^ b)), Is.EqualTo(1), "an edge joins corners differing in one axis");
    }
}
