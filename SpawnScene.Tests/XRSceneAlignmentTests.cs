using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>XR room-to-scene alignment (XRSceneAlignment, 2026-10-03): the head starts at the desktop camera, facing its way.</summary>
public class XRSceneAlignmentTests
{
    static Quaternion Yaw(float deg) => Quaternion.CreateFromAxisAngle(Vector3.UnitY, deg * MathF.PI / 180f);

    [TestCase(0f, 1f, 0f, 0f)]      // camera faces +X
    [TestCase(30f, -1f, 0f, -1f)]   // head turned, camera faces -X-Z
    [TestCase(-120f, 0f, 0f, 1f)]   // camera faces +Z
    public void HeadLandsOnTheCamera_FacingItsWay(float headYawDeg, float fx, float fy, float fz)
    {
        var head = new Vector3(0.3f, 1.7f, -0.2f);
        var headRot = Yaw(headYawDeg);
        var camPos = new Vector3(5, 2, 3);
        var camFwd = Vector3.Normalize(new Vector3(fx, fy, fz));
        var m = XRSceneAlignment.SceneFromRoom(head, headRot, camPos, camFwd);

        Assert.That(Vector3.Distance(Vector3.Transform(head, m), camPos), Is.LessThan(1e-4f), "head -> camera position");
        var headFwd = Vector3.Transform(-Vector3.UnitZ, headRot);
        var mapped = Vector3.Normalize(Vector3.TransformNormal(headFwd, m));
        Assert.That(Vector3.Dot(mapped, camFwd), Is.GreaterThan(0.9999f), "head facing -> camera facing");
        Assert.That(Vector3.Dot(Vector3.TransformNormal(Vector3.UnitY, m), Vector3.UnitY), Is.GreaterThan(0.9999f), "up stays up");
    }

    [Test]
    public void WebXRProjection_ProjectsLikeTheGLMatrix()
    {
        // A WebXR / GL perspective (column-major array, column vectors): fov 90 deg, near 0.1, far 100.
        float f = 1f / MathF.Tan(MathF.PI / 4), n = 0.1f, fa = 100f;
        var gl = new float[16];
        gl[0] = f; gl[5] = f; gl[10] = (fa + n) / (n - fa); gl[11] = -1; gl[14] = 2 * fa * n / (n - fa);
        var m = XRSceneAlignment.FromWebXRMatrix(gl);
        // A point 2 m ahead (-Z), 0.5 m right and 0.25 m up: NDC must be (f * 0.5 / 2, f * 0.25 / 2).
        var clip = Vector4.Transform(new Vector4(0.5f, 0.25f, -2f, 1f), m);
        Assert.That(clip.X / clip.W, Is.EqualTo(f * 0.5f / 2f).Within(1e-5f));
        Assert.That(clip.Y / clip.W, Is.EqualTo(f * 0.25f / 2f).Within(1e-5f));
        Assert.That(clip.W, Is.EqualTo(2f).Within(1e-5f), "w is the distance ahead");
    }

    [Test]
    public void SceneView_SeesFromTheCameraPosition()
    {
        // A room view at the head; in scene terms the eye must sit at the camera position.
        var head = new Vector3(0, 1.7f, 0);
        var m = XRSceneAlignment.SceneFromRoom(head, Quaternion.Identity, new Vector3(5, 2, 3), Vector3.UnitX);
        var roomView = Matrix4x4.CreateTranslation(-head);   // identity orientation
        var sceneView = XRSceneAlignment.SceneView(roomView, m);
        Matrix4x4.Invert(sceneView, out var eyeToScene);
        Assert.That(Vector3.Distance(Vector3.Transform(Vector3.Zero, eyeToScene), new Vector3(5, 2, 3)), Is.LessThan(1e-4f));
    }
}
