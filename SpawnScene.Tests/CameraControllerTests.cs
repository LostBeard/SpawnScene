using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for the CAMERA layer, which is where both viewer failures of 2026-09-21 actually lived.
///
/// Worth being precise, because the first write-up of that day was not: the display renderer was
/// handed a wrong camera on both occasions and drew exactly what it was asked to. It renders a
/// 14,000,000-splat scene and the DAv2-generated scenes correctly. The bugs were
/// CameraController.UpdateCamera replacing the camera basis with world +Y on every frame, and
/// CameraController.FitToScene aiming at a hardcoded TempleRing point.
///
/// Neither needed a GPU to catch, and neither was caught, because nothing tested this class.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter CameraController</c>
/// </summary>
public class CameraControllerTests
{
    static (CameraController Cam, SceneManager Scene) Fresh()
    {
        var scene = new SceneManager();
        return (new CameraController(scene), scene);
    }

    /// <summary>
    /// THE regression test. Setting a pose and then receiving any input must not change what the
    /// viewer is looking at.
    ///
    /// This is the bug: SetPose carefully computed a capture camera's basis including its roll and
    /// assigned it, and then the first mouse move called UpdateCamera and replaced the up vector
    /// with world +Y. A reconstruction has no gravity in it, so Bathroom's came out with its up at
    /// essentially -Y, and the room appeared rotated 90 degrees and tumbled as soon as anyone
    /// moved - while every PSNR and SSIM number stayed good, because those come from the trainer's
    /// rasteriser using the real camera basis.
    ///
    /// The contract is stability: if SetPose cannot keep a promise, it must not make it.
    /// </summary>
    [Test]
    public void APoseSurvivesTheFirstInputEvent()
    {
        var (cam, scene) = Fresh();

        // A capture pose from a reconstruction whose world +Y is NOT up - which is every
        // reconstruction, until something stands it up.
        var position = new Vector3(1.5f, -0.4f, 2.2f);
        var forward = Vector3.Normalize(new Vector3(0.3f, 0.1f, -1f));
        var up = Vector3.Normalize(new Vector3(0.2f, -0.94f, 0.1f));

        cam.SetPose(position, forward, up);

        // SNAPSHOT the values. CameraParams is a class, so holding the object would alias the
        // very thing under test and the comparison would be object-to-itself - vacuously green.
        var posedAt = scene.Camera.Position;
        var posedFwd = scene.Camera.Forward;
        var posedUp = scene.Camera.Up;

        // Any input at all. A zero-magnitude mouse move is still an input event.
        cam.OnMouseMove(0, 0, isPointerLocked: true);
        var afterInput = scene.Camera;

        Assert.That(Vector3.Distance(afterInput.Position, posedAt), Is.LessThan(1e-5f),
            "an input event must not move the camera that was just placed");
        Assert.That(Vector3.Dot(afterInput.Forward, posedFwd), Is.EqualTo(1f).Within(1e-4f),
            "nor turn it");
        Assert.That(Vector3.Dot(afterInput.Up, posedUp), Is.EqualTo(1f).Within(1e-4f),
            "nor roll it - this is the failure that put a room on its side while every number " +
            "said the reconstruction was the best that capture had produced");
    }

    [Test]
    public void APoseWithAnUprightUpIsUnchangedToo()
    {
        var (cam, scene) = Fresh();
        var forward = Vector3.Normalize(new Vector3(0.4f, -0.2f, -1f));

        cam.SetPose(new Vector3(2f, 1f, 3f), forward, Vector3.UnitY);
        var posedFwd = scene.Camera.Forward;
        var posedUp = scene.Camera.Up;
        cam.OnMouseMove(0, 0, isPointerLocked: true);

        Assert.That(Vector3.Dot(scene.Camera.Forward, posedFwd), Is.EqualTo(1f).Within(1e-4f));
        Assert.That(Vector3.Dot(scene.Camera.Up, posedUp), Is.EqualTo(1f).Within(1e-4f));
    }

    /// <summary>
    /// The camera must end up looking at the scene, not at a coordinate from another dataset.
    ///
    /// FitToScene's multi-view branch tried a literal TempleRing bounding-box midpoint first and
    /// fell back to the camera centroid only when that point happened to be behind the camera. On
    /// drjohnson it won that coin toss and the viewer aimed into empty space: 1,129,128 splats and
    /// a completely black frame.
    /// </summary>
    [Test]
    public void FitToSceneAimsAtTheScene()
    {
        var (cam, scene) = Fresh();

        // Cameras on one side of a subject that is nowhere near any hardcoded constant.
        var subject = new Vector3(40f, 12f, -25f);
        // GpuSplatCount matters: FitToScene returns immediately when scene.Count is 0, so
        // without it this test asserted against a camera FitToScene never touched.
        var gs = new SpawnScene.Models.GaussianScene
        {
            SourceName = "multi-view",
            GpuSplatCount = 10_000,
        };
        for (int i = -2; i <= 2; i++)
        {
            var pos = subject + new Vector3(i * 2f, 0f, 14f);
            gs.TrainingCameras.Add(new SpawnScene.Models.CameraParams
            {
                Width = 640, Height = 480, FocalX = 500, FocalY = 500,
                CenterX = 320, CenterY = 240,
                Position = pos,
                Forward = Vector3.Normalize(subject - pos),
                Up = Vector3.UnitY,
            });
        }
        scene.ActiveScene = gs;

        cam.FitToScene();

        var toSubject = Vector3.Normalize(subject - scene.Camera.Position);
        Assert.That(Vector3.Dot(scene.Camera.Forward, toSubject), Is.GreaterThan(0.7f),
            "the camera must face what the photographs faced, not a constant from another dataset");
    }
}
