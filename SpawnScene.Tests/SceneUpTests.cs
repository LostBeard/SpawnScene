using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for gravity-aligning a reconstruction that has none.
///
/// Why this exists: DAv3 and SfM both recover geometry up to an arbitrary rotation, and
/// everything downstream assumes world +Y is up - CameraController.UpdateCamera rebuilds the
/// camera's up as Vector3.UnitY on EVERY frame, so the capture pose's roll is discarded the
/// instant anyone moves. On Bathroom that showed as the room rotated about 90 degrees with the
/// floor up the side of the screen, and the scene tumbling when the camera moved. The numbers
/// never saw it: the trainer renders from real CameraParams with the correct up, so PSNR and
/// SSIM were measured on a correctly-oriented image the whole time.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter SceneUp</c>
/// </summary>
public class SceneUpTests
{
    static CameraParams Cam(Vector3 forward, Vector3 up) => new()
    {
        Width = 768, Height = 1024, FocalX = 900, FocalY = 900,
        CenterX = 384, CenterY = 512,
        Position = Vector3.Zero,
        Forward = Vector3.Normalize(forward),
        Up = Vector3.Normalize(up),
    };

    [Test]
    public void EstimatesUpFromCamerasThatRoughlyAgree()
    {
        // A handheld walk: the phone is held upright, give or take a few degrees of wobble.
        var trueUp = Vector3.Normalize(new Vector3(0.2f, 0.9f, -0.1f));
        var cams = new List<CameraParams>();
        var rng = new Random(11);
        for (int i = 0; i < 20; i++)
        {
            float J() => (float)(rng.NextDouble() - 0.5) * 0.15f;
            var wobbled = Vector3.Normalize(trueUp + new Vector3(J(), J(), J()));
            cams.Add(Cam(new Vector3(J(), J(), 1f), wobbled));
        }

        Assert.That(WorldSpaceGeometry.TryEstimateSceneUp(cams, out var up, out float conf), Is.True);
        Assert.That(conf, Is.GreaterThan(0.95f), "cameras that agree should report high confidence");
        Assert.That(Vector3.Dot(up, trueUp), Is.GreaterThan(0.99f));
    }

    [Test]
    public void RefusesWhenTheCamerasHaveNoConsistentUp()
    {
        // Rolled all the way round: the ups cancel and there is no gravity to find. Returning a
        // confident average of nothing would silently rotate the scene to an arbitrary axis.
        var cams = new List<CameraParams>();
        for (int i = 0; i < 8; i++)
        {
            float a = i * MathF.PI * 2f / 8f;
            cams.Add(Cam(-Vector3.UnitZ, new Vector3(MathF.Cos(a), MathF.Sin(a), 0f)));
        }

        Assert.That(WorldSpaceGeometry.TryEstimateSceneUp(cams, out _, out float conf), Is.False);
        Assert.That(conf, Is.LessThan(0.05f));
    }

    [Test]
    public void RotationPutsTheEstimatedUpOnWorldY()
    {
        foreach (var v in new[]
        {
            new Vector3(0.2f, 0.9f, -0.1f), new Vector3(1f, 0f, 0f),
            new Vector3(0f, 0f, 1f), new Vector3(-0.4f, 0.3f, 0.86f),
        })
        {
            var r = WorldSpaceGeometry.RotationBringingUpToY(v);
            var mapped = Vector3.Transform(Vector3.Normalize(v), r);
            Assert.That(Vector3.Dot(mapped, Vector3.UnitY), Is.EqualTo(1f).Within(1e-5f),
                $"{v} must land on +Y");
        }
    }

    [Test]
    public void RotationHandlesAnExactlyInvertedScene()
    {
        // Shortest arc is undefined at 180 degrees. It must still land on +Y, and it must be
        // deterministic - a run that reproduces is worth more than a prettier axis.
        var r = WorldSpaceGeometry.RotationBringingUpToY(-Vector3.UnitY);
        var mapped = Vector3.Transform(-Vector3.UnitY, r);
        Assert.That(Vector3.Dot(mapped, Vector3.UnitY), Is.EqualTo(1f).Within(1e-5f));

        var again = WorldSpaceGeometry.RotationBringingUpToY(-Vector3.UnitY);
        Assert.That(again, Is.EqualTo(r));
    }

    [Test]
    public void RotationIsIdentityWhenAlreadyUpright()
    {
        Assert.That(WorldSpaceGeometry.RotationBringingUpToY(Vector3.UnitY),
            Is.EqualTo(Matrix4x4.Identity),
            "an already-aligned scene must not be nudged - repeated runs would drift");
    }

    /// <summary>
    /// The alignment is a RIGID rotation of the whole reconstruction, so it must not change what
    /// any camera sees. If it did, it would be moving the geometry rather than re-expressing it,
    /// and every score measured before alignment would stop meaning anything.
    /// </summary>
    [Test]
    public void AligningChangesNothingAboutWhatACameraSees()
    {
        var trueUp = Vector3.Normalize(new Vector3(0.3f, 0.8f, 0.2f));
        var cam = Cam(new Vector3(0.1f, -0.2f, 1f), trueUp);
        cam.Position = new Vector3(1.5f, -0.4f, 2.2f);

        var r = WorldSpaceGeometry.RotationBringingUpToY(trueUp);
        var sim = new Similarity3(1f, r, Vector3.Zero);

        var worldPoint = new Vector3(0.7f, 1.1f, 3.4f);
        var before = WorldSpaceGeometry.WorldToCamera(cam, worldPoint);

        sim.ApplyToCamera(cam);
        var after = WorldSpaceGeometry.WorldToCamera(cam, sim.Apply(worldPoint));

        Assert.That(Vector3.Distance(before, after), Is.LessThan(1e-4f),
            "rotating the scene and its cameras together must be invisible to the renderer");
    }
}
