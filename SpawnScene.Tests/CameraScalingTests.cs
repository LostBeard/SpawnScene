using NUnit.Framework;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Resizing a camera with its image.
///
/// Training runs on downscaled copies of the capture - a 13 megapixel phone photo cannot be a
/// float RGB target, let alone 35 of them - so the intrinsics have to be scaled to match. The
/// failure this guards against is scaling the focal length but not the principal point: that
/// looks correct at the image centre and drifts progressively toward the edges, which is
/// exactly the kind of error that still trains and just converges worse.
///
/// So the tests project world points through both cameras and require the pixel coordinates to
/// be related by the resize factor, with probes deliberately spread to the corners.
/// </summary>
public class CameraScalingTests
{
    static CameraParams Phone() => new()
    {
        // A real Bathroom frame: portrait, and a principal point that is NOT centred.
        Width = 3120,
        Height = 4160,
        FocalX = 3180f,
        FocalY = 3172f,
        CenterX = 1548f,
        CenterY = 2095f,
        Near = 0.01f,
        Far = 100f,
        Position = new Vector3(0.3f, 1.2f, -0.7f),
        Forward = Vector3.Normalize(new Vector3(0.21f, -0.13f, -0.97f)),
        Up = Vector3.UnitY,
    };

    static (float U, float V) Project(CameraParams cam, Vector3 p)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var eye);
        var rel = p - eye;
        float z = Vector3.Dot(fwd, rel);
        Assert.That(z, Is.GreaterThan(1e-6f), "probe must be in front of the camera");
        return (cam.FocalX * Vector3.Dot(right, rel) / z + cam.CenterX,
                cam.CenterY - cam.FocalY * Vector3.Dot(up, rel) / z);
    }

    /// <summary>Probes spread to the frame corners, where a principal-point error shows up.</summary>
    static IEnumerable<Vector3> Probes(CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var eye);
        foreach (float r in new[] { -1.4f, -0.5f, 0f, 0.6f, 1.5f })
            foreach (float u in new[] { -1.9f, -0.7f, 0f, 0.8f, 2.0f })
                foreach (float d in new[] { 2f, 5f })
                    yield return eye + right * r + up * u + fwd * d;
    }

    [Test]
    public void ScalingMovesEveryPixelByTheResizeFactor()
    {
        var cam = Phone();
        var small = cam.ScaledTo(780, 1040);      // exactly a quarter
        float sx = 780f / cam.Width, sy = 1040f / cam.Height;

        int n = 0;
        foreach (var p in Probes(cam))
        {
            var (u, v) = Project(cam, p);
            var (u2, v2) = Project(small, p);
            Assert.That(u2, Is.EqualTo(u * sx).Within(1e-2f), $"u for {p}");
            Assert.That(v2, Is.EqualTo(v * sy).Within(1e-2f), $"v for {p}");
            n++;
        }
        Assert.That(n, Is.GreaterThan(40));
    }

    [Test]
    public void ScalingKeepsTheViewUnchanged()
    {
        // Pose must not move. A resize is a change of image sampling, not of where the camera is.
        var cam = Phone();
        var small = cam.ScaledTo(624, 832);
        Assert.That(small.Position, Is.EqualTo(cam.Position));
        Assert.That(small.Forward, Is.EqualTo(cam.Forward));
        Assert.That(small.Up, Is.EqualTo(cam.Up));
        Assert.That(small.Near, Is.EqualTo(cam.Near));
        Assert.That(small.Far, Is.EqualTo(cam.Far));
    }

    [Test]
    public void TheFieldOfViewIsPreserved()
    {
        // 2*atan(W / 2fx) must not change, or the downscaled render sees more or less of the
        // scene than the photograph it is being fitted to.
        var cam = Phone();
        var small = cam.ScaledTo(780, 1040);

        float FovX(CameraParams c) => 2f * MathF.Atan(c.Width / (2f * c.FocalX));
        float FovY(CameraParams c) => 2f * MathF.Atan(c.Height / (2f * c.FocalY));

        Assert.That(FovX(small), Is.EqualTo(FovX(cam)).Within(1e-5f));
        Assert.That(FovY(small), Is.EqualTo(FovY(cam)).Within(1e-5f));
    }

    [Test]
    public void AnOffCentrePrincipalPointScalesToo()
    {
        // The specific bug: scale the focal only, and the principal point stays put. Verify the
        // centre moves proportionally, not that it merely exists.
        var cam = Phone();
        var small = cam.ScaledTo(780, 1040);
        Assert.That(small.CenterX, Is.EqualTo(cam.CenterX * 0.25f).Within(1e-3f));
        Assert.That(small.CenterY, Is.EqualTo(cam.CenterY * 0.25f).Within(1e-3f));
        Assert.That(small.CenterX / small.Width,
            Is.EqualTo(cam.CenterX / (float)cam.Width).Within(1e-6f),
            "the principal point must stay at the same FRACTION of the frame");
    }

    [Test]
    public void FitWithinPreservesAspectAndNeverUpscales()
    {
        var cam = Phone();                                  // 3120x4160, 3:4
        var (w, h) = cam.FitWithin(1024);
        Assert.That(Math.Max(w, h), Is.LessThanOrEqualTo(1024));
        Assert.That(w / (float)h, Is.EqualTo(cam.Width / (float)cam.Height).Within(0.01f));
        Assert.That(w % 2, Is.Zero);
        Assert.That(h % 2, Is.Zero);

        // Already small enough: leave it exactly alone rather than rounding it about.
        var small = cam.ScaledTo(640, 480);
        Assert.That(small.FitWithin(1024), Is.EqualTo((640, 480)));
    }

    [Test]
    public void ScalingComposesWithARotation()
    {
        // Bathroom needs both: the photographs are portrait phone shots at 13 megapixels AND
        // the ring is not orientation-aligned. Doing one then the other must land in the same
        // place as doing them the other way round, or the training target and the render
        // disagree by a transpose.
        var cam = Phone();
        var rotThenScale = ImageOrientation.Rotate(cam, 1);
        rotThenScale = rotThenScale.ScaledTo(rotThenScale.Width / 4, rotThenScale.Height / 4);

        var scaleThenRot = cam.ScaledTo(cam.Width / 4, cam.Height / 4);
        scaleThenRot = ImageOrientation.Rotate(scaleThenRot, 1);

        Assert.That(rotThenScale.Width, Is.EqualTo(scaleThenRot.Width));
        Assert.That(rotThenScale.Height, Is.EqualTo(scaleThenRot.Height));
        Assert.That(rotThenScale.FocalX, Is.EqualTo(scaleThenRot.FocalX).Within(1e-2f));
        Assert.That(rotThenScale.FocalY, Is.EqualTo(scaleThenRot.FocalY).Within(1e-2f));
        // A quarter-turn's principal point uses (W-1), so the two orders differ by the resize
        // factor applied to that single pixel. Anything larger is a real disagreement.
        Assert.That(rotThenScale.CenterX, Is.EqualTo(scaleThenRot.CenterX).Within(1f));
        Assert.That(rotThenScale.CenterY, Is.EqualTo(scaleThenRot.CenterY).Within(1f));
    }

    [Test]
    public void ADegenerateViewportIsRejected()
    {
        var cam = Phone();
        Assert.Throws<ArgumentOutOfRangeException>(() => cam.ScaledTo(0, 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => cam.ScaledTo(100, -1));
    }

    /// <summary>
    /// Training above the IMPORT size (2026-10-02). Cameras are made at the import size (1024 on the longest edge) while
    /// training reloads its targets from the photo, so the photo's own size is the ceiling. FitWithin never upscales: with
    /// it, a 1600 px training setting silently trained TJ's 3120x4160 Bathroom photos at 768x1024.
    /// </summary>
    [Test]
    public void TrainingSize_ScalesUpToThePhotoNotTheImportSize()
    {
        var imported = Phone().ScaledTo(768, 1024);
        Assert.That(imported.TrainingSize(1600, 4160), Is.EqualTo((1200, 1600)), "1600 setting, 4160 px photo");
        Assert.That(imported.TrainingSize(720, 4160), Is.EqualTo((540, 720)), "720 setting shrinks");
        Assert.That(imported.TrainingSize(1024, 4160), Is.EqualTo((768, 1024)), "setting equal to the camera");
        Assert.That(imported.TrainingSize(8000, 4160), Is.EqualTo((3120, 4160)), "never above the photo");
        Assert.That(imported.TrainingSize(1600, 0), Is.EqualTo((768, 1024)), "unknown photo size: the camera is the ceiling");
        Assert.That(imported.TrainingSize(1600, 1024), Is.EqualTo((768, 1024)), "photo no larger than the camera");
    }

    /// <summary>
    /// At the photo's own size, training uses EXACTLY that size (parity, 2026-10-09): the even rounding stretched Mip-NeRF
    /// 360 Kitchen's 1558x1039 photos to 1558x1040, so the target was resampled and every render was a row off the photo
    /// it was scored against. A photo whose aspect is not the camera's still gets the rounded fit.
    /// </summary>
    [Test]
    public void TrainingSize_AtThePhotoSize_IsThePhotoSize()
    {
        Assert.That(CameraParams.TrainingSize(1024, 683, 1600, 1558, 1039), Is.EqualTo((1558, 1039)), "Kitchen");
        Assert.That(CameraParams.TrainingSize(1024, 683, 1600, 1557, 1038), Is.EqualTo((1557, 1038)), "Room");
        Assert.That(CameraParams.TrainingSize(683, 1024, 1600, 1558, 1039), Is.EqualTo((1039, 1558)), "portrait");
        Assert.That(CameraParams.TrainingSize(1024, 683, 1600, 1558), Is.EqualTo((1558, 1040)), "short side unknown: rounded");
        Assert.That(CameraParams.TrainingSize(1024, 683, 1200, 1558, 1039), Is.EqualTo((1200, 800)), "below the photo: rounded");
        Assert.That(CameraParams.TrainingSize(1024, 683, 1600, 1558, 900), Is.EqualTo((1558, 1040)), "other aspect: rounded");
    }
}
