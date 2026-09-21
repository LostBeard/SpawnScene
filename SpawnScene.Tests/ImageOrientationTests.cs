using NUnit.Framework;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// A quarter turn must move the pixels and the camera together.
///
/// The check that matters is not "does the image look upright" - it is that a world point lands
/// on the SAME PIECE OF THE SUBJECT before and after. Turn the pixels without turning the camera
/// and depth is estimated from one picture and unprojected as if it were another; turn the camera
/// without turning the pixels and the same thing happens the other way round. Either way the
/// reconstruction is silently wrong and the only symptom is that it is worse.
///
/// So every test here projects world points through the original camera, projects them through
/// the rotated camera, and requires the two pixel coordinates to be related by exactly the
/// mapping the pixel loop applies.
/// </summary>
public class ImageOrientationTests
{
    /// <summary>Project a world point to pixels through the pinhole the rest of the code uses.</summary>
    static (float U, float V, bool InFront) Project(CameraParams cam, Vector3 p)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var eye);
        var rel = p - eye;
        float x = Vector3.Dot(right, rel);
        float y = Vector3.Dot(up, rel);
        float z = Vector3.Dot(fwd, rel);
        if (z <= 1e-6f) return (0, 0, false);
        return (cam.FocalX * x / z + cam.CenterX, cam.CenterY - cam.FocalY * y / z, true);
    }

    /// <summary>
    /// One of the two TempleRing orientations, built exactly the way
    /// <c>WorldSpaceGeometry.ParseMiddleburyParams</c> builds it from templeR0001's entry.
    ///
    /// Middlebury stores OpenCV rows [right; DOWN; forward], so the parser NEGATES row 1 to get
    /// a y-up camera. Building this fixture with +row1 instead flips the extracted right vector
    /// and makes the detector ask for a turn in the opposite direction - which is how the first
    /// version of this test disagreed with the photograph.
    /// </summary>
    static CameraParams TempleLike() => new()
    {
        Width = 640, Height = 480,
        FocalX = 1520.4f, FocalY = 1525.9f,
        CenterX = 302.32f, CenterY = 246.87f,
        Near = 0.01f, Far = 100f,
        Position = new Vector3(-0.0007f, 0.1233f, 0.5094f),
        Forward = Vector3.Normalize(new Vector3(0.049f, -0.182f, -0.982f)),     // R[2,:]
        Up = Vector3.Normalize(new Vector3(-0.999f, 0.013f, -0.052f)),          // -R[1,:]
    };

    /// <summary>
    /// Independently of any rotation: the fixture must agree with the dataset about which way
    /// is right. Turning templeR0001 one quarter CCW stands the temple up - checked by eye on
    /// the actual pixels - so world-up has to come out pointing image-RIGHT here.
    /// </summary>
    [Test]
    public void TempleFixture_HasWorldUpPointingImageRight()
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(TempleLike().ViewMatrix,
            out var right, out var up, out _, out _);
        Assert.That(Vector3.Dot(right, Vector3.UnitY), Is.GreaterThan(0.95f),
            "image-right should be world-up");
        Assert.That(MathF.Abs(Vector3.Dot(up, Vector3.UnitY)), Is.LessThan(0.1f),
            "image-up should be nearly horizontal in the world");
    }

    static CameraParams Upright()
    {
        var fwd = Vector3.Normalize(new Vector3(0.3f, -0.15f, -0.94f));
        var right = Vector3.Normalize(Vector3.Cross(fwd, Vector3.UnitY));
        var up = Vector3.Normalize(Vector3.Cross(right, fwd));
        return new CameraParams
        {
            Width = 640, Height = 480,
            FocalX = 900f, FocalY = 910f,
            CenterX = 318f, CenterY = 244f,
            Near = 0.01f, Far = 100f,
            Position = new Vector3(0.2f, 0.4f, 1.5f),
            Forward = fwd, Up = up,
        };
    }

    static IEnumerable<Vector3> Probes(CameraParams cam)
    {
        // Points spread across the frame, placed in camera space so they are guaranteed visible.
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var eye);
        foreach (float rr in new[] { -0.12f, -0.04f, 0f, 0.05f, 0.13f })
            foreach (float uu in new[] { -0.09f, -0.03f, 0f, 0.04f, 0.10f })
                foreach (float dd in new[] { 0.6f, 1.1f })
                    yield return eye + right * rr + up * uu + fwd * dd;
    }

    [Test]
    public void TempleRing_IsDetectedAsNeedingOneQuarterTurn()
    {
        Assert.That(ImageOrientation.QuarterTurnsToUpright(TempleLike()), Is.EqualTo(1));
    }

    [Test]
    public void AnUprightCamera_NeedsNoTurn()
    {
        Assert.That(ImageOrientation.QuarterTurnsToUpright(Upright()), Is.EqualTo(0));
    }

    [Test]
    public void ACameraLookingStraightDown_IsLeftAlone()
    {
        // World-up projects to nothing, so its image direction is noise. Turning the picture on
        // the strength of that would be arbitrary and would differ frame to frame in a video.
        var cam = new CameraParams
        {
            Width = 640, Height = 480,
            FocalX = 900f, FocalY = 900f, CenterX = 320f, CenterY = 240f,
            Position = new Vector3(0f, 2f, 0f),
            Forward = -Vector3.UnitY,
            Up = -Vector3.UnitZ,
        };
        Assert.That(ImageOrientation.QuarterTurnsToUpright(cam), Is.EqualTo(0));
    }

    /// <summary>
    /// The load-bearing test: one turn of the camera reproduces exactly the pixel mapping the
    /// image loop applies, (u, v) -> (v, (W-1) - u).
    /// </summary>
    [Test]
    public void OneTurn_MovesPixelsTheSameWayItMovesTheCamera()
    {
        var cam = TempleLike();
        var rot = ImageOrientation.Rotate(cam, 1);

        Assert.That(rot.Width, Is.EqualTo(cam.Height));
        Assert.That(rot.Height, Is.EqualTo(cam.Width));

        int checkedCount = 0;
        foreach (var p in Probes(cam))
        {
            var (u, v, ok) = Project(cam, p);
            Assert.That(ok, Is.True);
            var (u2, v2, ok2) = Project(rot, p);
            Assert.That(ok2, Is.True, "the rotated camera must still see the point");

            Assert.That(u2, Is.EqualTo(v).Within(1e-2f), $"u' should be v for point {p}");
            Assert.That(v2, Is.EqualTo(cam.Width - 1 - u).Within(1e-2f), $"v' should be W-1-u for point {p}");
            checkedCount++;
        }
        Assert.That(checkedCount, Is.GreaterThan(40));
    }

    /// <summary>
    /// EVERY turn count, not just one.
    ///
    /// TempleRing needs 1 quarter turn for templeR0001-0031 and 3 for templeR0034-0046 - the
    /// camera ring flips orientation partway round - so the 3-turn path is not hypothetical,
    /// it is a third of the dataset. And FourTurns_AreTheIdentity cannot cover it: a pixel map
    /// that is consistently wrong in the SAME direction as the camera still composes to the
    /// identity after four, so that test passes for a rotation that turns the picture the
    /// wrong way.
    /// </summary>
    [TestCase(1)]
    [TestCase(2)]
    [TestCase(3)]
    public void EveryTurnCount_MovesPixelsTheSameWayItMovesTheCamera(int turns)
    {
        var cam = TempleLike();
        var rot = ImageOrientation.Rotate(cam, turns);

        Assert.That(rot.Width, Is.EqualTo(turns % 2 == 0 ? cam.Width : cam.Height));
        Assert.That(rot.Height, Is.EqualTo(turns % 2 == 0 ? cam.Height : cam.Width));

        int checkedCount = 0;
        foreach (var p in Probes(cam))
        {
            var (u, v, ok) = Project(cam, p);
            Assert.That(ok, Is.True);

            // Apply the pixel map one turn at a time, exactly as RotateRgba does.
            float pu = u, pv = v;
            int w = cam.Width;
            for (int t = 0; t < turns; t++)
            {
                (pu, pv, w) = (pv, w - 1 - pu, (t % 2 == 0) ? cam.Height : cam.Width);
            }

            var (u2, v2, ok2) = Project(rot, p);
            Assert.That(ok2, Is.True, "the rotated camera must still see the point");
            Assert.That(u2, Is.EqualTo(pu).Within(1e-2f), $"{turns} turn(s), u for {p}");
            Assert.That(v2, Is.EqualTo(pv).Within(1e-2f), $"{turns} turn(s), v for {p}");
            checkedCount++;
        }
        Assert.That(checkedCount, Is.GreaterThan(40));
    }

    /// <summary>
    /// The second orientation in the dataset. templeR0034 onward have world-up pointing image
    /// LEFT, not right, so they need three quarter turns and not one. An implementation that
    /// assumed a single dataset-wide rotation would turn a third of TempleRing upside down.
    /// </summary>
    [Test]
    public void TheOtherHalfOfTheRingNeedsThreeTurns()
    {
        // templeR0040: R[0,:] = (-0.047, -0.997, 0.061), R[1,:] = (-0.992, 0.040, -0.116),
        // R[2,:] = (0.114, -0.066, -0.991). Up is -R[1,:] as the parser builds it.
        var cam = new CameraParams
        {
            Width = 640, Height = 480,
            FocalX = 1520.4f, FocalY = 1525.9f,
            CenterX = 302.32f, CenterY = 246.87f,
            Near = 0.01f, Far = 100f,
            Position = new Vector3(0.24f, 0.10f, -0.56f),
            Forward = Vector3.Normalize(new Vector3(0.114f, -0.066f, -0.991f)),
            Up = Vector3.Normalize(new Vector3(0.992f, -0.040f, 0.116f)),
        };

        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out _, out _, out _);
        Assert.That(Vector3.Dot(right, Vector3.UnitY), Is.LessThan(-0.95f),
            "this half of the ring has image-right pointing world-DOWN");

        Assert.That(ImageOrientation.QuarterTurnsToUpright(cam), Is.EqualTo(3));
        Assert.That(ImageOrientation.QuarterTurnsToUpright(ImageOrientation.Rotate(cam, 3)),
            Is.EqualTo(0), "and three turns must leave it upright");
    }

    [Test]
    public void FourTurns_AreTheIdentity()
    {
        var cam = TempleLike();
        var rot = ImageOrientation.Rotate(cam, 4);

        Assert.That(rot.Width, Is.EqualTo(cam.Width));
        Assert.That(rot.Height, Is.EqualTo(cam.Height));
        Assert.That(rot.FocalX, Is.EqualTo(cam.FocalX).Within(1e-3f));
        Assert.That(rot.CenterX, Is.EqualTo(cam.CenterX).Within(1e-2f));
        Assert.That(rot.CenterY, Is.EqualTo(cam.CenterY).Within(1e-2f));

        foreach (var p in Probes(cam))
        {
            var (u, v, _) = Project(cam, p);
            var (u2, v2, _) = Project(rot, p);
            Assert.That(u2, Is.EqualTo(u).Within(1e-2f));
            Assert.That(v2, Is.EqualTo(v).Within(1e-2f));
        }
    }

    [Test]
    public void RotatingATempleCamera_MakesItUpright()
    {
        var rot = ImageOrientation.Rotate(TempleLike(), 1);
        Assert.That(ImageOrientation.QuarterTurnsToUpright(rot), Is.EqualTo(0));
    }

    /// <summary>
    /// The pixel loop, checked against the same mapping the camera test uses. A distinctive
    /// value per pixel means a transposed or mirrored copy cannot pass.
    /// </summary>
    [Test]
    public void RotateRgba_MovesEachPixelToTheDerivedPlace()
    {
        const int w = 7, h = 5;
        var src = new byte[w * h * 4];
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                int s = (j * w + i) * 4;
                src[s + 0] = (byte)i;
                src[s + 1] = (byte)j;
                src[s + 2] = (byte)(i * 10 + j);
                src[s + 3] = 255;
            }

        var dst = ImageOrientation.RotateRgba(src, w, h, 1);
        int dstW = h;
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                int di = j, dj = w - 1 - i;
                int d = (dj * dstW + di) * 4;
                Assert.That(dst[d + 0], Is.EqualTo((byte)i), $"({i},{j}) red");
                Assert.That(dst[d + 1], Is.EqualTo((byte)j), $"({i},{j}) green");
                Assert.That(dst[d + 2], Is.EqualTo((byte)(i * 10 + j)), $"({i},{j}) blue");
                Assert.That(dst[d + 3], Is.EqualTo((byte)255));
            }
    }

    [Test]
    public void RotateRgba_FourTurnsRestoresTheOriginal()
    {
        const int w = 9, h = 4;
        var rng = new Random(7);
        var src = new byte[w * h * 4];
        rng.NextBytes(src);

        var back = ImageOrientation.RotateRgba(src, w, h, 4);
        Assert.That(back, Is.EqualTo(src));
    }

    [Test]
    public void RotateRgba_ZeroTurnsDoesNotCopy()
    {
        var src = new byte[16];
        Assert.That(ImageOrientation.RotateRgba(src, 2, 2, 0), Is.SameAs(src));
    }
}
