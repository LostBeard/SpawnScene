using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for the 3D covariance / EWA projection shared by the WGSL vertex shaders and the
/// ILGPU unproject kernels.
///
/// Every assertion here is against an ANALYTIC answer, never against another implementation:
/// a surface disk turned 60 degrees away from the camera must foreshorten by exactly cos 60,
/// and projecting the same splat through a rotated camera must give the same ellipse. The
/// second one is what catches a transposed rotation - with an identity camera basis a
/// transpose is invisible, which is how a wrong Sigma survives a "looks fine" screenshot.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter SplatCovariance</c>
/// </summary>
public class SplatCovarianceTests
{
    const float Fx = 1520f, Fy = 1520f;   // Middlebury TempleRing intrinsics
    const float Depth = 0.6f;             // metres, typical temple standoff
    const float Radius = 3.6e-4f;         // measured TempleRing splat sigma (scale_probe meanSX)
    const float Flat = Radius * 0.15f;    // flattened surface axis

    static SplatCovariance.Cov2 ProjectDisk(
        Vector3 normal, Vector3 camPos,
        Vector3 right, Vector3 up, Vector3 forward)
    {
        var q = SplatCovariance.QuatFromNormal(normal.X, normal.Y, normal.Z);
        var cov3 = SplatCovariance.Cov3DFromScaleQuat(Radius, Radius, Flat, q);
        var camCov = SplatCovariance.RotateToCamera(cov3,
            right.X, right.Y, right.Z,
            up.X, up.Y, up.Z,
            forward.X, forward.Y, forward.Z);
        return SplatCovariance.ProjectCov2D(camCov, camPos.X, camPos.Y, camPos.Z, Fx, Fy);
    }

    // ── Sigma_2D has no EWA term in the analytic answer; strip it before comparing. ──
    static (float sx, float sy) SigmasPx(SplatCovariance.Cov2 c)
        => (MathF.Sqrt(MathF.Max(c.A - SplatCovariance.EwaFilterPx2, 0f)),
            MathF.Sqrt(MathF.Max(c.C - SplatCovariance.EwaFilterPx2, 0f)));

    [Test]
    public void QuatFromNormal_RotatesLocalZOntoTheNormal()
    {
        var normals = new[]
        {
            new Vector3(0, 0, 1),
            new Vector3(0, 0, -1),
            Vector3.Normalize(new Vector3(0.866f, 0f, -0.5f)),
            Vector3.Normalize(new Vector3(-0.3f, 0.7f, -0.64f)),
            Vector3.Normalize(new Vector3(1f, 1f, 1f)),
        };

        foreach (var n in normals)
        {
            var q = SplatCovariance.QuatFromNormal(n.X, n.Y, n.Z);
            var sq = new Quaternion(q.X, q.Y, q.Z, q.W);

            Assert.That(sq.Length(), Is.EqualTo(1f).Within(1e-5f), $"quat not unit for {n}");

            var rotated = Vector3.Transform(new Vector3(0, 0, 1), sq);
            Assert.That(rotated.X, Is.EqualTo(n.X).Within(1e-5f), $"x for {n}");
            Assert.That(rotated.Y, Is.EqualTo(n.Y).Within(1e-5f), $"y for {n}");
            Assert.That(rotated.Z, Is.EqualTo(n.Z).Within(1e-5f), $"z for {n}");
        }
    }

    [Test]
    public void FrontoParallelDisk_ProjectsToAnIsotropicEllipseOfRadiusRTimesFOverZ()
    {
        // Camera at the origin looking down +Z, disk straight ahead facing back at it.
        var cov2 = ProjectDisk(
            normal: new Vector3(0, 0, -1),
            camPos: new Vector3(0, 0, Depth),
            right: new Vector3(1, 0, 0), up: new Vector3(0, 1, 0), forward: new Vector3(0, 0, 1));

        var (sx, sy) = SigmasPx(cov2);
        float expected = Radius * Fx / Depth;

        Assert.That(sx, Is.EqualTo(expected).Within(expected * 1e-3f), "x sigma");
        Assert.That(sy, Is.EqualTo(expected).Within(expected * 1e-3f), "y sigma");
        Assert.That(cov2.B, Is.EqualTo(0f).Within(expected * expected * 1e-3f), "no tilt");
    }

    [Test]
    public void DiskTurned60Degrees_ForeshortensByExactlyCos60()
    {
        // Rotate the disk about the camera's vertical axis. Its horizontal extent must shrink by
        // cos(theta); its vertical extent must not move at all.
        foreach (float deg in new[] { 0f, 30f, 45f, 60f, 75f })
        {
            float rad = deg * MathF.PI / 180f;
            var normal = new Vector3(MathF.Sin(rad), 0f, -MathF.Cos(rad));

            var cov2 = ProjectDisk(normal, new Vector3(0, 0, Depth),
                new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1));

            var (sx, sy) = SigmasPx(cov2);
            float full = Radius * Fx / Depth;

            // The flattened axis contributes Flat*sin(theta) in quadrature with Radius*cos(theta).
            float cos = MathF.Cos(rad), sin = MathF.Sin(rad);
            float expectedX = full * MathF.Sqrt(cos * cos + (Flat / Radius) * (Flat / Radius) * sin * sin);

            Assert.That(sx, Is.EqualTo(expectedX).Within(full * 2e-3f), $"x sigma at {deg} deg");
            Assert.That(sy, Is.EqualTo(full).Within(full * 2e-3f), $"y sigma at {deg} deg");
        }
    }

    [Test]
    public void RotatingCameraAndSceneTogether_LeavesTheEllipseUnchanged()
    {
        // Equivariance. Sigma_2D depends only on the splat's pose RELATIVE to the camera, so
        // turning both by the same rotation must reproduce the same pixels. A transposed
        // RotateToCamera passes the fronto-parallel case and fails here.
        var normal = Vector3.Normalize(new Vector3(0.5f, -0.3f, -0.81f));
        var camPt = new Vector3(0.08f, -0.05f, Depth);   // off-centre, so J's third column matters

        var baseline = ProjectDisk(normal, camPt,
            new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1));

        // An arbitrary, deliberately asymmetric world rotation.
        var worldRot = Quaternion.CreateFromYawPitchRoll(0.7f, -0.4f, 1.1f);
        var right = Vector3.Transform(new Vector3(1, 0, 0), worldRot);
        var up = Vector3.Transform(new Vector3(0, 1, 0), worldRot);
        var fwd = Vector3.Transform(new Vector3(0, 0, 1), worldRot);

        var rotatedNormal = Vector3.Transform(normal, worldRot);
        var rotated = ProjectDisk(rotatedNormal, camPt, right, up, fwd);

        Assert.That(rotated.A, Is.EqualTo(baseline.A).Within(MathF.Abs(baseline.A) * 1e-3f + 1e-6f), "xx");
        Assert.That(rotated.B, Is.EqualTo(baseline.B).Within(MathF.Abs(baseline.A) * 1e-3f + 1e-6f), "xy");
        Assert.That(rotated.C, Is.EqualTo(baseline.C).Within(MathF.Abs(baseline.C) * 1e-3f + 1e-6f), "yy");
    }

    [Test]
    public void TurnedDiskProducesARealTilt_NotJustAnAxisAlignedShrink()
    {
        // Turn the disk about an axis 45 degrees between right and up: the ellipse must tilt,
        // i.e. the off-diagonal term must be clearly non-zero. This is the term an axis-aligned
        // billboard renderer cannot represent at all.
        float rad = 60f * MathF.PI / 180f;
        var tiltAxis = Vector3.Normalize(new Vector3(1, 1, 0));
        var normal = Vector3.Transform(new Vector3(0, 0, -1),
            Quaternion.CreateFromAxisAngle(tiltAxis, rad));

        var cov2 = ProjectDisk(normal, new Vector3(0, 0, Depth),
            new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1));

        float scale = Radius * Fx / Depth;
        Assert.That(MathF.Abs(cov2.B), Is.GreaterThan(0.15f * scale * scale),
            "a turned disk must produce an off-diagonal covariance term");

        var e = SplatCovariance.EigenAxes(cov2, 3f);
        Assert.That(e.Valid, Is.True);

        float major = MathF.Sqrt(e.Ax * e.Ax + e.Ay * e.Ay);
        float minor = MathF.Sqrt(e.Bx * e.Bx + e.By * e.By);
        Assert.That(major, Is.GreaterThan(minor), "major axis must be the longer one");
        Assert.That(e.Ax * e.Bx + e.Ay * e.By, Is.EqualTo(0f).Within(major * minor * 1e-4f),
            "axes must be orthogonal");

        // The long axis lies along the tilt axis (the direction that does NOT foreshorten),
        // which is the (1,1) diagonal in screen pixels.
        float cosToDiagonal = MathF.Abs(e.Ax + e.Ay) / (major * MathF.Sqrt(2f));
        Assert.That(cosToDiagonal, Is.EqualTo(1f).Within(1e-3f), "major axis should follow the tilt axis");
    }

    [Test]
    public void EigenAxes_OnADiagonalCovariance_ReturnsTheSquareRootsScaledByTheCutoff()
    {
        var c = new SplatCovariance.Cov2 { A = 16f, B = 0f, C = 4f };
        var e = SplatCovariance.EigenAxes(c, 3f);

        Assert.That(e.Valid, Is.True);
        float major = MathF.Sqrt(e.Ax * e.Ax + e.Ay * e.Ay);
        float minor = MathF.Sqrt(e.Bx * e.Bx + e.By * e.By);

        Assert.That(major, Is.EqualTo(12f).Within(1e-4f), "3 sigma of variance 16");
        Assert.That(minor, Is.EqualTo(6f).Within(1e-4f), "3 sigma of variance 4");
        Assert.That(MathF.Abs(e.Ax), Is.EqualTo(12f).Within(1e-4f), "major should be the x axis");
    }

    [Test]
    public void EigenAxes_RejectsADegenerateCovariance()
    {
        Assert.That(SplatCovariance.EigenAxes(new SplatCovariance.Cov2 { A = 0f, B = 0f, C = 0f }, 3f).Valid,
            Is.False);
        Assert.That(SplatCovariance.EigenAxes(new SplatCovariance.Cov2 { A = 4f, B = 4f, C = 4f }, 3f).Valid,
            Is.False, "singular (det = 0) must be rejected, not square-rooted");
    }

    [Test]
    public void NormalFromNeighbors_OnAFrontoParallelPatch_FacesTheCamera()
    {
        // Three camera-space points on a plane at constant z. OpenCV axes: x right, y DOWN, z fwd.
        var q = SplatCovariance.NormalQuatFromNeighbors(
            0f, 0f, Depth,
            0.01f, 0f, Depth,
            0f, 0.01f, Depth);

        var n = Vector3.Transform(new Vector3(0, 0, 1), new Quaternion(q.X, q.Y, q.Z, q.W));
        Assert.That(n.X, Is.EqualTo(0f).Within(1e-5f));
        Assert.That(n.Y, Is.EqualTo(0f).Within(1e-5f));
        Assert.That(n.Z, Is.EqualTo(-1f).Within(1e-5f), "must point back at the camera");
    }

    [Test]
    public void NormalFromNeighbors_OnATiltedPlane_MatchesThePlaneNormal()
    {
        // Plane through the origin-ish with true normal (sin30, 0, -cos30): z increases with x.
        float rad = 30f * MathF.PI / 180f;
        var expected = new Vector3(MathF.Sin(rad), 0f, -MathF.Cos(rad));

        // Points on that plane: z = Depth + x * tan(30)
        float step = 0.01f;
        float dz = step * MathF.Tan(rad);

        var q = SplatCovariance.NormalQuatFromNeighbors(
            0f, 0f, Depth,
            step, 0f, Depth + dz,
            0f, step, Depth);

        var n = Vector3.Transform(new Vector3(0, 0, 1), new Quaternion(q.X, q.Y, q.Z, q.W));
        Assert.That(n.X, Is.EqualTo(expected.X).Within(1e-4f), "x");
        Assert.That(n.Y, Is.EqualTo(expected.Y).Within(1e-4f), "y");
        Assert.That(n.Z, Is.EqualTo(expected.Z).Within(1e-4f), "z");
    }

    [Test]
    public void RotateQuatToWorld_TakesTheCameraSpaceNormalToTheWorldNormal()
    {
        // Camera-to-world columns are (right, down, forward) in world space.
        var worldRot = Quaternion.CreateFromYawPitchRoll(0.9f, 0.25f, -0.6f);
        var right = Vector3.Transform(new Vector3(1, 0, 0), worldRot);
        var down = Vector3.Transform(new Vector3(0, 1, 0), worldRot);
        var fwd = Vector3.Transform(new Vector3(0, 0, 1), worldRot);

        var camNormal = Vector3.Normalize(new Vector3(0.2f, -0.4f, -0.89f));
        var camQuat = SplatCovariance.QuatFromNormal(camNormal.X, camNormal.Y, camNormal.Z);

        // Arguments are the world-to-camera ROWS, exactly as SplatWorldParams stores them:
        // row0 = right, row1 = down, row2 = forward.
        var worldQuat = SplatCovariance.RotateQuatToWorld(camQuat,
            right.X, right.Y, right.Z,
            down.X, down.Y, down.Z,
            fwd.X, fwd.Y, fwd.Z);

        var got = Vector3.Transform(new Vector3(0, 0, 1),
            new Quaternion(worldQuat.X, worldQuat.Y, worldQuat.Z, worldQuat.W));
        var want = camNormal.X * right + camNormal.Y * down + camNormal.Z * fwd;

        Assert.That(got.X, Is.EqualTo(want.X).Within(1e-4f), "x");
        Assert.That(got.Y, Is.EqualTo(want.Y).Within(1e-4f), "y");
        Assert.That(got.Z, Is.EqualTo(want.Z).Within(1e-4f), "z");
    }

    [Test]
    public void ViewMatrixToCameraBasis_RecoversWhatCreateLookAtWasGiven()
    {
        var eye = new Vector3(0.31f, -0.22f, 1.4f);
        var target = new Vector3(0.028f, 0.042f, -0.054f);   // TempleRing lookAt
        var worldUp = new Vector3(0.1f, 1f, 0.05f);

        var view = Matrix4x4.CreateLookAt(eye, target, worldUp);
        WorldSpaceGeometry.ViewMatrixToCameraBasis(view, out var right, out var up, out var fwd, out var pos);

        var wantFwd = Vector3.Normalize(target - eye);
        var wantRight = Vector3.Normalize(Vector3.Cross(wantFwd, Vector3.Normalize(worldUp)));
        var wantUp = Vector3.Cross(wantRight, wantFwd);

        Assert.That(Vector3.Distance(pos, eye), Is.LessThan(1e-4f), $"position, got {pos}");
        Assert.That(Vector3.Dot(fwd, wantFwd), Is.EqualTo(1f).Within(1e-4f), $"forward, got {fwd}");
        Assert.That(Vector3.Dot(right, wantRight), Is.EqualTo(1f).Within(1e-4f), $"right, got {right}");
        Assert.That(Vector3.Dot(up, wantUp), Is.EqualTo(1f).Within(1e-4f), $"up, got {up}");

        // A .NET camera is x-right, y-up, z-BACKWARD, so (right, up, forward) is deliberately
        // left-handed. RotateToCamera only needs an invertible change of basis, not a rotation,
        // but pin the sign: flipping it silently mirrors every splat ellipse.
        Assert.That(Vector3.Dot(Vector3.Cross(right, up), fwd), Is.EqualTo(-1f).Within(1e-4f), "handedness");

        foreach (var (a, b, name) in new[] { (right, up, "right.up"), (right, fwd, "right.fwd"), (up, fwd, "up.fwd") })
            Assert.That(Vector3.Dot(a, b), Is.EqualTo(0f).Within(1e-4f), $"{name} must be orthogonal");
    }

    [Test]
    public void ViewMatrixToCameraBasis_AgreesWithTheProjectionItFeeds()
    {
        // The basis must reproduce the same camera-space point the view matrix itself produces,
        // or the covariance is projected at a different place than the splat centre is drawn.
        var view = Matrix4x4.CreateLookAt(
            new Vector3(-0.5f, 0.8f, 0.9f), new Vector3(0.03f, 0.04f, -0.05f), Vector3.UnitY);
        WorldSpaceGeometry.ViewMatrixToCameraBasis(view, out var right, out var up, out var fwd, out var pos);

        var world = new Vector3(0.11f, -0.07f, 0.22f);
        var viaMatrix = Vector3.Transform(world, view);   // eye space: looks down -Z
        var d = world - pos;
        var viaBasis = new Vector3(Vector3.Dot(right, d), Vector3.Dot(up, d), Vector3.Dot(fwd, d));

        Assert.That(viaBasis.X, Is.EqualTo(viaMatrix.X).Within(1e-4f), "x");
        Assert.That(viaBasis.Y, Is.EqualTo(viaMatrix.Y).Within(1e-4f), "y");
        Assert.That(viaBasis.Z, Is.EqualTo(-viaMatrix.Z).Within(1e-4f), "depth (view is -Z forward)");
    }

    [Test]
    public void SubPixelSplat_IsWidenedToTheEwaFloorInsteadOfVanishing()
    {
        // A splat far enough away to be well under a pixel. Without the EWA prefilter it would
        // be a flickering sub-pixel dot; with it the ellipse bottoms out near sqrt(0.3) px.
        var cov2 = ProjectDisk(new Vector3(0, 0, -1), new Vector3(0, 0, 500f),
            new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1));

        var e = SplatCovariance.EigenAxes(cov2, 3f);
        Assert.That(e.Valid, Is.True, "sub-pixel splats must still project");

        float minorSigma = MathF.Sqrt(e.Bx * e.Bx + e.By * e.By) / 3f;
        Assert.That(minorSigma, Is.EqualTo(MathF.Sqrt(SplatCovariance.EwaFilterPx2)).Within(0.02f),
            "EWA floor should dominate a sub-pixel splat");
    }
}
