using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for the off-axis WebGPU projection that replaced the symmetric fov/aspect one in
/// <see cref="GpuGaussianRenderer"/>.
///
/// Two things must both hold, and they pull in opposite directions:
///   1. For a centred principal point with fx == fy it must be EXACTLY the old matrix, so
///      swapping it in cannot change any existing scene.
///   2. For real intrinsics it must put a world point on the pixel the pinhole model says,
///      because the splat covariance Jacobian uses fx/fy/cx/cy directly and the two have to
///      agree about where a splat lands.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter CameraProjection</c>
/// </summary>
public class CameraProjectionTests
{
    // Middlebury TempleRing intrinsics: note fx != fy and the principal point is 18px off centre.
    const float Fx = 1520.4f, Fy = 1525.9f;
    const float Cx = 302.32f, Cy = 246.87f;
    const int W = 640, H = 480;
    const float Near = 0.1f, Far = 1000f;

    /// <summary>The symmetric projection this replaced, copied verbatim as the control.</summary>
    static Matrix4x4 LegacySymmetricPerspective(float fovY, float aspect, float near, float far)
    {
        float f = 1.0f / MathF.Tan(fovY * 0.5f);
        float rangeInv = 1.0f / (near - far);
        return new Matrix4x4(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * rangeInv, -1,
            0, 0, near * far * rangeInv, 0);
    }

    /// <summary>Project a world point through view*proj and return the pixel it lands on.</summary>
    static (float px, float py, float ndcZ, float w) ProjectToPixel(
        Matrix4x4 view, Matrix4x4 proj, Vector3 world, int width, int height)
    {
        var mvp = view * proj;
        var clip = Vector4.Transform(new Vector4(world, 1f), mvp);
        float ndcX = clip.X / clip.W;
        float ndcY = clip.Y / clip.W;
        return ((ndcX + 1f) * 0.5f * width, (1f - ndcY) * 0.5f * height, clip.Z / clip.W, clip.W);
    }

    [Test]
    public void CentredIntrinsics_MatchTheSymmetricPerspectiveItReplaces()
    {
        // The equivalence that makes this a safe swap: centred principal point, square pixels.
        float focal = 900f;
        var mine = CameraParams.CreateWebGpuProjection(focal, focal, W / 2f, H / 2f, W, H, Near, Far);

        float fovY = 2f * MathF.Atan(H / (2f * focal));
        var legacy = LegacySymmetricPerspective(fovY, (float)W / H, Near, Far);

        foreach (var (a, b, name) in new[]
        {
            (mine.M11, legacy.M11, "M11"), (mine.M22, legacy.M22, "M22"),
            (mine.M31, legacy.M31, "M31"), (mine.M32, legacy.M32, "M32"),
            (mine.M33, legacy.M33, "M33"), (mine.M34, legacy.M34, "M34"),
            (mine.M43, legacy.M43, "M43"), (mine.M44, legacy.M44, "M44"),
        })
            Assert.That(a, Is.EqualTo(b).Within(1e-4f), $"{name} must match the matrix it replaces");
    }

    [Test]
    public void RealIntrinsics_PutAPointOnThePixelThePinholeModelPredicts()
    {
        // Camera at the origin looking down -Z (eye space == world space).
        var view = Matrix4x4.CreateLookAt(Vector3.Zero, -Vector3.UnitZ, Vector3.UnitY);
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);

        // A point 2m ahead, offset right and up. Pinhole: px = fx*x/z + cx, py = fy*(-y)/z + cy.
        float depth = 2.0f;
        foreach (var (ox, oy) in new[] { (0f, 0f), (0.3f, 0.2f), (-0.25f, 0.1f), (0.1f, -0.4f) })
        {
            var world = new Vector3(ox, oy, -depth);
            var (px, py, _, w) = ProjectToPixel(view, proj, world, W, H);

            float wantPx = Fx * ox / depth + Cx;
            float wantPy = Fy * (-oy) / depth + Cy;

            Assert.That(w, Is.EqualTo(depth).Within(1e-3f), "clip.w must be the eye-space depth");
            Assert.That(px, Is.EqualTo(wantPx).Within(0.01f), $"pixel x for offset ({ox},{oy})");
            Assert.That(py, Is.EqualTo(wantPy).Within(0.01f), $"pixel y for offset ({ox},{oy})");
        }
    }

    [Test]
    public void PrincipalPointOffset_ActuallyShiftsTheImage()
    {
        // Red-check guard: if cx/cy were dropped (the old behaviour), a point on the optical
        // axis would land at the image centre instead of at the principal point.
        var view = Matrix4x4.CreateLookAt(Vector3.Zero, -Vector3.UnitZ, Vector3.UnitY);
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);

        var (px, py, _, _) = ProjectToPixel(view, proj, new Vector3(0, 0, -2f), W, H);

        Assert.That(px, Is.EqualTo(Cx).Within(0.01f), "on-axis point must land on the principal point");
        Assert.That(py, Is.EqualTo(Cy).Within(0.01f), "on-axis point must land on the principal point");

        // And it must be meaningfully away from the naive centre, or the test proves nothing.
        Assert.That(MathF.Abs(px - W / 2f), Is.GreaterThan(10f), "fixture must have an off-centre cx");
        Assert.That(MathF.Abs(py - H / 2f), Is.GreaterThan(5f), "fixture must have an off-centre cy");
    }

    [Test]
    public void NearAndFarPlanes_MapToWebGpuClipDepthZeroAndOne()
    {
        // WebGPU clip space is z in [0,1]. An OpenGL matrix would give -1 at the near plane,
        // which renders as everything clipped or depth-tested wrong.
        var view = Matrix4x4.CreateLookAt(Vector3.Zero, -Vector3.UnitZ, Vector3.UnitY);
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);

        var (_, _, zNear, _) = ProjectToPixel(view, proj, new Vector3(0, 0, -Near), W, H);
        var (_, _, zFar, _) = ProjectToPixel(view, proj, new Vector3(0, 0, -Far), W, H);

        Assert.That(zNear, Is.EqualTo(0f).Within(1e-3f), "near plane → 0");
        Assert.That(zFar, Is.EqualTo(1f).Within(1e-3f), "far plane → 1");
    }

    [Test]
    public void NonSquarePixels_AreNotForcedToASingleFocalLength()
    {
        // fx != fy by 0.36% on TempleRing. Small, but a symmetric projection cannot express it
        // at all, and the covariance Jacobian uses both separately.
        var view = Matrix4x4.CreateLookAt(Vector3.Zero, -Vector3.UnitZ, Vector3.UnitY);
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);

        float depth = 2f, off = 0.3f;
        var (pxRight, _, _, _) = ProjectToPixel(view, proj, new Vector3(off, 0, -depth), W, H);
        var (_, pyUp, _, _) = ProjectToPixel(view, proj, new Vector3(0, off, -depth), W, H);

        float dx = pxRight - Cx;
        float dy = Cy - pyUp;
        Assert.That(dx / dy, Is.EqualTo(Fx / Fy).Within(1e-4f),
            "horizontal/vertical pixel scale ratio must equal fx/fy");
    }

    [Test]
    public void ExtractIntrinsics_RoundTripsWhatCreateWebGpuProjectionWasGiven()
    {
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);
        CameraParams.ExtractIntrinsics(proj, W, H, out var fx, out var fy, out var cx, out var cy);

        Assert.That(fx, Is.EqualTo(Fx).Within(0.01f), "fx");
        Assert.That(fy, Is.EqualTo(Fy).Within(0.01f), "fy");
        Assert.That(cx, Is.EqualTo(Cx).Within(0.01f), "cx");
        Assert.That(cy, Is.EqualTo(Cy).Within(0.01f), "cy");
    }

    [Test]
    public void ExtractIntrinsics_RecoversTheOffCentreFrustumOfAVrEye()
    {
        // A real VR eye frustum is asymmetric: the principal point is pushed off centre so the
        // two eyes converge. Build one the way WebXR would (explicit l/r/b/t) and confirm we
        // recover its true principal point rather than assuming the image centre.
        const int eyeW = 1832, eyeH = 1920;
        float near = 0.1f, far = 1000f;
        // Left eye of a typical headset: more frustum on the outboard (left) side.
        float l = -0.0964f, r = 0.0762f, b = -0.0892f, t = 0.0892f;

        // Row-vector WebGPU off-axis frustum.
        var proj = new Matrix4x4(
            2 * near / (r - l), 0, 0, 0,
            0, 2 * near / (t - b), 0, 0,
            (r + l) / (r - l), (t + b) / (t - b), far / (near - far), -1,
            0, 0, near * far / (near - far), 0);

        CameraParams.ExtractIntrinsics(proj, eyeW, eyeH, out var fx, out var fy, out var cx, out var cy);

        // Ground truth from the frustum bounds themselves.
        float wantFx = near * eyeW / (r - l);
        float wantFy = near * eyeH / (t - b);
        float wantCx = -l * eyeW / (r - l);
        float wantCy = t * eyeH / (t - b);

        Assert.That(fx, Is.EqualTo(wantFx).Within(0.5f), "fx");
        Assert.That(fy, Is.EqualTo(wantFy).Within(0.5f), "fy");
        Assert.That(cx, Is.EqualTo(wantCx).Within(0.5f), "cx");
        Assert.That(cy, Is.EqualTo(wantCy).Within(0.5f), "cy");

        // The whole point: this eye's principal point is NOT the image centre. If it were, the
        // old |M11|*w/2 shortcut would have been fine and this test would prove nothing.
        Assert.That(MathF.Abs(cx - eyeW / 2f), Is.GreaterThan(50f),
            "fixture must be a genuinely asymmetric frustum");
    }

    [Test]
    public void TheProjectionAgreesWithTheCovarianceJacobiansPinholeModel()
    {
        // The whole point of this change: SplatCovariance.ProjectCov2D builds its Jacobian from
        // (fx, fy) about a camera-space centre, and the projection matrix decides where that
        // centre is drawn. If they disagree, a splat's ellipse is computed for one place and
        // rasterised at another. Check they agree on the same point.
        var eye = new Vector3(0.2f, 0.1f, 0.8f);
        var target = new Vector3(0.028f, 0.042f, -0.054f);
        var view = Matrix4x4.CreateLookAt(eye, target, Vector3.UnitY);
        var proj = CameraParams.CreateWebGpuProjection(Fx, Fy, Cx, Cy, W, H, Near, Far);

        WorldSpaceGeometry.ViewMatrixToCameraBasis(view, out var right, out var up, out var fwd, out var pos);

        var world = new Vector3(0.05f, 0.07f, -0.02f);
        var rel = world - pos;
        float camX = Vector3.Dot(right, rel);
        float camY = Vector3.Dot(up, rel);
        float camZ = Vector3.Dot(fwd, rel);

        // Where the shader's pinhole model says the centre is, in pixels from the principal point.
        float shaderPx = Fx * camX / camZ;
        float shaderPy = Fy * camY / camZ;

        // Where the projection matrix actually draws it.
        var (px, py, _, w) = ProjectToPixel(view, proj, world, W, H);

        Assert.That(w, Is.EqualTo(camZ).Within(1e-4f), "depth must agree");
        Assert.That(px - Cx, Is.EqualTo(shaderPx).Within(0.02f), "x offset from principal point");
        Assert.That(Cy - py, Is.EqualTo(shaderPy).Within(0.02f), "y offset from principal point (y up)");
    }
}
