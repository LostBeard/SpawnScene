using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Every geometric gradient against a central finite difference of the forward projection.
///
/// This is the only honest way to check this algebra. Re-deriving it a second time and comparing
/// the two derivations just tests whether the same mistake was made twice - and the mistakes
/// available here (a dropped transpose, a missing factor of two on a symmetric off-diagonal, the
/// perspective term of the Jacobian's own dependence on position) all produce gradients that
/// still point roughly downhill, so training would still improve and quietly converge worse.
///
/// The loss is a fixed linear functional of the five projected outputs, so the analytic answer
/// is exactly <c>Backward</c> with those weights as the upstream gradient.
/// </summary>
public class SplatGeometryGradientsTests
{
    static SplatGeometryGradients.View Camera(float eyeX, float eyeY, float eyeZ)
    {
        // A deliberately asymmetric basis. An axis-aligned camera makes a transposed rotation
        // indistinguishable from the correct one.
        float yaw = 0.37f, pitch = -0.21f;
        float cy = MathF.Cos(yaw), sy = MathF.Sin(yaw);
        float cp = MathF.Cos(pitch), sp = MathF.Sin(pitch);

        // forward, then right = normalize(cross(forward, worldUp)), up = cross(right, forward)
        float fx = sy * cp, fy = sp, fz = -cy * cp;
        float rx = -cy, ry = 0f, rz = -sy;
        float rl = MathF.Sqrt(rx * rx + ry * ry + rz * rz);
        rx /= rl; ry /= rl; rz /= rl;
        float ux = ry * fz - rz * fy;
        float uy = rz * fx - rx * fz;
        float uz = rx * fy - ry * fx;

        return new SplatGeometryGradients.View
        {
            EyeX = eyeX, EyeY = eyeY, EyeZ = eyeZ,
            Rx = rx, Ry = ry, Rz = rz,
            Ux = ux, Uy = uy, Uz = uz,
            Fx3 = fx, Fy3 = fy, Fz3 = fz,
            FocalX = 1520.4f, FocalY = 1525.9f,
            CenterX = 302.32f, CenterY = 246.87f,
            LimX = SplatCovariance.JacobianClampLimit(2 * 302.32f, 1520.4f),
            LimY = SplatCovariance.JacobianClampLimit(2 * 246.87f, 1525.9f),
        };
    }

    // Three weightings, and every fixture is checked under all three.
    //
    // One mixed weighting is not enough. The screen-centre gradient is ~1e3 while the covariance
    // gradients are ~1e-1, so any absolute tolerance wide enough for the first swallows a
    // complete failure of the second - which is exactly how a doubly-halved off-diagonal in the
    // Sigma_world chain survived the first version of this test on the quaternion.
    static readonly SplatGeometryGradients.UpstreamGrad Mixed = new()
    {
        // Nothing zero and nothing equal: a zero weight hides whichever path only that output
        // exercises, and equal weights let two swapped terms cancel.
        ScreenX = 0.731f, ScreenY = -0.417f,
        ConicA = 0.239f, ConicB = -0.583f, ConicC = 0.912f,
    };

    /// <summary>Only the conic. Isolates the whole covariance chain at its own scale.</summary>
    static readonly SplatGeometryGradients.UpstreamGrad ConicOnly = new()
    {
        ScreenX = 0f, ScreenY = 0f,
        ConicA = 0.239f, ConicB = -0.583f, ConicC = 0.912f,
    };

    /// <summary>Only the screen centre. Isolates the mean path and A^T.</summary>
    static readonly SplatGeometryGradients.UpstreamGrad ScreenOnly = new()
    {
        ScreenX = 0.731f, ScreenY = -0.417f,
        ConicA = 0f, ConicB = 0f, ConicC = 0f,
    };

    static double Loss(SplatGeometryGradients.Geometry g, SplatGeometryGradients.View v,
                       SplatGeometryGradients.UpstreamGrad Weights)
    {
        var p = SplatGeometryGradients.Project(g, v);
        if (!p.Valid) return double.NaN;
        // DOUBLE, deliberately. float*float is float arithmetic in C#, and the screen terms are
        // ~1e3 while a scale perturbation moves the conic by ~1e-6 - the whole signal lands
        // below float epsilon and every scale gradient reads as exactly zero.
        return (double)Weights.ScreenX * p.ScreenX + (double)Weights.ScreenY * p.ScreenY
             + (double)Weights.ConicA * p.ConicA + (double)Weights.ConicB * p.ConicB
             + (double)Weights.ConicC * p.ConicC;
    }

    delegate void Setter(ref SplatGeometryGradients.Geometry g, float value);

    static readonly (string Name, Func<SplatGeometryGradients.Geometry, float> Get,
                     Setter Set, Func<SplatGeometryGradients.Grad, float> Grad)[] Params =
    {
        ("PosX", g => g.PosX, (ref SplatGeometryGradients.Geometry g, float x) => g.PosX = x, d => d.PosX),
        ("PosY", g => g.PosY, (ref SplatGeometryGradients.Geometry g, float x) => g.PosY = x, d => d.PosY),
        ("PosZ", g => g.PosZ, (ref SplatGeometryGradients.Geometry g, float x) => g.PosZ = x, d => d.PosZ),
        ("ScaleX", g => g.ScaleX, (ref SplatGeometryGradients.Geometry g, float x) => g.ScaleX = x, d => d.ScaleX),
        ("ScaleY", g => g.ScaleY, (ref SplatGeometryGradients.Geometry g, float x) => g.ScaleY = x, d => d.ScaleY),
        ("ScaleZ", g => g.ScaleZ, (ref SplatGeometryGradients.Geometry g, float x) => g.ScaleZ = x, d => d.ScaleZ),
        ("QuatX", g => g.QuatX, (ref SplatGeometryGradients.Geometry g, float x) => g.QuatX = x, d => d.QuatX),
        ("QuatY", g => g.QuatY, (ref SplatGeometryGradients.Geometry g, float x) => g.QuatY = x, d => d.QuatY),
        ("QuatZ", g => g.QuatZ, (ref SplatGeometryGradients.Geometry g, float x) => g.QuatZ = x, d => d.QuatZ),
        ("QuatW", g => g.QuatW, (ref SplatGeometryGradients.Geometry g, float x) => g.QuatW = x, d => d.QuatW),
    };

    /// <summary>
    /// Compare every analytic gradient to a central difference. The step is scaled to the
    /// parameter so a position in metres and a quaternion component near 1 are both probed
    /// sensibly, and the tolerance is relative to the larger of the two magnitudes because the
    /// forward is evaluated in single precision.
    /// </summary>
    static void CheckAll(SplatGeometryGradients.Geometry g, SplatGeometryGradients.View v,
                         string label, double relTol = 0.02)
    {
        var fwd = SplatGeometryGradients.Project(g, v);
        Assert.That(fwd.Valid, Is.True, $"{label}: fixture does not project");

        CheckWith(g, v, ConicOnly, $"{label}/conic", relTol);
        CheckWith(g, v, ScreenOnly, $"{label}/screen", relTol);
        CheckWith(g, v, Mixed, $"{label}/mixed", relTol);
    }

    static void CheckWith(SplatGeometryGradients.Geometry g, SplatGeometryGradients.View v,
                          SplatGeometryGradients.UpstreamGrad weights, string label, double relTol)
    {
        var analytic = SplatGeometryGradients.Backward(g, v, weights);

        // The absolute floor is set from the largest gradient THIS weighting produces, so a
        // small gradient is still held to a tight bound when nothing in the fixture is large.
        double biggest = 0;
        foreach (var (_, _, _, grad) in Params) biggest = Math.Max(biggest, Math.Abs(grad(analytic)));
        // 1e-3 of the largest gradient in this weighting. Single-precision finite differences
        // on a conic built from a near-cancelling determinant are good to a few percent at best,
        // and holding a gradient three orders of magnitude below the fixture's largest to a 2%
        // relative bound just measures that noise. Red-checked: the doubly-halved off-diagonal
        // this test was written to catch produces 20-30% errors, far above this floor.
        double floor = Math.Max(1e-3 * biggest, 1e-9);

        foreach (var (name, get, set, grad) in Params)
        {
            float p0 = get(g);
            float step = 1e-3f * MathF.Max(MathF.Abs(p0), 1e-2f);

            var gp = g; set(ref gp, p0 + step);
            var gm = g; set(ref gm, p0 - step);
            double lp = Loss(gp, v, weights), lm = Loss(gm, v, weights);
            Assert.That(double.IsNaN(lp) || double.IsNaN(lm), Is.False,
                $"{label}/{name}: perturbed fixture stopped projecting");

            double numeric = (lp - lm) / (2.0 * step);
            double a = grad(analytic);
            double scale = Math.Max(Math.Abs(numeric), Math.Abs(a));
            double tol = Math.Max(relTol * scale, floor);

            Assert.That(a, Is.EqualTo(numeric).Within(tol),
                $"{label}/{name}: analytic {a:G6} vs numeric {numeric:G6}");
        }
    }

    /// <summary>
    /// Place a splat at a camera-space offset, so "on axis" means on axis by construction. The
    /// first version of this test hand-picked a world position and landed 1180 px off centre,
    /// which made the on-axis and off-axis cases the same test twice.
    /// </summary>
    static SplatGeometryGradients.Geometry AtCamera(
        SplatGeometryGradients.View v, float right, float up, float depth,
        float sx, float sy, float sz, float qx, float qy, float qz, float qw) => new()
        {
            PosX = v.EyeX + v.Rx * right + v.Ux * up + v.Fx3 * depth,
            PosY = v.EyeY + v.Ry * right + v.Uy * up + v.Fy3 * depth,
            PosZ = v.EyeZ + v.Rz * right + v.Uz * up + v.Fz3 * depth,
            ScaleX = sx, ScaleY = sy, ScaleZ = sz,
            QuatX = qx, QuatY = qy, QuatZ = qz, QuatW = qw,
        };

    static SplatGeometryGradients.Geometry Splat(
        float px, float py, float pz, float sx, float sy, float sz,
        float qx, float qy, float qz, float qw) => new()
        {
            PosX = px, PosY = py, PosZ = pz,
            ScaleX = sx, ScaleY = sy, ScaleZ = sz,
            QuatX = qx, QuatY = qy, QuatZ = qz, QuatW = qw,
        };

    [Test]
    public void Gradients_MatchFiniteDifference_NearOpticalAxis()
    {
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = AtCamera(v, 0.002f, -0.003f, 1.2f, 0.020f, 0.014f, 0.006f, 0.13f, -0.24f, 0.08f, 0.96f);
        var p = SplatGeometryGradients.Project(g, v);
        Assert.That(MathF.Abs(p.ScreenX - v.CenterX) + MathF.Abs(p.ScreenY - v.CenterY),
            Is.LessThan(20f), "fixture is not actually on-axis");
        CheckAll(g, v, "on-axis");
    }

    [Test]
    public void Gradients_MatchFiniteDifference_FarOffAxis()
    {
        // The perspective terms of the Jacobian scale with tx/tz and ty/tz. On the optical axis
        // they vanish, so a fixture near the centre cannot tell a correct implementation from one
        // that drops them entirely. This splat sits well out toward a frame corner.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = AtCamera(v, 0.20f, 0.15f, 1.2f, 0.020f, 0.014f, 0.006f, 0.13f, -0.24f, 0.08f, 0.96f);
        var p = SplatGeometryGradients.Project(g, v);
        Assert.That(p.Valid, Is.True);
        Assert.That(MathF.Abs(p.ScreenX - v.CenterX) + MathF.Abs(p.ScreenY - v.CenterY),
            Is.GreaterThan(100f), "fixture is not actually off-axis");
        CheckAll(g, v, "off-axis");
    }

    [Test]
    public void Gradients_MatchFiniteDifference_BeyondTheJacobianClamp()
    {
        // Outside the reference's 1.3 x half-FOV on BOTH axes, where the Jacobian is clamped: dL/dtx and dL/dty lose
        // their J terms and d/dtz changes. Far from the clamp boundary (x/z = 0.55, y/z = 0.45 against limits of
        // ~0.26 / ~0.21), so central differences never straddle the kink.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = AtCamera(v, 0.66f, 0.54f, 1.2f, 0.020f, 0.014f, 0.006f, 0.13f, -0.24f, 0.08f, 0.96f);
        Assert.That(0.66f / 1.2f, Is.GreaterThan(v.LimX * 1.5f), "fixture is not beyond the x clamp");
        Assert.That(0.54f / 1.2f, Is.GreaterThan(v.LimY * 1.5f), "fixture is not beyond the y clamp");
        Assert.That(SplatGeometryGradients.Project(g, v).Valid, Is.True);
        CheckAll(g, v, "clamped");
        // A mild splat's clamped J terms sit under the tolerance floor (red-check: a factor-2 d/dz error passed it).
        // Anisotropic covariances make them dominant: flat, and elongated.
        CheckAll(AtCamera(v, 0.66f, 0.54f, 1.2f, 0.030f, 0.028f, 0.0015f, -0.41f, 0.32f, 0.17f, 0.83f), v, "clamped-flat");
        CheckAll(AtCamera(v, -0.70f, 0.50f, 1.1f, 0.003f, 0.004f, 0.060f, 0.13f, -0.24f, 0.08f, 0.96f), v, "clamped-long");
    }

    [Test]
    public void JacobianClamp_BoundsTheFootprintOfAFarOffAxisSplat()
    {
        // The reason for the clamp: unclamped, a near splat far outside the frustum projects to an enormous
        // ellipse (the Truck probe measured 100-300 thousand px). Clamped, its footprint is what it would be at
        // the clamp boundary.
        var cam = new SplatCovariance.Cov3 { M00 = 1e-4f, M11 = 1e-4f, M22 = 1e-4f };
        float fx = 580f, fy = 580f, z = 0.3f, x = 3.0f;           // x/z = 10, limit ~1.1
        float lim = SplatCovariance.JacobianClampLimit(979f, fx);
        var un = SplatCovariance.ProjectCov2D(cam, x, 0f, z, fx, fy);
        var cl = SplatCovariance.ProjectCov2D(cam, x, 0f, z, fx, fy, lim, lim);
        var edge = SplatCovariance.ProjectCov2D(cam, lim * z, 0f, z, fx, fy);
        // Isotropic: A ~ (fx/z)^2 s (1 + (x/z)^2), so the ratio is (1 + 10^2) / (1 + 1.19^2) ~ 42.
        Assert.That(un.A, Is.GreaterThan(30f * cl.A), "unclamped footprint should be vastly larger");
        Assert.That(cl.A, Is.EqualTo(edge.A).Within(1e-3f * edge.A), "clamped = the footprint at the clamp boundary");
        // Inside the limits the clamp is the identity.
        var inside = SplatCovariance.ProjectCov2D(cam, 0.5f * lim * z, 0f, z, fx, fy, lim, lim);
        var inside0 = SplatCovariance.ProjectCov2D(cam, 0.5f * lim * z, 0f, z, fx, fy);
        Assert.That(inside.A, Is.EqualTo(inside0.A));
    }

    [Test]
    public void Gradients_MatchFiniteDifference_HighlyAnisotropic()
    {
        // A near-flat splat: the covariance is close to singular, which is where the conic
        // inversion's gradient is largest and any factor-of-two error shows up most.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = Splat(0.01f, -0.04f, 0.02f, 0.030f, 0.028f, 0.0015f, -0.41f, 0.32f, 0.17f, 0.83f);
        CheckAll(g, v, "flat");
    }

    [Test]
    public void Gradients_MatchFiniteDifference_UnnormalisedQuaternion()
    {
        // Stored quaternions drift off the unit sphere as Adam steps them, so the gradient has
        // to pass through the normalisation. With |q| = 2 a missing normalisation backward is a
        // clean factor-of-two error on all four components.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = Splat(0.02f, 0.05f, -0.03f, 0.020f, 0.014f, 0.006f, 0.26f, -0.48f, 0.16f, 1.92f);
        CheckAll(g, v, "unnormalised-q");
    }

    [Test]
    public void Gradients_MatchFiniteDifference_OverRandomSplats()
    {
        var rng = new Random(20260920);
        var v = Camera(0.35f, 0.22f, 1.15f);
        int checkedCount = 0;

        for (int t = 0; t < 40; t++)
        {
            var g = Splat(
                (float)(rng.NextDouble() - 0.5) * 0.5f,
                (float)(rng.NextDouble() - 0.5) * 0.4f,
                (float)(rng.NextDouble() - 0.5) * 0.5f,
                0.004f + (float)rng.NextDouble() * 0.030f,
                0.004f + (float)rng.NextDouble() * 0.030f,
                0.002f + (float)rng.NextDouble() * 0.015f,
                (float)rng.NextDouble() - 0.5f,
                (float)rng.NextDouble() - 0.5f,
                (float)rng.NextDouble() - 0.5f,
                (float)rng.NextDouble() - 0.5f);

            if (!SplatGeometryGradients.Project(g, v).Valid) continue;
            CheckAll(g, v, $"random[{t}]");
            checkedCount++;
        }
        Assert.That(checkedCount, Is.GreaterThan(30), "too few random fixtures projected");
    }

    [Test]
    public void Backward_IsZeroForACulledSplat()
    {
        // Behind the camera. A gradient here would drag a splat that contributes nothing to the
        // image, and the forward already refuses to produce one.
        var v = Camera(0f, 0f, 1f);
        var g = Splat(0f, 0f, 2f, 0.02f, 0.02f, 0.01f, 0f, 0f, 0f, 1f);
        Assert.That(SplatGeometryGradients.Project(g, v).Valid, Is.False, "fixture should be culled");

        var d = SplatGeometryGradients.Backward(g, v, Mixed);
        Assert.That(d.PosX, Is.Zero);
        Assert.That(d.PosY, Is.Zero);
        Assert.That(d.PosZ, Is.Zero);
        Assert.That(d.ScaleX, Is.Zero);
        Assert.That(d.QuatW, Is.Zero);
    }

    [Test]
    public void Backward_IsZeroForAClampedScale()
    {
        // MathF.Max cuts the dependency, so the true gradient IS zero. Reporting the unclamped
        // one would let Adam drive a degenerate splat further into the clamp every step.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = Splat(0.02f, 0.05f, -0.03f, 0.020f, 0.014f, -1e-4f, 0.13f, -0.24f, 0.08f, 0.96f);
        var d = SplatGeometryGradients.Backward(g, v, Mixed);
        Assert.That(d.ScaleZ, Is.Zero, "clamped scale must not receive a gradient");
        Assert.That(d.ScaleX, Is.Not.Zero, "unclamped scales must still receive one");
    }

    [Test]
    public void Project_CullsAtTheReferenceNearPlane_NotJustBehindTheEye()
    {
        // The exact splat the Truck forensics probe found in front of every dead view: 14.85 m off
        // to the camera's right, 9.3e-5 in front of its plane, scale 0.17. With a behind-the-eye
        // test (z <= 1e-6) this splat is "valid", projects to a footprint ~5e6 px across, clamps to
        // MaxAlpha at every pixel and zeroes the whole view's gradient. The reference culls at 0.2.
        Assert.That(SplatGeometryGradients.MinDepth, Is.EqualTo(0.2f), "reference in_frustum near plane");

        var v = Camera(0.4f, 0.25f, 1.2f);
        float side = 14.85f;
        SplatGeometryGradients.Geometry At(float depth) => Splat(
            v.EyeX + v.Rx * side + v.Fx3 * depth,
            v.EyeY + v.Ry * side + v.Fy3 * depth,
            v.EyeZ + v.Rz * side + v.Fz3 * depth,
            0.173f, 0.171f, 0.173f, 0f, 0f, 0f, 1f);

        Assert.That(SplatGeometryGradients.Project(At(9.3e-5f), v).Valid, Is.False, "in the camera plane");
        Assert.That(SplatGeometryGradients.Project(At(0.19f), v).Valid, Is.False, "inside the near plane");
        Assert.That(SplatGeometryGradients.Project(At(0.21f), v).Valid, Is.True, "just past the near plane");
        Assert.That(SplatGeometryGradients.Project(At(6.1f), v).Valid, Is.True, "ordinary scene depth");

        // Nothing inside the near plane may receive a gradient either: the backward re-projects.
        var d = SplatGeometryGradients.Backward(At(9.3e-5f), v, Mixed);
        Assert.That(d.PosX, Is.Zero);
        Assert.That(d.ScaleX, Is.Zero);
    }

    [Test]
    public void Project_MatchesTheRasterizersOwnProjection()
    {
        // The rasteriser gate projects splats with its own copy of this arithmetic. If the two
        // drift apart, the gradients are correct for a picture nobody renders.
        var v = Camera(0.4f, 0.25f, 1.2f);
        var g = Splat(0.02f, 0.05f, -0.03f, 0.020f, 0.014f, 0.006f, 0.13f, -0.24f, 0.08f, 0.96f);
        var p = SplatGeometryGradients.Project(g, v);

        var q = new SplatCovariance.Quat { X = g.QuatX, Y = g.QuatY, Z = g.QuatZ, W = g.QuatW };
        float len = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
        q = new SplatCovariance.Quat { X = q.X / len, Y = q.Y / len, Z = q.Z / len, W = q.W / len };

        float relX = g.PosX - v.EyeX, relY = g.PosY - v.EyeY, relZ = g.PosZ - v.EyeZ;
        float tx = v.Rx * relX + v.Ry * relY + v.Rz * relZ;
        float ty = v.Ux * relX + v.Uy * relY + v.Uz * relZ;
        float tz = v.Fx3 * relX + v.Fy3 * relY + v.Fz3 * relZ;

        var cov3 = SplatCovariance.Cov3DFromScaleQuat(g.ScaleX, g.ScaleY, g.ScaleZ, q);
        var camCov = SplatCovariance.RotateToCamera(cov3,
            v.Rx, v.Ry, v.Rz, v.Ux, v.Uy, v.Uz, v.Fx3, v.Fy3, v.Fz3);
        var cov2 = SplatCovariance.ProjectCov2D(camCov, tx, ty, tz, v.FocalX, v.FocalY, v.LimX, v.LimY);
        float invDet = 1f / (cov2.A * cov2.C - cov2.B * cov2.B);

        Assert.That(p.ScreenX, Is.EqualTo(v.FocalX * tx / tz + v.CenterX).Within(1e-3f));
        Assert.That(p.ScreenY, Is.EqualTo(v.CenterY - v.FocalY * ty / tz).Within(1e-3f));
        Assert.That(p.ConicA, Is.EqualTo(cov2.C * invDet).Within(1e-6f));
        Assert.That(p.ConicB, Is.EqualTo(-cov2.B * invDet).Within(1e-6f));
        Assert.That(p.ConicC, Is.EqualTo(cov2.A * invDet).Within(1e-6f));
    }
}
