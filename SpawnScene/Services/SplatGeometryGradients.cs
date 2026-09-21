namespace SpawnScene.Services;

/// <summary>
/// Gradients of the screen-space splat with respect to its GEOMETRY: world position, the three
/// scales, and the orientation quaternion.
///
/// <see cref="SplatRasterizer"/> already produces gradients at the 2D level - screen centre and
/// conic. This is the chain that carries those back through the projection to the parameters
/// that actually describe the scene:
///
///     q, s  ->  Sigma_world = R S S^T R^T
///           ->  Sigma_cam   = A Sigma_world A^T          (A rows = right, up, forward)
///           ->  Sigma_2D    = J Sigma_cam J^T + lambda I (J = Jacobian of the pinhole at t)
///           ->  conic       = Sigma_2D^-1
///     p     ->  t = A (p - eye)  ->  screen centre, AND J (which depends on t)
///
/// The second path from position is the one that is easy to forget: moving a splat sideways does
/// not only move its centre, it changes the perspective foreshortening of its footprint. Dropping
/// it leaves a gradient that is correct at the image centre and increasingly wrong toward the
/// edges, which is exactly the sort of error that still trains, just to a worse answer.
///
/// Everything here is scalar in / struct out with no arrays and no System.Numerics, matching the
/// rest of <see cref="SplatCovariance"/>, so the same code can be inlined into an ILGPU kernel
/// and transcribed line for line into WGSL.
///
/// Verified in <c>SpawnScene.Tests/SplatGeometryGradientsTests.cs</c> against central finite
/// differences of <see cref="Project"/> - never against another derivation of the same algebra.
/// </summary>
public static class SplatGeometryGradients
{
    /// <summary>Camera the splat is being projected into. Basis rows are right, up, forward.</summary>
    public struct View
    {
        public float EyeX, EyeY, EyeZ;
        public float Rx, Ry, Rz;       // right
        public float Ux, Uy, Uz;       // up
        public float Fx3, Fy3, Fz3;    // forward
        public float FocalX, FocalY;
        public float CenterX, CenterY;
    }

    /// <summary>A splat's geometry in world space. The quaternion need not be normalised.</summary>
    public struct Geometry
    {
        public float PosX, PosY, PosZ;
        public float ScaleX, ScaleY, ScaleZ;
        public float QuatX, QuatY, QuatZ, QuatW;
    }

    /// <summary>What the forward pass produces, and what the rasteriser differentiates.</summary>
    public struct Projected
    {
        public float ScreenX, ScreenY;
        public float ConicA, ConicB, ConicC;
        public float Depth;
        public bool Valid;
    }

    /// <summary>dL/d(geometry). Same field order as <see cref="Geometry"/>.</summary>
    public struct Grad
    {
        public float PosX, PosY, PosZ;
        public float ScaleX, ScaleY, ScaleZ;
        public float QuatX, QuatY, QuatZ, QuatW;
    }

    /// <summary>Incoming gradients from the rasteriser: dL/d(screen centre) and dL/d(conic).</summary>
    public struct UpstreamGrad
    {
        public float ScreenX, ScreenY;
        public float ConicA, ConicB, ConicC;
    }

    /// <summary>
    /// Smallest determinant of the 2D covariance treated as invertible. Below this the splat is
    /// edge-on to within floating-point noise and both the forward conic and every gradient
    /// through it blow up.
    /// </summary>
    public const float MinDet = 1e-20f;

    /// <summary>Camera-space z below which the splat is at or behind the eye.</summary>
    public const float MinDepth = 1e-6f;

    /// <summary>
    /// The forward pass, exactly as the rasteriser and the WGSL shader do it. Tests finite
    /// difference THIS function, so it is the definition the gradients must agree with.
    /// </summary>
    public static Projected Project(Geometry g, View v)
    {
        float relX = g.PosX - v.EyeX, relY = g.PosY - v.EyeY, relZ = g.PosZ - v.EyeZ;
        float tx = v.Rx * relX + v.Ry * relY + v.Rz * relZ;
        float ty = v.Ux * relX + v.Uy * relY + v.Uz * relZ;
        float tz = v.Fx3 * relX + v.Fy3 * relY + v.Fz3 * relZ;
        if (tz <= MinDepth) return default;

        var q = Normalised(g);
        var cov3 = SplatCovariance.Cov3DFromScaleQuat(
            MathF.Max(g.ScaleX, 1e-9f), MathF.Max(g.ScaleY, 1e-9f), MathF.Max(g.ScaleZ, 1e-9f), q);
        var camCov = SplatCovariance.RotateToCamera(cov3,
            v.Rx, v.Ry, v.Rz, v.Ux, v.Uy, v.Uz, v.Fx3, v.Fy3, v.Fz3);
        var cov2 = SplatCovariance.ProjectCov2D(camCov, tx, ty, tz, v.FocalX, v.FocalY);

        float det = cov2.A * cov2.C - cov2.B * cov2.B;
        if (!(det > MinDet)) return default;
        float invDet = 1f / det;

        return new Projected
        {
            // Screen y grows DOWN, camera y grows UP.
            ScreenX = v.FocalX * tx / tz + v.CenterX,
            ScreenY = v.CenterY - v.FocalY * ty / tz,
            ConicA = cov2.C * invDet,
            ConicB = -cov2.B * invDet,
            ConicC = cov2.A * invDet,
            Depth = tz,
            Valid = true,
        };
    }

    static SplatCovariance.Quat Normalised(Geometry g)
    {
        float len = MathF.Sqrt(
            g.QuatX * g.QuatX + g.QuatY * g.QuatY + g.QuatZ * g.QuatZ + g.QuatW * g.QuatW);
        if (!(len > 1e-20f)) return SplatCovariance.Quat.Identity;
        return new SplatCovariance.Quat
        {
            X = g.QuatX / len, Y = g.QuatY / len, Z = g.QuatZ / len, W = g.QuatW / len,
        };
    }

    /// <summary>
    /// Carry <paramref name="up"/> back to the splat's geometry. Returns all-zero for a splat
    /// the forward pass rejects - a culled splat has no gradient, and inventing one would drag
    /// it around from behind the camera.
    /// </summary>
    public static Grad Backward(Geometry g, View v, UpstreamGrad up)
    {
        // ── Forward again, keeping the intermediates ──
        float relX = g.PosX - v.EyeX, relY = g.PosY - v.EyeY, relZ = g.PosZ - v.EyeZ;
        float tx = v.Rx * relX + v.Ry * relY + v.Rz * relZ;
        float ty = v.Ux * relX + v.Uy * relY + v.Uz * relZ;
        float tz = v.Fx3 * relX + v.Fy3 * relY + v.Fz3 * relZ;
        if (tz <= MinDepth) return default;

        float sx = MathF.Max(g.ScaleX, 1e-9f);
        float sy = MathF.Max(g.ScaleY, 1e-9f);
        float sz = MathF.Max(g.ScaleZ, 1e-9f);
        var q = Normalised(g);

        var cov3 = SplatCovariance.Cov3DFromScaleQuat(sx, sy, sz, q);
        var camCov = SplatCovariance.RotateToCamera(cov3,
            v.Rx, v.Ry, v.Rz, v.Ux, v.Uy, v.Uz, v.Fx3, v.Fy3, v.Fz3);

        float invZ = 1f / tz;
        float invZ2 = invZ * invZ;
        float j00 = v.FocalX * invZ;
        float j02 = -v.FocalX * tx * invZ2;
        float j11 = v.FocalY * invZ;
        float j12 = -v.FocalY * ty * invZ2;

        var cov2 = SplatCovariance.ProjectCov2D(camCov, tx, ty, tz, v.FocalX, v.FocalY);
        float a = cov2.A, b = cov2.B, c = cov2.C;
        float det = a * c - b * b;
        if (!(det > MinDet)) return default;
        float iD = 1f / det;
        float iD2 = iD * iD;

        // ── 1. conic = Sigma_2D^-1, in the three unique components ──
        //   cA = c/D, cB = -b/D, cC = a/D,  D = ac - b^2
        float gA =
            up.ConicA * (-c * c * iD2) +
            up.ConicB * (b * c * iD2) +
            up.ConicC * (iD - a * c * iD2);
        float gB =
            up.ConicA * (2f * b * c * iD2) +
            up.ConicB * (-iD - 2f * b * b * iD2) +
            up.ConicC * (2f * a * b * iD2);
        float gC =
            up.ConicA * (iD - a * c * iD2) +
            up.ConicB * (a * b * iD2) +
            up.ConicC * (-a * a * iD2);

        // ── 2. Sigma_2D = J Sigma_cam J^T, expanded ──
        //   A = j00^2 S00 + 2 j00 j02 S02 + j02^2 S22 + lambda
        //   B = j00 j11 S01 + j00 j12 S02 + j02 j11 S12 + j02 j12 S22
        //   C = j11^2 S11 + 2 j11 j12 S12 + j12^2 S22 + lambda
        float S00 = camCov.M00, S01 = camCov.M01, S02 = camCov.M02;
        float S11 = camCov.M11, S12 = camCov.M12, S22 = camCov.M22;

        float gS00 = gA * j00 * j00;
        float gS01 = gB * j00 * j11;
        float gS02 = gA * 2f * j00 * j02 + gB * j00 * j12;
        float gS11 = gC * j11 * j11;
        float gS12 = gB * j02 * j11 + gC * 2f * j11 * j12;
        float gS22 = gA * j02 * j02 + gB * j02 * j12 + gC * j12 * j12;

        float gj00 = gA * 2f * (j00 * S00 + j02 * S02) + gB * (j11 * S01 + j12 * S02);
        float gj02 = gA * 2f * (j00 * S02 + j02 * S22) + gB * (j11 * S12 + j12 * S22);
        float gj11 = gB * (j00 * S01 + j02 * S12) + gC * 2f * (j11 * S11 + j12 * S12);
        float gj12 = gB * (j00 * S02 + j02 * S22) + gC * 2f * (j11 * S12 + j12 * S22);

        // ── 3. Sigma_cam = A Sigma_world A^T  =>  dL/dSigma_world = A^T (dL/dSigma_cam) A ──
        // The unique-component gradients are converted to a full symmetric matrix by halving the
        // off-diagonals, transformed, then converted back by doubling them. Skipping either half
        // of that bookkeeping scales the rotation and scale gradients by two on the off-diagonals
        // and by one on the diagonal, which finite differences catch immediately.
        float h01 = 0.5f * gS01, h02 = 0.5f * gS02, h12 = 0.5f * gS12;
        // K = A^T G  (A rows = right, up, forward, so A^T columns are those)
        float k00 = v.Rx * gS00 + v.Ux * h01 + v.Fx3 * h02;
        float k01 = v.Rx * h01 + v.Ux * gS11 + v.Fx3 * h12;
        float k02 = v.Rx * h02 + v.Ux * h12 + v.Fx3 * gS22;
        float k10 = v.Ry * gS00 + v.Uy * h01 + v.Fy3 * h02;
        float k11 = v.Ry * h01 + v.Uy * gS11 + v.Fy3 * h12;
        float k12 = v.Ry * h02 + v.Uy * h12 + v.Fy3 * gS22;
        float k20 = v.Rz * gS00 + v.Uz * h01 + v.Fz3 * h02;
        float k21 = v.Rz * h01 + v.Uz * gS11 + v.Fz3 * h12;
        float k22 = v.Rz * h02 + v.Uz * h12 + v.Fz3 * gS22;
        // W = K A  (row i of W = row i of K times A)
        float w00 = k00 * v.Rx + k01 * v.Ux + k02 * v.Fx3;
        float w01 = k00 * v.Ry + k01 * v.Uy + k02 * v.Fy3;
        float w02 = k00 * v.Rz + k01 * v.Uz + k02 * v.Fz3;
        float w11 = k10 * v.Ry + k11 * v.Uy + k12 * v.Fy3;
        float w12 = k10 * v.Rz + k11 * v.Uz + k12 * v.Fz3;
        float w22 = k20 * v.Rz + k21 * v.Uz + k22 * v.Fz3;

        // ── 4. Sigma_world = M M^T with M = R diag(s) ──
        // dL/dM = 2 (dL/dSigma_world)_full M, the full matrix again.
        float x = q.X, y = q.Y, z = q.Z, w = q.W;
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        float r00 = 1f - 2f * (yy + zz), r01 = 2f * (xy - wz), r02 = 2f * (xz + wy);
        float r10 = 2f * (xy + wz), r11 = 1f - 2f * (xx + zz), r12 = 2f * (yz - wx);
        float r20 = 2f * (xz - wy), r21 = 2f * (yz + wx), r22 = 1f - 2f * (xx + yy);

        float m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
        float m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
        float m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

        // w is ALREADY the full-matrix gradient of Sigma_world - A^T G A produces matrix
        // entries, not unique-parameter derivatives. Halving here as well (the conversion only
        // belongs where unique-parameter gradients enter, at gS01/gS02/gS12) scaled every scale
        // and rotation gradient by a fixture-dependent 1.2-1.3x, which finite differences catch
        // and a "does the loss go down" check never would.
        float f00 = w00, f01 = w01, f02 = w02;
        float f11 = w11, f12 = w12, f22 = w22;

        float gm00 = 2f * (f00 * m00 + f01 * m10 + f02 * m20);
        float gm01 = 2f * (f00 * m01 + f01 * m11 + f02 * m21);
        float gm02 = 2f * (f00 * m02 + f01 * m12 + f02 * m22);
        float gm10 = 2f * (f01 * m00 + f11 * m10 + f12 * m20);
        float gm11 = 2f * (f01 * m01 + f11 * m11 + f12 * m21);
        float gm12 = 2f * (f01 * m02 + f11 * m12 + f12 * m22);
        float gm20 = 2f * (f02 * m00 + f12 * m10 + f22 * m20);
        float gm21 = 2f * (f02 * m01 + f12 * m11 + f22 * m21);
        float gm22 = 2f * (f02 * m02 + f12 * m12 + f22 * m22);

        // M[i][k] = R[i][k] * s_k
        float gsx = gm00 * r00 + gm10 * r10 + gm20 * r20;
        float gsy = gm01 * r01 + gm11 * r11 + gm21 * r21;
        float gsz = gm02 * r02 + gm12 * r12 + gm22 * r22;

        float gr00 = gm00 * sx, gr10 = gm10 * sx, gr20 = gm20 * sx;
        float gr01 = gm01 * sy, gr11 = gm11 * sy, gr21 = gm21 * sy;
        float gr02 = gm02 * sz, gr12 = gm12 * sz, gr22 = gm22 * sz;

        // ── 5. R from the NORMALISED quaternion ──
        float gnx =
            2f * y * (gr01 + gr10) + 2f * z * (gr02 + gr20) -
            4f * x * (gr11 + gr22) + 2f * w * (gr21 - gr12);
        float gny =
            -4f * y * (gr00 + gr22) + 2f * x * (gr01 + gr10) +
            2f * w * (gr02 - gr20) + 2f * z * (gr12 + gr21);
        float gnz =
            -4f * z * (gr00 + gr11) + 2f * w * (gr10 - gr01) +
            2f * x * (gr02 + gr20) + 2f * y * (gr12 + gr21);
        float gnw =
            2f * z * (gr10 - gr01) + 2f * y * (gr02 - gr20) + 2f * x * (gr21 - gr12);

        // Through the normalisation: d/dq of (q / |q|) applied to the gradient.
        float qlen = MathF.Sqrt(
            g.QuatX * g.QuatX + g.QuatY * g.QuatY + g.QuatZ * g.QuatZ + g.QuatW * g.QuatW);
        float gqx = 0f, gqy = 0f, gqz = 0f, gqw = 0f;
        if (qlen > 1e-20f)
        {
            float dot = gnx * x + gny * y + gnz * z + gnw * w;
            float invLen = 1f / qlen;
            gqx = (gnx - dot * x) * invLen;
            gqy = (gny - dot * y) * invLen;
            gqz = (gnz - dot * z) * invLen;
            gqw = (gnw - dot * w) * invLen;
        }

        // ── 6. Camera-space position: through the screen centre AND through J ──
        float gtx = up.ScreenX * v.FocalX * invZ;
        float gty = up.ScreenY * (-v.FocalY * invZ);
        float gtz =
            up.ScreenX * (-v.FocalX * tx * invZ2) +
            up.ScreenY * (v.FocalY * ty * invZ2);

        gtx += gj02 * (-v.FocalX * invZ2);
        gty += gj12 * (-v.FocalY * invZ2);
        gtz +=
            gj00 * (-v.FocalX * invZ2) +
            gj02 * (2f * v.FocalX * tx * invZ2 * invZ) +
            gj11 * (-v.FocalY * invZ2) +
            gj12 * (2f * v.FocalY * ty * invZ2 * invZ);

        // t = A (p - eye)  =>  dL/dp = A^T dL/dt
        return new Grad
        {
            PosX = v.Rx * gtx + v.Ux * gty + v.Fx3 * gtz,
            PosY = v.Ry * gtx + v.Uy * gty + v.Fy3 * gtz,
            PosZ = v.Rz * gtx + v.Uz * gty + v.Fz3 * gtz,
            // A clamped scale is a constant: MathF.Max cut the dependency, so the gradient is
            // zero there too. Reporting one would let Adam march a degenerate splat further
            // into the clamp forever.
            ScaleX = g.ScaleX > 1e-9f ? gsx : 0f,
            ScaleY = g.ScaleY > 1e-9f ? gsy : 0f,
            ScaleZ = g.ScaleZ > 1e-9f ? gsz : 0f,
            QuatX = gqx, QuatY = gqy, QuatZ = gqz, QuatW = gqw,
        };
    }
}
