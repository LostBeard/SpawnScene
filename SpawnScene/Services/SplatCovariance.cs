namespace SpawnScene.Services;

/// <summary>
/// Packed splat buffer layout, shared by every producer and consumer.
///
/// NEVER hardcode the stride. Before this class the literal <c>10</c> appeared in 20+ places
/// across four files; widening the format meant finding all of them.
/// </summary>
public static class SplatFormat
{
    /// <summary>Floats per splat in the GPU-resident packed buffer.</summary>
    public const int Floats = 14;

    public const int OffPos = 0;      // x, y, z
    public const int OffColor = 3;    // r, g, b   (0..1)
    public const int OffScale = 6;    // sx, sy, sz (world units, 1 sigma)
    public const int OffOpacity = 9;
    public const int OffQuat = 10;    // x, y, z, w — rotates splat-local axes into world

    /// <summary>Bytes per vertex in the render-ready packed format written by the pack compute pass.</summary>
    public const int PackedBytes = 32; // 12 pos + 4 color/alpha + 8 scale f16x4 + 8 quat f16x4

    /// <summary>u32 words per vertex (<see cref="PackedBytes"/> / 4).</summary>
    public const int PackedWords = 8;
}

/// <summary>
/// CPU oracle for 3D Gaussian covariance projection (Zwicker EWA / Kerbl 3DGS).
///
/// The WGSL vertex shaders in <see cref="GpuGaussianRenderer"/> and the ILGPU kernels in
/// <see cref="DepthToGaussianKernel"/> MUST match these functions. Gate:
/// <c>SpawnScene.Tests/SplatCovarianceTests.cs</c>, which checks them against analytic
/// answers (a disk turned 60 degrees away foreshortens by cos 60), not against each other.
///
/// Camera convention here is the one the rest of this codebase uses (see
/// <see cref="WorldSpaceGeometry"/>): OpenCV pinhole, world to camera by the orthonormal rows
/// [right; up; forward], z forward and positive in front. Screen y is measured UPWARD so that a
/// pixel offset maps to NDC as (dx * 2 / width, dy * 2 / height) with no sign flip.
///
/// Every method is scalar-in / struct-out with no arrays, no System.Numerics and no branching on
/// reference types, so ILGPU can inline them straight into a kernel.
/// </summary>
public static class SplatCovariance
{
    /// <summary>Splat orientation. Unit quaternion, (x, y, z) vector part and w scalar part.</summary>
    public struct Quat
    {
        public float X, Y, Z, W;
        public static Quat Identity => new() { X = 0f, Y = 0f, Z = 0f, W = 1f };
    }

    /// <summary>Symmetric 3x3 covariance, upper triangle.</summary>
    public struct Cov3
    {
        public float M00, M01, M02, M11, M12, M22;
    }

    /// <summary>Symmetric 2x2 screen-space covariance in PIXELS squared.</summary>
    public struct Cov2
    {
        public float A;  // xx
        public float B;  // xy
        public float C;  // yy
    }

    /// <summary>
    /// Screen-space ellipse: two orthogonal half-axes in pixels, each already scaled to the
    /// requested sigma cutoff. A quad spanning +/-(AxisX, AxisY) covers the whole splat.
    /// </summary>
    public struct Ellipse
    {
        public float Ax, Ay;   // major half-axis (pixels)
        public float Bx, By;   // minor half-axis (pixels)
        public bool Valid;
    }

    /// <summary>
    /// EWA low-pass filter added to the screen-space covariance diagonal, in pixels squared.
    /// Zwicker's antialiasing prefilter: without it a splat smaller than a pixel aliases into
    /// a flickering dot. Replaces the old <c>max(radius, 0.25px)</c> clamp, which fattened
    /// every splat isotropically instead of only the sub-pixel ones.
    /// </summary>
    public const float EwaFilterPx2 = 0.3f;

    /// <summary>
    /// Shortest-arc quaternion taking the splat-local +Z axis onto <paramref name="nx"/>..
    /// <paramref name="nz"/>. The splat is flat in local Z, so its local +Z is its surface normal
    /// and the in-plane orientation is free (sx == sy at init, so any tangent basis is equivalent).
    /// Returns identity for a degenerate normal.
    /// </summary>
    public static Quat QuatFromNormal(float nx, float ny, float nz)
    {
        float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
        if (!(len > 1e-12f)) return Quat.Identity;
        nx /= len; ny /= len; nz /= len;

        // Shortest arc from (0,0,1) to n: axis = cross(z, n) = (-ny, nx, 0), w = 1 + dot(z, n).
        float w = 1f + nz;
        if (w < 1e-6f)
        {
            // n is antiparallel to +Z: a 180 degree turn about any perpendicular axis. Use +X.
            return new Quat { X = 1f, Y = 0f, Z = 0f, W = 0f };
        }

        float qx = -ny;
        float qy = nx;
        float qz = 0f;
        float qn = MathF.Sqrt(qx * qx + qy * qy + qz * qz + w * w);
        return new Quat { X = qx / qn, Y = qy / qn, Z = qz / qn, W = w / qn };
    }

    /// <summary>
    /// World-space covariance Sigma = R S S^T R^T for scale <paramref name="sx"/>,
    /// <paramref name="sy"/>, <paramref name="sz"/> (1 sigma, world units) and rotation q.
    /// </summary>
    public static Cov3 Cov3DFromScaleQuat(float sx, float sy, float sz, Quat q)
    {
        // Quaternion to rotation matrix, columns = rotated local axes.
        float xx = q.X * q.X, yy = q.Y * q.Y, zz = q.Z * q.Z;
        float xy = q.X * q.Y, xz = q.X * q.Z, yz = q.Y * q.Z;
        float wx = q.W * q.X, wy = q.W * q.Y, wz = q.W * q.Z;

        float r00 = 1f - 2f * (yy + zz), r01 = 2f * (xy - wz), r02 = 2f * (xz + wy);
        float r10 = 2f * (xy + wz), r11 = 1f - 2f * (xx + zz), r12 = 2f * (yz - wx);
        float r20 = 2f * (xz - wy), r21 = 2f * (yz + wx), r22 = 1f - 2f * (xx + yy);

        // M = R * S (scale the COLUMNS: column k is the k-th local axis times its scale).
        float m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
        float m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
        float m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

        // Sigma = M * M^T
        return new Cov3
        {
            M00 = m00 * m00 + m01 * m01 + m02 * m02,
            M01 = m00 * m10 + m01 * m11 + m02 * m12,
            M02 = m00 * m20 + m01 * m21 + m02 * m22,
            M11 = m10 * m10 + m11 * m11 + m12 * m12,
            M12 = m10 * m20 + m11 * m21 + m12 * m22,
            M22 = m20 * m20 + m21 * m21 + m22 * m22,
        };
    }

    /// <summary>
    /// Rotate a world covariance into camera space: Sigma_cam = A * Sigma * A^T, where A has the
    /// camera basis as its ROWS (row0 = right, row1 = up, row2 = forward, all world-space unit
    /// vectors). Getting this transpose backwards produces an ellipse that tilts the wrong way and
    /// is invisible on a symmetric test case, which is why the gate uses an asymmetric rotation.
    /// </summary>
    public static Cov3 RotateToCamera(
        Cov3 s,
        float rx, float ry, float rz,
        float ux, float uy, float uz,
        float fx3, float fy3, float fz3)
    {
        // T = A * Sigma (row i of T = row i of A times Sigma)
        float t00 = rx * s.M00 + ry * s.M01 + rz * s.M02;
        float t01 = rx * s.M01 + ry * s.M11 + rz * s.M12;
        float t02 = rx * s.M02 + ry * s.M12 + rz * s.M22;

        float t10 = ux * s.M00 + uy * s.M01 + uz * s.M02;
        float t11 = ux * s.M01 + uy * s.M11 + uz * s.M12;
        float t12 = ux * s.M02 + uy * s.M12 + uz * s.M22;

        float t20 = fx3 * s.M00 + fy3 * s.M01 + fz3 * s.M02;
        float t21 = fx3 * s.M01 + fy3 * s.M11 + fz3 * s.M12;
        float t22 = fx3 * s.M02 + fy3 * s.M12 + fz3 * s.M22;

        // Sigma_cam = T * A^T (column j of A^T = row j of A)
        return new Cov3
        {
            M00 = t00 * rx + t01 * ry + t02 * rz,
            M01 = t00 * ux + t01 * uy + t02 * uz,
            M02 = t00 * fx3 + t01 * fy3 + t02 * fz3,
            M11 = t10 * ux + t11 * uy + t12 * uz,
            M12 = t10 * fx3 + t11 * fy3 + t12 * fz3,
            M22 = t20 * fx3 + t21 * fy3 + t22 * fz3,
        };
    }

    /// <summary>
    /// Perspective-project a camera-space covariance to pixels: Sigma_2D = J * Sigma_cam * J^T,
    /// with J the Jacobian of (u, v) = (fx * x / z, fy * y / z) at the splat centre
    /// (<paramref name="camX"/>, <paramref name="camY"/>, <paramref name="camZ"/>).
    /// The EWA prefilter is added here, so the result is ready to decompose.
    /// </summary>
    public static Cov2 ProjectCov2D(Cov3 camCov, float camX, float camY, float camZ, float fx, float fy)
    {
        float invZ = 1f / camZ;
        float invZ2 = invZ * invZ;

        float j00 = fx * invZ;
        float j02 = -fx * camX * invZ2;
        float j11 = fy * invZ;
        float j12 = -fy * camY * invZ2;

        // Row 0 of J * Sigma_cam (J row 0 = [j00, 0, j02])
        float a0 = j00 * camCov.M00 + j02 * camCov.M02;
        float a1 = j00 * camCov.M01 + j02 * camCov.M12;
        float a2 = j00 * camCov.M02 + j02 * camCov.M22;

        // Row 1 of J * Sigma_cam (J row 1 = [0, j11, j12])
        float b0 = j11 * camCov.M01 + j12 * camCov.M02;
        float b1 = j11 * camCov.M11 + j12 * camCov.M12;
        float b2 = j11 * camCov.M12 + j12 * camCov.M22;

        return new Cov2
        {
            A = a0 * j00 + a2 * j02 + EwaFilterPx2,
            B = a1 * j11 + a2 * j12,
            C = b1 * j11 + b2 * j12 + EwaFilterPx2,
        };
    }

    /// <summary>
    /// Eigen-decompose the 2x2 screen covariance into two orthogonal half-axes, each
    /// <paramref name="sigmaCutoff"/> standard deviations long, in pixels.
    ///
    /// A quad spanning +/-(major, minor) then has the property that the Gaussian at quad
    /// coordinate (s, t) in [-1,1]^2 is exp(-0.5 * cutoff^2 * (s*s + t*t)) — so the fragment
    /// shader needs NO conic matrix, only dot(uv, uv).
    /// </summary>
    public static Ellipse EigenAxes(Cov2 c, float sigmaCutoff)
    {
        float det = c.A * c.C - c.B * c.B;
        if (!(det > 1e-20f)) return default; // Valid = false

        float mid = 0.5f * (c.A + c.C);
        float disc = MathF.Sqrt(MathF.Max(mid * mid - det, 0f));
        float l1 = mid + disc;          // major eigenvalue
        float l2 = mid - disc;          // minor eigenvalue
        if (!(l2 > 0f)) l2 = det / MathF.Max(l1, 1e-20f);
        if (!(l1 > 0f)) return default;

        // Eigenvector for l1. Both (B, l1 - A) and (l1 - C, B) span it; pick the longer one so a
        // near-isotropic covariance (B ~ 0) does not normalize noise into an arbitrary direction.
        float e1x, e1y;
        float p1x = c.B, p1y = l1 - c.A;
        float p2x = l1 - c.C, p2y = c.B;
        if (p1x * p1x + p1y * p1y >= p2x * p2x + p2y * p2y) { e1x = p1x; e1y = p1y; }
        else { e1x = p2x; e1y = p2y; }

        float elen = MathF.Sqrt(e1x * e1x + e1y * e1y);
        if (elen > 1e-12f) { e1x /= elen; e1y /= elen; }
        else { e1x = 1f; e1y = 0f; }

        float r1 = sigmaCutoff * MathF.Sqrt(l1);
        float r2 = sigmaCutoff * MathF.Sqrt(l2);

        return new Ellipse
        {
            Ax = e1x * r1,
            Ay = e1y * r1,
            Bx = -e1y * r2,
            By = e1x * r2,
            Valid = true,
        };
    }

    /// <summary>
    /// Camera-space surface normal from three unprojected depth samples (centre, +x neighbour,
    /// +y neighbour) in OpenCV camera axes (x right, y down, z forward). Result points BACK
    /// toward the camera. Returns the view direction when the triangle is degenerate, which
    /// yields the camera-facing disk the renderer drew before orientation existed.
    /// </summary>
    public static Quat NormalQuatFromNeighbors(
        float px, float py, float pz,
        float qx, float qy, float qz,
        float rx, float ry, float rz)
    {
        float ax = qx - px, ay = qy - py, az = qz - pz;
        float bx = rx - px, by = ry - py, bz = rz - pz;

        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;

        float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
        if (!(len > 1e-20f))
        {
            // Degenerate: face the camera.
            nx = -px; ny = -py; nz = -pz;
        }
        else
        {
            nx /= len; ny /= len; nz /= len;
        }

        // Orient toward the camera (camera sits at the origin, so the outward direction is +p).
        if (nx * px + ny * py + nz * pz > 0f) { nx = -nx; ny = -ny; nz = -nz; }

        return QuatFromNormal(nx, ny, nz);
    }

    /// <summary>
    /// Rotate a quaternion by a world rotation given as the OpenCV world-to-camera rows
    /// (r*, d*, f*) used by <c>UnprojectWorldSpaceKernel</c>: the camera-to-world matrix is that
    /// matrix transposed, so its COLUMNS are (r, d, f). Converts directly from the matrix to avoid
    /// a second quaternion multiply in the kernel.
    /// </summary>
    public static Quat RotateQuatToWorld(
        Quat q,
        float r0, float r1, float r2,
        float d0, float d1, float d2,
        float f0, float f1, float f2)
        // Camera-to-world is the world-to-camera rows read as COLUMNS.
        => RotateQuatByMatrix(q,
            r0, d0, f0,
            r1, d1, f1,
            r2, d2, f2);

    /// <summary>
    /// Pre-multiply a splat's orientation by a rotation matrix (row-major, acting on column
    /// vectors): q' represents M * R(q). Used when a whole cloud is rigidly transformed - the
    /// positions move, and the orientations must move with them or every splat keeps facing
    /// the direction it had in the old frame.
    /// </summary>
    public static Quat RotateQuatByMatrix(
        Quat q,
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
    {
        // Compose with q's matrix and convert back. Shepperd's method keeps this stable for all
        // four cases; a naive w = sqrt(1 + trace)/2 divides by zero at a 180 degree turn.
        float qxx = q.X * q.X, qyy = q.Y * q.Y, qzz = q.Z * q.Z;
        float qxy = q.X * q.Y, qxz = q.X * q.Z, qyz = q.Y * q.Z;
        float qwx = q.W * q.X, qwy = q.W * q.Y, qwz = q.W * q.Z;

        float a00 = 1f - 2f * (qyy + qzz), a01 = 2f * (qxy - qwz), a02 = 2f * (qxz + qwy);
        float a10 = 2f * (qxy + qwz), a11 = 1f - 2f * (qxx + qzz), a12 = 2f * (qyz - qwx);
        float a20 = 2f * (qxz - qwy), a21 = 2f * (qyz + qwx), a22 = 1f - 2f * (qxx + qyy);

        float c00 = m00 * a00 + m01 * a10 + m02 * a20;
        float c01 = m00 * a01 + m01 * a11 + m02 * a21;
        float c02 = m00 * a02 + m01 * a12 + m02 * a22;
        float c10 = m10 * a00 + m11 * a10 + m12 * a20;
        float c11 = m10 * a01 + m11 * a11 + m12 * a21;
        float c12 = m10 * a02 + m11 * a12 + m12 * a22;
        float c20 = m20 * a00 + m21 * a10 + m22 * a20;
        float c21 = m20 * a01 + m21 * a11 + m22 * a21;
        float c22 = m20 * a02 + m21 * a12 + m22 * a22;

        return QuatFromMatrix(c00, c01, c02, c10, c11, c12, c20, c21, c22);
    }

    /// <summary>Rotation matrix (row-major, acting on column vectors) to unit quaternion.</summary>
    public static Quat QuatFromMatrix(
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
    {
        float trace = m00 + m11 + m22;
        float qw, qx, qy, qz;

        if (trace > 0f)
        {
            float s = MathF.Sqrt(trace + 1f) * 2f;
            qw = 0.25f * s;
            qx = (m21 - m12) / s;
            qy = (m02 - m20) / s;
            qz = (m10 - m01) / s;
        }
        else if (m00 > m11 && m00 > m22)
        {
            float s = MathF.Sqrt(1f + m00 - m11 - m22) * 2f;
            qw = (m21 - m12) / s;
            qx = 0.25f * s;
            qy = (m01 + m10) / s;
            qz = (m02 + m20) / s;
        }
        else if (m11 > m22)
        {
            float s = MathF.Sqrt(1f + m11 - m00 - m22) * 2f;
            qw = (m02 - m20) / s;
            qx = (m01 + m10) / s;
            qy = 0.25f * s;
            qz = (m12 + m21) / s;
        }
        else
        {
            float s = MathF.Sqrt(1f + m22 - m00 - m11) * 2f;
            qw = (m10 - m01) / s;
            qx = (m02 + m20) / s;
            qy = (m12 + m21) / s;
            qz = 0.25f * s;
        }

        float n = MathF.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (!(n > 1e-20f)) return Quat.Identity;
        return new Quat { X = qx / n, Y = qy / n, Z = qz / n, W = qw / n };
    }
}
