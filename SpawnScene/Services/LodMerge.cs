using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Merging splats into one parent for the level-of-detail tree (Plans/lod-streaming.md): the parent's mean and
/// covariance are the moment-matched mixture of its children - Sigma_p = sum w (Sigma_i + (p_i - mu)(p_i - mu)^T) / sum w
/// - with weights w = opacity x footprint area (Spark 2.0's choice); colour is the w-weighted mean; opacity conserves
/// opacity x area (the hierarchical 3DGS rule), clamped to 1. Covariance back to scale + rotation is a Jacobi eigen
/// decomposition. Scalar code throughout, so a GPU kernel and the CPU tests run the same functions.
/// </summary>
public static class LodMerge
{
    const int F = SplatFormat.Floats;

    /// <summary>Running sums of one parent's children (doubles on the CPU oracle path).</summary>
    public struct Accum
    {
        public double W, PX, PY, PZ, CR, CG, CB, AlphaArea;
        // w-weighted second moments of position, plus each child's own covariance: sum w (Sigma_i + p p^T)
        public double Sxx, Sxy, Sxz, Syy, Syz, Szz;
        public int Count;
    }

    /// <summary>Footprint area of a splat: pi x its two largest 1-sigma scales - what it covers seen face-on.</summary>
    public static float Area(float sx, float sy, float sz)
    {
        float a = MathF.Max(sx, MathF.Max(sy, sz));
        float c = MathF.Min(sx, MathF.Min(sy, sz));
        float b = sx + sy + sz - a - c;
        return MathF.PI * a * b;
    }

    /// <summary>Add one packed splat row to <paramref name="acc"/>.</summary>
    public static void Add(ref Accum acc, ReadOnlySpan<float> row)
    {
        float sx = row[SplatFormat.OffScale], sy = row[SplatFormat.OffScale + 1], sz = row[SplatFormat.OffScale + 2];
        float alpha = row[SplatFormat.OffOpacity];
        float area = Area(sx, sy, sz);
        double w = Math.Max(1e-12, (double)alpha * area);
        var q = new SplatCovariance.Quat { X = row[SplatFormat.OffQuat], Y = row[SplatFormat.OffQuat + 1], Z = row[SplatFormat.OffQuat + 2], W = row[SplatFormat.OffQuat + 3] };
        var c = SplatCovariance.Cov3DFromScaleQuat(sx, sy, sz, Normalize(q));
        double x = row[0], y = row[1], z = row[2];
        acc.W += w;
        acc.PX += w * x; acc.PY += w * y; acc.PZ += w * z;
        acc.CR += w * row[SplatFormat.OffColor]; acc.CG += w * row[SplatFormat.OffColor + 1]; acc.CB += w * row[SplatFormat.OffColor + 2];
        acc.Sxx += w * (c.M00 + x * x); acc.Sxy += w * (c.M01 + x * y); acc.Sxz += w * (c.M02 + x * z);
        acc.Syy += w * (c.M11 + y * y); acc.Syz += w * (c.M12 + y * z); acc.Szz += w * (c.M22 + z * z);
        acc.AlphaArea += (double)alpha * area;
        acc.Count++;
    }

    /// <summary>The parent splat of everything added to <paramref name="acc"/>, as a packed row.</summary>
    public static void Finish(in Accum acc, Span<float> parent)
    {
        double inv = 1.0 / acc.W;
        double mx = acc.PX * inv, my = acc.PY * inv, mz = acc.PZ * inv;
        var cov = new SplatCovariance.Cov3
        {
            M00 = (float)(acc.Sxx * inv - mx * mx), M01 = (float)(acc.Sxy * inv - mx * my), M02 = (float)(acc.Sxz * inv - mx * mz),
            M11 = (float)(acc.Syy * inv - my * my), M12 = (float)(acc.Syz * inv - my * mz), M22 = (float)(acc.Szz * inv - mz * mz),
        };
        ScaleQuatFromCov(cov, out float sx, out float sy, out float sz, out var q);
        parent[0] = (float)mx; parent[1] = (float)my; parent[2] = (float)mz;
        parent[SplatFormat.OffColor] = (float)(acc.CR * inv);
        parent[SplatFormat.OffColor + 1] = (float)(acc.CG * inv);
        parent[SplatFormat.OffColor + 2] = (float)(acc.CB * inv);
        parent[SplatFormat.OffScale] = sx; parent[SplatFormat.OffScale + 1] = sy; parent[SplatFormat.OffScale + 2] = sz;
        float areaP = Area(sx, sy, sz);
        parent[SplatFormat.OffOpacity] = (float)Math.Clamp(acc.AlphaArea / Math.Max(areaP, 1e-20), 0.0, 1.0);
        parent[SplatFormat.OffQuat] = q.X; parent[SplatFormat.OffQuat + 1] = q.Y;
        parent[SplatFormat.OffQuat + 2] = q.Z; parent[SplatFormat.OffQuat + 3] = q.W;
    }

    /// <summary>Merge <paramref name="count"/> packed rows into one parent row (the CPU oracle).</summary>
    public static void Merge(ReadOnlySpan<float> rows, int count, Span<float> parent)
    {
        var acc = new Accum();
        for (int i = 0; i < count; i++) Add(ref acc, rows.Slice(i * F, F));
        Finish(acc, parent);
    }

    static SplatCovariance.Quat Normalize(SplatCovariance.Quat q)
    {
        float n = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
        if (!(n > 1e-20f)) return SplatCovariance.Quat.Identity;
        return new SplatCovariance.Quat { X = q.X / n, Y = q.Y / n, Z = q.Z / n, W = q.W / n };
    }

    /// <summary>
    /// A symmetric covariance as 1-sigma scales (ascending) and the rotation whose columns are the matching eigenvectors
    /// - the inverse of <see cref="SplatCovariance.Cov3DFromScaleQuat"/>. Cyclic Jacobi, a fixed number of sweeps (no
    /// data-dependent loop exit, so it suits a GPU kernel); a right-handed basis (det +1) so it is a rotation.
    /// </summary>
    public static void ScaleQuatFromCov(SplatCovariance.Cov3 c, out float sx, out float sy, out float sz, out SplatCovariance.Quat q)
    {
        double a00 = c.M00, a01 = c.M01, a02 = c.M02, a11 = c.M11, a12 = c.M12, a22 = c.M22;
        double v00 = 1, v01 = 0, v02 = 0, v10 = 0, v11 = 1, v12 = 0, v20 = 0, v21 = 0, v22 = 1;
        for (int sweep = 0; sweep < 8; sweep++)
        {
            Rotate(ref a00, ref a11, ref a01, ref a02, ref a12, ref v00, ref v01, ref v10, ref v11, ref v20, ref v21);   // (0,1)
            Rotate(ref a00, ref a22, ref a02, ref a01, ref a12, ref v00, ref v02, ref v10, ref v12, ref v20, ref v22);   // (0,2)
            Rotate(ref a11, ref a22, ref a12, ref a01, ref a02, ref v01, ref v02, ref v11, ref v12, ref v21, ref v22);   // (1,2)
        }
        // Eigenvalues a00, a11, a22 with eigenvectors the COLUMNS of V. Right-handed: flip column 2 if det < 0.
        double det = v00 * (v11 * v22 - v12 * v21) - v01 * (v10 * v22 - v12 * v20) + v02 * (v10 * v21 - v11 * v20);
        if (det < 0) { v02 = -v02; v12 = -v12; v22 = -v22; }
        sx = (float)Math.Sqrt(Math.Max(a00, 1e-24));
        sy = (float)Math.Sqrt(Math.Max(a11, 1e-24));
        sz = (float)Math.Sqrt(Math.Max(a22, 1e-24));
        q = QuatFromMatrix(v00, v01, v02, v10, v11, v12, v20, v21, v22);
    }

    /// <summary>
    /// One Jacobi rotation zeroing the off-diagonal a_pq of a symmetric 3x3 matrix (app, aqq, apq; apr/aqr the two
    /// entries coupling p and q to the third index r), accumulated into V's columns p and q.
    /// </summary>
    static void Rotate(ref double app, ref double aqq, ref double apq, ref double apr, ref double aqr,
        ref double v0p, ref double v0q, ref double v1p, ref double v1q, ref double v2p, ref double v2q)
    {
        if (Math.Abs(apq) < 1e-300) return;
        double theta = (aqq - app) / (2 * apq);
        double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
        if (theta == 0) t = 1;
        double cs = 1 / Math.Sqrt(t * t + 1), sn = t * cs;
        double nApp = app - t * apq, nAqq = aqq + t * apq;
        double nApr = cs * apr - sn * aqr, nAqr = sn * apr + cs * aqr;
        app = nApp; aqq = nAqq; apq = 0; apr = nApr; aqr = nAqr;
        double t0 = v0p, t1 = v1p, t2 = v2p;
        v0p = cs * t0 - sn * v0q; v0q = sn * t0 + cs * v0q;
        v1p = cs * t1 - sn * v1q; v1q = sn * t1 + cs * v1q;
        v2p = cs * t2 - sn * v2q; v2q = sn * t2 + cs * v2q;
    }

    /// <summary>Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method).</summary>
    static SplatCovariance.Quat QuatFromMatrix(double m00, double m01, double m02, double m10, double m11, double m12,
        double m20, double m21, double m22)
    {
        double tr = m00 + m11 + m22, x, y, z, w;
        if (tr > 0)
        {
            double s = Math.Sqrt(tr + 1) * 2;
            w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
        }
        else if (m00 > m11 && m00 > m22)
        {
            double s = Math.Sqrt(1 + m00 - m11 - m22) * 2;
            w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
        }
        else if (m11 > m22)
        {
            double s = Math.Sqrt(1 + m11 - m00 - m22) * 2;
            w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
        }
        else
        {
            double s = Math.Sqrt(1 + m22 - m00 - m11) * 2;
            w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
        }
        double n = Math.Sqrt(x * x + y * y + z * z + w * w);
        return new SplatCovariance.Quat { X = (float)(x / n), Y = (float)(y / n), Z = (float)(z / n), W = (float)(w / n) };
    }
}
