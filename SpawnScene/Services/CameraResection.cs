using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Place one camera from 2D-3D correspondences (perspective-n-point), known intrinsics: RANSAC over a minimal
/// P3P solver, then Gauss-Newton on reprojection error over the inliers.
///
/// Why: MEASURED 2026-09-24 on Truck, bundle adjustment cannot rescue a camera the DAv3 cascade placed
/// 60-156 degrees wrong; real SfM re-registers such a camera against the points everyone else agrees on.
///
/// Why P3P and not a DLT: the first version sampled 6 points into a DLT, which is DEGENERATE for coplanar
/// points - and the side of a truck is a plane. On real Truck views (Data/truck_resection_view79/34) it found
/// 19/87 and 0/51 correspondences where OpenCV's solvePnPRansac finds 80/87 and 27/51, and the pipeline dropped
/// those views. P3P needs 3 points and has no planar degeneracy.
/// </summary>
public static class CameraResection
{
    public static bool ResectRansac(
        IReadOnlyList<Vector3> world, IReadOnlyList<Vector2> pixels, CameraParams intrinsics,
        out CameraParams camera, out int inliers,
        double thresholdPx = 4.0, int iterations = 500, int seed = 1)
    {
        camera = new CameraParams();
        inliers = 0;
        int n = world.Count;
        if (n < 4) return false;
        var rng = new Random(seed);
        // Bearings in the camera frame (x right, y down, z forward), unit length.
        var f = new double[n * 3];
        for (int i = 0; i < n; i++)
        {
            double x = (pixels[i].X - intrinsics.CenterX) / intrinsics.FocalX;
            double y = (pixels[i].Y - intrinsics.CenterY) / intrinsics.FocalY;
            double len = Math.Sqrt(x * x + y * y + 1);
            f[i * 3] = x / len; f[i * 3 + 1] = y / len; f[i * 3 + 2] = 1 / len;
        }
        var solutions = new List<double[]>();
        double[]? bestP = null;
        int best = 0;
        for (int it = 0; it < iterations; it++)
        {
            int i0 = rng.Next(n), i1, i2;
            do { i1 = rng.Next(n); } while (i1 == i0);
            do { i2 = rng.Next(n); } while (i2 == i0 || i2 == i1);
            solutions.Clear();
            P3P(world, f, i0, i1, i2, solutions);
            foreach (var p in solutions)
            {
                int c = CountInliers(p, world, pixels, intrinsics, thresholdPx, null);
                if (c > best) { best = c; bestP = p; }
            }
        }
        if (bestP == null || best < 4) return false;

        // Refine on the inliers (reprojection error), then re-score; twice, as the inlier set grows.
        for (int round = 0; round < 2; round++)
        {
            var mask = new bool[n];
            CountInliers(bestP, world, pixels, intrinsics, thresholdPx, mask);
            var refined = Refine(bestP, world, pixels, intrinsics, mask);
            int c = CountInliers(refined, world, pixels, intrinsics, thresholdPx, null);
            if (c >= best) { best = c; bestP = refined; } else break;
        }
        inliers = best;
        camera = ToCamera(bestP, intrinsics);
        return true;
    }

    /// <summary>
    /// All real P3P poses for correspondences i0,i1,i2 as [R|t] (row-major 3x4, world -> camera).
    /// Depths s_k along bearings f_k satisfy the three law-of-cosines constraints. With u = s2/s1, v = s3/s1,
    /// the (1,2) and (1,3) constraints give v in closed form from u (two branches); the (2,3) constraint is then
    /// a scalar residual in u, scanned over (0, inf) via u = tan(theta) and bisected at every sign change.
    /// </summary>
    public static void P3P(IReadOnlyList<Vector3> w, double[] f, int i0, int i1, int i2, List<double[]> outPoses)
    {
        var p1 = w[i0]; var p2 = w[i1]; var p3 = w[i2];
        double a2 = Vector3.DistanceSquared(p2, p3), b2 = Vector3.DistanceSquared(p1, p3), c2 = Vector3.DistanceSquared(p1, p2);
        if (a2 < 1e-18 || b2 < 1e-18 || c2 < 1e-18) return;
        double cA = Dot(f, i1, i2), cB = Dot(f, i0, i2), cG = Dot(f, i0, i1);

        // For a given u: v from b^2 (1+u^2-2u cG) = c^2 (1+v^2-2v cB), residual of a^2 (..) = c^2 (u^2+v^2-2uv cA).
        double Res(double u, int branch, out double v)
        {
            double k = b2 / c2 * (1 + u * u - 2 * u * cG);
            double disc = cB * cB - 1 + k;
            v = double.NaN;
            if (disc < 0) return double.NaN;
            v = cB + branch * Math.Sqrt(disc);
            if (v <= 0) return double.NaN;
            return (a2 * (1 + u * u - 2 * u * cG) - c2 * (u * u + v * v - 2 * u * v * cA)) / c2;
        }

        const int steps = 1500;
        foreach (int branch in new[] { 1, -1 })
        {
            double prevU = 0, prevR = double.NaN;
            for (int s = 1; s < steps; s++)
            {
                double u = Math.Tan(s * (Math.PI / 2) / steps);
                double r = Res(u, branch, out _);
                if (double.IsFinite(r) && double.IsFinite(prevR) && Math.Sign(r) != Math.Sign(prevR))
                {
                    double lo = prevU, hi = u, rlo = prevR;
                    for (int b = 0; b < 60; b++)
                    {
                        double mid = 0.5 * (lo + hi);
                        double rm = Res(mid, branch, out _);
                        if (!double.IsFinite(rm)) break;
                        if (Math.Sign(rm) == Math.Sign(rlo)) { lo = mid; rlo = rm; } else hi = mid;
                    }
                    double uu = 0.5 * (lo + hi);
                    Res(uu, branch, out double vv);
                    if (double.IsFinite(vv)) AddPose(w, f, i0, i1, i2, uu, vv, c2, cG, outPoses);
                }
                prevU = u; prevR = r;
            }
        }
    }

    static double Dot(double[] f, int i, int j) => f[i * 3] * f[j * 3] + f[i * 3 + 1] * f[j * 3 + 1] + f[i * 3 + 2] * f[j * 3 + 2];

    static void AddPose(IReadOnlyList<Vector3> w, double[] f, int i0, int i1, int i2, double u, double v,
        double c2, double cG, List<double[]> outPoses)
    {
        double d = 1 + u * u - 2 * u * cG;
        if (d <= 1e-18) return;
        double s1 = Math.Sqrt(c2 / d), s2 = u * s1, s3 = v * s1;
        Span<double> q = stackalloc double[9];
        int[] ids = { i0, i1, i2 };
        double[] sc = { s1, s2, s3 };
        for (int k = 0; k < 3; k++) for (int a = 0; a < 3; a++) q[k * 3 + a] = sc[k] * f[ids[k] * 3 + a];
        Span<double> p = stackalloc double[9];
        for (int k = 0; k < 3; k++) { p[k * 3] = w[ids[k]].X; p[k * 3 + 1] = w[ids[k]].Y; p[k * 3 + 2] = w[ids[k]].Z; }
        var pose = new double[12];
        if (AbsoluteOrientation(p, q, 3, pose)) outPoses.Add(pose);
    }

    /// <summary>
    /// Horn's closed-form absolute orientation: R, t with Q = R P + t. The quaternion is the eigenvector of the
    /// LARGEST eigenvalue of Horn's N matrix, taken by cyclic Jacobi (power iteration can return the most
    /// negative one - fb-power-iteration-returns-worst-rotation).
    /// </summary>
    static bool AbsoluteOrientation(ReadOnlySpan<double> p, ReadOnlySpan<double> q, int n, double[] pose)
    {
        double pcx = 0, pcy = 0, pcz = 0, qcx = 0, qcy = 0, qcz = 0;
        for (int k = 0; k < n; k++)
        {
            pcx += p[k * 3]; pcy += p[k * 3 + 1]; pcz += p[k * 3 + 2];
            qcx += q[k * 3]; qcy += q[k * 3 + 1]; qcz += q[k * 3 + 2];
        }
        pcx /= n; pcy /= n; pcz /= n; qcx /= n; qcy /= n; qcz /= n;
        double sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;
        for (int k = 0; k < n; k++)
        {
            double px = p[k * 3] - pcx, py = p[k * 3 + 1] - pcy, pz = p[k * 3 + 2] - pcz;
            double qx = q[k * 3] - qcx, qy = q[k * 3 + 1] - qcy, qz = q[k * 3 + 2] - qcz;
            sxx += px * qx; sxy += px * qy; sxz += px * qz;
            syx += py * qx; syy += py * qy; syz += py * qz;
            szx += pz * qx; szy += pz * qy; szz += pz * qz;
        }
        Span<double> nm = stackalloc double[16]
        {
            sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
            syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
            szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
            sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
        };
        Span<double> vec = stackalloc double[16];
        Span<double> val = stackalloc double[4];
        JacobiEigen(nm, vec, val, 4);
        int m = 0;
        for (int i = 1; i < 4; i++) if (val[i] > val[m]) m = i;
        double qw = vec[0 * 4 + m], qx2 = vec[1 * 4 + m], qy2 = vec[2 * 4 + m], qz2 = vec[3 * 4 + m];
        double r00 = qw * qw + qx2 * qx2 - qy2 * qy2 - qz2 * qz2, r01 = 2 * (qx2 * qy2 - qw * qz2), r02 = 2 * (qx2 * qz2 + qw * qy2);
        double r10 = 2 * (qy2 * qx2 + qw * qz2), r11 = qw * qw - qx2 * qx2 + qy2 * qy2 - qz2 * qz2, r12 = 2 * (qy2 * qz2 - qw * qx2);
        double r20 = 2 * (qz2 * qx2 - qw * qy2), r21 = 2 * (qz2 * qy2 + qw * qx2), r22 = qw * qw - qx2 * qx2 - qy2 * qy2 + qz2 * qz2;
        pose[0] = r00; pose[1] = r01; pose[2] = r02; pose[3] = qcx - (r00 * pcx + r01 * pcy + r02 * pcz);
        pose[4] = r10; pose[5] = r11; pose[6] = r12; pose[7] = qcy - (r10 * pcx + r11 * pcy + r12 * pcz);
        pose[8] = r20; pose[9] = r21; pose[10] = r22; pose[11] = qcz - (r20 * pcx + r21 * pcy + r22 * pcz);
        foreach (var x in pose) if (!double.IsFinite(x)) return false;
        return true;
    }

    /// <summary>Gauss-Newton on reprojection error over the masked correspondences (rotation vector + translation).</summary>
    static double[] Refine(double[] pose, IReadOnlyList<Vector3> w, IReadOnlyList<Vector2> px, CameraParams k, bool[] mask)
    {
        var cur = (double[])pose.Clone();
        Span<double> jtj = stackalloc double[36];
        Span<double> jtr = stackalloc double[6];
        Span<double> e = stackalloc double[9];
        Span<double> ju = stackalloc double[6];
        Span<double> jv = stackalloc double[6];
        Span<double> d = stackalloc double[6];
        for (int it = 0; it < 15; it++)
        {
            jtj.Clear(); jtr.Clear();
            for (int i = 0; i < w.Count; i++)
            {
                if (!mask[i]) continue;
                double X = w[i].X, Y = w[i].Y, Z = w[i].Z;
                double xc = cur[0] * X + cur[1] * Y + cur[2] * Z + cur[3];
                double yc = cur[4] * X + cur[5] * Y + cur[6] * Z + cur[7];
                double zc = cur[8] * X + cur[9] * Y + cur[10] * Z + cur[11];
                if (zc <= 1e-9) continue;
                double iz = 1 / zc, iz2 = iz * iz;
                double ru = k.FocalX * xc * iz + k.CenterX - px[i].X;
                double rv = k.FocalY * yc * iz + k.CenterY - px[i].Y;
                double a0 = k.FocalX * iz, a2 = -k.FocalX * xc * iz2, b1 = k.FocalY * iz, b2 = -k.FocalY * yc * iz2;
                // Left-multiplied rotation update: dXc/dw = -[Xc]x ; dXc/dt = I
                ju[0] = a2 * yc; ju[1] = a0 * zc - a2 * xc; ju[2] = -a0 * yc; ju[3] = a0; ju[4] = 0; ju[5] = a2;
                jv[0] = -b1 * zc + b2 * yc; jv[1] = -b2 * xc; jv[2] = b1 * xc; jv[3] = 0; jv[4] = b1; jv[5] = b2;
                for (int a = 0; a < 6; a++)
                {
                    jtr[a] += ju[a] * ru + jv[a] * rv;
                    for (int b = 0; b < 6; b++) jtj[a * 6 + b] += ju[a] * ju[b] + jv[a] * jv[b];
                }
            }
            for (int a = 0; a < 6; a++) jtj[a * 7] += 1e-9 + 1e-6 * jtj[a * 7];
            if (!SolveSpd(jtj, jtr, d, 6)) break;
            // step = -d
            Rodrigues(-d[0], -d[1], -d[2], e);
            var next = new double[12];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                    next[r * 4 + c] = e[r * 3] * cur[c] + e[r * 3 + 1] * cur[4 + c] + e[r * 3 + 2] * cur[8 + c];
                next[r * 4 + 3] = e[r * 3] * cur[3] + e[r * 3 + 1] * cur[7] + e[r * 3 + 2] * cur[11] - d[3 + r];
            }
            cur = next;
            if (Math.Abs(d[0]) + Math.Abs(d[1]) + Math.Abs(d[2]) < 1e-10) break;
        }
        return cur;
    }

    static bool SolveSpd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> x, int n)
    {
        Span<double> l = stackalloc double[n * n];
        l.Clear();
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double s = a[i * n + j];
                for (int k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
                if (i == j) { if (!(s > 0)) return false; l[i * n + i] = Math.Sqrt(s); }
                else l[i * n + j] = s / l[j * n + j];
            }
        Span<double> y = stackalloc double[n];
        for (int i = 0; i < n; i++) { double s = b[i]; for (int k = 0; k < i; k++) s -= l[i * n + k] * y[k]; y[i] = s / l[i * n + i]; }
        for (int i = n - 1; i >= 0; i--) { double s = y[i]; for (int k = i + 1; k < n; k++) s -= l[k * n + i] * x[k]; x[i] = s / l[i * n + i]; }
        return true;
    }

    static void Rodrigues(double wx, double wy, double wz, Span<double> r)
    {
        double th = Math.Sqrt(wx * wx + wy * wy + wz * wz);
        if (th < 1e-12)
        {
            r[0] = 1; r[1] = -wz; r[2] = wy; r[3] = wz; r[4] = 1; r[5] = -wx; r[6] = -wy; r[7] = wx; r[8] = 1;
            return;
        }
        double kx = wx / th, ky = wy / th, kz = wz / th, c = Math.Cos(th), s = Math.Sin(th), t = 1 - c;
        r[0] = c + kx * kx * t; r[1] = kx * ky * t - kz * s; r[2] = kx * kz * t + ky * s;
        r[3] = ky * kx * t + kz * s; r[4] = c + ky * ky * t; r[5] = ky * kz * t - kx * s;
        r[6] = kz * kx * t - ky * s; r[7] = kz * ky * t + kx * s; r[8] = c + kz * kz * t;
    }

    static int CountInliers(double[] p, IReadOnlyList<Vector3> w, IReadOnlyList<Vector2> px, CameraParams k, double th, bool[]? mask)
    {
        int c = 0;
        double th2 = th * th;
        for (int i = 0; i < w.Count; i++)
        {
            double X = w[i].X, Y = w[i].Y, Z = w[i].Z;
            double zc = p[8] * X + p[9] * Y + p[10] * Z + p[11];
            bool ok = false;
            if (zc > 1e-9)
            {
                double u = k.FocalX * (p[0] * X + p[1] * Y + p[2] * Z + p[3]) / zc + k.CenterX;
                double vv = k.FocalY * (p[4] * X + p[5] * Y + p[6] * Z + p[7]) / zc + k.CenterY;
                double du = u - px[i].X, dv = vv - px[i].Y;
                ok = du * du + dv * dv <= th2;
            }
            if (mask != null) mask[i] = ok;
            if (ok) c++;
        }
        return c;
    }

    static CameraParams ToCamera(double[] p, CameraParams k)
    {
        var right = new Vector3((float)p[0], (float)p[1], (float)p[2]);
        var down = new Vector3((float)p[4], (float)p[5], (float)p[6]);
        var fwd = new Vector3((float)p[8], (float)p[9], (float)p[10]);
        var t = new Vector3((float)p[3], (float)p[7], (float)p[11]);
        // X_c = R X + t  ->  C = -R^T t
        var c = -(right * t.X + down * t.Y + fwd * t.Z);
        return new CameraParams
        {
            Width = k.Width, Height = k.Height, FocalX = k.FocalX, FocalY = k.FocalY, CenterX = k.CenterX, CenterY = k.CenterY,
            Near = k.Near, Far = k.Far,
            Position = c, Forward = Vector3.Normalize(fwd), Up = Vector3.Normalize(-down),
        };
    }

    static void JacobiEigen(Span<double> a, Span<double> v, Span<double> vals, int n)
    {
        v.Clear();
        for (int i = 0; i < n; i++) v[i * n + i] = 1;
        for (int sweep = 0; sweep < 60; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
            if (off < 1e-26) break;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
                    double t = theta == 0 ? 1 : Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k * n + p], akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p * n + k], aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k * n + p], vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
                    }
                }
        }
        for (int i = 0; i < n; i++) vals[i] = a[i * n + i];
    }
}
