using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Place one camera from 2D-3D correspondences (perspective-n-point), known intrinsics. RANSAC over a
/// 6-point DLT in normalised coordinates, the rotation recovered by polar decomposition, refit on inliers.
///
/// Why: MEASURED 2026-09-24 on Truck, bundle adjustment cannot rescue a camera the DAv3 cascade placed
/// 60-156 degrees wrong (views 41/123/79/38/83): with its tracks cut or corrupted, LM has nothing to pull
/// it with. Real SfM re-registers such a camera against the points everyone else agrees on - this.
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
        if (n < 6) return false;
        var rng = new Random(seed);
        Span<int> sample = stackalloc int[6];
        double[]? bestP = null;
        int best = 0;
        var p = new double[12];
        var all = new int[n];
        for (int i = 0; i < n; i++) all[i] = i;
        for (int it = 0; it < iterations; it++)
        {
            for (int k = 0; k < 6; k++)
            {
                int s;
                do { s = rng.Next(n); } while (Contains(sample, k, s));
                sample[k] = s;
            }
            if (!Dlt(world, pixels, intrinsics, sample, p)) continue;
            int c = CountInliers(p, world, pixels, intrinsics, thresholdPx, null);
            if (c > best) { best = c; bestP = (double[])p.Clone(); }
        }
        if (bestP == null || best < 6) return false;
        var mask = new bool[n];
        CountInliers(bestP, world, pixels, intrinsics, thresholdPx, mask);
        var idx = Enumerable.Range(0, n).Where(i => mask[i]).ToArray();
        if (idx.Length >= 6 && Dlt(world, pixels, intrinsics, idx, p))
        {
            int c = CountInliers(p, world, pixels, intrinsics, thresholdPx, null);
            if (c >= best) { bestP = (double[])p.Clone(); best = c; }
        }
        inliers = best;
        camera = ToCamera(bestP, intrinsics);
        return true;
    }

    static bool Contains(Span<int> s, int k, int v)
    {
        for (int i = 0; i < k; i++) if (s[i] == v) return true;
        return false;
    }

    /// <summary>P = [R | t] in normalised coordinates (rows right, down, forward), R orthonormal, det +1.</summary>
    static bool Dlt(IReadOnlyList<Vector3> w, IReadOnlyList<Vector2> px, CameraParams k, ReadOnlySpan<int> idx, double[] pOut)
    {
        // Hartley-normalise the world points too (centroid to origin, mean distance sqrt 3): an unconditioned
        // 12-column DLT found 122 of ~210 true inliers on the gate below and a 5 degree rotation error.
        double gx = 0, gy = 0, gz = 0;
        foreach (int i in idx) { gx += w[i].X; gy += w[i].Y; gz += w[i].Z; }
        gx /= idx.Length; gy /= idx.Length; gz /= idx.Length;
        double md = 0;
        foreach (int i in idx) md += Math.Sqrt((w[i].X - gx) * (w[i].X - gx) + (w[i].Y - gy) * (w[i].Y - gy) + (w[i].Z - gz) * (w[i].Z - gz));
        md /= idx.Length;
        double ws = md > 1e-12 ? Math.Sqrt(3) / md : 1;

        Span<double> ata = stackalloc double[144];
        ata.Clear();
        Span<double> row = stackalloc double[12];
        foreach (int i in idx)
        {
            double x = (px[i].X - k.CenterX) / k.FocalX, y = (px[i].Y - k.CenterY) / k.FocalY;
            double X = (w[i].X - gx) * ws, Y = (w[i].Y - gy) * ws, Z = (w[i].Z - gz) * ws;
            for (int pass = 0; pass < 2; pass++)
            {
                row.Clear();
                int o = pass == 0 ? 0 : 4;
                double q = pass == 0 ? x : y;
                row[o] = X; row[o + 1] = Y; row[o + 2] = Z; row[o + 3] = 1;
                row[8] = -q * X; row[9] = -q * Y; row[10] = -q * Z; row[11] = -q;
                for (int a = 0; a < 12; a++) for (int b = 0; b < 12; b++) ata[a * 12 + b] += row[a] * row[b];
            }
        }
        Span<double> vecs = stackalloc double[144];
        Span<double> vals = stackalloc double[12];
        JacobiEigen(ata, vecs, vals, 12);
        int sm = 0;
        for (int i = 1; i < 12; i++) if (vals[i] < vals[sm]) sm = i;
        Span<double> pn = stackalloc double[12];
        for (int i = 0; i < 12; i++) pn[i] = vecs[i * 12 + sm];
        // Undo the world normalisation: P = P' T, T = [s I | -s g].
        Span<double> pr = stackalloc double[12];
        for (int r = 0; r < 3; r++)
        {
            pr[r * 4] = pn[r * 4] * ws; pr[r * 4 + 1] = pn[r * 4 + 1] * ws; pr[r * 4 + 2] = pn[r * 4 + 2] * ws;
            pr[r * 4 + 3] = pn[r * 4 + 3] - ws * (pn[r * 4] * gx + pn[r * 4 + 1] * gy + pn[r * 4 + 2] * gz);
        }

        // M = pr[0..2; 4..6; 8..10] = lambda R. R = M (M^T M)^-1/2.
        Span<double> m = stackalloc double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) m[r * 3 + c] = pr[r * 4 + c];
        double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if (Math.Abs(det) < 1e-18) return false;
        double sign = det > 0 ? 1 : -1;
        for (int i = 0; i < 12; i++) pr[i] *= sign;
        for (int i = 0; i < 9; i++) m[i] *= sign;
        double lambda = Math.Cbrt(Math.Abs(det));

        Span<double> mtm = stackalloc double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            mtm[r * 3 + c] = m[0 * 3 + r] * m[0 * 3 + c] + m[1 * 3 + r] * m[1 * 3 + c] + m[2 * 3 + r] * m[2 * 3 + c];
        Span<double> v = stackalloc double[9];
        Span<double> e = stackalloc double[3];
        JacobiEigen(mtm, v, e, 3);
        Span<double> invSqrt = stackalloc double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
        {
            double acc = 0;
            for (int k2 = 0; k2 < 3; k2++) acc += v[r * 3 + k2] * (e[k2] > 1e-300 ? 1 / Math.Sqrt(e[k2]) : 0) * v[c * 3 + k2];
            invSqrt[r * 3 + c] = acc;
        }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
        {
            double acc = 0;
            for (int k2 = 0; k2 < 3; k2++) acc += m[r * 3 + k2] * invSqrt[k2 * 3 + c];
            pOut[r * 4 + c] = acc;
        }
        for (int r = 0; r < 3; r++) pOut[r * 4 + 3] = pr[r * 4 + 3] / lambda;

        // Cheirality: the points should be in FRONT. If most are behind, the whole solution is mirrored.
        int front = 0;
        foreach (int i in idx) if (pOut[8] * w[i].X + pOut[9] * w[i].Y + pOut[10] * w[i].Z + pOut[11] > 0) front++;
        if (front * 2 < idx.Length) return false;
        foreach (var val in pOut) if (!double.IsFinite(val)) return false;
        return true;
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
