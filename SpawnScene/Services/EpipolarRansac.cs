namespace SpawnScene.Services;

/// <summary>
/// Geometric verification of a pair's feature matches: RANSAC over the fundamental matrix
/// (normalised 8-point, rank 2 enforced), inliers by Sampson distance in pixels.
///
/// Why: MEASURED 2026-09-24 on Truck, cross-checked ratio-test matches sit at a noise floor of ~34 per
/// pair even for frames that share nothing (adjacent frames: median 216). 5,565 far pairs x ~34 chance
/// matches poisoned bundle adjustment's tracks - union-find chained them into tracks that claim two
/// features in one image, and BA was left 5,197 points for 126 cameras. A pair's true matches all obey
/// one epipolar geometry; chance matches do not.
/// </summary>
public static class EpipolarRansac
{
    public sealed record Result(double[] F, bool[] Inliers, int InlierCount, int Iterations);

    /// <summary>
    /// <paramref name="a"/>/<paramref name="b"/> are matched pixel coordinates (x0,y0,x1,y1,...).
    /// Null when no model reaches <paramref name="minInliers"/>.
    /// </summary>
    public static Result? Estimate(ReadOnlySpan<float> a, ReadOnlySpan<float> b, double thresholdPx = 2.0,
        int minInliers = 15, int maxIterations = 1000, double confidence = 0.999, int seed = 1)
    {
        int n = a.Length / 2;
        if (n < 8 || n < minInliers) return null;
        var rng = new Random(seed);
        double th2 = thresholdPx * thresholdPx;
        int best = 0;
        double[]? bestF = null;
        Span<int> sample = stackalloc int[8];
        var f = new double[9];
        int it = 0, needed = maxIterations;
        var pa = a.ToArray();
        var pb = b.ToArray();
        for (; it < Math.Min(needed, maxIterations); it++)
        {
            for (int k = 0; k < 8; k++)
            {
                int s;
                do { s = rng.Next(n); } while (Contains(sample, k, s));
                sample[k] = s;
            }
            if (!EightPoint(pa, pb, sample, f)) continue;
            int count = 0;
            for (int i = 0; i < n; i++) if (Sampson2(f, pa, pb, i) <= th2) count++;
            if (count > best)
            {
                best = count;
                bestF = (double[])f.Clone();
                double w = (double)best / n;
                double p8 = Math.Pow(w, 8);
                needed = p8 >= 1 ? 1 : (int)Math.Ceiling(Math.Log(1 - confidence) / Math.Log(Math.Max(1e-12, 1 - p8)));
            }
        }
        if (bestF == null || best < minInliers) return null;

        // Refit on all inliers, then re-score (one refinement pass).
        var inl = new List<int>();
        for (int i = 0; i < n; i++) if (Sampson2(bestF, pa, pb, i) <= th2) inl.Add(i);
        if (inl.Count >= 8 && EightPoint(pa, pb, inl.ToArray(), f))
        {
            int c = 0;
            for (int i = 0; i < n; i++) if (Sampson2(f, pa, pb, i) <= th2) c++;
            if (c >= best) { bestF = (double[])f.Clone(); best = c; }
        }
        var mask = new bool[n];
        int total = 0;
        for (int i = 0; i < n; i++) if (mask[i] = Sampson2(bestF, pa, pb, i) <= th2) total++;
        return total < minInliers ? null : new Result(bestF, mask, total, it);
    }

    static bool Contains(Span<int> s, int k, int v)
    {
        for (int i = 0; i < k; i++) if (s[i] == v) return true;
        return false;
    }

    /// <summary>Squared Sampson distance of match i to F (pixels^2).</summary>
    public static double Sampson2(double[] f, float[] a, float[] b, int i)
    {
        double x1 = a[i * 2], y1 = a[i * 2 + 1], x2 = b[i * 2], y2 = b[i * 2 + 1];
        double fx0 = f[0] * x1 + f[1] * y1 + f[2];
        double fx1 = f[3] * x1 + f[4] * y1 + f[5];
        double fx2 = f[6] * x1 + f[7] * y1 + f[8];
        double ftx0 = f[0] * x2 + f[3] * y2 + f[6];
        double ftx1 = f[1] * x2 + f[4] * y2 + f[7];
        double e = x2 * fx0 + y2 * fx1 + fx2;
        double d = fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1;
        return d <= 1e-300 ? double.MaxValue : e * e / d;
    }

    /// <summary>Normalised 8-point (Hartley) over the given match indices, rank 2 enforced. F maps pixels.</summary>
    static bool EightPoint(float[] a, float[] b, ReadOnlySpan<int> idx, double[] fOut)
    {
        int m = idx.Length;
        // Hartley normalisation per image.
        Normaliser(a, idx, out double ca0, out double ca1, out double sa);
        Normaliser(b, idx, out double cb0, out double cb1, out double sb);
        Span<double> ata = stackalloc double[81];
        ata.Clear();
        Span<double> row = stackalloc double[9];
        foreach (int i in idx)
        {
            double x1 = (a[i * 2] - ca0) * sa, y1 = (a[i * 2 + 1] - ca1) * sa;
            double x2 = (b[i * 2] - cb0) * sb, y2 = (b[i * 2 + 1] - cb1) * sb;
            row[0] = x2 * x1; row[1] = x2 * y1; row[2] = x2; row[3] = y2 * x1; row[4] = y2 * y1; row[5] = y2;
            row[6] = x1; row[7] = y1; row[8] = 1;
            for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) ata[r * 9 + c] += row[r] * row[c];
        }
        Span<double> vecs = stackalloc double[81];
        Span<double> vals = stackalloc double[9];
        JacobiEigen(ata, vecs, vals, 9);
        int s = 0;
        for (int i = 1; i < 9; i++) if (vals[i] < vals[s]) s = i;
        Span<double> fn = stackalloc double[9];
        for (int i = 0; i < 9; i++) fn[i] = vecs[i * 9 + s];

        // Rank 2: F = U diag(s1, s2, 0) V^T via the eigen-decomposition of F^T F.
        Span<double> ftf = stackalloc double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            ftf[r * 3 + c] = fn[0 * 3 + r] * fn[0 * 3 + c] + fn[1 * 3 + r] * fn[1 * 3 + c] + fn[2 * 3 + r] * fn[2 * 3 + c];
        Span<double> v = stackalloc double[9];
        Span<double> ev = stackalloc double[3];
        JacobiEigen(ftf, v, ev, 3);
        int z = 0;
        for (int i = 1; i < 3; i++) if (ev[i] < ev[z]) z = i;
        // F2 = F (I - v_z v_z^T): removes the smallest singular direction exactly.
        Span<double> f2 = stackalloc double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                double acc = 0;
                for (int k = 0; k < 3; k++)
                {
                    double proj = (k == c ? 1 : 0) - v[k * 3 + z] * v[c * 3 + z];
                    acc += fn[r * 3 + k] * proj;
                }
                f2[r * 3 + c] = acc;
            }
        // Denormalise: F = Tb^T F2 Ta, T = [[s,0,-s cx],[0,s,-s cy],[0,0,1]].
        double[] ta = { sa, 0, -sa * ca0, 0, sa, -sa * ca1, 0, 0, 1 };
        double[] tb = { sb, 0, -sb * cb0, 0, sb, -sb * cb1, 0, 0, 1 };
        Span<double> tmp = stackalloc double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            tmp[r * 3 + c] = f2[r * 3] * ta[c] + f2[r * 3 + 1] * ta[3 + c] + f2[r * 3 + 2] * ta[6 + c];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            fOut[r * 3 + c] = tb[r] * tmp[c] + tb[3 + r] * tmp[3 + c] + tb[6 + r] * tmp[6 + c];
        foreach (var x in fOut) if (!double.IsFinite(x)) return false;
        return true;
    }

    static void Normaliser(float[] p, ReadOnlySpan<int> idx, out double cx, out double cy, out double s)
    {
        cx = 0; cy = 0;
        foreach (int i in idx) { cx += p[i * 2]; cy += p[i * 2 + 1]; }
        cx /= idx.Length; cy /= idx.Length;
        double d = 0;
        foreach (int i in idx) d += Math.Sqrt((p[i * 2] - cx) * (p[i * 2] - cx) + (p[i * 2 + 1] - cy) * (p[i * 2 + 1] - cy));
        d /= idx.Length;
        s = d > 1e-12 ? Math.Sqrt(2) / d : 1;
    }

    /// <summary>Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix (in place); eigenvectors in columns.</summary>
    static void JacobiEigen(Span<double> a, Span<double> v, Span<double> vals, int n)
    {
        v.Clear();
        for (int i = 0; i < n; i++) v[i * n + i] = 1;
        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
            if (off < 1e-24) break;
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
