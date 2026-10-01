namespace SpawnScene.Services;

/// <summary>
/// Calibrated relative pose: Nistér's five-point essential-matrix solver ("An Efficient Solution to the Five-Point Relative
/// Pose Problem", PAMI 2004) and an E-RANSAC around it.
///
/// Why (2026-10-01, DrJohnson, the learned front end's real matches, COLMAP truth): relative rotations from F-RANSAC +
/// E = K^T F K were 8.25 deg off on true pairs (median; p75 28); refining on the F inliers fixed the median (1.30) but not
/// the tail (p75 22) - F-RANSAC lets planar-degenerate hypotheses win, and DrJohnson is walls. 5-point E-RANSAC + the same
/// refinement: 0.94 / 3.57. Through the rotation stage that is 35 of 44 cameras at 1.15 deg (p90 2.09, none above 5) against
/// 30 at 3.11 (seven above 5). An 8-point hypothesis projected onto the essential manifold did NOT help (8.07 median): the
/// five-point minimal set is what matters.
///
/// Conventions: rays are normalized image coordinates (x, y, 1); x_b^T E x_a = 0; E is row-major.
/// </summary>
public static class FivePoint
{
    // Nistér's monomial order: the first 10 are eliminated by Gauss-Jordan; the last 10 are the basis
    // [x z^2, x z, x, y z^2, y z, y, z^3, z^2, z, 1].
    static readonly (int X, int Y, int Z)[] Mono =
    {
        (3, 0, 0), (0, 3, 0), (2, 1, 0), (1, 2, 0), (2, 0, 1), (2, 0, 0), (0, 2, 1), (0, 2, 0), (1, 1, 1), (1, 1, 0),
        (1, 0, 2), (1, 0, 1), (1, 0, 0), (0, 1, 2), (0, 1, 1), (0, 1, 0), (0, 0, 3), (0, 0, 2), (0, 0, 1), (0, 0, 0),
    };

    // Polynomials in x, y, z of total degree <= 3 over 20 monomials, ordered by degree (1 | x y z | 6 quadratics | 10 cubics).
    // Mul only visits monomials up to each factor's degree and looks products up in a precomputed table.
    static readonly (int X, int Y, int Z)[] All = BuildMonomials();
    static readonly int[] DegStart = { 0, 1, 4, 10, 20 };
    static readonly int[,] ProductIndex = BuildProducts();
    static readonly int[] MonoIndex = Mono.Select(m => IndexOf(m.X, m.Y, m.Z)).ToArray();

    static (int, int, int)[] BuildMonomials()
    {
        var l = new List<(int, int, int)>();
        for (int d = 0; d <= 3; d++)
            for (int x = d; x >= 0; x--)
                for (int y = d - x; y >= 0; y--)
                    l.Add((x, y, d - x - y));
        return l.ToArray();
    }

    static int IndexOf(int x, int y, int z)
    {
        for (int i = 0; i < 20; i++) if (All[i] == (x, y, z)) return i;
        return -1;
    }

    static int[,] BuildProducts()
    {
        var t = new int[20, 20];
        for (int i = 0; i < 20; i++)
            for (int j = 0; j < 20; j++)
            {
                var (ax, ay, az) = All[i]; var (bx, by, bz) = All[j];
                t[i, j] = ax + ay + az + bx + by + bz > 3 ? -1 : IndexOf(ax + bx, ay + by, az + bz);
            }
        return t;
    }

    readonly struct P
    {
        public readonly double[] C;
        public readonly int Deg;
        public P(double[] c, int deg) { C = c; Deg = deg; }
        public static P Linear(double cx, double cy, double cz, double c1)
        {
            var c = new double[20];
            c[0] = c1; c[1] = cx; c[2] = cy; c[3] = cz;   // All[1..3] = x, y, z
            return new P(c, 1);
        }
        public static P operator *(P a, P b)
        {
            var c = new double[20];
            int na = DegStart[a.Deg + 1], nb = DegStart[b.Deg + 1];
            for (int i = 0; i < na; i++)
            {
                double ai = a.C[i];
                if (ai == 0) continue;
                for (int j = 0; j < nb; j++)
                {
                    int k = ProductIndex[i, j];
                    if (k >= 0) c[k] += ai * b.C[j];
                }
            }
            return new P(c, Math.Min(3, a.Deg + b.Deg));
        }
        public static P operator +(P a, P b) { var c = new double[20]; for (int i = 0; i < 20; i++) c[i] = a.C[i] + b.C[i]; return new P(c, Math.Max(a.Deg, b.Deg)); }
        public static P operator -(P a, P b) { var c = new double[20]; for (int i = 0; i < 20; i++) c[i] = a.C[i] - b.C[i]; return new P(c, Math.Max(a.Deg, b.Deg)); }
        public static P operator *(double s, P a) { var c = new double[20]; for (int i = 0; i < 20; i++) c[i] = s * a.C[i]; return new P(c, a.Deg); }
    }

    /// <summary>
    /// Every essential matrix (up to 10, row-major, unit Frobenius norm) consistent with the 5 correspondences
    /// <paramref name="ra"/>[i] -> <paramref name="rb"/>[i] (normalized rays, z = 1).
    /// </summary>
    public static List<double[]> Solve(ReadOnlySpan<(double X, double Y)> ra, ReadOnlySpan<(double X, double Y)> rb)
    {
        var sols = new List<double[]>();
        if (ra.Length != 5 || rb.Length != 5) throw new ArgumentException("exactly five correspondences");
        // 1. Null space of the 5 x 9 epipolar constraints: x_b^T E x_a = sum_ij xb_i E_ij xa_j.
        var q = new double[9 * 9];
        for (int i = 0; i < 5; i++)
        {
            double[] a = { ra[i].X, ra[i].Y, 1 }, b = { rb[i].X, rb[i].Y, 1 };
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) q[i * 9 + r * 3 + c] = b[r] * a[c];
        }
        var v = NullSpace4(q);   // 4 basis vectors of length 9: X, Y, Z, W
        // E = x X + y Y + z Z + W
        var e = new P[9];
        for (int k = 0; k < 9; k++) e[k] = P.Linear(v[0][k], v[1][k], v[2][k], v[3][k]);
        // 2. The ten cubic constraints: det(E) = 0 and 2 E E^T E - tr(E E^T) E = 0.
        var rows = new List<P>(10);
        rows.Add(e[0] * (e[4] * e[8] - e[5] * e[7]) - e[1] * (e[3] * e[8] - e[5] * e[6]) + e[2] * (e[3] * e[7] - e[4] * e[6]));
        var eet = new P[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                eet[r * 3 + c] = e[r * 3] * e[c * 3] + e[r * 3 + 1] * e[c * 3 + 1] + e[r * 3 + 2] * e[c * 3 + 2];
        var tr = eet[0] + eet[4] + eet[8];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                var s = eet[r * 3] * e[c] + eet[r * 3 + 1] * e[3 + c] + eet[r * 3 + 2] * e[6 + c];
                rows.Add(2.0 * s - tr * e[r * 3 + c]);
            }
        var a10 = new double[10 * 20];
        for (int r = 0; r < 10; r++)
            for (int m = 0; m < 20; m++) a10[r * 20 + m] = rows[r].C[MonoIndex[m]];
        // 3. Gauss-Jordan on the first 10 columns: row r becomes Mono[r] + (combination of the basis monomials) = 0.
        if (!GaussJordan(a10, 10, 20)) return sols;
        // 4. Nistér's hidden-variable resultant in z: <k> = <e> - z<f>, <l> = <g> - z<h>, <m> = <i> - z<j>, rows 4..9.
        //    Each is x * px(z) + y * py(z) + p1(z); coefficients ascending in z.
        var kx = new double[4]; var ky = new double[4]; var k1 = new double[5];
        var lx = new double[4]; var ly = new double[4]; var l1 = new double[5];
        var mx = new double[4]; var my = new double[4]; var m1 = new double[5];
        Resultant(a10, 4, 5, kx, ky, k1);
        Resultant(a10, 6, 7, lx, ly, l1);
        Resultant(a10, 8, 9, mx, my, m1);
        // det [[kx ky k1] [lx ly l1] [mx my m1]] = degree-10 polynomial in z.
        var n = PolyAdd(PolyAdd(
            PolyMul(kx, PolySub(PolyMul(ly, m1), PolyMul(l1, my))),
            PolyMul(ky, PolySub(PolyMul(l1, mx), PolyMul(lx, m1)))),
            PolyMul(k1, PolySub(PolyMul(lx, my), PolyMul(ly, mx))));
        foreach (double z in RealRoots(n))
        {
            double ex = Eval(kx, z), ey = Eval(ky, z), e1 = Eval(k1, z);
            double fx = Eval(lx, z), fy = Eval(ly, z), f1 = Eval(l1, z);
            double gx = Eval(mx, z), gy = Eval(my, z), g1 = Eval(m1, z);
            // (x, y, 1) is the null vector of the 3x3: the cross product of two rows (the best-conditioned pair).
            var c1 = Cross(ex, ey, e1, fx, fy, f1);
            var c2 = Cross(ex, ey, e1, gx, gy, g1);
            var c3 = Cross(fx, fy, f1, gx, gy, g1);
            var c = c1;
            if (Math.Abs(c2.Z) > Math.Abs(c.Z)) c = c2;
            if (Math.Abs(c3.Z) > Math.Abs(c.Z)) c = c3;
            if (Math.Abs(c.Z) < 1e-14) continue;
            double x = c.X / c.Z, y = c.Y / c.Z;
            var em = new double[9];
            double norm = 0;
            for (int k = 0; k < 9; k++) { em[k] = x * v[0][k] + y * v[1][k] + z * v[2][k] + v[3][k]; norm += em[k] * em[k]; }
            norm = Math.Sqrt(norm);
            if (!(norm > 0)) continue;
            for (int k = 0; k < 9; k++) em[k] /= norm;
            sols.Add(em);
        }
        return sols;
    }

    static (double X, double Y, double Z) Cross(double a0, double a1, double a2, double b0, double b1, double b2)
        => (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0);

    /// <summary>Row <paramref name="re"/> minus z times row <paramref name="rf"/>, split into the x, y and constant parts.</summary>
    static void Resultant(double[] a, int re, int rf, double[] px, double[] py, double[] p1)
    {
        // Basis columns 10..19: x z^2, x z, x, y z^2, y z, y, z^3, z^2, z, 1. After Gauss-Jordan the row reads
        // lead + sum b_k basis_k = 0, so the polynomial is sum b_k basis_k (the lead cancels in e - z f).
        double B(int row, int k) => a[row * 20 + 10 + k];
        // e part (ascending powers of z)
        px[0] = B(re, 2); px[1] = B(re, 1); px[2] = B(re, 0); px[3] = 0;
        py[0] = B(re, 5); py[1] = B(re, 4); py[2] = B(re, 3); py[3] = 0;
        p1[0] = B(re, 9); p1[1] = B(re, 8); p1[2] = B(re, 7); p1[3] = B(re, 6); p1[4] = 0;
        // minus z * f part
        px[1] -= B(rf, 2); px[2] -= B(rf, 1); px[3] -= B(rf, 0);
        py[1] -= B(rf, 5); py[2] -= B(rf, 4); py[3] -= B(rf, 3);
        p1[1] -= B(rf, 9); p1[2] -= B(rf, 8); p1[3] -= B(rf, 7); p1[4] -= B(rf, 6);
    }

    static double[] PolyMul(double[] a, double[] b)
    {
        var r = new double[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++) for (int j = 0; j < b.Length; j++) r[i + j] += a[i] * b[j];
        return r;
    }
    static double[] PolyAdd(double[] a, double[] b)
    {
        var r = new double[Math.Max(a.Length, b.Length)];
        for (int i = 0; i < a.Length; i++) r[i] += a[i];
        for (int i = 0; i < b.Length; i++) r[i] += b[i];
        return r;
    }
    static double[] PolySub(double[] a, double[] b)
    {
        var r = new double[Math.Max(a.Length, b.Length)];
        for (int i = 0; i < a.Length; i++) r[i] += a[i];
        for (int i = 0; i < b.Length; i++) r[i] -= b[i];
        return r;
    }
    static double Eval(double[] p, double z) { double s = 0; for (int i = p.Length - 1; i >= 0; i--) s = s * z + p[i]; return s; }

    /// <summary>
    /// The real roots of the polynomial (ascending coefficients): Sturm sequence to isolate them, bisection, then Newton
    /// polish. Degree <= 10 here.
    /// </summary>
    public static List<double> RealRoots(double[] coeffs)
    {
        var roots = new List<double>();
        int deg = coeffs.Length - 1;
        double maxAbs = 0;
        foreach (var c in coeffs) maxAbs = Math.Max(maxAbs, Math.Abs(c));
        while (deg > 0 && Math.Abs(coeffs[deg]) <= 1e-14 * maxAbs) deg--;
        if (deg < 1) return roots;
        var p = new double[deg + 1];
        for (int i = 0; i <= deg; i++) p[i] = coeffs[i] / coeffs[deg];   // monic
        // Cauchy bound.
        double bound = 0;
        for (int i = 0; i < deg; i++) bound = Math.Max(bound, Math.Abs(p[i]));
        bound += 1;
        // Sturm sequence p0 = p, p1 = p', p_{k+1} = -rem(p_{k-1}, p_k).
        var seq = new List<double[]> { p, Derivative(p) };
        while (seq[^1].Length > 1)
        {
            var r = Remainder(seq[^2], seq[^1]);
            for (int i = 0; i < r.Length; i++) r[i] = -r[i];
            if (r.Length == 0 || r.All(c => Math.Abs(c) < 1e-300)) break;
            seq.Add(r);
        }
        int SignChanges(double x)
        {
            int changes = 0; double last = 0;
            foreach (var s in seq)
            {
                double val = Eval(s, x);
                if (val == 0) continue;
                if (last != 0 && Math.Sign(val) != Math.Sign(last)) changes++;
                last = val;
            }
            return changes;
        }
        void Isolate(double lo, double hi, int clo, int chi, int depth)
        {
            int count = clo - chi;
            if (count <= 0) return;
            if (count == 1 || depth > 60 || hi - lo < 1e-12)
            {
                // One root in (lo, hi]: bisect on the sign of p (Sturm counts distinct roots, so p changes sign here
                // unless the root is of even multiplicity, which generic data does not produce).
                double a = lo, b = hi, fa = Eval(p, a), fb = Eval(p, b);
                if (Math.Sign(fa) == Math.Sign(fb)) { if (count == 1) roots.Add(0.5 * (a + b)); return; }
                for (int it = 0; it < 60 && b - a > 1e-9 * Math.Max(1, Math.Abs(a)); it++)   // Newton finishes below
                {
                    double m = 0.5 * (a + b), fm = Eval(p, m);
                    if (Math.Sign(fm) == Math.Sign(fa)) { a = m; fa = fm; } else { b = m; fb = fm; }
                }
                double x = 0.5 * (a + b);
                var d = Derivative(p);
                for (int it = 0; it < 4; it++) { double dv = Eval(d, x); if (dv == 0) break; double nx = x - Eval(p, x) / dv; if (nx >= lo && nx <= hi) x = nx; }
                roots.Add(x);
                return;
            }
            double mid = 0.5 * (lo + hi);
            int cm = SignChanges(mid);
            Isolate(lo, mid, clo, cm, depth + 1);
            Isolate(mid, hi, cm, chi, depth + 1);
        }
        Isolate(-bound, bound, SignChanges(-bound), SignChanges(bound), 0);
        return roots;
    }

    static double[] Derivative(double[] p)
    {
        if (p.Length <= 1) return new double[] { 0 };
        var d = new double[p.Length - 1];
        for (int i = 1; i < p.Length; i++) d[i - 1] = i * p[i];
        return d;
    }

    /// <summary>Remainder of a / b (ascending coefficients), trimmed of a vanishing leading term.</summary>
    static double[] Remainder(double[] a, double[] b)
    {
        int db = b.Length - 1;
        while (db > 0 && b[db] == 0) db--;
        var r = (double[])a.Clone();
        for (int i = r.Length - 1; i >= db; i--)
        {
            double f = r[i] / b[db];
            for (int j = 0; j <= db; j++) r[i - db + j] -= f * b[j];
        }
        int len = Math.Max(1, db);
        var o = new double[len];
        Array.Copy(r, o, Math.Min(len, r.Length));
        double scale = 0;
        foreach (var c in o) scale = Math.Max(scale, Math.Abs(c));
        int top = len - 1;
        while (top > 0 && Math.Abs(o[top]) <= 1e-13 * scale) top--;
        if (top < len - 1) Array.Resize(ref o, top + 1);
        return o;
    }

    /// <summary>Gauss-Jordan with partial pivoting on the first <paramref name="rows"/> columns (in place).</summary>
    static bool GaussJordan(double[] a, int rows, int cols)
    {
        for (int c = 0; c < rows; c++)
        {
            int piv = c; double best = Math.Abs(a[c * cols + c]);
            for (int r = c + 1; r < rows; r++) if (Math.Abs(a[r * cols + c]) > best) { best = Math.Abs(a[r * cols + c]); piv = r; }
            if (best < 1e-14) return false;
            if (piv != c) for (int k = 0; k < cols; k++) (a[c * cols + k], a[piv * cols + k]) = (a[piv * cols + k], a[c * cols + k]);
            double inv = 1 / a[c * cols + c];
            for (int k = 0; k < cols; k++) a[c * cols + k] *= inv;
            for (int r = 0; r < rows; r++)
            {
                if (r == c) continue;
                double f = a[r * cols + c];
                if (f == 0) continue;
                for (int k = 0; k < cols; k++) a[r * cols + k] -= f * a[c * cols + k];
            }
        }
        return true;
    }

    /// <summary>The 4-dimensional null space of the 5 x 9 constraint matrix (first 45 entries of <paramref name="q"/>):
    /// reduced row echelon form with full column pivoting, one basis vector per free column, orthonormalized.</summary>
    static double[][] NullSpace4(double[] q)
    {
        var m = new double[45];
        Array.Copy(q, m, 45);
        var pivotCol = new int[5];
        var isPivot = new bool[9];
        for (int r = 0; r < 5; r++)
        {
            int bc = -1, br = -1; double best = 0;
            for (int rr = r; rr < 5; rr++)
                for (int c = 0; c < 9; c++)
                    if (!isPivot[c] && Math.Abs(m[rr * 9 + c]) > best) { best = Math.Abs(m[rr * 9 + c]); bc = c; br = rr; }
            if (bc < 0 || best < 1e-14) { pivotCol[r] = -1; continue; }
            if (br != r) for (int k = 0; k < 9; k++) (m[r * 9 + k], m[br * 9 + k]) = (m[br * 9 + k], m[r * 9 + k]);
            double inv = 1 / m[r * 9 + bc];
            for (int k = 0; k < 9; k++) m[r * 9 + k] *= inv;
            for (int rr = 0; rr < 5; rr++)
            {
                if (rr == r) continue;
                double f = m[rr * 9 + bc];
                if (f == 0) continue;
                for (int k = 0; k < 9; k++) m[rr * 9 + k] -= f * m[r * 9 + k];
            }
            pivotCol[r] = bc; isPivot[bc] = true;
        }
        var res = new double[4][];
        int idx = 0;
        for (int free = 0; free < 9 && idx < 4; free++)
        {
            if (isPivot[free]) continue;
            var v = new double[9];
            v[free] = 1;
            for (int r = 0; r < 5; r++) if (pivotCol[r] >= 0) v[pivotCol[r]] = -m[r * 9 + free];
            for (int j = 0; j < idx; j++)
            {
                double d = 0; for (int k = 0; k < 9; k++) d += v[k] * res[j][k];
                for (int k = 0; k < 9; k++) v[k] -= d * res[j][k];
            }
            double nn = 0; for (int k = 0; k < 9; k++) nn += v[k] * v[k];
            nn = Math.Sqrt(nn);
            for (int k = 0; k < 9; k++) v[k] /= nn;
            res[idx++] = v;
        }
        while (idx < 4) res[idx++] = new double[9];   // degenerate sample: no solution comes out of it
        return res;
    }

    /// <summary>Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix: eigenvalues, eigenvectors as COLUMNS.</summary>
    internal static (double[] Values, double[] Vectors) Jacobi(double[] a, int n)
    {
        a = (double[])a.Clone();
        var v = new double[n * n];
        for (int i = 0; i < n; i++) v[i * n + i] = 1;
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += a[i * n + j] * a[i * n + j];
            if (off < 1e-30) break;
            for (int p = 0; p < n; p++)
                for (int qq = p + 1; qq < n; qq++)
                {
                    double apq = a[p * n + qq];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double app = a[p * n + p], aqq = a[qq * n + qq];
                    double theta = (aqq - app) / (2 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
                    double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k * n + p], akq = a[k * n + qq];
                        a[k * n + p] = c * akp - s * akq; a[k * n + qq] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p * n + k], aqk = a[qq * n + k];
                        a[p * n + k] = c * apk - s * aqk; a[qq * n + k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k * n + p], vkq = v[k * n + qq];
                        v[k * n + p] = c * vkp - s * vkq; v[k * n + qq] = s * vkp + c * vkq;
                    }
                }
        }
        var vals = new double[n];
        for (int i = 0; i < n; i++) vals[i] = a[i * n + i];
        return (vals, v);
    }

    /// <summary>Sampson distance of x_b^T E x_a = 0, in pixels (normalized rays x <paramref name="focal"/>).</summary>
    public static double SampsonPx(double[] e, (double X, double Y) a, (double X, double Y) b, double focal)
    {
        double ex0 = e[0] * a.X + e[1] * a.Y + e[2], ex1 = e[3] * a.X + e[4] * a.Y + e[5], ex2 = e[6] * a.X + e[7] * a.Y + e[8];
        double etx0 = e[0] * b.X + e[3] * b.Y + e[6], etx1 = e[1] * b.X + e[4] * b.Y + e[7];
        double num = b.X * ex0 + b.Y * ex1 + ex2;
        double den = ex0 * ex0 + ex1 * ex1 + etx0 * etx0 + etx1 * etx1;
        return den > 1e-300 ? Math.Abs(num) / Math.Sqrt(den) * focal : double.MaxValue;
    }

    /// <summary>Squared Sampson distance (normalized units) of x_b^T E x_a = 0 within <paramref name="thr2"/>, no sqrt.</summary>
    static bool WithinSampson(double[] e, (double X, double Y) a, (double X, double Y) b, double thr2)
    {
        double ex0 = e[0] * a.X + e[1] * a.Y + e[2], ex1 = e[3] * a.X + e[4] * a.Y + e[5], ex2 = e[6] * a.X + e[7] * a.Y + e[8];
        double etx0 = e[0] * b.X + e[3] * b.Y + e[6], etx1 = e[1] * b.X + e[4] * b.Y + e[7];
        double num = b.X * ex0 + b.Y * ex1 + ex2;
        return num * num <= thr2 * (ex0 * ex0 + ex1 * ex1 + etx0 * etx0 + etx1 * etx1);
    }

    /// <summary>
    /// E-RANSAC: five-point hypotheses, inliers by Sampson distance in pixels, adaptive stopping at <paramref name="confidence"/>.
    /// Null below <paramref name="minInliers"/>.
    /// </summary>
    public static (double[] E, bool[] Inliers, int Count)? Ransac(IReadOnlyList<(double X, double Y)> ra, IReadOnlyList<(double X, double Y)> rb,
        double focal, double thresholdPx = 2.0, int minInliers = 15, int maxIterations = 1000, double confidence = 0.999, int seed = 1)
    {
        int n = ra.Count;
        if (n < Math.Max(5, minInliers)) return null;
        var rng = new Random(seed);
        Span<(double X, double Y)> sa = stackalloc (double, double)[5];
        Span<(double X, double Y)> sb = stackalloc (double, double)[5];
        Span<int> idx = stackalloc int[5];
        double[]? best = null;
        int bestCount = -1;
        int needed = maxIterations;
        double thr2 = thresholdPx * thresholdPx / (focal * focal);
        for (int it = 0; it < needed && it < maxIterations; it++)
        {
            for (int k = 0; k < 5; k++)
            {
                int r;
                bool dup;
                do { r = rng.Next(n); dup = false; for (int j = 0; j < k; j++) if (idx[j] == r) dup = true; } while (dup);
                idx[k] = r; sa[k] = ra[r]; sb[k] = rb[r];
            }
            foreach (var e in Solve(sa, sb))
            {
                int count = 0;
                for (int i = 0; i < n; i++)
                {
                    if (WithinSampson(e, ra[i], rb[i], thr2)) count++;
                    else if (count + (n - 1 - i) <= bestCount) break;   // cannot beat the best any more
                }
                if (count > bestCount)
                {
                    bestCount = count; best = e;
                    double w = (double)count / n;
                    double p5 = Math.Pow(w, 5);
                    if (p5 >= 1 - 1e-12) needed = it + 1;
                    else if (p5 > 0) needed = (int)Math.Min(maxIterations, Math.Ceiling(Math.Log(1 - confidence) / Math.Log(1 - p5)));
                }
            }
        }
        if (best == null || bestCount < minInliers) return null;
        var inl = new bool[n];
        for (int i = 0; i < n; i++) inl[i] = SampsonPx(best, ra[i], rb[i], focal) <= thresholdPx;
        return (best, inl, bestCount);
    }
}
