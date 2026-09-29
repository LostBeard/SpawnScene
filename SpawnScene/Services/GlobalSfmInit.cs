using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Global structure-from-motion initialisation: camera poses from the verified image pairs alone - relative poses from
/// each pair's fundamental matrix, then robust ROTATION averaging, then sign-aware TRANSLATION averaging - so bundle
/// adjustment starts from a geometry that does not inherit the depth cascade's bend.
/// </summary>
/// <remarks>
/// Why (MEASURED 2026-09-28, TruckFull, per-stage pose-vs-COLMAP probe): bundle adjustment started from the DAv3 cascade
/// (11% of spread median off COLMAP, smoothly bent) settles in a nearby wrong minimum - 1.9-2.5% final, run-dependent,
/// occasionally 0.1% - while the SAME tracks and observations started from COLMAP's poses converge to 0.1%. Neither
/// COLMAP's rotations alone (5.5%) nor its positions alone (9.6%) suffice: the start needs both, consistent. Rotation
/// averaging + translation averaging is how global SfM (GLOMAP) builds that start from the pairs.
///
/// Conventions: world->camera rotation R has rows [right; down; forward] (OpenCV axes, as BundleAdjuster). A pair's
/// relative pose maps camera A to camera B: X_b = R_ab X_a + t_ab, so R_ab = R_b R_a^T and t_ab ~ R_b (C_a - C_b).
/// Everything is double and host-side: 251 cameras and ~4,600 pairs take milliseconds.
/// </remarks>
public static class GlobalSfmInit
{
    /// <summary>A pair's relative pose: R (row-major 3x3) maps camera-A coordinates to camera-B, T the unit translation
    /// (camera-B frame), Inliers its epipolar inlier count (the pair's weight).</summary>
    public sealed record RelativePose(int A, int B, double[] R, double[] T, int Inliers);

    // ── 3x3 linear algebra (row-major double[9]) ────────────────────────────────────────────────

    public static double[] Mul(double[] a, double[] b)
    {
        var r = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        return r;
    }

    public static double[] Transpose(double[] a) => new[] { a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8] };

    static double Det(double[] m) =>
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

    static double FrobeniusDiff(double[] a, double[] b)
    {
        double s = 0;
        for (int k = 0; k < 9; k++) { double d = a[k] - b[k]; s += d * d; }
        return Math.Sqrt(s);
    }

    /// <summary>Eigen-decomposition of a symmetric 3x3 (cyclic Jacobi): eigenvalues descending, eigenvectors as COLUMNS of v.</summary>
    static void SymEig3(double[] s, double[] eval, double[] v)
    {
        var a = (double[])s.Clone();
        for (int k = 0; k < 9; k++) v[k] = k % 4 == 0 ? 1 : 0;
        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
            if (off < 1e-30) break;
            for (int p = 0; p < 2; p++)
                for (int q = p + 1; q < 3; q++)
                {
                    double apq = a[p * 3 + q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double app = a[p * 3 + p], aqq = a[q * 3 + q];
                    double theta = (aqq - app) / (2 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
                    double c = 1 / Math.Sqrt(t * t + 1), sn = t * c;
                    for (int k = 0; k < 3; k++)
                    {
                        double akp = a[k * 3 + p], akq = a[k * 3 + q];
                        a[k * 3 + p] = c * akp - sn * akq;
                        a[k * 3 + q] = sn * akp + c * akq;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double apk = a[p * 3 + k], aqk = a[q * 3 + k];
                        a[p * 3 + k] = c * apk - sn * aqk;
                        a[q * 3 + k] = sn * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double vkp = v[k * 3 + p], vkq = v[k * 3 + q];
                        v[k * 3 + p] = c * vkp - sn * vkq;
                        v[k * 3 + q] = sn * vkp + c * vkq;
                    }
                }
        }
        eval[0] = a[0]; eval[1] = a[4]; eval[2] = a[8];
        // Sort descending (columns move with their values).
        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; j++)
                if (eval[j] > eval[i])
                {
                    (eval[i], eval[j]) = (eval[j], eval[i]);
                    for (int k = 0; k < 3; k++) (v[k * 3 + i], v[k * 3 + j]) = (v[k * 3 + j], v[k * 3 + i]);
                }
    }

    /// <summary>SVD of a 3x3: m = U diag(sigma) V^T, sigma descending, U and V orthonormal (possibly det -1).</summary>
    static void Svd3(double[] m, double[] u, double[] sigma, double[] v)
    {
        var mtm = Mul(Transpose(m), m);
        var ev = new double[3];
        SymEig3(mtm, ev, v);
        for (int k = 0; k < 3; k++) sigma[k] = Math.Sqrt(Math.Max(0, ev[k]));
        // u_k = m v_k / sigma_k; the smallest may be degenerate (rank 2): complete with a cross product.
        for (int k = 0; k < 2; k++)
        {
            double x = m[0] * v[k] + m[1] * v[3 + k] + m[2] * v[6 + k];
            double y = m[3] * v[k] + m[4] * v[3 + k] + m[5] * v[6 + k];
            double z = m[6] * v[k] + m[7] * v[3 + k] + m[8] * v[6 + k];
            double n = Math.Sqrt(x * x + y * y + z * z);
            if (n < 1e-300) { x = k == 0 ? 1 : 0; y = k == 1 ? 1 : 0; z = 0; n = 1; }
            u[k] = x / n; u[3 + k] = y / n; u[6 + k] = z / n;
        }
        // Re-orthogonalise u1 against u0, then u2 = u0 x u1.
        double d01 = u[0] * u[1] + u[3] * u[4] + u[6] * u[7];
        u[1] -= d01 * u[0]; u[4] -= d01 * u[3]; u[7] -= d01 * u[6];
        double n1 = Math.Sqrt(u[1] * u[1] + u[4] * u[4] + u[7] * u[7]);
        u[1] /= n1; u[4] /= n1; u[7] /= n1;
        u[2] = u[3] * u[7] - u[6] * u[4];
        u[5] = u[6] * u[1] - u[0] * u[7];
        u[8] = u[0] * u[4] - u[3] * u[1];
        // Keep m = U S V^T for the third column (its sign is free when sigma_2 ~ 0).
        double x2 = m[0] * v[2] + m[1] * v[5] + m[2] * v[8];
        double y2 = m[3] * v[2] + m[4] * v[5] + m[5] * v[8];
        double z2 = m[6] * v[2] + m[7] * v[5] + m[8] * v[8];
        if (x2 * u[2] + y2 * u[5] + z2 * u[8] < 0 && sigma[2] > 1e-12) { u[2] = -u[2]; u[5] = -u[5]; u[8] = -u[8]; }
    }

    /// <summary>Nearest rotation (polar projection onto SO(3)).</summary>
    public static double[] ProjectToRotation(double[] m)
    {
        var u = new double[9]; var s = new double[3]; var v = new double[9];
        Svd3(m, u, s, v);
        var vt = Transpose(v);
        var r = Mul(u, vt);
        if (Det(r) < 0)
        {
            u[2] = -u[2]; u[5] = -u[5]; u[8] = -u[8];
            r = Mul(u, vt);
        }
        return r;
    }

    /// <summary>World->camera rotation (rows right, down, forward) of a camera.</summary>
    public static double[] RotationOf(CameraParams cam)
    {
        WorldSpaceGeometry.GetOpenCvAxes(cam, out var right, out var down, out var fwd);
        return new double[] { right.X, right.Y, right.Z, down.X, down.Y, down.Z, fwd.X, fwd.Y, fwd.Z };
    }

    /// <summary>Set a camera's orientation from a world->camera rotation (rows right, down, forward).</summary>
    public static void SetRotation(CameraParams cam, double[] r)
    {
        cam.Forward = Vector3.Normalize(new Vector3((float)r[6], (float)r[7], (float)r[8]));
        cam.Up = Vector3.Normalize(new Vector3(-(float)r[3], -(float)r[4], -(float)r[5]));
    }

    /// <summary>Angle (degrees) of the rotation between two rotations.</summary>
    public static double AngleDeg(double[] a, double[] b)
    {
        var d = Mul(a, Transpose(b));
        double c = Math.Clamp((d[0] + d[4] + d[8] - 1) / 2, -1, 1);
        return Math.Acos(c) * 180 / Math.PI;
    }

    // ── relative pose from a fundamental matrix ─────────────────────────────────────────────────

    /// <summary>
    /// The relative pose of a verified pair: E = K_b^T F K_a (x_b^T F x_a = 0, pixels, row-major), decomposed into its
    /// four (R, t) candidates, the one putting most inlier points in front of BOTH cameras kept. Null when the pair is
    /// degenerate (too few points in front, or no clear winner).
    /// </summary>
    public static RelativePose? FromFundamental(int a, int b, double[] f, float[] xa, float[] xb, bool[] inliers,
        double focal, double cxA, double cyA, double cxB, double cyB)
    {
        var ka = new double[] { focal, 0, cxA, 0, focal, cyA, 0, 0, 1 };
        var kb = new double[] { focal, 0, cxB, 0, focal, cyB, 0, 0, 1 };
        var e = Mul(Mul(Transpose(kb), f), ka);
        var u = new double[9]; var s = new double[3]; var v = new double[9];
        Svd3(e, u, s, v);
        if (Det(u) < 0) { u[2] = -u[2]; u[5] = -u[5]; u[8] = -u[8]; }
        if (Det(v) < 0) { v[2] = -v[2]; v[5] = -v[5]; v[8] = -v[8]; }
        var w = new double[] { 0, -1, 0, 1, 0, 0, 0, 0, 1 };
        var vt = Transpose(v);
        var r1 = Mul(Mul(u, w), vt);
        var r2 = Mul(Mul(u, Transpose(w)), vt);
        var t = new[] { u[2], u[5], u[8] };

        // Normalised inlier rays, up to 300 (evenly strided).
        var ra = new List<(double X, double Y)>(); var rb = new List<(double X, double Y)>();
        int n = inliers.Length, count = 0;
        for (int i = 0; i < n; i++) if (inliers[i]) count++;
        if (count < 8) return null;
        int stride = Math.Max(1, count / 300);
        for (int i = 0, seen = 0; i < n; i++)
        {
            if (!inliers[i]) continue;
            if (seen++ % stride != 0) continue;
            ra.Add(((xa[i * 2] - cxA) / focal, (xa[i * 2 + 1] - cyA) / focal));
            rb.Add(((xb[i * 2] - cxB) / focal, (xb[i * 2 + 1] - cyB) / focal));
        }

        double[]? bestR = null; double[]? bestT = null;
        int best = -1, second = -1;
        foreach (var r in new[] { r1, r2 })
            foreach (double sign in new[] { 1.0, -1.0 })
            {
                var tt = new[] { t[0] * sign, t[1] * sign, t[2] * sign };
                int front = CountInFront(r, tt, ra, rb);
                if (front > best) { second = best; best = front; bestR = r; bestT = tt; }
                else if (front > second) second = front;
            }
        // A clear cheirality winner, and most points in front: otherwise the pair is (near) pure rotation or bad.
        if (bestR == null || best < 0.75 * ra.Count || best - second < 0.25 * ra.Count) return null;
        return new RelativePose(a, b, bestR, bestT!, count);
    }

    /// <summary>Points triangulated (midpoint) with positive depth in both cameras; camera A at the origin.</summary>
    static int CountInFront(double[] r, double[] t, List<(double X, double Y)> ra, List<(double X, double Y)> rb)
    {
        // Camera B's centre and ray directions in camera-A coordinates: C_b = -R^T t, d_b = R^T x_b.
        var rt = Transpose(r);
        double cbx = -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]);
        double cby = -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]);
        double cbz = -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]);
        int front = 0;
        for (int i = 0; i < ra.Count; i++)
        {
            double ax = ra[i].X, ay = ra[i].Y, az = 1;
            double bx0 = rb[i].X, by0 = rb[i].Y;
            double dx = rt[0] * bx0 + rt[1] * by0 + rt[2];
            double dy = rt[3] * bx0 + rt[4] * by0 + rt[5];
            double dz = rt[6] * bx0 + rt[7] * by0 + rt[8];
            // min |l1 a - (C + l2 d)|^2 -> [a.a  -a.d; a.d  -d.d] [l1 l2] = [a.C; d.C]
            double aa = ax * ax + ay * ay + az * az, ad = ax * dx + ay * dy + az * dz, dd = dx * dx + dy * dy + dz * dz;
            double ac = ax * cbx + ay * cby + az * cbz, dc = dx * cbx + dy * cby + dz * cbz;
            double det = -aa * dd + ad * ad;
            if (Math.Abs(det) < 1e-18) continue;
            double l1 = (-ac * dd + ad * dc) / det;
            double l2 = (aa * dc - ad * ac) / det;
            if (l1 > 0 && l2 > 0) front++;
        }
        return front;
    }

    // ── rotation averaging ───────────────────────────────────────────────────────────────────

    /// <summary>
    /// Global world->camera rotations from relative ones. A maximum-spanning-tree start (edges weighted by inliers), then
    /// IRLS rounds of GLOBAL chordal least squares: all rotations at once, min sum w_e |R_b - R_ab R_a|_F^2 as one
    /// block-Laplacian system (the three columns are three right-hand sides; the tree's root held), solved by PCG, each
    /// result projected onto SO(3). Weights sqrt(inliers) / sqrt(r^2 + delta^2) with delta ANNEALED from plain least
    /// squares (every pair heard, so no garbage edge the tree happened to use decides a camera) to <paramref name="delta"/>.
    /// Cameras in no pair keep <paramref name="fallback"/>; the result's world frame is the root's.
    /// </summary>
    /// <remarks>
    /// MEASURED (GlobalSfmInitTests, Truck's 126 cameras, pairs within 6 frames, 0.5 deg noise, 10% garbage): Gauss-Seidel
    /// neighbour averaging instead left 1.1-2.5 deg median - a local update removes the tree's accumulated twist like
    /// diffusion, ~(n/6)^2 sweeps - and IRLS from the tree start at a small delta locked a garbage tree edge in (175 deg).
    /// </remarks>
    public static double[][] AverageRotations(int n, IReadOnlyList<RelativePose> edges, double[][] fallback,
        out bool[] connected, int rounds = 20, double delta = 0.02)
    {
        var adj = new List<(int Other, int Edge, bool Out)>[n];
        for (int i = 0; i < n; i++) adj[i] = new();
        for (int e = 0; e < edges.Count; e++)
        {
            adj[edges[e].A].Add((edges[e].B, e, true));
            adj[edges[e].B].Add((edges[e].A, e, false));
        }
        int root = 0; double bestW = -1;
        for (int i = 0; i < n; i++)
        {
            double wsum = adj[i].Sum(e => (double)edges[e.Edge].Inliers);
            if (wsum > bestW) { bestW = wsum; root = i; }
        }
        // Maximum spanning tree (Prim): the start.
        var rot = new double[n][];
        var conn = new bool[n];
        rot[root] = (double[])fallback[root].Clone();
        conn[root] = true;
        var pq = new PriorityQueue<(int Edge, bool Out, int From), double>();
        void Push(int i) { foreach (var x in adj[i]) if (!conn[x.Other]) pq.Enqueue((x.Edge, x.Out, i), -edges[x.Edge].Inliers); }
        Push(root);
        while (pq.TryDequeue(out var item, out _))
        {
            var ed = edges[item.Edge];
            int to = item.Out ? ed.B : ed.A;
            if (conn[to]) continue;
            rot[to] = item.Out ? Mul(ed.R, rot[item.From]) : Mul(Transpose(ed.R), rot[item.From]);
            conn[to] = true;
            Push(to);
        }
        for (int i = 0; i < n; i++) if (!conn[i]) rot[i] = (double[])fallback[i].Clone();

        // Global IRLS. Unknown per camera: its 3x3 rotation (9 values, columns solved together). Edge energy
        // w |R_b - M R_a|^2 with M = R_ab: grad_a = w (R_a - M^T R_b), grad_b = w (R_b - M R_a).
        var free = new bool[n];
        for (int i = 0; i < n; i++) free[i] = conn[i] && i != root;
        int m = edges.Count;
        var w = new double[m];
        var diag = new double[n];
        var x = new double[n * 9]; var rhs = new double[n * 9]; var r = new double[n * 9];
        var z = new double[n * 9]; var p = new double[n * 9]; var ap = new double[n * 9];
        // y = K v over free cameras; v's fixed entries count as 0.
        void ApplyK(double[] v, double[] y)
        {
            Array.Clear(y);
            for (int i = 0; i < n; i++)
                if (free[i]) for (int k = 0; k < 9; k++) y[i * 9 + k] = diag[i] * v[i * 9 + k];
            for (int e = 0; e < m; e++)
            {
                var ed = edges[e]; var M = ed.R;
                int a = ed.A, b = ed.B;
                bool fa = free[a], fb = free[b];
                // row a: -w M^T v_b ; row b: -w M v_a  (v as 3x3 row-major, product M v)
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 3; col++)
                    {
                        if (fa && fb)
                        {
                            double mtvb = M[row] * v[b * 9 + col] + M[3 + row] * v[b * 9 + 3 + col] + M[6 + row] * v[b * 9 + 6 + col];
                            double mva = M[row * 3] * v[a * 9 + col] + M[row * 3 + 1] * v[a * 9 + 3 + col] + M[row * 3 + 2] * v[a * 9 + 6 + col];
                            y[a * 9 + row * 3 + col] -= w[e] * mtvb;
                            y[b * 9 + row * 3 + col] -= w[e] * mva;
                        }
                    }
            }
        }
        double Dot(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < n; i++) if (free[i]) for (int k = 0; k < 9; k++) s += a[i * 9 + k] * b[i * 9 + k];
            return s;
        }
        for (int round = 0; round < rounds; round++)
        {
            double dRound = round == 0 ? double.PositiveInfinity
                : Math.Max(delta, 1.0 * Math.Pow(delta, (double)round / Math.Max(1, rounds / 2)));
            Array.Clear(diag);
            Array.Clear(rhs);
            for (int e = 0; e < m; e++)
            {
                var ed = edges[e];
                double res = FrobeniusDiff(rot[ed.B], Mul(ed.R, rot[ed.A]));
                w[e] = double.IsPositiveInfinity(dRound) ? Math.Sqrt(ed.Inliers)
                    : Math.Sqrt(ed.Inliers) / Math.Sqrt(res * res + dRound * dRound);
                diag[ed.A] += w[e]; diag[ed.B] += w[e];
                // The root's rotation is fixed: its terms move to the right-hand side.
                if (ed.A == root && free[ed.B])
                {
                    var t = Mul(ed.R, rot[root]);
                    for (int k = 0; k < 9; k++) rhs[ed.B * 9 + k] += w[e] * t[k];
                }
                if (ed.B == root && free[ed.A])
                {
                    var t = Mul(Transpose(ed.R), rot[root]);
                    for (int k = 0; k < 9; k++) rhs[ed.A * 9 + k] += w[e] * t[k];
                }
            }
            // PCG (Jacobi: diag is a scalar per camera), warm-started from the current rotations.
            for (int i = 0; i < n; i++) for (int k = 0; k < 9; k++) x[i * 9 + k] = free[i] ? rot[i][k] : 0;
            ApplyK(x, ap);
            for (int i = 0; i < n * 9; i++) r[i] = free[i / 9] ? rhs[i] - ap[i] : 0;
            for (int i = 0; i < n * 9; i++) z[i] = free[i / 9] && diag[i / 9] > 0 ? r[i] / diag[i / 9] : 0;
            Array.Copy(z, p, n * 9);
            double rz = Dot(r, z), r0 = Math.Max(Dot(r, r), 1e-300);
            for (int it = 0; it < 1000 && Dot(r, r) > 1e-24 * r0; it++)
            {
                ApplyK(p, ap);
                double pap = Dot(p, ap);
                if (!(pap > 0)) break;
                double alpha = rz / pap;
                for (int i = 0; i < n * 9; i++) if (free[i / 9]) { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
                for (int i = 0; i < n * 9; i++) z[i] = free[i / 9] && diag[i / 9] > 0 ? r[i] / diag[i / 9] : 0;
                double rzNew = Dot(r, z);
                double beta = rzNew / rz;
                rz = rzNew;
                for (int i = 0; i < n * 9; i++) p[i] = free[i / 9] ? z[i] + beta * p[i] : 0;
            }
            for (int i = 0; i < n; i++)
                if (free[i]) rot[i] = ProjectToRotation(x.AsSpan(i * 9, 9).ToArray());
        }
        connected = conn;
        return rot;
    }

    /// <summary>The rotation Q (a change of world frame) that best maps averaged rotations onto reference ones:
    /// ref_i ~ avg_i Q. Robust (IRLS) so the reference's misplaced cameras do not tilt it.</summary>
    public static double[] AlignFrame(double[][] avg, double[][] reference, bool[] use)
    {
        var q = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        for (int round = 0; round < 4; round++)
        {
            var mm = new double[9];
            for (int i = 0; i < avg.Length; i++)
            {
                if (!use[i]) continue;
                double wt = round == 0 ? 1 : 1 / Math.Max(AngleDeg(Mul(avg[i], q), reference[i]), 0.5);
                var c = Mul(Transpose(avg[i]), reference[i]);
                for (int k = 0; k < 9; k++) mm[k] += wt * c[k];
            }
            q = ProjectToRotation(mm);
        }
        return q;
    }

    // ── global positioning (camera centres AND points, rotations known) ─────────────────────

    /// <summary>
    /// Camera centres and track points given the global rotations: every observation says the point lies along the
    /// camera's bearing b = R^T K^-1 [u v 1] (normalised). Per IRLS round, min over C and X of
    ///   sum w (|P_b (X - C)|^2 + lambda (b.(X - C) - s)^2),  P_b = I - b b^T,
    /// the perpendicular miss plus a whisper of pull along the bearing towards s (the observation's current depth, or the
    /// median when the point is behind) - only so a point seen along parallel bearings stays invertible. Scale and sign
    /// are fixed by pinning one camera-pair distance. Each point's 3x3 block is eliminated (Schur), leaving a sparse
    /// system over the camera centres (one camera held), solved by block-Jacobi PCG; the points are then back-substituted.
    /// Weights 1/(s^2 max(angle, 0.05 deg)) make the residual angular and down-weight mismatched observations.
    /// </summary>
    /// <remarks>
    /// Why tracks and not pairs: consecutive video frames have nearly parallel baselines, so pairwise translation
    /// directions leave each camera free to slide along the path. MEASURED (GlobalSfmInitTests, Truck path): pairwise
    /// averaging took a 20%-perturbed start only to 14.5%. Points triangulate that slide away - GLOMAP's global positioning.
    /// </remarks>
    public static (Vector3[] Centres, Vector3[] Points) GlobalPositioning(IReadOnlyList<CameraParams> cams, double[][] rot,
        bool[] connected, IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal,
        int rounds = 40, double lambda = 1.0, bool robust = true, bool bataScale = true, bool pinScale = false)
    {
        int n = cams.Count, np = pointCount, no = obs.Count;
        var c = new double[n * 3];
        for (int i = 0; i < n; i++) { c[i * 3] = cams[i].Position.X; c[i * 3 + 1] = cams[i].Position.Y; c[i * 3 + 2] = cams[i].Position.Z; }
        // World bearings.
        var bdir = new double[no * 3];
        for (int k = 0; k < no; k++)
        {
            var o = obs[k]; var cam = cams[o.Camera]; var rr = rot[o.Camera];
            double x = (o.U - cam.CenterX) / focal, y = (o.V - cam.CenterY) / focal, zz = 1;
            double bx = rr[0] * x + rr[3] * y + rr[6] * zz;
            double by = rr[1] * x + rr[4] * y + rr[7] * zz;
            double bz = rr[2] * x + rr[5] * y + rr[8] * zz;
            double bn = Math.Sqrt(bx * bx + by * by + bz * bz);
            bdir[k * 3] = bx / bn; bdir[k * 3 + 1] = by / bn; bdir[k * 3 + 2] = bz / bn;
        }
        var pointObs = new List<int>[np];
        for (int p = 0; p < np; p++) pointObs[p] = new();
        for (int k = 0; k < no; k++) if (connected[obs[k].Camera]) pointObs[obs[k].Point].Add(k);
        int anchor = Array.IndexOf(connected, true);
        // Scale (and the global sign) is pinned by ONE constraint - the anchor-to-far-camera distance along their current
        // direction - not by a pull on every depth. MEASURED (GlobalSfmInitTests, TruckFull-like 2-3 view tracks, bent
        // start): a lambda-0.01 pull on every depth towards the previous round's made each round a small proximal step -
        // 20 rounds left 9.2% median, 200 rounds 0.17%. With the pin, every round is an exact least-squares solve.
        int far = anchor;
        {
            double best = -1;
            for (int i = 0; i < n; i++)
            {
                if (!connected[i]) continue;
                double dx = c[i * 3] - c[anchor * 3], dy = c[i * 3 + 1] - c[anchor * 3 + 1], dz = c[i * 3 + 2] - c[anchor * 3 + 2];
                double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > best) { best = d2; far = i; }
            }
        }
        var free = new bool[n];
        for (int i = 0; i < n; i++) free[i] = connected[i] && i != anchor;
        var xpt = new double[np * 3];
        var w = new double[no];
        var s = new double[no];
        var vinv = new double[np * 9];
        var a = new double[no * 9];   // per observation A = w (I - (1 - lambda) b b^T)
        var g = new double[no * 3];   // per observation g = w lambda s b

        double Spread()
        {
            double mx = 0, my = 0, mz = 0; int k = 0;
            for (int i = 0; i < n; i++) if (connected[i]) { mx += c[i * 3]; my += c[i * 3 + 1]; mz += c[i * 3 + 2]; k++; }
            mx /= k; my /= k; mz /= k;
            double sum = 0;
            for (int i = 0; i < n; i++)
                if (connected[i]) { double dx = c[i * 3] - mx, dy = c[i * 3 + 1] - my, dz = c[i * 3 + 2] - mz; sum += dx * dx + dy * dy + dz * dz; }
            return Math.Sqrt(sum / k);
        }
        double spread0 = Spread();

        void BuildBlocks(double lam)
        {
            for (int k = 0; k < no; k++)
            {
                double kk = 1 - lam;
                int o9 = k * 9, o3 = k * 3;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        a[o9 + i * 3 + j] = w[k] * ((i == j ? 1 : 0) - kk * bdir[o3 + i] * bdir[o3 + j]);
                g[o3] = w[k] * lam * s[k] * bdir[o3]; g[o3 + 1] = w[k] * lam * s[k] * bdir[o3 + 1]; g[o3 + 2] = w[k] * lam * s[k] * bdir[o3 + 2];
            }
            for (int pp = 0; pp < np; pp++)
            {
                Span<double> v = stackalloc double[9];
                v.Clear();
                foreach (int k in pointObs[pp]) for (int q = 0; q < 9; q++) v[q] += a[k * 9 + q];
                var inv = pointObs[pp].Count > 0 ? Invert3(v) : new double[9];
                for (int q = 0; q < 9; q++) vinv[pp * 9 + q] = inv[q];
            }
        }
        // X_p = V^-1 sum (A C_i + g)
        void BackSubstitute()
        {
            for (int pp = 0; pp < np; pp++)
            {
                double hx = 0, hy = 0, hz = 0;
                foreach (int k in pointObs[pp])
                {
                    int i = obs[k].Camera, o9 = k * 9;
                    double cx = c[i * 3], cy = c[i * 3 + 1], cz = c[i * 3 + 2];
                    hx += a[o9] * cx + a[o9 + 1] * cy + a[o9 + 2] * cz + g[k * 3];
                    hy += a[o9 + 3] * cx + a[o9 + 4] * cy + a[o9 + 5] * cz + g[k * 3 + 1];
                    hz += a[o9 + 6] * cx + a[o9 + 7] * cy + a[o9 + 8] * cz + g[k * 3 + 2];
                }
                int b9 = pp * 9;
                xpt[pp * 3] = vinv[b9] * hx + vinv[b9 + 1] * hy + vinv[b9 + 2] * hz;
                xpt[pp * 3 + 1] = vinv[b9 + 3] * hx + vinv[b9 + 4] * hy + vinv[b9 + 5] * hz;
                xpt[pp * 3 + 2] = vinv[b9 + 6] * hx + vinv[b9 + 7] * hy + vinv[b9 + 8] * hz;
            }
        }

        // Start: triangulate every point from the start centres (pure perpendicular residual, a whisper of pull).
        for (int k = 0; k < no; k++) { w[k] = 1; s[k] = 0; }
        BuildBlocks(1e-6);
        BackSubstitute();

        // Reduced camera system storage: dense 3x3 blocks, visited through per-camera neighbour lists.
        var nb = new HashSet<int>[n];
        for (int i = 0; i < n; i++) nb[i] = new HashSet<int> { i };
        foreach (var list in pointObs)
            foreach (int k1 in list) foreach (int k2 in list) nb[obs[k1].Camera].Add(obs[k2].Camera);
        var nbl = nb.Select(h => h.ToArray()).ToArray();
        var S = new double[(long)n * n * 9];
        var rhs = new double[n * 3];
        var pre = new double[n * 9];
        var r = new double[n * 3]; var z = new double[n * 3]; var pv = new double[n * 3]; var ap = new double[n * 3];
        void ApplyS(double[] v, double[] y)
        {
            Array.Clear(y);
            for (int i = 0; i < n; i++)
            {
                if (!free[i]) continue;
                foreach (int j in nbl[i])
                {
                    if (!free[j]) continue;
                    long o = ((long)i * n + j) * 9;
                    y[i * 3] += S[o] * v[j * 3] + S[o + 1] * v[j * 3 + 1] + S[o + 2] * v[j * 3 + 2];
                    y[i * 3 + 1] += S[o + 3] * v[j * 3] + S[o + 4] * v[j * 3 + 1] + S[o + 5] * v[j * 3 + 2];
                    y[i * 3 + 2] += S[o + 6] * v[j * 3] + S[o + 7] * v[j * 3 + 1] + S[o + 8] * v[j * 3 + 2];
                }
            }
        }
        double Dot(double[] x1, double[] x2)
        {
            double sum = 0;
            for (int i = 0; i < n; i++) if (free[i]) sum += x1[i * 3] * x2[i * 3] + x1[i * 3 + 1] * x2[i * 3 + 1] + x1[i * 3 + 2] * x2[i * 3 + 2];
            return sum;
        }

        for (int round = 0; round < rounds; round++)
        {
            // Depths and weights from the current centres and points.
            var depths = new List<double>();
            for (int k = 0; k < no; k++)
            {
                var o = obs[k];
                if (!connected[o.Camera] || pointObs[o.Point].Count < 2) continue;
                double dx = xpt[o.Point * 3] - c[o.Camera * 3], dy = xpt[o.Point * 3 + 1] - c[o.Camera * 3 + 1], dz = xpt[o.Point * 3 + 2] - c[o.Camera * 3 + 2];
                double d = dx * bdir[k * 3] + dy * bdir[k * 3 + 1] + dz * bdir[k * 3 + 2];
                if (d > 0) depths.Add(d);
            }
            depths.Sort();
            double med = depths.Count > 0 ? depths[depths.Count / 2] : spread0;
            for (int k = 0; k < no; k++)
            {
                var o = obs[k];
                double dx = xpt[o.Point * 3] - c[o.Camera * 3], dy = xpt[o.Point * 3 + 1] - c[o.Camera * 3 + 1], dz = xpt[o.Point * 3 + 2] - c[o.Camera * 3 + 2];
                double len = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                double d = dx * bdir[k * 3] + dy * bdir[k * 3 + 1] + dz * bdir[k * 3 + 2];
                double ang = len > 1e-12 ? Math.Acos(Math.Clamp(d / len, -1, 1)) * 180 / Math.PI : 90;
                // BATA's scale (1/argmin_t |(X - C) t - b|): collapse-neutral - shrinking X - C shrinks s with it.
                s[k] = d > 0.05 * med ? (bataScale ? len * len / d : d) : med;
                w[k] = 1 / (s[k] * s[k]) / (round == 0 || !robust ? 1 : Math.Max(ang, 0.05));
            }
            // The pull along the bearings only fixes scale and sign; it targets the CURRENT depths, so it adds no bias at
            // convergence, but a strong pull slows weakly-constrained directions to a crawl (MEASURED, GlobalSfmInitTests:
            // lambda 0.1 for 8 rounds left 3.8% from a 20% start). Strong first, weak after.
            BuildBlocks(lambda);
            // Assemble S = blockdiag(sum A) - sum_p A_i V_p^-1 A_j, and rhs = -sum g + sum A_i V^-1 sum g.
            foreach (var i in Enumerable.Range(0, n)) foreach (int j in nbl[i]) Array.Clear(S, (int)(((long)i * n + j) * 9), 9);
            Array.Clear(rhs);
            Span<double> av = stackalloc double[9];
            foreach (var list in pointObs)
            {
                if (list.Count == 0) continue;
                int pp = obs[list[0]].Point, b9 = pp * 9;
                double gx = 0, gy = 0, gz = 0;
                foreach (int k in list) { gx += g[k * 3]; gy += g[k * 3 + 1]; gz += g[k * 3 + 2]; }
                foreach (int k1 in list)
                {
                    int i = obs[k1].Camera, o1 = k1 * 9;
                    // av = A_i V^-1
                    for (int r0 = 0; r0 < 3; r0++)
                        for (int c0 = 0; c0 < 3; c0++)
                            av[r0 * 3 + c0] = a[o1 + r0 * 3] * vinv[b9 + c0] + a[o1 + r0 * 3 + 1] * vinv[b9 + 3 + c0] + a[o1 + r0 * 3 + 2] * vinv[b9 + 6 + c0];
                    long d9 = ((long)i * n + i) * 9;
                    for (int q = 0; q < 9; q++) S[d9 + q] += a[o1 + q];
                    rhs[i * 3] += -g[k1 * 3] + av[0] * gx + av[1] * gy + av[2] * gz;
                    rhs[i * 3 + 1] += -g[k1 * 3 + 1] + av[3] * gx + av[4] * gy + av[5] * gz;
                    rhs[i * 3 + 2] += -g[k1 * 3 + 2] + av[6] * gx + av[7] * gy + av[8] * gz;
                    foreach (int k2 in list)
                    {
                        int j = obs[k2].Camera, o2 = k2 * 9;
                        long ij = ((long)i * n + j) * 9;
                        for (int r0 = 0; r0 < 3; r0++)
                            for (int c0 = 0; c0 < 3; c0++)
                                S[ij + r0 * 3 + c0] -= av[r0 * 3] * a[o2 + c0] + av[r0 * 3 + 1] * a[o2 + 3 + c0] + av[r0 * 3 + 2] * a[o2 + 6 + c0];
                    }
                }
            }
            // The scale pin: mu (d.(C_far - C_anchor) - L)^2, d and L from the current centres.
            if (pinScale && far != anchor && free[far])
            {
                double dx = c[far * 3] - c[anchor * 3], dy = c[far * 3 + 1] - c[anchor * 3 + 1], dz = c[far * 3 + 2] - c[anchor * 3 + 2];
                double L = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (L > 1e-12)
                {
                    double[] d = { dx / L, dy / L, dz / L };
                    long ff = ((long)far * n + far) * 9;
                    double mu = 10 * (S[ff] + S[ff + 4] + S[ff + 8]) / 3;
                    double proj = d[0] * c[anchor * 3] + d[1] * c[anchor * 3 + 1] + d[2] * c[anchor * 3 + 2] + L;
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++) S[ff + i * 3 + j] += mu * d[i] * d[j];
                        rhs[far * 3 + i] += mu * d[i] * proj;
                    }
                }
            }
            // The anchor is fixed: its column moves to the right-hand side.
            for (int i = 0; i < n; i++)
            {
                if (!free[i]) continue;
                long o = ((long)i * n + anchor) * 9;
                if (!nb[i].Contains(anchor)) continue;
                double ax = c[anchor * 3], ay = c[anchor * 3 + 1], az = c[anchor * 3 + 2];
                rhs[i * 3] -= S[o] * ax + S[o + 1] * ay + S[o + 2] * az;
                rhs[i * 3 + 1] -= S[o + 3] * ax + S[o + 4] * ay + S[o + 5] * az;
                rhs[i * 3 + 2] -= S[o + 6] * ax + S[o + 7] * ay + S[o + 8] * az;
            }
            for (int i = 0; i < n; i++)
            {
                if (!free[i]) continue;
                var inv = Invert3(new ReadOnlySpan<double>(S, (int)(((long)i * n + i) * 9), 9));
                for (int q = 0; q < 9; q++) pre[i * 9 + q] = inv[q];
            }
            // PCG warm-started from the current centres.
            var x0 = (double[])c.Clone();
            ApplyS(x0, ap);
            for (int i = 0; i < n * 3; i++) r[i] = free[i / 3] ? rhs[i] - ap[i] : 0;
            void Pre(double[] src, double[] dst)
            {
                for (int i = 0; i < n; i++)
                {
                    if (!free[i]) { dst[i * 3] = dst[i * 3 + 1] = dst[i * 3 + 2] = 0; continue; }
                    int o = i * 9;
                    dst[i * 3] = pre[o] * src[i * 3] + pre[o + 1] * src[i * 3 + 1] + pre[o + 2] * src[i * 3 + 2];
                    dst[i * 3 + 1] = pre[o + 3] * src[i * 3] + pre[o + 4] * src[i * 3 + 1] + pre[o + 5] * src[i * 3 + 2];
                    dst[i * 3 + 2] = pre[o + 6] * src[i * 3] + pre[o + 7] * src[i * 3 + 1] + pre[o + 8] * src[i * 3 + 2];
                }
            }
            Pre(r, z);
            Array.Copy(z, pv, n * 3);
            double rz = Dot(r, z), r0n = Math.Max(Dot(r, r), 1e-300);
            for (int it = 0; it < 1000 && Dot(r, r) > 1e-22 * r0n; it++)
            {
                ApplyS(pv, ap);
                double pap = Dot(pv, ap);
                if (!(pap > 0)) break;
                double alpha = rz / pap;
                for (int i = 0; i < n * 3; i++) if (free[i / 3]) { c[i] += alpha * pv[i]; r[i] -= alpha * ap[i]; }
                Pre(r, z);
                double rzNew = Dot(r, z);
                double beta = rzNew / rz;
                rz = rzNew;
                for (int i = 0; i < n * 3; i++) pv[i] = free[i / 3] ? z[i] + beta * pv[i] : 0;
            }
            BackSubstitute();
            // Keep the spread (scale is a gauge), points with it.
            double sp = Spread();
            if (sp > 1e-12)
            {
                double k = spread0 / sp;
                double ax = c[anchor * 3], ay = c[anchor * 3 + 1], az = c[anchor * 3 + 2];
                for (int i = 0; i < n; i++)
                    if (connected[i]) { c[i * 3] = ax + (c[i * 3] - ax) * k; c[i * 3 + 1] = ay + (c[i * 3 + 1] - ay) * k; c[i * 3 + 2] = az + (c[i * 3 + 2] - az) * k; }
                for (int pp = 0; pp < np; pp++)
                { xpt[pp * 3] = ax + (xpt[pp * 3] - ax) * k; xpt[pp * 3 + 1] = ay + (xpt[pp * 3 + 1] - ay) * k; xpt[pp * 3 + 2] = az + (xpt[pp * 3 + 2] - az) * k; }
            }
        }
        var centres = new Vector3[n];
        for (int i = 0; i < n; i++) centres[i] = new Vector3((float)c[i * 3], (float)c[i * 3 + 1], (float)c[i * 3 + 2]);
        var points = new Vector3[np];
        for (int pp = 0; pp < np; pp++) points[pp] = new Vector3((float)xpt[pp * 3], (float)xpt[pp * 3 + 1], (float)xpt[pp * 3 + 2]);
        return (centres, points);
    }

    static double[] Invert3(ReadOnlySpan<double> m)
    {
        double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
        double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
        double det = a * A + b * B + c * C;
        if (!(Math.Abs(det) > 1e-300)) return new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        double id = 1 / det;
        return new[]
        {
            A * id, -(b * i - c * h) * id, (b * f - c * e) * id,
            B * id, (a * i - c * g) * id, -(a * f - c * d) * id,
            C * id, -(a * h - b * g) * id, (a * e - b * d) * id,
        };
    }

    // ── the whole initialisation ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Replace <paramref name="cams"/>' poses with the global SfM solution, expressed in their current world frame: the
    /// averaged rotations are rotated onto the current ones (the gauge), then global positioning starts from the current
    /// centres and keeps their spread. Returns a one-line summary.
    /// </summary>
    public static string Apply(List<CameraParams> cams, IReadOnlyList<RelativePose> edges,
        IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal, double[][]? fixedRotations = null)
    {
        int n = cams.Count;
        var current = cams.Select(RotationOf).ToArray();
        double[][] rot;
        bool[] connected;
        if (fixedRotations != null)
        {
            // DIAGNOSIS: rotations given (already in the cameras' frame) - positioning only.
            rot = fixedRotations.Select(r => (double[])r.Clone()).ToArray();
            connected = Enumerable.Repeat(true, n).ToArray();
        }
        else
        {
            rot = AverageRotations(n, edges, current, out connected);
            var q = AlignFrame(rot, current, connected);
            for (int i = 0; i < n; i++) if (connected[i]) rot[i] = Mul(rot[i], q);
        }
        var resid = edges.Select(e => AngleDeg(rot[e.B], Mul(e.R, rot[e.A]))).OrderBy(x => x).ToList();
        var (centres, _) = GlobalPositioning(cams, rot, connected, obs, pointCount, focal);
        int moved = 0;
        for (int i = 0; i < n; i++)
        {
            if (!connected[i]) continue;
            SetRotation(cams[i], rot[i]);
            cams[i].Position = centres[i];
            moved++;
        }
        return $"{moved}/{n} cameras from {edges.Count} pair rotations and {obs.Count} track observations; pair rotation " +
               $"residual median {(resid.Count > 0 ? resid[resid.Count / 2] : double.NaN):F2} deg p90 " +
               $"{(resid.Count > 0 ? resid[resid.Count * 9 / 10] : double.NaN):F2} deg";
    }
}
