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

    /// <summary>
    /// Loop (triplet) consistency: keep an edge only when at least <paramref name="minConsistent"/> triangles through it
    /// compose to within <paramref name="maxDeg"/> of the identity (R_ca R_bc R_ab ~ I). An edge in no triangle cannot be
    /// checked and is dropped. Zach et al. 2010; the principle of GLOMAP's view-graph filtering.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-30 (DrJohnson, SIFT + 5-point E against COLMAP, Research/sfm-front-end-drjohnson-2026-09-30.md):
    /// 31% of verified pairs were off by 90-180 deg - REPEATED STRUCTURE (a wall matched to the opposite wall), which
    /// passes every two-view check. With 1 consistent triangle at 5 deg the kept pairs were 1.4 / 3.0 deg (median / p75)
    /// and the averaged rotations of the connected core 4.3 / 6.1 deg, against 2.8 / 110 unfiltered. The cost is
    /// coverage: cameras outside the consistent core are left to registration against its points.
    /// </remarks>
    public static List<RelativePose> FilterByLoops(IReadOnlyList<RelativePose> edges, double maxDeg = 5, int minConsistent = 1)
    {
        var rel = new Dictionary<(int, int), double[]>();
        var nb = new Dictionary<int, HashSet<int>>();
        foreach (var e in edges)
        {
            if (e.A == e.B || rel.ContainsKey((e.A, e.B)) || rel.ContainsKey((e.B, e.A))) continue;
            rel[(e.A, e.B)] = e.R;
            if (!nb.TryGetValue(e.A, out var na)) nb[e.A] = na = new HashSet<int>();
            if (!nb.TryGetValue(e.B, out var nbB)) nb[e.B] = nbB = new HashSet<int>();
            na.Add(e.B); nbB.Add(e.A);
        }
        double[] Rel(int i, int j) => rel.TryGetValue((i, j), out var r) ? r : Transpose(rel[(j, i)]);
        var identity = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        var kept = new List<RelativePose>();
        foreach (var e in edges)
        {
            if (!rel.TryGetValue((e.A, e.B), out var rab) || !ReferenceEquals(rab, e.R)) continue;   // first of duplicates
            int ok = 0;
            foreach (int c in nb[e.A])
            {
                if (c == e.B || !nb[e.B].Contains(c)) continue;
                var cycle = Mul(Rel(c, e.A), Mul(Rel(e.B, c), e.R));
                if (AngleDeg(cycle, identity) <= maxDeg && ++ok >= minConsistent) break;
            }
            if (ok >= minConsistent) kept.Add(e);
        }
        return kept;
    }

    /// <summary>
    /// View-graph focal calibration (GLOMAP view_graph_calibration.cc, after Fetzer et al.): one focal shared by every
    /// view, the one for which each pair's E = K^T F K is closest to a valid essential matrix, under a Cauchy loss of
    /// scale 1e-2. The per-pair residual is GLOMAP's FetzerFocalLengthSameCameraCost, ported as is. Null when too few
    /// pairs support it or the estimate leaves [0.1, 10] x the prior (GLOMAP's thres_lower/higher_ratio).
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-30 (Python port of the same cost on SIFT F-RANSAC pairs): DrJohnson 1046.9 vs COLMAP 1035.5
    /// (+1.1%) from DAv3's 812 (-22%); TruckFull 588.0 vs 581.9 (+1.0%) from 609.7. A 22% focal error made DrJohnson's
    /// pair rotations 3-4x worse (Research/sfm-front-end-drjohnson-2026-09-30.md).
    /// </remarks>
    public static (double Focal, int Supporting, int Pairs)? CalibrateFocal(IReadOnlyList<double[]> fundamentals,
        double cx, double cy, double prior, int minSupporting = 8)
    {
        if (fundamentals.Count == 0 || !(prior > 0)) return null;
        var k = new double[] { 1, 0, cx, 0, 1, cy, 0, 0, 1 };
        var ds = new List<(double[] D01, double[] D12)>(fundamentals.Count);
        foreach (var f in fundamentals)
        {
            var g = Mul(Mul(Transpose(k), f), k);
            var u = new double[9]; var sv = new double[3]; var v = new double[9];
            Svd3(g, u, sv, v);
            double v00 = v[0], v01 = v[3], v02 = v[6], v10 = v[1], v11 = v[4], v12 = v[7];   // columns 0 and 1 of V
            double u00 = u[0], u01 = u[3], u02 = u[6], u10 = u[1], u11 = u[4], u12 = u[7];   // columns 0 and 1 of U
            double s0 = sv[0], s1 = sv[1];
            var ai = new[] { s0 * s0 * (v00 * v00 + v01 * v01), s0 * s1 * (v00 * v10 + v01 * v11), s1 * s1 * (v10 * v10 + v11 * v11) };
            var aj = new[] { u10 * u10 + u11 * u11, -(u00 * u10 + u01 * u11), u00 * u00 + u01 * u01 };
            var bi = new[] { s0 * s0 * v02 * v02, s0 * s1 * v02 * v12, s1 * s1 * v12 * v12 };
            var bj = new[] { u12 * u12, -(u02 * u12), u02 * u02 };
            double[] D(int a, int b) => new[] { ai[a] * aj[b] - ai[b] * aj[a], ai[a] * bj[b] - ai[b] * bj[a],
                                                bi[a] * aj[b] - bi[b] * aj[a], bi[a] * bj[b] - bi[b] * bj[a] };
            ds.Add((D(1, 0), D(2, 1)));
        }
        static double Res2(double[] d01, double[] d12, double f)
        {
            double ff = f * f;
            double di = ff * d01[0] + d01[1], dj = ff * d12[0] + d12[2];
            if (di == 0) di = 1e-6;
            if (dj == 0) dj = 1e-6;
            double k0 = -(ff * d01[2] + d01[3]) / di, k1 = -(ff * d12[1] + d12[3]) / dj;
            double r0 = (ff - k0) / ff, r1 = (ff - k1) / ff;
            return r0 * r0 + r1 * r1;
        }
        double Cost(double f) { double c = 0; foreach (var (a, b) in ds) c += 1e-4 * Math.Log(1 + Res2(a, b, f) / 1e-4); return c; }
        // 1-D: a log-spaced scan over [0.3, 3] x prior, then two local refinements (the cost is cheap, the scan global).
        double best = prior, bestCost = double.MaxValue;
        for (int i = 0; i < 400; i++)
        {
            double f = prior * Math.Exp(Math.Log(0.3) + (Math.Log(3) - Math.Log(0.3)) * i / 399.0), c = Cost(f);
            if (c < bestCost) { bestCost = c; best = f; }
        }
        foreach (double step in new[] { 0.01, 0.001 })
        {
            double center = best;
            for (int i = -20; i <= 20; i++) { double f = center * Math.Exp(step * i), c = Cost(f); if (c < bestCost) { bestCost = c; best = f; } }
        }
        if (best / prior > 10 || best / prior < 0.1) return null;
        int supporting = ds.Count(d => Math.Sqrt(Res2(d.D01, d.D12, best)) <= 2.0);
        return supporting >= minSupporting ? (best, supporting, ds.Count) : null;
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

    // ── robust global positioning (GLOMAP) ──────────────────────────────────────────────────

    // Shared with GpuGlobalPositioner's kernels, so the two solvers run the SAME algorithm and can be proven equal. No
    // transcendental functions: WebGPU emulates f64 as a pair of f32, where acos/log would come back at float precision
    // (acos of a cosine within 1e-8 of 1 is meaningless in float) - these need only +, *, / and sqrt.

    /// <summary>Deterministic start coordinate in [-1, 1): a 32-bit integer hash (Wellons' lowbias32) of (seed, stream,
    /// index). Stream 0 = camera centres (index = camera * 3 + axis), 1 = points (point * 3 + axis).</summary>
    public static double StartCoord(int seed, int stream, int index)
    {
        uint x = (uint)index * 0x9E3779B9u + (uint)seed * 0x85EBCA6Bu + (uint)stream * 0xC2B2AE35u;
        x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
        return (int)(x >> 8) * (2.0 / 16777216.0) - 1;
    }

    /// <summary>tan(theta / 2) of the angle between unit bearing v and u = X - C: |v x u| / (|u| + v.u). Monotone in the angle
    /// on [0, pi], exact for small angles, 1e30 when u is 0 or points straight back.</summary>
    public static double TanHalfAngle(double v0, double v1, double v2, double ux, double uy, double uz)
    {
        double cx = v1 * uz - v2 * uy, cy = v2 * ux - v0 * uz, cz = v0 * uy - v1 * ux;
        double un = Math.Sqrt(ux * ux + uy * uy + uz * uz);
        double den = un + (v0 * ux + v1 * uy + v2 * uz);
        double num = Math.Sqrt(cx * cx + cy * cy + cz * cz);
        // num <= |u|, so past the guard num / den <= 1e30: no product can leave WebGPU double-float's (float's) range.
        return den > 1e-30 * (un + 1e-30) ? Math.Min(num / den, 1e30) : 1e30;
    }

    /// <summary>Histogram bins for the median angle: 128 per octave of tan(theta/2) from 2^-30 up (bins 0 and 4095 catch the
    /// ends), keyed on a float's exponent and top 7 mantissa bits - a piecewise-linear log2, identical on host and device.</summary>
    public const int AngleBins = 4096;
    public const int AngleKeyMin = (127 - 30) << 7;   // key of 2^-30

    public static int AngleBin(double tanHalf)
    {
        float f = (float)Math.Min(tanHalf, 1e30);
        int key = BitConverter.SingleToInt32Bits(Math.Max(f, 1e-30f)) >> 16;
        return Math.Clamp(key - AngleKeyMin, 0, AngleBins - 1);
    }

    /// <summary>The upper edge of <see cref="AngleBin"/> bin b (tan(theta/2)).</summary>
    public static double AngleBinUpper(int b) => BitConverter.Int32BitsToSingle((b + 1 + AngleKeyMin) << 16);

    /// <summary>The inlier limit (tan(theta/2)) from the median bin's upper edge: max(0.1 deg, 5 x the median angle).</summary>
    public static double InlierLimit(double medianTanHalf)
    {
        double median = 2 * Math.Atan(medianTanHalf);
        return Math.Tan(0.5 * Math.Max(0.1 * Math.PI / 180, Math.Min(5 * median, 0.99 * Math.PI)));
    }

    /// <summary>
    /// GLOMAP's global positioning (Pan et al. 2024). With the rotations known, observation k (camera i, point j) says the
    /// point lies along the camera's world bearing v_k. Minimise over the centres C, the points X and one scale d_k &gt;= 0
    /// per observation
    ///   sum_k huber(|v_k - d_k (X_j - C_i)|)
    /// by Levenberg-Marquardt from a RANDOM start. d_k makes the residual collapse-proof (X - C -&gt; 0 leaves |v| = 1) and
    /// the Huber loss caps the pull of a mismatched observation - what the linear formulation above lacks (5% mismatches
    /// took it to 40% median error). Each step Schur-eliminates every d_k (1x1), then every point (3x3), leaving a dense
    /// system over the centres, solved by Cholesky. The translation gauge is a centroid term (mu/2)|sum C|^2 / n, NOT a fixed
    /// camera: from a random start a fixed camera sits wherever it was drawn, its observations get rejected as mismatches and
    /// it is orphaned (MEASURED: camera 0 left 79.5% off while every other camera was within 1.2%). The scale is free (d absorbs it), so
    /// the result is returned at the connected cameras' current centroid and spread.
    /// Why tracks and not pairs: consecutive video frames have nearly parallel baselines, so pairwise translation directions
    /// leave each camera free to slide along the path (MEASURED: pairwise averaging took a 20% start only to 14.5%).
    /// </summary>
    /// <summary><see cref="GlobalPositioningRobust"/>'s result. Converged: the last round reached the relative-decrease stop,
    /// not the iteration cap. Outliers: observations outside the final inlier selection (a retired point's included).</summary>
    public sealed record PositioningResult(Vector3[] Centres, string Summary, bool Converged, int Outliers, int Observations);

    public static PositioningResult GlobalPositioningRobust(IReadOnlyList<CameraParams> cams,
        double[][] rot, bool[] connected, IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal,
        int maxIterations = 50, double huber = 0.003, int seed = 1)
    {
        int n = cams.Count, np = pointCount;
        var (oc, op, v) = PositioningObservations(cams, rot, connected, obs, np, focal);
        int no = oc.Length;
        var pointObs = new List<int>[np];
        for (int p = 0; p < np; p++) pointObs[p] = new();
        for (int t = 0; t < no; t++) pointObs[op[t]].Add(t);
        var col = new int[n];
        int nf = 0;
        for (int i = 0; i < n; i++) col[i] = connected[i] ? nf++ : -1;
        int dim = nf * 3;

        // Random start (GLOMAP): centres and points uniform in [-1, 1]^3.
        var c = new double[n * 3]; var xp = new double[np * 3]; var d = new double[no];
        for (int i = 0; i < n * 3; i++) c[i] = StartCoord(seed, 0, i);
        for (int i = 0; i < np * 3; i++) xp[i] = StartCoord(seed, 1, i);
        for (int t = 0; t < no; t++)   // each scale at its optimum (see the variable projection below)
        {
            int i = oc[t] * 3, j = op[t] * 3;
            double ux = xp[j] - c[i], uy = xp[j + 1] - c[i + 1], uz = xp[j + 2] - c[i + 2];
            double uu = ux * ux + uy * uy + uz * uz;
            d[t] = uu > 0 ? Math.Max(0, (v[t * 3] * ux + v[t * 3 + 1] * uy + v[t * 3 + 2] * uz) / uu) : 0;
        }

        var r = new double[no * 3]; var w = new double[no];
        // Rejected observations (w = 0, no cost) and each point's count of live ones - a point needs two.
        var off = new bool[no]; var live = new int[np];
        foreach (int t0 in Enumerable.Range(0, no)) live[op[t0]]++;
        double mu = 0;   // centroid gauge weight, set from the first system's camera diagonal
        double Evaluate(double[] cc, double[] xx, double[] dd, bool keep)
        {
            double cost = 0;
            if (mu > 0)
            {
                double sx = 0, sy = 0, sz = 0;
                for (int i = 0; i < n; i++) if (col[i] >= 0) { sx += cc[i * 3]; sy += cc[i * 3 + 1]; sz += cc[i * 3 + 2]; }
                cost += 0.5 * mu * (sx * sx + sy * sy + sz * sz) / nf;
            }
            for (int t = 0; t < no; t++)
            {
                if (off[t]) { if (keep) { r[t * 3] = r[t * 3 + 1] = r[t * 3 + 2] = 0; w[t] = 0; } continue; }
                int i = oc[t] * 3, j = op[t] * 3;
                double r0 = v[t * 3] - dd[t] * (xx[j] - cc[i]);
                double r1 = v[t * 3 + 1] - dd[t] * (xx[j + 1] - cc[i + 1]);
                double r2 = v[t * 3 + 2] - dd[t] * (xx[j + 2] - cc[i + 2]);
                double e = Math.Sqrt(r0 * r0 + r1 * r1 + r2 * r2);
                cost += e <= huber ? 0.5 * e * e : huber * (e - 0.5 * huber);
                if (keep) { r[t * 3] = r0; r[t * 3 + 1] = r1; r[t * 3 + 2] = r2; w[t] = e <= huber ? 1 : huber / e; }
            }
            return cost;
        }

        var hdd = new double[no]; var gd = new double[no]; var a = new double[no * 3]; var e2 = new double[no];
        var hxx = new double[np * 6]; var gx = new double[np * 3];
        var hcc = new double[n * 6]; var gc = new double[n * 3];
        var S = new double[dim * dim]; var rhs = new double[dim];
        var pinv = new double[np * 9];
        var dc = new double[dim]; var cNew = new double[n * 3]; var xNew = new double[np * 3]; var dNew = new double[no];
        double lambda = 1e-4, cost0 = Evaluate(c, xp, d, true), cost = cost0;
        int iter = 0, accepted = 0, rejectedTotal = 0, round = 0, roundsRun = 0;
        var roundLog = new List<string>();
        bool roundConverged = false;   // the last round reached the relative-decrease stop, not the iteration cap
        // Rounds: converge, then reject observations whose angle to their point is far past the typical one (a
        // mismatch in a 2-3 view track cannot be outvoted inside its track - it only shows once the cameras settle),
        // and re-solve from where the previous round ended.
        for (; round < 6; round++)
        {
        roundsRun++;
        int roundStart = iter, acceptedStart = accepted;
        roundConverged = false;
        for (int roundIter = 0; roundIter < maxIterations; roundIter++, iter++)
        {
            // Normal equations of the IRLS-weighted linearisation, gradient g = J^T W r, step H delta = -g.
            Array.Clear(hxx); Array.Clear(gx); Array.Clear(hcc); Array.Clear(gc);
            for (int t = 0; t < no; t++)
            {
                int i = oc[t], j = op[t];
                double ux = xp[j * 3] - c[i * 3], uy = xp[j * 3 + 1] - c[i * 3 + 1], uz = xp[j * 3 + 2] - c[i * 3 + 2];
                double wt = w[t], dt = d[t], r0 = r[t * 3], r1 = r[t * 3 + 1], r2 = r[t * 3 + 2];
                hdd[t] = wt * (ux * ux + uy * uy + uz * uz);
                gd[t] = -wt * (ux * r0 + uy * r1 + uz * r2);
                a[t * 3] = wt * dt * ux; a[t * 3 + 1] = wt * dt * uy; a[t * 3 + 2] = wt * dt * uz;   // H_Xd; H_cd = -a
                double dd2 = wt * dt * dt;
                e2[t] = -dd2;                                                                         // H_cX = -w d^2 I
                hxx[j * 6] += dd2; hxx[j * 6 + 3] += dd2; hxx[j * 6 + 5] += dd2;
                hcc[i * 6] += dd2; hcc[i * 6 + 3] += dd2; hcc[i * 6 + 5] += dd2;
                gx[j * 3] -= wt * dt * r0; gx[j * 3 + 1] -= wt * dt * r1; gx[j * 3 + 2] -= wt * dt * r2;
                gc[i * 3] += wt * dt * r0; gc[i * 3 + 1] += wt * dt * r1; gc[i * 3 + 2] += wt * dt * r2;
            }
            if (mu == 0)
            {
                double sum = 0;
                for (int i = 0; i < n; i++) if (col[i] >= 0) sum += hcc[i * 6];
                mu = Math.Max(sum / nf, 1e-12);
                cost0 = cost = Evaluate(c, xp, d, true);
            }
            // Marquardt damping on the diagonal (every diagonal entry of H_XX / H_cc is the same sum).
            for (int j = 0; j < np; j++) { double s = hxx[j * 6] * lambda; hxx[j * 6] += s; hxx[j * 6 + 3] += s; hxx[j * 6 + 5] += s; }
            for (int i = 0; i < n; i++) { double s = hcc[i * 6] * lambda; hcc[i * 6] += s; hcc[i * 6 + 3] += s; hcc[i * 6 + 5] += s; }
            // Eliminate every d_t: H_XX -= s a a^T, H_cc -= s a a^T, H_cX = e I + s a a^T, g_X -= s a g_d, g_c += s a g_d.
            for (int t = 0; t < no; t++)
            {
                hdd[t] = hdd[t] * (1 + lambda) + 1e-300;
                double s = 1 / hdd[t], a0 = a[t * 3], a1 = a[t * 3 + 1], a2 = a[t * 3 + 2];
                int j = op[t] * 6, i = oc[t] * 6;
                double q00 = s * a0 * a0, q01 = s * a0 * a1, q02 = s * a0 * a2, q11 = s * a1 * a1, q12 = s * a1 * a2, q22 = s * a2 * a2;
                hxx[j] -= q00; hxx[j + 1] -= q01; hxx[j + 2] -= q02; hxx[j + 3] -= q11; hxx[j + 4] -= q12; hxx[j + 5] -= q22;
                hcc[i] -= q00; hcc[i + 1] -= q01; hcc[i + 2] -= q02; hcc[i + 3] -= q11; hcc[i + 4] -= q12; hcc[i + 5] -= q22;
                double sg = s * gd[t];
                gx[op[t] * 3] -= sg * a0; gx[op[t] * 3 + 1] -= sg * a1; gx[op[t] * 3 + 2] -= sg * a2;
                gc[oc[t] * 3] += sg * a0; gc[oc[t] * 3 + 1] += sg * a1; gc[oc[t] * 3 + 2] += sg * a2;
            }
            // Eliminate every point: S = H_cc - sum B P B^T, rhs = g_c - sum B P g_X, B = H_cX of each observation.
            Array.Clear(S); Array.Clear(rhs);
            for (int i = 0; i < n; i++)
            {
                if (col[i] < 0) continue;
                int b = col[i] * 3;
                S[b * dim + b] = hcc[i * 6]; S[b * dim + b + 1] = hcc[i * 6 + 1]; S[b * dim + b + 2] = hcc[i * 6 + 2];
                S[(b + 1) * dim + b] = hcc[i * 6 + 1]; S[(b + 1) * dim + b + 1] = hcc[i * 6 + 3]; S[(b + 1) * dim + b + 2] = hcc[i * 6 + 4];
                S[(b + 2) * dim + b] = hcc[i * 6 + 2]; S[(b + 2) * dim + b + 1] = hcc[i * 6 + 4]; S[(b + 2) * dim + b + 2] = hcc[i * 6 + 5];
                rhs[b] = gc[i * 3]; rhs[b + 1] = gc[i * 3 + 1]; rhs[b + 2] = gc[i * 3 + 2];
            }
            {
                // Centroid gauge: Hessian mu / n between every pair of cameras on each axis, gradient mu / n * sum C.
                double sx = 0, sy = 0, sz = 0;
                for (int i = 0; i < n; i++) if (col[i] >= 0) { sx += c[i * 3]; sy += c[i * 3 + 1]; sz += c[i * 3 + 2]; }
                double hg = mu / nf;
                for (int i = 0; i < nf; i++)
                {
                    for (int k = 0; k < nf; k++)
                        for (int ax = 0; ax < 3; ax++) S[(i * 3 + ax) * dim + k * 3 + ax] += hg;
                    rhs[i * 3] += hg * sx; rhs[i * 3 + 1] += hg * sy; rhs[i * 3 + 2] += hg * sz;
                }
            }
            Span<double> B1 = stackalloc double[9], B2 = stackalloc double[9], Q = stackalloc double[9], H9 = stackalloc double[9];
            for (int j = 0; j < np; j++)
            {
                var list = pointObs[j];
                if (live[j] < 2) continue;
                var h = hxx.AsSpan(j * 6, 6);
                H9[0] = h[0]; H9[1] = h[1]; H9[2] = h[2]; H9[3] = h[1]; H9[4] = h[3]; H9[5] = h[4]; H9[6] = h[2]; H9[7] = h[4]; H9[8] = h[5];
                var P = Invert3(H9);
                P.CopyTo(pinv, j * 9);
                double g0 = gx[j * 3], g1 = gx[j * 3 + 1], g2 = gx[j * 3 + 2];
                foreach (int t1 in list)
                {
                    int c1 = col[oc[t1]];
                    if (c1 < 0) continue;
                    ObsBlock(t1, B1);
                    Mul3(B1, P, Q);   // Q = B1 P
                    int b1 = c1 * 3;
                    for (int rr = 0; rr < 3; rr++) rhs[b1 + rr] -= Q[rr * 3] * g0 + Q[rr * 3 + 1] * g1 + Q[rr * 3 + 2] * g2;
                    foreach (int t2 in list)
                    {
                        int c2 = col[oc[t2]];
                        if (c2 < 0) continue;
                        ObsBlock(t2, B2);
                        int b2 = c2 * 3;
                        for (int rr = 0; rr < 3; rr++)
                            for (int cc2 = 0; cc2 < 3; cc2++)   // (Q B2^T)[rr, cc2]; B2 is symmetric
                                S[(b1 + rr) * dim + b2 + cc2] -= Q[rr * 3] * B2[cc2 * 3] + Q[rr * 3 + 1] * B2[cc2 * 3 + 1] + Q[rr * 3 + 2] * B2[cc2 * 3 + 2];
                    }
                }
            }
            for (int k = 0; k < dim; k++) dc[k] = -rhs[k];
            if (!CholeskySolve(S, dim, dc))
            {
                lambda *= 10;
                Evaluate(c, xp, d, true);
                if (lambda > 1e12) break;
                continue;
            }
            // Back-substitution: dX = -P (g_X + sum B dc), dd = -(g_d + a.dX - a.dc) / H_dd.
            Array.Copy(c, cNew, c.Length);
            for (int i = 0; i < n; i++)
                if (col[i] >= 0) { int b = col[i] * 3; cNew[i * 3] += dc[b]; cNew[i * 3 + 1] += dc[b + 1]; cNew[i * 3 + 2] += dc[b + 2]; }
            Array.Copy(xp, xNew, xp.Length);
            var dxAll = new double[np * 3];
            for (int j = 0; j < np; j++)
            {
                var list = pointObs[j];
                if (live[j] < 2) continue;
                double s0 = gx[j * 3], s1 = gx[j * 3 + 1], s2 = gx[j * 3 + 2];
                foreach (int t in list)
                {
                    int ci = col[oc[t]];
                    if (ci < 0) continue;
                    ObsBlock(t, B1);
                    double y0 = dc[ci * 3], y1 = dc[ci * 3 + 1], y2 = dc[ci * 3 + 2];
                    s0 += B1[0] * y0 + B1[1] * y1 + B1[2] * y2;
                    s1 += B1[3] * y0 + B1[4] * y1 + B1[5] * y2;
                    s2 += B1[6] * y0 + B1[7] * y1 + B1[8] * y2;
                }
                var P = pinv.AsSpan(j * 9, 9);
                dxAll[j * 3] = -(P[0] * s0 + P[1] * s1 + P[2] * s2);
                dxAll[j * 3 + 1] = -(P[3] * s0 + P[4] * s1 + P[5] * s2);
                dxAll[j * 3 + 2] = -(P[6] * s0 + P[7] * s1 + P[8] * s2);
                xNew[j * 3] += dxAll[j * 3]; xNew[j * 3 + 1] += dxAll[j * 3 + 1]; xNew[j * 3 + 2] += dxAll[j * 3 + 2];
            }
            // Variable projection: each d_t at its exact optimum for the candidate X, C (closed form: the scale that best
            // stretches X - C onto the bearing; the residual is then sin of their angle, or 1 past 90 deg), NOT the
            // linearised d step. The residual is bilinear in d and X - C, so a linearised step along a poorly
            // constrained point depth s misses by (ds/s)^2 and LM rejects it: MEASURED 2026-09-29 (Truck, 10% mismatches)
            // the linearised version alternated accept/reject for 600 iterations per round without converging (1,529
            // iterations, 66.8 s, 0.278% median); projected, a round after the first rejection converges in 6-8 (45
            // iterations at a cap of 15, 1.9 s, 0.254%). Same minimiser: d only ever sits at its optimum.
            for (int t = 0; t < no; t++)
            {
                int i = oc[t] * 3, j = op[t] * 3;
                double ux = xNew[j] - cNew[i], uy = xNew[j + 1] - cNew[i + 1], uz = xNew[j + 2] - cNew[i + 2];
                double uu = ux * ux + uy * uy + uz * uz;
                dNew[t] = uu > 0 ? Math.Max(0, (v[t * 3] * ux + v[t * 3 + 1] * uy + v[t * 3 + 2] * uz) / uu) : 0;
            }
            double newCost = Evaluate(cNew, xNew, dNew, false);
            if (newCost < cost)
            {
                bool converged = (cost - newCost) < 1e-10 * cost;
                (c, cNew) = (cNew, c); (xp, xNew) = (xNew, xp); (d, dNew) = (dNew, d);
                cost = Evaluate(c, xp, d, true);
                lambda = Math.Max(lambda / 3, 1e-12);
                accepted++;
                if (converged) { roundConverged = true; iter++; break; }
            }
            else
            {
                lambda *= 4;
                if (lambda > 1e12) break;
            }
        }
            // Re-select the inliers from ALL observations each round - an observation rejected by an earlier round can come
            // back. A round stops at the iteration cap before converging, so its first selection also rejects good
            // observations still at large angles: MEASURED 2026-09-29, with rejection permanent the CLEAN Truck tracks lost
            // 6,353 of 30,000 observations in round 0. Retired points (fewer than two inliers, so the solve left them
            // behind) are first re-triangulated from all their bearings: min sum |(I - v v^T)(X - C)|^2.
            Span<double> A9 = stackalloc double[9];
            for (int j = 0; j < np; j++)
            {
                if (live[j] >= 2 || pointObs[j].Count < 2) continue;
                A9.Clear();
                double b0 = 0, b1 = 0, b2 = 0;
                foreach (int t in pointObs[j])
                {
                    double v0 = v[t * 3], v1 = v[t * 3 + 1], v2 = v[t * 3 + 2];
                    int i = oc[t] * 3;
                    double c0 = c[i], c1 = c[i + 1], c2 = c[i + 2];
                    double p00 = 1 - v0 * v0, p01 = -v0 * v1, p02 = -v0 * v2, p11 = 1 - v1 * v1, p12 = -v1 * v2, p22 = 1 - v2 * v2;
                    A9[0] += p00; A9[1] += p01; A9[2] += p02; A9[4] += p11; A9[5] += p12; A9[8] += p22;
                    b0 += p00 * c0 + p01 * c1 + p02 * c2; b1 += p01 * c0 + p11 * c1 + p12 * c2; b2 += p02 * c0 + p12 * c1 + p22 * c2;
                }
                A9[3] = A9[1]; A9[6] = A9[2]; A9[7] = A9[5];
                // Parallel bearings leave A singular along them: the scaled determinant says so, and the point keeps its X.
                double det = A9[0] * (A9[4] * A9[8] - A9[5] * A9[7]) - A9[1] * (A9[3] * A9[8] - A9[5] * A9[6]) + A9[2] * (A9[3] * A9[7] - A9[4] * A9[6]);
                if (!(det > 1e-12 * A9[0] * A9[4] * A9[8])) continue;
                var ai = Invert3(A9);
                xp[j * 3] = ai[0] * b0 + ai[1] * b1 + ai[2] * b2;
                xp[j * 3 + 1] = ai[3] * b0 + ai[4] * b1 + ai[5] * b2;
                xp[j * 3 + 2] = ai[6] * b0 + ai[7] * b1 + ai[8] * b2;
            }
            // Angle (as tan(theta/2)) between every observation's bearing and its point, against max(0.1 deg, 5x the
            // inliers' median) - the median from a histogram (AngleBin), as the GPU solver computes it.
            var ang = new double[no];
            var hist = new int[AngleBins];
            int liveCount = 0;
            for (int t = 0; t < no; t++)
            {
                int i = oc[t] * 3, j = op[t] * 3;
                ang[t] = TanHalfAngle(v[t * 3], v[t * 3 + 1], v[t * 3 + 2], xp[j] - c[i], xp[j + 1] - c[i + 1], xp[j + 2] - c[i + 2]);
                if (!off[t]) { hist[AngleBin(ang[t])]++; liveCount++; }
            }
            int medianBin = 0;
            for (int acc = 0; medianBin < AngleBins; medianBin++) { acc += hist[medianBin]; if (acc > liveCount / 2) break; }
            double limit = InlierLimit(AngleBinUpper(Math.Min(medianBin, AngleBins - 1)));
            var newOff = new bool[no];
            Array.Clear(live);
            for (int t = 0; t < no; t++) { newOff[t] = ang[t] > limit; if (!newOff[t]) live[op[t]]++; }
            // A point with one inlier carries no information: retire it.
            for (int t = 0; t < no; t++)
                if (!newOff[t] && live[op[t]] < 2) { newOff[t] = true; live[op[t]]--; }
            int dropped = 0, readmitted = 0, outliers = 0;
            for (int t = 0; t < no; t++)
            {
                if (newOff[t] && !off[t]) dropped++;
                if (!newOff[t] && off[t]) readmitted++;
                if (newOff[t]) outliers++;
            }
            Array.Copy(newOff, off, no);
            rejectedTotal = outliers;
            roundLog.Add($"{iter - roundStart}/{accepted - acceptedStart}{(roundConverged ? "" : " (cap)")} -{dropped}+{readmitted}");
            if (dropped == 0 && readmitted == 0) break;
            // Every scale at its optimum again (re-triangulated points moved).
            for (int t = 0; t < no; t++)
            {
                int i = oc[t] * 3, j = op[t] * 3;
                double ux = xp[j] - c[i], uy = xp[j + 1] - c[i + 1], uz = xp[j + 2] - c[i + 2];
                double uu = ux * ux + uy * uy + uz * uz;
                d[t] = uu > 0 ? Math.Max(0, (v[t * 3] * ux + v[t * 3 + 1] * uy + v[t * 3 + 2] * uz) / uu) : 0;
            }
            cost = Evaluate(c, xp, d, true);
            lambda = 1e-4;
        }

        void ObsBlock(int t, Span<double> m)   // H_cX of observation t after the d elimination: e I + a a^T / H_dd
        {
            double s = 1 / hdd[t], a0 = a[t * 3], a1 = a[t * 3 + 1], a2 = a[t * 3 + 2], ee = e2[t];
            m[0] = ee + s * a0 * a0; m[1] = s * a0 * a1; m[2] = s * a0 * a2;
            m[3] = m[1]; m[4] = ee + s * a1 * a1; m[5] = s * a1 * a2;
            m[6] = m[2]; m[7] = m[5]; m[8] = ee + s * a2 * a2;
        }

        var centres = ToCurrentFrame(cams, connected, i => new Vector3((float)c[i * 3], (float)c[i * 3 + 1], (float)c[i * 3 + 2]));
        return new PositioningResult(centres, $"robust positioning: {no} obs, {roundsRun} rounds [iterations/accepted -dropped+readmitted: {string.Join(", ", roundLog)}], {iter} iterations ({accepted} accepted), {rejectedTotal} outliers, cost {cost0:G4} -> {cost:G4}", roundConverged, rejectedTotal, no);
    }

    /// <summary>The positioning's observations (shared by the managed and GPU solvers): those of a connected camera on a
    /// point seen by at least two connected cameras, each as its camera, point and unit WORLD bearing R^T K^-1 [u v 1].</summary>
    internal static (int[] Oc, int[] Op, double[] V) PositioningObservations(IReadOnlyList<CameraParams> cams, double[][] rot,
        bool[] connected, IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal)
    {
        var seen = new int[pointCount];
        foreach (var o in obs) if (connected[o.Camera]) seen[o.Point]++;
        var use = new List<int>();
        for (int k = 0; k < obs.Count; k++) if (connected[obs[k].Camera] && seen[obs[k].Point] >= 2) use.Add(k);
        int no = use.Count;
        var oc = new int[no]; var op = new int[no]; var v = new double[no * 3];
        for (int t = 0; t < no; t++)
        {
            var o = obs[use[t]]; var cam = cams[o.Camera]; var rr = rot[o.Camera];
            oc[t] = o.Camera; op[t] = o.Point;
            double x = (o.U - cam.CenterX) / focal, y = (o.V - cam.CenterY) / focal;
            double bx = rr[0] * x + rr[3] * y + rr[6], by = rr[1] * x + rr[4] * y + rr[7], bz = rr[2] * x + rr[5] * y + rr[8];
            double bn = Math.Sqrt(bx * bx + by * by + bz * bz);
            v[t * 3] = bx / bn; v[t * 3 + 1] = by / bn; v[t * 3 + 2] = bz / bn;
        }
        return (oc, op, v);
    }

    /// <summary>
    /// Leave-one-out reprojection per camera: each 3+-view track is triangulated from its OTHER placed cameras and
    /// reprojected into this one (at most <paramref name="maxPerCamera"/> samples each); a point behind the camera counts
    /// 1e6 px. Returns each camera's median miss (NaN: no sample) and the sample counts. Cameras in
    /// <paramref name="unplaced"/> have no pose: they are neither evaluated nor used to triangulate. Evaluating them (until
    /// 2026-10-01) let 21 placeholder cameras of DrJohnson's 44 set the "typical" miss to 1e6 px, so the misplaced limit
    /// (4x typical) was 4e6 px and the check was silently off.
    /// </summary>
    public static (double[] Medians, int[] Counts) LeaveOneOutMedians(IReadOnlyList<CameraParams> cams,
        IReadOnlyList<IReadOnlyList<(int Camera, float U, float V)>> tracks, ISet<int> unplaced, int maxPerCamera = 400)
    {
        int n = cams.Count;
        var errs = new List<double>[n];
        for (int c = 0; c < n; c++) errs[c] = new List<double>();
        var others = new List<(int Camera, float U, float V)>();
        foreach (var track in tracks)
        {
            if (track.Count < 3) continue;
            for (int k = 0; k < track.Count; k++)
            {
                var me = track[k];
                if (unplaced.Contains(me.Camera) || errs[me.Camera].Count >= maxPerCamera) continue;
                others.Clear();
                for (int j = 0; j < track.Count; j++)
                    if (j != k && !unplaced.Contains(track[j].Camera)) others.Add(track[j]);
                if (others.Count < 2 || !BundleAdjuster.Triangulate(cams, others, out var x)) continue;
                if (!WorldSpaceGeometry.Project(cams[me.Camera], x, out var u, out var v, out _)) { errs[me.Camera].Add(1e6); continue; }
                errs[me.Camera].Add(Math.Sqrt((u - me.U) * (u - me.U) + (v - me.V) * (v - me.V)));
            }
        }
        var medians = new double[n];
        var counts = new int[n];
        for (int c = 0; c < n; c++)
        {
            counts[c] = errs[c].Count;
            if (counts[c] == 0) { medians[c] = double.NaN; continue; }
            errs[c].Sort();
            medians[c] = errs[c][errs[c].Count / 2];
        }
        return (medians, counts);
    }

    /// <summary>A positioning solution (free scale and translation) expressed at the connected cameras' CURRENT centre and
    /// spread; unconnected cameras keep their position. Centre = coordinate-wise median, spread = median distance from it:
    /// ROBUST, because the solution can leave a few cameras wildly off. With the mean and the RMS spread (until 2026-09-30),
    /// TruckFull's 3 misplaced cameras (leave-one-out up to 1e6 px) carried nearly all the spread, the matched scale shrank the
    /// real cluster ~390x, and BA then solved that cluster 0.10% from COLMAP at 1/390 of the scene's scale: held PSNR 4.7.</summary>
    /// <param name="trusted">Cameras whose CURRENT pose means something (the depth cascade posed them); null = all. Views
    /// the cascade could not pose enter SfM with placeholder poses (2026-09-30, DrJohnson) and must not decide the frame.
    /// When no connected camera is trusted, the solution is matched to ALL trusted cameras' centre and spread - the scene
    /// scale, which every scale-dependent step downstream assumes (a wrong one cost held-out PSNR 4.7, b52).</param>
    public static Vector3[] ToCurrentFrame(IReadOnlyList<CameraParams> cams, bool[] connected, Func<int, Vector3> solution,
        bool[]? trusted = null)
    {
        int n = cams.Count;
        var buf = new List<float>(n);
        float Median() { buf.Sort(); int m = buf.Count; return m == 0 ? 0 : (m & 1) == 1 ? buf[m / 2] : 0.5f * (buf[m / 2 - 1] + buf[m / 2]); }
        float MedianOver(Func<int, bool> set, Func<int, float> f) { buf.Clear(); for (int i = 0; i < n; i++) if (set(i)) buf.Add(f(i)); return Median(); }
        Vector3 Centroid(Func<int, bool> set, Func<int, Vector3> at) =>
            new(MedianOver(set, i => at(i).X), MedianOver(set, i => at(i).Y), MedianOver(set, i => at(i).Z));
        float Spread(Func<int, bool> set, Func<int, Vector3> at, Vector3 mid) => MedianOver(set, i => (at(i) - mid).Length());
        bool IsTrusted(int i) => trusted == null || trusted[i];
        int connectedTrusted = 0, trustedCount = 0;
        for (int i = 0; i < n; i++) { if (IsTrusted(i)) trustedCount++; if (connected[i] && IsTrusted(i)) connectedTrusted++; }
        // Both sides over the SAME cameras when there are enough of them to have a spread (connected and trusted, 3+);
        // otherwise the component solution onto the trusted cameras' frame; with nothing trusted at all, the old
        // behaviour (every connected camera). One shared camera has spread 0: scale 0 put every connected camera on one
        // point (DrJohnson b73/b75, 2026-10-01: cascade posed 6 of 44, global init connected 23, overlap 1).
        const int MinSharedCameras = 3;
        bool shared = connectedTrusted >= MinSharedCameras;
        Func<int, bool> curSet = shared ? i => connected[i] && IsTrusted(i) : trustedCount >= 2 ? IsTrusted : i => connected[i];
        Func<int, bool> solSet = shared ? curSet : i => connected[i];
        Vector3 Cur(int i) => cams[i].Position;
        var curMid = Centroid(curSet, Cur); var solMid = Centroid(solSet, solution);
        float scale = Spread(curSet, Cur, curMid) / Math.Max(Spread(solSet, solution, solMid), 1e-30f);
        var centres = new Vector3[n];
        for (int i = 0; i < n; i++) centres[i] = connected[i] ? curMid + (solution(i) - solMid) * scale : cams[i].Position;
        return centres;
    }

    static void Mul3(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> o)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                o[i * 3 + j] = x[i * 3] * y[j] + x[i * 3 + 1] * y[3 + j] + x[i * 3 + 2] * y[6 + j];
    }

    /// <summary>In-place dense Cholesky of the symmetric positive-definite m (row-major, lower half used), then solves
    /// m x = b into b. False when m is not positive definite.</summary>
    static bool CholeskySolve(double[] m, int dim, double[] b)
    {
        for (int j = 0; j < dim; j++)
        {
            double s = m[j * dim + j];
            for (int k = 0; k < j; k++) s -= m[j * dim + k] * m[j * dim + k];
            if (!(s > 0)) return false;
            double l = Math.Sqrt(s);
            m[j * dim + j] = l;
            for (int i = j + 1; i < dim; i++)
            {
                double t = m[i * dim + j];
                for (int k = 0; k < j; k++) t -= m[i * dim + k] * m[j * dim + k];
                m[i * dim + j] = t / l;
            }
        }
        for (int i = 0; i < dim; i++)
        {
            double t = b[i];
            for (int k = 0; k < i; k++) t -= m[i * dim + k] * b[k];
            b[i] = t / m[i * dim + i];
        }
        for (int i = dim - 1; i >= 0; i--)
        {
            double t = b[i];
            for (int k = i + 1; k < dim; k++) t -= m[k * dim + i] * b[k];
            b[i] = t / m[i * dim + i];
        }
        return true;
    }

    // ── the whole initialisation ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Replace <paramref name="cams"/>' poses with the global SfM solution, expressed in their current world frame: the
    /// averaged rotations are rotated onto the current ones (the gauge), then global positioning starts from the current
    /// centres and keeps their spread. Returns a one-line summary.
    /// </summary>
    public static string Apply(List<CameraParams> cams, IReadOnlyList<RelativePose> edges,
        IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal, double[][]? fixedRotations = null) =>
        // No accelerator: the managed positioning, which completes synchronously.
        ApplyAsync(cams, edges, obs, pointCount, focal, fixedRotations).GetAwaiter().GetResult();

    /// <summary><see cref="Apply"/>, with the positioning on <paramref name="accelerator"/> (<see cref="GpuGlobalPositioner"/>,
    /// the same algorithm) when one is given.</summary>
    public static async Task<string> ApplyAsync(List<CameraParams> cams, IReadOnlyList<RelativePose> edges,
        IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal, double[][]? fixedRotations = null,
        ILGPU.Runtime.Accelerator? accelerator = null)
        => (await ApplyAsync(cams, edges, obs, pointCount, focal, fixedRotations, accelerator, null)).Summary;

    /// <summary>As the overload above, with only the cameras whose current pose is meaningful (<paramref name="trusted"/>;
    /// null = all) setting the frame, and returning which cameras the solution placed.</summary>
    public static async Task<(string Summary, bool[] Connected)> ApplyAsync(List<CameraParams> cams,
        IReadOnlyList<RelativePose> edges, IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal,
        double[][]? fixedRotations, ILGPU.Runtime.Accelerator? accelerator, bool[]? trusted)
    {
        int n = cams.Count;
        var current = cams.Select(RotationOf).ToArray();
        double[][] rot;
        bool[] connected;
        string loopNote = "";
        if (fixedRotations != null)
        {
            // DIAGNOSIS: rotations given (already in the cameras' frame) - positioning only.
            rot = fixedRotations.Select(r => (double[])r.Clone()).ToArray();
            connected = Enumerable.Repeat(true, n).ToArray();
        }
        else
        {
            var consistent = FilterByLoops(edges);
            loopNote = $"loop-consistent pairs {consistent.Count} of {edges.Count}; ";
            edges = consistent;
            rot = AverageRotations(n, edges, current, out connected);
            var gauge = new bool[n];
            bool anyGauge = false;
            for (int i = 0; i < n; i++) { gauge[i] = connected[i] && (trusted == null || trusted[i]); anyGauge |= gauge[i]; }
            // A placeholder rotation says nothing; with no trusted camera in the component its frame stays the root one.
            if (anyGauge)
            {
                var q = AlignFrame(rot, current, gauge);
                for (int i = 0; i < n; i++) if (connected[i]) rot[i] = Mul(rot[i], q);
            }
        }
        var resid = edges.Select(e => AngleDeg(rot[e.B], Mul(e.R, rot[e.A]))).OrderBy(x => x).ToList();
        // A camera the positioning has fewer than 2 observations of has an (almost) empty block in the camera system:
        // the damped Cholesky fails on every step and NO camera moves (TruckFull b63). It is not placed here; the
        // pipeline's re-registration places it against the adjusted points, like any unconnected camera.
        var obsPerCam = new int[n];
        foreach (var o in obs) obsPerCam[o.Camera]++;
        int starved = 0;
        for (int i = 0; i < n; i++) if (connected[i] && obsPerCam[i] < 2) { connected[i] = false; starved++; }
        PositioningResult gp;
        int connectedCount = connected.Count(c => c);
        if (obs.Count == 0 || connectedCount < 2)
            gp = new PositioningResult(cams.Select(c => c.Position).ToArray(),
                $"positioning skipped: {obs.Count} observations, {connectedCount} connected camera(s)", false, 0, obs.Count);
        else if (accelerator != null)
        {
            using var gpu = new GpuGlobalPositioner(accelerator, cams, rot, connected, obs, pointCount, focal);
            gp = await gpu.SolveAsync();
        }
        else gp = GlobalPositioningRobust(cams, rot, connected, obs, pointCount, focal);
        var (centres, positioning) = (gp.Centres, gp.Summary);
        // The positioners map onto every connected camera; re-map onto the trusted ones (a translation + scale, so
        // this equals mapping the raw solution).
        if (trusted != null) { var mapped = centres; centres = ToCurrentFrame(cams, connected, i => mapped[i], trusted); }
        int moved = 0;
        for (int i = 0; i < n; i++)
        {
            if (!connected[i]) continue;
            SetRotation(cams[i], rot[i]);
            cams[i].Position = centres[i];
            moved++;
        }
        return ($"{moved}/{n} cameras from {edges.Count} pair rotations and {obs.Count} track observations; pair rotation " +
               (resid.Count > 0 ? $"residual median {resid[resid.Count / 2]:F2} deg p90 {resid[resid.Count * 9 / 10]:F2} deg; " : "residual n/a (no pairs); ") +
               loopNote +
               (starved > 0 ? $"{starved} connected camera(s) without 2 positioning observations left to re-registration; " : "") +
               $"{positioning}", connected);
    }
}
