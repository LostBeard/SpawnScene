using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Bundle adjustment: jointly refine every camera pose and every 3D feature point so the points
/// reproject onto the pixels they were matched at.
///
/// Why it exists: DAv3's joint passes place cameras to a few percent of the camera spread (Truck:
/// median 7.8%, p90 27% vs COLMAP) and the chunk fold adds its own error. MEASURED 2026-09-24, Truck
/// 2K: with COLMAP poses held-out views render as a recognisable truck; with the cascade's poses the
/// same trainer produces shredded ghost layers. Splatting needs SfM-grade poses; this is the SfM
/// refinement step, seeded by the cascade instead of by incremental registration.
///
/// Model = <see cref="WorldSpaceGeometry.Project"/> exactly: R rows are [right; down; forward], Xc =
/// R (X - C), u = fx x/z + cx, v = fy y/z + cy. Intrinsics are held fixed (DAv3 focal is within ~3% on
/// real captures once it is in the right pixel grid, and a free focal per view is a gauge the tracks
/// here cannot always pin down).
///
/// Solver: Levenberg-Marquardt on the normal equations with the point block eliminated (Schur
/// complement), the reduced camera system solved by block-Jacobi preconditioned conjugate gradients.
/// Huber-weighted residuals (IRLS) so mismatched features do not drag the solution; observations far
/// past the Huber scale are dropped between rounds. Camera 0 is held fixed (6 of the 7 gauge freedoms);
/// the remaining scale freedom is absorbed by the damping.
/// </summary>
public sealed class BundleAdjuster
{
    /// <summary>A pixel observation of <see cref="Point"/> in <see cref="Camera"/>.</summary>
    public readonly record struct Observation(int Camera, int Point, float U, float V);

    public sealed class Options
    {
        public int MaxIterations { get; init; } = 30;
        /// <summary>Huber scale in pixels.</summary>
        public double HuberPixels { get; init; } = 2.0;
        /// <summary>After each round, drop observations whose residual exceeds this many Huber scales.</summary>
        public double OutlierHubers { get; init; } = 4.0;
        public int Rounds { get; init; } = 3;
        public int MaxCgIterations { get; init; } = 200;
        public double CgTolerance { get; init; } = 1e-6;
        /// <summary>Called after each round: (round, LM iterations, inlier RMS px, observations kept).</summary>
        public Action<int, int, double, int>? RoundLog { get; init; }
    }

    public sealed record Result(
        double InitialRmsPixels, double FinalRmsPixels, int Iterations, int Observations, int ObservationsKept,
        int Points, double Seconds);

    // Per camera: R (row-major 3x3), C, intrinsics.
    readonly double[] _r;
    readonly double[] _c;
    readonly double[] _k; // fx fy cx cy
    readonly bool[] _fixed;
    readonly double[] _x; // points, 3 per point
    readonly List<Observation> _obs;
    readonly bool[] _keep;
    readonly int _nc, _np;

    // Shared focal length (one camera, many frames): fx = fy = _f for every view, solved with the poses.
    // DAv3 predicts a different focal per frame; with those held fixed, BA bends the geometry to absorb
    // them - MEASURED on Truck: 1.51 px self-consistent and still 5.5% (median) off COLMAP.
    readonly bool _sharedFocal;
    double _f;

    // Parameter layout: camera ci pose at [ci*6, ci*6+6); the shared focal (if any) at index _nc*6.
    int ParamCount => _nc * 6 + (_sharedFocal ? 1 : 0);
    int FocalIndex => _nc * 6;
    const int Cols = 7; // per observation: 6 pose + focal

    public int CameraCount => _nc;
    public int PointCount => _np;
    public double SharedFocal => _f;

    public BundleAdjuster(IReadOnlyList<CameraParams> cameras, IReadOnlyList<Vector3> points,
        IReadOnlyList<Observation> observations, int fixedCamera = 0, bool sharedFocal = false)
    {
        _nc = cameras.Count;
        _np = points.Count;
        _r = new double[_nc * 9];
        _c = new double[_nc * 3];
        _k = new double[_nc * 4];
        _fixed = new bool[_nc];
        if (fixedCamera >= 0 && fixedCamera < _nc) _fixed[fixedCamera] = true;
        var focals = new List<double>();
        for (int i = 0; i < _nc; i++)
        {
            WorldSpaceGeometry.GetOpenCvAxes(cameras[i], out var right, out var down, out var fwd);
            SetRow(_r, i, 0, right); SetRow(_r, i, 1, down); SetRow(_r, i, 2, fwd);
            _c[i * 3] = cameras[i].Position.X; _c[i * 3 + 1] = cameras[i].Position.Y; _c[i * 3 + 2] = cameras[i].Position.Z;
            _k[i * 4] = cameras[i].FocalX; _k[i * 4 + 1] = cameras[i].FocalY;
            _k[i * 4 + 2] = cameras[i].CenterX; _k[i * 4 + 3] = cameras[i].CenterY;
            focals.Add(0.5 * (cameras[i].FocalX + cameras[i].FocalY));
        }
        _sharedFocal = sharedFocal;
        focals.Sort();
        _f = focals.Count > 0 ? focals[focals.Count / 2] : 1;
        _x = new double[_np * 3];
        for (int p = 0; p < _np; p++) { _x[p * 3] = points[p].X; _x[p * 3 + 1] = points[p].Y; _x[p * 3 + 2] = points[p].Z; }
        _obs = observations.ToList();
        _keep = new bool[_obs.Count];
        Array.Fill(_keep, true);
    }

    static void SetRow(double[] r, int cam, int row, Vector3 v)
    {
        r[cam * 9 + row * 3] = v.X; r[cam * 9 + row * 3 + 1] = v.Y; r[cam * 9 + row * 3 + 2] = v.Z;
    }

    double Fx(int ci, double f) => _sharedFocal ? f : _k[ci * 4];
    double Fy(int ci, double f) => _sharedFocal ? f : _k[ci * 4 + 1];

    /// <summary>Write the refined pose (and, with a shared focal, the focal) of camera <paramref name="i"/>.</summary>
    public void WriteCamera(int i, CameraParams cam)
    {
        var down = new Vector3((float)_r[i * 9 + 3], (float)_r[i * 9 + 4], (float)_r[i * 9 + 5]);
        var fwd = new Vector3((float)_r[i * 9 + 6], (float)_r[i * 9 + 7], (float)_r[i * 9 + 8]);
        cam.Forward = Vector3.Normalize(fwd);
        cam.Up = Vector3.Normalize(-down);
        cam.Position = new Vector3((float)_c[i * 3], (float)_c[i * 3 + 1], (float)_c[i * 3 + 2]);
        if (_sharedFocal) { cam.FocalX = (float)_f; cam.FocalY = (float)_f; }
    }

    public Vector3 PointAt(int p) => new((float)_x[p * 3], (float)_x[p * 3 + 1], (float)_x[p * 3 + 2]);

    /// <summary>
    /// Per camera: observations given, observations surviving the outlier rounds, and the median reprojection
    /// error of ALL its observations (px; a misplaced camera's median is large even when its few kept ones fit).
    /// </summary>
    public (int Total, int Kept, double MedianError)[] CameraStats()
    {
        var errs = new List<double>[_nc];
        var kept = new int[_nc];
        for (int c = 0; c < _nc; c++) errs[c] = new List<double>();
        for (int i = 0; i < _obs.Count; i++)
        {
            var o = _obs[i];
            if (_keep[i]) kept[o.Camera]++;
            errs[o.Camera].Add(Residual(o, _r, _c, _x, _f, out var ru, out var rv, out _, out _, out _)
                ? Math.Sqrt(ru * ru + rv * rv) : 1e6);
        }
        var stats = new (int, int, double)[_nc];
        for (int c = 0; c < _nc; c++)
        {
            errs[c].Sort();
            stats[c] = (errs[c].Count, kept[c], errs[c].Count == 0 ? double.NaN : errs[c][errs[c].Count / 2]);
        }
        return stats;
    }

    /// <summary>Per point: how many of its observations survived the outlier rounds.</summary>
    public int[] KeptObservationsPerPoint()
    {
        var n = new int[_np];
        for (int i = 0; i < _obs.Count; i++) if (_keep[i]) n[_obs[i].Point]++;
        return n;
    }

    /// <summary>Residual (pixels) of observation <paramref name="o"/>; false if the point is behind the camera.</summary>
    bool Residual(in Observation o, double[] r, double[] c, double[] x, double f, out double ru, out double rv,
        out double xc, out double yc, out double zc)
    {
        int ci = o.Camera, pi = o.Point;
        double dx = x[pi * 3] - c[ci * 3], dy = x[pi * 3 + 1] - c[ci * 3 + 1], dz = x[pi * 3 + 2] - c[ci * 3 + 2];
        int b = ci * 9;
        xc = r[b] * dx + r[b + 1] * dy + r[b + 2] * dz;
        yc = r[b + 3] * dx + r[b + 4] * dy + r[b + 5] * dz;
        zc = r[b + 6] * dx + r[b + 7] * dy + r[b + 8] * dz;
        if (zc <= 1e-9) { ru = rv = 0; return false; }
        ru = Fx(ci, f) * xc / zc + _k[ci * 4 + 2] - o.U;
        rv = Fy(ci, f) * yc / zc + _k[ci * 4 + 3] - o.V;
        return true;
    }

    double RobustCost(double[] r, double[] c, double[] x, double f)
    {
        double cost = 0;
        double d = _opts.HuberPixels;
        for (int i = 0; i < _obs.Count; i++)
        {
            if (!_keep[i]) continue;
            if (!Residual(_obs[i], r, c, x, f, out var ru, out var rv, out _, out _, out _)) { cost += 1e6; continue; }
            double e = Math.Sqrt(ru * ru + rv * rv);
            cost += e <= d ? 0.5 * e * e : d * (e - 0.5 * d);
        }
        return cost;
    }

    public double RmsPixels()
    {
        double s = 0; int n = 0;
        for (int i = 0; i < _obs.Count; i++)
        {
            if (!_keep[i]) continue;
            if (!Residual(_obs[i], _r, _c, _x, _f, out var ru, out var rv, out _, out _, out _)) continue;
            s += ru * ru + rv * rv; n++;
        }
        return n == 0 ? double.NaN : Math.Sqrt(s / n);
    }

    Options _opts = new();
    int[][]? _cameraNeighbours;

    /// <summary>
    /// Camera blocks of S that can be non-zero: each camera and every camera it shares a point with (over all
    /// observations, so a superset of any round's). Computed once - it was a HashSet insert per point pair per
    /// LM attempt.
    /// </summary>
    int[][] CameraNeighbours()
    {
        var sets = new HashSet<int>[_nc];
        for (int c = 0; c < _nc; c++) sets[c] = new HashSet<int> { c };
        var camsOfPoint = new List<int>?[_np];
        foreach (var o in _obs) (camsOfPoint[o.Point] ??= new List<int>()).Add(o.Camera);
        foreach (var list in camsOfPoint)
        {
            if (list == null) continue;
            foreach (int a in list) foreach (int b in list) sets[a].Add(b);
        }
        var cols = new int[_nc][];
        for (int c = 0; c < _nc; c++) { cols[c] = sets[c].ToArray(); Array.Sort(cols[c]); }
        return cols;
    }
    long _tBuild, _tSolve, _tCost, _tCg;
    int _attempts;

    /// <summary>Where the time went (seconds): normal equations, damped solve (Schur + CG), of which CG, cost evaluation; LM attempts.</summary>
    public string TimingSummary()
    {
        double f = System.Diagnostics.Stopwatch.Frequency;
        return $"build {_tBuild / f:F2}s, solve {_tSolve / f:F2}s (CG {_tCg / f:F2}s), cost {_tCost / f:F2}s, {_attempts} attempts";
    }

    public Result Solve(Options? options = null)
    {
        _opts = options ?? new Options();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double initialRms = RmsPixels();
        int iters = 0;
        for (int round = 0; round < _opts.Rounds; round++)
        {
            int roundIters = RunLevenbergMarquardt();
            iters += roundIters;
            _opts.RoundLog?.Invoke(round, roundIters, RmsPixels(), _keep.Count(k => k));
            // Drop what the fit says is a mismatch, then refit without it.
            double limit = _opts.OutlierHubers * _opts.HuberPixels;
            int dropped = 0;
            for (int i = 0; i < _obs.Count; i++)
            {
                if (!_keep[i]) continue;
                if (!Residual(_obs[i], _r, _c, _x, _f, out var ru, out var rv, out _, out _, out _)
                    || ru * ru + rv * rv > limit * limit) { _keep[i] = false; dropped++; }
            }
            if (dropped == 0) break;
        }
        return new Result(initialRms, RmsPixels(), iters, _obs.Count, _keep.Count(k => k), _np, sw.Elapsed.TotalSeconds);
    }

    int RunLevenbergMarquardt()
    {
        double lambda = 1e-3;
        double cost = RobustCost(_r, _c, _x, _f);
        int it = 0;
        var nr = new double[_r.Length];
        var nc = new double[_c.Length];
        var nx = new double[_x.Length];
        for (; it < _opts.MaxIterations; it++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var ne = BuildNormalEquations();
            _tBuild += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            bool improved = false;
            for (int attempt = 0; attempt < 10; attempt++)
            {
                long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                bool solved = SolveDamped(ne, lambda, out var dCam, out var dPt);
                _tSolve += System.Diagnostics.Stopwatch.GetTimestamp() - t1;
                _attempts++;
                if (!solved) { lambda *= 10; continue; }
                double nf = ApplyStep(dCam, dPt, nr, nc, nx);
                long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                double newCost = RobustCost(nr, nc, nx, nf);
                _tCost += System.Diagnostics.Stopwatch.GetTimestamp() - t2;
                if (newCost < cost)
                {
                    Array.Copy(nr, _r, _r.Length); Array.Copy(nc, _c, _c.Length); Array.Copy(nx, _x, _x.Length);
                    _f = nf;
                    double rel = (cost - newCost) / Math.Max(cost, 1e-30);
                    cost = newCost;
                    lambda = Math.Max(lambda / 3, 1e-9);
                    improved = true;
                    if (rel < 1e-7) return it + 1;
                    break;
                }
                lambda *= 10;
            }
            if (!improved) break;
        }
        return it;
    }

    sealed class NormalEquations
    {
        public required double[] U;       // dense ParamCount^2: camera-side J^T W J (pose blocks + focal couplings)
        public required double[] G;       // ParamCount
        public required double[] V;       // 9 per point
        public required double[] GP;      // 3 per point
        public required double[] Wb;      // Cols x 3 per observation
        public required List<int>?[] ByPoint;
    }

    /// <summary>Global parameter index of column <paramref name="a"/> (0..6) of an observation in camera <paramref name="ci"/>; -1 if fixed/absent.</summary>
    int Col(int ci, int a) => a < 6 ? (_fixed[ci] ? -1 : ci * 6 + a) : (_sharedFocal ? FocalIndex : -1);

    NormalEquations BuildNormalEquations()
    {
        int n = ParamCount;
        var u = new double[n * n];
        var g = new double[n];
        var v = new double[_np * 9];
        var gP = new double[_np * 3];
        var wb = new double[_obs.Count * Cols * 3];
        var byPoint = new List<int>?[_np];
        Span<double> jc = stackalloc double[2 * Cols]; // row 0 = du, row 1 = dv
        Span<double> jp = stackalloc double[6];
        Span<int> col = stackalloc int[Cols];
        double d = _opts.HuberPixels;
        for (int i = 0; i < _obs.Count; i++)
        {
            if (!_keep[i]) continue;
            var o = _obs[i];
            if (!Residual(o, _r, _c, _x, _f, out var ru, out var rv, out var xc, out var yc, out var zc)) continue;
            (byPoint[o.Point] ??= new List<int>()).Add(i);
            double e = Math.Sqrt(ru * ru + rv * rv);
            double w = e <= d ? 1.0 : d / e; // Huber IRLS weight
            int ci = o.Camera, pi = o.Point;
            double fx = Fx(ci, _f), fy = Fy(ci, _f);
            double iz = 1.0 / zc, iz2 = iz * iz;
            double a0 = fx * iz, a2 = -fx * xc * iz2;
            double b1 = fy * iz, b2 = -fy * yc * iz2;
            int rb = ci * 9;
            for (int k = 0; k < 3; k++)
            {
                double r0 = _r[rb + k], r1 = _r[rb + 3 + k], r2 = _r[rb + 6 + k];
                jp[k] = a0 * r0 + a2 * r2;
                jp[3 + k] = b1 * r1 + b2 * r2;
            }
            // d/domega = -[Xc]x  (R' = exp([w]x) R); d/dC = -d/dX; d/df = (x/z, y/z)
            jc[0] = a2 * yc; jc[1] = a0 * zc + a2 * -xc; jc[2] = -a0 * yc;
            jc[Cols + 0] = -b1 * zc + b2 * yc; jc[Cols + 1] = b2 * -xc; jc[Cols + 2] = b1 * xc;
            for (int k = 0; k < 3; k++) { jc[3 + k] = -jp[k]; jc[Cols + 3 + k] = -jp[3 + k]; }
            jc[6] = xc * iz; jc[Cols + 6] = yc * iz;

            for (int a = 0; a < Cols; a++) col[a] = Col(ci, a);
            for (int a = 0; a < Cols; a++)
            {
                if (col[a] < 0) continue;
                g[col[a]] += w * (jc[a] * ru + jc[Cols + a] * rv);
                for (int bb = 0; bb < Cols; bb++)
                {
                    if (col[bb] < 0) continue;
                    u[col[a] * n + col[bb]] += w * (jc[a] * jc[bb] + jc[Cols + a] * jc[Cols + bb]);
                }
                for (int bb = 0; bb < 3; bb++)
                    wb[(i * Cols + a) * 3 + bb] = w * (jc[a] * jp[bb] + jc[Cols + a] * jp[3 + bb]);
            }
            for (int a = 0; a < 3; a++)
            {
                gP[pi * 3 + a] += w * (jp[a] * ru + jp[3 + a] * rv);
                for (int bb = 0; bb < 3; bb++)
                    v[pi * 9 + a * 3 + bb] += w * (jp[a] * jp[bb] + jp[3 + a] * jp[3 + bb]);
            }
        }
        return new NormalEquations { U = u, G = g, V = v, GP = gP, Wb = wb, ByPoint = byPoint };
    }

    bool SolveDamped(NormalEquations ne, double lambda, out double[] dCam, out double[] dPt)
    {
        int n = ParamCount;
        dCam = new double[n];
        dPt = new double[_np * 3];

        var vInv = new double[_np * 9];
        Span<double> m = stackalloc double[9];
        for (int p = 0; p < _np; p++)
        {
            for (int a = 0; a < 9; a++) m[a] = ne.V[p * 9 + a];
            for (int a = 0; a < 3; a++) m[a * 4] += lambda * m[a * 4] + 1e-9;
            if (!Invert3(m, vInv.AsSpan(p * 9, 9))) return false;
        }

        // S = U* - sum_p W_p V*_p^-1 W_p^T ; b = -g + sum_p W_p V*_p^-1 gP_p
        var s = (double[])ne.U.Clone();
        var rhs = new double[n];
        for (int k = 0; k < n; k++)
        {
            rhs[k] = -ne.G[k];
            double diag = ne.U[k * n + k];
            bool isFixedPose = k < _nc * 6 && _fixed[k / 6];
            s[k * n + k] += lambda * diag + (isFixedPose ? 1.0 : 1e-9);
        }
        Span<double> wv = stackalloc double[Cols * 3];
        Span<int> colI = stackalloc int[Cols];
        Span<int> colJ = stackalloc int[Cols];
        for (int p = 0; p < _np; p++)
        {
            var list = ne.ByPoint[p];
            if (list == null) continue;
            var vi = vInv.AsSpan(p * 9, 9);
            for (int ii = 0; ii < list.Count; ii++)
            {
                int i = list[ii];
                int ci = _obs[i].Camera;
                for (int a = 0; a < Cols; a++) colI[a] = Col(ci, a);
                for (int a = 0; a < Cols; a++)
                {
                    int ga = colI[a];
                    if (ga < 0) continue;
                    int wbase = (i * Cols + a) * 3;
                    for (int bb = 0; bb < 3; bb++)
                        wv[a * 3 + bb] = ne.Wb[wbase] * vi[bb] + ne.Wb[wbase + 1] * vi[3 + bb] + ne.Wb[wbase + 2] * vi[6 + bb];
                    rhs[ga] += wv[a * 3] * ne.GP[p * 3] + wv[a * 3 + 1] * ne.GP[p * 3 + 1] + wv[a * 3 + 2] * ne.GP[p * 3 + 2];
                }
                // S is symmetric (V^-1 is): visit each unordered pair once, write the block and its transpose.
                for (int jj = ii; jj < list.Count; jj++)
                {
                    int j = list[jj];
                    int cj = _obs[j].Camera;
                    for (int bb = 0; bb < Cols; bb++) colJ[bb] = Col(cj, bb);
                    for (int a = 0; a < Cols; a++)
                    {
                        int ga = colI[a];
                        if (ga < 0) continue;
                        double w0 = wv[a * 3], w1 = wv[a * 3 + 1], w2 = wv[a * 3 + 2];
                        int rowA = ga * n;
                        for (int bb = 0; bb < Cols; bb++)
                        {
                            int gb = colJ[bb];
                            if (gb < 0) continue;
                            int jb = (j * Cols + bb) * 3;
                            double val = w0 * ne.Wb[jb] + w1 * ne.Wb[jb + 1] + w2 * ne.Wb[jb + 2];
                            s[rowA + gb] -= val;
                            if (jj != ii) s[gb * n + ga] -= val;
                        }
                    }
                }
            }
        }

        var cols = _cameraNeighbours ??= CameraNeighbours();
        long tc = System.Diagnostics.Stopwatch.GetTimestamp();
        bool cgOk = BlockJacobiCg(s, rhs, dCam, n, cols);
        _tCg += System.Diagnostics.Stopwatch.GetTimestamp() - tc;
        if (!cgOk) return false;
        for (int ci = 0; ci < _nc; ci++) if (_fixed[ci]) for (int a = 0; a < 6; a++) dCam[ci * 6 + a] = 0;

        // Back-substitute: dP = V*^-1 (-gP - sum W^T dC)
        for (int p = 0; p < _np; p++)
        {
            double t0 = -ne.GP[p * 3], t1 = -ne.GP[p * 3 + 1], t2 = -ne.GP[p * 3 + 2];
            var list = ne.ByPoint[p];
            if (list != null)
                foreach (int i in list)
                {
                    int ci = _obs[i].Camera;
                    for (int a = 0; a < Cols; a++)
                    {
                        int ga = Col(ci, a);
                        if (ga < 0) continue;
                        double dc = dCam[ga];
                        int wbase = (i * Cols + a) * 3;
                        t0 -= ne.Wb[wbase] * dc; t1 -= ne.Wb[wbase + 1] * dc; t2 -= ne.Wb[wbase + 2] * dc;
                    }
                }
            var vi = vInv.AsSpan(p * 9, 9);
            dPt[p * 3] = vi[0] * t0 + vi[1] * t1 + vi[2] * t2;
            dPt[p * 3 + 1] = vi[3] * t0 + vi[4] * t1 + vi[5] * t2;
            dPt[p * 3 + 2] = vi[6] * t0 + vi[7] * t1 + vi[8] * t2;
        }
        return true;
    }

    bool BlockJacobiCg(double[] s, double[] b, double[] x, int n, int[][] cols)
    {
        // 6x6 blocks for the poses, scalar blocks for anything after them (the shared focal).
        int nb = _nc;
        var pre = new double[nb * 36];
        Span<double> blk = stackalloc double[36];
        for (int k = 0; k < nb; k++)
        {
            for (int a = 0; a < 6; a++)
                for (int c = 0; c < 6; c++)
                    blk[a * 6 + c] = s[(k * 6 + a) * n + k * 6 + c];
            if (!InvertSpd(blk, pre.AsSpan(k * 36, 36), 6)) return false;
        }
        int tail0 = nb * 6;
        var tailInv = new double[n - tail0];
        for (int k = tail0; k < n; k++) tailInv[k - tail0] = s[k * n + k] > 0 ? 1 / s[k * n + k] : 0;

        void Pre(double[] r, double[] z)
        {
            for (int k = 0; k < nb; k++)
                for (int a = 0; a < 6; a++)
                {
                    double acc = 0;
                    for (int c = 0; c < 6; c++) acc += pre[k * 36 + a * 6 + c] * r[k * 6 + c];
                    z[k * 6 + a] = acc;
                }
            for (int k = tail0; k < n; k++) z[k] = tailInv[k - tail0] * r[k];
        }

        var r = (double[])b.Clone();
        var z = new double[n];
        Pre(r, z);
        var p = (double[])z.Clone();
        var ap = new double[n];
        double rz = Dot(r, z);
        double b2 = Math.Max(Dot(b, b), 1e-300);
        for (int it = 0; it < _opts.MaxCgIterations; it++)
        {
            SparseMatVec(s, p, ap, n, cols);
            double pap = Dot(p, ap);
            if (pap <= 0) break;
            double alpha = rz / pap;
            for (int i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
            if (Dot(r, r) / b2 < _opts.CgTolerance) break;
            Pre(r, z);
            double rzNew = Dot(r, z);
            double beta = rzNew / rz;
            rz = rzNew;
            for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        }
        foreach (var val in x) if (!double.IsFinite(val)) return false;
        return true;
    }

    /// <summary>
    /// y = S x touching only the camera blocks that can be non-zero (<paramref name="cols"/>), plus the
    /// trailing global columns/rows (the shared focal), which couple to everything.
    /// </summary>
    void SparseMatVec(double[] s, double[] x, double[] y, int n, int[][] cols)
    {
        int tail0 = _nc * 6;
        for (int ci = 0; ci < _nc; ci++)
        {
            var nb = cols[ci];
            for (int a = 0; a < 6; a++)
            {
                int row = (ci * 6 + a) * n;
                double acc = 0;
                foreach (int cj in nb)
                {
                    int c0 = row + cj * 6;
                    int x0 = cj * 6;
                    acc += s[c0] * x[x0] + s[c0 + 1] * x[x0 + 1] + s[c0 + 2] * x[x0 + 2]
                         + s[c0 + 3] * x[x0 + 3] + s[c0 + 4] * x[x0 + 4] + s[c0 + 5] * x[x0 + 5];
                }
                for (int k = tail0; k < n; k++) acc += s[row + k] * x[k];
                y[ci * 6 + a] = acc;
            }
        }
        for (int k = tail0; k < n; k++)
        {
            double acc = 0;
            int row = k * n;
            for (int j = 0; j < n; j++) acc += s[row + j] * x[j];
            y[k] = acc;
        }
    }

    static void MatVec(double[] s, double[] x, double[] y, int n)
    {
        for (int i = 0; i < n; i++)
        {
            double acc = 0;
            int row = i * n;
            for (int j = 0; j < n; j++) acc += s[row + j] * x[j];
            y[i] = acc;
        }
    }

    static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary>Candidate state after a step; returns the candidate shared focal.</summary>
    double ApplyStep(double[] dCam, double[] dPt, double[] nr, double[] nc, double[] nx)
    {
        Array.Copy(_r, nr, _r.Length);
        Array.Copy(_c, nc, _c.Length);
        Span<double> e = stackalloc double[9];
        for (int ci = 0; ci < _nc; ci++)
        {
            if (_fixed[ci]) continue;
            Rodrigues(dCam[ci * 6], dCam[ci * 6 + 1], dCam[ci * 6 + 2], e);
            // R' = exp([w]x) R
            for (int a = 0; a < 3; a++)
                for (int bb = 0; bb < 3; bb++)
                    nr[ci * 9 + a * 3 + bb] = e[a * 3] * _r[ci * 9 + bb] + e[a * 3 + 1] * _r[ci * 9 + 3 + bb] + e[a * 3 + 2] * _r[ci * 9 + 6 + bb];
            for (int a = 0; a < 3; a++) nc[ci * 3 + a] = _c[ci * 3 + a] + dCam[ci * 6 + 3 + a];
        }
        for (int i = 0; i < _x.Length; i++) nx[i] = _x[i] + dPt[i];
        return _sharedFocal ? Math.Max(1e-3, _f + dCam[FocalIndex]) : _f;
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

    static bool Invert3(ReadOnlySpan<double> m, Span<double> o)
    {
        double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
        double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
        double det = a * A + b * B + c * C;
        if (!(Math.Abs(det) > 1e-300)) return false;
        double id = 1 / det;
        o[0] = A * id; o[1] = -(b * i - c * h) * id; o[2] = (b * f - c * e) * id;
        o[3] = B * id; o[4] = (a * i - c * g) * id; o[5] = -(a * f - c * d) * id;
        o[6] = C * id; o[7] = -(a * h - b * g) * id; o[8] = (a * e - b * d) * id;
        return true;
    }

    /// <summary>Inverse of a small SPD matrix via Cholesky.</summary>
    static bool InvertSpd(ReadOnlySpan<double> m, Span<double> inv, int n)
    {
        Span<double> l = stackalloc double[n * n];
        l.Clear();
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = m[i * n + j];
                for (int k = 0; k < j; k++) sum -= l[i * n + k] * l[j * n + k];
                if (i == j) { if (!(sum > 0)) return false; l[i * n + i] = Math.Sqrt(sum); }
                else l[i * n + j] = sum / l[j * n + j];
            }
        Span<double> col = stackalloc double[n];
        for (int c = 0; c < n; c++)
        {
            for (int i = 0; i < n; i++) col[i] = i == c ? 1 : 0;
            for (int i = 0; i < n; i++) { double s2 = col[i]; for (int k = 0; k < i; k++) s2 -= l[i * n + k] * col[k]; col[i] = s2 / l[i * n + i]; }
            for (int i = n - 1; i >= 0; i--) { double s2 = col[i]; for (int k = i + 1; k < n; k++) s2 -= l[k * n + i] * col[k]; col[i] = s2 / l[i * n + i]; }
            for (int i = 0; i < n; i++) inv[i * n + c] = col[i];
        }
        return true;
    }

    // ---- tracks and triangulation ----

    /// <summary>
    /// Merge pairwise matches into multi-view tracks (union-find over (image, feature)). A track that
    /// claims two different features in one image is inconsistent - some match in it is wrong - and is
    /// dropped whole rather than guessed at.
    /// </summary>
    public static List<List<(int Image, int Feature)>> BuildTracks(
        IEnumerable<(int ImageA, int FeatureA, int ImageB, int FeatureB)> matches, int minLength = 2)
    {
        var id = new Dictionary<(int, int), int>();
        var parent = new List<int>();
        int Node((int, int) key)
        {
            if (!id.TryGetValue(key, out var n)) { n = parent.Count; id[key] = n; parent.Add(n); }
            return n;
        }
        int Find(int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
        foreach (var (ia, fa, ib, fb) in matches)
        {
            int a = Find(Node((ia, fa))), b = Find(Node((ib, fb)));
            if (a != b) parent[a] = b;
        }
        var groups = new Dictionary<int, List<(int, int)>>();
        foreach (var (key, n) in id)
        {
            int root = Find(n);
            if (!groups.TryGetValue(root, out var g)) groups[root] = g = new List<(int, int)>();
            g.Add(key);
        }
        var tracks = new List<List<(int, int)>>();
        foreach (var g in groups.Values)
        {
            if (g.Count < minLength) continue;
            if (g.Select(t => t.Item1).Distinct().Count() != g.Count) continue;
            g.Sort();
            tracks.Add(g);
        }
        return tracks;
    }

    /// <summary>
    /// Linear (DLT) triangulation of one track from the given cameras, in normalised image
    /// coordinates. False if the point lands behind any observing camera.
    /// </summary>
    public static bool Triangulate(IReadOnlyList<CameraParams> cams, IReadOnlyList<(int Camera, float U, float V)> obs, out Vector3 point)
    {
        point = default;
        if (obs.Count < 2) return false;
        Span<double> ata = stackalloc double[16];
        ata.Clear();
        Span<double> row = stackalloc double[4];
        foreach (var (ci, u, v) in obs)
        {
            var cam = cams[ci];
            WorldSpaceGeometry.GetOpenCvAxes(cam, out var rr, out var dd, out var ff);
            double x = (u - cam.CenterX) / cam.FocalX, y = (v - cam.CenterY) / cam.FocalY;
            // P = [R | -R C]
            double[] p0 = { rr.X, rr.Y, rr.Z, -Vector3.Dot(rr, cam.Position) };
            double[] p1 = { dd.X, dd.Y, dd.Z, -Vector3.Dot(dd, cam.Position) };
            double[] p2 = { ff.X, ff.Y, ff.Z, -Vector3.Dot(ff, cam.Position) };
            for (int pass = 0; pass < 2; pass++)
            {
                for (int k = 0; k < 4; k++) row[k] = pass == 0 ? x * p2[k] - p0[k] : y * p2[k] - p1[k];
                for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) ata[a * 4 + b] += row[a] * row[b];
            }
        }
        // Smallest eigenvector of the 4x4 by cyclic Jacobi (power iteration is not safe here; see
        // fb-power-iteration-returns-worst-rotation).
        Span<double> vecs = stackalloc double[16];
        JacobiEigen4(ata, vecs, out int smallest);
        double w = vecs[3 * 4 + smallest];
        if (Math.Abs(w) < 1e-12) return false;
        point = new Vector3((float)(vecs[0 * 4 + smallest] / w), (float)(vecs[1 * 4 + smallest] / w), (float)(vecs[2 * 4 + smallest] / w));
        foreach (var (ci, _, _) in obs)
            if (Vector3.Dot(point - cams[ci].Position, Vector3.Normalize(cams[ci].Forward)) <= 0) return false;
        return float.IsFinite(point.X) && float.IsFinite(point.Y) && float.IsFinite(point.Z);
    }

    static void JacobiEigen4(Span<double> a, Span<double> v, out int smallest)
    {
        const int n = 4;
        v.Clear();
        for (int i = 0; i < n; i++) v[i * n + i] = 1;
        for (int sweep = 0; sweep < 60; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
            if (off < 1e-30) break;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
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
        smallest = 0;
        for (int i = 1; i < n; i++) if (a[i * n + i] < a[smallest * n + smallest]) smallest = i;
    }
}
