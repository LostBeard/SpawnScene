using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
#if BA_FLOAT
using Real = float;
#else
using Real = double;
#endif

namespace SpawnScene.Services;

/// <summary>
/// <see cref="BundleAdjuster"/>'s solver with the heavy work on the GPU: the same model, the same Levenberg-Marquardt
/// loop, damping, Huber IRLS, outlier rounds, fixed camera and shared focal - but the Schur complement is never
/// built. Every product with it is formed on the fly from per-observation Jacobians:
/// <c>S x = U x - Σ_p W_p V_p⁻¹ W_pᵀ x</c> as a per-point pass (<c>z_p = V_p⁻¹ Σ Wᵀx</c>) and a per-camera pass
/// (<c>Σ W z</c>), both one thread per point / camera over CSR observation lists. The preconditioned CG over the
/// camera system runs on the device too (one workgroup; each thread owns whole cameras, so the 6x6 block-Jacobi step
/// touches only its own entries, and the dot products are shared-memory reductions). The host reads back a few
/// scalars every <see cref="CgCheckEvery"/> CG iterations, the camera step once per attempt, and the cost as 1,024
/// partial sums.
/// </summary>
/// <remarks>
/// Why: the managed solver runs on the Mono WASM interpreter in the browser - TruckFull's first solve took 1,358.8 s
/// (b24, 2026-09-28: build 189 s, solve 1,157 s of which CG 583 s). A first GPU version kept the CG on the host: on the
/// Truck-scale test that was 22,561 CG iterations and 25,068 GPU-to-host round trips for 497 LM attempts - one per CG
/// iteration. Device precision is double by default (see Precision; BA_FLOAT builds an f32 variant for A/B timing). Proven
/// against the managed solver on the ILGPU CPU accelerator (SpawnScene.Tests GpuBundleAdjusterTests).
/// </remarks>
public sealed class GpuBundleAdjuster : IBundleSolution, IDisposable
{
    const int Cols = 7;
    const int J = 23; // per observation: w, ru, rv, jc[2x7], jp[2x3]

    /// <summary>
    /// Device precision: double unless built with BA_FLOAT. WebGPU runs double as Dekker double-float (a pair of f32:
    /// ~48-bit mantissa, float's exponent range), so no constant here may go below ~1e-38 - the managed solver's 1e-300
    /// singularity guard would flush to 0.
    /// </summary>
    public static readonly string Precision = typeof(Real) == typeof(double) ? "f64" : "f32";

#if BA_FLOAT
    static Real Sqrt(Real v) => MathF.Sqrt(v);
    static Real Abs(Real v) => MathF.Abs(v);
#else
    static Real Sqrt(Real v) => Math.Sqrt(v);
    static Real Abs(Real v) => Math.Abs(v);
#endif

    readonly Accelerator _acc;
    readonly double[] _r, _c, _k;
    readonly bool[] _fixed;
    readonly double[] _x;
    readonly List<BundleAdjuster.Observation> _obs;
    readonly bool[] _keep;
    readonly int _nc, _np;
    readonly bool _sharedFocal;
    double _f;
    BundleAdjuster.Options _opts = new();
    long _tDevice, _tCg, _tCost;
    int _attempts, _cgIterations, _readbacks;
    int _focalPreNonPositive, _cgCapHits; double _focalPreMinRatio = double.PositiveInfinity; // diagnostics: S_ff / U*_ff

    /// <summary>
    /// &amp;batrace=1: log every LM attempt that is not accepted (why, CG iterations and residual, lambda) and, when a
    /// candidate's cost is not finite or not lower, the device state behind it (largest |V|, non-finite point inverses,
    /// candidate points and per-observation costs). Written 2026-09-28 for b32: on WebGPU the first TruckFull BA stopped
    /// after 2 LM iterations at 646 px while the same solver on the CPU accelerator matches the managed one exactly.
    /// </summary>
    public static bool TraceAttempts { get; set; }

    /// <summary>
    /// On WebGPU, record one CG batch (<see cref="CgCheckEvery"/> iterations, 5 dispatches each) per round with
    /// <see cref="WebGPUAccelerator.BeginDispatchCapture"/> and REPLAY it for every later batch: one interop crossing
    /// instead of 40 dispatches. MEASURED 2026-09-28 (b36, TruckFull): with the per-camera kernels as workgroups a CG
    /// iteration still took 6.85 ms, and the host enqueue of a single kernel was ~2 ms - launch cost, not GPU work.
    /// Valid because a CG batch touches only round-lifetime buffers and its scalar parameters (tolerance, sizes) are
    /// fixed for the round. &amp;bacapture=0 turns it off (A/B).
    /// </summary>
    public static bool UseDispatchCapture { get; set; } = true;
    WebGPUDispatchPlan? _cgPlan;
    bool _cgPlanTimed;
    int _replays;
    string _failReason = "", _cgTrace = "";
    int _traced;
    const int MaxTracedAttempts = 40;

    public int CameraCount => _nc;
    public int PointCount => _np;
    public double SharedFocal => _f;
    int ParamCount => _nc * 6 + (_sharedFocal ? 1 : 0);

    public GpuBundleAdjuster(Accelerator accelerator, IReadOnlyList<CameraParams> cameras, IReadOnlyList<Vector3> points,
        IReadOnlyList<BundleAdjuster.Observation> observations, int fixedCamera = 0, bool sharedFocal = false)
    {
        _acc = accelerator;
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

    // ── host residual (double), for the outlier rounds and the statistics - same as BundleAdjuster ──

    double Fx(int ci) => _sharedFocal ? _f : _k[ci * 4];
    double Fy(int ci) => _sharedFocal ? _f : _k[ci * 4 + 1];

    bool Residual(in BundleAdjuster.Observation o, out double ru, out double rv)
    {
        int ci = o.Camera, pi = o.Point;
        double dx = _x[pi * 3] - _c[ci * 3], dy = _x[pi * 3 + 1] - _c[ci * 3 + 1], dz = _x[pi * 3 + 2] - _c[ci * 3 + 2];
        int b = ci * 9;
        double xc = _r[b] * dx + _r[b + 1] * dy + _r[b + 2] * dz;
        double yc = _r[b + 3] * dx + _r[b + 4] * dy + _r[b + 5] * dz;
        double zc = _r[b + 6] * dx + _r[b + 7] * dy + _r[b + 8] * dz;
        if (zc <= 1e-9) { ru = rv = 0; return false; }
        ru = Fx(ci) * xc / zc + _k[ci * 4 + 2] - o.U;
        rv = Fy(ci) * yc / zc + _k[ci * 4 + 3] - o.V;
        return true;
    }

    public double RmsPixels()
    {
        double s = 0; int n = 0;
        for (int i = 0; i < _obs.Count; i++)
        {
            if (!_keep[i] || !Residual(_obs[i], out var ru, out var rv)) continue;
            s += ru * ru + rv * rv; n++;
        }
        return n == 0 ? double.NaN : Math.Sqrt(s / n);
    }

    public (int Total, int Kept, double MedianError)[] CameraStats()
    {
        var errs = new List<double>[_nc];
        var kept = new int[_nc];
        for (int c = 0; c < _nc; c++) errs[c] = new List<double>();
        for (int i = 0; i < _obs.Count; i++)
        {
            var o = _obs[i];
            if (_keep[i]) kept[o.Camera]++;
            errs[o.Camera].Add(Residual(o, out var ru, out var rv) ? Math.Sqrt(ru * ru + rv * rv) : 1e6);
        }
        var stats = new (int, int, double)[_nc];
        for (int c = 0; c < _nc; c++)
        {
            errs[c].Sort();
            stats[c] = (errs[c].Count, kept[c], errs[c].Count == 0 ? double.NaN : errs[c][errs[c].Count / 2]);
        }
        return stats;
    }

    public int[] KeptObservationsPerPoint()
    {
        var n = new int[_np];
        for (int i = 0; i < _obs.Count; i++) if (_keep[i]) n[_obs[i].Point]++;
        return n;
    }

    public string TimingSummary()
    {
        double f = System.Diagnostics.Stopwatch.Frequency;
        return $"{Precision}, device {_tDevice / f:F2}s (CG {_tCg / f:F2}s, cost {_tCost / f:F2}s), {_attempts} attempts, " +
               $"{_cgIterations} CG iterations, {_readbacks} readbacks, {_replays} CG-batch replays" +
               $", CG at its cap in {_cgCapHits} of {_attempts} attempts" +
               (_sharedFocal ? $", focal S_ff <= 0 in {_focalPreNonPositive} attempts, min S_ff/U*_ff {_focalPreMinRatio:E2}" : "");
    }

    // ── kernels ──────────────────────────────────────────────────────────────────────────────

    // cams: 16 values per camera = R (9, row-major), C (3), fx, fy, cx, cy. flags: bit0 fixed, bit1 shared focal.
    static void JacobianKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> obsIdx, ArrayView1D<Real, Stride1D.Dense> obsUv,
        ArrayView1D<Real, Stride1D.Dense> cams, ArrayView1D<int, Stride1D.Dense> camFlags,
        ArrayView1D<Real, Stride1D.Dense> pts, ArrayView1D<Real, Stride1D.Dense> jac, Real huber)
    {
        int ci = obsIdx[i * 2], pi = obsIdx[i * 2 + 1];
        int cb = ci * 16, o = i * J;
        Real dx = pts[pi * 3] - cams[cb + 9], dy = pts[pi * 3 + 1] - cams[cb + 10], dz = pts[pi * 3 + 2] - cams[cb + 11];
        Real xc = cams[cb] * dx + cams[cb + 1] * dy + cams[cb + 2] * dz;
        Real yc = cams[cb + 3] * dx + cams[cb + 4] * dy + cams[cb + 5] * dz;
        Real zc = cams[cb + 6] * dx + cams[cb + 7] * dy + cams[cb + 8] * dz;
        if (zc <= (Real)1e-9)
        {
            for (int k = 0; k < J; k++) jac[o + k] = (Real)0;
            return;
        }
        Real fx = cams[cb + 12], fy = cams[cb + 13];
        Real ru = fx * xc / zc + cams[cb + 14] - obsUv[i * 2];
        Real rv = fy * yc / zc + cams[cb + 15] - obsUv[i * 2 + 1];
        Real e = Sqrt(ru * ru + rv * rv);
        Real w = e <= huber ? (Real)1 : huber / e;
        Real iz = (Real)1 / zc, iz2 = iz * iz;
        Real a0 = fx * iz, a2 = -fx * xc * iz2;
        Real b1 = fy * iz, b2 = -fy * yc * iz2;
        jac[o] = w; jac[o + 1] = ru; jac[o + 2] = rv;
        // jp: row 0 at o+17, row 1 at o+20
        for (int k = 0; k < 3; k++)
        {
            Real r0 = cams[cb + k], r1 = cams[cb + 3 + k], r2 = cams[cb + 6 + k];
            jac[o + 17 + k] = a0 * r0 + a2 * r2;
            jac[o + 20 + k] = b1 * r1 + b2 * r2;
        }
        int flags = camFlags[ci];
        bool fixedPose = (flags & 1) != 0, shared = (flags & 2) != 0;
        // jc: row 0 at o+3 .. o+9, row 1 at o+10 .. o+16 (6 pose + focal); fixed pose / no shared focal -> 0.
        Real p0 = fixedPose ? (Real)0 : (Real)1;
        jac[o + 3] = p0 * (a2 * yc);
        jac[o + 4] = p0 * (a0 * zc + a2 * -xc);
        jac[o + 5] = p0 * (-a0 * yc);
        jac[o + 10] = p0 * (-b1 * zc + b2 * yc);
        jac[o + 11] = p0 * (b2 * -xc);
        jac[o + 12] = p0 * (b1 * xc);
        for (int k = 0; k < 3; k++)
        {
            jac[o + 6 + k] = p0 * -jac[o + 17 + k];
            jac[o + 13 + k] = p0 * -jac[o + 20 + k];
        }
        jac[o + 9] = shared ? xc * iz : (Real)0;
        jac[o + 16] = shared ? yc * iz : (Real)0;
    }

    // Per point: V (9) and gP (3) over its observations.
    static void PointAssembleKernel(Index1D p, ArrayView1D<int, Stride1D.Dense> pStart, ArrayView1D<int, Stride1D.Dense> pObs,
        ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> v, ArrayView1D<Real, Stride1D.Dense> gp)
    {
        Real v00 = 0, v01 = 0, v02 = 0, v11 = 0, v12 = 0, v22 = 0, g0 = 0, g1 = 0, g2 = 0;
        for (int t = pStart[p]; t < pStart[p + 1]; t++)
        {
            int o = pObs[t] * J;
            Real w = jac[o], ru = jac[o + 1], rv = jac[o + 2];
            Real a0 = jac[o + 17], a1 = jac[o + 18], a2 = jac[o + 19];
            Real b0 = jac[o + 20], b1 = jac[o + 21], b2 = jac[o + 22];
            v00 += w * (a0 * a0 + b0 * b0); v01 += w * (a0 * a1 + b0 * b1); v02 += w * (a0 * a2 + b0 * b2);
            v11 += w * (a1 * a1 + b1 * b1); v12 += w * (a1 * a2 + b1 * b2); v22 += w * (a2 * a2 + b2 * b2);
            g0 += w * (a0 * ru + b0 * rv); g1 += w * (a1 * ru + b1 * rv); g2 += w * (a2 * ru + b2 * rv);
        }
        v[p * 9] = v00; v[p * 9 + 1] = v01; v[p * 9 + 2] = v02;
        v[p * 9 + 3] = v01; v[p * 9 + 4] = v11; v[p * 9 + 5] = v12;
        v[p * 9 + 6] = v02; v[p * 9 + 7] = v12; v[p * 9 + 8] = v22;
        gp[p * 3] = g0; gp[p * 3 + 1] = g1; gp[p * 3 + 2] = g2;
    }

    // 🔴 Per-camera passes run ONE WORKGROUP PER CAMERA (Grid.IdxX = camera): its threads stride over the camera's
    // observations, each reads an observation's Jacobian once and accumulates every output, then shared-memory sums.
    // MEASURED 2026-09-28 (b35, TruckFull, WebGPU): with one THREAD per camera (251 threads for the GPU, each walking up
    // to ~1,500 observations and re-reading each one's Jacobian per output column) a CG iteration took 10.4 ms and the
    // first BA 975 s - 1.33x the managed solver.

    // Per camera: U (7x7 = pose + focal, symmetric: 28 sums) and g (7) over its observations.
    static void CameraAssembleKernel(ArrayView1D<int, Stride1D.Dense> cStart, ArrayView1D<int, Stride1D.Dense> cObs, ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> u, ArrayView1D<Real, Stride1D.Dense> g)
    {
        var sh = SharedMemory.Allocate<Real>(GroupSize);
        var acc = LocalMemory.Allocate<Real>(35); // g[7], then U's upper triangle row by row (28)
        int c = Grid.IdxX, t = Group.IdxX, dim = Group.DimX;
        for (int k = 0; k < 35; k++) acc[k] = (Real)0;
        for (int q = cStart[c] + t; q < cStart[c + 1]; q += dim)
        {
            int o = cObs[q] * J;
            Real w = jac[o], ru = jac[o + 1], rv = jac[o + 2];
            int k = 7;
            for (int a = 0; a < Cols; a++)
            {
                Real ca = jac[o + 3 + a], cb = jac[o + 10 + a];
                acc[a] += w * (ca * ru + cb * rv);
                for (int b = a; b < Cols; b++)
                {
                    acc[k] += w * (ca * jac[o + 3 + b] + cb * jac[o + 10 + b]);
                    k++;
                }
            }
        }
        for (int a = 0; a < Cols; a++)
        {
            Real total = GroupSum(sh, acc[a]);
            if (t == 0) g[c * Cols + a] = total;
        }
        int kk = 7;
        for (int a = 0; a < Cols; a++)
            for (int b = a; b < Cols; b++)
            {
                Real total = GroupSum(sh, acc[kk]);
                if (t == 0)
                {
                    u[c * 49 + a * 7 + b] = total;
                    u[c * 49 + b * 7 + a] = total;
                }
                kk++;
            }
    }

    // Per point: V* = V + lambda diag(V) + 1e-9 I, its inverse, and t = V*⁻¹ gP.
    //
    // 🔴 Inverted through the Jacobi-scaled matrix V' = D V* D, D = diag(1/sqrt(V*_ii)): unit diagonal, |off-diagonal| <= 1,
    // det' in (0, 1] - then V*⁻¹ = D V'⁻¹ D. The plain cofactor inverse forms det(V*) ~ V³: MEASURED 2026-09-28 (b33,
    // TruckFull, &batrace=1) V reached 5.6e11, so det ~ 1.8e35 - past what WebGPU's Dekker double-float can hold (float's
    // exponent range; its f64_mul goes NaN once an operand passes ~8.3e34). Every such point failed the whole LM attempt
    // ("not invertible") and a larger lambda only made it worse (the right-hand side went NaN), so the first BA stopped
    // after 2 iterations at 646 px. The scaled inverse is the same inverse in exact arithmetic, at any magnitude.
    // det' is floored at 1e-14 (double-float's resolution of these O(1) cofactor sums): below it the point is singular
    // to working precision, where the managed solver's own inverse is equally meaningless but finite. Only a non-finite
    // V fails the attempt.
    static void PointSolveKernel(Index1D p, ArrayView1D<Real, Stride1D.Dense> v, ArrayView1D<Real, Stride1D.Dense> gp,
        ArrayView1D<Real, Stride1D.Dense> vinv, ArrayView1D<Real, Stride1D.Dense> tp, ArrayView1D<Real, Stride1D.Dense> cgs, Real lambda)
    {
        int b = p * 9;
        Real m0 = v[b] + lambda * v[b] + (Real)1e-9;
        Real m4 = v[b + 4] + lambda * v[b + 4] + (Real)1e-9;
        Real m8 = v[b + 8] + lambda * v[b + 8] + (Real)1e-9;
        Real d0 = (Real)1 / Sqrt(m0), d1 = (Real)1 / Sqrt(m4), d2 = (Real)1 / Sqrt(m8);
        // V is symmetric (PointAssembleKernel writes both halves): scaled off-diagonals.
        Real sa = v[b + 1] * d0 * d1, sb = v[b + 2] * d0 * d2, sc = v[b + 5] * d1 * d2;
        Real det = (Real)1 + (Real)2 * sa * sb * sc - sa * sa - sb * sb - sc * sc;
        if (!(det > (Real)(-1) && det < (Real)2))
        {
            // Non-finite (NaN compares false): a broken V, not a conditioning question.
            cgs[7] = (Real)1; // failed attempt (every writer stores the same value)
            for (int k = 0; k < 9; k++) vinv[b + k] = (Real)0;
            tp[p * 3] = (Real)0; tp[p * 3 + 1] = (Real)0; tp[p * 3 + 2] = (Real)0;
            return;
        }
        if (!(det > (Real)1e-14))
        {
            det = (Real)1e-14;
            cgs[8] = (Real)1; // a point singular to working precision (traced)
        }
        Real id = (Real)1 / det;
        Real i0 = ((Real)1 - sc * sc) * id * d0 * d0;
        Real i1 = (sb * sc - sa) * id * d0 * d1;
        Real i2 = (sa * sc - sb) * id * d0 * d2;
        Real i4 = ((Real)1 - sb * sb) * id * d1 * d1;
        Real i5 = (sa * sb - sc) * id * d1 * d2;
        Real i8 = ((Real)1 - sa * sa) * id * d2 * d2;
        vinv[b] = i0; vinv[b + 1] = i1; vinv[b + 2] = i2;
        vinv[b + 3] = i1; vinv[b + 4] = i4; vinv[b + 5] = i5;
        vinv[b + 6] = i2; vinv[b + 7] = i5; vinv[b + 8] = i8;
        Real g0 = gp[p * 3], g1 = gp[p * 3 + 1], g2 = gp[p * 3 + 2];
        tp[p * 3] = i0 * g0 + i1 * g1 + i2 * g2;
        tp[p * 3 + 1] = i1 * g0 + i4 * g1 + i5 * g2;
        tp[p * 3 + 2] = i2 * g0 + i5 * g1 + i8 * g2;
    }

    // W_i row a (1x3) = w (jc0[a] jp0 + jc1[a] jp1). Per camera (one workgroup): out_c[a] = Σ_i W_i[a] · vec3_{p(i)}.
    static void CameraGatherKernel(ArrayView1D<int, Stride1D.Dense> cStart, ArrayView1D<int, Stride1D.Dense> cObs, ArrayView1D<int, Stride1D.Dense> obsIdx, ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> vec3, ArrayView1D<Real, Stride1D.Dense> outCam)
    {
        var sh = SharedMemory.Allocate<Real>(GroupSize);
        var acc = LocalMemory.Allocate<Real>(7);
        int c = Grid.IdxX, t = Group.IdxX, dim = Group.DimX;
        for (int a = 0; a < Cols; a++) acc[a] = (Real)0;
        for (int q = cStart[c] + t; q < cStart[c + 1]; q += dim)
        {
            int i = cObs[q];
            int o = i * J, p = obsIdx[i * 2 + 1];
            Real w = jac[o];
            Real y0 = vec3[p * 3], y1 = vec3[p * 3 + 1], y2 = vec3[p * 3 + 2];
            // Jᵖ·y for both residual rows, then each column's W·y = w (jc0[a] (Jᵖ₀·y) + jc1[a] (Jᵖ₁·y)).
            Real py0 = jac[o + 17] * y0 + jac[o + 18] * y1 + jac[o + 19] * y2;
            Real py1 = jac[o + 20] * y0 + jac[o + 21] * y1 + jac[o + 22] * y2;
            for (int a = 0; a < Cols; a++)
                acc[a] += w * (jac[o + 3 + a] * py0 + jac[o + 10 + a] * py1);
        }
        for (int a = 0; a < Cols; a++)
        {
            Real total = GroupSum(sh, acc[a]);
            if (t == 0) outCam[c * Cols + a] = total;
        }
    }

    // Per camera (one workgroup): the 7x7 block Σ_i W_i V*_p⁻¹ W_iᵀ (symmetric: 28 sums), for the block-Jacobi
    // preconditioner.
    static void CameraDiagKernel(ArrayView1D<int, Stride1D.Dense> cStart, ArrayView1D<int, Stride1D.Dense> cObs, ArrayView1D<int, Stride1D.Dense> obsIdx, ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> vinv, ArrayView1D<Real, Stride1D.Dense> outBlk)
    {
        var sh = SharedMemory.Allocate<Real>(GroupSize);
        var acc = LocalMemory.Allocate<Real>(28);
        var wr = LocalMemory.Allocate<Real>(21); // W_i rows (7 x 3)
        int c = Grid.IdxX, t = Group.IdxX, dim = Group.DimX;
        for (int k = 0; k < 28; k++) acc[k] = (Real)0;
        for (int q = cStart[c] + t; q < cStart[c + 1]; q += dim)
        {
            int i = cObs[q];
            int o = i * J, vb = obsIdx[i * 2 + 1] * 9;
            Real w = jac[o];
            for (int a = 0; a < Cols; a++)
            {
                Real ca = jac[o + 3 + a], cb = jac[o + 10 + a];
                wr[a * 3] = w * (ca * jac[o + 17] + cb * jac[o + 20]);
                wr[a * 3 + 1] = w * (ca * jac[o + 18] + cb * jac[o + 21]);
                wr[a * 3 + 2] = w * (ca * jac[o + 19] + cb * jac[o + 22]);
            }
            int k = 0;
            for (int b = 0; b < Cols; b++)
            {
                // z = V*⁻¹ W_b, then W_a · z for every a <= b.
                Real z0 = vinv[vb] * wr[b * 3] + vinv[vb + 1] * wr[b * 3 + 1] + vinv[vb + 2] * wr[b * 3 + 2];
                Real z1 = vinv[vb + 3] * wr[b * 3] + vinv[vb + 4] * wr[b * 3 + 1] + vinv[vb + 5] * wr[b * 3 + 2];
                Real z2 = vinv[vb + 6] * wr[b * 3] + vinv[vb + 7] * wr[b * 3 + 1] + vinv[vb + 8] * wr[b * 3 + 2];
                for (int a = 0; a <= b; a++)
                {
                    acc[k] += wr[a * 3] * z0 + wr[a * 3 + 1] * z1 + wr[a * 3 + 2] * z2;
                    k++;
                }
            }
        }
        int kk = 0;
        for (int b = 0; b < Cols; b++)
            for (int a = 0; a <= b; a++)
            {
                Real total = GroupSum(sh, acc[kk]);
                if (t == 0)
                {
                    outBlk[c * 49 + a * 7 + b] = total;
                    outBlk[c * 49 + b * 7 + a] = total;
                }
                kk++;
            }
    }

    // Per point: y = Σ_i W_iᵀ x_{c(i)} (7-vectors per camera), then z = V*⁻¹ (sign·gP + y)·mode:
    //   mode 0 (matvec):   z = V*⁻¹ y
    //   mode 1 (backsub):  dP = V*⁻¹ (-gP - y), and the candidate point = x + dP.
    static void PointScatterKernel(Index1D p, ArrayView1D<int, Stride1D.Dense> pStart, ArrayView1D<int, Stride1D.Dense> pObs,
        ArrayView1D<int, Stride1D.Dense> obsIdx, ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> camVec,
        ArrayView1D<Real, Stride1D.Dense> vinv, ArrayView1D<Real, Stride1D.Dense> gp, ArrayView1D<Real, Stride1D.Dense> pts,
        ArrayView1D<Real, Stride1D.Dense> outPt, int mode)
    {
        Real y0 = 0, y1 = 0, y2 = 0;
        for (int t = pStart[p]; t < pStart[p + 1]; t++)
        {
            int i = pObs[t];
            int o = i * J, c = obsIdx[i * 2] * Cols;
            // Wᵀx = w Jᵖᵀ (Jᶜ x): the two residual rows' camera dot products first, then Jᵖᵀ once. MEASURED
            // 2026-09-28 (b37, WebGPU timestamps): the per-column form (8 Jacobian loads and 12 multiply-adds per
            // column) took 2.3 ms per call, 65% of a CG iteration's GPU time.
            Real s0 = 0, s1 = 0;
            for (int a = 0; a < Cols; a++)
            {
                Real xa = camVec[c + a];
                s0 += jac[o + 3 + a] * xa;
                s1 += jac[o + 10 + a] * xa;
            }
            Real w = jac[o];
            s0 *= w;
            s1 *= w;
            y0 += jac[o + 17] * s0 + jac[o + 20] * s1;
            y1 += jac[o + 18] * s0 + jac[o + 21] * s1;
            y2 += jac[o + 19] * s0 + jac[o + 22] * s1;
        }
        if (mode == 1) { y0 = -gp[p * 3] - y0; y1 = -gp[p * 3 + 1] - y1; y2 = -gp[p * 3 + 2] - y2; }
        int b = p * 9;
        Real z0 = vinv[b] * y0 + vinv[b + 1] * y1 + vinv[b + 2] * y2;
        Real z1 = vinv[b + 3] * y0 + vinv[b + 4] * y1 + vinv[b + 5] * y2;
        Real z2 = vinv[b + 6] * y0 + vinv[b + 7] * y1 + vinv[b + 8] * y2;
        if (mode == 1) { z0 += pts[p * 3]; z1 += pts[p * 3 + 1]; z2 += pts[p * 3 + 2]; }
        outPt[p * 3] = z0; outPt[p * 3 + 1] = z1; outPt[p * 3 + 2] = z2;
    }

    // Per observation: Huber cost with candidate cameras + points (behind the camera = 1e6, as BundleAdjuster).
    static void CostKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> obsIdx, ArrayView1D<Real, Stride1D.Dense> obsUv,
        ArrayView1D<Real, Stride1D.Dense> cams, ArrayView1D<Real, Stride1D.Dense> pts, ArrayView1D<Real, Stride1D.Dense> cost, Real huber)
    {
        int ci = obsIdx[i * 2], pi = obsIdx[i * 2 + 1];
        int cb = ci * 16;
        Real dx = pts[pi * 3] - cams[cb + 9], dy = pts[pi * 3 + 1] - cams[cb + 10], dz = pts[pi * 3 + 2] - cams[cb + 11];
        Real xc = cams[cb] * dx + cams[cb + 1] * dy + cams[cb + 2] * dz;
        Real yc = cams[cb + 3] * dx + cams[cb + 4] * dy + cams[cb + 5] * dz;
        Real zc = cams[cb + 6] * dx + cams[cb + 7] * dy + cams[cb + 8] * dz;
        if (zc <= (Real)1e-9) { cost[i] = (Real)1e6; return; }
        Real ru = cams[cb + 12] * xc / zc + cams[cb + 14] - obsUv[i * 2];
        Real rv = cams[cb + 13] * yc / zc + cams[cb + 15] - obsUv[i * 2 + 1];
        Real e = Sqrt(ru * ru + rv * rv);
        cost[i] = e <= huber ? (Real)0.5 * e * e : huber * (e - (Real)0.5 * huber);
    }

    // 1,024 strided partial sums (the host adds them in double).
    static void PartialSumKernel(Index1D t, ArrayView1D<Real, Stride1D.Dense> values, ArrayView1D<Real, Stride1D.Dense> partial, int n)
    {
        Real s = 0;
        for (int i = t; i < n; i += 1024) s += values[i];
        partial[t] = s;
    }

    // ── the PCG on the device ───────────────────────────────────────────────────────────────
    // Parameter vector (n = 6 per camera + the shared focal, if any): camera c's pose at [c*6, c*6+6), the focal at nc*6.
    //   cgv  = x | r | z | p | ap           (5 n)
    //   sysv = rhs | damped diagonal        (2 n)
    //   pre  = 6x6 inverse per camera, then the focal's scalar inverse at nc*36
    //   cgs  = rz, |b|^2, done, iterations, (U_ff + focal damping), last |r|²/|b|², focal S_ff (traced), failed, singular point seen
    const int GroupSize = 256; // shared-memory size; the launch uses the largest power of two the accelerator allows

    /// <summary>
    /// CG iterations launched between reads of the convergence flag (steps after convergence are no-ops). MEASURED
    /// 2026-09-28 (b38, TruckFull, WebGPU, batches replayed): a batch of 8 cost 9.15 ms wall of which ~2.3 ms GPU - the
    /// rest is the replay + readback round trip, and an attempt averages ~136 CG iterations. 32 trades ~1/4 of the round
    /// trips for at most 31 no-op steps.
    /// </summary>
    public const int CgCheckEvery = 32;

    /// <summary>
    /// LM stops once an accepted step lowers the cost by less than this (relative): the managed solver's 1e-7. Only the
    /// f32 build (BA_FLOAT) uses 1e-6 - 1e-7 is below float's resolution of a summed cost: MEASURED on the Truck-scale
    /// test, the f32 cost kept finding "decreases" at that level and ran 330 LM iterations to the managed solver's 222.
    /// </summary>
    public static readonly double RelativeStop = typeof(Real) == typeof(double) ? 1e-7 : 1e-6;

    /// <summary>&amp;barel=X: overrides <see cref="RelativeStop"/> (diagnosis: does LM stop in a flat valley before the geometry
    /// settles? 2026-09-28: final BA converged by this rule at 2.1% of spread vs COLMAP, the one run that hit the iteration
    /// cap instead ended at 0.1%).</summary>
    public static double? RelativeStopOverride { get; set; }

    static void ResetScalarsKernel(Index1D i, ArrayView1D<Real, Stride1D.Dense> cgs) => cgs[i] = (Real)0;

    // A camera-space vector for the per-point/per-camera passes: 7 per camera (pose, focal), fixed poses zeroed.
    static void ExpandKernel(Index1D i, ArrayView1D<Real, Stride1D.Dense> cgv, ArrayView1D<int, Stride1D.Dense> camFlags, ArrayView1D<Real, Stride1D.Dense> camVec, int srcOffset, int nc)
    {
        int idx = i;
        int c = idx / 7, a = idx - c * 7;
        int flags = camFlags[c];
        Real v;
        if (a < 6) v = (flags & 1) != 0 ? (Real)0 : cgv[srcOffset + c * 6 + a];
        else v = (flags & 2) != 0 ? cgv[srcOffset + nc * 6] : (Real)0;
        camVec[idx] = v;
    }

    // Pose block (a, b) of camera c of the damped S: U - Σ W V*⁻¹ Wᵀ (per-camera part) + diagonal damping.
    static Real DampedPoseBlock(ArrayView1D<Real, Stride1D.Dense> u, ArrayView1D<Real, Stride1D.Dense> blk, int c, int a, int b, Real lambda, bool fixedPose)
    {
        Real s = u[c * 49 + a * 7 + b] - blk[c * 49 + a * 7 + b];
        if (a == b) s += fixedPose ? (Real)1 : lambda * u[c * 49 + a * 8] + (Real)1e-9;
        return s;
    }

    // Per camera: the pose rows of rhs (-g + Σ W V*⁻¹ gP) and of the damped diagonal, and the inverse of its 6x6 block
    // (Cholesky, as BundleAdjuster.InvertSpd; a non-SPD block fails the attempt).
    static void PrepareSystemKernel(Index1D ci, ArrayView1D<Real, Stride1D.Dense> u, ArrayView1D<Real, Stride1D.Dense> g, ArrayView1D<Real, Stride1D.Dense> wt, ArrayView1D<Real, Stride1D.Dense> blk, ArrayView1D<int, Stride1D.Dense> camFlags, ArrayView1D<Real, Stride1D.Dense> sysv, ArrayView1D<Real, Stride1D.Dense> pre, ArrayView1D<Real, Stride1D.Dense> cgs, Real lambda, int n)
    {
        int c = ci;
        bool fixedPose = (camFlags[c] & 1) != 0;
        for (int a = 0; a < 6; a++)
        {
            sysv[c * 6 + a] = fixedPose ? (Real)0 : -g[c * 7 + a] + wt[c * 7 + a];
            sysv[n + c * 6 + a] = fixedPose ? (Real)1 : lambda * u[c * 49 + a * 8] + (Real)1e-9;
        }
        var l = LocalMemory.Allocate<Real>(36);
        var y = LocalMemory.Allocate<Real>(6);
        bool ok = true;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j <= i; j++)
            {
                Real sum = DampedPoseBlock(u, blk, c, i, j, lambda, fixedPose);
                for (int k = 0; k < j; k++) sum -= l[i * 6 + k] * l[j * 6 + k];
                if (i == j)
                {
                    if (!(sum > (Real)0)) { ok = false; sum = (Real)1; }
                    l[i * 6 + i] = Sqrt(sum);
                }
                else l[i * 6 + j] = sum / l[j * 6 + j];
            }
        for (int col = 0; col < 6; col++)
        {
            for (int i = 0; i < 6; i++)
            {
                Real s = i == col ? (Real)1 : (Real)0;
                for (int k = 0; k < i; k++) s -= l[i * 6 + k] * y[k];
                y[i] = s / l[i * 6 + i];
            }
            for (int i = 5; i >= 0; i--)
            {
                Real s = y[i];
                for (int k = i + 1; k < 6; k++) s -= l[k * 6 + i] * y[k];
                y[i] = s / l[i * 6 + i];
            }
            for (int i = 0; i < 6; i++) pre[c * 36 + i * 6 + col] = y[i];
        }
        if (!ok) cgs[7] = (Real)1;
    }

    // Per point: aᵀ V*⁻¹ a with a = Σ_(i in p) W_i[focal] - the point's share of the Schur complement's focal diagonal,
    // INCLUDING the cross-camera terms of a point seen by several views.
    static void FocalSchurKernel(Index1D p, ArrayView1D<int, Stride1D.Dense> pStart, ArrayView1D<int, Stride1D.Dense> pObs, ArrayView1D<Real, Stride1D.Dense> jac, ArrayView1D<Real, Stride1D.Dense> vinv, ArrayView1D<Real, Stride1D.Dense> outp)
    {
        Real a0 = 0, a1 = 0, a2 = 0;
        for (int t = pStart[p]; t < pStart[p + 1]; t++)
        {
            int o = pObs[t] * J;
            Real w = jac[o], cf0 = jac[o + 9], cf1 = jac[o + 16];
            a0 += w * (cf0 * jac[o + 17] + cf1 * jac[o + 20]);
            a1 += w * (cf0 * jac[o + 18] + cf1 * jac[o + 21]);
            a2 += w * (cf0 * jac[o + 19] + cf1 * jac[o + 22]);
        }
        int b = p * 9;
        outp[p] = a0 * (vinv[b] * a0 + vinv[b + 1] * a1 + vinv[b + 2] * a2)
                + a1 * (vinv[b + 3] * a0 + vinv[b + 4] * a1 + vinv[b + 5] * a2)
                + a2 * (vinv[b + 6] * a0 + vinv[b + 7] * a1 + vinv[b + 8] * a2);
    }

    // One thread: the shared focal's rhs, damping and preconditioner - the EXACT diagonal S_ff, as the managed
    // BlockJacobiCg's tailInv. fsPartial = 1,024 partial sums of FocalSchurKernel. The first version summed only the
    // per-camera Schur blocks (no cross-camera terms): with CG capped at 16 iterations that left the GPU solver 0.23% of
    // spread from the managed one on the same input (2026-09-28); converged, 0.004%.
    static void PrepareFocalKernel(Index1D t, ArrayView1D<Real, Stride1D.Dense> u, ArrayView1D<Real, Stride1D.Dense> g, ArrayView1D<Real, Stride1D.Dense> wt, ArrayView1D<Real, Stride1D.Dense> fsPartial, ArrayView1D<Real, Stride1D.Dense> sysv, ArrayView1D<Real, Stride1D.Dense> pre, ArrayView1D<Real, Stride1D.Dense> cgs, Real lambda, int nc, int n)
    {
        Real uff = 0, sff = 0, rf = 0;
        for (int c = 0; c < nc; c++)
        {
            uff += u[c * 49 + 48];
            rf += -g[c * 7 + 6] + wt[c * 7 + 6];
        }
        for (int k = 0; k < 1024; k++) sff += fsPartial[k];
        Real df = lambda * uff + (Real)1e-9;
        sysv[nc * 6] = rf;
        sysv[n + nc * 6] = df;
        Real s = uff - sff + df;
        pre[nc * 36] = s > (Real)0 ? (Real)1 / s : (Real)0;
        cgs[4] = uff + df;
        cgs[6] = s; // traced: the raw focal diagonal of the damped Schur complement
    }

    // Per camera: the pose rows of ap = S p (U p + damping - the device's Σ W V*⁻¹ Wᵀ p in wz), and this camera's part
    // of the focal row (summed in CgStep).
    static void ApPoseKernel(Index1D ci, ArrayView1D<Real, Stride1D.Dense> u, ArrayView1D<Real, Stride1D.Dense> cgv, ArrayView1D<Real, Stride1D.Dense> wz, ArrayView1D<Real, Stride1D.Dense> sysv, ArrayView1D<Real, Stride1D.Dense> fpart, int nc, int n, int hasF)
    {
        int c = ci;
        int P = 3 * n, AP = 4 * n;
        Real pf = hasF != 0 ? cgv[P + nc * 6] : (Real)0;
        int ub = c * 49;
        for (int a = 0; a < 6; a++)
        {
            Real acc = u[ub + a * 7 + 6] * pf;
            for (int b = 0; b < 6; b++) acc += u[ub + a * 7 + b] * cgv[P + c * 6 + b];
            cgv[AP + c * 6 + a] = acc - wz[c * 7 + a] + sysv[n + c * 6 + a] * cgv[P + c * 6 + a];
        }
        Real fp = -wz[c * 7 + 6];
        for (int b = 0; b < 6; b++) fp += u[ub + 42 + b] * cgv[P + c * 6 + b];
        fpart[c] = fp;
    }

    // Sum of one value per thread of the (single) workgroup. The loop's trip count is a constant, so every barrier is in
    // uniform control flow (WGSL requires it); lanes past the group size never take part.
    static Real GroupSum(ArrayView<Real> sh, Real v)
    {
        int t = Group.IdxX, dim = Group.DimX;
        sh[t] = v;
        Group.Barrier();
        for (int s = GroupSize / 2; s > 0; s >>= 1)
        {
            // Only the steps this group size needs. dim is uniform (WGSL: the module const workgroup_size), so the
            // barrier stays in uniform control flow; a group of 4 ran 8 barrier steps instead of 2 - on the ILGPU CPU
            // accelerator every one is a real thread rendezvous.
            if (s < dim)
            {
                if (t < s) sh[t] += sh[t + s];
                Group.Barrier();
            }
        }
        Real total = sh[0];
        Group.Barrier();
        return total;
    }

    // x = 0, r = rhs, z = M⁻¹ r, p = z, rz = r·z, |b|² (one workgroup; thread t owns cameras t, t + dim, ...; thread 0
    // also owns the focal).
    static void CgInitKernel(ArrayView1D<Real, Stride1D.Dense> sysv, ArrayView1D<Real, Stride1D.Dense> pre, ArrayView1D<Real, Stride1D.Dense> cgv, ArrayView1D<Real, Stride1D.Dense> cgs, int nc, int n, int hasF)
    {
        var sh = SharedMemory.Allocate<Real>(GroupSize);
        int t = Group.IdxX, dim = Group.DimX;
        int R = n, Z = 2 * n, P = 3 * n, AP = 4 * n;
        Real rz = 0, b2 = 0;
        for (int c = t; c < nc; c += dim)
            for (int a = 0; a < 6; a++)
            {
                int k = c * 6 + a;
                Real rhs = sysv[k];
                Real z = 0;
                for (int b = 0; b < 6; b++) z += pre[c * 36 + a * 6 + b] * sysv[c * 6 + b];
                cgv[k] = (Real)0; cgv[R + k] = rhs; cgv[Z + k] = z; cgv[P + k] = z; cgv[AP + k] = (Real)0;
                b2 += rhs * rhs; rz += rhs * z;
            }
        if (t == 0 && hasF != 0)
        {
            int k = nc * 6;
            Real rhs = sysv[k], z = pre[nc * 36] * rhs;
            cgv[k] = (Real)0; cgv[R + k] = rhs; cgv[Z + k] = z; cgv[P + k] = z; cgv[AP + k] = (Real)0;
            b2 += rhs * rhs; rz += rhs * z;
        }
        Real rzT = GroupSum(sh, rz);
        Real b2T = GroupSum(sh, b2);
        if (t == 0)
        {
            cgs[0] = rzT;
            cgs[1] = b2T > (Real)1e-30 ? b2T : (Real)1e-30;
            cgs[2] = (Real)0;
            cgs[3] = (Real)0;
        }
    }

    // One PCG iteration after ap (pose rows) and the focal parts are in: the managed BlockJacobiCg's loop body, including
    // its two exits (p·Ap <= 0 before the update, |r|²/|b|² < tol after it). Once done, a step changes nothing.
    static void CgStepKernel(ArrayView1D<Real, Stride1D.Dense> cgv, ArrayView1D<Real, Stride1D.Dense> pre, ArrayView1D<Real, Stride1D.Dense> fpart, ArrayView1D<Real, Stride1D.Dense> cgs, Real tol, int nc, int n, int hasF)
    {
        var sh = SharedMemory.Allocate<Real>(GroupSize);
        int t = Group.IdxX, dim = Group.DimX;
        int R = n, Z = 2 * n, P = 3 * n, AP = 4 * n, f = nc * 6;
        Real rz = cgs[0], b2 = cgs[1];
        bool wasDone = cgs[2] != (Real)0;

        Real fs = 0;
        for (int c = t; c < nc; c += dim) fs += fpart[c];
        Real fsum = GroupSum(sh, fs);
        Real apf = hasF != 0 ? fsum + cgs[4] * cgv[P + f] : (Real)0;

        Real pap = 0;
        for (int c = t; c < nc; c += dim)
            for (int a = 0; a < 6; a++) pap += cgv[P + c * 6 + a] * cgv[AP + c * 6 + a];
        if (t == 0 && hasF != 0) pap += cgv[P + f] * apf;
        Real papT = GroupSum(sh, pap);
        bool active = !wasDone && papT > (Real)0;
        Real alpha = active ? rz / papT : (Real)0;

        Real rr = 0;
        for (int c = t; c < nc; c += dim)
            for (int a = 0; a < 6; a++)
            {
                int k = c * 6 + a;
                cgv[k] += alpha * cgv[P + k];
                Real r = cgv[R + k] - alpha * cgv[AP + k];
                cgv[R + k] = r;
                rr += r * r;
            }
        if (t == 0 && hasF != 0)
        {
            cgv[AP + f] = apf;
            cgv[f] += alpha * cgv[P + f];
            Real r = cgv[R + f] - alpha * apf;
            cgv[R + f] = r;
            rr += r * r;
        }
        Real rrT = GroupSum(sh, rr);
        bool converged = active && rrT / b2 < tol;
        bool go = active && !converged;

        Real rzn = 0;
        for (int c = t; c < nc; c += dim)
            for (int a = 0; a < 6; a++)
            {
                Real z = 0;
                for (int b = 0; b < 6; b++) z += pre[c * 36 + a * 6 + b] * cgv[R + c * 6 + b];
                cgv[Z + c * 6 + a] = z;
                rzn += cgv[R + c * 6 + a] * z;
            }
        if (t == 0 && hasF != 0)
        {
            Real z = pre[nc * 36] * cgv[R + f];
            cgv[Z + f] = z;
            rzn += cgv[R + f] * z;
        }
        Real rznT = GroupSum(sh, rzn);
        Real beta = go ? rznT / rz : (Real)0;
        if (go)
        {
            for (int c = t; c < nc; c += dim)
                for (int a = 0; a < 6; a++)
                {
                    int k = c * 6 + a;
                    cgv[P + k] = cgv[Z + k] + beta * cgv[P + k];
                }
            if (t == 0 && hasF != 0) cgv[P + f] = cgv[Z + f] + beta * cgv[P + f];
        }
        if (t == 0)
        {
            if (active) cgs[5] = rrT / b2;
            cgs[0] = go ? rznT : rz;
            cgs[2] = go ? (Real)0 : (Real)1;
            cgs[3] = cgs[3] + (active ? (Real)1 : (Real)0);
        }
    }

    // ── state on the device ─────────────────────────────────────────────────────────────────

    sealed class Kernels
    {
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real> Jac = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>> PointAsm = null!;
        public Action<KernelConfig, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>> CamAsm = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real> PointSolve = null!;
        public Action<KernelConfig, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>> CamGather = null!;
        public Action<KernelConfig, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>> CamDiag = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int> PointScatter = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real> Cost = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int> PartialSum = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>> ResetScalars = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int> Expand = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int> PrepareSystem = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int, int> PrepareFocal = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>> FocalSchur = null!;
        public Action<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int, int> ApPose = null!;
        public Action<KernelConfig, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int, int> CgInit = null!;
        public Action<KernelConfig, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int, int, int> CgStep = null!;
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    private static Kernels For(Accelerator a) => s_kernels.GetValue(a, acc => new Kernels
    {
        Jac = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real>(JacobianKernel),
        PointAsm = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>>(PointAssembleKernel),
        CamAsm = acc.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>>(CameraAssembleKernel),
        PointSolve = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real>(PointSolveKernel),
        CamGather = acc.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>>(CameraGatherKernel),
        CamDiag = acc.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>>(CameraDiagKernel),
        PointScatter = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int>(PointScatterKernel),
        Cost = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real>(CostKernel),
        PartialSum = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int>(PartialSumKernel),
        ResetScalars = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>>(ResetScalarsKernel),
        Expand = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int>(ExpandKernel),
        PrepareSystem = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int>(PrepareSystemKernel),
        PrepareFocal = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int, int>(PrepareFocalKernel),
        FocalSchur = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>>(FocalSchurKernel),
        ApPose = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int, int>(ApPoseKernel),
        CgInit = acc.LoadStreamKernel<ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, int, int, int>(CgInitKernel),
        CgStep = acc.LoadStreamKernel<ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, ArrayView1D<Real, Stride1D.Dense>, Real, int, int, int>(CgStepKernel),
    });

    // Round-lifetime device buffers (the kept observation set changes between rounds).
    MemoryBuffer1D<int, Stride1D.Dense>? _obsIdx, _pStart, _pObs, _cStart, _cObs, _camFlags;
    MemoryBuffer1D<Real, Stride1D.Dense>? _obsUv, _jac, _cams, _pts, _ptsCand, _v, _gp, _vinv, _tp, _camVec, _camOut, _camBlk,
        _ptVec, _uCam, _gCam, _cost, _partial, _cgv, _sysv, _pre, _fpart, _cgs, _fsPoint, _fsPartial;
    int _m; // kept observations this round
    int _group; // the CG workgroup: the largest power of two <= GroupSize the accelerator allows
    int _camGroup; // per-camera workgroup: the largest power of two <= CamGroupSize the accelerator allows
    const int CamGroupSize = 64;
    KernelConfig CamConfig => new KernelConfig(_nc, _camGroup);

    void FreeRound()
    {
        _cgPlan?.Dispose();
        _cgPlan = null;
        foreach (var b in new MemoryBuffer?[] { _obsIdx, _pStart, _pObs, _cStart, _cObs, _obsUv, _jac, _v, _gp, _vinv, _tp, _ptVec, _cost, _fsPoint })
            b?.Dispose();
        _obsIdx = _pStart = _pObs = _cStart = _cObs = null;
        _obsUv = _jac = _v = _gp = _vinv = _tp = _ptVec = _cost = _fsPoint = null;
    }

    /// <summary>Frees every device buffer. The solution itself (cameras, points, statistics) lives on the host.</summary>
    public void Dispose()
    {
        FreeRound();
        foreach (var b in new MemoryBuffer?[] { _camFlags, _cams, _pts, _ptsCand, _camVec, _camOut, _camBlk, _uCam, _gCam, _partial,
            _cgv, _sysv, _pre, _fpart, _cgs, _fsPartial })
            b?.Dispose();
        _camFlags = null;
        _cams = _pts = _ptsCand = _camVec = _camOut = _camBlk = _uCam = _gCam = _partial = null;
        _cgv = _sysv = _pre = _fpart = _cgs = _fsPartial = null;
    }

    /// <summary>Upload the kept observations and their per-point / per-camera CSR lists.</summary>
    void BuildRound()
    {
        FreeRound();
        var kept = new List<int>();
        for (int i = 0; i < _obs.Count; i++) if (_keep[i]) kept.Add(i);
        _m = kept.Count;
        var idx = new int[Math.Max(1, _m * 2)];
        var uv = new Real[Math.Max(1, _m * 2)];
        var pCount = new int[_np + 1];
        var cCount = new int[_nc + 1];
        for (int t = 0; t < _m; t++)
        {
            var o = _obs[kept[t]];
            idx[t * 2] = o.Camera; idx[t * 2 + 1] = o.Point;
            uv[t * 2] = o.U; uv[t * 2 + 1] = o.V;
            pCount[o.Point + 1]++; cCount[o.Camera + 1]++;
        }
        for (int p = 0; p < _np; p++) pCount[p + 1] += pCount[p];
        for (int c = 0; c < _nc; c++) cCount[c + 1] += cCount[c];
        var pObs = new int[Math.Max(1, _m)];
        var cObs = new int[Math.Max(1, _m)];
        var pFill = (int[])pCount.Clone();
        var cFill = (int[])cCount.Clone();
        // Observation order within a point / camera = the managed solver's order (ascending observation index).
        for (int t = 0; t < _m; t++)
        {
            var o = _obs[kept[t]];
            pObs[pFill[o.Point]++] = t;
            cObs[cFill[o.Camera]++] = t;
        }
        _obsIdx = _acc.Allocate1D(idx);
        _obsUv = _acc.Allocate1D(uv);
        _pStart = _acc.Allocate1D(pCount);
        _pObs = _acc.Allocate1D(pObs);
        _cStart = _acc.Allocate1D(cCount);
        _cObs = _acc.Allocate1D(cObs);
        _jac = _acc.Allocate1D<Real>(Math.Max(1, (long)_m * J));
        _v = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 9));
        _gp = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 3));
        _vinv = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 9));
        _tp = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 3));
        _ptVec = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 3));
        _cost = _acc.Allocate1D<Real>(Math.Max(1, _m));
        _fsPoint = _acc.Allocate1D<Real>(Math.Max(1, (long)_np));
        if (_cams == null)
        {
            var flags = new int[_nc];
            for (int c = 0; c < _nc; c++) flags[c] = (_fixed[c] ? 1 : 0) | (_sharedFocal ? 2 : 0);
            _camFlags = _acc.Allocate1D(flags);
            int n = ParamCount;
            _cgv = _acc.Allocate1D<Real>(5 * n);
            _sysv = _acc.Allocate1D<Real>(2 * n);
            _pre = _acc.Allocate1D<Real>(_nc * 36 + 1);
            _fpart = _acc.Allocate1D<Real>(_nc);
            _cgs = _acc.Allocate1D<Real>(9);
            _fsPartial = _acc.Allocate1D<Real>(1024);
            // The ILGPU CPU accelerator runs a group's threads as OS threads that meet at every barrier: 64 per camera
            // (x 251 cameras, ~40 barriers per call) burned 11,317 CPU-seconds in one equivalence test (2026-09-28). Small
            // groups there still exercise the multi-thread reductions; the GPU sizes are for GPUs.
            bool cpu = _acc.AcceleratorType == AcceleratorType.CPU;
            int grp = Math.Min(cpu ? 16 : GroupSize, _acc.MaxNumThreadsPerGroup);
            while ((grp & (grp - 1)) != 0) grp &= grp - 1;
            _group = grp;
            int cg = Math.Min(cpu ? 4 : CamGroupSize, _acc.MaxNumThreadsPerGroup);
            while ((cg & (cg - 1)) != 0) cg &= cg - 1;
            _camGroup = cg;
            _cams = _acc.Allocate1D<Real>(_nc * 16);
            _pts = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 3));
            _ptsCand = _acc.Allocate1D<Real>(Math.Max(1, (long)_np * 3));
            _camVec = _acc.Allocate1D<Real>(_nc * Cols);
            _camOut = _acc.Allocate1D<Real>(_nc * Cols);
            _camBlk = _acc.Allocate1D<Real>(_nc * 49);
            _uCam = _acc.Allocate1D<Real>(_nc * 49);
            _gCam = _acc.Allocate1D<Real>(_nc * Cols);
            _partial = _acc.Allocate1D<Real>(1024);
        }
        // Points go up once per round (they stay on the device between LM steps).
        var pts = new Real[Math.Max(1, _np * 3)];
        for (int i = 0; i < _np * 3; i++) pts[i] = (Real)_x[i];
        _pts!.View.BaseView.CopyFromCPU(pts);
    }

    Real[] CamsFloat(double[] r, double[] c, double f)
    {
        var a = new Real[_nc * 16];
        for (int i = 0; i < _nc; i++)
        {
            for (int k = 0; k < 9; k++) a[i * 16 + k] = (Real)r[i * 9 + k];
            for (int k = 0; k < 3; k++) a[i * 16 + 9 + k] = (Real)c[i * 3 + k];
            a[i * 16 + 12] = (Real)(_sharedFocal ? f : _k[i * 4]);
            a[i * 16 + 13] = (Real)(_sharedFocal ? f : _k[i * 4 + 1]);
            a[i * 16 + 14] = (Real)_k[i * 4 + 2];
            a[i * 16 + 15] = (Real)_k[i * 4 + 3];
        }
        return a;
    }

    async Task<Real[]> ReadAsync(MemoryBuffer1D<Real, Stride1D.Dense> buf)
    {
        _readbacks++;
        // WebGPU: the readback flushes pending kernels itself and its copy is queue-ordered after them, so a
        // SynchronizeAsync first was a SECOND GPU-to-host wait per read (b38: ~7 ms round trip per CG batch).
        if (_acc is not WebGPUAccelerator) await _acc.SynchronizeAsync();
        return await buf.CopyToHostAsync<Real>();
    }

    async Task<double> DeviceCostAsync(double[] r, double[] c, MemoryBuffer1D<Real, Stride1D.Dense> pts, double f)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var k = For(_acc);
        _cams!.View.BaseView.CopyFromCPU(CamsFloat(r, c, f));
        double cost = 0;
        if (_m > 0)
        {
            k.Cost(_m, _obsIdx!.View, _obsUv!.View, _cams.View, pts.View, _cost!.View, (Real)_opts.HuberPixels);
            k.PartialSum(1024, _cost.View, _partial!.View, _m);
            foreach (var s in await ReadAsync(_partial)) cost += s;
        }
        _tCost += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        return cost;
    }

    // ── solve ───────────────────────────────────────────────────────────────────────────────

    public async Task<BundleAdjuster.Result> SolveAsync(BundleAdjuster.Options? options = null)
    {
        _opts = options ?? new BundleAdjuster.Options();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double initialRms = RmsPixels();
        int iters;
        try
        {
            iters = await RunRoundsAsync();
        }
        finally
        {
            // Device memory is held only for the solve: the host keeps the solution.
            Dispose();
        }
        return new BundleAdjuster.Result(initialRms, RmsPixels(), iters, _obs.Count, _keep.Count(k => k), _np, sw.Elapsed.TotalSeconds);
    }

    async Task<int> RunRoundsAsync()
    {
        int iters = 0;
        for (int round = 0; round < _opts.Rounds; round++)
        {
            BuildRound();
            int roundIters = await RunLevenbergMarquardtAsync();
            iters += roundIters;
            // Points come back once per round, for the outlier test and the statistics.
            var pts = await ReadAsync(_pts!);
            for (int i = 0; i < _np * 3; i++) _x[i] = pts[i];
            _opts.RoundLog?.Invoke(round, roundIters, RmsPixels(), _keep.Count(k => k));
            double limit = _opts.OutlierHubers * _opts.HuberPixels;
            int dropped = 0;
            for (int i = 0; i < _obs.Count; i++)
            {
                if (!_keep[i]) continue;
                if (!Residual(_obs[i], out var ru, out var rv) || ru * ru + rv * rv > limit * limit) { _keep[i] = false; dropped++; }
            }
            if (dropped == 0) break;
        }
        return iters;
    }

    async Task<int> RunLevenbergMarquardtAsync()
    {
        var k = For(_acc);
        int n = ParamCount;
        double lambda = 1e-3;
        double cost = await DeviceCostAsync(_r, _c, _pts!, _f);
        int it = 0;
        var nr = new double[_r.Length];
        var ncc = new double[_c.Length];
        for (; it < _opts.MaxIterations; it++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _cams!.View.BaseView.CopyFromCPU(CamsFloat(_r, _c, _f));
            if (_m > 0)
            {
                k.Jac(_m, _obsIdx!.View, _obsUv!.View, _cams.View, _camFlags!.View, _pts!.View, _jac!.View, (Real)_opts.HuberPixels);
                k.PointAsm(_np, _pStart!.View, _pObs!.View, _jac.View, _v!.View, _gp!.View);
                k.CamAsm(CamConfig, _cStart!.View, _cObs!.View, _jac.View, _uCam!.View, _gCam!.View);
            }
            _tDevice += System.Diagnostics.Stopwatch.GetTimestamp() - t0;

            bool improved = false;
            for (int attempt = 0; attempt < 10; attempt++)
            {
                _attempts++;
                var dCam = await SolveDampedAsync(lambda, n);
                if (dCam == null)
                {
                    if (TraceAttempts && _traced++ < MaxTracedAttempts)
                        Console.WriteLine($"[GpuBA] it {it} attempt {attempt} lambda {lambda:E1}: REJECTED - {_failReason}; {_cgTrace}");
                    lambda *= 10;
                    continue;
                }

                // Candidate points on the device (back-substitution), candidate cameras on the host.
                long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                // The camera step is still on the device (x in cgv): no upload.
                k.Expand(_nc * Cols, _cgv!.View, _camFlags!.View, _camVec!.View, 0, _nc);
                if (_np > 0)
                    k.PointScatter(_np, _pStart!.View, _pObs!.View, _obsIdx!.View, _jac!.View, _camVec.View, _vinv!.View, _gp!.View,
                        _pts!.View, _ptsCand!.View, 1);
                double nf = ApplyCameraStep(dCam, nr, ncc);
                _tDevice += System.Diagnostics.Stopwatch.GetTimestamp() - t1;
                double newCost = await DeviceCostAsync(nr, ncc, _ptsCand!, nf);
                if (TraceAttempts && !(newCost < cost) && _traced++ < MaxTracedAttempts)
                {
                    Console.WriteLine($"[GpuBA] it {it} attempt {attempt} lambda {lambda:E1}: cost {cost:R} -> {newCost:R}; " +
                        $"max|dCam| {dCam.Max(Math.Abs):E2}; {_cgTrace}");
                    await TraceDeviceStateAsync();
                }
                if (newCost < cost)
                {
                    Array.Copy(nr, _r, _r.Length); Array.Copy(ncc, _c, _c.Length);
                    _f = nf;
                    (_pts, _ptsCand) = (_ptsCand, _pts);
                    double rel = (cost - newCost) / Math.Max(cost, 1e-30);
                    _opts.IterationLog?.Invoke(it, newCost, rel, dCam.Max(Math.Abs), double.NaN);
                    cost = newCost;
                    lambda = Math.Max(lambda / 3, 1e-9);
                    improved = true;
                    if (rel < (RelativeStopOverride ?? RelativeStop)) return it + 1;
                    break;
                }
                lambda *= 10;
            }
            if (!improved) break;
        }
        return it;
    }

    async Task TraceDeviceStateAsync()
    {
        static (int NonFinite, double MaxAbs) Scan(Real[] a, int n)
        {
            int bad = 0; double max = 0;
            for (int i = 0; i < n; i++)
            {
                double v = a[i];
                if (!double.IsFinite(v)) bad++;
                else if (Math.Abs(v) > max) max = Math.Abs(v);
            }
            return (bad, max);
        }
        var v = Scan(await ReadAsync(_v!), _np * 9);
        var vi = Scan(await ReadAsync(_vinv!), _np * 9);
        var pc = Scan(await ReadAsync(_ptsCand!), _np * 3);
        var co = Scan(await ReadAsync(_cost!), _m);
        Console.WriteLine($"[GpuBA]   device: V non-finite {v.NonFinite} max {v.MaxAbs:E2}; V*⁻¹ non-finite {vi.NonFinite} max {vi.MaxAbs:E2}; " +
            $"candidate points non-finite {pc.NonFinite} max {pc.MaxAbs:E2}; obs cost non-finite {co.NonFinite} max {co.MaxAbs:E2}");
    }

    double ApplyCameraStep(double[] dCam, double[] nr, double[] nc)
    {
        Array.Copy(_r, nr, _r.Length);
        Array.Copy(_c, nc, _c.Length);
        Span<double> e = stackalloc double[9];
        for (int ci = 0; ci < _nc; ci++)
        {
            if (_fixed[ci]) continue;
            BundleAdjuster.Rodrigues(dCam[ci * 6], dCam[ci * 6 + 1], dCam[ci * 6 + 2], e);
            for (int a = 0; a < 3; a++)
                for (int bb = 0; bb < 3; bb++)
                    nr[ci * 9 + a * 3 + bb] = e[a * 3] * _r[ci * 9 + bb] + e[a * 3 + 1] * _r[ci * 9 + 3 + bb] + e[a * 3 + 2] * _r[ci * 9 + 6 + bb];
            for (int a = 0; a < 3; a++) nc[ci * 3 + a] = _c[ci * 3 + a] + dCam[ci * 6 + 3 + a];
        }
        return _sharedFocal ? Math.Max(1e-3, _f + dCam[_nc * 6]) : _f;
    }

    /// <summary>
    /// The damped reduced camera system, solved by block-Jacobi PCG entirely on the device (n = 6 per camera + focal):
    /// the system, the preconditioner and every CG step are kernels; the host reads the CG scalars every
    /// <see cref="CgCheckEvery"/> iterations and the camera step once. Null = a singular point block, a non-SPD camera
    /// block or a non-finite step (the caller raises lambda).
    /// </summary>
    async Task<double[]?> SolveDampedAsync(double lambda, int n)
    {
        var k = For(_acc);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        int hasF = _sharedFocal ? 1 : 0;
        Real lam = (Real)lambda;
        k.ResetScalars(9, _cgs!.View);
        if (_np > 0) k.PointSolve(_np, _v!.View, _gp!.View, _vinv!.View, _tp!.View, _cgs.View, lam);
        // wt = Σ W V*⁻¹ gP (for the rhs); the per-camera blocks of Σ W V*⁻¹ Wᵀ (for the preconditioner).
        k.CamGather(CamConfig, _cStart!.View, _cObs!.View, _obsIdx!.View, _jac!.View, _tp!.View, _camOut!.View);
        k.CamDiag(CamConfig, _cStart.View, _cObs.View, _obsIdx.View, _jac.View, _vinv!.View, _camBlk!.View);
        k.PrepareSystem(_nc, _uCam!.View, _gCam!.View, _camOut.View, _camBlk.View, _camFlags!.View, _sysv!.View, _pre!.View, _cgs.View, lam, n);
        if (hasF != 0)
        {
            k.FocalSchur(_np, _pStart!.View, _pObs!.View, _jac.View, _vinv.View, _fsPoint!.View);
            k.PartialSum(1024, _fsPoint.View, _fsPartial!.View, _np);
            k.PrepareFocal(1, _uCam.View, _gCam.View, _camOut.View, _fsPartial.View, _sysv.View, _pre.View, _cgs.View, lam, _nc, n);
        }
        var cfg = new KernelConfig(1, _group);
        k.CgInit(cfg, _sysv.View, _pre.View, _cgv!.View, _cgs.View, _nc, n, hasF);

        Real[] sc = new Real[9];
        bool failed = false;
        for (int launched = 0; launched < _opts.MaxCgIterations;)
        {
            int batch = Math.Min(CgCheckEvery, _opts.MaxCgIterations - launched);
            if (_cgPlan != null && batch == CgCheckEvery)
            {
                _replays++;
                if (TraceAttempts && !_cgPlanTimed)
                {
                    _cgPlanTimed = true;
                    Console.WriteLine($"[GpuBA] CG batch GPU time ({CgCheckEvery} iterations, {_cgPlan.DispatchCount} dispatches): {await _cgPlan.ReplayTimedAsync()}");
                }
                else await _cgPlan.ReplayAsync();
                launched += batch;
                sc = await ReadAsync(_cgs);
                if (sc[7] != (Real)0) { failed = true; break; }
                if (sc[2] != (Real)0) break;
                continue;
            }
            var wg = UseDispatchCapture && batch == CgCheckEvery ? _acc as WebGPUAccelerator : null;
            bool cacheWas = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching;
            if (wg != null)
            {
                SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = false;
                wg.BeginDispatchCapture();
            }
            try
            {
            for (int b = 0; b < batch; b++)
            {
                // ap = S p: the Σ W V*⁻¹ Wᵀ p part per point then per camera, the rest per camera, then the CG step.
                k.Expand(_nc * Cols, _cgv.View, _camFlags.View, _camVec!.View, 3 * n, _nc);
                k.PointScatter(_np, _pStart!.View, _pObs!.View, _obsIdx.View, _jac.View, _camVec.View, _vinv.View, _gp!.View,
                    _pts!.View, _ptVec!.View, 0);
                k.CamGather(CamConfig, _cStart.View, _cObs.View, _obsIdx.View, _jac.View, _ptVec.View, _camOut.View);
                k.ApPose(_nc, _uCam.View, _cgv.View, _camOut.View, _sysv.View, _fpart!.View, _nc, n, hasF);
                k.CgStep(cfg, _cgv.View, _pre.View, _fpart.View, _cgs.View, (Real)_opts.CgTolerance, _nc, n, hasF);
            }
            }
            finally
            {
                if (wg != null)
                {
                    _cgPlan = wg.EndDispatchCapture();
                    SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = cacheWas;
                }
            }
            launched += batch;
            sc = await ReadAsync(_cgs);
            if (sc[7] != (Real)0) { failed = true; break; }
            if (sc[2] != (Real)0) break;
        }
        _cgIterations += (int)sc[3];
        if (!failed && sc[2] == (Real)0) _cgCapHits++;
        _tCg += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        if (_sharedFocal)
        {
            double sff = sc[6], uff = sc[4];
            if (!(sff > 0)) _focalPreNonPositive++;
            if (uff > 0) _focalPreMinRatio = Math.Min(_focalPreMinRatio, sff / uff);
        }
        _cgTrace = $"CG {(int)sc[3]} iters, done {sc[2] != 0}, |r|²/|b|² {(double)sc[5]:E2}, |b|² {(double)sc[1]:E3}" +
            (sc[8] != 0 ? ", a point singular to working precision" : "");
        if (failed) { _failReason = "a point or camera block is not invertible"; return null; }

        var v = await ReadAsync(_cgv);
        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = v[i];
            if (!double.IsFinite(x[i])) { _failReason = $"non-finite step at parameter {i}"; return null; }
        }
        for (int c = 0; c < _nc; c++) if (_fixed[c]) for (int a = 0; a < 6; a++) x[c * 6 + a] = 0;
        return x;
    }

    static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }
}
