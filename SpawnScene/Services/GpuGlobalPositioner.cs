using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using IV = ILGPU.Runtime.ArrayView1D<int, ILGPU.Stride1D.Dense>;
using RV = ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>;

namespace SpawnScene.Services;

/// <summary>
/// <see cref="GlobalSfmInit.GlobalPositioningRobust"/> on the GPU: the same model (per-observation scales, Huber, centroid
/// gauge, variable projection of the scales), the same Levenberg-Marquardt loop and damping, the same re-admitting inlier
/// rounds, the same deterministic start (<see cref="GlobalSfmInit.StartCoord"/>), the same histogram median and the same
/// exact solve of the reduced camera system (dense Cholesky) - but every array lives on the device. Per LM attempt the
/// scales and points are eliminated per observation / per point, the camera system
/// <c>S = U' + gauge - Σ_p B V_p⁻¹ Bᵀ</c> is BUILT (one thread per camera writes its own three rows) and factored in one
/// workgroup. The host reads back only scalars (the cost and a failure flag per attempt, the median bin and the inlier
/// changes per round) and, once, the centres.
/// </summary>
/// <remarks>
/// Why: the managed solver runs on the Mono WASM interpreter in the browser; natively it takes 5-12 s on Truck's 126 views
/// (2026-09-29, after variable projection), far more under the interpreter.
/// Why a direct solve and not GpuBundleAdjuster's matrix-free PCG: the camera system is small (3 per camera: 378 on Truck,
/// 753 on TruckFull) and badly conditioned for block Jacobi - consecutive views on short tracks form a chain. MEASURED
/// 2026-09-29 (ILGPU CPU accelerator, the first version of this class was PCG): ~57 CG iterations per attempt on long
/// tracks, ~340 on TruckFull's real track lengths (13,503 in 40 LM attempts) - on a GPU at ~1 ms per replayed CG iteration,
/// ~0.3 s per attempt. The Cholesky is ~7 barriers per unknown in one workgroup and gives the managed solver's exact step.
/// Double precision: on WebGPU a Dekker pair of f32 (float's exponent range, NaN past ~8.3e34), so no constant here is
/// below 1e-30 and the tan-half-angle is capped at 1e30.
/// </remarks>
public sealed class GpuGlobalPositioner : IDisposable
{
    const int GroupSize = 256;      // shared-memory size of the single-workgroup kernels
    const int CamGroupSize = 64;    // per-camera workgroup
    const int K = 8;                // per observation: q = w d², û (3), gdn = -w d (û·r), gr = w d r (3)
    const int PS = 15;              // per point: V*⁻¹ (9), t = V*⁻¹ g' (3), g' (3)
    const int CS = 9;               // per camera: U' (00 01 02 11 12 22), rhs (3)
    // Scalars: failure flag, gauge weight, cost, median bin edge, camera sum of C, 1 + λ of the current attempt.
    const int S_FAILED = 7, S_MU = 8, S_COST = 9, S_MEDIAN = 11, S_CSUM = 12, S_F = 15, NScalars = 16;

    /// <summary>Per-attempt trace (round, iteration, lambda, cost -> candidate): the solver's only view of a device run.</summary>
    public static Action<string>? Trace { get; set; }

    readonly Accelerator _acc;
    readonly IReadOnlyList<CameraParams> _cams;
    readonly bool[] _connected;
    readonly int _n, _np, _no, _nf, _dim;
    readonly int[] _camOrig;          // compact camera -> original index
    readonly int _maxIterations, _seed;
    readonly double _huber;
    int _group, _camGroup;
    long _tSolve, _tTotal;
    int _attempts, _readbacks;

    // Device state. Fixed for the solve: observation topology and bearings. Swapped on an accepted step: C/Cn, X/Xn, d/dn.
    MemoryBuffer1D<int, Stride1D.Dense>? _oc, _op, _pStart, _pObs, _cStart, _cObs, _camOrigBuf, _off, _live, _hist, _counts;
    // _sys = the camera system S (3 nf x 3 nf, factored in place), _rhs = its right-hand side, _step = the solution dc.
    MemoryBuffer1D<double, Stride1D.Dense>? _v, _c, _cn, _x, _xn, _d, _dn, _ob, _rawP, _rawC, _ps, _cs, _sys, _rhs, _step,
        _cost, _partial, _sc, _ang;

    /// <summary>Same inputs as <see cref="GlobalSfmInit.GlobalPositioningRobust"/>.</summary>
    public GpuGlobalPositioner(Accelerator accelerator, IReadOnlyList<CameraParams> cams, double[][] rot, bool[] connected,
        IReadOnlyList<BundleAdjuster.Observation> obs, int pointCount, double focal, int maxIterations = 50,
        double huber = 0.003, int seed = 1)
    {
        _acc = accelerator;
        _cams = cams;
        _connected = connected;
        _n = cams.Count;
        _np = pointCount;
        _maxIterations = maxIterations;
        _huber = huber;
        _seed = seed;
        var col = new int[_n];
        var orig = new List<int>();
        for (int i = 0; i < _n; i++) { col[i] = connected[i] ? orig.Count : -1; if (connected[i]) orig.Add(i); }
        _camOrig = orig.ToArray();
        _nf = _camOrig.Length;
        _dim = _nf * 3;

        var (oc, op, v) = GlobalSfmInit.PositioningObservations(cams, rot, connected, obs, pointCount, focal);
        _no = oc.Length;
        var occ = new int[Math.Max(1, _no)];
        for (int t = 0; t < _no; t++) occ[t] = col[oc[t]];
        // CSR lists by point and by camera, in ascending observation order (the managed solver's order).
        var pStart = new int[_np + 1]; var cStart = new int[_nf + 1];
        for (int t = 0; t < _no; t++) { pStart[op[t] + 1]++; cStart[occ[t] + 1]++; }
        for (int p = 0; p < _np; p++) pStart[p + 1] += pStart[p];
        for (int c = 0; c < _nf; c++) cStart[c + 1] += cStart[c];
        var pObs = new int[Math.Max(1, _no)]; var cObs = new int[Math.Max(1, _no)];
        var pFill = (int[])pStart.Clone(); var cFill = (int[])cStart.Clone();
        for (int t = 0; t < _no; t++) { pObs[pFill[op[t]]++] = t; cObs[cFill[occ[t]]++] = t; }
        var live = new int[Math.Max(1, _np)];
        for (int t = 0; t < _no; t++) live[op[t]]++;

        // CPU transfer: the observation topology and bearings are this solver's INPUT (host tracks); they go up once.
        _oc = _acc.Allocate1D(occ);
        _op = _acc.Allocate1D(_no > 0 ? op : new int[1]);
        _v = _acc.Allocate1D(_no > 0 ? v : new double[3]);
        _pStart = _acc.Allocate1D(pStart);
        _pObs = _acc.Allocate1D(pObs);
        _cStart = _acc.Allocate1D(cStart);
        _cObs = _acc.Allocate1D(cObs);
        _camOrigBuf = _acc.Allocate1D(_camOrig.Length > 0 ? _camOrig : new int[1]);
        _live = _acc.Allocate1D(live);
        _off = _acc.Allocate1D<int>(Math.Max(1, _no));
        _off.MemSetToZero();
        _hist = _acc.Allocate1D<int>(GlobalSfmInit.AngleBins);
        _counts = _acc.Allocate1D<int>(4);
        long no = Math.Max(1, _no), np = Math.Max(1, _np), nf = Math.Max(1, _nf);
        _c = _acc.Allocate1D<double>(nf * 3); _cn = _acc.Allocate1D<double>(nf * 3);
        _x = _acc.Allocate1D<double>(np * 3); _xn = _acc.Allocate1D<double>(np * 3);
        _d = _acc.Allocate1D<double>(no); _dn = _acc.Allocate1D<double>(no);
        _ob = _acc.Allocate1D<double>(no * K);
        _rawP = _acc.Allocate1D<double>(np * 4); _rawC = _acc.Allocate1D<double>(nf * 4);
        _ps = _acc.Allocate1D<double>(np * PS); _cs = _acc.Allocate1D<double>(nf * CS);
        _sys = _acc.Allocate1D<double>(nf * 3 * nf * 3);
        _rhs = _acc.Allocate1D<double>(nf * 3);
        _step = _acc.Allocate1D<double>(nf * 3);
        _cost = _acc.Allocate1D<double>(no);
        _ang = _acc.Allocate1D<double>(no);
        _partial = _acc.Allocate1D<double>(1024);
        _sc = _acc.Allocate1D<double>(NScalars);
        _sc.MemSetToZero();

        // The ILGPU CPU accelerator runs a group's threads as OS threads meeting at every barrier (see GpuBundleAdjuster).
        bool cpu = _acc.AcceleratorType == AcceleratorType.CPU;
        _group = Pow2Floor(Math.Min(cpu ? 16 : GroupSize, _acc.MaxNumThreadsPerGroup));
        _camGroup = Pow2Floor(Math.Min(cpu ? 4 : CamGroupSize, _acc.MaxNumThreadsPerGroup));
        _sc.MemSetToZero();
    }

    static int Pow2Floor(int v) { while ((v & (v - 1)) != 0) v &= v - 1; return v; }

    public void Dispose()
    {
        // Submit whatever is still queued (the constructor's clears at least) BEFORE destroying the buffers: on WebGPU an
        // ILGPU clear is deferred to the next submit, and a buffer destroyed first fails that submit - DrJohnson b67
        // (2026-09-30): no observations, SolveAsync returned at once, and the 4-byte _off's pending clear broke the next
        // unrelated dispatch ("Storage 4B used in submit while destroyed").
        _acc.Flush();
        foreach (var b in new MemoryBuffer?[] { _oc, _op, _pStart, _pObs, _cStart, _cObs, _camOrigBuf, _off, _live, _hist, _counts,
            _v, _c, _cn, _x, _xn, _d, _dn, _ob, _rawP, _rawC, _ps, _cs, _sys, _rhs, _step, _cost, _partial, _sc, _ang })
            b?.Dispose();
        _oc = _op = _pStart = _pObs = _cStart = _cObs = _camOrigBuf = _off = _live = _hist = _counts = null;
        _v = _c = _cn = _x = _xn = _d = _dn = _ob = _rawP = _rawC = _ps = _cs = _sys = _rhs = _step = _cost = _partial = _sc = _ang = null;
    }

    public string TimingSummary()
    {
        double f = System.Diagnostics.Stopwatch.Frequency;
        return $"total {_tTotal / f:F2}s (solves {_tSolve / f:F2}s), {_attempts} attempts, {_readbacks} readbacks";
    }

    // ── kernels ──────────────────────────────────────────────────────────────────────────────

    static void InitCamsKernel(Index1D k, IV camOrig, RV c, int seed)
    {
        for (int a = 0; a < 3; a++) c[k * 3 + a] = GlobalSfmInit.StartCoord(seed, 0, camOrig[k] * 3 + a);
    }

    static void InitPointsKernel(Index1D i, RV x, int seed) => x[i] = GlobalSfmInit.StartCoord(seed, 1, i);

    // Variable projection: each scale at its optimum max(0, v·u / |u|²) for the given C, X (every observation).
    static void ProjectKernel(Index1D t, IV oc, IV op, RV v, RV c, RV x, RV dOut)
    {
        int i = oc[t] * 3, j = op[t] * 3;
        double ux = x[j] - c[i], uy = x[j + 1] - c[i + 1], uz = x[j + 2] - c[i + 2];
        double uu = ux * ux + uy * uy + uz * uz;
        double vu = v[t * 3] * ux + v[t * 3 + 1] * uy + v[t * 3 + 2] * uz;
        bool ahead = uu > 0;
        ahead = ahead & vu > 0;
        dOut[t] = ahead ? vu / uu : 0;
    }

    // Per observation, the IRLS-weighted Gauss-Newton terms in normalised form (no 1/(w|u|²) that could leave the
    // double-float range): with û = u/|u|, the d elimination's s a aᵀ = q ûûᵀ / (1+λ) and s g_d a = gdn û / (1+λ).
    static void ObsTermsKernel(Index1D t, IV oc, IV op, RV v, RV c, RV x, RV d, IV off, RV ob, double huber)
    {
        int o = t * K;
        if (off[t] != 0)
        {
            for (int k = 0; k < K; k++) ob[o + k] = 0;
            return;
        }
        int i = oc[t] * 3, j = op[t] * 3;
        double ux = x[j] - c[i], uy = x[j + 1] - c[i + 1], uz = x[j + 2] - c[i + 2];
        double dt = d[t];
        double r0 = v[t * 3] - dt * ux, r1 = v[t * 3 + 1] - dt * uy, r2 = v[t * 3 + 2] - dt * uz;
        double e = Math.Sqrt(r0 * r0 + r1 * r1 + r2 * r2);
        double w = e <= huber ? 1 : huber / e;
        double m = Math.Sqrt(ux * ux + uy * uy + uz * uz);
        double inv = m > 1e-30 ? 1 / m : 0;
        double h0 = ux * inv, h1 = uy * inv, h2 = uz * inv;
        double wd = w * dt;
        ob[o] = wd * dt;
        ob[o + 1] = h0; ob[o + 2] = h1; ob[o + 3] = h2;
        ob[o + 4] = -wd * (h0 * r0 + h1 * r1 + h2 * r2);
        ob[o + 5] = wd * r0; ob[o + 6] = wd * r1; ob[o + 7] = wd * r2;
    }

    // Per point: Σq and g_X = -Σ gr.
    static void PointRawKernel(Index1D p, IV pStart, IV pObs, RV ob, RV rawP)
    {
        double sq = 0, g0 = 0, g1 = 0, g2 = 0;
        for (int k = pStart[p]; k < pStart[p + 1]; k++)
        {
            int o = pObs[k] * K;
            sq += ob[o]; g0 -= ob[o + 5]; g1 -= ob[o + 6]; g2 -= ob[o + 7];
        }
        rawP[p * 4] = sq; rawP[p * 4 + 1] = g0; rawP[p * 4 + 2] = g1; rawP[p * 4 + 3] = g2;
    }

    // Per camera (one workgroup): Σq and g_C = Σ gr.
    static void CamRawKernel(IV cStart, IV cObs, RV ob, RV rawC)
    {
        var sh = SharedMemory.Allocate<double>(GroupSize);
        int c = Grid.IdxX, t = Group.IdxX, dim = Group.DimX;
        double sq = 0, g0 = 0, g1 = 0, g2 = 0;
        for (int k = cStart[c] + t; k < cStart[c + 1]; k += dim)
        {
            int o = cObs[k] * K;
            sq += ob[o]; g0 += ob[o + 5]; g1 += ob[o + 6]; g2 += ob[o + 7];
        }
        double a = GroupSum(sh, sq), b = GroupSum(sh, g0), e = GroupSum(sh, g1), f = GroupSum(sh, g2);
        if (t == 0) { rawC[c * 4] = a; rawC[c * 4 + 1] = b; rawC[c * 4 + 2] = e; rawC[c * 4 + 3] = f; }
    }

    // One workgroup: the centroid gauge weight mu = mean camera Σq (the managed solver's first camera diagonal).
    static void MuKernel(RV rawC, RV sc, int nf)
    {
        var sh = SharedMemory.Allocate<double>(GroupSize);
        int t = Group.IdxX, dim = Group.DimX;
        double s = 0;
        for (int c = t; c < nf; c += dim) s += rawC[c * 4];
        double total = GroupSum(sh, s);
        if (t == 0) sc[S_MU] = Math.Max(total / nf, 1e-12);
    }

    // One workgroup: sc[S_CSUM..] = Σ C.
    static void CSumKernel(RV c, RV sc, int nf)
    {
        var sh = SharedMemory.Allocate<double>(GroupSize);
        int t = Group.IdxX, dim = Group.DimX;
        double s0 = 0, s1 = 0, s2 = 0;
        for (int k = t; k < nf; k += dim) { s0 += c[k * 3]; s1 += c[k * 3 + 1]; s2 += c[k * 3 + 2]; }
        double a = GroupSum(sh, s0), b = GroupSum(sh, s1), e = GroupSum(sh, s2);
        if (t == 0) { sc[S_CSUM] = a; sc[S_CSUM + 1] = b; sc[S_CSUM + 2] = e; }
    }

    /// <summary>Inverse of a symmetric 3x3 (a00 a01 a02 a11 a12 a22) through its Jacobi-scaled form (unit diagonal), as
    /// GpuBundleAdjuster's point solve: exact in exact arithmetic at any magnitude. Returns 0 when the matrix is not finite
    /// (the attempt fails), 1 when singular - a zero inverse, or det' floored - and 2 when fine.
    /// A zero diagonal is not a failure: a point whose every scale is clamped at 0 (all its bearings more than 90 deg off)
    /// has a zero block AND zero gradient and coupling, so it simply does not move. MEASURED 2026-09-29: failing the attempt
    /// there (the first version) rejected every LM step from the random start - 0 accepted, lambda to 1e12. The managed
    /// solver's Invert3 returns identity for it, which moves the point by the same zero.</summary>
    static int InvertSym3(double a00, double a01, double a02, double a11, double a12, double a22,
        out double i0, out double i1, out double i2, out double i4, out double i5, out double i8)
    {
        i0 = i1 = i2 = i4 = i5 = i8 = 0;
        double probe = a00 + a01 + a02 + a11 + a12 + a22;
        if (!(probe - probe == 0)) return 0;   // NaN or infinity
        bool positive = a00 > 0;
        positive = positive & a11 > 0;
        positive = positive & a22 > 0;
        if (!positive) return 1;
        double d0 = 1 / Math.Sqrt(a00), d1 = 1 / Math.Sqrt(a11), d2 = 1 / Math.Sqrt(a22);
        double sa = a01 * d0 * d1, sb = a02 * d0 * d2, sc = a12 * d1 * d2;
        double det = 1 + 2 * sa * sb * sc - sa * sa - sb * sb - sc * sc;
        bool finiteDet = det > -1;
        finiteDet = finiteDet & det < 2;
        if (!finiteDet) return 0;
        int status = 2;
        if (!(det > 1e-14)) { det = 1e-14; status = 1; }
        double id = 1 / det;
        i0 = (1 - sc * sc) * id * d0 * d0;
        i1 = (sb * sc - sa) * id * d0 * d1;
        i2 = (sa * sc - sb) * id * d0 * d2;
        i4 = (1 - sb * sb) * id * d1 * d1;
        i5 = (sa * sb - sc) * id * d1 * d2;
        i8 = (1 - sa * sa) * id * d2 * d2;
        return status;
    }

    // Per point (λ): V* = (1+λ)Σq I - Σ q/(1+λ) ûûᵀ, g' = g_X - Σ gdn/(1+λ) û; stores V*⁻¹, V*⁻¹ g', g'. A retired point
    // (fewer than two inliers) stores zeros: it drops out of every product.
    static void PointSolveKernel(Index1D p, IV pStart, IV pObs, RV ob, RV rawP, IV live, RV ps, RV sc, double lambda)
    {
        int b = p * PS;
        if (live[p] < 2)
        {
            for (int k = 0; k < PS; k++) ps[b + k] = 0;
            return;
        }
        double f = 1 + lambda, sq = rawP[p * 4] * f;
        double a00 = sq, a01 = 0, a02 = 0, a11 = sq, a12 = 0, a22 = sq;
        double g0 = rawP[p * 4 + 1], g1 = rawP[p * 4 + 2], g2 = rawP[p * 4 + 3];
        for (int k = pStart[p]; k < pStart[p + 1]; k++)
        {
            int o = pObs[k] * K;
            double qf = ob[o] / f, h0 = ob[o + 1], h1 = ob[o + 2], h2 = ob[o + 3], gf = ob[o + 4] / f;
            a00 -= qf * h0 * h0; a01 -= qf * h0 * h1; a02 -= qf * h0 * h2;
            a11 -= qf * h1 * h1; a12 -= qf * h1 * h2; a22 -= qf * h2 * h2;
            g0 -= gf * h0; g1 -= gf * h1; g2 -= gf * h2;
        }
        int status = InvertSym3(a00, a01, a02, a11, a12, a22, out var i0, out var i1, out var i2, out var i4, out var i5, out var i8);
        if (status == 0) sc[S_FAILED] = 1;
        ps[b] = i0; ps[b + 1] = i1; ps[b + 2] = i2;
        ps[b + 3] = i1; ps[b + 4] = i4; ps[b + 5] = i5;
        ps[b + 6] = i2; ps[b + 7] = i5; ps[b + 8] = i8;
        ps[b + 9] = i0 * g0 + i1 * g1 + i2 * g2;
        ps[b + 10] = i1 * g0 + i4 * g1 + i5 * g2;
        ps[b + 11] = i2 * g0 + i5 * g1 + i8 * g2;
        ps[b + 12] = g0; ps[b + 13] = g1; ps[b + 14] = g2;
    }

    // Per camera (one workgroup, λ): U' = (1+λ)Σq I - Σ q/(1+λ) ûûᵀ and rhs = g_C + Σ gdn/(1+λ) û - Σ B V*⁻¹ g', where
    // B = -q I + q/(1+λ) ûûᵀ is the observation's camera-point block.
    static void CamPrepKernel(IV cStart, IV cObs, IV op, RV ob, RV ps, RV rawC, RV cs, double lambda)
    {
        var sh = SharedMemory.Allocate<double>(GroupSize);
        var acc = LocalMemory.Allocate<double>(12);   // U' part (6), g part (3), Σ B t (3)
        int c = Grid.IdxX, t = Group.IdxX, dim = Group.DimX;
        double f = 1 + lambda;
        for (int k = 0; k < 12; k++) acc[k] = 0;
        for (int k = cStart[c] + t; k < cStart[c + 1]; k += dim)
        {
            int obs = cObs[k], o = obs * K, pb = op[obs] * PS;
            double q = ob[o], qf = q / f, h0 = ob[o + 1], h1 = ob[o + 2], h2 = ob[o + 3], gf = ob[o + 4] / f;
            acc[0] -= qf * h0 * h0; acc[1] -= qf * h0 * h1; acc[2] -= qf * h0 * h2;
            acc[3] -= qf * h1 * h1; acc[4] -= qf * h1 * h2; acc[5] -= qf * h2 * h2;
            acc[6] += gf * h0; acc[7] += gf * h1; acc[8] += gf * h2;
            // B t = -q t + qf û (û·t)
            double t0 = ps[pb + 9], t1 = ps[pb + 10], t2 = ps[pb + 11];
            double ht = h0 * t0 + h1 * t1 + h2 * t2;
            acc[9] += -q * t0 + qf * h0 * ht; acc[10] += -q * t1 + qf * h1 * ht; acc[11] += -q * t2 + qf * h2 * ht;
        }
        double sq = rawC[c * 4] * f;
        int b = c * CS;
        for (int k = 0; k < 12; k++)
        {
            double total = GroupSum(sh, acc[k]);
            if (t == 0)
            {
                if (k < 6) cs[b + k] = total + (k * (k - 3) * (k - 5) == 0 ? sq : 0);   // diagonal entries 0, 3, 5
                else if (k < 9) cs[b + k] = rawC[c * 4 + 1 + (k - 6)] + total;   // g'_C, then minus Σ B t below
                else cs[b + k - 3] = cs[b + k - 3] - total;
            }
        }
    }

    // Per camera: the system's right-hand side -(rhs + hg Σ C), hg = mu/nf (the centroid gauge).
    static void RhsKernel(Index1D k, RV cs, RV sc, RV rhs, int nf)
    {
        double hg = sc[S_MU] / nf;
        for (int a = 0; a < 3; a++) rhs[k * 3 + a] = -(cs[k * CS + 6 + a] + hg * sc[S_CSUM + a]);
    }

    // One thread per camera i builds ITS three rows of S (no other thread writes them): U'_i on the diagonal block, minus
    // B_1 V*⁻¹ B_2 for every pair of observations (camera i's, any camera's) of the same point - the point's Schur
    // complement. The gauge is added by the factorisation. Rejected observations (q = 0) and retired points (V*⁻¹ = 0)
    // contribute nothing.
    static void BuildSystemKernel(Index1D i, IV cStart, IV cObs, IV pStart, IV pObs, IV oc, IV op, RV ob, RV ps, RV cs, RV sys,
        double lambda, int nf)
    {
        int n = nf * 3, r0 = i * 3 * n;
        double f = 1 + lambda;
        for (int k = 0; k < 3 * n; k++) sys[r0 + k] = 0;
        int b = i * CS, d0 = r0 + i * 3;
        sys[d0] = cs[b]; sys[d0 + 1] = cs[b + 1]; sys[d0 + 2] = cs[b + 2];
        sys[d0 + n] = cs[b + 1]; sys[d0 + n + 1] = cs[b + 3]; sys[d0 + n + 2] = cs[b + 4];
        sys[d0 + 2 * n] = cs[b + 2]; sys[d0 + 2 * n + 1] = cs[b + 4]; sys[d0 + 2 * n + 2] = cs[b + 5];
        for (int k1 = cStart[i]; k1 < cStart[i + 1]; k1++)
        {
            int t1 = cObs[k1], o1 = t1 * K;
            double q1 = ob[o1];
            if (!(q1 > 0)) continue;
            int p = op[t1], pb = p * PS;
            double qf1 = q1 / f, a0 = ob[o1 + 1], a1 = ob[o1 + 2], a2 = ob[o1 + 3];
            double p00 = ps[pb], p01 = ps[pb + 1], p02 = ps[pb + 2], p11 = ps[pb + 4], p12 = ps[pb + 5], p22 = ps[pb + 8];
            // M = B_1 P = -q1 P + qf1 a (P a)ᵀ   (P symmetric)
            double w0 = p00 * a0 + p01 * a1 + p02 * a2, w1 = p01 * a0 + p11 * a1 + p12 * a2, w2 = p02 * a0 + p12 * a1 + p22 * a2;
            double m00 = -q1 * p00 + qf1 * a0 * w0, m01 = -q1 * p01 + qf1 * a0 * w1, m02 = -q1 * p02 + qf1 * a0 * w2;
            double m10 = -q1 * p01 + qf1 * a1 * w0, m11 = -q1 * p11 + qf1 * a1 * w1, m12 = -q1 * p12 + qf1 * a1 * w2;
            double m20 = -q1 * p02 + qf1 * a2 * w0, m21 = -q1 * p12 + qf1 * a2 * w1, m22 = -q1 * p22 + qf1 * a2 * w2;
            for (int k2 = pStart[p]; k2 < pStart[p + 1]; k2++)
            {
                int t2 = pObs[k2], o2 = t2 * K;
                double q2 = ob[o2];
                if (!(q2 > 0)) continue;
                double qf2 = q2 / f, h0 = ob[o2 + 1], h1 = ob[o2 + 2], h2 = ob[o2 + 3];
                // M B_2 = -q2 M + qf2 (M h) hᵀ
                double mh0 = m00 * h0 + m01 * h1 + m02 * h2, mh1 = m10 * h0 + m11 * h1 + m12 * h2, mh2 = m20 * h0 + m21 * h1 + m22 * h2;
                int cb = r0 + oc[t2] * 3;
                sys[cb] -= -q2 * m00 + qf2 * mh0 * h0; sys[cb + 1] -= -q2 * m01 + qf2 * mh0 * h1; sys[cb + 2] -= -q2 * m02 + qf2 * mh0 * h2;
                sys[cb + n] -= -q2 * m10 + qf2 * mh1 * h0; sys[cb + n + 1] -= -q2 * m11 + qf2 * mh1 * h1; sys[cb + n + 2] -= -q2 * m12 + qf2 * mh1 * h2;
                sys[cb + 2 * n] -= -q2 * m20 + qf2 * mh2 * h0; sys[cb + 2 * n + 1] -= -q2 * m21 + qf2 * mh2 * h1; sys[cb + 2 * n + 2] -= -q2 * m22 + qf2 * mh2 * h2;
            }
        }
    }

    // One workgroup: add the centroid gauge (hg on every camera pair, per axis), factor S = L Lᵀ in place (lower triangle,
    // right-looking: the column, then each thread updates whole trailing rows), then L y = b, Lᵀ x = y. Every loop bound
    // is the uniform n, so every barrier is in uniform control flow (WGSL requires it). A pivot <= 0 (not positive
    // definite) flags the attempt as failed, as the managed CholeskySolve returning false.
    static void CholeskyKernel(RV sys, RV rhs, RV x, RV sc, int nf)
    {
        int t = Group.IdxX, dim = Group.DimX, n = nf * 3;
        double hg = sc[S_MU] / nf;
        for (int idx = t; idx < n * n; idx += dim)
        {
            int r = idx / n, c = idx - r * n;
            if (r % 3 == c % 3) sys[idx] += hg;
        }
        Group.Barrier();
        for (int j = 0; j < n; j++)
        {
            if (t == 0)
            {
                double s = sys[j * n + j];
                if (!(s > 0)) { sc[S_FAILED] = 1; s = 1; }
                sys[j * n + j] = Math.Sqrt(s);
            }
            Group.Barrier();
            double l = sys[j * n + j];
            for (int i = j + 1 + t; i < n; i += dim) sys[i * n + j] /= l;
            Group.Barrier();
            for (int i = j + 1 + t; i < n; i += dim)
            {
                double lij = sys[i * n + j];
                for (int k = j + 1; k <= i; k++) sys[i * n + k] -= lij * sys[k * n + j];
            }
            Group.Barrier();
        }
        for (int i = t; i < n; i += dim) x[i] = rhs[i];
        Group.Barrier();
        for (int j = 0; j < n; j++)
        {
            if (t == 0) x[j] = x[j] / sys[j * n + j];
            Group.Barrier();
            double xj = x[j];
            for (int i = j + 1 + t; i < n; i += dim) x[i] -= sys[i * n + j] * xj;
            Group.Barrier();
        }
        for (int j = n - 1; j >= 0; j--)
        {
            if (t == 0) x[j] = x[j] / sys[j * n + j];
            Group.Barrier();
            double xj = x[j];
            for (int i = t; i < j; i += dim) x[i] -= sys[j * n + i] * xj;
            Group.Barrier();
        }
    }

    // Per point: the candidate X + V*⁻¹ (-g' - Σ_obs B dc_cam), dc = the camera step.
    static void PointBacksubKernel(Index1D p, IV pStart, IV pObs, IV oc, RV ob, RV ps, RV step, RV x, RV xn, RV sc)
    {
        double f = sc[S_F];
        int b = p * PS;
        double y0 = -ps[b + 12], y1 = -ps[b + 13], y2 = -ps[b + 14];
        for (int k = pStart[p]; k < pStart[p + 1]; k++)
        {
            int obs = pObs[k], o = obs * K, cb = oc[obs] * 3;
            double q = ob[o], h0 = ob[o + 1], h1 = ob[o + 2], h2 = ob[o + 3];
            double x0 = step[cb], x1 = step[cb + 1], x2 = step[cb + 2];
            double qfh = q / f * (h0 * x0 + h1 * x1 + h2 * x2);
            y0 -= -q * x0 + qfh * h0; y1 -= -q * x1 + qfh * h1; y2 -= -q * x2 + qfh * h2;
        }
        xn[p * 3] = x[p * 3] + ps[b] * y0 + ps[b + 1] * y1 + ps[b + 2] * y2;
        xn[p * 3 + 1] = x[p * 3 + 1] + ps[b + 3] * y0 + ps[b + 4] * y1 + ps[b + 5] * y2;
        xn[p * 3 + 2] = x[p * 3 + 2] + ps[b + 6] * y0 + ps[b + 7] * y1 + ps[b + 8] * y2;
    }

    static void CandCamKernel(Index1D i, RV c, RV step, RV cn) => cn[i] = c[i] + step[i];

    // Per observation: Huber cost of |v - d (X - C)| (0 for a rejected observation).
    static void CostKernel(Index1D t, IV oc, IV op, RV v, RV c, RV x, RV d, IV off, RV cost, double huber)
    {
        if (off[t] != 0) { cost[t] = 0; return; }
        int i = oc[t] * 3, j = op[t] * 3;
        double dt = d[t];
        double r0 = v[t * 3] - dt * (x[j] - c[i]), r1 = v[t * 3 + 1] - dt * (x[j + 1] - c[i + 1]), r2 = v[t * 3 + 2] - dt * (x[j + 2] - c[i + 2]);
        double e = Math.Sqrt(r0 * r0 + r1 * r1 + r2 * r2);
        cost[t] = e <= huber ? 0.5 * e * e : huber * (e - 0.5 * huber);
    }

    static void PartialSumKernel(Index1D t, RV values, RV partial, int n)
    {
        double s = 0;
        for (int i = t; i < n; i += 1024) s += values[i];
        partial[t] = s;
    }

    // One workgroup: sc[S_COST] = Σ partial + the centroid gauge (mu/2)|Σ C|²/nf once mu is set.
    static void FinalCostKernel(RV partial, RV c, RV sc, int nf)
    {
        var sh = SharedMemory.Allocate<double>(GroupSize);
        int t = Group.IdxX, dim = Group.DimX;
        double s = 0, c0 = 0, c1 = 0, c2 = 0;
        for (int k = t; k < 1024; k += dim) s += partial[k];
        for (int k = t; k < nf; k += dim) { c0 += c[k * 3]; c1 += c[k * 3 + 1]; c2 += c[k * 3 + 2]; }
        double total = GroupSum(sh, s), a = GroupSum(sh, c0), b = GroupSum(sh, c1), e = GroupSum(sh, c2);
        if (t == 0)
        {
            double mu = sc[S_MU];
            sc[S_COST] = total + (mu > 0 ? 0.5 * mu * (a * a + b * b + e * e) / nf : 0);
        }
    }

    // A retired point (fewer than two inliers, left behind by the solve) from ALL its bearings:
    // min Σ |(I - v vᵀ)(X - C)|² (parallel bearings: the point keeps its X).
    static void RetriangulateKernel(Index1D p, IV pStart, IV pObs, IV oc, RV v, RV c, IV live, RV x)
    {
        // WebGPU: a compound predicate inline in an early return dropped every thread once (fb-wgsl-inline-predicate-drops-all).
        bool skip = live[p] >= 2;
        skip = skip | pStart[p + 1] - pStart[p] < 2;
        if (skip) return;
        double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0, b0 = 0, b1 = 0, b2 = 0;
        for (int k = pStart[p]; k < pStart[p + 1]; k++)
        {
            int t = pObs[k], i = oc[t] * 3;
            double v0 = v[t * 3], v1 = v[t * 3 + 1], v2 = v[t * 3 + 2];
            double c0 = c[i], c1 = c[i + 1], c2 = c[i + 2];
            double p00 = 1 - v0 * v0, p01 = -v0 * v1, p02 = -v0 * v2, p11 = 1 - v1 * v1, p12 = -v1 * v2, p22 = 1 - v2 * v2;
            a00 += p00; a01 += p01; a02 += p02; a11 += p11; a12 += p12; a22 += p22;
            b0 += p00 * c0 + p01 * c1 + p02 * c2; b1 += p01 * c0 + p11 * c1 + p12 * c2; b2 += p02 * c0 + p12 * c1 + p22 * c2;
        }
        double det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
        if (!(det > 1e-12 * a00 * a11 * a22)) return;
        double id = 1 / det;
        double i0 = (a11 * a22 - a12 * a12) * id, i1 = (a02 * a12 - a01 * a22) * id, i2 = (a01 * a12 - a02 * a11) * id;
        double i4 = (a00 * a22 - a02 * a02) * id, i5 = (a01 * a02 - a00 * a12) * id, i8 = (a00 * a11 - a01 * a01) * id;
        x[p * 3] = i0 * b0 + i1 * b1 + i2 * b2;
        x[p * 3 + 1] = i1 * b0 + i4 * b1 + i5 * b2;
        x[p * 3 + 2] = i2 * b0 + i5 * b1 + i8 * b2;
    }

    // Per observation: tan(theta/2) to its point; inliers go into the median histogram (GlobalSfmInit.AngleBin).
    static void AngleKernel(Index1D t, IV oc, IV op, RV v, RV c, RV x, IV off, RV ang, IV hist)
    {
        int i = oc[t] * 3, j = op[t] * 3;
        double a = GlobalSfmInit.TanHalfAngle(v[t * 3], v[t * 3 + 1], v[t * 3 + 2], x[j] - c[i], x[j + 1] - c[i + 1], x[j + 2] - c[i + 2]);
        ang[t] = a;
        if (off[t] == 0)
        {
            float fa = (float)Math.Min(a, 1e30);
            if (!(fa > 1e-30f)) fa = 1e-30f;
            int key = (int)(Interop.FloatAsInt(fa) >> 16) - GlobalSfmInit.AngleKeyMin;
            int bin = key < 0 ? 0 : key > GlobalSfmInit.AngleBins - 1 ? GlobalSfmInit.AngleBins - 1 : key;
            Atomic.Add(ref hist[bin], 1);
        }
    }

    static void ClearIntKernel(Index1D i, IV a) => a[i] = 0;

    static void SetScalarKernel(Index1D i, RV sc, int slot, double value) => sc[slot] = value;

    // One thread: the upper edge of the bin holding the median inlier angle (GlobalSfmInit's rule).
    static void MedianKernel(Index1D i, IV hist, RV sc)
    {
        int total = 0;
        for (int b = 0; b < GlobalSfmInit.AngleBins; b++) total += hist[b];
        int acc = 0, bin = GlobalSfmInit.AngleBins - 1;
        for (int b = 0; b < GlobalSfmInit.AngleBins; b++)
        {
            acc += hist[b];
            if (acc > total / 2) { bin = b; break; }
        }
        sc[S_MEDIAN] = Interop.IntAsFloat((uint)((bin + 1 + GlobalSfmInit.AngleKeyMin) << 16));
    }

    // Per point: re-select its inliers (tan(theta/2) <= limit), retire it with fewer than two, count the changes.
    static void SelectKernel(Index1D p, IV pStart, IV pObs, RV ang, IV off, IV live, IV counts, double limit)
    {
        int cnt = 0;
        for (int k = pStart[p]; k < pStart[p + 1]; k++) if (!(ang[pObs[k]] > limit)) cnt++;
        bool retire = cnt < 2;
        int dropped = 0, readmitted = 0, outliers = 0;
        for (int k = pStart[p]; k < pStart[p + 1]; k++)
        {
            int t = pObs[k];
            bool outside = ang[t] > limit;
            outside = outside | retire;
            int nowOff = outside ? 1 : 0;
            int was = off[t];
            if (nowOff > was) dropped++;
            if (nowOff < was) readmitted++;
            if (nowOff != 0) outliers++;
            off[t] = nowOff;
        }
        live[p] = retire ? 0 : cnt;
        if (dropped > 0) Atomic.Add(ref counts[0], dropped);
        if (readmitted > 0) Atomic.Add(ref counts[1], readmitted);
        if (outliers > 0) Atomic.Add(ref counts[2], outliers);
    }

    // ── kernel loading ──────────────────────────────────────────────────────────────────────

    sealed class Kernels
    {
        public Action<Index1D, IV, RV, int> InitCams = null!;
        public Action<Index1D, RV, int> InitPoints = null!;
        public Action<Index1D, IV, IV, RV, RV, RV, RV> Project = null!;
        public Action<Index1D, IV, IV, RV, RV, RV, RV, IV, RV, double> ObsTerms = null!;
        public Action<Index1D, IV, IV, RV, RV> PointRaw = null!;
        public Action<KernelConfig, IV, IV, RV, RV> CamRaw = null!;
        public Action<KernelConfig, RV, RV, int> Mu = null!;
        public Action<KernelConfig, RV, RV, int> CSum = null!;
        public Action<Index1D, IV, IV, RV, RV, IV, RV, RV, double> PointSolve = null!;
        public Action<KernelConfig, IV, IV, IV, RV, RV, RV, RV, double> CamPrep = null!;
        public Action<Index1D, RV, RV, RV, int> Rhs = null!;
        public Action<Index1D, IV, IV, IV, IV, IV, IV, RV, RV, RV, RV, double, int> BuildSystem = null!;
        public Action<KernelConfig, RV, RV, RV, RV, int> Cholesky = null!;
        public Action<Index1D, IV, IV, IV, RV, RV, RV, RV, RV, RV> PointBacksub = null!;
        public Action<Index1D, RV, RV, RV> CandCam = null!;
        public Action<Index1D, IV, IV, RV, RV, RV, RV, IV, RV, double> Cost = null!;
        public Action<Index1D, RV, RV, int> PartialSum = null!;
        public Action<KernelConfig, RV, RV, RV, int> FinalCost = null!;
        public Action<Index1D, IV, IV, IV, RV, RV, IV, RV> Retriangulate = null!;
        public Action<Index1D, IV, IV, RV, RV, RV, IV, RV, IV> Angle = null!;
        public Action<Index1D, IV> ClearInt = null!;
        public Action<Index1D, RV, int, double> SetScalar = null!;
        public Action<Index1D, IV, RV> Median = null!;
        public Action<Index1D, IV, IV, RV, IV, IV, IV, double> Select = null!;
    }
    static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    static Kernels For(Accelerator a) => s_kernels.GetValue(a, acc => new Kernels
    {
        InitCams = acc.LoadAutoGroupedStreamKernel<Index1D, IV, RV, int>(InitCamsKernel),
        InitPoints = acc.LoadAutoGroupedStreamKernel<Index1D, RV, int>(InitPointsKernel),
        Project = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV, RV, RV>(ProjectKernel),
        ObsTerms = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV, RV, RV, IV, RV, double>(ObsTermsKernel),
        PointRaw = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV>(PointRawKernel),
        CamRaw = acc.LoadStreamKernel<IV, IV, RV, RV>(CamRawKernel),
        Mu = acc.LoadStreamKernel<RV, RV, int>(MuKernel),
        CSum = acc.LoadStreamKernel<RV, RV, int>(CSumKernel),
        PointSolve = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV, IV, RV, RV, double>(PointSolveKernel),
        CamPrep = acc.LoadStreamKernel<IV, IV, IV, RV, RV, RV, RV, double>(CamPrepKernel),
        Rhs = acc.LoadAutoGroupedStreamKernel<Index1D, RV, RV, RV, int>(RhsKernel),
        BuildSystem = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, IV, RV, RV, RV, RV, double, int>(BuildSystemKernel),
        Cholesky = acc.LoadStreamKernel<RV, RV, RV, RV, int>(CholeskyKernel),
        PointBacksub = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, RV, RV, RV, RV, RV, RV>(PointBacksubKernel),
        CandCam = acc.LoadAutoGroupedStreamKernel<Index1D, RV, RV, RV>(CandCamKernel),
        Cost = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV, RV, RV, IV, RV, double>(CostKernel),
        PartialSum = acc.LoadAutoGroupedStreamKernel<Index1D, RV, RV, int>(PartialSumKernel),
        FinalCost = acc.LoadStreamKernel<RV, RV, RV, int>(FinalCostKernel),
        Retriangulate = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, RV, RV, IV, RV>(RetriangulateKernel),
        Angle = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, RV, RV, IV, RV, IV>(AngleKernel),
        ClearInt = acc.LoadAutoGroupedStreamKernel<Index1D, IV>(ClearIntKernel),
        SetScalar = acc.LoadAutoGroupedStreamKernel<Index1D, RV, int, double>(SetScalarKernel),
        Median = acc.LoadAutoGroupedStreamKernel<Index1D, IV, RV>(MedianKernel),
        Select = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, RV, IV, IV, IV, double>(SelectKernel),
    });

    // ── the solve (host: the LM / rounds control flow and a few scalars) ─────────────────────

    async Task<double[]> ReadScalarsAsync()
    {
        _readbacks++;
        // WebGPU: the readback flushes pending kernels and is queue-ordered after them (see GpuBundleAdjuster.ReadAsync).
        if (_acc is not WebGPUAccelerator) await _acc.SynchronizeAsync();
        return await _sc!.CopyToHostAsync<double>();
    }

    async Task<int[]> ReadCountsAsync()
    {
        _readbacks++;
        if (_acc is not WebGPUAccelerator) await _acc.SynchronizeAsync();
        return await _counts!.CopyToHostAsync<int>();
    }

    KernelConfig One => new(1, _group);
    KernelConfig PerCamera => new(_nf, _camGroup);

    /// <summary>The cost of (C, X, d) with the current inlier set (and the gauge term once mu is set), and whether the
    /// attempt that produced it failed (a non-finite point block or a camera system that is not positive definite).</summary>
    async Task<(double Cost, bool Failed)> CostAsync(MemoryBuffer1D<double, Stride1D.Dense> c, MemoryBuffer1D<double, Stride1D.Dense> x,
        MemoryBuffer1D<double, Stride1D.Dense> d)
    {
        var k = For(_acc);
        k.Cost(_no, _oc!.View, _op!.View, _v!.View, c.View, x.View, d.View, _off!.View, _cost!.View, _huber);
        k.PartialSum(1024, _cost.View, _partial!.View, _no);
        k.FinalCost(One, _partial.View, c.View, _sc!.View, _nf);
        var sc = await ReadScalarsAsync();
        return (sc[S_COST], sc[S_FAILED] != 0);
    }

    /// <summary>Enqueue the damped step into _step (no readback: a failure shows in the candidate's cost read).</summary>
    void EnqueueDampedStep(double lambda)
    {
        var k = For(_acc);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        _attempts++;
        k.SetScalar(1, _sc!.View, S_F, 1 + lambda);
        k.SetScalar(1, _sc.View, S_FAILED, 0);
        k.PointSolve(_np, _pStart!.View, _pObs!.View, _ob!.View, _rawP!.View, _live!.View, _ps!.View, _sc.View, lambda);
        k.CamPrep(PerCamera, _cStart!.View, _cObs!.View, _op!.View, _ob.View, _ps.View, _rawC!.View, _cs!.View, lambda);
        k.CSum(One, _c!.View, _sc.View, _nf);
        k.Rhs(_nf, _cs.View, _sc.View, _rhs!.View, _nf);
        k.BuildSystem(_nf, _cStart.View, _cObs.View, _pStart.View, _pObs.View, _oc!.View, _op.View, _ob.View, _ps.View, _cs.View,
            _sys!.View, lambda, _nf);
        k.Cholesky(One, _sys.View, _rhs.View, _step!.View, _sc.View, _nf);
        _tSolve += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
    }

    public async Task<GlobalSfmInit.PositioningResult> SolveAsync()
    {
        long tStart = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_no == 0 || _nf == 0 || _np == 0)
            return new GlobalSfmInit.PositioningResult(_cams.Select(c => c.Position).ToArray(), "robust positioning [GPU]: no observations", false, 0, 0);
        var k = For(_acc);
        k.InitCams(_nf, _camOrigBuf!.View, _c!.View, _seed);
        k.InitPoints(_np * 3, _x!.View, _seed);
        k.Project(_no, _oc!.View, _op!.View, _v!.View, _c.View, _x.View, _d!.View);
        double lambda = 1e-4, cost0 = (await CostAsync(_c, _x, _d)).Cost, cost = cost0;
        bool muSet = false, dirty = true, roundConverged = false;
        int iter = 0, accepted = 0, outliers = 0, round = 0, roundsRun = 0;
        var roundLog = new List<string>();
        for (; round < 6; round++)
        {
            roundsRun++;
            int roundStart = iter, acceptedStart = accepted;
            roundConverged = false;
            for (int roundIter = 0; roundIter < _maxIterations; roundIter++, iter++)
            {
                if (dirty)
                {
                    // The IRLS-weighted terms at the current state (an unaccepted step leaves them unchanged).
                    k.ObsTerms(_no, _oc.View, _op.View, _v.View, _c.View, _x.View, _d.View, _off!.View, _ob!.View, _huber);
                    k.PointRaw(_np, _pStart!.View, _pObs!.View, _ob.View, _rawP!.View);
                    k.CamRaw(PerCamera, _cStart!.View, _cObs!.View, _ob.View, _rawC!.View);
                    dirty = false;
                    if (!muSet)
                    {
                        k.Mu(One, _rawC.View, _sc!.View, _nf);
                        muSet = true;
                        cost0 = cost = (await CostAsync(_c, _x, _d)).Cost;
                    }
                }
                EnqueueDampedStep(lambda);
                k.CandCam(_dim, _c.View, _step!.View, _cn!.View);
                k.PointBacksub(_np, _pStart.View, _pObs.View, _oc.View, _ob.View, _ps!.View, _step.View, _x.View, _xn!.View, _sc!.View);
                k.Project(_no, _oc.View, _op.View, _v.View, _cn.View, _xn.View, _dn!.View);
                var (newCost, failed) = await CostAsync(_cn, _xn, _dn);
                if (failed)
                {
                    Trace?.Invoke($"r{round} it{roundIter} lambda {lambda:E1}: the damped system failed (not positive definite)");
                    lambda *= 10;
                    if (lambda > 1e12) break;
                    continue;
                }
                Trace?.Invoke($"r{round} it{roundIter} lambda {lambda:E1} cost {cost:G6} -> {newCost:G6} {(newCost < cost ? "ACC" : "rej")}");
                if (newCost < cost)
                {
                    bool converged = (cost - newCost) < 1e-10 * cost;
                    (_c, _cn) = (_cn, _c); (_x, _xn) = (_xn, _x); (_d, _dn) = (_dn, _d);
                    cost = newCost;
                    lambda = Math.Max(lambda / 3, 1e-12);
                    accepted++;
                    dirty = true;
                    if (converged) { roundConverged = true; iter++; break; }
                }
                else
                {
                    lambda *= 4;
                    if (lambda > 1e12) break;
                }
            }
            // Re-select the inliers from all observations (GlobalSfmInit's rule), retired points re-triangulated first.
            k.ClearInt(GlobalSfmInit.AngleBins, _hist!.View);
            k.ClearInt(4, _counts!.View);
            k.Retriangulate(_np, _pStart.View, _pObs.View, _oc.View, _v.View, _c.View, _live!.View, _x.View);
            k.Angle(_no, _oc.View, _op.View, _v.View, _c.View, _x.View, _off.View, _ang!.View, _hist.View);
            k.Median(1, _hist.View, _sc.View);
            double limit = GlobalSfmInit.InlierLimit((await ReadScalarsAsync())[S_MEDIAN]);
            k.Select(_np, _pStart.View, _pObs.View, _ang.View, _off.View, _live.View, _counts.View, limit);
            var counts = await ReadCountsAsync();
            int dropped = counts[0], readmitted = counts[1];
            outliers = counts[2];
            roundLog.Add($"{iter - roundStart}/{accepted - acceptedStart}{(roundConverged ? "" : " (cap)")} -{dropped}+{readmitted}");
            if (dropped == 0 && readmitted == 0) break;
            k.Project(_no, _oc.View, _op.View, _v.View, _c.View, _x.View, _d.View);
            cost = (await CostAsync(_c, _x, _d)).Cost;
            lambda = 1e-4;
            dirty = true;
        }
        // CPU transfer: the camera centres are the result (nf x 3, once).
        var cc = await _c!.CopyToHostAsync<double>();
        var compact = new int[_n];
        for (int i = 0; i < _nf; i++) compact[_camOrig[i]] = i;
        var centres = GlobalSfmInit.ToCurrentFrame(_cams, _connected,
            i => new Vector3((float)cc[compact[i] * 3], (float)cc[compact[i] * 3 + 1], (float)cc[compact[i] * 3 + 2]));
        _tTotal = System.Diagnostics.Stopwatch.GetTimestamp() - tStart;
        return new GlobalSfmInit.PositioningResult(centres,
            $"robust positioning [GPU]: {_no} obs, {roundsRun} rounds [iterations/accepted -dropped+readmitted: {string.Join(", ", roundLog)}], " +
            $"{iter} iterations ({accepted} accepted), {outliers} outliers, cost {cost0:G4} -> {cost:G4}; {TimingSummary()}",
            roundConverged, outliers, _no);
    }

    static double GroupSum(ArrayView<double> sh, double v)
    {
        int t = Group.IdxX, dim = Group.DimX;
        sh[t] = v;
        Group.Barrier();
        for (int s = GroupSize / 2; s > 0; s >>= 1)
        {
            if (s < dim)
            {
                if (t < s) sh[t] += sh[t + s];
                Group.Barrier();
            }
        }
        double total = sh[0];
        Group.Barrier();
        return total;
    }
}
