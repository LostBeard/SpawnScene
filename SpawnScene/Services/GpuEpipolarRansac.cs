using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnScene.Services;

/// <summary>
/// <see cref="EpipolarRansac"/> for every candidate pair at once: all hypotheses of all pairs are generated and
/// scored on the device, one thread per (pair, hypothesis); the host keeps only the per-pair refit and final mask
/// (<see cref="EpipolarRansac.Finish"/>, in double - the same tail the CPU estimator runs).
///
/// Why: MEASURED 2026-09-27 on TruckFull (251 views): CPU verification of 8,716 candidate pairs took 208 s. Nearly
/// every pair runs the full 1,000 iterations (far pairs are mostly chance matches, so the adaptive stop never
/// fires): ~8.7M 8-point solves, each a 9x9 Jacobi eigen-decomposition in managed doubles, ~24 us apiece.
///
/// Differences from the CPU estimator, on purpose:
/// - Every pair runs <c>hypotheses</c> samples (no adaptive stop): parallel hypotheses cost nothing extra, and the
///   best of more samples is never worse.
/// - The minimal 8-point solve is f32 and finds the 8x9 null space by elimination with complete pivoting on the
///   Hartley-normalised system, instead of the eigenvectors of A^T A (squaring the condition number in f32 is not
///   affordable). A hypothesis only has to find the inlier set; the refit that decides the answer is double.
/// - Samples come from a counter hash (pair seed, hypothesis, draw), so a hypothesis can be rebuilt exactly by the
///   pick-best kernel instead of storing 9 floats per hypothesis.
/// GpuEpipolarRansacTests gates it against the CPU estimator on Truck's real cameras and points.
/// </summary>
public sealed class GpuEpipolarRansac : IDisposable
{
    readonly Accelerator _accel;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, float,
        ArrayView<int>> _score;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int,
        ArrayView<float>, ArrayView<int>> _pick;

    /// <summary>
    /// Hard upper bound on hypothesis threads per dispatch (the adaptive sizing works under it). Each thread's
    /// 8-point solve (8x9 elimination + 3x3 Jacobi
    /// over dynamically indexed private arrays) outweighs its scoring for small pairs, so evaluations alone do not
    /// bound a dispatch's time. MEASURED 2026-09-27: an evaluation-only bound let one TruckFull dispatch run past
    /// the Windows GPU watchdog (~2 s) - device lost ("A valid external Instance reference no longer exists").
    /// </summary>
    public int MaxThreadsPerDispatch { get; set; } = 1 << 20;

    /// <summary>Batch time the adaptive sizing aims for (far under the ~2 s OS GPU watchdog).</summary>
    public double TargetBatchMs { get; set; } = 100;

    /// <summary>Work units (Sampson evaluations) of the first batch, before any speed is measured.</summary>
    public long FirstBatchUnits { get; set; } = 16_384L * 400;

    /// <summary>One hypothesis's 8-point solve, in Sampson evaluations (for batch sizing only).</summary>
    public long HypothesisCostInEvaluations { get; set; } = 200;

    /// <summary>Progress/diagnostic lines (first batches, and any batch over 3x the target).</summary>
    public Action<string>? OnBatch { get; set; }

    /// <summary>Per batch of the last <see cref="EstimateAsync"/>: pairs, hypothesis threads, Sampson evaluations, ms.</summary>
    public List<(int Pairs, long Threads, long Evaluations, double Ms)> LastBatches { get; } = new();

    public GpuEpipolarRansac(Accelerator accelerator)
    {
        _accel = accelerator;
        _score = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, int, float, ArrayView<int>>(ScoreKernel);
        _pick = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, int, ArrayView<float>, ArrayView<int>>(PickKernel);
    }

    /// <summary>One pair's matched pixel coordinates, as <see cref="EpipolarRansac.Estimate"/> takes them.</summary>
    public readonly record struct Pair(float[] A, float[] B, int Seed);

    /// <summary>
    /// Verify every pair. Result i is null when pair i has no model with <paramref name="minInliers"/> inliers,
    /// exactly as <see cref="EpipolarRansac.Estimate"/> reports it.
    /// </summary>
    public async Task<EpipolarRansac.Result?[]> EstimateAsync(IReadOnlyList<Pair> pairs, double thresholdPx = 2.0,
        int minInliers = 15, int hypotheses = 1000, double minInlierRatio = EpipolarRansac.DefaultMinInlierRatio)
    {
        var results = new EpipolarRansac.Result?[pairs.Count];
        // Pairs too small for a model never reach the device (EpipolarRansac.Estimate returns null for them too).
        var live = new List<int>();
        for (int i = 0; i < pairs.Count; i++)
        {
            int n = pairs[i].A.Length / 2;
            if (n >= 8 && n >= minInliers) live.Add(i);
        }
        if (live.Count == 0) return results;

        // Every live pair's matches, packed x1,y1,x2,y2 - uploaded once.
        var offsets = new int[live.Count];
        var counts = new int[live.Count];
        var seeds = new int[live.Count];
        long total = 0;
        for (int k = 0; k < live.Count; k++)
        {
            var p = pairs[live[k]];
            offsets[k] = (int)total;
            counts[k] = p.A.Length / 2;
            seeds[k] = p.Seed;
            total += counts[k];
        }
        var pts = new float[total * 4];
        for (int k = 0; k < live.Count; k++)
        {
            var p = pairs[live[k]];
            int o = offsets[k] * 4;
            for (int i = 0; i < counts[k]; i++)
            {
                pts[o + i * 4] = p.A[i * 2]; pts[o + i * 4 + 1] = p.A[i * 2 + 1];
                pts[o + i * 4 + 2] = p.B[i * 2]; pts[o + i * 4 + 3] = p.B[i * 2 + 1];
            }
        }

        float th2 = (float)(thresholdPx * thresholdPx);
        // CPU transfer: the matched coordinates live on the host (features + matches are host lists); uploaded once.
        using var dPts = _accel.Allocate1D(pts);
        using var dOffsets = _accel.Allocate1D(offsets);
        using var dCounts = _accel.Allocate1D(counts);
        using var dSeeds = _accel.Allocate1D(seeds);

        // Batches of consecutive pairs, sized ADAPTIVELY from the measured speed of the previous batch: a batch's
        // cost ~ Sampson evaluations + HypothesisCostInEvaluations per hypothesis thread. The first batch is small
        // (it also pays the shader compile); each later one targets TargetBatchMs. A fixed bound cannot be right on
        // every GPU - one that was fine on CUDA (1-3 ms batches) ran past the Windows watchdog in the browser.
        long hardThreads = Math.Min((long)MaxThreadsPerDispatch, (long)live.Count * hypotheses);
        using var dScores = _accel.Allocate1D<int>(hardThreads);
        using var dF = _accel.Allocate1D<float>(live.Count * 9L);
        using var dBest = _accel.Allocate1D<int>(live.Count);

        LastBatches.Clear();
        var batchWatch = new System.Diagnostics.Stopwatch();
        long budget = FirstBatchUnits;
        int next = 0, batchIndex = 0;
        while (next < live.Count)
        {
            int start = next;
            long units = 0, evals = 0;
            while (next < live.Count)
            {
                long e = (long)counts[next] * hypotheses;
                long u = e + HypothesisCostInEvaluations * hypotheses;
                bool full = units + u > budget || (long)(next - start + 1) * hypotheses > hardThreads;
                if (next > start && full) break;
                units += u; evals += e; next++;
            }
            int count = next - start;
            batchWatch.Restart();
            // Views offset to the batch: pair k of the batch is live pair start + k.
            _score((Index1D)(count * hypotheses), dPts.View,
                dOffsets.View.SubView(start, count), dCounts.View.SubView(start, count), dSeeds.View.SubView(start, count),
                hypotheses, th2, dScores.View);
            _pick((Index1D)count, dPts.View,
                dOffsets.View.SubView(start, count), dCounts.View.SubView(start, count), dSeeds.View.SubView(start, count),
                dScores.View, hypotheses, dF.View.SubView(start * 9L, count * 9L), dBest.View.SubView(start, count));
            // One submission per batch, completed before the next: a browser GPU queue runs a whole submission
            // under the OS watchdog, so batches must stay short individually, not just be many.
            await _accel.SynchronizeAsync();
            double ms = batchWatch.Elapsed.TotalMilliseconds;
            LastBatches.Add((count, (long)count * hypotheses, evals, ms));
            if (batchIndex < 3 || ms > 3 * TargetBatchMs)
                OnBatch?.Invoke($"batch {batchIndex}: {count} pairs, {(long)count * hypotheses} threads, {evals / 1e6:F1}M evaluations, {ms:F1} ms");
            // Next batch from this one's rate (ms per unit), clamped to [first batch, hard cap].
            double msPerUnit = Math.Max(ms, 0.05) / units;
            budget = (long)Math.Clamp(TargetBatchMs / msPerUnit, FirstBatchUnits, (double)hardThreads * (HypothesisCostInEvaluations + 2000));
            batchIndex++;
        }

        // CPU transfer: 9 floats + 1 int per pair (the winning hypothesis); the refit needs the host's matches.
        var f = await dF.View.CopyToHostAsync();
        var best = await dBest.View.CopyToHostAsync();
        double th2d = thresholdPx * thresholdPx;
        var fd = new double[9];
        for (int k = 0; k < live.Count; k++)
        {
            if (best[k] < 0) continue;   // no non-degenerate hypothesis
            for (int j = 0; j < 9; j++) fd[j] = f[k * 9 + j];
            var p = pairs[live[k]];
            results[live[k]] = EpipolarRansac.Finish(fd, p.A, p.B, th2d, minInliers, hypotheses, minInlierRatio);
        }
        return results;
    }

    /// <summary>Thread t = (pair, hypothesis): build hypothesis h of the pair and count its inliers (-1: degenerate).</summary>
    static void ScoreKernel(Index1D t, ArrayView<float> pts, ArrayView<int> offsets, ArrayView<int> counts,
        ArrayView<int> seeds, int hypotheses, float th2, ArrayView<int> scores)
    {
        int p = t / hypotheses;
        int h = t - p * hypotheses;
        int n = counts[p];
        int off = offsets[p];
        var a = LocalMemory.Allocate<float>(72);
        var perm = LocalMemory.Allocate<int>(9);
        var sample = LocalMemory.Allocate<int>(8);
        var f = LocalMemory.Allocate<float>(9);
        var m = LocalMemory.Allocate<float>(9);
        var v = LocalMemory.Allocate<float>(9);
        int result = -1;
        if (Hypothesis(pts, off, n, seeds[p], h, a, perm, sample, f, m, v))
        {
            result = 0;
            for (int i = 0; i < n; i++)
                if (Sampson2(f, pts, (off + i) * 4) <= th2) result++;
        }
        scores[t] = result;
    }

    /// <summary>Thread = pair: the first hypothesis with the most inliers, rebuilt and written out (best -1: none).</summary>
    static void PickKernel(Index1D p, ArrayView<float> pts, ArrayView<int> offsets, ArrayView<int> counts,
        ArrayView<int> seeds, ArrayView<int> scores, int hypotheses, ArrayView<float> outF, ArrayView<int> outBest)
    {
        int bestH = -1, best = -1;
        int b0 = p * hypotheses;
        for (int h = 0; h < hypotheses; h++)
        {
            int s = scores[b0 + h];
            if (s > best) { best = s; bestH = h; }
        }
        var a = LocalMemory.Allocate<float>(72);
        var perm = LocalMemory.Allocate<int>(9);
        var sample = LocalMemory.Allocate<int>(8);
        var f = LocalMemory.Allocate<float>(9);
        var m = LocalMemory.Allocate<float>(9);
        var v = LocalMemory.Allocate<float>(9);
        if (best < 0 || !Hypothesis(pts, offsets[p], counts[p], seeds[p], bestH, a, perm, sample, f, m, v))
            best = -1;
        for (int j = 0; j < 9; j++) outF[p * 9 + j] = best < 0 ? 0f : f[j];
        outBest[p] = best;
    }

    /// <summary>Counter-based hash (lowbias32): sample draws are a pure function of (seed, hypothesis, draw).</summary>
    static uint Hash(uint x)
    {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    /// <summary>Squared Sampson distance of the match at pts[o..o+3] to F (pixels^2), as EpipolarRansac.Sampson2.</summary>
    static float Sampson2(ArrayView<float> f, ArrayView<float> pts, int o)
    {
        float x1 = pts[o], y1 = pts[o + 1], x2 = pts[o + 2], y2 = pts[o + 3];
        float fx0 = f[0] * x1 + f[1] * y1 + f[2];
        float fx1 = f[3] * x1 + f[4] * y1 + f[5];
        float fx2 = f[6] * x1 + f[7] * y1 + f[8];
        float ftx0 = f[0] * x2 + f[3] * y2 + f[6];
        float ftx1 = f[1] * x2 + f[4] * y2 + f[7];
        float e = x2 * fx0 + y2 * fx1 + fx2;
        float d = fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1;
        return d <= 1e-30f ? float.MaxValue : e * e / d;
    }

    /// <summary>
    /// Hypothesis h of a pair: 8 distinct matches drawn from the hash, normalised 8-point, rank 2 enforced, F in
    /// pixels written to <paramref name="f"/>. False when the sample or the system is degenerate.
    /// </summary>
    static bool Hypothesis(ArrayView<float> pts, int off, int n, int seed, int h,
        ArrayView<float> a, ArrayView<int> perm, ArrayView<int> sample, ArrayView<float> f,
        ArrayView<float> m, ArrayView<float> v)
    {
        // 8 distinct indices by rejection; bounded draws.
        uint baseKey = Hash((uint)seed * 0x9E3779B9u + Hash((uint)h));
        int got = 0;
        for (int draw = 0; draw < 128 && got < 8; draw++)
        {
            int s = (int)(Hash(baseKey + (uint)draw * 0x632BE5ABu) % (uint)n);
            bool dup = false;
            for (int k = 0; k < got; k++) if (sample[k] == s) dup = true;
            if (!dup) { sample[got] = s; got++; }
        }
        if (got < 8) return false;

        // Hartley normalisation per image over the sample.
        float ca0 = 0, ca1 = 0, cb0 = 0, cb1 = 0;
        for (int k = 0; k < 8; k++)
        {
            int o = (off + sample[k]) * 4;
            ca0 += pts[o]; ca1 += pts[o + 1]; cb0 += pts[o + 2]; cb1 += pts[o + 3];
        }
        ca0 *= 0.125f; ca1 *= 0.125f; cb0 *= 0.125f; cb1 *= 0.125f;
        float da = 0, db = 0;
        for (int k = 0; k < 8; k++)
        {
            int o = (off + sample[k]) * 4;
            float ax = pts[o] - ca0, ay = pts[o + 1] - ca1, bx = pts[o + 2] - cb0, by = pts[o + 3] - cb1;
            da += XMath.Sqrt(ax * ax + ay * ay);
            db += XMath.Sqrt(bx * bx + by * by);
        }
        da *= 0.125f; db *= 0.125f;
        if (da < 1e-6f || db < 1e-6f) return false;
        float sa = 1.41421356f / da, sb = 1.41421356f / db;

        // A (8x9): row = [x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1] in normalised coordinates.
        for (int k = 0; k < 8; k++)
        {
            int o = (off + sample[k]) * 4;
            float x1 = (pts[o] - ca0) * sa, y1 = (pts[o + 1] - ca1) * sa;
            float x2 = (pts[o + 2] - cb0) * sb, y2 = (pts[o + 3] - cb1) * sb;
            int r = k * 9;
            a[r] = x2 * x1; a[r + 1] = x2 * y1; a[r + 2] = x2;
            a[r + 3] = y2 * x1; a[r + 4] = y2 * y1; a[r + 5] = y2;
            a[r + 6] = x1; a[r + 7] = y1; a[r + 8] = 1f;
        }
        for (int c = 0; c < 9; c++) perm[c] = c;

        // Null space by Gaussian elimination with complete pivoting: 8 pivots, the column left over is free.
        for (int r = 0; r < 8; r++)
        {
            int pr = r, pc = r;
            float pv = 0;
            for (int i = r; i < 8; i++)
                for (int c = r; c < 9; c++)
                {
                    float x = XMath.Abs(a[i * 9 + c]);
                    if (x > pv) { pv = x; pr = i; pc = c; }
                }
            if (pv < 1e-7f) return false;   // rank < 8: the sample does not fix F
            if (pr != r)
                for (int c = 0; c < 9; c++) { float tmp = a[r * 9 + c]; a[r * 9 + c] = a[pr * 9 + c]; a[pr * 9 + c] = tmp; }
            if (pc != r)
            {
                for (int i = 0; i < 8; i++) { float tmp = a[i * 9 + r]; a[i * 9 + r] = a[i * 9 + pc]; a[i * 9 + pc] = tmp; }
                int tp = perm[r]; perm[r] = perm[pc]; perm[pc] = tp;
            }
            float inv = 1f / a[r * 9 + r];
            for (int i = r + 1; i < 8; i++)
            {
                float fac = a[i * 9 + r] * inv;
                if (fac == 0f) continue;
                for (int c = r; c < 9; c++) a[i * 9 + c] -= fac * a[r * 9 + c];
            }
        }
        // Back substitution with the free (permuted) column 8 = 1; m holds the solution in permuted order.
        m[8] = 1f;
        for (int r = 7; r >= 0; r--)
        {
            float acc = 0;
            for (int c = r + 1; c < 9; c++) acc += a[r * 9 + c] * m[c];
            m[r] = -acc / a[r * 9 + r];
        }
        float norm = 0;
        for (int c = 0; c < 9; c++) norm += m[c] * m[c];
        norm = XMath.Rsqrt(norm);
        for (int c = 0; c < 9; c++) f[perm[c]] = m[c] * norm;   // fn, normalised F

        // Rank 2: F2 = F (I - v v^T), v = eigenvector of F^T F with the smallest eigenvalue (3x3 cyclic Jacobi).
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                m[r * 3 + c] = f[r] * f[c] + f[3 + r] * f[3 + c] + f[6 + r] * f[6 + c];
        for (int i = 0; i < 9; i++) v[i] = 0f;
        v[0] = 1f; v[4] = 1f; v[8] = 1f;
        for (int sweep = 0; sweep < 12; sweep++)
        {
            float off2 = m[1] * m[1] + m[2] * m[2] + m[5] * m[5];
            if (off2 < 1e-20f) break;
            for (int pq = 0; pq < 3; pq++)
            {
                int pp = pq == 2 ? 1 : 0;
                int qq = pq == 0 ? 1 : 2;
                float apq = m[pp * 3 + qq];
                if (XMath.Abs(apq) < 1e-30f) continue;
                float theta = (m[qq * 3 + qq] - m[pp * 3 + pp]) / (2f * apq);
                float tt = (theta >= 0 ? 1f : -1f) / (XMath.Abs(theta) + XMath.Sqrt(theta * theta + 1f));
                float cs = XMath.Rsqrt(tt * tt + 1f), sn = tt * cs;
                for (int k = 0; k < 3; k++)
                {
                    float akp = m[k * 3 + pp], akq = m[k * 3 + qq];
                    m[k * 3 + pp] = cs * akp - sn * akq; m[k * 3 + qq] = sn * akp + cs * akq;
                }
                for (int k = 0; k < 3; k++)
                {
                    float apk = m[pp * 3 + k], aqk = m[qq * 3 + k];
                    m[pp * 3 + k] = cs * apk - sn * aqk; m[qq * 3 + k] = sn * apk + cs * aqk;
                }
                for (int k = 0; k < 3; k++)
                {
                    float vkp = v[k * 3 + pp], vkq = v[k * 3 + qq];
                    v[k * 3 + pp] = cs * vkp - sn * vkq; v[k * 3 + qq] = sn * vkp + cs * vkq;
                }
            }
        }
        int z = 0;
        if (m[4] < m[0]) z = 1;
        if (m[8] < m[z * 4]) z = 2;
        float z0 = v[z], z1 = v[3 + z], z2 = v[6 + z];
        // m <- F2 = F - (F v) v^T
        for (int r = 0; r < 3; r++)
        {
            float fv = f[r * 3] * z0 + f[r * 3 + 1] * z1 + f[r * 3 + 2] * z2;
            m[r * 3] = f[r * 3] - fv * z0;
            m[r * 3 + 1] = f[r * 3 + 1] - fv * z1;
            m[r * 3 + 2] = f[r * 3 + 2] - fv * z2;
        }
        // Denormalise: F = Tb^T F2 Ta, T = [[s,0,-s cx],[0,s,-s cy],[0,0,1]].
        // F2 Ta: columns 0,1 scale by s_a; column 2 = -s_a (cx col0 + cy col1) + col2.
        for (int r = 0; r < 3; r++)
        {
            float c0 = m[r * 3], c1 = m[r * 3 + 1], c2 = m[r * 3 + 2];
            m[r * 3] = sa * c0;
            m[r * 3 + 1] = sa * c1;
            m[r * 3 + 2] = c2 - sa * (ca0 * c0 + ca1 * c1);
        }
        // Tb^T (F2 Ta): rows 0,1 scale by s_b; row 2 = -s_b (cx row0 + cy row1) + row2.
        for (int c = 0; c < 3; c++)
        {
            float r0 = m[c], r1 = m[3 + c], r2 = m[6 + c];
            f[c] = sb * r0;
            f[3 + c] = sb * r1;
            f[6 + c] = r2 - sb * (cb0 * r0 + cb1 * r1);
        }
        for (int i = 0; i < 9; i++) if (!(XMath.Abs(f[i]) < float.MaxValue)) return false;
        return true;
    }

    public void Dispose() { }
}
