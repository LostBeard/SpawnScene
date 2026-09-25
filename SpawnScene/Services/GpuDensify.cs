using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Density control (clone / split / prune) on the device: the decision of
/// <see cref="SplatDensityControl.Decide"/> and the rebuild of <see cref="SplatDensityControl.Apply"/>, without
/// the scene ever crossing to the host.
///
/// The host path read the whole scene back every densify (every 100 iterations), unpacked it into structs,
/// built the grown list and repacked it - four full copies in the wasm heap at once. MEASURED 2026-09-24: it
/// threw OutOfMemoryException at ~2.0M splats on Truck, and a 1.2M cap was still binding (growth stopped only
/// at the cap). Here the only transfers are a few counters and a 2 KiB gradient histogram.
///
/// Differences from the host oracle, both deliberate and both bounded:
/// <list type="bullet">
/// <item>When candidates exceed the budget the host takes them highest-gradient-first. This takes whole
/// histogram bins from the top (log2 bins, 1/64 octave = 1.1% wide) and, inside the one bin that straddles
/// the budget, the lowest indices first. So a selected candidate never has a gradient more than one bin below
/// an unselected one.</item>
/// <item>New splats are appended in index order, not gradient order, and split children are drawn from a
/// hash RNG. Where a child lands is a sample from the parent's ellipsoid either way.</item>
/// </list>
/// </summary>
public sealed class GpuDensify : IDisposable
{
    const int Floats = SplatFormat.Floats;
    const int Bins = 2048;
    const float BinsPerOctave = 64f;

    // Action codes written by Classify.
    const int Keep = 0, PruneFaint = 1, PruneBig = 2, CloneCandidate = 3, SplitCandidate = 4;
    // Counter slots (atomic): one per action code, plus visible.
    const int CounterSlots = 6, VisibleSlot = 5;

    readonly Accelerator _accel;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, ArrayView<int>, ClassifyParams> _classify;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _boundary;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, SelectParams> _select;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<float>, ArrayView<int>, ArrayView<int>, CompactParams> _compact;
    readonly Scan<int, Stride1D.Dense, Stride1D.Dense> _scan;

    public GpuDensify(Accelerator accelerator)
    {
        _accel = accelerator;
        _classify = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ClassifyParams>(ClassifyKernel);
        _boundary = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(
            BoundaryKernel);
        _select = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, SelectParams>(SelectKernel);
        _compact = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, CompactParams>(CompactKernel);
        _scan = accelerator.CreateScan<int, Stride1D.Dense, Stride1D.Dense, AddInt32>(ScanKind.Exclusive);
    }

    /// <summary>What to decide with. Mirrors the arguments and statics of <see cref="SplatDensityControl.Decide"/>.</summary>
    public readonly record struct Options(
        float SceneExtent,
        bool AfterFirstOpacityReset,
        int MaxSplats,
        bool ResetOpacity,
        uint Seed,
        bool NoOp = false);

    /// <summary>
    /// The grown set. <see cref="Packed"/>, <see cref="AdamSources"/> and <see cref="FeatureSources"/> are owned
    /// by the caller. The source maps are the ones <see cref="SplatDensityControl.Apply"/> returns: the OLD index
    /// whose Adam moments / SH features a NEW splat keeps, or -1.
    /// </summary>
    public sealed record Result(
        MemoryBuffer1D<float, Stride1D.Dense> Packed, int Count,
        MemoryBuffer1D<int, Stride1D.Dense> AdamSources, MemoryBuffer1D<int, Stride1D.Dense> FeatureSources,
        int Visible, int Cloned, int Split, int PrunedFaint, int PrunedBig, int Candidates)
    {
        public int Removed => PrunedFaint + PrunedBig + Split;
        public int Added => Cloned + 2 * Split;

        public override string ToString() =>
            $"clone {Cloned}, split {Split}, prune {PrunedFaint} faint + {PrunedBig} bloated, net {Added - Removed:+#;-#;0}" +
            (Candidates > Cloned + Split ? $" ({Candidates - Cloned - Split} candidates over budget)" : "");
    }

    public struct ClassifyParams
    {
        public int Count;
        public float MinOpacity, MaxWorldSize, MaxScreenRadiusPx, GradientThreshold, SizeSplit;
        public int AfterReset, NoOp;
    }

    public struct SelectParams
    {
        public int Count;
        public int CutoffBin;      // -1: every candidate is selected
        public int Remaining;      // growth allowed inside the cutoff bin
    }

    public struct CompactParams
    {
        public int Count;
        public int KeptTotal;
        public int ResetOpacity;
        public float OpacityResetTo;
        public uint Seed;
        public float SplitScaleDivisor;
    }

    /// <summary>
    /// Decide and rebuild. <paramref name="densifyStats"/> is the trainer's 2-per-splat accumulator
    /// (pixel-gradient sum, visible count) and <paramref name="maxRadius"/> its per-splat max screen radius.
    /// </summary>
    public async Task<Result> RunAsync(
        ArrayView<float> packed, int n, ArrayView<float> densifyStats, ArrayView<float> maxRadius, Options o)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        var stream = _accel.DefaultStream;

        using var action = _accel.Allocate1D<int>(n);
        using var bin = _accel.Allocate1D<int>(n);
        using var hist = _accel.Allocate1D<int>(Bins);
        using var counters = _accel.Allocate1D<int>(CounterSlots);
        hist.MemSetToZero();
        counters.MemSetToZero();

        var cp = new ClassifyParams
        {
            Count = n,
            MinOpacity = SplatDensityControl.MinOpacity,
            MaxWorldSize = SplatDensityControl.MaxWorldSizeFraction * o.SceneExtent,
            MaxScreenRadiusPx = float.IsPositiveInfinity(SplatDensityControl.MaxScreenRadiusPx)
                ? float.MaxValue : SplatDensityControl.MaxScreenRadiusPx,
            GradientThreshold = SplatDensityControl.GradientThreshold,
            SizeSplit = SplatDensityControl.PercentDense * o.SceneExtent,
            AfterReset = o.AfterFirstOpacityReset ? 1 : 0,
            NoOp = o.NoOp ? 1 : 0,
        };
        _classify((Index1D)n, packed, densifyStats, maxRadius, action.View, bin.View, hist.View, counters.View, cp);

        // CPU transfer: 6 counters + the 2048-bin growth histogram (8 KiB). Everything per-splat stays put.
        int[] c = await counters.CopyToHostAsync<int>(0, CounterSlots);
        int[] h = await hist.CopyToHostAsync<int>(0, Bins);

        int candidates = c[CloneCandidate] + c[SplitCandidate];
        long wanted = 0;
        foreach (int v in h) wanted += v;
        int budget = Math.Max(0, o.MaxSplats - n);
        float frac = Math.Clamp(SplatDensityControl.GrowthSelectFraction, 0f, 1f);
        long allowed = Math.Min(budget, frac >= 0.999f ? wanted : (long)MathF.Ceiling(wanted * frac));

        int cutoff = -1, remaining = 0;
        if (wanted > allowed)
        {
            // Whole bins from the top while they fit; the first that does not is the cutoff.
            long taken = 0;
            cutoff = 0;
            for (int b = Bins - 1; b >= 0; b--)
            {
                if (taken + h[b] > allowed) { cutoff = b; remaining = (int)(allowed - taken); break; }
                taken += h[b];
            }
        }

        // Inside the cutoff bin, lowest index first: an exclusive scan of each candidate's growth there.
        using var partial = _accel.Allocate1D<int>(n);
        using var prefix = _accel.Allocate1D<int>(n);
        using var scanTemp = _accel.Allocate1D<int>(Math.Max(1, _accel.ComputeScanTempStorageSize<int>(n)));
        _boundary((Index1D)n, action.View, bin.View, partial.View, cutoff);
        _scan(stream, partial.View, prefix.View, scanTemp.View);

        using var keep = _accel.Allocate1D<int>(n);
        using var add = _accel.Allocate1D<int>(n);
        _select((Index1D)n, action.View, bin.View, prefix.View, partial.View, keep.View, add.View,
            new SelectParams { Count = n, CutoffBin = cutoff, Remaining = remaining });

        using var keepDst = _accel.Allocate1D<int>(n);
        using var addDst = _accel.Allocate1D<int>(n);
        _scan(stream, keep.View, keepDst.View, scanTemp.View);
        _scan(stream, add.View, addDst.View, scanTemp.View);

        // CPU transfer: the two totals (last exclusive prefix + last element), 16 bytes.
        int[] kd = await keepDst.CopyToHostAsync<int>(n - 1, 1);
        int[] kl = await keep.CopyToHostAsync<int>(n - 1, 1);
        int[] ad = await addDst.CopyToHostAsync<int>(n - 1, 1);
        int[] al = await add.CopyToHostAsync<int>(n - 1, 1);
        int kept = kd[0] + kl[0], added = ad[0] + al[0];
        int m = kept + added;

        var outPacked = _accel.Allocate1D<float>((long)Math.Max(1, m) * Floats);
        var adamSrc = _accel.Allocate1D<int>(Math.Max(1, m));
        var featSrc = _accel.Allocate1D<int>(Math.Max(1, m));
        _compact((Index1D)n, packed, keep.View, add.View, keepDst.View, addDst.View,
            outPacked.View, adamSrc.View, featSrc.View,
            new CompactParams
            {
                Count = n,
                KeptTotal = kept,
                ResetOpacity = o.ResetOpacity ? 1 : 0,
                OpacityResetTo = SplatDensityControl.OpacityResetTo,
                Seed = o.Seed,
                SplitScaleDivisor = SplatDensityControl.SplitScaleDivisor,
            });

        // Clones and splits from the totals: added = clones + 2 splits, and every split parent left the kept set.
        int prunedFaint = c[PruneFaint], prunedBig = c[PruneBig];
        int splitCount = n - prunedFaint - prunedBig - kept;
        int cloneCount = added - 2 * splitCount;
        await _accel.SynchronizeAsync();

        return new Result(outPacked, m, adamSrc, featSrc,
            c[VisibleSlot], cloneCount, splitCount, prunedFaint, prunedBig, candidates);
    }

    // ---------------------------------------------------------------- kernels

    static int GradientBin(float avg, float threshold)
    {
        float octaves = XMath.Log2(avg / threshold);
        int b = (int)(octaves * BinsPerOctave);
        return b < 0 ? 0 : (b >= Bins ? Bins - 1 : b);
    }

    static void ClassifyKernel(Index1D i, ArrayView<float> packed, ArrayView<float> stats, ArrayView<float> maxRadius,
        ArrayView<int> action, ArrayView<int> bin, ArrayView<int> hist, ArrayView<int> counters, ClassifyParams p)
    {
        if (i >= p.Count) return;
        long o = (long)i.X * Floats;
        float opacity = packed[o + 9];
        float s0 = packed[o + 6], s1 = packed[o + 7], s2 = packed[o + 8];
        float maxScale = XMath.Max(s0, XMath.Max(s1, s2));
        float gradSum = stats[2 * i];
        float vis = stats[2 * i + 1];
        float avg = vis > 0f ? gradSum / vis : 0f;
        if (vis > 0f) Atomic.Add(ref counters[VisibleSlot], 1);

        int a = Keep;
        int b = 0;
        if (p.NoOp == 0)
        {
            // Same order as the host: prune first, a splat being removed is never densified.
            if (opacity < p.MinOpacity) a = PruneFaint;
            else if (p.AfterReset != 0 && (maxScale > p.MaxWorldSize || maxRadius[i] > p.MaxScreenRadiusPx)) a = PruneBig;
            else if (avg >= p.GradientThreshold)
            {
                a = maxScale > p.SizeSplit ? SplitCandidate : CloneCandidate;
                b = GradientBin(avg, p.GradientThreshold);
                Atomic.Add(ref hist[b], a == SplitCandidate ? 2 : 1);
            }
        }
        action[i] = a;
        bin[i] = b;
        Atomic.Add(ref counters[a], 1);
    }

    static void BoundaryKernel(Index1D i, ArrayView<int> action, ArrayView<int> bin, ArrayView<int> partial, int cutoff)
    {
        int a = action[i];
        bool candidate = a == CloneCandidate || a == SplitCandidate;
        partial[i] = candidate && bin[i] == cutoff ? (a == SplitCandidate ? 2 : 1) : 0;
    }

    static void SelectKernel(Index1D i, ArrayView<int> action, ArrayView<int> bin, ArrayView<int> prefix,
        ArrayView<int> partial, ArrayView<int> keep, ArrayView<int> add, SelectParams p)
    {
        int a = action[i];
        if (a == PruneFaint || a == PruneBig) { keep[i] = 0; add[i] = 0; return; }
        if (a == Keep) { keep[i] = 1; add[i] = 0; return; }

        bool selected = p.CutoffBin < 0 || bin[i] > p.CutoffBin
            || (bin[i] == p.CutoffBin && prefix[i] + partial[i] <= p.Remaining);
        if (!selected) { keep[i] = 1; add[i] = 0; return; }
        if (a == CloneCandidate) { keep[i] = 1; add[i] = 1; }
        else { keep[i] = 0; add[i] = 2; }
    }

    static uint Hash(uint x)
    {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    /// <summary>A standard normal deviate from (seed, splat, slot), Box-Muller.</summary>
    static float Normal(uint seed, int i, int slot)
    {
        uint h1 = Hash(seed ^ Hash((uint)i * 16u + (uint)slot * 2u));
        uint h2 = Hash(seed ^ Hash((uint)i * 16u + (uint)slot * 2u + 1u) ^ 0x9e3779b9u);
        float u1 = (h1 + 1f) * (1f / 4294967296f);   // (0, 1]
        float u2 = h2 * (1f / 4294967296f);
        return XMath.Sqrt(-2f * XMath.Log(u1)) * XMath.Cos(6.28318530718f * u2);
    }

    static void CopyRow(ArrayView<float> src, long so, ArrayView<float> dst, long d0, int reset, float resetTo)
    {
        for (int k = 0; k < Floats; k++) dst[d0 + k] = src[so + k];
        if (reset != 0) dst[d0 + 9] = XMath.Min(dst[d0 + 9], resetTo);
    }

    static void CompactKernel(Index1D i, ArrayView<float> packed, ArrayView<int> keep, ArrayView<int> add,
        ArrayView<int> keepDst, ArrayView<int> addDst, ArrayView<float> outPacked,
        ArrayView<int> adamSrc, ArrayView<int> featSrc, CompactParams p)
    {
        if (i >= p.Count) return;
        long so = (long)i.X * Floats;
        if (keep[i] != 0)
        {
            int d = keepDst[i];
            CopyRow(packed, so, outPacked, (long)d * Floats, p.ResetOpacity, p.OpacityResetTo);
            adamSrc[d] = i;
            featSrc[d] = i;
        }
        int na = add[i];
        if (na == 0) return;
        int first = p.KeptTotal + addDst[i];
        if (na == 1)
        {
            // Clone: an exact copy; the optimiser separates them.
            CopyRow(packed, so, outPacked, (long)first * Floats, p.ResetOpacity, p.OpacityResetTo);
            adamSrc[first] = -1;
            featSrc[first] = i;
            return;
        }

        // Split: two children drawn from the parent's own ellipsoid, shrunk (SplatDensityControl.Child).
        float qx = packed[so + 10], qy = packed[so + 11], qz = packed[so + 12], qw = packed[so + 13];
        float len = XMath.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (len > 1e-20f) { qx /= len; qy /= len; qz /= len; qw /= len; }
        else { qx = 0f; qy = 0f; qz = 0f; qw = 1f; }
        float xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz;
        float wx = qw * qx, wy = qw * qy, wz = qw * qz;
        float r00 = 1f - 2f * (yy + zz), r01 = 2f * (xy - wz), r02 = 2f * (xz + wy);
        float r10 = 2f * (xy + wz), r11 = 1f - 2f * (xx + zz), r12 = 2f * (yz - wx);
        float r20 = 2f * (xz - wy), r21 = 2f * (yz + wx), r22 = 1f - 2f * (xx + yy);
        float sx = packed[so + 6], sy = packed[so + 7], sz = packed[so + 8];

        for (int c = 0; c < 2; c++)
        {
            int d = first + c;
            long dO = (long)d * Floats;
            CopyRow(packed, so, outPacked, dO, p.ResetOpacity, p.OpacityResetTo);
            float lx = Normal(p.Seed, i, c * 3 + 0) * sx;
            float ly = Normal(p.Seed, i, c * 3 + 1) * sy;
            float lz = Normal(p.Seed, i, c * 3 + 2) * sz;
            outPacked[dO + 0] = packed[so + 0] + r00 * lx + r01 * ly + r02 * lz;
            outPacked[dO + 1] = packed[so + 1] + r10 * lx + r11 * ly + r12 * lz;
            outPacked[dO + 2] = packed[so + 2] + r20 * lx + r21 * ly + r22 * lz;
            outPacked[dO + 6] = sx / p.SplitScaleDivisor;
            outPacked[dO + 7] = sy / p.SplitScaleDivisor;
            outPacked[dO + 8] = sz / p.SplitScaleDivisor;
            adamSrc[d] = -1;
            featSrc[d] = i;
        }
    }

    public void Dispose() { }
}
