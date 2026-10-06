using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The density step of "3D Gaussian Splatting as Markov Chain Monte Carlo" (Kheradmand et al. 2024), as gsplat's
/// MCMCStrategy runs it, on the device: every dead Gaussian (opacity at or under <see cref="MinOpacity"/>) is moved
/// onto a live one sampled in proportion to opacity, and the set grows by <see cref="GrowthFactor"/> toward the cap
/// by the same sampling. A Gaussian sampled c times becomes c + 1 identical copies whose opacity and scale are
/// chosen so that together they render like the original (<see cref="NewOpacity"/>, <see cref="ScaleFactor"/>) - the
/// image does not jump, unlike clone/split, and growth goes where the scene is already opaque.
/// <para>
/// One pass where gsplat runs two (relocate, then add): the dead slots and the new rows are drawn together from the
/// same distribution and a parent's ratio counts both. That is the paper's rule - N copies replace one - applied
/// once instead of twice; gsplat's second pass can resample a copy the first just made.
/// </para>
/// <para>
/// The result has <see cref="GpuDensify.Result"/>'s shape, so the trainer installs it the same way: every sampled
/// parent and every placed copy gets fresh Adam moments (adam source -1, as gsplat zeroes them), copies take their
/// parent's SH. Splats outside a partitioned block's trainable volume are frozen context: never dead, never sampled.
/// </para>
/// </summary>
public sealed class GpuMcmc
{
    const int Floats = SplatFormat.Floats;

    /// <summary>gsplat MCMCStrategy.min_opacity: at or under this a Gaussian is dead and relocated.</summary>
    public const float MinOpacity = 0.005f;
    /// <summary>Copies one Gaussian can become in one step (gsplat's n_max: its binomial table is 51 x 51).</summary>
    public const int MaxRatio = 51;
    /// <summary>Growth per step toward the cap (gsplat: n_target = min(cap_max, int(1.05 * n))).</summary>
    public const float GrowthFactor = 1.05f;

    readonly Accelerator _accel;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ClassifyParams, SplatEditor.Volume> _classify;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> _deadSlots;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, SampleParams> _sample;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>> _parents;
    readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, PlaceParams> _place;
    readonly Action<Index1D, ArrayView<float>, float, float> _reinit;
    readonly Scan<int, Stride1D.Dense, Stride1D.Dense> _scan;

    public GpuMcmc(Accelerator accelerator)
    {
        _accel = accelerator;
        _classify = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ClassifyParams, SplatEditor.Volume>(ClassifyKernel);
        _deadSlots = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(DeadSlotsKernel);
        _sample = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, SampleParams>(SampleKernel);
        _parents = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>>(ParentsKernel);
        _place = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, PlaceParams>(PlaceKernel);
        _reinit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, float>(ReinitKernel);
        _scan = accelerator.CreateScan<int, Stride1D.Dense, Stride1D.Dense, AddInt32>(ScanKind.Exclusive);
    }

    public struct ClassifyParams
    {
        public int Count;
        public float MinOpacity;
        public int WeightScale;      // opacity -> integer sampling weight (the scan is int)
        public int HasTrainable;
    }

    public struct SampleParams
    {
        public int Count;
        public int Samples;
        public uint Total;           // sum of the weights
        public uint Limit;           // largest multiple of Total that fits in 2^32: rejection keeps the draw unbiased
        public uint Seed;
    }

    public struct PlaceParams
    {
        public int Count;
        public int Samples;
        public int Dead;
    }

    /// <summary>One step's counts, for the log.</summary>
    public readonly record struct Step(int Relocated, int Added)
    {
        public override string ToString() => $"relocated {Relocated:N0} dead, added {Added:N0}";
    }

    /// <summary>
    /// Relocate the dead and grow to min(<paramref name="cap"/>, 1.05 n) when <paramref name="grow"/>. Null when
    /// there is nothing to do (no dead, no growth) or nothing alive to sample from.
    /// </summary>
    public async Task<(GpuDensify.Result Result, Step Step)?> RunAsync(ArrayView<float> packed, int n, int cap, bool grow,
        uint seed, SplatEditor.Volume? trainable = null)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        // Integer weights: opacity in 1/255ths (at least 1 for anything alive), as fine as the scan's int total allows.
        int weightScale = (int)Math.Clamp(int.MaxValue / (long)n, 1, 255);

        using var dead = _accel.Allocate1D<int>(n);
        using var weight = _accel.Allocate1D<int>(n);
        using var counters = _accel.Allocate1D<int>(1);
        counters.MemSetToZero();
        var vol = trainable ?? SplatEditor.Volume.Rows(0, 0);
        _classify((Index1D)n, packed, dead.View, weight.View, counters.View,
            new ClassifyParams { Count = n, MinOpacity = MinOpacity, WeightScale = weightScale, HasTrainable = trainable.HasValue ? 1 : 0 }, vol);

        using var deadDst = _accel.Allocate1D<int>(n);
        using var cdf = _accel.Allocate1D<int>(n);
        using var scanTemp = _accel.Allocate1D<int>(Math.Max(1, _accel.ComputeScanTempStorageSize<int>(n)));
        _scan(_accel.DefaultStream, dead.View, deadDst.View, scanTemp.View);
        _scan(_accel.DefaultStream, weight.View, cdf.View, scanTemp.View);

        // CPU transfer: the dead count and the weight total (last exclusive prefix + last weight), 12 bytes.
        int deadCount = (await counters.CopyToHostAsync<int>(0, 1))[0];
        long total = (long)(await cdf.CopyToHostAsync<int>(n - 1, 1))[0] + (await weight.CopyToHostAsync<int>(n - 1, 1))[0];
        int added = grow ? Math.Max(0, Math.Min(cap, (int)(GrowthFactor * n)) - n) : 0;
        int samples = deadCount + added;
        if (samples == 0 || total <= 0) return null;

        using var deadSlots = _accel.Allocate1D<int>(Math.Max(1, deadCount));
        _deadSlots((Index1D)n, dead.View, deadDst.View, deadSlots.View);

        using var parent = _accel.Allocate1D<int>(samples);
        using var count = _accel.Allocate1D<int>(n);
        count.MemSetToZero();
        uint t = (uint)total;
        _sample((Index1D)samples, cdf.View, weight.View, parent.View, count.View, new SampleParams
        {
            Count = n, Samples = samples, Total = t, Limit = (uint)(4294967296UL / t * t - 1), Seed = seed,
        });

        int m = n + added;
        var outPacked = _accel.Allocate1D<float>((long)m * Floats);
        var adamSrc = _accel.Allocate1D<int>(m);
        var featSrc = _accel.Allocate1D<int>(m);
        _parents((Index1D)n, packed, count.View, outPacked.View, adamSrc.View, featSrc.View);
        _place((Index1D)samples, packed, count.View, parent.View, deadSlots.View, outPacked.View, adamSrc.View, featSrc.View,
            new PlaceParams { Count = n, Samples = samples, Dead = deadCount });
        await _accel.SynchronizeAsync();

        // Shaped as a densify result: the dead count as prunes, every placed copy as a clone, so net = growth.
        var result = new GpuDensify.Result(outPacked, m, adamSrc, featSrc, 0, samples, 0, deadCount, 0, 0);
        return (result, new Step(deadCount, added));
    }

    /// <summary>
    /// MCMC's starting point (gsplat MCMC config: init_opa 0.5, init_scale 0.1): every opacity set to
    /// <paramref name="opacity"/>, every scale multiplied by <paramref name="scaleMul"/>.
    /// </summary>
    public void Reinitialise(ArrayView<float> packed, int n, float opacity, float scaleMul)
        => _reinit((Index1D)n, packed, opacity, scaleMul);

    // ---------------------------------------------------------------- the relocation rule (host + kernels)

    /// <summary>Opacity of each of <paramref name="ratio"/> copies so that, stacked, they are as opaque as one at <paramref name="opacity"/>.</summary>
    public static float NewOpacity(float opacity, int ratio) => 1f - XMath.Pow(1f - opacity, 1f / ratio);

    /// <summary>
    /// The paper's scale correction (gsplat relocation_kernel): <c>opacity / sum_{i=1..N} sum_{k=0..i-1}
    /// C(i-1,k) (-1)^k newOpacity^(k+1) / sqrt(k+1)</c>. N = 1 gives 1. The binomials are built along each row
    /// rather than read from a table.
    /// </summary>
    public static float ScaleFactor(float opacity, float newOpacity, int ratio)
    {
        float denom = 0f;
        for (int i = 1; i <= ratio; i++)
        {
            float binom = 1f, sign = 1f, power = newOpacity;
            for (int k = 0; k <= i - 1; k++)
            {
                denom += binom * sign * power / XMath.Sqrt(k + 1f);
                binom = binom * (i - 1 - k) / (k + 1f);
                sign = -sign;
                power *= newOpacity;
            }
        }
        return opacity / denom;
    }

    /// <summary>Clamp gsplat applies to the new opacity (after the scale is computed from the unclamped one).</summary>
    public static float ClampOpacity(float o) => XMath.Clamp(o, MinOpacity, 1f - 1.1920929e-7f);

    // ---------------------------------------------------------------- kernels

    internal static uint Hash(uint x)
    {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    static void ClassifyKernel(Index1D i, ArrayView<float> packed, ArrayView<int> dead, ArrayView<int> weight,
        ArrayView<int> counters, ClassifyParams p, SplatEditor.Volume trainable)
    {
        if (i >= p.Count) return;
        long o = (long)i.X * Floats;
        bool frozen = p.HasTrainable != 0 && !SplatEditor.Inside(trainable, packed[o], packed[o + 1], packed[o + 2]);
        float a = packed[o + 9];
        bool isDead = !frozen && a <= p.MinOpacity;
        dead[i] = isDead ? 1 : 0;
        int w = (int)(a * p.WeightScale);
        weight[i] = frozen || isDead ? 0 : (w < 1 ? 1 : w);
        if (isDead) Atomic.Add(ref counters[0], 1);
    }

    static void DeadSlotsKernel(Index1D i, ArrayView<int> dead, ArrayView<int> deadDst, ArrayView<int> deadSlots)
    {
        if (i >= dead.Length) return;
        if (dead[i] != 0) deadSlots[deadDst[i]] = i;
    }

    /// <summary>
    /// Draw sample k: an unbiased integer below the weight total (rejection above the last whole multiple), then the
    /// last index whose exclusive prefix is at or under it. A zero-weight index never wins: the next one has the same
    /// prefix, and the last one's prefix is the total.
    /// </summary>
    static void SampleKernel(Index1D k, ArrayView<int> cdf, ArrayView<int> weight, ArrayView<int> parent,
        ArrayView<int> count, SampleParams p)
    {
        if (k >= p.Samples) return;
        uint h = 0;
        for (uint attempt = 0; attempt < 32; attempt++)
        {
            h = Hash(p.Seed ^ Hash((uint)k.X * 32u + attempt));
            if (h <= p.Limit) break;
        }
        int u = (int)(h % p.Total);
        int lo = 0, hi = p.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (cdf[mid] <= u) lo = mid; else hi = mid - 1;
        }
        parent[k] = lo;
        Atomic.Add(ref count[lo], 1);
    }

    /// <summary>Every row as it was, except a sampled parent: its share of the opacity and the corrected scale, fresh Adam.</summary>
    static void ParentsKernel(Index1D i, ArrayView<float> packed, ArrayView<int> count, ArrayView<float> outPacked,
        ArrayView<int> adamSrc, ArrayView<int> featSrc)
    {
        if (i >= count.Length) return;
        long o = (long)i.X * Floats;
        for (int c = 0; c < Floats; c++) outPacked[o + c] = packed[o + c];
        featSrc[i] = i;
        int copies = count[i];
        if (copies == 0) { adamSrc[i] = i; return; }
        WriteShare(packed, o, outPacked, o, copies);
        adamSrc[i] = -1;
    }

    /// <summary>Sample k lands in the k-th dead slot, or past the old end once those are used: the parent's share.</summary>
    static void PlaceKernel(Index1D k, ArrayView<float> packed, ArrayView<int> count, ArrayView<int> parent,
        ArrayView<int> deadSlots, ArrayView<float> outPacked, ArrayView<int> adamSrc, ArrayView<int> featSrc, PlaceParams p)
    {
        if (k >= p.Samples) return;
        int src = parent[k];
        int dst = k < p.Dead ? deadSlots[k] : p.Count + (k - p.Dead);
        long so = (long)src * Floats, d = (long)dst * Floats;
        for (int c = 0; c < Floats; c++) outPacked[d + c] = packed[so + c];
        WriteShare(packed, so, outPacked, d, count[src]);
        adamSrc[dst] = -1;
        featSrc[dst] = src;
    }

    static void WriteShare(ArrayView<float> packed, long so, ArrayView<float> outPacked, long d, int copies)
    {
        int ratio = copies + 1;
        if (ratio > MaxRatio) ratio = MaxRatio;
        float a = packed[so + 9];
        float na = NewOpacity(a, ratio);
        float f = ScaleFactor(a, na, ratio);
        outPacked[d + 9] = ClampOpacity(na);
        outPacked[d + 6] = packed[so + 6] * f;
        outPacked[d + 7] = packed[so + 7] * f;
        outPacked[d + 8] = packed[so + 8] * f;
    }

    static void ReinitKernel(Index1D i, ArrayView<float> packed, float opacity, float scaleMul)
    {
        long o = (long)i.X * Floats;
        if (o + Floats > packed.Length) return;
        packed[o + 9] = opacity;
        packed[o + 6] *= scaleMul;
        packed[o + 7] *= scaleMul;
        packed[o + 8] *= scaleMul;
    }
}
