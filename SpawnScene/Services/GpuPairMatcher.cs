using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// All-pairs binary-descriptor matching in batches: every image's descriptors uploaded ONCE, then hundreds of
/// pairs per dispatch, with the ratio test and the cross-check done on the device.
///
/// <see cref="GpuFeatureMatcher.MatchAsync"/> matches one pair per call: repack both descriptor sets on the CPU,
/// upload them, two kernels, two waits, four readbacks. MEASURED on Truck (126 images, 7,875 pairs): 212.5 s,
/// ~27 ms a pair, for arithmetic worth well under a millisecond. The rules here are exactly that matcher's -
/// same scan order and strict tie-breaking (the first minimum wins), same ratio test, same distance cap, same
/// cross-check - so the matches are identical (GpuPairMatcherTests).
/// </summary>
public sealed class GpuPairMatcher : IDisposable
{
    const int IntsPerDesc = 8;

    readonly Accelerator _accel;
    readonly float _ratio;
    readonly int _maxDistance;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _best;
    readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int,
        float, ArrayView<int>, ArrayView<int>> _filter;

    public GpuPairMatcher(Accelerator accelerator, float ratioThreshold = 0.75f, int maxDistance = 64)
    {
        _accel = accelerator;
        _ratio = ratioThreshold;
        _maxDistance = maxDistance;
        _best = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, int, int, ArrayView<int>, ArrayView<int>, ArrayView<int>>(BestKernel);
        _filter = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, int, int, float, ArrayView<int>, ArrayView<int>>(FilterKernel);
    }

    /// <summary>
    /// Match every listed pair. <paramref name="onPair"/> gets (pair index, matches) for each pair, in order.
    /// </summary>
    public async Task MatchPairsAsync(
        IReadOnlyList<IReadOnlyList<ImageFeature>> images, IReadOnlyList<(int A, int B)> pairs,
        Action<int, List<FeatureMatch>> onPair, Func<int, Task>? onBatch = null)
    {
        if (pairs.Count == 0) return;
        int nImg = images.Count;
        var offsets = new int[nImg];
        var counts = new int[nImg];
        int total = 0, maxF = 0;
        for (int i = 0; i < nImg; i++)
        {
            offsets[i] = total;
            counts[i] = images[i].Count;
            total += counts[i];
            maxF = Math.Max(maxF, counts[i]);
        }
        if (maxF == 0) { for (int p = 0; p < pairs.Count; p++) onPair(p, new List<FeatureMatch>()); return; }

        var desc = new int[Math.Max(1, total) * IntsPerDesc];
        for (int i = 0; i < nImg; i++)
            for (int f = 0; f < counts[i]; f++)
            {
                var d = images[i][f].Descriptor;
                int b = (offsets[i] + f) * IntsPerDesc;
                for (int w = 0; w < IntsPerDesc; w++)
                    desc[b + w] = d[w * 4] | (d[w * 4 + 1] << 8) | (d[w * 4 + 2] << 16) | (d[w * 4 + 3] << 24);
            }

        using var dDesc = _accel.Allocate1D(desc);
        using var dOffsets = _accel.Allocate1D(offsets);
        using var dCounts = _accel.Allocate1D(counts);

        // Pairs per dispatch: ~4M query threads (both directions) - one dispatch's worth, bounded memory.
        int batch = Math.Max(1, Math.Min(pairs.Count, (4 << 20) / (2 * maxF)));
        using var dPairs = _accel.Allocate1D<int>(batch * 2);
        using var dBest = _accel.Allocate1D<int>((long)batch * 2 * maxF);
        using var dDist = _accel.Allocate1D<int>((long)batch * 2 * maxF);
        using var dSecond = _accel.Allocate1D<int>((long)batch * 2 * maxF);
        using var dOutIdx = _accel.Allocate1D<int>((long)batch * maxF);
        using var dOutDist = _accel.Allocate1D<int>((long)batch * maxF);
        var pairBuf = new int[batch * 2];

        for (int start = 0; start < pairs.Count; start += batch)
        {
            int n = Math.Min(batch, pairs.Count - start);
            for (int k = 0; k < n; k++) { pairBuf[k * 2] = pairs[start + k].A; pairBuf[k * 2 + 1] = pairs[start + k].B; }
            dPairs.View.SubView(0, n * 2).CopyFromCPU(pairBuf.AsSpan(0, n * 2).ToArray());

            _best((Index1D)(n * 2 * maxF), dDesc.View, dOffsets.View, dCounts.View, dPairs.View, maxF, IntsPerDesc,
                dBest.View, dDist.View, dSecond.View);
            _filter((Index1D)(n * maxF), dBest.View, dDist.View, dSecond.View, dCounts.View, dPairs.View, maxF,
                _maxDistance, _ratio, dOutIdx.View, dOutDist.View);

            // CPU transfer: one int (matched index or -1) and its distance per query feature of the batch.
            var outIdx = await dOutIdx.View.SubView(0, (long)n * maxF).CopyToHostAsync();
            var outDist = await dOutDist.View.SubView(0, (long)n * maxF).CopyToHostAsync();
            for (int k = 0; k < n; k++)
            {
                int ca = counts[pairs[start + k].A];
                var list = new List<FeatureMatch>();
                int b0 = k * maxF;
                for (int a = 0; a < ca; a++)
                {
                    int j = outIdx[b0 + a];
                    if (j >= 0) list.Add(new FeatureMatch { IndexA = a, IndexB = j, Distance = outDist[b0 + a] });
                }
                onPair(start + k, list);
            }
            if (onBatch != null) await onBatch(start + n);
        }
    }

    /// <summary>
    /// Thread t = (pair, direction, query feature). Direction 0: a feature of A against all of B; 1: B against A.
    /// Identical to GpuFeatureMatcher.HammingMatchKernel: ascending scan, strict less-than, first minimum wins.
    /// </summary>
    static void BestKernel(Index1D t, ArrayView<int> desc, ArrayView<int> offsets, ArrayView<int> counts,
        ArrayView<int> pairs, int maxF, int ints, ArrayView<int> bestIdx, ArrayView<int> bestDist, ArrayView<int> second)
    {
        int per = 2 * maxF;
        int p = t / per;
        int rem = t - p * per;
        int dir = rem / maxF;
        int f = rem - dir * maxF;
        int imgQ = dir == 0 ? pairs[p * 2] : pairs[p * 2 + 1];
        int imgC = dir == 0 ? pairs[p * 2 + 1] : pairs[p * 2];
        int best = 999999, sec = 999999, bestJ = -1;
        if (f < counts[imgQ])
        {
            int qo = (offsets[imgQ] + f) * ints;
            int co = offsets[imgC] * ints;
            int nc = counts[imgC];
            for (int j = 0; j < nc; j++)
            {
                int bo = co + j * ints;
                int dist = 0;
                for (int k = 0; k < ints; k++)
                    dist += IntrinsicMath.BitOperations.PopCount(desc[qo + k] ^ desc[bo + k]);
                if (dist < best) { sec = best; best = dist; bestJ = j; }
                else if (dist < sec) sec = dist;
            }
        }
        bestIdx[t] = bestJ;
        bestDist[t] = best;
        second[t] = sec;
    }

    /// <summary>
    /// Thread t = (pair, feature a of A). GpuFeatureMatcher's filter: distance cap, Lowe ratio against the second
    /// best, then the cross-check - B's best match for a's best must be a.
    /// </summary>
    static void FilterKernel(Index1D t, ArrayView<int> bestIdx, ArrayView<int> bestDist, ArrayView<int> second,
        ArrayView<int> counts, ArrayView<int> pairs, int maxF, int maxDistance, float ratio,
        ArrayView<int> outIdx, ArrayView<int> outDist)
    {
        int p = t / maxF;
        int a = t - p * maxF;
        int per = 2 * maxF;
        int fwd = p * per + a;
        int j = bestIdx[fwd];
        int result = -1;
        if (a < counts[pairs[p * 2]] && j >= 0 && bestDist[fwd] < maxDistance
            && second[fwd] > 0 && bestDist[fwd] < ratio * second[fwd])
        {
            int rev = p * per + maxF + j;
            if (bestIdx[rev] == a) result = j;
        }
        outIdx[t] = result;
        outDist[t] = bestDist[fwd];
    }

    public void Dispose() { }
}
