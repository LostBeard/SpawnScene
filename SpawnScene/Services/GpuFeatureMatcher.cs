using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// GPU-accelerated feature matching using SpawnDev.ILGPU.
/// Uses a compute kernel to perform brute-force Hamming distance
/// matching on the GPU, achieving massive parallelism for the O(n×m)
/// comparison between descriptor sets.
/// SpawnDev.ILGPU always provides a backend (WebGPU/WebGL/Wasm).
/// </summary>
public class GpuFeatureMatcher
{
    private readonly GpuService _gpu;
    private readonly float _ratioThreshold;
    private readonly int _maxDistance;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, int, int, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _matchKernel;
    private bool _kernelLoaded;

    public GpuFeatureMatcher(GpuService gpu, float ratioThreshold = 0.75f, int maxDistance = 64)
    {
        _gpu = gpu;
        _ratioThreshold = ratioThreshold;
        _maxDistance = maxDistance;
    }

    /// <summary>
    /// Match features between two images using GPU-accelerated Hamming distance.
    /// Each GPU thread handles one feature from image A, comparing it against
    /// all features from image B to find the best and second-best match.
    /// </summary>
    public async Task<List<FeatureMatch>> MatchAsync(List<ImageFeature> featuresA, List<ImageFeature> featuresB)
    {
        if (featuresA.Count == 0 || featuresB.Count == 0)
            return [];

        return await MatchGpuAsync(featuresA, featuresB);
    }

    /// <summary>
    /// GPU-accelerated matching implementation.
    /// </summary>
    private async Task<List<FeatureMatch>> MatchGpuAsync(List<ImageFeature> featuresA, List<ImageFeature> featuresB)
    {
        var accelerator = _gpu.Accelerator;
        int countA = featuresA.Count;
        int countB = featuresB.Count;

        // Pack descriptors as int arrays (32 bytes = 8 ints per descriptor)
        const int intsPerDesc = 8;
        var descDataA = PackDescriptors(featuresA);
        var descDataB = PackDescriptors(featuresB);

        // Allocate GPU buffers
        using var bufDescA = accelerator.Allocate1D(descDataA);
        using var bufDescB = accelerator.Allocate1D(descDataB);
        using var bufBestIdx = accelerator.Allocate1D<int>(countA);
        using var bufBestDist = accelerator.Allocate1D<int>(countA);
        using var bufSecondDist = accelerator.Allocate1D<int>(countA);

        // Load kernel once
        if (!_kernelLoaded)
        {
            _matchKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, int, int,
                ArrayView<int>, ArrayView<int>, ArrayView<int>>(HammingMatchKernel);
            _kernelLoaded = true;
        }

        // Launch: one thread per feature in A
        _matchKernel!((Index1D)countA,
            bufDescA.View, bufDescB.View,
            countB, intsPerDesc,
            bufBestIdx.View, bufBestDist.View, bufSecondDist.View);

        await accelerator.SynchronizeAsync();

        // Read results back
        var bestIdx = await bufBestIdx.CopyToHostAsync();
        var bestDist = await bufBestDist.CopyToHostAsync();
        var secondDist = await bufSecondDist.CopyToHostAsync();

        // Apply ratio test on CPU (fast — just filtering arrays)
        var forwardMatches = new List<FeatureMatch>();
        for (int i = 0; i < countA; i++)
        {
            if (bestIdx[i] < 0 || bestDist[i] >= _maxDistance) continue;

            // Ratio test
            if (secondDist[i] > 0 && bestDist[i] < _ratioThreshold * secondDist[i])
            {
                forwardMatches.Add(new FeatureMatch
                {
                    IndexA = i,
                    IndexB = bestIdx[i],
                    Distance = bestDist[i],
                });
            }
        }

        // Cross-check: run reverse matching on GPU
        using var bufBestIdxR = accelerator.Allocate1D<int>(countB);
        using var bufBestDistR = accelerator.Allocate1D<int>(countB);
        using var bufSecondDistR = accelerator.Allocate1D<int>(countB);

        _matchKernel!((Index1D)countB,
            bufDescB.View, bufDescA.View,
            countA, intsPerDesc,
            bufBestIdxR.View, bufBestDistR.View, bufSecondDistR.View);

        await accelerator.SynchronizeAsync();

        var reverseIdx = await bufBestIdxR.CopyToHostAsync();

        // Filter by cross-check
        var result = new List<FeatureMatch>();
        foreach (var m in forwardMatches)
        {
            if (m.IndexB < reverseIdx.Length && reverseIdx[m.IndexB] == m.IndexA)
            {
                result.Add(m);
            }
        }

        return result;
    }

    /// <summary>
    /// ILGPU kernel: for each feature i in A, find the best and second-best
    /// matching feature in B by Hamming distance on packed int descriptors.
    /// </summary>
    private static void HammingMatchKernel(
        Index1D idx,
        ArrayView<int> descA,
        ArrayView<int> descB,
        int countB,
        int intsPerDesc,
        ArrayView<int> bestIdx,
        ArrayView<int> bestDist,
        ArrayView<int> secondDist)
    {
        int offsetA = idx * intsPerDesc;
        int best = 999999;
        int second = 999999;
        int bestJ = -1;

        for (int j = 0; j < countB; j++)
        {
            int offsetB = j * intsPerDesc;
            int dist = 0;

            // Hamming distance: XOR + popcount on 8 ints (256 bits)
            for (int k = 0; k < intsPerDesc; k++)
            {
                int xorVal = descA[offsetA + k] ^ descB[offsetB + k];
                // Bit count using ILGPU intrinsic
                dist += IntrinsicMath.BitOperations.PopCount(xorVal);
            }

            if (dist < best)
            {
                second = best;
                best = dist;
                bestJ = j;
            }
            else if (dist < second)
            {
                second = dist;
            }
        }

        bestIdx[idx] = bestJ;
        bestDist[idx] = best;
        secondDist[idx] = second;
    }

    /// <summary>
    /// Pack feature descriptors (byte[32]) into int[] for GPU transfer.
    /// Each descriptor becomes 8 consecutive ints.
    /// </summary>
    private static int[] PackDescriptors(List<ImageFeature> features)
    {
        const int intsPerDesc = 8;
        var data = new int[features.Count * intsPerDesc];

        for (int i = 0; i < features.Count; i++)
        {
            var desc = features[i].Descriptor;
            int baseIdx = i * intsPerDesc;

            for (int w = 0; w < intsPerDesc; w++)
            {
                int byteOffset = w * 4;
                data[baseIdx + w] =
                    desc[byteOffset] |
                    (desc[byteOffset + 1] << 8) |
                    (desc[byteOffset + 2] << 16) |
                    (desc[byteOffset + 3] << 24);
            }
        }

        return data;
    }
}
