using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Pair retrieval for the learned front end (2026-10-01): LearnedFeatureMatcher.PairScoresAsync on the device vs a host
/// reference with the same rules (mutual nearest neighbours by dot product of unit descriptors, ratio test on Euclidean
/// distance), and TopPartnerPairs. Synthetic: 5 images x 256 random unit descriptors; image 1 carries noisy copies of 120
/// of image 0's, image 3 noisy copies of 60 of image 2's, and 40 of image 1's are copies of ONE of image 4's.
/// </summary>
public class LearnedRetrievalTests
{
    sealed class NoModels : SpawnDev.ILGPU.ML.Hub.IModelSource
    {
        public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    }

    static float[] Unit(Random rng)
    {
        var v = new float[128]; double s = 0;
        for (int d = 0; d < 128; d++) { v[d] = (float)(rng.NextDouble() * 2 - 1); s += v[d] * v[d]; }
        float inv = (float)(1 / Math.Sqrt(s));
        for (int d = 0; d < 128; d++) v[d] *= inv;
        return v;
    }

    static int Reference(float[][] a, float[][] b, float ratio)
    {
        int k = a.Length;
        int[] Best(float[][] x, float[][] y, out float[] s1, out float[] s2)
        {
            var arg = new int[x.Length]; s1 = new float[x.Length]; s2 = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                float b1 = -2, b2 = -2; int ai = -1;
                for (int j = 0; j < y.Length; j++)
                {
                    float dot = 0; for (int d = 0; d < 128; d++) dot += x[i][d] * y[j][d];
                    if (dot > b1) { b2 = b1; b1 = dot; ai = j; } else if (dot > b2) b2 = dot;
                }
                arg[i] = ai; s1[i] = b1; s2[i] = b2;
            }
            return arg;
        }
        var ab = Best(a, b, out var bs1, out var bs2);
        var ba = Best(b, a, out _, out _);
        int c = 0;
        for (int i = 0; i < k; i++)
        {
            if (ba[ab[i]] != i) continue;
            float d1 = MathF.Sqrt(MathF.Max(0, 2 - 2 * bs1[i])), d2 = MathF.Sqrt(MathF.Max(0, 2 - 2 * bs2[i]));
            if (d1 < ratio * d2) c++;
        }
        return c;
    }

    [Test]
    public async Task PairScores_LargeK_CompareAStridedSubset()
    {
        // K=2048: retrieval compares every 2nd descriptor (LearnedFeatureMatcher.RetrievalDescriptors = 1024), so its
        // cost stays at the K=1024 level (b140: all 3072 took 691 s on TruckFull). The device must equal the host
        // reference run on exactly that subset.
        const int n = 3, k = 2048;
        var rng = new Random(5);
        var desc = new float[n][][];
        for (int i = 0; i < n; i++) desc[i] = Enumerable.Range(0, k).Select(_ => Unit(rng)).ToArray();
        for (int q = 0; q < 300; q++)
        {
            var v = desc[0][q * 2].Select(x => x + (float)(rng.NextDouble() - 0.5) * 0.02f).ToArray();
            float inv = 1 / MathF.Sqrt(v.Sum(x => x * x));
            desc[1][(q * 14) % k] = v.Select(x => x * inv).ToArray(); // even slots: inside the compared subset
        }
        int stride = k / LearnedFeatureMatcher.RetrievalDescriptors;
        Assert.That(stride, Is.EqualTo(2));
        float[][] Sub(float[][] d) => Enumerable.Range(0, k / stride).Select(i => d[i * stride]).ToArray();

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var matcher = new LearnedFeatureMatcher(() => accel, new NoModels());
        var images = new List<ImportedImage>();
        try
        {
            for (int i = 0; i < n; i++)
                images.Add(new ImportedImage
                {
                    FileName = $"img{i}",
                    LearnedDescriptors = new LearnedFeatureMatcher.ImageDescriptors
                    {
                        K = k,
                        Normalized = accel.Allocate1D<float>(k * 2),
                        Descriptors = accel.Allocate1D(desc[i].SelectMany(x => x).ToArray()),
                    },
                });
            var scores = await matcher.PairScoresAsync(images);
            for (int a = 0; a < n; a++)
                for (int b = a + 1; b < n; b++)
                    Assert.That(scores[a, b], Is.EqualTo(Reference(Sub(desc[a]), Sub(desc[b]), LearnedFeatureMatcher.RetrievalRatio)), $"pair {a}-{b}");
            Assert.That(scores[0, 1], Is.GreaterThan(250), "the planted overlap ranks first");
            Assert.That(scores[0, 2], Is.LessThan(20));
        }
        finally { foreach (var im in images) im.DisposeSource(); }
    }

    [Test]
    public async Task PairScores_MatchHostReference_AndRankTheOverlappingPairs()
    {
        const int n = 5, k = 256;
        var rng = new Random(3);
        var desc = new float[n][][];
        for (int i = 0; i < n; i++) desc[i] = Enumerable.Range(0, k).Select(_ => Unit(rng)).ToArray();
        void Plant(int from, int to, int count)
        {
            for (int q = 0; q < count; q++)
            {
                var v = desc[from][q].Select(x => x + (float)(rng.NextDouble() - 0.5) * 0.02f).ToArray();
                float inv = 1 / MathF.Sqrt(v.Sum(x => x * x));
                desc[to][(q * 7) % k] = v.Select(x => x * inv).ToArray();
            }
        }
        Plant(0, 1, 120);
        Plant(2, 3, 60);
        // Many-to-one: 40 of image 1's descriptors are noisy copies of ONE of image 4's - each one's nearest neighbour is
        // that descriptor, but only one of them is its mutual nearest. Without the mutual check pair 1-4 would score ~40.
        for (int q = 0; q < 40; q++)
        {
            var v = desc[4][5].Select(x => x + (float)(rng.NextDouble() - 0.5) * 0.02f).ToArray();
            float inv = 1 / MathF.Sqrt(v.Sum(x => x * x));
            desc[1][200 + q] = v.Select(x => x * inv).ToArray();
        }

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var matcher = new LearnedFeatureMatcher(() => accel, new NoModels());
        var images = new List<ImportedImage>();
        try
        {
            for (int i = 0; i < n; i++)
                images.Add(new ImportedImage
                {
                    FileName = $"img{i}",
                    LearnedDescriptors = new LearnedFeatureMatcher.ImageDescriptors
                    {
                        K = k,
                        Normalized = accel.Allocate1D<float>(k * 2),
                        Descriptors = accel.Allocate1D(desc[i].SelectMany(x => x).ToArray()),
                    },
                });
            var scores = await matcher.PairScoresAsync(images, batchPairs: 3);   // 10 pairs in batches of 3: a short last batch
            for (int a = 0; a < n; a++)
                for (int b = a + 1; b < n; b++)
                {
                    int want = Reference(desc[a], desc[b], LearnedFeatureMatcher.RetrievalRatio);
                    Assert.That(scores[a, b], Is.EqualTo(want), $"pair {a}-{b}");
                    Assert.That(scores[b, a], Is.EqualTo(scores[a, b]));
                }
            TestContext.Out.WriteLine($"scores 0-1 {scores[0, 1]}, 2-3 {scores[2, 3]}, 0-2 {scores[0, 2]}, 1-4 {scores[1, 4]}");
            Assert.That(scores[0, 1], Is.GreaterThan(100));
            Assert.That(scores[1, 4], Is.LessThanOrEqualTo(2), "many-to-one matches count once (mutual)");
            Assert.That(scores[2, 3], Is.GreaterThan(45));
            var top1 = LearnedFeatureMatcher.TopPartnerPairs(scores, 1);
            Assert.That(top1, Does.Contain((0, 1)));
            Assert.That(top1, Does.Contain((2, 3)));
        }
        finally { foreach (var im in images) im.DisposeSource(); }
    }
}
