using ILGPU;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Equivalence gate: <see cref="GpuPairMatcher"/> (all pairs, batched) must return EXACTLY the matches of the
/// per-pair matcher (GpuFeatureMatcher: Hamming best/second, distance cap, ratio test, cross-check), on the ILGPU
/// CPU accelerator. The reference below is that matcher's logic transcribed, including its tie rule.
/// </summary>
public class GpuPairMatcherTests
{
    const float Ratio = 0.75f;
    const int MaxDistance = 64;

    static List<ImageFeature> Features(Random rng, int count, List<byte[]> shared)
    {
        var list = new List<ImageFeature>(count);
        for (int i = 0; i < count; i++)
        {
            var d = new byte[32];
            if (rng.NextDouble() < 0.5)
            {
                // A noisy copy of a shared "landmark" descriptor, so pairs have real matches, near ties and all.
                shared[rng.Next(shared.Count)].CopyTo(d, 0);
                for (int f = 0; f < rng.Next(0, 24); f++) d[rng.Next(32)] ^= (byte)(1 << rng.Next(8));
            }
            else rng.NextBytes(d);
            list.Add(new ImageFeature { X = i, Y = i, Descriptor = d });
        }
        return list;
    }

    static (int[] best, int[] dist, int[] second) Best(List<ImageFeature> q, List<ImageFeature> c)
    {
        var best = new int[q.Count]; var dist = new int[q.Count]; var second = new int[q.Count];
        for (int i = 0; i < q.Count; i++)
        {
            int b = 999999, s = 999999, bj = -1;
            for (int j = 0; j < c.Count; j++)
            {
                int dd = 0;
                for (int k = 0; k < 32; k++) dd += System.Numerics.BitOperations.PopCount((uint)(q[i].Descriptor[k] ^ c[j].Descriptor[k]));
                if (dd < b) { s = b; b = dd; bj = j; }
                else if (dd < s) s = dd;
            }
            best[i] = bj; dist[i] = b; second[i] = s;
        }
        return (best, dist, second);
    }

    /// <summary>GpuFeatureMatcher.MatchGpuAsync's filtering, transcribed.</summary>
    static List<FeatureMatch> Reference(List<ImageFeature> a, List<ImageFeature> b)
    {
        if (a.Count == 0 || b.Count == 0) return new();
        var (bi, bd, sd) = Best(a, b);
        var fwd = new List<FeatureMatch>();
        for (int i = 0; i < a.Count; i++)
        {
            if (bi[i] < 0 || bd[i] >= MaxDistance) continue;
            if (sd[i] > 0 && bd[i] < Ratio * sd[i])
                fwd.Add(new FeatureMatch { IndexA = i, IndexB = bi[i], Distance = bd[i] });
        }
        var (ri, _, _) = Best(b, a);
        return fwd.Where(m => m.IndexB < ri.Length && ri[m.IndexB] == m.IndexA).ToList();
    }

    [Test]
    public async Task AllPairs_IdenticalToThePerPairMatcher()
    {
        var rng = new Random(42);
        var shared = Enumerable.Range(0, 60).Select(_ => { var d = new byte[32]; rng.NextBytes(d); return d; }).ToList();
        // Different feature counts per image, including an empty one.
        var counts = new[] { 180, 220, 150, 0, 200, 95, 240 };
        var images = counts.Select(n => Features(rng, n, shared)).ToList();
        var pairs = new List<(int, int)>();
        for (int i = 0; i < images.Count - 1; i++) for (int j = i + 1; j < images.Count; j++) pairs.Add((i, j));

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var matcher = new GpuPairMatcher(accel, Ratio, MaxDistance);
        var got = new List<FeatureMatch>[pairs.Count];
        await matcher.MatchPairsAsync(images.Cast<IReadOnlyList<ImageFeature>>().ToList(), pairs, (p, m) => got[p] = m);

        int total = 0;
        for (int p = 0; p < pairs.Count; p++)
        {
            var (i, j) = pairs[p];
            var want = Reference(images[i], images[j]);
            total += want.Count;
            Assert.That(got[p].Select(m => (m.IndexA, m.IndexB, m.Distance)),
                Is.EqualTo(want.Select(m => (m.IndexA, m.IndexB, m.Distance))), $"pair {i}-{j}");
        }
        Assert.That(total, Is.GreaterThan(200), "the fixture must produce real matches");
        TestContext.Out.WriteLine($"{pairs.Count} pairs, {total} matches, all identical");
    }
}
