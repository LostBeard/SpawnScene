using System.Numerics;
using ILGPU;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate: <see cref="GpuEpipolarRansac"/> (all pairs, f32 hypotheses on the device, double refit on the host) must
/// verify pairs as well as the CPU <see cref="EpipolarRansac"/> it replaces in production. Built on Truck's real
/// COLMAP cameras and points: pairs 1-12 frames apart (the verification candidates), 0.5 px noise, 10-70% chance
/// matches, plus pure-noise pairs at Truck's measured noise floor (~34 chance matches) that must be rejected.
/// Run on the ILGPU CPU accelerator.
/// </summary>
public class GpuEpipolarRansacTests
{
    const int W = 979, H = 546;

    static string DatasetFile(string name) => Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
        "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", "Truck", name));

    static float Gauss(Random rng) =>
        (float)(Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rng.NextDouble()));

    internal sealed record TestPair(float[] A, float[] B, bool[] Truth, int Seed, string Kind);

    internal static List<TestPair>? BuildPairs(int seed)
    {
        string posesPath = DatasetFile("poses.par"), pointsPath = DatasetFile("points3d.bin");
        if (!File.Exists(posesPath) || !File.Exists(pointsPath)) return null;
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(posesPath), 1957, 1091)
            .OrderBy(e => e.filename, StringComparer.Ordinal)
            .Select(e => e.camera.ScaledTo(W, H)).ToList();
        var cloud = SparsePointCloudInit.Parse(File.ReadAllBytes(pointsPath)).Positions;
        var rng = new Random(seed);
        var pairs = new List<TestPair>();
        int[] gaps = { 1, 2, 3, 5, 8, 12 };
        double[] outlierFractions = { 0.1, 0.3, 0.5, 0.7 };
        for (int i = 0; i + 12 < cams.Count; i += 5)
        {
            foreach (int gap in gaps)
            {
                int j = i + gap;
                var both = new List<(float u1, float v1, float u2, float v2)>();
                foreach (var p in cloud)
                {
                    if (!WorldSpaceGeometry.Project(cams[i], p, out var u1, out var v1, out var z1) || z1 <= 0.1f) continue;
                    if (!WorldSpaceGeometry.Project(cams[j], p, out var u2, out var v2, out var z2) || z2 <= 0.1f) continue;
                    if (u1 < 0 || v1 < 0 || u1 >= W || v1 >= H || u2 < 0 || v2 < 0 || u2 >= W || v2 >= H) continue;
                    both.Add((u1, v1, u2, v2));
                }
                if (both.Count < 30) continue;
                // Truck's verified pairs carry ~20-400 true matches; draw a count in that range.
                int inliers = Math.Min(both.Count, 20 + rng.Next(380));
                double frac = outlierFractions[rng.Next(outlierFractions.Length)];
                int outliers = (int)Math.Round(inliers * frac / (1 - frac));
                var a = new List<float>(); var b = new List<float>(); var t = new List<bool>();
                foreach (var k in Enumerable.Range(0, both.Count).OrderBy(_ => rng.Next()).Take(inliers))
                {
                    var (u1, v1, u2, v2) = both[k];
                    a.Add(u1 + 0.5f * Gauss(rng)); a.Add(v1 + 0.5f * Gauss(rng));
                    b.Add(u2 + 0.5f * Gauss(rng)); b.Add(v2 + 0.5f * Gauss(rng));
                    t.Add(true);
                }
                for (int k = 0; k < outliers; k++)
                {
                    a.Add((float)(rng.NextDouble() * W)); a.Add((float)(rng.NextDouble() * H));
                    b.Add((float)(rng.NextDouble() * W)); b.Add((float)(rng.NextDouble() * H));
                    t.Add(false);
                }
                // Shuffle so inliers are not a prefix.
                var order = Enumerable.Range(0, t.Count).OrderBy(_ => rng.Next()).ToArray();
                pairs.Add(new TestPair(
                    order.SelectMany(k => new[] { a[k * 2], a[k * 2 + 1] }).ToArray(),
                    order.SelectMany(k => new[] { b[k * 2], b[k * 2 + 1] }).ToArray(),
                    order.Select(k => t[k]).ToArray(), i * 7919 + j, $"gap{gap} out{frac:F1}"));
            }
            // Pure noise at Truck's measured floor (and above it): must never verify.
            foreach (int count in new[] { 34, 60 })
            {
                var a = new float[count * 2]; var b = new float[count * 2];
                for (int k = 0; k < count; k++)
                {
                    a[k * 2] = (float)(rng.NextDouble() * W); a[k * 2 + 1] = (float)(rng.NextDouble() * H);
                    b[k * 2] = (float)(rng.NextDouble() * W); b[k * 2 + 1] = (float)(rng.NextDouble() * H);
                }
                pairs.Add(new TestPair(a, b, new bool[count], i * 7919 + 100000 + count, $"noise{count}"));
            }
        }
        return pairs;
    }

    [Test]
    public async Task VerifiesTruckPairs_AsWellAsTheCpuEstimator()
    {
        var pairs = BuildPairs(seed: 3);
        if (pairs == null) Assert.Ignore("Truck dataset not in this checkout");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var cpu = pairs!.Select(p => EpipolarRansac.Estimate(p.A, p.B, thresholdPx: 2.0, minInliers: 15, seed: p.Seed)).ToArray();
        double cpuSec = sw.Elapsed.TotalSeconds;

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var gpuRansac = new GpuEpipolarRansac(accel) { FirstBatchUnits = 1L << 20, TargetBatchMs = 1 };  // many adaptive batches
        sw.Restart();
        var gpu = await gpuRansac.EstimateAsync(
            pairs.Select(p => new GpuEpipolarRansac.Pair(p.A, p.B, p.Seed)).ToList(), thresholdPx: 2.0, minInliers: 15);
        double gpuSec = sw.Elapsed.TotalSeconds;

        int real = 0, noise = 0, disagree = 0, noiseAccepted = 0, gpuOnlyRejected = 0;
        // Per chance-match rate: (true matches, CPU true positives, GPU true positives, CPU false, GPU false).
        var byRate = new SortedDictionary<string, long[]>();
        for (int k = 0; k < pairs.Count; k++)
        {
            var p = pairs[k];
            if (p.Kind.StartsWith("noise")) { noise++; if (gpu[k] != null) noiseAccepted++; continue; }
            real++;
            string rate = p.Kind[(p.Kind.IndexOf("out") + 3)..];
            // At 70% chance matches (true inlier ratio 0.30) both estimators land on partial models near the 0.25
            // inlier-ratio bar, so accept/reject is borderline per pair on either side - compare decisions only where
            // RANSAC is reliable.
            bool decisive = double.Parse(rate, System.Globalization.CultureInfo.InvariantCulture) <= 0.5;
            if (decisive && (cpu[k] == null) != (gpu[k] == null))
            {
                disagree++;
                if (gpu[k] == null) gpuOnlyRejected++;
                TestContext.Out.WriteLine($"  decision differs: pair {k} {p.Kind} n={p.Truth.Length} " +
                    $"cpu={(cpu[k]?.InlierCount.ToString() ?? "null")} gpu={(gpu[k]?.InlierCount.ToString() ?? "null")}");
            }
            if (!byRate.TryGetValue(rate, out var acc)) byRate[rate] = acc = new long[5];
            for (int i = 0; i < p.Truth.Length; i++)
            {
                if (p.Truth[i]) acc[0]++;
                if (cpu[k] != null && cpu[k]!.Inliers[i]) { if (p.Truth[i]) acc[1]++; else acc[3]++; }
                if (gpu[k] != null && gpu[k]!.Inliers[i]) { if (p.Truth[i]) acc[2]++; else acc[4]++; }
            }
        }
        TestContext.Out.WriteLine(
            $"{real} real pairs + {noise} noise pairs; CPU {cpuSec:F2}s, GPU-kernels-on-CPU {gpuSec:F2}s; " +
            $"decisions differ on {disagree}; noise pairs accepted by GPU: {noiseAccepted}");
        foreach (var (rate, a) in byRate)
            TestContext.Out.WriteLine($"  chance-match rate {rate}: {a[0]} true | recall CPU {(double)a[1] / a[0]:P1} " +
                $"GPU {(double)a[2] / a[0]:P1} | false inliers CPU {a[3]} GPU {a[4]}");

        Assert.That(real, Is.GreaterThan(100), "the gate must cover Truck's neighbour pairs");
        Assert.That(noiseAccepted, Is.Zero, "a pair of chance matches must not verify");
        Assert.That(gpuOnlyRejected, Is.Zero, "GPU rejected a real pair the CPU verified");
        Assert.That(disagree, Is.LessThanOrEqualTo(Math.Max(1, real / 100)), "accept/reject decisions must match the CPU estimator");
        foreach (var (rate, a) in byRate)
        {
            Assert.That(a[2], Is.GreaterThanOrEqualTo((long)(a[1] * 0.99)), $"recall below the CPU estimator at rate {rate}");
            // Precision, not a raw false count: a model that recovers more true matches also puts more chance
            // matches inside its 2 px band (rate 0.5: GPU +156 true, +11 false).
            double cpuPrecision = (double)a[1] / Math.Max(1, a[1] + a[3]), gpuPrecision = (double)a[2] / Math.Max(1, a[2] + a[4]);
            Assert.That(gpuPrecision, Is.GreaterThanOrEqualTo(cpuPrecision - 0.005), $"precision below the CPU estimator at rate {rate}");
            // 8-point RANSAC at 1,000 hypotheses: an all-inlier sample is near-certain up to 50% chance matches
            // (0.5^8 x 1000 = 3.9 expected), near-impossible at 70% (0.3^8 x 1000 = 0.07) - there only >= CPU holds.
            if (double.Parse(rate, System.Globalization.CultureInfo.InvariantCulture) <= 0.5)
                Assert.That((double)a[2] / a[0], Is.GreaterThanOrEqualTo(0.95), $"recall at chance-match rate {rate}");
        }
    }

    /// <summary>
    /// CPU estimator regression: a large pair whose FIRST hypothesis is poor (4 of 584 inliers, p8 ~ 5e-18) made
    /// 1 - p8 round to 1, the iteration bound -infinity, the int cast int.MinValue - RANSAC stopped after one
    /// iteration and rejected a real pair with 292 true matches. Found by the gate above (pairs 4 and 12).
    /// </summary>
    [Test]
    public void CpuEstimator_LargePairWithPoorFirstHypothesis_KeepsSampling()
    {
        var pairs = BuildPairs(seed: 3);
        if (pairs == null) Assert.Ignore("Truck dataset not in this checkout");
        foreach (int k in new[] { 4, 12 })
        {
            var p = pairs![k];
            var r = EpipolarRansac.Estimate(p.A, p.B, thresholdPx: 2.0, minInliers: 15, seed: p.Seed);
            int truth = p.Truth.Count(x => x);
            TestContext.Out.WriteLine($"pair {k} {p.Kind}: {truth} true, " + (r == null ? "rejected" : $"{r.InlierCount} inliers in {r.Iterations} iterations"));
            Assert.That(r, Is.Not.Null, $"pair {k} ({truth} true matches) must verify");
            Assert.That(r!.InlierCount, Is.GreaterThanOrEqualTo((int)(truth * 0.95)));
        }
    }

    /// <summary>
    /// Dispatch timing on a real GPU (ILGPU CUDA), at TruckFull scale: 8,716 candidate pairs. Sizes the per-dispatch
    /// bounds so no single submission approaches the Windows GPU watchdog (~2 s) in the browser, where one did.
    /// </summary>
    [Test, Explicit("benchmark: needs an NVIDIA GPU")]
    public async Task TruckFullScale_DispatchTimes_OnCuda()
    {
        var baseline = BuildPairs(seed: 3);
        if (baseline == null) Assert.Ignore("Truck dataset not in this checkout");
        var pairs = new List<GpuEpipolarRansac.Pair>();
        for (int rep = 0; pairs.Count < 8716; rep++)
            foreach (var p in baseline!)
                if (pairs.Count < 8716) pairs.Add(new GpuEpipolarRansac.Pair(p.A, p.B, p.Seed + rep * 1_000_003));
        using var context = Context.Create(b => b.Cuda().EnableAlgorithms());
        if (context.GetCudaDevices().Count == 0) Assert.Ignore("no CUDA device");
        using var accel = context.CreateCudaAccelerator(0);
        using var ransac = new GpuEpipolarRansac(accel);
        await ransac.EstimateAsync(pairs.Take(64).ToList());   // JIT/compile warm-up
        foreach (double target in new[] { 20.0, 100.0 })
        {
            ransac.TargetBatchMs = target;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var r = await ransac.EstimateAsync(pairs);
            double total = sw.Elapsed.TotalMilliseconds;
            var b = ransac.LastBatches;
            TestContext.Out.WriteLine($"target {target} ms: {b.Count} batches, total {total:F0} ms, verified {r.Count(x => x != null)}, " +
                $"batch max {b.Max(x => x.Ms):F1} ms, first {b[0].Ms:F1} ms ({b[0].Threads} threads), " +
                $"~{b.Sum(x => x.Ms) * 1e6 / b.Sum(x => x.Threads):F0} ns/thread, evals {b.Sum(x => x.Evaluations) / 1e9:F2}G");
            Assert.That(b.Skip(1).Max(x => x.Ms), Is.LessThan(target * 5), "adaptive batches must stay near their target");
        }
    }
}
