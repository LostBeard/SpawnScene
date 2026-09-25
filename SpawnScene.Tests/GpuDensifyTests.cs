using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Equivalence gate: <see cref="GpuDensify"/> against the host oracle (<see cref="SplatDensityControl.Decide"/>
/// + <see cref="SplatDensityControl.Apply"/>) on the same splats and statistics, on the ILGPU CPU accelerator.
///
/// Within budget the decision must be identical - same survivors in the same order, the same clones, the same
/// split parents. Split children are drawn from different random numbers on each side, so they are checked for
/// what they must be (parent's features and shrunk scale, a draw from the parent's ellipsoid), not for position.
/// </summary>
public class GpuDensifyTests
{
    const int F = SplatFormat.Floats;
    const float Extent = 5f;

    static Context NewContext() => Context.Create(b => b.CPU().EnableAlgorithms());

    /// <summary>A mix of faint, big, small-and-busy, big-and-busy and quiet splats, with distinct gradients.</summary>
    static (float[] packed, float[] stats, float[] radius, SplatDensityControl.Splat[] splats,
        SplatDensityControl.Accumulator[] acc) Scene(int n, int seed)
    {
        var rng = new Random(seed);
        var packed = new float[n * F];
        var stats = new float[n * 2];
        var radius = new float[n];
        var splats = new SplatDensityControl.Splat[n];
        var acc = new SplatDensityControl.Accumulator[n];
        for (int i = 0; i < n; i++)
        {
            int o = i * F;
            float scale = rng.NextDouble() < 0.4 ? 0.08f + (float)rng.NextDouble() * 0.2f : 0.01f + (float)rng.NextDouble() * 0.02f;
            float opacity = rng.NextDouble() < 0.1 ? 0.002f : 0.05f + (float)rng.NextDouble() * 0.9f;
            var q = new[] { (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, 1f };
            packed[o + 0] = (float)rng.NextDouble() * 4f; packed[o + 1] = (float)rng.NextDouble() * 4f; packed[o + 2] = (float)rng.NextDouble() * 4f;
            packed[o + 3] = (float)rng.NextDouble(); packed[o + 4] = (float)rng.NextDouble(); packed[o + 5] = (float)rng.NextDouble();
            packed[o + 6] = scale; packed[o + 7] = scale * 0.7f; packed[o + 8] = scale * 0.5f;
            packed[o + 9] = opacity;
            packed[o + 10] = q[0]; packed[o + 11] = q[1]; packed[o + 12] = q[2]; packed[o + 13] = q[3];

            // Distinct gradients, a third of them over the bar; a few never visible.
            int vis = rng.NextDouble() < 0.05 ? 0 : 1 + rng.Next(20);
            float avg = rng.NextDouble() < 0.35
                ? SplatDensityControl.GradientThreshold * (1.01f + (float)rng.NextDouble() * 30f)
                : SplatDensityControl.GradientThreshold * (float)rng.NextDouble() * 0.9f;
            stats[i * 2] = avg * vis;
            stats[i * 2 + 1] = vis;
            radius[i] = (float)rng.NextDouble() * 100f;

            splats[i] = new SplatDensityControl.Splat
            {
                PosX = packed[o], PosY = packed[o + 1], PosZ = packed[o + 2],
                ColR = packed[o + 3], ColG = packed[o + 4], ColB = packed[o + 5],
                ScaleX = packed[o + 6], ScaleY = packed[o + 7], ScaleZ = packed[o + 8],
                Opacity = opacity,
                QuatX = q[0], QuatY = q[1], QuatZ = q[2], QuatW = q[3],
            };
            acc[i] = new SplatDensityControl.Accumulator
            {
                GradientSum = stats[i * 2], VisibleCount = vis, MaxScreenRadiusPx = radius[i],
            };
        }
        return (packed, stats, radius, splats, acc);
    }

    static async Task<(float[] packed, int[] adam, int[] feat, GpuDensify.Result r)> RunGpu(
        float[] packed, float[] stats, float[] radius, int n, GpuDensify.Options o)
    {
        using var context = NewContext();
        using var accel = context.CreateCPUAccelerator(0);
        using var p = accel.Allocate1D(packed);
        using var s = accel.Allocate1D(stats);
        using var r = accel.Allocate1D(radius);
        using var d = new GpuDensify(accel);
        var res = await d.RunAsync(p.View, n, s.View, r.View, o);
        var outPacked = res.Packed.GetAsArray1D();
        var adam = res.AdamSources.GetAsArray1D();
        var feat = res.FeatureSources.GetAsArray1D();
        res.Packed.Dispose(); res.AdamSources.Dispose(); res.FeatureSources.Dispose();
        return (outPacked, adam, feat, res);
    }

    [TestCase(false)]
    [TestCase(true)]
    public async Task WithinBudget_SameDecisionAsTheHostOracle(bool afterReset)
    {
        const int n = 3000;
        var (packed, stats, radius, splats, acc) = Scene(n, 7);
        float savedRadius = SplatDensityControl.MaxScreenRadiusPx;
        try
        {
            // A radius bar some of these exceed, so the post-reset screen prune is exercised too.
            SplatDensityControl.MaxScreenRadiusPx = 80f;
            var plan = SplatDensityControl.Decide(splats, acc, Extent, afterReset, () => 0f);
            var cpu = SplatDensityControl.Apply(splats, plan, out var cpuAdam, out var cpuFeat);

            var (g, adam, feat, r) = await RunGpu(packed, stats, radius, n,
                new GpuDensify.Options(Extent, afterReset, int.MaxValue, ResetOpacity: false, Seed: 1));

            Assert.That(r.Count, Is.EqualTo(cpu.Count), "grown count");
            Assert.That(r.Cloned, Is.EqualTo(plan.Cloned), "clones");
            Assert.That(r.Split, Is.EqualTo(plan.Split), "splits");
            Assert.That(r.PrunedFaint, Is.EqualTo(plan.PrunedOpacity), "faint prunes");
            Assert.That(r.PrunedBig, Is.EqualTo(plan.PrunedTooBig), "size prunes");
            if (afterReset) Assert.That(r.PrunedBig, Is.GreaterThan(0), "the size prune must be exercised");

            // Survivors: identical rows, identical order, identical source maps.
            int kept = cpu.Count - plan.Add.Count;
            for (int i = 0; i < kept; i++)
            {
                Assert.That(adam[i], Is.EqualTo(cpuAdam[i]), $"survivor {i} adam source");
                Assert.That(feat[i], Is.EqualTo(cpuFeat[i]), $"survivor {i} feature source");
                for (int k = 0; k < F; k++)
                    Assert.That(g[i * F + k], Is.EqualTo(packed[cpuAdam[i] * F + k]), $"survivor {i} float {k}");
            }

            // Added: same parents (as a multiset), fresh Adam, and each row is what its kind must be.
            var gpuParents = feat.Skip(kept).OrderBy(x => x).ToArray();
            var cpuParents = cpuFeat.Skip(kept).OrderBy(x => x).ToArray();
            Assert.That(gpuParents, Is.EqualTo(cpuParents), "parents of the added splats");
            Assert.That(adam.Skip(kept).All(a => a == -1), "added splats start with fresh moments");

            var splitParents = plan.Remove.Where(i => !(acc[i].AverageGradient < SplatDensityControl.GradientThreshold)
                && splats[i].Opacity >= SplatDensityControl.MinOpacity).ToHashSet();
            for (int j = kept; j < r.Count; j++)
            {
                int parent = feat[j];
                int po = parent * F, o = j * F;
                bool split = splitParents.Contains(parent);
                for (int k = 3; k < F; k++)
                {
                    if (k is 6 or 7 or 8) continue;
                    Assert.That(g[o + k], Is.EqualTo(packed[po + k]), $"added {j} feature {k}");
                }
                for (int k = 6; k <= 8; k++)
                {
                    float want = split ? packed[po + k] / SplatDensityControl.SplitScaleDivisor : packed[po + k];
                    Assert.That(g[o + k], Is.EqualTo(want).Within(1e-6f), $"added {j} scale {k}");
                }
                float dist = MathF.Sqrt(MathF.Pow(g[o] - packed[po], 2) + MathF.Pow(g[o + 1] - packed[po + 1], 2) + MathF.Pow(g[o + 2] - packed[po + 2], 2));
                if (split) Assert.That(dist, Is.LessThan(6f * packed[po + 6]), $"split child {j} lands inside the parent's ellipsoid");
                else Assert.That(dist, Is.EqualTo(0f), $"clone {j} sits exactly on its parent");
            }
        }
        finally { SplatDensityControl.MaxScreenRadiusPx = savedRadius; }
    }

    [Test]
    public async Task OverBudget_TakesTheHighestGradientsAndStaysInBudget()
    {
        const int n = 4000;
        var (packed, stats, radius, splats, acc) = Scene(n, 11);
        var plan = SplatDensityControl.Decide(splats, acc, Extent, false, () => 0f);
        int wanted = plan.Cloned + 2 * plan.Split;
        int budget = wanted / 3;

        var (_, _, feat, r) = await RunGpu(packed, stats, radius, n,
            new GpuDensify.Options(Extent, false, n + budget, ResetOpacity: false, Seed: 1));

        Assert.That(r.Added, Is.LessThanOrEqualTo(budget), "growth within budget");
        Assert.That(r.Added, Is.GreaterThan(budget * 9 / 10), "and close to it");

        // No unselected candidate more than one histogram bin (1/64 octave) above a selected one.
        var selectedParents = feat.Skip(r.Count - r.Added).ToHashSet();
        var candidates = Enumerable.Range(0, n).Where(i => splats[i].Opacity >= SplatDensityControl.MinOpacity
            && acc[i].AverageGradient >= SplatDensityControl.GradientThreshold).ToList();
        float minSelected = candidates.Where(selectedParents.Contains).Min(i => acc[i].AverageGradient);
        float maxUnselected = candidates.Where(i => !selectedParents.Contains(i)).Max(i => acc[i].AverageGradient);
        Assert.That(maxUnselected, Is.LessThan(minSelected * MathF.Pow(2f, 1f / 64f) * 1.0001f),
            $"selected down to {minSelected:G6}, an unselected candidate at {maxUnselected:G6}");
    }

    [Test]
    public async Task OpacityReset_CapsEveryOutputOpacity()
    {
        const int n = 1500;
        var (packed, stats, radius, _, _) = Scene(n, 3);
        var (g, _, _, r) = await RunGpu(packed, stats, radius, n,
            new GpuDensify.Options(Extent, false, int.MaxValue, ResetOpacity: true, Seed: 1));
        for (int j = 0; j < r.Count; j++)
            Assert.That(g[j * F + 9], Is.LessThanOrEqualTo(SplatDensityControl.OpacityResetTo), $"splat {j}");
    }

    [Test]
    public async Task NoOp_IsTheIdentity()
    {
        const int n = 800;
        var (packed, stats, radius, _, _) = Scene(n, 5);
        var (g, adam, feat, r) = await RunGpu(packed, stats, radius, n,
            new GpuDensify.Options(Extent, true, int.MaxValue, ResetOpacity: false, Seed: 1, NoOp: true));
        Assert.That(r.Count, Is.EqualTo(n));
        Assert.That(g.Take(n * F), Is.EqualTo(packed));
        Assert.That(adam.Take(n), Is.EqualTo(Enumerable.Range(0, n)));
        Assert.That(feat.Take(n), Is.EqualTo(Enumerable.Range(0, n)));
    }
}
