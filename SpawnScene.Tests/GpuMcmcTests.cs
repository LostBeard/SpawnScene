using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// MCMC density (GpuMcmc): the relocation rule against gsplat's own formula, and the device step on the ILGPU CPU
/// accelerator - every dead slot refilled, growth to 1.05 n, every touched row exactly its parent's share, the rest
/// untouched, and sampling that follows opacity.
/// </summary>
public class GpuMcmcTests
{
    const int F = SplatFormat.Floats;

    /// <summary>gsplat relocation_kernel, in double with an exact binomial table.</summary>
    static double ReferenceScaleFactor(double opacity, int ratio)
    {
        double na = 1 - Math.Pow(1 - opacity, 1.0 / ratio);
        double denom = 0;
        for (int i = 1; i <= ratio; i++)
            for (int k = 0; k <= i - 1; k++)
            {
                double binom = 1;
                for (int j = 0; j < k; j++) binom = binom * (i - 1 - j) / (j + 1);
                denom += binom * Math.Pow(-1, k) / Math.Sqrt(k + 1) * Math.Pow(na, k + 1);
            }
        return opacity / denom;
    }

    [TestCase(0.9f)] [TestCase(0.5f)] [TestCase(0.05f)] [TestCase(0.999f)]
    public void Relocation_CopiesStackToTheOriginalOpacity_AndMatchGsplatsScale(float o)
    {
        Assert.That(GpuMcmc.ScaleFactor(o, GpuMcmc.NewOpacity(o, 1), 1), Is.EqualTo(1f).Within(1e-5f), "one copy is the original");
        foreach (int n in new[] { 2, 3, 5, 10, 25, GpuMcmc.MaxRatio })
        {
            float na = GpuMcmc.NewOpacity(o, n);
            Assert.That(1 - Math.Pow(1 - na, n), Is.EqualTo(o).Within(2e-4), $"{n} copies stacked");
            double want = ReferenceScaleFactor(o, n);
            Assert.That(GpuMcmc.ScaleFactor(o, na, n), Is.EqualTo(want).Within(Math.Abs(want) * 2e-3 + 1e-5), $"scale factor, {n} copies");
        }
    }

    static float[] Scene(int n, int seed, double deadFraction)
    {
        var rows = LodTreeTests.Scene(n, seed);
        var rng = new Random(seed + 1);
        for (int i = 0; i < n; i++)
            rows[i * F + 9] = rng.NextDouble() < deadFraction ? (float)rng.NextDouble() * GpuMcmc.MinOpacity : 0.01f + (float)rng.NextDouble() * 0.99f;
        return rows;
    }

    [TestCase(4000, 3, true)]
    [TestCase(20000, 4, false)]
    public void Step_RefillsTheDead_GrowsAndWritesExactShares(int n, int seed, bool grow)
    {
        var rows = Scene(n, seed, 0.1);
        int deadIn = Enumerable.Range(0, n).Count(i => rows[i * F + 9] <= GpuMcmc.MinOpacity);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var packed = accel.Allocate1D(rows);
        var mcmc = new GpuMcmc(accel);
        int cap = n + n / 100;   // binds: growth is 1% here, not 5%
        var step = mcmc.RunAsync(packed.View, n, cap, grow, 77).GetAwaiter().GetResult();
        Assert.That(step, Is.Not.Null);
        var (r, s) = step!.Value;
        using var outPacked = r.Packed; using var adamSrc = r.AdamSources; using var featSrc = r.FeatureSources;
        int added = grow ? cap - n : 0;
        Assert.That(s.Relocated, Is.EqualTo(deadIn));
        Assert.That(s.Added, Is.EqualTo(added));
        Assert.That(r.Count, Is.EqualTo(n + added));

        var o = outPacked.GetAsArray1D();
        var a = adamSrc.GetAsArray1D();
        var f = featSrc.GetAsArray1D();
        var copies = new int[n];
        for (int d = 0; d < r.Count; d++)
            if (f[d] != d) copies[f[d]]++;
        Assert.That(copies.Sum(), Is.EqualTo(deadIn + added), "one copy per sample");

        double sampledOpacity = 0, liveOpacity = 0; int live = 0;
        for (int d = 0; d < r.Count; d++)
        {
            int src = f[d];
            Assert.That(src, Is.InRange(0, n - 1));
            bool wasDead = d < n && rows[d * F + 9] <= GpuMcmc.MinOpacity;
            if (wasDead) Assert.That(src, Is.Not.EqualTo(d), $"dead slot {d} refilled");
            Assert.That(rows[src * F + 9], Is.GreaterThan(GpuMcmc.MinOpacity), $"row {d}: a dead parent was sampled");
            int ratio = Math.Min(copies[src] + 1, GpuMcmc.MaxRatio);
            bool touched = copies[src] > 0;
            Assert.That(a[d], Is.EqualTo(touched ? -1 : d), $"row {d}: Adam source");
            float pa = rows[src * F + 9];
            float na = touched ? GpuMcmc.ClampOpacity(GpuMcmc.NewOpacity(pa, ratio)) : pa;
            float k = touched ? GpuMcmc.ScaleFactor(pa, GpuMcmc.NewOpacity(pa, ratio), ratio) : 1f;
            for (int c = 0; c < F; c++)
            {
                float want = c == 9 ? na : c is >= 6 and <= 8 ? rows[src * F + c] * k : rows[src * F + c];
                Assert.That(o[d * F + c], Is.EqualTo(want).Within(Math.Abs(want) * 1e-5f + 1e-7f), $"row {d} float {c}");
            }
        }
        for (int i = 0; i < n; i++)
        {
            float op = rows[i * F + 9];
            if (op <= GpuMcmc.MinOpacity) continue;
            live++; liveOpacity += op; sampledOpacity += op * copies[i];
        }
        // Sampled in proportion to opacity: the mean opacity of a draw is E[o^2]/E[o], well above E[o] for o ~ U(0.01, 1).
        double meanDraw = sampledOpacity / copies.Sum(), meanLive = liveOpacity / live;
        Assert.That(meanDraw, Is.GreaterThan(meanLive * 1.2), $"draws average opacity {meanDraw:F3} vs {meanLive:F3} live");
    }

    [Test]
    public void Step_WithNothingToDo_ReturnsNull()
    {
        var rows = Scene(1000, 9, 0);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var packed = accel.Allocate1D(rows);
        Assert.That(new GpuMcmc(accel).RunAsync(packed.View, 1000, 1000, true, 1).GetAwaiter().GetResult(), Is.Null);
    }
}
