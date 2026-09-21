using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Does the optimiser actually optimise?
///
/// The gradient tests prove the derivatives are right; these prove the loop built on them
/// converges. Both are needed: correct gradients wired into a broken loop (bad bias correction,
/// wrong activation chain, stale depth order) still produce a flat loss curve, and on the GPU
/// that would read as a shader bug.
///
/// The decisive test recovers KNOWN parameters: render a target from a scene whose colours and
/// opacities we chose, perturb them, and check the optimiser walks back. An "it went down a bit"
/// assertion passes for an optimiser that is merely dimming everything.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter SplatOptimizer</c>
/// </summary>
public class SplatOptimizerTests
{
    const int W = 16, H = 14;

    static List<SplatRasterizer.Splat2D> Scene(int n, int seed)
    {
        var rng = new Random(seed);
        var list = new List<SplatRasterizer.Splat2D>();
        for (int i = 0; i < n; i++)
        {
            float s = 2.0f + (float)rng.NextDouble();
            float inv = 1f / (s * s);
            list.Add(new SplatRasterizer.Splat2D
            {
                Px = 3f + (float)rng.NextDouble() * (W - 6f),
                Py = 3f + (float)rng.NextDouble() * (H - 6f),
                ConicA = inv, ConicB = 0f, ConicC = inv,
                R = 0.2f + (float)rng.NextDouble() * 0.6f,
                G = 0.2f + (float)rng.NextDouble() * 0.6f,
                B = 0.2f + (float)rng.NextDouble() * 0.6f,
                Opacity = 0.3f + (float)rng.NextDouble() * 0.4f,
                Depth = (float)rng.NextDouble(),
            });
        }
        return list;
    }

    [Test]
    public void Sigmoid_And_Logit_RoundTrip()
    {
        foreach (float p in new[] { 0.01f, 0.1f, 0.5f, 0.9f, 0.99f })
            Assert.That(SplatOptimizer.Sigmoid(SplatOptimizer.Logit(p)), Is.EqualTo(p).Within(1e-5f),
                $"opacity {p} must survive the logit round trip");
    }

    [Test]
    public void LossDecreasesMonotonicallyOverall()
    {
        var truth = Scene(8, seed: 3);
        var target = SplatRasterizer.Render(truth, W, H).Colour;

        // Same geometry, wrong appearance.
        var splats = Scene(8, seed: 3);
        for (int i = 0; i < splats.Count; i++)
        {
            var s = splats[i];
            s.R = 0.5f; s.G = 0.5f; s.B = 0.5f; s.Opacity = 0.5f;
            splats[i] = s;
        }

        var history = SplatOptimizer.FitColourOpacity(splats, target, W, H, iterations: 400);

        TestContext.Out.WriteLine($"loss {history[0]:F6} -> {history[^1]:F6}");
        Assert.That(history[^1], Is.LessThan(history[0] * 0.5f),
            $"loss should at least halve: {history[0]:F6} -> {history[^1]:F6}");

        // Not strictly monotonic (Adam overshoots), but the trend must be down.
        float firstQuarter = history.Take(history.Length / 4).Average();
        float lastQuarter = history.Skip(3 * history.Length / 4).Average();
        Assert.That(lastQuarter, Is.LessThan(firstQuarter),
            "average loss in the last quarter must beat the first quarter");
    }

    [Test]
    public void RecoversKnownColours()
    {
        // The real test: the target was rendered from parameters we know, so we can check the
        // optimiser walks BACK to them rather than merely reducing a number.
        var truth = Scene(5, seed: 11);
        var target = SplatRasterizer.Render(truth, W, H).Colour;

        var splats = Scene(5, seed: 11);
        for (int i = 0; i < splats.Count; i++)
        {
            var s = splats[i];
            s.R = 0.5f; s.G = 0.5f; s.B = 0.5f;
            splats[i] = s;
        }

        float errBefore = ColourError(splats, truth);
        SplatOptimizer.FitColourOpacity(splats, target, W, H, iterations: 1500);
        float errAfter = ColourError(splats, truth);

        TestContext.Out.WriteLine($"mean |colour - truth|: {errBefore:F4} -> {errAfter:F4}");
        Assert.That(errAfter, Is.LessThan(errBefore * 0.6f),
            $"colours should move toward the truth: {errBefore:F4} -> {errAfter:F4}");

        static float ColourError(List<SplatRasterizer.Splat2D> a, List<SplatRasterizer.Splat2D> b)
        {
            float e = 0;
            for (int i = 0; i < a.Count; i++)
                e += MathF.Abs(a[i].R - b[i].R) + MathF.Abs(a[i].G - b[i].G) + MathF.Abs(a[i].B - b[i].B);
            return e / (a.Count * 3);
        }
    }

    [Test]
    public void OpacityIsOptimisedThroughTheSigmoid()
    {
        // Opacity lives as a logit. If the sigmoid chain rule (da/dlogit = a(1-a)) is missing,
        // the step size is wrong by a factor that varies with opacity, and opacity barely moves
        // where it matters most.
        var truth = Scene(4, seed: 5);
        for (int i = 0; i < truth.Count; i++)
        {
            var s = truth[i]; s.Opacity = 0.85f; truth[i] = s;
        }
        var target = SplatRasterizer.Render(truth, W, H).Colour;

        var splats = Scene(4, seed: 5);
        for (int i = 0; i < splats.Count; i++)
        {
            var s = splats[i]; s.Opacity = 0.2f; splats[i] = s;
        }

        SplatOptimizer.FitColourOpacity(splats, target, W, H, iterations: 1200);

        float mean = splats.Average(s => s.Opacity);
        TestContext.Out.WriteLine($"mean opacity 0.20 -> {mean:F3} (truth 0.85)");
        Assert.That(mean, Is.GreaterThan(0.35f),
            $"opacity should climb toward the truth, got {mean:F3}");
        Assert.That(splats.All(s => s.Opacity is > 0f and <= 1f), Is.True,
            "sigmoid must keep opacity in range");
    }

    [Test]
    public void AnAlreadyCorrectSceneIsLeftAlone()
    {
        // Fitting a scene to its own render must not degrade it. A missing bias correction or a
        // sign error often shows up here as drift away from a perfect starting point.
        var splats = Scene(6, seed: 21);
        var target = SplatRasterizer.Render(splats, W, H).Colour;

        float before = SplatRasterizer.L1(SplatRasterizer.Render(splats, W, H).Colour, target);
        var history = SplatOptimizer.FitColourOpacity(splats, target, W, H, iterations: 200);

        Assert.That(before, Is.EqualTo(0f).Within(1e-6f), "fixture should start perfect");
        Assert.That(history[^1], Is.LessThan(0.02f),
            $"a perfect scene should stay near-perfect, ended at {history[^1]:F6}");
    }

    /// <summary>
    /// A splat with no gradient must not move when the guard is on - and MUST move when it is
    /// off, because that is the behaviour being questioned.
    ///
    /// Asserting both directions is the point. The geometry optimiser has always refused to step
    /// a splat this view never touched; the colour path never did, on the stated judgement that
    /// stale momentum is "harmless for colour". At batch size 1 over 26 views, a splat visible in
    /// one of them takes about 25 zero-gradient steps per cycle, and opacity is optimised in
    /// LOGIT space - so stale momentum walks splats in and out of visibility. Whether that is
    /// what makes held-out quality oscillate by 2.74 dB within a run is a measurement; this test
    /// only pins what the flag does.
    /// </summary>
    [Test]
    public void ZeroGradientSplatMovesUnlessGuarded()
    {
        static (List<SplatRasterizer.Splat2D> Splats, float[] Logits) Fresh()
        {
            var splats = new List<SplatRasterizer.Splat2D>
            {
                new() { R = 0.30f, G = 0.40f, B = 0.50f, Opacity = 0.60f },
                new() { R = 0.30f, G = 0.40f, B = 0.50f, Opacity = 0.60f },
            };
            return (splats, new[] { SplatOptimizer.Logit(0.60f), SplatOptimizer.Logit(0.60f) });
        }

        // Splat 0 carries a real gradient every step. Splat 1 is seen ONCE and then never again -
        // which is the round robin, and is the only way stale momentum exists at all.
        //
        // An earlier version of this test gave splat 1 zeros from the start and found it did not
        // move, which is correct and uninteresting: Adam's m and v both stay at zero, so the
        // update is 0/(0+eps). The drag comes from momentum a splat ALREADY has, decaying at
        // beta1 = 0.9 per step - about 7% of it still left 25 steps later, which is one cycle of
        // 26 views.
        var both = new float[6];
        var bothOp = new float[2];
        both[0] = 0.10f; both[1] = -0.05f; both[2] = 0.07f; bothOp[0] = 0.09f;
        both[3] = 0.08f; both[4] = 0.06f; both[5] = -0.04f; bothOp[1] = 0.05f;

        var onlyFirst = new float[6];
        var onlyFirstOp = new float[2];
        onlyFirst[0] = 0.10f; onlyFirst[1] = -0.05f; onlyFirst[2] = 0.07f; onlyFirstOp[0] = 0.09f;

        (var guarded, var guardedLogits) = Fresh();
        var withGuard = new SplatOptimizer(2) { SkipZeroGradient = true };
        withGuard.Step(guarded, guardedLogits, both, bothOp);
        var seenOnce = (guarded[1].R, guarded[1].G, guarded[1].B, guarded[1].Opacity);
        for (int i = 0; i < 10; i++) withGuard.Step(guarded, guardedLogits, onlyFirst, onlyFirstOp);

        (var plain, var plainLogits) = Fresh();
        var without = new SplatOptimizer(2);
        without.Step(plain, plainLogits, both, bothOp);
        for (int i = 0; i < 10; i++) without.Step(plain, plainLogits, onlyFirst, onlyFirstOp);

        // The gradient-bearing splat must move under BOTH, or the guard has broken the optimiser
        // rather than narrowed it.
        Assert.That(guarded[0].R, Is.Not.EqualTo(0.30f), "guarded run must still optimise splat 0");
        Assert.That(plain[0].R, Is.Not.EqualTo(0.30f));

        // Guarded: splat 1 took its one real step and then froze, exactly.
        Assert.That(guarded[1].R, Is.EqualTo(seenOnce.R));
        Assert.That(guarded[1].G, Is.EqualTo(seenOnce.G));
        Assert.That(guarded[1].B, Is.EqualTo(seenOnce.B));
        Assert.That(guarded[1].Opacity, Is.EqualTo(seenOnce.Opacity));

        // Unguarded: it moves. This is the current shipped behaviour, asserted so the diff
        // documents what is being questioned rather than silently changing it.
        bool moved = plain[1].R != seenOnce.R || plain[1].G != seenOnce.G
                     || plain[1].B != seenOnce.B || plain[1].Opacity != seenOnce.Opacity;
        Assert.That(moved, Is.True,
            "without the guard, a splat that was seen once keeps being dragged by the momentum " +
            "it earned then - if this ever stops being true the guard has nothing left to do");

        // And by how much, so the effect has a size rather than a direction. This is the drag a
        // splat takes per cycle it is not seen.
        double drift = Math.Abs(plain[1].R - seenOnce.R) + Math.Abs(plain[1].G - seenOnce.G)
                     + Math.Abs(plain[1].B - seenOnce.B) + Math.Abs(plain[1].Opacity - seenOnce.Opacity);
        TestContext.Out.WriteLine($"stale drift over 10 unseen steps: {drift:F6}");
    }
}
