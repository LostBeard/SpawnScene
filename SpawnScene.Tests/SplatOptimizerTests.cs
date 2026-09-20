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
}
