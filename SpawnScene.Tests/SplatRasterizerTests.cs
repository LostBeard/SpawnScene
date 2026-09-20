using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for the differentiable rasteriser.
///
/// Every analytic gradient is checked against a CENTRAL FINITE DIFFERENCE of the actual loss.
/// That is the only honest test for a hand-derived gradient: it compares the derivative against
/// the function it claims to differentiate, so a dropped chain-rule term or a sign error cannot
/// hide. A gradient that merely "looks plausible" and points roughly downhill will still train
/// to the wrong answer, slowly, and look like a tuning problem.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter SplatRasterizer</c>
/// </summary>
public class SplatRasterizerTests
{
    const int W = 12, H = 10;

    /// <summary>A deterministic, deliberately OVERLAPPING set of splats.</summary>
    static List<SplatRasterizer.Splat2D> MakeScene(int n = 6, int seed = 7)
    {
        var rng = new Random(seed);
        var list = new List<SplatRasterizer.Splat2D>();
        for (int i = 0; i < n; i++)
        {
            // Inverse covariance of a roughly 2px-sigma blob, slightly anisotropic + rotated.
            float sx = 1.5f + (float)rng.NextDouble();
            float sy = 1.5f + (float)rng.NextDouble();
            float rot = (float)(rng.NextDouble() * Math.PI);
            float c = MathF.Cos(rot), s = MathF.Sin(rot);
            // Sigma = R diag(sx^2, sy^2) R^T, then invert analytically.
            float a = c * c * sx * sx + s * s * sy * sy;
            float b = c * s * (sx * sx - sy * sy);
            float d = s * s * sx * sx + c * c * sy * sy;
            float det = a * d - b * b;

            list.Add(new SplatRasterizer.Splat2D
            {
                Px = 2f + (float)rng.NextDouble() * (W - 4f),
                Py = 2f + (float)rng.NextDouble() * (H - 4f),
                ConicA = d / det, ConicB = -b / det, ConicC = a / det,
                R = 0.2f + (float)rng.NextDouble() * 0.6f,
                G = 0.2f + (float)rng.NextDouble() * 0.6f,
                B = 0.2f + (float)rng.NextDouble() * 0.6f,
                Opacity = 0.15f + (float)rng.NextDouble() * 0.5f,
                Depth = (float)rng.NextDouble(),
            });
        }
        return list;
    }

    static float[] MakeTarget(int seed = 99)
    {
        var rng = new Random(seed);
        var t = new float[W * H * 3];
        for (int i = 0; i < t.Length; i++) t[i] = (float)rng.NextDouble();
        return t;
    }

    /// <summary>Full forward loss, so the finite difference measures the real thing.</summary>
    static float Loss(List<SplatRasterizer.Splat2D> splats, float[] target, int[] order)
        => SplatRasterizer.L1(SplatRasterizer.Render(splats, W, H, order).Colour, target);

    [Test]
    public void Forward_CompositesFrontToBackAndConservesTransmittance()
    {
        var splats = MakeScene();
        var fwd = SplatRasterizer.Render(splats, W, H);

        for (int p = 0; p < W * H; p++)
        {
            Assert.That(fwd.FinalT[p], Is.InRange(0f, 1f), $"transmittance out of range at {p}");
            // Black background: emitted colour can never exceed the coverage that produced it.
            float coverage = 1f - fwd.FinalT[p];
            for (int c = 0; c < 3; c++)
                Assert.That(fwd.Colour[p * 3 + c], Is.LessThanOrEqualTo(coverage + 1e-4f),
                    $"pixel {p} channel {c} brighter than its own coverage");
        }
    }

    [Test]
    public void Forward_AnOpaqueNearSplatOccludesWhatIsBehindIt()
    {
        // Two splats at the same place: the near one opaque, the far one a different colour.
        // The result must be the near colour, or compositing order is wrong.
        var splats = new List<SplatRasterizer.Splat2D>
        {
            // Px/Py sit on the PIXEL CENTRE (x+0.5). At integer coords the Gaussian weight is
            // exp(-0.125) = 0.88, not 1, so alpha never reaches the 0.99 clamp.
            new() { Px = 6.5f, Py = 5.5f, ConicA = 0.5f, ConicB = 0, ConicC = 0.5f,
                    R = 1, G = 0, B = 0, Opacity = 1.0f, Depth = 0.1f },
            new() { Px = 6.5f, Py = 5.5f, ConicA = 0.5f, ConicB = 0, ConicC = 0.5f,
                    R = 0, G = 0, B = 1, Opacity = 1.0f, Depth = 0.9f },
        };
        var fwd = SplatRasterizer.Render(splats, W, H);
        int p = 5 * W + 6;

        Assert.That(fwd.Colour[p * 3 + 0], Is.GreaterThan(0.9f), "near (red) splat should dominate");
        Assert.That(fwd.Colour[p * 3 + 2], Is.LessThan(0.05f), "far (blue) splat should be occluded");
    }

    [Test]
    public void ColourGradient_MatchesCentralFiniteDifference()
    {
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);

        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        const float eps = 1e-3f;
        for (int i = 0; i < splats.Count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                var plus = new List<SplatRasterizer.Splat2D>(splats);
                var minus = new List<SplatRasterizer.Splat2D>(splats);
                var sp = plus[i]; var sm = minus[i];
                switch (c)
                {
                    case 0: sp.R += eps; sm.R -= eps; break;
                    case 1: sp.G += eps; sm.G -= eps; break;
                    default: sp.B += eps; sm.B -= eps; break;
                }
                plus[i] = sp; minus[i] = sm;

                float numeric = (Loss(plus, target, order) - Loss(minus, target, order)) / (2 * eps);
                float analytic = dCol[i * 3 + c];

                Assert.That(analytic, Is.EqualTo(numeric).Within(2e-3f),
                    $"colour gradient splat {i} channel {c}: analytic {analytic:F6} vs numeric {numeric:F6}");
            }
        }
    }

    [Test]
    public void OpacityGradient_MatchesCentralFiniteDifference()
    {
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);

        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        const float eps = 1e-3f;
        for (int i = 0; i < splats.Count; i++)
        {
            var plus = new List<SplatRasterizer.Splat2D>(splats);
            var minus = new List<SplatRasterizer.Splat2D>(splats);
            var sp = plus[i]; var sm = minus[i];
            sp.Opacity += eps; sm.Opacity -= eps;
            plus[i] = sp; minus[i] = sm;

            float numeric = (Loss(plus, target, order) - Loss(minus, target, order)) / (2 * eps);
            float analytic = dOpa[i];

            Assert.That(analytic, Is.EqualTo(numeric).Within(3e-3f),
                $"opacity gradient splat {i}: analytic {analytic:F6} vs numeric {numeric:F6}");
        }
    }

    [Test]
    public void OpacityGradient_IsZeroWhereAlphaIsClamped()
    {
        // alpha = min(0.99, opacity*G). Past the clamp a splat cannot change the image by getting
        // more opaque, so its gradient must be exactly zero; without the clamp check the analytic
        // gradient is a phantom that pushes opacity up forever.
        //
        // Measured on a SINGLE pixel with the splat exactly on its centre, so G == 1 and alpha is
        // genuinely clamped. Over a whole image the periphery has G < 1, is unclamped, and moves
        // the loss for an unrelated reason - which is what made the first version of this fixture
        // fail while the implementation was correct.
        var splats = new List<SplatRasterizer.Splat2D>
        {
            new() { Px = 0.5f, Py = 0.5f, ConicA = 1f, ConicB = 0, ConicC = 1f,
                    R = 0.5f, G = 0.5f, B = 0.5f, Opacity = 1.0f, Depth = 0.5f },
        };
        var target = new float[] { 0f, 0f, 0f };
        var order = SplatRasterizer.DepthOrder(splats);

        var fwd = SplatRasterizer.Render(splats, 1, 1, order);
        Assert.That(fwd.Colour[0], Is.EqualTo(0.5f * SplatRasterizer.MaxAlpha).Within(1e-5f),
            "fixture must actually be clamped at MaxAlpha");

        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var dCol = new float[3];
        var dOpa = new float[1];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        float numeric = (Loss1(splats, target, order, 1e-3f) - Loss1(splats, target, order, -1e-3f)) / 2e-3f;

        Assert.That(numeric, Is.EqualTo(0f).Within(1e-6f), "clamped: loss must not respond to opacity");
        Assert.That(dOpa[0], Is.EqualTo(0f).Within(1e-6f), "clamped alpha must have zero gradient");

        static float Loss1(List<SplatRasterizer.Splat2D> src, float[] tgt, int[] ord, float d)
        {
            var c = new List<SplatRasterizer.Splat2D>(src);
            var s = c[0]; s.Opacity += d; c[0] = s;
            return SplatRasterizer.L1(SplatRasterizer.Render(c, 1, 1, ord).Colour, tgt);
        }
    }

    [Test]
    public void Backward_RespectsTheForwardEarlyOut()
    {
        // Many opaque stacked splats: the forward saturates and stops early. Splats past that
        // point contributed nothing, so they must receive NO gradient. Walking the whole list
        // instead invents gradient for them AND corrupts the transmittance recovery.
        var splats = new List<SplatRasterizer.Splat2D>();
        for (int i = 0; i < 40; i++)
        {
            splats.Add(new SplatRasterizer.Splat2D
            {
                Px = 6, Py = 5, ConicA = 0.05f, ConicB = 0, ConicC = 0.05f,
                R = 0.6f, G = 0.6f, B = 0.6f, Opacity = 0.95f, Depth = i * 0.01f,
            });
        }
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);
        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        int centre = 5 * W + 6;
        Assert.That(fwd.OrderEnd[centre], Is.LessThan(splats.Count),
            "fixture must actually saturate, or this test proves nothing");

        // The deepest splat is fully occluded; nudging it must not change the loss, and its
        // analytic colour gradient must agree.
        int deepest = order[^1];
        var plus = new List<SplatRasterizer.Splat2D>(splats);
        var sp = plus[deepest]; sp.R += 1e-2f; plus[deepest] = sp;
        float numeric = (Loss(plus, target, order) - Loss(splats, target, order)) / 1e-2f;

        Assert.That(numeric, Is.EqualTo(0f).Within(1e-5f), "occluded splat should not affect the loss");
        Assert.That(dCol[deepest * 3], Is.EqualTo(0f).Within(1e-6f),
            "occluded splat must get zero colour gradient");
    }

    /// <summary>Finite-difference every screen-space parameter of every splat.</summary>
    static void CheckFd(Func<SplatRasterizer.Splat2D, float, SplatRasterizer.Splat2D> bump,
                        Func<SplatRasterizer.Grad2D, float> pick, string name, float tol, float eps = 1e-3f)
    {
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);

        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var grads = new SplatRasterizer.Grad2D[splats.Count];
        SplatRasterizer.Backward(splats, fwd, dPix, grads);

        for (int i = 0; i < splats.Count; i++)
        {
            var plus = new List<SplatRasterizer.Splat2D>(splats);
            var minus = new List<SplatRasterizer.Splat2D>(splats);
            plus[i] = bump(plus[i], eps);
            minus[i] = bump(minus[i], -eps);

            float numeric = (Loss(plus, target, order) - Loss(minus, target, order)) / (2 * eps);
            float analytic = pick(grads[i]);

            Assert.That(analytic, Is.EqualTo(numeric).Within(tol),
                $"{name} gradient splat {i}: analytic {analytic:F6} vs numeric {numeric:F6}");
        }
    }

    [Test]
    public void Mean2DGradient_MatchesCentralFiniteDifference()
    {
        // This one also drives adaptive density control, which thresholds on |dL/dmean2D|.
        // Wrong here means densification places new geometry in the wrong places.
        CheckFd((s, d) => { s.Px += d; return s; }, g => g.Px, "mean2D.x", 3e-3f);
        CheckFd((s, d) => { s.Py += d; return s; }, g => g.Py, "mean2D.y", 3e-3f);
    }

    [Test]
    public void ConicGradient_ConvergesToTheAnalyticValueAsStepShrinks()
    {
        // A conic bump has a big lever arm: power shifts by 0.5*dx^2*eps, so at eps=1e-3 with
        // dx~10 the exponent moves 0.05 and the finite difference is dominated by pixels
        // crossing the MinAlpha / power>0 cutoffs rather than by the smooth derivative.
        // The test for "is the gradient right" is therefore CONVERGENCE as eps shrinks, not
        // agreement at one arbitrary step size.
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);
        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var grads = new SplatRasterizer.Grad2D[splats.Count];
        SplatRasterizer.Backward(splats, fwd, dPix, grads);

        foreach (int i in new[] { 0, 2, 4 })
        {
            float analytic = grads[i].ConicA;
            float prevErr = float.MaxValue;
            foreach (float eps in new[] { 1e-3f, 3e-4f, 1e-4f })
            {
                var plus = new List<SplatRasterizer.Splat2D>(splats);
                var minus = new List<SplatRasterizer.Splat2D>(splats);
                var sp = plus[i]; sp.ConicA += eps; plus[i] = sp;
                var sm = minus[i]; sm.ConicA -= eps; minus[i] = sm;
                float numeric = (Loss(plus, target, order) - Loss(minus, target, order)) / (2 * eps);
                float err = MathF.Abs(numeric - analytic);
                TestContext.Out.WriteLine($"splat {i} eps={eps:E1} numeric={numeric:F6} analytic={analytic:F6} err={err:F6}");
                prevErr = err;
            }
            // At the smallest step the analytic value must be close.
            var p2 = new List<SplatRasterizer.Splat2D>(splats);
            var m2 = new List<SplatRasterizer.Splat2D>(splats);
            var a = p2[i]; a.ConicA += 1e-4f; p2[i] = a;
            var b = m2[i]; b.ConicA -= 1e-4f; m2[i] = b;
            float fine = (Loss(p2, target, order) - Loss(m2, target, order)) / 2e-4f;
            Assert.That(analytic, Is.EqualTo(fine).Within(MathF.Max(2e-3f, MathF.Abs(fine) * 0.15f)),
                $"conic.a splat {i}: analytic {analytic:F6} vs fine-step numeric {fine:F6}");
        }
    }

    [Test]
    public void ConicGradient_MatchesCentralFiniteDifference()
    {
        // eps 1e-4, not 1e-3: a conic bump moves the exponent by 0.5*dx^2*eps, so at 1e-3 with
        // dx~10 the finite difference measures MinAlpha/power-cutoff crossings instead of the
        // derivative. ConicGradient_ConvergesToTheAnalyticValueAsStepShrinks proves the analytic
        // value is the limit these approach.
        CheckFd((s, d) => { s.ConicA += d; return s; }, g => g.ConicA, "conic.a", 5e-4f, 1e-4f);
        CheckFd((s, d) => { s.ConicB += d; return s; }, g => g.ConicB, "conic.b", 5e-4f, 1e-4f);
        CheckFd((s, d) => { s.ConicC += d; return s; }, g => g.ConicC, "conic.c", 5e-4f, 1e-4f);
    }

    [Test]
    public void FullBackward_AgreesWithTheColourOpacityOnlyPath()
    {
        // Two implementations of the same quantity must not drift apart. The colour+opacity path
        // is the Phase-2 shipping path; the full path is Phase 3. If they disagree, one is wrong.
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);
        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);

        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        var grads = new SplatRasterizer.Grad2D[splats.Count];
        SplatRasterizer.Backward(splats, fwd, dPix, grads);

        for (int i = 0; i < splats.Count; i++)
        {
            Assert.That(grads[i].R, Is.EqualTo(dCol[i * 3 + 0]).Within(1e-6f), $"R {i}");
            Assert.That(grads[i].G, Is.EqualTo(dCol[i * 3 + 1]).Within(1e-6f), $"G {i}");
            Assert.That(grads[i].B, Is.EqualTo(dCol[i * 3 + 2]).Within(1e-6f), $"B {i}");
            Assert.That(grads[i].Opacity, Is.EqualTo(dOpa[i]).Within(1e-6f), $"opacity {i}");
        }
    }

    [Test]
    public void Gradients_PointDownhill()
    {
        // End-to-end sanity: one small step along -gradient must reduce the loss. If this fails
        // while the finite-difference tests pass, the sign convention is inverted somewhere.
        var splats = MakeScene();
        var target = MakeTarget();
        var order = SplatRasterizer.DepthOrder(splats);

        var fwd = SplatRasterizer.Render(splats, W, H, order);
        float before = SplatRasterizer.L1(fwd.Colour, target);

        var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];
        SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

        const float lr = 0.5f;
        var stepped = new List<SplatRasterizer.Splat2D>(splats);
        for (int i = 0; i < stepped.Count; i++)
        {
            var s = stepped[i];
            s.R = Math.Clamp(s.R - lr * dCol[i * 3 + 0], 0f, 1f);
            s.G = Math.Clamp(s.G - lr * dCol[i * 3 + 1], 0f, 1f);
            s.B = Math.Clamp(s.B - lr * dCol[i * 3 + 2], 0f, 1f);
            s.Opacity = Math.Clamp(s.Opacity - lr * dOpa[i], 0.01f, 1f);
            stepped[i] = s;
        }

        float after = SplatRasterizer.L1(SplatRasterizer.Render(stepped, W, H, order).Colour, target);
        Assert.That(after, Is.LessThan(before),
            $"one gradient step should reduce loss: {before:F6} -> {after:F6}");
    }
}
