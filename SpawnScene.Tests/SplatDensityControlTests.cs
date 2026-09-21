using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The clone / split / prune rules.
///
/// These decide what geometry exists, so the failure mode is not a wrong pixel - it is a scene
/// that grows without bound, or never grows at all, or quietly deletes the Gaussians that were
/// doing the work. Each rule is therefore tested in isolation on a fixture where only that rule
/// can fire, and the interactions (prune beating densify, the budget) are tested separately.
/// </summary>
public class SplatDensityControlTests
{
    const float Extent = 1.0f;                                   // scene extent
    const float Small = SplatDensityControl.PercentDense * Extent * 0.5f;   // clones
    const float Large = SplatDensityControl.PercentDense * Extent * 2f;     // splits
    const float BigGrad = SplatDensityControl.GradientThresholdNdc * 2f;
    const float SmallGrad = SplatDensityControl.GradientThresholdNdc * 0.5f;

    static SplatDensityControl.Splat Splat(float scale, float opacity = 0.5f) => new()
    {
        PosX = 1f, PosY = 2f, PosZ = 3f,
        ScaleX = scale, ScaleY = scale, ScaleZ = scale,
        QuatX = 0f, QuatY = 0f, QuatZ = 0f, QuatW = 1f,
        Opacity = opacity,
    };

    static SplatDensityControl.Accumulator Acc(float avgGrad, int visible = 4, float radiusPx = 1f) => new()
    {
        GradientSum = avgGrad * visible,
        VisibleCount = visible,
        MaxScreenRadiusPx = radiusPx,
    };

    /// <summary>Deterministic "normal" draws, so a split's placement is checkable.</summary>
    static Func<float> Deviates(params float[] values)
    {
        int i = 0;
        return () => values[i++ % values.Length];
    }

    static SplatDensityControl.Plan Decide(
        SplatDensityControl.Splat[] splats,
        SplatDensityControl.Accumulator[] stats,
        bool afterReset = false,
        Func<float>? deviates = null,
        int maxSplats = int.MaxValue)
        => SplatDensityControl.Decide(
            splats, stats, Extent, afterReset, deviates ?? Deviates(0.5f), maxSplats);

    [Test]
    public void ASmallHighGradientGaussianIsCloned()
    {
        var plan = Decide(new[] { Splat(Small) }, new[] { Acc(BigGrad) });

        Assert.That(plan.Cloned, Is.EqualTo(1));
        Assert.That(plan.Split, Is.Zero);
        Assert.That(plan.Remove, Is.Empty, "a clone keeps its parent");
        Assert.That(plan.Add, Has.Count.EqualTo(1));
        // The copy starts exactly where the parent is; the optimiser separates them.
        Assert.That(plan.Add[0].PosX, Is.EqualTo(1f).Within(1e-6f));
        Assert.That(plan.Add[0].ScaleX, Is.EqualTo(Small).Within(1e-9f));
    }

    [Test]
    public void ALargeHighGradientGaussianIsSplit()
    {
        var plan = Decide(new[] { Splat(Large) }, new[] { Acc(BigGrad) });

        Assert.That(plan.Split, Is.EqualTo(1));
        Assert.That(plan.Cloned, Is.Zero);
        Assert.That(plan.Add, Has.Count.EqualTo(2), "a split makes two children");
        Assert.That(plan.Remove, Is.EqualTo(new[] { 0 }), "and removes the parent");

        foreach (var c in plan.Add)
            Assert.That(c.ScaleX, Is.EqualTo(Large / SplatDensityControl.SplitScaleDivisor).Within(1e-9f));
    }

    [Test]
    public void ALowGradientGaussianIsLeftAlone()
    {
        // The whole point of the gradient test. Densifying regardless would double the scene
        // every step and reach the memory ceiling before it reached quality.
        var plan = Decide(
            new[] { Splat(Small), Splat(Large) },
            new[] { Acc(SmallGrad), Acc(SmallGrad) });

        Assert.That(plan.Cloned, Is.Zero);
        Assert.That(plan.Split, Is.Zero);
        Assert.That(plan.Add, Is.Empty);
        Assert.That(plan.Remove, Is.Empty);
    }

    [Test]
    public void SplitChildrenAreDisplacedByTheParentsOwnDistribution()
    {
        // An anisotropic, ROTATED parent: a child offset that ignored the rotation, or used one
        // scale for all three axes, still looks plausible and is wrong. Half a turn about Z
        // maps local +x to world -x and local +y to world -y.
        var parent = new SplatDensityControl.Splat
        {
            PosX = 0f, PosY = 0f, PosZ = 0f,
            ScaleX = 0.4f, ScaleY = 0.2f, ScaleZ = 0.1f,
            QuatX = 0f, QuatY = 0f, QuatZ = 1f, QuatW = 0f,   // 180 degrees about Z
            Opacity = 0.5f,
        };
        // One deviate per axis per child: child A gets (1,0,0), child B gets (0,1,0).
        var plan = Decide(new[] { parent }, new[] { Acc(BigGrad) },
            deviates: Deviates(1f, 0f, 0f, 0f, 1f, 0f));

        Assert.That(plan.Add, Has.Count.EqualTo(2));
        var a = plan.Add[0];
        var b = plan.Add[1];

        // local (0.4, 0, 0) rotated 180 about Z -> world (-0.4, 0, 0)
        Assert.That(a.PosX, Is.EqualTo(-0.4f).Within(1e-5f));
        Assert.That(a.PosY, Is.EqualTo(0f).Within(1e-5f));
        // local (0, 0.2, 0) rotated 180 about Z -> world (0, -0.2, 0)
        Assert.That(b.PosX, Is.EqualTo(0f).Within(1e-5f));
        Assert.That(b.PosY, Is.EqualTo(-0.2f).Within(1e-5f));
    }

    [Test]
    public void AFaintGaussianIsPrunedAndNeverDensified()
    {
        // Opacity below the floor AND a large gradient. Densifying something that is about to
        // be deleted would leave orphan children with no parent geometry behind them.
        var plan = Decide(
            new[] { Splat(Small, opacity: SplatDensityControl.MinOpacity * 0.5f) },
            new[] { Acc(BigGrad) });

        Assert.That(plan.PrunedOpacity, Is.EqualTo(1));
        Assert.That(plan.Remove, Is.EqualTo(new[] { 0 }));
        Assert.That(plan.Add, Is.Empty, "a pruned Gaussian must not also be cloned");
        Assert.That(plan.Cloned, Is.Zero);
    }

    [Test]
    public void SizePrunesOnlyApplyAfterTheFirstOpacityReset()
    {
        // Before the first reset a big Gaussian may simply not have had the chance to shrink.
        // Pruning it then deletes real geometry the optimiser was still working on.
        var bloated = Splat(SplatDensityControl.MaxWorldSizeFraction * Extent * 2f);
        var wide = Acc(SmallGrad, radiusPx: SplatDensityControl.MaxScreenRadiusPx * 2f);

        var before = Decide(new[] { bloated }, new[] { wide }, afterReset: false);
        Assert.That(before.PrunedTooBig, Is.Zero);
        Assert.That(before.Remove, Is.Empty);

        var after = Decide(new[] { bloated }, new[] { wide }, afterReset: true);
        Assert.That(after.PrunedTooBig, Is.EqualTo(1));
        Assert.That(after.Remove, Is.EqualTo(new[] { 0 }));
    }

    [Test]
    public void TheAverageIsOverVISIBLEIterationsOnly()
    {
        // Two identical Gaussians with the same per-visible-iteration gradient, one seen four
        // times as often. Dividing by the iteration count instead would make the rarely-seen
        // one look four times less urgent purely because of where the cameras are.
        var often = new SplatDensityControl.Accumulator
        { GradientSum = BigGrad * 8, VisibleCount = 8 };
        var rarely = new SplatDensityControl.Accumulator
        { GradientSum = BigGrad * 2, VisibleCount = 2 };

        Assert.That(often.AverageGradient, Is.EqualTo(rarely.AverageGradient).Within(1e-9f));

        var plan = Decide(new[] { Splat(Small), Splat(Small) }, new[] { often, rarely });
        Assert.That(plan.Cloned, Is.EqualTo(2), "both should densify, or neither");
    }

    [Test]
    public void AGaussianNeverSeenHasNoGradientAndIsNotDensified()
    {
        var unseen = new SplatDensityControl.Accumulator { GradientSum = 0f, VisibleCount = 0 };
        Assert.That(unseen.AverageGradient, Is.Zero, "must not divide by zero");

        var plan = Decide(new[] { Splat(Small) }, new[] { unseen });
        Assert.That(plan.Add, Is.Empty);
        Assert.That(plan.Remove, Is.Empty);
    }

    [Test]
    public void TheBudgetCapsGrowthButNotPruning()
    {
        // A browser tab has a memory ceiling that densification does not know about. Growth
        // must stop at the cap while pruning still runs, so the set can recover.
        var splats = new[]
        {
            Splat(Small, opacity: SplatDensityControl.MinOpacity * 0.5f),
            Splat(Small), Splat(Small), Splat(Small),
        };
        var stats = new[] { Acc(BigGrad), Acc(BigGrad), Acc(BigGrad), Acc(BigGrad) };

        var plan = Decide(splats, stats, maxSplats: splats.Length + 1);

        Assert.That(plan.Add, Has.Count.EqualTo(1), "only one clone fits in the budget");
        Assert.That(plan.PrunedOpacity, Is.EqualTo(1), "pruning is not budgeted");
    }

    [Test]
    public void ApplyRemovesAndAppendsAgainstTheORIGINALIndices()
    {
        // Remove indices refer to the input array. Applying them one at a time would shift
        // every later index and delete the wrong Gaussians.
        var splats = new[] { Splat(Small, 0.1f), Splat(Small, 0.2f), Splat(Small, 0.3f) };
        var plan = new SplatDensityControl.Plan();
        plan.Remove.Add(0);
        plan.Remove.Add(2);
        plan.Add.Add(Splat(Small, 0.9f));

        var result = SplatDensityControl.Apply(splats, plan);

        Assert.That(result, Has.Count.EqualTo(2));
        Assert.That(result[0].Opacity, Is.EqualTo(0.2f).Within(1e-6f), "the survivor is index 1");
        Assert.That(result[1].Opacity, Is.EqualTo(0.9f).Within(1e-6f), "then the addition");
    }

    [Test]
    public void ResetOpacityCapsWithoutRaising()
    {
        var splats = new[] { Splat(Small, 0.9f), Splat(Small, 0.001f) };
        SplatDensityControl.ResetOpacity(splats);

        Assert.That(splats[0].Opacity, Is.EqualTo(SplatDensityControl.OpacityResetTo).Within(1e-9f));
        Assert.That(splats[1].Opacity, Is.EqualTo(0.001f).Within(1e-9f),
            "a reset is a cap, not an assignment - it must never RAISE an opacity");
    }

    [Test]
    public void PixelGradientsAreConvertedToNdcBeforeComparison()
    {
        // The published threshold is in NDC, where the frame spans [-1, 1]. Comparing a raw
        // pixel-space gradient against it sets the bar about 320x too high at 640 wide, and
        // nothing ever densifies - which reads as "densification does not help".
        const int w = 640, h = 480;

        float ndc = SplatDensityControl.PixelGradientToNdc(1f, 0f, w, h);
        Assert.That(ndc, Is.EqualTo(w * 0.5f).Within(1e-3f));

        // The pixel gradient that sits exactly on the threshold.
        float onThreshold = SplatDensityControl.GradientThresholdNdc / (w * 0.5f);
        Assert.That(SplatDensityControl.PixelGradientToNdc(onThreshold, 0f, w, h),
            Is.EqualTo(SplatDensityControl.GradientThresholdNdc).Within(1e-9f));

        // Both axes contribute; it is a magnitude, not a per-axis test.
        float both = SplatDensityControl.PixelGradientToNdc(2f / w, 2f / h, w, h);
        Assert.That(both, Is.EqualTo(MathF.Sqrt(2f)).Within(1e-4f));
    }

    [Test]
    public void MismatchedInputsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => SplatDensityControl.Decide(
            new[] { Splat(Small), Splat(Small) },
            new[] { Acc(BigGrad) },
            Extent, false, Deviates(0f)));
    }
}
