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
    // Not const: GradientThresholdNdc is a tunable static so a run can A/B it.
    static float BigGrad => SplatDensityControl.GradientThresholdNdc * 2f;
    static float SmallGrad => SplatDensityControl.GradientThresholdNdc * 0.5f;

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
    public void DensifyGradientIsTheReferenceNdcNorm()
    {
        // Kerbl's backward.cu accumulates dL/dmean2D scaled by ddelx_dx = 0.5*W and
        // ddely_dy = 0.5*H, and densify_grad_threshold = 0.0002 is applied to the norm of that.
        // The helper is the CPU mirror of densify_accum and must apply the same scaling.
        Assert.That(SplatDensityControl.PixelGradientMagnitude(3f, 4f), Is.EqualTo(5f).Within(1e-6f));

        // 640x480: x scaled by 320, y by 240 -> (960, 960) -> norm 960*sqrt(2).
        Assert.That(SplatDensityControl.PixelGradientToNdc(3f, 4f, 640, 480),
            Is.EqualTo(960f * MathF.Sqrt(2f)).Within(1e-2f));

        // Default bar is the published one.
        Assert.That(SplatDensityControl.GradientThreshold, Is.EqualTo(2e-4f).Within(1e-12f));
    }

    [Test]
    public void GrowthSelectFractionKeepsOnlyTheHighestGradientCandidates()
    {
        // Brush densifies a fraction of above-threshold candidates; without that cap a weak
        // gradient signal clones most of the scene every densify step.
        float prevFrac = SplatDensityControl.GrowthSelectFraction;
        float prevThr = SplatDensityControl.GradientThreshold;
        try
        {
            SplatDensityControl.GrowthSelectFraction = 0.25f;
            SplatDensityControl.GradientThreshold = 0.001f;

            var splats = new SplatDensityControl.Splat[8];
            var stats = new SplatDensityControl.Accumulator[8];
            for (int i = 0; i < 8; i++)
            {
                splats[i] = Splat(Small);
                // All above threshold; grads 0.008 .. 0.001 so the top 25% are indices 0 and 1.
                stats[i] = Acc(0.008f - i * 0.001f);
            }

            var plan = SplatDensityControl.Decide(splats, stats, Extent, false, Deviates(0f));
            Assert.That(plan.Cloned, Is.EqualTo(2),
                "0.25 of 8 candidates is 2 clones, highest gradient first");
            Assert.That(plan.Split, Is.EqualTo(0));
        }
        finally
        {
            SplatDensityControl.GrowthSelectFraction = prevFrac;
            SplatDensityControl.GradientThreshold = prevThr;
        }
    }

    [Test]
    public void MismatchedInputsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => SplatDensityControl.Decide(
            new[] { Splat(Small), Splat(Small) },
            new[] { Acc(BigGrad) },
            Extent, false, Deviates(0f)));
    }

    /// <summary>
    /// The survivor map is what lets Adam momentum cross a densification. Getting it wrong is
    /// silent - every splat keeps training, just with someone else's momentum, or none.
    /// </summary>
    [Test]
    public void ApplyReportsWhereEverySurvivorCameFrom()
    {
        var splats = new SplatDensityControl.Splat[6];
        for (int i = 0; i < 6; i++)
            splats[i] = new SplatDensityControl.Splat
            {
                PosX = i, ScaleX = 0.01f, ScaleY = 0.01f, ScaleZ = 0.01f,
                QuatW = 1f, Opacity = 0.5f, ColR = i / 10f,
            };

        var plan = new SplatDensityControl.Plan();
        plan.Remove.Add(1);
        plan.Remove.Add(4);
        plan.Add.Add(splats[0]);   // a clone
        plan.Add.Add(splats[3]);   // another

        var grown = SplatDensityControl.Apply(splats, plan, out var survivors);

        Assert.That(grown, Has.Count.EqualTo(6));
        Assert.That(survivors, Has.Length.EqualTo(grown.Count),
            "one entry per NEW index, or the Adam copy walks off the end");

        // Survivors keep their old index, in order, with the removed ones gone.
        Assert.That(survivors[..4], Is.EqualTo(new[] { 0, 2, 3, 5 }));
        // Added splats have no prior Adam state.
        Assert.That(survivors[4], Is.EqualTo(-1));
        Assert.That(survivors[5], Is.EqualTo(-1));

        // And the mapping actually identifies the right splat, not just a plausible index.
        for (int i = 0; i < 4; i++)
            Assert.That(grown[i].PosX, Is.EqualTo((float)survivors[i]),
                $"new splat {i} maps to old index {survivors[i]} but is not that splat");
    }

    [Test]
    public void RemapFloatRows_CopiesSourcesAndZerosMissing()
    {
        // Two old splats, 3 floats each. New layout: keep 0, drop 1, add child of 0.
        float[] prior = [1f, 2f, 3f, 10f, 20f, 30f];
        int[] sources = [0, -1, 0]; // survivor 0, new child, clone of 0
        var next = SplatDensityControl.RemapFloatRows(prior, sources, stride: 3);
        Assert.That(next, Is.EqualTo(new float[] { 1f, 2f, 3f, 0f, 0f, 0f, 1f, 2f, 3f }));
    }

    [Test]
    public void Apply_CopiesFeatureSourceFromAddParent()
    {
        var splats = new SplatDensityControl.Splat[3];
        for (int i = 0; i < 3; i++)
            splats[i] = new SplatDensityControl.Splat
            {
                PosX = i, ScaleX = 0.01f, ScaleY = 0.01f, ScaleZ = 0.01f,
                QuatW = 1f, Opacity = 0.5f,
            };

        var plan = new SplatDensityControl.Plan();
        plan.Add.Add(splats[1]);
        plan.AddParent.Add(1);
        plan.Add.Add(splats[1]);
        plan.AddParent.Add(1);
        plan.Remove.Add(1); // split-style: parent gone, two children

        var grown = SplatDensityControl.Apply(splats, plan, out var adam, out var features);
        Assert.That(grown, Has.Count.EqualTo(4)); // 0,2 kept + 2 children
        Assert.That(adam, Is.EqualTo(new[] { 0, 2, -1, -1 }));
        Assert.That(features, Is.EqualTo(new[] { 0, 2, 1, 1 }),
            "children must inherit parent 1's SH rest, not start at zero");
    }

    [Test]
    public void Decide_RecordsAddParentForCloneAndSplit()
    {
        var splats = new SplatDensityControl.Splat[2];
        splats[0] = new SplatDensityControl.Splat
        {
            PosX = 0, ScaleX = 0.001f, ScaleY = 0.001f, ScaleZ = 0.001f,
            QuatW = 1f, Opacity = 0.5f,
        };
        // Large enough to split at sceneExtent=10 (PercentDense*10=0.1)
        splats[1] = new SplatDensityControl.Splat
        {
            PosX = 1, ScaleX = 0.2f, ScaleY = 0.2f, ScaleZ = 0.2f,
            QuatW = 1f, Opacity = 0.5f,
        };
        var stats = new SplatDensityControl.Accumulator[2];
        stats[0] = new SplatDensityControl.Accumulator { GradientSum = 1e-3f, VisibleCount = 1 };
        stats[1] = new SplatDensityControl.Accumulator { GradientSum = 1e-3f, VisibleCount = 1 };

        var plan = SplatDensityControl.Decide(splats, stats, sceneExtent: 10f,
            afterFirstOpacityReset: false, sampleUnitNormal: () => 0f);

        Assert.That(plan.AddParent, Has.Count.EqualTo(plan.Add.Count));
        Assert.That(plan.Cloned + plan.Split * 2, Is.EqualTo(plan.Add.Count));
        foreach (int p in plan.AddParent)
            Assert.That(p, Is.EqualTo(0).Or.EqualTo(1));
    }

    [Test]
    public void ApplyWithoutASurvivorMapStillWorks()
    {
        // The two-argument overload is used where the mapping is not needed; it must not
        // diverge from the three-argument one.
        var splats = new SplatDensityControl.Splat[3];
        for (int i = 0; i < 3; i++) splats[i] = new SplatDensityControl.Splat { PosX = i, QuatW = 1f };
        var plan = new SplatDensityControl.Plan();
        plan.Remove.Add(0);
        plan.Add.Add(splats[2]);

        var a = SplatDensityControl.Apply(splats, plan);
        var b = SplatDensityControl.Apply(splats, plan, out _);
        Assert.That(a.Select(x => x.PosX), Is.EqualTo(b.Select(x => x.PosX)));
    }

    [Test]
    public void ClonesAndSplitChildrenCarryTheParentColour()
    {
        // Colour lives in the Splat struct precisely so `var child = parent` propagates it.
        // Without it every new Gaussian is born black, which reads as the render darkening.
        var splats = new[]
        {
            new SplatDensityControl.Splat
            {
                PosX = 0, ScaleX = 0.001f, ScaleY = 0.001f, ScaleZ = 0.001f,
                QuatW = 1f, Opacity = 0.9f, ColR = 0.25f, ColG = 0.5f, ColB = 0.75f,
            },
        };
        var stats = new[]
        {
            new SplatDensityControl.Accumulator { GradientSum = 1f, VisibleCount = 1 },
        };

        var plan = SplatDensityControl.Decide(
            splats, stats, sceneExtent: 1f, afterFirstOpacityReset: false, () => 0.5f);

        Assert.That(plan.Add, Is.Not.Empty, "a high-gradient small splat should densify");
        foreach (var child in plan.Add)
        {
            Assert.That(child.ColR, Is.EqualTo(0.25f));
            Assert.That(child.ColG, Is.EqualTo(0.5f));
            Assert.That(child.ColB, Is.EqualTo(0.75f));
        }
    }

    [Test]
    public void PruneUnconstrainedDropsOnlyZeroSupport()
    {
        // The support census is the signal: a splat never constrained by any supervised view
        // cannot improve under photometric loss and is free to be wrong on held-out views.
        uint[] support = { 0, 1, 0, 2, 0, 4 };
        var plan = SplatDensityControl.PruneUnconstrained(support);

        Assert.That(plan.PrunedUnconstrained, Is.EqualTo(3));
        Assert.That(plan.Remove, Is.EqualTo(new[] { 0, 2, 4 }));
        Assert.That(plan.Add, Is.Empty);

        var splats = new SplatDensityControl.Splat[6];
        for (int i = 0; i < 6; i++)
            splats[i] = new SplatDensityControl.Splat { PosX = i, QuatW = 1f, Opacity = 0.5f };

        var grown = SplatDensityControl.Apply(splats, plan, out var survivors);
        Assert.That(grown.Select(s => (int)s.PosX), Is.EqualTo(new[] { 1, 3, 5 }));
        Assert.That(survivors, Is.EqualTo(new[] { 1, 3, 5 }));
    }

    [Test]
    public void PruneUnconstrainedIsANoOpWhenEverySplatIsSeen()
    {
        uint[] support = { 1, 2, 3 };
        var plan = SplatDensityControl.PruneUnconstrained(support);
        Assert.That(plan.Remove, Is.Empty);
        Assert.That(plan.PrunedUnconstrained, Is.EqualTo(0));
    }

    [Test]
    public void AbsCentreMagnitudeSurvivesOpposingPullsThatCancelInASignedSum()
    {
        // AbsGS densify must not use a signed tile sum (cancels) or a sum of |dCentre| over
        // pixels (scales with footprint → only LARGE Gaussians densify → 0 clones on Truck).
        // Peak |dCentre| is size-fair so small under-reconstructed Gaussians can CLONE.
        float thr = SplatDensityControl.GradientThreshold;
        float signedMean = ((+thr) + (-thr)) / 2f;
        float absMean = (Math.Abs(+thr) + Math.Abs(-thr)) / 2f;

        Assert.That(signedMean, Is.EqualTo(0f).Within(1e-9f));
        Assert.That(absMean, Is.EqualTo(thr).Within(1e-9f));

        var splat = new[] { Splat(Small) };
        var signedStats = new[] { Acc(signedMean, visible: 1) };
        var absStats = new[] { Acc(absMean, visible: 1) };

        Assert.That(Decide(splat, signedStats).Cloned, Is.EqualTo(0),
            "signed cancel must stay below the densify bar");
        Assert.That(Decide(splat, absStats).Cloned, Is.EqualTo(1),
            "peak absgrad magnitude must clear the densify bar for a SMALL splat");
    }

    [Test]
    public void LargeAndSmallHighGradBothDensifyAsSplitAndClone()
    {
        // Footprint decides clone vs split; the gradient bar must not silently exclude small.
        var splats = new[] { Splat(Small), Splat(Large) };
        var stats = new[] { Acc(BigGrad), Acc(BigGrad) };
        var plan = Decide(splats, stats);
        Assert.That(plan.Cloned, Is.EqualTo(1), "small + high grad -> clone");
        Assert.That(plan.Split, Is.EqualTo(1), "large + high grad -> split");
    }
}
