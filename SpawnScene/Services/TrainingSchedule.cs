namespace SpawnScene.Services;

/// <summary>
/// When a training loop samples itself.
///
/// Separated from the page so it can be tested without a browser: a sampling schedule is
/// arithmetic, and the bug it exists to prevent - a schedule that aliases with the round robin
/// it samples - is exactly the kind a test catches and a run does not.
/// </summary>
public static class TrainingSchedule
{
    /// <summary>
    /// Iterations to probe gradient health at, chosen to land on DIFFERENT views.
    ///
    /// The probe used to fire at <c>it == 0</c>, <c>it == supervised.Count * 2</c> and the last
    /// iteration. The first two are the same phase of the round robin, so two of the three
    /// samples were the same view - and on Bathroom that view produced no gradients at all,
    /// which read as "the run has no gradients" rather than "view 0 has none". A schedule whose
    /// samples alias with the thing being sampled measures one point three times.
    ///
    /// So: one probe early, one mid, one late, at three distinct phases. With a single
    /// supervised view they necessarily collapse, which is correct - there is only one view.
    /// </summary>
    public static HashSet<int> ProbeIterations(int iterations, int viewCount)
    {
        var probes = new HashSet<int>();
        if (iterations <= 0 || viewCount <= 0) return probes;

        // Distinct phases, spread across the cycle rather than adjacent, so the views sampled
        // are unrelated in the capture order too.
        int[] phases = [0, viewCount / 3, 2 * viewCount / 3];
        int[] anchors = [0, iterations / 2, iterations - 1];

        for (int i = 0; i < 3; i++)
        {
            int phase = phases[i] % viewCount;
            int anchor = anchors[i];
            // Nearest iteration at or before the anchor with that phase, clamped into range.
            int it = anchor - ((anchor - phase) % viewCount + viewCount) % viewCount;
            if (it < 0) it += viewCount;
            if (it < iterations) probes.Add(it);
        }

        // Fewer than three distinct views available: a duplicate is honest, an empty set is not.
        if (probes.Count == 0) probes.Add(0);
        return probes;
    }

    /// <summary>
    /// Reorder <paramref name="order"/> into a fresh uniform random permutation: the order the views of one
    /// epoch are trained in. The reference (train.py) pops a random view from <c>viewpoint_stack</c> each
    /// iteration and refills the stack when it empties, which is exactly one uniform permutation per epoch.
    ///
    /// The alternative, file order every epoch, feeds Adam neighbouring views back to back whenever the
    /// photos were taken along a path (Truck, video frames): the moments then average a sweep of one side of
    /// the scene instead of the whole rig. Fisher-Yates; permutes whatever it is given, so the caller's
    /// previous order is irrelevant to the result's distribution.
    /// </summary>
    public static void ShuffleEpoch(int[] order, Random rng)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    /// <summary>
    /// Exponential learning-rate decay, the reference's <c>get_expon_lr_func</c>.
    ///
    /// 3DGS decays the POSITION rate by 100x across training - 1.6e-4 to 1.6e-6 - and holds the
    /// others fixed, because position is the only parameter whose units are world-scale and the
    /// only one that can keep jittering geometry that should be settling. A constant rate means
    /// the last iteration moves a splat as far as the first one did.
    ///
    /// The interpolation is LOGARITHMIC, not linear: a rate is a multiplier, so the meaningful
    /// midpoint of 1.6e-4 and 1.6e-6 is 1.6e-5, not 8.1e-5.
    /// </summary>
    public static float ExponentialLr(float lrInit, float lrFinal, int step, int maxSteps)
    {
        if (lrInit <= 0f || lrFinal <= 0f) return 0f;
        if (maxSteps <= 0) return lrInit;

        float t = Math.Clamp(step / (float)maxSteps, 0f, 1f);
        float lr = MathF.Exp(MathF.Log(lrInit) * (1f - t) + MathF.Log(lrFinal) * t);

        // exp(log(x)) does not round-trip exactly in f32: at t=0 this returns 1.60000005e-4 for
        // an init of 1.6e-4. Tiny, but a rate that can exceed its own declared start is a
        // property nobody should have to reason about.
        return Math.Clamp(lr, MathF.Min(lrInit, lrFinal), MathF.Max(lrInit, lrFinal));
    }
}
