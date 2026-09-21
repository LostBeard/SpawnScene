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
}
