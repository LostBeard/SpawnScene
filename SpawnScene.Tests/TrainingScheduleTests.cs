using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The probe schedule's whole job is to NOT alias with the round robin it samples. That is a
/// property, so assert the property rather than the three numbers it currently produces.
/// </summary>
[TestFixture]
public class TrainingScheduleTests
{
    [Test]
    public void ProbesLandOnDistinctViews()
    {
        // 26 is Bathroom's supervised count, 44 is drjohnson's, and the rest are there to catch
        // a divisor that happens to work for one cycle length and not another.
        foreach (int views in new[] { 3, 4, 5, 7, 8, 12, 26, 33, 44, 100 })
        {
            var probes = TrainingSchedule.ProbeIterations(1600, views);
            var sampled = probes.Select(it => it % views).ToHashSet();

            Assert.That(sampled, Has.Count.EqualTo(probes.Count),
                $"{views} views: two probes sample the SAME view - " +
                $"iterations [{string.Join(", ", probes.Order())}] " +
                $"give phases [{string.Join(", ", probes.Select(it => it % views).Order())}]");
            Assert.That(probes, Has.Count.EqualTo(3), $"{views} views: expected three probes");
        }
    }

    [Test]
    public void ProbesAreSpreadAcrossTheRun()
    {
        var probes = TrainingSchedule.ProbeIterations(1600, 26).Order().ToList();

        // One early, one around the middle, one near the end. A schedule that measured three
        // points in the first cycle would satisfy the distinct-view test and still be useless.
        Assert.That(probes[0], Is.LessThan(26), "first probe should be in the first cycle");
        Assert.That(probes[1], Is.InRange(700, 900), "second probe should be near the middle");
        Assert.That(probes[2], Is.GreaterThan(1600 - 26), "third probe should be in the last cycle");
    }

    [Test]
    public void TheOldScheduleWouldHaveFailed()
    {
        // it == 0 and it == viewCount * 2 are the same phase. This is the bug, written down as
        // a test so the diff shows what was wrong rather than only what replaced it.
        const int views = 26;
        Assert.That(0 % views, Is.EqualTo(views * 2 % views),
            "the old probe schedule sampled view 0 twice out of three times");
    }

    [Test]
    public void StaysInRangeOnShortAndDegenerateRuns()
    {
        foreach (var (iterations, views) in new[] { (1, 1), (5, 26), (26, 26), (1600, 1), (0, 26) })
        {
            var probes = TrainingSchedule.ProbeIterations(iterations, views);
            Assert.That(probes, Is.All.InRange(0, Math.Max(iterations - 1, 0)),
                $"{iterations} iterations over {views} views: probe outside the loop");
            if (iterations > 0)
                Assert.That(probes, Is.Not.Empty,
                    $"{iterations} iterations over {views} views: no probe at all");
        }
    }

    [Test]
    public void ExponentialLrHitsBothEndpoints()
    {
        const float init = 1.6e-4f, final = 1.6e-6f;
        Assert.That(TrainingSchedule.ExponentialLr(init, final, 0, 30000),
            Is.EqualTo(init).Within(1e-9f));
        Assert.That(TrainingSchedule.ExponentialLr(init, final, 30000, 30000),
            Is.EqualTo(final).Within(1e-9f));
    }

    [Test]
    public void ExponentialLrIsLogarithmicNotLinear()
    {
        // A rate is a multiplier, so the meaningful midpoint of 1.6e-4 and 1.6e-6 is 1.6e-5.
        // Linear interpolation would give 8.08e-5 - five times too high half way through the
        // run, which is exactly the geometry jitter this decay exists to remove.
        const float init = 1.6e-4f, final = 1.6e-6f;
        float mid = TrainingSchedule.ExponentialLr(init, final, 15000, 30000);
        Assert.That(mid, Is.EqualTo(1.6e-5f).Within(1e-9f));
        Assert.That(mid, Is.LessThan((init + final) * 0.5f * 0.5f), "this looks linear");
    }

    [Test]
    public void ExponentialLrIsMonotonicAndClamped()
    {
        const float init = 1.6e-4f, final = 1.6e-6f;
        float prev = float.MaxValue;
        for (int step = 0; step <= 40000; step += 500)
        {
            float lr = TrainingSchedule.ExponentialLr(init, final, step, 30000);
            Assert.That(lr, Is.LessThanOrEqualTo(prev + 1e-12f), $"rose at step {step}");
            Assert.That(lr, Is.InRange(final, init), $"out of range at step {step}");
            prev = lr;
        }
    }

    [Test]
    public void ExponentialLrHandlesDegenerateInputs()
    {
        Assert.That(TrainingSchedule.ExponentialLr(0f, 0f, 5, 100), Is.EqualTo(0f));
        Assert.That(TrainingSchedule.ExponentialLr(1e-4f, 1e-6f, 5, 0), Is.EqualTo(1e-4f));
    }
}
