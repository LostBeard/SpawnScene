using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The training size a run uses under a GPU memory budget, and the project page's training-time estimate.
/// </summary>
public class TrainingTimeEstimateTests
{
    const long MiB = 1024 * 1024;

    [Test]
    public void FitsTheBudget_TrainsAtTheSetting()
    {
        // TJ's Bathroom: 35 portrait 3120x4160 photos at the Standard 1024 px.
        var (w, h, shrunk) = TrainingTimeEstimate.TrainingSize(3120, 4160, 3120, 1024, 35, 512 * MiB);
        Assert.That((w, h, shrunk), Is.EqualTo((768, 1024, false)));
    }

    [Test]
    public void OverTheBudget_TrainsAtTheLargestSizeThatFits()
    {
        // TruckFull: 251 views of 979x546 photos at the High 1600 px, 256 MB for the photo stack. The old 3/4 steps from
        // 1600 px gave 676x376; 692x386 fits (268,180,448 bytes of 268,435,456).
        var (w, h, shrunk) = TrainingTimeEstimate.TrainingSize(979, 546, 979, 1600, 251, 256 * MiB);
        Assert.That((w, h, shrunk), Is.EqualTo((692, 386, true)));
        Assert.That(251L * w * h * 4, Is.LessThanOrEqualTo(256 * MiB));
        // One even step larger does not fit.
        Assert.That(251L * (w + 2) * (h + 2) * 4, Is.GreaterThan(256 * MiB));
    }

    [Test]
    public void TinyBudget_StopsAt128()
    {
        var (w, h, shrunk) = TrainingTimeEstimate.TrainingSize(1000, 1000, 1000, 1000, 100_000, 1);
        Assert.That((w, h, shrunk), Is.EqualTo((128, 128, true)));
    }

    [Test]
    public void Estimate_NothingRecorded_IsNull() =>
        Assert.That(TrainingTimeEstimate.Estimate(new Dictionary<int, double>(), 7000, 0.5), Is.Null);

    [Test]
    public void Estimate_InterpolatesTheMarks()
    {
        // Per-megapixel seconds that grow faster than the iterations, like a densifying run.
        var marks = new Dictionary<int, double> { [1000] = 20, [3000] = 100, [7000] = 400 };
        // At a mark, exactly that mark; scaled by the megapixels.
        Assert.That(TrainingTimeEstimate.Estimate(marks, 3000, 0.5), Is.EqualTo(new TrainingTimeEstimate.Duration(50, false)));
        // Between 3000 and 7000: 100 + 300 * 2000/4000 = 250 per megapixel.
        Assert.That(TrainingTimeEstimate.Estimate(marks, 5000, 1.0), Is.EqualTo(new TrainingTimeEstimate.Duration(250, false)));
        // Before the first mark: from (0, 0).
        Assert.That(TrainingTimeEstimate.Estimate(marks, 500, 1.0), Is.EqualTo(new TrainingTimeEstimate.Duration(10, false)));
    }

    [Test]
    public void Estimate_PastTheLongestRun_IsALowerBoundFromTheLastStretch()
    {
        // b123's real case: one 1,000-iteration run of a small scene (99 it/s). A single rate from it predicted a 7K
        // TruckFull run at ~1 min; real 7K runs take minutes because the splat count keeps growing. So past the last mark
        // the estimate extends the last stretch and says AT LEAST.
        var marks = new Dictionary<int, double> { [1000] = 20, [3000] = 100 };
        var d = TrainingTimeEstimate.Estimate(marks, 7000, 1.0);
        Assert.That(d, Is.EqualTo(new TrainingTimeEstimate.Duration(100 + 4000 * (80.0 / 2000), true)));
        var one = TrainingTimeEstimate.Estimate(new Dictionary<int, double> { [1000] = 20 }, 3000, 1.0);
        Assert.That(one, Is.EqualTo(new TrainingTimeEstimate.Duration(60, true)));
    }

    [Test]
    public void Merge_ShortRunKeepsLongerMarks_DropsInconsistentOnes()
    {
        var old = new Dictionary<int, double> { [1000] = 20, [3000] = 100, [7000] = 400 };
        // A later 3K run: its two marks replace, 7000 stays.
        var merged = TrainingTimeEstimate.Merge(old, new Dictionary<int, double> { [1000] = 25, [3000] = 110 });
        Assert.That(merged, Is.EqualTo(new Dictionary<int, double> { [1000] = 25, [3000] = 110, [7000] = 400 }));
        // A run whose 1000 mark (150) is slower than the old 3000 (100): that 3000 would make time run backwards, so it
        // goes; the old 7000 (400) is still later and stays.
        merged = TrainingTimeEstimate.Merge(old, new Dictionary<int, double> { [1000] = 150 });
        Assert.That(merged, Is.EqualTo(new Dictionary<int, double> { [1000] = 150, [7000] = 400 }));
    }

    [TestCase(30, false, "under a minute")]
    [TestCase(30, true, "a minute or more")]
    [TestCase(421, false, "about 7 min")]
    [TestCase(1881.7, false, "about 31 min")]
    [TestCase(1881.7, true, "at least about 31 min")]
    [TestCase(3590, false, "about 1 h")]
    [TestCase(4200, false, "about 1 h 10 min")]
    [TestCase(7000, false, "about 1 h 55 min")]
    [TestCase(7150, false, "about 2 h")]
    public void Describe(double seconds, bool atLeast, string expected) =>
        Assert.That(TrainingTimeEstimate.Describe(new TrainingTimeEstimate.Duration(seconds, atLeast)), Is.EqualTo(expected));
}
