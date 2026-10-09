using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The size a run trains at, and how long its training takes on this device.
///
/// An iteration's cost grows with the splat count, and densification grows that count through the run: TruckFull went
/// 46K -> 1.1M splats in its first half (b108/b121), and a 1,000-iteration run of 89K splats ran at 99 it/s where a full
/// one runs at 16-24 (b123). So one rate does not stretch across run lengths. Each run records instead its elapsed time
/// at the <see cref="Marks"/> it reaches, per training megapixel. An estimate interpolates those marks (the growth as it
/// happened on this device). Past the last mark it extends the last stretch's rate and says "at least", since splats are
/// still growing there.
/// </summary>
public static class TrainingTimeEstimate
{
    /// <summary>The iteration counts a run records its elapsed time at (the presets' iteration choices).</summary>
    public static readonly int[] Marks = [1000, 3000, 7000, 15000, 30000];

    /// <summary>A training time: <see cref="AtLeast"/> when it extends past the longest run this device has measured.</summary>
    public readonly record struct Duration(double Seconds, bool AtLeast);

    /// <summary>
    /// The training size for <paramref name="viewCount"/> views of <paramref name="width"/> x <paramref name="height"/>:
    /// <see cref="CameraParams.TrainingSize(int, int, int, int)"/>, or, when the resident target stack (every view, 4 bytes
    /// a pixel) would not fit <paramref name="targetStackBytes"/>, the largest size that does (128 px at the least).
    /// Returns the size and whether the budget shrank it.
    /// </summary>
    public static (int Width, int Height, bool Shrunk) TrainingSize(int width, int height, int sourceLongestSide,
        int maxDimension, int viewCount, long targetStackBytes, int sourceShortestSide = 0)
    {
        long Bytes(int w, int h) => (long)viewCount * w * h * sizeof(uint);
        var (tw, th) = CameraParams.TrainingSize(width, height, maxDimension, sourceLongestSide, sourceShortestSide);
        if (Bytes(tw, th) <= targetStackBytes) return (tw, th, false);
        // Bytes grow with the square of the longest side, so the fitting side is one square root away. This replaced 3/4
        // steps down from the SETTING, which landed wherever the steps fell: TruckFull's 251 views at 1600 px (photos 979)
        // in 256 MB trained at 676 px where 692 fits.
        int fit = Math.Max(128, (int)(Math.Max(tw, th) * Math.Sqrt((double)targetStackBytes / Bytes(tw, th))));
        (tw, th) = CameraParams.TrainingSize(width, height, fit, sourceLongestSide);
        while (fit > 128 && Bytes(tw, th) > targetStackBytes) // rounding to even sizes can land just over
        {
            fit--;
            (tw, th) = CameraParams.TrainingSize(width, height, fit, sourceLongestSide);
        }
        return (tw, th, true);
    }

    /// <summary>Training megapixels of one <paramref name="width"/> x <paramref name="height"/> view.</summary>
    public static double Megapixels(int width, int height) => width * (double)height / 1e6;

    /// <summary>
    /// Training time for <paramref name="iterations"/> at <paramref name="megapixels"/> from the recorded
    /// <paramref name="secondsPerMegapixelAtMark"/> (mark -> elapsed seconds per training megapixel), or null when nothing
    /// is recorded yet.
    /// </summary>
    public static Duration? Estimate(IReadOnlyDictionary<int, double> secondsPerMegapixelAtMark, int iterations, double megapixels)
    {
        var marks = secondsPerMegapixelAtMark.Where(kv => kv.Key > 0 && kv.Value > 0).OrderBy(kv => kv.Key).ToList();
        if (marks.Count == 0 || iterations <= 0) return null;
        // Linear between marks, from (0, 0).
        int prevIt = 0;
        double prevS = 0;
        foreach (var (mark, s) in marks)
        {
            if (iterations <= mark)
                return new Duration((prevS + (s - prevS) * (iterations - prevIt) / (mark - prevIt)) * megapixels, false);
            prevIt = mark;
            prevS = s;
        }
        // Past the longest measured run: the last stretch's per-iteration rate, a lower bound while splats still grow.
        var (lastIt, lastS) = marks[^1];
        var (beforeIt, beforeS) = marks.Count > 1 ? (marks[^2].Key, marks[^2].Value) : (0, 0.0);
        double perIteration = (lastS - beforeS) / (lastIt - beforeIt);
        return new Duration((lastS + perIteration * (iterations - lastIt)) * megapixels, true);
    }

    /// <summary>
    /// The marks after a run: <paramref name="recorded"/> with the ones this run reached replaced. A shorter run replaces
    /// only its own marks and keeps a longer run's later ones.
    /// </summary>
    public static Dictionary<int, double> Merge(IReadOnlyDictionary<int, double> recorded, IReadOnlyDictionary<int, double> run)
    {
        var merged = new Dictionary<int, double>(recorded);
        foreach (var (mark, s) in run) merged[mark] = s;
        // A mark later than a fresh one but no slower would be from a faster (smaller) scene: drop it rather than
        // interpolate backwards in time.
        double floor = 0;
        foreach (var mark in merged.Keys.OrderBy(k => k).ToList())
        {
            if (merged[mark] <= floor) merged.Remove(mark);
            else floor = merged[mark];
        }
        return merged;
    }

    /// <summary>A rough duration for the UI: "under a minute", "about 7 min", "at least about 1 h 10 min".</summary>
    public static string Describe(Duration duration)
    {
        double seconds = duration.Seconds;
        string prefix = duration.AtLeast ? "at least about " : "about ";
        if (seconds < 60) return duration.AtLeast ? "a minute or more" : "under a minute";
        int minutes = (int)Math.Round(seconds / 60);
        if (minutes < 60) return $"{prefix}{minutes} min";
        int h = minutes / 60, m = minutes % 60;
        // Past an hour, five-minute steps: the model is not that precise.
        m = (int)Math.Round(m / 5.0) * 5;
        if (m == 60) { h++; m = 0; }
        return m == 0 ? $"{prefix}{h} h" : $"{prefix}{h} h {m} min";
    }
}
