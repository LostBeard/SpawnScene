using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// <see cref="FeatureDetector"/> (per pyramid level: FAST-9 + grid non-max suppression + quota, intensity-centroid orientation,
/// steered BRIEF on a Gaussian-smoothed level) on
/// the device, reading a grayscale frame that never leaves it. Only the small results come back: the per-cell NMS
/// winners (3 ints per 8x8 cell) and the chosen features' 256-bit descriptors.
/// </summary>
/// <remarks>
/// Every step is the CPU detector's logic, transcribed: the same quick reject, the same 9-contiguous test and score,
/// the same first-strict-maximum per cell in row-major order, the SAME host sort for the top-N (so ties order exactly
/// as before), the same blur weights and accumulation order, the same BRIEF pairs and bit layout. On the ILGPU CPU
/// accelerator the output is identical to <see cref="FeatureDetector.Detect"/> (SpawnScene.Tests
/// GpuFeatureDetectorTests). On a GPU the blur is float math a shader compiler may contract into fused multiply-adds,
/// so a smoothed pixel can occasionally round one level differently and flip a BRIEF bit; FAST and NMS are integer.
/// </remarks>
public sealed class GpuFeatureDetector
{
    const int Margin = 4, CellSize = 8;

    private readonly int _maxFeatures, _threshold, _levels, _coarseFeatures;
    private readonly bool _oriented;
    /// <summary>See <see cref="FeatureDetector"/>'s constructor: level 0 keeps <paramref name="maxFeatures"/>, the
    /// coarser levels add <paramref name="coarseFeatures"/>.</summary>
    public GpuFeatureDetector(int maxFeatures = 2000, int fastThreshold = 25, int levels = 1, int coarseFeatures = 0,
        bool oriented = false)
    {
        _maxFeatures = maxFeatures;
        _coarseFeatures = coarseFeatures;
        _oriented = oriented;
        _threshold = fastThreshold;
        _levels = Math.Max(1, levels);
    }

    // ── kernels ──────────────────────────────────────────────────────────────────────────────

    static int At(ArrayView1D<int, Stride1D.Dense> gray, ArrayView1D<int, Stride1D.Dense> circle, int w, int x, int y, int k)
        => gray[(y + circle[k * 2 + 1]) * w + (x + circle[k * 2])];

    static bool Nine(ArrayView1D<int, Stride1D.Dense> gray, ArrayView1D<int, Stride1D.Dense> circle, int w, int x, int y,
        int limit, bool brighter)
    {
        int maxRun = 0, run = 0;
        for (int i = 0; i < 32; i++)
        {
            int v = At(gray, circle, w, x, y, i % 16);
            bool passes = brighter ? v > limit : v < limit;
            if (passes)
            {
                run++;
                if (run > maxRun) maxRun = run;
                if (maxRun >= 9) return true;
            }
            else run = 0;
        }
        return false;
    }

    static void FastKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> gray, ArrayView1D<int, Stride1D.Dense> circle,
        ArrayView1D<int, Stride1D.Dense> score, int w, int h, int threshold)
    {
        int y = i / w, x = i - y * w;
        if (x < Margin || x >= w - Margin || y < Margin || y >= h - Margin) { score[i] = 0; return; }
        int center = gray[i];
        int ct = center + threshold, cd = center - threshold;
        int p0 = At(gray, circle, w, x, y, 0), p4 = At(gray, circle, w, x, y, 4);
        int p8 = At(gray, circle, w, x, y, 8), p12 = At(gray, circle, w, x, y, 12);
        int bright = (p0 > ct ? 1 : 0) + (p4 > ct ? 1 : 0) + (p8 > ct ? 1 : 0) + (p12 > ct ? 1 : 0);
        int dark = (p0 < cd ? 1 : 0) + (p4 < cd ? 1 : 0) + (p8 < cd ? 1 : 0) + (p12 < cd ? 1 : 0);
        if (bright < 2 && dark < 2) { score[i] = 0; return; }

        int s = 0;
        if (Nine(gray, circle, w, x, y, ct, true))
        {
            int min = int.MaxValue;
            for (int k = 0; k < 16; k++)
            {
                int d = At(gray, circle, w, x, y, k) - center - threshold;
                if (d > 0 && d < min) min = d;
            }
            s = min == int.MaxValue ? 0 : min;
        }
        else if (Nine(gray, circle, w, x, y, cd, false))
        {
            int min = int.MaxValue;
            for (int k = 0; k < 16; k++)
            {
                int d = center - threshold - At(gray, circle, w, x, y, k);
                if (d > 0 && d < min) min = d;
            }
            s = min == int.MaxValue ? 0 : min;
        }
        score[i] = s;
    }

    // Keep the first STRICT maximum of each 8x8 cell in row-major order - the CPU path appends corners row-major and
    // replaces a cell's entry only on a strictly greater score.
    static void CellMaxKernel(Index1D c, ArrayView1D<int, Stride1D.Dense> score, ArrayView1D<int, Stride1D.Dense> cells,
        int w, int h, int gridW)
    {
        int cy = c / gridW, cx = c - cy * gridW;
        int best = 0, bx = 0, by = 0;
        int y0 = cy * CellSize, x0 = cx * CellSize;
        int y1 = y0 + CellSize < h ? y0 + CellSize : h;
        int x1 = x0 + CellSize < w ? x0 + CellSize : w;
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
            {
                int s = score[y * w + x];
                if (s > best) { best = s; bx = x; by = y; }
            }
        cells[c * 3] = bx;
        cells[c * 3 + 1] = by;
        cells[c * 3 + 2] = best;
    }

    static void BlurRowsKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> gray, ArrayView1D<float, Stride1D.Dense> k9,
        ArrayView1D<float, Stride1D.Dense> tmp, int w)
    {
        int y = i / w, x = i - y * w, row = y * w;
        float acc = 0;
        for (int t = -4; t <= 4; t++)
        {
            int xx = x + t;
            xx = xx < 0 ? 0 : (xx > w - 1 ? w - 1 : xx);
            acc += k9[t + 4] * gray[row + xx];
        }
        tmp[i] = acc;
    }

    static void BlurColsKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> tmp, ArrayView1D<float, Stride1D.Dense> k9,
        ArrayView1D<int, Stride1D.Dense> smooth, int w, int h)
    {
        int y = i / w, x = i - y * w;
        float acc = 0;
        for (int t = -4; t <= 4; t++)
        {
            int yy = y + t;
            yy = yy < 0 ? 0 : (yy > h - 1 ? h - 1 : yy);
            acc += k9[t + 4] * tmp[yy * w + x];
        }
        float r = MathF.Round(acc);
        smooth[i] = (int)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }

    // One pyramid level from the previous: FeatureDetector.ResizePixel's float expression, operation for operation.
    static void ResizeKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> dst,
        int sw, int sh, int dw, int dh)
    {
        int y = i / dw, x = i - y * dw;
        float rx = sw / (float)dw, ry = sh / (float)dh;
        float fx = (x + 0.5f) * rx - 0.5f, fy = (y + 0.5f) * ry - 0.5f;
        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
        float ax = fx - x0, ay = fy - y0;
        int xa = x0 < 0 ? 0 : (x0 > sw - 1 ? sw - 1 : x0), xb = x0 + 1 < 0 ? 0 : (x0 + 1 > sw - 1 ? sw - 1 : x0 + 1);
        int ya = y0 < 0 ? 0 : (y0 > sh - 1 ? sh - 1 : y0), yb = y0 + 1 < 0 ? 0 : (y0 + 1 > sh - 1 ? sh - 1 : y0 + 1);
        float top = (1 - ax) * src[ya * sw + xa] + ax * src[ya * sw + xb];
        float bot = (1 - ax) * src[yb * sw + xa] + ax * src[yb * sw + xb];
        float r = MathF.Round((1 - ay) * top + ay * bot);
        dst[i] = (int)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }

    // Intensity-centroid orientation bin (FeatureDetector.OrientationBin): 32-bit integer moments and argmax.
    static void OrientKernel(Index1D f, ArrayView1D<int, Stride1D.Dense> xy, ArrayView1D<int, Stride1D.Dense> img,
        ArrayView1D<int, Stride1D.Dense> disc, ArrayView1D<int, Stride1D.Dense> dirs, ArrayView1D<int, Stride1D.Dense> bins, int w)
    {
        int x = xy[f * 2], y = xy[f * 2 + 1];
        int m10 = 0, m01 = 0;
        for (int dy = -15; dy <= 15; dy++)
        {
            int half = disc[dy + 15];
            for (int dx = -half; dx <= half; dx++)
            {
                int v = img[(y + dy) * w + x + dx];
                m10 += dx * v; m01 += dy * v;
            }
        }
        int best = 0, bestDot = int.MinValue;
        for (int k = 0; k < FeatureDetector.AngleBins; k++)
        {
            int d = m10 * dirs[k * 2] + m01 * dirs[k * 2 + 1];
            if (d > bestDot) { bestDot = d; best = k; }
        }
        bins[f] = best;
    }

    // Steered BRIEF-256 on the smoothed level: the pair table rotated to the feature's bin.
    static void SteeredBriefKernel(Index1D f, ArrayView1D<int, Stride1D.Dense> xy, ArrayView1D<int, Stride1D.Dense> bins,
        ArrayView1D<int, Stride1D.Dense> smooth, ArrayView1D<int, Stride1D.Dense> steered, ArrayView1D<int, Stride1D.Dense> desc, int w)
    {
        int fx = xy[f * 2], fy = xy[f * 2 + 1], bin = bins[f];
        for (int wi = 0; wi < 8; wi++)
        {
            int word = 0;
            for (int b = 0; b < 32; b++)
            {
                int o = (bin * 256 + wi * 32 + b) * 4;
                int p1 = smooth[(fy + steered[o + 1]) * w + fx + steered[o]];
                int p2 = smooth[(fy + steered[o + 3]) * w + fx + steered[o + 2]];
                if (p1 < p2) word |= 1 << b;
            }
            desc[f * 8 + wi] = word;
        }
    }

    private sealed class Kernels
    {
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int> Fast = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int> CellMax = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> BlurRows = null!;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int> BlurCols = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int> Resize = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> Orient = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> Brief = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Circle = null!, Steered = null!, Disc = null!, Dirs = null!;
        public MemoryBuffer1D<float, Stride1D.Dense> Blur = null!;
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    private static Kernels For(Accelerator a) => s_kernels.GetValue(a, acc => new Kernels
    {
        Fast = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int>(FastKernel),
        CellMax = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int>(CellMaxKernel),
        BlurRows = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(BlurRowsKernel),
        BlurCols = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(BlurColsKernel),
        Resize = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int>(ResizeKernel),
        Orient = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(OrientKernel),
        Brief = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(SteeredBriefKernel),
        Circle = acc.Allocate1D(FeatureDetector.CircleTable()),
        Steered = acc.Allocate1D(FeatureDetector.SteeredPairTable()),
        Disc = acc.Allocate1D(FeatureDetector.DiscHalfWidthTable()),
        Dirs = acc.Allocate1D(FeatureDetector.AngleDirTable()),
        Blur = acc.Allocate1D(FeatureDetector.BlurKernel()),
    });

    /// <summary>
    /// Detect features in a device-resident grayscale frame (one int 0..255 per pixel, row-major).
    /// </summary>
    public async Task<List<ImageFeature>> DetectAsync(Accelerator accelerator,
        ArrayView1D<int, Stride1D.Dense> gray, int width, int height)
    {
        var k = For(accelerator);
        var sizes = FeatureDetector.LevelSizes(width, height, _levels);
        var quota = FeatureDetector.LevelQuota(_maxFeatures, _coarseFeatures, _levels);
        var result = new List<ImageFeature>();
        var levelBufs = new List<MemoryBuffer1D<int, Stride1D.Dense>>();
        const int Edge = FeatureDetector.EdgeBorder;
        try
        {
            var level = gray;
            for (int l = 0; l < _levels; l++)
            {
                var (w, h) = sizes[l];
                // Checked BEFORE the resize: a level buffer disposed in finally while its dispatch is still queued is a
                // use-after-free on WebGPU.
                if (w <= 2 * Edge || h <= 2 * Edge) break;
                if (l > 0)
                {
                    var (pw, ph) = sizes[l - 1];
                    var buf = accelerator.Allocate1D<int>((long)w * h);
                    levelBufs.Add(buf);
                    k.Resize(w * h, level, buf.View, pw, ph, w, h);
                    level = buf.View;
                }
                int n = w * h;
                int gridW = (w + CellSize - 1) / CellSize, gridH = (h + CellSize - 1) / CellSize;

                // FAST score per pixel, first strict maximum per cell - on the device; the cells come back (small).
                int[] cells;
                using (var score = accelerator.Allocate1D<int>(n))
                using (var cellBuf = accelerator.Allocate1D<int>((long)gridW * gridH * 3))
                {
                    k.Fast(n, level, k.Circle.View, score.View, w, h, _threshold);
                    k.CellMax(gridW * gridH, score.View, cellBuf.View, w, h, gridW);
                    await accelerator.SynchronizeAsync();
                    cells = await cellBuf.CopyToHostAsync<int>();
                }
                var features = new List<ImageFeature>();
                for (int c = 0; c < gridW * gridH; c++)
                    if (cells[c * 3 + 2] > 0)
                        features.Add(new ImageFeature { X = cells[c * 3], Y = cells[c * 3 + 1], Score = cells[c * 3 + 2] });

                // The CPU detector's own border filter, sort and quota, on the same list order: ties resolve identically.
                features.RemoveAll(c => c.X < Edge || c.X >= w - Edge || c.Y < Edge || c.Y >= h - Edge);
                features.Sort((a, b) => b.Score.CompareTo(a.Score));
                if (features.Count > quota[l]) features = features.GetRange(0, quota[l]);
                if (features.Count == 0) continue;

                // Orientation on the level; steered BRIEF on its smoothed copy - on the device, descriptors back.
                var xy = new int[features.Count * 2];
                for (int f = 0; f < features.Count; f++) { xy[f * 2] = (int)features[f].X; xy[f * 2 + 1] = (int)features[f].Y; }
                int[] words;
                using (var tmp = accelerator.Allocate1D<float>(n))
                using (var smooth = accelerator.Allocate1D<int>(n))
                using (var xyBuf = accelerator.Allocate1D(xy))
                using (var bins = accelerator.Allocate1D<int>(features.Count))
                using (var desc = accelerator.Allocate1D<int>((long)features.Count * 8))
                {
                    if (_oriented) k.Orient(features.Count, xyBuf.View, level, k.Disc.View, k.Dirs.View, bins.View, w);
                    else bins.MemSetToZero();
                    k.BlurRows(n, level, k.Blur.View, tmp.View, w);
                    k.BlurCols(n, tmp.View, k.Blur.View, smooth.View, w, h);
                    k.Brief(features.Count, xyBuf.View, bins.View, smooth.View, k.Steered.View, desc.View, w);
                    await accelerator.SynchronizeAsync();
                    words = await desc.CopyToHostAsync<int>();
                }
                float sx = width / (float)w, sy = height / (float)h;
                for (int f = 0; f < features.Count; f++)
                {
                    var d = new byte[32];
                    for (int j = 0; j < 32; j++) d[j] = (byte)((words[f * 8 + j / 4] >> (8 * (j % 4))) & 0xFF);
                    var ft = features[f];
                    ft.Descriptor = d;
                    ft.Octave = l;
                    ft.X = ((int)ft.X + 0.5f) * sx - 0.5f;
                    ft.Y = ((int)ft.Y + 0.5f) * sy - 0.5f;
                    result.Add(ft);
                }
            }
        }
        finally
        {
            foreach (var b in levelBufs) b.Dispose();
        }
        return result;
    }
}
