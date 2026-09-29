using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// <see cref="FeatureDetector"/> (FAST-9 + grid non-max suppression + top-N + BRIEF on a Gaussian-smoothed image) on
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
    const int Margin = 4, CellSize = 8, PatchRadius = 15;

    private readonly int _maxFeatures, _threshold;
    public GpuFeatureDetector(int maxFeatures = 2000, int fastThreshold = 25)
    {
        _maxFeatures = maxFeatures;
        _threshold = fastThreshold;
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

    static void BriefKernel(Index1D f, ArrayView1D<int, Stride1D.Dense> xy, ArrayView1D<int, Stride1D.Dense> smooth,
        ArrayView1D<int, Stride1D.Dense> pairs, ArrayView1D<int, Stride1D.Dense> desc, int w, int h)
    {
        int fx = xy[f * 2], fy = xy[f * 2 + 1];
        bool border = fx < PatchRadius || fx >= w - PatchRadius || fy < PatchRadius || fy >= h - PatchRadius;
        for (int wi = 0; wi < 8; wi++)
        {
            int word = 0;
            if (!border)
                for (int b = 0; b < 32; b++)
                {
                    int p = (wi * 32 + b) * 4;
                    int p1 = smooth[(fy + pairs[p + 1]) * w + (fx + pairs[p])];
                    int p2 = smooth[(fy + pairs[p + 3]) * w + (fx + pairs[p + 2])];
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
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int> Brief = null!;
        public MemoryBuffer1D<int, Stride1D.Dense> Circle = null!, Pairs = null!;
        public MemoryBuffer1D<float, Stride1D.Dense> Blur = null!;
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    private static Kernels For(Accelerator a) => s_kernels.GetValue(a, acc => new Kernels
    {
        Fast = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int>(FastKernel),
        CellMax = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int>(CellMaxKernel),
        BlurRows = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(BlurRowsKernel),
        BlurCols = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(BlurColsKernel),
        Brief = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(BriefKernel),
        Circle = acc.Allocate1D(FeatureDetector.CircleTable()),
        Pairs = acc.Allocate1D(FeatureDetector.BriefPairTable()),
        Blur = acc.Allocate1D(FeatureDetector.BlurKernel()),
    });

    /// <summary>
    /// Detect features in a device-resident grayscale frame (one int 0..255 per pixel, row-major).
    /// </summary>
    public async Task<List<ImageFeature>> DetectAsync(Accelerator accelerator,
        ArrayView1D<int, Stride1D.Dense> gray, int width, int height)
    {
        var k = For(accelerator);
        int n = width * height;
        int gridW = (width + CellSize - 1) / CellSize, gridH = (height + CellSize - 1) / CellSize;

        // 1-3. FAST score per pixel, first strict maximum per cell - on the device; the cells come back (small).
        int[] cells;
        using (var score = accelerator.Allocate1D<int>(n))
        using (var cellBuf = accelerator.Allocate1D<int>((long)gridW * gridH * 3))
        {
            k.Fast(n, gray, k.Circle.View, score.View, width, height, _threshold);
            k.CellMax(gridW * gridH, score.View, cellBuf.View, width, height, gridW);
            await accelerator.SynchronizeAsync();
            cells = await cellBuf.CopyToHostAsync<int>();
        }
        var features = new List<ImageFeature>();
        for (int c = 0; c < gridW * gridH; c++)
            if (cells[c * 3 + 2] > 0)
                features.Add(new ImageFeature { X = cells[c * 3], Y = cells[c * 3 + 1], Score = cells[c * 3 + 2] });

        // Top-N with the CPU detector's own sort, on the same list order, so ties resolve identically.
        features.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (features.Count > _maxFeatures) features = features.GetRange(0, _maxFeatures);
        if (features.Count == 0) return features;

        // 4. BRIEF on the smoothed frame - smoothing and sampling on the device, descriptors back.
        var xy = new int[features.Count * 2];
        for (int f = 0; f < features.Count; f++) { xy[f * 2] = (int)features[f].X; xy[f * 2 + 1] = (int)features[f].Y; }
        int[] words;
        using (var tmp = accelerator.Allocate1D<float>(n))
        using (var smooth = accelerator.Allocate1D<int>(n))
        using (var xyBuf = accelerator.Allocate1D(xy))
        using (var desc = accelerator.Allocate1D<int>((long)features.Count * 8))
        {
            k.BlurRows(n, gray, k.Blur.View, tmp.View, width);
            k.BlurCols(n, tmp.View, k.Blur.View, smooth.View, width, height);
            k.Brief(features.Count, xyBuf.View, smooth.View, k.Pairs.View, desc.View, width, height);
            await accelerator.SynchronizeAsync();
            words = await desc.CopyToHostAsync<int>();
        }
        for (int f = 0; f < features.Count; f++)
        {
            var d = new byte[32];
            for (int j = 0; j < 32; j++) d[j] = (byte)((words[f * 8 + j / 4] >> (8 * (j % 4))) & 0xFF);
            features[f].Descriptor = d;
        }
        return features;
    }
}
