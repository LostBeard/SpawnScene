using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// ORB-style features: FAST-9 on an image pyramid, intensity-centroid orientation, steered BRIEF-256 on a smoothed level.
/// The CPU reference; GpuFeatureDetector reproduces it on the device.
/// </summary>
public class FeatureDetector
{
    // FAST circle offsets (16-point Bresenham circle of radius 3)
    private static readonly (int dx, int dy)[] CircleOffsets = new (int, int)[]
    {
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0),  (3, 1),  (2, 2),  (1, 3),
        (0, 3),  (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1),(-2, -2),(-1, -3),
    };

    // BRIEF descriptor sampling pairs (256 pairs for 256-bit descriptor)
    // Generated with a fixed seed for reproducibility
    private static readonly (int, int, int, int)[] BriefPairs = GenerateBriefPairs(256, 42);

    /// <summary>The BRIEF test pairs flattened (dx1, dy1, dx2, dy2) x 256 - shared with GpuFeatureDetector so both
    /// detectors sample the identical pattern.</summary>
    internal static int[] BriefPairTable()
    {
        var t = new int[BriefPairs.Length * 4];
        for (int i = 0; i < BriefPairs.Length; i++)
        {
            var (a, b, c, d) = BriefPairs[i];
            t[i * 4] = a; t[i * 4 + 1] = b; t[i * 4 + 2] = c; t[i * 4 + 3] = d;
        }
        return t;
    }

    /// <summary>The FAST circle as (dx, dy) x 16, shared with GpuFeatureDetector.</summary>
    internal static int[] CircleTable()
    {
        var t = new int[32];
        for (int i = 0; i < 16; i++) { t[i * 2] = CircleOffsets[i].dx; t[i * 2 + 1] = CircleOffsets[i].dy; }
        return t;
    }

    /// <summary>The BRIEF pre-smoothing kernel (sigma 2, 9 taps, normalised) exactly as GaussianBlur computes it.</summary>
    internal static float[] BlurKernel()
    {
        const int r = 4;
        var k = new float[2 * r + 1];
        float sum = 0;
        for (int i = -r; i <= r; i++) { k[i + r] = MathF.Exp(-(i * i) / (2f * 2f * 2f)); sum += k[i + r]; }
        for (int i = 0; i < k.Length; i++) k[i] /= sum;
        return k;
    }

    public int MaxFeatures => _maxFeatures;
    public int FastThreshold => _fastThreshold;
    public int Levels => _levels;

    private readonly int _maxFeatures;
    private readonly int _fastThreshold;
    private readonly int _levels;

    /// <summary>Pyramid step between levels (as ORB).</summary>
    public const float ScaleFactor = 1.2f;
    /// <summary>Steered-BRIEF orientation bins (12 degree steps, as ORB).</summary>
    public const int AngleBins = 30;
    /// <summary>Keypoints stay this far from a level's border: the orientation disc (radius 15) and the steered test
    /// pairs (offsets up to 15 px per axis, so up to 15 sqrt 2 = 21.2 px once rotated) fit inside.</summary>
    public const int EdgeBorder = 22;
    const int OrientRadius = 15;

    /// <summary>
    /// ORB-style detection (2026-09-30): FAST-9 on an 8-level pyramid (scale 1.2), intensity-centroid orientation, BRIEF
    /// steered to it. The single-scale, unoriented FAST+BRIEF made 7 of DrJohnson's 119 truly overlapping pairs
    /// verifiable (15+ matches correct under COLMAP's geometry); OpenCV ORB on the same data: orientation alone 19, scale
    /// alone 36, both 70 (DrJohnsonMatchingTests). <paramref name="levels"/> = 1 is single-scale FAST + steered BRIEF.
    /// </summary>
    public FeatureDetector(int maxFeatures = 2000, int fastThreshold = 25, int levels = 8)
    {
        _maxFeatures = maxFeatures;
        _fastThreshold = fastThreshold;
        _levels = Math.Max(1, levels);
    }

    /// <summary>Level sizes: each level resized from the previous by <see cref="ScaleFactor"/>, rounded.</summary>
    public static (int W, int H)[] LevelSizes(int width, int height, int levels)
    {
        var s = new (int W, int H)[levels];
        s[0] = (width, height);
        for (int l = 1; l < levels; l++)
            s[l] = ((int)MathF.Round(s[l - 1].W / ScaleFactor), (int)MathF.Round(s[l - 1].H / ScaleFactor));
        return s;
    }

    /// <summary>The per-level feature quota (as ORB): a geometric series in 1/scale summing to <paramref name="total"/>.</summary>
    public static int[] LevelQuota(int total, int levels)
    {
        var q = new int[levels];
        double factor = 1.0 / ScaleFactor;
        double perLevel = total * (1 - factor) / (1 - Math.Pow(factor, levels));
        int sum = 0;
        for (int l = 0; l < levels - 1; l++) { q[l] = (int)Math.Round(perLevel); sum += q[l]; perLevel *= factor; }
        q[levels - 1] = Math.Max(total - sum, 0);
        return q;
    }

    /// <summary>
    /// One bilinear pixel of a sw x sh image at destination (x, y) of a dw x dh resize. The GPU pyramid kernel evaluates
    /// this same float expression, so on the ILGPU CPU accelerator the two detectors agree bit for bit.
    /// </summary>
    internal static int ResizePixel(byte[] src, int sw, int sh, int dw, int dh, int x, int y)
    {
        float rx = sw / (float)dw, ry = sh / (float)dh;
        float fx = (x + 0.5f) * rx - 0.5f, fy = (y + 0.5f) * ry - 0.5f;
        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
        float ax = fx - x0, ay = fy - y0;
        int xa = Math.Clamp(x0, 0, sw - 1), xb = Math.Clamp(x0 + 1, 0, sw - 1);
        int ya = Math.Clamp(y0, 0, sh - 1), yb = Math.Clamp(y0 + 1, 0, sh - 1);
        float top = (1 - ax) * src[ya * sw + xa] + ax * src[ya * sw + xb];
        float bot = (1 - ax) * src[yb * sw + xa] + ax * src[yb * sw + xb];
        float r = MathF.Round((1 - ay) * top + ay * bot);
        return (int)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }

    /// <summary>Unit directions of the <see cref="AngleBins"/> bins x 256, rounded: (cos, sin) x bins. The orientation
    /// bin is the argmax of the integer moment vector against these - integer math, identical on every backend (an
    /// atan2 rounds differently on a GPU and would flip bins at their edges). x256 keeps the dot product in 32 bits:
    /// |m10|, |m01| &lt;= 255 x sum|dx| over the disc (~5,000) = 1.3M, so |dot| &lt;= 1.3M x 256 x 2 = 6.6e8 - no emulated i64
    /// on WebGPU.</summary>
    internal static int[] AngleDirTable()
    {
        var t = new int[AngleBins * 2];
        for (int k = 0; k < AngleBins; k++)
        {
            double a = 2 * Math.PI * k / AngleBins;
            t[k * 2] = (int)Math.Round(Math.Cos(a) * 256); t[k * 2 + 1] = (int)Math.Round(Math.Sin(a) * 256);
        }
        return t;
    }

    /// <summary>The BRIEF pairs rotated to every bin, rounded: (dx1, dy1, dx2, dy2) x 256 x bins.</summary>
    internal static int[] SteeredPairTable()
    {
        var t = new int[AngleBins * BriefPairs.Length * 4];
        for (int k = 0; k < AngleBins; k++)
        {
            double a = 2 * Math.PI * k / AngleBins, c = Math.Cos(a), s = Math.Sin(a);
            for (int i = 0; i < BriefPairs.Length; i++)
            {
                var (x1, y1, x2, y2) = BriefPairs[i];
                int o = (k * BriefPairs.Length + i) * 4;
                t[o] = (int)Math.Round(c * x1 - s * y1); t[o + 1] = (int)Math.Round(s * x1 + c * y1);
                t[o + 2] = (int)Math.Round(c * x2 - s * y2); t[o + 3] = (int)Math.Round(s * x2 + c * y2);
            }
        }
        return t;
    }

    /// <summary>Half-width of the orientation disc per row: (dy + 15) -> the largest dx with dx^2 + dy^2 &lt;= 15^2.</summary>
    internal static int[] DiscHalfWidthTable()
    {
        var t = new int[2 * OrientRadius + 1];
        for (int dy = -OrientRadius; dy <= OrientRadius; dy++)
        {
            int h = 0;
            while ((h + 1) * (h + 1) + dy * dy <= OrientRadius * OrientRadius) h++;
            t[dy + OrientRadius] = h;
        }
        return t;
    }

    private static readonly int[] s_dirs = AngleDirTable();
    private static readonly int[] s_steered = SteeredPairTable();
    private static readonly int[] s_disc = DiscHalfWidthTable();

    /// <summary>The orientation bin of the patch at (x, y): intensity centroid over the radius-15 disc.</summary>
    internal static int OrientationBin(byte[] img, int w, int x, int y)
    {
        int m10 = 0, m01 = 0;
        for (int dy = -OrientRadius; dy <= OrientRadius; dy++)
        {
            int half = s_disc[dy + OrientRadius];
            for (int dx = -half; dx <= half; dx++)
            {
                int v = img[(y + dy) * w + x + dx];
                m10 += dx * v; m01 += dy * v;
            }
        }
        int best = 0, bestDot = int.MinValue;
        for (int k = 0; k < AngleBins; k++)
        {
            int d = m10 * s_dirs[k * 2] + m01 * s_dirs[k * 2 + 1];
            if (d > bestDot) { bestDot = d; best = k; }
        }
        return best;
    }

    /// <summary>
    /// Detect features in a grayscale image. X/Y are level-0 pixels; <see cref="ImageFeature.Octave"/> is the level.
    /// </summary>
    public List<ImageFeature> Detect(byte[] gray, int width, int height)
    {
        var sizes = LevelSizes(width, height, _levels);
        var quota = LevelQuota(_maxFeatures, _levels);
        var result = new List<ImageFeature>();
        byte[] level = gray;
        for (int l = 0; l < _levels; l++)
        {
            var (w, h) = sizes[l];
            if (l > 0)
            {
                var (pw, ph) = sizes[l - 1];
                var prev = level;
                level = new byte[w * h];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        level[y * w + x] = (byte)ResizePixel(prev, pw, ph, w, h, x, y);
            }
            if (w <= 2 * EdgeBorder || h <= 2 * EdgeBorder) break;

            // FAST + the first strict maximum per 8x8 cell, then only corners whose disc and steered pairs fit.
            var corners = NonMaxSuppression(DetectFastCorners(level, w, h, _fastThreshold), w, h);
            corners.RemoveAll(c => c.X < EdgeBorder || c.X >= w - EdgeBorder || c.Y < EdgeBorder || c.Y >= h - EdgeBorder);
            corners.Sort((a, b) => b.Score.CompareTo(a.Score));
            if (corners.Count > quota[l]) corners = corners.GetRange(0, quota[l]);
            if (corners.Count == 0) continue;

            // Orientation on the level image; steered BRIEF on its SMOOTHED copy. BRIEF's binary tests compare single
            // pixels, so without smoothing sensor noise flips bits (the reference, Calonder et al., smooths first).
            // Unsmoothed, true correspondences sat ~59 of 256 bits apart, at the matcher's 64-bit cutoff.
            var smooth = GaussianBlur(level, w, h);
            float sx = width / (float)w, sy = height / (float)h;
            foreach (var c in corners)
            {
                int fx = (int)c.X, fy = (int)c.Y;
                int bin = OrientationBin(level, w, fx, fy);
                var desc = new byte[32];
                for (int i = 0; i < 256; i++)
                {
                    int o = (bin * 256 + i) * 4;
                    int p1 = smooth[(fy + s_steered[o + 1]) * w + fx + s_steered[o]];
                    int p2 = smooth[(fy + s_steered[o + 3]) * w + fx + s_steered[o + 2]];
                    if (p1 < p2) desc[i / 8] |= (byte)(1 << (i % 8));
                }
                c.Descriptor = desc;
                c.Octave = l;
                c.X = (fx + 0.5f) * sx - 0.5f;
                c.Y = (fy + 0.5f) * sy - 0.5f;
                result.Add(c);
            }
        }
        return result;
    }

    /// <summary>
    /// FAST-9 corner detection.
    /// A pixel is a corner if N contiguous pixels on the Bresenham circle
    /// are all brighter or all darker than the center pixel by threshold.
    /// </summary>
    private List<ImageFeature> DetectFastCorners(byte[] gray, int width, int height, int threshold)
    {
        var corners = new List<ImageFeature>();
        int margin = 4; // Border for circle + descriptor sampling

        for (int y = margin; y < height - margin; y++)
        {
            for (int x = margin; x < width - margin; x++)
            {
                int center = gray[y * width + x];
                int ct = center + threshold;
                int cd = center - threshold;

                // Quick rejection: check pixels at 0°, 90°, 180°, 270°
                int p0 = gray[(y + CircleOffsets[0].dy) * width + (x + CircleOffsets[0].dx)];
                int p4 = gray[(y + CircleOffsets[4].dy) * width + (x + CircleOffsets[4].dx)];
                int p8 = gray[(y + CircleOffsets[8].dy) * width + (x + CircleOffsets[8].dx)];
                int p12 = gray[(y + CircleOffsets[12].dy) * width + (x + CircleOffsets[12].dx)];

                int brightCount = (p0 > ct ? 1 : 0) + (p4 > ct ? 1 : 0) + (p8 > ct ? 1 : 0) + (p12 > ct ? 1 : 0);
                int darkCount = (p0 < cd ? 1 : 0) + (p4 < cd ? 1 : 0) + (p8 < cd ? 1 : 0) + (p12 < cd ? 1 : 0);

                // FAST-9: any 9 contiguous pixels of the 16 contain at least TWO of the four compass
                // points (they sit 4 apart), so 2-of-4 is the correct quick reject. This used to be
                // 3-of-4, the FAST-12 test, which rejects every convex 90-degree corner (exactly two
                // compass points fall outside it) - door, frame and window corners never survived, and
                // DrJohnson's pair matches were noise (adjacent frames matched no better than distant).
                if (brightCount < 2 && darkCount < 2) continue;

                // Full check: need 9 contiguous pixels
                int score = ComputeCornerScore(gray, width, x, y, threshold);
                if (score > 0)
                {
                    corners.Add(new ImageFeature { X = x, Y = y, Score = score });
                }
            }
        }

        return corners;
    }

    private int ComputeCornerScore(byte[] gray, int width, int x, int y, int threshold)
    {
        int center = gray[y * width + x];

        // Check for 9 contiguous brighter or darker pixels
        int[] circleValues = new int[16];
        for (int i = 0; i < 16; i++)
        {
            circleValues[i] = gray[(y + CircleOffsets[i].dy) * width + (x + CircleOffsets[i].dx)];
        }

        // Check brighter
        if (Check9Contiguous(circleValues, center, threshold, true))
        {
            // Score = minimum difference
            int minDiff = int.MaxValue;
            for (int i = 0; i < 16; i++)
            {
                int diff = circleValues[i] - center - threshold;
                if (diff > 0) minDiff = Math.Min(minDiff, diff);
            }
            return minDiff == int.MaxValue ? 0 : minDiff;
        }

        // Check darker
        if (Check9Contiguous(circleValues, center, threshold, false))
        {
            int minDiff = int.MaxValue;
            for (int i = 0; i < 16; i++)
            {
                int diff = center - threshold - circleValues[i];
                if (diff > 0) minDiff = Math.Min(minDiff, diff);
            }
            return minDiff == int.MaxValue ? 0 : minDiff;
        }

        return 0;
    }

    private bool Check9Contiguous(int[] circle, int center, int threshold, bool brighter)
    {
        int limit = brighter ? center + threshold : center - threshold;
        int maxContiguous = 0;
        int contiguous = 0;

        // Check twice around the circle for wrap-around
        for (int i = 0; i < 32; i++)
        {
            int val = circle[i % 16];
            bool passes = brighter ? val > limit : val < limit;

            if (passes)
            {
                contiguous++;
                maxContiguous = Math.Max(maxContiguous, contiguous);
                if (maxContiguous >= 9) return true;
            }
            else
            {
                contiguous = 0;
            }
        }

        return false;
    }

    private List<ImageFeature> NonMaxSuppression(List<ImageFeature> corners, int width, int height)
    {
        // Simple grid-based suppression (keep best per cell)
        int cellSize = 8;
        int gridW = (width + cellSize - 1) / cellSize;
        int gridH = (height + cellSize - 1) / cellSize;
        var grid = new ImageFeature?[gridW * gridH];

        foreach (var corner in corners)
        {
            int gx = (int)(corner.X / cellSize);
            int gy = (int)(corner.Y / cellSize);
            int gIdx = gy * gridW + gx;

            if (grid[gIdx] == null || corner.Score > grid[gIdx]!.Score)
            {
                grid[gIdx] = corner;
            }
        }

        return grid.Where(g => g != null).Select(g => g!).ToList();
    }

    /// <summary>
    /// Separable Gaussian, sigma 2, 9 taps, clamped borders - the BRIEF paper's pre-smoothing.
    /// </summary>
    private static byte[] GaussianBlur(byte[] gray, int width, int height)
    {
        const int r = 4;
        Span<float> k = stackalloc float[2 * r + 1];
        float sum = 0;
        for (int i = -r; i <= r; i++) { k[i + r] = MathF.Exp(-(i * i) / (2f * 2f * 2f)); sum += k[i + r]; }
        for (int i = 0; i < k.Length; i++) k[i] /= sum;

        var tmp = new float[width * height];
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                float acc = 0;
                for (int i = -r; i <= r; i++)
                    acc += k[i + r] * gray[row + Math.Clamp(x + i, 0, width - 1)];
                tmp[row + x] = acc;
            }
        }
        var outp = new byte[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float acc = 0;
                for (int i = -r; i <= r; i++)
                    acc += k[i + r] * tmp[Math.Clamp(y + i, 0, height - 1) * width + x];
                outp[y * width + x] = (byte)Math.Clamp(MathF.Round(acc), 0, 255);
            }
        return outp;
    }

    /// <summary>
    /// Generate BRIEF sampling pairs using a Gaussian-like distribution.
    /// </summary>
    private static (int, int, int, int)[] GenerateBriefPairs(int count, int seed)
    {
        var rng = new Random(seed);
        var pairs = new (int, int, int, int)[count];
        // Test offsets ~ isotropic Gaussian with sigma = S/5 over an S=31 patch (BRIEF's G II, as ORB uses),
        // clamped to 15 px per axis (EdgeBorder covers them once steered). This was sigma
        // 12/5 = 2.4 px - the RADIUS where the paper means the patch SIZE - so every test looked at a
        // 5x5 neighbourhood and the descriptor carried almost no structure.
        const int patchSize = 31;
        int patchRadius = patchSize / 2;

        for (int i = 0; i < count; i++)
        {
            // Gaussian-distributed offsets (Box-Muller)
            double u1 = rng.NextDouble();
            double u2 = rng.NextDouble();
            double r1 = Math.Sqrt(-2 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2 * Math.PI * u2);
            double r2 = Math.Sqrt(-2 * Math.Log(Math.Max(u1, 1e-10))) * Math.Sin(2 * Math.PI * u2);

            u1 = rng.NextDouble();
            u2 = rng.NextDouble();
            double r3 = Math.Sqrt(-2 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2 * Math.PI * u2);
            double r4 = Math.Sqrt(-2 * Math.Log(Math.Max(u1, 1e-10))) * Math.Sin(2 * Math.PI * u2);

            int dx1 = Math.Clamp((int)Math.Round(r1 * patchSize / 5.0), -patchRadius, patchRadius);
            int dy1 = Math.Clamp((int)Math.Round(r2 * patchSize / 5.0), -patchRadius, patchRadius);
            int dx2 = Math.Clamp((int)Math.Round(r3 * patchSize / 5.0), -patchRadius, patchRadius);
            int dy2 = Math.Clamp((int)Math.Round(r4 * patchSize / 5.0), -patchRadius, patchRadius);

            pairs[i] = (dx1, dy1, dx2, dy2);
        }

        return pairs;
    }
}
