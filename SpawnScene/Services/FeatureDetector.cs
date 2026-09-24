using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Feature detection using FAST corner detection + BRIEF-like binary descriptors.
/// This is a CPU implementation suitable for small image sets.
/// For larger sets, ILGPU acceleration can be added.
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

    private readonly int _maxFeatures;
    private readonly int _fastThreshold;

    public FeatureDetector(int maxFeatures = 2000, int fastThreshold = 25)
    {
        _maxFeatures = maxFeatures;
        _fastThreshold = fastThreshold;
    }

    /// <summary>
    /// Detect features in a grayscale image.
    /// </summary>
    public List<ImageFeature> Detect(byte[] gray, int width, int height)
    {
        // Step 1: FAST corner detection
        var corners = DetectFastCorners(gray, width, height, _fastThreshold);

        // Step 2: Non-maximum suppression
        var suppressed = NonMaxSuppression(corners, width, height);

        // Step 3: Keep top-N by score
        suppressed.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (suppressed.Count > _maxFeatures)
            suppressed = suppressed.GetRange(0, _maxFeatures);

        // Step 4: Compute BRIEF descriptors - on a SMOOTHED image. BRIEF's binary tests compare single
        // pixels, so without smoothing sensor noise flips bits: the reference (Calonder et al.) smooths
        // with a Gaussian first. Unsmoothed, true correspondences sat ~59 of 256 bits apart, at the
        // matcher's 64-bit cutoff, and a neighbouring view matched no better than an unrelated one.
        var smooth = GaussianBlur(gray, width, height);
        ComputeDescriptors(suppressed, smooth, width, height);

        return suppressed;
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
    /// Compute BRIEF-like binary descriptors for each feature.
    /// </summary>
    private void ComputeDescriptors(List<ImageFeature> features, byte[] gray, int width, int height)
    {
        int patchRadius = 15;

        foreach (var feat in features)
        {
            int fx = (int)feat.X;
            int fy = (int)feat.Y;

            // Skip if too close to border
            if (fx < patchRadius || fx >= width - patchRadius ||
                fy < patchRadius || fy >= height - patchRadius)
            {
                feat.Descriptor = new byte[32];
                continue;
            }

            var desc = new byte[32];
            for (int i = 0; i < 256; i++)
            {
                var (dx1, dy1, dx2, dy2) = BriefPairs[i];
                int p1 = gray[(fy + dy1) * width + (fx + dx1)];
                int p2 = gray[(fy + dy2) * width + (fx + dx2)];

                if (p1 < p2)
                {
                    desc[i / 8] |= (byte)(1 << (i % 8));
                }
            }

            feat.Descriptor = desc;
        }
    }

    /// <summary>
    /// Generate BRIEF sampling pairs using a Gaussian-like distribution.
    /// </summary>
    private static (int, int, int, int)[] GenerateBriefPairs(int count, int seed)
    {
        var rng = new Random(seed);
        var pairs = new (int, int, int, int)[count];
        // Test offsets ~ isotropic Gaussian with sigma = S/5 over an S=31 patch (BRIEF's G II, as ORB uses),
        // clamped to the 15 px radius ComputeDescriptors keeps clear of the border. This was sigma
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
