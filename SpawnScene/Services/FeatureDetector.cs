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

        // Step 4: Compute BRIEF descriptors
        ComputeDescriptors(suppressed, gray, width, height);

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

                if (brightCount < 3 && darkCount < 3) continue;

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
        int patchRadius = 12;

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

            int dx1 = Math.Clamp((int)(r1 * patchRadius / 5), -patchRadius, patchRadius);
            int dy1 = Math.Clamp((int)(r2 * patchRadius / 5), -patchRadius, patchRadius);
            int dx2 = Math.Clamp((int)(r3 * patchRadius / 5), -patchRadius, patchRadius);
            int dy2 = Math.Clamp((int)(r4 * patchRadius / 5), -patchRadius, patchRadius);

            pairs[i] = (dx1, dy1, dx2, dy2);
        }

        return pairs;
    }
}
