using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Initialise splats from a sparse SfM point cloud, the way 3D Gaussian Splatting actually does it.
///
/// This replaces unprojecting a monocular depth map per view. That produced one private depth
/// shell per camera - measured on drjohnson, 91.9% of the resulting splats were constrained by
/// at most one view and 49.5% by none at all, so supervised loss fell while held-out loss rose
/// and no optimiser setting could have changed it.
///
/// A COLMAP point is triangulated from at least two images by construction, which is exactly
/// the property that was missing. The cloud is small - 79,922 points for drjohnson against
/// 1,115,136 depth splats - because it is meant to be a STARTING point that density control
/// grows, not a finished scene.
///
/// Follows <c>GaussianModel.create_from_pcd</c> in the Kerbl et al. reference implementation:
/// isotropic scale from the local point spacing, low starting opacity, identity rotation,
/// colour straight from the point.
/// </summary>
public static class SparsePointCloudInit
{
    /// <summary>Starting opacity. The reference uses inverse_sigmoid(0.1); our format is linear.</summary>
    public const float InitialOpacity = 0.1f;

    /// <summary>Neighbours used to estimate local spacing (reference: distCUDA2 takes 3).</summary>
    public const int SpacingNeighbours = 3;

    /// <summary>
    /// Floor on the squared spacing, mirroring the reference clamp. Duplicate points give a
    /// distance of zero, and a zero-scale Gaussian has no gradient and can never recover.
    /// </summary>
    public const float MinSpacingSq = 1e-7f;

    /// <summary>Header is an i32 count, then per point 3x f32 position and 4x u8 RGBA.</summary>
    public const int BytesPerPoint = 16;

    /// <summary>
    /// Parse the flat cloud written by <c>tools/colmap_to_dataset.py</c>.
    /// </summary>
    public static PointCloud Parse(byte[] data)
    {
        if (data.Length < 4)
            throw new ArgumentException("point cloud is shorter than its header", nameof(data));

        int count = BitConverter.ToInt32(data, 0);
        long needed = 4L + (long)count * BytesPerPoint;
        if (count < 0 || needed > data.Length)
            throw new ArgumentException(
                $"point cloud header says {count} points ({needed} bytes) but the file is " +
                $"{data.Length} bytes", nameof(data));

        var positions = new Vector3[count];
        var colors = new Vector3[count];
        for (int i = 0; i < count; i++)
        {
            int o = 4 + i * BytesPerPoint;
            positions[i] = new Vector3(
                BitConverter.ToSingle(data, o),
                BitConverter.ToSingle(data, o + 4),
                BitConverter.ToSingle(data, o + 8));
            colors[i] = new Vector3(data[o + 12] / 255f, data[o + 13] / 255f, data[o + 14] / 255f);
        }
        return new PointCloud { Positions = positions, Colors = colors };
    }

    /// <summary>
    /// Build a packed splat buffer in <see cref="SplatFormat"/> layout.
    ///
    /// Scale is the RMS distance to the nearest <see cref="SpacingNeighbours"/> points, so a
    /// splat is about as big as the gap it has to cover. A single constant cannot do this: the
    /// same cloud has dense points on a textured wall and sparse ones across a floor, and one
    /// size either leaves holes or smears detail.
    /// </summary>
    public static float[] BuildPacked(PointCloud cloud)
    {
        int n = cloud.Count;
        var packed = new float[(long)n * SplatFormat.Floats];
        if (n == 0) return packed;

        float[] spacing = LocalSpacing(cloud.Positions);

        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var p = cloud.Positions[i];
            var c = i < cloud.Colors.Length ? cloud.Colors[i] : new Vector3(0.5f);
            float s = spacing[i];

            packed[o + SplatFormat.OffPos] = p.X;
            packed[o + SplatFormat.OffPos + 1] = p.Y;
            packed[o + SplatFormat.OffPos + 2] = p.Z;
            packed[o + SplatFormat.OffColor] = c.X;
            packed[o + SplatFormat.OffColor + 1] = c.Y;
            packed[o + SplatFormat.OffColor + 2] = c.Z;

            // Isotropic: an SfM point measures a LOCATION, not a surface, so there is no normal
            // to flatten along. The depth path flattens to 0.15 because a depth sample does
            // measure a surface. Anisotropy here is the optimiser job, not the initialiser one.
            packed[o + SplatFormat.OffScale] = s;
            packed[o + SplatFormat.OffScale + 1] = s;
            packed[o + SplatFormat.OffScale + 2] = s;

            packed[o + SplatFormat.OffOpacity] = InitialOpacity;

            // Identity quaternion, xyzw.
            packed[o + SplatFormat.OffQuat] = 0f;
            packed[o + SplatFormat.OffQuat + 1] = 0f;
            packed[o + SplatFormat.OffQuat + 2] = 0f;
            packed[o + SplatFormat.OffQuat + 3] = 1f;
        }
        return packed;
    }

    /// <summary>
    /// RMS distance from each point to its three nearest neighbours.
    ///
    /// Through a uniform grid rather than all pairs: 80k points is 6.4 billion pair tests, which
    /// is minutes of wasm. The grid makes the cost proportional to the points actually nearby,
    /// and the answer is identical rather than approximate, because the search widens until
    /// every remaining cell is provably farther than the current third-best.
    /// </summary>
    public static float[] LocalSpacing(Vector3[] points)
    {
        int n = points.Length;
        var result = new float[n];
        if (n == 0) return result;
        if (n <= SpacingNeighbours)
        {
            // Too few to estimate spacing from. Use the extent, which at least has the right
            // order of magnitude, rather than a constant nobody measured.
            var (lo0, hi0) = Bounds(points);
            float fallback = MathF.Max((hi0 - lo0).Length() / MathF.Max(n, 1),
                                       MathF.Sqrt(MinSpacingSq));
            Array.Fill(result, fallback);
            return result;
        }

        // Size the grid from a ROBUST extent, not the bounding box.
        //
        // An SfM cloud has far outliers - a few points triangulated out into the sky. The
        // bounding box is then enormous, the cell computed from it is enormous, every real
        // point lands in one or two cells, and the grid degenerates to the all-pairs search it
        // exists to avoid. MEASURED: drjohnson's 79,922 points took 349 SECONDS this way.
        // Percentile bounds ignore the outliers; they are still placed, clamped into the edge
        // cells, and still get an exact answer - they just do not get to set the cell size.
        var (lo, hi) = RobustBounds(points);
        Vector3 extent = Vector3.Max(hi - lo, new Vector3(1e-6f));

        // Aim at a handful of points per cell: enough that a 3x3x3 block usually settles the
        // answer, few enough that scanning a cell stays cheap.
        double volume = (double)extent.X * extent.Y * extent.Z;
        float cell = (float)Math.Cbrt(Math.Max(volume, 1e-12) * 8.0 / n);
        if (!float.IsFinite(cell) || cell <= 0) cell = extent.Length() / 64f;

        // Cap the axis counts so a pathological aspect ratio cannot allocate a huge grid, and
        // widen the cell to match rather than silently building a grid that does not cover.
        const int MaxCellsPerAxis = 256;
        cell = MathF.Max(cell, extent.X / MaxCellsPerAxis);
        cell = MathF.Max(cell, extent.Y / MaxCellsPerAxis);
        cell = MathF.Max(cell, extent.Z / MaxCellsPerAxis);

        int gx = Math.Clamp((int)(extent.X / cell) + 1, 1, MaxCellsPerAxis);
        int gy = Math.Clamp((int)(extent.Y / cell) + 1, 1, MaxCellsPerAxis);
        int gz = Math.Clamp((int)(extent.Z / cell) + 1, 1, MaxCellsPerAxis);

        // Counting sort into cells: two passes and two arrays, no per-cell List allocations.
        long cells = (long)gx * gy * gz;
        var cellOf = new int[n];
        var starts = new int[cells + 1];
        for (int i = 0; i < n; i++)
        {
            var d = points[i] - lo;
            int cx = Math.Clamp((int)(d.X / cell), 0, gx - 1);
            int cy = Math.Clamp((int)(d.Y / cell), 0, gy - 1);
            int cz = Math.Clamp((int)(d.Z / cell), 0, gz - 1);
            int c = (cz * gy + cy) * gx + cx;
            cellOf[i] = c;
            starts[c + 1]++;
        }
        for (long c = 0; c < cells; c++) starts[c + 1] += starts[c];
        var order = new int[n];
        var cursor = (int[])starts.Clone();
        for (int i = 0; i < n; i++) order[cursor[cellOf[i]]++] = i;

        var best = new float[SpacingNeighbours];

        for (int i = 0; i < n; i++)
        {
            var d = points[i] - lo;
            int cx = Math.Clamp((int)(d.X / cell), 0, gx - 1);
            int cy = Math.Clamp((int)(d.Y / cell), 0, gy - 1);
            int cz = Math.Clamp((int)(d.Z / cell), 0, gz - 1);

            Array.Fill(best, float.MaxValue);
            int found = 0;

            for (int ring = 0; ; ring++)
            {
                ScanRing(points, order, starts, gx, gy, gz, cx, cy, cz, ring, i, best, ref found);

                // Anything outside this ring is at least ring*cell away, so once the third-best
                // beats that, widening cannot change the answer. This is what makes the grid
                // exact rather than approximate.
                float guaranteed = ring * cell;
                if (found >= SpacingNeighbours &&
                    best[SpacingNeighbours - 1] <= guaranteed * guaranteed)
                    break;
                if (ring > gx + gy + gz) break;   // exhausted the grid
            }

            float sum = 0;
            int used = 0;
            for (int k = 0; k < SpacingNeighbours; k++)
                if (best[k] < float.MaxValue) { sum += best[k]; used++; }

            float meanSq = used > 0 ? sum / used : MinSpacingSq;
            result[i] = MathF.Sqrt(MathF.Max(meanSq, MinSpacingSq));
        }
        return result;
    }

    /// <summary>Scan the shell of cells exactly <paramref name="ring"/> steps from the centre.</summary>
    static void ScanRing(
        Vector3[] points, int[] order, int[] starts, int gx, int gy, int gz,
        int cx, int cy, int cz, int ring, int self, float[] best, ref int found)
    {
        int x0 = cx - ring, x1 = cx + ring;
        int y0 = cy - ring, y1 = cy + ring;
        int z0 = cz - ring, z1 = cz + ring;

        for (int z = Math.Max(z0, 0); z <= Math.Min(z1, gz - 1); z++)
        for (int y = Math.Max(y0, 0); y <= Math.Min(y1, gy - 1); y++)
        {
            bool edgeZY = z == z0 || z == z1 || y == y0 || y == y1;
            for (int x = Math.Max(x0, 0); x <= Math.Min(x1, gx - 1); x++)
            {
                // Only the SHELL: interior cells were covered by an earlier ring.
                if (!edgeZY && x != x0 && x != x1) continue;

                int c = (z * gy + y) * gx + x;
                for (int s = starts[c]; s < starts[c + 1]; s++)
                {
                    int j = order[s];
                    if (j == self) continue;
                    float d2 = Vector3.DistanceSquared(points[self], points[j]);
                    Insert(best, d2, ref found);
                }
            }
        }
    }

    /// <summary>Keep the smallest <see cref="SpacingNeighbours"/> squared distances seen.</summary>
    static void Insert(float[] best, float d2, ref int found)
    {
        if (d2 >= best[SpacingNeighbours - 1]) return;
        int k = SpacingNeighbours - 1;
        while (k > 0 && best[k - 1] > d2) { best[k] = best[k - 1]; k--; }
        best[k] = d2;
        if (found < SpacingNeighbours) found++;
    }

    /// <summary>
    /// Per-axis 1st and 99th percentile, so a handful of far outliers cannot set the scale.
    /// </summary>
    static (Vector3 lo, Vector3 hi) RobustBounds(Vector3[] points)
    {
        int n = points.Length;
        var xs = new float[n];
        var ys = new float[n];
        var zs = new float[n];
        for (int i = 0; i < n; i++) { xs[i] = points[i].X; ys[i] = points[i].Y; zs[i] = points[i].Z; }
        Array.Sort(xs); Array.Sort(ys); Array.Sort(zs);

        int lo = n / 100;
        int hi = n - 1 - lo;
        if (hi <= lo) { lo = 0; hi = n - 1; }
        return (new Vector3(xs[lo], ys[lo], zs[lo]), new Vector3(xs[hi], ys[hi], zs[hi]));
    }

    static (Vector3 lo, Vector3 hi) Bounds(Vector3[] points)
    {
        var lo = new Vector3(float.MaxValue);
        var hi = new Vector3(float.MinValue);
        foreach (var p in points) { lo = Vector3.Min(lo, p); hi = Vector3.Max(hi, p); }
        return (lo, hi);
    }
}
