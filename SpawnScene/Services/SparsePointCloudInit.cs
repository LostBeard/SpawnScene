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
    ///
    /// <paramref name="maxScale"/> caps the initial size. Truck's SfM cloud has far outliers
    /// whose 3-NN spacing is huge (MEASURED p90 0.26 with median 0.017); without a cap those
    /// start at the Adam MaxScale ceiling and the scene is blobby before the first step.
    /// Pass the training MaxScale (or densify split size) when known; 0 = no cap.
    /// </summary>
    public static float[] BuildPacked(PointCloud cloud, float maxScale = 0f)
    {
        int n = cloud.Count;
        var packed = new float[(long)n * SplatFormat.Floats];
        if (n == 0) return packed;

        float[] spacing = LocalSpacing(cloud.Positions);

        // Cap outlier spacings. Truck MEASURED: median 0.017, p90 0.265 - the tail is SfM
        // outliers, not real surface spacing. Without a cap those Gaussians start at the Adam
        // MaxScale ceiling and the scene is blobby before step 0. Kerbl's distCUDA2 has the
        // same formula but their clouds are cleaner; we clamp to 10x the median (or an
        // explicit maxScale when the caller knows the training bound).
        float[] sorted = (float[])spacing.Clone();
        Array.Sort(sorted);
        float median = sorted[n / 2];
        float cap = maxScale > 0f ? maxScale : MathF.Max(median * 10f, 1e-4f);

        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var p = cloud.Positions[i];
            var c = i < cloud.Colors.Length ? cloud.Colors[i] : new Vector3(0.5f);
            float s = MathF.Min(spacing[i], cap);

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
    /// Exact, through a k-d tree. The previous uniform grid was exact too, but its cost for a point is the
    /// number of cells within its third-nearest distance - and an SfM cloud's outliers (points triangulated
    /// into the sky) are far from everything, so each one scanned most of a 256^3 grid. MEASURED 2026-09-25:
    /// Truck's 133k-point cloud took 28 s natively and 266 s in the browser, before every GT-pose run. The
    /// tree's bound is the distance to the splitting plane, which prunes the same way wherever the point is.
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

        var tree = new KdTree(points);
        var best = new float[SpacingNeighbours];
        for (int i = 0; i < n; i++)
        {
            Array.Fill(best, float.MaxValue);
            int found = 0;
            tree.Nearest(i, best, ref found);

            float sum = 0;
            int used = 0;
            for (int k = 0; k < SpacingNeighbours; k++)
                if (best[k] < float.MaxValue) { sum += best[k]; used++; }

            float meanSq = used > 0 ? sum / used : MinSpacingSq;
            result[i] = MathF.Sqrt(MathF.Max(meanSq, MinSpacingSq));
        }
        return result;
    }

    /// <summary>
    /// Static k-d tree over point indices, stored implicitly: the node for range [lo, hi) is the median at
    /// mid = (lo + hi) / 2, split on <c>_axis[mid]</c> (the range's widest axis), with [lo, mid) on the low
    /// side and (mid, hi) on the high side. Ranges of <see cref="Leaf"/> or fewer are scanned directly.
    /// </summary>
    sealed class KdTree
    {
        const int Leaf = 8;
        readonly float[] _x, _y, _z;
        readonly int[] _idx;
        readonly byte[] _axis;

        public KdTree(Vector3[] points)
        {
            int n = points.Length;
            _x = new float[n]; _y = new float[n]; _z = new float[n];
            for (int i = 0; i < n; i++) { _x[i] = points[i].X; _y[i] = points[i].Y; _z[i] = points[i].Z; }
            _idx = new int[n];
            for (int i = 0; i < n; i++) _idx[i] = i;
            _axis = new byte[n];
            Build(0, n);
        }

        float Coord(int point, int axis) => axis == 0 ? _x[point] : axis == 1 ? _y[point] : _z[point];

        void Build(int lo, int hi)
        {
            // Recurse on the smaller side, loop on the larger: depth stays log2(n) whatever the data.
            while (hi - lo > Leaf)
            {
                float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
                float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
                for (int k = lo; k < hi; k++)
                {
                    int p = _idx[k];
                    minX = MathF.Min(minX, _x[p]); maxX = MathF.Max(maxX, _x[p]);
                    minY = MathF.Min(minY, _y[p]); maxY = MathF.Max(maxY, _y[p]);
                    minZ = MathF.Min(minZ, _z[p]); maxZ = MathF.Max(maxZ, _z[p]);
                }
                float ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
                int axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
                int mid = (lo + hi) >> 1;
                Select(lo, hi - 1, mid, axis);
                _axis[mid] = (byte)axis;
                if (mid - lo < hi - mid - 1) { Build(lo, mid); lo = mid + 1; }
                else { Build(mid + 1, hi); hi = mid; }
            }
        }

        /// <summary>
        /// Quickselect: afterwards _idx[k] holds the k-th smallest on <paramref name="axis"/> within [l, r], with
        /// nothing larger before it and nothing smaller after it.
        /// </summary>
        void Select(int l, int r, int k, int axis)
        {
            while (r > l)
            {
                // Median-of-three pivot: sorted or clustered input cannot drive it quadratic.
                int m = (l + r) >> 1;
                if (Coord(_idx[m], axis) < Coord(_idx[l], axis)) Swap(l, m);
                if (Coord(_idx[r], axis) < Coord(_idx[l], axis)) Swap(l, r);
                if (Coord(_idx[r], axis) < Coord(_idx[m], axis)) Swap(m, r);
                float pivot = Coord(_idx[m], axis);
                int i = l, j = r;
                while (i <= j)
                {
                    while (Coord(_idx[i], axis) < pivot) i++;
                    while (Coord(_idx[j], axis) > pivot) j--;
                    if (i <= j) { Swap(i, j); i++; j--; }
                }
                if (k <= j) r = j;
                else if (k >= i) l = i;
                else return;
            }
        }

        void Swap(int a, int b) => (_idx[a], _idx[b]) = (_idx[b], _idx[a]);

        /// <summary>The <see cref="SpacingNeighbours"/> smallest squared distances from point <paramref name="self"/> to any other.</summary>
        public void Nearest(int self, float[] best, ref int found)
            => Search(0, _idx.Length, self, _x[self], _y[self], _z[self], best, ref found);

        void Search(int lo, int hi, int self, float qx, float qy, float qz, float[] best, ref int found)
        {
            if (hi - lo <= Leaf)
            {
                for (int k = lo; k < hi; k++) Visit(_idx[k], self, qx, qy, qz, best, ref found);
                return;
            }
            int mid = (lo + hi) >> 1;
            int p = _idx[mid];
            Visit(p, self, qx, qy, qz, best, ref found);
            int axis = _axis[mid];
            float diff = (axis == 0 ? qx : axis == 1 ? qy : qz) - Coord(p, axis);
            // Near side first, so the far side is usually pruned by a bound that is already tight.
            if (diff < 0)
            {
                Search(lo, mid, self, qx, qy, qz, best, ref found);
                if (diff * diff < best[SpacingNeighbours - 1]) Search(mid + 1, hi, self, qx, qy, qz, best, ref found);
            }
            else
            {
                Search(mid + 1, hi, self, qx, qy, qz, best, ref found);
                if (diff * diff < best[SpacingNeighbours - 1]) Search(lo, mid, self, qx, qy, qz, best, ref found);
            }
        }

        void Visit(int p, int self, float qx, float qy, float qz, float[] best, ref int found)
        {
            if (p == self) return;
            float dx = _x[p] - qx, dy = _y[p] - qy, dz = _z[p] - qz;
            Insert(best, dx * dx + dy * dy + dz * dz, ref found);
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

    static (Vector3 lo, Vector3 hi) Bounds(Vector3[] points)
    {
        var lo = new Vector3(float.MaxValue);
        var hi = new Vector3(float.MinValue);
        foreach (var p in points) { lo = Vector3.Min(lo, p); hi = Vector3.Max(hi, p); }
        return (lo, hi);
    }
}
