using System.Numerics;
using System.Runtime.InteropServices;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The level-of-detail splat tree (Plans/lod-streaming.md), CPU builder and cut - the oracle the GPU versions are
/// checked against. Leaves are the scene's splats; each level merges the splats that share a grid cell (step growing
/// x1.5 a level, Spark's Tiny-LoD) into a parent (<see cref="LodMerge"/>), until one root is left.
/// <para>
/// Every node carries a bounding sphere around all its descendants and an LOD size at least its largest child's, so
/// the view metric size / distance-to-sphere never grows from a parent to a child. That is what makes the parallel cut
/// exact: a node is drawn when it is small enough on screen and its parent is not, and along every root-to-leaf path
/// exactly one node satisfies that, from any viewpoint and at any threshold.
/// </para>
/// </summary>
public sealed class LodTree
{
    const int F = SplatFormat.Floats;

    /// <summary>Packed rows of every node: leaves first (the input order), then each level's parents.</summary>
    public float[] Rows = Array.Empty<float>();
    /// <summary>Parent of each node, -1 for a root.</summary>
    public int[] Parent = Array.Empty<int>();
    /// <summary>First child and child count of each node (children are contiguous); count 0 for a leaf.</summary>
    public int[] FirstChild = Array.Empty<int>(), ChildCount = Array.Empty<int>();
    /// <summary>Bounding sphere of each node's whole subtree: centre xyz + radius, 4 floats a node.</summary>
    public float[] Bounds = Array.Empty<float>();
    /// <summary>World size used by the cut: 2 x the largest 1-sigma scale, raised to its largest child's.</summary>
    public float[] LodSize = Array.Empty<float>();
    public int LeafCount, NodeCount;

    /// <summary>Grid growth per level (Spark's default r = 1.5).</summary>
    public const float LevelGrowth = 1.5f;

    /// <summary>World size of a splat row: 2 x its largest 1-sigma scale.</summary>
    public static float SizeOf(ReadOnlySpan<float> row) =>
        2f * MathF.Max(row[SplatFormat.OffScale], MathF.Max(row[SplatFormat.OffScale + 1], row[SplatFormat.OffScale + 2]));

    /// <summary>Build the tree over <paramref name="n"/> packed rows.</summary>
    public static LodTree Build(ReadOnlySpan<float> leaves, int n, int maxLevels = 64)
    {
        var rows = new List<float>(n * F * 2);
        var parent = new List<int>(n * 2);
        var first = new List<int>(n * 2);
        var count = new List<int>(n * 2);
        var size = new List<float>(n * 2);
        var bounds = new List<float>(n * 8);
        for (int i = 0; i < n; i++)
        {
            var r = leaves.Slice(i * F, F);
            for (int k = 0; k < F; k++) rows.Add(r[k]);
            parent.Add(-1); first.Add(-1); count.Add(0);
            float s = SizeOf(r);
            size.Add(s);
            // A splat's own sphere: its centre, radius 3 sigma of its largest axis.
            bounds.Add(r[0]); bounds.Add(r[1]); bounds.Add(r[2]); bounds.Add(1.5f * s);
        }

        // The base grid step: the median leaf size, so most splats take part from the first level.
        var sorted = new float[n];
        for (int i = 0; i < n; i++) sorted[i] = size[i];
        Array.Sort(sorted);
        float step = n > 0 ? MathF.Max(sorted[n / 2], 1e-6f) : 1f;

        var frontier = Enumerable.Range(0, n).ToList();
        var row = new float[F];
        for (int level = 0; level < maxLevels && frontier.Count > 1; level++, step *= LevelGrowth)
        {
            // Splats no larger than this level's cell take part; larger ones wait for a coarser level.
            var groups = new Dictionary<(long, long, long), List<int>>();
            var next = new List<int>();
            foreach (int node in frontier)
            {
                if (size[node] > step && level < maxLevels - 1) { next.Add(node); continue; }
                int o = node * F;
                var key = ((long)MathF.Floor(rows[o] / step), (long)MathF.Floor(rows[o + 1] / step), (long)MathF.Floor(rows[o + 2] / step));
                if (!groups.TryGetValue(key, out var list)) groups[key] = list = new List<int>();
                list.Add(node);
            }
            foreach (var list in groups.Values)
            {
                if (list.Count == 1) { next.Add(list[0]); continue; }
                int p = parent.Count;
                var acc = new LodMerge.Accum();
                float maxChildSize = 0f;
                foreach (int c in list)
                {
                    LodMerge.Add(ref acc, CollectionsMarshal.AsSpan(rows).Slice(c * F, F));
                    maxChildSize = MathF.Max(maxChildSize, size[c]);
                }
                LodMerge.Finish(acc, row);
                for (int k = 0; k < F; k++) rows.Add(row[k]);
                parent.Add(-1); first.Add(-1); count.Add(list.Count);
                size.Add(MathF.Max(SizeOf(row), maxChildSize));
                // Sphere around the children's spheres (centred on the parent's mean).
                float cx = row[0], cy = row[1], cz = row[2], rad = 0f;
                foreach (int c in list)
                {
                    int b = c * 4;
                    float d = MathF.Sqrt((bounds[b] - cx) * (bounds[b] - cx) + (bounds[b + 1] - cy) * (bounds[b + 1] - cy) + (bounds[b + 2] - cz) * (bounds[b + 2] - cz));
                    rad = MathF.Max(rad, d + bounds[b + 3]);
                }
                bounds.Add(cx); bounds.Add(cy); bounds.Add(cz); bounds.Add(rad);
                foreach (int c in list) parent[c] = p;   // FixChildRuns lists each node's children
                next.Add(p);
            }
            frontier = next;   // a cell with nothing to merge waits for the next, coarser level
        }

        var tree = new LodTree
        {
            LeafCount = n, NodeCount = parent.Count,
            Rows = rows.ToArray(), Parent = parent.ToArray(), LodSize = size.ToArray(), Bounds = bounds.ToArray(),
            ChildCount = count.ToArray(), FirstChild = first.ToArray(),
        };
        tree.FixChildRuns();
        return tree;
    }

    /// <summary>Children as an explicit list (not necessarily contiguous in node order) - rebuilt from Parent.</summary>
    public int[] ChildIndex = Array.Empty<int>();

    void FixChildRuns()
    {
        // Children of p listed contiguously in ChildIndex[FirstChild[p] .. + ChildCount[p]].
        var start = new int[NodeCount];
        int run = 0;
        for (int p = 0; p < NodeCount; p++) { start[p] = run; run += ChildCount[p]; }
        ChildIndex = new int[run];
        var fill = (int[])start.Clone();
        for (int c = 0; c < NodeCount; c++) if (Parent[c] >= 0) ChildIndex[fill[Parent[c]]++] = c;
        for (int p = 0; p < NodeCount; p++) FirstChild[p] = ChildCount[p] > 0 ? start[p] : -1;
    }

    /// <summary>
    /// The view metric of node <paramref name="i"/>: its LOD size over the distance from the camera to its bounding
    /// sphere (0.1 near floor), in pixels for a focal length <paramref name="focal"/>. Never larger for a child than
    /// for its parent.
    /// </summary>
    public float PixelSize(int i, Vector3 cam, float focal)
    {
        int b = i * 4;
        float d = Vector3.Distance(cam, new Vector3(Bounds[b], Bounds[b + 1], Bounds[b + 2])) - Bounds[b + 3];
        return LodSize[i] * focal / MathF.Max(d, 0.1f);
    }

    /// <summary>
    /// The cut, node by node (the GPU does exactly this in parallel): node i is drawn when it is no larger than
    /// <paramref name="tau"/> pixels (or a leaf) and its parent is larger (or it is a root).
    /// </summary>
    public bool InCut(int i, Vector3 cam, float focal, float tau)
    {
        bool smallEnough = ChildCount[i] == 0 || PixelSize(i, cam, focal) <= tau;
        bool parentTooBig = Parent[i] < 0 || PixelSize(Parent[i], cam, focal) > tau;
        return smallEnough && parentTooBig;
    }
}
