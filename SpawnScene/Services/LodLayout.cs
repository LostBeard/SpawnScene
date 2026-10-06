namespace SpawnScene.Services;

/// <summary>
/// The streaming layout of an LOD tree (Plans/lod-streaming.md, phase B): nodes in BREADTH-FIRST order - the roots, then
/// every node's children in turn - so each parent comes before its children, siblings are contiguous, and cutting the
/// order into fixed-size chunks gives chunk 0 = the top of the tree (a whole coarse scene) and later chunks = finer
/// detail of spatially coherent regions (siblings come from one grid cell). CPU oracle for the GPU layout.
/// <para>
/// In this order the leaves are no longer the first rows, so the layout zeroes a leaf's LOD size instead: a zero size
/// is 0 px from anywhere, always "small enough", which is exactly the leaf rule of the cut (<see cref="InCut"/>) with no
/// leaf count or flag to carry.
/// </para>
/// </summary>
public static class LodLayout
{
    /// <summary>The nodes of <paramref name="t"/> in breadth-first order: <c>order[newIndex] = oldIndex</c>.</summary>
    public static int[] BreadthFirst(LodTree t)
    {
        var order = new int[t.NodeCount];
        int tail = 0;
        for (int i = 0; i < t.NodeCount; i++) if (t.Parent[i] < 0) order[tail++] = i;
        for (int head = 0; head < tail; head++)
        {
            int p = order[head];
            for (int k = 0; k < t.ChildCount[p]; k++) order[tail++] = t.ChildIndex[t.FirstChild[p] + k];
        }
        if (tail != t.NodeCount) throw new InvalidOperationException($"{t.NodeCount - tail} nodes are not reachable from a root");
        return order;
    }

    /// <summary>
    /// <paramref name="t"/> rearranged into <paramref name="order"/> (<c>order[newIndex] = oldIndex</c>, a parent-first
    /// order such as <see cref="BreadthFirst"/>): rows, bounds and sizes moved, parent and child indices renumbered, each
    /// node's children contiguous (ChildIndex is the identity), and every leaf's LOD size 0.
    /// </summary>
    public static LodTree Reorder(LodTree t, int[] order)
    {
        const int F = SplatFormat.Floats;
        int n = t.NodeCount;
        var newIndex = new int[n];
        for (int k = 0; k < n; k++) newIndex[order[k]] = k;
        var r = new LodTree
        {
            Rows = new float[n * F], Parent = new int[n], FirstChild = new int[n], ChildCount = new int[n],
            Bounds = new float[n * 4], LodSize = new float[n], ChildIndex = new int[n],
            LeafCount = t.LeafCount, NodeCount = n,
        };
        for (int k = 0; k < n; k++)
        {
            int o = order[k];
            Array.Copy(t.Rows, o * F, r.Rows, k * F, F);
            Array.Copy(t.Bounds, o * 4, r.Bounds, k * 4, 4);
            r.Parent[k] = t.Parent[o] < 0 ? -1 : newIndex[t.Parent[o]];
            r.ChildCount[k] = t.ChildCount[o];
            r.LodSize[k] = t.ChildCount[o] == 0 ? 0f : t.LodSize[o];
            r.ChildIndex[k] = k;
            // Children are contiguous in a parent-first order only when the order put them so (BreadthFirst does);
            // the first one is the smallest new index among them.
            int first = int.MaxValue;
            for (int c = 0; c < t.ChildCount[o]; c++) first = Math.Min(first, newIndex[t.ChildIndex[t.FirstChild[o] + c]]);
            r.FirstChild[k] = t.ChildCount[o] > 0 ? first : -1;
        }
        return r;
    }

    /// <summary>
    /// Chunk boundaries for streaming a laid-out tree (any order with parents first and each node's children
    /// contiguous): chunks of at most <paramref name="maxNodes"/> nodes that never split a sibling run - a run being
    /// consecutive nodes with the same parent, the roots one run - so a node's children always share one chunk.
    /// Splitting runs made each chunk need the next one (its last run straddled), and a streamed view loaded every
    /// chunk (2026-10-05). Returns the chunk starts, then NodeCount.
    /// <para>
    /// Measured against TREELET chunks (each chunk a subtree expanded breadth-first) on a 20-room corridor seen from
    /// one end: breadth-first needed 58 / 18 of 130 chunks at tau 10 / 50 px, treelets 67 / 20 - not better.
    /// </para>
    /// </summary>
    public static int[] ChunkStarts(LodTree t, int maxNodes)
    {
        var starts = new List<int> { 0 };
        int chunkStart = 0, runStart = 0;
        for (int i = 1; i <= t.NodeCount; i++)
        {
            if (i < t.NodeCount && t.Parent[i] == t.Parent[i - 1]) continue;
            // [runStart, i) is a run.
            if (i - runStart > maxNodes) throw new InvalidOperationException($"a sibling run of {i - runStart} nodes is over the chunk size {maxNodes}");
            if (i - chunkStart > maxNodes) { starts.Add(runStart); chunkStart = runStart; }
            runStart = i;
        }
        starts.Add(t.NodeCount);
        return starts.ToArray();
    }

    /// <summary>The chunk holding node <paramref name="i"/>, for chunk starts from <see cref="ChunkStarts"/>.</summary>
    public static int ChunkOf(int[] starts, int i)
    {
        int k = Array.BinarySearch(starts, i);
        return k >= 0 ? k : ~k - 1;
    }

    /// <summary>
    /// The chunks a chunk needs resident with it: those holding the parents of its nodes (in breadth-first order a
    /// contiguous span before it). Loading a chunk only with these, transitively,
    /// keeps every resident node's ancestors resident, which is what makes <see cref="InCutPaged"/>'s local test exact.
    /// </summary>
    public static int[] ParentChunks(LodTree t, int[] starts, int chunk)
    {
        var set = new SortedSet<int>();
        for (int i = starts[chunk]; i < starts[chunk + 1]; i++)
        {
            int p = t.Parent[i];
            if (p < 0) continue;
            int c = ChunkOf(starts, p);
            if (c != chunk) set.Add(c);
        }
        return set.ToArray();
    }

    /// <summary>
    /// The cut over a PARTLY resident tree (paging), node by node: drawn when its chunk is resident, its parent is too
    /// big (or it is a root), and it is small enough on screen OR its children's chunk is not resident - it stands in
    /// for them, and <paramref name="wantsChildren"/> says that chunk is the one to load. With the resident chunks
    /// closed under <see cref="ParentChunks"/>, every leaf's root path has exactly one drawn node.
    /// </summary>
    public static bool InCutPaged(LodTree t, int i, System.Numerics.Vector3 cam, float focal, float tau, int[] starts,
        Func<int, bool> chunkResident, out bool wantsChildren)
    {
        wantsChildren = false;
        if (!chunkResident(ChunkOf(starts, i))) return false;
        int p = t.Parent[i];
        if (p >= 0 && t.PixelSize(p, cam, focal) <= tau) return false;
        if (t.PixelSize(i, cam, focal) <= tau) return true;   // leaves: size 0
        wantsChildren = !chunkResident(ChunkOf(starts, t.FirstChild[i]));
        return wantsChildren;
    }

    /// <summary>
    /// The cut over a laid-out tree, as the GPU does it: node i is drawn when its view size is at most
    /// <paramref name="tau"/> (a leaf's is 0) and its parent's is larger (or it is a root).
    /// </summary>
    public static bool InCut(LodTree t, int i, System.Numerics.Vector3 cam, float focal, float tau)
    {
        bool smallEnough = t.PixelSize(i, cam, focal) <= tau;
        bool parentTooBig = t.Parent[i] < 0 || t.PixelSize(t.Parent[i], cam, focal) > tau;
        return smallEnough && parentTooBig;
    }

    /// <summary>
    /// Phase D (Plans/lod-streaming.md): one LOD tree a training block, joined under a top - so no step needs the whole
    /// scene at once. Each block's tree is laid out breadth-first on its own; its roots merge into a BLOCK NODE (LodMerge
    /// over the roots, a sphere around theirs, LOD size raised to their largest), and one root merges the block nodes.
    /// Order: the root, the block nodes, then each block's nodes - so a block node's children are its block's root
    /// run, contiguous and first in the block's region; every parent comes first. Chunks: the top on its own, then each
    /// block's own run-aligned chunks (a block never shares a chunk with another).
    /// </summary>
    public static (LodTree Laid, int[] Starts) Forest(IReadOnlyList<LodTree> blocks, int maxNodes)
    {
        const int F = SplatFormat.Floats;
        int b = blocks.Count;
        if (b < 1 || b + 1 > maxNodes) throw new ArgumentException($"{b} blocks do not fit a {maxNodes}-node top chunk");
        var laid = blocks.Select(t => Reorder(t, BreadthFirst(t))).ToArray();
        int nodes = 1 + b + laid.Sum(l => l.NodeCount);
        var r = new LodTree
        {
            NodeCount = nodes, LeafCount = laid.Sum(l => l.LeafCount),
            Rows = new float[nodes * F], Parent = new int[nodes], FirstChild = new int[nodes], ChildCount = new int[nodes],
            Bounds = new float[nodes * 4], LodSize = new float[nodes], ChildIndex = Enumerable.Range(0, nodes).ToArray(),
        };
        var starts = new List<int> { 0, 1 + b };   // the top (root + block nodes) is chunk 0
        int at = 1 + b;
        var blockRows = new float[b * F];
        for (int k = 0; k < b; k++)
        {
            var l = laid[k];
            int roots = 0;
            while (roots < l.NodeCount && l.Parent[roots] < 0) roots++;
            // The block's nodes, renumbered by its offset; its roots hang off block node 1 + k.
            Array.Copy(l.Rows, 0, r.Rows, at * F, l.NodeCount * F);
            Array.Copy(l.Bounds, 0, r.Bounds, at * 4, l.NodeCount * 4);
            Array.Copy(l.LodSize, 0, r.LodSize, at, l.NodeCount);
            for (int i = 0; i < l.NodeCount; i++)
            {
                r.Parent[at + i] = l.Parent[i] < 0 ? 1 + k : l.Parent[i] + at;
                r.ChildCount[at + i] = l.ChildCount[i];
                r.FirstChild[at + i] = l.ChildCount[i] > 0 ? l.FirstChild[i] + at : -1;
            }
            // Block node 1 + k: the merge of the block's roots, enclosing them.
            int bn = 1 + k;
            LodMerge.Merge(l.Rows.AsSpan(0, roots * F), roots, r.Rows.AsSpan(bn * F, F));
            Enclose(r, bn, at, roots);
            r.Parent[bn] = 0; r.FirstChild[bn] = at; r.ChildCount[bn] = roots;
            Array.Copy(r.Rows, bn * F, blockRows, k * F, F);
            // Its chunks: the block's own run-aligned starts, offset (its first is the previous chunk's end).
            var bs = ChunkStarts(l, maxNodes);
            for (int c = 1; c < bs.Length; c++) starts.Add(at + bs[c]);
            at += l.NodeCount;
        }
        // The root: the merge of the block nodes.
        LodMerge.Merge(blockRows, b, r.Rows.AsSpan(0, F));
        Enclose(r, 0, 1, b);
        r.Parent[0] = -1; r.FirstChild[0] = 1; r.ChildCount[0] = b;
        return (r, starts.ToArray());
    }

    /// <summary>Node p's sphere around its children's [first, first+count) spheres; its LOD size at least theirs.</summary>
    static void Enclose(LodTree t, int p, int first, int count)
    {
        float cx = t.Rows[p * SplatFormat.Floats], cy = t.Rows[p * SplatFormat.Floats + 1], cz = t.Rows[p * SplatFormat.Floats + 2];
        float rad = 0f, size = LodTree.SizeOf(t.Rows.AsSpan(p * SplatFormat.Floats, SplatFormat.Floats));
        for (int c = first; c < first + count; c++)
        {
            float dx = t.Bounds[c * 4] - cx, dy = t.Bounds[c * 4 + 1] - cy, dz = t.Bounds[c * 4 + 2] - cz;
            rad = MathF.Max(rad, MathF.Sqrt(dx * dx + dy * dy + dz * dz) + t.Bounds[c * 4 + 3]);
            size = MathF.Max(size, t.LodSize[c]);
        }
        t.Bounds[p * 4] = cx; t.Bounds[p * 4 + 1] = cy; t.Bounds[p * 4 + 2] = cz; t.Bounds[p * 4 + 3] = rad;
        t.LodSize[p] = size;
    }
}
