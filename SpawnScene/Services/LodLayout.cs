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
    /// The cut over a laid-out tree, as the GPU does it: node i is drawn when its view size is at most
    /// <paramref name="tau"/> (a leaf's is 0) and its parent's is larger (or it is a root).
    /// </summary>
    public static bool InCut(LodTree t, int i, System.Numerics.Vector3 cam, float focal, float tau)
    {
        bool smallEnough = t.PixelSize(i, cam, focal) <= tau;
        bool parentTooBig = t.Parent[i] < 0 || t.PixelSize(t.Parent[i], cam, focal) > tau;
        return smallEnough && parentTooBig;
    }
}
