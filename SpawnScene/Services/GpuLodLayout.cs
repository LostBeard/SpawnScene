using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// <see cref="LodLayout"/> on the GPU: a built <see cref="GpuLodTree"/> rearranged into breadth-first order (roots in
/// node order, then each node's children in its child-list order), every leaf's LOD size zeroed, children contiguous.
/// One depth at a time: the frontier's child counts are scanned, each frontier node places its children after the
/// frontier, then every array is gathered through the order once. Plans/lod-streaming.md, phase B.
/// </summary>
public static class GpuLodLayout
{
    const int F = SplatFormat.Floats;

    /// <summary>
    /// Lay out <paramref name="t"/> (left as it is). The result's ChildList is unused (children are the runs
    /// FirstChild .. + ChildCount); <paramref name="order"/> receives <c>order[newIndex] = oldIndex</c> for whatever else
    /// travels with the nodes (SH). CPU transfers: two ints per tree depth (the frontier's size).
    /// </summary>
    public static async Task<GpuLodTree> BuildAsync(Accelerator a, GpuLodTree t, Action<MemoryBuffer1D<int, Stride1D.Dense>>? order = null)
    {
        int n = t.NodeCount;
        var ord = a.Allocate1D<int>(Math.Max(1, n));
        using var newIndex = a.Allocate1D<int>(Math.Max(1, n));
        using var flag = a.Allocate1D<int>(Math.Max(1, n));
        using var slot = a.Allocate1D<int>(Math.Max(1, n));
        using var scanTemp = a.Allocate1D<int>(Math.Max(1, a.ComputeScanTempStorageSize<int>(Math.Max(1, n))));
        var scan = a.CreateScan<int, Stride1D.Dense, Stride1D.Dense, AddInt32>(ScanKind.Exclusive);
        var stream = a.DefaultStream;

        var rootFlag = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(RootFlagKernel);
        var rootPlace = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(RootPlaceKernel);
        var count = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(CountKernel);
        var expand = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int, int>(ExpandKernel);

        // Roots, in node order.
        rootFlag(n, t.Parent.View.SubView(0, n), flag.View);
        scan(stream, flag.View.SubView(0, n), slot.View.SubView(0, n), scanTemp.View);
        rootPlace(n, t.Parent.View.SubView(0, n), slot.View, ord.View, newIndex.View);
        int lo = 0, hi = await TotalAsync(a, flag, slot, n);

        // Then each depth's children after it.
        while (hi > lo && hi < n)
        {
            int m = hi - lo;
            count(m, ord.View, t.ChildCount.View, flag.View, lo);
            scan(stream, flag.View.SubView(0, m), slot.View.SubView(0, m), scanTemp.View);
            expand(m, ord.View, newIndex.View, t.FirstChild.View, t.ChildList.View, flag.View, slot.View, lo, hi);
            int added = await TotalAsync(a, flag, slot, m);
            if (added == 0) break;
            lo = hi; hi += added;
        }
        if (hi != n) throw new InvalidOperationException($"LOD layout reached {hi:N0} of {n:N0} nodes from the roots");

        var r = new GpuLodTree { LeafCount = t.LeafCount, NodeCount = n, Levels = t.Levels };
        r.Nodes = a.Allocate1D<float>((long)n * F);
        r.Parent = a.Allocate1D<int>(n);
        r.Bounds = a.Allocate1D<float>((long)n * 4);
        r.LodSize = a.Allocate1D<float>(n);
        r.FirstChild = a.Allocate1D<int>(n);
        r.ChildCount = a.Allocate1D<int>(n);
        r.ChildList = a.Allocate1D<int>(1);
        var rows = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ScatterRowsKernel);
        var topo = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ScatterTopologyKernel);
        rows(n, ord.View, t.Nodes.View, t.Bounds.View, r.Nodes.View, r.Bounds.View);
        topo(n, ord.View, newIndex.View, t.Parent.View, t.LodSize.View, t.FirstChild.View, t.ChildCount.View, t.ChildList.View,
            r.Parent.View, r.LodSize.View, r.FirstChild.View, r.ChildCount.View);
        await a.SynchronizeAsync();
        if (order != null) order(ord); else ord.Dispose();
        return r;
    }

    /// <summary>CPU transfer: two ints - the last flag and its exclusive prefix, i.e. the sum over the range.</summary>
    static async Task<int> TotalAsync(Accelerator a, MemoryBuffer1D<int, Stride1D.Dense> flag, MemoryBuffer1D<int, Stride1D.Dense> slot, int m)
    {
        await a.SynchronizeAsync();
        var f = await flag.CopyToHostAsync<int>(m - 1, 1);
        var s = await slot.CopyToHostAsync<int>(m - 1, 1);
        return f[0] + s[0];
    }

    static void RootFlagKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> parent, ArrayView1D<int, Stride1D.Dense> flag)
        => flag[i] = parent[i] < 0 ? 1 : 0;

    static void RootPlaceKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> parent, ArrayView1D<int, Stride1D.Dense> slot,
        ArrayView1D<int, Stride1D.Dense> order, ArrayView1D<int, Stride1D.Dense> newIndex)
    {
        if (parent[i] >= 0) return;
        int k = slot[i];
        order[k] = i;
        newIndex[i] = k;
    }

    /// <summary>Frontier entry j (node order[lo + j]): its child count.</summary>
    static void CountKernel(Index1D j, ArrayView1D<int, Stride1D.Dense> order, ArrayView1D<int, Stride1D.Dense> childCount,
        ArrayView1D<int, Stride1D.Dense> counts, int lo)
        => counts[j] = childCount[order[lo + j]];

    /// <summary>Frontier entry j places its children, in child-list order, at hi + its prefix.</summary>
    static void ExpandKernel(Index1D j, ArrayView1D<int, Stride1D.Dense> order, ArrayView1D<int, Stride1D.Dense> newIndex,
        ArrayView1D<int, Stride1D.Dense> firstChild, ArrayView1D<int, Stride1D.Dense> childList,
        ArrayView1D<int, Stride1D.Dense> counts, ArrayView1D<int, Stride1D.Dense> prefix, int lo, int hi)
    {
        int o = order[lo + j];
        int c = counts[j];
        int at = hi + prefix[j];
        int first = firstChild[o];
        for (int k = 0; k < c; k++)
        {
            int child = childList[first + k];
            order[at + k] = child;
            newIndex[child] = at + k;
        }
    }

    /// <summary>New node k takes old node order[k]'s row and bounding sphere.</summary>
    static void ScatterRowsKernel(Index1D k, ArrayView1D<int, Stride1D.Dense> order, ArrayView1D<float, Stride1D.Dense> nodes,
        ArrayView1D<float, Stride1D.Dense> bounds, ArrayView1D<float, Stride1D.Dense> outNodes, ArrayView1D<float, Stride1D.Dense> outBounds)
    {
        int o = order[k];
        for (int f = 0; f < F; f++) outNodes[k * F + f] = nodes[o * F + f];
        for (int f = 0; f < 4; f++) outBounds[k * 4 + f] = bounds[o * 4 + f];
    }

    /// <summary>New node k: LOD size (0 for a leaf), renumbered parent and first child, child count.</summary>
    static void ScatterTopologyKernel(Index1D k, ArrayView1D<int, Stride1D.Dense> order, ArrayView1D<int, Stride1D.Dense> newIndex,
        ArrayView1D<int, Stride1D.Dense> parent, ArrayView1D<float, Stride1D.Dense> lodSize, ArrayView1D<int, Stride1D.Dense> firstChild,
        ArrayView1D<int, Stride1D.Dense> childCount, ArrayView1D<int, Stride1D.Dense> childList,
        ArrayView1D<int, Stride1D.Dense> outParent, ArrayView1D<float, Stride1D.Dense> outLodSize,
        ArrayView1D<int, Stride1D.Dense> outFirstChild, ArrayView1D<int, Stride1D.Dense> outChildCount)
    {
        int o = order[k];
        int c = childCount[o];
        outLodSize[k] = c == 0 ? 0f : lodSize[o];
        int p = parent[o];
        outParent[k] = p < 0 ? -1 : newIndex[p];
        outChildCount[k] = c;
        outFirstChild[k] = c == 0 ? -1 : newIndex[childList[firstChild[o]]];
    }
}
