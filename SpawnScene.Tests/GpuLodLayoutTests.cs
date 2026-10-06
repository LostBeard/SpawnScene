using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The GPU breadth-first layout (GpuLodLayout) on the ILGPU CPU accelerator: for a tree GpuLodTree built, it must give
/// exactly what the CPU oracle (LodLayout.BreadthFirst + Reorder) gives for the same tree - every array, bit for bit -
/// and the order it reports must be that permutation.
/// </summary>
public class GpuLodLayoutTests
{
    const int F = SplatFormat.Floats;

    static void HostSort(MemoryBuffer1D<uint, Stride1D.Dense> keys, MemoryBuffer1D<uint, Stride1D.Dense> values, int count)
    {
        var k = keys.GetAsArray1D();
        var v = values.GetAsArray1D();
        var order = Enumerable.Range(0, count).OrderBy(i => k[i]).ThenBy(i => i).ToArray();
        var k2 = (uint[])k.Clone(); var v2 = (uint[])v.Clone();
        for (int i = 0; i < count; i++) { k2[i] = k[order[i]]; v2[i] = v[order[i]]; }
        keys.CopyFromCPU(k2); values.CopyFromCPU(v2);
    }

    [TestCase(3000, 21)]
    [TestCase(20000, 22)]
    public void GpuLayout_EqualsTheCpuOracle(int n, int seed)
    {
        var rows = LodTreeTests.Scene(n, seed);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var leaves = accel.Allocate1D(rows);
        var sizes = Enumerable.Range(0, n).Select(i => LodTree.SizeOf(rows.AsSpan(i * F, F))).OrderBy(s => s).ToArray();
        using var gpu = GpuLodTree.BuildAsync(accel, leaves, n, sizes[n / 2], HostSort).GetAwaiter().GetResult();
        int nodes = gpu.NodeCount;
        var tree = new LodTree
        {
            LeafCount = n, NodeCount = nodes,
            Rows = gpu.Nodes.GetAsArray1D()[..(nodes * F)],
            Parent = gpu.Parent.GetAsArray1D()[..nodes],
            Bounds = gpu.Bounds.GetAsArray1D()[..(nodes * 4)],
            LodSize = gpu.LodSize.GetAsArray1D()[..nodes],
            ChildCount = gpu.ChildCount.GetAsArray1D()[..nodes],
            FirstChild = gpu.FirstChild.GetAsArray1D()[..nodes],
            ChildIndex = gpu.ChildList.GetAsArray1D(),
        };
        var expectOrder = LodLayout.BreadthFirst(tree);
        var expect = LodLayout.Reorder(tree, expectOrder);

        MemoryBuffer1D<int, Stride1D.Dense>? orderBuf = null;
        using var laid = GpuLodLayout.BuildAsync(accel, gpu, o => orderBuf = o).GetAwaiter().GetResult();
        using (orderBuf)
        {
            Assert.That(orderBuf!.GetAsArray1D()[..nodes], Is.EqualTo(expectOrder), "the breadth-first order");
            Assert.That(laid.Nodes.GetAsArray1D()[..(nodes * F)], Is.EqualTo(expect.Rows), "rows");
            Assert.That(laid.Parent.GetAsArray1D()[..nodes], Is.EqualTo(expect.Parent), "parents");
            Assert.That(laid.Bounds.GetAsArray1D()[..(nodes * 4)], Is.EqualTo(expect.Bounds), "bounds");
            Assert.That(laid.LodSize.GetAsArray1D()[..nodes], Is.EqualTo(expect.LodSize), "LOD sizes (leaves 0)");
            Assert.That(laid.ChildCount.GetAsArray1D()[..nodes], Is.EqualTo(expect.ChildCount), "child counts");
            Assert.That(laid.FirstChild.GetAsArray1D()[..nodes], Is.EqualTo(expect.FirstChild), "first children");
        }
        TestContext.Out.WriteLine($"{n} leaves -> {nodes} nodes laid out breadth-first");
    }

    [Test]
    public void MergedSh_EqualsTheCpuOracle_AndKeepsSharedBands()
    {
        const int n = 3000, W = 15;
        var rows = LodTreeTests.Scene(n, 23);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var leaves = accel.Allocate1D(rows);
        var sizes = Enumerable.Range(0, n).Select(i => LodTree.SizeOf(rows.AsSpan(i * F, F))).OrderBy(s => s).ToArray();
        using var gpu = GpuLodTree.BuildAsync(accel, leaves, n, sizes[n / 2], HostSort).GetAwaiter().GetResult();
        MemoryBuffer1D<int, Stride1D.Dense>? orderBuf = null;
        using var laid = GpuLodLayout.BuildAsync(accel, gpu, o => orderBuf = o).GetAwaiter().GetResult();
        using (orderBuf)
        {
            int nodes = laid.NodeCount;
            var rng = new Random(9);
            var leafSh = Enumerable.Range(0, n * W).Select(_ => (float)rng.NextDouble() * 2f - 1f).ToArray();
            using var leafShBuf = accel.Allocate1D(leafSh);
            using var sh = GpuLodLayout.GatherLeafRows(accel, leafShBuf, orderBuf!, n, nodes, W);
            GpuLodLayout.MergeSh(accel, laid, sh, W);
            accel.Synchronize();
            var got = sh.GetAsArray1D()[..(nodes * W)];

            var l = new LodTree
            {
                NodeCount = nodes, LeafCount = n,
                Rows = laid.Nodes.GetAsArray1D()[..(nodes * F)],
                ChildCount = laid.ChildCount.GetAsArray1D()[..nodes],
                FirstChild = laid.FirstChild.GetAsArray1D()[..nodes],
            };
            var order = orderBuf!.GetAsArray1D()[..nodes];
            var expect = new float[nodes * W];
            for (int k = 0; k < nodes; k++)
                if (order[k] < n) Array.Copy(leafSh, order[k] * W, expect, k * W, W);
            LodMerge.MergeShBottomUp(l, expect, W);
            for (int i = 0; i < got.Length; i++)
                Assert.That(got[i], Is.EqualTo(expect[i]).Within(1e-5f), $"node {i / W} band float {i % W}");
            int merged = Enumerable.Range(0, nodes).Count(k => l.ChildCount[k] > 0 && Enumerable.Range(0, W).Any(f => got[k * W + f] != 0f));
            Assert.That(merged, Is.EqualTo(nodes - n), "every merge got bands");

            // Children that share one set of bands hand it to their parent unchanged (a weighted mean of equals).
            var same = Enumerable.Range(0, n * W).Select(i => 0.25f + (i % W) * 0.01f).ToArray();
            using var sameBuf = accel.Allocate1D(same);
            using var sh2 = GpuLodLayout.GatherLeafRows(accel, sameBuf, orderBuf!, n, nodes, W);
            GpuLodLayout.MergeSh(accel, laid, sh2, W);
            accel.Synchronize();
            var got2 = sh2.GetAsArray1D();
            for (int k = 0; k < nodes; k++)
                for (int f = 0; f < W; f++)
                    Assert.That(got2[k * W + f], Is.EqualTo(0.25f + f * 0.01f).Within(1e-5f), $"node {k} band {f}");
        }
    }
}
