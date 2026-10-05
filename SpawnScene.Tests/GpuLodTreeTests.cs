using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The GPU LOD tree build (GpuLodTree) on the ILGPU CPU accelerator, with a host stable sort standing in for the WGSL
/// radix sort: the tree it builds must satisfy what LodTree's oracle guarantees - exactly one drawn node on every
/// root-to-leaf path from any view, parents equal to the merge of their children (float vs the double oracle).
/// </summary>
public class GpuLodTreeTests
{
    const int F = SplatFormat.Floats;

    static void HostSort(MemoryBuffer1D<uint, Stride1D.Dense> keys, MemoryBuffer1D<uint, Stride1D.Dense> values, int count)
    {
        var k = keys.GetAsArray1D();
        var v = values.GetAsArray1D();
        var order = Enumerable.Range(0, count).OrderBy(i => k[i]).ThenBy(i => i).ToArray();   // stable
        var k2 = (uint[])k.Clone(); var v2 = (uint[])v.Clone();
        for (int i = 0; i < count; i++) { k2[i] = k[order[i]]; v2[i] = v[order[i]]; }
        keys.CopyFromCPU(k2); values.CopyFromCPU(v2);
    }

    static (LodTree tree, GpuLodTree gpu) Build(int n, int seed)
    {
        var rows = LodTreeTestsScene(n, seed);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var leaves = accel.Allocate1D(rows);
        var sizes = Enumerable.Range(0, n).Select(i => LodTree.SizeOf(rows.AsSpan(i * F, F))).OrderBy(s => s).ToArray();
        var gpu = GpuLodTree.BuildAsync(accel, leaves, n, sizes[n / 2], HostSort).GetAwaiter().GetResult();
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
        gpu.Dispose();
        return (tree, gpu);
    }

    // Same synthetic scene as LodTreeTests (two rooms and a corridor of mixed splat sizes).
    static float[] LodTreeTestsScene(int n, int seed)
    {
        var rng = new Random(seed);
        var rows = new float[n * F];
        for (int i = 0; i < n; i++)
        {
            int o = i * F;
            float room = rng.Next(3) * 6f;
            rows[o] = room + (float)rng.NextDouble() * 5f; rows[o + 1] = (float)rng.NextDouble() * 3f; rows[o + 2] = (float)rng.NextDouble() * 5f;
            rows[o + 3] = (float)rng.NextDouble(); rows[o + 4] = (float)rng.NextDouble(); rows[o + 5] = (float)rng.NextDouble();
            float s = rng.NextDouble() < 0.1 ? 0.2f : 0.01f + (float)rng.NextDouble() * 0.04f;
            rows[o + 6] = s; rows[o + 7] = s * 0.7f; rows[o + 8] = s * 0.2f;
            rows[o + 9] = 0.1f + (float)rng.NextDouble() * 0.9f;
            var q = Quaternion.Normalize(new Quaternion((float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, 1f));
            rows[o + 10] = q.X; rows[o + 11] = q.Y; rows[o + 12] = q.Z; rows[o + 13] = q.W;
        }
        return rows;
    }

    [Test]
    public void GpuTree_CutIsExact_AndParentsAreTheirChildrensMerge()
    {
        var (t, _) = Build(3000, 2);
        int roots = Enumerable.Range(0, t.NodeCount).Count(i => t.Parent[i] < 0);
        TestContext.Out.WriteLine($"{t.LeafCount} leaves, {t.NodeCount - t.LeafCount} internal nodes, {roots} root(s)");
        Assert.That(t.NodeCount, Is.GreaterThan(t.LeafCount));
        Assert.That(roots, Is.LessThanOrEqualTo(4));

        // Parents vs the double-precision oracle.
        for (int p = t.LeafCount; p < t.NodeCount; p++)
        {
            Assert.That(t.ChildCount[p], Is.GreaterThanOrEqualTo(2), $"node {p}");
            var kids = t.ChildIndex.AsSpan(t.FirstChild[p], t.ChildCount[p]).ToArray();
            Assert.That(kids.All(c => t.Parent[c] == p), $"node {p}'s children point back");
            var expect = new float[F];
            LodMerge.Merge(kids.SelectMany(c => t.Rows.AsSpan(c * F, F).ToArray()).ToArray(), kids.Length, expect);
            var got = t.Rows.AsSpan(p * F, F).ToArray();
            for (int k = 0; k < 6; k++) Assert.That(got[k], Is.EqualTo(expect[k]).Within(1e-4f), $"node {p} float {k}");
            var cg = SplatCovariance.Cov3DFromScaleQuat(got[6], got[7], got[8], new SplatCovariance.Quat { X = got[10], Y = got[11], Z = got[12], W = got[13] });
            var ce = SplatCovariance.Cov3DFromScaleQuat(expect[6], expect[7], expect[8], new SplatCovariance.Quat { X = expect[10], Y = expect[11], Z = expect[12], W = expect[13] });
            float tol = 1e-3f * MathF.Max(ce.M00, MathF.Max(ce.M11, ce.M22)) + 1e-9f;
            Assert.That(cg.M00, Is.EqualTo(ce.M00).Within(tol), $"node {p} cov xx");
            Assert.That(cg.M01, Is.EqualTo(ce.M01).Within(tol), $"node {p} cov xy");
            Assert.That(cg.M12, Is.EqualTo(ce.M12).Within(tol), $"node {p} cov yz");
            Assert.That(cg.M22, Is.EqualTo(ce.M22).Within(tol), $"node {p} cov zz");
            Assert.That(got[9], Is.EqualTo(expect[9]).Within(2e-3f), $"node {p} opacity");
        }

        // The cut: exactly one drawn node on every path.
        var rng = new Random(7);
        for (int trial = 0; trial < 30; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 4 - 1);
            var drawn = new bool[t.NodeCount];
            for (int i = 0; i < t.NodeCount; i++) drawn[i] = t.InCut(i, cam, 1000f, tau);
            for (int leaf = 0; leaf < t.LeafCount; leaf++)
            {
                int hits = 0;
                for (int nd = leaf; nd >= 0; nd = t.Parent[nd]) if (drawn[nd]) hits++;
                Assert.That(hits, Is.EqualTo(1), $"trial {trial} leaf {leaf}");
            }
        }
    }
}
