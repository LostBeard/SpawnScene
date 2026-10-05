using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The breadth-first streaming layout of an LOD tree (LodLayout, Plans/lod-streaming.md phase B): a permutation in
/// which every parent precedes its children and siblings are contiguous, so fixed-size chunks load top-down and a
/// node's children sit in at most two chunks; and the same cut as the tree it came from, with leaves told apart by a
/// zero LOD size instead of their position.
/// </summary>
public class LodLayoutTests
{
    const int F = SplatFormat.Floats;

    static (LodTree Tree, int[] Order, LodTree Laid) Build(int n, int seed)
    {
        var t = LodTree.Build(LodTreeTests.Scene(n, seed), n);
        var order = LodLayout.BreadthFirst(t);
        return (t, order, LodLayout.Reorder(t, order));
    }

    [Test]
    public void BreadthFirst_IsAPermutation_ParentsFirst_SiblingsContiguous()
    {
        var (t, order, l) = Build(3000, 11);
        Assert.That(order.OrderBy(i => i), Is.EqualTo(Enumerable.Range(0, t.NodeCount)), "every node exactly once");
        for (int i = 0; i < l.NodeCount; i++)
        {
            if (l.Parent[i] >= 0) Assert.That(l.Parent[i], Is.LessThan(i), $"node {i}'s parent comes first");
            for (int k = 0; k < l.ChildCount[i]; k++)
                Assert.That(l.Parent[l.FirstChild[i] + k], Is.EqualTo(i), $"node {i}'s children are the run at {l.FirstChild[i]}");
            // Rows, bounds and the non-leaf sizes travel with their node.
            int o = order[i];
            for (int f = 0; f < F; f++) Assert.That(l.Rows[i * F + f], Is.EqualTo(t.Rows[o * F + f]));
            for (int f = 0; f < 4; f++) Assert.That(l.Bounds[i * 4 + f], Is.EqualTo(t.Bounds[o * 4 + f]));
            Assert.That(l.LodSize[i], Is.EqualTo(t.ChildCount[o] == 0 ? 0f : t.LodSize[o]));
        }
        // Breadth-first: depth never decreases along the order.
        var depth = new int[l.NodeCount];
        for (int i = 0; i < l.NodeCount; i++)
        {
            depth[i] = l.Parent[i] < 0 ? 0 : depth[l.Parent[i]] + 1;
            if (i > 0) Assert.That(depth[i], Is.GreaterThanOrEqualTo(depth[i - 1]), $"node {i}: depth went back");
        }
    }

    [Test]
    public void Chunks_LoadTopDown_ChildrenSpanAtMostTwo()
    {
        var (_, _, l) = Build(3000, 12);
        foreach (int chunk in new[] { 64, 257, 1024 })
        {
            for (int i = 0; i < l.NodeCount; i++)
            {
                if (l.Parent[i] >= 0) Assert.That(l.Parent[i] / chunk, Is.LessThanOrEqualTo(i / chunk), "a parent never lives in a later chunk");
                if (l.ChildCount[i] > 0)
                {
                    int a = l.FirstChild[i] / chunk, b = (l.FirstChild[i] + l.ChildCount[i] - 1) / chunk;
                    Assert.That(b - a, Is.LessThanOrEqualTo(1), $"chunk {chunk}: node {i}'s {l.ChildCount[i]} children span chunks {a}..{b}");
                }
            }
        }
    }

    [Test]
    public void Cut_IsTheTreesCut_LeavesByZeroSize()
    {
        var (t, order, l) = Build(3000, 13);
        var rng = new Random(5);
        for (int trial = 0; trial < 40; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 4 - 2);   // 0.01 .. 100 px
            for (int i = 0; i < l.NodeCount; i++)
                Assert.That(LodLayout.InCut(l, i, cam, 1000f, tau), Is.EqualTo(t.InCut(order[i], cam, 1000f, tau)),
                    $"trial {trial} (tau {tau:G3}): node {i} (was {order[i]})");
        }
    }

    [Test]
    public void ChunkZero_IsACoarseWholeScene()
    {
        // The first chunk alone, drawn as the cut of what it holds, covers every leaf exactly once. The paging rule:
        // a node's children are used only when ALL of them are resident (they span at most two chunks); otherwise the
        // node stands in for its whole subtree. So a resident node is usable when it is a root or its siblings are all
        // resident, and it is drawn (the frontier) when it is usable and its own children are not all resident.
        var (_, _, l) = Build(3000, 14);
        const int chunk = 256;
        int end = Math.Min(chunk, l.NodeCount);
        bool AllChildrenResident(int i) => l.ChildCount[i] > 0 && l.FirstChild[i] + l.ChildCount[i] <= end;
        bool Usable(int i) => i < end && (l.Parent[i] < 0 || AllChildrenResident(l.Parent[i]));
        bool Frontier(int i) => Usable(i) && !AllChildrenResident(i);
        // A straddling parent is what the rule is for: make sure this tree has one at the boundary.
        Assert.That(Enumerable.Range(0, end).Any(i => l.ChildCount[i] > 0 && l.FirstChild[i] < end
            && l.FirstChild[i] + l.ChildCount[i] > end), "some node's children straddle the end of chunk 0");
        int leaves = 0;
        for (int i = 0; i < l.NodeCount; i++)
        {
            if (l.ChildCount[i] != 0) continue;
            leaves++;
            int hits = 0;
            for (int a = i; a >= 0; a = l.Parent[a]) if (Frontier(a)) hits++;
            Assert.That(hits, Is.EqualTo(1), $"leaf {i}: {hits} frontier ancestors in chunk 0");
        }
        Assert.That(leaves, Is.EqualTo(l.LeafCount));
    }
}
