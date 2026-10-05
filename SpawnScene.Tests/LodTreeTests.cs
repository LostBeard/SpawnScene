using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The LOD tree and its cut (LodTree, Plans/lod-streaming.md): every leaf's root path must contain EXACTLY one drawn
/// node from any viewpoint at any threshold (no hole, no double layer); tau -> 0 draws the leaves, tau -> infinity the
/// roots; each parent is the merge of its children.
/// </summary>
public class LodTreeTests
{
    const int F = SplatFormat.Floats;

    static float[] Scene(int n, int seed)
    {
        var rng = new Random(seed);
        var rows = new float[n * F];
        for (int i = 0; i < n; i++)
        {
            int o = i * F;
            // Two "rooms" and a corridor of splats of mixed sizes.
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
    public void Tree_IsWellFormed_AndParentsAreTheirChildrensMerge()
    {
        var rows = Scene(3000, 1);
        var t = LodTree.Build(rows, 3000);
        Assert.That(t.NodeCount, Is.GreaterThan(t.LeafCount), "the tree has internal nodes");
        int roots = Enumerable.Range(0, t.NodeCount).Count(i => t.Parent[i] < 0);
        TestContext.Out.WriteLine($"{t.LeafCount} leaves, {t.NodeCount - t.LeafCount} internal nodes, {roots} root(s)");
        Assert.That(roots, Is.LessThanOrEqualTo(4), "the scene converges to a handful of roots");
        for (int p = t.LeafCount; p < t.NodeCount; p++)
        {
            Assert.That(t.ChildCount[p], Is.GreaterThanOrEqualTo(2), $"node {p} merges at least two");
            var kids = t.ChildIndex.AsSpan(t.FirstChild[p], t.ChildCount[p]).ToArray();
            Assert.That(kids.All(c => t.Parent[c] == p), $"node {p}'s children point back to it");
            var merged = new float[F];
            var childRows = kids.SelectMany(c => t.Rows.AsSpan(c * F, F).ToArray()).ToArray();
            LodMerge.Merge(childRows, kids.Length, merged);
            for (int k = 0; k < F; k++) Assert.That(t.Rows[p * F + k], Is.EqualTo(merged[k]).Within(1e-5f), $"node {p} float {k}");
        }
    }

    [Test]
    public void Cut_DrawsExactlyOneNodeOnEveryPath()
    {
        var rows = Scene(3000, 2);
        var t = LodTree.Build(rows, 3000);
        var rng = new Random(7);
        for (int trial = 0; trial < 40; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 4 - 1);   // 0.1 .. 1000 px
            var drawn = new bool[t.NodeCount];
            for (int i = 0; i < t.NodeCount; i++) drawn[i] = t.InCut(i, cam, 1000f, tau);
            for (int leaf = 0; leaf < t.LeafCount; leaf++)
            {
                int hits = 0;
                for (int n = leaf; n >= 0; n = t.Parent[n]) if (drawn[n]) hits++;
                Assert.That(hits, Is.EqualTo(1), $"trial {trial} (tau {tau:G3}) leaf {leaf}: drawn {hits} times on its path");
            }
        }
    }

    [Test]
    public void Cut_Limits_AreTheLeavesAndTheRoots()
    {
        var rows = Scene(1000, 3);
        var t = LodTree.Build(rows, 1000);
        var cam = new Vector3(8f, 1.5f, -3f);
        var fine = Enumerable.Range(0, t.NodeCount).Where(i => t.InCut(i, cam, 1000f, 1e-9f)).ToArray();
        Assert.That(fine, Is.EqualTo(Enumerable.Range(0, t.LeafCount).ToArray()), "tau -> 0 draws exactly the leaves");
        var coarse = Enumerable.Range(0, t.NodeCount).Where(i => t.InCut(i, cam, 1000f, 1e12f)).ToArray();
        Assert.That(coarse, Is.EqualTo(Enumerable.Range(0, t.NodeCount).Where(i => t.Parent[i] < 0).ToArray()), "tau -> infinity draws the roots");
    }

    [Test]
    public void PixelSize_NeverGrowsFromParentToChild()
    {
        var rows = Scene(3000, 4);
        var t = LodTree.Build(rows, 3000);
        var rng = new Random(9);
        for (int trial = 0; trial < 20; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            for (int c = 0; c < t.NodeCount; c++)
            {
                int p = t.Parent[c];
                if (p < 0) continue;
                Assert.That(t.PixelSize(c, cam, 1000f), Is.LessThanOrEqualTo(t.PixelSize(p, cam, 1000f) * (1 + 1e-5f)), $"node {c} under {p}");
            }
        }
    }
}
