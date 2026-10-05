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

    /// <summary>A tree in the streaming order (breadth-first) with its run-aligned chunk starts.</summary>
    static (LodTree Tree, int[] Order, LodTree Laid, int[] Starts) Chunked(float[] rows, int n, int chunk)
    {
        var t = LodTree.Build(rows, n);
        var order = LodLayout.BreadthFirst(t);
        var l = LodLayout.Reorder(t, order);
        return (t, order, l, LodLayout.ChunkStarts(l, chunk));
    }

    /// <summary>Add, transitively, the chunks each resident chunk needs (LodLayout.ParentChunks).</summary>
    static HashSet<int> Close(LodTree l, int[] starts, IEnumerable<int> seed)
    {
        var set = new HashSet<int>(seed);
        var todo = new Stack<int>(set);
        while (todo.Count > 0)
            foreach (int c in LodLayout.ParentChunks(l, starts, todo.Pop()))
                if (set.Add(c)) todo.Push(c);
        return set;
    }

    [Test]
    public void ChunkStarts_RunsWholeInOneChunk_ChunksFull()
    {
        foreach (int max in new[] { 64, 200, 1024 })
        {
            var (t, order, l, starts) = Chunked(LodTreeTests.Scene(3000, 17), 3000, max);
            Assert.That(order.OrderBy(i => i), Is.EqualTo(Enumerable.Range(0, t.NodeCount)), $"max {max}: every node once");
            Assert.That(starts[0], Is.EqualTo(0));
            Assert.That(starts[^1], Is.EqualTo(l.NodeCount));
            for (int c = 0; c + 1 < starts.Length; c++)
                Assert.That(starts[c + 1] - starts[c], Is.InRange(1, max), $"max {max}: chunk {c} size");
            for (int i = 0; i < l.NodeCount; i++)
            {
                if (l.Parent[i] >= 0) Assert.That(l.Parent[i], Is.LessThan(i), $"max {max}: node {i}'s parent comes first");
                if (l.ChildCount[i] == 0) continue;
                for (int k = 0; k < l.ChildCount[i]; k++) Assert.That(l.Parent[l.FirstChild[i] + k], Is.EqualTo(i), "children contiguous");
                Assert.That(LodLayout.ChunkOf(starts, l.FirstChild[i]), Is.EqualTo(LodLayout.ChunkOf(starts, l.FirstChild[i] + l.ChildCount[i] - 1)),
                    $"max {max}: node {i}'s children share a chunk");
            }
            int chunks = starts.Length - 1;
            Assert.That(chunks, Is.LessThanOrEqualTo((int)Math.Ceiling(l.NodeCount / (double)max * 1.5)), $"max {max}: {chunks} chunks, not mostly full");
        }
    }

    [Test]
    public void PagedCut_AnyClosedResidentSet_DrawsEveryPathOnce()
    {
        var (_, _, l, starts) = Chunked(LodTreeTests.Scene(3000, 15), 3000, 64);
        int chunks = starts.Length - 1;
        var rng = new Random(9);
        for (int trial = 0; trial < 60; trial++)
        {
            var seed = Enumerable.Range(0, chunks).Where(_ => rng.NextDouble() < 0.15).Append(0);
            var resident = Close(l, starts, seed);
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 4 - 2);
            var drawn = new bool[l.NodeCount];
            for (int i = 0; i < l.NodeCount; i++) drawn[i] = LodLayout.InCutPaged(l, i, cam, 1000f, tau, starts, resident.Contains, out _);
            for (int i = 0; i < l.NodeCount; i++)
            {
                if (l.ChildCount[i] != 0) continue;
                int hits = 0;
                for (int a = i; a >= 0; a = l.Parent[a]) if (drawn[a]) hits++;
                Assert.That(hits, Is.EqualTo(1), $"trial {trial} ({resident.Count}/{chunks} chunks, tau {tau:G3}): leaf {i} drawn {hits} times");
            }
        }
    }

    [Test]
    public void PagedCut_StreamingConverges_ToTheFullCut()
    {
        // From chunk 0 alone, load (with their parents' chunks) the chunks the cut asks for until it asks for none: the
        // drawn set is then exactly the fully resident cut.
        var (t, order, l, starts) = Chunked(LodTreeTests.Scene(3000, 16), 3000, 64);
        var rng = new Random(3);
        for (int trial = 0; trial < 20; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 2);   // 1 .. 100 px
            var (resident, rounds) = Stream(l, starts, cam, tau);
            for (int i = 0; i < l.NodeCount; i++)
                Assert.That(LodLayout.InCutPaged(l, i, cam, 1000f, tau, starts, resident.Contains, out _),
                    Is.EqualTo(t.InCut(order[i], cam, 1000f, tau)), $"trial {trial}: node {i} after {rounds} rounds");
        }
    }

    [Test]
    public void PagedCut_AViewLoadsOnlyTheChunksItsCutNeeds()
    {
        // A long building - 20 rooms along x over 400 units - seen from one end. Streaming must load exactly the chunks
        // that hold a node of the full cut or an ancestor of one, closed under ParentChunks (the smallest resident set
        // the paged cut is exact on - nothing speculative), and a coarse view few of them.
        const int n = 20000;
        var rows = LodTreeTests.Scene(n, 18);
        var rng = new Random(18);
        for (int i = 0; i < n; i++) rows[i * F] += rng.Next(20) * 20f;
        var (_, _, l, starts) = Chunked(rows, n, 256);
        int chunks = starts.Length - 1;
        var cam = new Vector3(-3f, 1.5f, 2.5f);
        foreach (float tau in new[] { 2f, 10f, 50f })
        {
            var needed = new HashSet<int>();
            var on = new bool[l.NodeCount];
            for (int i = 0; i < l.NodeCount; i++)
                if (LodLayout.InCut(l, i, cam, 1000f, tau))
                    for (int a = i; a >= 0 && !on[a]; a = l.Parent[a]) { on[a] = true; needed.Add(LodLayout.ChunkOf(starts, a)); }
            var (resident, rounds) = Stream(l, starts, cam, tau);
            var minimal = Close(l, starts, needed);
            TestContext.Out.WriteLine($"tau {tau}: {on.Count(x => x)} of {l.NodeCount} nodes needed in {needed.Count} chunks " +
                $"({minimal.Count} closed); streamed {resident.Count} of {chunks} in {rounds} rounds");
            Assert.That(resident.OrderBy(c => c).ToArray(), Is.EqualTo(minimal.OrderBy(c => c).ToArray()), $"tau {tau}: exactly the closed chunks the cut needs");
            if (tau >= 50f) Assert.That(resident.Count, Is.LessThan(chunks / 4), "a coarse view streams few chunks");
        }
    }

    /// <summary>Stream from chunk 0 until the cut asks for nothing more: the resident set and the rounds it took.</summary>
    static (HashSet<int> Resident, int Rounds) Stream(LodTree l, int[] starts, Vector3 cam, float tau)
    {
        var resident = Close(l, starts, new[] { 0 });
        int rounds = 0;
        while (true)
        {
            var want = new HashSet<int>();
            for (int i = 0; i < l.NodeCount; i++)
                if (LodLayout.InCutPaged(l, i, cam, 1000f, tau, starts, resident.Contains, out bool wants) && wants)
                    want.Add(LodLayout.ChunkOf(starts, l.FirstChild[i]));
            if (want.Count == 0) return (resident, rounds);
            resident = Close(l, starts, resident.Concat(want));
            if (++rounds >= 200) throw new AssertionException("streaming did not settle");
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
