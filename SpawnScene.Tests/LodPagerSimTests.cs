using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// GpuLodPager's slot bookkeeping on the ILGPU CPU accelerator: chunks loaded into pool pages by LodPageKernels (parent
/// SLOT, children's chunk), with the pager's rules - a chunk after the chunks it needs, eviction of the least recently
/// wanted chunk nothing resident needs, chunk 0 pinned - and the paged cull's decision per slot (as
/// GpuSplatSorter.CullAndDistanceLodPagedKernel takes it, without the frustum). The slots it draws, mapped back to
/// nodes, must be LodLayout.InCutPaged's for the resident chunks; with room for every chunk the stream settles on the
/// full cut; with a small pool every leaf path is still drawn exactly once.
/// </summary>
public class LodPagerSimTests
{
    const float Focal = 1000f;

    sealed class Sim : IDisposable
    {
        readonly Accelerator _a;
        readonly LodTree _l;
        readonly int[] _starts;
        public readonly int PageNodes, Pages, Chunks;
        public readonly MemoryBuffer1D<int, Stride1D.Dense> ParentSlot, ChildChunk, ChunkPage, StartsBuf;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Bounds, Size;
        public readonly int[] PageChunk, ChunkPageCpu, Dependents;
        public readonly long[] LastWanted;
        public bool[] PageUsed;
        long _clock;
        readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int> _fill;
        readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, LodPageKernels.SlotParams> _slots;

        public Sim(Accelerator a, LodTree l, int[] starts, int pages)
        {
            _a = a; _l = l; _starts = starts;
            Chunks = starts.Length - 1;
            PageNodes = Enumerable.Range(0, Chunks).Max(c => starts[c + 1] - starts[c]);
            Pages = pages;
            int slots = PageNodes * pages;
            ParentSlot = a.Allocate1D<int>(slots); ChildChunk = a.Allocate1D<int>(slots);
            Bounds = a.Allocate1D<float>(slots * 4L); Size = a.Allocate1D<float>(slots);
            ChunkPage = a.Allocate1D<int>(Chunks);
            StartsBuf = a.Allocate1D(starts);
            PageChunk = Enumerable.Repeat(-1, pages).ToArray();
            ChunkPageCpu = Enumerable.Repeat(-1, Chunks).ToArray();
            Dependents = new int[Chunks];
            LastWanted = new long[Chunks];
            PageUsed = new bool[pages];
            _fill = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int>(LodPageKernels.FillKernel);
            _slots = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LodPageKernels.SlotParams>(LodPageKernels.SlotKernel);
            _fill(slots, ParentSlot.View, 0, -2);
            _fill(Chunks, ChunkPage.View, 0, -1);
            a.Synchronize();
        }

        public void Want(int c) => LastWanted[c] = ++_clock;

        /// <summary>GpuLodPager.LoadWithNeedsAsync, synchronously.</summary>
        public bool Load(int c)
        {
            if (ChunkPageCpu[c] >= 0) return true;
            var needs = LodLayout.ParentChunks(_l, _starts, c);
            foreach (int need in needs) if (!Load(need)) return false;
            foreach (int need in needs) Dependents[need]++;
            int page = Array.IndexOf(PageChunk, -1);
            if (page < 0)
            {
                int victim = -1;
                for (int p = 0; p < Pages; p++)
                {
                    int v = PageChunk[p];
                    if (v <= 0 || Dependents[v] > 0 || PageUsed[p]) continue;
                    if (victim < 0 || LastWanted[v] < LastWanted[PageChunk[victim]]) victim = p;
                }
                if (victim < 0) { foreach (int need in needs) Dependents[need]--; return false; }
                Evict(PageChunk[victim]);
                page = victim;
            }
            int first = _starts[c], n = _starts[c + 1] - first;
            using var parent = _a.Allocate1D(_l.Parent[first..(first + n)]);
            using var firstChild = _a.Allocate1D(_l.FirstChild[first..(first + n)]);
            using var bounds = _a.Allocate1D(_l.Bounds[(first * 4)..((first + n) * 4)]);
            using var size = _a.Allocate1D(_l.LodSize[first..(first + n)]);
            int slot0 = page * PageNodes;
            _fill(PageNodes, ParentSlot.View.SubView(slot0, PageNodes), 0, -2);
            _slots(n, parent.View, firstChild.View, bounds.View, size.View, StartsBuf.View, ChunkPage.View,
                ParentSlot.View, ChildChunk.View, Bounds.View, Size.View,
                new LodPageKernels.SlotParams { Slot0 = slot0, PageNodes = PageNodes, Chunks = Chunks });
            _fill(1, ChunkPage.View.SubView(c, 1), 0, page);
            _a.Synchronize();
            ChunkPageCpu[c] = page; PageChunk[page] = c;
            Want(c);
            return true;
        }

        void Evict(int c)
        {
            int page = ChunkPageCpu[c];
            _fill(PageNodes, ParentSlot.View.SubView(page * PageNodes, PageNodes), 0, -2);
            _fill(1, ChunkPage.View.SubView(c, 1), 0, -1);
            _a.Synchronize();
            ChunkPageCpu[c] = -1; PageChunk[page] = -1;
            foreach (int need in LodLayout.ParentChunks(_l, _starts, c)) Dependents[need]--;
        }

        /// <summary>The paged cull per slot (no frustum): drawn slots and wanted chunks; pages drawn from are marked used.</summary>
        public (bool[] Drawn, HashSet<int> Wanted) Cull(Vector3 cam, float tau)
        {
            var ps = ParentSlot.GetAsArray1D(); var cc = ChildChunk.GetAsArray1D(); var cp = ChunkPage.GetAsArray1D();
            var b = Bounds.GetAsArray1D(); var s = Size.GetAsArray1D();
            float Px(int i)
            {
                float d = Vector3.Distance(cam, new Vector3(b[i * 4], b[i * 4 + 1], b[i * 4 + 2])) - b[i * 4 + 3];
                return s[i] * Focal / MathF.Max(d, 0.1f);
            }
            var drawn = new bool[ps.Length];
            var wanted = new HashSet<int>();
            for (int i = 0; i < ps.Length; i++)
            {
                int parent = ps[i];
                bool take = parent != -2 && (parent < 0 || Px(parent) > tau);
                if (take && Px(i) > tau)
                {
                    if (cc[i] >= 0 && cp[cc[i]] >= 0) take = false;
                    else if (cc[i] >= 0) wanted.Add(cc[i]);
                }
                drawn[i] = take;
            }
            PageUsed = new bool[Pages];
            for (int i = 0; i < drawn.Length; i++) if (drawn[i]) PageUsed[i / PageNodes] = true;
            foreach (int p in Enumerable.Range(0, Pages).Where(p => PageUsed[p] && PageChunk[p] >= 0)) Want(PageChunk[p]);
            return (drawn, wanted);
        }

        /// <summary>The node a slot holds, or -1.</summary>
        public int NodeOf(int slot)
        {
            int c = PageChunk[slot / PageNodes];
            if (c < 0) return -1;
            int off = slot % PageNodes;
            return off < _starts[c + 1] - _starts[c] ? _starts[c] + off : -1;
        }

        public void Dispose()
        {
            ParentSlot.Dispose(); ChildChunk.Dispose(); ChunkPage.Dispose(); StartsBuf.Dispose(); Bounds.Dispose(); Size.Dispose();
        }
    }

    static (LodTree Tree, int[] Order, LodTree Laid, int[] Starts) Build(int n, int seed, int chunk)
    {
        var t = LodTree.Build(LodTreeTests.Scene(n, seed), n);
        var order = LodLayout.BreadthFirst(t);
        var l = LodLayout.Reorder(t, order);
        return (t, order, l, LodLayout.ChunkStarts(l, chunk));
    }

    /// <summary>Stream like the pager: cull, load what is wanted (with needs), until nothing is wanted or nothing loads.</summary>
    static int Stream(Sim sim, Vector3 cam, float tau)
    {
        sim.Load(0);
        for (int round = 0; round < 300; round++)
        {
            var (_, wanted) = sim.Cull(cam, tau);
            int loaded = 0;
            foreach (int c in wanted) { sim.Want(c); if (sim.ChunkPageCpu[c] < 0 && sim.Load(c)) loaded++; }
            if (loaded == 0) return round;
        }
        throw new AssertionException("the stream did not settle");
    }

    [Test]
    public void Pool_WithRoomForEverything_SettlesOnTheFullCut()
    {
        var (t, order, l, starts) = Build(3000, 31, 64);
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        using var sim = new Sim(accel, l, starts, starts.Length - 1);
        var rng = new Random(4);
        for (int trial = 0; trial < 8; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 2);
            int rounds = Stream(sim, cam, tau);
            var (drawn, _) = sim.Cull(cam, tau);
            var drawnNodes = new HashSet<int>(Enumerable.Range(0, drawn.Length).Where(i => drawn[i]).Select(sim.NodeOf));
            Assert.That(drawnNodes.Contains(-1), Is.False, "only slots holding a node draw");
            var expect = Enumerable.Range(0, l.NodeCount).Where(i => t.InCut(order[i], cam, Focal, tau)).ToHashSet();
            Assert.That(drawnNodes.SetEquals(expect), $"trial {trial} (tau {tau:G3}, {rounds} rounds): {drawnNodes.Count} drawn vs the cut's {expect.Count}");
        }
    }

    [Test]
    public void Pool_TooSmall_EvictsAndStillDrawsEveryPathOnce()
    {
        var (_, _, l, starts) = Build(3000, 32, 64);
        using var context = Context.Create(b => b.CPU());
        using var accel = context.CreateCPUAccelerator(0);
        using var sim = new Sim(accel, l, starts, 10);   // of ~60 chunks
        var rng = new Random(5);
        int evictedViews = 0;
        for (int trial = 0; trial < 12; trial++)
        {
            var cam = new Vector3((float)rng.NextDouble() * 20f - 2f, (float)rng.NextDouble() * 4f, (float)rng.NextDouble() * 8f - 1.5f);
            float tau = (float)Math.Pow(10, rng.NextDouble() * 1.5);
            Stream(sim, cam, tau);
            var resident = Enumerable.Range(0, sim.Chunks).Where(c => sim.ChunkPageCpu[c] >= 0).ToHashSet();
            var (drawn, _) = sim.Cull(cam, tau);
            var drawnNodes = Enumerable.Range(0, drawn.Length).Where(i => drawn[i]).Select(sim.NodeOf).ToHashSet();
            // The slots agree with the oracle on the same resident chunks...
            var expect = Enumerable.Range(0, l.NodeCount).Where(i => LodLayout.InCutPaged(l, i, cam, Focal, tau, starts, resident.Contains, out _)).ToHashSet();
            Assert.That(drawnNodes.SetEquals(expect), $"trial {trial}: slots draw {drawnNodes.Count}, the paged cut {expect.Count}");
            // ...and every leaf path is drawn exactly once, whatever was evicted.
            for (int i = 0; i < l.NodeCount; i++)
            {
                if (l.ChildCount[i] != 0) continue;
                int hits = 0;
                for (int a = i; a >= 0; a = l.Parent[a]) if (drawnNodes.Contains(a)) hits++;
                Assert.That(hits, Is.EqualTo(1), $"trial {trial}: leaf {i} drawn {hits} times ({resident.Count} chunks resident)");
            }
            if (resident.Count == sim.Pages) evictedViews++;
        }
        Assert.That(evictedViews, Is.GreaterThan(0), "the pool filled up (eviction ran)");
    }
}
