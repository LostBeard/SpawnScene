using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Streams a .spawnscene v3 LOD tree (<see cref="LodChunkFile"/>) through a fixed GPU pool of pages, one chunk a page
/// (Plans/lod-streaming.md, phase B): the renderer draws the pool through the paged cut (LodLayout.InCutPaged,
/// GpuSplatSorter.SetLodPaged), which draws a node in place of children whose chunk is not resident and asks for that
/// chunk; the pager loads it - after the chunks it needs (its nodes' parents), so the resident set stays closed -
/// biggest stand-in first, and when the pool is full evicts a page nothing resident depends on whose loss would matter
/// less (by screen size) than the chunk it makes room for. Chunk 0 (the top of the tree) is loaded first and never
/// evicted, so a coarse whole scene is on screen at once.
/// </summary>
public sealed class GpuLodPager : IDisposable
{
    const int F = SplatFormat.Floats;

    /// <summary>A chunk's bytes, gunzipped (LodChunkFile.Layout).</summary>
    public delegate Task<ArrayBuffer> ChunkBytes(LodChunkFile.Chunk chunk);

    readonly WebGPUAccelerator _a;
    readonly GpuGaussianRenderer _r;
    readonly LodChunkFile.Header3 _h;
    readonly ChunkBytes _bytes;
    readonly bool _withSh;

    /// <summary>Slots a page (the file's largest chunk) and pages in the pool.</summary>
    public int PageNodes { get; }
    public int Pages { get; }

    // GPU, per slot: parent slot (-1 root, -2 empty), child chunk (-1 leaf), sphere, LOD size; per chunk: page, want flag.
    readonly MemoryBuffer1D<int, Stride1D.Dense> _parentSlot, _childChunk, _chunkPage, _want, _starts;
    readonly MemoryBuffer1D<float, Stride1D.Dense> _bounds, _size;

    // CPU mirror of the residency.
    readonly int[] _pageChunk, _chunkPageCpu, _dependents;
    readonly long[] _lastWanted;
    int[] _pagePriority;   // latest cut: 0 = not drawn from on screen, else 1 + 16 x px of what eviction would show
    readonly int[] _wantPriority;
    // Cut readbacks so far, and the count when each page was filled: a page is not judged unused until two readbacks
    // have come in since its load (the latest one may be from a cut sorted before the load). Judging it by the older
    // cut evicted fresh chunks before any cut saw them: a 600K pool on TruckFull loaded 2, evicted 2, forever.
    long _readbacks;
    readonly long[] _pageLoadedAt;
    readonly List<int> _queue = new();   // wanted chunks, highest priority first
    bool _pumping, _poolFullLogged, _disposed;

    public int ResidentChunks { get; private set; }
    public int ChunkCount => _h.Chunks.Length;
    public int Loads { get; private set; }
    public int Evictions { get; private set; }

    readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int> _fill;
    readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, LodPageKernels.SlotParams> _slots;

    GpuLodPager(WebGPUAccelerator a, GpuGaussianRenderer r, LodChunkFile.Header3 h, ChunkBytes bytes, int pages)
    {
        _a = a; _r = r; _h = h; _bytes = bytes;
        _withSh = h.ShDegree > 0;
        PageNodes = h.Chunks.Max(c => c.Count);
        Pages = pages;
        int slots = PageNodes * Pages, chunks = h.Chunks.Length;
        _parentSlot = a.Allocate1D<int>(slots);
        _childChunk = a.Allocate1D<int>(slots);
        _bounds = a.Allocate1D<float>((long)slots * 4);
        _size = a.Allocate1D<float>(slots);
        _chunkPage = a.Allocate1D<int>(chunks);
        _want = a.Allocate1D<int>(chunks + pages);   // want flag a chunk, then used flag a page
        _pagePriority = new int[pages];
        _wantPriority = new int[chunks];
        _pageLoadedAt = new long[pages];
        _starts = a.Allocate1D(h.Chunks.Select(c => c.First).Append(h.NodeCount).ToArray());
        _pageChunk = Enumerable.Repeat(-1, pages).ToArray();
        _chunkPageCpu = Enumerable.Repeat(-1, chunks).ToArray();
        _dependents = new int[chunks];
        _lastWanted = new long[chunks];
        _fill = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, int, int>(LodPageKernels.FillKernel);
        _slots = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LodPageKernels.SlotParams>(LodPageKernels.SlotKernel);
        _fill(slots, _parentSlot.View, 0, -2);
        _fill(chunks, _chunkPage.View, 0, -1);
        _want.MemSetToZero();
    }

    /// <summary>
    /// Stream <paramref name="h"/>'s tree through a pool of about <paramref name="poolNodes"/> slots (whole pages, at
    /// least enough for chunk 0 and a few more), drawn at <paramref name="tau"/> px; chunk 0 is on screen when this returns.
    /// </summary>
    public static async Task<GpuLodPager> CreateAsync(WebGPUAccelerator a, GpuGaussianRenderer r, LodChunkFile.Header3 h,
        ChunkBytes bytes, int poolNodes, float tau)
    {
        int pageNodes = h.Chunks.Max(c => c.Count);
        int pages = Math.Clamp((poolNodes + pageNodes - 1) / pageNodes, Math.Min(4, h.Chunks.Length), h.Chunks.Length);
        var pager = new GpuLodPager(a, r, h, bytes, pages);
        var pool = a.Allocate1D<float>((long)pages * pageNodes * F);
        pool.MemSetToZero();
        await a.SynchronizeAsync();
        await r.InstallPagedLodAsync(pool, pages * pageNodes, h.ShDegree, pager._parentSlot, pager._bounds, pager._size,
            pager._childChunk, pager._chunkPage, pager._want, h.Chunks.Length, pageNodes, tau);
        await pager.LoadWithNeedsAsync(0);
        // Only now: a want arriving during chunk 0's load would start a second, concurrent load of it.
        r.LodChunksWanted += pager.OnWanted;
        Console.WriteLine($"[LOD] pager: {h.Chunks.Length} chunks, pool {pages} pages x {pageNodes:N0} slots " +
            $"({(long)pages * pageNodes:N0} of {h.NodeCount:N0} nodes)");
        return pager;
    }

    void OnWanted(int[] chunkPriority, int[] pagePriority)
    {
        if (_disposed) return;
        _pagePriority = pagePriority;
        _readbacks++;
        long now = Environment.TickCount64;
        for (int p = 0; p < Pages && p < pagePriority.Length; p++)
            if (pagePriority[p] > 0 && _pageChunk[p] >= 0) _lastWanted[_pageChunk[p]] = now;
        // The queue is what THIS cut wants, biggest stand-in first (the previous cut's wants are stale).
        _queue.Clear();
        for (int c = 0; c < chunkPriority.Length; c++)
        {
            _wantPriority[c] = chunkPriority[c];
            if (chunkPriority[c] > 0 && _chunkPageCpu[c] < 0) { _lastWanted[c] = now; _queue.Add(c); }
        }
        _queue.Sort((x, y) => chunkPriority[y].CompareTo(chunkPriority[x]));
        if (!_pumping && _queue.Count > 0) _ = PumpAsync();
    }

    async Task PumpAsync()
    {
        _pumping = true;
        int loads = Loads, evictions = Evictions;
        long t0 = Environment.TickCount64;
        try
        {
            while (_queue.Count > 0 && !_disposed)
            {
                int c = _queue[0];
                _queue.RemoveAt(0);
                if (!await LoadWithNeedsAsync(c, _wantPriority[c]))
                {
                    // The pool holds what matters more than anything still queued: stop; the next cut asks again.
                    _queue.Clear();
                    break;
                }
            }
        }
        catch (Exception ex) { if (!_disposed) Console.WriteLine($"[LOD] pager: {ex.Message}"); }
        finally
        {
            if (!_disposed && Loads > loads)
                Console.WriteLine($"[LOD] pager: +{Loads - loads} chunks ({Evictions - evictions} evicted) in " +
                    $"{(Environment.TickCount64 - t0) / 1000.0:F1}s; {ResidentChunks} of {_h.Chunks.Length} resident");
            _pumping = false;
            if (_disposed) _starts.Dispose();
        }
    }

    /// <summary>
    /// Make chunk <paramref name="c"/> resident, the chunks it needs first, for a stand-in of priority
    /// <paramref name="priority"/>. False when no page may be given up for it.
    /// </summary>
    async Task<bool> LoadWithNeedsAsync(int c, int priority = int.MaxValue)
    {
        if (_chunkPageCpu[c] >= 0) return true;
        var needs = _h.Chunks[c].Needs;
        foreach (int need in needs)
            if (!await LoadWithNeedsAsync(need, priority)) return false;
        // Hold what it needs while a page is found (they must not be evicted for it).
        foreach (int need in needs) _dependents[need]++;
        int page = FreePage(priority);
        if (page < 0)
        {
            foreach (int need in needs) _dependents[need]--;
            if (!_poolFullLogged) { _poolFullLogged = true; Console.WriteLine($"[LOD] pager: pool full ({Pages} pages), the view keeps coarser nodes"); }
            return false;
        }
        await LoadIntoPageAsync(c, page);
        return true;
    }

    /// <summary>
    /// A free page, or one freed by evicting a chunk nothing resident needs, judged by the latest cut: one it did not
    /// draw from (least recently used first), else the one whose loss would show the smallest stand-in - and only if
    /// that matters well under <paramref name="priority"/> (1.5x: without the gap two chunks of a full pool trade
    /// places forever; a 600K pool on TruckFull did, and filled itself with background while the near truck stayed
    /// coarse). A page loaded since the previous readback is never judged (no cut has seen it yet). A sort still in
    /// flight may draw from an evicted page: for that one sort its slots show the new chunk's rows - a rare glitch.
    /// </summary>
    int FreePage(int priority)
    {
        for (int p = 0; p < Pages; p++) if (_pageChunk[p] < 0) return p;
        int victim = -1;
        for (int p = 0; p < Pages; p++)
        {
            int c = _pageChunk[p];
            if (c <= 0 || _dependents[c] > 0 || _readbacks - _pageLoadedAt[p] < 2) continue;
            int pp = p < _pagePriority.Length ? _pagePriority[p] : 0;
            if (pp > 0 && (long)pp * 3 >= (long)priority * 2) continue;   // matters too much to give up for this
            if (victim < 0) { victim = p; continue; }
            int vp = victim < _pagePriority.Length ? _pagePriority[victim] : 0;
            if (pp < vp || (pp == vp && _lastWanted[c] < _lastWanted[_pageChunk[victim]])) victim = p;
        }
        if (victim < 0) return -1;
        Evict(_pageChunk[victim]);
        return victim;
    }

    void Evict(int c)
    {
        int page = _chunkPageCpu[c];
        _fill(PageNodes, _parentSlot.View.SubView((long)page * PageNodes, PageNodes), 0, -2);
        _fill(1, _chunkPage.View.SubView(c, 1), 0, -1);
        _chunkPageCpu[c] = -1;
        _pageChunk[page] = -1;
        foreach (int need in _h.Chunks[c].Needs) _dependents[need]--;
        ResidentChunks--;
        Evictions++;
    }

    async Task LoadIntoPageAsync(int c, int page)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuLodPager));
        var chunk = _h.Chunks[c];
        int n = chunk.Count;
        using var raw = await _bytes(chunk);
        if (_disposed) throw new ObjectDisposedException(nameof(GpuLodPager));   // the scene changed while it was read
        var L = LodChunkFile.Layout.Of(n, _withSh);
        if (raw.ByteLength != L.End) throw new InvalidDataException($"chunk {c}: {raw.ByteLength} bytes, expected {L.End}");
        var temps = new List<IDisposable>();
        try
        {
            var geo = _a.Allocate1D<uint>((long)n * SceneCodec.GeoWords); temps.Add(geo);
            var app = _a.Allocate1D<uint>((long)n * SceneCodec.AppWords); temps.Add(app);
            var shw = _withSh ? _a.Allocate1D<uint>((long)n * SceneCodec.ShWords) : null; if (shw != null) temps.Add(shw);
            var parent = _a.Allocate1D<int>(n); temps.Add(parent);
            var firstChild = _a.Allocate1D<int>(n); temps.Add(firstChild);
            var bounds = _a.Allocate1D<float>((long)n * 4); temps.Add(bounds);
            var size = _a.Allocate1D<float>(n); temps.Add(size);
            // File I/O: the chunk's parts straight to the GPU.
            _r.WriteIlgpu(geo, 0, raw, (int)L.Geo, L.App - L.Geo);
            _r.WriteIlgpu(app, 0, raw, (int)L.App, L.Sh - L.App);
            if (shw != null) _r.WriteIlgpu(shw, 0, raw, (int)L.Sh, L.Parent - L.Sh);
            _r.WriteIlgpu(parent, 0, raw, (int)L.Parent, L.FirstChild - L.Parent);
            _r.WriteIlgpu(firstChild, 0, raw, (int)L.FirstChild, L.Bounds - L.FirstChild);
            _r.WriteIlgpu(bounds, 0, raw, (int)L.Bounds, L.LodSize - L.Bounds);
            _r.WriteIlgpu(size, 0, raw, (int)L.LodSize, L.End - L.LodSize);
            var (rows, rowsSh) = SceneCodec.Decode(_a, geo, app, shw, LodChunkFile.FrameOf(chunk));
            temps.Add(rows);
            if (rowsSh != null) temps.AddRange(rowsSh);

            long slot0 = (long)page * PageNodes;
            var pool = _r.PackedSplatBuffer ?? throw new InvalidOperationException("the LOD pool is gone");
            pool.View.SubView(slot0 * F, (long)n * F).CopyFrom(rows.View);
            _fill(PageNodes, _parentSlot.View.SubView(slot0, PageNodes), 0, -2);
            _slots(n, parent.View, firstChild.View, bounds.View, size.View, _starts.View, _chunkPage.View,
                _parentSlot.View, _childChunk.View, _bounds.View, _size.View,
                new LodPageKernels.SlotParams { Slot0 = (int)slot0, PageNodes = PageNodes, Chunks = _h.Chunks.Length });
            _fill(1, _chunkPage.View.SubView(c, 1), 0, page);
            await _a.SynchronizeAsync();
            if (rowsSh != null) for (int p = 0; p < rowsSh.Length; p++) _r.WriteShRows(p, rowsSh[p], slot0, n);
            await _a.SynchronizeAsync();
        }
        finally { foreach (var t in temps) t.Dispose(); }
        _chunkPageCpu[c] = page;
        _pageChunk[page] = c;
        _pageLoadedAt[page] = _readbacks;
        _lastWanted[c] = Environment.TickCount64;
        ResidentChunks++;
        Loads++;
        _r.RequestResort();
    }

    /// <summary>Stop streaming (another scene is opening). The cut's arrays are the sorter's, dropped with the scene.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _r.LodChunksWanted -= OnWanted;
        if (!_pumping) _starts.Dispose();
    }
}
