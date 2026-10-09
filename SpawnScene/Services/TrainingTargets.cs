using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// The training photographs when they do not all fit the GPU target budget: every view's pixels stay in browser memory
/// (packed RGBA8, as the stack stores them - JS typed arrays, never the .NET heap) and a fixed number of GPU slots holds
/// the views in use, least recently used out. A training step needs ONE photo (SetTargetFrom copies it into the working
/// target), so a miss costs one w x h x 4 byte upload, not a decode.
/// <para>
/// TJ 2026-10-08: "there is no reason to cut corners". Before this, a run whose photos did not fit the budget was trained
/// at a SMALLER RESOLUTION (TrainingTimeEstimate.TrainingSize shrank every view): Mip-NeRF 360 Kitchen trained at 1470x980
/// instead of 1558x1039, Bonsai and Room likewise - and lost to gsplat there.
/// </para>
/// </summary>
public sealed class TrainingTargets : IDisposable
{
    readonly WebGPUAccelerator _accel;
    readonly SplatTrainerGpu _trainer;
    readonly Uint8Array?[] _host;
    readonly int[] _slotView;
    readonly Dictionary<int, int> _viewSlot = new();
    readonly LinkedList<int> _lru = new();               // slots, most recently used first
    readonly Dictionary<int, LinkedListNode<int>> _lruNode = new();

    /// <summary>The GPU stack: <see cref="Slots"/> views of w x h packed RGBA8.</summary>
    public MemoryBuffer1D<uint, Stride1D.Dense> Buffer { get; }
    public int Slots { get; }
    public int Views => _host.Length;
    public long Uploads { get; private set; }

    public TrainingTargets(WebGPUAccelerator accel, SplatTrainerGpu trainer, int views, int slots, long pixelsPerView)
    {
        _accel = accel;
        _trainer = trainer;
        _host = new Uint8Array?[views];
        Slots = Math.Max(1, Math.Min(slots, views));
        _slotView = Enumerable.Repeat(-1, Slots).ToArray();
        Buffer = accel.Allocate1D<uint>((long)Slots * pixelsPerView);
    }

    /// <summary>Keep a view's pixels (a JS-side copy of <paramref name="rgba"/>).</summary>
    public void Store(int view, Uint8Array rgba)
    {
        _host[view]?.Dispose();
        _host[view] = new Uint8Array(rgba);
    }

    /// <summary>The slot holding <paramref name="view"/>, uploading it into the least recently used one if it is not
    /// resident. Work already queued on the GPU reads the evicted slot before the upload lands (queue order).</summary>
    public int Slot(int view)
    {
        if (_viewSlot.TryGetValue(view, out int s))
        {
            Touch(s);
            return s;
        }
        var pixels = _host[view] ?? throw new InvalidOperationException($"training view {view} has no stored pixels");
        int slot = System.Array.IndexOf(_slotView, -1);
        if (slot < 0)
        {
            slot = _lru.Last!.Value;
            _viewSlot.Remove(_slotView[slot]);
        }
        // ILGPU kernels still pending would otherwise run AFTER this upload (it goes straight to the queue).
        _accel.FlushPendingCommands();
        _trainer.UploadTargetFrom(Buffer, slot, pixels);
        Uploads++;
        _slotView[slot] = view;
        _viewSlot[view] = slot;
        Touch(slot);
        return slot;
    }

    void Touch(int slot)
    {
        if (_lruNode.TryGetValue(slot, out var node)) _lru.Remove(node);
        _lruNode[slot] = _lru.AddFirst(slot);
    }

    public void Dispose()
    {
        if (ReferenceEquals(_trainer.StreamedTargets, this)) _trainer.StreamedTargets = null;
        foreach (var h in _host) h?.Dispose();
        Buffer.Dispose();
    }
}
