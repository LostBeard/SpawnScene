using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Floater census (SplatTrainerShaders.FloaterCensus): per splat, how much of its blending weight over the training
/// photos sat in FRONT of the surface those photos saw. TJ's Bicycle (2026-10-07) looked right at every photo's pose and
/// was full of faint floaters the moment the camera moved; every held-out score sat beside a photo and missed them.
/// Everything stays on the GPU; the host reads a 23-word histogram.
/// </summary>
public sealed partial class SplatTrainerGpu
{
    /// <summary>Fixed-point scale of the per-view weight sums (1/1024 of a pixel).</summary>
    const float FloaterFixedScale = 1024f;

    GPUComputePipeline? _floaterCensus, _floaterFold, _floaterClassify;
    MemoryBuffer1D<uint, Stride1D.Dense>? _floaterFixed;     // 2 per splat, one view, fixed point
    MemoryBuffer1D<float, Stride1D.Dense>? _floaterTotals;   // 2 per splat over the views: total, front
    MemoryBuffer1D<uint, Stride1D.Dense>? _floaterHist;      // 23 words
    GPUBuffer? _floaterCfgBuf;
    int _floaterViews;

    /// <summary>The census after <see cref="ClassifyFloatersAsync"/>: splats by front fraction, and the floaters.</summary>
    public sealed record FloaterReport(int Views, uint[] ByFront, double[] WeightByFront, uint Unseen, uint Floaters,
        double FloaterWeight)
    {
        public uint Seen => (uint)ByFront.Sum(x => (long)x);
        public override string ToString()
        {
            double w = WeightByFront.Sum();
            string bins = string.Join(" ", Enumerable.Range(0, 10).Select(b =>
                $"{b * 10}-{b * 10 + 10}%:{ByFront[b]:N0}/{(w > 0 ? WeightByFront[b] / w : 0):P1}"));
            return $"over {Views} views: {Seen:N0} seen splats ({Unseen:N0} unseen) by front share [count/weight share] {bins}; " +
                $"floaters {Floaters:N0} ({(Seen > 0 ? Floaters / (double)Seen : 0):P1} of seen) carrying " +
                $"{(w > 0 ? FloaterWeight / w : 0):P2} of all blending weight";
        }
    }

    /// <summary>Start a census over <paramref name="splatCount"/> splats.</summary>
    public void ResetFloaterCensus(int splatCount)
    {
        var accel = _gpu.WebGPUAccelerator;
        _floaterCensus ??= MakePipeline(SplatTrainerShaders.FloaterCensus, "floater_census");
        _floaterFold ??= MakePipeline(SplatTrainerShaders.FloaterFold, "floater_fold");
        _floaterClassify ??= MakePipeline(SplatTrainerShaders.FloaterClassify, "floater_classify");
        _floaterCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        long words = (long)splatCount * 2;
        if (_floaterFixed == null || _floaterFixed.Length < words)
        {
            _floaterFixed?.Dispose(); _floaterTotals?.Dispose();
            _floaterFixed = accel.Allocate1D<uint>(words);
            _floaterTotals = accel.Allocate1D<float>(words);
        }
        _floaterHist ??= accel.Allocate1D<uint>(32);
        _floaterFixed.MemSetToZero();
        _floaterTotals!.MemSetToZero();
        accel.FlushPendingCommands();   // the clears ahead of the raw dispatches below
        _floaterViews = 0;
    }

    /// <summary>
    /// Add one training view to the census: the forward pass builds that view's sorted keys and tile ranges, the census
    /// walks them, the fold moves the fixed-point sums into the run's totals. <paramref name="frontMargin"/> is how far
    /// in front of the photo's surface (a fraction of its depth) counts as in front.
    /// </summary>
    public async Task AccumulateFloaterCensusAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar, float frontMargin)
    {
        await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
        if (LastKeyCount == 0) return;
        WriteVec4(_floaterCfgBuf!, frontMargin, FloaterFixedScale, 0f, 0f);
        Dispatch(_floaterCensus!, _tilesX, _tilesY, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatBuf.GetGPUBuffer()!), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _floaterFixed!.GetGPUBuffer()!), Buf(5, _floaterCfgBuf!),
            Buf(11, _splatColour!.GetGPUBuffer()!),
        });
        WriteVec4(_floaterCfgBuf!, splatCount, FloaterFixedScale, 0f, 0f);
        DispatchLinear(_floaterFold!, splatCount, groupSize: 256, entries: new[]
        {
            Buf(0, _floaterFixed!.GetGPUBuffer()!), Buf(1, _floaterTotals!.GetGPUBuffer()!), Buf(2, _floaterCfgBuf!),
        });
        _floaterViews++;
    }

    /// <summary>
    /// Classify every splat from the census (front share at least <paramref name="frontShare"/> of a total weight at
    /// least <paramref name="minWeight"/> pixels = floater) and, with <paramref name="carve"/>, set the floaters'
    /// opacity to 0 in <paramref name="splatBuf"/>. Returns the histogram (CPU transfer: 23 words).
    /// </summary>
    public async Task<FloaterReport> ClassifyFloatersAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        float frontShare, float minWeight, bool carve)
    {
        var accel = _gpu.WebGPUAccelerator;
        _floaterHist!.MemSetToZero();
        accel.FlushPendingCommands();
        // The cfg uniform is rewritten by the next queue.WriteBuffer only after this dispatch is submitted, and WebGPU
        // orders writeBuffer against submissions, so one buffer serves every pass.
        WriteVec4(_floaterCfgBuf!, frontShare, minWeight, carve ? 1f : 0f, splatCount);
        DispatchLinear(_floaterClassify!, splatCount, groupSize: 256, entries: new[]
        {
            Buf(0, splatBuf.GetGPUBuffer()!), Buf(1, _floaterTotals!.GetGPUBuffer()!),
            Buf(2, _floaterHist!.GetGPUBuffer()!), Buf(3, _floaterCfgBuf!),
        });
        // CPU transfer: 23 histogram words.
        uint[] h = await _floaterHist.CopyToHostAsync<uint>(0, 23);
        return new FloaterReport(_floaterViews, h[..10], h[10..20].Select(x => x / 10.0).ToArray(), h[20], h[21], h[22] / 10.0);
    }

    /// <summary>The census totals, 2 per splat (total, front). CPU transfer: the trainer gate only, at gate scale.</summary>
    public Task<float[]> ReadFloaterTotalsAsync(int splatCount) =>
        _floaterTotals!.CopyToHostAsync<float>(0, (long)splatCount * 2);

    void DisposeFloaterCensus()
    {
        _floaterFixed?.Dispose(); _floaterFixed = null;
        _floaterTotals?.Dispose(); _floaterTotals = null;
        _floaterHist?.Dispose(); _floaterHist = null;
        _floaterCfgBuf?.Destroy(); _floaterCfgBuf = null;
    }
}
