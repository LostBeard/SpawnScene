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
    GPUBuffer? _floaterDepthCfgBuf;                          // DepthRef: the view's depth map (or none)
    MemoryBuffer1D<uint, Stride1D.Dense>? _floaterAgree;     // 13 bins: rendered surface / map depth, per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _floaterNoMap;    // bound when a view has no map (the shader reads nothing)
    int _floaterViews, _floaterDepthViews;

    /// <summary>
    /// A view's depth map as the census's surface (FloaterCensus with a map): <paramref name="Width"/> x
    /// <paramref name="Height"/> floats at <paramref name="Offset"/> in <paramref name="Maps"/>, camera depth times
    /// <paramref name="Scale"/> = scene units (DepthFusionInit.DepthTargets); the surface at a pixel is the minimum over
    /// (2 <paramref name="MinRadius"/> + 1)^2 map pixels around it.
    /// </summary>
    public readonly record struct CensusDepth(MemoryBuffer1D<float, Stride1D.Dense> Maps, long Offset, int Width, int Height,
        float Scale, int MinRadius);

    /// <summary>Pixels by rendered surface / photo depth (FloaterCensus's agreement bins) over the census's mapped views.</summary>
    public sealed record DepthAgreement(int Views, uint[] Bins)
    {
        static readonly string[] Names = { "<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.05", "1.05-1.1",
            "1.1-1.25", "1.25-1.5", "1.5-2", ">2", "no render surface", "no map depth" };
        public override string ToString()
        {
            double all = Bins.Sum(x => (double)x);
            return $"{Views} views, {all:N0} px; rendered / photo depth: " +
                string.Join(" ", Bins.Select((b, i) => $"{Names[i]}:{(all > 0 ? b / all : 0):P1}"));
        }
    }

    /// <summary>The census after <see cref="ClassifyFloatersAsync"/>: splats by front fraction, and the floaters.</summary>
    public sealed record FloaterReport(int Views, uint[] ByFront, double[] WeightByFront, uint Unseen, uint Floaters,
        double FloaterWeight, double FrontWeight)
    {
        public uint Seen => (uint)ByFront.Sum(x => (long)x);
        public override string ToString()
        {
            double w = WeightByFront.Sum();
            string bins = string.Join(" ", Enumerable.Range(0, 10).Select(b =>
                $"{b * 10}-{b * 10 + 10}%:{ByFront[b]:N0}/{(w > 0 ? WeightByFront[b] / w : 0):P1}"));
            return $"over {Views} views: {Seen:N0} seen splats ({Unseen:N0} unseen) by front share [count/weight share] {bins}; " +
                $"floaters {Floaters:N0} ({(Seen > 0 ? Floaters / (double)Seen : 0):P1} of seen) carrying " +
                $"{(w > 0 ? FloaterWeight / w : 0):P2} of all blending weight; all splats: {(w > 0 ? FrontWeight / w : 0):P2} of the weight in front";
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
        _floaterAgree ??= accel.Allocate1D<uint>(16);
        _floaterNoMap ??= accel.Allocate1D<float>(4);
        _floaterDepthCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 32,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _floaterAgree.MemSetToZero();
        _floaterFixed.MemSetToZero();
        _floaterTotals!.MemSetToZero();
        accel.FlushPendingCommands();   // the clears ahead of the raw dispatches below
        _floaterViews = 0;
        _floaterDepthViews = 0;
    }

    /// <summary>
    /// Add one training view to the census: the forward pass builds that view's sorted keys and tile ranges, the census
    /// walks them, the fold moves the fixed-point sums into the run's totals. <paramref name="frontMargin"/> is how far
    /// in front of the photo's surface (a fraction of its depth) counts as in front. With <paramref name="depth"/> the
    /// surface is that view's depth map wherever it has one (FloaterCensus), not the render's own.
    /// </summary>
    public async Task AccumulateFloaterCensusAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar, float frontMargin, CensusDepth? depth = null)
    {
        await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
        if (LastKeyCount == 0) return;
        WriteVec4(_floaterCfgBuf!, frontMargin, FloaterFixedScale, 0f, 0f);
        bool mapped = depth is { } dm && dm.Width > 0 && dm.Height > 0 && dm.Scale > 0f && float.IsFinite(dm.Scale);
        var f = mapped ? new[] { depth!.Value.Scale, depth.Value.Width, depth.Value.Height, Math.Max(0, depth.Value.MinRadius) }
            : new[] { 0f, 1f, 1f, 0f };
        var nu = new uint[] { (uint)_width, (uint)_height, mapped ? (uint)depth!.Value.Offset : 0u, 0u };
        var bytes = new byte[32];
        Buffer.BlockCopy(f, 0, bytes, 0, 16);
        Buffer.BlockCopy(nu, 0, bytes, 16, 16);
        _queue!.WriteBuffer(_floaterDepthCfgBuf!, 0, bytes);
        if (mapped) _floaterDepthViews++;
        Dispatch(_floaterCensus!, _tilesX, _tilesY, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatBuf.GetGPUBuffer()!), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _floaterFixed!.GetGPUBuffer()!), Buf(5, _floaterCfgBuf!),
            Buf(6, (mapped ? depth!.Value.Maps : _floaterNoMap!).GetGPUBuffer()!), Buf(7, _floaterDepthCfgBuf!),
            Buf(8, _floaterAgree!.GetGPUBuffer()!), Buf(11, _splatColour!.GetGPUBuffer()!),
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
    /// opacity to 0 in <paramref name="splatBuf"/> (with <paramref name="carveUnseen"/> also every splat under
    /// <paramref name="minWeight"/>). Returns the histogram (CPU transfer: 23 words).
    /// </summary>
    public async Task<FloaterReport> ClassifyFloatersAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        float frontShare, float minWeight, bool carve, bool carveUnseen = false)
    {
        var accel = _gpu.WebGPUAccelerator;
        _floaterHist!.MemSetToZero();
        accel.FlushPendingCommands();
        // The cfg uniform is rewritten by the next queue.WriteBuffer only after this dispatch is submitted, and WebGPU
        // orders writeBuffer against submissions, so one buffer serves every pass.
        WriteVec4(_floaterCfgBuf!, frontShare, minWeight, (carve ? 1f : 0f) + (carveUnseen ? 2f : 0f), splatCount);
        DispatchLinear(_floaterClassify!, splatCount, groupSize: 256, entries: new[]
        {
            Buf(0, splatBuf.GetGPUBuffer()!), Buf(1, _floaterTotals!.GetGPUBuffer()!),
            Buf(2, _floaterHist!.GetGPUBuffer()!), Buf(3, _floaterCfgBuf!),
        });
        // CPU transfer: 23 histogram words.
        uint[] h = await _floaterHist.CopyToHostAsync<uint>(0, 24);
        return new FloaterReport(_floaterViews, h[..10], h[10..20].Select(x => x / 10.0).ToArray(), h[20], h[21], h[22] / 10.0,
            h[23] / 10.0);
    }

    /// <summary>The census's depth agreement since <see cref="ResetFloaterCensus"/> (CPU transfer: 13 words).</summary>
    public async Task<DepthAgreement> ReadDepthAgreementAsync()
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        var b = await _floaterAgree!.CopyToHostAsync<uint>(0, 13);
        return new DepthAgreement(_floaterDepthViews, b);
    }

    /// <summary>The census totals, 2 per splat (total, front). CPU transfer: the trainer gate only, at gate scale.</summary>
    public Task<float[]> ReadFloaterTotalsAsync(int splatCount) =>
        _floaterTotals!.CopyToHostAsync<float>(0, (long)splatCount * 2);

    GPUComputePipeline? _pickPixel;
    MemoryBuffer1D<float, Stride1D.Dense>? _pickOut;

    /// <summary>One splat painting a picked pixel: its index, blending weight there, alpha, camera depth.</summary>
    public readonly record struct PickedSplat(int Index, float Weight, float Alpha, float Depth);

    /// <summary>
    /// Diagnostic: the splats that paint pixel (<paramref name="x"/>, <paramref name="y"/>) of <paramref name="cam"/>, front to
    /// back, with blending weight at least <paramref name="minWeight"/> (at most 16). CPU transfer: 65 floats.
    /// </summary>
    public async Task<PickedSplat[]> PickPixelAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar, int x, int y, float minWeight = 0.02f)
    {
        _pickPixel ??= MakePipeline(SplatTrainerShaders.PickPixel, "pick_pixel");
        _pickOut ??= _gpu.WebGPUAccelerator.Allocate1D<float>(1 + 16 * 4);
        _floaterCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
        WriteVec4(_floaterCfgBuf, x, y, minWeight, 0f);
        Dispatch(_pickPixel, 1, 1, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatBuf.GetGPUBuffer()!), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _pickOut.GetGPUBuffer()!), Buf(5, _floaterCfgBuf),
            Buf(11, _splatColour!.GetGPUBuffer()!),
        });
        var f = await _pickOut.CopyToHostAsync<float>(0, 1 + 16 * 4);
        int count = (int)f[0];
        var r = new PickedSplat[Math.Min(count, 16)];
        for (int k = 0; k < r.Length; k++)
            r[k] = new PickedSplat(BitConverter.SingleToInt32Bits(f[1 + k * 4]), f[2 + k * 4], f[3 + k * 4], f[4 + k * 4]);
        return r;
    }

    /// <summary>Census totals (total, front) of a few splats. CPU transfer: 2 floats each.</summary>
    public async Task<(float Total, float Front)> ReadFloaterTotalsOfAsync(int index)
    {
        if (_floaterTotals == null) return (float.NaN, float.NaN);
        var t = await _floaterTotals.CopyToHostAsync<float>((long)index * 2, 2);
        return (t[0], t[1]);
    }

    void DisposeFloaterCensus()
    {
        _floaterFixed?.Dispose(); _floaterFixed = null;
        _floaterTotals?.Dispose(); _floaterTotals = null;
        _floaterHist?.Dispose(); _floaterHist = null;
        _floaterCfgBuf?.Destroy(); _floaterCfgBuf = null;
        _floaterDepthCfgBuf?.Destroy(); _floaterDepthCfgBuf = null;
        _floaterAgree?.Dispose(); _floaterAgree = null;
        _floaterNoMap?.Dispose(); _floaterNoMap = null;
        _pickOut?.Dispose(); _pickOut = null;
    }
}
