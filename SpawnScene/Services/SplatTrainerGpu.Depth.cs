using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Depth supervision (PLANS 3b; the reference 3DGS's <c>-d</c>, Kerbl et al. 2024 update): an L1 loss between the rendered
/// inverse depth sum(w_i / z_i) and the photo's monocular depth (DAv3, scaled to the scene by DepthFusionInit), weight
/// decaying over training (the caller sets <see cref="DepthLossWeight"/> every step; the reference goes 1.0 -> 0.01).
/// Rooms from phone photos are thin coverage: a wall two photos saw at a glance has colour to fit but no geometry to hold
/// it, and the depth says where it is.
/// <para>
/// Inverse depth blends exactly like a colour channel, so the raster passes carry it as a fourth one in their
/// depth-supervised builds (SplatTrainerShaders.DepthVariant, the <c>//DEPTH:</c> lines): the forward writes it, the
/// backward adds its share to dL/d(alpha) and writes each key's dL/d(1/z), the scatter sums those per splat and the
/// geometry step takes them into the position through d(1/z)/dz = -1/z^2. With the weight at 0 or no target set, the
/// default pipelines run untouched.
/// </para>
/// <para>
/// One device buffer <c>depth_io</c> carries all of it, so each pass needs one more binding: [0, P) the rendered inverse
/// depth, [P, 2P) dL/d(it) from the loss, [2P, 2P + key capacity) the per-key gradient (P = pixels).
/// </para>
/// </summary>
public sealed partial class SplatTrainerGpu
{
    GPUComputePipeline? _rasterForwardDepth, _rasterBackwardDepth, _scatterGradDepth, _adamGeometryDepth, _invDepthL1;
    MemoryBuffer1D<float, Stride1D.Dense>? _depthIo;     // 2 per pixel + 1 per key
    MemoryBuffer1D<float, Stride1D.Dense>? _gradInvz;    // 1 per splat: dL/d(1/z), f32 bits
    GPUBuffer? _depthCfgBuf;
    MemoryBuffer1D<float, Stride1D.Dense>? _depthMaps;
    long _depthMapOffset;
    int _depthMapW, _depthMapH;
    float _depthScale;
    bool _depthUnsupported;

    /// <summary>The depth loss's weight for the next steps (0 = off). The caller decays it over training.</summary>
    public float DepthLossWeight { get; set; }

    /// <summary>
    /// The current view's depth target: a <paramref name="width"/> x <paramref name="height"/> map at
    /// <paramref name="offset"/> floats into <paramref name="maps"/>, camera depth in the view's own unit, times
    /// <paramref name="scale"/> = scene units. The map covers the whole photo (its camera is the view's, scaled). Null
    /// clears it: the step after runs without the depth loss.
    /// </summary>
    public void SetDepthTarget(MemoryBuffer1D<float, Stride1D.Dense>? maps, long offset = 0, int width = 0, int height = 0, float scale = 0f)
    {
        bool ok = maps != null && width > 0 && height > 0 && scale > 0f && float.IsFinite(scale);
        _depthMaps = ok ? maps : null;
        _depthMapOffset = offset; _depthMapW = width; _depthMapH = height; _depthScale = scale;
    }

    /// <summary>True when the next step runs the depth loss (weight on, a target set, buffers ready).</summary>
    bool PrepareDepthStep(int splatCount)
    {
        if (DepthLossWeight <= 0f || _depthMaps == null || _depthUnsupported) return false;
        // The depth-supervised backward binds 12 storage buffers (the plain one 11): a device that allows fewer has no
        // depth loss rather than a failed pipeline.
        int maxStorage = _gpu.WebGPUAccelerator.NativeAccelerator.MaxStorageBuffersPerShaderStage;
        if (maxStorage < 12)
        {
            _depthUnsupported = true;
            Console.WriteLine($"[Trainer] depth loss off: the device allows {maxStorage} storage buffers a shader, it needs 12");
            return false;
        }
        _rasterForwardDepth ??= MakePipeline(SplatTrainerShaders.DepthVariant(SplatTrainerShaders.RasterForward), "raster_forward");
        _rasterBackwardDepth ??= MakePipeline(SplatTrainerShaders.DepthVariant(SplatTrainerShaders.RasterBackward), "raster_backward");
        _scatterGradDepth ??= MakePipeline(SplatTrainerShaders.DepthVariant(SplatTrainerShaders.ScatterGradients), "scatter_gradients");
        _adamGeometryDepth ??= MakePipeline(SplatTrainerShaders.DepthVariant(SplatTrainerShaders.GeometryAdam), "adam_geometry");
        _invDepthL1 ??= MakePipeline(SplatTrainerShaders.InvDepthL1, "inv_depth_l1");
        _depthCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 32, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        EnsureDepthIo();
        if (_gradInvz == null || _gradInvz.Length < splatCount)
        {
            _gradInvz?.Dispose();
            _gradInvz = _gpu.WebGPUAccelerator.Allocate1D<float>(Math.Max(1, splatCount));
        }
        return true;
    }

    /// <summary>depth_io sized for this viewport and the current key capacity (called again when the keys grow).</summary>
    void EnsureDepthIo()
    {
        if (_rasterForwardDepth == null) return;   // depth never used: nothing to keep in step
        long need = 2L * _width * _height + _keyCapacity;
        if (_depthIo != null && _depthIo.Length >= need) return;
        _depthIo?.Dispose();
        _depthIo = _gpu.WebGPUAccelerator.Allocate1D<float>(need);
    }

    /// <summary>The depth loss for this view into depth_io[P, 2P) and the step's loss sum.</summary>
    void DispatchInvDepthLoss(int pixels, int groupsX, int groupsY, int groups)
    {
        var f = new[] { DepthLossWeight, _depthScale, _depthMapW, _depthMapH };
        var n = new uint[] { (uint)_width, (uint)_height, (uint)_depthMapOffset, 0u };
        var bytes = new byte[32];
        Buffer.BlockCopy(f, 0, bytes, 0, 16);
        Buffer.BlockCopy(n, 0, bytes, 16, 16);
        _queue!.WriteBuffer(_depthCfgBuf!, 0, bytes);
        Dispatch(_invDepthL1!, groupsX, groupsY, new[]
        {
            Buf(0, _depthIo!.GetGPUBuffer()!), Buf(1, _depthMaps!.GetGPUBuffer()!),
            Buf(2, _lossPartial!.GetGPUBuffer()!), Buf(3, _lossDimsBuf!), Buf(4, _depthCfgBuf!),
        });
        // loss_reduce adds weights.x * sum / (3 pixels) - written for the 3-channel L1 - so lambda x 3 here makes it
        // lambda x mean |diff|, the depth loss's own value, in the step's reported loss.
        WriteVec4(_lossWeightsBuf!, 3f * DepthLossWeight, 0f, 0f, 0f);
        Dispatch(_lossReduce!, 1, 1, new[]
        {
            Buf(0, _lossPartial!.GetGPUBuffer()!), Buf(1, _lossSum!.GetGPUBuffer()!),
            Buf(2, _lossDimsBuf!), Buf(3, _lossWeightsBuf!),
        });
    }

    /// <summary>
    /// Plain forwards with <c>depth: true</c> also write the rendered inverse depth, without the depth loss: the depth
    /// variant's forward pipeline and depth_io (UnseenFill renders colour, transmittance and depth of views no photo
    /// took). The first width x height floats of <see cref="RenderedInverseDepth"/> are sum(T a / z) per pixel.
    /// </summary>
    public void EnableDepthRender()
    {
        _rasterForwardDepth ??= MakePipeline(SplatTrainerShaders.DepthVariant(SplatTrainerShaders.RasterForward), "raster_forward");
        EnsureDepthIo();
    }

    /// <summary>depth_io; its first width x height floats are the last depth forward's sum(T a / z). GPU-resident.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? RenderedInverseDepth => _depthIo;

    /// <summary>The rendered inverse depth of the last depth-supervised forward. CPU transfer: gate only.</summary>
    public async Task<float[]> ReadRenderedInverseDepthAsync()
    {
        if (_depthIo == null) return System.Array.Empty<float>();
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        return await _depthIo.CopyToHostAsync<float>(0, (long)_width * _height);
    }

    void DisposeDepth()
    {
        _depthIo?.Dispose(); _depthIo = null;
        _gradInvz?.Dispose(); _gradInvz = null;
        _depthCfgBuf?.Destroy(); _depthCfgBuf?.Dispose(); _depthCfgBuf = null;
        _rasterForwardDepth = _rasterBackwardDepth = _scatterGradDepth = _adamGeometryDepth = _invDepthL1 = null;
        _depthMaps = null;
    }
}
