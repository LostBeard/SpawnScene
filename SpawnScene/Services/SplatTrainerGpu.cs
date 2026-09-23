using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Host side of the differentiable tile rasteriser: buffers, pipelines and dispatch for the
/// kernels in <see cref="SplatTrainerShaders"/>.
///
/// The algorithm these kernels implement is modelled and verified on CPU
/// (<see cref="SplatTileRasterizer"/>), which is in turn checked against the
/// finite-difference-verified <see cref="SplatRasterizer"/>. So the job here is plumbing, and
/// the first gate is simply: does the GPU forward reproduce the CPU model.
///
/// Buffers the sort touches are allocated through ILGPU so the existing radix sort works on
/// them directly, and bound to WGSL via <c>GetGPUBuffer()</c> - the same bridge the display
/// renderer already uses. Readback goes through ILGPU's <c>CopyToHostAsync</c> rather than
/// hand-rolled staging buffers.
/// </summary>
public sealed class SplatTrainerGpu : IDisposable
{
    readonly GpuService _gpu;

    GPUDevice? _device;
    GPUQueue? _queue;

    GPUComputePipeline? _emitKeys;
    GPUComputePipeline? _tileRanges;
    GPUComputePipeline? _rasterForward;
    GPUComputePipeline? _rasterBackward;
    GPUComputePipeline? _scatterGrad;
    GPUComputePipeline? _lossL1;
    GPUComputePipeline? _adamStep;
    GPUComputePipeline? _initLogits;
    GPUComputePipeline? _adamGeometry;
    GPUComputePipeline? _evalSse;
    GPUComputePipeline? _ssimRowsPipe;
    GPUComputePipeline? _ssimReducePipe;
    GPUComputePipeline? _ssimWinGradPipe;
    GPUComputePipeline? _ssimRowsBwdPipe;
    GPUComputePipeline? _ssimPixBwdPipe;
    GPUComputePipeline? _gradStats;
    GPUComputePipeline? _maxMagnitude;
    GPUComputePipeline? _sampleStride;
    GPUComputePipeline? _densifyAccum;
    GPUComputePipeline? _remapFloatRows;
    GPUComputePipeline? _accumulateSupport;
    GPUComputePipeline? _supportHistogram;
    GPUComputePipeline? _unpackTarget;
    GPUComputePipeline? _initRgbToDc;
    GPUComputePipeline? _scatterShGrad;
    GPUComputePipeline? _adamShRest;

    GPUBuffer? _uniformBuf;     // TrainUniforms
    GPUBuffer? _capsBuf;        // vec4<u32>: key capacity
    GPUBuffer? _countBuf;       // vec4<u32>: key count, for the ranges pass
    byte[]? _uniformBytes;
    Uint32Array? _scratch4;

    // ILGPU-owned so the radix sort and CopyToHostAsync both work on them.
    MemoryBuffer1D<uint, Stride1D.Dense>? _keys;
    MemoryBuffer1D<uint, Stride1D.Dense>? _values;
    MemoryBuffer1D<int, Stride1D.Dense>? _counter;
    MemoryBuffer1D<uint, Stride1D.Dense>? _ranges;      // 2 per tile
    MemoryBuffer1D<float, Stride1D.Dense>? _outColour;  // 3 per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _outFinalT;  // 1 per pixel
    MemoryBuffer1D<uint, Stride1D.Dense>? _outEnd;      // 1 per pixel
    MemoryBuffer1D<int, Stride1D.Dense>? _sortTemp;   // radix temp is int-typed
    MemoryBuffer1D<float, Stride1D.Dense>? _target;      // 3 per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _dLdPix;      // 3 per pixel
    // Three bindings of three floats per key, not one of nine: maxStorageBufferBindingSize
    // applies per BINDING, so splitting triples how many keys fit. See Resize.
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyA;
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyB;
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyC;
    MemoryBuffer1D<int, Stride1D.Dense>? _gradFixed;     // 9 per splat, f32 bit patterns (CAS-summed)
    MemoryBuffer1D<int, Stride1D.Dense>? _densifyAbs;    // 2 per splat: peak |dPx|,|dPy|

    MemoryBuffer1D<float, Stride1D.Dense>? _opacityLogit;
    MemoryBuffer1D<float, Stride1D.Dense>? _logScale;    // 3 per splat
    MemoryBuffer1D<float, Stride1D.Dense>? _geomOut;     // 10 per splat, diagnostics + gate
    MemoryBuffer1D<float, Stride1D.Dense>? _ssePartials; // 1 per 256-pixel workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimRows;    // 5 filtered channels per (window col, row)
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimPartials;// 1 per 256-window workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimWinGrad; // 5 per valid window (dL/dM)
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimDRows;   // 5 per (window col, row) - row adjoint
    MemoryBuffer1D<float, Stride1D.Dense>? _gradStatsPartials; // 6 per workgroup, 256 workgroups
    MemoryBuffer1D<float, Stride1D.Dense>? _fitSample; // stride gather for the scale histogram
    const int FitSampleCount = 4096;
    MemoryBuffer1D<float, Stride1D.Dense>? _densifyStats;   // 2 per splat: pixel grad sum, visible count
    MemoryBuffer1D<uint, Stride1D.Dense>? _viewSupport;        // views that ever moved each splat
    MemoryBuffer1D<float, Stride1D.Dense>? _supportPartials;   // 6 per workgroup, 256 workgroups
    MemoryBuffer1D<float, Stride1D.Dense>? _adamM;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamV;
    MemoryBuffer1D<float, Stride1D.Dense>? _shRest;
    MemoryBuffer1D<int, Stride1D.Dense>? _gradShRest;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamShM;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamShV;
    MemoryBuffer1D<int, Stride1D.Dense>? _lossFixed;
    GPUBuffer? _dimsBuf;
    GPUBuffer? _ssimDimsBuf;
    GPUBuffer? _ssimCfgBuf;
    GPUBuffer? _lossWeightsBuf;
    GPUBuffer? _adamFlagsBuf;
    GPUBuffer? _adamCfgBuf;
    GPUBuffer? _geomCfgBuf;
    GPUBuffer? _targetBytes;   // one frame of packed RGBA, straight from the canvas
    int _adamStepCount;

    /// <summary>Gradient slots per splat. Must match GRADS_PER_SPLAT in the shaders.</summary>
    public const int GradsPerSplat = SplatTileRasterizer.GradsPerKey;

    /// <summary>Adam moment slots per splat: 3 colour, 1 opacity, 3 position, 3 scale, 4 quaternion.</summary>
    const int AdamSlots = 14;
    /// <summary>Adam layout: RGB(0..2), opacity(3), pos(4..6), log-scale(7..9), quat(10..13).</summary>
    const int AdamOpacitySlot = 3;

    /// <summary>Geometry gradients reported per splat: position xyz, scale xyz, quaternion xyzw.</summary>
    public const int GeomGradsPerSplat = 10;

    // The gradient accumulator (_gradFixed, 9 per splat) holds IEEE f32 bit patterns summed by
    // a compare-exchange loop in scatter_gradients. There is NO fixed-point scale any more.
    //
    // History, so nobody reintroduces one: the i32 fixed-point accumulator needed one scale per
    // slot, and no scale worked for the conic. Too coarse and it rounded to zero (no scale or
    // rotation learning - init-sized soft blobs). Too fine and ~200 keys per splat wrapped the
    // i32 (wrong-sign steps). The per-key +-1e5 clamp that stopped the wrap saturated typical
    // conic keys, which turns each conic component into a sign count and hands adam_geometry a
    // distorted a:b:c ratio to chain into scale and rotation. Twelve Truck 2K gates on
    // 2026-09-22 were spent fitting that scale; supervised PSNR never passed ~17 dB.

    /// <summary>
    /// Linear-scale p10/median/p90 from log-scale params (GPU stride sample). Geometry learning
    /// proof: median must shrink vs init (~0.017 on Truck points) once conic grads live.
    /// </summary>
    public async Task<(float P10, float Median, float P90)> ReadScaleHistogramAsync(int splatCount)
    {
        if (_logScale == null || splatCount <= 0 || _fitSample == null || _sampleStride == null)
            return (0, 0, 0);
        long count = (long)splatCount * 3;
        int n = (int)Math.Min(FitSampleCount, count);
        int stride = Math.Max(1, (int)(count / n));
        n = (int)Math.Min(FitSampleCount, (count + stride - 1) / stride);
        WriteU32x4(_dimsBuf!, (uint)n, (uint)stride, (uint)Math.Min(count, _logScale.Length), 0);
        Dispatch(_sampleStride, (n + 255) / 256, 1, new[]
        {
            Buf(0, _logScale.GetGPUBuffer()!), Buf(1, _fitSample.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // CPU transfer: scale histogram for train logs (gate: median vs init).
        float[] host = await _fitSample.CopyToHostAsync<float>(0, n);
        var lin = new float[n];
        for (int i = 0; i < n; i++) lin[i] = MathF.Exp(host[i]);
        System.Array.Sort(lin);
        float At(double p) => lin[(int)Math.Clamp(Math.Round((n - 1) * p), 0, n - 1)];
        return (At(0.10), At(0.50), At(0.90));
    }

    RadixSortPairs<uint, Stride1D.Dense, uint, Stride1D.Dense>? _sortPairs;

    int _width, _height, _tilesX, _tilesY, _keyCapacity;

    /// <summary>
    /// Keys per splat actually budgeted, which may be lower than requested: the ceiling is the
    /// storage BINDING size limit, not free memory. Reported so a log line cannot claim a
    /// budget that was not used.
    /// </summary>
    public int KeysPerSplat { get; private set; }

    /// <summary>Keys emitted by the last render. Exceeding capacity is reported, never silent.</summary>
    public int LastKeyCount { get; private set; }

    /// <summary>True when the last render overflowed the key buffer and is therefore incomplete.</summary>
    public bool LastOverflowed { get; private set; }

    /// <summary>
    /// Keys the last render actually WANTED, which exceeds <see cref="LastKeyCount"/> when it
    /// overflowed. Lets a caller size the budget from a measurement instead of a guess: how
    /// many tiles a splat covers depends on the scene, and a room is not a turntable.
    /// </summary>
    public int LastKeyDemand { get; private set; }

    /// <summary>Active SH degree 0..3, sent in <c>TrainUniforms.sh_degree</c>.</summary>
    public int ActiveShDegree { get; set; }

    bool _rgbToDcDone;

    public SplatTrainerGpu(GpuService gpu) => _gpu = gpu;

    const int UniformFloats = 28;   // 112 bytes: 4 vec4 + 3 vec2 + 2 u32 + 2 f32 + u32 + pad

    public void Initialize()
    {
        var accel = _gpu.WebGPUAccelerator;
        var native = accel.NativeAccelerator;
        _device = native.NativeDevice ?? throw new InvalidOperationException("no WebGPU device");
        _queue = native.Queue ?? throw new InvalidOperationException("no WebGPU queue");

        _emitKeys = MakePipeline(SplatTrainerShaders.EmitKeys, "emit_keys");
        _tileRanges = MakePipeline(SplatTrainerShaders.TileRanges, "tile_ranges");
        _rasterForward = MakePipeline(SplatTrainerShaders.RasterForward, "raster_forward");
        _rasterBackward = MakePipeline(SplatTrainerShaders.RasterBackward, "raster_backward");
        _scatterGrad = MakePipeline(SplatTrainerShaders.ScatterGradients, "scatter_gradients");
        _lossL1 = MakePipeline(SplatTrainerShaders.LossL1, "loss_l1");
        _adamStep = MakePipeline(SplatTrainerShaders.AdamStep, "adam_step");
        _initLogits = MakePipeline(SplatTrainerShaders.InitLogits, "init_logits");
        _adamGeometry = MakePipeline(SplatTrainerShaders.GeometryAdam, "adam_geometry");
        _evalSse = MakePipeline(SplatTrainerShaders.EvalSse, "eval_sse");
        _ssimRowsPipe = MakePipeline(SplatTrainerShaders.SsimRows, "ssim_rows");
        _ssimReducePipe = MakePipeline(SplatTrainerShaders.SsimReduce, "ssim_reduce");
        _ssimWinGradPipe = MakePipeline(SplatTrainerShaders.SsimWinGrad, "ssim_win_grad");
        _ssimRowsBwdPipe = MakePipeline(SplatTrainerShaders.SsimRowsBwd, "ssim_rows_bwd");
        _ssimPixBwdPipe = MakePipeline(SplatTrainerShaders.SsimPixBwd, "ssim_pix_bwd");
        _gradStats = MakePipeline(SplatTrainerShaders.GradStats, "grad_stats");
        _maxMagnitude = MakePipeline(SplatTrainerShaders.MaxMagnitude, "max_magnitude");
        _sampleStride = MakePipeline(SplatTrainerShaders.SampleStride, "sample_stride");
        _densifyAccum = MakePipeline(SplatTrainerShaders.DensifyAccum, "densify_accum");
        _remapFloatRows = MakePipeline(SplatTrainerShaders.RemapFloatRows, "remap_float_rows");
        _accumulateSupport = MakePipeline(SplatTrainerShaders.AccumulateSupport, "accumulate_support");
        _supportHistogram = MakePipeline(SplatTrainerShaders.SupportHistogram, "support_histogram");
        _unpackTarget = MakePipeline(SplatTrainerShaders.UnpackTarget, "unpack_target");
        _initRgbToDc = MakePipeline(SplatTrainerShaders.InitRgbToDc, "init_rgb_to_dc");
        _scatterShGrad = MakePipeline(SplatTrainerShaders.ScatterShGrad, "scatter_sh_grad");
        _adamShRest = MakePipeline(SplatTrainerShaders.AdamShRest, "adam_sh_rest");

        _uniformBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = UniformFloats * sizeof(float),
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _capsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _countBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _uniformBytes = new byte[UniformFloats * sizeof(float)];
        _scratch4 = new Uint32Array(4);
        _dimsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _adamCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        // Separate from _adamCfgBuf rather than widening it: growing a uniform without updating
        // its allocation gives the driver's unhelpful "[Invalid CommandBuffer] ... previous
        // error", reported from whichever dispatch runs next rather than from the guilty one.
        _adamFlagsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _geomCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 32,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _ssimDimsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        // SsimCfg: vec4 luma + vec4 consts + array<vec4,3> weights = 80 bytes.
        _ssimCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 80,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _lossWeightsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        Console.WriteLine("[Trainer] pipelines created: emit_keys, tile_ranges, raster_forward, " +
            "raster_backward, scatter_gradients, loss_l1, adam_step, init_logits, adam_geometry, " +
            "eval_sse, unpack_target, ssim_rows, ssim_reduce, ssim_win_grad, ssim_rows_bwd, " +
            "ssim_pix_bwd, grad_stats");
    }

    GPUComputePipeline MakePipeline(string wgsl, string entry)
    {
        using var module = _device!.CreateShaderModule(new GPUShaderModuleDescriptor { Code = wgsl });
        return _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage { Module = module, EntryPoint = entry },
        });
    }

    /// <summary>
    /// Size buffers for a viewport and splat count.
    /// <paramref name="keysPerSplat"/> is the worst-case tile overlap budget; the emit kernel
    /// refuses to write past capacity and the host reports the overflow rather than rendering
    /// a silently truncated scene.
    /// </summary>
    /// <summary>
    /// The most splats this trainer can hold at a given key budget.
    ///
    /// Densification must not grow past it. Resize clamps keysPerSplat downwards to fit, and a
    /// clamped budget overflows - so growing beyond this does not buy splats, it buys frames
    /// trained on incomplete renders.
    /// </summary>
    public int MaxTrainableSplats(int keysPerSplat) =>
        (int)Math.Max(1, MaxStorageBindingBytes() / (3 * sizeof(float)) / Math.Max(1, keysPerSplat));

    long _maxBindingBytes;

    /// <summary>
    /// The device's real <c>maxStorageBufferBindingSize</c>, not the spec minimum.
    ///
    /// 128 MiB is what WebGPU GUARANTEES; it is not what hardware provides. Sizing to the
    /// guarantee is the right default for a shader that must behave identically everywhere, but
    /// here it is a hard ceiling on the splat count: at the ~50 keys per splat this scene
    /// actually demands, 128 MiB stops the reconstruction near 220,000 splats while the
    /// reference finishes a room in the millions. That is the guarantee deciding the quality.
    ///
    /// So ask. The guarantee stays the floor, so a device that reports something smaller or
    /// nothing at all behaves exactly as before.
    /// </summary>
    long MaxStorageBindingBytes()
    {
        const long Guaranteed = 128L * 1024 * 1024;
        if (_maxBindingBytes > 0) return _maxBindingBytes;
        _maxBindingBytes = Guaranteed;
        try
        {
            using var limits = _device?.JSRef?.Get<SpawnDev.SpawnJS.SpawnJSObject>("limits");
            double? reported = limits?.JSRef?.Get<double?>("maxStorageBufferBindingSize");
            if (reported is > 0)
            {
                long bytes = (long)reported.Value;
                if (bytes > _maxBindingBytes)
                {
                    _maxBindingBytes = bytes;
                    Console.WriteLine(
                        $"[Trainer] device maxStorageBufferBindingSize is " +
                        $"{bytes / (1024 * 1024)} MiB, not the {Guaranteed / (1024 * 1024)} MiB " +
                        $"guarantee - key capacity scales with it");
                }
            }
        }
        catch (Exception ex)
        {
            // A device that will not answer keeps the guarantee. Never fatal: this is an
            // optimisation over a value that is already correct.
            Console.WriteLine($"[Trainer] could not read device limits ({ex.Message}); " +
                              "using the 128 MiB guarantee");
        }
        return _maxBindingBytes;
    }

    public void Resize(int width, int height, int splatCount, int keysPerSplat = 8)
    {
        var accel = _gpu.WebGPUAccelerator;

        _width = width;
        _height = height;
        _tilesX = (width + SplatTrainerShaders.TileSize - 1) / SplatTrainerShaders.TileSize;
        _tilesY = (height + SplatTrainerShaders.TileSize - 1) / SplatTrainerShaders.TileSize;

        // The key packs the tile id above 18 depth bits, so the tile count has a hard ceiling.
        int tileCount = _tilesX * _tilesY;
        if (tileCount >= (1 << 14))
            throw new InvalidOperationException(
                $"{tileCount} tiles exceeds the {1 << 14} the 18-bit depth key leaves room for " +
                $"({width}x{height}). Reduce the training resolution or widen the key.");

        // The key budget is bounded by what a single storage BINDING may be, not by free VRAM.
        // The nine per-key gradients are split across three bindings of three floats, so the
        // widest key-indexed binding is 12 bytes rather than 36. As one buffer, 580k splats at
        // 8 keys each came to 159 MiB against a guaranteed 128 MiB, and the driver rejected the
        // bind group with "[Invalid CommandBuffer] is invalid due to a previous error" - naming
        // neither the buffer nor the limit, and surfacing from whichever dispatch ran next.
        //
        // 128 MiB is the WebGPU guaranteed minimum for maxStorageBufferBindingSize. Sizing to
        // the guarantee rather than querying means this behaves the same on every device.
        long MaxBindingBytes = MaxStorageBindingBytes();
        const long bytesPerKey = 3 * sizeof(float);   // the widest single key-indexed binding
        long maxKeys = MaxBindingBytes / bytesPerKey;

        // Clamp DOWN to what the binding allows, and no further.
        //
        // I changed this to also grow keysPerSplat up to the affordable maximum, on the theory
        // that leaving capacity unused was causing the overflow I saw. That was WRONG twice over.
        // The overflow self-corrects: the caller measures actual peak demand after the first
        // cycle and re-sizes with headroom ("peak demand 5,991,085 keys for 725,452 splats
        // (8.3 per splat); re-sizing to 11 per splat with 25% headroom"), so exactly one frame
        // is incomplete, not the run. And taking the maximum instead allocated 10.9M keys and
        // lost the device outright.
        //
        // Measured demand plus headroom is the right rule. "Use everything available" is not a
        // budget either - it is the same mistake as a fixed default, pointing the other way.
        int requested = keysPerSplat;
        if ((long)splatCount * keysPerSplat > maxKeys)
        {
            keysPerSplat = Math.Max(1, (int)(maxKeys / Math.Max(1, splatCount)));
            Console.WriteLine(
                $"[Trainer] keysPerSplat {requested} -> {keysPerSplat}: {splatCount:N0} splats " +
                $"would need {(long)splatCount * requested * bytesPerKey / (1024 * 1024)} MiB for " +
                $"one binding, over the {MaxBindingBytes / (1024 * 1024)} MiB this device allows. " +
                "A tighter budget can overflow, which is reported per frame, not hidden.");
        }

        KeysPerSplat = keysPerSplat;
        _keyCapacity = Math.Max(1024, splatCount * keysPerSplat);
        if (_keyCapacity > maxKeys)
            throw new InvalidOperationException(
                $"{splatCount:N0} splats cannot be trained: even one key each needs " +
                $"{_keyCapacity * bytesPerKey / (1024 * 1024)} MiB for a single binding. " +
                "Reduce the splat count or the training resolution.");

        DisposeBuffers();
        _keys = accel.Allocate1D<uint>(_keyCapacity);
        _values = accel.Allocate1D<uint>(_keyCapacity);
        _counter = accel.Allocate1D<int>(1);
        _ranges = accel.Allocate1D<uint>(tileCount * 2);
        _outColour = accel.Allocate1D<float>((long)width * height * 3);
        _outFinalT = accel.Allocate1D<float>((long)width * height);
        _outEnd = accel.Allocate1D<uint>((long)width * height);

        _target = accel.Allocate1D<float>((long)width * height * 3);
        _targetBytes?.Destroy(); _targetBytes?.Dispose();
        _targetBytes = _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)width * (ulong)height * 4UL,
            Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
        });
        _dLdPix = accel.Allocate1D<float>((long)width * height * 3);
        _gradKeyA = accel.Allocate1D<float>((long)_keyCapacity * 3);
        _gradKeyB = accel.Allocate1D<float>((long)_keyCapacity * 3);
        _gradKeyC = accel.Allocate1D<float>((long)_keyCapacity * 3);
        _gradFixed = accel.Allocate1D<int>((long)splatCount * GradsPerSplat);
        _densifyAbs = accel.Allocate1D<int>((long)splatCount * 2);
        _opacityLogit = accel.Allocate1D<float>(splatCount);
        _logScale = accel.Allocate1D<float>((long)splatCount * 3);
        _geomOut = accel.Allocate1D<float>((long)splatCount * GeomGradsPerSplat);
        _ssePartials = accel.Allocate1D<float>(SseWorkgroups);
        _gradStatsPartials = accel.Allocate1D<float>(GradStatsWorkgroups * GradStatsSlots);
        _fitSample = accel.Allocate1D<float>(FitSampleCount);
        _densifyStats = accel.Allocate1D<float>((long)splatCount * 2);
        _viewSupport = accel.Allocate1D<uint>(splatCount);
        _supportPartials = accel.Allocate1D<float>(SupportWorkgroups * SupportSlots);

        // SSIM works on 'valid' windows, so a viewport smaller than the window has none and the
        // metric is genuinely undefined there - the Python oracle raises rather than inventing a
        // number, and ScoreAgainstAsync reports NaN for the same reason.
        if (HasSsimWindows)
        {
            _ssimRows = accel.Allocate1D<float>((long)SsimWindowsX * _height * 5);
            _ssimPartials = accel.Allocate1D<float>(SsimWorkgroups);
            _ssimWinGrad = accel.Allocate1D<float>((long)SsimWindowsX * SsimWindowsY * 5);
            _ssimDRows = accel.Allocate1D<float>((long)SsimWindowsX * _height * 5);
            WriteSsimCfg();
        }
        _adamM = accel.Allocate1D<float>((long)splatCount * AdamSlots);
        _adamV = accel.Allocate1D<float>((long)splatCount * AdamSlots);
        _shRest = accel.Allocate1D<float>((long)splatCount * SphericalHarmonics.RestFloatsPerSplat);
        _gradShRest = accel.Allocate1D<int>((long)splatCount * SphericalHarmonics.RestFloatsPerSplat);
        _adamShM = accel.Allocate1D<float>((long)splatCount * SphericalHarmonics.RestFloatsPerSplat);
        _adamShV = accel.Allocate1D<float>((long)splatCount * SphericalHarmonics.RestFloatsPerSplat);
        _shRest.MemSetToZero();
        _adamShM.MemSetToZero();
        _adamShV.MemSetToZero();
        _lossFixed = accel.Allocate1D<int>(1);
        _adamStepCount = 0;

        int temp = accel.ComputeRadixSortPairsTempStorageSize<uint, uint, AscendingUInt32>((Index1D)_keyCapacity);
        _sortTemp = accel.Allocate1D<int>(Math.Max(1, temp));
        _sortPairs = accel.CreateRadixSortPairs<uint, Stride1D.Dense, uint, Stride1D.Dense, AscendingUInt32>();

        Console.WriteLine($"[Trainer] sized {width}x{height} = {_tilesX}x{_tilesY} tiles, " +
            $"{splatCount:N0} splats, key capacity {_keyCapacity:N0}");
    }

    /// <summary>
    /// Run the forward pass: emit keys, sort, find tile ranges, rasterise.
    /// Returns the rendered RGB (3 floats per pixel).
    /// </summary>
    public async Task<float[]> RenderForwardAsync(
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar, bool readback = true)
    {
        if (_device == null || _emitKeys == null) throw new InvalidOperationException("not initialized");

        var accel = _gpu.WebGPUAccelerator;
        var splatGpu = splatBuf.GetGPUBuffer() ?? throw new InvalidOperationException("splat buffer has no GPU handle");

        WriteUniforms(cam, depthNear, depthFar, splatCount);
        WriteU32(_capsBuf!, (uint)_keyCapacity);

        // Clear the counter and the tile ranges. Tiles with no keys are never written by the
        // ranges kernel, so stale values from a previous frame would be read as real spans.
        // Also clear colour / T / end_idx: a view that paints nothing must not leave the
        // previous view's image in the loss (MEASURED Truck STAGE PROBE: mean|rgb| live,
        // max|gradPerKey|==0 on a subset of views - stale colour + consumed=0 fits that split).
        // Clear keys/values too: overflow frames write only a prefix; the rest used to keep
        // the previous view's splat ids, and sort then fed mixed lists into raster.
        _counter!.MemSetToZero();
        _ranges!.MemSetToZero();
        _outColour!.MemSetToZero();
        _outFinalT!.MemSetToZero();
        _outEnd!.MemSetToZero();
        _keys!.MemSetToZero();
        _values!.MemSetToZero();
        await accel.SynchronizeAsync();

        // ── 1. Emit (tile, depth) keys ──
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_emitKeys);
            using var layout = _emitKeys.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuf! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = splatGpu } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _keys!.GetGPUBuffer()! } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _values!.GetGPUBuffer()! } },
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = _counter!.GetGPUBuffer()! } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _capsBuf! } },
                    ShRestBindEntry(12),
                },
            });
            pass.SetBindGroup(0, bg);
            var (ekX, ekY) = LinearGrid(splatCount);
            pass.DispatchWorkgroups((uint)ekX, (uint)ekY, 1);
            pass.End();
            using var cmd = enc.Finish();
            _queue!.Submit(new[] { cmd });
        }
        await accel.SynchronizeAsync();

        // 4 bytes back to learn how many keys exist. A scalar, not bulk data.
        int[] counted = await _counter.CopyToHostAsync<int>(0, 1);
        int keyCount = counted[0];
        LastKeyDemand = keyCount;
        PeakKeyDemand = Math.Max(PeakKeyDemand, keyCount);
        LastOverflowed = keyCount > _keyCapacity;
        LastKeyCount = Math.Min(keyCount, _keyCapacity);
        if (LastOverflowed)
        {
            Console.WriteLine($"[Trainer] KEY OVERFLOW: {keyCount:N0} needed, capacity {_keyCapacity:N0}. " +
                "Raise keysPerSplat; this frame is incomplete.");
        }
        if (LastKeyCount == 0)
        {
            Console.WriteLine("[Trainer] no keys emitted - nothing visible from this view");
            return readback ? new float[_width * _height * 3] : System.Array.Empty<float>();
        }

        // ── 2. Sort by key: groups by tile AND orders front-to-back within each tile ──
        _sortPairs!(accel.DefaultStream, _keys!.View.SubView(0, LastKeyCount),
            _values!.View.SubView(0, LastKeyCount), _sortTemp!.View);
        await accel.SynchronizeAsync();

        // ── 3. Tile ranges ──
        WriteU32(_countBuf!, (uint)LastKeyCount);
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_tileRanges!);
            using var layout = _tileRanges!.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _keys!.GetGPUBuffer()! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = _ranges!.GetGPUBuffer()! } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _countBuf! } },
                },
            });
            pass.SetBindGroup(0, bg);
            var (trX, trY) = LinearGrid(LastKeyCount);
            pass.DispatchWorkgroups((uint)trX, (uint)trY, 1);
            pass.End();
            using var cmd = enc.Finish();
            _queue!.Submit(new[] { cmd });
        }

        // ── 4. Rasterise: one workgroup per tile ──
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_rasterForward!);
            using var layout = _rasterForward!.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuf! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = splatGpu } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _ranges!.GetGPUBuffer()! } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _values!.GetGPUBuffer()! } },
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = _outColour!.GetGPUBuffer()! } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _outFinalT!.GetGPUBuffer()! } },
                    new() { Binding = 6, Resource = new GPUBufferBinding { Buffer = _outEnd!.GetGPUBuffer()! } },
                    ShRestBindEntry(12),
                },
            });
            pass.SetBindGroup(0, bg);
            pass.DispatchWorkgroups((uint)_tilesX, (uint)_tilesY, 1);
            pass.End();
            using var cmd = enc.Finish();
            _queue!.Submit(new[] { cmd });
        }
        await accel.SynchronizeAsync();

        // CPU transfer: gate comparison only. Training passes readback:false and the colour
        // stays on the GPU - at 640x480 this copy is 3.7 MB, which would dwarf the iteration.
        if (!readback) return System.Array.Empty<float>();
        return await _outColour!.CopyToHostAsync<float>(0, _width * _height * 3);
    }

    /// <summary>Upload the target image this view is being fitted to.</summary>
    public void SetTarget(float[] rgb)
    {
        if (rgb.Length != _width * _height * 3)
            throw new ArgumentException($"target is {rgb.Length}, expected {_width * _height * 3}");
        _target!.CopyFromCPU(rgb);
    }

    /// <summary>
    /// Point the loss at one image inside a GPU-resident stack of targets (view-major,
    /// width*height*3 floats each). Device-to-device, so a multi-view run pays the upload once.
    /// </summary>
    public void SetTargetFrom(MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex)
    {
        long pixels = (long)_width * _height;
        long off = (long)viewIndex * pixels;
        if (off + pixels > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} needs pixels [{off},{off + pixels}) of a {stack.Length}-pixel stack");

        // Unpack one frame out of the packed stack into the working float target.
        //
        // The stack holds RGBA8, four bytes a pixel, not three floats. That is a THIRD of the
        // memory, and target memory is what caps how many views can supervise a run: 256 MiB
        // held 62 views as floats and holds 187 packed. Views are the scarce resource here -
        // the reference trains drjohnson on about 230 images and we were using 33.
        WriteU32x4(_dimsBuf!, (uint)pixels, 0, (uint)off, 0);
        Dispatch(_unpackTarget!, (int)((pixels + 63) / 64), 1, new[]
        {
            Buf(0, stack.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
    }

    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _copyKernel;

    static void CopyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst)
    {
        int i = index;
        if (i >= dst.Length) return;
        dst[i] = src[i];
    }

    /// <summary>
    /// Gradient coverage over every splat. Magnitudes are the true float gradients.
    /// </summary>
    public readonly record struct GradientStats(
        long Splats, long ColourLive, long CentreLive, long ConicLive,
        double MeanCentreAbs, double MaxCentreAbs, double MaxConicAbs,
        double MaxColourAbs)
    {
        /// <summary>
        /// Splats that will take an Adam step on a gradient of exactly zero this iteration.
        ///
        /// With batch size 1 and a round robin over the supervised views, a splat visible in one
        /// view of 26 is invisible for the other 25 steps of the cycle. adam_geometry guards
        /// against stepping those; adam_step does not, on the stated judgement that it is
        /// "harmless for colour". This number is what makes that judgement checkable.
        /// </summary>
        public double StaleColourFraction =>
            Splats > 0 ? 1.0 - (double)ColourLive / Splats : 0.0;
    }

    const int GradStatsWorkgroups = 256;
    const int GradStatsSlots = 7;

    /// <summary>
    /// Reduce the gradient accumulator on the GPU and bring back 6 KB.
    ///
    /// Must be called AFTER TrainStepAsync returns and before the next one begins: the
    /// accumulator is cleared mid-step (between raster_backward and scatter_gradients), so it
    /// holds the completed step's totals until the following step reaches that point.
    /// </summary>
    public async Task<GradientStats> ReadGradientStatsAsync(int splatCount)
    {
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        Dispatch(_gradStats!, GradStatsWorkgroups, 1, new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _gradStatsPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();

        // CPU transfer: 6 floats per workgroup. Slots 0-3 are SUMS, slots 4-5 are MAXES - two
        // reduction operators in one array, so they must be combined differently here too.
        float[] p = await _gradStatsPartials!.CopyToHostAsync<float>(
            0, GradStatsWorkgroups * GradStatsSlots);

        double colour = 0, centre = 0, conic = 0, sumCentre = 0;
        double maxCentre = 0, maxConic = 0, maxColour = 0;
        for (int i = 0; i < GradStatsWorkgroups; i++)
        {
            int o = i * GradStatsSlots;
            colour += p[o];
            centre += p[o + 1];
            conic += p[o + 2];
            sumCentre += p[o + 3];
            maxCentre = Math.Max(maxCentre, p[o + 4]);
            maxConic = Math.Max(maxConic, p[o + 5]);
            maxColour = Math.Max(maxColour, p[o + 6]);
        }

        return new GradientStats(
            splatCount, (long)colour, (long)centre, (long)conic,
            centre > 0 ? sumCentre / centre : 0.0, maxCentre, maxConic, maxColour);
    }

    /// <summary>
    /// Steal optimizer row buffers before <see cref="Resize"/> so densify can remap on GPU
    /// without a host round-trip. Caller must Dispose the returned buffers after remap.
    /// </summary>
    public (MemoryBuffer1D<float, Stride1D.Dense>? AdamM,
            MemoryBuffer1D<float, Stride1D.Dense>? AdamV,
            MemoryBuffer1D<float, Stride1D.Dense>? ShRest,
            MemoryBuffer1D<float, Stride1D.Dense>? ShAdamM,
            MemoryBuffer1D<float, Stride1D.Dense>? ShAdamV,
            int StepCount) DetachOptimizerRows()
    {
        var t = (_adamM, _adamV, _shRest, _adamShM, _adamShV, _adamStepCount);
        _adamM = null; _adamV = null; _shRest = null; _adamShM = null; _adamShV = null;
        return t;
    }

    public void SetAdamStepCount(int step) => _adamStepCount = Math.Max(0, step);

    /// <summary>
    /// Densify restore: host RemapFloatRows for Adam (proven; GPU path killed opacity), GPU
    /// RemapFloatRows for SH banks with a readback fence so ILGPU SynchronizeAsync is not
    /// trusted to drain the raw WebGPU <c>_queue</c> Submit used by <see cref="Dispatch"/>.
    /// Host SH CopyToHost OOM'd at ~780k (MEASURED growhost).
    /// </summary>
    public async Task RemapOptimizerRowsHybridAsync(
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamV,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShRest,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamV,
        int priorCount,
        int priorStep,
        int[] adamSurvivors,
        int[] featureSources,
        int zeroAdamSlot = -1)
    {
        if (priorCount <= 0 || adamSurvivors.Length <= 0) return;
        int m = adamSurvivors.Length;
        await _gpu.WebGPUAccelerator.SynchronizeAsync();

        if (priorAdamM != null && priorAdamV != null)
        {
            long adamLen = (long)priorCount * AdamSlots;
            // CPU transfer: Adam m/v only (~87 MB at 780k) — opacity-safe path.
            var state = new AdamState(
                await priorAdamM.CopyToHostAsync<float>(0, adamLen),
                await priorAdamV.CopyToHostAsync<float>(0, adamLen),
                priorStep);
            RestoreAdamState(state, adamSurvivors, zeroAdamSlot);
        }
        else
            SetAdamStepCount(priorStep);

        if (_remapFloatRows == null) return;
        var accel = _gpu.WebGPUAccelerator;
        var adamSrc = accel.Allocate1D<int>(m);
        adamSrc.CopyFromCPU(adamSurvivors);
        var featSrc = accel.Allocate1D<int>(m);
        featSrc.CopyFromCPU(featureSources);
        // CPU transfer: 1 float fence after each SH remap so Dispose cannot race _queue.
        try
        {
            await RemapGpuFencedAsync(priorShRest, _shRest, featSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
            await RemapGpuFencedAsync(priorShAdamM, _adamShM, adamSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
            await RemapGpuFencedAsync(priorShAdamV, _adamShV, adamSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
        }
        finally
        {
            adamSrc.Dispose();
            featSrc.Dispose();
        }
    }

    async Task RemapGpuFencedAsync(
        MemoryBuffer1D<float, Stride1D.Dense>? prior,
        MemoryBuffer1D<float, Stride1D.Dense>? next,
        MemoryBuffer1D<int, Stride1D.Dense> sources,
        int priorCount,
        int newCount,
        int stride,
        int zeroSlot = -1)
    {
        if (prior == null || next == null || _remapFloatRows == null) return;
        next.MemSetToZero();
        uint z = zeroSlot >= 0 && zeroSlot < stride ? (uint)zeroSlot : uint.MaxValue;
        WriteU32x4(_dimsBuf!, (uint)newCount, (uint)stride, (uint)priorCount, z);
        Dispatch(_remapFloatRows!, (newCount + 63) / 64, 1, new[]
        {
            Buf(0, prior.GetGPUBuffer()!), Buf(1, next.GetGPUBuffer()!),
            Buf(2, sources.GetGPUBuffer()!), Buf(3, _dimsBuf!),
        });
        // CPU transfer: 4-byte fence — drains WebGPU queue after Dispatch Submit.
        _ = await next.CopyToHostAsync<float>(0, 1);
    }

    /// <summary>
    /// Full GPU densify remap. Prefer <see cref="RemapOptimizerRowsHybridAsync"/> — pure GPU
    /// Adam remap still fails opacity (MEASURED) even with SynchronizeAsync.
    /// </summary>
    public async Task RemapOptimizerRowsGpuAsync(
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamV,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShRest,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamV,
        int priorCount,
        int[] adamSurvivors,
        int[] featureSources,
        int zeroAdamSlot = -1)
    {
        int m = adamSurvivors.Length;
        if (m <= 0 || _remapFloatRows == null) return;
        var adamSrc = _gpu.WebGPUAccelerator.Allocate1D<int>(m);
        adamSrc.CopyFromCPU(adamSurvivors);
        var featSrc = _gpu.WebGPUAccelerator.Allocate1D<int>(m);
        featSrc.CopyFromCPU(featureSources);
        try
        {
            await RemapGpuFencedAsync(priorAdamM, _adamM, adamSrc, priorCount, m, AdamSlots, zeroAdamSlot);
            await RemapGpuFencedAsync(priorAdamV, _adamV, adamSrc, priorCount, m, AdamSlots, zeroAdamSlot);
            await RemapGpuFencedAsync(priorShRest, _shRest, featSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
            await RemapGpuFencedAsync(priorShAdamM, _adamShM, adamSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
            await RemapGpuFencedAsync(priorShAdamV, _adamShV, adamSrc, priorCount, m,
                SphericalHarmonics.RestFloatsPerSplat);
        }
        finally
        {
            adamSrc.Dispose();
            featSrc.Dispose();
        }
    }

    /// <summary>
    /// Densify Adam/SH restore via host RemapFloatRows (proven for Adam). Prefer hybrid for
    /// large N — full host SH CopyToHost OOM'd at ~780k (MEASURED growhost).
    /// </summary>
    public async Task RemapOptimizerRowsHostAsync(
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorAdamV,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShRest,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamM,
        MemoryBuffer1D<float, Stride1D.Dense>? priorShAdamV,
        int priorCount,
        int priorStep,
        int[] adamSurvivors,
        int[] featureSources,
        int zeroAdamSlot = -1)
    {
        // Same as hybrid Adam path + host SH (small scenes).
        await RemapOptimizerRowsHybridAsync(
            priorAdamM, priorAdamV, null, null, null,
            priorCount, priorStep, adamSurvivors, featureSources, zeroAdamSlot);
        long shLen = (long)priorCount * SphericalHarmonics.RestFloatsPerSplat;
        if (priorShRest != null)
            RestoreShRest(await priorShRest.CopyToHostAsync<float>(0, shLen), featureSources);
        if (priorShAdamM != null && priorShAdamV != null)
        {
            RestoreShAdamState(
                new ShAdamState(
                    await priorShAdamM.CopyToHostAsync<float>(0, shLen),
                    await priorShAdamV.CopyToHostAsync<float>(0, shLen)),
                adamSurvivors);
        }
    }

    /// <summary>Adam moments and the global step count, as one movable blob.</summary>
    public readonly record struct AdamState(float[] M, float[] V, int StepCount);

    /// <summary>
    /// Read the Adam moments so densification can carry them across a resize.
    ///
    /// CPU transfer: 14 moments x 2 per splat. Densification changes the splat count, which
    /// reallocates these buffers regardless, so the choice is between moving the state and
    /// throwing it away - not between moving it and leaving it alone.
    /// </summary>
    public async Task<AdamState> ReadAdamStateAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        long len = (long)splatCount * AdamSlots;
        return new AdamState(
            await _adamM!.CopyToHostAsync<float>(0, len),
            await _adamV!.CopyToHostAsync<float>(0, len),
            _adamStepCount);
    }

    /// <summary>
    /// Restore Adam moments after a resize, keeping the survivors' momentum.
    ///
    /// <paramref name="survivors"/> maps each NEW splat index to the OLD index it came from, or
    /// -1 for a splat that did not exist before. A clone and a split child both inherit their
    /// parent's momentum in the reference implementation only in the sense that they start at
    /// zero and the parent keeps its own; so new splats get zeros here, which is what the
    /// reference does when it concatenates zeroed rows.
    ///
    /// The step count is carried too. Bias correction divides by 1 - beta^t, and restarting t
    /// at zero makes the first step after every densification about ten times larger than it
    /// should be - a visible kick, nine times in a run.
    /// </summary>
    public void RestoreAdamState(AdamState prior, int[] survivors, int zeroSlot = -1)
    {
        int n = survivors.Length;
        var m = new float[(long)n * AdamSlots];
        var v = new float[(long)n * AdamSlots];
        int oldCount = prior.M.Length / AdamSlots;

        for (int i = 0; i < n; i++)
        {
            int src = survivors[i];
            if (src < 0 || src >= oldCount) continue;   // new splat: zeros, as allocated
            System.Array.Copy(prior.M, (long)src * AdamSlots, m, (long)i * AdamSlots, AdamSlots);
            System.Array.Copy(prior.V, (long)src * AdamSlots, v, (long)i * AdamSlots, AdamSlots);
        }

        // One parameter's momentum can be dropped while the rest is kept - used by the
        // opacity reset, which momentum would otherwise undo within a few steps.
        if (zeroSlot >= 0 && zeroSlot < AdamSlots)
            for (long i = 0; i < n; i++) { m[i * AdamSlots + zeroSlot] = 0f; v[i * AdamSlots + zeroSlot] = 0f; }

        _adamM!.CopyFromCPU(m);
        _adamV!.CopyFromCPU(v);
        _adamStepCount = prior.StepCount;
    }

    /// <summary>
    /// Peak key demand seen since it was last cleared, so a caller can re-size on evidence.
    ///
    /// <see cref="LastOverflowed"/> is per-frame and a densification window spans hundreds of
    /// frames, so a caller that only looked at the last one would miss every overflow but one.
    /// </summary>
    public int PeakKeyDemand { get; private set; }

    /// <summary>Clear the peak, at the start of a new densification window.</summary>
    public void ResetPeakKeyDemand() => PeakKeyDemand = 0;

    /// <summary>
    /// Where a view's gradient dies: magnitudes at each stage of the chain, plus whether the
    /// forward pass painted anything. keys + loss + zero per-key can mean "forward was black"
    /// or "backward dropped a live render" - those are different files.
    /// </summary>
    public async Task<(double DLdPix, double PerKey, double MeanColour, double MeanFinalT,
        int NanKeys, int NanT, double FracTOneOpaque)> ReadBackwardStagesAsync()
    {
        long pixels = (long)_width * _height;
        double dl = await MaxMagnitudeAsync(_dLdPix!, pixels * 3);
        double pk = await MaxMagnitudeAsync(_gradKeyA!, (long)LastKeyCount * 3);
        var (meanC, _, _) = await SampleStatsAsync(_outColour!, pixels * 3);
        // max_magnitude reports 0 for an all-NaN buffer (max(0, NaN) drops the NaN), so a NaN
        // backward looks identical to an empty one. Count NaNs explicitly. T == 1-MAX_ALPHA at
        // a pixel means exactly one splat was applied at the alpha cap and nothing after it;
        // that is what min(MAX_ALPHA, NaN) produces in the forward.
        var (_, nanKeys, _) = await SampleStatsAsync(_gradKeyA!, (long)LastKeyCount * 3);
        var (meanT, nanT, fracT) = await SampleStatsAsync(_outFinalT!, pixels, 0.0100000179f, 1e-6f);
        return (dl, pk, meanC, meanT, nanKeys, nanT, fracT);
    }

    /// <summary>
    /// Stride-sampled (mean |x|, NaN count, fraction within <paramref name="tol"/> of
    /// <paramref name="match"/>) over a GPU buffer. Stage-probe only: it runs on the handful
    /// of views that produced no gradient, never in the training loop.
    /// </summary>
    async Task<(double MeanAbs, int NanCount, double FracMatch)> SampleStatsAsync(
        MemoryBuffer1D<float, Stride1D.Dense> buf, long count, float match = float.NaN, float tol = 0f)
    {
        if (count <= 0 || _fitSample == null || _sampleStride == null) return (0, 0, 0);
        int n = (int)Math.Min(FitSampleCount, count);
        int stride = Math.Max(1, (int)(count / n));
        n = (int)Math.Min(FitSampleCount, (count + stride - 1) / stride);
        WriteU32x4(_dimsBuf!, (uint)n, (uint)stride, (uint)Math.Min(count, buf.Length), 0);
        Dispatch(_sampleStride, (n + 255) / 256, 1, new[]
        {
            Buf(0, buf.GetGPUBuffer()!), Buf(1, _fitSample.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // CPU transfer: 4K-float stride sample for the dead-view stage probe.
        float[] host = await _fitSample.CopyToHostAsync<float>(0, n);
        double sum = 0; int nan = 0, hit = 0;
        for (int i = 0; i < host.Length; i++)
        {
            float v = host[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
            sum += Math.Abs(v);
            if (!float.IsNaN(match) && Math.Abs(v - match) <= tol) hit++;
        }
        int finite = host.Length - nan;
        return (finite > 0 ? sum / finite : 0, nan, host.Length > 0 ? (double)hit / host.Length : 0);
    }

    async Task<double> MaxMagnitudeAsync(MemoryBuffer1D<float, Stride1D.Dense> buf, long count)
    {
        if (count <= 0) return 0;
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)Math.Min(count, buf.Length), 0, 0, 0);
        Dispatch(_maxMagnitude!, GradStatsWorkgroups, 1, new[]
        {
            Buf(0, buf.GetGPUBuffer()!), Buf(1, _gradStatsPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();
        float[] p = await _gradStatsPartials!.CopyToHostAsync<float>(0, GradStatsWorkgroups);
        double m = 0;
        foreach (float v in p) m = Math.Max(m, v);
        return m;
    }

    /// <summary>Start a fresh densification window. Call after each densify step.</summary>
    public void ResetDensifyStats() => _densifyStats!.MemSetToZero();

    /// <summary>
    /// Fold the step that just finished into the densification statistics. One dispatch, no sync.
    ///
    /// Same timing constraint as <see cref="ReadGradientStatsAsync"/>: the accumulator is cleared
    /// mid-step, so this must run after TrainStepAsync returns and before the next one begins.
    /// </summary>
    public void AccumulateDensifyStats(int splatCount)
    {
        WriteU32x4(_dimsBuf!, (uint)splatCount, (uint)_width, (uint)_height, 0);
        Dispatch(_densifyAccum!, (splatCount + 255) / 256, 1, new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _densifyStats!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
    }

    /// <summary>
    /// Read the accumulated densification statistics.
    ///
    /// CPU transfer: 8 bytes per splat, and the decision it feeds changes the splat COUNT, which
    /// means reallocating every buffer anyway. At 80k splats this is 640 KB every hundred
    /// iterations. Doing the clone/split decision on the GPU would need a compaction pass and a
    /// prefix sum; that is worth writing when the count makes it worth writing.
    /// </summary>
    public async Task<SplatDensityControl.Accumulator[]> ReadDensifyStatsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        float[] raw = await _densifyStats!.CopyToHostAsync<float>(0, (long)splatCount * 2);

        var stats = new SplatDensityControl.Accumulator[splatCount];
        for (int i = 0; i < splatCount; i++)
            stats[i] = new SplatDensityControl.Accumulator
            {
                GradientSum = raw[i * 2],
                VisibleCount = (int)raw[i * 2 + 1],
            };
        return stats;
    }

    const int SupportWorkgroups = 256;
    const int SupportSlots = 6;

    /// <summary>
    /// How many distinct views constrain each splat, bucketed.
    ///
    /// <paramref name="Views"/> is how many views were accumulated, so the buckets can be read
    /// against the maximum a splat could possibly have reached.
    /// </summary>
    public readonly record struct ViewSupport(
        long Splats, int Views,
        long Unconstrained, long OneView, long TwoViews, long ThreeViews, long FourOrMore,
        double MeanViews)
    {
        /// <summary>
        /// Splats that no two views agree on - the ones a second view could contradict but
        /// never does. If this is most of the scene, the reconstruction is a stack of per-view
        /// shells and cannot generalise to a view nobody trained on, whatever the optimiser does.
        /// </summary>
        public double SingleViewFraction =>
            Splats > 0 ? (double)(Unconstrained + OneView) / Splats : 0.0;
    }

    /// <summary>Start a fresh support count. Call once, then accumulate over a full cycle.</summary>
    public void ResetViewSupport(int splatCount)
    {
        _viewSupport!.MemSetToZero();
        _supportViewsAccumulated = 0;
    }

    int _supportViewsAccumulated;

    /// <summary>
    /// Fold the step that just finished into the support count. One dispatch, no sync.
    ///
    /// Same timing constraint as <see cref="ReadGradientStatsAsync"/>: the accumulator is
    /// cleared mid-step, so this must run after TrainStepAsync returns and before the next.
    /// </summary>
    public void AccumulateViewSupport(int splatCount)
    {
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        Dispatch(_accumulateSupport!, (splatCount + 255) / 256, 1, new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _viewSupport!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        _supportViewsAccumulated++;
    }

    /// <summary>Reduce the support counts to buckets and bring back 6 KB.</summary>
    public async Task<ViewSupport> ReadViewSupportAsync(int splatCount)
    {
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        Dispatch(_supportHistogram!, SupportWorkgroups, 1, new[]
        {
            Buf(0, _viewSupport!.GetGPUBuffer()!), Buf(1, _supportPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();

        // CPU transfer: 6 floats per workgroup. Every slot is a sum.
        float[] p = await _supportPartials!.CopyToHostAsync<float>(
            0, SupportWorkgroups * SupportSlots);

        double c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, total = 0;
        for (int i = 0; i < SupportWorkgroups; i++)
        {
            int o = i * SupportSlots;
            c0 += p[o]; c1 += p[o + 1]; c2 += p[o + 2];
            c3 += p[o + 3]; c4 += p[o + 4]; total += p[o + 5];
        }

        return new ViewSupport(
            splatCount, _supportViewsAccumulated,
            (long)c0, (long)c1, (long)c2, (long)c3, (long)c4,
            splatCount > 0 ? total / splatCount : 0.0);
    }

    /// <summary>
    /// Per-splat view-support counts after a full cycle. Used to compact out the unconstrained
    /// half before densification starts spending budget on them.
    ///
    /// CPU transfer: 4 bytes per splat. One read per run, and the alternative is leaving half
    /// the scene as dead weight that can only hurt held-out views.
    /// </summary>
    public async Task<uint[]> ReadViewSupportCountsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        return await _viewSupport!.CopyToHostAsync<uint>(0, splatCount);
    }

    /// <summary>
    /// Skip the colour/opacity Adam step for splats whose gradient is exactly zero.
    ///
    /// Off by default. adam_geometry has always done this for position; whether it helps colour
    /// and opacity at batch size 1 is the measurement, and a default that changes quietly would
    /// make every earlier run incomparable.
    /// </summary>
    public bool SkipZeroGradientSteps { get; set; }

    /// <summary>The viewport the trainer is currently sized for.</summary>
    public (int Width, int Height) Size => (_width, _height);

    int SseWorkgroups => (_width * _height + 255) / 256;

    /// <summary>Count of 'valid' SSIM window positions across and down.</summary>
    int SsimWindowsX => _width - ImageQuality.WindowSize + 1;
    int SsimWindowsY => _height - ImageQuality.WindowSize + 1;
    bool HasSsimWindows => _width >= ImageQuality.WindowSize && _height >= ImageQuality.WindowSize;
    int SsimWorkgroups => (SsimWindowsX * SsimWindowsY + 255) / 256;

    /// <summary>
    /// Upload every SSIM constant from <see cref="ImageQuality"/>, so the shader holds none.
    ///
    /// The one value WGSL must duplicate is the window size, because it sizes a loop and a
    /// shader cannot read a C# constant. Asserting it here rather than trusting it: if the
    /// oracle's window ever changes, this throws with the file to edit instead of silently
    /// scoring two different metrics.
    /// </summary>
    void WriteSsimCfg()
    {
        if (ImageQuality.WindowSize != 11)
            throw new InvalidOperationException(
                $"ImageQuality.WindowSize is {ImageQuality.WindowSize} but the WGSL in " +
                "SplatTrainerShaders.SsimRows/SsimReduce declares 'const WINDOW : u32 = 11u'. " +
                "Change both or they measure different things.");

        var k = ImageQuality.GaussianKernel1D();
        var f = new float[20];                       // 80 bytes: luma, consts, 3 x vec4 of taps
        f[0] = (float)ImageQuality.LumaR;
        f[1] = (float)ImageQuality.LumaG;
        f[2] = (float)ImageQuality.LumaB;
        f[4] = (float)ImageQuality.C1;
        f[5] = (float)ImageQuality.C2;
        for (int i = 0; i < k.Length; i++) f[8 + i] = (float)k[i];

        var bytes = new byte[80];
        Buffer.BlockCopy(f, 0, bytes, 0, 80);
        _queue!.WriteBuffer(_ssimCfgBuf!, 0, bytes);
    }

    /// <summary>
    /// PSNR of the current render against one image in a GPU-resident target stack.
    ///
    /// The whole comparison happens on the GPU; only one partial sum per 256 pixels comes back,
    /// a few KB rather than the two full images this used to copy. On a 35-view capture that is
    /// the difference between about 650 MB of readback per evaluation pass and about 600 KB.
    /// </summary>
    public async Task<double> PsnrAgainstAsync(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex)
        => (await ScoreAgainstAsync(stack, viewIndex, withSsim: false)).Psnr;

    /// <summary>
    /// PSNR and SSIM of the current render against one image in the target stack.
    ///
    /// Both in one call because they share the render and can share the sync: three dispatches
    /// are queued, then ONE SynchronizeAsync, then both partial buffers come back. Scoring them
    /// separately would double the round trips per view for no benefit.
    ///
    /// SSIM matters because PSNR does not see the failure this trainer currently has. MEASURED
    /// on Bathroom: held out went 12.50 -> 12.21 dB across a run, essentially flat, while the
    /// render melted from a recognisable room into fog. PSNR on a sparse reconstruction is
    /// dominated by large smooth regions, so smoothing structure away barely moves it.
    ///
    /// Returns NaN for SSIM when the viewport is smaller than the window - there are no valid
    /// window positions and the metric is undefined, which the Python oracle signals by raising.
    /// </summary>
    public async Task<(double Psnr, double Ssim)> ScoreAgainstAsync(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex, bool withSsim = true)
    {
        var accel = _gpu.WebGPUAccelerator;

        // Unpack the view being scored into the working target, then compare against THAT.
        // Scoring straight out of the stack would mean teaching eval_sse and ssim_rows to
        // unpack as well, which is two more places for the pixel format to drift apart from
        // its oracle. One unpack, one format, one gate.
        SetTargetFrom(stack, viewIndex);
        uint targetOffset = 0;

        WriteU32x2(_dimsBuf!, (uint)(_width * _height), targetOffset);
        Dispatch(_evalSse!, SseWorkgroups, 1, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
            Buf(2, _ssePartials!.GetGPUBuffer()!), Buf(3, _dimsBuf!),
        });

        bool doSsim = withSsim && HasSsimWindows && _ssimRows != null;
        if (doSsim)
        {
            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY,
                (uint)_width, targetOffset);

            // Horizontal pass over every source ROW (not just window rows): the vertical pass
            // reads eleven rows above each window, so the rows above the last window position
            // are still needed.
            int rowThreads = SsimWindowsX * _height;
            Dispatch(_ssimRowsPipe!, (rowThreads + 63) / 64, 1, new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimRows!.GetGPUBuffer()!), Buf(3, _ssimDimsBuf!), Buf(4, _ssimCfgBuf!),
            });
            Dispatch(_ssimReducePipe!, SsimWorkgroups, 1, new[]
            {
                Buf(0, _ssimRows!.GetGPUBuffer()!), Buf(1, _ssimPartials!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!),
            });
        }

        await accel.SynchronizeAsync();

        // CPU transfer: one float per 256 pixels / per 256 windows, summed in double because a
        // million f32 adds in sequence loses the tail.
        float[] partials = await _ssePartials!.CopyToHostAsync<float>(0, SseWorkgroups);
        double sse = 0;
        foreach (float v in partials) sse += v;
        double mse = sse / ((long)_width * _height * 3);
        double psnr = mse <= 1e-12 ? 99.0 : 10.0 * Math.Log10(1.0 / mse);

        double ssim = double.NaN;
        if (doSsim)
        {
            float[] sp = await _ssimPartials!.CopyToHostAsync<float>(0, SsimWorkgroups);
            double total = 0;
            foreach (float v in sp) total += v;
            ssim = total / ((double)SsimWindowsX * SsimWindowsY);
        }
        return (psnr, ssim);
    }

    /// <summary>
    /// Put one target photograph into a GPU-resident stack, straight from the canvas.
    ///
    /// <paramref name="rgba"/> is the JS typed array from <c>getImageData</c>; its bytes go to
    /// the GPU without passing through the managed heap, and a kernel expands them into floats.
    /// Reading them into .NET to convert in a loop was a million iterations and about 6 MB of
    /// managed allocation per view.
    /// </summary>
    public void UploadTargetFrom(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex, Uint8Array rgba)
    {
        long pixels = (long)_width * _height;
        long off = (long)viewIndex * pixels;
        if (off + pixels > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} does not fit a {stack.Length}-pixel stack");

        // Straight into the stack. The bytes are already the format the stack stores, so there
        // is nothing to expand at upload time - the unpack moved to SetTargetFrom, where it
        // runs on one frame instead of all of them.
        _queue!.WriteBuffer(stack.GetGPUBuffer()!, (ulong)(off * 4), rgba);
    }

    /// <summary>Seed opacity logits from the splats' current opacity. Call once before training.</summary>
    public void InitOptimizerState(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        var splatGpu = splatBuf.GetGPUBuffer()!;
        WriteU32(_dimsBuf!, (uint)splatCount);
        Dispatch(_initLogits!, (splatCount + 63) / 64, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _opacityLogit!.GetGPUBuffer()!), Buf(2, _dimsBuf!),
            Buf(3, _logScale!.GetGPUBuffer()!),
        });
        _adamM!.MemSetToZero();
        _adamV!.MemSetToZero();
        _adamShM?.MemSetToZero();
        _adamShV?.MemSetToZero();
        _adamStepCount = 0;
    }

    /// <summary>Convert packed linear RGB to SH DC once before the first training step.</summary>
    public void EnsureRgbConvertedToShDc(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        if (_rgbToDcDone) return;
        var splatGpu = splatBuf.GetGPUBuffer()!;
        WriteU32(_dimsBuf!, (uint)splatCount);
        Dispatch(_initRgbToDc!, (splatCount + 63) / 64, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _dimsBuf!),
        });
        _rgbToDcDone = true;
    }

    public async Task<float[]> ReadShRestAsync(int splatCount)
    {
        if (_shRest == null) return System.Array.Empty<float>();
        return await _shRest.CopyToHostAsync<float>(0, (long)splatCount * SphericalHarmonics.RestFloatsPerSplat);
    }

    /// <summary>
    /// SH-rest Adam moments across a densify. Same survivor map as colour Adam: keep for
    /// survivors, zero for densified children. Resize zeros these; without a restore every
    /// densify (every 100 iters) throws away SH momentum while colour Adam is carefully kept.
    /// </summary>
    public readonly record struct ShAdamState(float[] M, float[] V);

    public async Task<ShAdamState> ReadShAdamStateAsync(int splatCount)
    {
        if (_adamShM == null || _adamShV == null)
            return new ShAdamState(System.Array.Empty<float>(), System.Array.Empty<float>());
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        long len = (long)splatCount * SphericalHarmonics.RestFloatsPerSplat;
        return new ShAdamState(
            await _adamShM.CopyToHostAsync<float>(0, len),
            await _adamShV.CopyToHostAsync<float>(0, len));
    }

    public void RestoreShRest(ReadOnlySpan<float> prior, int[] featureSources)
    {
        if (_shRest == null) return;
        _shRest.CopyFromCPU(SplatDensityControl.RemapFloatRows(prior, featureSources, SphericalHarmonics.RestFloatsPerSplat));
    }

    public void RestoreShAdamState(ShAdamState prior, int[] adamSurvivors)
    {
        if (_adamShM == null || _adamShV == null) return;
        int stride = SphericalHarmonics.RestFloatsPerSplat;
        _adamShM.CopyFromCPU(SplatDensityControl.RemapFloatRows(prior.M, adamSurvivors, stride));
        _adamShV.CopyFromCPU(SplatDensityControl.RemapFloatRows(prior.V, adamSurvivors, stride));
    }

    GPUBindGroupEntry ShRestBindEntry(int binding) =>
        new()
        {
            Binding = (uint)binding,
            Resource = new GPUBufferBinding { Buffer = _shRest!.GetGPUBuffer()! },
        };

    /// <summary>
    /// One training iteration on the current target: forward, loss, backward, scatter, Adam.
    /// Returns the L1 loss BEFORE the step, so a caller can watch it fall.
    ///
    /// Deliberately one iteration per call. A long-running dispatch chain can trip the driver
    /// watchdog and lose the device, so the host yields between iterations rather than queueing
    /// a whole training run.
    /// </summary>
    public async Task<float> TrainStepAsync(
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar,
        float colourLr = SplatOptimizer.DefaultColourLr,
        float opacityLr = SplatOptimizer.DefaultOpacityLr,
        GeometryStep? geometry = null)
    {
        var accel = _gpu.WebGPUAccelerator;
        var splatGpu = splatBuf.GetGPUBuffer()!;

        // Forward also refreshes the tile binning for this view.
        await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
        if (LastKeyCount == 0)
        {
            // Clear before returning: the view-support census and densify accum read these
            // buffers after TrainStepAsync returns. Leaving the previous view's gradients here
            // made a zero-key view look live (Sonnet audit 2026-09-21).
            _gradFixed!.MemSetToZero();
            _densifyAbs!.MemSetToZero();
            return 0f;
        }

        int pixels = _width * _height;

        // ── Loss and dL/d(pixel): 0.8 L1 + 0.2 D-SSIM, matching the reference ──
        _lossFixed!.MemSetToZero();
        await accel.SynchronizeAsync();
        WriteU32(_dimsBuf!, (uint)pixels);
        WriteVec4(_lossWeightsBuf!, ImageQuality.LambdaL1, ImageQuality.LambdaDssim, 0f, 0f);
        Dispatch(_lossL1!, (pixels + 63) / 64, 1, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
            Buf(2, _dLdPix!.GetGPUBuffer()!), Buf(3, _lossFixed!.GetGPUBuffer()!),
            Buf(4, _dimsBuf!), Buf(5, _lossWeightsBuf!),
        });

        if (HasSsimWindows && _ssimRows != null && _ssimWinGrad != null && _ssimDRows != null)
        {
            // Reuse the scoring forward's horizontal pass, then the three adjoint passes that
            // mirror ImageQuality.AddMeanSsimLumaGradient (FD-gated).
            // ssim_rows dims: wx, wy (=height-10), srcW, targetOffset. srcH = wy+10.
            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_width, 0);
            int rowThreads = SsimWindowsX * _height;
            Dispatch(_ssimRowsPipe!, (rowThreads + 63) / 64, 1, new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimRows!.GetGPUBuffer()!), Buf(3, _ssimDimsBuf!), Buf(4, _ssimCfgBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_width, 0);
            WriteVec4(_lossWeightsBuf!, ImageQuality.LambdaDssim, 0f, 0f, 0f);
            int winThreads = SsimWindowsX * SsimWindowsY;
            Dispatch(_ssimWinGradPipe!, (winThreads + 255) / 256, 1, new[]
            {
                Buf(0, _ssimRows!.GetGPUBuffer()!), Buf(1, _ssimWinGrad!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!), Buf(4, _lossWeightsBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_height, 0);
            Dispatch(_ssimRowsBwdPipe!, (rowThreads + 255) / 256, 1, new[]
            {
                Buf(0, _ssimWinGrad!.GetGPUBuffer()!), Buf(1, _ssimDRows!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)_width, (uint)_height, 0);
            Dispatch(_ssimPixBwdPipe!, (pixels + 255) / 256, 1, new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimDRows!.GetGPUBuffer()!), Buf(3, _dLdPix!.GetGPUBuffer()!),
                Buf(4, _ssimDimsBuf!), Buf(5, _ssimCfgBuf!),
            });
        }

        // ── Backward: one workgroup per tile, no atomics for signed grads ──
        // densify_abs is filled HERE with peak per-pixel |dCentre| (AbsGS). Clear first so a
        // previous view cannot leak into densify_accum after this step.
        _densifyAbs!.MemSetToZero();
        await accel.SynchronizeAsync();
        // grad_per_key is written for every key this frame, so stale values cannot leak in.
        Dispatch(_rasterBackward!, _tilesX, _tilesY, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _outFinalT!.GetGPUBuffer()!),
            Buf(5, _outEnd!.GetGPUBuffer()!), Buf(6, _dLdPix!.GetGPUBuffer()!),
            Buf(7, _gradKeyA!.GetGPUBuffer()!), Buf(8, _gradKeyB!.GetGPUBuffer()!),
            Buf(9, _gradKeyC!.GetGPUBuffer()!),
            Buf(10, _densifyAbs!.GetGPUBuffer()!),
            ShRestBindEntry(12),
        });

        // Cleared here, not at the top of the step: the census and densify accum read the
        // completed step's totals after TrainStepAsync returns.
        _gradFixed!.MemSetToZero();
        await accel.SynchronizeAsync();

        // ── Scatter per-key gradients into per-splat totals (f32 CAS add) ──
        WriteU32(_countBuf!, (uint)LastKeyCount);
        DispatchLinear(_scatterGrad!, LastKeyCount, new[]
        {
            Buf(0, _gradKeyA!.GetGPUBuffer()!), Buf(1, _gradKeyB!.GetGPUBuffer()!),
            Buf(2, _gradKeyC!.GetGPUBuffer()!), Buf(3, _values!.GetGPUBuffer()!),
            Buf(4, _gradFixed!.GetGPUBuffer()!), Buf(5, _countBuf!),
        });

        // ── Adam ──
        _adamStepCount++;
        WriteVec4(_adamCfgBuf!, colourLr, opacityLr, _adamStepCount, splatCount);
        WriteVec4(_adamFlagsBuf!, SkipZeroGradientSteps ? 1f : 0f, 0f, 0f, 0f);
        Dispatch(_adamStep!, (splatCount + 63) / 64, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _gradFixed!.GetGPUBuffer()!),
            Buf(2, _opacityLogit!.GetGPUBuffer()!), Buf(3, _adamM!.GetGPUBuffer()!),
            Buf(4, _adamV!.GetGPUBuffer()!), Buf(5, _adamCfgBuf!), Buf(6, _adamFlagsBuf!),
        });

        if (ActiveShDegree >= 1 && _scatterShGrad != null && _adamShRest != null)
        {
            _gradShRest!.MemSetToZero();
            await accel.SynchronizeAsync();
            // Scatter cfg: .w = splat count (matches shader).
            WriteVec4(_adamCfgBuf!, 0f, 0f, 0f, splatCount);
            Dispatch(_scatterShGrad, (splatCount + 63) / 64, 1, new[]
            {
                Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _gradFixed!.GetGPUBuffer()!),
                Buf(3, _gradShRest!.GetGPUBuffer()!), Buf(4, _adamCfgBuf!),
            });
            // Rest bands use feature_lr / 20 (Kerbl).
            WriteVec4(_adamCfgBuf!, colourLr / 20f, 0f, _adamStepCount, splatCount);
            Dispatch(_adamShRest, (splatCount + 63) / 64, 1, new[]
            {
                Buf(0, _shRest!.GetGPUBuffer()!), Buf(1, _gradShRest!.GetGPUBuffer()!),
                Buf(2, _adamShM!.GetGPUBuffer()!), Buf(3, _adamShV!.GetGPUBuffer()!),
                Buf(4, _adamCfgBuf!),
            });
        }

        // -- Geometry: the 2D gradients chained back to position, scale and rotation --
        // Separate dispatch, and optional, so a run can isolate whether a change came from
        // the colours or from the geometry moving.
        if (geometry is { } geo)
        {
            WriteVec4x2(_geomCfgBuf!,
                geo.PositionLr, geo.LogScaleLr, geo.RotationLr, _adamStepCount,
                splatCount, geo.MaxScale, geo.MinScale, 0f);
            Dispatch(_adamGeometry!, (splatCount + 63) / 64, 1, new[]
            {
                Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _gradFixed!.GetGPUBuffer()!),
                Buf(3, _logScale!.GetGPUBuffer()!), Buf(4, _adamM!.GetGPUBuffer()!),
                Buf(5, _adamV!.GetGPUBuffer()!), Buf(6, _geomCfgBuf!),
                Buf(7, _geomOut!.GetGPUBuffer()!),
            });
        }

        await accel.SynchronizeAsync();

        int[] lossRaw = await _lossFixed!.CopyToHostAsync<int>(0, 1);
        return lossRaw[0] / 1048576f;
    }

    /// <summary>
    /// Learning rates for the geometric parameters. Position is scaled by the scene extent, as
    /// the reference does - a learning rate in world units means nothing without it. The scale
    /// bounds stop a splat that stops being constrained from collapsing to nothing or swelling
    /// to cover the frame; the reference prunes those instead, which needs density control.
    /// </summary>
    public readonly record struct GeometryStep(
        float PositionLr,
        float LogScaleLr,
        float RotationLr,
        float MinScale,
        float MaxScale);

    /// <summary>
    /// Per-splat accumulated gradients. Nine per splat, in the shader's order:
    /// colour RGB, opacity, screen centre x and y, conic a, b and c. For the GPU gate only.
    /// </summary>
    /// <remarks>
    /// Reads the WHOLE accumulator, so this is for the gate's few-hundred-splat scene, not for a
    /// real one. There used to be a maxSplats prefix option here for summary statistics; it is
    /// gone because a prefix of a VIEW-MAJOR splat buffer is not a sample - it is the top of
    /// view 0's depth map, which can be entirely zero while the buffer is full, and that is
    /// exactly how the health probe went blind. Use <see cref="ReadGradientStatsAsync"/>.
    /// </remarks>
    public async Task<float[]> ReadGradientsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // The buffer is f32 bit patterns in an int-typed allocation; reinterpret, do not convert.
        int[] raw = await _gradFixed!.CopyToHostAsync<int>(0, (long)splatCount * GradsPerSplat);
        var outp = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++) outp[i] = BitConverter.Int32BitsToSingle(raw[i]);
        return outp;
    }

    /// <summary>
    /// Geometry gradients from the last step: 10 per splat, position xyz then scale xyz then
    /// quaternion xyzw. What the GPU gate compares against the CPU oracle.
    /// </summary>
    public Task<float[]> ReadGeometryGradientsAsync(int splatCount) =>
        _geomOut!.CopyToHostAsync<float>(0, (long)splatCount * GeomGradsPerSplat);

    static GPUBindGroupEntry Buf(uint binding, GPUBuffer b) =>
        new() { Binding = binding, Resource = new GPUBufferBinding { Buffer = b } };

    /// <summary>
    /// Largest workgroup count WebGPU guarantees in any one dimension. Exceeding it does not
    /// clamp - the dispatch is rejected, and the driver reports it from whatever runs next.
    /// </summary>
    const int MaxWorkgroupsPerDim = 65535;

    /// <summary>
    /// Dispatch enough 64-thread workgroups to cover <paramref name="threads"/>, wrapping into
    /// a second dimension past the per-dimension limit. One dimension tops out at 4.19M threads
    /// and a room-scale scene emits more keys than that, so every key-indexed pass needs this.
    /// The shader recovers the flat index from workgroup_id and num_workgroups.
    /// </summary>
    void DispatchLinear(GPUComputePipeline pipeline, long threads, GPUBindGroupEntry[] entries)
    {
        var (wgX, wgY) = LinearGrid(threads);
        Dispatch(pipeline, wgX, wgY, entries);
    }

    /// <summary>Workgroup grid covering <paramref name="threads"/> at 64 per group.</summary>
    static (int X, int Y) LinearGrid(long threads)
    {
        long groups = Math.Max(1, (threads + 63) / 64);
        int x = (int)Math.Min(groups, MaxWorkgroupsPerDim);
        int y = (int)((groups + MaxWorkgroupsPerDim - 1) / MaxWorkgroupsPerDim);
        return (x, y);
    }

    void Dispatch(GPUComputePipeline pipeline, int wgX, int wgY, GPUBindGroupEntry[] entries)
    {
        using var enc = _device!.CreateCommandEncoder();
        using var pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        using var layout = pipeline.GetBindGroupLayout(0);
        using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor { Layout = layout, Entries = entries });
        pass.SetBindGroup(0, bg);
        pass.DispatchWorkgroups((uint)Math.Max(1, wgX), (uint)Math.Max(1, wgY), 1);
        pass.End();
        using var cmd = enc.Finish();
        _queue!.Submit(new[] { cmd });
    }

    void WriteU32x4(GPUBuffer buf, uint x, uint y, uint z, uint w)
    {
        var v = new uint[] { x, y, z, w };
        var bytes = new byte[16];
        Buffer.BlockCopy(v, 0, bytes, 0, 16);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    void WriteVec4(GPUBuffer buf, float x, float y, float z, float w)
    {
        var f = new[] { x, y, z, w };
        var bytes = new byte[16];
        Buffer.BlockCopy(f, 0, bytes, 0, 16);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    void WriteVec4x2(GPUBuffer buf, float a, float b, float c, float d,
                     float e, float f, float g, float h)
    {
        var v = new[] { a, b, c, d, e, f, g, h };
        var bytes = new byte[32];
        Buffer.BlockCopy(v, 0, bytes, 0, 32);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    void WriteUniforms(CameraParams cam, float depthNear, float depthFar, int splatCount)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);

        var f = new float[UniformFloats];
        void V4(int o, Vector3 v, float w) { f[o] = v.X; f[o + 1] = v.Y; f[o + 2] = v.Z; f[o + 3] = w; }
        V4(0, right, 0f);
        V4(4, up, 0f);
        V4(8, fwd, 0f);
        V4(12, pos, 1f);
        f[16] = cam.FocalX; f[17] = cam.FocalY;
        f[18] = cam.CenterX; f[19] = cam.CenterY;
        f[20] = _width; f[21] = _height;
        f[22] = BitConverter.Int32BitsToSingle(_tilesX);
        f[23] = BitConverter.Int32BitsToSingle(_tilesY);
        f[24] = depthNear;
        f[25] = depthFar;
        f[26] = BitConverter.Int32BitsToSingle(splatCount);
        f[27] = BitConverter.Int32BitsToSingle(Math.Clamp(ActiveShDegree, 0, SphericalHarmonics.MaxDegree));

        Buffer.BlockCopy(f, 0, _uniformBytes!, 0, _uniformBytes!.Length);
        _queue!.WriteBuffer(_uniformBuf!, 0, _uniformBytes);
    }

    void WriteU32x2(GPUBuffer buf, uint x, uint y)
    {
        _scratch4![0] = x;
        _scratch4[1] = y;
        _scratch4[2] = 0; _scratch4[3] = 0;
        _queue!.WriteBuffer(buf, 0, _scratch4);
    }

    void WriteU32(GPUBuffer buf, uint value)
    {
        _scratch4![0] = value;
        _scratch4[1] = 0; _scratch4[2] = 0; _scratch4[3] = 0;
        _queue!.WriteBuffer(buf, 0, _scratch4);
    }

    void DisposeBuffers()
    {
        _keys?.Dispose(); _keys = null;
        _values?.Dispose(); _values = null;
        _counter?.Dispose(); _counter = null;
        _ranges?.Dispose(); _ranges = null;
        _outColour?.Dispose(); _outColour = null;
        _outFinalT?.Dispose(); _outFinalT = null;
        _outEnd?.Dispose(); _outEnd = null;
        _sortTemp?.Dispose(); _sortTemp = null;
        _target?.Dispose(); _target = null;
        _dLdPix?.Dispose(); _dLdPix = null;
        _gradKeyA?.Dispose(); _gradKeyA = null;
        _gradKeyB?.Dispose(); _gradKeyB = null;
        _gradKeyC?.Dispose(); _gradKeyC = null;
        _gradFixed?.Dispose(); _gradFixed = null;
        _densifyAbs?.Dispose(); _densifyAbs = null;
        _opacityLogit?.Dispose(); _opacityLogit = null;
        _logScale?.Dispose(); _logScale = null;
        _geomOut?.Dispose(); _geomOut = null;
        _ssePartials?.Dispose(); _ssePartials = null;
        _ssimRows?.Dispose(); _ssimRows = null;
        _ssimPartials?.Dispose(); _ssimPartials = null;
        _ssimWinGrad?.Dispose(); _ssimWinGrad = null;
        _ssimDRows?.Dispose(); _ssimDRows = null;
        _gradStatsPartials?.Dispose(); _gradStatsPartials = null;
        _fitSample?.Dispose(); _fitSample = null;
        _densifyStats?.Dispose(); _densifyStats = null;
        _viewSupport?.Dispose(); _viewSupport = null;
        _supportPartials?.Dispose(); _supportPartials = null;
        _adamM?.Dispose(); _adamM = null;
        _adamV?.Dispose(); _adamV = null;
        _shRest?.Dispose(); _shRest = null;
        _gradShRest?.Dispose(); _gradShRest = null;
        _adamShM?.Dispose(); _adamShM = null;
        _adamShV?.Dispose(); _adamShV = null;
        _lossFixed?.Dispose(); _lossFixed = null;
    }

    public void Dispose()
    {
        DisposeBuffers();
        _uniformBuf?.Destroy(); _uniformBuf?.Dispose();
        _capsBuf?.Destroy(); _capsBuf?.Dispose();
        _geomCfgBuf?.Destroy(); _geomCfgBuf?.Dispose();
        _targetBytes?.Destroy(); _targetBytes?.Dispose();
        _countBuf?.Destroy(); _countBuf?.Dispose();
        _dimsBuf?.Destroy(); _dimsBuf?.Dispose();
        _adamCfgBuf?.Destroy(); _adamCfgBuf?.Dispose();
        _adamFlagsBuf?.Destroy(); _adamFlagsBuf?.Dispose();
        _ssimDimsBuf?.Destroy(); _ssimDimsBuf?.Dispose();
        _ssimCfgBuf?.Destroy(); _ssimCfgBuf?.Dispose();
        _lossWeightsBuf?.Destroy(); _lossWeightsBuf?.Dispose();
        _scratch4?.Dispose();
        _emitKeys?.Dispose();
        _tileRanges?.Dispose();
        _rasterForward?.Dispose();
        _rasterBackward?.Dispose();
        _scatterGrad?.Dispose();
        _lossL1?.Dispose();
        _ssimRowsPipe?.Dispose();
        _ssimReducePipe?.Dispose();
        _ssimWinGradPipe?.Dispose();
        _ssimRowsBwdPipe?.Dispose();
        _ssimPixBwdPipe?.Dispose();
        _adamStep?.Dispose();
        _initLogits?.Dispose();
    }
}
