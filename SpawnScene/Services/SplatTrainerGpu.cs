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
    GPUComputePipeline? _gradStats;
    GPUComputePipeline? _unpackTarget;

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
    MemoryBuffer1D<int, Stride1D.Dense>? _gradFixed;     // 4 per splat, fixed point
    MemoryBuffer1D<float, Stride1D.Dense>? _opacityLogit;
    MemoryBuffer1D<float, Stride1D.Dense>? _logScale;    // 3 per splat
    MemoryBuffer1D<float, Stride1D.Dense>? _geomOut;     // 10 per splat, diagnostics + gate
    MemoryBuffer1D<float, Stride1D.Dense>? _ssePartials; // 1 per 256-pixel workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimRows;    // 5 filtered channels per (window col, row)
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimPartials;// 1 per 256-window workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _gradStatsPartials; // 6 per workgroup, 256 workgroups
    MemoryBuffer1D<float, Stride1D.Dense>? _adamM;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamV;
    MemoryBuffer1D<int, Stride1D.Dense>? _lossFixed;
    GPUBuffer? _dimsBuf;
    GPUBuffer? _ssimDimsBuf;
    GPUBuffer? _ssimCfgBuf;
    GPUBuffer? _adamFlagsBuf;
    GPUBuffer? _gradScaleBuf;
    GPUBuffer? _adamCfgBuf;
    GPUBuffer? _geomCfgBuf;
    GPUBuffer? _targetBytes;   // one frame of packed RGBA, straight from the canvas
    int _adamStepCount;

    /// <summary>Gradient slots per splat. Must match GRADS_PER_SPLAT in the shaders.</summary>
    public const int GradsPerSplat = SplatTileRasterizer.GradsPerKey;

    /// <summary>Adam moment slots per splat: 3 colour, 1 opacity, 3 position, 3 scale, 4 quaternion.</summary>
    const int AdamSlots = 14;

    /// <summary>Geometry gradients reported per splat: position xyz, scale xyz, quaternion xyzw.</summary>
    public const int GeomGradsPerSplat = 10;

    /// <summary>
    /// Fixed-point scale for each of the nine gradient slots. Must match the shaders.
    /// Slots 0..5 (colour, opacity, screen centre) are bounded small and get 2^26; the conic
    /// grows with a splat's pixel area and keeps 2^20 for the range. See the note in
    /// SplatTrainerShaders.UniformsBlock.
    /// </summary>
    /// <summary>Scale for slots 0..3 - colour and opacity. Bounded: colour gradients are &lt;= 1/3.</summary>
    public float ColourScale { get; private set; } = DefaultColourScale;

    /// <summary>
    /// Scale for slots 4..5, the screen centre.
    ///
    /// Separate from colour because it is orders of magnitude smaller. They shared a scale, and
    /// fitting that shared scale to the centre put COLOUR past the i32 wrap - measured, by doing
    /// exactly that: held-out PSNR fell 12.33 -> 9.95 dB on drjohnson in the run that tried it.
    /// </summary>
    public float CentreScale { get; private set; } = DefaultCentreScale;

    /// <summary>Scale for slots 6..8, the conic - the one that grows with splat AREA.</summary>
    public float ConicScale { get; private set; } = DefaultConicScale;

    public const float DefaultColourScale = 67108864f;   // 2^26
    public const float DefaultCentreScale = 67108864f;   // 2^26
    public const float DefaultConicScale = 1048576f;     // 2^20

    /// <summary>
    /// The scale a gradient slot is stored at. Was a static with two literals in it, which is
    /// precisely why it ended up wrong for a scene nobody had tried yet.
    /// </summary>
    public float FixedScaleFor(int slot) => slot < 4 ? ColourScale : slot < 6 ? CentreScale : ConicScale;

    /// <summary>
    /// Re-scale the fixed-point gradients from what the last step actually produced.
    ///
    /// Both scales were consts chosen for TempleRing and Bathroom, and on drjohnson they were
    /// wrong in OPPOSITE directions at once: the screen-centre gradient carried ONE quantum of
    /// 32, so position was rounded away and the geometry could not move, while the conic sat at
    /// 96% of the i32 wrap point - and the atomic WRAPS rather than clamping, so those gradients
    /// were wrong rather than merely coarse. The conic grows with a splat's screen AREA, so no
    /// single constant can serve a 3.8-unit scene and a 14.5-unit one.
    ///
    /// Aim each scale so the largest value seen lands at <see cref="TargetSaturation"/> of the
    /// ceiling: far enough from the wrap to be safe, close enough that the small values still
    /// land on several quanta rather than rounding to zero. Driven by the measurement rather
    /// than by a formula, because the thing being measured is the whole problem.
    ///
    /// Returns true when anything moved, so a caller can say so.
    /// </summary>
    public bool RecalibrateGradientScales(GradientStats stats)
    {
        const float TargetSaturation = 0.25f;
        float ceiling = int.MaxValue * TargetSaturation;

        static float Retarget(float current, double maxQuanta, float ceiling, float fallback)
        {
            // Nothing was measured, so there is nothing to fit to - leave it alone rather than
            // inventing a scale from an empty sample.
            if (maxQuanta <= 0) return fallback;
            double maxValue = maxQuanta / current;
            double wanted = ceiling / maxValue;
            // Keep it a power of two: the quantisation error is then exactly representable and
            // a scale change cannot introduce a rounding difference of its own.
            double exp = Math.Round(Math.Log2(wanted));
            return (float)Math.Pow(2, Math.Clamp(exp, 4, 40));
        }

        // Colour has an ANALYTIC bound the measurement cannot see: with an L1 loss averaged over
        // the image, |dL/dcolour| <= 1/3. Fitting purely to the largest value observed SO FAR
        // would pick a scale that wraps the moment a gradient grows - and drjohnson's first
        // reading, 8.55e3 quanta at 2^26, is 1.3e-4, which is 2600x below that bound. A scale
        // chosen from it alone leaves no room for the run to change.
        //
        // So the measurement sets the scale and the bound caps it: never finer than the point
        // where the largest gradient the loss can produce still fits.
        const float ColourGradientBound = 1f / 3f;
        float colourCap = (float)Math.Pow(2, Math.Floor(Math.Log2(int.MaxValue / ColourGradientBound)));
        float colour = Math.Min(
            Retarget(ColourScale, stats.MaxColourQuanta, ceiling, ColourScale), colourCap);
        float centre = Retarget(CentreScale, stats.MaxCentreQuanta, ceiling, CentreScale);
        float conic = Retarget(ConicScale, stats.MaxConicQuanta, ceiling, ConicScale);

        bool moved = Math.Abs(Math.Log2(colour / ColourScale)) >= 0.5
                  || Math.Abs(Math.Log2(centre / CentreScale)) >= 0.5
                  || Math.Abs(Math.Log2(conic / ConicScale)) >= 0.5;
        if (!moved) return false;

        Console.WriteLine(
            $"[Trainer] gradient scales: colour 2^{Math.Log2(ColourScale):F0} -> 2^{Math.Log2(colour):F0}, " +
            $"centre 2^{Math.Log2(CentreScale):F0} -> 2^{Math.Log2(centre):F0}, " +
            $"conic 2^{Math.Log2(ConicScale):F0} -> 2^{Math.Log2(conic):F0} " +
            $"(max quanta: colour {stats.MaxColourQuanta:G3}, centre {stats.MaxCentreQuanta:G3}, " +
            $"conic {stats.MaxConicQuanta:G3}; aiming at {TargetSaturation:P0} of the i32 ceiling)");

        ColourScale = colour;
        CentreScale = centre;
        ConicScale = conic;
        return true;
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
        _gradStats = MakePipeline(SplatTrainerShaders.GradStats, "grad_stats");
        _unpackTarget = MakePipeline(SplatTrainerShaders.UnpackTarget, "unpack_target");

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

        // Shared by scatter_gradients (writer) and both adam passes (readers). One buffer so a
        // step cannot write at one scale and read at another.
        _gradScaleBuf = _device.CreateBuffer(new GPUBufferDescriptor
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

        Console.WriteLine("[Trainer] pipelines created: emit_keys, tile_ranges, raster_forward, " +
            "raster_backward, scatter_gradients, loss_l1, adam_step, init_logits, adam_geometry, " +
            "eval_sse, unpack_target, ssim_rows, ssim_reduce, grad_stats");
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
        const long MaxBindingBytes = 128L * 1024 * 1024;
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
                $"one binding, over the {MaxBindingBytes / (1024 * 1024)} MiB guarantee. " +
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
        _opacityLogit = accel.Allocate1D<float>(splatCount);
        _logScale = accel.Allocate1D<float>((long)splatCount * 3);
        _geomOut = accel.Allocate1D<float>((long)splatCount * GeomGradsPerSplat);
        _ssePartials = accel.Allocate1D<float>(SseWorkgroups);
        _gradStatsPartials = accel.Allocate1D<float>(GradStatsWorkgroups * GradStatsSlots);

        // SSIM works on 'valid' windows, so a viewport smaller than the window has none and the
        // metric is genuinely undefined there - the Python oracle raises rather than inventing a
        // number, and ScoreAgainstAsync reports NaN for the same reason.
        if (HasSsimWindows)
        {
            _ssimRows = accel.Allocate1D<float>((long)SsimWindowsX * _height * 5);
            _ssimPartials = accel.Allocate1D<float>(SsimWorkgroups);
            WriteSsimCfg();
        }
        _adamM = accel.Allocate1D<float>((long)splatCount * AdamSlots);
        _adamV = accel.Allocate1D<float>((long)splatCount * AdamSlots);
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
        _counter!.MemSetToZero();
        _ranges!.MemSetToZero();
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
    public void SetTargetFrom(MemoryBuffer1D<float, Stride1D.Dense> stack, int viewIndex)
    {
        long len = (long)_width * _height * 3;
        long off = (long)viewIndex * len;
        if (off + len > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} needs floats [{off},{off + len}) of a {stack.Length}-float stack");
        // A kernel, not ArrayView.CopyTo: the WebGPU backend has no synchronous device-to-device
        // copy ("Synchronous GPU to CPU copies are not supported"), and this stays on the GPU.
        var accel = _gpu.WebGPUAccelerator;
        _copyKernel ??= accel.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(CopyKernel);
        _copyKernel((int)len, stack.View.SubView(off, len), _target!.View);
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
    /// Gradient coverage over every splat, in fixed-point QUANTA.
    ///
    /// Quanta rather than values because an i32 atomic WRAPS rather than clamping, so how close
    /// a gradient sits to its ceiling is the question that matters, and the ceiling is an
    /// integer one.
    /// </summary>
    public readonly record struct GradientStats(
        long Splats, long ColourLive, long CentreLive, long ConicLive,
        double MeanCentreQuanta, double MaxCentreQuanta, double MaxConicQuanta,
        double MaxColourQuanta)
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
        MemoryBuffer1D<float, Stride1D.Dense> stack, int viewIndex)
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
        MemoryBuffer1D<float, Stride1D.Dense> stack, int viewIndex, bool withSsim = true)
    {
        var accel = _gpu.WebGPUAccelerator;
        long frameFloats = (long)_width * _height * 3;
        uint targetOffset = (uint)(viewIndex * frameFloats);

        WriteU32x2(_dimsBuf!, (uint)(_width * _height), targetOffset);
        Dispatch(_evalSse!, SseWorkgroups, 1, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, stack.GetGPUBuffer()!),
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
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, stack.GetGPUBuffer()!),
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
        double mse = sse / frameFloats;
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
        MemoryBuffer1D<float, Stride1D.Dense> stack, int viewIndex, Uint8Array rgba)
    {
        long frameFloats = (long)_width * _height * 3;
        long off = (long)viewIndex * frameFloats;
        if (off + frameFloats > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} does not fit a {stack.Length}-float stack");

        _queue!.WriteBuffer(_targetBytes!, 0, rgba);
        WriteU32x2(_dimsBuf!, (uint)(_width * _height), (uint)off);
        Dispatch(_unpackTarget!, (_width * _height + 63) / 64, 1, new[]
        {
            Buf(0, _targetBytes!), Buf(1, stack.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
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
        _adamStepCount = 0;
    }

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
        if (LastKeyCount == 0) return 0f;

        int pixels = _width * _height;

        // ── Loss and dL/d(pixel) ──
        _lossFixed!.MemSetToZero();
        await accel.SynchronizeAsync();
        WriteU32(_dimsBuf!, (uint)pixels);
        Dispatch(_lossL1!, (pixels + 63) / 64, 1, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
            Buf(2, _dLdPix!.GetGPUBuffer()!), Buf(3, _lossFixed!.GetGPUBuffer()!), Buf(4, _dimsBuf!),
        });

        // ── Backward: one workgroup per tile, no atomics ──
        // grad_per_key is written for every key this frame, so stale values cannot leak in.
        Dispatch(_rasterBackward!, _tilesX, _tilesY, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _outFinalT!.GetGPUBuffer()!),
            Buf(5, _outEnd!.GetGPUBuffer()!), Buf(6, _dLdPix!.GetGPUBuffer()!),
            Buf(7, _gradKeyA!.GetGPUBuffer()!), Buf(8, _gradKeyB!.GetGPUBuffer()!),
            Buf(9, _gradKeyC!.GetGPUBuffer()!),
        });

        // ── Scatter per-key gradients into per-splat totals ──
        _gradFixed!.MemSetToZero();
        await accel.SynchronizeAsync();
        WriteU32(_countBuf!, (uint)LastKeyCount);
        DispatchLinear(_scatterGrad!, LastKeyCount, new[]
        {
            Buf(0, _gradKeyA!.GetGPUBuffer()!), Buf(1, _gradKeyB!.GetGPUBuffer()!),
            Buf(2, _gradKeyC!.GetGPUBuffer()!), Buf(3, _values!.GetGPUBuffer()!),
            Buf(4, _gradFixed!.GetGPUBuffer()!), Buf(5, _countBuf!), Buf(6, _gradScaleBuf!),
        });

        // ── Adam ──
        _adamStepCount++;
        WriteVec4(_adamCfgBuf!, colourLr, opacityLr, _adamStepCount, splatCount);
        WriteVec4(_adamFlagsBuf!, SkipZeroGradientSteps ? 1f : 0f, 0f, 0f, 0f);
        // One buffer for writer and readers, written before the scatter, so a step cannot
        // accumulate at one scale and divide by another.
        WriteVec4(_gradScaleBuf!, ColourScale, CentreScale, ConicScale, 0f);
        Dispatch(_adamStep!, (splatCount + 63) / 64, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _gradFixed!.GetGPUBuffer()!),
            Buf(2, _opacityLogit!.GetGPUBuffer()!), Buf(3, _adamM!.GetGPUBuffer()!),
            Buf(4, _adamV!.GetGPUBuffer()!), Buf(5, _adamCfgBuf!), Buf(6, _adamFlagsBuf!),
            Buf(7, _gradScaleBuf!),
        });

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
                Buf(7, _geomOut!.GetGPUBuffer()!), Buf(8, _gradScaleBuf!),
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
    /// Per-splat accumulated gradients, dequantised. Nine per splat, in the shader's order:
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
        int[] raw = await _gradFixed!.CopyToHostAsync<int>(0, (long)splatCount * GradsPerSplat);
        var outp = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++) outp[i] = raw[i] / FixedScaleFor(i % GradsPerSplat);
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
        f[27] = 0f;

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
        _opacityLogit?.Dispose(); _opacityLogit = null;
        _logScale?.Dispose(); _logScale = null;
        _geomOut?.Dispose(); _geomOut = null;
        _ssePartials?.Dispose(); _ssePartials = null;
        _ssimRows?.Dispose(); _ssimRows = null;
        _ssimPartials?.Dispose(); _ssimPartials = null;
        _gradStatsPartials?.Dispose(); _gradStatsPartials = null;
        _adamM?.Dispose(); _adamM = null;
        _adamV?.Dispose(); _adamV = null;
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
        _gradScaleBuf?.Destroy(); _gradScaleBuf?.Dispose();
        _ssimDimsBuf?.Destroy(); _ssimDimsBuf?.Dispose();
        _ssimCfgBuf?.Destroy(); _ssimCfgBuf?.Dispose();
        _scratch4?.Dispose();
        _emitKeys?.Dispose();
        _tileRanges?.Dispose();
        _rasterForward?.Dispose();
        _rasterBackward?.Dispose();
        _scatterGrad?.Dispose();
        _lossL1?.Dispose();
        _adamStep?.Dispose();
        _initLogits?.Dispose();
    }
}
