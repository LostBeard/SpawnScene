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
    MemoryBuffer1D<float, Stride1D.Dense>? _gradPerKey;  // 4 per key
    MemoryBuffer1D<int, Stride1D.Dense>? _gradFixed;     // 4 per splat, fixed point
    MemoryBuffer1D<float, Stride1D.Dense>? _opacityLogit;
    MemoryBuffer1D<float, Stride1D.Dense>? _logScale;    // 3 per splat
    MemoryBuffer1D<float, Stride1D.Dense>? _geomOut;     // 10 per splat, diagnostics + gate
    MemoryBuffer1D<float, Stride1D.Dense>? _adamM;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamV;
    MemoryBuffer1D<int, Stride1D.Dense>? _lossFixed;
    GPUBuffer? _dimsBuf;
    GPUBuffer? _adamCfgBuf;
    GPUBuffer? _geomCfgBuf;
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
    public static float FixedScaleFor(int slot) => slot < 6 ? 67108864f : 1048576f;

    RadixSortPairs<uint, Stride1D.Dense, uint, Stride1D.Dense>? _sortPairs;

    int _width, _height, _tilesX, _tilesY, _keyCapacity;

    /// <summary>Keys emitted by the last render. Exceeding capacity is reported, never silent.</summary>
    public int LastKeyCount { get; private set; }

    /// <summary>True when the last render overflowed the key buffer and is therefore incomplete.</summary>
    public bool LastOverflowed { get; private set; }

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

        _geomCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 32,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        Console.WriteLine("[Trainer] pipelines created: emit_keys, tile_ranges, raster_forward, " +
            "raster_backward, scatter_gradients, loss_l1, adam_step, init_logits, adam_geometry");
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

        _keyCapacity = Math.Max(1024, splatCount * keysPerSplat);

        DisposeBuffers();
        _keys = accel.Allocate1D<uint>(_keyCapacity);
        _values = accel.Allocate1D<uint>(_keyCapacity);
        _counter = accel.Allocate1D<int>(1);
        _ranges = accel.Allocate1D<uint>(tileCount * 2);
        _outColour = accel.Allocate1D<float>((long)width * height * 3);
        _outFinalT = accel.Allocate1D<float>((long)width * height);
        _outEnd = accel.Allocate1D<uint>((long)width * height);

        _target = accel.Allocate1D<float>((long)width * height * 3);
        _dLdPix = accel.Allocate1D<float>((long)width * height * 3);
        _gradPerKey = accel.Allocate1D<float>((long)_keyCapacity * GradsPerSplat);
        _gradFixed = accel.Allocate1D<int>((long)splatCount * GradsPerSplat);
        _opacityLogit = accel.Allocate1D<float>(splatCount);
        _logScale = accel.Allocate1D<float>((long)splatCount * 3);
        _geomOut = accel.Allocate1D<float>((long)splatCount * GeomGradsPerSplat);
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
            pass.DispatchWorkgroups((uint)((splatCount + 63) / 64), 1, 1);
            pass.End();
            using var cmd = enc.Finish();
            _queue!.Submit(new[] { cmd });
        }
        await accel.SynchronizeAsync();

        // 4 bytes back to learn how many keys exist. A scalar, not bulk data.
        int[] counted = await _counter.CopyToHostAsync<int>(0, 1);
        int keyCount = counted[0];
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
            pass.DispatchWorkgroups((uint)((LastKeyCount + 63) / 64), 1, 1);
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

    /// <summary>The viewport the trainer is currently sized for.</summary>
    public (int Width, int Height) Size => (_width, _height);

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
            Buf(7, _gradPerKey!.GetGPUBuffer()!),
        });

        // ── Scatter per-key gradients into per-splat totals ──
        _gradFixed!.MemSetToZero();
        await accel.SynchronizeAsync();
        WriteU32(_countBuf!, (uint)LastKeyCount);
        Dispatch(_scatterGrad!, (LastKeyCount + 63) / 64, 1, new[]
        {
            Buf(0, _gradPerKey!.GetGPUBuffer()!), Buf(1, _values!.GetGPUBuffer()!),
            Buf(2, _gradFixed!.GetGPUBuffer()!), Buf(3, _countBuf!),
        });

        // ── Adam ──
        _adamStepCount++;
        WriteVec4(_adamCfgBuf!, colourLr, opacityLr, _adamStepCount, splatCount);
        Dispatch(_adamStep!, (splatCount + 63) / 64, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _gradFixed!.GetGPUBuffer()!),
            Buf(2, _opacityLogit!.GetGPUBuffer()!), Buf(3, _adamM!.GetGPUBuffer()!),
            Buf(4, _adamV!.GetGPUBuffer()!), Buf(5, _adamCfgBuf!),
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
    /// Per-splat accumulated gradients, dequantised. Nine per splat, in the shader's order:
    /// colour RGB, opacity, screen centre x and y, conic a, b and c. For the GPU gate only.
    /// </summary>
    public async Task<float[]> ReadGradientsAsync(int splatCount)
    {
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
        _gradPerKey?.Dispose(); _gradPerKey = null;
        _gradFixed?.Dispose(); _gradFixed = null;
        _opacityLogit?.Dispose(); _opacityLogit = null;
        _logScale?.Dispose(); _logScale = null;
        _geomOut?.Dispose(); _geomOut = null;
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
        _countBuf?.Destroy(); _countBuf?.Dispose();
        _dimsBuf?.Destroy(); _dimsBuf?.Dispose();
        _adamCfgBuf?.Destroy(); _adamCfgBuf?.Dispose();
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
