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

        Console.WriteLine("[Trainer] pipelines created (emit_keys, tile_ranges, raster_forward)");
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
        CameraParams cam, float depthNear, float depthFar)
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
            return new float[_width * _height * 3];
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

        // CPU transfer: gate comparison only. Training keeps this on the GPU.
        return await _outColour!.CopyToHostAsync<float>(0, _width * _height * 3);
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
    }

    public void Dispose()
    {
        DisposeBuffers();
        _uniformBuf?.Destroy(); _uniformBuf?.Dispose();
        _capsBuf?.Destroy(); _capsBuf?.Dispose();
        _countBuf?.Destroy(); _countBuf?.Dispose();
        _scratch4?.Dispose();
        _emitKeys?.Dispose();
        _tileRanges?.Dispose();
        _rasterForward?.Dispose();
    }
}
