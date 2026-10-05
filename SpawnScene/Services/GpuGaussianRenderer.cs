using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>Controls whether adaptive half-resolution mode is applied.</summary>
public enum AdaptiveResMode
{
    Auto,       // velocity-gated: enters half-res above LowResEnterVelocity
    ForceFull,  // always render at full physical resolution
    ForceHalf,  // always render at half physical resolution
}

/// <summary>
/// Native WebGPU Gaussian splat renderer with GPU-sorted splats.
/// Architecture:
///   - ILGPU = data generation + sorting (GpuSplatSorter)
///   - WebGPU = rendering only (this class)
///   - GPU-side buffer copy (no CPU round-trips)
///   - EWA anti-alias filter for distant splats
///   - CAS post-processing sharpening
///   - Adaptive resolution: halves canvas pixel dims during fast movement, restores when still
/// </summary>
public class GpuGaussianRenderer : IDisposable
{
    private readonly GpuService _gpu;
    private readonly GpuSplatSorter _sorter;
    private readonly DepthToGaussianKernel _gaussianKernel; // owns the packed-layout GPU kernels

    // WebGPU objects
    private GPUDevice? _device;
    private GPUQueue? _queue;
    private GPUCanvasContext? _context;
    private GPURenderPipeline? _splatPipeline;
    private GPURenderPipeline? _casPipeline;
    private string _canvasFormat = "bgra8unorm";

    // Sorted mode blends into this, not the canvas. Blending straight into an 8-bit target rounds after
    // every splat, and a faint splat that moves a pixel by less than half a step moves it by NOTHING - a
    // trained scene is full of those, and the trainer composites in f32. MEASURED 2026-09-24 on Truck
    // 7K/1.2M: the viewer scored 0.7-3.6 dB under the trainer on the same views, growing with splat count.
    private const string SortedTargetFormat = "rgba16float";

    /// <summary>
    /// The stochastic accumulation texture's format. It was the canvas format (bgra8unorm): a running average at weight
    /// 1/n stops moving in 8 bits once 1/n is below a quantisation step, so a still camera froze on an early, under-
    /// converged average - a regular crosshatch at the splat grid, darker than the scene (MEASURED 2026-10-02, the single-
    /// photo Room sample: 320 samples accumulated, still crosshatched). Half floats keep averaging.
    /// </summary>
    private const string AccumTargetFormat = "rgba16float";

    // Gaussian vertex buffer: packed format (position f32x3 + color_alpha u8x4 + scale f16x4 + quat f16x4)
    private GPUBuffer? _splatBuffer;
    private int _splatCount;
    private const int PackedBytesPerSplat = SplatFormat.PackedBytes; // 12 pos + 8 color/alpha f16 + 12 scale + 16 quat f32

    // Pack compute pipeline: converts Float32 sort output → packed vertex format
    private GPUComputePipeline? _packPipeline;
    private GPUBindGroup? _packBindGroup;

    // LOD cut, drawn indirect: a one-thread pass turns the cull's drawn count into the pack dispatch (u32 0..2) and the
    // splat draw (u32 4..7) arguments, so frame cost follows the cut, not the tree's size. Valid after a cut was packed.
    private GPUComputePipeline? _lodArgsPipeline;
    private GPUBindGroup? _lodArgsBindGroup;
    private GPUBuffer? _lodArgsBuf;
    private GPUBuffer? _lodArgsCountCached;
    private bool _lodArgsValid;
    private GPUBuffer? _srcDataCached; // cached ILGPU data buffer handle for bind group invalidation
    private GPUBuffer? _srcIdxCached;  // cached ILGPU index buffer handle for bind group invalidation

    // View-dependent colour. The trainer learns SH bands 1..3 per splat; the viewer used to draw DC only,
    // i.e. one colour per splat from every angle. Owned here (a GPU copy handed over after training).
    // SphericalHarmonics.Parts buffers of PartFloatsPerSplat floats a splat (one 45-float buffer capped a binding at
    // 11.9M splats).
    private GPUBuffer[]? _shRest;
    private GPUBuffer? _shRestCached;
    private GPUBuffer? _shDummy;
    private GPUComputePipeline? _shSplitPipeline;
    private int _shDegree;
    private System.Numerics.Vector3 _packCameraPos;
    private long _lastShRepack;

    /// <summary>
    /// Give the viewer the scene's SH rest coefficients (SphericalHarmonics.Parts buffers of PartFloatsPerSplat floats
    /// a splat) and the active degree; the renderer takes ownership. Null / degree 0 = DC colour only.
    /// </summary>
    public void SetShRest(GPUBuffer[]? parts, int degree)
    {
        if (!ReferenceEquals(_shRest, parts)) DestroyShRest();
        _shRest = parts;
        _shDegree = parts == null ? 0 : Math.Clamp(degree, 0, SphericalHarmonics.MaxDegree);
        _packBindGroup?.Dispose();
        _packBindGroup = null;
    }

    void DestroyShRest()
    {
        if (_shRest == null) return;
        foreach (var b in _shRest) { b.Destroy(); b.Dispose(); }
        _shRest = null;
    }

    GPUBuffer NewShPart(ulong bytes) => _device!.CreateBuffer(new GPUBufferDescriptor
    {
        Size = Math.Max(16UL, bytes),
        // CopySrc: ReadShRestPartsAsync (the autotest's legacy-load check) reads them back.
        Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst | GPUBufferUsage.CopySrc,
    });

    /// <summary>The SH degree the viewer is drawing with (0 = DC only).</summary>
    public int ShDegree => _shDegree;

    /// <summary>The SH part buffers the viewer owns (PartFloatsPerSplat floats a splat each), or null.</summary>
    public GPUBuffer[]? ShRestBuffers => _shRest;

    /// <summary>SH part <paramref name="part"/>'s first <paramref name="splatCount"/> rows copied into a new ILGPU buffer
    /// (scene editing gathers and grows them with kernels). The caller owns it.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> CopyShPartToIlgpu(Accelerator a, int part, int splatCount)
    {
        long floats = (long)splatCount * SphericalHarmonics.PartFloatsPerSplat;
        var dst = a.Allocate1D<float>(floats);
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var encoder = _device!.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(_shRest![part], 0, dst.GetGPUBuffer()!, 0, (ulong)floats * sizeof(float));
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, _submitArray);
        return dst;
    }

    /// <summary>A new renderer-owned SH part buffer holding <paramref name="splatCount"/> rows copied from an ILGPU
    /// buffer (for <see cref="SetShRest"/>).</summary>
    public GPUBuffer NewShPartFrom(MemoryBuffer1D<float, Stride1D.Dense> src, int splatCount)
    {
        ulong bytes = (ulong)splatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
        var part = NewShPart(bytes);
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var encoder = _device!.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(src.GetGPUBuffer()!, 0, part, 0, bytes);
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, _submitArray);
        return part;
    }

    /// <summary>
    /// The next scene's colours are plain RGB with no SH bands - every scene load that is NOT a trained scene
    /// must say so. Training sets <see cref="ColoursAreShDc"/> and nothing reset it, so any scene loaded after a
    /// training run in the same tab had its RGB colours drawn as SH DC coefficients.
    /// </summary>
    public void UseRgbColours()
    {
        ColoursAreShDc = false;
        SetShRest(null, 0);
    }

    /// <summary>
    /// Load a trained scene's SH rest parts (one ArrayBuffer per SphericalHarmonics part, as saved from the trainer)
    /// from JS memory straight into GPU buffers - the bytes never enter the .NET heap.
    /// </summary>
    public void LoadShRest(ArrayBuffer[] parts, int degree)
    {
        if (_device == null || _queue == null) return;
        if (degree <= 0 || parts.Length != SphericalHarmonics.Parts || parts.Any(p => p.ByteLength <= 0))
        {
            SetShRest(null, 0);
            return;
        }
        var buffers = new GPUBuffer[parts.Length];
        for (int part = 0; part < parts.Length; part++)
        {
            buffers[part] = NewShPart((ulong)parts[part].ByteLength);
            _queue.WriteBuffer(buffers[part], 0L, parts[part]);
        }
        SetShRest(buffers, degree);
    }

    /// <summary>
    /// <see cref="LoadShRest"/> for a scene made of blocks: each block's parts (one ArrayBuffer per SphericalHarmonics
    /// part) written one after another into the same part buffers, in block order - the order the blocks' packed rows
    /// were merged in. JS memory straight to the GPU.
    /// </summary>
    public void LoadShRestBlocks(IReadOnlyList<ArrayBuffer[]> blocks, int degree)
    {
        if (_device == null || _queue == null) return;
        if (degree <= 0 || blocks.Count == 0 || blocks.Any(b => b.Length != SphericalHarmonics.Parts))
        {
            SetShRest(null, 0);
            return;
        }
        var buffers = new GPUBuffer[SphericalHarmonics.Parts];
        for (int part = 0; part < buffers.Length; part++)
        {
            long total = blocks.Sum(b => (long)b[part].ByteLength);
            buffers[part] = NewShPart((ulong)Math.Max(4L, total));
            long offset = 0;
            foreach (var b in blocks)
            {
                if (b[part].ByteLength > 0) _queue.WriteBuffer(buffers[part], offset, b[part]);
                offset += (long)b[part].ByteLength;
            }
        }
        SetShRest(buffers, degree);
    }

    /// <summary>
    /// Show the scene through its LOD tree (Plans/lod-streaming.md): the tree's nodes become the drawn rows (the sorter
    /// takes ownership of them and of the cut arrays), each frame draws only the cut at <paramref name="tau"/> pixels,
    /// and the SH bands grow to the node count - the leaves keep theirs, internal nodes get none (view-independent
    /// colour for merged distant detail; WebGPU buffers start zeroed). The cut runs in the sorted path's cull, so LOD
    /// draws sorted: stochastic mode would draw every node at once.
    /// </summary>
    public async Task InstallLodAsync(GpuLodTree tree, float tau)
    {
        var oldSh = _shRest;
        int degree = _shDegree;
        _shRest = null;   // kept alive across the upload, re-sized below
        var nodes = tree.Nodes;
        tree.Nodes = null!;
        RenderMode = SplatRenderMode.Sorted;
        await UploadSceneFromGpuBuffer(nodes, tree.NodeCount);
        _sorter.SetLod(tree.Parent, tree.Bounds, tree.LodSize, tree.LeafCount);
        tree.Parent = null!; tree.Bounds = null!; tree.LodSize = null!;
        _sorter.LodTau = tau;
        if (oldSh != null && degree > 0 && _device != null && _queue != null)
        {
            ulong leafBytes = (ulong)tree.LeafCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
            ulong nodeBytes = (ulong)tree.NodeCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
            var parts = new GPUBuffer[oldSh.Length];
            _gpu.WebGPUAccelerator.FlushPendingCommands();
            using (var encoder = _device.CreateCommandEncoder())
            {
                for (int p = 0; p < parts.Length; p++)
                {
                    parts[p] = NewShPart(nodeBytes);
                    encoder.CopyBufferToBuffer(oldSh[p], 0, parts[p], 0, leafBytes);
                }
                using var cmd = encoder.Finish();
                _submitArray[0] = cmd;
                RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
            }
            foreach (var b in oldSh) { b.Destroy(); b.Dispose(); }
            SetShRest(parts, degree);
        }
        else
        {
            if (oldSh != null) foreach (var b in oldSh) { b.Destroy(); b.Dispose(); }
            SetShRest(null, 0);
        }
        RepackForDisplay();
        Console.WriteLine($"[LOD] drawing {tree.LeafCount:N0} splats through a {tree.NodeCount:N0}-node tree " +
            $"({tree.Levels} levels), cut at {tau} px, sorted mode");
    }

    /// <summary>
    /// Show an LOD tree already laid out breadth-first (LodLayout: leaves anywhere, told apart by LOD size 0) - a
    /// .spawnscene v3 file. Takes the tree's nodes and cut arrays like <see cref="InstallLodAsync"/>; the SH bands come
    /// node-ordered in <paramref name="nodeSh"/> (copied on the GPU, the caller keeps them).
    /// </summary>
    public async Task InstallLaidLodAsync(GpuLodTree laid, MemoryBuffer1D<float, Stride1D.Dense>[]? nodeSh, int shDegree, float tau)
    {
        var nodes = laid.Nodes;
        laid.Nodes = null!;
        RenderMode = SplatRenderMode.Sorted;
        await UploadSceneFromGpuBuffer(nodes, laid.NodeCount);
        _sorter.SetLod(laid.Parent, laid.Bounds, laid.LodSize, 0);
        laid.Parent = null!; laid.Bounds = null!; laid.LodSize = null!;
        _sorter.LodTau = tau;
        if (nodeSh is { Length: SphericalHarmonics.Parts } && shDegree > 0 && _device != null && _queue != null)
        {
            ulong nodeBytes = (ulong)laid.NodeCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
            var parts = new GPUBuffer[nodeSh.Length];
            _gpu.WebGPUAccelerator.FlushPendingCommands();
            using (var encoder = _device.CreateCommandEncoder())
            {
                for (int p = 0; p < parts.Length; p++)
                {
                    parts[p] = NewShPart(nodeBytes);
                    encoder.CopyBufferToBuffer(nodeSh[p].GetGPUBuffer()!, 0, parts[p], 0, nodeBytes);
                }
                using var cmd = encoder.Finish();
                _submitArray[0] = cmd;
                RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
            }
            SetShRest(parts, shDegree);
        }
        else SetShRest(null, 0);
        RepackForDisplay();
        Console.WriteLine($"[LOD] drawing {laid.LeafCount:N0} splats through a {laid.NodeCount:N0}-node laid-out tree, " +
            $"cut at {tau} px, sorted mode");
    }

    /// <summary>
    /// Show a STREAMED LOD tree: <paramref name="poolRows"/> (taken) is a pool of <paramref name="poolNodes"/> slots a
    /// GpuLodPager fills page by page; SH bands get zeroed pool-sized parts the pager writes into
    /// (<see cref="WriteShRows"/>). The paged cut's arrays stay the pager's (GpuSplatSorter.SetLodPaged).
    /// </summary>
    public async Task InstallPagedLodAsync(MemoryBuffer1D<float, Stride1D.Dense> poolRows, int poolNodes, int shDegree,
        MemoryBuffer1D<int, Stride1D.Dense> parentSlot, MemoryBuffer1D<float, Stride1D.Dense> bounds,
        MemoryBuffer1D<float, Stride1D.Dense> size, MemoryBuffer1D<int, Stride1D.Dense> childChunk,
        MemoryBuffer1D<int, Stride1D.Dense> chunkPage, MemoryBuffer1D<int, Stride1D.Dense> want, int chunkCount, float tau)
    {
        RenderMode = SplatRenderMode.Sorted;
        await UploadSceneFromGpuBuffer(poolRows, poolNodes);
        _sorter.SetLodPaged(parentSlot, bounds, size, childChunk, chunkPage, want, chunkCount);
        _sorter.LodTau = tau;
        if (shDegree > 0 && _device != null)
        {
            ulong bytes = (ulong)poolNodes * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
            SetShRest(Enumerable.Range(0, SphericalHarmonics.Parts).Select(_ => NewShPart(bytes)).ToArray(), shDegree);
        }
        else SetShRest(null, 0);
        Console.WriteLine($"[LOD] streaming through a {poolNodes:N0}-slot pool, cut at {tau} px, sorted mode");
    }

    /// <summary>Copy <paramref name="rows"/> SH rows of part <paramref name="part"/> from an ILGPU buffer to pool row <paramref name="dstRow"/>.</summary>
    public void WriteShRows(int part, MemoryBuffer1D<float, Stride1D.Dense> src, long dstRow, int rows)
    {
        if (_shRest == null || _device == null || _queue == null) return;
        ulong rowBytes = SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var encoder = _device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(src.GetGPUBuffer()!, 0, _shRest[part], (ulong)dstRow * rowBytes, (ulong)rows * rowBytes);
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    /// <summary>Cut and sort again on the next frame even with a still camera (the drawn rows changed).</summary>
    public void RequestResort() => _sorter.RequestResort();

    /// <summary>The streamed tree's cut wants these chunks loaded (GpuSplatSorter.LodChunksWanted).</summary>
    public event Action<int[]>? LodChunksWanted
    {
        add => _sorter.LodChunksWanted += value;
        remove => _sorter.LodChunksWanted -= value;
    }

    /// <summary>
    /// Copy <paramref name="bytes"/> bytes of <paramref name="src"/> from <paramref name="srcOffset"/> into an ILGPU
    /// buffer at byte <paramref name="dstOffset"/> (file I/O: a decoded file part straight to its place on the GPU).
    /// </summary>
    public void WriteIlgpu(MemoryBuffer dst, long dstOffset, ArrayBuffer src, int srcOffset, long bytes)
    {
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        _queue!.WriteBuffer(dst.GetGPUBuffer()!, dstOffset, src, srcOffset, bytes);
    }

    /// <summary>The LOD cut threshold in pixels (when a tree is installed).</summary>
    public float LodTau { get => _sorter.LodTau; set => _sorter.LodTau = value; }

    /// <summary>Splats a frame the LOD cut aims for (0 = a fixed <see cref="LodTau"/>).</summary>
    public int LodBudget { get => _sorter.LodBudget; set => _sorter.LodBudget = value; }

    /// <summary>Nodes the last LOD cut drew, -1 before the first count.</summary>
    public int LodDrawn => _sorter.LodDrawn;

    /// <summary>JS bytes into an ILGPU buffer at a byte offset (a parked block's rows into a merged scene).</summary>
    public void WriteIlgpuBytes(MemoryBuffer1D<float, Stride1D.Dense> dst, long byteOffset, ArrayBuffer bytes)
    {
        if (bytes.ByteLength <= 0) return;
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        _queue!.WriteBuffer(dst.GetGPUBuffer()!, byteOffset, bytes);
    }

    /// <summary>
    /// Load SH saved before the part split (one file of RestFloatsPerSplat floats a splat, row-major): upload it and
    /// split it into the part buffers ON THE GPU. Such scenes are at most 11.9M splats - the old one-binding limit -
    /// so the row buffer still fits one binding.
    /// </summary>
    public void LoadShRestRows(ArrayBuffer rows, int degree)
    {
        if (_device == null || _queue == null) return;
        long bytes = (long)rows.ByteLength;
        if (bytes <= 0 || degree <= 0) { SetShRest(null, 0); return; }
        uint splats = (uint)(bytes / (SphericalHarmonics.RestFloatsPerSplat * sizeof(float)));
        using var src = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)bytes,
            Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
        });
        _queue.WriteBuffer(src, 0L, rows);
        var parts = new GPUBuffer[SphericalHarmonics.Parts];
        for (int part = 0; part < parts.Length; part++)
            parts[part] = NewShPart((ulong)splats * SphericalHarmonics.PartFloatsPerSplat * sizeof(float));
        if (_shSplitPipeline == null)
        {
            using var module = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = ShSplitWgsl });
            _shSplitPipeline = _device.CreateComputePipeline(new GPUComputePipelineDescriptor
            {
                Layout = "auto",
                Compute = new GPUProgrammableStage { Module = module, EntryPoint = "split_sh_rows" },
            });
        }
        using var cfg = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _queue.WriteBuffer(cfg, 0, new byte[] { (byte)splats, (byte)(splats >> 8), (byte)(splats >> 16), (byte)(splats >> 24), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        using var layout = _shSplitPipeline.GetBindGroupLayout(0);
        using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = layout,
            Entries = new GPUBindGroupEntry[]
            {
                new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = src } },
                new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = parts[0] } },
                new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = parts[1] } },
                new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = parts[2] } },
                new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = cfg } },
            },
        });
        using var encoder = _device.CreateCommandEncoder();
        using (var pass = encoder.BeginComputePass())
        {
            pass.SetPipeline(_shSplitPipeline);
            pass.SetBindGroup(0, bg);
            uint groups = (splats + 63) / 64;
            const uint maxWG = 65535u;
            pass.DispatchWorkgroups(Math.Min(groups, maxWG), (groups + maxWG - 1) / maxWG, 1);
            pass.End();
        }
        using var cmd = encoder.Finish();
        _queue.Submit(new[] { cmd });
        src.Destroy();
        cfg.Destroy();
        SetShRest(parts, degree);
    }

    /// <summary>
    /// The SH part buffers the viewer is drawing with, read back (autotest / diagnostics only: CPU transfer of
    /// ~60 bytes a splat a part). Null when it has none.
    /// </summary>
    public async Task<float[][]?> ReadShRestPartsAsync()
    {
        if (_shRest == null || _device == null || _queue == null) return null;
        var parts = new float[_shRest.Length][];
        for (int part = 0; part < parts.Length; part++)
        {
            ulong bytes = _shRest[part].Size;
            using var staging = _device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = bytes,
                Usage = GPUBufferUsage.MapRead | GPUBufferUsage.CopyDst,
            });
            using (var encoder = _device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(_shRest[part], 0, staging, 0, bytes);
                using var cmd = encoder.Finish();
                _queue.Submit(new[] { cmd });
            }
            await staging.MapAsync(GPUMapMode.Read, 0, (long)bytes);
            using (var mapped = staging.GetMappedRange())
            using (var floats = new Float32Array(mapped))
                parts[part] = floats.ToArray();
            staging.Unmap();
            staging.Destroy();
        }
        return parts;
    }

    /// <summary>
    /// The SH part buffers the viewer draws with, as JS Uint8Arrays (each a JS-side copy of the mapped range: the
    /// bytes never enter the .NET heap - a 14M-splat scene's bands are ~2.5 GB). For saving a scene that was loaded,
    /// not trained here (edited, then saved). Null when the viewer has no SH bands.
    /// </summary>
    public async Task<Uint8Array[]?> ReadShRestUint8ArraysAsync()
    {
        if (_shRest == null || _device == null || _queue == null) return null;
        var parts = new Uint8Array[_shRest.Length];
        for (int part = 0; part < parts.Length; part++)
        {
            ulong bytes = _shRest[part].Size;
            using var staging = _device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = bytes,
                Usage = GPUBufferUsage.MapRead | GPUBufferUsage.CopyDst,
            });
            using (var encoder = _device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(_shRest[part], 0, staging, 0, bytes);
                using var cmd = encoder.Finish();
                _queue.Submit(new[] { cmd });
            }
            await staging.MapAsync(GPUMapMode.Read, 0, (long)bytes);
            using (var mapped = staging.GetMappedRange())
            using (var view = new Uint8Array(mapped))
                parts[part] = new Uint8Array(view);   // copy: unmapping detaches the mapped range
            staging.Unmap();
            staging.Destroy();
        }
        return parts;
    }

    const string ShSplitWgsl = @"
@group(0) @binding(0) var<storage, read>       rows  : array<f32>;   // 45 floats a splat
@group(0) @binding(1) var<storage, read_write> part0 : array<f32>;   // 15 a splat each
@group(0) @binding(2) var<storage, read_write> part1 : array<f32>;
@group(0) @binding(3) var<storage, read_write> part2 : array<f32>;
@group(0) @binding(4) var<uniform>             cfg   : vec4<u32>;    // x = splat count

@compute @workgroup_size(64)
fn split_sh_rows(@builtin(workgroup_id) wg : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>,
                 @builtin(local_invocation_index) li : u32) {
    let i = (wg.y * nwg.x + wg.x) * 64u + li;
    if (i >= cfg.x) { return; }
    for (var k = 0u; k < 15u; k = k + 1u) {
        part0[i * 15u + k] = rows[i * 45u + k];
        part1[i * 15u + k] = rows[i * 45u + 15u + k];
        part2[i * 15u + k] = rows[i * 45u + 30u + k];
    }
}
";

    /// <summary>Lower the SH degree in use (diagnostic A/B); takes effect at the next pack.</summary>
    public void CapShDegree(int degree)
    {
        _shDegree = Math.Clamp(Math.Min(_shDegree, degree), 0, SphericalHarmonics.MaxDegree);
        _packBindGroup?.Dispose();
        _packBindGroup = null;
    }
    private GPUBuffer? _packCountBuf;  // uniform: visible count for pack dispatch guard
    private Uint32Array? _packCountJsArray; // cached JS array for WriteBuffer (no per-frame alloc)

    // ── Splat uniform block ──
    // Named slots, because these indices are written from four call sites (sorted, stochastic and
    // two XR paths) and a silent off-by-one between them is invisible until something renders wrong.
    // Layout must match `struct Uniforms` in SplatShaderSource / StochasticSplatShaderSource.
    private const int UMvp = 0;         // mat4x4            [0..15]
    private const int UCamRight = 16;   // vec4 (xyz used)   [16..19]
    private const int UCamUp = 20;      // vec4              [20..23]
    private const int UCamFwd = 24;     // vec4              [24..27]
    private const int UCamPos = 28;     // vec4              [28..31]
    private const int UViewport = 32;   // vec2              [32..33]
    private const int UFocal = 34;      // vec2              [34..35]
    private const int UFrameIndex = 36; // u32 (bitcast)
    private const int UDilation = 37;   // f32
    private const int UMinAlpha = 38;   // f32
    private const int UniformFloats = 40; // + 1 pad, 160 bytes

    private GPUBuffer? _uniformBuffer;
    private GPUBindGroup? _uniformBindGroup;        // for stochastic pipeline
    private GPUBindGroup? _uniformBindGroupSorted;  // for sorted pipeline (separate auto-layout)
    private readonly float[] _uniformData = new float[UniformFloats];
    private byte[]? _uniformByteData; // pre-allocated byte mirror of _uniformData for direct WriteBuffer

    // CAS sharpening pass
    private GPUTexture? _offscreenTexture;
    private GPUTextureView? _offscreenView;
    private GPUBindGroup? _casBindGroup;           // for sorted mode (reads _offscreenTexture)
    private GPUBindGroup? _casBindGroupStochastic; // for stochastic mode (reads _accumTexture)
    private GPUBuffer? _casUniformBuffer;
    private GPUSampler? _casSampler;
    private float _sharpeningStrength = 0.5f;
    private readonly float[] _casData = new float[4];
    private byte[]? _casByteData; // pre-allocated byte mirror of _casData for direct WriteBuffer

    // Stochastic rasterization — sort-free rendering with temporal accumulation
    private GPURenderPipeline? _stochasticSplatPipeline;
    private GPUTexture? _stochasticTexture;      // per-frame stochastic render target (cleared each frame)
    private GPUTextureView? _stochasticView;
    private GPUTexture? _accumTexture;            // persistent accumulation texture (NOT cleared per frame)
    private GPUTextureView? _accumView;
    private GPURenderPipeline? _accumPipeline;    // fullscreen blend pass for temporal accumulation
    private GPUBindGroup? _accumBindGroup;        // samples _stochasticTexture
    private GPUBuffer? _accumUniformBuffer;       // accumulation weight uniform
    private readonly float[] _accumData = new float[4]; // weight + padding
    private byte[]? _accumByteData;
    private int _accumFrameCount;                 // velocity-adaptive: capped by movement speed
    private int _globalFrameCount;                // monotonically increasing, never resets (hash seed)

    // Velocity-adaptive dilation: subtle splat fattening to bridge sub-pixel spatial gaps
    private const float DilationScale = 5f;       // sqrt(velocity) * DilationScale
    private const float MaxDilationFactor = 0.05f; // max additional scale (0.05 = up to 5% larger)

    // Multi-SPP: render multiple stochastic passes per frame for faster convergence
    /// <summary>Max samples per pixel per frame (1-4). Higher = faster convergence, lower FPS. Default 2 is optimal for 60fps.</summary>
    public int StochasticSPP { get; set; } = 2;

    // Cached render pass descriptors for stochastic mode
    private GPURenderPassColorAttachment? _stochasticColorAttach;
    private GPURenderPassDescriptor? _stochasticPassDesc;
    private GPURenderPassColorAttachment? _accumColorAttach;
    private GPURenderPassDescriptor? _accumPassDesc;

    // Depth texture
    private GPUTexture? _depthTexture;
    private GPUTextureView? _depthView;

    private int _canvasWidth;
    private int _canvasHeight;

    // Adaptive resolution — physical dims track the true canvas pixel size.
    // _canvasWidth/_canvasHeight may be half of physical during fast movement.
    private ElementReference _canvasRef;
    private int _physicalWidth;
    private int _physicalHeight;
    private bool _lowResActive;
    // Thresholds calibrated for per-frame DistanceSquared (see GpuSplatSorter for scale reference).
    // Fast movement (aggressive mouse or Shift+WASD) pushes _smoothedVelocity above 0.0001.
    private const float LowResEnterVelocity = 0.0002f; // enter half-res when velocity exceeds this
    private const float LowResExitVelocity  = 0.00005f; // exit half-res once velocity drops below this (hysteresis)

    private bool _disposed;

    // XR bridge: WebGPU OffscreenCanvas used to pass rendered frames to WebGL XR
    private OffscreenCanvas? _xrBridgeCanvas;
    private GPUCanvasContext? _xrBridgeContext;
    private GPUTexture? _xrBridgeDepth;
    private GPUTexture? _xrBridgeStochasticTex; // intermediate texture when CAS is enabled
    private GPUTextureView? _xrBridgeStochasticView;
    private GPUBindGroup? _xrCasBindGroup;
    private int _xrBridgeWidth;
    private int _xrBridgeHeight;

    /// <summary>The OffscreenCanvas that WebGL reads from via texImage2D for XR blit.</summary>
    public OffscreenCanvas? XRBridgeCanvas => _xrBridgeCanvas;

    // Reused 1-element array for Submit (avoids per-frame allocation)
    private static readonly GPUCommandBuffer[] _submitArray = new GPUCommandBuffer[1];

    // Cached render pass descriptors — rebuilt on resize, reused every frame.
    // GPURenderPassColorAttachment.View is { get; set; } so we update it per frame for swapchain targets.
    // GPURenderPassDepthStencilAttachment.View is { get; init; } so we recreate on resize.
    private GPURenderPassColorAttachment? _splatColorAttachCas;    // View = _offscreenView (stable)
    private GPURenderPassColorAttachment? _splatColorAttachDirect; // View updated per frame
    private GPURenderPassColorAttachment? _casColorAttach;          // View updated per frame
    private GPURenderPassDescriptor? _splatPassDescCas;            // fully stable (CAS path)
    private GPURenderPassDescriptor? _splatPassDescDirect;          // color View updated per frame
    private GPURenderPassDescriptor? _casPassDesc;                  // color View updated per frame

    /// <summary>Sharpening intensity (0 = off, 1 = maximum).</summary>
    public float SharpeningStrength
    {
        get => _sharpeningStrength;
        set => _sharpeningStrength = Math.Clamp(value, 0f, 1f);
    }

    /// <summary>Controls adaptive resolution behavior.</summary>
    public AdaptiveResMode AdaptiveResMode { get; set; } = AdaptiveResMode.Auto;

    // ── Background ──
    // Was hardcoded at seven sites, two of which are attachment objects CACHED at texture
    // creation - so setting it has to refresh those or half the passes keep the old colour.
    // Needed beyond taste: scoring a render against a dataset shot on black is dominated by a
    // background mismatch (61% of a TempleRing frame is near-black), and SuperSplat exposes a
    // custom background too (NOTES.md parity list).
    //
    // Default BLACK, the background every splat scene is trained over (ours, and the reference trainer's
    // default). This was a dark navy (0.04, 0.04, 0.10), and wherever splats do not fully cover a pixel -
    // sky, foliage, the far background - the navy showed through as a blue tint the trainer never had.
    // MEASURED 2026-09-24 (trainer-vs-viewer dumps, Truck): a smooth purple/blue cast over the upper frame,
    // blue +9 levels on a held-out view, while LOD cull, blend precision, CAS and the far plane moved nothing.
    private double _bgR = 0.0, _bgG = 0.0, _bgB = 0.0;

    /// <summary>Scene clear colour, linear 0..1. Alpha is always 1.</summary>
    public (double R, double G, double B) BackgroundColor
    {
        get => (_bgR, _bgG, _bgB);
        set
        {
            (_bgR, _bgG, _bgB) = value;
            ApplyBackgroundToCachedAttachments();
        }
    }

    /// <summary>
    /// Push the background onto every CACHED colour attachment.
    ///
    /// There are four, and they are built at different times: the stochastic and accumulation
    /// ones when the textures are (re)created, the direct and CAS ones in
    /// <see cref="RebuildCachedDescriptors"/>. Missing any of them means the colour changes on
    /// some passes and not others - which is exactly what happened when only the two stochastic
    /// attachments were refreshed and the SORTED path kept rendering on the old blue.
    /// Anything added here must also be listed here.
    /// </summary>
    private void ApplyBackgroundToCachedAttachments()
    {
        if (_stochasticColorAttach != null) _stochasticColorAttach.ClearValue = NewClear();
        if (_accumColorAttach != null) _accumColorAttach.ClearValue = NewClear();
        if (_splatColorAttachDirect != null) _splatColorAttachDirect.ClearValue = NewClear();
        if (_splatColorAttachCas != null) _splatColorAttachCas.ClearValue = NewClear();
    }

    /// <summary>Fresh clear-colour POCO. Never share one instance across descriptors.</summary>
    private GPUColorDict NewClear() => new() { R = _bgR, G = _bgG, B = _bgB, A = 1.0 };

    /// <summary>Diagnostics (&amp;xrclear=1): clear XR eye views to magenta, so an empty view tells apart from a broken copy.</summary>
    public static bool XRDebugClear { get; set; }
    private GPUColorDict XRClear() => XRDebugClear ? new() { R = 1, G = 0, B = 1, A = 1 }
        : XRTransparent ? new() { R = 0, G = 0, B = 0, A = 0 } : NewClear();

    /// <summary>
    /// AR passthrough: XR eye views keep the splats' coverage as alpha (premultiplied) instead of drawing over the
    /// background colour, so the real world shows wherever the scene has nothing. The sorted target clears to
    /// transparent, back-to-front SrcAlpha/OneMinusSrcAlpha blending then leaves premultiplied colour + coverage, and the
    /// CAS pass passes that alpha through (CASUniforms.keep_alpha) to a premultiplied bridge canvas.
    /// </summary>
    public bool XRTransparent { get; set; }

    /// <summary>Controls whether to use sorted alpha blending or stochastic rasterization.</summary>
    private SplatRenderMode _renderMode = SplatRenderMode.Stochastic;
    public SplatRenderMode RenderMode
    {
        get => _renderMode;
        set
        {
            if (_renderMode == value) return;
            _renderMode = value;
            _accumFrameCount = 0; // reset accumulation on mode switch
        }
    }

    /// <summary>Sort precision passthrough: true = 4-pass 16-bit (faster), false = 8-pass 32-bit.</summary>
    public bool Use16BitSort
    {
        get => _sorter.Use16BitSort;
        set => _sorter.Use16BitSort = value;
    }

    /// <summary>Screen-space LOD cull threshold in pixels (0 = draw every splat). See GpuSplatSorter.</summary>
    public float LodCullPixels
    {
        get => _sorter.LodCullPixels;
        set => _sorter.LodCullPixels = value;
    }

    /// <summary>Diagnostic: skip radix sort entirely (render unsorted).</summary>
    public bool SkipSort
    {
        get => _sorter.SkipSort;
        set => _sorter.SkipSort = value;
    }

    public GpuGaussianRenderer(GpuService gpuService, GpuSplatSorter sorter, DepthToGaussianKernel gaussianKernel)
    {
        _gpu = gpuService;
        _sorter = sorter;
        _gaussianKernel = gaussianKernel;
    }

    /// <summary>
    /// When true, packed colour slots hold SH DC (after
    /// <c>EnsureRgbConvertedToShDc</c>), not linear RGB. Pack must run
    /// <c>max(C0*dc+0.5,0)</c> before unorm8 or the viewer shows clamped DC as
    /// washed blobs (MEASURED: Truck looks recognisable but messy without this).
    /// </summary>
    public bool ColoursAreShDc { get; set; }

    /// <summary>Whether the GPU has a valid packed splat buffer ready to render.</summary>
    public bool HasGpuData => _splatCount > 0;

    /// <summary>
    /// The live packed splat buffer (SplatFormat.Floats per splat), for the optimiser to write
    /// through. Training mutates these values in place, so callers must
    /// <see cref="RepackForDisplay"/> afterwards or the viewer keeps showing the pre-training
    /// vertex data.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? PackedSplatBuffer => _sorter.PackedDataBuf;

    /// <summary>Splats currently uploaded.</summary>
    public int SplatCount => _splatCount;

    /// <summary>
    /// Rebuild the display vertex buffer from the packed splat data. Needed after anything
    /// mutates the splats behind the renderer's back - the optimiser does exactly that.
    /// </summary>
    public void RepackForDisplay(Vector3? cameraPosition = null)
    {
        if (cameraPosition.HasValue) _packCameraPos = cameraPosition.Value;
        PackAtUpload();
        _accumFrameCount = 0;
    }

    /// <summary>
    /// The splats were edited in the packed buffer (SplatEditor): rebuild the display vertices for the stochastic
    /// path and make the sorted paths (desktop sorted, XR) sort again, even with a still camera.
    /// </summary>
    public void SplatsEdited(Vector3 cameraPosition)
    {
        RepackForDisplay(cameraPosition);
        _sorter.RequestResort();
    }

    // ── Selection highlight (Edit tools) ─────────────────────────────────────────────────────────────────────
    GPUBuffer? _selBuf;
    readonly float[] _selData = new float[28];   // mat4 (16) + lo (4) + hi (4) + rows (4, as i32 bits)

    GPUBuffer EnsureSelectionBuffer()
    {
        if (_selBuf == null)
        {
            _selBuf = _device!.CreateBuffer(new GPUBufferDescriptor
            {
                Size = (ulong)(_selData.Length * sizeof(float)),
                Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
            });
            WriteSelection();
        }
        return _selBuf;
    }

    void WriteSelection()
    {
        if (_selBuf == null || _queue == null) return;
        var bytes = new byte[_selData.Length * sizeof(float)];
        Buffer.BlockCopy(_selData, 0, bytes, 0, bytes.Length);
        _queue.WriteBuffer(_selBuf, 0, bytes);
    }

    /// <summary>
    /// Tint the splats a selection takes (null = none) - drawn by the pack pass, so the scene data is untouched. Takes
    /// effect at once: the display vertices are repacked and the sorted paths sort again.
    /// </summary>
    public void SetSelectionHighlight(SplatEditor.Volume? selection)
    {
        System.Array.Clear(_selData);
        if (selection is { } v)
        {
            // Row-major: WGSL reads column-major, so m * p in the shader is p * M here.
            float[] m = { v.M11, v.M12, v.M13, v.M14, v.M21, v.M22, v.M23, v.M24, v.M31, v.M32, v.M33, v.M34, v.M41, v.M42, v.M43, v.M44 };
            System.Array.Copy(m, _selData, 16);
            _selData[16] = v.X0; _selData[17] = v.Y0; _selData[18] = v.Z0; _selData[19] = 1f;
            _selData[20] = v.X1; _selData[21] = v.Y1; _selData[22] = v.Z1;
            _selData[24] = BitConverter.Int32BitsToSingle(v.RowFrom);
            _selData[25] = BitConverter.Int32BitsToSingle(v.RowTo);
        }
        if (_device == null) return;
        EnsureSelectionBuffer();
        WriteSelection();
        RepackForDisplay();
        _sorter.RequestResort();
    }

    /// <summary>
    /// Draw more on top of the current XR eye view in a submit of its own (after the eye's main one), with the bridge
    /// canvas and its depth: for a second world-space UI batch, which cannot share a command buffer with the first
    /// (GameUI's EndWorldSpace writes one set of buffers per batch).
    /// </summary>
    public void RenderXROverlayToCanvas(Action<GPUCommandEncoder, GPUTextureView, GPUTextureView> overlay)
    {
        if (_device == null || _xrBridgeContext == null || _xrBridgeDepth == null) return;
        using var encoder = _device.CreateCommandEncoder();
        using var canvasTexture = _xrBridgeContext.GetCurrentTexture();
        using var canvasView = canvasTexture.CreateView();
        using var depthView = _xrBridgeDepth.CreateView();
        overlay(encoder, canvasView, depthView);
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, _submitArray);
    }

    /// <summary>
    /// Initialize the WebGPU render pipeline. Called once when canvas is attached.
    /// <paramref name="canvasRef"/> is stored for adaptive-resolution canvas pixel resizing.
    /// </summary>
    public void AttachCanvas(HTMLCanvasElement canvas, ElementReference canvasRef)
    {
        _canvasRef = canvasRef;

        var webGpuAccel = _gpu.WebGPUAccelerator;
        var nativeAccel = webGpuAccel.NativeAccelerator;
        _device = nativeAccel.NativeDevice
            ?? throw new InvalidOperationException("WebGPU native device is null");
        _queue = nativeAccel.Queue
            ?? throw new InvalidOperationException("WebGPU queue is null");

        _context = canvas.GetContext<GPUCanvasContext>("webgpu");

        using var navigator = SpawnJSRuntime.Instance.Get<Navigator>("navigator");
        using var gpu = navigator.Gpu;
        if (gpu is not null)
            _canvasFormat = gpu.GetPreferredCanvasFormat();

        _context.Configure(new GPUCanvasConfiguration
        {
            Device = _device,
            Format = _canvasFormat,
        });

        _physicalWidth = canvas.Width;
        _physicalHeight = canvas.Height;
        _canvasWidth = _physicalWidth;
        _canvasHeight = _physicalHeight;

        // ── Pack Compute Pipeline (Float32 → Float16/Unorm8 packing) ──
        using var packShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = PackComputeSource });
        _packPipeline = _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage
            {
                Module = packShader,
                EntryPoint = "pack_splats",
            }
        });

        // ── Splat Pipeline (packed vertex format) ──
        using var splatShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = SplatShaderSource });
        _splatPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = splatShader,
                EntryPoint = "vs_trainer",
                Buffers = new[]
                {
                    new GPUVertexBufferLayout
                    {
                        ArrayStride = (ulong)PackedBytesPerSplat,
                        StepMode = GPUVertexStepMode.Instance,
                        Attributes = new GPUVertexAttribute[]
                        {
                            new() { ShaderLocation = 0, Offset = 0,  Format = GPUVertexFormat.Float32x3 },  // position (12B)
                            new() { ShaderLocation = 1, Offset = 12, Format = GPUVertexFormat.Float16x4 },  // color+alpha (8B)
                            new() { ShaderLocation = 2, Offset = 20, Format = GPUVertexFormat.Float32x3 },  // scale sx,sy,sz (12B f32)
                            new() { ShaderLocation = 3, Offset = 32, Format = GPUVertexFormat.Float32x4 },  // rotation quat (16B f32)
                        }
                    }
                }
            },
            Fragment = new GPUFragmentState
            {
                Module = splatShader,
                EntryPoint = "fs_trainer",
                Targets = new[]
                {
                    new GPUColorTargetState
                    {
                        Format = SortedTargetFormat,
                        Blend = new GPUBlendState
                        {
                            Color = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.SrcAlpha,
                                DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                Operation = GPUBlendOperation.Add,
                            },
                            Alpha = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.One,
                                DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                Operation = GPUBlendOperation.Add,
                            }
                        }
                    }
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
            DepthStencil = new GPUDepthStencilState
            {
                Format = "depth24plus",
                DepthWriteEnabled = false,
                DepthCompare = "less",
            }
        });

        // ── Stochastic Splat Pipeline (sort-free, depth-tested, opaque writes) ──
        var splatVertexBuffers = new[]
        {
            new GPUVertexBufferLayout
            {
                ArrayStride = (ulong)PackedBytesPerSplat,
                StepMode = GPUVertexStepMode.Instance,
                Attributes = new GPUVertexAttribute[]
                {
                    new() { ShaderLocation = 0, Offset = 0,  Format = GPUVertexFormat.Float32x3 },
                    new() { ShaderLocation = 1, Offset = 12, Format = GPUVertexFormat.Float16x4 },
                    new() { ShaderLocation = 2, Offset = 20, Format = GPUVertexFormat.Float32x3 },
                    new() { ShaderLocation = 3, Offset = 32, Format = GPUVertexFormat.Float32x4 },
                }
            }
        };
        using var stochasticShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = StochasticSplatShaderSource });
        _stochasticSplatPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = stochasticShader,
                EntryPoint = "vs_trainer", // the trainer's footprint, as the sorted path
                Buffers = splatVertexBuffers,
            },
            Fragment = new GPUFragmentState
            {
                Module = stochasticShader,
                EntryPoint = "fs_stochastic_trainer",
                Targets = new[]
                {
                    new GPUColorTargetState { Format = _canvasFormat } // No blend — opaque writes
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
            DepthStencil = new GPUDepthStencilState
            {
                Format = "depth24plus",
                DepthWriteEnabled = true, // Stochastic NEEDS depth writes (hardware selects closest surviving sample)
                DepthCompare = "less",
            }
        });

        // ── CAS Pipeline ──
        using var casShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = CasShaderSource });
        _casPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState { Module = casShader, EntryPoint = "vs_fullscreen" },
            Fragment = new GPUFragmentState
            {
                Module = casShader,
                EntryPoint = "fs_cas",
                Targets = new[] { new GPUColorTargetState { Format = _canvasFormat } }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
        });

        // ── Accumulation Pipeline (fullscreen EMA blend for temporal convergence) ──
        using var accumShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = AccumulateShaderSource });
        _accumPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState { Module = accumShader, EntryPoint = "vs_fullscreen" },
            Fragment = new GPUFragmentState
            {
                Module = accumShader,
                EntryPoint = "fs_accum",
                Targets = new[]
                {
                    new GPUColorTargetState
                    {
                        Format = AccumTargetFormat,
                        Blend = new GPUBlendState
                        {
                            Color = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.SrcAlpha,
                                DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                Operation = GPUBlendOperation.Add,
                            },
                            Alpha = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.One,
                                DstFactor = GPUBlendFactor.Zero,
                                Operation = GPUBlendOperation.Add,
                            }
                        }
                    }
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
        });

        // Depth texture
        CreateDepthTexture();

        // Uniform buffer: see the U* slot constants (mat4 + 4 camera vec4s + viewport/focal/flags).
        _uniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)UniformFloats * sizeof(float),
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // Each pipeline has its own auto-generated bind group layout (even if structurally identical)
        _uniformBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _stochasticSplatPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry
                {
                    Binding = 0,
                    Resource = new GPUBufferBinding { Buffer = _uniformBuffer }
                }
            }
        });
        _uniformBindGroupSorted = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _splatPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry
                {
                    Binding = 0,
                    Resource = new GPUBufferBinding { Buffer = _uniformBuffer }
                }
            }
        });

        // Pre-allocate reusable byte buffers for direct WriteBuffer — avoids HeapView/PrimeHeap on every frame
        _uniformByteData = new byte[_uniformData.Length * sizeof(float)];
        _casByteData = new byte[_casData.Length * sizeof(float)];
        _packCountJsArray = new Uint32Array(8);

        // Pack uniforms: count, colours_are_sh_dc, sh_degree, pad | cam_pos.xyz (f32 bits), pad.
        _packCountBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 32,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // CAS uniform (16 bytes aligned: sharpening strength + texel size)
        _casUniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // Sampler for CAS
        _casSampler = _device.CreateSampler(new GPUSamplerDescriptor
        {
            MinFilter = "linear",
            MagFilter = "linear",
        });

        // Accumulation uniform (16 bytes aligned: weight + padding)
        _accumUniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _accumByteData = new byte[_accumData.Length * sizeof(float)];

        // Offscreen texture for CAS input (must be after CAS resources are created)
        CreateOffscreenTexture();

        // Stochastic textures (render target + accumulation)
        CreateStochasticTextures();

        // If splat data was uploaded before the canvas was attached, create the vertex buffer now.
        EnsureSplatBuffer();
        // Pack vertex buffer if upload already happened (deferred pack-at-upload)
        PackAtUpload();

        Console.WriteLine($"[GpuRenderer] Pipeline created: sorted + stochastic + CAS. Format: {_canvasFormat}");
    }

    private void CreateDepthTexture()
    {
        _depthView?.Dispose();
        _depthTexture?.Destroy();
        _depthTexture?.Dispose();

        _depthTexture = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = "depth24plus",
            Usage = GPUTextureUsage.RenderAttachment,
        });
        _depthView = _depthTexture.CreateView();
        RebuildCachedDescriptors();
    }

    private void CreateOffscreenTexture()
    {
        _offscreenView?.Dispose();
        _offscreenTexture?.Destroy();
        _offscreenTexture?.Dispose();

        _offscreenTexture = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = SortedTargetFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        });
        _offscreenView = _offscreenTexture.CreateView();
        RebuildCachedDescriptors();

        // Rebuild CAS bind group when texture changes
        if (_casPipeline != null && _casSampler != null && _casUniformBuffer != null)
        {
            _casBindGroup?.Dispose();
            _casBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _casPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _offscreenView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler },
                    new GPUBindGroupEntry
                    {
                        Binding = 2,
                        Resource = new GPUBufferBinding { Buffer = _casUniformBuffer }
                    }
                }
            });
        }
    }

    private void CreateStochasticTextures()
    {
        _stochasticView?.Dispose();
        _stochasticTexture?.Destroy();
        _stochasticTexture?.Dispose();
        _accumView?.Dispose();
        _accumTexture?.Destroy();
        _accumTexture?.Dispose();

        var desc = new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = _canvasFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        };

        _stochasticTexture = _device!.CreateTexture(desc);
        _stochasticView = _stochasticTexture.CreateView();
        _accumTexture = _device.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = AccumTargetFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        });
        _accumView = _accumTexture.CreateView();

        _accumFrameCount = 0;

        RebuildStochasticDescriptors();
    }

    /// <summary>Rebuild stochastic render pass descriptors and bind groups after texture recreation.</summary>
    private void RebuildStochasticDescriptors()
    {
        if (_stochasticView == null || _depthView == null || _accumView == null) return;

        // Stochastic splat pass → _stochasticTexture (cleared each frame, with depth)
        _stochasticColorAttach = new GPURenderPassColorAttachment
        {
            View = _stochasticView,
            LoadOp = GPULoadOp.Clear,
            StoreOp = GPUStoreOp.Store,
            ClearValue = NewClear(),
        };
        _stochasticPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _stochasticColorAttach },
            DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
            {
                View = _depthView,
                DepthLoadOp = "clear",
                DepthStoreOp = "store",
                DepthClearValue = 1.0f,
            },
        };

        // Accumulation pass → _accumTexture (LoadOp toggled per frame: clear on reset, load normally)
        _accumColorAttach = new GPURenderPassColorAttachment
        {
            View = _accumView,
            LoadOp = GPULoadOp.Clear, // toggled per frame
            StoreOp = GPUStoreOp.Store,
            ClearValue = NewClear(),
        };
        _accumPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _accumColorAttach },
        };

        // Accumulation bind group: samples _stochasticTexture
        if (_accumPipeline != null && _casSampler != null && _accumUniformBuffer != null)
        {
            _accumBindGroup?.Dispose();
            _accumBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _accumPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _stochasticView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler }, // reuse sampler
                    new GPUBindGroupEntry
                    {
                        Binding = 2,
                        Resource = new GPUBufferBinding { Buffer = _accumUniformBuffer }
                    }
                }
            });
        }

        // CAS reads _accumTexture in stochastic mode (rebuilt here since _accumView changed)
        RebuildCasBindGroupForAccum();
    }

    /// <summary>Rebuild CAS bind group for stochastic mode (samples _accumTexture).</summary>
    private void RebuildCasBindGroupForAccum()
    {
        if (_casPipeline == null || _casSampler == null || _casUniformBuffer == null || _accumView == null) return;

        // Stochastic CAS reads from accumulation texture (separate from sorted CAS which reads _offscreenView)
        _casBindGroupStochastic?.Dispose();
        _casBindGroupStochastic = _device!.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _casPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry { Binding = 0, Resource = _accumView },
                new GPUBindGroupEntry { Binding = 1, Resource = _casSampler },
                new GPUBindGroupEntry
                {
                    Binding = 2,
                    Resource = new GPUBufferBinding { Buffer = _casUniformBuffer }
                }
            }
        });
    }

    /// <summary>
    /// Rebuilds the cached render pass descriptor objects.
    /// Called after depth or offscreen texture recreation (canvas resize).
    /// On cache hit frames, these objects are reused directly — only the swapchain View field
    /// is updated per frame in Render() for the direct and CAS pass targets.
    /// </summary>
    private void RebuildCachedDescriptors()
    {
        // Every attachment below is constructed with NewClear(), so a rebuild already picks up
        // the current background. ApplyBackgroundToCachedAttachments() covers the other
        // direction: a background change AFTER the rebuild.

        // Direct splat pass (no CAS) — depth stencil is stable; color View updated per frame
        if (_depthView != null)
        {
            _splatColorAttachDirect = new GPURenderPassColorAttachment
            {
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = NewClear(),
            };
            _splatPassDescDirect = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { _splatColorAttachDirect },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = _depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
        }

        // CAS splat pass — renders to offscreen texture (fully stable, no per-frame update needed)
        if (_offscreenView != null && _depthView != null)
        {
            _splatColorAttachCas = new GPURenderPassColorAttachment
            {
                View = _offscreenView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = NewClear(),
            };
            _splatPassDescCas = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { _splatColorAttachCas },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = _depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
        }

        // CAS post-process pass — renders to swapchain; color View updated per frame
        _casColorAttach = new GPURenderPassColorAttachment
        {
            LoadOp = GPULoadOp.Clear,
            StoreOp = GPUStoreOp.Store,
            ClearValue = new GPUColorDict { R = 0.0, G = 0.0, B = 0.0, A = 1.0 },
        };
        _casPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _casColorAttach },
        };
    }

    /// <summary>
    /// Creates _splatBuffer if splat data is ready and device is available.
    /// Safe to call multiple times — no-ops if buffer already exists or data not ready.
    /// </summary>
    private void EnsureSplatBuffer()
    {
        if (_device == null || _splatCount == 0) return;

        _splatBuffer?.Destroy();
        _splatBuffer?.Dispose();
        _splatBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)_splatCount * PackedBytesPerSplat,
            Usage = GPUBufferUsage.Vertex | GPUBufferUsage.Storage,
        });

        // Invalidate pack bind group whenever the vertex buffer is recreated
        _packBindGroup?.Dispose();
        _packBindGroup = null;
        _srcDataCached = null;
        _srcIdxCached = null;
    }

    /// <summary>
    /// Upload scene data to GPU sorter and create vertex buffer.
    /// Called once when scene changes.
    /// </summary>
    public async Task UploadScene(GaussianScene scene)
    {
        UseRgbColours(); // a CPU-side scene (.ply / .splat / depth) is always RGB, never a trained scene
        await _sorter.UploadAsync(scene);

        _splatCount = _sorter.SplatCount;
        if (_splatCount == 0) return;

        EnsureSplatBuffer();

        // Fill identity indices and pack vertex buffer at upload time (for stochastic mode)
        await _sorter.FillIdentityIndicesAsync();
        PackAtUpload();

        // Reset accumulation so the new scene renders immediately (not blended with old scene)
        _accumFrameCount = 0;

        Console.WriteLine($"[GpuRenderer] Packed vertex buffer: {_splatCount:N0} splats ({_splatCount * PackedBytesPerSplat / 1024}KB, was {_splatCount * 40 / 1024}KB)");
    }

    /// <summary>
    /// Upload a GPU-resident packed buffer directly (GPU fast path, no CPU involvement).
    /// Transfers ownership of packedBuf to the sorter — caller must NOT dispose it.
    /// Safe to call before AttachCanvas — vertex buffer is deferred until canvas is ready.
    /// </summary>
    /// <summary>
    /// Stream a packed splat scene straight from a <see cref="Stream"/> (OPFS <c>BlobStream</c>, WebTorrent,
    /// etc.) into a GPU buffer, then upload it for rendering. // GPU load: file I/O
    /// When the stream is an <c>IJSReadStream</c> (browser OPFS/torrent), <c>CopyFromStreamAsync</c> streams
    /// the bytes JS-side chunk-by-chunk directly into the GPU buffer — they never enter the .NET/WASM managed
    /// heap. This is the only scalable load path for large scenes (a 5K image ≈ 14.7M splats ≈ 588 MB).
    /// </summary>
    /// <summary>
    /// Stream a saved scene .bin straight into a GPU buffer and upload it.
    /// <paramref name="floatsPerSplat"/> is the stride the FILE was written at, which is not
    /// necessarily the current one: scenes saved before splats carried a rotation are 10 floats
    /// wide and get widened on the GPU. Reading an old file at the new stride would not fail,
    /// it would just render noise, so the stride is always passed explicitly.
    /// </summary>
    /// <summary>A saved scene's packed splats streamed into a new GPU buffer at the current stride (widened as on
    /// load) WITHOUT uploading it - for inserting one scene into another. The caller owns the buffer.</summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> LoadPackedFromStreamAsync(Stream sceneStream, int splatCount, int floatsPerSplat)
    {
        var packedBuf = _gpu.WebGPUAccelerator.Allocate1D<float>((long)splatCount * floatsPerSplat);
        await packedBuf.View.CopyFromStreamAsync(sceneStream);
        return await _gaussianKernel.WidenPackedAsync(packedBuf, splatCount, floatsPerSplat);
    }

    /// <summary>Bytes as a new ILGPU word buffer (a .spawnscene v2 stream); the caller owns it.</summary>
    public MemoryBuffer1D<uint, Stride1D.Dense> IlgpuWordsFromArrayBuffer(Accelerator a, ArrayBuffer bytes)
    {
        var dst = a.Allocate1D<uint>(Math.Max(1L, bytes.ByteLength / sizeof(uint)));
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        _queue!.WriteBuffer(dst.GetGPUBuffer()!, 0L, bytes);
        return dst;
    }

    /// <summary>A saved SH part (bytes from the project store) as a new ILGPU buffer; the caller owns it.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> IlgpuFromArrayBuffer(Accelerator a, ArrayBuffer bytes)
    {
        var dst = a.Allocate1D<float>(bytes.ByteLength / sizeof(float));
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        _queue!.WriteBuffer(dst.GetGPUBuffer()!, 0L, bytes);
        return dst;
    }

    public async Task UploadSceneFromStream(Stream sceneStream, int splatCount, int floatsPerSplat)
    {
        var accelerator = _gpu.WebGPUAccelerator;
        var packedBuf = accelerator.Allocate1D<float>((long)splatCount * floatsPerSplat);
        await packedBuf.View.CopyFromStreamAsync(sceneStream);
        packedBuf = await _gaussianKernel.WidenPackedAsync(packedBuf, splatCount, floatsPerSplat);
        await UploadSceneFromGpuBuffer(packedBuf, splatCount);
    }

    public async Task UploadSceneFromGpuBuffer(
        MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)
    {
        await _sorter.UploadFromGpuBufferAsync(packedBuf, splatCount);

        _splatCount = _sorter.SplatCount;
        if (_splatCount == 0) return;

        EnsureSplatBuffer();

        // Fill identity indices and pack vertex buffer at upload time (for stochastic mode)
        await _sorter.FillIdentityIndicesAsync();
        PackAtUpload();

        // Reset accumulation so the new scene renders immediately (not blended with old scene)
        _accumFrameCount = 0;

        Console.WriteLine($"[GpuRenderer] GPU fast-path upload: {_splatCount:N0} splats" +
            (_splatBuffer != null ? $", {_splatCount * PackedBytesPerSplat / 1024}KB vertex buffer" : " (vertex buffer deferred)"));
    }

    /// <summary>
    /// Read packed splat data back from GPU to CPU as a .NET float[]. // CPU transfer: file I/O
    /// Returns float[splatCount * SplatFormat.Floats] or null if buffer unavailable.
    /// NOTE: This marshals every byte into the .NET/WASM managed heap — only use for SMALL,
    /// genuinely CPU-bound needs (e.g. PLY export). For OPFS save, use
    /// <see cref="ReadPackedUint8ArrayAsync"/> so the bytes stay in JS.
    /// </summary>
    public async Task<float[]?> ReadPackedDataAsync(int splatCount)
    {
        var buf = _sorter.PackedDataBuf;
        if (buf == null) return null;
        try
        {
            var accelerator = _gpu.WebGPUAccelerator;
            await accelerator.SynchronizeAsync();
            return await buf.CopyToHostAsync<float>(0, splatCount * SplatFormat.Floats);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuRenderer] ReadPackedData failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Read packed splat data off the GPU as a JS <see cref="Uint8Array"/>. // GPU→JS: file I/O (OPFS)
    /// The bytes land in the browser JS heap and NEVER enter the .NET/WASM managed heap — the
    /// whole point of the SpawnDev browser stack. Hand the returned Uint8Array straight to
    /// OPFS (FileSystemWritableFileStream.Write) so the data flows GPU → JS → disk with zero
    /// managed-heap copies. Caller owns the returned Uint8Array (dispose it).
    /// Returns null if the packed buffer is unavailable.
    /// </summary>
    public async Task<Uint8Array?> ReadPackedUint8ArrayAsync(int splatCount)
    {
        var buf = _sorter.PackedDataBuf;
        if (buf == null) return null;
        try
        {
            var accelerator = _gpu.WebGPUAccelerator;
            await accelerator.SynchronizeAsync();
            long byteCount = (long)splatCount * SplatFormat.Floats * sizeof(float);
            return await buf.CopyToHostUint8ArrayAsync(0, byteCount);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuRenderer] ReadPackedUint8Array failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// One-time pack at upload: converts ILGPU Float32 data → packed vertex buffer using identity indices.
    /// Used by stochastic mode so the vertex buffer is ready without per-frame sort+pack.
    /// Also provides initial data for sorted mode's first frame before sort completes.
    /// </summary>
    private void PackAtUpload()
    {
        var dataBuf = _sorter.PackedDataBuf;
        var idxBuf = _sorter.IndicesBuf;
        if (_device == null || _splatBuffer == null || _packPipeline == null || dataBuf == null || idxBuf == null) return;

        using var encoder = _device.CreateCommandEncoder();
        AppendPackComputePass(encoder, dataBuf, idxBuf, _splatCount);
        _lodArgsValid = false;   // the buffer holds every row now, not a cut
        using var cmdBuf = encoder.Finish();
        _submitArray[0] = cmdBuf;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, _submitArray);

        Console.WriteLine($"[GpuRenderer] Pack-at-upload complete: {_splatCount:N0} splats packed");
    }

    /// <summary>
    /// Resize canvas and recreate GPU textures for new dimensions.
    /// Called when the browser window is resized.
    /// Preserves the current adaptive resolution mode (low-res active = render at half the new physical size).
    /// </summary>
    public void ResizeCanvas(int newWidth, int newHeight)
    {
        if (_device == null || newWidth <= 0 || newHeight <= 0) return;

        _physicalWidth = newWidth;
        _physicalHeight = newHeight;

        int renderW = _lowResActive ? Math.Max(1, newWidth / 2) : newWidth;
        int renderH = _lowResActive ? Math.Max(1, newHeight / 2) : newHeight;

        if (renderW == _canvasWidth && renderH == _canvasHeight) return;

        _canvasWidth = renderW;
        _canvasHeight = renderH;

        CreateDepthTexture();
        CreateOffscreenTexture();
        CreateStochasticTextures();

        Console.WriteLine($"[GpuRenderer] Resized GPU textures: {renderW}×{renderH} (physical: {newWidth}×{newHeight})");
    }

    /// <summary>
    /// Render one frame. Fully synchronous — no GPU drain.
    /// Sorted mode: GPU sort → GPU pack → splat render → CAS sharpen.
    /// Stochastic mode: stochastic render → temporal accumulate → CAS display.
    /// </summary>
    private bool _renderLogged;
    public void Render(GaussianScene scene, CameraParams camera)
    {
        if (_device == null || _context == null || _splatBuffer == null || _splatCount == 0)
        {
            if (!_renderLogged)
            {
                _renderLogged = true;
                Console.WriteLine($"[Render] Early exit: device={_device != null} ctx={_context != null} buf={_splatBuffer != null} count={_splatCount}");
            }
            return;
        }
        if (!_renderLogged) { _renderLogged = true; Console.WriteLine($"[Render] First frame ({RenderMode}): {_splatCount} splats, cam={camera.Position} fwd={camera.Forward}"); }

        // ── Adaptive Resolution: enter/exit half-res based on mode and camera velocity ──
        {
            float velocity = _sorter.SmoothedVelocity;
            bool wantLowRes = AdaptiveResMode switch
            {
                AdaptiveResMode.ForceFull => false,
                AdaptiveResMode.ForceHalf => true,
                _                         => _lowResActive ? velocity > LowResExitVelocity : velocity > LowResEnterVelocity,
            };
            if (wantLowRes != _lowResActive && _physicalWidth > 0)
            {
                _lowResActive = wantLowRes;
                int rw = _lowResActive ? Math.Max(1, _physicalWidth / 2) : _physicalWidth;
                int rh = _lowResActive ? Math.Max(1, _physicalHeight / 2) : _physicalHeight;

                using var canvasEl = _canvasRef.As<HTMLCanvasElement>();
                canvasEl.Width = rw;
                canvasEl.Height = rh;

                _canvasWidth = rw;
                _canvasHeight = rh;
                CreateDepthTexture();
                CreateOffscreenTexture();
                CreateStochasticTextures();

                Console.WriteLine($"[GpuRenderer] Adaptive res: {(wantLowRes ? "LOW" : "FULL")} {rw}×{rh} (velocity={velocity:F4})");
            }
        }

        // Ensure camera dimensions match canvas
        if (camera.Width == 0 || camera.Height == 0)
        {
            camera.Width = _canvasWidth;
            camera.Height = _canvasHeight;
            camera.CenterX = _canvasWidth / 2f;
            camera.CenterY = _canvasHeight / 2f;
            camera.FocalX = CameraParams.ViewerFocal(_canvasHeight);
            camera.FocalY = camera.FocalX;
        }

        // ── Build MVP (needed by both modes) ──
        // Scale the camera's intrinsics to the CANVAS resolution, then build the projection and
        // the shader uniforms from the SAME numbers. Previously the projection came from a
        // symmetric fovY (which uses fy for both axes and cannot carry a principal point) while
        // the shader was handed fx and fy separately -- identical for a centred square-pixel
        // camera, but they disagree for any real one: a dataset GT pose or an AR passthrough
        // camera. Equivalence for the centred case is pinned by
        // CameraProjectionTests.CentredIntrinsics_MatchTheSymmetricPerspectiveItReplaces.
        var view = camera.ViewMatrix;
        float focalScaleX = camera.Width > 0 ? (float)_canvasWidth / camera.Width : 1f;
        float focalScaleY = camera.Height > 0 ? (float)_canvasHeight / camera.Height : 1f;
        float fx = camera.FocalX * focalScaleX;
        float fy = camera.FocalY * focalScaleY;
        float cx = camera.CenterX * focalScaleX;
        float cy = camera.CenterY * focalScaleY;

        var proj = CameraParams.CreateWebGpuProjection(
            fx, fy, cx, cy, _canvasWidth, _canvasHeight, camera.Near, camera.Far);
        var mvp = view * proj;

        // ── Upload MVP + camera basis + viewport uniforms ──
        WriteCameraUniforms(mvp, view, _canvasWidth, _canvasHeight, fx, fy);

        if (RenderMode == SplatRenderMode.Stochastic)
        {
            RenderStochastic(camera, mvp);
        }
        else
        {
            RenderSorted(camera, mvp);
        }
        FramesSubmitted++;
    }

    /// <summary>Stochastic mode: samples accumulated into the current image (resets while the camera moves), and the
    /// smoothed camera velocity that decides "moving". Diagnostics for convergence.</summary>
    public int StochasticAccumulatedSamples => _accumFrameCount;
    public float SmoothedCameraVelocity => _sorter.SmoothedVelocity;

    /// <summary>Frames submitted by <see cref="Render"/> so far (a capture waits for frames AFTER a camera jump).</summary>
    public long FramesSubmitted { get; private set; }

    /// <summary>
    /// The camera position the sorted path's current draw order was packed for. After a camera jump the sorted path
    /// keeps drawing the previous order until its asynchronous sort catches up; a frame shows the new pose only once
    /// this equals it.
    /// </summary>
    public Vector3 PackCameraPosition => _packCameraPos;

    /// <summary>
    /// Fill the shared uniform slots from a view/projection pair. Every render path goes through
    /// here: the splat vertex shader needs the camera BASIS (not just the MVP) to build the
    /// screen-space covariance, and the XR paths only ever have a view matrix to get it from.
    /// </summary>
    private void WriteCameraUniforms(Matrix4x4 mvp, Matrix4x4 view,
        float viewportW, float viewportH, float focalX, float focalY)
    {
        // .NET stores row-major and WGSL reads a mat4x4 column-major, so a straight row-order copy
        // transposes it — which is exactly what turns `world * M` into `M * world` in the shader.
        _uniformData[UMvp + 0] = mvp.M11; _uniformData[UMvp + 1] = mvp.M12; _uniformData[UMvp + 2] = mvp.M13; _uniformData[UMvp + 3] = mvp.M14;
        _uniformData[UMvp + 4] = mvp.M21; _uniformData[UMvp + 5] = mvp.M22; _uniformData[UMvp + 6] = mvp.M23; _uniformData[UMvp + 7] = mvp.M24;
        _uniformData[UMvp + 8] = mvp.M31; _uniformData[UMvp + 9] = mvp.M32; _uniformData[UMvp + 10] = mvp.M33; _uniformData[UMvp + 11] = mvp.M34;
        _uniformData[UMvp + 12] = mvp.M41; _uniformData[UMvp + 13] = mvp.M42; _uniformData[UMvp + 14] = mvp.M43; _uniformData[UMvp + 15] = mvp.M44;

        WorldSpaceGeometry.ViewMatrixToCameraBasis(view, out var right, out var up, out var fwd, out var pos);
        _uniformData[UCamRight + 0] = right.X; _uniformData[UCamRight + 1] = right.Y; _uniformData[UCamRight + 2] = right.Z; _uniformData[UCamRight + 3] = 0f;
        _uniformData[UCamUp + 0] = up.X; _uniformData[UCamUp + 1] = up.Y; _uniformData[UCamUp + 2] = up.Z; _uniformData[UCamUp + 3] = 0f;
        _uniformData[UCamFwd + 0] = fwd.X; _uniformData[UCamFwd + 1] = fwd.Y; _uniformData[UCamFwd + 2] = fwd.Z; _uniformData[UCamFwd + 3] = 0f;
        _uniformData[UCamPos + 0] = pos.X; _uniformData[UCamPos + 1] = pos.Y; _uniformData[UCamPos + 2] = pos.Z; _uniformData[UCamPos + 3] = 1f;

        _uniformData[UViewport + 0] = viewportW;
        _uniformData[UViewport + 1] = viewportH;
        _uniformData[UFocal + 0] = focalX;
        _uniformData[UFocal + 1] = focalY;
    }

    /// <summary>Sorted alpha-blend rendering: cull → sort → pack → render → optional CAS.</summary>
    private void RenderSorted(CameraParams camera, Matrix4x4 mvp)
    {
        var (dataBuf, idxBuf, sortRan, visibleCount) = _sorter.Sort(camera, mvp);
        if (sortRan) _packCameraPos = camera.Position;

        // Upload uniforms (frame_index/dilation/min_alpha not used in sorted mode)
        _uniformData[UFrameIndex] = 0f;
        _uniformData[UDilation] = 1f; // no dilation
        _uniformData[UMinAlpha] = 0f; // no alpha floor
        Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

        using var colorTexture = _context!.GetCurrentTexture();
        using var colorView = colorTexture.CreateView();
        using var encoder = _device!.CreateCommandEncoder();

        if (sortRan && dataBuf != null && idxBuf != null)
            AppendPackComputePass(encoder, dataBuf, idxBuf, visibleCount, _sorter.LodDrawnCountBuffer);

        // Always the f32-precision path: splats blend into the rgba16float offscreen target, then the CAS
        // pass writes the canvas. Strength 0 (or low-res motion) makes CAS an exact copy.
        using var splatPass = encoder.BeginRenderPass(_splatPassDescCas!);
        splatPass.SetPipeline(_splatPipeline!);
        splatPass.SetBindGroup(0, _uniformBindGroupSorted!);
        splatPass.SetVertexBuffer(0, _splatBuffer!);
        DrawSortedSplats(splatPass, visibleCount);
        splatPass.End();

        {
            _casData[0] = _lowResActive ? 0f : _sharpeningStrength;
            _casData[1] = 1f / _canvasWidth;
            _casData[2] = 1f / _canvasHeight;
            _casData[3] = 0f;
            Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
            _queue.WriteBuffer(_casUniformBuffer!, 0, _casByteData);

            _casColorAttach!.View = colorView;
            using var casPass = encoder.BeginRenderPass(_casPassDesc!);
            casPass.SetPipeline(_casPipeline!);
            casPass.SetBindGroup(0, _casBindGroup!);
            casPass.Draw(3, 1, 0, 0);
            casPass.End();
        }

        using var commandBuffer = encoder.Finish();
        _submitArray[0] = commandBuffer;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    /// <summary>
    /// Stochastic rasterization with velocity-adaptive dilation, multi-SPP, and temporal accumulation.
    /// Per frame: SPP × (stochastic render + accumulate) + 1 display pass.
    /// Each sub-sample uses a unique hash seed. Accumulation weight = 1/totalSamples (running average).
    /// </summary>
    private void RenderStochastic(CameraParams camera, Matrix4x4 mvp)
    {
        // Velocity tracking (no sort needed)
        _sorter.UpdateVelocity(camera.Position, camera.Forward);
        float velocity = _sorter.SmoothedVelocity;
        bool moving = velocity > 1e-7f;

        // View-dependent colour is baked at pack time for the pack's camera. Stochastic mode packs once at
        // upload and never sorts, so re-pack (identity indices) when the camera has moved, at most every
        // 50 ms - otherwise SH would be evaluated for whatever camera the upload saw.
        if (_shDegree > 0 && _shRest != null
            && Vector3.DistanceSquared(camera.Position, _packCameraPos) > 1e-10f
            && System.Diagnostics.Stopwatch.GetElapsedTime(_lastShRepack).TotalMilliseconds >= 50)
        {
            _packCameraPos = camera.Position;
            _lastShRepack = System.Diagnostics.Stopwatch.GetTimestamp();
            PackAtUpload();
        }

        // ── Velocity-adaptive parameters ──

        // Dilation: very subtle splat fattening to bridge sub-pixel spatial gaps (max +5%)
        _uniformData[UDilation] = 1f + MathF.Min(MathF.Sqrt(velocity) * DilationScale, MaxDilationFactor);

        // Min alpha floor: boost survival of low-alpha edge fragments during movement.
        // At splat edges, Gaussian alpha drops to 0.05-0.1 → 90%+ discard rate → holes.
        // Floor of 0.15 ensures at least 15% survival at edges, filling gaps between splats.
        // During convergence: floor=0 restores exact Monte Carlo sampling for correct result.
        _uniformData[UMinAlpha] = moving ? 0.15f : 0f;

        // When moving: RESET accumulation each frame to prevent ghosting entirely.
        // Each frame's SPP sub-samples are still properly averaged (weight = 1/1, 1/2, 1/3...),
        // but no inter-frame blending occurs since frameCount resets to 0.
        // When still: accumulation grows across frames for progressive convergence.
        if (moving)
            _accumFrameCount = 0;

        // Multi-SPP: more samples per frame to fill stochastic holes.
        // Moving: SPP=2 (two independent samples, much fewer holes than 1).
        // Just stopped: brief SPP burst for fast initial convergence.
        // Converged: SPP=1 (image is clean, maximize FPS).
        // SPP=2@60fps = 120 samples/sec > SPP=4@25fps = 100 samples/sec, so keep SPP low.
        int spp;
        if (moving)
            spp = StochasticSPP; // default 2 during movement
        else if (_accumFrameCount < 60) // first ~1 second after stopping
            spp = StochasticSPP + 1;    // convergence burst (e.g., 3)
        else
            spp = 1;                    // converged, save GPU

        // ── Multi-SPP loop: each sub-sample gets stochastic render + accumulate ──
        // Each queue.submit() includes preceding writeBuffer operations, so uniform updates
        // between sub-samples are correctly sequenced by the GPU.
        _accumColorAttach!.LoadOp = GPULoadOp.Load;

        for (int s = 0; s < spp; s++)
        {
            _accumFrameCount = Math.Min(_accumFrameCount + 1, 1024);
            float accumWeight = 1f / _accumFrameCount;

            // Upload uniforms: unique seed per sub-sample (global counter, never repeats)
            _globalFrameCount++;
            _uniformData[UFrameIndex] = BitConverter.Int32BitsToSingle(_globalFrameCount);
            Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
            _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

            // Upload accumulation weight for this sub-sample
            _accumData[0] = accumWeight;
            Buffer.BlockCopy(_accumData, 0, _accumByteData!, 0, _accumByteData!.Length);
            _queue.WriteBuffer(_accumUniformBuffer!, 0, _accumByteData);

            using var encoder = _device!.CreateCommandEncoder();

            // Stochastic splat render → _stochasticTexture (cleared each sub-sample)
            {
                using var pass = encoder.BeginRenderPass(_stochasticPassDesc!);
                pass.SetPipeline(_stochasticSplatPipeline!);
                pass.SetBindGroup(0, _uniformBindGroup!);
                pass.SetVertexBuffer(0, _splatBuffer!);
                pass.Draw(6, (uint)_splatCount, 0, 0);
                pass.End();
            }

            // Accumulate blend → _accumTexture (load previous, blend with weight)
            {
                using var pass = encoder.BeginRenderPass(_accumPassDesc!);
                pass.SetPipeline(_accumPipeline!);
                pass.SetBindGroup(0, _accumBindGroup!);
                pass.Draw(3, 1, 0, 0);
                pass.End();
            }

            using var cmdBuf = encoder.Finish();
            _submitArray[0] = cmdBuf;
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
        }

        // ── Display pass (once per frame): CAS reads _accumTexture → canvas ──
        {
            float displayStrength = _lowResActive ? 0f : _sharpeningStrength;
            _casData[0] = displayStrength;
            _casData[1] = 1f / _canvasWidth;
            _casData[2] = 1f / _canvasHeight;
            _casData[3] = 0f;
            Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
            _queue!.WriteBuffer(_casUniformBuffer!, 0, _casByteData);

            using var colorTexture = _context!.GetCurrentTexture();
            using var colorView = colorTexture.CreateView();
            using var displayEncoder = _device!.CreateCommandEncoder();

            _casColorAttach!.View = colorView;
            using var pass = displayEncoder.BeginRenderPass(_casPassDesc!);
            pass.SetPipeline(_casPipeline!);
            pass.SetBindGroup(0, _casBindGroupStochastic!);
            pass.Draw(3, 1, 0, 0);
            pass.End();

            using var displayCmdBuf = displayEncoder.Finish();
            _submitArray[0] = displayCmdBuf;
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
        }
    }

    /// <summary>
    /// Render a single XR eye view to the given color texture.
    /// Called twice per XR frame (left + right eye) with different view/proj matrices.
    /// Uses stochastic rendering without accumulation (single-sample per eye, no temporal blending).
    /// </summary>
    public void RenderXRView(Matrix4x4 viewMatrix, Matrix4x4 projMatrix,
        GPUTexture colorTexture, GPUTexture? depthTexture,
        int viewportX, int viewportY, int viewportWidth, int viewportHeight)
    {
        if (_device == null || _splatBuffer == null || _splatCount == 0 || _stochasticSplatPipeline == null) return;

        var mvp = viewMatrix * projMatrix;

        // Upload uniforms for this eye
        // WebXR gives no intrinsics, but its per-eye frustum encodes them. Recover rather than
        // assume: a headset eye is ASYMMETRIC (off-centre principal point, that is how the eyes
        // converge), so anything that assumes a centred frustum is wrong in one eye each way.
        CameraParams.ExtractIntrinsics(projMatrix, viewportWidth, viewportHeight,
            out float eyeFx, out float eyeFy, out _, out _);
        WriteCameraUniforms(mvp, viewMatrix, viewportWidth, viewportHeight,
            MathF.Abs(eyeFx), MathF.Abs(eyeFy));
        _globalFrameCount++;
        _uniformData[UFrameIndex] = BitConverter.Int32BitsToSingle(_globalFrameCount);
        _uniformData[UDilation] = 1f; // no dilation in XR
        _uniformData[UMinAlpha] = 0.1f; // slight min_alpha floor for XR (reduce holes)

        Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

        // Create a depth texture for this view if the XR layer doesn't provide one
        // For now, use a temporary depth texture
        using var colorView = colorTexture.CreateView();
        using var tempDepth = depthTexture == null ? _device.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { viewportWidth, viewportHeight },
            Format = "depth24plus",
            Usage = GPUTextureUsage.RenderAttachment,
        }) : null;
        using var depthView = (depthTexture ?? tempDepth!).CreateView();

        using var encoder = _device.CreateCommandEncoder();

        // Single stochastic render pass (no accumulation — each XR frame is independent)
        var colorAttach = new GPURenderPassColorAttachment
        {
            View = colorView,
            LoadOp = GPULoadOp.Clear,
            StoreOp = GPUStoreOp.Store,
            ClearValue = NewClear(),
        };
        var passDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { colorAttach },
            DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
            {
                View = depthView,
                DepthLoadOp = "clear",
                DepthStoreOp = "store",
                DepthClearValue = 1.0f,
            },
        };

        using var pass = encoder.BeginRenderPass(passDesc);
        pass.SetPipeline(_stochasticSplatPipeline);
        pass.SetBindGroup(0, _uniformBindGroup!);
        pass.SetVertexBuffer(0, _splatBuffer);
        pass.SetViewport(viewportX, viewportY, viewportWidth, viewportHeight, 0, 1);
        pass.Draw(6, (uint)_splatCount, 0, 0);
        pass.End();

        using var cmdBuf = encoder.Finish();
        _submitArray[0] = cmdBuf;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    /// <summary>
    /// Render a single XR eye to the internal bridge OffscreenCanvas (for WebGL XR fallback).
    /// WebGPU renders stochastic splats → OffscreenCanvas, then WebGLXRBlit reads it via texImage2D.
    /// No accumulation (each eye is independent). Optional CAS sharpening pass.
    /// </summary>
    public void RenderXRViewToCanvas(Matrix4x4 viewMatrix, Matrix4x4 projMatrix,
        int width, int height, bool applyCAS = false)
    {
        if (_device == null || _splatBuffer == null || _splatCount == 0 || _stochasticSplatPipeline == null) return;

        EnsureXRBridge(width, height, applyCAS);

        var mvp = viewMatrix * projMatrix;

        // Upload uniforms for this eye
        CameraParams.ExtractIntrinsics(projMatrix, width, height,
            out float eyeFx, out float eyeFy, out _, out _);
        WriteCameraUniforms(mvp, viewMatrix, width, height,
            MathF.Abs(eyeFx), MathF.Abs(eyeFy));
        _globalFrameCount++;
        _uniformData[UFrameIndex] = BitConverter.Int32BitsToSingle(_globalFrameCount);
        _uniformData[UDilation] = 1f; // no dilation in XR
        _uniformData[UMinAlpha] = 0.1f; // slight min_alpha floor for XR (reduce holes)

        Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

        using var encoder = _device.CreateCommandEncoder();

        // Determine color target: directly to canvas if no CAS, otherwise to intermediate texture
        using var canvasTexture = _xrBridgeContext!.GetCurrentTexture();

        if (applyCAS && _xrBridgeStochasticView != null && _xrCasBindGroup != null)
        {
            // Pass 1: stochastic render → intermediate texture
            using var depthView = _xrBridgeDepth!.CreateView();
            var colorAttach = new GPURenderPassColorAttachment
            {
                View = _xrBridgeStochasticView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = XRClear(),
            };
            var passDesc = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { colorAttach },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
            using var splatPass = encoder.BeginRenderPass(passDesc);
            splatPass.SetPipeline(_stochasticSplatPipeline);
            splatPass.SetBindGroup(0, _uniformBindGroup!);
            splatPass.SetVertexBuffer(0, _splatBuffer);
            splatPass.Draw(6, (uint)_splatCount, 0, 0);
            splatPass.End();

            // Pass 2: CAS → canvas texture
            _casData[0] = _sharpeningStrength;
            _casData[1] = 1f / width;
            _casData[2] = 1f / height;
            _casData[3] = 0f;
            Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
            _queue.WriteBuffer(_casUniformBuffer!, 0, _casByteData);

            using var canvasView = canvasTexture.CreateView();
            var casAttach = new GPURenderPassColorAttachment
            {
                View = canvasView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = new GPUColorDict { R = 0, G = 0, B = 0, A = 1.0 },
            };
            var casDesc = new GPURenderPassDescriptor { ColorAttachments = new[] { casAttach } };
            using var casPass = encoder.BeginRenderPass(casDesc);
            casPass.SetPipeline(_casPipeline!);
            casPass.SetBindGroup(0, _xrCasBindGroup);
            casPass.Draw(3, 1, 0, 0);
            casPass.End();
        }
        else
        {
            // Single pass: stochastic render directly → canvas texture
            using var canvasView = canvasTexture.CreateView();
            using var depthView = _xrBridgeDepth!.CreateView();
            var colorAttach = new GPURenderPassColorAttachment
            {
                View = canvasView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = XRClear(),
            };
            var passDesc = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { colorAttach },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
            using var splatPass = encoder.BeginRenderPass(passDesc);
            splatPass.SetPipeline(_stochasticSplatPipeline);
            splatPass.SetBindGroup(0, _uniformBindGroup!);
            splatPass.SetVertexBuffer(0, _splatBuffer);
            splatPass.Draw(6, (uint)_splatCount, 0, 0);
            splatPass.End();
        }

        using var xrCmdBuf = encoder.Finish();
        _submitArray[0] = xrCmdBuf;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    // ── XR, sorted: one depth sort per frame from the HEAD, shared by both eyes ────────────────────────────────
    // The stochastic XR path draws one sample per eye per frame and the head never holds still, so nothing
    // accumulates: the first emulator session (2026-10-03) showed sparse dots where the desktop shows the room. Sorted
    // alpha blending needs no accumulation; one sort from between the eyes serves both (the usual WebXR splat-viewer
    // approach), and the sorter runs asynchronously, so a frame never waits on it.
    private int _xrSortedVisible;
    private GPUTexture? _xrSortedTex;
    private GPUTextureView? _xrSortedView;
    private GPUBindGroup? _xrSortedCasBindGroup;
    private int _xrSortedW, _xrSortedH;

    /// <summary>Use sorted alpha blending in XR (default) rather than the stochastic path.</summary>
    public bool XRSorted { get; set; } = true;

    /// <summary>
    /// Start an XR frame on the sorted path: hand the sorter the head pose (in scene space) and a frustum wide enough for
    /// both eyes; when a sort has completed, repack the vertex buffer in its order (SH colour for the head position).
    /// </summary>
    public void BeginXRFrameSorted(CameraParams head, Matrix4x4 cullMvp)
    {
        if (_device == null || _splatBuffer == null || _splatCount == 0) return;
        // No sub-pixel LOD cull in XR: a scene shrunk by the grips or placed as an AR miniature is made of splats far
        // under 0.3 px (more so through this wide ~182 px-focal sort camera), and culling them removed the whole
        // miniature once it was set on the floor 1.5 m away (emulator, 2026-10-03). Together they ARE the image.
        float lod = _sorter.LodCullPixels;
        _sorter.LodCullPixels = 0f;
        (MemoryBuffer1D<float, Stride1D.Dense>? dataBuf, MemoryBuffer1D<int, Stride1D.Dense>? idxBuf, bool sortRan, int visibleCount) sorted;
        try { sorted = _sorter.Sort(head, cullMvp); }
        finally { _sorter.LodCullPixels = lod; }
        var (dataBuf, idxBuf, sortRan, visibleCount) = sorted;
        _xrSortedVisible = visibleCount;
        if (!sortRan || dataBuf == null || idxBuf == null) return;
        _packCameraPos = head.Position;
        using var encoder = _device.CreateCommandEncoder();
        AppendPackComputePass(encoder, dataBuf, idxBuf, visibleCount, _sorter.LodDrawnCountBuffer);
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, _submitArray);
    }

    /// <summary>One eye on the sorted path: back-to-front alpha blend into an rgba16float target, then CAS (or a plain
    /// copy at strength 0) into the XR bridge canvas for <see cref="WebGLXRBlit"/>.</summary>
    /// <param name="overlay">Optional: draws on top of the finished eye view (XR menus and pointers) with the bridge
    /// canvas view and a depth24plus view cleared to 1 (sorted splats write no depth), in the same command buffer.</param>
    public void RenderXRViewSortedToCanvas(Matrix4x4 viewMatrix, Matrix4x4 projMatrix, int width, int height, bool applyCAS,
        Action<GPUCommandEncoder, GPUTextureView, GPUTextureView>? overlay = null)
    {
        if (_device == null || _splatBuffer == null || _splatCount == 0 || _splatPipeline == null || _casPipeline == null) return;
        EnsureXRBridge(width, height, needsCAS: false);
        if (_xrSortedTex == null || _xrSortedW != width || _xrSortedH != height)
        {
            _xrSortedCasBindGroup?.Dispose();
            _xrSortedView?.Dispose();
            _xrSortedTex?.Destroy(); _xrSortedTex?.Dispose();
            _xrSortedTex = _device.CreateTexture(new GPUTextureDescriptor
            {
                Size = new[] { width, height },
                Format = SortedTargetFormat,
                Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
            });
            _xrSortedView = _xrSortedTex.CreateView();
            _xrSortedCasBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _casPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _xrSortedView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler! },
                    new GPUBindGroupEntry { Binding = 2, Resource = new GPUBufferBinding { Buffer = _casUniformBuffer! } },
                },
            });
            _xrSortedW = width; _xrSortedH = height;
        }

        var mvp = viewMatrix * projMatrix;
        CameraParams.ExtractIntrinsics(projMatrix, width, height, out float eyeFx, out float eyeFy, out _, out _);
        WriteCameraUniforms(mvp, viewMatrix, width, height, MathF.Abs(eyeFx), MathF.Abs(eyeFy));
        _uniformData[UFrameIndex] = 0f;
        _uniformData[UDilation] = 1f;
        _uniformData[UMinAlpha] = 0f;
        Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

        using var encoder = _device.CreateCommandEncoder();
        // The sorted pipeline declares a depth24plus attachment (compare "less", no writes): give it the bridge's.
        using var depthView = _xrBridgeDepth!.CreateView();
        using (var splatPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
        {
            ColorAttachments = new[]
            {
                new GPURenderPassColorAttachment
                {
                    View = _xrSortedView!, LoadOp = GPULoadOp.Clear, StoreOp = GPUStoreOp.Store, ClearValue = XRClear(),
                },
            },
            DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
            {
                View = depthView, DepthLoadOp = "clear", DepthStoreOp = "store", DepthClearValue = 1.0f,
            },
        }))
        {
            splatPass.SetPipeline(_splatPipeline);
            splatPass.SetBindGroup(0, _uniformBindGroupSorted!);
            splatPass.SetVertexBuffer(0, _splatBuffer);
            DrawSortedSplats(splatPass, _xrSortedVisible);
            splatPass.End();
        }
        _casData[0] = applyCAS ? _sharpeningStrength : 0f;
        _casData[1] = 1f / width;
        _casData[2] = 1f / height;
        _casData[3] = XRTransparent ? 1f : 0f;
        Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
        _queue.WriteBuffer(_casUniformBuffer!, 0, _casByteData);
        using var canvasTexture = _xrBridgeContext!.GetCurrentTexture();
        using var canvasView = canvasTexture.CreateView();
        using (var casPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
        {
            ColorAttachments = new[]
            {
                new GPURenderPassColorAttachment
                {
                    View = canvasView, LoadOp = GPULoadOp.Clear, StoreOp = GPUStoreOp.Store,
                    ClearValue = new GPUColorDict { R = 0, G = 0, B = 0, A = XRTransparent ? 0 : 1 },
                },
            },
        }))
        {
            casPass.SetPipeline(_casPipeline);
            casPass.SetBindGroup(0, _xrSortedCasBindGroup!);
            casPass.Draw(3, 1, 0, 0);
            casPass.End();
        }
        overlay?.Invoke(encoder, canvasView, depthView);
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    /// <summary>Lazily create or resize the XR bridge OffscreenCanvas + associated resources.</summary>
    private void EnsureXRBridge(int width, int height, bool needsCAS)
    {
        if (_xrBridgeCanvas != null && _xrBridgeWidth == width && _xrBridgeHeight == height)
        {
            // Size matches — only create CAS resources if newly requested
            if (needsCAS && _xrBridgeStochasticTex == null)
                CreateXRBridgeCASResources(width, height);
            return;
        }

        // Dispose old resources
        DisposeXRBridge();

        _xrBridgeCanvas = new OffscreenCanvas(width, height);
        _xrBridgeContext = _xrBridgeCanvas.GetWebGPUContext();
        _xrBridgeContext.Configure(new GPUCanvasConfiguration
        {
            Device = _device!,
            Format = _canvasFormat,
            // Alpha survives the copy into the XR layer (AR passthrough); VR views write alpha 1 anyway.
            AlphaMode = "premultiplied",
        });

        _xrBridgeDepth = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { width, height },
            Format = "depth24plus",
            Usage = GPUTextureUsage.RenderAttachment,
        });

        _xrBridgeWidth = width;
        _xrBridgeHeight = height;

        if (needsCAS)
            CreateXRBridgeCASResources(width, height);
    }

    private void CreateXRBridgeCASResources(int width, int height)
    {
        _xrBridgeStochasticTex?.Destroy();
        _xrBridgeStochasticTex?.Dispose();
        _xrBridgeStochasticView?.Dispose();
        _xrCasBindGroup?.Dispose();

        _xrBridgeStochasticTex = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { width, height },
            Format = _canvasFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        });
        _xrBridgeStochasticView = _xrBridgeStochasticTex.CreateView();

        // CAS bind group reads from the XR stochastic texture
        if (_casPipeline != null && _casSampler != null && _casUniformBuffer != null)
        {
            _xrCasBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _casPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _xrBridgeStochasticView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler },
                    new GPUBindGroupEntry
                    {
                        Binding = 2,
                        Resource = new GPUBufferBinding { Buffer = _casUniformBuffer }
                    }
                }
            });
        }
    }

    /// <summary>Release XR bridge resources. Called on session end and in Dispose.</summary>
    public void DisposeXRBridge()
    {
        _xrSortedCasBindGroup?.Dispose(); _xrSortedCasBindGroup = null;
        _xrSortedView?.Dispose(); _xrSortedView = null;
        _xrSortedTex?.Destroy(); _xrSortedTex?.Dispose(); _xrSortedTex = null;
        _xrSortedW = _xrSortedH = 0;
        _xrCasBindGroup?.Dispose();
        _xrCasBindGroup = null;
        _xrBridgeStochasticView?.Dispose();
        _xrBridgeStochasticView = null;
        _xrBridgeStochasticTex?.Destroy();
        _xrBridgeStochasticTex?.Dispose();
        _xrBridgeStochasticTex = null;
        _xrBridgeDepth?.Destroy();
        _xrBridgeDepth?.Dispose();
        _xrBridgeDepth = null;
        _xrBridgeContext?.Dispose();
        _xrBridgeContext = null;
        _xrBridgeCanvas?.Dispose();
        _xrBridgeCanvas = null;
        _xrBridgeWidth = 0;
        _xrBridgeHeight = 0;
    }

    /// <summary>
    /// Appends a pack compute pass to the supplied encoder.
    /// Converts ILGPU Float32 splat data → packed vertex buffer using the sorted index buffer.
    /// Only packs visibleCount splats — culled sentinels at [visibleCount..N-1] are skipped.
    /// Caller is responsible for submitting the encoder.
    /// Called only on frames where the sort ran (indices changed).
    /// </summary>
    private void AppendPackComputePass(
        GPUCommandEncoder encoder,
        MemoryBuffer1D<float, Stride1D.Dense> dataBuf,
        MemoryBuffer1D<int, Stride1D.Dense> idxBuf,
        int visibleCount,
        MemoryBuffer1D<int, Stride1D.Dense>? lodDrawnCount = null)
    {
        if (_splatBuffer == null || _device == null || _packPipeline == null || _packCountBuf == null) return;

        var srcDataBuffer = dataBuf.GetGPUBuffer();
        if (srcDataBuffer == null) return;

        var srcIdxBuffer = idxBuf.GetGPUBuffer();
        if (srcIdxBuffer == null) return;

        // Write pack uniforms (before encoder submit, queue.writeBuffer runs first).
        _packCountJsArray![0] = (uint)visibleCount;
        _packCountJsArray[1] = ColoursAreShDc ? 1u : 0u;
        _packCountJsArray[2] = ColoursAreShDc && _shRest != null ? (uint)_shDegree : 0u;
        _packCountJsArray[3] = 0u;
        _packCountJsArray[4] = BitConverter.SingleToUInt32Bits(_packCameraPos.X);
        _packCountJsArray[5] = BitConverter.SingleToUInt32Bits(_packCameraPos.Y);
        _packCountJsArray[6] = BitConverter.SingleToUInt32Bits(_packCameraPos.Z);
        _packCountJsArray[7] = 0u;
        _queue!.WriteBuffer(_packCountBuf, 0, _packCountJsArray);

        _shDummy ??= _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Storage,
        });
        var sh0 = _shRest?[0] ?? _shDummy;
        var shBinding = sh0;

        // Create or reuse pack bind group (invalidate only when GPU buffer refs change)
        if (_packBindGroup == null || _srcDataCached != srcDataBuffer || _srcIdxCached != srcIdxBuffer
            || !ReferenceEquals(_shRestCached, shBinding))
        {
            _packBindGroup?.Dispose();
            _srcDataCached = srcDataBuffer;
            _srcIdxCached = srcIdxBuffer;
            _shRestCached = shBinding;

            using var layout = _packPipeline.GetBindGroupLayout(0);
            _packBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = srcDataBuffer } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = srcIdxBuffer } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _splatBuffer } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _packCountBuf } },
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = sh0 } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _shRest?[1] ?? _shDummy } },
                    new() { Binding = 6, Resource = new GPUBufferBinding { Buffer = _shRest?[2] ?? _shDummy } },
                    new() { Binding = 7, Resource = new GPUBufferBinding { Buffer = EnsureSelectionBuffer() } },
                }
            });
        }

        // 2D dispatch to stay within WebGPU's maxComputeWorkgroupsPerDimension (65535).
        // For scenes ≤ 4.2M splats: wgY=1 (identical to old 1D path).
        // For larger scenes (e.g. 5K full-res = 14.7M splats): wgY=4.
        // Out-of-bounds threads hit the i >= u.count guard in the shader and return early.
        const uint maxWG = 65535u;
        uint totalWG = (uint)((visibleCount + 63) / 64);
        uint wgX = Math.Min(totalWG, maxWG);
        uint wgY = (totalWG + maxWG - 1) / maxWG;
        var countBuffer = lodDrawnCount?.GetGPUBuffer();
        if (countBuffer != null)
        {
            AppendLodArgsPass(encoder, countBuffer);
            using var lodPass = encoder.BeginComputePass();
            lodPass.SetPipeline(_packPipeline);
            lodPass.SetBindGroup(0, _packBindGroup);
            lodPass.DispatchWorkgroupsIndirect(_lodArgsBuf!, 0);
            lodPass.End();
            _lodArgsValid = true;
            return;
        }
        using var pass = encoder.BeginComputePass();
        pass.SetPipeline(_packPipeline);
        pass.SetBindGroup(0, _packBindGroup);
        pass.DispatchWorkgroups(wgX, wgY, 1);
        pass.End();
    }

    /// <summary>Write the LOD cut's indirect arguments from the cull's drawn count (one GPU thread, no readback).</summary>
    private void AppendLodArgsPass(GPUCommandEncoder encoder, GPUBuffer countBuffer)
    {
        if (_lodArgsPipeline == null)
        {
            using var module = _device!.CreateShaderModule(new GPUShaderModuleDescriptor { Code = LodArgsSource });
            _lodArgsPipeline = _device.CreateComputePipeline(new GPUComputePipelineDescriptor
            {
                Layout = "auto",
                Compute = new GPUProgrammableStage { Module = module, EntryPoint = "lod_args" },
            });
            _lodArgsBuf = _device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = 32,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.Indirect,
            });
        }
        if (_lodArgsBindGroup == null || !ReferenceEquals(_lodArgsCountCached, countBuffer))
        {
            _lodArgsBindGroup?.Dispose();
            _lodArgsCountCached = countBuffer;
            using var layout = _lodArgsPipeline.GetBindGroupLayout(0);
            _lodArgsBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = countBuffer } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = _lodArgsBuf! } },
                }
            });
        }
        using var pass = encoder.BeginComputePass();
        pass.SetPipeline(_lodArgsPipeline);
        pass.SetBindGroup(0, _lodArgsBindGroup);
        pass.DispatchWorkgroups(1, 1, 1);
        pass.End();
    }

    /// <summary>The sorted splat draw: only the LOD cut (indirect) once one was packed, else every packed slot.</summary>
    private void DrawSortedSplats(GPURenderPassEncoder pass, int count)
    {
        if (_lodArgsValid && _sorter.LodActive && _lodArgsBuf != null) pass.DrawIndirect(_lodArgsBuf, 16);
        else pass.Draw(6, (uint)count, 0, 0);
    }

    // The pack dispatch (2D past 65535 groups, as AppendPackComputePass) and the draw (6 vertices an instance) for the
    // first n sorted rows, n = the LOD cull's drawn count.
    private const string LodArgsSource = @"
@group(0) @binding(0) var<storage, read>       drawn : array<u32>;
@group(0) @binding(1) var<storage, read_write> args  : array<u32>;

@compute @workgroup_size(1)
fn lod_args() {
    let n = drawn[0];
    let groups = (n + 63u) / 64u;
    let wgX = min(groups, 65535u);
    args[0] = wgX;
    args[1] = select(1u, (groups + 65534u) / 65535u, groups > 0u);
    args[2] = 1u;
    args[3] = 0u;
    args[4] = 6u;
    args[5] = n;
    args[6] = 0u;
    args[7] = 0u;
}
";

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _splatBuffer?.Destroy();
        _splatBuffer?.Dispose();
        DestroyShRest();
        _shSplitPipeline?.Dispose();
        _shDummy?.Destroy(); _shDummy?.Dispose();
        _uniformBuffer?.Destroy();
        _uniformBuffer?.Dispose();
        _uniformBindGroup?.Dispose();
        _uniformBindGroupSorted?.Dispose();
        _depthTexture?.Destroy();
        _depthTexture?.Dispose();
        _depthView?.Dispose();
        _offscreenTexture?.Destroy();
        _offscreenTexture?.Dispose();
        _offscreenView?.Dispose();
        _casBindGroup?.Dispose();
        _casBindGroupStochastic?.Dispose();
        _casUniformBuffer?.Destroy();
        _casUniformBuffer?.Dispose();
        _casSampler?.Dispose();
        _splatPipeline?.Dispose();
        _casPipeline?.Dispose();
        _packCountBuf?.Destroy();
        _lodArgsBuf?.Destroy(); _lodArgsBuf?.Dispose();
        _lodArgsBindGroup?.Dispose();
        _lodArgsPipeline?.Dispose();
        _packCountBuf?.Dispose();
        _packCountJsArray?.Dispose();

        // XR bridge resources
        DisposeXRBridge();

        // Stochastic rasterization resources
        _stochasticSplatPipeline?.Dispose();
        _stochasticTexture?.Destroy();
        _stochasticTexture?.Dispose();
        _stochasticView?.Dispose();
        _accumTexture?.Destroy();
        _accumTexture?.Dispose();
        _accumView?.Dispose();
        _accumPipeline?.Dispose();
        _accumBindGroup?.Dispose();
        _accumUniformBuffer?.Destroy();
        _accumUniformBuffer?.Dispose();
    }

    // ============================================================
    //  WGSL Splat Vertex Stage - 3D covariance EWA (Zwicker / Kerbl 3DGS)
    //
    //  Shared verbatim by the sorted and stochastic pipelines: the projection is the part that
    //  is easy to get subtly wrong, so there is exactly one copy of it. Only the fragment stage
    //  differs between the two.
    //
    //  MUST match Services/SplatCovariance.cs, which is unit-tested against analytic answers
    //  (SpawnScene.Tests/SplatCovarianceTests.cs). Change one, change both.
    //
    //  The splat is emitted as a quad aligned to the eigenvectors of the screen-space covariance
    //  and spanning SIGMA_CUTOFF sigmas along each. That makes the quad coordinate `uv` a
    //  whitened coordinate, so the fragment stage evaluates the Gaussian with dot(uv, uv) and
    //  needs no conic matrix at all.
    // ============================================================
    private const string SplatVertexWgsl = @"
struct Uniforms {
    mvp         : mat4x4<f32>,
    cam_right   : vec4<f32>,   // world-space camera basis; xyz used, w padding
    cam_up      : vec4<f32>,
    cam_fwd     : vec4<f32>,   // direction the camera LOOKS
    cam_pos     : vec4<f32>,
    viewport    : vec2<f32>,
    focal       : vec2<f32>,
    frame_index : u32,
    dilation    : f32,
    min_alpha   : f32,
    _pad3       : u32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) position    : vec3<f32>,
    @location(1) color_alpha : vec4<f32>,  // Unorm8x4
    @location(2) scale       : vec4<f32>,  // Float32x3: sx, sy, sz (w reads 1) (world units, 1 sigma)
    @location(3) quat        : vec4<f32>,  // Float32x4: x, y, z, w
};

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) color   : vec3<f32>,
    @location(1) opacity : f32,
    @location(2) uv      : vec2<f32>,      // whitened splat coordinate; unit disk = the footprint
    @location(3) cut     : f32,            // footprint radius in sigmas (the unit disk's edge)
    // vs_trainer only: the trainer's conic (inverse 2D covariance) and centre, framebuffer pixels (y down).
    @location(4) @interpolate(flat) conic     : vec3<f32>,
    @location(5) @interpolate(flat) centre_px : vec2<f32>,
};

// The trainer's footprint, not a fixed ellipse: like the reference rasteriser it has NO sigma cutoff, only
// alpha >= 1/255 (and alpha <= 0.99). A splat of opacity o therefore reaches sqrt(2 ln(255 o)) sigmas - 3.33
// for an opaque one - and the optimiser fitted colours with those fringes present. The viewer cut every splat
// at 3 sigma and missed them: MEASURED 2026-09-24, trainer-vs-viewer dumps on Truck differed on every edge.
const VIEW_MIN_ALPHA : f32 = 0.00392156862;   // 1/255, SplatTrainerShaders.MIN_ALPHA
const VIEW_MAX_ALPHA : f32 = 0.99;            // SplatTrainerShaders.MAX_ALPHA
const TRAINER_TILE : f32 = 16.0;              // SplatTrainerShaders TILE
const TRAINER_TILE_SIGMAS : f32 = 3.0;        // SplatTrainerShaders SIGMA_CUTOFF (tile overlap extent)

// Footprint cutoff in standard deviations. 3 sigma captures 98.9% of the mass; below ~2.5 the
// truncation shows up as a visible hard edge on large splats.
const SIGMA_CUTOFF : f32 = 3.0;

// Zwicker EWA antialiasing prefilter, in pixels squared. Keeps a sub-pixel splat at about a
// half-pixel sigma instead of letting it alias into a flickering dot. Replaces the old
// max(radius, 0.25px) clamp, which fattened every splat instead of only the sub-pixel ones.
const EWA_FILTER_PX2 : f32 = 0.3;
// Viewer near cull, scene units: the trainer's and the reference rasteriser's 0.2. (Lowering it to 0.01 changed
// nothing measurable on gsplat's imported Truck, 2026-10-04.)
const VIEWER_NEAR : f32 = 0.2;

// A splat this large is either sitting on the near plane or numerically broken; either way it
// can only cost fill rate. A rasterizer guard, not a workaround for a defect elsewhere.
const MAX_AXIS_PX_FACTOR : f32 = 4.0;

fn quad_corner(vid : u32) -> vec2<f32> {
    var quad_pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0)
    );
    return quad_pos[vid];
}

// Degenerate vertex behind the near plane with zero opacity: cheapest possible discard.
fn splat_reject(uv : vec2<f32>, color : vec3<f32>) -> VertexOutput {
    var out : VertexOutput;
    out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.color = color;
    out.opacity = 0.0;
    out.uv = uv;
    out.cut = 1.0;
    return out;
}

@vertex
fn vs_trainer(input : VertexInput, @builtin(vertex_index) vid : u32) -> VertexOutput {
    let uv = quad_corner(vid);
    let rgb = input.color_alpha.rgb;

    let center_clip = u.mvp * vec4<f32>(input.position, 1.0);
    if (center_clip.w <= 0.001) { return splat_reject(uv, rgb); }

    // -- Camera-space centre. Basis rows are (right, up, forward), z forward and positive,
    //    y measured UPWARD so a pixel offset maps to NDC with no sign flip. --
    let rel = input.position - u.cam_pos.xyz;
    let cx = dot(u.cam_right.xyz, rel);
    let cy = dot(u.cam_up.xyz, rel);
    let cz = dot(u.cam_fwd.xyz, rel);
    // Near plane 0.2 scene units, same as the trainer and the reference rasteriser. A splat at
    // depth 1e-5 projects to a frame-covering quad whose f32 conic is garbage: a full-screen flash.
    if (cz <= VIEWER_NEAR) { return splat_reject(uv, rgb); }

    // -- Sigma_world = R S S^T R^T --
    let q = normalize(input.quat);
    let s = max(input.scale.xyz, vec3<f32>(1e-9, 1e-9, 1e-9)) * u.dilation;

    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
    let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;

    // Columns of R, each pre-scaled by its axis: M = R * S, so Sigma = M * M^T.
    let m0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)) * s.x;
    let m1 = vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)) * s.y;
    let m2 = vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)) * s.z;
    let M = mat3x3<f32>(m0, m1, m2);
    let sigma_world = M * transpose(M);

    // -- Rotate into camera space: Sigma_cam = A * Sigma * A^T, A rows = (right, up, fwd) --
    let A = transpose(mat3x3<f32>(u.cam_right.xyz, u.cam_up.xyz, u.cam_fwd.xyz));
    let sc = A * sigma_world * transpose(A);

    // -- Perspective Jacobian of (u, v) = (fx * x / z, fy * y / z) at the centre, x/z and y/z clamped to
    //    1.3 x tan(fov/2) exactly as the trainer (and the reference's computeCov2D) do. Without it the viewer drew
    //    far-off-axis splats as the same huge smears the trainer did - differently, since it culls by world margin. --
    let invz = 1.0 / cz;
    let invz2 = invz * invz;
    let lim = 1.3 * 0.5 * u.viewport / u.focal;
    let cxc = clamp(cx * invz, -lim.x, lim.x) * cz;
    let cyc = clamp(cy * invz, -lim.y, lim.y) * cz;
    let j00 = u.focal.x * invz;
    let j02 = -u.focal.x * cxc * invz2;
    let j11 = u.focal.y * invz;
    let j12 = -u.focal.y * cyc * invz2;

    // mat[col][row]; sigma_cam is symmetric so the order does not matter.
    let s00 = sc[0][0]; let s01 = sc[1][0]; let s02 = sc[2][0];
    let s11 = sc[1][1]; let s12 = sc[2][1]; let s22 = sc[2][2];

    let a0 = j00 * s00 + j02 * s02;
    let a1 = j00 * s01 + j02 * s12;
    let a2 = j00 * s02 + j02 * s22;
    let b1 = j11 * s11 + j12 * s12;
    let b2 = j11 * s12 + j12 * s22;

    let cov_a = a0 * j00 + a2 * j02 + EWA_FILTER_PX2;
    let cov_b = a1 * j11 + a2 * j12;
    let cov_c = b1 * j11 + b2 * j12 + EWA_FILTER_PX2;

    // -- Eigen-decompose the 2x2 into principal screen axes --
    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 1e-20) { return splat_reject(uv, rgb); }

    // -- From here on, exactly the trainer's footprint (SplatTrainerShaders project() + emit_keys + splat_weight):
    //    the conic from the same det, the centre in pixels, and ONLY the pixels of the 16-px tiles overlapped by
    //    centre +- 3 sqrt(lambda_max). The eigen-decomposed whitened quad of vs_main disagreed with it for thin,
    //    near splats (MEASURED 2026-09-25 TruckFull 30K: the near post and near pavement smeared in the viewer
    //    and not in the trainer; held-out captures 1.6-2.3 dB under the trainer's own render). --
    // The covariance above is in camera axes with y UP (j11/j12 differentiate +fy*y/z), but centre_px and the fragment's
    // pixel offsets have y DOWN: flipping y negates the off-diagonal, so in pixel space the conic's xy term is +cov_b/det.
    // It was -cov_b/det: every tilted ellipse drawn MIRRORED about the horizontal - invisible on round splats, fur on
    // needles (gsplat's Truck at the exact camera gsplat drew cleanly, 2026-10-04).
    let conic = vec3<f32>(cov_c / det, cov_b / det, cov_a / det);
    let mid = 0.5 * (cov_a + cov_c);
    let l1 = mid + sqrt(max(mid * mid - det, 0.0));
    let op = input.color_alpha.a;
    if (op <= 0.0) { return splat_reject(uv, rgb); }

    let ndc_center = center_clip.xyz / center_clip.w;
    if (ndc_center.z < -0.1 || ndc_center.z > 1.1) { return splat_reject(uv, rgb); }
    let centre_px = vec2<f32>((ndc_center.x + 1.0) * 0.5 * u.viewport.x, (1.0 - ndc_center.y) * 0.5 * u.viewport.y);
    let r = TRAINER_TILE_SIGMAS * sqrt(max(l1, 1e-20));
    if (!(r < 1e30)) { return splat_reject(uv, rgb); }
    let tiles_end = ceil(u.viewport / TRAINER_TILE) * TRAINER_TILE;
    var lo = max(floor((centre_px - vec2<f32>(r, r)) / TRAINER_TILE) * TRAINER_TILE, vec2<f32>(0.0, 0.0));
    var hi = min((floor((centre_px + vec2<f32>(r, r)) / TRAINER_TILE) + 1.0) * TRAINER_TILE, tiles_end);
    // Overdraw only: a pixel survives fs_trainer iff alpha >= 1/255, i.e. Mahalanobis^2 <= k^2 = 2 ln(255 op), whose
    // bounding box has half-widths k sqrt(cov_a), k sqrt(cov_c). Intersecting with it (1 px margin for rounding)
    // changes no pixel's value - the fragment test still decides - but a thin splat no longer rasterises the whole
    // 3-sigma square of its long axis.
    let k2 = 2.0 * log(255.0 * op);
    if (k2 <= 0.0) { return splat_reject(uv, rgb); }
    let half = sqrt(k2 * vec2<f32>(cov_a, cov_c)) + vec2<f32>(1.0, 1.0);
    lo = max(lo, floor(centre_px - half));
    hi = min(hi, ceil(centre_px + half));
    if (hi.x <= lo.x || hi.y <= lo.y) { return splat_reject(uv, rgb); }

    let corner_px = select(lo, hi, uv > vec2<f32>(0.0, 0.0));
    let ndc = vec2<f32>(corner_px.x / u.viewport.x * 2.0 - 1.0, 1.0 - corner_px.y / u.viewport.y * 2.0);

    var out : VertexOutput;
    out.clip_pos = vec4<f32>(ndc * center_clip.w, ndc_center.z * center_clip.w, center_clip.w);
    out.color = rgb;
    out.opacity = op;
    out.uv = uv;
    out.cut = 1.0;
    out.conic = conic;
    out.centre_px = centre_px;
    return out;
}

@vertex
fn vs_main(input : VertexInput, @builtin(vertex_index) vid : u32) -> VertexOutput {
    let uv = quad_corner(vid);
    let rgb = input.color_alpha.rgb;

    let center_clip = u.mvp * vec4<f32>(input.position, 1.0);
    if (center_clip.w <= 0.001) { return splat_reject(uv, rgb); }

    // -- Camera-space centre. Basis rows are (right, up, forward), z forward and positive,
    //    y measured UPWARD so a pixel offset maps to NDC with no sign flip. --
    let rel = input.position - u.cam_pos.xyz;
    let cx = dot(u.cam_right.xyz, rel);
    let cy = dot(u.cam_up.xyz, rel);
    let cz = dot(u.cam_fwd.xyz, rel);
    // Near plane 0.2 scene units, same as the trainer and the reference rasteriser. A splat at
    // depth 1e-5 projects to a frame-covering quad whose f32 conic is garbage: a full-screen flash.
    if (cz <= VIEWER_NEAR) { return splat_reject(uv, rgb); }

    // -- Sigma_world = R S S^T R^T --
    let q = normalize(input.quat);
    let s = max(input.scale.xyz, vec3<f32>(1e-9, 1e-9, 1e-9)) * u.dilation;

    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
    let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;

    // Columns of R, each pre-scaled by its axis: M = R * S, so Sigma = M * M^T.
    let m0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)) * s.x;
    let m1 = vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)) * s.y;
    let m2 = vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)) * s.z;
    let M = mat3x3<f32>(m0, m1, m2);
    let sigma_world = M * transpose(M);

    // -- Rotate into camera space: Sigma_cam = A * Sigma * A^T, A rows = (right, up, fwd) --
    let A = transpose(mat3x3<f32>(u.cam_right.xyz, u.cam_up.xyz, u.cam_fwd.xyz));
    let sc = A * sigma_world * transpose(A);

    // -- Perspective Jacobian of (u, v) = (fx * x / z, fy * y / z) at the centre, x/z and y/z clamped to
    //    1.3 x tan(fov/2) exactly as the trainer (and the reference's computeCov2D) do. Without it the viewer drew
    //    far-off-axis splats as the same huge smears the trainer did - differently, since it culls by world margin. --
    let invz = 1.0 / cz;
    let invz2 = invz * invz;
    let lim = 1.3 * 0.5 * u.viewport / u.focal;
    let cxc = clamp(cx * invz, -lim.x, lim.x) * cz;
    let cyc = clamp(cy * invz, -lim.y, lim.y) * cz;
    let j00 = u.focal.x * invz;
    let j02 = -u.focal.x * cxc * invz2;
    let j11 = u.focal.y * invz;
    let j12 = -u.focal.y * cyc * invz2;

    // mat[col][row]; sigma_cam is symmetric so the order does not matter.
    let s00 = sc[0][0]; let s01 = sc[1][0]; let s02 = sc[2][0];
    let s11 = sc[1][1]; let s12 = sc[2][1]; let s22 = sc[2][2];

    let a0 = j00 * s00 + j02 * s02;
    let a1 = j00 * s01 + j02 * s12;
    let a2 = j00 * s02 + j02 * s22;
    let b1 = j11 * s11 + j12 * s12;
    let b2 = j11 * s12 + j12 * s22;

    let cov_a = a0 * j00 + a2 * j02 + EWA_FILTER_PX2;
    let cov_b = a1 * j11 + a2 * j12;
    let cov_c = b1 * j11 + b2 * j12 + EWA_FILTER_PX2;

    // -- Eigen-decompose the 2x2 into principal screen axes --
    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 1e-20) { return splat_reject(uv, rgb); }

    let mid = 0.5 * (cov_a + cov_c);
    let disc = sqrt(max(mid * mid - det, 0.0));
    let l1 = mid + disc;
    var l2 = mid - disc;
    if (l2 <= 0.0) { l2 = det / max(l1, 1e-20); }
    if (l1 <= 0.0) { return splat_reject(uv, rgb); }

    // Both candidates span the l1 eigenspace; take the longer so a near-isotropic covariance
    // does not normalize rounding noise into an arbitrary direction.
    let p1 = vec2<f32>(cov_b, l1 - cov_a);
    let p2 = vec2<f32>(l1 - cov_c, cov_b);
    var e1 = select(p2, p1, dot(p1, p1) >= dot(p2, p2));
    let elen = length(e1);
    e1 = select(vec2<f32>(1.0, 0.0), e1 / max(elen, 1e-20), elen > 1e-12);

    // Where opacity * exp(-cut^2 / 2) falls to 1/255. Faint splats get a smaller quad than before.
    let op = input.color_alpha.a;
    if (op <= VIEW_MIN_ALPHA) { return splat_reject(uv, rgb); }
    let cut = min(sqrt(2.0 * log(op / VIEW_MIN_ALPHA)), 4.0);
    let r1 = cut * sqrt(l1);
    let r2 = cut * sqrt(l2);
    if (r1 > MAX_AXIS_PX_FACTOR * max(u.viewport.x, u.viewport.y)) { return splat_reject(uv, rgb); }

    let axis_major = e1 * r1;                          // pixels
    let axis_minor = vec2<f32>(-e1.y, e1.x) * r2;      // pixels, orthogonal by construction

    // -- Cull against NDC using the ellipse's real extent --
    let px_to_ndc = vec2<f32>(2.0 / u.viewport.x, 2.0 / u.viewport.y);
    let ndc_center = center_clip.xyz / center_clip.w;
    let ext = (abs(axis_major) + abs(axis_minor)) * px_to_ndc;

    if (ndc_center.x + ext.x < -1.0 || ndc_center.x - ext.x > 1.0 ||
        ndc_center.y + ext.y < -1.0 || ndc_center.y - ext.y > 1.0 ||
        ndc_center.z < -0.1 || ndc_center.z > 1.1) {
        return splat_reject(uv, rgb);
    }

    let offset_ndc = (uv.x * axis_major + uv.y * axis_minor) * px_to_ndc;
    let final_ndc = vec3<f32>(ndc_center.xy + offset_ndc, ndc_center.z);

    var out : VertexOutput;
    out.clip_pos = vec4<f32>(final_ndc * center_clip.w, center_clip.w);
    out.color = rgb;
    out.opacity = input.color_alpha.a;
    out.uv = uv;
    out.cut = cut;
    return out;
}
";

    // ============================================================
    //  Sorted pipeline - classic back-to-front alpha blending.
    // ============================================================
    private const string SplatShaderSource = SplatVertexWgsl + @"
// The trainer's per-pixel weight (SplatTrainerShaders splat_weight) at this pixel's centre, with its alpha rules.
@fragment
fn fs_trainer(input : VertexOutput) -> @location(0) vec4<f32> {
    let d = input.clip_pos.xy - input.centre_px;
    let c = input.conic;
    let power = -0.5 * (c.x * d.x * d.x + c.z * d.y * d.y) - c.y * d.x * d.y;
    if (power > 0.0) { discard; }
    let alpha = min(VIEW_MAX_ALPHA, input.opacity * exp(power));
    if (alpha < VIEW_MIN_ALPHA) { discard; }
    return vec4<f32>(input.color, alpha);
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // uv is whitened: the unit disk IS the cut-sigma ellipse, so the Mahalanobis distance
    // squared is simply cut^2 * dot(uv, uv). No conic, no inverse covariance.
    let r2 = dot(input.uv, input.uv);
    if (r2 > 1.0) { discard; }

    let alpha = min(VIEW_MAX_ALPHA, input.opacity * exp(-0.5 * input.cut * input.cut * r2));
    if (alpha < VIEW_MIN_ALPHA) { discard; }

    return vec4<f32>(input.color, alpha);
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL CAS (Contrast Adaptive Sharpening) Post-Processing
    // ════════════════════════════════════════════════════════════
    private const string CasShaderSource = @"
struct CASUniforms {
    strength   : f32,
    texel_x    : f32,
    texel_y    : f32,
    keep_alpha : f32,   // > 0.5: pass alpha through (premultiplied; AR passthrough), else opaque
};

@group(0) @binding(0) var t_color : texture_2d<f32>;
@group(0) @binding(1) var s_color : sampler;
@group(0) @binding(2) var<uniform> cas : CASUniforms;

struct VSOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> VSOutput {
    // Fullscreen triangle (covers entire screen with 3 vertices)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let pos = positions[vid];
    var out : VSOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    // Flip Y for WebGPU UV convention
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_cas(input : VSOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let tx = cas.texel_x;
    let ty = cas.texel_y;

    // Sample center and 8 neighbors (cross + diagonals)
    let c  = textureSample(t_color, s_color, uv);
    let n  = textureSample(t_color, s_color, uv + vec2<f32>(0.0, -ty));
    let s  = textureSample(t_color, s_color, uv + vec2<f32>(0.0,  ty));
    let e  = textureSample(t_color, s_color, uv + vec2<f32>( tx, 0.0));
    let w  = textureSample(t_color, s_color, uv + vec2<f32>(-tx, 0.0));
    let ne = textureSample(t_color, s_color, uv + vec2<f32>( tx, -ty));
    let nw = textureSample(t_color, s_color, uv + vec2<f32>(-tx, -ty));
    let se = textureSample(t_color, s_color, uv + vec2<f32>( tx,  ty));
    let sw = textureSample(t_color, s_color, uv + vec2<f32>(-tx,  ty));

    // CAS: find min/max of full 8-neighbor pattern
    let mn_cross = min(min(n, s), min(e, w));
    let mx_cross = max(max(n, s), max(e, w));
    let mn_diag  = min(min(ne, nw), min(se, sw));
    let mx_diag  = max(max(ne, nw), max(se, sw));
    let mn = min(mn_cross, mn_diag);
    let mx = max(mx_cross, mx_diag);

    // Adaptive sharpening weight (sharper where neighbors are similar)
    let amp = clamp(min(mn, vec4<f32>(2.0) - mx) / mx, vec4<f32>(0.0), vec4<f32>(1.0));
    let sharp = amp * cas.strength;

    // Weighted average: cross neighbors 2x weight, diagonals 1x (total = 12)
    let avg = (n + s + e + w) * 0.166666 + (ne + nw + se + sw) * 0.083333;
    let result = mix(c, c + (c - avg) * sharp, vec4<f32>(cas.strength));

    if (cas.keep_alpha > 0.5) {
        // Premultiplied: colour can never exceed coverage.
        let a = clamp(result.a, 0.0, 1.0);
        return vec4<f32>(clamp(result.rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
    }
    return vec4<f32>(clamp(result.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
";

    // ============================================================
    //  WGSL Stochastic Splat Shader - sort-free rendering via stochastic transparency
    //  Same EWA vertex stage as the sorted pipeline. The fragment stage does a stochastic
    //  discard plus the hardware depth test; over accumulated frames this converges to the
    //  correct alpha-blended result without ever sorting.
    // ============================================================
    private const string StochasticSplatShaderSource = SplatVertexWgsl + @"
// lowbias32 hash - fast, good avalanche properties
fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    return x;
}

// The trainer's footprint and alpha (as fs_trainer, the sorted path's), then the stochastic test.
//
// The random number must be independent PER FRAGMENT. It was seeded by pixel and frame only, so every splat over a pixel
// drew the same u: the pixel came out empty whenever u >= the LARGEST alpha there, i.e. with probability 1 - max(a_i)
// instead of compositing's prod(1 - a_i) (four splats of alpha 0.5: covered 50% of the time, not 94%). The converged
// image was darker and patterned at the splat grid (MEASURED 2026-10-02, single-photo Room sample: PSNR 17.1 dB against
// the sorted render, mean brightness 81 vs 109). The splat's own projected centre now enters the seed.
@fragment
fn fs_stochastic_trainer(input : VertexOutput) -> @location(0) vec4<f32> {
    let d = input.clip_pos.xy - input.centre_px;
    let c = input.conic;
    let power = -0.5 * (c.x * d.x * d.x + c.z * d.y * d.y) - c.y * d.x * d.y;
    if (power > 0.0) { discard; }
    let alpha = min(VIEW_MAX_ALPHA, input.opacity * exp(power));
    if (alpha < VIEW_MIN_ALPHA) { discard; }
    let effective_alpha = max(alpha, u.min_alpha);
    let pixel = vec2<u32>(input.clip_pos.xy);
    let splat_id = hash_u32((bitcast<u32>(input.centre_px.x) * 73856093u) ^ (bitcast<u32>(input.centre_px.y) * 19349663u));
    let seed = hash_u32(pixel.x + pixel.y * 65537u + u.frame_index * 2654435761u) ^ splat_id;
    let u_rand = f32(hash_u32(seed)) / 4294967295.0;
    if (u_rand >= effective_alpha) { discard; }
    return vec4<f32>(input.color, 1.0);
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(input.uv, input.uv);
    if (r2 > 1.0) { discard; }

    let alpha = min(VIEW_MAX_ALPHA, input.opacity * exp(-0.5 * input.cut * input.cut * r2));
    if (alpha < VIEW_MIN_ALPHA) { discard; }

    // Stochastic transparency: discard with probability (1 - effective_alpha).
    // min_alpha floor: during movement, boost survival of low-alpha edge fragments
    // to fill holes. During convergence, min_alpha=0 restores exact Monte Carlo sampling.
    let effective_alpha = max(alpha, u.min_alpha);
    let pixel = vec2<u32>(input.clip_pos.xy);
    let seed = pixel.x + pixel.y * 65537u + u.frame_index * 2654435761u;
    let u_rand = f32(hash_u32(seed)) / 4294967295.0;
    if (u_rand >= effective_alpha) { discard; }

    return vec4<f32>(input.color, 1.0);  // Opaque write - no alpha blending
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL Temporal Accumulation — fullscreen EMA blend
    //  Reads per-frame stochastic render, blends into persistent accumulation texture.
    //  Uses SrcAlpha/OneMinusSrcAlpha blend: output alpha = 1/frameCount = running average weight.
    // ════════════════════════════════════════════════════════════
    private const string AccumulateShaderSource = @"
struct AccumUniforms {
    weight   : f32,
    _pad1    : f32,
    _pad2    : f32,
    _pad3    : f32,
};

@group(0) @binding(0) var t_current : texture_2d<f32>;
@group(0) @binding(1) var s_current : sampler;
@group(0) @binding(2) var<uniform> accum : AccumUniforms;

struct VSOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> VSOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let pos = positions[vid];
    var out : VSOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_accum(input : VSOutput) -> @location(0) vec4<f32> {
    let current = textureSample(t_current, s_current, input.uv);
    return vec4<f32>(current.rgb, accum.weight);
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL Pack Compute — Float32 sort output → packed vertex format
    //  Input:  SplatFormat.Floats per splat (pos3, color3, scale3, opacity1, quat4)
    //  Output: SplatFormat.PackedWords u32s per splat
    //          (pos3_bitcast, color_alpha_u8x4, scale_f16x4, quat_f16x4) = 32 bytes
    // ════════════════════════════════════════════════════════════
    private const string PackComputeSource = @"
struct PackUniforms {
    count             : u32,
    colours_are_sh_dc : u32,
    sh_degree         : u32,   // 0 = DC only; 1..3 = evaluate SH bands for the view direction
    _pad0             : u32,
    cam_pos           : vec4<f32>,   // the camera the current sort (and so this pack) is for
}

@group(0) @binding(0) var<storage, read>       src     : array<f32>;  // SplatFormat.Floats per splat
@group(0) @binding(1) var<storage, read>       idx     : array<i32>;  // sorted indices; -1 = culled sentinel
@group(0) @binding(2) var<storage, read_write> dst     : array<u32>;  // packed vertex output (8 u32s/splat)
@group(0) @binding(3) var<uniform>             u       : PackUniforms;
// SH rest in SphericalHarmonics.Parts buffers of 15 floats a splat (or dummies).
@group(0) @binding(4) var<storage, read>       sh_rest0 : array<f32>;
@group(0) @binding(5) var<storage, read>       sh_rest1 : array<f32>;
@group(0) @binding(6) var<storage, read>       sh_rest2 : array<f32>;

// The edit selection (SplatEditor.Volume), drawn tinted: the same test as the editor's kernels. lo.w = 1 when on.
struct Selection {
    m    : mat4x4<f32>,   // row-major copy of the System.Numerics matrix: m * p == p * M
    lo   : vec4<f32>,     // x0, y0, z0, enabled
    hi   : vec4<f32>,     // x1, y1, z1, -
    rows : vec4<i32>,     // rowFrom, rowTo (a row range when rowTo > rowFrom), -, -
}
@group(0) @binding(7) var<uniform> sel : Selection;

fn is_selected(i : u32, p : vec3<f32>) -> bool {
    if (sel.lo.w == 0.0) { return false; }
    if (sel.rows.y > sel.rows.x) { return i32(i) >= sel.rows.x && i32(i) < sel.rows.y; }
    let c = sel.m * vec4<f32>(p, 1.0);
    if (c.w <= 1e-7) { return false; }
    let n = c.xyz / c.w;
    return all(n >= sel.lo.xyz) && all(n <= sel.hi.xyz);
}

const SH_C0 : f32 = 0.28209479177387814;
const SH_C1 : f32 = 0.4886025119029199;
" + SphericalHarmonics.WgslPartAccess + SphericalHarmonics.WgslViewRgb + @"

@compute @workgroup_size(64)
fn pack_splats(@builtin(global_invocation_id) gid : vec3<u32>,
               @builtin(num_workgroups) nwg : vec3<u32>) {
    // 2D dispatch: recompute linear index from row (gid.y) and column (gid.x).
    // nwg.x = wgX (workgroups in X), so nwg.x * 64 = total threads per row.
    let i = gid.y * nwg.x * 64u + gid.x;
    if (i >= u.count) { return; }

    let dstOff = i * 12u;

    // Culled splats have idx=-1 sentinel (sorted last by DescendingInt32).
    // Write a fully-transparent vertex so the fragment shader discards it cheaply.
    let origIdx = idx[i];
    if (origIdx < 0) {
        dst[dstOff + 0u] = 0u;
        dst[dstOff + 1u] = 0u;
        dst[dstOff + 2u] = 0u;
        dst[dstOff + 3u] = 0u;  // opacity = 0 → fragment discard
        dst[dstOff + 4u] = 0u;
        dst[dstOff + 5u] = 0u;
        dst[dstOff + 6u] = 0u;
        dst[dstOff + 7u] = 0u;
        dst[dstOff + 8u] = 0u;
        dst[dstOff + 9u] = 0u;
        dst[dstOff + 10u] = 0u;
        dst[dstOff + 11u] = 0u;
        return;
    }

    // Index lookup: maps sorted position i to original splat data — eliminates CPU reorder pass
    let srcOff = u32(origIdx) * 14u;

    // Position: 3 floats bitcast to 3 u32s (preserve full precision)
    dst[dstOff + 0u] = bitcast<u32>(src[srcOff + 0u]);  // pos.x
    dst[dstOff + 1u] = bitcast<u32>(src[srcOff + 1u]);  // pos.y
    dst[dstOff + 2u] = bitcast<u32>(src[srcOff + 2u]);  // pos.z

    // Colour: linear RGB, OR SH DC -> RGB when training converted the buffer (feeding raw DC made every
    // trained scene look like washed blobs). f16, unclamped above 1 like the trainer's compositing.
    var rgb = vec3<f32>(src[srcOff + 3u], src[srcOff + 4u], src[srcOff + 5u]);
    if (u.colours_are_sh_dc != 0u) {
        // Same function the trainer renders with, for the direction from this pack's camera.
        let pos = vec3<f32>(src[srcOff + 0u], src[srcOff + 1u], src[srcOff + 2u]);
        rgb = sh_view_rgb(u32(origIdx), normalize(pos - u.cam_pos.xyz), rgb, u.sh_degree);
    }
    // Selected (Edit tools): amber-tinted, so what Delete / Move / Copy will take is visible first.
    if (is_selected(u32(origIdx), vec3<f32>(src[srcOff + 0u], src[srcOff + 1u], src[srcOff + 2u]))) {
        rgb = mix(rgb, vec3<f32>(1.0, 0.72, 0.15), 0.5);
    }
    dst[dstOff + 3u] = pack2x16float(vec2<f32>(max(rgb.r, 0.0), max(rgb.g, 0.0)));
    dst[dstOff + 4u] = pack2x16float(vec2<f32>(max(rgb.b, 0.0), clamp(src[srcOff + 9u], 0.0, 1.0)));

    // Scale and rotation in FULL f32. They were f16 on the claim that ~3 digits of a unit quaternion is
    // 'invisible at any splat size' - true for round splats, false for needles: a 3DGS-trained splat 1000x longer
    // than wide, tilted by f16's 1e-3, moves its long axis by ten times its own width. gsplat's own Truck imported
    // rendered as fur and streaks at the exact camera gsplat drew cleanly (2026-10-04).
    dst[dstOff + 5u] = bitcast<u32>(src[srcOff + 6u]);
    dst[dstOff + 6u] = bitcast<u32>(src[srcOff + 7u]);
    dst[dstOff + 7u] = bitcast<u32>(src[srcOff + 8u]);
    dst[dstOff + 8u] = bitcast<u32>(src[srcOff + 10u]);
    dst[dstOff + 9u] = bitcast<u32>(src[srcOff + 11u]);
    dst[dstOff + 10u] = bitcast<u32>(src[srcOff + 12u]);
    dst[dstOff + 11u] = bitcast<u32>(src[srcOff + 13u]);
}
";
}
