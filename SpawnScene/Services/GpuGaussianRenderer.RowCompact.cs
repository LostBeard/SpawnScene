using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

// Source splat rows kept compact in VRAM (Research/viewer-memory-2026-10-11.md): SplatFormat.CompactWords (36 B) a splat
// instead of SplatFormat.Floats (56 B). Only the frame's readers see the compact rows - the pack shader (its sv()
// accessor, PackComputeSource's compact variant) and GpuSplatSorter's flat cull. Every other reader asks for
// PackedSplatBuffer (36 call sites: training, Edit, export, SfM views, stats) or reads the rows back to save them: that
// expands them to f32 and keeps them f32 for the rest of the scene (pinned), so no tool ever holds a buffer the next
// frame frees. A new scene upload unpins. LOD trees and the streamed pool are never compacted.
public partial class GpuGaussianRenderer
{
    /// <summary>Draw from compact source rows (positions f32, colour / opacity / scale f16, rotation snorm16). The default
    /// since 2026-10-12; <c>&amp;rowq=0</c> keeps the f32 rows.</summary>
    public static bool RowCompact { get; set; } = true;

    bool _rowsPinned;
    int _rowsHold;   // leases out (BorrowRowsF32): no compaction under them
    GPUComputePipeline? _rowCompactPipeline, _rowExpandPipeline;
    GPUBuffer? _rowParams;

    const string RowQWgsl = @"
// Source rows: SplatFormat.Floats (14) f32 <-> SplatFormat.CompactWords (9) u32 - pos x y z f32, (r g) (b opacity)
// (sx sy) (sz -) f16 pairs, quaternion (x y) (z w) snorm16 pairs, normalized. f16 is clamped to its finite range.
@group(0) @binding(0) var<storage, read_write> rows32 : array<f32>;
@group(0) @binding(1) var<storage, read_write> rowsq  : array<u32>;
@group(0) @binding(2) var<uniform>             n      : vec4<u32>;

fn h2(a : f32, b : f32) -> u32 { return pack2x16float(clamp(vec2<f32>(a, b), vec2<f32>(-65504.0), vec2<f32>(65504.0))); }

@compute @workgroup_size(256)
fn compact_rows(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * 256u + gid.x;
    if (i >= n.x) { return; }
    let a = i * 14u;
    let b = i * 9u;
    rowsq[b] = bitcast<u32>(rows32[a]);
    rowsq[b + 1u] = bitcast<u32>(rows32[a + 1u]);
    rowsq[b + 2u] = bitcast<u32>(rows32[a + 2u]);
    rowsq[b + 3u] = h2(rows32[a + 3u], rows32[a + 4u]);
    rowsq[b + 4u] = h2(rows32[a + 5u], rows32[a + 9u]);
    rowsq[b + 5u] = h2(rows32[a + 6u], rows32[a + 7u]);
    rowsq[b + 6u] = h2(rows32[a + 8u], 0.0);
    var q = vec4<f32>(rows32[a + 10u], rows32[a + 11u], rows32[a + 12u], rows32[a + 13u]);
    if (dot(q, q) > 0.0) { q = normalize(q); }
    rowsq[b + 7u] = pack2x16snorm(q.xy);
    rowsq[b + 8u] = pack2x16snorm(q.zw);
}

@compute @workgroup_size(256)
fn expand_rows(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * 256u + gid.x;
    if (i >= n.x) { return; }
    let a = i * 14u;
    let b = i * 9u;
    rows32[a] = bitcast<f32>(rowsq[b]);
    rows32[a + 1u] = bitcast<f32>(rowsq[b + 1u]);
    rows32[a + 2u] = bitcast<f32>(rowsq[b + 2u]);
    let c = unpack2x16float(rowsq[b + 3u]);
    let bo = unpack2x16float(rowsq[b + 4u]);
    let s = unpack2x16float(rowsq[b + 5u]);
    rows32[a + 3u] = c.x;
    rows32[a + 4u] = c.y;
    rows32[a + 5u] = bo.x;
    rows32[a + 6u] = s.x;
    rows32[a + 7u] = s.y;
    rows32[a + 8u] = unpack2x16float(rowsq[b + 6u]).x;
    rows32[a + 9u] = bo.y;
    let qxy = unpack2x16snorm(rowsq[b + 7u]);
    let qzw = unpack2x16snorm(rowsq[b + 8u]);
    rows32[a + 10u] = qxy.x;
    rows32[a + 11u] = qxy.y;
    rows32[a + 12u] = qzw.x;
    rows32[a + 13u] = qzw.y;
}
";

    /// <summary>The pack shader's row accessor over f32 rows (part of PackComputeSource).</summary>
    const string PackRowsF32 = @"const SRC_STRIDE : u32 = 14u;   // SplatFormat.Floats
fn sv(o : u32, k : u32) -> f32 { return src[o + k]; }";

    /// <summary>The pack shader's row accessor over compact rows (replaces PackRowsF32 in the compact variant).</summary>
    const string PackRowsCompact = @"const SRC_STRIDE : u32 = 9u;   // SplatFormat.CompactWords (&rowq)
fn sv(o : u32, k : u32) -> f32 {
    switch (k) {
        case 0u, 1u, 2u: { return bitcast<f32>(src[o + k]); }
        case 3u: { return unpack2x16float(src[o + 3u]).x; }
        case 4u: { return unpack2x16float(src[o + 3u]).y; }
        case 5u: { return unpack2x16float(src[o + 4u]).x; }
        case 9u: { return unpack2x16float(src[o + 4u]).y; }
        case 6u: { return unpack2x16float(src[o + 5u]).x; }
        case 7u: { return unpack2x16float(src[o + 5u]).y; }
        case 8u: { return unpack2x16float(src[o + 6u]).x; }
        case 10u: { return unpack2x16snorm(src[o + 7u]).x; }
        case 11u: { return unpack2x16snorm(src[o + 7u]).y; }
        case 12u: { return unpack2x16snorm(src[o + 8u]).x; }
        default: { return unpack2x16snorm(src[o + 8u]).y; }
    }
}";

    void EnsureRowPipelines()
    {
        if (_rowCompactPipeline != null) return;
        using var module = _device!.CreateShaderModule(new GPUShaderModuleDescriptor { Code = RowQWgsl });
        GPUComputePipeline Make(string entry) => _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage { Module = module, EntryPoint = entry },
        });
        _rowCompactPipeline = Make("compact_rows");
        _rowExpandPipeline = Make("expand_rows");
        _rowParams = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
    }

    void ConvertRows(GPUComputePipeline pipeline, GPUBuffer rows32, GPUBuffer rowsq, int count)
    {
        var p = new byte[16];
        BitConverter.TryWriteBytes(p.AsSpan(0), (uint)count);
        _queue!.WriteBuffer(_rowParams!, 0, p);
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var layout = pipeline.GetBindGroupLayout(0);
        using var bg = _device!.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = layout,
            Entries = new GPUBindGroupEntry[]
            {
                new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = rows32 } },
                new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = rowsq } },
                new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _rowParams! } },
            },
        });
        using var encoder = _device.CreateCommandEncoder();
        using (var pass = encoder.BeginComputePass())
        {
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, bg);
            uint groups = ((uint)count + 255) / 256;
            const uint maxWG = 65535u;
            pass.DispatchWorkgroups(Math.Min(groups, maxWG), (groups + maxWG - 1) / maxWG, 1);
            pass.End();
        }
        using var cmd = encoder.Finish();
        _submitArray[0] = cmd;
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
    }

    /// <summary>With <see cref="RowCompact"/>, before a frame: the f32 rows to compact rows, the f32 freed.</summary>
    void CompactRowsIfAsked()
    {
        // The sorter's count, and only when the renderer agrees: an upload sets the renderer's count after an await, and a
        // frame in that gap compacted a trainer's grown buffer at the OLD count (2026-10-11, densify: 165,438 of 198,447
        // rows kept). Uploads also hold compaction off (_rowsHold).
        int n = _sorter.SplatCount;
        if (!RowCompact || _rowsPinned || _rowsHold > 0 || _sorter.RowsCompact || _sorter.LodActive || n <= 0 || n != _splatCount
            || _device == null || _queue == null || _sorter.PackedDataBuf is not { } rows32 || rows32.Length < (long)n * SplatFormat.Floats) return;
        EnsureRowPipelines();
        var a = _gpu.WebGPUAccelerator;
        var rowsq = a.Allocate1D<uint>((long)n * SplatFormat.CompactWords);
        ConvertRows(_rowCompactPipeline!, rows32.GetGPUBuffer()!, rowsq.GetGPUBuffer()!, n);
        _sorter.SwapToCompactRows(rowsq);
        Console.WriteLine($"[Renderer] source rows kept compact (&rowq): {n:N0} splats, " +
            $"{(long)n * SplatFormat.Floats * 4 / (1024 * 1024)} -> {(long)n * SplatFormat.CompactWords * 4 / (1024 * 1024)} MB");
    }

    /// <summary>
    /// Someone reads the source rows as f32 (PackedSplatBuffer, a readback): expand them if compact, and keep them f32
    /// for the rest of this scene - the reader may hold the buffer across frames (training, Edit, export).
    /// </summary>
    void PinRowsF32()
    {
        _rowsPinned = true;
        if (!_sorter.RowsCompact || _sorter.CompactRows is not { } rowsq || _device == null || _queue == null) return;
        EnsureRowPipelines();
        int n = (int)(rowsq.Length / SplatFormat.CompactWords);   // the rows the codes hold, whatever the counts say
        var rows32 = _gpu.WebGPUAccelerator.Allocate1D<float>((long)n * SplatFormat.Floats);
        ConvertRows(_rowExpandPipeline!, rows32.GetGPUBuffer()!, rowsq.GetGPUBuffer()!, n);
        _sorter.SwapToF32Rows(rows32);
        Console.WriteLine($"[Renderer] source rows expanded to f32 for this scene ({n:N0} splats): a tool reads them");
    }

    /// <summary>
    /// The source rows as f32 for a brief READ (bounds, a distance) without keeping them f32: the live rows when they are
    /// f32 (compaction held off until the lease is disposed - the read awaits across frames), a temporary expanded copy
    /// when they are compact. Anything that keeps or writes the rows takes <see cref="PackedSplatBuffer"/>.
    /// </summary>
    public RowsLease BorrowRowsF32()
    {
        if (_sorter.RowsCompact && _sorter.CompactRows is { } rowsq && _device != null && _queue != null)
        {
            EnsureRowPipelines();
            int n = (int)(rowsq.Length / SplatFormat.CompactWords);
            var copy = _gpu.WebGPUAccelerator.Allocate1D<float>((long)n * SplatFormat.Floats);
            ConvertRows(_rowExpandPipeline!, copy.GetGPUBuffer()!, rowsq.GetGPUBuffer()!, n);
            return new RowsLease(this, copy, owned: true);
        }
        _rowsHold++;
        return new RowsLease(this, _sorter.PackedDataBuf, owned: false);
    }

    /// <summary>A brief f32 view of the source rows (<see cref="BorrowRowsF32"/>); dispose it when the read is done.</summary>
    public sealed class RowsLease : IDisposable
    {
        readonly GpuGaussianRenderer _r;
        readonly bool _owned;
        bool _done;
        public MemoryBuffer1D<float, Stride1D.Dense>? Rows { get; }

        internal RowsLease(GpuGaussianRenderer r, MemoryBuffer1D<float, Stride1D.Dense>? rows, bool owned)
        {
            _r = r; Rows = rows; _owned = owned;
        }

        public void Dispose()
        {
            if (_done) return;
            _done = true;
            if (_owned) Rows?.Dispose();
            else _r._rowsHold--;
        }
    }

    void DisposeRowQ()
    {
        _rowCompactPipeline?.Dispose(); _rowExpandPipeline?.Dispose();
        _rowParams?.Destroy(); _rowParams?.Dispose();
    }
}
