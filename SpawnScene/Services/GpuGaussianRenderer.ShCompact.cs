using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

// SH bands kept as 8-bit codes in VRAM (Research/viewer-memory-2026-10-11.md): 45 bytes a splat instead of 180, the
// largest item of the viewer's ~300 B/splat at SH 3. The pack pass dequantizes where it reads a coefficient
// (SphericalHarmonics.WgslPartAccessQ8). Measured before building it (&shq8sim=1, a fixed 1/128 step): 57.0 dB vs the
// f32 render on Train 7K, max 2/255; built (per-band step): 60.1 dB, max 1/255, 95 MiB less on that scene.
//
// The f32 parts stay the interface: every reader of them (ShRestBuffers, CopyShPartToIlgpu, the readbacks, LOD
// install) expands the codes back to f32 first, and the next draw compacts again. A scene edited, saved or trained
// from while compact therefore carries 8-bit-rounded bands from then on (round-trip stable bit for bit: sh_step;
// &shq8cycles checks it). LOD trees and the streamed pool stay f32 (the pager writes f32 rows).
public partial class GpuGaussianRenderer
{
    /// <summary>&amp;shq8=1: keep SH bands as 8-bit codes with one scale per band (bands 1..3).</summary>
    public static bool ShCompact { get; set; }

    /// <summary>&amp;shq8cycles=N (harness): after the first compaction, expand to f32 and compact again N times - what an
    /// Edit, save or train-from does. A round trip must not drift: the render is to be pixel-identical to N = 0.</summary>
    public static int ShCompactCycles { get; set; }

    bool _shPacked;          // _shRest holds 8-bit codes (15 a row a part, 4 to a u32), not f32
    int _shRows;             // the rows the codes hold (an expand allocates this many f32 rows)
    int _shHold;             // readbacks in flight: no compaction under them (they await between parts)
    GPUBuffer? _shScale;     // 4 x u32: bands 1..3 largest |v| (f32 bits), written on the GPU by band_max
    GPUBuffer[]? _shQParams;
    GPUComputePipeline? _shBandMax, _shPackQ8, _shUnpackQ8;

    const string ShQ8Wgsl = @"
// A SH part (15 floats a row: 5 basis functions x RGB, SphericalHarmonics.WgslPartAccess's layout) <-> 8-bit codes,
// 4 to a word, one step per band: code = round(v / s), v = code * s, s = sh_step(m), m = the band's largest |v|.
// The step s = m / 127 keeps 16 mantissa bits (rounded): code * s is then EXACT in f32 (7 + 17 bits), so an expanded
// band's largest |v| is exactly 127 * s, and m' / 127 rounds back to the same s even through the GPU's inexact
// division - an expand + compact (Edit, save, train-from) reproduces the codes and the values bit for bit.
// Coefficient j of part p is basis b = p * 5 + j / 3: band 1 (b < 3), 2 (b < 8), 3.
struct QParams { floats : u32, part : u32, words : u32, _pad : u32 }
@group(0) @binding(0) var<storage, read_write> f32s  : array<f32>;
@group(0) @binding(1) var<storage, read_write> codes : array<u32>;
@group(0) @binding(2) var<storage, read_write> amax  : array<atomic<u32>, 4>;
@group(0) @binding(3) var<uniform>             q     : QParams;
var<workgroup> wmax : array<atomic<u32>, 3>;

" + SphericalHarmonics.WgslShStep + @"
fn band_of(part : u32, j : u32) -> u32 {
    let b = part * 5u + j / 3u;
    return select(select(2u, 1u, b < 8u), 0u, b < 3u);
}

// Largest |v| per band: workgroup atomics, then one global atomic per band per workgroup. Non-negative floats order
// like their bit patterns; NaN / inf are left out (they would take the whole band's scale).
@compute @workgroup_size(256)
fn band_max(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_index) lid : u32,
            @builtin(num_workgroups) nwg : vec3<u32>) {
    if (lid < 3u) { atomicStore(&wmax[lid], 0u); }
    workgroupBarrier();
    let o = gid.y * nwg.x * 256u + gid.x;
    if (o < q.floats) {
        let a = abs(f32s[o]);
        if (a <= 3.0e38) { atomicMax(&wmax[band_of(q.part, o % 15u)], bitcast<u32>(a)); }
    }
    workgroupBarrier();
    if (lid < 3u) {
        let m = atomicLoad(&wmax[lid]);
        if (m != 0u) { atomicMax(&amax[lid], m); }
    }
}

@compute @workgroup_size(256)
fn pack_q8(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
    let w = gid.y * nwg.x * 256u + gid.x;
    if (w >= q.words) { return; }
    var word = 0u;
    for (var t = 0u; t < 4u; t++) {
        let o = w * 4u + t;
        if (o < q.floats) {
            let s = sh_step(bitcast<f32>(atomicLoad(&amax[band_of(q.part, o % 15u)])));
            var c = 0;
            if (s > 0.0) { c = i32(clamp(round(f32s[o] / s), -127.0, 127.0)); }
            word = word | ((bitcast<u32>(c) & 0xffu) << (8u * t));
        }
    }
    codes[w] = word;
}

@compute @workgroup_size(256)
fn unpack_q8(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
    let o = gid.y * nwg.x * 256u + gid.x;
    if (o >= q.floats) { return; }
    let s = sh_step(bitcast<f32>(atomicLoad(&amax[band_of(q.part, o % 15u)])));
    f32s[o] = f32(extractBits(bitcast<i32>(codes[o >> 2u]), (o & 3u) * 8u, 8u)) * s;
}
";

    static ulong ShCodeBytes(int rows) => ((ulong)rows * SphericalHarmonics.PartFloatsPerSplat + 3) / 4 * 4;

    void EnsureShQ8Pipelines()
    {
        if (_shBandMax != null) return;
        using var module = _device!.CreateShaderModule(new GPUShaderModuleDescriptor { Code = ShQ8Wgsl });
        GPUComputePipeline Make(string entry) => _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage { Module = module, EntryPoint = entry },
        });
        _shBandMax = Make("band_max");
        _shPackQ8 = Make("pack_q8");
        _shUnpackQ8 = Make("unpack_q8");
        _shScale = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
        });
        _shQParams = Enumerable.Range(0, SphericalHarmonics.Parts).Select(_ => _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        })).ToArray();
    }

    void WriteShQParams(int rows)
    {
        uint floats = (uint)rows * SphericalHarmonics.PartFloatsPerSplat;
        uint words = (uint)(ShCodeBytes(rows) / 4);
        for (int part = 0; part < SphericalHarmonics.Parts; part++)
        {
            var bytes = new byte[16];
            BitConverter.TryWriteBytes(bytes.AsSpan(0), floats);
            BitConverter.TryWriteBytes(bytes.AsSpan(4), (uint)part);
            BitConverter.TryWriteBytes(bytes.AsSpan(8), words);
            _queue!.WriteBuffer(_shQParams![part], 0, bytes);
        }
    }

    void DispatchShQ8(GPUCommandEncoder encoder, GPUComputePipeline pipeline, GPUBuffer f32s, GPUBuffer? codes, int part,
        uint threads)
    {
        using var layout = pipeline.GetBindGroupLayout(0);
        var entries = new List<GPUBindGroupEntry>
        {
            new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = f32s } },
            new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _shScale! } },
            new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _shQParams![part] } },
        };
        if (codes != null) entries.Add(new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = codes } });
        using var bg = _device!.CreateBindGroup(new GPUBindGroupDescriptor { Layout = layout, Entries = entries.ToArray() });
        using var pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bg);
        uint groups = (threads + 255) / 256;
        const uint maxWG = 65535u;
        pass.DispatchWorkgroups(Math.Min(groups, maxWG), (groups + maxWG - 1) / maxWG, 1);
        pass.End();
    }

    /// <summary>
    /// With <see cref="ShCompact"/>: turn the f32 SH parts into 8-bit codes (band scales found on the GPU, never read
    /// back) and free the f32. Called before each pack; a no-op when already compact, under an LOD tree or a readback.
    /// </summary>
    void CompactShIfAsked()
    {
        if (!ShCompact || _shPacked || _shHold > 0 || _shRest == null || _shDegree <= 0 || _sorter.LodActive
            || _device == null || _queue == null) return;
        int rows = (int)(_shRest[0].Size / (SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        if (rows <= 0) return;
        EnsureShQ8Pipelines();
        WriteShQParams(rows);
        _queue.WriteBuffer(_shScale!, 0, new byte[16]);
        var codes = new GPUBuffer[_shRest.Length];
        uint floats = (uint)rows * SphericalHarmonics.PartFloatsPerSplat;
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using (var encoder = _device.CreateCommandEncoder())
        {
            for (int part = 0; part < codes.Length; part++)
                DispatchShQ8(encoder, _shBandMax!, _shRest[part], null, part, floats);
            for (int part = 0; part < codes.Length; part++)
            {
                codes[part] = NewShPart(ShCodeBytes(rows));
                DispatchShQ8(encoder, _shPackQ8!, _shRest[part], codes[part], part, (uint)(ShCodeBytes(rows) / 4));
            }
            using var cmd = encoder.Finish();
            _submitArray[0] = cmd;
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
        }
        DestroyShRest();   // the f32 (freed once the passes above have run); keeps the degree
        _shRest = codes;
        _shPacked = true;
        _shRows = rows;
        _packBindGroup?.Dispose();
        _packBindGroup = null;
        Console.WriteLine($"[Renderer] SH bands kept as 8-bit codes (&shq8): {rows:N0} rows, " +
            $"{(long)rows * SphericalHarmonics.RestFloatsPerSplat * sizeof(float) / (1024 * 1024)} -> " +
            $"{(long)ShCodeBytes(rows) * codes.Length / (1024 * 1024)} MB");
        if (ShCompactCycles > 0)
        {
            ShCompactCycles--;
            ExpandShRest();
            CompactShIfAsked();
        }
    }

    /// <summary>The 8-bit codes back to f32 parts (the f32 interface every SH reader expects). No-op when not compact.</summary>
    void ExpandShRest()
    {
        if (!_shPacked || _shRest == null || _device == null || _queue == null) return;
        EnsureShQ8Pipelines();
        WriteShQParams(_shRows);
        var parts = new GPUBuffer[_shRest.Length];
        uint floats = (uint)_shRows * SphericalHarmonics.PartFloatsPerSplat;
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using (var encoder = _device.CreateCommandEncoder())
        {
            for (int part = 0; part < parts.Length; part++)
            {
                parts[part] = NewShPart((ulong)floats * sizeof(float));
                DispatchShQ8(encoder, _shUnpackQ8!, parts[part], _shRest[part], part, floats);
            }
            using var cmd = encoder.Finish();
            _submitArray[0] = cmd;
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, _submitArray);
        }
        SetShRest(parts, _shDegree);
    }

    void DisposeShQ8()
    {
        _shBandMax?.Dispose(); _shPackQ8?.Dispose(); _shUnpackQ8?.Dispose();
        _shScale?.Destroy(); _shScale?.Dispose();
        if (_shQParams != null) foreach (var b in _shQParams) { b.Destroy(); b.Dispose(); }
    }
}
