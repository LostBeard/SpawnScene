using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Stable LSD radix sort of u32 key / u32 value pairs, in WGSL, 8 bits per pass.
///
/// Why not ILGPU.Algorithms' RadixSortPairs: it sorts 2 bits per pass - 16 passes over a u32 key, each a
/// clear, a histogram, a multi-kernel scan and a scatter, plus a gather and a scatter packing the pairs -
/// about 100 dispatches per sort. MEASURED (&amp;trainprofile=1, Truck 979x546, 184k-500k splats): the sort was
/// 101-126 ms of a 165-201 ms training step, 60-67% of it. This is 5 dispatches per 8-bit pass, every pass
/// recorded into ONE command buffer and submitted once.
///
/// Per pass (reduce-then-scan):
///   histogram   - per 1024-key block, 256 digit counts, stored digit-major: hist[digit * blocks + block]
///   scan_reduce / scan_sums / scan_down - exclusive scan of hist, so hist[d * blocks + b] becomes the
///                 output index of block b's first key with digit d
///   scatter     - each block ranks its keys stably and writes them. WebGPU has no guaranteed subgroups,
///                 so the rank is a ballot emulated in shared memory: every thread ORs its bit into its
///                 digit's 256-bit mask, and its rank is the popcount of the mask below it.
/// Stability comes from ranking in element order within a block and ordering blocks by the scan.
/// </summary>
public sealed class GpuRadixSort : IDisposable
{
    const int Block = 1024;          // keys per workgroup: 256 threads x 4
    const int MaxWorkgroupsPerDim = 65535;
    const int MaxPasses = 4;

    readonly GPUDevice _device;
    readonly GPUQueue _queue;
    GPUComputePipeline? _histogram, _scanReduce, _scanSums, _scanDown, _scatter;
    readonly GPUBuffer?[] _params = new GPUBuffer?[MaxPasses];
    GPUBuffer? _hist, _sums, _keysAlt, _valuesAlt;
    int _capacity;

    public GpuRadixSort(GPUDevice device, GPUQueue queue)
    {
        _device = device;
        _queue = queue;
    }

    /// <summary>Keys this sorter can take without reallocating.</summary>
    public int Capacity => _capacity;

    /// <summary>Size the scratch buffers (2 x 4 bytes per key plus ~1 byte per key of histogram).</summary>
    public void EnsureCapacity(int keys)
    {
        if (keys <= _capacity) return;
        EnsurePipelines();
        DestroyScratch();
        _capacity = keys;
        int blocks = Math.Max(1, (keys + Block - 1) / Block);
        long histLen = 256L * blocks;
        int scanBlocks = (int)((histLen + Block - 1) / Block);
        _hist = Storage(histLen * 4);
        _sums = Storage(Math.Max(1, scanBlocks) * 4L);
        _keysAlt = Storage((long)keys * 4);
        _valuesAlt = Storage((long)keys * 4);
    }

    /// <summary>
    /// Sort the first <paramref name="count"/> pairs of <paramref name="keys"/> / <paramref name="values"/>
    /// ascending by key, stably, on the low <paramref name="keyBits"/> bits. Records and submits one command
    /// buffer; the caller must have submitted whatever wrote the inputs (queue order does the rest).
    /// </summary>
    public void Sort(GPUBuffer keys, GPUBuffer values, int count, int keyBits = 32)
    {
        if (count <= 1) return;
        if (keyBits < 1 || keyBits > 32) throw new ArgumentOutOfRangeException(nameof(keyBits));
        EnsureCapacity(count);

        int blocks = (count + Block - 1) / Block;
        long histLen = 256L * blocks;
        int scanBlocks = (int)((histLen + Block - 1) / Block);
        int passes = (keyBits + 7) / 8;

        using var enc = _device.CreateCommandEncoder();
        GPUBuffer srcK = keys, srcV = values, dstK = _keysAlt!, dstV = _valuesAlt!;
        for (int pass = 0; pass < passes; pass++)
        {
            var prm = _params[pass] ??= _device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = 16,
                Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
            });
            // One params buffer per pass: every pass is in the same submission, so one shared buffer
            // would be read by all of them with the LAST value written.
            var bytes = new byte[16];
            BitConverter.TryWriteBytes(bytes.AsSpan(0), (uint)count);
            BitConverter.TryWriteBytes(bytes.AsSpan(4), (uint)(pass * 8));
            BitConverter.TryWriteBytes(bytes.AsSpan(8), (uint)blocks);
            BitConverter.TryWriteBytes(bytes.AsSpan(12), (uint)scanBlocks);
            _queue.WriteBuffer(prm, 0, bytes);

            Pass(enc, _histogram!, blocks, prm, srcK, _hist!);
            Pass(enc, _scanReduce!, scanBlocks, prm, _hist!, _sums!);
            Pass(enc, _scanSums!, 1, prm, _sums!);
            Pass(enc, _scanDown!, scanBlocks, prm, _hist!, _sums!);
            Pass(enc, _scatter!, blocks, prm, srcK, srcV, dstK, dstV, _hist!);

            (srcK, dstK) = (dstK, srcK);
            (srcV, dstV) = (dstV, srcV);
        }
        if (passes % 2 == 1)
        {
            // An odd pass count leaves the result in the scratch pair.
            enc.CopyBufferToBuffer(srcK, 0, keys, 0, (ulong)count * 4);
            enc.CopyBufferToBuffer(srcV, 0, values, 0, (ulong)count * 4);
        }
        using var cmd = enc.Finish();
        _queue.Submit(new[] { cmd });
    }

    void Pass(GPUCommandEncoder enc, GPUComputePipeline pipeline, int workgroups, params GPUBuffer[] buffers)
    {
        using var pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        using var layout = pipeline.GetBindGroupLayout(0);
        var entries = new GPUBindGroupEntry[buffers.Length];
        for (int i = 0; i < buffers.Length; i++)
            entries[i] = new GPUBindGroupEntry { Binding = (uint)i, Resource = new GPUBufferBinding { Buffer = buffers[i] } };
        using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor { Layout = layout, Entries = entries });
        pass.SetBindGroup(0, bg);
        int x = Math.Min(workgroups, MaxWorkgroupsPerDim);
        int y = (workgroups + MaxWorkgroupsPerDim - 1) / MaxWorkgroupsPerDim;
        pass.DispatchWorkgroups((uint)Math.Max(1, x), (uint)Math.Max(1, y), 1);
        pass.End();
    }

    GPUBuffer Storage(long bytes) => _device.CreateBuffer(new GPUBufferDescriptor
    {
        Size = (ulong)Math.Max(16, (bytes + 15) / 16 * 16),
        Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopySrc | GPUBufferUsage.CopyDst,
    });

    void EnsurePipelines()
    {
        if (_histogram != null) return;
        _histogram = Make("histogram");
        _scanReduce = Make("scan_reduce");
        _scanSums = Make("scan_sums");
        _scanDown = Make("scan_down");
        _scatter = Make("scatter");

        GPUComputePipeline Make(string entry)
        {
            using var module = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = Wgsl(entry) });
            return _device.CreateComputePipeline(new GPUComputePipelineDescriptor
            {
                Layout = "auto",
                Compute = new GPUProgrammableStage { Module = module, EntryPoint = entry },
            });
        }
    }

    // One module per entry point: "auto" layouts only include the bindings an entry point uses, and the
    // kernels bind different buffers at the same slots, so each gets its own declarations.
    static string Wgsl(string entry) => Common + entry switch
    {
        "histogram" => Histogram,
        "scan_reduce" => ScanReduce,
        "scan_sums" => ScanSums,
        "scan_down" => ScanDown,
        "scatter" => Scatter,
        _ => throw new ArgumentOutOfRangeException(nameof(entry)),
    };

    const string Common = @"
struct SortParams {
    n           : u32,   // keys to sort
    shift       : u32,   // bit offset of this pass's digit
    blocks      : u32,   // ceil(n / 1024)
    scan_blocks : u32,   // ceil(256 * blocks / 1024)
}
@group(0) @binding(0) var<uniform> p : SortParams;
const WG : u32 = 256u;
const PER : u32 = 4u;
const BLOCK : u32 = 1024u;
";

    const string Histogram = @"
@group(0) @binding(1) var<storage, read> keys_in : array<u32>;
@group(0) @binding(2) var<storage, read_write> hist : array<u32>;
var<workgroup> counts : array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram(@builtin(local_invocation_id) lid : vec3<u32>,
             @builtin(workgroup_id) wid : vec3<u32>,
             @builtin(num_workgroups) nwg : vec3<u32>) {
    let block = wid.x + wid.y * nwg.x;
    atomicStore(&counts[lid.x], 0u);
    workgroupBarrier();
    if (block < p.blocks) {
        for (var k = 0u; k < PER; k = k + 1u) {
            let i = block * BLOCK + k * WG + lid.x;
            if (i < p.n) {
                atomicAdd(&counts[(keys_in[i] >> p.shift) & 255u], 1u);
            }
        }
    }
    workgroupBarrier();
    if (block < p.blocks) {
        hist[lid.x * p.blocks + block] = atomicLoad(&counts[lid.x]);
    }
}
";

    const string ScanReduce = @"
@group(0) @binding(1) var<storage, read> data : array<u32>;
@group(0) @binding(2) var<storage, read_write> sums : array<u32>;
var<workgroup> red : array<u32, 256>;

@compute @workgroup_size(256)
fn scan_reduce(@builtin(local_invocation_id) lid : vec3<u32>,
               @builtin(workgroup_id) wid : vec3<u32>,
               @builtin(num_workgroups) nwg : vec3<u32>) {
    let b = wid.x + wid.y * nwg.x;
    let len = 256u * p.blocks;
    var s = 0u;
    if (b < p.scan_blocks) {
        let base = b * BLOCK + lid.x * PER;
        for (var k = 0u; k < PER; k = k + 1u) {
            if (base + k < len) { s = s + data[base + k]; }
        }
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
        workgroupBarrier();
    }
    if (lid.x == 0u && b < p.scan_blocks) { sums[b] = red[0]; }
}
";

    const string ScanSums = @"
@group(0) @binding(1) var<storage, read_write> sums : array<u32>;
var<workgroup> tmp : array<u32, 256>;
var<workgroup> carry : u32;

// One workgroup: exclusive scan of the per-block sums, 256 at a time with a running carry.
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_id) lid : vec3<u32>) {
    if (lid.x == 0u) { carry = 0u; }
    workgroupBarrier();
    let chunks = (p.scan_blocks + 255u) / 256u;
    for (var c = 0u; c < chunks; c = c + 1u) {
        let i = c * 256u + lid.x;
        var v = 0u;
        if (i < p.scan_blocks) { v = sums[i]; }
        tmp[lid.x] = v;
        workgroupBarrier();
        for (var off = 1u; off < 256u; off = off << 1u) {
            var add = 0u;
            if (lid.x >= off) { add = tmp[lid.x - off]; }
            workgroupBarrier();
            tmp[lid.x] = tmp[lid.x] + add;
            workgroupBarrier();
        }
        let base = carry;
        if (i < p.scan_blocks) { sums[i] = base + tmp[lid.x] - v; }
        workgroupBarrier();
        if (lid.x == 255u) { carry = base + tmp[255u]; }
        workgroupBarrier();
    }
}
";

    const string ScanDown = @"
@group(0) @binding(1) var<storage, read_write> data : array<u32>;
@group(0) @binding(2) var<storage, read> sums : array<u32>;
var<workgroup> tsum : array<u32, 256>;

@compute @workgroup_size(256)
fn scan_down(@builtin(local_invocation_id) lid : vec3<u32>,
             @builtin(workgroup_id) wid : vec3<u32>,
             @builtin(num_workgroups) nwg : vec3<u32>) {
    let b = wid.x + wid.y * nwg.x;
    let len = 256u * p.blocks;
    let live = b < p.scan_blocks;
    let base = b * BLOCK + lid.x * PER;
    var v : array<u32, 4>;
    var s = 0u;
    for (var k = 0u; k < PER; k = k + 1u) {
        var x = 0u;
        if (live && base + k < len) { x = data[base + k]; }
        v[k] = x;
        s = s + x;
    }
    tsum[lid.x] = s;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
        var add = 0u;
        if (lid.x >= off) { add = tsum[lid.x - off]; }
        workgroupBarrier();
        tsum[lid.x] = tsum[lid.x] + add;
        workgroupBarrier();
    }
    if (live) {
        var run = sums[b] + tsum[lid.x] - s;
        for (var k = 0u; k < PER; k = k + 1u) {
            if (base + k < len) { data[base + k] = run; }
            run = run + v[k];
        }
    }
}
";

    const string Scatter = @"
@group(0) @binding(1) var<storage, read> keys_in : array<u32>;
@group(0) @binding(2) var<storage, read> vals_in : array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out : array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out : array<u32>;
@group(0) @binding(5) var<storage, read> offsets : array<u32>;
var<workgroup> masks : array<atomic<u32>, 2048>;   // 256 digits x 256 thread bits
var<workgroup> running : array<u32, 256>;          // next output index per digit for this block

@compute @workgroup_size(256)
fn scatter(@builtin(local_invocation_id) lid : vec3<u32>,
           @builtin(workgroup_id) wid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let block = wid.x + wid.y * nwg.x;
    let valid_block = block < p.blocks;
    if (valid_block) { running[lid.x] = offsets[lid.x * p.blocks + block]; }
    let word = lid.x >> 5u;
    let bit = 1u << (lid.x & 31u);
    for (var k = 0u; k < PER; k = k + 1u) {
        for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&masks[lid.x * 8u + w], 0u); }
        workgroupBarrier();
        let i = block * BLOCK + k * WG + lid.x;
        let live = valid_block && i < p.n;
        var key = 0u;
        var d = 0u;
        if (live) {
            key = keys_in[i];
            d = (key >> p.shift) & 255u;
            atomicOr(&masks[d * 8u + word], bit);
        }
        workgroupBarrier();
        var rank = 0u;
        var total = 0u;
        if (live) {
            for (var w = 0u; w < 8u; w = w + 1u) {
                let c = countOneBits(atomicLoad(&masks[d * 8u + w]));
                total = total + c;
                if (w < word) { rank = rank + c; }
            }
            rank = rank + countOneBits(atomicLoad(&masks[d * 8u + word]) & (bit - 1u));
            let dst = running[d] + rank;
            keys_out[dst] = key;
            vals_out[dst] = vals_in[i];
        }
        workgroupBarrier();
        // The last key of each digit in this chunk advances the digit's cursor for the next chunk.
        if (live && rank + 1u == total) { running[d] = running[d] + total; }
        workgroupBarrier();
    }
}
";

    /// <summary>Free the scratch buffers; the next Sort reallocates them. Pipelines are kept.</summary>
    public void ReleaseScratch() => DestroyScratch();

    void DestroyScratch()
    {
        foreach (var b in new[] { _hist, _sums, _keysAlt, _valuesAlt })
        {
            b?.Destroy();
            b?.Dispose();
        }
        _hist = _sums = _keysAlt = _valuesAlt = null;
        _capacity = 0;
    }

    public void Dispose()
    {
        DestroyScratch();
        for (int i = 0; i < _params.Length; i++) { _params[i]?.Destroy(); _params[i]?.Dispose(); _params[i] = null; }
        _histogram?.Dispose(); _scanReduce?.Dispose(); _scanSums?.Dispose(); _scanDown?.Dispose(); _scatter?.Dispose();
        _histogram = _scanReduce = _scanSums = _scanDown = _scatter = null;
    }
}
