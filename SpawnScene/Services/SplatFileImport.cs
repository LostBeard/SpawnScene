using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// antimatter15's .splat (github.com/antimatter15/splat convert.py; many web viewers read and write it) into SpawnScene's
/// splat rows, on the GPU. No header: 32 bytes a splat - position f32 x3, scale f32 x3 (LINEAR, exp of the PLY's log),
/// colour u8 x4 (rgb = 0.5 + C0 * f_dc, a = sigmoid(opacity)), rotation u8 x4 (w x y z of the unit quaternion, q * 128 + 128).
/// No SH bands. Converted from a 3DGS PLY without turning, so it is y down like one: <c>flipToYUp</c> as the PLY import.
/// (Formats/SplatParser, the legacy Viewer's reader, read the rotation as signed bytes and called the scale log-space.)
/// </summary>
public static class SplatFileImport
{
    public const int BytesPerSplat = 32;

    static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _kernel;
    static Accelerator? _loadedFor;

    static void Kernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> packed, int count, int flip)
    {
        if (i >= count) return;
        int b = i * 8;   // 32 bytes = 8 words, always aligned
        float sgn = flip != 0 ? -1f : 1f;
        int o = i * SplatFormat.Floats;
        packed[o] = Interop.IntAsFloat(w[b]);
        packed[o + 1] = Interop.IntAsFloat(w[b + 1]) * sgn;
        packed[o + 2] = Interop.IntAsFloat(w[b + 2]) * sgn;
        packed[o + 6] = XMath.Abs(Interop.IntAsFloat(w[b + 3]));
        packed[o + 7] = XMath.Abs(Interop.IntAsFloat(w[b + 4]));
        packed[o + 8] = XMath.Abs(Interop.IntAsFloat(w[b + 5]));
        uint c = w[b + 6], r = w[b + 7];
        const float C0 = 0.28209479177387814f;
        packed[o + 3] = ((c & 0xFFu) / 255f - 0.5f) / C0;
        packed[o + 4] = (((c >> 8) & 0xFFu) / 255f - 0.5f) / C0;
        packed[o + 5] = (((c >> 16) & 0xFFu) / 255f - 0.5f) / C0;
        packed[o + 9] = ((c >> 24) & 0xFFu) / 255f;
        float qw = ((r & 0xFFu) - 128f) / 128f, qx = (((r >> 8) & 0xFFu) - 128f) / 128f;
        float qy = (((r >> 16) & 0xFFu) - 128f) / 128f, qz = (((r >> 24) & 0xFFu) - 128f) / 128f;
        float len = XMath.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (!(len > 1e-6f)) { qw = 1f; qx = 0f; qy = 0f; qz = 0f; len = 1f; }
        qw /= len; qx /= len; qy /= len; qz /= len;
        if (flip != 0)
        {
            float nw = -qx, nx = qw, ny = -qz, nz = qy;   // (1,0,0,0) * q, as GaussianPlyImport
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        packed[o + 10] = qx; packed[o + 11] = qy; packed[o + 12] = qz; packed[o + 13] = qw;
    }

    /// <summary>Decode <paramref name="count"/> splats already on the device. Public for the CPU tests.</summary>
    public static void Run(Accelerator a, ArrayView1D<uint, Stride1D.Dense> words, int count, bool flipToYUp, ArrayView1D<float, Stride1D.Dense> packed)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _kernel = null; _loadedFor = a; }
        _kernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(Kernel);
        _kernel(count, words, packed, count, flipToYUp ? 1 : 0);
    }

    /// <summary>The file (JS-side) to rows on the device, a chunk at a time (the caller owns the result).</summary>
    public static async Task<MemoryBuffer1D<float, Stride1D.Dense>> ConvertAsync(WebGPUAccelerator a, ArrayBuffer file, bool flipToYUp)
    {
        long n = file.ByteLength / BytesPerSplat;
        if (n <= 0 || n > int.MaxValue / SplatFormat.Floats) throw new FormatException($"{n:N0} splats");
        var packed = a.Allocate1D<float>(n * SplatFormat.Floats);
        const int PerChunk = (128 << 20) / BytesPerSplat;
        using var chunk = a.Allocate1D<uint>(PerChunk * 8);
        var queue = a.NativeAccelerator.Queue!;
        for (long v0 = 0; v0 < n; v0 += PerChunk)
        {
            int count = (int)Math.Min(PerChunk, n - v0);
            a.FlushPendingCommands();   // the previous chunk's kernel is queued before its bytes are overwritten
            queue.WriteBuffer(chunk.GetGPUBuffer()!, 0L, file, (int)(v0 * BytesPerSplat), (long)count * BytesPerSplat);
            Run(a, chunk.View, count, flipToYUp, packed.View.SubView(v0 * SplatFormat.Floats, (long)count * SplatFormat.Floats));
        }
        await a.SynchronizeAsync();
        return packed;
    }
}
