using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Formats;

namespace SpawnScene.Services;

/// <summary>
/// A 3DGS .ply (GaussianPly) into SpawnScene's splat rows (SplatFormat: position, SH DC, linear scale, linear opacity,
/// quaternion x y z w) and SH-rest parts (SphericalHarmonics: band-major, three parts), ON THE GPU: the file's bytes go
/// from the browser's ArrayBuffer to the device in chunks and a kernel converts each vertex. A 30K-iteration scene's
/// .ply is 0.5-0.8 GB - copied through .NET it would not fit the wasm heap.
/// <para>
/// <b>Up:</b> a 3DGS .ply is in its SfM frame (COLMAP: y DOWN, z forward - "RDF"); SpawnScene, SPZ and WebXR are y up.
/// <c>flipToYUp</c> turns the scene 180 degrees about X (y, z negated): positions, rotations
/// (q' = (1,0,0,0) * q) and SH (that turn is diagonal in the real SH basis: each coefficient keeps or flips its sign by
/// whether its basis function is odd in y and z together - <see cref="FlipSignMask"/>).
/// </para>
/// </summary>
public static class GaussianPlyImport
{
    /// <summary>Bit k set: SH coefficient k (1..15, the reference's order) changes sign under (x, y, z) -> (x, -y, -z).
    /// Band 1: y, z, x -> -, -, +. Band 2: xy, yz, 3z^2-r^2, xz, x^2-y^2 -> -, +, +, -, +. Band 3: y(3x^2-y^2), xyz,
    /// y(4z^2-x^2-y^2), z(2z^2-3x^2-3y^2), x(4z^2-x^2-y^2), z(x^2-y^2), x(x^2-3y^2) -> -, +, -, -, +, -, +.</summary>
    public const int FlipSignMask = (1 << 1) | (1 << 2) | (1 << 4) | (1 << 7) | (1 << 9) | (1 << 11) | (1 << 12) | (1 << 14);

    public struct Params
    {
        public int Count, V0, First, Stride;
        public int X, Y, Z, Dc0, Dc1, Dc2, Opacity, S0, S1, S2, R0, R1, R2, R3, RestFirst, RestPerChannel, Flip;
    }

    static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _kernel;
    static Accelerator? _loadedFor;

    /// <summary>The u32 at any byte address (a PLY header has any length, so vertex fields are not word aligned).</summary>
    static uint ReadU32(ArrayView1D<uint, Stride1D.Dense> w, int byteAddr)
    {
        int wi = byteAddr >> 2;
        int sh = (byteAddr & 3) * 8;
        uint a = w[wi];
        if (sh == 0) return a;
        uint b = w[wi + 1];
        return (a >> sh) | (b << (32 - sh));
    }

    static float F(ArrayView1D<uint, Stride1D.Dense> w, int byteAddr) => Interop.IntAsFloat(ReadU32(w, byteAddr));

    static void Kernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2, Params p)
    {
        if (i >= p.Count) return;
        int b = p.First + i * p.Stride;
        float sgn = p.Flip != 0 ? -1f : 1f;
        int o = (p.V0 + i) * SplatFormat.Floats;
        float x = F(w, b + p.X), y = F(w, b + p.Y), z = F(w, b + p.Z);
        // |v| <= 3e38 is false for NaN and both infinities.
        bool finite = XMath.Abs(x) <= 3e38f && XMath.Abs(y) <= 3e38f && XMath.Abs(z) <= 3e38f;
        packed[o] = finite ? x : 0f; packed[o + 1] = finite ? y * sgn : 0f; packed[o + 2] = finite ? z * sgn : 0f;
        packed[o + 3] = F(w, b + p.Dc0); packed[o + 4] = F(w, b + p.Dc1); packed[o + 5] = F(w, b + p.Dc2);
        packed[o + 6] = XMath.Exp(XMath.Clamp(F(w, b + p.S0), -30f, 30f));
        packed[o + 7] = XMath.Exp(XMath.Clamp(F(w, b + p.S1), -30f, 30f));
        packed[o + 8] = XMath.Exp(XMath.Clamp(F(w, b + p.S2), -30f, 30f));
        float logit = XMath.Clamp(F(w, b + p.Opacity), -30f, 30f);
        packed[o + 9] = finite ? 1f / (1f + XMath.Exp(-logit)) : 0f;   // a non-finite splat is kept, invisible
        // PLY: rot_0 = w, rot_1..3 = x y z, not normalised. Ours: x y z w, unit.
        float qw = F(w, b + p.R0), qx = F(w, b + p.R1), qy = F(w, b + p.R2), qz = F(w, b + p.R3);
        float len = XMath.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (!(len > 1e-12f)) { qw = 1f; qx = 0f; qy = 0f; qz = 0f; len = 1f; }
        qw /= len; qx /= len; qy /= len; qz /= len;
        if (p.Flip != 0)
        {
            // (1,0,0,0) (x y z, w = 0) times q: w' = -qx, v' = (qw, -qz, qy).
            float nw = -qx, nx = qw, ny = -qz, nz = qy;
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        packed[o + 10] = qx; packed[o + 11] = qy; packed[o + 12] = qz; packed[o + 13] = qw;

        if (p.RestPerChannel == 0) return;
        int partBase = (p.V0 + i) * SphericalHarmonics.PartFloatsPerSplat;
        for (int k = 1; k <= 15; k++)
            for (int c = 0; c < 3; c++)
            {
                // PLY f_rest: channel-major (c * bandsInFile + band). Ours: band-major, (band - 1) * 3 + c, in parts of 15.
                float v = k <= p.RestPerChannel ? F(w, b + p.RestFirst + 4 * (c * p.RestPerChannel + k - 1)) : 0f;
                if (p.Flip != 0 && ((FlipSignMask >> k) & 1) != 0) v = -v;
                int f = (k - 1) * 3 + c;
                int part = f / SphericalHarmonics.PartFloatsPerSplat, idx = partBase + f % SphericalHarmonics.PartFloatsPerSplat;
                if (part == 0) sh0[idx] = v;
                else if (part == 1) sh1[idx] = v;
                else sh2[idx] = v;
            }
    }

    /// <summary>Convert vertices [v0, v0 + count) whose bytes sit in <paramref name="words"/> from byte
    /// <paramref name="first"/> on (the caller uploaded the chunk). Public for the CPU tests.</summary>
    public static void RunChunk(Accelerator a, ArrayView1D<uint, Stride1D.Dense> words, int first, int v0, int count,
        GaussianPly.Layout L, bool flipToYUp, ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _kernel = null; _loadedFor = a; }
        _kernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(Kernel);
        _kernel(count, words, packed, sh0, sh1, sh2, new Params
        {
            Count = count, V0 = v0, First = first, Stride = L.StrideBytes,
            X = L.X, Y = L.Y, Z = L.Z, Dc0 = L.Dc0, Dc1 = L.Dc1, Dc2 = L.Dc2, Opacity = L.Opacity,
            S0 = L.Scale0, S1 = L.Scale1, S2 = L.Scale2, R0 = L.Rot0, R1 = L.Rot1, R2 = L.Rot2, R3 = L.Rot3,
            RestFirst = L.RestFirst, RestPerChannel = L.RestPerChannel, Flip = flipToYUp ? 1 : 0,
        });
    }

    /// <summary>Bytes of the file on the device at once.</summary>
    const int ChunkBytes = 128 << 20;

    /// <summary>
    /// The whole file (<paramref name="file"/>, JS-side) to rows and SH parts on the device (the caller owns them; SH null
    /// at degree 0). The bytes cross to the GPU a chunk at a time and never enter .NET.
    /// </summary>
    public static async Task<(MemoryBuffer1D<float, Stride1D.Dense> Packed, MemoryBuffer1D<float, Stride1D.Dense>[]? Sh)> ConvertAsync(
        WebGPUAccelerator a, ArrayBuffer file, GaussianPly.Layout L, bool flipToYUp, Action<double>? progress = null)
    {
        if (L.Count > int.MaxValue / SplatFormat.Floats) throw new FormatException($"{L.Count:N0} splats is more than one scene holds");
        if (L.HeaderBytes + L.DataBytes > file.ByteLength) throw new FormatException("the PLY is shorter than its header says (cut off?)");
        if (L.HeaderBytes + L.DataBytes > int.MaxValue) throw new FormatException("PLY files over 2 GB are not read yet");
        int n = (int)L.Count;
        var packed = a.Allocate1D<float>(Math.Max(1L, (long)n * SplatFormat.Floats));
        MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
        using var dummy = a.Allocate1D<float>(1);
        if (L.ShDegree > 0)
        {
            sh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
            for (int p = 0; p < sh.Length; p++) sh[p] = a.Allocate1D<float>(Math.Max(1L, (long)n * SphericalHarmonics.PartFloatsPerSplat));
        }
        int perChunk = Math.Max(1, (ChunkBytes - 8) / L.StrideBytes);
        // One vertex past the chunk's last word: ReadU32 of an unaligned field reads the next word too.
        using var chunk = a.Allocate1D<uint>(ChunkBytes / 4 + L.StrideBytes / 4 + 4);
        var queue = a.NativeAccelerator.Queue!;
        var gpuChunk = chunk.GetGPUBuffer()!;
        for (int v0 = 0; v0 < n; v0 += perChunk)
        {
            int count = Math.Min(perChunk, n - v0);
            long start = L.HeaderBytes + (long)v0 * L.StrideBytes;
            long baseByte = start & ~3L;
            long end = start + (long)count * L.StrideBytes;
            long full = (end - baseByte) & ~3L;
            a.FlushPendingCommands();   // the previous chunk's kernel is queued before its bytes are overwritten
            if (full > 0) queue.WriteBuffer(gpuChunk, 0L, file, (int)baseByte, full);
            if (end - baseByte > full)
            {
                // The last 1-3 bytes of the file: writeBuffer moves whole words, so pad them to one.
                using var tail = new Uint8Array(4);
                using var src = new Uint8Array(file, baseByte + full, end - baseByte - full);
                tail.Set(src);
                using var tailBuf = tail.Buffer;
                queue.WriteBuffer(gpuChunk, full, tailBuf, 0, 4L);
            }
            RunChunk(a, chunk.View, (int)(start - baseByte), v0, count, L, flipToYUp, packed.View,
                sh?[0].View ?? dummy.View, sh?[1].View ?? dummy.View, sh?[2].View ?? dummy.View);
            progress?.Invoke((v0 + count) / (double)n);
        }
        await a.SynchronizeAsync();
        return (packed, sh);
    }
}
