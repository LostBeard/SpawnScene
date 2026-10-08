using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Niantic's .spz (MIT, github.com/nianticlabs/spz; Scaniverse and others publish it) into SpawnScene's splat rows and SH
/// parts, decoded on the GPU from the gunzipped bytes. Versions 2 and 3, the gzip single-stream files in circulation
/// (v1's float16 positions were never released; v4 is a zstd container, not yet released and not decompressible by the
/// browser's DecompressionStream). Layout after the 16-byte header, attribute by attribute for all points: positions
/// (3 x 24-bit fixed point, fractionalBits), alphas (sigmoid x 255), colours (SH DC: (c/255 - 0.5) / 0.15), scales
/// (log: s/16 - 10), rotations (v2: x y z at r/127.5 - 1, w = sqrt(1 - |xyz|^2); v3: "smallest three" in 4 bytes), SH
/// ((s - 128) / 128, coefficient-major with the colour channel inner - our order; degree 4's band is dropped).
/// <para>
/// <b>Up:</b> the SPZ spec calls its frame RUB (y up), but the files in circulation carry the PLY's y-down frame - Spark's
/// own examples turn every one 180 degrees about X (<c>quaternion.set(1, 0, 0, 0)</c>), and its butterfly.spz drew upside
/// down here without it (2026-10-08). So <c>flipToYUp</c> applies the same turn as GaussianPlyImport (positions,
/// rotations, SH signs), on by default.
/// </para>
/// </summary>
public static class SpzImport
{
    public const uint Magic = 0x5053474e;   // "NGSP" little-endian

    public sealed record Header(int Version, int Count, int ShDegree, int FractionalBits, bool Antialiased)
    {
        public int ShDim => ShDegree switch { 0 => 0, 1 => 3, 2 => 8, 3 => 15, _ => 24 };
        public int RotBytes => Version >= 3 ? 4 : 3;
        public long PosOff => 16;
        public long AlphaOff => PosOff + 9L * Count;
        public long ColourOff => AlphaOff + Count;
        public long ScaleOff => ColourOff + 3L * Count;
        public long RotOff => ScaleOff + 3L * Count;
        public long ShOff => RotOff + (long)RotBytes * Count;
        public long Bytes => ShOff + 3L * ShDim * Count;
        /// <summary>The SH degree we keep (we draw up to 3).</summary>
        public int KeptShDegree => Math.Min(3, ShDegree);
    }

    /// <summary>Read the header of gunzipped SPZ bytes; <see cref="FormatException"/> with a reason when we cannot read it.</summary>
    public static Header ParseHeader(ReadOnlySpan<byte> d)
    {
        if (d.Length < 16 || BitConverter.ToUInt32(d) != Magic) throw new FormatException("not an SPZ file");
        int version = (int)BitConverter.ToUInt32(d[4..]);
        uint count = BitConverter.ToUInt32(d[8..]);
        int shDegree = d[12], frac = d[13], flags = d[14];
        if (version is < 2 or > 3)
            throw new FormatException(version == 1 ? "SPZ version 1 (float16 positions) is not read" :
                $"SPZ version {version} is not read yet (versions 2 and 3 are)");
        if (count == 0 || count > int.MaxValue / 16) throw new FormatException($"SPZ claims {count} points");
        if (shDegree > 4) throw new FormatException($"SPZ SH degree {shDegree}");
        if ((flags & 0x2) != 0) Console.WriteLine("[SPZ] the file has extensions; they are skipped");
        return new Header(version, (int)count, shDegree, frac, (flags & 0x1) != 0);
    }

    public struct Params
    {
        public int Count, PosOff, AlphaOff, ColourOff, ScaleOff, RotOff, ShOff, ShDim, KeptBands, Smallest3, Flip;
        public float PosScale;
    }

    static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _kernel;
    static Accelerator? _loadedFor;

    static uint B(ArrayView1D<uint, Stride1D.Dense> w, int addr) => (w[addr >> 2] >> ((addr & 3) * 8)) & 0xFFu;

    static void Kernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2, Params p)
    {
        if (i >= p.Count) return;
        int o = i * SplatFormat.Floats;
        for (int a = 0; a < 3; a++)
        {
            int at = p.PosOff + (i * 3 + a) * 3;
            int fixed32 = (int)(B(w, at) | (B(w, at + 1) << 8) | (B(w, at + 2) << 16));
            if ((fixed32 & 0x800000) != 0) fixed32 |= unchecked((int)0xFF000000);   // sign extension
            packed[o + a] = fixed32 * p.PosScale * (a > 0 && p.Flip != 0 ? -1f : 1f);
        }
        for (int c = 0; c < 3; c++)
        {
            packed[o + 3 + c] = (B(w, p.ColourOff + i * 3 + c) / 255f - 0.5f) / 0.15f;
            packed[o + 6 + c] = XMath.Exp(B(w, p.ScaleOff + i * 3 + c) / 16f - 10f);
        }
        packed[o + 9] = B(w, p.AlphaOff + i) / 255f;   // stored as sigmoid(alpha) x 255: already our linear opacity
        float qx, qy, qz, qw;
        if (p.Smallest3 != 0)
        {
            int at = p.RotOff + i * 4;
            uint comp = B(w, at) | (B(w, at + 1) << 8) | (B(w, at + 2) << 16) | (B(w, at + 3) << 24);
            int largest = (int)(comp >> 30);
            const uint Mask = (1u << 9) - 1u;
            float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f, sum = 0f;
            for (int k = 3; k >= 0; k--)
            {
                if (k == largest) continue;
                float v = 0.70710678f * (comp & Mask) / Mask;
                if (((comp >> 9) & 1u) != 0) v = -v;
                comp >>= 10;
                sum += v * v;
                if (k == 0) r0 = v; else if (k == 1) r1 = v; else if (k == 2) r2 = v; else r3 = v;
            }
            float big = XMath.Sqrt(XMath.Max(0f, 1f - sum));
            if (largest == 0) r0 = big; else if (largest == 1) r1 = big; else if (largest == 2) r2 = big; else r3 = big;
            qx = r0; qy = r1; qz = r2; qw = r3;
        }
        else
        {
            int at = p.RotOff + i * 3;
            qx = B(w, at) / 127.5f - 1f; qy = B(w, at + 1) / 127.5f - 1f; qz = B(w, at + 2) / 127.5f - 1f;
            qw = XMath.Sqrt(XMath.Max(0f, 1f - (qx * qx + qy * qy + qz * qz)));
        }
        float len = XMath.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (!(len > 1e-12f)) { qx = 0f; qy = 0f; qz = 0f; qw = 1f; len = 1f; }
        if (p.Flip != 0)
        {
            // (1,0,0,0) (x y z, w = 0) times q, as GaussianPlyImport.
            float nw = -qx, nx = qw, ny = -qz, nz = qy;
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        packed[o + 10] = qx / len; packed[o + 11] = qy / len; packed[o + 12] = qz / len; packed[o + 13] = qw / len;

        if (p.KeptBands == 0) return;
        int partBase = i * SphericalHarmonics.PartFloatsPerSplat;
        for (int f = 0; f < 45; f++)
        {
            int band = f / 3;   // 0-based coefficient index (band k = band + 1)
            float v = band < p.KeptBands ? (B(w, p.ShOff + (i * p.ShDim + band) * 3 + f % 3) - 128f) / 128f : 0f;
            if (p.Flip != 0 && ((GaussianPlyImport.FlipSignMask >> (band + 1)) & 1) != 0) v = -v;
            int part = f / SphericalHarmonics.PartFloatsPerSplat, idx = partBase + f % SphericalHarmonics.PartFloatsPerSplat;
            if (part == 0) sh0[idx] = v;
            else if (part == 1) sh1[idx] = v;
            else sh2[idx] = v;
        }
    }

    /// <summary>Decode all points from gunzipped bytes already on the device. Public for the CPU tests.</summary>
    public static void Run(Accelerator a, ArrayView1D<uint, Stride1D.Dense> words, Header h, ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2,
        bool flipToYUp = false)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _kernel = null; _loadedFor = a; }
        _kernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(Kernel);
        int kept = h.KeptShDegree switch { 0 => 0, 1 => 3, 2 => 8, _ => 15 };
        _kernel(h.Count, words, packed, sh0, sh1, sh2, new Params
        {
            Count = h.Count, PosOff = (int)h.PosOff, AlphaOff = (int)h.AlphaOff, ColourOff = (int)h.ColourOff,
            ScaleOff = (int)h.ScaleOff, RotOff = (int)h.RotOff, ShOff = (int)h.ShOff, ShDim = h.ShDim, KeptBands = kept,
            Smallest3 = h.Version >= 3 ? 1 : 0, PosScale = 1f / (1 << h.FractionalBits), Flip = flipToYUp ? 1 : 0,
        });
    }

    /// <summary>Gunzipped SPZ bytes (JS-side) to rows and SH parts on the device (the caller owns them; SH null at degree 0).</summary>
    public static async Task<(MemoryBuffer1D<float, Stride1D.Dense> Packed, MemoryBuffer1D<float, Stride1D.Dense>[]? Sh)> ConvertAsync(
        WebGPUAccelerator a, ArrayBuffer raw, Header h, bool flipToYUp)
    {
        if (h.Bytes > raw.ByteLength) throw new FormatException("the SPZ is shorter than its header says (cut off?)");
        if (h.Bytes > int.MaxValue - 16) throw new FormatException("SPZ data over 2 GB is not read yet");
        var packed = a.Allocate1D<float>((long)h.Count * SplatFormat.Floats);
        MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
        using var dummy = a.Allocate1D<float>(1);
        if (h.KeptShDegree > 0)
        {
            sh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
            for (int p = 0; p < sh.Length; p++) sh[p] = a.Allocate1D<float>((long)h.Count * SphericalHarmonics.PartFloatsPerSplat);
        }
        long full = h.Bytes & ~3L;
        using var words = a.Allocate1D<uint>(full / 4 + 2);
        var queue = a.NativeAccelerator.Queue!;
        a.FlushPendingCommands();
        // CPU transfer: none - the gunzipped bytes go from the browser's ArrayBuffer to the device.
        if (full > 0) queue.WriteBuffer(words.GetGPUBuffer()!, 0L, raw, 0, full);
        if (h.Bytes > full)
        {
            using var tail = new Uint8Array(4);
            using var src = new Uint8Array(raw, full, h.Bytes - full);
            tail.Set(src);
            using var tailBuf = tail.Buffer;
            queue.WriteBuffer(words.GetGPUBuffer()!, full, tailBuf, 0, 4L);
        }
        Run(a, words.View, h, packed.View, sh?[0].View ?? dummy.View, sh?[1].View ?? dummy.View, sh?[2].View ?? dummy.View, flipToYUp);
        await a.SynchronizeAsync();
        return (packed, sh);
    }
}
