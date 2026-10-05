using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnScene.Services;

/// <summary>
/// Quantized splat encoding for .spawnscene v2 (SPZ-style: Niantic's format stores 24-bit positions, 8-bit alpha, 8-bit
/// log scales, smallest-three rotations and 4-5 bit SH and reports ~10x smaller than PLY with no visible loss). A splat
/// is 236 bytes raw here (14 packed floats + 45 SH floats); encoded it is 19 words = 76 bytes, in three streams the file
/// gzips separately: geometry (4 words), appearance (3) and SH (12). Chrome cancelled the download of a raw 737 MB export
/// (bicycle, 3.28M splats, 2026-10-04).
/// <para>
/// Per splat, all 4-byte aligned so a GPU kernel writes words:
/// geometry  w0 = x (24 bits, see <see cref="QuantPosP"/>) | opacity (8) &lt;&lt; 24; w1 = y (24); w2 = z (24);
///           w3 = rotation, smallest three of the unit quaternion: largest component's index (2 bits, 30-31) and the
///           other three in order (10 bits each, [-1/sqrt2, 1/sqrt2]), the largest made positive.
/// appearance w4 = f16 log scale x | f16 log scale y &lt;&lt; 16; w5 = f16 log scale z | f16 colour r &lt;&lt; 16;
///           w6 = f16 colour g | f16 colour b &lt;&lt; 16 (colour is whatever the packed slots hold: RGB or SH DC).
/// SH        45 coefficients as bytes round(clamp(v, -1, 1) * 127) + 128, 4 to a word, 12 words (3 spare bytes) - finer
///           than SPZ's 4-5 bits.
/// </para>
/// Encode and decode are plain static functions over scalars, so the ILGPU kernels and the CPU tests run the same code.
/// </summary>
public static class SceneCodec
{
    public const int GeoWords = 4, AppWords = 3, ShWords = 12;
    public const float ShRange = 1f;
    const float InvSqrt2 = 0.70710678118654752f;
    const uint PosMax = (1u << 24) - 1u;

    /// <summary>
    /// Quantization frame. Per axis an inner box [Min, Min + Size] gets linear codes, and when <see cref="Piecewise"/> is
    /// set the splats outside it get log-spaced codes out to TailLo below and TailHi above (<see cref="QuantPosP"/>).
    /// Linear over the whole bounds let a few stray splats set the step for everything: the Truck's bounds run to
    /// z = 13,737 while 98% of it is within 25 units, so the step was 1.2e-3 units and the truck's edges moved a median
    /// 0.3 px, up to 13.7 px, after a round trip (2026-10-04).
    /// </summary>
    public struct Frame
    {
        public float MinX, MinY, MinZ, SizeX, SizeY, SizeZ;
        public float TailLoX, TailLoY, TailLoZ, TailHiX, TailHiY, TailHiZ;
        public int Count, Piecewise;

        static float Size(float lo, float hi) => hi > lo ? hi - lo : 1f;

        /// <summary>Linear over <paramref name="b"/> (files written before the inner box: Header2.Inner null).</summary>
        public static Frame From(SplatBounds.Aabb b, int count) => new Frame
        {
            MinX = b.MinX, MinY = b.MinY, MinZ = b.MinZ,
            SizeX = Size(b.MinX, b.MaxX), SizeY = Size(b.MinY, b.MaxY), SizeZ = Size(b.MinZ, b.MaxZ),
            Count = count,
        };

        /// <summary>Linear over <paramref name="inner"/> (clipped to <paramref name="outer"/>), log tails out to <paramref name="outer"/>.</summary>
        public static Frame From(SplatBounds.Aabb outer, SplatBounds.Aabb inner, int count)
        {
            float lx = MathF.Max(inner.MinX, outer.MinX), ly = MathF.Max(inner.MinY, outer.MinY), lz = MathF.Max(inner.MinZ, outer.MinZ);
            float hx = MathF.Min(inner.MaxX, outer.MaxX), hy = MathF.Min(inner.MaxY, outer.MaxY), hz = MathF.Min(inner.MaxZ, outer.MaxZ);
            if (hx <= lx) { lx = outer.MinX; hx = outer.MaxX; }
            if (hy <= ly) { ly = outer.MinY; hy = outer.MaxY; }
            if (hz <= lz) { lz = outer.MinZ; hz = outer.MaxZ; }
            var f = new Frame
            {
                MinX = lx, MinY = ly, MinZ = lz,
                SizeX = Size(lx, hx), SizeY = Size(ly, hy), SizeZ = Size(lz, hz),
                Count = count, Piecewise = 1,
            };
            f.TailLoX = MathF.Max(0f, lx - outer.MinX); f.TailHiX = MathF.Max(0f, outer.MaxX - (lx + f.SizeX));
            f.TailLoY = MathF.Max(0f, ly - outer.MinY); f.TailHiY = MathF.Max(0f, outer.MaxY - (ly + f.SizeY));
            f.TailLoZ = MathF.Max(0f, lz - outer.MinZ); f.TailHiZ = MathF.Max(0f, outer.MaxZ - (lz + f.SizeZ));
            return f;
        }

        /// <summary>The inner box as min x,y,z then max x,y,z (Header2.Inner).</summary>
        public readonly float[] InnerArray() => new[] { MinX, MinY, MinZ, MinX + SizeX, MinY + SizeY, MinZ + SizeZ };
    }

    // ── scalar building blocks (ILGPU-safe) ─────────────────────────────────────────────────────────────────

    public static uint QuantPos(float v, float min, float size)
    {
        float t = (v - min) / size;
        t = t < 0f ? 0f : t > 1f ? 1f : t;
        // Clamped: at t = 1, PosMax + 0.5 rounds UP to 2^24 in float, and masking that to 24 bits wrapped the far edge of
        // the bounds to the near one (caught by SceneCodecTests).
        uint q = (uint)(t * PosMax + 0.5f);
        return q > PosMax ? PosMax : q;
    }

    public static float DequantPos(uint q, float min, float size) => min + (q & PosMax) / (float)PosMax * size;

    /// <summary>Codes per log tail; the inner box gets the other 2^24 - 2^21 (step = size / 14.7M).</summary>
    const uint TailCodes = 1u << 20;
    const uint InnerCodes = PosMax - 2u * TailCodes;
    /// <summary>Log tails are spaced in units of size / 16, so a code just outside the box is ~1e-5 size.</summary>
    const float TailScaleDiv = 16f;

    /// <summary>
    /// Piecewise position code: [min, min + size] linear over the middle 2^24 - 2^21 codes; below and above, the
    /// distance e past the box as u = log(1 + e/s) / log(1 + tail/s), s = size/16, over 2^20 codes each. Floaters keep
    /// relative precision without setting the step inside the box.
    /// </summary>
    public static uint QuantPosP(float v, float min, float size, float tailLo, float tailHi)
    {
        float hi = min + size;
        if (v < min && tailLo > 0f)
        {
            float s = size / TailScaleDiv;
            float u = XMath.Log(1f + (min - v) / s) / XMath.Log(1f + tailLo / s);
            u = u > 1f ? 1f : u;
            return TailCodes - 1u - (uint)(u * (TailCodes - 1u) + 0.5f);
        }
        if (v > hi && tailHi > 0f)
        {
            float s = size / TailScaleDiv;
            float u = XMath.Log(1f + (v - hi) / s) / XMath.Log(1f + tailHi / s);
            u = u > 1f ? 1f : u;
            return PosMax - TailCodes + 1u + (uint)(u * (TailCodes - 1u) + 0.5f);
        }
        float t = (v - min) / size;
        t = t < 0f ? 0f : t > 1f ? 1f : t;
        uint q = (uint)(t * InnerCodes + 0.5f);
        return TailCodes + (q > InnerCodes ? InnerCodes : q);
    }

    public static float DequantPosP(uint q, float min, float size, float tailLo, float tailHi)
    {
        q &= PosMax;
        float s = size / TailScaleDiv;
        if (q < TailCodes)
        {
            float u = (TailCodes - 1u - q) / (float)(TailCodes - 1u);
            return min - s * (XMath.Exp(u * XMath.Log(1f + tailLo / s)) - 1f);
        }
        if (q > TailCodes + InnerCodes)
        {
            float u = (q - (PosMax - TailCodes + 1u)) / (float)(TailCodes - 1u);
            return min + size + s * (XMath.Exp(u * XMath.Log(1f + tailHi / s)) - 1f);
        }
        return min + (q - TailCodes) / (float)InnerCodes * size;
    }

    static uint EncodeAxis(float v, float min, float size, float tailLo, float tailHi, int piecewise)
        => piecewise != 0 ? QuantPosP(v, min, size, tailLo, tailHi) : QuantPos(v, min, size);

    static float DecodeAxis(uint q, float min, float size, float tailLo, float tailHi, int piecewise)
        => piecewise != 0 ? DequantPosP(q, min, size, tailLo, tailHi) : DequantPos(q, min, size);

    /// <summary>IEEE half from float, round to nearest even-ish (adds half an ulp), flushing below the half range to 0.</summary>
    public static uint FloatToHalf(float f)
    {
        uint x = Interop.FloatAsInt(f);
        uint sign = (x >> 16) & 0x8000u;
        int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
        uint mant = x & 0x7FFFFFu;
        if (exp <= 0) return sign;                                   // below the half normal range: 0
        if (exp >= 31) return sign | 0x7C00u;                        // overflow: inf (none of our values get here)
        uint h = sign | ((uint)exp << 10) | (mant >> 13);
        if ((mant & 0x1000u) != 0u) h += 1u;                         // round on the first dropped bit
        return h;
    }

    public static float HalfToFloat(uint h)
    {
        uint sign = (h & 0x8000u) << 16;
        uint exp = (h >> 10) & 0x1Fu;
        uint mant = h & 0x3FFu;
        if (exp == 0u) return Interop.IntAsFloat(sign);              // zero (we never write subnormals)
        if (exp == 31u) return Interop.IntAsFloat(sign | 0x7F800000u | (mant << 13));
        return Interop.IntAsFloat(sign | ((exp - 15u + 127u) << 23) | (mant << 13));
    }

    public static uint PackRotation(float x, float y, float z, float w)
    {
        float len = XMath.Sqrt(x * x + y * y + z * z + w * w);
        if (len < 1e-12f) { x = 0f; y = 0f; z = 0f; w = 1f; len = 1f; }
        x /= len; y /= len; z /= len; w /= len;
        float ax = XMath.Abs(x), ay = XMath.Abs(y), az = XMath.Abs(z), aw = XMath.Abs(w);
        uint largest = 0u; float best = ax;
        if (ay > best) { largest = 1u; best = ay; }
        if (az > best) { largest = 2u; best = az; }
        if (aw > best) { largest = 3u; }
        float lv = largest == 0u ? x : largest == 1u ? y : largest == 2u ? z : w;
        if (lv < 0f) { x = -x; y = -y; z = -z; w = -w; }
        // The other three, in x, y, z, w order.
        float c0 = largest == 0u ? y : x;
        float c1 = largest <= 1u ? z : y;
        float c2 = largest <= 2u ? w : z;
        return (largest << 30) | (Q10(c0) << 20) | (Q10(c1) << 10) | Q10(c2);
    }

    static uint Q10(float v)
    {
        float t = v / InvSqrt2;
        t = t < -1f ? -1f : t > 1f ? 1f : t;
        return (uint)((int)XMath.Round(t * 511f) + 511);
    }

    static float D10(uint q) => ((int)(q & 0x3FFu) - 511) / 511f * InvSqrt2;

    public static void UnpackRotation(uint r, out float x, out float y, out float z, out float w)
    {
        uint largest = r >> 30;
        float c0 = D10(r >> 20), c1 = D10(r >> 10), c2 = D10(r);
        float l = XMath.Sqrt(XMath.Max(0f, 1f - c0 * c0 - c1 * c1 - c2 * c2));
        if (largest == 0u) { x = l; y = c0; z = c1; w = c2; }
        else if (largest == 1u) { x = c0; y = l; z = c1; w = c2; }
        else if (largest == 2u) { x = c0; y = c1; z = l; w = c2; }
        else { x = c0; y = c1; z = c2; w = l; }
    }

    public static uint QuantSh(float v)
    {
        float t = v / ShRange;
        t = t < -1f ? -1f : t > 1f ? 1f : t;
        return (uint)((int)XMath.Round(t * 127f) + 128);
    }

    public static float DequantSh(uint b) => ((int)(b & 0xFFu) - 128) / 127f * ShRange;

    // ── kernels ─────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Splat i of a packed buffer (SplatFormat: pos 0-2, colour 3-5, scale 6-8, opacity 9, quat 10-13 x,y,z,w).</summary>
    public static void EncodeKernel(Index1D i,
        ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2,
        ArrayView1D<uint, Stride1D.Dense> geo, ArrayView1D<uint, Stride1D.Dense> app, ArrayView1D<uint, Stride1D.Dense> sh,
        Frame f, int withSh)
    {
        if (i >= f.Count) return;
        long ii = (int)i;
        long o = ii * SplatFormat.Floats;
        float op = packed[o + 9];
        op = op < 0f ? 0f : op > 1f ? 1f : op;
        uint opq = (uint)(op * 255f + 0.5f);
        geo[ii * GeoWords + 0] = EncodeAxis(packed[o + 0], f.MinX, f.SizeX, f.TailLoX, f.TailHiX, f.Piecewise) | (opq << 24);
        geo[ii * GeoWords + 1] = EncodeAxis(packed[o + 1], f.MinY, f.SizeY, f.TailLoY, f.TailHiY, f.Piecewise);
        geo[ii * GeoWords + 2] = EncodeAxis(packed[o + 2], f.MinZ, f.SizeZ, f.TailLoZ, f.TailHiZ, f.Piecewise);
        geo[ii * GeoWords + 3] = PackRotation(packed[o + 10], packed[o + 11], packed[o + 12], packed[o + 13]);

        float sx = XMath.Max(packed[o + 6], 1e-9f), sy = XMath.Max(packed[o + 7], 1e-9f), sz = XMath.Max(packed[o + 8], 1e-9f);
        app[ii * AppWords + 0] = FloatToHalf(XMath.Log(sx)) | (FloatToHalf(XMath.Log(sy)) << 16);
        app[ii * AppWords + 1] = FloatToHalf(XMath.Log(sz)) | (FloatToHalf(packed[o + 3]) << 16);
        app[ii * AppWords + 2] = FloatToHalf(packed[o + 4]) | (FloatToHalf(packed[o + 5]) << 16);

        if (withSh == 0) return;
        long so = ii * SphericalHarmonics.PartFloatsPerSplat;
        for (int w = 0; w < ShWords; w++)
        {
            uint word = 0u;
            for (int b = 0; b < 4; b++)
            {
                int k = w * 4 + b;
                uint q = 128u;   // spare bytes 45-47 encode 0
                if (k < 15) q = QuantSh(sh0[so + k]);
                else if (k < 30) q = QuantSh(sh1[so + k - 15]);
                else if (k < 45) q = QuantSh(sh2[so + k - 30]);
                word |= q << (8 * b);
            }
            sh[ii * ShWords + w] = word;
        }
    }

    public static void DecodeKernel(Index1D i,
        ArrayView1D<uint, Stride1D.Dense> geo, ArrayView1D<uint, Stride1D.Dense> app, ArrayView1D<uint, Stride1D.Dense> sh,
        ArrayView1D<float, Stride1D.Dense> packed,
        ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2,
        Frame f, int withSh)
    {
        if (i >= f.Count) return;
        long ii = (int)i;
        long o = ii * SplatFormat.Floats;
        uint g0 = geo[ii * GeoWords + 0];
        packed[o + 0] = DecodeAxis(g0, f.MinX, f.SizeX, f.TailLoX, f.TailHiX, f.Piecewise);
        packed[o + 1] = DecodeAxis(geo[ii * GeoWords + 1], f.MinY, f.SizeY, f.TailLoY, f.TailHiY, f.Piecewise);
        packed[o + 2] = DecodeAxis(geo[ii * GeoWords + 2], f.MinZ, f.SizeZ, f.TailLoZ, f.TailHiZ, f.Piecewise);
        packed[o + 9] = (g0 >> 24) / 255f;
        UnpackRotation(geo[ii * GeoWords + 3], out float qx, out float qy, out float qz, out float qw);
        packed[o + 10] = qx; packed[o + 11] = qy; packed[o + 12] = qz; packed[o + 13] = qw;

        uint a0 = app[ii * AppWords + 0], a1 = app[ii * AppWords + 1], a2 = app[ii * AppWords + 2];
        packed[o + 6] = XMath.Exp(HalfToFloat(a0 & 0xFFFFu));
        packed[o + 7] = XMath.Exp(HalfToFloat(a0 >> 16));
        packed[o + 8] = XMath.Exp(HalfToFloat(a1 & 0xFFFFu));
        packed[o + 3] = HalfToFloat(a1 >> 16);
        packed[o + 4] = HalfToFloat(a2 & 0xFFFFu);
        packed[o + 5] = HalfToFloat(a2 >> 16);

        if (withSh == 0) return;
        long so = ii * SphericalHarmonics.PartFloatsPerSplat;
        for (int k = 0; k < 45; k++)
        {
            uint word = sh[ii * ShWords + k / 4];
            float v = DequantSh(word >> (8 * (k % 4)));
            if (k < 15) sh0[so + k] = v;
            else if (k < 30) sh1[so + k - 15] = v;
            else sh2[so + k - 30] = v;
        }
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
        ArrayView1D<uint, Stride1D.Dense>, Frame, int>? _encode;
    static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, Frame, int>? _decode;

    /// <summary>Encode <paramref name="f"/>.Count splats (and their SH parts, if given) into three new word buffers.</summary>
    public static (MemoryBuffer1D<uint, Stride1D.Dense> Geo, MemoryBuffer1D<uint, Stride1D.Dense> App, MemoryBuffer1D<uint, Stride1D.Dense> Sh)
        Encode(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> packed, MemoryBuffer1D<float, Stride1D.Dense>[]? shParts, Frame f)
    {
        _encode ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
            ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, Frame, int>(EncodeKernel);
        bool withSh = shParts is { Length: 3 };
        var geo = a.Allocate1D<uint>((long)f.Count * GeoWords);
        var app = a.Allocate1D<uint>((long)f.Count * AppWords);
        var sh = a.Allocate1D<uint>(withSh ? (long)f.Count * ShWords : 1);
        var dummy = DummyF(a);
        _encode(f.Count, packed.View,
            (withSh ? shParts![0] : dummy).View, (withSh ? shParts![1] : dummy).View, (withSh ? shParts![2] : dummy).View,
            geo.View, app.View, sh.View, f, withSh ? 1 : 0);
        return (geo, app, sh);
    }

    /// <summary>Decode into new buffers: the packed splats and (with SH) the three SH parts.</summary>
    public static (MemoryBuffer1D<float, Stride1D.Dense> Packed, MemoryBuffer1D<float, Stride1D.Dense>[]? Sh)
        Decode(Accelerator a, MemoryBuffer1D<uint, Stride1D.Dense> geo, MemoryBuffer1D<uint, Stride1D.Dense> app,
            MemoryBuffer1D<uint, Stride1D.Dense>? sh, Frame f)
    {
        _decode ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
            ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Frame, int>(DecodeKernel);
        bool withSh = sh != null;
        var packed = a.Allocate1D<float>((long)f.Count * SplatFormat.Floats);
        MemoryBuffer1D<float, Stride1D.Dense>[]? parts = withSh
            ? Enumerable.Range(0, 3).Select(_ => a.Allocate1D<float>((long)f.Count * SphericalHarmonics.PartFloatsPerSplat)).ToArray()
            : null;
        var dummyF = DummyF(a);
        _dummyU ??= a.Allocate1D<uint>(1);
        _decode(f.Count, geo.View, app.View, (sh ?? _dummyU).View, packed.View,
            (parts?[0] ?? dummyF).View, (parts?[1] ?? dummyF).View, (parts?[2] ?? dummyF).View, f, withSh ? 1 : 0);
        return (packed, parts);
    }

    // Stand-ins for unused bindings. Persistent: the kernels run asynchronously, so a buffer disposed right after the
    // launch could be gone before the kernel reads its binding.
    static MemoryBuffer1D<float, Stride1D.Dense>? _dummyF;
    static MemoryBuffer1D<uint, Stride1D.Dense>? _dummyU;
    static MemoryBuffer1D<float, Stride1D.Dense> DummyF(Accelerator a) => _dummyF ??= a.Allocate1D<float>(1);
}
