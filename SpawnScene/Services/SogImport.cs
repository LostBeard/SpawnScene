using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Formats;

namespace SpawnScene.Services;

/// <summary>
/// PlayCanvas's SOG (SogMeta) into SpawnScene's splat rows and SH parts on the GPU. Each lossless WebP is decoded by the
/// browser straight into a WebGPU texture (createImageBitmap with premultiplyAlpha "none" and colorSpaceConversion "none",
/// copyExternalImageToTexture with premultipliedAlpha false - a 2D canvas would premultiply, and sh0's alpha is the
/// opacity its colour bytes ride on), then copied into one device buffer at 256-byte-aligned offsets; one kernel decodes
/// every splat as playcanvas/engine gsplat-sog-data.js GSplatSogIterator.read does, formula for formula.
/// </summary>
public static class SogImport
{
    public struct Tex
    {
        public int Off, Row, Width;
    }

    public struct Params
    {
        public int Count, Version, Coeffs, Flip;
        public Tex MeansL, MeansU, Quats, Scales, Sh0, Labels, Centroids;
        public float MinX, MinY, MinZ, MaxX, MaxY, MaxZ;
        public float SMinX, SMinY, SMinZ, SMaxX, SMaxY, SMaxZ;
        public float CMinR, CMinG, CMinB, CMinA, CMaxR, CMaxG, CMaxB, CMaxA;
        public float NMin, NMax;
    }

    static Action<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _kernel;
    static Accelerator? _loadedFor;

    static uint Px(ArrayView1D<uint, Stride1D.Dense> t, Tex x, int i) => t[x.Off + (i / x.Width) * x.Row + i % x.Width];
    static float Lerp(float a, float b, float t) => a * (1f - t) + b * t;
    static float Unlog(float n) => n >= 0f ? XMath.Exp(n) - 1f : -(XMath.Exp(-n) - 1f);

    static void Kernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> book,
        ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> sh0, ArrayView1D<float, Stride1D.Dense> sh1,
        ArrayView1D<float, Stride1D.Dense> sh2, Params p)
    {
        if (i >= p.Count) return;
        int o = i * SplatFormat.Floats;
        float sgn = p.Flip != 0 ? -1f : 1f;
        uint l = Px(t, p.MeansL, i), u = Px(t, p.MeansU, i);
        packed[o] = Unlog(Lerp(p.MinX, p.MaxX, (((u & 0xFFu) << 8) + (l & 0xFFu)) / 65535f));
        packed[o + 1] = Unlog(Lerp(p.MinY, p.MaxY, ((((u >> 8) & 0xFFu) << 8) + ((l >> 8) & 0xFFu)) / 65535f)) * sgn;
        packed[o + 2] = Unlog(Lerp(p.MinZ, p.MaxZ, ((((u >> 16) & 0xFFu) << 8) + ((l >> 16) & 0xFFu)) / 65535f)) * sgn;

        uint q = Px(t, p.Quats, i);
        const float Norm = 1.41421356f;
        float a = ((q & 0xFFu) / 255f - 0.5f) * Norm, b = (((q >> 8) & 0xFFu) / 255f - 0.5f) * Norm, c = (((q >> 16) & 0xFFu) / 255f - 0.5f) * Norm;
        float d = XMath.Sqrt(XMath.Max(0f, 1f - (a * a + b * b + c * c)));
        int mode = (int)((q >> 24) & 0xFFu) - 252;
        float qx, qy, qz, qw;
        if (mode == 1) { qx = d; qy = b; qz = c; qw = a; }
        else if (mode == 2) { qx = b; qy = d; qz = c; qw = a; }
        else if (mode == 3) { qx = b; qy = c; qz = d; qw = a; }
        else { qx = a; qy = b; qz = c; qw = d; }
        float len = XMath.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (!(len > 1e-12f)) { qx = 0f; qy = 0f; qz = 0f; qw = 1f; len = 1f; }
        qx /= len; qy /= len; qz /= len; qw /= len;
        if (p.Flip != 0)
        {
            float nw = -qx, nx = qw, ny = -qz, nz = qy;   // (1,0,0,0) * q, as GaussianPlyImport
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        packed[o + 10] = qx; packed[o + 11] = qy; packed[o + 12] = qz; packed[o + 13] = qw;

        uint s = Px(t, p.Scales, i);
        if (p.Version == 2)
        {
            packed[o + 6] = XMath.Exp(book[(int)(s & 0xFFu)]);
            packed[o + 7] = XMath.Exp(book[(int)((s >> 8) & 0xFFu)]);
            packed[o + 8] = XMath.Exp(book[(int)((s >> 16) & 0xFFu)]);
        }
        else
        {
            packed[o + 6] = XMath.Exp(Lerp(p.SMinX, p.SMaxX, (s & 0xFFu) / 255f));
            packed[o + 7] = XMath.Exp(Lerp(p.SMinY, p.SMaxY, ((s >> 8) & 0xFFu) / 255f));
            packed[o + 8] = XMath.Exp(Lerp(p.SMinZ, p.SMaxZ, ((s >> 16) & 0xFFu) / 255f));
        }

        uint col = Px(t, p.Sh0, i);
        if (p.Version == 2)
        {
            packed[o + 3] = book[256 + (int)(col & 0xFFu)];
            packed[o + 4] = book[256 + (int)((col >> 8) & 0xFFu)];
            packed[o + 5] = book[256 + (int)((col >> 16) & 0xFFu)];
            packed[o + 9] = ((col >> 24) & 0xFFu) / 255f;
        }
        else
        {
            packed[o + 3] = Lerp(p.CMinR, p.CMaxR, (col & 0xFFu) / 255f);
            packed[o + 4] = Lerp(p.CMinG, p.CMaxG, ((col >> 8) & 0xFFu) / 255f);
            packed[o + 5] = Lerp(p.CMinB, p.CMaxB, ((col >> 16) & 0xFFu) / 255f);
            packed[o + 9] = 1f / (1f + XMath.Exp(-Lerp(p.CMinA, p.CMaxA, ((col >> 24) & 0xFFu) / 255f)));
        }

        if (p.Coeffs == 0) return;
        uint lab = Px(t, p.Labels, i);
        int n = (int)((lab & 0xFFu) + (((lab >> 8) & 0xFFu) << 8));
        int cu = (n % 64) * p.Coeffs, cv = n / 64;
        int partBase = i * SphericalHarmonics.PartFloatsPerSplat;
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++)
            {
                float val = 0f;
                if (k < p.Coeffs)
                {
                    uint texel = t[p.Centroids.Off + cv * p.Centroids.Row + cu + k];
                    uint by = (texel >> (8 * ch)) & 0xFFu;
                    val = p.Version == 2 ? book[512 + (int)by] : Lerp(p.NMin, p.NMax, by / 255f);
                }
                if (p.Flip != 0 && ((GaussianPlyImport.FlipSignMask >> (k + 1)) & 1) != 0) val = -val;
                int f = k * 3 + ch;
                int part = f / SphericalHarmonics.PartFloatsPerSplat, idx = partBase + f % SphericalHarmonics.PartFloatsPerSplat;
                if (part == 0) sh0[idx] = val;
                else if (part == 1) sh1[idx] = val;
                else sh2[idx] = val;
            }
    }

    /// <summary>Decode every splat from textures already in <paramref name="texels"/> (laid out as <paramref name="p"/>
    /// says). Public for the CPU tests.</summary>
    public static void Run(Accelerator a, ArrayView1D<uint, Stride1D.Dense> texels, ArrayView1D<float, Stride1D.Dense> codebooks,
        Params p, ArrayView1D<float, Stride1D.Dense> packed, ArrayView1D<float, Stride1D.Dense> sh0,
        ArrayView1D<float, Stride1D.Dense> sh1, ArrayView1D<float, Stride1D.Dense> sh2)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _kernel = null; _loadedFor = a; }
        _kernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, Params>(Kernel);
        _kernel(p.Count, texels, codebooks, packed, sh0, sh1, sh2, p);
    }

    /// <summary>The kernel's parameters from the meta and the textures' placement.</summary>
    public static Params MakeParams(SogMeta m, Tex meansL, Tex meansU, Tex quats, Tex scales, Tex sh0, Tex labels, Tex centroids,
        int coeffs, bool flipToYUp) => new()
    {
        Count = m.Count, Version = m.Version, Coeffs = coeffs, Flip = flipToYUp ? 1 : 0,
        MeansL = meansL, MeansU = meansU, Quats = quats, Scales = scales, Sh0 = sh0, Labels = labels, Centroids = centroids,
        MinX = m.MeansMin[0], MinY = m.MeansMin[1], MinZ = m.MeansMin[2], MaxX = m.MeansMax[0], MaxY = m.MeansMax[1], MaxZ = m.MeansMax[2],
        SMinX = m.ScalesMin[0], SMinY = m.ScalesMin[1], SMinZ = m.ScalesMin[2], SMaxX = m.ScalesMax[0], SMaxY = m.ScalesMax[1], SMaxZ = m.ScalesMax[2],
        CMinR = m.Sh0Min[0], CMinG = m.Sh0Min[1], CMinB = m.Sh0Min[2], CMinA = m.Sh0Min.Length > 3 ? m.Sh0Min[3] : 0f,
        CMaxR = m.Sh0Max[0], CMaxG = m.Sh0Max[1], CMaxB = m.Sh0Max[2], CMaxA = m.Sh0Max.Length > 3 ? m.Sh0Max[3] : 0f,
        NMin = m.ShNMin, NMax = m.ShNMax,
    };

    /// <summary>The three codebooks in one array (scales, sh0, shN; zeros where absent - version 1).</summary>
    public static float[] Codebooks(SogMeta m)
    {
        var cb = new float[768];
        m.ScalesCodebook?.CopyTo(cb, 0);
        m.Sh0Codebook?.CopyTo(cb, 256);
        m.ShNCodebook?.CopyTo(cb, 512);
        return cb;
    }

    /// <summary>
    /// The SOG whose files <paramref name="open"/> returns (as Blobs: a zip entry, or a sibling URL's body) into rows and SH
    /// parts on the device (the caller owns them; SH null without rest bands).
    /// </summary>
    public static async Task<(MemoryBuffer1D<float, Stride1D.Dense> Packed, MemoryBuffer1D<float, Stride1D.Dense>[]? Sh, int ShDegree)> ConvertAsync(
        WebGPUAccelerator a, Window window, SogMeta m, Func<string, Task<Blob>> open, bool flipToYUp)
    {
        var device = a.NativeAccelerator.NativeDevice!;
        var queue = a.NativeAccelerator.Queue!;
        var names = new List<string> { m.MeansFiles[0], m.MeansFiles[1], m.QuatsFiles[0], m.ScalesFiles[0], m.Sh0Files[0] };
        if (m.ShNFiles.Length == 2) { names.Add(m.ShNFiles[1]); names.Add(m.ShNFiles[0]); }   // labels, centroids
        // Decode every texture first (sizes decide the layout), then one buffer for all of them.
        var bitmaps = new List<ImageBitmap>();
        try
        {
            foreach (var name in names)
            {
                using var blob = await open(name);
                bitmaps.Add(await window.CreateImageBitmap(blob, new ImageBitmapOptions { PremultiplyAlpha = "none", ColorSpaceConversion = "none" }));
            }
            var tex = new Tex[7];
            long words = 0;
            for (int k = 0; k < bitmaps.Count; k++)
            {
                int w = (int)bitmaps[k].Width, h = (int)bitmaps[k].Height;
                int rowBytes = (w * 4 + 255) / 256 * 256;   // copyTextureToBuffer rows are 256-byte aligned
                tex[k] = new Tex { Off = (int)words, Row = rowBytes / 4, Width = w };
                words += (long)rowBytes / 4 * h;
                words = (words + 63) / 64 * 64;   // the next texture starts 256-byte aligned
            }
            for (int k = 0; k < 5; k++)
                if ((long)bitmaps[k].Width * bitmaps[k].Height < m.Count) throw new FormatException($"SOG texture {names[k]} is smaller than {m.Count:N0} splats");
            int coeffs = bitmaps.Count == 7 ? Math.Min(15, SogMeta.BandsForCentroidWidth((int)bitmaps[6].Width) switch { 1 => 3, 2 => 8, 3 => 15, _ => 0 }) : 0;
            var texels = a.Allocate1D<uint>(Math.Max(64L, words));
            try
            {
                a.FlushPendingCommands();
                using var enc = device.CreateCommandEncoder();
                var textures = new List<GPUTexture>();
                for (int k = 0; k < bitmaps.Count; k++)
                {
                    uint w = (uint)bitmaps[k].Width, h = (uint)bitmaps[k].Height;
                    var gt = device.CreateTexture(new GPUTextureDescriptor
                    {
                        Size = new[] { (int)w, (int)h }, Format = "rgba8unorm",
                        Usage = GPUTextureUsage.CopyDst | GPUTextureUsage.CopySrc | GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
                    });
                    textures.Add(gt);
                    queue.CopyExternalImageToTexture(new GPUCopyExternalImageSourceInfo { Source = bitmaps[k] },
                        new GPUCopyExternalImageDestInfo { Texture = gt, PremultipliedAlpha = false }, new uint[] { w, h });
                    enc.CopyTextureToBuffer(new GPUTexelCopyTextureInfo { Texture = gt },
                        new GPUTexelCopyBufferInfo { Buffer = texels.GetGPUBuffer()!, Offset = (ulong)tex[k].Off * 4, BytesPerRow = (uint)tex[k].Row * 4, RowsPerImage = h },
                        new uint[] { w, h });
                }
                using (var cmd = enc.Finish()) queue.Submit(new[] { cmd });
                foreach (var gt in textures) { gt.Destroy(); gt.Dispose(); }

                using var books = a.Allocate1D(Codebooks(m));
                var packed = a.Allocate1D<float>((long)m.Count * SplatFormat.Floats);
                MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
                using var d0 = a.Allocate1D<float>(1);
                using var d1 = a.Allocate1D<float>(1);
                using var d2 = a.Allocate1D<float>(1);
                if (coeffs > 0)
                {
                    sh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
                    for (int pi = 0; pi < sh.Length; pi++) sh[pi] = a.Allocate1D<float>((long)m.Count * SphericalHarmonics.PartFloatsPerSplat);
                }
                var none = new Tex { Off = 0, Row = 1, Width = 1 };
                var prm = MakeParams(m, tex[0], tex[1], tex[2], tex[3], tex[4], coeffs > 0 ? tex[5] : none, coeffs > 0 ? tex[6] : none, coeffs, flipToYUp);
                Run(a, texels.View, books.View, prm, packed.View, sh?[0].View ?? d0.View, sh?[1].View ?? d1.View, sh?[2].View ?? d2.View);
                await a.SynchronizeAsync();
                return (packed, sh, coeffs switch { 3 => 1, 8 => 2, 15 => 3, _ => 0 });
            }
            finally { texels.Dispose(); }
        }
        finally { foreach (var b in bitmaps) { b.Close(); b.Dispose(); } }
    }
}
