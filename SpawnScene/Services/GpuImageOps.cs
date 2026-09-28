using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Per-image work the import used to do on managed pixel arrays, done on the device against
/// <see cref="ImportedImage.GpuRgba"/>. Only the small results come back to the host: the grayscale frame the
/// (CPU) feature detector reads, and one colour per feature.
/// </summary>
public static class GpuImageOps
{
    // Nearest-neighbour downsample + integer BT.601 luma: EXACTLY ImageImportService.DownsampleGrayscale /
    // RgbaToGrayscale (same float scale, same truncation, same (r*77 + g*150 + b*29) >> 8), so the detector sees
    // the same bytes it saw from the managed path and finds the same features.
    static void GrayKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<byte, Stride1D.Dense> gray,
        int srcW, int srcH, int dstW, float scaleX, float scaleY)
    {
        int dy = i / dstW, dx = i - dy * dstW;
        int sy = Math.Min((int)(dy * scaleY), srcH - 1);
        int sx = Math.Min((int)(dx * scaleX), srcW - 1);
        int p = rgba[sy * srcW + sx];
        int r = p & 0xFF, g = (p >> 8) & 0xFF, b = (p >> 16) & 0xFF;
        gray[i] = (byte)((r * 77 + g * 150 + b * 29) >> 8);
    }

    static void GatherKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<int, Stride1D.Dense> index,
        ArrayView1D<int, Stride1D.Dense> colour)
        => colour[i] = rgba[index[i]];

    private sealed class Kernels
    {
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, int, int, int, float, float> Gray = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> Gather = null!;
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    private static Kernels For(Accelerator accelerator) => s_kernels.GetValue(accelerator, a => new Kernels
    {
        Gray = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, int, int, int, float, float>(GrayKernel),
        Gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GatherKernel),
    });

    /// <summary>
    /// The grayscale frame the feature detector reads, at <paramref name="dstW"/> x <paramref name="dstH"/>. This IS
    /// host data - the detector runs on the CPU - but it is one frame at a time, transient, a quarter of the RGBA
    /// size, and nothing keeps it after detection.
    /// </summary>
    public static async Task<byte[]> GrayscaleAsync(Accelerator accelerator,
        MemoryBuffer1D<int, Stride1D.Dense> rgba, int srcW, int srcH, int dstW, int dstH)
    {
        var k = For(accelerator);
        using var gray = accelerator.Allocate1D<byte>((long)dstW * dstH);
        float scaleX = (float)srcW / dstW, scaleY = (float)srcH / dstH;
        k.Gray(dstW * dstH, rgba.View, gray.View, srcW, srcH, dstW, scaleX, scaleY);
        await accelerator.SynchronizeAsync();
        return await gray.CopyToHostAsync<byte>();
    }

    /// <summary>
    /// Stamp each feature with the photo's colour at (round(X), round(Y)) - the pixel the BA sparse-cloud init always
    /// read. One small upload of indices, one small readback of colours; the image stays on the device.
    /// </summary>
    public static async Task SampleFeatureColoursAsync(Accelerator accelerator,
        MemoryBuffer1D<int, Stride1D.Dense> rgba, int width, int height, IReadOnlyList<ImageFeature> features)
    {
        if (features.Count == 0) return;
        var index = new int[features.Count];
        for (int f = 0; f < features.Count; f++) index[f] = PixelIndex(features[f], width, height);
        var k = For(accelerator);
        using var indexBuf = accelerator.Allocate1D(index);
        using var colourBuf = accelerator.Allocate1D<int>(features.Count);
        k.Gather(features.Count, rgba.View, indexBuf.View, colourBuf.View);
        await accelerator.SynchronizeAsync();
        var colours = await colourBuf.CopyToHostAsync<int>();
        for (int f = 0; f < features.Count; f++) features[f].PackedColor = colours[f];
    }

    /// <summary>The same stamp from managed pixels (legacy CPU imports).</summary>
    public static void SampleFeatureColours(byte[] rgba, int width, int height, IReadOnlyList<ImageFeature> features)
    {
        foreach (var f in features)
        {
            int o = PixelIndex(f, width, height) * 4;
            if (o + 3 >= rgba.Length) continue;
            f.PackedColor = rgba[o] | (rgba[o + 1] << 8) | (rgba[o + 2] << 16) | (rgba[o + 3] << 24);
        }
    }

    private static int PixelIndex(ImageFeature f, int width, int height)
    {
        int px = Math.Clamp((int)MathF.Round(f.X), 0, width - 1);
        int py = Math.Clamp((int)MathF.Round(f.Y), 0, height - 1);
        return py * width + px;
    }
}
