using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Per-image work the import used to do on managed pixel arrays, done on the device against
/// <see cref="ImportedImage.GpuRgba"/>: the detector's grayscale (which stays there - GpuFeatureDetector reads it) and
/// one colour per feature (the only thing that comes back).
/// </summary>
public static class GpuImageOps
{
    // Nearest-neighbour downsample + integer BT.601 luma: EXACTLY ImageImportService.DownsampleGrayscale /
    // RgbaToGrayscale (same float scale, same truncation, same (r*77 + g*150 + b*29) >> 8), so the detector sees
    // the same bytes it saw from the managed path and finds the same features.
    static void GrayKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<int, Stride1D.Dense> gray,
        int srcW, int srcH, int dstW, float scaleX, float scaleY)
    {
        int dy = i / dstW, dx = i - dy * dstW;
        int sy = Math.Min((int)(dy * scaleY), srcH - 1);
        int sx = Math.Min((int)(dx * scaleX), srcW - 1);
        int p = rgba[sy * srcW + sx];
        int r = p & 0xFF, g = (p >> 8) & 0xFF, b = (p >> 16) & 0xFF;
        gray[i] = (r * 77 + g * 150 + b * 29) >> 8;
    }

    static void GatherKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<int, Stride1D.Dense> index,
        ArrayView1D<int, Stride1D.Dense> colour)
        => colour[i] = rgba[index[i]];

    // A packed RGBA image turned 180 degrees is the pixel array reversed.
    static void ReverseKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> dst, int n)
        => dst[i] = src[n - 1 - i];

    private sealed class Kernels
    {
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> Reverse = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, float, float> Gray = null!;
        public Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> Gather = null!;
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> s_kernels = new();

    private static Kernels For(Accelerator accelerator) => s_kernels.GetValue(accelerator, a => new Kernels
    {
        Reverse = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(ReverseKernel),
        Gray = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, float, float>(GrayKernel),
        Gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GatherKernel),
    });

    /// <summary>A copy of a packed RGBA image turned 180 degrees, on the device. Caller disposes.</summary>
    public static MemoryBuffer1D<int, Stride1D.Dense> Turn180(Accelerator accelerator, MemoryBuffer1D<int, Stride1D.Dense> rgba)
    {
        var turned = accelerator.Allocate1D<int>(rgba.Length);
        For(accelerator).Reverse((int)rgba.Length, rgba.View, turned.View, (int)rgba.Length);
        return turned;
    }

    /// <summary>
    /// The feature detector's grayscale frame (one int 0..255 per pixel) at <paramref name="dstW"/> x
    /// <paramref name="dstH"/>, ON THE DEVICE - GpuFeatureDetector reads it there; it never comes back. Caller disposes.
    /// </summary>
    public static MemoryBuffer1D<int, Stride1D.Dense> GrayscaleToDevice(Accelerator accelerator,
        MemoryBuffer1D<int, Stride1D.Dense> rgba, int srcW, int srcH, int dstW, int dstH)
    {
        var k = For(accelerator);
        var gray = accelerator.Allocate1D<int>((long)dstW * dstH);
        float scaleX = (float)srcW / dstW, scaleY = (float)srcH / dstH;
        k.Gray(dstW * dstH, rgba.View, gray.View, srcW, srcH, dstW, scaleX, scaleY);
        return gray;
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

    /// <summary>
    /// Make sure <paramref name="img"/> has its pixels on the device: decode <see cref="ImportedImage.Source"/> (browser
    /// decode + canvas resize + CopyFromJS - nothing through .NET) when there is no device copy yet. Returns true when
    /// it decoded NOW, so the caller that caused the decode can release it with <see cref="ImportedImage.DisposeGpu"/>
    /// once its dispatches have completed. A legacy managed image (no Source, no GpuRgba) returns false untouched.
    /// </summary>
    public static async Task<bool> EnsureOnDeviceAsync(Accelerator accelerator, ImportedImage img)
    {
        if (img.GpuRgba != null || img.Source == null) return false;
        var (rgba, w, h, _, _) = await SpawnDev.ILGPU.ML.Preprocessing.MediaInterop.DecodeToDeviceAsync(
            img.Source, accelerator, img.DecodeMaxEdge);
        if (w != img.Width || h != img.Height)
        {
            rgba.Dispose();
            throw new InvalidOperationException(
                $"{img.FileName}: re-decoded at {w}x{h}, but the image was imported at {img.Width}x{img.Height}");
        }
        img.GpuRgba = rgba;
        return true;
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
