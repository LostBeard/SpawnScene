using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using System.Runtime.InteropServices;

namespace SpawnScene.Services;

/// <summary>
/// ILGPU kernel for GPU-only depth-to-Gaussian conversion.
/// Reads GPU-resident depth (from DepthEstimationService) and GPU-uploaded RGBA,
/// writes 10 floats per splat directly into a packed output buffer.
/// No CPU readback — the packed buffer is passed straight to GpuSplatSorter.
/// </summary>
public class DepthToGaussianKernel
{
    private readonly GpuService _gpu;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depthValues (GPU-resident raw depth)
        ArrayView1D<int, Stride1D.Dense>,    // packedRGBA (1 int per pixel)
        ArrayView1D<float, Stride1D.Dense>,  // outPacked (10 floats per splat)
        ArrayView1D<float, Stride1D.Dense>,  // params
        int>?                                // totalPoints
        _unprojectAndPackKernel;

    public DepthToGaussianKernel(GpuService gpu) => _gpu = gpu;

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernel
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: unproject depth + RGBA → packed 10-float splat (pos3, color3, scale3, opacity).
    /// Invalid pixels get opacity=0, pos=(0,0,0) — discarded by the fragment shader (alpha<0.002).
    /// Params: [0]=width [1]=height [2]=fx [3]=fy [4]=cx [5]=cy [6]=subsample
    ///         [7]=minDepth [8]=maxDepth (for on-GPU normalization of relative depth)
    /// </summary>
    private static void UnprojectAndPackKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> depthValues,
        ArrayView1D<int, Stride1D.Dense> packedRGBA,
        ArrayView1D<float, Stride1D.Dense> outPacked,
        ArrayView1D<float, Stride1D.Dense> p,
        int totalPoints)
    {
        int width = (int)p[0];
        float fx = p[2]; float fy = p[3];
        float cx = p[4]; float cy = p[5];
        int subsample = (int)p[6];
        float minDepth = p[7];
        float maxDepth = p[8];

        int sampledW = width / subsample;
        int sx = index % sampledW;
        int sy = index / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;
        int imgIdx = imgY * width + imgX;

        float rawDepth = depthValues[imgIdx];

        // Normalize raw disparity → [0,1] using min/max computed on GPU
        float range = maxDepth - minDepth;
        float normalizedD = (range > 1e-6f) ? (rawDepth - minDepth) / range : 0f;
        // Disparity → depth (larger disparity = closer)
        float invD = 1.0f / (normalizedD + 0.01f);
        float d = invD;
        int valid = (normalizedD >= 0.01f && d > 0.01f && d < 100f) ? 1 : 0;

        int outOff = index * 10;

        if (valid == 1)
        {
            float posX = -((imgX - cx) * d / fx);
            float posY = -((imgY - cy) * d / fy);
            float posZ = d;

            // Per-splat scale: world size of one pixel at this depth
            float pixelScale = d * subsample / fx;
            float splatScale = pixelScale > 0.001f ? pixelScale : 0.001f;

            int packed = packedRGBA[imgIdx];
            float r = (packed & 0xFF) / 255f;
            float g = ((packed >> 8) & 0xFF) / 255f;
            float b = ((packed >> 16) & 0xFF) / 255f;

            outPacked[outOff + 0] = posX;
            outPacked[outOff + 1] = posY;
            outPacked[outOff + 2] = posZ;
            outPacked[outOff + 3] = r;
            outPacked[outOff + 4] = g;
            outPacked[outOff + 5] = b;
            outPacked[outOff + 6] = splatScale;
            outPacked[outOff + 7] = splatScale;
            outPacked[outOff + 8] = splatScale * 0.5f;
            outPacked[outOff + 9] = 0.9f;
        }
        else
        {
            // Invisible splat: opacity=0 → discarded in fs_main (alpha < 0.002)
            outPacked[outOff + 0] = 0f;
            outPacked[outOff + 1] = 0f;
            outPacked[outOff + 2] = 0f;
            outPacked[outOff + 3] = 0f;
            outPacked[outOff + 4] = 0f;
            outPacked[outOff + 5] = 0f;
            outPacked[outOff + 6] = 0.001f;
            outPacked[outOff + 7] = 0.001f;
            outPacked[outOff + 8] = 0.0005f;
            outPacked[outOff + 9] = 0f;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Generate a GPU-packed splat buffer from GPU-resident depth + CPU RGBA.
    /// Returns (packedBuf, totalSplatCount) — ownership of packedBuf transfers to caller.
    /// No CPU readback of splat data at any point.
    /// RGBA is uploaded once from image source data (acceptable: image loaded from disk).
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, ImportedImage image, int subsample = 2)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;

        _unprojectAndPackKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(UnprojectAndPackKernel);

        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        float fx = MathF.Max(w, h) * 1.2f;
        float fy = fx;
        float cx = w / 2f;
        float cy = h / 2f;

        // Upload RGBA to GPU — justified: image data from file/picker (CPU source boundary).
        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        var paramArr = new float[]
        {
            w, h, fx, fy, cx, cy,
            subsample,
            depth.MinDepth,
            depth.MaxDepth,
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);

        // Output buffer: 10 floats per splat (pos3, color3, scale3, opacity).
        // Ownership transfers to caller → GpuSplatSorter.
        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * 10);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null — GPU path requires GPU-resident depth.");

        _unprojectAndPackKernel(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            outPackedBuf.View,
            paramBuf.View,
            numPoints);

        await accelerator.SynchronizeAsync();

        Console.WriteLine($"[DepthGPU] Generated {numPoints:N0} splat slots (subsample={subsample}) — no CPU readback");
        return (outPackedBuf, numPoints);
    }
}
