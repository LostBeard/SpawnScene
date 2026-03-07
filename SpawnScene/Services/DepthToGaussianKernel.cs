using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using System.Runtime.InteropServices;

namespace SpawnScene.Services;

/// <summary>
/// ILGPU kernel for GPU-only depth-to-Gaussian conversion with atomic compaction.
///
/// Pipeline:
///   1. Unproject depth + RGBA → 10-float splat for each valid pixel.
///   2. Invalid pixels (bad depth range) are skipped entirely via Atomic.Add compaction.
///   3. 4-byte counter readback → actual valid splat count (no wasted slots in output buffer).
///   4. Optional edge-sharpening: depth gradient magnitude shrinks splat scale at edges.
/// </summary>
public class DepthToGaussianKernel
{
    private readonly GpuService _gpu;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depthValues
        ArrayView1D<int, Stride1D.Dense>,    // packedRGBA
        ArrayView1D<float, Stride1D.Dense>,  // outPacked (compacted)
        ArrayView1D<int, Stride1D.Dense>,    // counter [0] = valid splat count
        ArrayView1D<float, Stride1D.Dense>>? // params
        _unprojectAndPackKernel;

    public DepthToGaussianKernel(GpuService gpu) => _gpu = gpu;

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernel
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: unproject depth + RGBA → compacted packed splat buffer.
    /// Only valid pixels write output (Atomic.Add compaction — no zero-opacity dummy splats).
    /// Params: [0]=width [1]=height [2]=fx [3]=fy [4]=cx [5]=cy [6]=subsample
    ///         [7]=minDepth [8]=maxDepth [9]=edgeSharpness (0=disabled, 0.3=default)
    /// </summary>
    private static void UnprojectAndPackKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> depthValues,
        ArrayView1D<int, Stride1D.Dense> packedRGBA,
        ArrayView1D<float, Stride1D.Dense> outPacked,
        ArrayView1D<int, Stride1D.Dense> counter,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        int width = (int)p[0];
        int height = (int)p[1];
        float fx = p[2]; float fy = p[3];
        float cx = p[4]; float cy = p[5];
        int subsample = (int)p[6];
        float minDepth = p[7];
        float maxDepth = p[8];
        float edgeSharpness = p[9];

        int sampledW = width / subsample;
        int sx = index % sampledW;
        int sy = index / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;
        int imgIdx = imgY * width + imgX;

        float rawDepth = depthValues[imgIdx];

        // Normalize raw disparity → [0,1]
        float range = maxDepth - minDepth;
        float normalizedD = (range > 1e-6f) ? (rawDepth - minDepth) / range : 0f;
        // Disparity → metric-like depth (larger disparity = closer)
        float invD = 1.0f / (normalizedD + 0.01f);
        float d = invD;

        // Validity check: skip extreme depths and background
        if (normalizedD < 0.01f || d <= 0.01f || d >= 100f) return;

        // Per-splat scale: world size of one pixel at this depth
        float pixelScale = d * subsample / fx;
        float splatScale = pixelScale > 0.001f ? pixelScale : 0.001f;

        // Phase 4b: Edge-adaptive scale — shrink splats at depth discontinuities.
        // Central difference gradient on raw depth (normalized by range for unit independence).
        if (edgeSharpness > 0f && range > 1e-6f)
        {
            int x0 = (imgX > 0) ? imgX - subsample : imgX;
            int x1 = (imgX + subsample < width) ? imgX + subsample : imgX;
            int y0 = (imgY > 0) ? imgY - subsample : imgY;
            int y1 = (imgY + subsample < height) ? imgY + subsample : imgY;

            float gx = (depthValues[imgY * width + x1] - depthValues[imgY * width + x0]) / range;
            float gy = (depthValues[y1 * width + imgX] - depthValues[y0 * width + imgX]) / range;
            float gradMag = MathF.Sqrt(gx * gx + gy * gy);
            // Reduce scale at edges: high gradient → smaller splats → sharper edges
            splatScale /= (1f + gradMag * edgeSharpness);
        }

        int packed = packedRGBA[imgIdx];
        float r = (packed & 0xFF) / 255f;
        float g = ((packed >> 8) & 0xFF) / 255f;
        float b = ((packed >> 16) & 0xFF) / 255f;

        float posX = -((imgX - cx) * d / fx);
        float posY = -((imgY - cy) * d / fy);
        float posZ = d;

        // Atomic compaction: each valid splat gets a unique dense output slot.
        // Zero-opacity dummy splats no longer exist — the output buffer has no gaps.
        int slot = Atomic.Add(ref counter[0], 1);
        int outOff = slot * 10;

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

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer from GPU-resident depth + CPU RGBA.
    /// Returns (packedBuf, validSplatCount) — ownership of packedBuf transfers to caller.
    /// No CPU readback of splat data. Buffer contains only valid splats (no zero-opacity gaps).
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, ImportedImage image, int subsample = 2,
            float edgeSharpness = 0.3f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;

        _unprojectAndPackKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(UnprojectAndPackKernel);

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
            edgeSharpness,
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);

        // Atomic compaction counter
        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        // Output buffer: worst case all pixels are valid (over-allocated, compacted on GPU).
        // Ownership transfers to caller → GpuSplatSorter.
        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * 10);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null — GPU path requires GPU-resident depth.");

        _unprojectAndPackKernel(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            outPackedBuf.View,
            counterBuf.View,
            paramBuf.View);

        // Readback valid splat count (flushes ILGPU stream internally)
        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] Compacted: {validCount:N0} valid / {numPoints:N0} candidate splats " +
            $"(subsample={subsample}, edgeSharpness={edgeSharpness:F2})");

        return (outPackedBuf, validCount);
    }
}
