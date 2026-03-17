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
    ///         [10..13]=exclusion rect (exclX0,exclY0,exclX1,exclY1) — skip pixels inside this region (0=disabled)
    ///         [14]=depthScaleCorrection (multiplied into depth to match reference at seams, default 1.0)
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

        // Exclusion rectangle: skip pixels whose original coordinates fall inside this region.
        // Used by multi-view to avoid generating duplicate splats in the overlap with the reference.
        int exclX0 = (int)p[10]; int exclY0 = (int)p[11];
        int exclX1 = (int)p[12]; int exclY1 = (int)p[13];
        float depthScaleCorr = p[14];

        int globalIndex = index;

        int sampledW = width / subsample;
        int sx = globalIndex % sampledW;
        int sy = globalIndex / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;

        // Exclusion check: if this pixel maps inside the reference image's coverage, skip it.
        if (exclX1 > exclX0 && exclY1 > exclY0)
        {
            if (imgX >= exclX0 && imgX < exclX1 && imgY >= exclY0 && imgY < exclY1)
                return;
        }

        int imgIdx = imgY * width + imgX;

        float rawDepth = depthValues[imgIdx];

        // Normalize raw disparity → [0,1]
        float range = maxDepth - minDepth;
        float normalizedD = (range > 1e-6f) ? (rawDepth - minDepth) / range : 0f;
        // Disparity → metric-like depth (larger disparity = closer), with seam-matching correction
        float invD = 1.0f / (normalizedD + 0.01f);
        float d = invD * depthScaleCorr;

        // Validity check: skip extreme depths and background
        if (normalizedD < 0.01f || d <= 0.01f || d >= 100f * depthScaleCorr) return;

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

    /// <summary>
    /// GPU kernel: unproject depth → camera space → world space using SfM-recovered camera params.
    /// Params: [0]=width [1]=height [2]=fx [3]=fy [4]=cx [5]=cy [6]=subsample
    ///         [7]=minDepth [8]=maxDepth [9]=edgeSharpness
    ///         [10..18]=rotation matrix R (3x3, row-major)
    ///         [19..21]=camera world position (3 floats)
    ///         [22]=depthScale (scales inverted disparity to match SfM coordinate system)
    /// </summary>
    private static void UnprojectWorldSpaceKernel(
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

        // Camera rotation matrix R (world→camera, row-major)
        float r00 = p[10]; float r01 = p[11]; float r02 = p[12];
        float r10 = p[13]; float r11 = p[14]; float r12 = p[15];
        float r20 = p[16]; float r21 = p[17]; float r22 = p[18];
        // Camera world position
        float tx = p[19]; float ty = p[20]; float tz = p[21];
        float depthScale = p[22];

        int globalIndex = index;
        int sampledW = width / subsample;
        int sx = globalIndex % sampledW;
        int sy = globalIndex / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;
        int imgIdx = imgY * width + imgX;

        float rawDepth = depthValues[imgIdx];

        // Normalize raw disparity → [0,1], invert to metric-like depth, apply SfM scale
        float range = maxDepth - minDepth;
        float normalizedD = (range > 1e-6f) ? (rawDepth - minDepth) / range : 0f;
        float invD = 1.0f / (normalizedD + 0.01f);
        float d = invD * depthScale; // Scale to match SfM coordinate system

        if (normalizedD < 0.01f || d <= 0.001f || d >= 1000f) return;

        float pixelScale = d * subsample / fx;
        float splatScale = pixelScale > 0.001f ? pixelScale : 0.001f;

        // Edge-adaptive scale
        if (edgeSharpness > 0f && range > 1e-6f)
        {
            int x0 = (imgX > 0) ? imgX - subsample : imgX;
            int x1 = (imgX + subsample < width) ? imgX + subsample : imgX;
            int y0 = (imgY > 0) ? imgY - subsample : imgY;
            int y1 = (imgY + subsample < height) ? imgY + subsample : imgY;

            float gx = (depthValues[imgY * width + x1] - depthValues[imgY * width + x0]) / range;
            float gy = (depthValues[y1 * width + imgX] - depthValues[y0 * width + imgX]) / range;
            float gradMag = MathF.Sqrt(gx * gx + gy * gy);
            splatScale /= (1f + gradMag * edgeSharpness);
        }

        int packed = packedRGBA[imgIdx];
        float r = (packed & 0xFF) / 255f;
        float g = ((packed >> 8) & 0xFF) / 255f;
        float b = ((packed >> 16) & 0xFF) / 255f;

        // Unproject to camera space (same as single-image path)
        float camX = -((imgX - cx) * d / fx);
        float camY = -((imgY - cy) * d / fy);
        float camZ = d;

        // Transform camera space → world space: pos_world = R^T * pos_cam + C_world
        // R is the world→camera rotation matrix, so R^T rotates camera→world.
        // C_world is the camera center in world coordinates (passed as tx, ty, tz).
        float worldX = r00 * camX + r10 * camY + r20 * camZ + tx;
        float worldY = r01 * camX + r11 * camY + r21 * camZ + ty;
        float worldZ = r02 * camX + r12 * camY + r22 * camZ + tz;

        int slot = Atomic.Add(ref counter[0], 1);
        int outOff = slot * 10;

        outPacked[outOff + 0] = worldX;
        outPacked[outOff + 1] = worldY;
        outPacked[outOff + 2] = worldZ;
        outPacked[outOff + 3] = r;
        outPacked[outOff + 4] = g;
        outPacked[outOff + 5] = b;
        outPacked[outOff + 6] = splatScale;
        outPacked[outOff + 7] = splatScale;
        outPacked[outOff + 8] = splatScale * 0.5f;
        outPacked[outOff + 9] = 0.9f;
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _unprojectWorldSpaceKernel;

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer from GPU-resident depth + CPU RGBA.
    /// Returns (packedBuf, validSplatCount) — ownership of packedBuf transfers to caller.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, ImportedImage image, int subsample = 2,
            float edgeSharpness = 0.3f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        // Upload RGBA to GPU — justified: image data from file/picker (CPU source boundary).
        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        return await RunUnprojectAsync(accelerator, depth, rgbaBuf.View, subsample, edgeSharpness);
    }

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer from GPU-resident depth + GPU-resident RGBA.
    /// SR fast path — skips CPU→GPU upload.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, GpuImage gpuImage, int subsample = 2,
            float edgeSharpness = 0.3f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        return await RunUnprojectAsync(accelerator, depth, gpuImage.PackedRgba.View, subsample, edgeSharpness);
    }

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer in WORLD space using SfM-recovered camera parameters.
    /// Each splat position is transformed from camera-local to world coordinates via (R, t).
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferWorldSpaceAsync(DepthResult depth, ImportedImage image,
            CameraParams camera, int subsample = 2, float edgeSharpness = 0.3f,
            float depthScale = 1.0f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        // Upload RGBA to GPU
        var packedRgba = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        // Build rotation matrix from camera Forward/Up/Right
        // Camera axes: Right=X, Up=Y, Forward=Z (view direction)
        var right = camera.Right;
        var up = camera.Up;
        var fwd = camera.Forward;

        // R = [right | up | forward] as rows (world→camera rotation)
        var paramArr = new float[]
        {
            w, h, camera.FocalX, camera.FocalY, camera.CenterX, camera.CenterY,
            subsample,
            depth.MinDepth,
            depth.MaxDepth,
            edgeSharpness,
            // Rotation matrix (row-major, world→camera)
            right.X, right.Y, right.Z,
            -up.X, -up.Y, -up.Z,      // negate up because camera Y is typically flipped
            fwd.X, fwd.Y, fwd.Z,
            // Camera world position
            camera.Position.X, camera.Position.Y, camera.Position.Z,
            // Depth scale (maps inverted disparity to SfM coordinate system)
            depthScale,
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);

        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * 10);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null — GPU path requires GPU-resident depth.");

        _unprojectWorldSpaceKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            outPackedBuf.View,
            counterBuf.View,
            paramBuf.View);

        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] World-space: {validCount:N0} splats, pos={camera.Position}, scale={depthScale:F4}");
        Console.WriteLine($"[DepthGPU]   R=[{right.X:F3},{right.Y:F3},{right.Z:F3} | {-up.X:F3},{-up.Y:F3},{-up.Z:F3} | {fwd.X:F3},{fwd.Y:F3},{fwd.Z:F3}]");

        return (outPackedBuf, validCount);
    }

    /// <summary>
    /// Generate splats using the single-image pipeline but with a pixel offset applied.
    /// Used by multi-view fusion: the offset shifts each view's splats into the reference frame.
    /// The offset (dx, dy) represents the pixel displacement of this image relative to the reference.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferWithOffsetAsync(DepthResult depth, ImportedImage image,
            float pixelOffsetX, float pixelOffsetY,
            int subsample = 2, float edgeSharpness = 0.3f,
            int refWidth = 0, int refHeight = 0,
            float depthScaleCorrection = 1.0f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        var packedRgba = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        float fx = MathF.Max(w, h) * 1.2f;
        float fy = fx;
        float cx = w / 2f + pixelOffsetX;
        float cy = h / 2f + pixelOffsetY;

        // Compute exclusion rectangle: the region in THIS image that overlaps with the reference.
        // Pixel (x, y) in this image maps to reference pixel (x + dx, y + dy).
        // Overlap = where (x + dx) ∈ [0, refW) AND (y + dy) ∈ [0, refH)
        // → x ∈ [-dx, refW - dx) clamped to [0, w)
        int exclX0 = 0, exclY0 = 0, exclX1 = 0, exclY1 = 0;
        if (refWidth > 0 && refHeight > 0 && (pixelOffsetX != 0 || pixelOffsetY != 0))
        {
            exclX0 = Math.Clamp((int)(-pixelOffsetX), 0, w);
            exclY0 = Math.Clamp((int)(-pixelOffsetY), 0, h);
            exclX1 = Math.Clamp((int)(refWidth - pixelOffsetX), 0, w);
            exclY1 = Math.Clamp((int)(refHeight - pixelOffsetY), 0, h);
        }

        var paramArr = new float[]
        {
            w, h, fx, fy, cx, cy,
            subsample,
            depth.MinDepth,
            depth.MaxDepth,
            edgeSharpness,
            exclX0, exclY0, exclX1, exclY1,
            depthScaleCorrection,
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);

        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * 10);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null.");

        _unprojectAndPackKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            outPackedBuf.View,
            counterBuf.View,
            paramBuf.View);

        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        // Readback param buffer to verify GPU sees correct values
        float[] gpuParams;
        try { gpuParams = await paramBuf.CopyToHostAsync<float>(0, paramArr.Length); }
        catch { gpuParams = Array.Empty<float>(); }
        string gpuExcl = gpuParams.Length >= 14 ? $"gpu_excl=[{gpuParams[10]:F0},{gpuParams[11]:F0}→{gpuParams[12]:F0},{gpuParams[13]:F0}]" : "gpu_readback_failed";

        Console.WriteLine($"[DepthGPU] Offset: {validCount:N0} splats (offset=({pixelOffsetX:F1},{pixelOffsetY:F1}), excl=[{exclX0},{exclY0}→{exclX1},{exclY1}], {gpuExcl}, img={w}x{h})");

        return (outPackedBuf, validCount);
    }

    private void EnsureKernelLoaded(WebGPUAccelerator accelerator)
    {
        _unprojectAndPackKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(UnprojectAndPackKernel);

        _unprojectWorldSpaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(UnprojectWorldSpaceKernel);
    }

    /// <summary>
    /// Shared unprojection pipeline: GPU-resident depth + GPU-resident packed RGBA → compacted splat buffer.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        RunUnprojectAsync(WebGPUAccelerator accelerator, DepthResult depth,
            ArrayView1D<int, Stride1D.Dense> rgbaView, int subsample, float edgeSharpness)
    {
        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        float fx = MathF.Max(w, h) * 1.2f;
        float fy = fx;
        float cx = w / 2f;
        float cy = h / 2f;

        var paramArr = new float[]
        {
            w, h, fx, fy, cx, cy,
            subsample,
            depth.MinDepth,
            depth.MaxDepth,
            edgeSharpness,
            0, 0, 0, 0, // no exclusion for single-image path
            1.0f,       // no depth scale correction
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

        _unprojectAndPackKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaView,
            outPackedBuf.View,
            counterBuf.View,
            paramBuf.View);

        // Readback valid splat count only (4 bytes)
        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] Compacted: {validCount:N0} valid / {numPoints:N0} candidate splats " +
            $"(subsample={subsample}, edgeSharpness={edgeSharpness:F2})");

        return (outPackedBuf, validCount);
    }
}
