using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.BlazorJS.OnnxRuntimeWeb;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Available depth estimation model definition.
/// All models use the same 518×518 ViT input and ImageNet normalization.
/// </summary>
/// <param name="IsDirectDepth">True if model outputs depth (high=far), false if disparity (high=close).</param>
public record DepthModelInfo(string Id, string Name, string Path, string SizeLabel, bool IsDirectDepth = false);

/// <summary>
/// Monocular depth estimation using ONNX models (DepthAnythingV2, DistillAnyDepth).
/// Full GPU pipeline: RGBA→NCHW preprocessing on GPU, ONNX inference on the shared
/// GPUDevice (zero-copy input via TensorFromGpuBuffer), output kept GPU-resident via
/// ExternalWebGPUMemoryBuffer, depth resized + min/max computed on GPU.
/// Only 2 floats (min/max) are ever read back to CPU.
/// </summary>
public class DepthEstimationService : IAsyncDisposable
{
    public static readonly DepthModelInfo[] AvailableModels = new[]
    {
        new DepthModelInfo("distill-any-depth-small", "DistillAnyDepth Small", "models/distill_any_depth_small.onnx", "~99 MB"),
        new DepthModelInfo("depth-anything-v2-small", "DepthAnythingV2 Small", "models/depth_anything_v2_small.onnx", "~99 MB"),
        new DepthModelInfo("depth-anything-v3-small", "DepthAnythingV3 Small", "models/depth_anything_v3_small_fp16.onnx", "~50 MB", IsDirectDepth: true),
    };

    public static readonly string DefaultModelId = "depth-anything-v3-small";

    private readonly GpuService _gpu;
    private OnnxRuntime? _ort;
    private OrtInferenceSession? _session;
    public string? LoadedModelId { get; private set; }
    public string? LoadedModelName { get; private set; }
    public bool LoadedModelIsDirectDepth { get; private set; }

    // GPU kernel delegates — loaded lazily, cached across calls
    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,    // srcRgba (packed int per pixel)
        ArrayView1D<float, Stride1D.Dense>,  // dstNchw (NCHW float output)
        ArrayView1D<float, Stride1D.Dense>>? // params [inputSize, origW, origH, mean×3, std×3]
        _preprocessKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // srcDepth
        ArrayView1D<int, Stride1D.Dense>,    // guideRgba (high-res color for edge guidance)
        ArrayView1D<float, Stride1D.Dense>,  // dstDepth
        int, int, int, int, int>?            // offset, srcW, srcH, dstW, dstH
        _guidedUpsampleKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depth values
        ArrayView1D<int, Stride1D.Dense>,    // minMaxOut [0]=min bits, [1]=max bits
        int>?                                // offset
        _minMaxKernel;

    // Flips depth values: out[i] = (min + max) - in[i], converting direct depth → disparity-like
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depth buffer (in-place)
        float, float, int>?                  // min, max, offset
        _flipDepthKernel;

    // WebGPU hard limit: maxComputeWorkgroupsPerDimension = 65535.
    // ILGPU WebGPU 1D auto-grouped kernels use group size 64.
    // Batch large dispatches to stay within (65535 * 64 = 4,194,240) elements per call.
    private const int MaxDispatchElements = 65535 * 64;

    public event Action? OnStateChanged;
    public string Status { get; private set; } = "";
    public bool IsLoading { get; private set; }
    public bool IsReady => _session != null;

    public DepthEstimationService(GpuService gpu)
    {
        _gpu = gpu;
    }

    /// <summary>
    /// Initialize ONNX Runtime and load the specified depth model.
    /// Injects the ILGPU GPUDevice into ort.env.webgpu so ORT and ILGPU share one
    /// device, enabling zero-copy buffer exchange.
    /// </summary>
    public async Task LoadModelAsync(string modelId)
    {
        var model = AvailableModels.FirstOrDefault(m => m.Id == modelId);
        if (model == null)
        {
            Status = $"❌ Unknown model: {modelId}";
            OnStateChanged?.Invoke();
            return;
        }

        // Already loaded this exact model
        if (_session != null && LoadedModelId == modelId) return;

        // Switching models: dispose old session
        if (_session != null)
        {
            _session.Dispose();
            _session = null;
            LoadedModelId = null;
            LoadedModelName = null;
        }

        IsLoading = true;
        OnStateChanged?.Invoke();

        try
        {
            if (_ort == null)
            {
                Status = "Initializing ONNX Runtime...";
                OnStateChanged?.Invoke();
                await Task.Yield();
                _ort = await OnnxRuntime.Init();
            }

            // Request GPU-resident output tensors so we never copy depth to CPU
            using var env = _ort.Env;
            env.SetPreferredOutputLocation("gpu-buffer");

            Status = $"Loading {model.Name}...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            _session = await _ort.CreateInferenceSessionAsync(model.Path, new SessionCreateOptions
            {
                ExecutionProviders = new[] { "webgpu", "wasm" },
                GraphOptimizationLevel = "all",
                LogSeverityLevel = 3,
            });

            LoadedModelId = modelId;
            LoadedModelName = model.Name;
            LoadedModelIsDirectDepth = model.IsDirectDepth;
            Status = $"✅ {model.Name} loaded — inputs: [{string.Join(", ", _session.InputNames)}], outputs: [{string.Join(", ", _session.OutputNames)}]";
            Console.WriteLine($"[Depth] {Status}");
        }
        catch (Exception ex)
        {
            Status = $"❌ Failed to load model: {ex.Message}";
            Console.WriteLine($"[Depth] Error: {ex}");
        }
        finally
        {
            IsLoading = false;
            OnStateChanged?.Invoke();
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernels
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: bicubic (Catmull-Rom) resize + NCHW layout + ImageNet normalize.
    /// Input:  packed RGBA ints [origW*origH]
    /// Output: NCHW float32 [3*inputSize*inputSize] ready for depth model
    /// Params: [0]=inputSize [1]=origW [2]=origH [3-5]=mean [6-8]=std
    /// </summary>
    private static void PreprocessRgbaKernel(
        Index1D idx,
        ArrayView1D<int, Stride1D.Dense> srcRgba,
        ArrayView1D<float, Stride1D.Dense> dstNchw,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        int inputSize = (int)p[0];
        int origW = (int)p[1];
        int origH = (int)p[2];

        int totalPix = inputSize * inputSize;
        int c = idx / totalPix;
        int rem = idx % totalPix;
        int dstY = rem / inputSize;
        int dstX = rem % inputSize;
        int shift = c * 8; // R=0, G=8, B=16

        // Map destination pixel to source coordinates (center-aligned)
        float srcXf = (dstX + 0.5f) * origW / inputSize - 0.5f;
        float srcYf = (dstY + 0.5f) * origH / inputSize - 0.5f;
        int x0 = (int)srcXf - (srcXf < (int)srcXf ? 1 : 0); // floor
        int y0 = (int)srcYf - (srcYf < (int)srcYf ? 1 : 0);
        float fx = srcXf - x0;
        float fy = srcYf - y0;

        // Catmull-Rom weights for 4 sample points at positions -1, 0, 1, 2
        float fx2 = fx * fx, fx3 = fx2 * fx;
        float fy2 = fy * fy, fy3 = fy2 * fy;

        float wx0 = -0.5f * fx3 + fx2 - 0.5f * fx;
        float wx1 = 1.5f * fx3 - 2.5f * fx2 + 1f;
        float wx2 = -1.5f * fx3 + 2f * fx2 + 0.5f * fx;
        float wx3 = 0.5f * fx3 - 0.5f * fx2;

        float wy0 = -0.5f * fy3 + fy2 - 0.5f * fy;
        float wy1 = 1.5f * fy3 - 2.5f * fy2 + 1f;
        float wy2 = -1.5f * fy3 + 2f * fy2 + 0.5f * fy;
        float wy3 = 0.5f * fy3 - 0.5f * fy2;

        // Sample 4×4 neighborhood with bicubic weights
        float val = 0f;
        for (int j = -1; j <= 2; j++)
        {
            float wy = j == -1 ? wy0 : j == 0 ? wy1 : j == 1 ? wy2 : wy3;
            int sy = y0 + j;
            sy = sy < 0 ? 0 : (sy >= origH ? origH - 1 : sy);

            for (int i = -1; i <= 2; i++)
            {
                float wx = i == -1 ? wx0 : i == 0 ? wx1 : i == 1 ? wx2 : wx3;
                int sx = x0 + i;
                sx = sx < 0 ? 0 : (sx >= origW ? origW - 1 : sx);

                int packed = srcRgba[sy * origW + sx];
                float pixVal = ((packed >> shift) & 0xFF) / 255f;
                val += pixVal * wx * wy;
            }
        }

        // Clamp to [0,1] before normalization (bicubic can overshoot)
        val = val < 0f ? 0f : (val > 1f ? 1f : val);

        dstNchw[idx] = (val - p[3 + c]) / p[6 + c];
    }

    /// <summary>
    /// GPU kernel: joint bilateral upsampling — uses the high-res color image as an
    /// edge guide so depth boundaries align with color edges.
    /// Samples a 5×5 window in low-res depth, weighted by spatial distance and
    /// color similarity (Cauchy kernel, no exp needed).
    /// </summary>
    private static void GuidedDepthUpsampleKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> srcDepth,
        ArrayView1D<int, Stride1D.Dense> guideRgba,
        ArrayView1D<float, Stride1D.Dense> dstDepth,
        int offset, int srcW, int srcH, int dstW, int dstH)
    {
        int absIdx = idx + offset;
        int dstY = absIdx / dstW;
        int dstX = absIdx % dstW;

        // Map destination pixel to source coordinates
        float srcXf = (dstX + 0.5f) * srcW / dstW - 0.5f;
        float srcYf = (dstY + 0.5f) * srcH / dstH - 0.5f;
        int cx = (int)(srcXf + 0.5f);
        int cy = (int)(srcYf + 0.5f);
        cx = cx < 0 ? 0 : (cx >= srcW ? srcW - 1 : cx);
        cy = cy < 0 ? 0 : (cy >= srcH ? srcH - 1 : cy);

        // Guide color at this output pixel
        int centerPacked = guideRgba[dstY * dstW + dstX];
        float cR = (centerPacked & 0xFF) / 255f;
        float cG = ((centerPacked >> 8) & 0xFF) / 255f;
        float cB = ((centerPacked >> 16) & 0xFF) / 255f;

        // Joint bilateral: spatial × color similarity weights (Cauchy kernel)
        const float invSigmaSpace2 = 1f / (1.5f * 1.5f);  // spatial sigma = 1.5 low-res pixels
        const float invSigmaColor2 = 1f / (0.05f * 0.05f); // color sigma = 0.05 in [0,1]
        const int radius = 2; // 5×5 window

        float weightSum = 0f;
        float depthSum = 0f;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int sx = cx + dx;
                int sy = cy + dy;
                sx = sx < 0 ? 0 : (sx >= srcW ? srcW - 1 : sx);
                sy = sy < 0 ? 0 : (sy >= srcH ? srcH - 1 : sy);

                // Spatial weight: Cauchy kernel 1/(1 + d²/σ²)
                float dist2 = (float)(dx * dx + dy * dy);
                float ws = 1f / (1f + dist2 * invSigmaSpace2);

                // Color similarity: compare guide at output pixel vs guide at sample's position
                int gx = sx * dstW / srcW;
                int gy = sy * dstH / srcH;
                gx = gx >= dstW ? dstW - 1 : gx;
                gy = gy >= dstH ? dstH - 1 : gy;
                int samplePacked = guideRgba[gy * dstW + gx];
                float sR = (samplePacked & 0xFF) / 255f;
                float sG = ((samplePacked >> 8) & 0xFF) / 255f;
                float sB = ((samplePacked >> 16) & 0xFF) / 255f;

                float colorDiff2 = (cR - sR) * (cR - sR) + (cG - sG) * (cG - sG) + (cB - sB) * (cB - sB);
                float wc = 1f / (1f + colorDiff2 * invSigmaColor2);

                float w = ws * wc;
                weightSum += w;
                depthSum += srcDepth[sy * srcW + sx] * w;
            }
        }

        dstDepth[absIdx] = weightSum > 0f ? depthSum / weightSum : srcDepth[cy * srcW + cx];
    }

    /// <summary>
    /// GPU kernel: parallel min/max reduction using atomic operations.
    /// For positive depth values, IEEE 754 bit patterns preserve float ordering,
    /// so we atomically track min/max as int bit patterns.
    /// Output: [0] = min depth bits, [1] = max depth bits
    /// </summary>
    private static void MinMaxKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> minMaxOut,
        int offset)
    {
        float v = depth[idx + offset];
        if (v > 0f)
        {
            // Interop.FloatAsInt returns uint; IEEE 754 positive float ordering is preserved
            // under uint bit-pattern comparison, so atomic min/max on int works correctly.
            int bits = (int)Interop.FloatAsInt(v);
            Atomic.Min(ref minMaxOut[0], bits);
            Atomic.Max(ref minMaxOut[1], bits);
        }
    }

    /// <summary>
    /// GPU kernel: flip depth values in-place. Converts direct depth (high=far) to disparity-like (high=close).
    /// Formula: buf[i] = (min + max) - buf[i]
    /// </summary>
    private static void FlipDepthKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        float min, float max, int offset)
    {
        int i = idx + offset;
        depth[i] = (min + max) - depth[i];
    }

    // ─────────────────────────────────────────────────────────────
    //  Inference
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Run depth estimation on a CPU-resident image. Returns a GPU-resident DepthResult.
    /// Uploads RGBA to GPU, then runs the shared depth pipeline.
    /// </summary>
    public async Task<DepthResult?> EstimateDepthAsync(ImportedImage image)
    {
        if (_session == null || _ort == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelsLoaded(accelerator);

        // Upload RGBA to GPU — justified: image data from disk/file picker (CPU source boundary).
        var packedRgba = System.Runtime.InteropServices.MemoryMarshal
            .Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        return await RunDepthPipelineAsync(accelerator, rgbaBuf.View, image.Width, image.Height);
    }

    /// <summary>
    /// Run depth estimation on a GPU-resident image (SR fast path).
    /// Skips CPU→GPU upload — packed RGBA is already on GPU.
    /// </summary>
    public async Task<DepthResult?> EstimateDepthAsync(GpuImage gpuImage)
    {
        if (_session == null || _ort == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelsLoaded(accelerator);

        return await RunDepthPipelineAsync(accelerator, gpuImage.PackedRgba.View, gpuImage.Width, gpuImage.Height);
    }

    private void EnsureKernelsLoaded(WebGPUAccelerator accelerator)
    {
        _preprocessKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(PreprocessRgbaKernel);

        _guidedUpsampleKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int>(GuidedDepthUpsampleKernel);

        _minMaxKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            int>(MinMaxKernel);

        _flipDepthKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float, float, int>(FlipDepthKernel);
    }

    /// <summary>
    /// Shared depth pipeline: preprocess GPU-resident packed RGBA → ORT inference → resize + min/max.
    /// </summary>
    private async Task<DepthResult?> RunDepthPipelineAsync(
        WebGPUAccelerator accelerator,
        ArrayView1D<int, Stride1D.Dense> rgbaView,
        int origW, int origH)
    {
        const int inputSize = 518; // ViT patch size × 37

        Status = "Preprocessing image on GPU...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        // GPU preprocess — RGBA → NCHW float32 (resize + ImageNet normalize)
        var paramArr = new float[]
        {
            inputSize, origW, origH,
            0.485f, 0.456f, 0.406f, // ImageNet mean (R, G, B)
            0.229f, 0.224f, 0.225f, // ImageNet std  (R, G, B)
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);
        using var preprocessBuf = accelerator.Allocate1D<float>(3 * inputSize * inputSize);

        _preprocessKernel!(3 * inputSize * inputSize, rgbaView, preprocessBuf.View, paramBuf.View);

        await accelerator.SynchronizeAsync();

        var gpuInputBuffer = preprocessBuf.GetGPUBuffer();
        if (gpuInputBuffer == null)
        {
            Status = "❌ Could not access GPU buffer for ORT input.";
            return null;
        }

        Status = "Running depth inference...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            // ── Step 3: Zero-copy ORT inference ────────────────────────────
            // TensorFromGpuBuffer wraps our ILGPU buffer — no CPU copy of input data.
            // DAv3 expects [batch, num_images, 3, H, W] (5D); DAv2/DistillAnyDepth expect [1, 3, H, W] (4D).
            // Same physical buffer — only the logical shape changes.
            bool isDav3 = LoadedModelId?.StartsWith("depth-anything-v3") == true;
            var inputDims = isDav3
                ? new long[] { 1, 1, 3, inputSize, inputSize }
                : new long[] { 1, 3, inputSize, inputSize };

            using var inputTensor = _ort.TensorFromGpuBuffer(gpuInputBuffer,
                new TensorFromGpuBufferOptions
                {
                    DataType = "float32",
                    Dims = inputDims,
                });

            using var feeds = new OrtFeeds();
            feeds.Set(_session.InputNames[0], inputTensor);

            using var ortResult = await _session.Run(feeds);
            using var outputTensor = ortResult.GetTensor(_session.OutputNames[0]);

            var dims = outputTensor.Dims;
            int outH = (int)(dims.Length >= 2 ? dims[dims.Length - 2] : inputSize);
            int outW = (int)(dims.Length >= 1 ? dims[dims.Length - 1] : inputSize);

            Console.WriteLine($"[Depth] Output dims: [{string.Join(", ", dims)}], location: {outputTensor.Location}");

            // ── Step 4: Keep depth on GPU; run resize + min/max ─────────────
            if (outputTensor.Location == "gpu-buffer")
            {
                // Zero-copy: wrap ORT's GPUBuffer for use in ILGPU resize/minmax kernels.
                // Shared GPUDevice (set in LoadModelAsync) guarantees buffer compatibility.
                var ortGpuBuffer = outputTensor.GPUBuffer;
                long outElements = (long)outH * outW;
                using var externalBuf = new ExternalWebGPUMemoryBuffer(
                    accelerator, ortGpuBuffer, outElements, sizeof(float));

                // externalBuf + outputTensor remain alive while kernels execute and sync inside
                return await RunResizeMinMaxAsync(accelerator,
                    externalBuf.AsArrayView<float>(0, outElements),
                    rgbaView, outW, outH, origW, origH);
            }
            else
            {
                // WASM fallback: ORT ran on CPU — upload depth to GPU once then continue.
                // Acceptable CPU transfer: unavoidable when WebGPU EP is unavailable
                // (WebGPU is required by the app, but ORT's WebGPU EP may lag browser support).
                Console.WriteLine("[Depth] WASM fallback: uploading ORT output to GPU.");
                using var outputData = outputTensor.GetData<Float32Array>();
                float[] cpuDepth = outputData.ToArray();

                using var rawUploadBuf = accelerator.Allocate1D(cpuDepth);
                return await RunResizeMinMaxAsync(accelerator, rawUploadBuf.View,
                    rgbaView, outW, outH, origW, origH);
            }
        }
        catch (Exception ex)
        {
            Status = $"❌ Inference failed: {ex.Message}";
            Console.WriteLine($"[Depth] Error: {ex}");
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────

    private async Task<DepthResult?> RunResizeMinMaxAsync(
        WebGPUAccelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> rawView,
        ArrayView1D<int, Stride1D.Dense> guideRgba,
        int srcW, int srcH, int dstW, int dstH)
    {
        var resizedBuf = accelerator.Allocate1D<float>(dstW * dstH);
        int totalPixels = dstW * dstH;
        for (int offset = 0; offset < totalPixels; offset += MaxDispatchElements)
        {
            int count = Math.Min(MaxDispatchElements, totalPixels - offset);
            _guidedUpsampleKernel!(count, rawView, guideRgba, resizedBuf.View, offset, srcW, srcH, dstW, dstH);
        }

        using var minMaxBuf = accelerator.Allocate1D<int>(2);
        minMaxBuf.CopyFromCPU(new int[] { BitConverter.SingleToInt32Bits(float.MaxValue), 0 });
        for (int offset = 0; offset < totalPixels; offset += MaxDispatchElements)
        {
            int count = Math.Min(MaxDispatchElements, totalPixels - offset);
            _minMaxKernel!(count, resizedBuf.View, minMaxBuf.View, offset);
        }

        await accelerator.SynchronizeAsync();

        // Only 8 bytes of CPU readback: scalar metadata for display and kernel params
        int[] mmResult = await minMaxBuf.CopyToHostAsync<int>(0, 2);
        float minD = BitConverter.Int32BitsToSingle(mmResult[0]);
        float maxD = mmResult[1] != 0 ? BitConverter.Int32BitsToSingle(mmResult[1]) : 1f;
        if (minD >= float.MaxValue - 1f || minD >= maxD) { minD = 0f; maxD = 1f; }

        Console.WriteLine($"[Depth] Min/Max: [{minD:F6}, {maxD:F6}], range={maxD - minD:F6}");

        // Direct-depth models (DAv3): flip buffer so high value = close (disparity-like).
        // The unprojection kernel assumes disparity ordering (high=close, inverts to get depth).
        if (LoadedModelIsDirectDepth && maxD > minD)
        {
            Console.WriteLine($"[Depth] Flipping direct depth → disparity (min+max-val)");
            for (int offset = 0; offset < totalPixels; offset += MaxDispatchElements)
            {
                int count = Math.Min(MaxDispatchElements, totalPixels - offset);
                _flipDepthKernel!(count, resizedBuf.View, minD, maxD, offset);
            }
            await accelerator.SynchronizeAsync();
            // After flip: old min becomes new max, old max becomes new min, but numerical range is preserved
        }

        Status = $"✅ Depth estimated — range: [{minD:F3}, {maxD:F3}]";
        OnStateChanged?.Invoke();

        return new DepthResult
        {
            RawDepthGpu = resizedBuf,
            Width = dstW,
            Height = dstH,
            MinDepth = minD,
            MaxDepth = maxD,
        };
    }


    public async ValueTask DisposeAsync()
    {
        _session?.Dispose();
        _session = null;
        _ort?.Dispose();
        _ort = null;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// GPU-resident result of depth estimation.
/// The depth map lives entirely in GPU memory — no CPU float arrays.
/// Caller must Dispose() to release the GPU buffer.
/// </summary>
public class DepthResult : IDisposable
{
    /// <summary>
    /// GPU-resident raw depth values at original image resolution.
    /// Owned by this instance — disposed with it.
    /// For metric models: depth in meters.
    /// For relative models: raw disparity (use MinDepth/MaxDepth to normalize on GPU).
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? RawDepthGpu { get; set; }

    /// <summary>Width of the depth map (matches source image).</summary>
    public int Width { get; set; }

    /// <summary>Height of the depth map (matches source image).</summary>
    public int Height { get; set; }

    /// <summary>
    /// Minimum raw depth value (GPU-computed, 8-byte readback).
    /// Used by GPU kernels for on-GPU normalization — not for CPU processing.
    /// </summary>
    public float MinDepth { get; set; }

    /// <summary>
    /// Maximum raw depth value (GPU-computed, 8-byte readback).
    /// Used by GPU kernels for on-GPU normalization — not for CPU processing.
    /// </summary>
    public float MaxDepth { get; set; }



    public void Dispose()
    {
        RawDepthGpu?.Dispose();
        RawDepthGpu = null;
        GC.SuppressFinalize(this);
    }
}
