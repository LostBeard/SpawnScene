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
/// <param name="IsNative">True if this model runs via native ILGPU inference (no ORT).</param>
public record DepthModelInfo(string Id, string Name, string Path, string SizeLabel, bool IsDirectDepth = false, bool IsNative = false);

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
        new DepthModelInfo("depth-anything-v3-small-native", "DAv3 Native (ILGPU)", "models/dav3_weights", "~50 MB", IsDirectDepth: true, IsNative: true),
    };

    public static readonly string DefaultModelId = "depth-anything-v3-small";

    private readonly GpuService _gpu;
    private readonly HttpClient _http;
    private OnnxRuntime? _ort;
    private OrtInferenceSession? _session;
    private SpawnDev.ILGPU.ML.Dav3Inference? _nativeInference;
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

    // Unguided bilinear upsample for native depth pipeline (matches the Interpolate op in ORT DAv3)
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // src depth
        ArrayView1D<float, Stride1D.Dense>,  // dst depth
        int, int, int, int>?                 // srcW, srcH, dstW, dstH
        _bilinearDepthKernel;

    // In-place exp() for native depth: ONNX graph applies Exp to raw DPT depth channel.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _expInPlaceKernel;

    // WebGPU hard limit: maxComputeWorkgroupsPerDimension = 65535.
    // ILGPU WebGPU 1D auto-grouped kernels use group size 64.
    // Batch large dispatches to stay within (65535 * 64 = 4,194,240) elements per call.
    private const int MaxDispatchElements = 65535 * 64;

    public event Action? OnStateChanged;
    public string Status { get; private set; } = "";
    public bool IsLoading { get; private set; }
    public bool IsReady => _session != null || _nativeInference?.IsInitialized == true;

    public DepthEstimationService(GpuService gpu, HttpClient http)
    {
        _gpu = gpu;
        _http = http;
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
        if (LoadedModelId == modelId && IsReady) return;

        // Switching models: dispose old session / native inference
        if (_session != null)
        {
            _session.Dispose();
            _session = null;
        }
        if (_nativeInference != null)
        {
            _nativeInference.Dispose();
            _nativeInference = null;
        }
        LoadedModelId = null;
        LoadedModelName = null;

        IsLoading = true;
        OnStateChanged?.Invoke();

        // ── Native ILGPU path ──────────────────────────────────────────
        if (model.IsNative)
        {
            try
            {
                if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
                var accelerator = _gpu.WebGPUAccelerator;

                Status = $"Loading {model.Name} weights...";
                OnStateChanged?.Invoke();
                await Task.Yield();

                var weightLoader = new SpawnDev.ILGPU.ML.WeightLoader(accelerator, _http);
                await weightLoader.LoadAsync();

                _nativeInference = new SpawnDev.ILGPU.ML.Dav3Inference(accelerator, weightLoader);
                _nativeInference.Initialize();

                LoadedModelId = modelId;
                LoadedModelName = model.Name;
                LoadedModelIsDirectDepth = model.IsDirectDepth;
                Status = $"✅ {model.Name} ready";
                Console.WriteLine($"[Depth] {Status}");
            }
            catch (Exception ex)
            {
                Status = $"❌ Failed to load native model: {ex.Message}";
                Console.WriteLine($"[Depth] Native load error: {ex}");
            }
            finally
            {
                IsLoading = false;
                OnStateChanged?.Invoke();
            }
            return;
        }

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
    /// GPU kernel: simple bilinear depth upsample (no color guidance).
    /// Matches the Interpolate op in the DAv3 ONNX graph that upsamples DPT output to 518×518.
    /// Align-corners = false (same as PyTorch interpolate default).
    /// </summary>
    private static void BilinearDepthKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int srcW, int srcH, int dstW, int dstH)
    {
        int dstX = idx % dstW;
        int dstY = idx / dstW;

        float fy = ((dstY + 0.5f) * srcH / dstH) - 0.5f;
        float fx = ((dstX + 0.5f) * srcW / dstW) - 0.5f;

        int y0 = (int)fy; if (y0 < 0) y0 = 0;
        int y1 = y0 + 1;  if (y1 >= srcH) y1 = srcH - 1;
        int x0 = (int)fx; if (x0 < 0) x0 = 0;
        int x1 = x0 + 1;  if (x1 >= srcW) x1 = srcW - 1;

        float ty = fy - y0; if (ty < 0f) ty = 0f;
        float tx = fx - x0; if (tx < 0f) tx = 0f;

        dst[idx] = src[y0 * srcW + x0] * (1f - ty) * (1f - tx)
                 + src[y0 * srcW + x1] * (1f - ty) * tx
                 + src[y1 * srcW + x0] * ty * (1f - tx)
                 + src[y1 * srcW + x1] * ty * tx;
    }

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
    /// Uses sortable-int encoding so IEEE 754 float ordering is preserved for both
    /// positive AND negative values under signed int comparison.
    /// Output: [0] = min sortable bits, [1] = max sortable bits
    /// </summary>
    private static void MinMaxKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> minMaxOut,
        int offset)
    {
        float v = depth[idx + offset];
        if (v != v) return; // skip NaN
        int bits = (int)Interop.FloatAsInt(v);
        // Positive floats: bit pattern already sorts correctly as signed int.
        // Negative floats: flip all except sign bit so more-negative → smaller int.
        int sortable = bits >= 0 ? bits : (bits ^ 0x7FFFFFFF);
        Atomic.Min(ref minMaxOut[0], sortable);
        Atomic.Max(ref minMaxOut[1], sortable);
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

    /// <summary>In-place exp(x) — matches the Exp op in the ONNX graph after the DPT head.</summary>
    private static void ExpInPlaceKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        data[idx] = MathF.Exp(data[idx]);
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
        if (!IsReady)
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
    /// Multi-view depth estimation result: per-view depth maps + DAv3-predicted camera poses.
    /// </summary>
    public class MultiViewDepthResult : IDisposable
    {
        public List<DepthResult> DepthResults { get; set; } = new();
        /// <summary>DAv3 extrinsics: [R|t] 3×4 matrix per view. null if model doesn't output them.</summary>
        public float[,][]? Extrinsics { get; set; }
        public void Dispose()
        {
            foreach (var d in DepthResults) d.Dispose();
            DepthResults.Clear();
        }
    }

    /// <summary>
    /// Run DAv3 multi-view inference: all images in one pass → consistent depth + camera poses.
    /// Requires DAv3 model (5D input). Falls back to per-image inference for other models.
    /// </summary>
    public async Task<MultiViewDepthResult?> EstimateDepthMultiViewAsync(IReadOnlyList<ImportedImage> images)
    {
        if (!IsReady)
        {
            Status = "Model not loaded.";
            return null;
        }

        // Native model doesn't support 5D batch — fall back to per-image.
        // Non-DAv3 ORT models also fall back (no multi-view input shape).
        bool isDav3Ort = LoadedModelId == "depth-anything-v3-small" && _session != null;
        if (!isDav3Ort)
        {
            // Fallback: run each image independently
            var result = new MultiViewDepthResult();
            foreach (var img in images)
            {
                var d = await EstimateDepthAsync(img);
                if (d != null) result.DepthResults.Add(d);
            }
            return result;
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelsLoaded(accelerator);

        const int inputSize = 518;
        int N = images.Count;
        int pixelsPerImage = 3 * inputSize * inputSize;

        Status = $"Preprocessing {N} images on GPU...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        // Preprocess all N images into one contiguous buffer: [N * 3 * 518 * 518]
        using var preprocessBuf = accelerator.Allocate1D<float>(N * pixelsPerImage);
        var paramArr = new float[]
        {
            inputSize, 0, 0, // origW/origH filled per image
            0.485f, 0.456f, 0.406f,
            0.229f, 0.224f, 0.225f,
        };

        for (int i = 0; i < N; i++)
        {
            var img = images[i];
            paramArr[1] = img.Width;
            paramArr[2] = img.Height;

            // Upload RGBA for this image
            var packedRgba = System.Runtime.InteropServices.MemoryMarshal
                .Cast<byte, int>(img.RgbaPixels.AsSpan()).ToArray();
            using var rgbaBuf = accelerator.Allocate1D(packedRgba);
            using var paramBuf = accelerator.Allocate1D(paramArr);

            // Write into the correct slice of the combined buffer
            var sliceView = preprocessBuf.View.SubView(i * pixelsPerImage, pixelsPerImage);
            _preprocessKernel!(pixelsPerImage, rgbaBuf.View, sliceView, paramBuf.View);
        }

        await accelerator.SynchronizeAsync();

        var gpuInputBuffer = preprocessBuf.GetGPUBuffer();
        if (gpuInputBuffer == null)
        {
            Status = "Could not access GPU buffer for multi-view input.";
            return null;
        }

        Status = $"Running DAv3 multi-view inference ({N} images)...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            using var inputTensor = _ort.TensorFromGpuBuffer(gpuInputBuffer,
                new TensorFromGpuBufferOptions
                {
                    DataType = "float32",
                    Dims = new long[] { 1, N, 3, inputSize, inputSize },
                });

            using var feeds = new OrtFeeds();
            feeds.Set(_session.InputNames[0], inputTensor);

            using var ortResult = await _session.Run(feeds);

            // Parse predicted_depth: [1, N, H, W]
            using var depthTensor = ortResult.GetTensor("predicted_depth");
            var depthDims = depthTensor.Dims;
            int outH = (int)depthDims[depthDims.Length - 2];
            int outW = (int)depthDims[depthDims.Length - 1];
            Console.WriteLine($"[Depth] Multi-view output: [{string.Join(", ", depthDims)}], location: {depthTensor.Location}");

            // Read depth to CPU for per-view slicing → re-upload per view for guided upsampling.
            // CPU transfer justified: splitting [1,N,H,W] into N separate buffers.
            using var depthData = depthTensor.GetData<SpawnDev.BlazorJS.JSObjects.Float32Array>();
            float[] allDepth = depthData.ToArray();
            Console.WriteLine($"[Depth] Multi-view depth: {allDepth.Length} total floats for {N} views");

            // Parse extrinsics: [1, N, 3, 4] if available
            float[][]? extrinsicsPerView = null;
            try
            {
                using var extTensor = ortResult.GetTensor("extrinsics");
                float[] extData;
                // CPU readback of extrinsics (12 floats per view — tiny, always OK)
                using var ed = extTensor.GetData<SpawnDev.BlazorJS.JSObjects.Float32Array>();
                extData = ed.ToArray();

                extrinsicsPerView = new float[N][];
                for (int i = 0; i < N; i++)
                {
                    extrinsicsPerView[i] = new float[12];
                    System.Array.Copy(extData, i * 12, extrinsicsPerView[i], 0, 12);
                    Console.WriteLine($"[Depth] View {i} extrinsics: [{string.Join(", ", extrinsicsPerView[i].Select(v => v.ToString("F4")))}]");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Depth] No extrinsics output: {ex.Message}");
            }

            // Build per-view DepthResults with guided upsampling
            var result = new MultiViewDepthResult();
            int depthSliceSize = outH * outW;

            for (int i = 0; i < N; i++)
            {
                int origW = images[i].Width;
                int origH = images[i].Height;

                // Upload this view's depth slice to GPU
                var viewDepth = new float[depthSliceSize];
                System.Array.Copy(allDepth, i * depthSliceSize, viewDepth, 0, depthSliceSize);
                using var rawBuf = accelerator.Allocate1D(viewDepth);

                // Upload RGBA guide for this view
                var packedRgba = System.Runtime.InteropServices.MemoryMarshal
                    .Cast<byte, int>(images[i].RgbaPixels.AsSpan()).ToArray();
                using var guideBuf = accelerator.Allocate1D(packedRgba);

                // Guided upsample + min/max
                var depthResult = await RunResizeMinMaxAsync(accelerator, rawBuf.View,
                    guideBuf.View, outW, outH, origW, origH);

                if (depthResult != null)
                    result.DepthResults.Add(depthResult);
            }

            if (extrinsicsPerView != null)
            {
                // Store as jagged array for now; MultiViewGenerationService will parse [R|t]
                result.Extrinsics = new float[1, N][];
                for (int i = 0; i < N; i++)
                    result.Extrinsics[0, i] = extrinsicsPerView[i];
            }

            return result;
        }
        catch (Exception ex)
        {
            Status = $"Multi-view inference failed: {ex.Message}";
            Console.WriteLine($"[Depth] Multi-view error: {ex}");
            return null;
        }
    }

    /// <summary>
    /// Run depth estimation on a GPU-resident image (SR fast path).
    /// Skips CPU→GPU upload — packed RGBA is already on GPU.
    /// </summary>
    public async Task<DepthResult?> EstimateDepthAsync(GpuImage gpuImage)
    {
        if (!IsReady)
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

        _bilinearDepthKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(BilinearDepthKernel);

        _expInPlaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>>(ExpInPlaceKernel);
    }

    /// <summary>
    /// Native ILGPU depth pipeline: preprocess → Dav3Inference.RunFull → resize + min/max.
    /// </summary>
    private async Task<DepthResult?> RunNativeDepthPipelineAsync(
        WebGPUAccelerator accelerator,
        ArrayView1D<int, Stride1D.Dense> rgbaView,
        int origW, int origH)
    {
        const int inputSize = 518;

        Status = "Preprocessing image (native)...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        EnsureKernelsLoaded(accelerator);

        var paramArr = new float[]
        {
            inputSize, origW, origH,
            0.485f, 0.456f, 0.406f,
            0.229f, 0.224f, 0.225f,
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);
        using var preprocessBuf = accelerator.Allocate1D<float>(3 * inputSize * inputSize);
        _preprocessKernel!(3 * inputSize * inputSize, rgbaView, preprocessBuf.View, paramBuf.View);

        Status = "Running DAv3 native inference...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        // RunFull enqueues all GPU commands (backbone + DPT head) synchronously.
        // preprocessBuf stays alive until after SynchronizeAsync (inside RunResizeMinMaxAsync).
        var depthBuf = _nativeInference!.RunFull(preprocessBuf.View);

        // depthBuf is [2, OutputH, OutputW] — channel 0 = depth, channel 1 = confidence.
        int outH = SpawnDev.ILGPU.ML.DptHead.OutputH;  // 296
        int outW = SpawnDev.ILGPU.ML.DptHead.OutputW;  // 296
        var depthCh0 = depthBuf.View.SubView(0, outH * outW);

        // ONNX graph applies Exp() to raw DPT depth channel (converts log-depth to depth).
        _expInPlaceKernel!(outH * outW, depthCh0);

        // DptHead now outputs at 518×518 (resize moved inside head to match ONNX graph).
        // No separate bilinear upsample needed — go straight to guided bilateral upsample.
        return await RunResizeMinMaxAsync(accelerator, depthCh0, rgbaView, outW, outH, origW, origH);
    }

    /// <summary>
    /// Shared depth pipeline: preprocess GPU-resident packed RGBA → ORT inference → resize + min/max.
    /// </summary>
    private async Task<DepthResult?> RunDepthPipelineAsync(
        WebGPUAccelerator accelerator,
        ArrayView1D<int, Stride1D.Dense> rgbaView,
        int origW, int origH)
    {
        if (_nativeInference != null)
            return await RunNativeDepthPipelineAsync(accelerator, rgbaView, origW, origH);

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
        // Initialize with sortable-int sentinels: int.MaxValue for min (any value is smaller),
        // int.MinValue for max (any value is larger).
        minMaxBuf.CopyFromCPU(new int[] { int.MaxValue, int.MinValue });
        for (int offset = 0; offset < totalPixels; offset += MaxDispatchElements)
        {
            int count = Math.Min(MaxDispatchElements, totalPixels - offset);
            _minMaxKernel!(count, resizedBuf.View, minMaxBuf.View, offset);
        }

        await accelerator.SynchronizeAsync();

        // Only 8 bytes of CPU readback: scalar metadata for display and kernel params.
        // Convert from sortable-int encoding back to float bits.
        int[] mmResult = await minMaxBuf.CopyToHostAsync<int>(0, 2);
        int sortMin = mmResult[0], sortMax = mmResult[1];
        int bitsMin = sortMin >= 0 ? sortMin : (sortMin ^ 0x7FFFFFFF);
        int bitsMax = sortMax >= 0 ? sortMax : (sortMax ^ 0x7FFFFFFF);
        float minD = BitConverter.Int32BitsToSingle(bitsMin);
        float maxD = BitConverter.Int32BitsToSingle(bitsMax);
        if (float.IsNaN(minD) || float.IsNaN(maxD) || minD >= maxD) { minD = 0f; maxD = 1f; }

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
        _nativeInference?.Dispose();
        _nativeInference = null;
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
