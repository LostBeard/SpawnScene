using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.BlazorJS.OnnxRuntimeWeb;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Monocular depth estimation using DepthAnythingV2 ONNX models.
/// Full GPU pipeline: RGBA→NCHW preprocessing on GPU, ONNX inference on the shared
/// GPUDevice (zero-copy input via TensorFromGpuBuffer), output kept GPU-resident via
/// ExternalWebGPUMemoryBuffer, depth resized + min/max computed on GPU.
/// Only 2 floats (min/max) are ever read back to CPU.
/// </summary>
public class DepthEstimationService : IAsyncDisposable
{
    private const string ModelPath = "models/depth_anything_v2_small.onnx";
    private const string ModelName = "DepthAnythingV2 Small";

    private readonly GpuService _gpu;
    private OnnxRuntime? _ort;
    private OrtInferenceSession? _session;

    // GPU kernel delegates — loaded lazily, cached across calls
    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,    // srcRgba (packed int per pixel)
        ArrayView1D<float, Stride1D.Dense>,  // dstNchw (NCHW float output)
        ArrayView1D<float, Stride1D.Dense>>? // params [inputSize, origW, origH, mean×3, std×3]
        _preprocessKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // srcDepth
        ArrayView1D<float, Stride1D.Dense>,  // dstDepth
        int, int, int, int>?                 // srcW, srcH, dstW, dstH
        _resizeKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depth values
        ArrayView1D<int, Stride1D.Dense>>?   // minMaxOut [0]=min bits, [1]=max bits
        _minMaxKernel;

    public event Action? OnStateChanged;
    public string Status { get; private set; } = "";
    public bool IsLoading { get; private set; }
    public bool IsReady => _session != null;

    public DepthEstimationService(GpuService gpu)
    {
        _gpu = gpu;
    }

    /// <summary>
    /// Initialize ONNX Runtime and load the DepthAnythingV2 Small model.
    /// Injects the ILGPU GPUDevice into ort.env.webgpu so ORT and ILGPU share one
    /// device, enabling zero-copy buffer exchange.
    /// </summary>
    public async Task LoadModelAsync()
    {
        if (_session != null) return;

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

            Status = $"Loading {ModelName}...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            _session = await _ort.CreateInferenceSessionAsync(ModelPath, new SessionCreateOptions
            {
                ExecutionProviders = new[] { "webgpu", "wasm" },
                GraphOptimizationLevel = "all",
                LogSeverityLevel = 3,
            });

            Status = $"✅ {ModelName} loaded — inputs: [{string.Join(", ", _session.InputNames)}], outputs: [{string.Join(", ", _session.OutputNames)}]";
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
    /// GPU kernel: nearest-neighbor resize + NCHW layout + ImageNet normalize.
    /// Input:  packed RGBA ints [origW*origH]
    /// Output: NCHW float32 [3*inputSize*inputSize] ready for DepthAnythingV2
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
        int y = rem / inputSize;
        int x = rem % inputSize;

        int sx = x * origW / inputSize;
        int sy = y * origH / inputSize;
        sx = sx < 0 ? 0 : (sx >= origW ? origW - 1 : sx);
        sy = sy < 0 ? 0 : (sy >= origH ? origH - 1 : sy);

        int packed = srcRgba[sy * origW + sx];
        int shift = c * 8; // R=0, G=8, B=16
        float pixVal = (float)((packed >> shift) & 0xFF) / 255f;

        dstNchw[idx] = (pixVal - p[3 + c]) / p[6 + c];
    }

    /// <summary>
    /// GPU kernel: nearest-neighbor depth resize.
    /// Input:  depth [srcW*srcH]
    /// Output: depth [dstW*dstH]
    /// </summary>
    private static void ResizeDepthKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> srcDepth,
        ArrayView1D<float, Stride1D.Dense> dstDepth,
        int srcW, int srcH, int dstW, int dstH)
    {
        int y = idx / dstW;
        int x = idx % dstW;
        int sx = x * srcW / dstW;
        int sy = y * srcH / dstH;
        sx = sx < 0 ? 0 : (sx >= srcW ? srcW - 1 : sx);
        sy = sy < 0 ? 0 : (sy >= srcH ? srcH - 1 : sy);
        dstDepth[idx] = srcDepth[sy * srcW + sx];
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
        ArrayView1D<int, Stride1D.Dense> minMaxOut)
    {
        float v = depth[idx];
        if (v > 0f)
        {
            // Interop.FloatAsInt returns uint; IEEE 754 positive float ordering is preserved
            // under uint bit-pattern comparison, so atomic min/max on int works correctly.
            int bits = (int)Interop.FloatAsInt(v);
            Atomic.Min(ref minMaxOut[0], bits);
            Atomic.Max(ref minMaxOut[1], bits);
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Inference
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Run depth estimation on an image. Returns a GPU-resident DepthResult.
    /// Full GPU pipeline:
    ///   1. Upload RGBA bytes to GPU (one-time — image loaded from disk)
    ///   2. GPU preprocess: RGBA → NCHW float32 normalized (kernel)
    ///   3. ONNX inference on shared GPUDevice (zero-copy tensor input)
    ///   4. Output stays GPU-resident via ExternalWebGPUMemoryBuffer
    ///   5. GPU resize: 518×518 → origW×origH (kernel)
    ///   6. GPU min/max reduce → 8 bytes CPU readback for scalar metadata only
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

        // Load kernels on first call (cached for reuse)
        _preprocessKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(PreprocessRgbaKernel);

        _resizeKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(ResizeDepthKernel);

        _minMaxKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(MinMaxKernel);

        int origW = image.Width;
        int origH = image.Height;
        const int inputSize = 518; // ViT patch size × 37

        Status = "Preprocessing image on GPU...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        // ── Step 1: Upload RGBA to GPU ──────────────────────────────────────
        // Justified: image data loaded from disk/file picker — unavoidable source boundary.
        // Reinterpret byte[W*H*4] as int[W*H] (packed RGBA, zero-copy in managed memory).
        var packedRgba = System.Runtime.InteropServices.MemoryMarshal
            .Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        // ── Step 2: GPU preprocess — RGBA → NCHW float32 (normalized) ──────
        var paramArr = new float[]
        {
            inputSize, origW, origH,
            0.485f, 0.456f, 0.406f, // ImageNet mean (R, G, B)
            0.229f, 0.224f, 0.225f, // ImageNet std  (R, G, B)
        };
        using var paramBuf = accelerator.Allocate1D(paramArr);
        using var preprocessBuf = accelerator.Allocate1D<float>(3 * inputSize * inputSize);

        _preprocessKernel(3 * inputSize * inputSize, rgbaBuf.View, preprocessBuf.View, paramBuf.View);

        await accelerator.SynchronizeAsync();

        // Get the underlying GPUBuffer to create a zero-copy ORT input tensor.
        // Both preprocessBuf and the session use the same GPUDevice (set in LoadModelAsync).
        var gpuInputBuffer = GetNativeBuffer(preprocessBuf);
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
            using var inputTensor = _ort.TensorFromGpuBuffer(gpuInputBuffer,
                new TensorFromGpuBufferOptions
                {
                    DataType = "float32",
                    Dims = new long[] { 1, 3, inputSize, inputSize },
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
                    outW, outH, origW, origH);
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
                    outW, outH, origW, origH);
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
        int srcW, int srcH, int dstW, int dstH)
    {
        var resizedBuf = accelerator.Allocate1D<float>(dstW * dstH);
        _resizeKernel!(dstW * dstH, rawView, resizedBuf.View, srcW, srcH, dstW, dstH);

        using var minMaxBuf = accelerator.Allocate1D<int>(2);
        minMaxBuf.CopyFromCPU(new int[] { BitConverter.SingleToInt32Bits(float.MaxValue), 0 });
        _minMaxKernel!(dstW * dstH, resizedBuf.View, minMaxBuf.View);

        await accelerator.SynchronizeAsync();

        // Only 8 bytes of CPU readback: scalar metadata for display and kernel params
        int[] mmResult = await minMaxBuf.CopyToHostAsync<int>(0, 2);
        float minD = BitConverter.Int32BitsToSingle(mmResult[0]);
        float maxD = mmResult[1] != 0 ? BitConverter.Int32BitsToSingle(mmResult[1]) : 1f;
        if (minD >= float.MaxValue - 1f || minD >= maxD) { minD = 0f; maxD = 1f; }

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


    /// <summary>Extract the native WebGPU GPUBuffer from an ILGPU MemoryBuffer.</summary>
    private static GPUBuffer? GetNativeBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> buf)
        where T : unmanaged
    {
        var iView = (IArrayView)(MemoryBuffer)buf;
        if (iView.Buffer is WebGPUMemoryBuffer webGpuMem)
            return webGpuMem.NativeBuffer?.NativeBuffer;
        return null;
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
