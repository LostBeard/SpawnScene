using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.BlazorJS.OnnxRuntimeWeb;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;
using System.Runtime.InteropServices;

namespace SpawnScene.Services;

/// <summary>
/// Optional super-resolution upscaling using an ONNX SR model.
/// Upscales the input image before depth estimation, producing a higher-resolution
/// depth map and more detailed Gaussian splats.
///
/// Compatible models: any ONNX SR model that accepts float32 NCHW [1,3,H,W] ∈ [0,1]
/// and outputs float32 NCHW [1,3,H*scale,W*scale] ∈ [0,1].
/// Suggested: Real-ESRGAN-x2plus or realesr-animevideov3-x2 (~3-5 MB ONNX export).
/// Place model files at wwwroot/models/sr_x2.onnx and wwwroot/models/sr_x4.onnx.
/// </summary>
public class SuperResolutionService : IAsyncDisposable
{
    public const string Model2xPath = "models/sr_x2.onnx";
    public const string Model4xPath = "models/sr_x4.onnx";

    private readonly GpuService _gpu;
    private OnnxRuntime? _ort;
    private OrtInferenceSession? _session;

    // GPU kernel delegates — loaded lazily, cached across calls
    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,    // srcRgba (packed RGBA int per pixel)
        ArrayView1D<float, Stride1D.Dense>,  // dstNchw (3*W*H float, values in [0,1])
        int, int, int>? _rgbaToNchwKernel;   // w, h, offset

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // srcNchw (3*W*H float, values in [0,1])
        ArrayView1D<int, Stride1D.Dense>,    // dstRgba (packed RGBA int per pixel)
        int, int, int>? _nchwToRgbaKernel;   // w, h, offset

    // WebGPU hard limit: maxComputeWorkgroupsPerDimension = 65535, group size 64.
    private const int MaxDispatchElements = 65535 * 64;

    public event Action? OnStateChanged;
    public string Status { get; private set; } = "";
    public bool IsLoading { get; private set; }
    public bool IsReady => _session != null;
    public int LoadedScale { get; private set; }

    public SuperResolutionService(GpuService gpu) => _gpu = gpu;

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernels
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: packed RGBA int → NCHW float32 [0,1].
    /// No resize (SR models process at full resolution).
    /// Channel order: R=ch0, G=ch1, B=ch2.
    /// </summary>
    private static void RgbaToNchwKernel(
        Index1D idx,
        ArrayView1D<int, Stride1D.Dense> srcRgba,
        ArrayView1D<float, Stride1D.Dense> dstNchw,
        int w, int h, int offset)
    {
        int absIdx = idx + offset;
        int pixCount = w * h;
        int c = absIdx / pixCount;
        int pix = absIdx % pixCount;
        int packed = srcRgba[pix];
        dstNchw[absIdx] = ((packed >> (c * 8)) & 0xFF) / 255f;
    }

    /// <summary>
    /// GPU kernel: NCHW float32 [0,1] → packed RGBA int.
    /// Each invocation writes one pixel (R+G+B channels, A=255).
    /// </summary>
    private static void NchwToRgbaKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> srcNchw,
        ArrayView1D<int, Stride1D.Dense> dstRgba,
        int w, int h, int offset)
    {
        int pixIdx = idx + offset;
        int pixCount = w * h;
        float r = srcNchw[0 * pixCount + pixIdx];
        float g = srcNchw[1 * pixCount + pixIdx];
        float b = srcNchw[2 * pixCount + pixIdx];

        int rv = r <= 0f ? 0 : r >= 1f ? 255 : (int)(r * 255f + 0.5f);
        int gv = g <= 0f ? 0 : g >= 1f ? 255 : (int)(g * 255f + 0.5f);
        int bv = b <= 0f ? 0 : b >= 1f ? 255 : (int)(b * 255f + 0.5f);

        dstRgba[pixIdx] = rv | (gv << 8) | (bv << 16) | (255 << 24);
    }

    // ─────────────────────────────────────────────────────────────
    //  Model Loading
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Load the SR ONNX model for the given scale (2 or 4).
    /// Uses the shared GPUDevice from GpuService (zero-copy buffer exchange with ORT).
    /// </summary>
    public async Task LoadModelAsync(int scale = 2)
    {
        if (_session != null && LoadedScale == scale) return;

        IsLoading = true;
        Status = $"Loading {scale}x SR model…";
        OnStateChanged?.Invoke();

        try
        {
            if (_ort == null)
            {
                Status = "Initializing ONNX Runtime…";
                OnStateChanged?.Invoke();
                await Task.Yield();
                _ort = await OnnxRuntime.Init();
            }

            using var env = _ort.Env;
            env.SetPreferredOutputLocation("gpu-buffer");

            string modelPath = scale == 4 ? Model4xPath : Model2xPath;
            Status = $"Downloading {scale}x SR model…";
            OnStateChanged?.Invoke();
            await Task.Yield();

            _session?.Dispose();
            _session = await _ort.CreateInferenceSessionAsync(modelPath, new SessionCreateOptions
            {
                ExecutionProviders = new[] { "webgpu", "wasm" },
                GraphOptimizationLevel = "all",
                LogSeverityLevel = 3,
            });

            LoadedScale = scale;
            Status = $"✅ SR ×{scale} model loaded — inputs: [{string.Join(", ", _session.InputNames)}]";
            Console.WriteLine($"[SR] {Status}");
        }
        catch (Exception ex)
        {
            Status = $"❌ Failed to load SR model: {ex.Message}";
            Console.WriteLine($"[SR] Error: {ex}");
            _session?.Dispose();
            _session = null;
        }
        finally
        {
            IsLoading = false;
            OnStateChanged?.Invoke();
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Inference
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Upscale an image using the loaded SR model.
    /// GPU pipeline:
    ///   1. RGBA bytes → GPU packed RGBA buffer (one CPU boundary per image)
    ///   2. GPU RGBA → NCHW float [0,1] (kernel, batched for large images)
    ///   3. ORT inference on shared GPUDevice (zero-copy tensor input)
    ///   4. GPU NCHW float [0,1] → packed RGBA (kernel, batched)
    ///   5. CPU readback of RGBA bytes (one CPU boundary per image)
    /// Returns a new ImportedImage at upscaled dimensions, or null on failure.
    /// </summary>
    public async Task<ImportedImage?> UpscaleAsync(ImportedImage image)
    {
        if (_session == null || _ort == null)
        {
            Status = "SR model not loaded. Load a model first.";
            return null;
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;

        _rgbaToNchwKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(RgbaToNchwKernel);

        _nchwToRgbaKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            int, int, int>(NchwToRgbaKernel);

        int origW = image.Width;
        int origH = image.Height;

        Status = $"Upscaling {origW}×{origH} → {origW * LoadedScale}×{origH * LoadedScale}…";
        OnStateChanged?.Invoke();
        await Task.Yield();

        // ── Step 1: Upload RGBA bytes to GPU ──────────────────────────────
        // Reinterpret byte[W*H*4] as int[W*H] — packed RGBA, zero-copy.
        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        // ── Step 2: RGBA → NCHW float [0,1] (batched for large images) ───
        int nchwTotal = 3 * origW * origH;
        using var nchwBuf = accelerator.Allocate1D<float>(nchwTotal);
        for (int off = 0; off < nchwTotal; off += MaxDispatchElements)
        {
            int count = Math.Min(MaxDispatchElements, nchwTotal - off);
            _rgbaToNchwKernel(count, rgbaBuf.View, nchwBuf.View, origW, origH, off);
        }

        await accelerator.SynchronizeAsync();

        var gpuInputBuffer = GetNativeBuffer(nchwBuf);
        if (gpuInputBuffer == null)
        {
            Status = "❌ Could not access GPU buffer for SR input.";
            return null;
        }

        Status = $"Running {LoadedScale}x SR inference…";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            // ── Step 3: Zero-copy ORT SR inference ────────────────────────
            using var inputTensor = _ort.TensorFromGpuBuffer(gpuInputBuffer, new TensorFromGpuBufferOptions
            {
                DataType = "float32",
                Dims = new long[] { 1, 3, origH, origW },
            });

            using var feeds = new OrtFeeds();
            feeds.Set(_session.InputNames[0], inputTensor);

            using var ortResult = await _session.Run(feeds);
            using var outputTensor = ortResult.GetTensor(_session.OutputNames[0]);

            var dims = outputTensor.Dims;
            int outH = (int)dims[dims.Length - 2];
            int outW = (int)dims[dims.Length - 1];
            int outPixCount = outH * outW;

            Console.WriteLine($"[SR] Output dims: [{string.Join(", ", dims)}], location: {outputTensor.Location}");

            // ── Step 4: NCHW float → packed RGBA ──────────────────────────
            using var dstRgbaBuf = accelerator.Allocate1D<int>(outPixCount);

            if (outputTensor.Location == "gpu-buffer")
            {
                var ortGpuBuffer = outputTensor.GPUBuffer;
                long outElements = 3L * outPixCount;
                using var externalBuf = new ExternalWebGPUMemoryBuffer(
                    accelerator, ortGpuBuffer, outElements, sizeof(float));

                var outView = externalBuf.AsArrayView<float>(0, outElements);
                for (int off = 0; off < outPixCount; off += MaxDispatchElements)
                {
                    int count = Math.Min(MaxDispatchElements, outPixCount - off);
                    _nchwToRgbaKernel(count, outView, dstRgbaBuf.View, outW, outH, off);
                }
            }
            else
            {
                // WASM fallback: ORT ran on CPU — upload SR output to GPU once.
                Console.WriteLine("[SR] WASM fallback: uploading ORT output to GPU.");
                using var outputData = outputTensor.GetData<Float32Array>();
                float[] cpuData = outputData.ToArray();
                using var cpuBuf = accelerator.Allocate1D(cpuData);
                for (int off = 0; off < outPixCount; off += MaxDispatchElements)
                {
                    int count = Math.Min(MaxDispatchElements, outPixCount - off);
                    _nchwToRgbaKernel(count, cpuBuf.View, dstRgbaBuf.View, outW, outH, off);
                }
            }

            await accelerator.SynchronizeAsync();

            // ── Step 5: Readback to CPU ────────────────────────────────────
            // One CPU boundary per image — unavoidable for the ImportedImage contract.
            // Future optimization: pass NCHW GPU buffer directly to depth estimator.
            int[] packedResult = await dstRgbaBuf.CopyToHostAsync<int>(0, outPixCount);
            byte[] rgbaBytes = MemoryMarshal.Cast<int, byte>(packedResult.AsSpan()).ToArray();

            Status = $"✅ SR ×{LoadedScale}: {origW}×{origH} → {outW}×{outH}";
            OnStateChanged?.Invoke();

            return new ImportedImage
            {
                FileName = $"{image.FileName} ×{LoadedScale}SR",
                Width = outW,
                Height = outH,
                RgbaPixels = rgbaBytes,
            };
        }
        catch (Exception ex)
        {
            Status = $"❌ SR inference failed: {ex.Message}";
            Console.WriteLine($"[SR] Error: {ex}");
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────

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
