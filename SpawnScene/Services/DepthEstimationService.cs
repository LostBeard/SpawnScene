using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Available depth estimation model definition.
/// </summary>
/// <param name="IsDirectDepth">True if model outputs depth (high=far), false if disparity (high=close).</param>
/// <param name="IsNative">Unused (legacy) — kept for record shape compatibility.</param>
public record DepthModelInfo(string Id, string Name, string Path, string SizeLabel, bool IsDirectDepth = false, bool IsNative = false);

/// <summary>
/// Monocular depth estimation via the SpawnDev.ILGPU.ML <see cref="DepthEstimationPipeline"/>.
/// The pipeline owns everything: model download (hub HTTP + OPFS cache via <see cref="IModelSource"/>),
/// zero-copy stream-to-GPU load, ImageNet preprocessing, DAv3 inference, and a GPU-resident depth
/// result. SpawnScene wires two calls — no ORT, no hand-built backbone, no model bytes in the
/// .NET/WASM managed heap.
/// </summary>
public class DepthEstimationService : IAsyncDisposable
{
    /// <summary>HuggingFace repo id for the depth model the pipeline downloads + caches.</summary>
    // DAv3 (onnx-community/depth-anything-v3-small, 5-D) currently throws in GraphExecutor:
    //   Tensor '/backbone/Transpose_output_0' not found (needed by Resize)
    // with producerOp=NONE elideBlocked=True — a SpawnDev.ILGPU.ML graph issue, not SpawnScene.
    // DAv2 Small is the Depth demo's working path (4-D [1,3,518,518]) until that is fixed.
    private const string RepoId = "onnx-community/depth-anything-v2-small";

    public static readonly DepthModelInfo[] AvailableModels = new[]
    {
        new DepthModelInfo("depth-anything-v2-small", "Depth Anything V2 Small", RepoId, "~100 MB"),
    };

    public static readonly string DefaultModelId = "depth-anything-v2-small";

    private readonly GpuService _gpu;
    private readonly SpawnDev.ILGPU.ML.Hub.IModelSource _modelSource;
    private DepthEstimationPipeline? _pipe;

    public string? LoadedModelId { get; private set; }
    public string? LoadedModelName { get; private set; }

    public event Action? OnStateChanged;
    public string Status { get; private set; } = "";
    public bool IsLoading { get; private set; }
    public bool IsReady => _pipe != null;

    public DepthEstimationService(GpuService gpu, SpawnDev.ILGPU.ML.Hub.IModelSource modelSource)
    {
        _gpu = gpu;
        _modelSource = modelSource;
    }

    /// <summary>
    /// Build the depth pipeline for the given model. The pipeline downloads + OPFS-caches the model
    /// via <see cref="IModelSource"/> and streams its weights straight to the GPU — the weights
    /// never enter .NET.
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

        // Already loaded this exact model.
        if (_pipe != null && LoadedModelId == modelId) return;

        // Switching models: dispose the old pipeline.
        _pipe?.Dispose();
        _pipe = null;
        LoadedModelId = null;
        LoadedModelName = null;

        IsLoading = true;
        Status = $"Loading {model.Name} (download + cache)...";
        OnStateChanged?.Invoke();

        try
        {
            if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
            var accelerator = _gpu.WebGPUAccelerator;

            await Task.Yield();

            // 4-D input [batch, 3, H, W] for DAv2. (DAv3 is 5-D [batch, num_images, 3, H, W].)
            _pipe = await DepthEstimationPipeline.CreateFromHubAsync(
                accelerator, _modelSource, RepoId,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 518, 518 } },
                externalDataFile: ""); // DAv2 is a single-file ONNX (no model.onnx_data)
            // One-shot photo path: capture/replay warmup is for video.
            _pipe.EnableGraphCapture = false;

            LoadedModelId = modelId;
            LoadedModelName = model.Name;
            Status = $"✅ {model.Name} ready";
            Console.WriteLine($"[Depth] {Status}");
        }
        catch (Exception ex)
        {
            Status = $"❌ Failed to load model: {ex.Message}";
            Console.WriteLine($"[Depth] Load error: {ex}");
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
    /// Run depth estimation from packed RGBA ints already on the .NET heap.
    /// Prefer feeding a JS TypedArray via <see cref="EstimateDepthFromJsRgbaAsync"/> when the
    /// pixels are still JS-side — this entry exists because <c>EstimateGpuRawAsync</c> takes <c>int[]</c>.
    /// </summary>
    public Task<DepthResult?> EstimateDepthFromPackedRgbaAsync(int[] packedRgba, int width, int height)
        => RunPipelineAsync(packedRgba, width, height);

    /// <summary>
    /// Run depth from a JS TypedArray (e.g. ImageData.Data). One JS→.NET <c>Read&lt;int&gt;</c> for the
    /// pipeline's <c>int[]</c> API — does not first upload to GPU and read back.
    /// </summary>
    public Task<DepthResult?> EstimateDepthFromJsRgbaAsync(TypedArray rgbaBytes, int width, int height)
    {
        // CPU transfer: DepthEstimationPipeline.EstimateGpuRawAsync currently requires int[].
        int[] packedRgba = rgbaBytes.Read<int>();
        return RunPipelineAsync(packedRgba, width, height);
    }

    /// <summary>
    /// Run depth estimation on a CPU-resident image. Returns a GPU-resident <see cref="DepthResult"/>.
    /// </summary>
    public async Task<DepthResult?> EstimateDepthAsync(ImportedImage image)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }

        // CPU transfer: ImportedImage already holds managed bytes (feature/SfM path).
        var packedRgba = System.Runtime.InteropServices.MemoryMarshal
            .Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();

        return await RunPipelineAsync(packedRgba, image.Width, image.Height);
    }

    /// <summary>
    /// Run depth estimation on a GPU-resident image.
    /// ⚠️ Round-trips GPU→JS→.NET because the pipeline only accepts <c>int[]</c>. Prefer
    /// <see cref="EstimateDepthFromJsRgbaAsync"/> / <see cref="EstimateDepthFromPackedRgbaAsync"/>
    /// when the source TypedArray is still available.
    /// </summary>
    public async Task<DepthResult?> EstimateDepthAsync(GpuImage gpuImage)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }

        int pixelCount = gpuImage.Width * gpuImage.Height;
        // GPU → JS TypedArray (not .NET), then one Read<int> for the pipeline API.
        // Call on the typed MemoryBuffer1D — .Buffer is the raw backend buffer and is not IArrayView.
        using var u8 = await gpuImage.PackedRgba.CopyToHostUint8ArrayAsync(0, (long)pixelCount * 4);
        int[] packedRgba = u8.Read<int>();

        return await RunPipelineAsync(packedRgba, gpuImage.Width, gpuImage.Height);
    }

    /// <summary>
    /// Shared pipeline call: RGBA int[] → GPU-resident depth (bit-exact DAv3) + min/max.
    /// DAv3 <c>predicted_depth</c> is inverse/relative depth (high = close) = disparity, which is
    /// exactly what the unprojection kernel expects — so no flip is applied.
    /// </summary>
    private async Task<DepthResult?> RunPipelineAsync(int[] packedRgba, int width, int height)
    {
        Status = "Running depth inference...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            var (rawDepth, minD, maxD, outW, outH) = await _pipe!.EstimateGpuRawAsync(
                packedRgba, width, height, width, height);

            Console.WriteLine($"[Depth] {outW}x{outH} min/max: [{minD:F6}, {maxD:F6}], range={maxD - minD:F6}");

            Status = $"✅ Depth estimated — range: [{minD:F3}, {maxD:F3}]";
            OnStateChanged?.Invoke();

            return new DepthResult
            {
                RawDepthGpu = rawDepth,
                Width = outW,
                Height = outH,
                MinDepth = minD,
                MaxDepth = maxD,
            };
        }
        catch (Exception ex)
        {
            Status = $"❌ Inference failed: {ex.Message}";
            Console.WriteLine($"[Depth] Error: {ex}");
            return null;
        }
    }

    /// <summary>
    /// Multi-view depth estimation result: per-view depth maps + (optional) camera poses.
    /// </summary>
    public class MultiViewDepthResult : IDisposable
    {
        public List<DepthResult> DepthResults { get; set; } = new();
        /// <summary>DAv3 extrinsics: [R|t] 3×4 matrix per view. null when unavailable.</summary>
        public float[,][]? Extrinsics { get; set; }
        public void Dispose()
        {
            foreach (var d in DepthResults) d.Dispose();
            DepthResults.Clear();
        }
    }

    /// <summary>
    /// Multi-view depth: per-image fallback. Each image is estimated independently.
    /// NOTE: joint depth + camera extrinsics (the old ORT 5-D multi-view batch) are intentionally
    /// dropped in the pipeline migration; <see cref="MultiViewDepthResult.Extrinsics"/> is null until
    /// the pipeline exposes a multi-view entry point.
    /// </summary>
    public async Task<MultiViewDepthResult?> EstimateDepthMultiViewAsync(IReadOnlyList<ImportedImage> images)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded.";
            return null;
        }

        var result = new MultiViewDepthResult();
        foreach (var img in images)
        {
            var d = await EstimateDepthAsync(img);
            if (d != null) result.DepthResults.Add(d);
        }
        result.Extrinsics = null;
        return result;
    }

    public ValueTask DisposeAsync()
    {
        _pipe?.Dispose();
        _pipe = null;
        GC.SuppressFinalize(this);
        return ValueTask.CompletedTask;
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
    /// Relative/inverse depth (disparity-like, high = close); use MinDepth/MaxDepth to normalize on GPU.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? RawDepthGpu { get; set; }

    /// <summary>Width of the depth map (matches source image).</summary>
    public int Width { get; set; }

    /// <summary>Height of the depth map (matches source image).</summary>
    public int Height { get; set; }

    /// <summary>Minimum raw depth value (GPU-computed). Used by GPU kernels for on-GPU normalization.</summary>
    public float MinDepth { get; set; }

    /// <summary>Maximum raw depth value (GPU-computed). Used by GPU kernels for on-GPU normalization.</summary>
    public float MaxDepth { get; set; }

    public void Dispose()
    {
        RawDepthGpu?.Dispose();
        RawDepthGpu = null;
        GC.SuppressFinalize(this);
    }
}
