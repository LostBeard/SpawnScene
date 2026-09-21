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
    // DAv3 Small: native 5-D [batch, num_images, 3, H, W] + external model.onnx_data.
    // (DAv2 was a temporary SpawnScene workaround while ML dropped the pos-embed weight on the hub/stream path.)
    private const string RepoId = "onnx-community/depth-anything-v3-small";

    public static readonly DepthModelInfo[] AvailableModels = new[]
    {
        new DepthModelInfo("depth-anything-v3-small", "Depth Anything V3 Small", RepoId, "~100 MB", IsDirectDepth: true),
    };

    public static readonly string DefaultModelId = "depth-anything-v3-small";

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
    /// <summary>Vision-transformer patch size. Every input dimension must be a multiple.</summary>
    public const int PatchSize = 14;

    /// <summary>
    /// Input width and height bound at load, both multiples of <see cref="PatchSize"/>.
    /// Defaults to the familiar 518x518. Set <see cref="MatchAspect"/> before loading to have
    /// this follow the source images instead.
    /// </summary>
    public static (int Width, int Height) InputShape { get; private set; } = (518, 518);

    /// <summary>
    /// Choose an input shape with the same aspect as <paramref name="srcW"/> x
    /// <paramref name="srcH"/> and about the same number of patches as the square default, so
    /// the cost is unchanged and no budget is spent on padding. Rounds to
    /// <see cref="PatchSize"/> and clamps to at least 4 patches a side.
    ///
    /// Has no effect on an already-loaded model - the shape is bound at session creation.
    /// </summary>
    public static void MatchAspect(int srcW, int srcH, int patchBudget = 37 * 37)
    {
        if (srcW <= 0 || srcH <= 0) { InputShape = (518, 518); return; }

        double aspect = (double)srcW / srcH;
        // pw * ph ~= budget with pw/ph == aspect
        int ph = Math.Max(4, (int)Math.Round(Math.Sqrt(patchBudget / aspect)));
        int pw = Math.Max(4, (int)Math.Round(patchBudget / (double)ph));
        InputShape = (pw * PatchSize, ph * PatchSize);
    }

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

            // Native DAv3 shape: 5-D [batch, num_images, 3, H, W]. External weights via default
            // onnx/model.onnx_data (do NOT pass externalDataFile: "" — that is the DAv2 single-file path).
            //
            // 518x518 is not a model limit, it is a shape WE bind at load. DA3 pads to the ViT
            // patch size and crops back, so any multiple of 14 is valid - and a SQUARE input
            // spends its token budget on letterbox padding. An aspect-matched shape costs the
            // same and carries more picture: 448x602 is 1,376 patches against 518x518's 1,369.
            var (inW, inH) = InputShape;
            Console.WriteLine($"[Depth] binding pixel_values to [1,1,3,{inH},{inW}] " +
                $"({inW / PatchSize}x{inH / PatchSize} patches)");
            _pipe = await DepthEstimationPipeline.CreateFromHubAsync(
                accelerator, _modelSource, RepoId,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, inH, inW } });
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
    /// Prefer <see cref="EstimateDepthFromJsRgbaAsync"/> when pixels are still JS-side.
    /// </summary>
    public Task<DepthResult?> EstimateDepthFromPackedRgbaAsync(int[] packedRgba, int width, int height)
        => RunPipelineAsync(() => _pipe!.EstimateGpuRawAsync(packedRgba, width, height, width, height));

    /// <summary>
    /// Run depth from a JS TypedArray (e.g. ImageData.Data) — JS → GPU via CopyFromJS,
    /// no managed <c>int[]</c> crossing.
    /// </summary>
    public Task<DepthResult?> EstimateDepthFromJsRgbaAsync(TypedArray rgbaBytes, int width, int height)
        => RunPipelineAsync(() => _pipe!.EstimateGpuRawAsync(rgbaBytes, width, height, width, height));

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

        // ImportedImage already holds managed bytes (feature/SfM path).
        var packedRgba = System.Runtime.InteropServices.MemoryMarshal
            .Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();

        return await EstimateDepthFromPackedRgbaAsync(packedRgba, image.Width, image.Height);
    }

    /// <summary>
    /// Run depth estimation on a GPU-resident image — no download/readback; preprocess reads
    /// <see cref="GpuImage.PackedRgba"/> in place.
    /// </summary>
    public Task<DepthResult?> EstimateDepthAsync(GpuImage gpuImage)
        => RunPipelineAsync(() => _pipe!.EstimateGpuRawAsync(
            gpuImage.PackedRgba.View, gpuImage.Width, gpuImage.Height, gpuImage.Width, gpuImage.Height));

    /// <summary>
    /// Shared pipeline call: RGBA → GPU-resident depth + min/max.
    /// DAv3 <c>predicted_depth</c> is relative/direct depth (high = far), not DAv2 disparity.
    /// Unprojection kernels consume it as direct depth (no invert / no FlipDepthKernel).
    /// </summary>
    private async Task<DepthResult?> RunPipelineAsync(
        Func<Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>> run)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }

        Status = "Running depth inference...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            var (rawDepth, minD, maxD, outW, outH) = await run().ConfigureAwait(false);

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
        /// <summary>Per-view 3×4 [R|t] as length-12 row-major arrays. Null when unavailable.</summary>
        public float[][]? Extrinsics { get; set; }
        /// <summary>Per-view 3×3 intrinsics as length-9 row-major arrays. Null when unavailable.</summary>
        public float[][]? Intrinsics { get; set; }
        /// <summary>True when every view has a GPU confidence map (same W×H as depth).</summary>
        public bool HasConfidence =>
            DepthResults.Count > 0 && DepthResults.All(d => d.ConfidenceGpu != null);
        public void Dispose()
        {
            foreach (var d in DepthResults) d.Dispose();
            DepthResults.Clear();
        }
    }

    /// <summary>Max views for a single joint DAv3 forward (WebGPU memory / compile cost).</summary>
    public const int MaxMultiViewImages = 6;

    /// <summary>
    /// Joint multi-view depth via DAv3 <c>[1,N,3,H,W]</c>. Fills <see cref="MultiViewDepthResult.Extrinsics"/>
    /// when the model emits usable poses. Caps at <see cref="MaxMultiViewImages"/>.
    /// </summary>
    public async Task<MultiViewDepthResult?> EstimateDepthMultiViewAsync(IReadOnlyList<ImportedImage> images)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded.";
            return null;
        }
        if (images.Count == 0) return null;

        int n = Math.Min(images.Count, MaxMultiViewImages);
        if (images.Count > MaxMultiViewImages)
            Console.WriteLine($"[Depth] Cap multi-view at {MaxMultiViewImages} (got {images.Count})");

        Status = $"Running joint DAv3 multi-view ({n} images)...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            var rgbaFrames = new int[n][];
            var widths = new int[n];
            var heights = new int[n];
            for (int i = 0; i < n; i++)
            {
                rgbaFrames[i] = System.Runtime.InteropServices.MemoryMarshal
                    .Cast<byte, int>(images[i].RgbaPixels.AsSpan()).ToArray();
                widths[i] = images[i].Width;
                heights[i] = images[i].Height;
            }

            // Output at first image resolution (splat unproject expects matching WxH).
            int outW = widths[0], outH = heights[0];
            using var mv = await _pipe.EstimateMultiViewGpuAsync(
                rgbaFrames, widths, heights, outputWidth: outW, outputHeight: outH).ConfigureAwait(false);

            var result = new MultiViewDepthResult
            {
                Extrinsics = mv.Extrinsics,
                Intrinsics = mv.Intrinsics,
            };

            for (int i = 0; i < mv.ViewCount; i++)
            {
                var (raw, minD, maxD, w, h) = mv.Views[i];
                result.DepthResults.Add(new DepthResult
                {
                    RawDepthGpu = raw,
                    ConfidenceGpu = mv.ConfidenceMaps != null && i < mv.ConfidenceMaps.Count
                        ? mv.ConfidenceMaps[i]
                        : null,
                    Width = w,
                    Height = h,
                    MinDepth = minD,
                    MaxDepth = maxD,
                });
            }
            // Caller owns depth (+ confidence) buffers now.
            mv.DetachDepthViews();
            mv.DetachConfidenceMaps();

            Status = $"✅ Multi-view depth: {result.DepthResults.Count} views" +
                     (result.HasConfidence ? " + confidence" : "") +
                     (result.Extrinsics != null ? " + extrinsics" : "");
            Console.WriteLine($"[Depth] {Status}");
            OnStateChanged?.Invoke();
            return result;
        }
        catch (Exception ex)
        {
            Status = $"❌ Multi-view inference failed: {ex.Message}";
            Console.WriteLine($"[Depth] Multi-view error: {ex}");
            OnStateChanged?.Invoke();
            return null;
        }
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
    /// Relative/direct depth from DAv3 (high = far). Use MinDepth/MaxDepth for range; do not invert.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? RawDepthGpu { get; set; }

    /// <summary>
    /// Optional per-pixel confidence (same W×H as depth). Owned by this instance when set.
    /// Joint DAv3 multi-view fills this; monocular EstimateDepthAsync leaves it null.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense>? ConfidenceGpu { get; set; }

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
        ConfidenceGpu?.Dispose();
        ConfidenceGpu = null;
        GC.SuppressFinalize(this);
    }
}
