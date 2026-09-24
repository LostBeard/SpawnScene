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
    /// Input width and height bound at load, both multiples of <see cref="PatchSize"/>. Always a
    /// square (<see cref="SetSquareInput"/>).
    /// </summary>
    public static (int Width, int Height) InputShape { get; private set; } = (48 * PatchSize, 48 * PatchSize);

    /// <summary>
    /// How a picture becomes the model's input tensor. <see cref="DepthResizeMode.Letterbox"/> pads it into the
    /// bound square; <see cref="DepthResizeMode.NativeAspect"/> is Depth Anything 3's own preprocessing (long side
    /// = the bound square's side, aspect kept, no pad). Applied on every call, so an autotest can A/B it
    /// (<c>&amp;resize=native|letterbox</c>) without reloading the model.
    /// </summary>
    public static DepthResizeMode ResizeMode { get; set; } = DepthResizeMode.Letterbox;

    /// <summary>
    /// Patch grid that joint multi-view inference has been shown to survive.
    ///
    /// Measured on the 5K living room, single image, same code, only this changed:
    ///   37x37 (518px) - the recessed room is one smooth blob. No pendant lamps, no stool,
    ///                   armchairs and sofa merged. This is what "DAv3 looks worse than DAv2"
    ///                   was: 518 simply is not enough for a room.
    ///   64x64 (896px) - three pendant lamps resolved as separate spheres, armchairs and sofa
    ///                   separated, the stool by the doorway appears. Detail comparable to the
    ///                   DAv2 reference and better in places.
    ///
    /// 64 is NOT safe for the 6-view joint path: it loses the WebGPU device outright
    /// ("A valid external Instance reference no longer exists") at 896x896 x 6. 48 runs on both
    /// paths, so it is the default. Raise it for single-image work.
    ///
    /// TempleRing shows no reconstruction change between 37 and 48 (21.11 vs 21.00 dB held
    /// out), which is expected - at 480x640 it only upsamples 1.2x, so it was never
    /// resolution-limited. Absence of a gain there is not evidence against; it is the wrong
    /// test for this.
    /// </summary>
    public const int SafeMultiViewPatches = 48;

    /// <summary>
    /// Bind a SQUARE input of <paramref name="patchesPerSide"/> x <paramref name="patchesPerSide"/>
    /// ViT patches. 37 is the familiar 518.
    ///
    /// This is the axis to push for detail. It is also the long side NativeAspect resizes to (see
    /// <see cref="ResizeMode"/>) - an aspect-preserving tensor comes from that mode, not from a
    /// non-square binding, which the pipeline refuses.
    /// Cost grows with the square of this, and attention with its fourth power.
    /// </summary>
    public static void SetSquareInput(int patchesPerSide)
    {
        int n = Math.Max(4, patchesPerSide);
        InputShape = (n * PatchSize, n * PatchSize);
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
            // 518x518 is not a model limit, it is a shape WE bind at load; any multiple of 14 is
            // valid. The binding is always SQUARE (the pipeline refuses anything else): Letterbox
            // pads into it, NativeAspect resizes to its side on the long axis and feeds the model a
            // non-square tensor that follows the picture (see ResizeMode).
            var (inW, inH) = InputShape;
            Console.WriteLine($"[Depth] binding pixel_values to [1,1,3,{inH},{inW}] " +
                $"({inW / PatchSize}x{inH / PatchSize} patches)");
            _pipe = await DepthEstimationPipeline.CreateFromHubAsync(
                accelerator, _modelSource, RepoId,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, inH, inW } });
            // One-shot photo path: capture/replay warmup is for video.
            _pipe.EnableGraphCapture = false;
            _pipe.ResizeMode = ResizeMode;

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
    private void ApplyResizeMode()
    {
        if (_pipe != null && _pipe.ResizeMode != ResizeMode)
        {
            Console.WriteLine($"[Depth] resize mode {_pipe.ResizeMode} -> {ResizeMode}");
            _pipe.ResizeMode = ResizeMode;
        }
    }

    private async Task<DepthResult?> RunPipelineAsync(
        Func<Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>> run)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded. Load a model first.";
            return null;
        }
        ApplyResizeMode();

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
            // Null-tolerant: a caller may take ownership of individual views (the chunked pose
            // pass keeps the ones it posed and lets the rest go) by nulling the slot.
            foreach (var d in DepthResults) d?.Dispose();
            DepthResults.Clear();
        }
    }

    /// <summary>
    /// GPU-resident RGBA uploads, reused across joint passes.
    ///
    /// The chunked pose pass runs the same ANCHOR images in every chunk, so without this each of
    /// them is re-copied and re-uploaded once per chunk. Owned and disposed by the caller, since
    /// only the caller knows when the run is over.
    /// </summary>
    public sealed class MultiViewUploadCache : IDisposable
    {
        private readonly GpuService _gpu;
        private readonly Dictionary<ImportedImage, MemoryBuffer1D<int, Stride1D.Dense>> _cache = new();

        public MultiViewUploadCache(GpuService gpu) => _gpu = gpu;

        public int Uploads { get; private set; }
        public int Reuses { get; private set; }

        public async Task<IReadOnlyList<ArrayView1D<int, Stride1D.Dense>>> ViewsForAsync(
            IReadOnlyList<ImportedImage> images, int count)
        {
            if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
            var accelerator = _gpu.WebGPUAccelerator;

            var views = new ArrayView1D<int, Stride1D.Dense>[count];
            for (int i = 0; i < count; i++)
            {
                var image = images[i];
                if (_cache.TryGetValue(image, out var existing))
                {
                    Reuses++;
                    views[i] = existing.View;
                    continue;
                }

                // One managed span -> one GPU buffer. The span is a VIEW of the existing pixels,
                // so nothing is copied on the .NET heap on the way.
                var pixels = System.Runtime.InteropServices.MemoryMarshal
                    .Cast<byte, int>(image.RgbaPixels.AsSpan());
                var buffer = accelerator.Allocate1D<int>(pixels.Length);
                buffer.View.BaseView.CopyFromCPU(pixels);
                _cache[image] = buffer;
                Uploads++;
                views[i] = buffer.View;
            }
            await accelerator.SynchronizeAsync();
            return views;
        }

        public void Dispose()
        {
            foreach (var b in _cache.Values) b.Dispose();
            _cache.Clear();
        }
    }

    private static int[][] BuildManagedFrames(IReadOnlyList<ImportedImage> images, int n)
    {
        var frames = new int[n][];
        for (int i = 0; i < n; i++)
            frames[i] = System.Runtime.InteropServices.MemoryMarshal
                .Cast<byte, int>(images[i].RgbaPixels.AsSpan()).ToArray();
        return frames;
    }

    /// <summary>
    /// Views per joint DAv3 forward. A STARTING POINT, not a measured limit.
    ///
    /// This was a <c>const 6</c> introduced with the comment "(WebGPU memory / compile cost)" and
    /// no measurement anywhere behind it - the ML library's own DAv3 multi-view test runs N=2,
    /// and nothing in either repo queries a device limit on this path. A compile-time constant
    /// here decides how much of a stranger's capture gets posed, on hardware we have never seen:
    /// too high and the forward fails outright rather than degrading, too low and views are
    /// discarded for nothing.
    ///
    /// So it is a knob, and callers are expected to back off rather than trust it - see
    /// <c>MultiViewGenerationService.PoseAllViewsChunkedAsync</c>, which retries at a smaller N
    /// and still poses every view, just in more chunks. The number that matters for coverage is
    /// no longer this one.
    ///
    /// Note the model recompiles per distinct N ("Session shape-recompile handles N != the
    /// compile-time num_images"), so a caller should keep N the same across a run.
    /// </summary>
    public static int MaxMultiViewImages { get; set; } = 6;

    /// <summary>
    /// Joint multi-view depth via DAv3 <c>[1,N,3,H,W]</c>. Fills <see cref="MultiViewDepthResult.Extrinsics"/>
    /// when the model emits usable poses.
    ///
    /// <paramref name="maxViews"/> overrides <see cref="MaxMultiViewImages"/> for this call, so a
    /// caller that has just watched a forward fail can retry smaller instead of giving up. Zero
    /// or less means use the default.
    /// </summary>
    public async Task<MultiViewDepthResult?> EstimateDepthMultiViewAsync(
        IReadOnlyList<ImportedImage> images, int maxViews = 0, MultiViewUploadCache? uploads = null)
    {
        if (_pipe == null)
        {
            Status = "Model not loaded.";
            return null;
        }
        if (images.Count == 0) return null;
        ApplyResizeMode();

        int cap = maxViews > 0 ? maxViews : MaxMultiViewImages;
        int n = Math.Min(images.Count, cap);
        if (images.Count > cap)
            Console.WriteLine($"[Depth] Cap multi-view at {cap} (got {images.Count})");

        Status = $"Running joint DAv3 multi-view ({n} images)...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        try
        {
            var widths = new int[n];
            var heights = new int[n];
            for (int i = 0; i < n; i++)
            {
                widths[i] = images[i].Width;
                heights[i] = images[i].Height;
            }

            // Output at first image resolution (splat unproject expects matching WxH).
            int outW = widths[0], outH = heights[0];

            // Upload once, through a cache the caller owns.
            //
            // This used to be MemoryMarshal.Cast(...).ToArray() per view per call - a full RGBA
            // copy on the managed heap every time, and the chunked pose pass multiplied that by
            // the number of chunks because the ANCHOR views are in every single one. 35 frames at
            // 768x1024 is 105 MB resident before any copy, in a 2 GB WASM heap that has already
            // thrown an OutOfMemoryException on this exact class of thing. Uploading each frame
            // once and reusing the GPU buffer removes both the copies and the repeat uploads.
            using var mv = uploads != null
                ? await _pipe.EstimateMultiViewGpuAsync(
                    await uploads.ViewsForAsync(images, n), widths, heights,
                    outputWidth: outW, outputHeight: outH).ConfigureAwait(false)
                : await _pipe.EstimateMultiViewGpuAsync(
                    BuildManagedFrames(images, n), widths, heights,
                    outputWidth: outW, outputHeight: outH).ConfigureAwait(false);

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

    /// <summary>
    /// Give back the GPU memory the loaded model keeps parked between runs (its activation arena and
    /// any capture plan), keeping the weights loaded. Returns the bytes freed, 0 if no model is loaded.
    ///
    /// Call this when depth work is finished and the GPU is about to be used for something else.
    /// MEASURED 2026-09-23, DrJohnson 2000: after the 14-pass DAv3 cascade the GPU process held
    /// 6.1 GB dedicated VRAM, 3.9 GB of it this arena; the trainer's own 1.2 GB on top took Chrome
    /// to 7.4 GB and it dropped the device on the trainer's second resize.
    /// </summary>
    public long ReleaseWorkingMemory()
    {
        if (_pipe == null) return 0;
        long freed = _pipe.ReleaseWorkingMemory();
        Console.WriteLine($"[Depth] released {freed / 1048576.0:F0} MB of depth working memory (weights kept)");
        return freed;
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
