using ILGPU.Runtime;
using ILGPU;
using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Manages image import, pixel data extraction, feature detection, and matching.
/// Coordinates the pipeline: Load → Grayscale → Detect → Match.
/// All heavy operations yield to the UI thread to prevent freezing.
/// </summary>
public class ImageImportService : IDisposable
{
    private readonly FeatureDetector _detector = new();
    private readonly GpuFeatureDetector _gpuDetector = new();
    private readonly GpuFeatureMatcher _gpuMatcher;
    private readonly GpuService _gpu;
    private readonly HttpClient _http;
    private readonly List<ImportedImage> _images = [];
    private readonly List<ImagePair> _pairs = [];

    private readonly VideoFrameExtractor _video;
    private readonly LearnedFeatureMatcher _learned;

    public ImageImportService(GpuFeatureMatcher gpuMatcher, GpuService gpu, HttpClient http, VideoFrameExtractor video,
        LearnedFeatureMatcher learned)
    {
        _gpuMatcher = gpuMatcher;
        _gpu = gpu;
        _video = video;
        _http = http;
        _learned = learned;
        _learned.OnStatus = line => { Status = line; Console.WriteLine($"[Import] {line}"); NotifyChanged(); };
    }

    /// <summary>
    /// Features and pair matches from the learned front end (<see cref="LearnedFeatureMatcher"/>: RaCo-ALIKED +
    /// LightGlue+) instead of FAST/BRIEF + Hamming matching. &amp;features=learned. Pair verification and everything after
    /// it are the same for both.
    /// </summary>
    public bool UseLearnedFeatures { get; set; }

    /// <summary>All imported images.</summary>
    public IReadOnlyList<ImportedImage> Images => _images;

    /// <summary>All matched image pairs.</summary>
    public IReadOnlyList<ImagePair> MatchedPairs => _pairs;

    /// <summary>Fired when an image is added or processing state changes.</summary>
    public event Action? OnStateChanged;

    /// <summary>Current processing status message.</summary>
    public string Status { get; private set; } = "";

    /// <summary>Whether processing is currently running.</summary>
    public bool IsProcessing { get; private set; }

    /// <summary>Progress 0.0 – 1.0</summary>
    public float Progress { get; private set; }

    /// <summary>Current image being processed (1-based).</summary>
    public int CurrentImageIndex { get; private set; }

    /// <summary>Total images in current batch.</summary>
    public int TotalImages { get; private set; }

    /// <summary>
    /// Import images from browser file input.
    /// Reads pixel data, converts to grayscale, detects features.
    /// Yields control to UI between each image to prevent freezing.
    /// </summary>
    public async Task ImportImagesAsync(IReadOnlyList<IBrowserFile> files)
    {
        IsProcessing = true;
        Progress = 0;
        TotalImages = files.Count;
        CurrentImageIndex = 0;
        NotifyChanged();

        // Ensure GPU is initialized before matching
        if (!_gpu.IsInitialized)
        {
            Status = "Initializing GPU...";
            NotifyChanged();
            await Task.Yield();
            try
            {
                await _gpu.InitializeAsync();
                Console.WriteLine($"[Import] GPU initialized: {_gpu.DeviceName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Import] GPU init failed, will use CPU: {ex.Message}");
            }
        }

        try
        {
            int imageCount = files.Count;

            for (int fi = 0; fi < imageCount; fi++)
            {
                var file = files[fi];
                if (!file.ContentType.StartsWith("image/")) continue;

                CurrentImageIndex = fi + 1;
                Progress = (float)fi / imageCount;

                // --- Step 1: Read file bytes ---
                Status = $"Reading {file.Name} ({fi + 1}/{imageCount})...";
                NotifyChanged();
                await Task.Yield(); // Let UI update

                const long maxSize = 50 * 1024 * 1024;
                byte[] bytes;
                try
                {
                    using var stream = file.OpenReadStream(maxAllowedSize: maxSize);
                    using var ms = new MemoryStream();
                    await stream.CopyToAsync(ms);
                    bytes = ms.ToArray();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Import] Failed to read {file.Name}: {ex.Message}");
                    continue;
                }

                // --- Step 2: Decode image ---
                Status = $"Decoding {file.Name} ({fi + 1}/{imageCount})...";
                NotifyChanged();

                var result = await DecodeImageAsync(bytes, file.ContentType);
                if (result == null) continue;

                var (rgba, imgWidth, imgHeight) = result.Value;

                // For large images, downsample for feature detection (performance)
                byte[] grayPixels;
                int featureWidth = imgWidth, featureHeight = imgHeight;

                if (imgWidth > 1024 || imgHeight > 1024)
                {
                    // Downsample for feature detection only (keep full RGBA for viewing)
                    float ds = 1024f / Math.Max(imgWidth, imgHeight);
                    featureWidth = (int)(imgWidth * ds);
                    featureHeight = (int)(imgHeight * ds);
                    grayPixels = DownsampleGrayscale(rgba, imgWidth, imgHeight, featureWidth, featureHeight);
                }
                else
                {
                    grayPixels = RgbaToGrayscale(rgba, imgWidth, imgHeight);
                }

                var imported = new ImportedImage
                {
                    FileName = file.Name,
                    Width = imgWidth,
                    Height = imgHeight,
                    RgbaPixels = rgba,
                    GrayPixels = grayPixels,
                    FeatureWidth = featureWidth,
                    FeatureHeight = featureHeight,
                };

                // --- Step 3: Feature detection (CPU-heavy, yield before and after) ---
                Status = $"Detecting features in {file.Name} ({fi + 1}/{imageCount})...";
                NotifyChanged();
                await Task.Yield(); // Critical: let UI render before CPU work

                imported.Features = UseLearnedFeatures
                    ? await DetectLearnedFromManagedAsync(imported)
                    : _detector.Detect(imported.GrayPixels, featureWidth, featureHeight);

                // Scale feature coordinates back to full image resolution
                if (featureWidth != imgWidth)
                {
                    float scaleBackX = (float)imgWidth / featureWidth;
                    float scaleBackY = (float)imgHeight / featureHeight;
                    foreach (var feat in imported.Features)
                    {
                        feat.X *= scaleBackX;
                        feat.Y *= scaleBackY;
                    }
                }

                Console.WriteLine($"[Import] {file.Name}: {imported.Features.Count} features ({featureWidth}×{featureHeight})");
                _images.Add(imported);

                Progress = (float)(fi + 1) / imageCount;
                NotifyChanged();
                await Task.Yield(); // Let UI show the new image
            }

            // --- Step 4: Match features across pairs (yield between pairs) ---
            if (_images.Count >= 2)
            {
                if (!SkipPairMatching) await MatchAllPairsAsync();
            }

            Progress = 1.0f;
        }
        catch (Exception ex)
        {
            Status = $"Import error: {ex.Message}";
            Console.WriteLine($"[Import] Error: {ex}");
        }
        finally
        {
            IsProcessing = false;
            Status = $"{_images.Count} images imported, {_pairs.Count} pairs matched";
            NotifyChanged();
        }
    }

    /// <summary>
    /// Features for a GPU-resident / Source-backed photo, entirely on the device: decode if needed, grayscale, FAST,
    /// NMS, smoothing, BRIEF (GpuFeatureDetector) and one colour per feature. Only the per-cell winners, the descriptors
    /// and the colours come back. A Source-backed device copy is released afterwards (it can be re-decoded).
    /// Features are at feature resolution (the caller scales them to the image, as for the managed path).
    /// </summary>
    private async Task DetectOnDeviceAsync(ImportedImage img)
    {
        var accel = _gpu.WebGPUAccelerator;
        await GpuImageOps.EnsureOnDeviceAsync(accel, img);
        try
        {
            if (UseLearnedFeatures)
                img.Features = await _learned.ExtractAsync(img, img.GpuRgba!, img.Width, img.Height,
                    img.FeatureWidth, img.FeatureHeight);
            else
                using (var grayDev = GpuImageOps.GrayscaleToDevice(accel, img.GpuRgba!,
                           img.Width, img.Height, img.FeatureWidth, img.FeatureHeight))
                    img.Features = await _gpuDetector.DetectAsync(accel, grayDev.View, img.FeatureWidth, img.FeatureHeight);
            // Colours are sampled at IMAGE resolution after the caller's scale-back in the managed path; with the
            // capped decode the feature and image resolutions are the same, so sampling here is the same pixel.
            if (img.FeatureWidth == img.Width && img.FeatureHeight == img.Height)
                await GpuImageOps.SampleFeatureColoursAsync(accel, img.GpuRgba!, img.Width, img.Height, img.Features);
            else
            {
                float sx = (float)img.Width / img.FeatureWidth, sy = (float)img.Height / img.FeatureHeight;
                var scaled = img.Features.Select(f => new ImageFeature { X = f.X * sx, Y = f.Y * sy }).ToList();
                await GpuImageOps.SampleFeatureColoursAsync(accel, img.GpuRgba!, img.Width, img.Height, scaled);
                for (int i = 0; i < scaled.Count; i++) img.Features[i].PackedColor = scaled[i].PackedColor;
            }
        }
        finally
        {
            if (img.Source != null) img.DisposeGpu();
        }
    }

    /// <summary>
    /// Learned features for a legacy MANAGED image (file picker / byte arrays: its pixels are on the host already).
    /// At feature resolution, like <see cref="FeatureDetector.Detect"/>.
    /// </summary>
    private async Task<List<ImageFeature>> DetectLearnedFromManagedAsync(ImportedImage img)
    {
        var accel = _gpu.WebGPUAccelerator;
        // CPU transfer: the decoded photo, uploaded once for the extractor (this path's pixels only exist on the host).
        using var rgba = accel.Allocate1D(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(img.RgbaPixels).ToArray());
        var features = await _learned.ExtractAsync(img, rgba, img.Width, img.Height, img.FeatureWidth, img.FeatureHeight);
        await accel.SynchronizeAsync();
        return features;
    }

    /// <summary>
    /// Import pre-loaded images (from OPFS or byte arrays). Detects features and matches pairs.
    /// Used by the multi-view pipeline where images are already decoded.
    /// </summary>
    public async Task ImportFromImagesAsync(IReadOnlyList<ImportedImage> images)
    {
        IsProcessing = true;
        Progress = 0;
        TotalImages = images.Count;
        CurrentImageIndex = 0;
        NotifyChanged();

        if (!_gpu.IsInitialized)
        {
            Status = "Initializing GPU...";
            NotifyChanged();
            await Task.Yield();
            try { await _gpu.InitializeAsync(); }
            catch (Exception ex) { Console.WriteLine($"[Import] GPU init failed: {ex.Message}"); }
        }

        try
        {
            for (int i = 0; i < images.Count; i++)
            {
                var img = images[i];
                CurrentImageIndex = i + 1;
                Progress = (float)i / images.Count;

                // Ensure grayscale + features exist
                bool onDevice = img.GpuRgba != null || img.Source != null;
                if ((img.Features == null || img.Features.Count == 0) && (img.GrayPixels == null || img.GrayPixels.Length == 0))
                {
                    int featureWidth = img.Width, featureHeight = img.Height;
                    if (img.Width > 1024 || img.Height > 1024)
                    {
                        float ds = 1024f / Math.Max(img.Width, img.Height);
                        featureWidth = (int)(img.Width * ds);
                        featureHeight = (int)(img.Height * ds);
                    }
                    // A GPU-resident photo never produces a host grayscale frame: GpuFeatureDetector works on the
                    // device (below). The managed path keeps its CPU grayscale.
                    if (onDevice) { }
                    else if (featureWidth != img.Width)
                        img.GrayPixels = DownsampleGrayscale(img.RgbaPixels, img.Width, img.Height, featureWidth, featureHeight);
                    else
                        img.GrayPixels = RgbaToGrayscale(img.RgbaPixels, img.Width, img.Height);
                    img.FeatureWidth = featureWidth;
                    img.FeatureHeight = featureHeight;
                }

                if (img.Features == null || img.Features.Count == 0)
                {
                    Status = $"Detecting features: {img.FileName} ({i + 1}/{images.Count})...";
                    NotifyChanged();
                    await Task.Yield();

                    if (onDevice)
                        await DetectOnDeviceAsync(img);
                    else if (UseLearnedFeatures)
                        img.Features = await DetectLearnedFromManagedAsync(img);
                    else
                        img.Features = _detector.Detect(img.GrayPixels, img.FeatureWidth, img.FeatureHeight);

                    // Scale feature coordinates back to full image resolution
                    if (img.FeatureWidth != img.Width)
                    {
                        float scaleBackX = (float)img.Width / img.FeatureWidth;
                        float scaleBackY = (float)img.Height / img.FeatureHeight;
                        foreach (var feat in img.Features)
                        {
                            feat.X *= scaleBackX;
                            feat.Y *= scaleBackY;
                        }
                    }
                    // Colour each feature now, while the photo is at hand, so nothing later needs the pixels on the host.
                    if (!onDevice)
                        GpuImageOps.SampleFeatureColours(img.RgbaPixels, img.Width, img.Height, img.Features);
                    Console.WriteLine($"[Import] {img.FileName}: {img.Features.Count} features");
                }

                if (!_images.Contains(img))
                    _images.Add(img);

                NotifyChanged();
                await Task.Yield();
            }

            if (_images.Count >= 2 && !SkipPairMatching)
                await MatchAllPairsAsync();
            else if (SkipPairMatching)
                Console.WriteLine(
                    $"[Import] skipped {_images.Count * (_images.Count - 1) / 2:N0} pair matches " +
                    "- this run uses ground-truth poses and never reads them");

            Progress = 1.0f;
        }
        catch (Exception ex)
        {
            Status = $"Import error: {ex.Message}";
            Console.WriteLine($"[Import] Error: {ex}");
        }
        finally
        {
            IsProcessing = false;
            Status = $"{_images.Count} images imported, {_pairs.Count} pairs matched";
            NotifyChanged();
        }
    }

    /// <summary>
    /// Decode image bytes into RGBA pixel data using a temporary canvas.
    /// Uses SpawnDev.SpawnJS's Blob and OffscreenCanvas for efficient interop.
    /// </summary>
    /// <summary>
    /// Longest edge, in pixels, that an imported photograph is decoded at.
    ///
    /// A modern phone frame is 13 megapixels; 35 of them as RGBA in the managed heap is 1.8 GB
    /// against a 2 GB WASM ceiling, and Bathroom died exactly there with an
    /// OutOfMemoryException. Nothing downstream wants that resolution: the depth model resizes
    /// to 518x518, feature detection already downsamples to 1024, and the unprojection makes
    /// one splat per pixel, so full resolution would be 13 million splats per view.
    ///
    /// The resize happens in the CANVAS during decode, so the full-size bitmap never becomes a
    /// managed array at all - it is not read and then shrunk.
    /// </summary>
    public static int MaxImportDimension { get; set; } = 1024;

    /// <summary>The Bathroom capture's 35 phone photos (the dataset has no manifest).</summary>
    public static readonly string[] BathroomImages = {
                    "IMG_20260223_133436884.jpg", "IMG_20260223_133439584.jpg",
                    "IMG_20260223_133441993_HDR.jpg", "IMG_20260223_133446428_HDR.jpg",
                    "IMG_20260223_133449608.jpg", "IMG_20260223_133453518.jpg",
                    "IMG_20260223_133455077.jpg", "IMG_20260223_133457370.jpg",
                    "IMG_20260223_133459619.jpg", "IMG_20260223_133501789.jpg",
                    "IMG_20260223_133504521.jpg", "IMG_20260223_133506760.jpg",
                    "IMG_20260223_133509247.jpg", "IMG_20260223_133512126.jpg",
                    "IMG_20260223_133520304.jpg", "IMG_20260223_133524853.jpg",
                    "IMG_20260223_133528946.jpg", "IMG_20260223_133531494.jpg",
                    "IMG_20260223_133534018.jpg", "IMG_20260223_133535902.jpg",
                    "IMG_20260223_133538361.jpg", "IMG_20260223_133541652.jpg",
                    "IMG_20260223_133544880.jpg", "IMG_20260223_133546894.jpg",
                    "IMG_20260223_133548727.jpg", "IMG_20260223_133551070.jpg",
                    "IMG_20260223_133553527.jpg", "IMG_20260223_133556013.jpg",
                    "IMG_20260223_133558348.jpg", "IMG_20260223_133601527.jpg",
                    "IMG_20260223_133603411.jpg", "IMG_20260223_133609313.jpg",
                    "IMG_20260223_133612910_HDR.jpg", "IMG_20260223_133616395.jpg",
                    "IMG_20260223_133618729.jpg"
                };

    /// <summary>Decode an image at most <see cref="MaxImportDimension"/> on its longest edge (resized in the canvas,
    /// so the full-size bitmap never becomes a managed array). Every multi-image path must decode through this.</summary>
    internal async Task<(byte[] rgba, int width, int height)?> DecodeImageAsync(byte[] bytes, string mimeType)
    {
        try
        {
            // Create a Blob from the raw bytes (SpawnDev.SpawnJS efficient interop)
            using var blob = new Blob(new[] { bytes }, new BlobOptions { Type = mimeType });

            // Decode via createImageBitmap (async, off main thread in the browser)
            using var imageBitmap = await SpawnJSRuntime.Instance.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);

            int srcW = (int)imageBitmap.Width;
            int srcH = (int)imageBitmap.Height;

            // Cap the decode size. Aspect is preserved, and an image already small enough is
            // left exactly alone rather than resampled for nothing.
            int width = srcW, height = srcH;
            int longest = Math.Max(srcW, srcH);
            if (MaxImportDimension > 0 && longest > MaxImportDimension)
            {
                float s = (float)MaxImportDimension / longest;
                width = Math.Max(1, (int)MathF.Round(srcW * s));
                height = Math.Max(1, (int)MathF.Round(srcH * s));
            }

            // Draw to an OffscreenCanvas to extract pixel data
            using var canvas = new OffscreenCanvas(width, height);
            using var ctx = canvas.Get2DContext();
            ctx.JSRef!.CallVoid("drawImage", imageBitmap, 0, 0, width, height);

            // Get the pixel data
            using var imageData = ctx.GetImageData(0, 0, width, height);
            using var data = imageData.Data;

            // Read the pixel data into a byte array
            var rgba = data.ReadBytes();

            return (rgba, width, height);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Import] Failed to decode image: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Convert RGBA pixel data to grayscale.
    /// </summary>
    private static byte[] RgbaToGrayscale(byte[] rgba, int width, int height)
    {
        var gray = new byte[width * height];
        for (int i = 0; i < width * height; i++)
        {
            int r = rgba[i * 4];
            int g = rgba[i * 4 + 1];
            int b = rgba[i * 4 + 2];
            // ITU-R BT.601 luminance coefficients
            gray[i] = (byte)((r * 77 + g * 150 + b * 29) >> 8);
        }
        return gray;
    }

    /// <summary>
    /// Downsample RGBA to a smaller grayscale image for faster feature detection.
    /// Uses area averaging for quality downsampling.
    /// </summary>
    private static byte[] DownsampleGrayscale(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
    {
        var gray = new byte[dstW * dstH];
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        for (int dy = 0; dy < dstH; dy++)
        {
            int sy = Math.Min((int)(dy * scaleY), srcH - 1);
            for (int dx = 0; dx < dstW; dx++)
            {
                int sx = Math.Min((int)(dx * scaleX), srcW - 1);
                int srcIdx = (sy * srcW + sx) * 4;
                int r = rgba[srcIdx];
                int g = rgba[srcIdx + 1];
                int b = rgba[srcIdx + 2];
                gray[dy * dstW + dx] = (byte)((r * 77 + g * 150 + b * 29) >> 8);
            }
        }
        return gray;
    }

    /// <summary>
    /// Match features across all image pairs.
    /// Yields to UI between pairs to prevent freezing.
    /// </summary>
    // The last full match: which images, in which order, by which front end - and the result. The multi-view overlap pass
    // clears and re-imports the SAME images (BuildOverlapMatrixAsync); matching is deterministic for given features, so it
    // reuses the pairs. MEASURED 2026-10-01: DrJohnson's learned matching ran twice, 559 s + 645 s.
    private List<ImportedImage>? _matchedImages;
    private string? _matchedBy;
    private List<ImagePair>? _matchedResult;

    private string FrontEndKey => UseLearnedFeatures ? $"learned K={LearnedFeatureMatcher.KeypointBudget}" : "BRIEF";

    /// <summary>
    /// The learned front end matches every pair up to this many images; above it, only each image's
    /// <see cref="LearnedRetrievalTopK"/> best partners by <see cref="LearnedFeatureMatcher.PairScoresAsync"/> (LightGlue is
    /// ~0.6 s a pair in the browser: TruckFull's 31,375 pairs would be ~5 h). &amp;lgretrieval=N sets it (0 = always all pairs).
    /// </summary>
    public int LearnedAllPairsUpTo { get; set; } = 60;

    /// <summary>Partners per image when retrieval chooses the pairs (&amp;lgtopk=N).</summary>
    public int LearnedRetrievalTopK { get; set; } = 30;

    private async Task MatchAllPairsAsync()
    {
        _pairs.Clear();
        if (_matchedResult != null && _matchedBy == FrontEndKey && _matchedImages != null
            && _matchedImages.Count == _images.Count && _matchedImages.Zip(_images).All(t => ReferenceEquals(t.First, t.Second)))
        {
            _pairs.AddRange(_matchedResult);
            Console.WriteLine($"[Import] reused the {_pairs.Count} matched pairs of these {_images.Count} images ({FrontEndKey})");
            NotifyChanged();
            return;
        }
        var pairs = new List<(int A, int B)>();
        for (int i = 0; i < _images.Count - 1; i++)
            for (int j = i + 1; j < _images.Count; j++)
                pairs.Add((i, j));
        if (UseLearnedFeatures && LearnedAllPairsUpTo > 0 && _images.Count > LearnedAllPairsUpTo)
        {
            var tr = System.Diagnostics.Stopwatch.StartNew();
            Status = $"Choosing image pairs ({_images.Count} images)...";
            NotifyChanged();
            var scores = await _learned.PairScoresAsync(_images);
            int all = pairs.Count;
            pairs = LearnedFeatureMatcher.TopPartnerPairs(scores, LearnedRetrievalTopK);
            Console.WriteLine($"[Import] retrieval: {pairs.Count} of {all} pairs (top {LearnedRetrievalTopK} partners per image) " +
                $"in {tr.Elapsed.TotalSeconds:F1}s");
        }
        int totalPairs = pairs.Count;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        void OnPair(int p, List<FeatureMatch> matches)
        {
            if (matches.Count < 8) return;
            var (i, j) = pairs[p];
            _pairs.Add(new ImagePair
            {
                ImageIndexA = i,
                ImageIndexB = j,
                Matches = matches,
                InlierCount = matches.Count,
            });
            Console.WriteLine($"[Import] Matched {_images[i].FileName} ↔ {_images[j].FileName}: {matches.Count} matches");
        }
        bool firstBatch = true;
        async Task OnProgress(int done)
        {
            if (firstBatch)
            {
                // The learned matcher's attention is O(K^2) per pair: its first batch's footprint is the number that decides
                // pairs-per-run (8 pairs x K=1024 lost the device on DrJohnson, 2026-10-01).
                firstBatch = false;
                Console.WriteLine($"[Import] first match batch ({done} pairs) after {sw.Elapsed.TotalSeconds:F1}s; [GPU] {GpuService.MemoryReport(4)}");
            }
            Status = $"Matching pairs {done}/{totalPairs}...";
            Progress = (float)done / totalPairs;
            NotifyChanged();
            await Task.Yield();
        }

        if (UseLearnedFeatures)
        {
            // Every image is extracted by now: free the extractors before the matcher allocates, and the matcher after,
            // so the depth model that runs next has the device (4.6 GB of model pools held there lost it, 2026-10-01).
            _learned.ReleaseExtractors();
            try { await _learned.MatchPairsAsync(_images, pairs, OnPair, OnProgress); }
            finally { _learned.ReleaseMatcher(); }
            Console.WriteLine($"[Import] learned models released; [GPU] {GpuService.MemoryReport(4)}");
        }
        else
            // Batched on the device: every image's descriptors uploaded once, hundreds of pairs per dispatch. Pair by
            // pair this was 212.5 s for Truck's 7,875 pairs (~27 ms of round trips each); the matches are identical.
            await _gpuMatcher.MatchPairsAsync(
                _images.Select(im => (IReadOnlyList<ImageFeature>)im.Features).ToList(), pairs, OnPair, OnProgress);
        _matchedImages = new List<ImportedImage>(_images);
        _matchedBy = FrontEndKey;
        _matchedResult = new List<ImagePair>(_pairs);
        Console.WriteLine($"[Import] {(UseLearnedFeatures ? $"LightGlue+ (K={LearnedFeatureMatcher.KeypointBudget})" : "BRIEF")} matched {totalPairs} pairs in {sw.Elapsed.TotalSeconds:F1}s ({_pairs.Count} with >= 8 matches); {HeapReport()}");

        NotifyChanged();
    }

    private void NotifyChanged() => OnStateChanged?.Invoke();

    /// <summary>Managed heap now: in use, committed, and what the runtime says is available. For finding an OOM.</summary>
    public static string HeapReport()
    {
        long before = GC.GetTotalMemory(false);
        long live = GC.GetTotalMemory(true);   // after a full collection: what is actually still referenced
        var gi = GC.GetGCMemoryInfo();
        return $"heap {before / (1024 * 1024)} MB in use, {live / (1024 * 1024)} MB live after full GC, " +
            $"{gi.TotalCommittedBytes / (1024 * 1024)} MB committed, pinned {gi.PinnedObjectsCount}, " +
            $"fragmented {gi.FragmentedBytes / (1024 * 1024)} MB, {gi.TotalAvailableMemoryBytes / (1024 * 1024)} MB available";
    }

    /// <summary>
    /// Clear all imported images and matches.
    /// </summary>
    public void Clear()
    {
        // Does NOT dispose the images: callers clear and re-import the SAME images (the multi-view overlap pass does
        // exactly that), and project images belong to the project flow. Images this service created are released
        // when the next dataset load replaces them (ReleaseOwnedImages).
        _images.Clear();
        _pairs.Clear();
        Status = "";
        Progress = 0;
        NotifyChanged();
    }

    /// <summary>
    /// Load a sample dataset from wwwroot/datasets/ for testing.
    /// </summary>
    /// <summary>
    /// Skip pairwise feature matching on load.
    ///
    /// Matching is O(n^2) in images - 132 images is 8,646 pairs - and it exists to feed SfM
    /// pose recovery. A run using ground-truth poses never looks at the result, so every one of
    /// those pairs is wall clock spent on an answer nobody reads, and it grows quadratically
    /// exactly as more views are added to improve quality.
    /// </summary>
    public bool SkipPairMatching { get; set; }

    /// <summary>Images the dataset loader created (it owns their encoded Sources and any device copies).</summary>
    private readonly List<ImportedImage> _ownedImages = new();

    /// <summary>Release every image this service created (JS Blobs, GPU buffers). Their users must be done with them.</summary>
    public void ReleaseOwnedImages()
    {
        _matchedImages = null;
        _matchedResult = null;
        foreach (var img in _ownedImages) img.DisposeSource();
        _ownedImages.Clear();
    }

    public async Task LoadSampleDatasetAsync(string datasetName)
    {
        Clear();
        ReleaseOwnedImages(); // the previous dataset's photos: nothing uses them once a new load starts
        IsProcessing = true;
        NotifyChanged();

        // Ensure GPU is initialized
        if (!_gpu.IsInitialized)
        {
            Status = "Initializing GPU...";
            NotifyChanged();
            await Task.Yield();
            try { await _gpu.InitializeAsync(); }
            catch (Exception ex) { Console.WriteLine($"[Import] GPU init failed: {ex.Message}"); }
        }

        try
        {
            // Fetch the file list from the dataset
            var imageNames = new List<string>();
            string basePath;

            // A manifest, if the dataset has one, so adding a dataset costs no code.
            //
            // Everything below this is an if-else chain with filenames pasted into C#, which
            // means a new capture cannot be tried without editing and rebuilding the app. The
            // COLMAP converter (tools/colmap_to_dataset.py) writes a manifest instead, and the
            // static server reads the same file to mount the images from wherever they actually
            // live rather than copying 168 MB into wwwroot.
            var manifest = await TryLoadManifestAsync(datasetName);
            List<VideoFrameExtractor.Frame>? videoFrames = null;
            if (manifest != null && !string.IsNullOrEmpty(manifest.Video))
            {
                // A video dataset: the frames are chosen and decoded in the browser, then go through exactly the
                // same decode / feature / match path as photographs.
                basePath = VideoFrameStore.Prefix;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                int want = manifest.VideoFrames > 0 ? manifest.VideoFrames : 120;
                Status = $"Extracting {want} frames from {manifest.Video}...";
                NotifyChanged();
                videoFrames = await _video.ExtractAsync($"datasets/{datasetName}/{manifest.Video}", want,
                    manifest.VideoCandidates > 0 ? manifest.VideoCandidates : 3, 1600);
                foreach (var f in videoFrames)
                {
                    VideoFrameStore.Put(f.Name, f.Jpeg);
                    imageNames.Add(f.Name);
                }
                Console.WriteLine(
                    $"[Import] {datasetName}: {videoFrames.Count} frames from {manifest.Video} in " +
                    $"{sw.Elapsed.TotalSeconds:F1}s (sharpest of {Math.Max(1, manifest.VideoCandidates > 0 ? manifest.VideoCandidates : 3)} per slot; " +
                    $"times {string.Join(" ", videoFrames.Take(4).Select(f => f.TimeSeconds.ToString("F2")))} ...)");
                Console.WriteLine($"[Import] after extraction ({videoFrames.Sum(f => (long)f.Jpeg.Length) / (1024 * 1024)} MB of JPEG): {HeapReport()}");
            }
            else if (manifest != null)
            {
                basePath = $"datasets/{datasetName}/{manifest.ImageDir}/";
                imageNames.AddRange(manifest.Images);
                Console.WriteLine(
                    $"[Import] {datasetName}: manifest lists {imageNames.Count} images " +
                    $"at {manifest.Width}x{manifest.Height}" +
                    (string.IsNullOrEmpty(manifest.Poses) ? "" : $", poses in {manifest.Poses}"));
            }
            else if (datasetName == "Skull")
            {
                // Skull: 01.JPG - 75.JPG directly in datasets/Skull/
                // Every 3rd for ~25 images — good overlap for matching
                basePath = $"datasets/{datasetName}/";
                for (int i = 1; i <= 75; i += 3)
                    imageNames.Add($"{i:D2}.JPG");
            }
            else if (datasetName == "Bathroom")
            {
                basePath = $"datasets/{datasetName}/";
                // All 35 bathroom images
                imageNames.AddRange(BathroomImages);
            }
            else if (datasetName == "TempleRing")
            {
                basePath = $"datasets/{datasetName}/";
                // 16 views sampled from 47-view ring around temple model (every 3rd)
                foreach (var n in new[] { 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46 })
                    imageNames.Add($"templeR{n:D4}.png");
            }
            else if (datasetName == "DinoSparseRing")
            {
                basePath = $"datasets/{datasetName}/";
                // All 16 views of sparse ring around dinosaur model
                for (int i = 1; i <= 16; i++)
                    imageNames.Add($"dinoSR{i:D4}.png");
            }
            else if (datasetName == "SouthBuilding")
            {
                basePath = $"datasets/{datasetName}/";
                // 22 views sampled from 128-image outdoor building dataset (every 6th)
                imageNames.AddRange(new[] {
                    "P1180141.JPG", "P1180147.JPG", "P1180153.JPG", "P1180159.JPG",
                    "P1180165.JPG", "P1180171.JPG", "P1180177.JPG", "P1180183.JPG",
                    "P1180189.JPG", "P1180195.JPG", "P1180201.JPG", "P1180207.JPG",
                    "P1180213.JPG", "P1180219.JPG", "P1180225.JPG", "P1180310.JPG",
                    "P1180316.JPG", "P1180322.JPG", "P1180328.JPG", "P1180334.JPG",
                    "P1180340.JPG", "P1180346.JPG"
                });
            }
            else
            {
                basePath = $"datasets/{datasetName}/Images/";
                for (int i = 3472; i <= 3485; i++)
                    imageNames.Add($"IMG_{i}.JPG");
            }

            TotalImages = imageNames.Count;
            CurrentImageIndex = 0;

            for (int fi = 0; fi < imageNames.Count; fi++)
            {
                var fileName = imageNames[fi];
                CurrentImageIndex = fi + 1;
                Progress = (float)fi / imageNames.Count;
                Status = $"Loading {fileName} ({fi + 1}/{imageNames.Count})...";
                NotifyChanged();
                await Task.Yield();

                // The photo as a JS Blob - fetched by the BROWSER (the response body never enters .NET), or a video frame's
                // JPEG. It stays compressed on the image (ImportedImage.Source) and is decoded straight to the device
                // wherever its pixels are needed. This used to GetByteArrayAsync + decode + ReadBytes every photo into
                // the managed heap and keep it there for the whole run (TruckFull: 251 x 2.3 MB).
                SpawnDev.SpawnJS.JSObjects.Blob source;
                try
                {
                    if (videoFrames != null)
                        source = new SpawnDev.SpawnJS.JSObjects.Blob(new byte[][] { videoFrames[fi].Jpeg },
                            new BlobOptions { Type = "image/jpeg" });
                    else
                    {
                        using var response = await SpawnJSRuntime.Instance.CallAsync<string, Response>("fetch", basePath + fileName);
                        if (!response.Ok) throw new InvalidOperationException($"HTTP {response.Status}");
                        source = await response.Blob();
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Import] Failed to fetch {fileName}: {ex.Message}");
                    continue;
                }

                MemoryBuffer1D<int, Stride1D.Dense> rgbaDev;
                int imgWidth, imgHeight;
                try
                {
                    (rgbaDev, imgWidth, imgHeight, _, _) = await SpawnDev.ILGPU.ML.Preprocessing.MediaInterop.DecodeToDeviceAsync(
                        source, _gpu.WebGPUAccelerator, MaxImportDimension);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Import] Failed to decode image: {ex.Message}");
                    source.Dispose();
                    continue;
                }

                int featureWidth = imgWidth, featureHeight = imgHeight;
                if (imgWidth > 1024 || imgHeight > 1024)
                {
                    float ds = 1024f / Math.Max(imgWidth, imgHeight);
                    featureWidth = (int)(imgWidth * ds);
                    featureHeight = (int)(imgHeight * ds);
                }

                var imported = new ImportedImage
                {
                    FileName = fileName,
                    SourceUrl = basePath + fileName,
                    Width = imgWidth,
                    Height = imgHeight,
                    Source = source,
                    DecodeMaxEdge = MaxImportDimension,
                    GpuRgba = rgbaDev,
                    FeatureWidth = featureWidth,
                    FeatureHeight = featureHeight,
                };

                Status = $"Detecting features in {fileName} ({fi + 1}/{imageNames.Count})...";
                NotifyChanged();
                await Task.Yield();

                _ownedImages.Add(imported);
                await DetectOnDeviceAsync(imported); // also releases the device copy (re-decoded when needed)
                if (featureWidth != imgWidth)
                {
                    float scaleBackX = (float)imgWidth / featureWidth;
                    float scaleBackY = (float)imgHeight / featureHeight;
                    foreach (var feat in imported.Features)
                    {
                        feat.X *= scaleBackX;
                        feat.Y *= scaleBackY;
                    }
                }

                Console.WriteLine($"[Import] {fileName}: {imported.Features.Count} features ({featureWidth}×{featureHeight})");
                _images.Add(imported);
                Progress = (float)(fi + 1) / imageNames.Count;
                NotifyChanged();
                await Task.Yield();
            }

            Console.WriteLine($"[Import] {_images.Count} images decoded + features: {HeapReport()}");
            if (_images.Count >= 2 && !SkipPairMatching)
                await MatchAllPairsAsync();

            Progress = 1.0f;
        }
        catch (Exception ex)
        {
            Status = $"Import error: {ex.Message}";
            Console.WriteLine($"[Import] Error: {ex}");
        }
        finally
        {
            IsProcessing = false;
            Status = $"{_images.Count} images imported, {_pairs.Count} pairs matched";
            NotifyChanged();
        }
    }

    public void Dispose()
    {
        Clear();
        ReleaseOwnedImages();
        GC.SuppressFinalize(this);
    }

    /// <summary>A dataset described by a file rather than by a branch in this method.</summary>
    public sealed class DatasetManifest
    {
        public string Name { get; set; } = "";
        public string ImageDir { get; set; } = "images";
        public List<string> Images { get; set; } = new();
        public int Width { get; set; }
        public int Height { get; set; }
        /// <summary>Middlebury-format camera parameters, or empty when the capture is unposed.</summary>
        public string Poses { get; set; } = "";

        /// <summary>
        /// Sparse SfM point cloud, or empty when the dataset has none.
        ///
        /// This is what 3DGS initialises from. Every point is triangulated from at least two
        /// images, which is the property a per-view depth unprojection does not have.
        /// </summary>
        public string Points { get; set; } = "";

        /// <summary>Points in <see cref="Points"/>, for reporting before the file is fetched.</summary>
        public int PointCount { get; set; }

        /// <summary>A video file in the dataset folder to take the images from, instead of <see cref="Images"/>.</summary>
        public string Video { get; set; } = "";

        /// <summary>Frames to take from <see cref="Video"/> (evenly spaced slots). 0 = 120.</summary>
        public int VideoFrames { get; set; }

        /// <summary>Candidate frames scored per slot; the sharpest is kept. 0 = 3.</summary>
        public int VideoCandidates { get; set; }
    }

    /// <summary>
    /// Fetch a dataset's sparse SfM point cloud. Null when it has none.
    /// </summary>
    public async Task<byte[]?> TryLoadPointCloudAsync(string datasetName, string file)
    {
        if (string.IsNullOrEmpty(file)) return null;
        try
        {
            using var resp = await _http.GetAsync($"datasets/{datasetName}/{file}");
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadAsByteArrayAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Import] point cloud {datasetName}/{file} failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Read <c>datasets/&lt;name&gt;/manifest.json</c> if it exists. A missing manifest is the
    /// normal case for the datasets that predate it, not an error.
    /// </summary>
    public async Task<DatasetManifest?> TryLoadManifestAsync(string datasetName)
    {
        try
        {
            using var resp = await _http.GetAsync($"datasets/{datasetName}/manifest.json");
            if (!resp.IsSuccessStatusCode) return null;
            var json = await resp.Content.ReadAsStringAsync();
            var m = System.Text.Json.JsonSerializer.Deserialize<DatasetManifest>(json,
                new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                });
            return m is { Images.Count: > 0 } || !string.IsNullOrEmpty(m?.Video) ? m : null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Import] manifest for {datasetName} not usable: {ex.Message}");
            return null;
        }
    }
}
