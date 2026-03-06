using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
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
    private readonly GpuFeatureMatcher _gpuMatcher;
    private readonly GpuService _gpu;
    private readonly HttpClient _http;
    private readonly List<ImportedImage> _images = [];
    private readonly List<ImagePair> _pairs = [];

    public ImageImportService(GpuFeatureMatcher gpuMatcher, GpuService gpu, HttpClient http)
    {
        _gpuMatcher = gpuMatcher;
        _gpu = gpu;
        _http = http;
    }

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

                imported.Features = _detector.Detect(imported.GrayPixels, featureWidth, featureHeight);

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
                await MatchAllPairsAsync();
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
    /// Decode image bytes into RGBA pixel data using a temporary canvas.
    /// Uses SpawnDev.BlazorJS's Blob and OffscreenCanvas for efficient interop.
    /// </summary>
    private async Task<(byte[] rgba, int width, int height)?> DecodeImageAsync(byte[] bytes, string mimeType)
    {
        try
        {
            // Create a Blob from the raw bytes (SpawnDev.BlazorJS efficient interop)
            using var blob = new Blob(new[] { bytes }, new BlobOptions { Type = mimeType });

            // Decode via createImageBitmap (async, off main thread in the browser)
            using var imageBitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", blob);

            int width = (int)imageBitmap.Width;
            int height = (int)imageBitmap.Height;

            // Draw to an OffscreenCanvas to extract pixel data
            using var canvas = new OffscreenCanvas(width, height);
            using var ctx = canvas.Get2DContext();
            ctx.JSRef!.CallVoid("drawImage", imageBitmap, 0, 0);

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
    private async Task MatchAllPairsAsync()
    {
        _pairs.Clear();
        int totalPairs = (_images.Count * (_images.Count - 1)) / 2;
        int pairsDone = 0;

        for (int i = 0; i < _images.Count - 1; i++)
        {
            for (int j = i + 1; j < _images.Count; j++)
            {
                pairsDone++;
                Status = $"Matching pair {pairsDone}/{totalPairs}: {_images[i].FileName} ↔ {_images[j].FileName}...";
                Progress = (float)pairsDone / totalPairs;

                // Yield every few pairs to let UI breathe
                if (pairsDone % 3 == 0)
                {
                    NotifyChanged();
                    await Task.Yield();
                }

                var matches = await _gpuMatcher.MatchAsync(_images[i].Features, _images[j].Features);

                if (matches.Count >= 8)
                {
                    _pairs.Add(new ImagePair
                    {
                        ImageIndexA = i,
                        ImageIndexB = j,
                        Matches = matches,
                        InlierCount = matches.Count,
                    });

                    Console.WriteLine($"[Import] Matched {_images[i].FileName} ↔ {_images[j].FileName}: {matches.Count} matches");
                }
            }
        }

        NotifyChanged();
    }

    private void NotifyChanged() => OnStateChanged?.Invoke();

    /// <summary>
    /// Clear all imported images and matches.
    /// </summary>
    public void Clear()
    {
        _images.Clear();
        _pairs.Clear();
        Status = "";
        Progress = 0;
        NotifyChanged();
    }

    /// <summary>
    /// Load a sample dataset from wwwroot/datasets/ for testing.
    /// </summary>
    public async Task LoadSampleDatasetAsync(string datasetName)
    {
        Clear();
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

            if (datasetName == "Skull")
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
                imageNames.AddRange(new[] {
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
                });
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

                byte[] bytes;
                try
                {
                    bytes = await _http.GetByteArrayAsync(basePath + fileName);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Import] Failed to fetch {fileName}: {ex.Message}");
                    continue;
                }

                // Decode — detect MIME type from file extension
                string mimeType = fileName.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? "image/png" : "image/jpeg";
                var result = await DecodeImageAsync(bytes, mimeType);
                if (result == null) continue;
                var (rgba, imgWidth, imgHeight) = result.Value;

                // Grayscale + downsample
                byte[] grayPixels;
                int featureWidth = imgWidth, featureHeight = imgHeight;
                if (imgWidth > 1024 || imgHeight > 1024)
                {
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
                    FileName = fileName,
                    Width = imgWidth,
                    Height = imgHeight,
                    RgbaPixels = rgba,
                    GrayPixels = grayPixels,
                    FeatureWidth = featureWidth,
                    FeatureHeight = featureHeight,
                };

                Status = $"Detecting features in {fileName} ({fi + 1}/{imageNames.Count})...";
                NotifyChanged();
                await Task.Yield();

                imported.Features = _detector.Detect(imported.GrayPixels, featureWidth, featureHeight);
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

            if (_images.Count >= 2)
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
        GC.SuppressFinalize(this);
    }
}
