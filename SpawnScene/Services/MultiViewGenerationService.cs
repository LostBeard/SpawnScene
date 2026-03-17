using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Multi-view scene generation using feature-based 2D alignment.
///
/// For near-parallel views (typical phone photos of a room):
/// 1. Pick the first image as the reference frame
/// 2. Generate depth + splats for the reference (single-image pipeline)
/// 3. For each additional image, compute 2D pixel offset via feature matches
/// 4. Generate splats for additional images with the offset applied (extends coverage)
///
/// This avoids the SfM rotation/scale alignment problems that plague near-parallel views
/// while still leveraging multi-image coverage to extend the scene beyond a single photo.
/// </summary>
public class MultiViewGenerationService
{
    private readonly BlazorJSRuntime _js;
    private readonly GpuService _gpu;
    private readonly ImageImportService _importService;
    private readonly SfmReconstructor _sfm;
    private readonly DepthEstimationService _depthService;
    private readonly DepthToGaussianKernel _gaussianKernel;

    public string Status { get; private set; } = "";
    public event Action? OnStatusChanged;

    /// <summary>Camera poses recovered by the last SfM run (index matches input images).</summary>
    public CameraParams?[] SfmCameraPoses => _sfm.CameraPoses;

    public MultiViewGenerationService(
        BlazorJSRuntime js,
        GpuService gpu,
        ImageImportService importService,
        SfmReconstructor sfm,
        DepthEstimationService depthService,
        DepthToGaussianKernel gaussianKernel)
    {
        _js = js;
        _gpu = gpu;
        _importService = importService;
        _sfm = sfm;
        _depthService = depthService;
        _gaussianKernel = gaussianKernel;
    }

    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateAsync(IReadOnlyList<ImportedImage> images, int subsample = 2, float edgeSharpness = 0.3f)
    {
        if (images.Count < 2)
            throw new ArgumentException("Multi-view generation requires at least 2 images.");

        // DAv3 native multi-view: single inference → consistent depth + predicted camera poses
        bool isDav3 = _depthService.LoadedModelId?.StartsWith("depth-anything-v3") == true;
        if (isDav3)
        {
            return await GenerateWithDav3MultiViewAsync(images, subsample, edgeSharpness);
        }

        // ─── Legacy path: feature matching + 2D offset ───

        // ─── Step 1: Feature detection + matching ───
        SetStatus($"Detecting features in {images.Count} images...");
        _importService.Clear();
        await _importService.ImportFromImagesAsync(images);

        if (_importService.MatchedPairs.Count == 0)
        {
            SetStatus("Error: No feature matches found between images.");
            return null;
        }
        SetStatus($"Matched {_importService.MatchedPairs.Count} image pairs.");

        // ─── Step 2: Compute 2D pixel offsets relative to reference image (image 0) ───
        // Use feature matches to find the median pixel displacement between each image and the reference.
        var refIdx = 0;
        var pixelOffsets = new Dictionary<int, (float dx, float dy)>();
        pixelOffsets[refIdx] = (0, 0);

        for (int i = 0; i < images.Count; i++)
        {
            if (i == refIdx) continue;

            var offset = ComputePixelOffset(refIdx, i, images);
            if (offset.HasValue)
            {
                pixelOffsets[i] = offset.Value;
                Console.WriteLine($"[MultiView] Image {i} → ref offset: dx={offset.Value.dx:F1}, dy={offset.Value.dy:F1} pixels");
            }
            else
            {
                // Try indirect: ref→A→i
                Console.WriteLine($"[MultiView] No direct matches for image {i}, trying indirect...");
                bool found = false;
                foreach (var mid in pixelOffsets.Keys)
                {
                    if (mid == refIdx) continue;
                    var midToI = ComputePixelOffset(mid, i, images);
                    if (midToI.HasValue)
                    {
                        var midOff = pixelOffsets[mid];
                        pixelOffsets[i] = (midOff.dx + midToI.Value.dx, midOff.dy + midToI.Value.dy);
                        Console.WriteLine($"[MultiView] Image {i} → ref offset (via {mid}): dx={pixelOffsets[i].dx:F1}, dy={pixelOffsets[i].dy:F1} pixels");
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    Console.WriteLine($"[MultiView] Skipping image {i}: no path to reference.");
                }
            }
        }

        // ─── Step 3: Ensure depth model loaded ───
        if (!_depthService.IsReady)
        {
            SetStatus("Loading depth model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        // ─── Step 4: Pass 1 — generate reference depth, count splats per view ───
        int refW = images[refIdx].Width;
        int refH = images[refIdx].Height;
        var viewCounts = new List<(int imageIndex, float dx, float dy, int count, float depthScale)>();
        int totalSplats = 0;

        // Generate reference depth first and keep it for seam matching
        SetStatus($"Depth estimation: {images[refIdx].FileName} (reference)...");
        var refDepthResult = await _depthService.EstimateDepthAsync(images[refIdx]);
        float[]? refDepthData = null;
        if (refDepthResult != null)
        {
            try { refDepthData = await refDepthResult.RawDepthGpu!.CopyToHostAsync<float>(0, refDepthResult.RawDepthGpu.Length); }
            catch { Console.WriteLine("[MultiView] Reference depth readback failed"); }
        }

        foreach (var (imgIdx, (dx, dy)) in pixelOffsets)
        {
            SetStatus($"Depth estimation: {images[imgIdx].FileName} ({imgIdx + 1}/{images.Count})...");
            DepthResult? depthResult;
            if (imgIdx == refIdx && refDepthResult != null)
            {
                depthResult = refDepthResult;
            }
            else
            {
                depthResult = await _depthService.EstimateDepthAsync(images[imgIdx]);
            }
            if (depthResult == null)
            {
                Console.WriteLine($"[MultiView] Depth failed for image {imgIdx}, skipping.");
                continue;
            }

            // Depth scale correction: 1.0 = no correction (each view uses its own depth scale).
            // TODO: improve seam matching once SfM poses are more accurate.
            float depthScaleCorrection = 1.0f;
            bool isRef = (imgIdx == refIdx);

            SetStatus($"Generating splats: {images[imgIdx].FileName}...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWithOffsetAsync(
                depthResult, images[imgIdx], dx, dy, subsample, edgeSharpness,
                isRef ? 0 : refW, isRef ? 0 : refH, depthScaleCorrection);

            viewCounts.Add((imgIdx, dx, dy, count, depthScaleCorrection));
            totalSplats += count;
            buf.Dispose();
            if (imgIdx != refIdx) depthResult.Dispose();
            Console.WriteLine($"[MultiView] View {imgIdx} ({images[imgIdx].FileName}): {count:N0} splats, offset=({dx:F1},{dy:F1}), depthScale={depthScaleCorrection:F4}");
        }

        refDepthResult?.Dispose();

        if (viewCounts.Count == 0)
        {
            SetStatus("Error: No views produced splats.");
            return null;
        }

        // ─── Step 5: Pass 2 — allocate merged buffer, regenerate + copy ───
        SetStatus($"Merging {totalSplats:N0} splats from {viewCounts.Count} views...");
        var accelerator = _gpu.WebGPUAccelerator;
        var nativeAccel = accelerator.NativeAccelerator;
        var device = nativeAccel.NativeDevice!;
        var queue = nativeAccel.Queue!;

        var merged = accelerator.Allocate1D<float>(totalSplats * 10);
        var mergedGpuBuf = merged.GetGPUBuffer();

        ulong byteOffset = 0;
        foreach (var (imgIdx, dx, dy, expectedCount, viewDepthScale) in viewCounts)
        {
            SetStatus($"Fusing view {imgIdx + 1}/{images.Count} into scene...");

            var depthResult = await _depthService.EstimateDepthAsync(images[imgIdx]);
            if (depthResult == null) continue;

            bool isRef = (imgIdx == refIdx);
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWithOffsetAsync(
                depthResult, images[imgIdx], dx, dy, subsample, edgeSharpness,
                isRef ? 0 : refW, isRef ? 0 : refH, viewDepthScale);
            depthResult.Dispose();

            var srcGpuBuf = buf.GetGPUBuffer();
            ulong byteCount = (ulong)count * 10 * sizeof(float);

            if (srcGpuBuf != null && mergedGpuBuf != null)
            {
                using var encoder = device.CreateCommandEncoder();
                encoder.CopyBufferToBuffer(srcGpuBuf, 0, mergedGpuBuf, byteOffset, byteCount);
                using var cmdBuf = encoder.Finish();
                queue.Submit(new[] { cmdBuf });
            }

            byteOffset += byteCount;
            buf.Dispose();
        }

        int actualTotal = (int)(byteOffset / (10 * sizeof(float)));
        SetStatus($"Multi-view generation complete: {actualTotal:N0} splats from {viewCounts.Count} views.");
        Console.WriteLine($"[MultiView] Total: {actualTotal:N0} splats from {viewCounts.Count} views");

        return (merged, totalSplats);
    }

    /// <summary>
    /// DAv3 native multi-view: all images in one forward pass → consistent depth + camera extrinsics.
    /// No feature matching, no 2D offsets, no per-view depth scale mismatch.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithDav3MultiViewAsync(IReadOnlyList<ImportedImage> images, int subsample, float edgeSharpness)
    {
        // Ensure DAv3 model loaded
        if (!_depthService.IsReady)
        {
            SetStatus("Loading DAv3 model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        // Run multi-view inference
        SetStatus($"Running DAv3 multi-view inference ({images.Count} images)...");
        var mvResult = await _depthService.EstimateDepthMultiViewAsync(images);
        if (mvResult == null || mvResult.DepthResults.Count == 0)
        {
            SetStatus("Error: DAv3 multi-view inference failed.");
            return null;
        }

        Console.WriteLine($"[MultiView-DAv3] Got {mvResult.DepthResults.Count} depth maps" +
            (mvResult.Extrinsics != null ? " + extrinsics" : ""));

        // Parse extrinsics into CameraParams (if available)
        var cameras = new CameraParams?[images.Count];
        bool hasExtrinsics = mvResult.Extrinsics != null;

        if (hasExtrinsics)
        {
            for (int i = 0; i < images.Count && i < mvResult.DepthResults.Count; i++)
            {
                var ext = mvResult.Extrinsics![0, i]; // [R00,R01,R02,tx, R10,R11,R12,ty, R20,R21,R22,tz]
                var cam = images[i].EstimatedCamera ?? CameraParams.CreateDefault(images[i].Width, images[i].Height);

                // Parse 3×4 [R|t] matrix
                cam.Forward = new Vector3(ext[8], ext[9], ext[10]); // third row of R
                cam.Up = new Vector3(-ext[4], -ext[5], -ext[6]);    // negated second row (camera Y flipped)
                cam.Position = new Vector3(
                    -(ext[0] * ext[3] + ext[4] * ext[7] + ext[8] * ext[11]),   // -R^T * t
                    -(ext[1] * ext[3] + ext[5] * ext[7] + ext[9] * ext[11]),
                    -(ext[2] * ext[3] + ext[6] * ext[7] + ext[10] * ext[11])
                );

                cameras[i] = cam;
                Console.WriteLine($"[MultiView-DAv3] View {i}: pos={cam.Position}, fwd={cam.Forward}");
            }
        }

        // Generate splats per view
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        var device = accelerator.NativeAccelerator.NativeDevice!;
        var queue = accelerator.NativeAccelerator.Queue!;

        // Pass 1: count splats per view
        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0;

        for (int i = 0; i < mvResult.DepthResults.Count && i < images.Count; i++)
        {
            SetStatus($"Generating splats: {images[i].FileName} ({i + 1}/{images.Count})...");
            var depth = mvResult.DepthResults[i];
            var cam = cameras[i] ?? images[i].EstimatedCamera;

            (MemoryBuffer1D<float, Stride1D.Dense> buf, int count) result;

            if (hasExtrinsics && cameras[i] != null)
            {
                // World-space unprojection using DAv3's predicted camera poses
                result = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                    depth, images[i], cameras[i]!, subsample, edgeSharpness, depthScale: 1.0f);
            }
            else
            {
                // Fallback: camera-local unprojection with EXIF focal length
                result = await _gaussianKernel.GeneratePackedGpuBufferAsync(
                    depth, images[i], subsample, edgeSharpness, cam);
            }

            viewResults.Add(result);
            totalSplats += result.count;
            Console.WriteLine($"[MultiView-DAv3] View {i}: {result.count:N0} splats");
        }

        if (totalSplats == 0)
        {
            foreach (var (buf, _) in viewResults) buf.Dispose();
            mvResult.Dispose();
            SetStatus("Error: No splats generated.");
            return null;
        }

        // Merge all view buffers into one
        SetStatus($"Merging {totalSplats:N0} splats from {viewResults.Count} views...");
        var merged = accelerator.Allocate1D<float>(totalSplats * 10);
        var mergedGpuBuf = merged.GetGPUBuffer();
        ulong byteOffset = 0;

        foreach (var (buf, count) in viewResults)
        {
            var srcGpuBuf = buf.GetGPUBuffer();
            ulong byteCount = (ulong)count * 10 * sizeof(float);

            if (srcGpuBuf != null && mergedGpuBuf != null)
            {
                using var encoder = device.CreateCommandEncoder();
                encoder.CopyBufferToBuffer(srcGpuBuf, 0, mergedGpuBuf, byteOffset, byteCount);
                using var cmdBuf = encoder.Finish();
                queue.Submit(new[] { cmdBuf });
            }

            byteOffset += byteCount;
            buf.Dispose();
        }

        mvResult.Dispose();

        int actualTotal = (int)(byteOffset / (10 * sizeof(float)));
        SetStatus($"DAv3 multi-view complete: {actualTotal:N0} splats from {viewResults.Count} views.");
        Console.WriteLine($"[MultiView-DAv3] Total: {actualTotal:N0} splats from {viewResults.Count} views");

        return (merged, totalSplats);
    }

    /// <summary>
    /// Compute the median pixel displacement from image A to image B using matched features.
    /// Returns (dx, dy) in pixels where B's features are shifted by (dx, dy) relative to A's.
    /// </summary>
    private (float dx, float dy)? ComputePixelOffset(int idxA, int idxB, IReadOnlyList<ImportedImage> images)
    {
        // Find the matched pair (could be A→B or B→A)
        ImagePair? pair = null;
        bool swapped = false;
        foreach (var p in _importService.MatchedPairs)
        {
            if (p.ImageIndexA == idxA && p.ImageIndexB == idxB) { pair = p; break; }
            if (p.ImageIndexA == idxB && p.ImageIndexB == idxA) { pair = p; swapped = true; break; }
        }
        if (pair == null || pair.Matches.Count < 5) return null;

        var imgA = images[idxA];
        var imgB = images[idxB];
        var featA = swapped ? images[idxB].Features : imgA.Features;
        var featB = swapped ? imgA.Features : images[idxB].Features;

        var dxList = new List<float>();
        var dyList = new List<float>();

        foreach (var m in pair.Matches)
        {
            var fA = featA[m.IndexA];
            var fB = featB[m.IndexB];

            if (swapped)
            {
                dxList.Add(fA.X - fB.X);
                dyList.Add(fA.Y - fB.Y);
            }
            else
            {
                dxList.Add(fB.X - fA.X);
                dyList.Add(fB.Y - fA.Y);
            }
        }

        dxList.Sort();
        dyList.Sort();
        return (dxList[dxList.Count / 2], dyList[dyList.Count / 2]);
    }

    /// <summary>
    /// Compute depth scale correction for a non-reference view by comparing depth values
    /// at the seam boundary between the reference and extension views.
    /// Samples depth along the boundary where both views have coverage, computes
    /// median(refDepth / extDepth) as the scale factor.
    /// </summary>
    private async Task<float> ComputeSeamDepthScaleAsync(
        float[] refDepthData, DepthResult refDepth, DepthResult extDepth,
        float dx, float dy, int refW, int refH)
    {
        int extW = extDepth.Width, extH = extDepth.Height;

        float[] extDepthData;
        try { extDepthData = await extDepth.RawDepthGpu!.CopyToHostAsync<float>(0, extDepth.RawDepthGpu.Length); }
        catch { return 1.0f; }

        float refRange = refDepth.MaxDepth - refDepth.MinDepth;
        float extRange = extDepth.MaxDepth - extDepth.MinDepth;
        if (refRange < 1e-6f || extRange < 1e-6f) return 1.0f;

        // Sample along the exclusion boundary (the edge where ref coverage meets extension)
        // The boundary in the extension image is where the exclusion rect edge is.
        int exclX0 = Math.Clamp((int)(-dx), 0, extW);
        int exclY0 = Math.Clamp((int)(-dy), 0, extH);
        int exclX1 = Math.Clamp((int)(refW - dx), 0, extW);
        int exclY1 = Math.Clamp((int)(refH - dy), 0, extH);

        var ratios = new List<float>();
        int step = 8; // Sample every 8 pixels along the boundary

        // Sample along the vertical boundaries (left and right edges of exclusion)
        foreach (int bx in new[] { exclX0, exclX1 - 1 })
        {
            if (bx < 0 || bx >= extW) continue;
            for (int ey = exclY0; ey < exclY1; ey += step)
            {
                if (ey < 0 || ey >= extH) continue;

                // Extension depth at this pixel
                float extRaw = extDepthData[ey * extW + bx];
                float extNorm = (extRaw - extDepth.MinDepth) / extRange;
                if (extNorm < 0.01f) continue;
                float extD = 1.0f / (extNorm + 0.01f);

                // Corresponding reference pixel
                int refX = (int)(bx + dx);
                int refY = (int)(ey + dy);
                if (refX < 0 || refX >= refW || refY < 0 || refY >= refH) continue;

                float refRaw = refDepthData[refY * refW + refX];
                float refNorm = (refRaw - refDepth.MinDepth) / refRange;
                if (refNorm < 0.01f) continue;
                float refD = 1.0f / (refNorm + 0.01f);

                if (extD > 0.01f && refD > 0.01f)
                    ratios.Add(refD / extD);
            }
        }

        // Sample along horizontal boundaries
        foreach (int by in new[] { exclY0, exclY1 - 1 })
        {
            if (by < 0 || by >= extH) continue;
            for (int ex = exclX0; ex < exclX1; ex += step)
            {
                if (ex < 0 || ex >= extW) continue;

                float extRaw = extDepthData[by * extW + ex];
                float extNorm = (extRaw - extDepth.MinDepth) / extRange;
                if (extNorm < 0.01f) continue;
                float extD = 1.0f / (extNorm + 0.01f);

                int refX = (int)(ex + dx);
                int refY = (int)(by + dy);
                if (refX < 0 || refX >= refW || refY < 0 || refY >= refH) continue;

                float refRaw = refDepthData[refY * refW + refX];
                float refNorm = (refRaw - refDepth.MinDepth) / refRange;
                if (refNorm < 0.01f) continue;
                float refD = 1.0f / (refNorm + 0.01f);

                if (extD > 0.01f && refD > 0.01f)
                    ratios.Add(refD / extD);
            }
        }

        if (ratios.Count < 10)
        {
            Console.WriteLine($"[MultiView] Seam depth: only {ratios.Count} samples, using scale=1.0");
            return 1.0f;
        }

        ratios.Sort();
        // Use median for robustness against outliers
        float scale = ratios[ratios.Count / 2];
        Console.WriteLine($"[MultiView] Seam depth alignment: {ratios.Count} samples, scale={scale:F4} (range: {ratios[0]:F4} to {ratios[^1]:F4})");
        return scale;
    }

    /// <summary>
    /// Generate a multi-view scene using ground truth camera parameters (bypasses SfM).
    /// Used for testing/validation with datasets like TempleRing that have known R/t/K.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithGroundTruthAsync(IReadOnlyList<ImportedImage> images, IReadOnlyList<CameraParams> cameras,
            int subsample = 2, float edgeSharpness = 0.3f)
    {
        if (images.Count != cameras.Count)
            throw new ArgumentException("Image count must match camera count.");

        if (!_depthService.IsReady)
        {
            SetStatus("Loading depth model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        // Log camera positions and compute scene center + depth scale
        var camCentroid = Vector3.Zero;
        for (int i = 0; i < cameras.Count; i++)
        {
            var c = cameras[i];
            camCentroid += c.Position;
            Console.WriteLine($"[MultiView-GT] Cam[{i}] pos=({c.Position.X:F4},{c.Position.Y:F4},{c.Position.Z:F4}) fwd=({c.Forward.X:F3},{c.Forward.Y:F3},{c.Forward.Z:F3}) fx={c.FocalX:F1}");
        }
        camCentroid /= cameras.Count;

        // Estimate the scene center: the point all cameras are looking at.
        // For a ring dataset, the scene center is roughly the centroid of cameras
        // plus the average forward direction scaled by the inter-camera distance.
        // Simpler: the centroid of all cameras IS roughly the scene center for ring captures.
        // The typical depth from camera to scene center is the average distance from each camera to the centroid.
        float avgDist = 0;
        for (int i = 0; i < cameras.Count; i++)
            avgDist += Vector3.Distance(cameras[i].Position, camCentroid);
        avgDist /= cameras.Count;

        // MDE inverted disparity median is roughly 2.0 (for normalized=0.5 → d=1/(0.5+0.01)≈2.0)
        // So depthScale = avgDist / typicalMdeDepth
        float depthScale = avgDist / 2.0f;
        Console.WriteLine($"[MultiView-GT] Camera centroid=({camCentroid.X:F4},{camCentroid.Y:F4},{camCentroid.Z:F4}), avgDist={avgDist:F4}, depthScale={depthScale:F4}");

        // Pass 1: estimate depth + compute per-view scale by projecting scene center into each view
        var viewCounts = new List<(int idx, int count, float viewScale)>();
        int totalSplats = 0;

        for (int i = 0; i < images.Count; i++)
        {
            SetStatus($"Depth estimation: {images[i].FileName} ({i + 1}/{images.Count})...");
            var depthResult = await _depthService.EstimateDepthAsync(images[i]);
            if (depthResult == null) { Console.WriteLine($"[MultiView-GT] Depth failed for {i}"); continue; }

            // Compute per-view depth scale: project scene center into this camera,
            // sample MDE depth there, and scale so MDE depth = actual depth to center.
            float viewScale = depthScale; // fallback
            try
            {
                var cam = cameras[i];
                float[] depthData = await depthResult.RawDepthGpu!.CopyToHostAsync<float>(0, depthResult.RawDepthGpu.Length);
                float range = depthResult.MaxDepth - depthResult.MinDepth;

                // Project scene center into this camera
                var delta = camCentroid - cam.Position;
                float camZ = Vector3.Dot(cam.Forward, delta);

                if (camZ > 0.01f && range > 1e-6f)
                {
                    float camX = Vector3.Dot(cam.Right, delta);
                    float camY = Vector3.Dot(-cam.Up, delta);
                    float px = cam.FocalX * camX / camZ + cam.CenterX;
                    float py = cam.FocalY * camY / camZ + cam.CenterY;
                    int ix = Math.Clamp((int)px, 0, depthResult.Width - 1);
                    int iy = Math.Clamp((int)py, 0, depthResult.Height - 1);

                    float rawMde = depthData[iy * depthResult.Width + ix];
                    float norm = (rawMde - depthResult.MinDepth) / range;
                    if (norm > 0.01f)
                    {
                        float mdeDepth = 1.0f / (norm + 0.01f);
                        viewScale = camZ / mdeDepth;
                        Console.WriteLine($"[MultiView-GT] View {i} depth align: center at pixel ({ix},{iy}), camZ={camZ:F4}, mdeDepth={mdeDepth:F2}, scale={viewScale:F4}");
                    }
                }
            }
            catch { }

            SetStatus($"Generating splats: {images[i].FileName} ({i + 1}/{images.Count})...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                depthResult, images[i], cameras[i], subsample, edgeSharpness, viewScale);

            viewCounts.Add((i, count, viewScale));
            totalSplats += count;
            buf.Dispose();
            depthResult.Dispose();
            Console.WriteLine($"[MultiView-GT] View {i}: {count:N0} splats, scale={viewScale:F4}");
        }

        if (viewCounts.Count == 0 || totalSplats == 0)
        {
            SetStatus("Error: No views produced splats.");
            return null;
        }

        // Pass 2: regenerate + merge
        SetStatus($"Merging {totalSplats:N0} splats from {viewCounts.Count} views...");
        var accelerator = _gpu.WebGPUAccelerator;
        var nativeAccel = accelerator.NativeAccelerator;
        var device = nativeAccel.NativeDevice!;
        var queue = nativeAccel.Queue!;

        var merged = accelerator.Allocate1D<float>(totalSplats * 10);
        var mergedGpuBuf = merged.GetGPUBuffer();

        ulong byteOffset = 0;
        foreach (var (idx, expectedCount, viewScale) in viewCounts)
        {
            SetStatus($"Fusing view {idx + 1}/{images.Count}...");
            var depthResult = await _depthService.EstimateDepthAsync(images[idx]);
            if (depthResult == null) continue;

            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                depthResult, images[idx], cameras[idx], subsample, edgeSharpness, viewScale);
            depthResult.Dispose();

            var srcGpuBuf = buf.GetGPUBuffer();
            ulong byteCount = (ulong)count * 10 * sizeof(float);

            if (srcGpuBuf != null && mergedGpuBuf != null)
            {
                using var encoder = device.CreateCommandEncoder();
                encoder.CopyBufferToBuffer(srcGpuBuf, 0, mergedGpuBuf, byteOffset, byteCount);
                using var cmdBuf = encoder.Finish();
                queue.Submit(new[] { cmdBuf });
            }

            byteOffset += byteCount;
            buf.Dispose();
        }

        int actualTotal = (int)(byteOffset / (10 * sizeof(float)));
        SetStatus($"GT generation complete: {actualTotal:N0} splats from {viewCounts.Count} views.");
        Console.WriteLine($"[MultiView-GT] Total: {actualTotal:N0} splats from {viewCounts.Count} views");

        return (merged, totalSplats);
    }

    /// <summary>
    /// Parse a Middlebury-format camera parameter file (templeR_par.txt, dinoSR_par.txt).
    /// Returns (filename, CameraParams) for each camera line.
    /// Format: filename K[0..8] R[0..8] t[0..2]
    /// </summary>
    public static List<(string filename, CameraParams camera)> ParseMiddleburyParams(string parFileContent, int imageWidth, int imageHeight)
    {
        var results = new List<(string, CameraParams)>();
        var lines = parFileContent.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        // First line is count
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 22) continue;

            string filename = parts[0];
            // K matrix (row-major): parts[1..9]
            float fx = float.Parse(parts[1]);
            float fy = float.Parse(parts[5]);
            float cx = float.Parse(parts[3]);
            float cy = float.Parse(parts[6]);

            // R matrix (row-major): parts[10..18]
            var R = new double[3, 3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R[r, c] = double.Parse(parts[10 + r * 3 + c]);

            // t vector: parts[19..21]
            var t = new double[] {
                double.Parse(parts[19]),
                double.Parse(parts[20]),
                double.Parse(parts[21])
            };

            var cam = new CameraParams
            {
                Width = imageWidth,
                Height = imageHeight,
                FocalX = fx,
                FocalY = fy,
                CenterX = cx,
                CenterY = cy,
            };

            // Use the same SetCameraPose logic: Position = -R^T * t, Forward = R[2,:], Up = -R[1,:]
            cam.Forward = new Vector3((float)R[2, 0], (float)R[2, 1], (float)R[2, 2]);
            cam.Up = new Vector3(-(float)R[1, 0], -(float)R[1, 1], -(float)R[1, 2]);
            cam.Position = new Vector3(
                -((float)(R[0, 0] * t[0] + R[1, 0] * t[1] + R[2, 0] * t[2])),
                -((float)(R[0, 1] * t[0] + R[1, 1] * t[1] + R[2, 1] * t[2])),
                -((float)(R[0, 2] * t[0] + R[1, 2] * t[1] + R[2, 2] * t[2])));

            results.Add((filename, cam));
        }

        return results;
    }

    private void SetStatus(string status)
    {
        Status = status;
        Console.WriteLine($"[MultiView] {status}");
        OnStatusChanged?.Invoke();
    }
}
