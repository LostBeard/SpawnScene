using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Multi-view scene generation: joint DAv3 depths + hybrid poses (SfM → DAv3 extrinsics → fallback).
/// World-space unproject when poses are shared; soft border fade + seam/SfM scale for fusion.
/// </summary>
public class MultiViewGenerationService
{
    private readonly SpawnJSRuntime _js;
    private readonly GpuService _gpu;
    private readonly ImageImportService _importService;
    private readonly SfmReconstructor _sfm;
    private readonly DepthEstimationService _depthService;
    private readonly DepthToGaussianKernel _gaussianKernel;

    public string Status { get; private set; } = "";
    public event Action? OnStatusChanged;

    /// <summary>Camera poses recovered by the last SfM run (index matches input images).</summary>
    public CameraParams?[] SfmCameraPoses => _sfm.CameraPoses;

    /// <summary>
    /// The poses the SfM -> DAv3 -> fallback cascade settled on, one per input image, null where
    /// a view could not be posed. Without these the optimiser cannot touch an unposed capture:
    /// the cascade resolved them internally and then threw them away, so only the TempleRing
    /// path (which has a calibration file) could ever be trained.
    /// </summary>
    public CameraParams?[] LastCameras { get; private set; } = [];

    /// <summary>Where <see cref="LastCameras"/> came from: sfm, dav3 or fallback.</summary>
    public string LastPoseSource { get; private set; } = "none";

    /// <summary>
    /// Which pose source to try first.
    ///
    /// The cascade preferred SfM, and that is arguably backwards when the depth comes from a
    /// JOINT multi-view model. DAv3's joint inference puts every view's depth in ONE shared
    /// frame - that is the whole point of it - and it reports the extrinsics for that frame.
    /// Taking the depths from DAv3 and the cameras from SfM means the geometry and the
    /// cameras live in different coordinate systems at different scales, reconciled by a
    /// median depth ratio. On Bathroom that produced a cloud that rendered as soup.
    ///
    /// "dav3" keeps depth and poses in the same frame. "sfm" is the old behaviour. "auto"
    /// prefers SfM and falls back, which is what shipped.
    /// </summary>
    public string PosePreference { get; set; } = "auto";

    /// <summary>
    /// Pose every view by running the joint depth model in chunks that share anchor views, rather
    /// than posing only the handful one forward pass accepts. See
    /// <see cref="MultiViewChunkPlan"/> for why this removes the geometry-versus-supervision fork
    /// instead of picking a side of it.
    /// </summary>
    public bool ChunkedPoses { get; set; } = true;

    /// <summary>
    /// Views the chunked pass shares between every chunk. Three is the minimum that determines a
    /// similarity; more costs capacity for new views per chunk (and so more chunks) but makes the
    /// fold more tolerant of one anchor coming back badly posed.
    /// </summary>
    public int ChunkAnchorCount { get; set; } = MultiViewChunkPlan.MinAnchors;

    /// <summary>How many views the last chunked run actually put in one forward pass.</summary>
    public int LastChunkSize { get; private set; }

    public MultiViewGenerationService(
        SpawnJSRuntime js,
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

        // Ensure depth model is loaded before checking model type
        if (!_depthService.IsReady)
        {
            SetStatus("Loading depth model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

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

        var merged = accelerator.Allocate1D<float>(totalSplats * SplatFormat.Floats);
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
            ulong byteCount = (ulong)count * SplatFormat.Floats * sizeof(float);

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
    /// Hybrid multi-view: joint DAv3 depths + pose from SfM (if reliable) else DAv3 extrinsics else camera-local fallback.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithDav3MultiViewAsync(IReadOnlyList<ImportedImage> images, int subsample, float edgeSharpness)
    {
        // Chunked poses supersede the single-pass cascade whenever there are more views than one
        // forward accepts: same model, same unproject, every view posed instead of six.
        if (ChunkedPoses
            && !string.Equals(PosePreference, "sfm", StringComparison.OrdinalIgnoreCase)
            && images.Count > DepthEstimationService.MaxMultiViewImages)
        {
            var chunked = await GenerateWithChunkedDav3Async(images, subsample, edgeSharpness);
            if (chunked != null) return chunked;
            Console.WriteLine(
                "[MultiView] chunked poses produced nothing; falling back to the single-pass cascade.");
        }

        if (!_depthService.IsReady)
        {
            SetStatus("Loading DAv3 model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        int n = Math.Min(images.Count, DepthEstimationService.MaxMultiViewImages);

        // Spread the depth views across the sequence instead of taking the first n. Consecutive
        // handheld frames are near-parallel, which is the worst case for both the joint depth
        // model and for SfM - the TempleRing path learned this already ("first 4 consecutive
        // frames are near-identical viewpoints -> floating near-duplicates").
        var viewIdx = new int[n];
        for (int i = 0; i < n; i++)
            viewIdx[i] = n == 1 ? 0 : (int)Math.Round(i * (images.Count - 1) / (double)(n - 1));
        var viewImages = viewIdx.Select(i => images[i]).ToList();

        SetStatus($"Running joint DAv3 multi-view ({n} images)...");
        var mvResult = await _depthService.EstimateDepthMultiViewAsync(viewImages);
        if (mvResult == null || mvResult.DepthResults.Count == 0)
        {
            SetStatus("Error: DAv3 multi-view inference failed.");
            return null;
        }

        // ── Pose selection: SfM → DAv3 extrinsics → fallback ──
        string poseSource = "fallback";
        var cameras = new CameraParams?[n];           // the depth subset
        var allPoses = new CameraParams?[images.Count];  // every image SfM could register

        bool preferDav3 = string.Equals(PosePreference, "dav3", StringComparison.OrdinalIgnoreCase);
        if (preferDav3)
            Console.WriteLine(
                "[MultiView] Pose preference=dav3: skipping SfM so the depths and the cameras " +
                "stay in the SAME frame (joint DAv3 inference already shares one).");

        // Try SfM on feature matches (often fails on near-parallel phone snaps — that is expected).
        try
        {
            if (preferDav3) throw new OperationCanceledException("dav3 preferred");

            SetStatus($"Trying SfM poses ({images.Count} images)...");
            _importService.Clear();

            // EVERY image, not just the depth subset. Depth initialisation and photometric
            // supervision are different budgets: the depth model takes 6 views, but every view
            // SfM can register is another photograph the optimiser can be held to. Running SfM
            // on the subset threw away 29 of Bathroom's 35 frames and left 5 supervision views
            // against 197k Gaussians, which is why held-out quality went backwards.
            await _importService.ImportFromImagesAsync(images);
            if (_importService.MatchedPairs.Count > 0)
            {
                await _sfm.ReconstructAsync();
                for (int i = 0; i < images.Count && i < _sfm.CameraPoses.Length; i++)
                    allPoses[i] = _sfm.CameraPoses[i];

                int posed = allPoses.Count(c => c != null);
                int posedInSubset = 0;
                for (int i = 0; i < n; i++)
                {
                    cameras[i] = allPoses[viewIdx[i]];
                    if (cameras[i] != null) posedInSubset++;
                }

                // The depth subset still has to be posed - it is what the geometry is built
                // from. Extra posed views only add supervision.
                if (posedInSubset >= 2 && _sfm.Points3D.Count >= 10)
                {
                    poseSource = "sfm";
                    Console.WriteLine(
                        $"[MultiView] Pose source=sfm cameras={posed}/{images.Count} " +
                        $"({posedInSubset}/{n} of the depth views) pts={_sfm.Points3D.Count}");
                }
                else
                {
                    System.Array.Clear(cameras);
                    System.Array.Clear(allPoses);
                    Console.WriteLine($"[MultiView] SfM weak (cams={posedInSubset}/{n} of the depth views, pts={_sfm.Points3D.Count}) — trying DAv3 extrinsics");
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Preference, not a failure.
        }
        catch (Exception ex)
        {
            System.Array.Clear(cameras);
            System.Array.Clear(allPoses);
            Console.WriteLine($"[MultiView] SfM failed: {ex.Message} — trying DAv3 extrinsics");
        }

        if (poseSource != "sfm" && mvResult.Extrinsics != null && mvResult.Extrinsics.Length >= n)
        {
            bool sane = true;
            for (int i = 0; i < n; i++)
            {
                var ext = mvResult.Extrinsics[i];
                if (ext == null || ext.Length < 12 || !IsSaneExtrinsics(ext))
                { sane = false; break; }
            }
            if (sane)
            {
                for (int i = 0; i < n; i++)
                {
                    var ext = mvResult.Extrinsics[i];
                    var cam = viewImages[i].EstimatedCamera
                        ?? CameraParams.CreateDefault(viewImages[i].Width, viewImages[i].Height);
                    ApplyExtrinsicsToCamera(cam, ext);
                    if (mvResult.Intrinsics != null && i < mvResult.Intrinsics.Length
                        && mvResult.Intrinsics[i] is { Length: >= 9 } K)
                    {
                        cam.FocalX = K[0];
                        cam.FocalY = K[4];
                        cam.CenterX = K[2];
                        cam.CenterY = K[5];
                    }
                    cameras[i] = cam;
                }
                poseSource = "dav3";
                // DAv3 only sees the depth subset, so only those views get poses.
                for (int i = 0; i < n; i++) allPoses[viewIdx[i]] = cameras[i];
                Console.WriteLine($"[MultiView] Pose source=dav3 extrinsics ({n} views)");
            }
        }

        if (poseSource == "fallback")
            Console.WriteLine("[MultiView] Pose source=fallback (camera-local / no shared world)");

        // Per-view depth scale: center-projection style when we have shared poses (TempleRing idea).
        float[] depthScales = Enumerable.Repeat(1.0f, n).ToArray();
        if (poseSource != "fallback")
        {
            depthScales = ComputeHybridDepthScales(cameras, mvResult.DepthResults, viewImages);

            // Only when SfM supplied the poses. If DAv3 did, its depths are already in that
            // same frame and rescaling them to a sparse SfM cloud would reintroduce exactly the
            // frame mismatch this is meant to avoid.
            if (poseSource == "sfm" && _sfm.Points3D.Count >= 10)
            {
                try
                {
                    var (perView, global, support) = await FitSfmDepthScalesAsync(
                        cameras, mvResult.DepthResults, _sfm.Points3D);

                    int fitted = 0;
                    for (int i = 0; i < n; i++)
                    {
                        float sc = perView[i];
                        if (!(sc > 0.05f && sc < 50f)) sc = global;
                        if (!(sc > 0.05f && sc < 50f)) continue;
                        depthScales[i] *= sc;
                        if (support[i] >= 12) fitted++;
                    }
                    Console.WriteLine(
                        $"[MultiView] SfM depth scale: {fitted}/{n} views fitted individually, " +
                        $"global={global:F4}, per-view=[" +
                        string.Join(", ", Enumerable.Range(0, n)
                            .Select(i => $"{perView[i]:F3}({support[i]})")) + "] (pts=" +
                        $"{_sfm.Points3D.Count})");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MultiView] SfM sparse scale skipped: {ex.Message}");
                }
            }
            // Joint DAv3 depths already share one relative frame — do NOT apply per-view seam
            // scales (that separates clouds into floating fragments). Global baseScale is enough.
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        var device = accelerator.NativeAccelerator.NativeDevice!;
        var queue = accelerator.NativeAccelerator.Queue!;

        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0;
        LastCameras = allPoses;
        LastPoseSource = poseSource;

        bool useWorld = poseSource != "fallback";
        int nonRefIn = 0, nonRefKept = 0;


        int unposedSkipped = 0;
        for (int i = 0; i < mvResult.DepthResults.Count && i < n; i++)
        {
            // A view with no recovered pose must not contribute to a WORLD-SPACE merge.
            //
            // The fallback here was viewImages[i].EstimatedCamera - a default camera invented
            // from the image dimensions - which is fine when every view is camera-local and
            // catastrophic when the others are in a shared world frame: the cloud is placed
            // by guesswork and merged in as though it were measured. On Bathroom, SfM posed
            // only 4 of the 6 depth views and the other two contributed 393,216 splats, 67% of
            // the scene, from fabricated poses. That is most of why it rendered as soup.
            //
            // Dropping them loses coverage. Inventing them loses the reconstruction.
            if (useWorld && cameras[i] == null)
            {
                unposedSkipped++;
                Console.WriteLine(
                    $"[MultiView] View {i} ({viewImages[i].FileName}) has no pose - skipped. " +
                    "Unprojecting it would place a full cloud from a camera that was guessed.");
                continue;
            }

            SetStatus($"Generating splats: {viewImages[i].FileName} ({i + 1}/{n}, pose={poseSource})...");
            var depth = mvResult.DepthResults[i];
            var cam = cameras[i] ?? viewImages[i].EstimatedCamera;

            (MemoryBuffer1D<float, Stride1D.Dense> buf, int count) result;
            if (useWorld && cameras[i] != null)
            {
                result = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                    depth, viewImages[i], cameras[i]!, subsample, edgeSharpness, depthScales[i]);

                // Fuse non-ref views against view 0 (same shared-frame depths).
                if (i > 0 && result.count > 0 && cameras[0] != null)
                {
                    nonRefIn += result.count;
                    result = await _gaussianKernel.FuseConsistencyVsRefAsync(
                        result.buf, result.count, mvResult.DepthResults[0], cameras[0]!,
                        depthScales[0], relThresh: 0.06f);
                    nonRefKept += result.count;
                }
            }
            else
            {
                result = await _gaussianKernel.GeneratePackedGpuBufferAsync(
                    depth, viewImages[i], subsample, edgeSharpness, cam);
            }

            viewResults.Add(result);
            totalSplats += result.count;
            Console.WriteLine($"[MultiView] View {i}: {result.count:N0} splats scale={depthScales[i]:F3}");
        }
        if (nonRefIn > 0)
            Console.WriteLine($"[MultiView] Consistency: non-ref kept {nonRefKept:N0}/{nonRefIn:N0} ({(float)nonRefKept / nonRefIn:P0})");

        if (totalSplats == 0)
        {
            foreach (var (buf, _) in viewResults) buf.Dispose();
            mvResult.Dispose();
            SetStatus("Error: No splats generated.");
            return null;
        }

        SetStatus($"Merging {totalSplats:N0} splats from {viewResults.Count} views (pose={poseSource})...");
        var merged = accelerator.Allocate1D<float>(totalSplats * SplatFormat.Floats);
        long offsetBytes = 0;
        int actualTotal = 0;
        foreach (var (buf, count) in viewResults)
        {
            if (count <= 0) { buf.Dispose(); continue; }
            ulong byteCount = (ulong)count * SplatFormat.Floats * sizeof(float);
            using (var encoder = device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(
                    buf.GetGPUBuffer()!, 0,
                    merged.GetGPUBuffer()!, (ulong)offsetBytes,
                    byteCount);
                queue.Submit(new[] { encoder.Finish() });
            }
            offsetBytes += (long)byteCount;
            actualTotal += count;
            buf.Dispose();
        }
        await accelerator.SynchronizeAsync();
        mvResult.Dispose();

        SetStatus($"Multi-view complete: {actualTotal:N0} splats from {viewResults.Count} views (pose={poseSource}).");
        Console.WriteLine($"[MultiView] Total: {actualTotal:N0} splats pose={poseSource}");
        return (merged, actualTotal);
    }

    /// <summary>
    /// Generate from every view the chunked pose pass could place, all in one world frame.
    ///
    /// The difference from <see cref="GenerateWithDav3MultiViewAsync"/> is coverage, not method:
    /// the same joint depth model, the same world-space unproject, but run over all 35 of
    /// Bathroom's frames instead of the 6 one forward pass accepts.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithChunkedDav3Async(IReadOnlyList<ImportedImage> images, int subsample, float edgeSharpness)
    {
        using var poses = await PoseAllViewsChunkedAsync(images);
        if (poses == null || poses.PosedCount == 0)
        {
            SetStatus("Error: the chunked pose pass placed no views.");
            return null;
        }

        var posed = new List<int>();
        for (int i = 0; i < images.Count; i++)
            if (poses.Cameras[i] != null && poses.Depths[i] != null) posed.Add(i);

        if (posed.Count == 0)
        {
            SetStatus("Error: posed views have no depth maps.");
            return null;
        }

        // Depth scale composes in one order and only one. A view's raw depth is in ITS CHUNK's
        // frame, so it takes the fold's scale first; only then are the views comparable enough to
        // share one base scale mapping relative depth onto the camera rig. Computing the base
        // scale over raw values from different chunk frames would average numbers that do not
        // share a unit.
        var scales = new float[images.Count];
        {
            var placedCams = posed.Select(i => poses.Cameras[i]!).ToList();
            var centroid = new Vector3(
                placedCams.Average(c => c.Position.X),
                placedCams.Average(c => c.Position.Y),
                placedCams.Average(c => c.Position.Z));
            float avgDist = placedCams.Average(c => Vector3.Distance(c.Position, centroid));
            if (avgDist < 1e-3f) avgDist = 1f;

            double midInFrame = posed.Average(i =>
                (poses.Depths[i]!.MinDepth + poses.Depths[i]!.MaxDepth) * 0.5f * poses.FrameScales[i]);
            if (midInFrame < 1e-4) midInFrame = 1.0;

            float baseScale = (float)(avgDist / midInFrame);
            foreach (int i in posed) scales[i] = poses.FrameScales[i] * baseScale;

            Console.WriteLine(
                $"[MultiView] depth scale: base={baseScale:F4} (avg camera distance {avgDist:F4}, " +
                $"mid depth in frame {midInFrame:F4}); per-view span " +
                $"{posed.Min(i => scales[i]):F4} to {posed.Max(i => scales[i]):F4}");
        }

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        var device = accelerator.NativeAccelerator.NativeDevice!;
        var queue = accelerator.NativeAccelerator.Queue!;

        // Coverage now scales with the capture, so the splat budget has to as well: at subsample 2
        // Bathroom's 35 views of 768x1024 would emit 6.9M splats where 6 views emitted 1.2M.
        int effectiveSub = ChooseSubsample(subsample, posed.Select(i => poses.Depths[i]!).ToList());

        int refView = posed[0];
        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0, nonRefIn = 0, nonRefKept = 0;

        foreach (int i in posed)
        {
            SetStatus($"Generating splats: {images[i].FileName} ({viewResults.Count + 1}/{posed.Count})...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                poses.Depths[i]!, images[i], poses.Cameras[i]!, effectiveSub, edgeSharpness, scales[i]);

            if (i != refView && count > 0)
            {
                nonRefIn += count;
                (buf, count) = await _gaussianKernel.FuseConsistencyVsRefAsync(
                    buf, count, poses.Depths[refView]!, poses.Cameras[refView]!,
                    scales[refView], relThresh: 0.06f);
                nonRefKept += count;
            }

            viewResults.Add((buf, count));
            totalSplats += count;
        }

        if (nonRefIn > 0)
            Console.WriteLine(
                $"[MultiView] consistency: non-ref kept {nonRefKept:N0}/{nonRefIn:N0} " +
                $"({(float)nonRefKept / nonRefIn:P0})");

        if (totalSplats == 0)
        {
            foreach (var (buf, _) in viewResults) buf.Dispose();
            SetStatus("Error: No splats generated.");
            return null;
        }

        SetStatus($"Merging {totalSplats:N0} splats from {viewResults.Count} views...");
        var merged = accelerator.Allocate1D<float>((long)totalSplats * SplatFormat.Floats);
        long offsetBytes = 0;
        int actualTotal = 0;
        foreach (var (buf, count) in viewResults)
        {
            if (count <= 0) { buf.Dispose(); continue; }
            ulong byteCount = (ulong)count * SplatFormat.Floats * sizeof(float);
            using (var encoder = device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(
                    buf.GetGPUBuffer()!, 0, merged.GetGPUBuffer()!, (ulong)offsetBytes, byteCount);
                queue.Submit(new[] { encoder.Finish() });
            }
            offsetBytes += (long)byteCount;
            actualTotal += count;
            buf.Dispose();
        }
        await accelerator.SynchronizeAsync();

        LastCameras = poses.Cameras;
        LastPoseSource = "dav3-chunked";

        Console.WriteLine(
            $"[MultiView] Total: {actualTotal:N0} splats from {posed.Count}/{images.Count} views " +
            $"pose=dav3-chunked N={poses.ChunkSize} subsample={effectiveSub}");
        SetStatus($"Multi-view complete: {actualTotal:N0} splats from {posed.Count} views.");
        return (merged, actualTotal);
    }

    /// <summary>
    /// Target splats for one initialisation. A 3DGS optimiser densifies from an initialisation;
    /// emitting a splat per pixel per view is not a better start, it is a bigger one, and the
    /// key-indexed training buffers are bounded by a single storage binding rather than by VRAM
    /// (see <c>SplatTrainerGpu</c>). Coarser sampling across MORE views beats dense sampling of
    /// a few, because the extra views are the new information.
    /// </summary>
    public int SplatBudget { get; set; } = 1_500_000;

    private int ChooseSubsample(int requested, IReadOnlyList<DepthResult> depths)
    {
        int sub = Math.Max(1, requested);
        long PixelsAt(int s) => depths.Sum(d => (long)(d.Width / s) * (d.Height / s));

        long at = PixelsAt(sub);
        if (at <= SplatBudget) return sub;

        int chosen = sub;
        while (chosen < 16 && PixelsAt(chosen) > SplatBudget) chosen++;
        Console.WriteLine(
            $"[MultiView] subsample {sub} -> {chosen}: {depths.Count} views would emit " +
            $"{at:N0} splats against a {SplatBudget:N0} budget, now {PixelsAt(chosen):N0}");
        return chosen;
    }

    /// <summary>
    /// What a chunked pose pass recovered, indexed by GLOBAL image index. A null camera means
    /// that view was never placed in the world frame and must not contribute to the merge.
    /// </summary>
    private sealed class ChunkedPoseResult : IDisposable
    {
        public required CameraParams?[] Cameras { get; init; }
        public required DepthResult?[] Depths { get; init; }

        /// <summary>Factor carrying this view's raw depth into the reference frame.</summary>
        public required float[] FrameScales { get; init; }

        /// <summary>Views per forward pass this device actually sustained.</summary>
        public int ChunkSize { get; set; }
        public int ChunkCount { get; set; }
        public int ChunksRejected { get; set; }
        public int PosedCount => Cameras.Count(c => c != null);

        public void Dispose()
        {
            foreach (var d in Depths) d?.Dispose();
        }
    }

    /// <summary>
    /// Pose EVERY view in one world frame by running the joint depth model in chunks that share
    /// anchor views, folding each chunk in with a similarity fitted to those anchors.
    ///
    /// See <see cref="MultiViewChunkPlan"/> for why this is the answer to the fork commit eeedfff
    /// left open, rather than a third option alongside it.
    ///
    /// The chunk size is PROVEN on the device by the first pass rather than assumed: if that
    /// forward fails it is retried smaller, and only the size that actually ran is used for the
    /// rest. Every view still gets posed either way, just in more chunks - which is the whole
    /// reason the cap stopped being a coverage decision.
    /// </summary>
    private async Task<ChunkedPoseResult?> PoseAllViewsChunkedAsync(IReadOnlyList<ImportedImage> images)
    {
        if (!_depthService.IsReady)
        {
            SetStatus("Loading DAv3 model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        var shapes = images.Select(im => (im.Width, im.Height)).ToArray();
        int anchors = Math.Max(MultiViewChunkPlan.MinAnchors, ChunkAnchorCount);
        int chunkSize = Math.Max(anchors + 1, DepthEstimationService.MaxMultiViewImages);

        var result = new ChunkedPoseResult
        {
            Cameras = new CameraParams?[images.Count],
            Depths = new DepthResult?[images.Count],
            FrameScales = Enumerable.Repeat(1f, images.Count).ToArray(),
        };

        IReadOnlyList<MultiViewChunkPlan.MultiViewShapeGroup>? groups = null;
        DepthEstimationService.MultiViewDepthResult? firstRun = null;

        while (chunkSize > anchors)
        {
            groups = MultiViewChunkPlan.PlanByShape(shapes, chunkSize, anchors);
            SetStatus($"Joint depth pass 1 of {groups[0].Chunks.Count} (N={chunkSize})...");
            firstRun = await TryRunChunkAsync(images, groups[0].Chunks[0], chunkSize);
            if (firstRun != null) break;

            int next = chunkSize - 1;
            Console.WriteLine(
                $"[MultiView] the joint forward did not run at N={chunkSize}; retrying at N={next}. " +
                "That cap is a starting point, not a measured limit - every view still gets " +
                "posed, in more chunks.");
            chunkSize = next;
        }

        if (firstRun == null || groups == null)
        {
            result.Dispose();
            Console.WriteLine(
                $"[MultiView] no joint depth pass ran, down to N={anchors + 1}. " +
                "Nothing can be posed in a shared frame.");
            return null;
        }

        var reference = groups[0];
        result.ChunkSize = chunkSize;
        result.ChunkCount = reference.Chunks.Count;
        Console.WriteLine(
            $"[MultiView] chunked poses: {reference.ViewCount} views at " +
            $"{reference.Width}x{reference.Height} in {reference.Chunks.Count} pass(es) of N={chunkSize}, " +
            $"{anchors} shared anchors [{string.Join(",", reference.Chunks[0].Anchors.ToArray())}]");

        // Chunk 0 defines the world frame: its own output, untransformed.
        AdoptChunk(result, reference.Chunks[0],
            CamerasFromRun(images, reference.Chunks[0], firstRun), firstRun,
            Similarity3.Identity, adoptAnchors: true);
        firstRun.Dispose();

        var placed = BuildPlacedLookup(result);
        if (placed.Count < MultiViewChunkPlan.MinAnchors && reference.Chunks.Count > 1)
        {
            Console.WriteLine(
                $"[MultiView] the reference pass posed only {placed.Count} view(s), fewer than the " +
                $"{MultiViewChunkPlan.MinAnchors} anchors a fold needs. Later chunks cannot be placed.");
        }

        for (int ci = 1; ci < reference.Chunks.Count; ci++)
        {
            var chunk = reference.Chunks[ci];
            SetStatus($"Joint depth pass {ci + 1} of {reference.Chunks.Count} (N={chunkSize})...");

            var run = await TryRunChunkAsync(images, chunk, chunkSize);
            if (run == null)
            {
                // One failed pass costs its own new views, not the run. Every other chunk holds
                // the same anchors and folds in independently.
                result.ChunksRejected++;
                Console.WriteLine(
                    $"[MultiView] chunk {ci} did not run; its " +
                    $"{chunk.NewViews.Length} view(s) stay unposed.");
                continue;
            }

            var cams = CamerasFromRun(images, chunk, run);
            bool fitted = MultiViewChunkPlan.TryFitChunkToReference(
                chunk, cams, placed, out var sim, out float rms, out int used, out float spread);

            // Always report residual AGAINST the spread. The threshold is otherwise a judgement
            // call nobody can check, and at MinAnchors the fit is over-determined by only two -
            // so a non-zero residual is not rounding, it says the model gave a differently SHAPED
            // anchor triangle in this pass than in the reference one.
            string fitLine =
                $"anchors {used}/{chunk.AnchorCount}, residual {rms:F4} on a spread of {spread:F4} " +
                $"({(spread > 0 ? rms / spread : float.NaN):P1} of it, limit " +
                $"{MultiViewChunkPlan.MaxAnchorRmsFraction:P0})";

            LogAnchorTriangle(ci, chunk, cams, placed);

            if (!fitted)
            {
                result.ChunksRejected++;
                Console.WriteLine(
                    $"[MultiView] chunk {ci} REJECTED: {fitLine}. Placing it anyway would put a " +
                    "full cloud in the wrong part of the scene looking measured.");
                run.Dispose();
                continue;
            }

            Console.WriteLine(
                $"[MultiView] chunk {ci} folded: {fitLine}, depth scale {sim.Scale:F4}, " +
                $"{chunk.NewViews.Length} new view(s)");
            AdoptChunk(result, chunk, cams, run, sim, adoptAnchors: false);
            run.Dispose();
        }

        foreach (var group in groups.Skip(1))
            Console.WriteLine(
                $"[MultiView] {group.ViewCount} view(s) at {group.Width}x{group.Height} are a " +
                "different shape from the reference group. The joint pass emits every view at its " +
                "FIRST view's resolution, so they cannot share a pass, and sharing no anchors they " +
                "cannot be folded in either. Skipped rather than placed.");

        LastChunkSize = chunkSize;
        Console.WriteLine(
            $"[MultiView] chunked poses done: {result.PosedCount}/{images.Count} views posed in one " +
            $"frame, {result.ChunksRejected} chunk(s) rejected");
        return result;
    }

    /// <summary>One joint forward, returning null rather than throwing when the device refuses it.</summary>
    private async Task<DepthEstimationService.MultiViewDepthResult?> TryRunChunkAsync(
        IReadOnlyList<ImportedImage> images, MultiViewChunk chunk, int chunkSize)
    {
        var views = chunk.Views.Select(v => images[v]).ToList();
        try
        {
            var run = await _depthService.EstimateDepthMultiViewAsync(views, maxViews: chunkSize);
            if (run == null || run.DepthResults.Count < views.Count)
            {
                Console.WriteLine(
                    $"[MultiView] joint forward returned {run?.DepthResults.Count ?? 0} depth view(s) " +
                    $"for {views.Count} image(s)");
                run?.Dispose();
                return null;
            }
            return run;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MultiView] joint forward failed at N={views.Count}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Cameras for one chunk's slots, in that chunk's own frame. Each is a FRESH object: an
    /// anchor image appears in every chunk, and writing its pose back into the shared
    /// <c>ImportedImage.EstimatedCamera</c> would let one pass overwrite another's answer.
    /// </summary>
    private static CameraParams?[] CamerasFromRun(
        IReadOnlyList<ImportedImage> images, MultiViewChunk chunk, DepthEstimationService.MultiViewDepthResult run)
    {
        var cams = new CameraParams?[chunk.Views.Length];
        if (run.Extrinsics == null) return cams;

        for (int slot = 0; slot < chunk.Views.Length && slot < run.Extrinsics.Length; slot++)
        {
            var ext = run.Extrinsics[slot];
            if (ext == null || ext.Length < 12 || !IsSaneExtrinsics(ext)) continue;

            var img = images[chunk.Views[slot]];
            var seed = img.EstimatedCamera;
            var cam = CameraParams.CreateDefault(img.Width, img.Height);
            if (seed != null)
            {
                cam.FocalX = seed.FocalX; cam.FocalY = seed.FocalY;
                cam.CenterX = seed.CenterX; cam.CenterY = seed.CenterY;
            }
            ApplyExtrinsicsToCamera(cam, ext);

            if (run.Intrinsics != null && slot < run.Intrinsics.Length
                && run.Intrinsics[slot] is { Length: >= 9 } K)
            {
                cam.FocalX = K[0];
                cam.FocalY = K[4];
                cam.CenterX = K[2];
                cam.CenterY = K[5];
            }
            cams[slot] = cam;
        }
        return cams;
    }

    /// <summary>
    /// Move a chunk's posed views into the world frame, taking ownership of their depth maps.
    /// A view already placed by an earlier chunk is left alone, so anchors keep the reference
    /// pass's answer rather than the last one to run.
    /// </summary>
    private static void AdoptChunk(
        ChunkedPoseResult into, MultiViewChunk chunk, CameraParams?[] cams,
        DepthEstimationService.MultiViewDepthResult run, Similarity3 sim, bool adoptAnchors)
    {
        int from = adoptAnchors ? 0 : chunk.AnchorCount;
        for (int slot = from; slot < chunk.Views.Length; slot++)
        {
            int g = chunk.Views[slot];
            if (cams[slot] == null || into.Cameras[g] != null) continue;

            sim.ApplyToCamera(cams[slot]!);
            into.Cameras[g] = cams[slot];
            into.FrameScales[g] = sim.Scale;

            if (slot < run.DepthResults.Count)
            {
                into.Depths[g] = run.DepthResults[slot];
                run.DepthResults[slot] = null!;   // ownership moved; the run no longer disposes it
            }
        }
    }

    /// <summary>
    /// The anchor triangle in this pass versus in the reference pass, as pairwise distances.
    ///
    /// A similarity preserves SHAPE, so it can absorb any difference in the triangle's size,
    /// position or orientation but none in its proportions. Printing the distance RATIOS
    /// separates the two things a residual conflates: a fold that is merely rescaling (all
    /// ratios equal, one number) from a model that reported genuinely different relative
    /// geometry for the same three physical cameras (ratios that disagree with each other).
    /// The second cannot be fixed by any transform, and knowing which one we have decides
    /// whether the threshold is wrong or the approach is.
    /// </summary>
    private static void LogAnchorTriangle(
        int chunkIndex, MultiViewChunk chunk, CameraParams?[] cams,
        IReadOnlyDictionary<int, CameraParams> reference)
    {
        var parts = new List<string>();
        var ratios = new List<float>();

        for (int a = 0; a < chunk.AnchorCount; a++)
            for (int b = a + 1; b < chunk.AnchorCount; b++)
            {
                if (cams[a] == null || cams[b] == null) continue;
                if (!reference.TryGetValue(chunk.Views[a], out var refA)) continue;
                if (!reference.TryGetValue(chunk.Views[b], out var refB)) continue;

                float here = Vector3.Distance(cams[a]!.Position, cams[b]!.Position);
                float there = Vector3.Distance(refA.Position, refB.Position);
                if (!(there > 1e-6f)) continue;

                float ratio = here / there;
                ratios.Add(ratio);
                parts.Add($"{chunk.Views[a]}-{chunk.Views[b]} {here:F4}/{there:F4}={ratio:F3}");
            }

        if (ratios.Count < 2) return;
        float spreadPct = (ratios.Max() - ratios.Min()) / ratios.Average();
        Console.WriteLine(
            $"[MultiView]   chunk {chunkIndex} anchor triangle: {string.Join("  ", parts)} " +
            $"-> ratios disagree by {spreadPct:P1} " +
            (spreadPct < 0.02f
                ? "(a pure rescale; a similarity absorbs this)"
                : "(a SHAPE difference; no similarity can absorb this)"));
    }

    private static Dictionary<int, CameraParams> BuildPlacedLookup(ChunkedPoseResult result)
    {
        var placed = new Dictionary<int, CameraParams>();
        for (int i = 0; i < result.Cameras.Length; i++)
            if (result.Cameras[i] is { } cam) placed[i] = cam;
        return placed;
    }

    private static bool IsSaneExtrinsics(float[] ext)
    {
        // Reject all-zero / NaN; require a roughly unit-ish rotation row.
        float row0 = MathF.Sqrt(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]);
        float row2 = MathF.Sqrt(ext[8] * ext[8] + ext[9] * ext[9] + ext[10] * ext[10]);
        return row0 > 0.5f && row0 < 1.5f && row2 > 0.5f && row2 < 1.5f;
    }

    private static void ApplyExtrinsicsToCamera(CameraParams cam, float[] ext)
    {
        // ext = row-major 3×4 [R|t]
        cam.Forward = new Vector3(ext[8], ext[9], ext[10]);
        cam.Up = new Vector3(-ext[4], -ext[5], -ext[6]);
        cam.Position = new Vector3(
            -(ext[0] * ext[3] + ext[4] * ext[7] + ext[8] * ext[11]),
            -(ext[1] * ext[3] + ext[5] * ext[7] + ext[9] * ext[11]),
            -(ext[2] * ext[3] + ext[6] * ext[7] + ext[10] * ext[11]));
    }

    /// <summary>
    /// TempleRing-style depth scale: place a unit depth at a typical scene distance from camera centroid.
    /// </summary>
    private static float[] ComputeHybridDepthScales(
        CameraParams?[] cameras, List<DepthResult> depths, IReadOnlyList<ImportedImage> images)
    {
        var scales = new float[depths.Count];
        System.Array.Fill(scales, 1.0f);

        var posed = cameras.Where(c => c != null).Select(c => c!).ToList();
        if (posed.Count < 1) return scales;

        var centroid = new Vector3(
            posed.Average(c => c.Position.X),
            posed.Average(c => c.Position.Y),
            posed.Average(c => c.Position.Z));
        float avgDist = posed.Average(c => Vector3.Distance(c.Position, centroid));
        if (avgDist < 1e-3f) avgDist = 1.0f;
        // Relative MDE mid-depth ≈ mid of min/max — scale so that maps into avg camera distance.
        float midRaw = depths.Average(d => (d.MinDepth + d.MaxDepth) * 0.5f);
        if (midRaw < 1e-4f) midRaw = 1.0f;
        float baseScale = avgDist / midRaw;
        for (int i = 0; i < scales.Length; i++)
            scales[i] = baseScale;
        Console.WriteLine($"[MultiView] Hybrid depthScale base={baseScale:F4} (avgCamDist={avgDist:F4}, midRaw={midRaw:F4})");
        return scales;
    }

    /// <summary>
    /// Fit global MDE→metric scale from SfM sparse points: median(Z_cam / inv(normalized MDE)).
    /// </summary>
    /// <summary>
    /// Metric scale for each view's monocular depth, from the SfM points it can see.
    ///
    /// Monocular depth is only defined up to scale, and that scale is NOT shared between views -
    /// the network sees each photograph independently. Pooling every camera's ratios into one
    /// median, which is what this used to return, forces one number onto views that genuinely
    /// disagree, and the result is a union of shells that do not line up. On Bathroom the
    /// cross-view consistency screen then threw two of six views away entirely and kept 2% of a
    /// third, and what survived rendered as soup.
    ///
    /// Each view is now fitted against the SfM points IT can see. Views with too few points
    /// fall back to the pooled median, which is still better than nothing and is reported as a
    /// fallback rather than passed off as a fit.
    ///
    /// Returns one scale per camera; 1.0 where nothing could be fitted.
    /// </summary>
    private static async Task<(float[] PerView, float Global, int[] Support)> FitSfmDepthScalesAsync(
        CameraParams?[] cameras, List<DepthResult> depths, List<ReconstructedPoint> points)
    {
        var perViewRatios = new List<float>[cameras.Length];
        for (int i = 0; i < cameras.Length; i++) perViewRatios[i] = new List<float>();

        var ratios = new List<float>();
        int maxPts = Math.Min(points.Count, 4000);

        for (int ci = 0; ci < cameras.Length && ci < depths.Count; ci++)
        {
            var cam = cameras[ci];
            var depth = depths[ci];
            if (cam == null || depth.RawDepthGpu == null) continue;

            float[] host;
            try { host = await depth.RawDepthGpu.CopyToHostAsync<float>(0, depth.RawDepthGpu.Length); }
            catch { continue; }

            int w = depth.Width, h = depth.Height;
            var right = Vector3.Normalize(Vector3.Cross(cam.Forward, cam.Up));
            var up = Vector3.Normalize(Vector3.Cross(right, cam.Forward));

            for (int pi = 0; pi < maxPts; pi++)
            {
                var world = points[pi].Position;
                var delta = world - cam.Position;
                float zCam = Vector3.Dot(delta, cam.Forward);
                if (zCam < 0.05f) continue;

                float xCam = Vector3.Dot(delta, right);
                float yCam = Vector3.Dot(delta, -up); // OpenCV Y-down
                // Match UnprojectWorldSpaceKernel: u = cx + fx*xCam/z, v = cy + fy*yCam/z
                float u = cam.CenterX + cam.FocalX * xCam / zCam;
                float v = cam.CenterY + cam.FocalY * yCam / zCam;
                int ix = (int)MathF.Round(u);
                int iy = (int)MathF.Round(v);
                if (ix < 0 || iy < 0 || ix >= w || iy >= h) continue;

                float raw = host[iy * w + ix];
                if (raw < 1e-4f) continue;
                float r = zCam / raw;
                ratios.Add(r);
                perViewRatios[ci].Add(r);
            }
        }

        float global = 1.0f;
        if (ratios.Count >= 8)
        {
            ratios.Sort();
            global = ratios[ratios.Count / 2];
        }

        // A median needs enough samples to beat the outliers that a sparse cloud is full of.
        const int MinSupport = 12;
        var scales = new float[cameras.Length];
        var support = new int[cameras.Length];
        for (int i = 0; i < cameras.Length; i++)
        {
            support[i] = perViewRatios[i].Count;
            if (perViewRatios[i].Count >= MinSupport)
            {
                perViewRatios[i].Sort();
                scales[i] = perViewRatios[i][perViewRatios[i].Count / 2];
            }
            else
            {
                scales[i] = global;
            }
        }
        return (scales, global, support);
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

                // Extension depth at this pixel (DAv3 direct depth)
                float extRaw = extDepthData[ey * extW + bx];
                if (extRaw < 1e-4f) continue;

                // Corresponding reference pixel
                int refX = (int)(bx + dx);
                int refY = (int)(ey + dy);
                if (refX < 0 || refX >= refW || refY < 0 || refY >= refH) continue;

                float refRaw = refDepthData[refY * refW + refX];
                if (refRaw < 1e-4f) continue;

                ratios.Add(refRaw / extRaw);
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
                if (extRaw < 1e-4f) continue;

                int refX = (int)(ex + dx);
                int refY = (int)(by + dy);
                if (refX < 0 || refX >= refW || refY < 0 || refY >= refH) continue;

                float refRaw = refDepthData[refY * refW + refX];
                if (refRaw < 1e-4f) continue;

                ratios.Add(refRaw / extRaw);
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
    /// TempleRing / GT regression:
    /// Joint DAv3 depths + GT cameras + DN-Splatter per-view affine scale + MVSNet/COLMAP FB fuse.
    /// DAv3 extrinsics on TempleRing are often degenerate — do not unproject with them.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithGroundTruthAsync(IReadOnlyList<ImportedImage> images, IReadOnlyList<CameraParams> cameras,
            int subsample = 1, float edgeSharpness = 0.3f, int onlyView = -1, bool globalScale = false)
    {
        if (images.Count != cameras.Count)
            throw new ArgumentException("Image count must match camera count.");

        if (!_depthService.IsReady)
        {
            SetStatus("Loading depth model...");
            await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
        }

        int n = images.Count;
        bool isTemple = images.Any(im => im.FileName.StartsWith("templeR", StringComparison.OrdinalIgnoreCase));
        var lookAt = isTemple
            ? new Vector3(0.028f, 0.042f, -0.054f)
            : cameras.Aggregate(Vector3.Zero, (a, c) => a + c.Position) / cameras.Count;

        for (int i = 0; i < n; i++)
        {
            var c = cameras[i];
            Console.WriteLine($"[MultiView-GT] GT Cam[{i}] {images[i].FileName} pos=({c.Position.X:F4},{c.Position.Y:F4},{c.Position.Z:F4}) fwd=({c.Forward.X:F3},{c.Forward.Y:F3},{c.Forward.Z:F3})");
        }
        Console.WriteLine($"[MultiView-GT] lookAt=({lookAt.X:F4},{lookAt.Y:F4},{lookAt.Z:F4}) subsample={subsample}");

        SetStatus($"Joint DAv3 multi-view depth ({n} images)...");
        var mvResult = await _depthService.EstimateDepthMultiViewAsync(images);
        if (mvResult == null || mvResult.DepthResults.Count < n)
        {
            int got = mvResult?.DepthResults.Count ?? 0;
            mvResult?.Dispose();
            if (isTemple)
            {
                // Monocular fallback reintroduces ghost sheets — fail closed on TempleRing.
                SetStatus($"Error: Joint DAv3 returned {got}/{n} views — fail-closed (no monocular).");
                Console.WriteLine($"[MultiView-GT] FAIL joint_depth got={got}/{n}");
                return null;
            }
            Console.WriteLine("[MultiView-GT] Joint depth failed — monocular fallback");
            return await GenerateWithGroundTruthMonocularFallbackAsync(images, cameras, subsample, edgeSharpness);
        }

        int w = mvResult.DepthResults[0].Width;
        int h = mvResult.DepthResults[0].Height;
        if (w <= 0 || h <= 0)
        {
            mvResult.Dispose();
            SetStatus("Error: Invalid depth map size.");
            return null;
        }

        // ── Depth stays on the device ──
        // ⚠️ This used to CopyToHostAsync every view (4.9 MB at 640x480 x4), run metricization,
        // the scale probe and the FB fuse in WASM, then upload the same maps back for the unproject.
        // The whole middle is kernels now; see MvsFusionGpu. Gate: MvsFusionGpuTests.
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var acc = _gpu.WebGPUAccelerator;
        using var fusionGpu = new MvsFusionGpu(acc);

        bool hasConf = mvResult.HasConfidence;
        int pixelsPerView = w * h;
        var rawViews = new List<MemoryBuffer1D<float, Stride1D.Dense>>(n);
        var confViews = new List<MemoryBuffer1D<float, Stride1D.Dense>>(n);
        for (int i = 0; i < n; i++)
        {
            var d = mvResult.DepthResults[i];
            if (d.RawDepthGpu == null || d.Width != w || d.Height != h)
            {
                mvResult.Dispose();
                SetStatus("Error: Depth view size mismatch.");
                return null;
            }
            rawViews.Add(d.RawDepthGpu);
            if (hasConf && d.ConfidenceGpu != null) confViews.Add(d.ConfidenceGpu);
        }

        using var rawPacked = fusionGpu.PackViews(rawViews, pixelsPerView);
        using var confPacked = confViews.Count == n
            ? fusionGpu.PackViews(confViews, pixelsPerView)
            : acc.Allocate1D<float>(1);
        using var camBuf = fusionGpu.UploadCameras(cameras);

        // ── Multi-anchor metric depths for DN-Splatter affine fit ──
        // The anchors plus lookAt are a few dozen pixel samples, gathered on the device. That is the
        // scalar exemption used properly: n * (anchors + 1) * 2 floats cross, never a map.
        var anchors = BuildTempleAnchors(lookAt, isTemple);
        var probePoints = new List<Vector3>(anchors) { lookAt };
        int lookAtIdx = probePoints.Count - 1;
        var rawSamples = await fusionGpu.GatherAnchorsAsync(
            rawPacked.View, camBuf.View, probePoints, w, h, n);

        var scaleA = new float[n];
        var scaleB = new float[n];
        for (int i = 0; i < n; i++)
        {
            var rawList = new List<float>();
            var metList = new List<float>();
            for (int ai = 0; ai < anchors.Count; ai++)
            {
                var (raw, zCam) = rawSamples[i][ai];
                if (raw < 0f || zCam <= 1e-4f) continue;
                rawList.Add(raw);
                metList.Add(zCam);
            }

            float a = 1f, b = 0f;
            bool fitted = false;
            if (MvsGeometricFusion.TryFitScaleOnly(rawList, metList, out float aOnly))
            {
                a = aOnly; b = 0f; fitted = true;
                // Affine only when it stays near scale-only (tiny-a + large-b flattens relative MDE → sheets).
                if (MvsGeometricFusion.TryFitAffineScaleShift(rawList, metList, out float aAb, out float bAb)
                    && aAb > 0.5f * aOnly && aAb < 2f * aOnly)
                {
                    float medZ = metList.OrderBy(z => z).ElementAt(metList.Count / 2);
                    if (MathF.Abs(bAb) < 0.25f * MathF.Max(medZ, 1e-3f))
                    {
                        double residAb = 0, residA = 0;
                        for (int k = 0; k < rawList.Count; k++)
                        {
                            float e1 = aAb * rawList[k] + bAb - metList[k];
                            float e2 = aOnly * rawList[k] - metList[k];
                            residAb += e1 * e1; residA += e2 * e2;
                        }
                        if (residAb < residA * 0.85)
                        { a = aAb; b = bAb; }
                    }
                }
            }
            else
            {
                var (raw, zCam) = rawSamples[i][lookAtIdx];
                if (raw > 1e-4f && zCam > 1e-4f) { a = zCam / raw; b = 0f; fitted = true; }
            }
            if (!fitted) { a = 1f; b = 0f; }

            scaleA[i] = a; scaleB[i] = b;
            Console.WriteLine($"[MultiView-GT] View {i} affine a={a:F4} b={b:F4} anchors={rawList.Count}");
        }

        // ── One global fit, pooled across every view ──
        // HYPOTHESIS under test: DAv3 JOINT inference already emits depth that is consistent
        // ACROSS views in one shared relative frame. Fitting a separate affine per view (the
        // DN-Splatter recipe, designed for INDEPENDENTLY estimated monocular depths) would then
        // manufacture divergence rather than remove it - and the measured near-depths span
        // 0.313 to 0.467 across four cameras that all sit ~0.52 from the object, which is
        // exactly that signature. Pool the anchors and fit once.
        var poolRaw = new List<float>();
        var poolMet = new List<float>();
        for (int i = 0; i < n; i++)
        {
            for (int ai = 0; ai < anchors.Count; ai++)
            {
                var (raw, zCam) = rawSamples[i][ai];
                if (raw < 0f || zCam <= 1e-4f) continue;
                poolRaw.Add(raw);
                poolMet.Add(zCam);
            }
        }
        float gA = 1f, gB = 0f;
        bool gFitted = MvsGeometricFusion.TryFitScaleOnly(poolRaw, poolMet, out float gAOnly);
        if (gFitted) gA = gAOnly;
        float spread = 0f;
        {
            float mn = float.MaxValue, mx = float.MinValue;
            for (int i = 0; i < n; i++) { mn = MathF.Min(mn, scaleA[i]); mx = MathF.Max(mx, scaleA[i]); }
            spread = mn > 0 ? (mx - mn) / mn : 0f;
        }
        Console.WriteLine($"[MultiView-GT] global_fit a={gA:F4} (pooled {poolRaw.Count} anchors, fitted={gFitted}) " +
            $"| per-view a spread={spread:P1}");

        if (globalScale && gFitted)
        {
            for (int i = 0; i < n; i++) { scaleA[i] = gA; scaleB[i] = 0f; }
            Console.WriteLine("[MultiView-GT] USING GLOBAL SCALE for all views (per-view affine overridden)");
        }

        // metric = a*raw + b for every view, on device.
        using var metricPacked = fusionGpu.Metricize(rawPacked.View, scaleA, scaleB, pixelsPerView);
        Console.WriteLine($"[MultiView-GT] pose=gt+mvs scaleMode={(globalScale ? "GLOBAL" : "per-view")}");

        // Fail-closed: lookAt residual after affine must be ≤5%.
        // If multi-anchor SSI drifts, re-anchor each view to lookAt scale-only (still per-view, not shared).
        float maxLookAtResid = 0f;
        bool reAnchored = false;
        for (int i = 0; i < n; i++)
        {
            var (raw, zCam) = rawSamples[i][lookAtIdx];
            if (!(zCam > 1e-4f)) continue;
            // metric at lookAt is a*raw+b by construction; no need to sample the map back.
            float zm = raw > 1e-6f ? scaleA[i] * raw + scaleB[i] : 0f;
            float rel = zm > 1e-4f ? MathF.Abs(zm - zCam) / zCam : 1f;
            if (rel > 0.05f)
            {
                if (!(raw > 1e-4f))
                {
                    Console.WriteLine($"[MultiView-GT] FAIL affine_probe view {i} lookAt raw invalid");
                    mvResult.Dispose();
                    SetStatus($"Error: View {i} lookAt depth invalid.");
                    return null;
                }
                float aFix = zCam / raw;
                scaleA[i] = aFix; scaleB[i] = 0f;
                reAnchored = true;
                rel = 0f;
                Console.WriteLine($"[MultiView-GT] View {i} re-anchored lookAt scale={aFix:F4} (affine residual was high)");
            }
            maxLookAtResid = MathF.Max(maxLookAtResid, rel);
            Console.WriteLine($"[MultiView-GT] View {i} lookAt residual={rel:P2}");
        }
        if (reAnchored)
        {
            // Rebuild in place. ⚠️ NOT via a temp buffer + CopyFrom: the temp would be disposed
            // before WebGPU's batched submit ran, which is the 'used in submit while destroyed'
            // trap that cost this session a CDP cycle.
            fusionGpu.MetricizeInto(rawPacked.View, scaleA, scaleB, pixelsPerView, metricPacked.View);
        }
        Console.WriteLine($"[MultiView-GT] affine_probe maxLookAtResid={maxLookAtResid:P2}");
        if (isTemple && maxLookAtResid > 0.05f)
        {
            SetStatus($"Error: Affine lookAt residual {maxLookAtResid:P1} >5% — fail-closed.");
            Console.WriteLine($"[MultiView-GT] FAIL affine_probe residual={maxLookAtResid:P2}");
            mvResult.Dispose();
            return null;
        }

        // ── Scale refine: maximize FB agreements per view ──
        SetStatus("Refining per-view scales for multi-view consistency...");
        var refine = new float[n];
        for (int i = 0; i < n; i++)
        {
            refine[i] = await fusionGpu.OptimizeScaleFactorAsync(
                metricPacked.View, camBuf.View, i, w, h, n,
                maxDepthError: 0.05f, maxReprojPx: 2f, probeSubsample: 8);
            if (MathF.Abs(refine[i] - 1f) > 1e-3f)
            {
                scaleA[i] *= refine[i];
                Console.WriteLine($"[MultiView-GT] View {i} scale refine x{refine[i]:F3} -> a={scaleA[i]:F4}");
            }
        }
        fusionGpu.ApplyScales(metricPacked.View, refine, pixelsPerView);

        // Probe: COLMAP FB keep-ratio (fail-closed)
        const float mvsDepthErr = 0.05f;
        int minViews = Math.Min(MvsGeometricFusion.DefaultMinViews, n);
        SetStatus($"MVS consistency probe ({mvsDepthErr:P0}, minViews={minViews})...");
        // Depth range per view going INTO the probe. If these are zero the metric maps were wiped
        // upstream and the keep-ratio gate below will fail with kept=0/0 for the wrong reason.
        var ranges = await fusionGpu.MinMaxPerViewAsync(metricPacked.View, pixelsPerView, n);
        for (int i = 0; i < n; i++)
            Console.WriteLine($"[MultiView-GT] metric view {i} depth=[{ranges[i].min:F5}, {ranges[i].max:F5}]");

        // ⚠️ This was MvsGeometricFusion.FuseDepthMaps with its point list DISCARDED: a full-resolution
        // CPU fusion run purely to produce these stats and the gate below. It is a kernel now, and only
        // the 5-int stats buffer comes back.
        var fuseStats = await fusionGpu.ForwardBackAsync(
            metricPacked.View, confPacked.View, camBuf.View,
            w, h, n,
            subsample: Math.Max(2, subsample),
            maxDepthError: mvsDepthErr,
            maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
            minViews: minViews,
            hasConf: false, confMin: 0f);
        float keepRatio = fuseStats.Input > 0 ? (float)fuseStats.Kept / fuseStats.Input : 0f;
        Console.WriteLine(
            $"[MultiView-GT] mvs_fuse kept={fuseStats.Kept}/{fuseStats.Input} keepRatio={keepRatio:P1} " +
            $"minview_rej={fuseStats.MinViewReject} (thresh={mvsDepthErr:P0}/2px/minViews={minViews}) "
            + $"[gpu threads={fusionGpu.LastFbThreads}]");

        // ── POSITIVE CONTROL: does this probe report agreement AT ALL? ──
        // I have been reading keepRatio=0.8% as "the views disagree" without ever checking the
        // probe can return a high number. Feed it n copies of view 0 with n copies of view 0's
        // CAMERA: every pixel then agrees with itself by construction, so a correct probe must
        // keep nearly everything. If this comes back low, keepRatio was never evidence about
        // the scene and every conclusion drawn from it is void.
        var ctlAcc = _gpu.WebGPUAccelerator;
        using (var selfDepth = ctlAcc.Allocate1D<float>((long)n * pixelsPerView))
        using (var selfCams = ctlAcc.Allocate1D<float>(camBuf.Length))
        {
            for (int i = 0; i < n; i++)
                selfDepth.View.SubView((long)i * pixelsPerView, pixelsPerView)
                    .CopyFrom(metricPacked.View.SubView(0, pixelsPerView));
            long camStride = camBuf.Length / n;
            for (int i = 0; i < n; i++)
                selfCams.View.SubView(i * camStride, camStride).CopyFrom(camBuf.View.SubView(0, camStride));
            await ctlAcc.SynchronizeAsync();

            var ctl = await fusionGpu.ForwardBackAsync(
                selfDepth.View, confPacked.View, selfCams.View,
                w, h, n,
                subsample: Math.Max(4, subsample),
                maxDepthError: 0.05f,
                maxReprojPx: MvsGeometricFusion.DefaultMaxReprojPx,
                minViews: minViews,
                hasConf: false, confMin: 0f);
            float ctlKeep = ctl.Input > 0 ? (float)ctl.Kept / ctl.Input : 0f;
            Console.WriteLine($"[MultiView-GT] CONTROL self-vs-self keep={ctlKeep:P2} " +
                $"({ctl.Kept}/{ctl.Input}) - expect ~100%; anything low means the PROBE is broken");
        }

        // ── Alignment diagnostic: keep-ratio vs tolerance ──
        // MEASURED 2026-09-20: one view alone reproduces its own photo at 27.7 dB, four views
        // merged drop it to 19.2 dB, and this probe keeps only ~0.8% at 5%. The merge is
        // destructive, so the question is WHY the views disagree, and the shape of this sweep
        // answers it without guessing:
        //   climbs steeply with tolerance -> a per-view SCALE/offset error, fixable by a better
        //                                    metric fit (the depth SHAPE is right)
        //   stays flat                    -> the per-view depth shape itself disagrees, and no
        //                                    scalar correction will ever reconcile them
        // Cheap: reuses the same kernel, one extra GPU pass per threshold, stats only.
        // Sweep BOTH thresholds. The first sweep moved only maxDepthError and came back dead
        // flat (0.78% -> 0.79% across a 20x relaxation), which does not mean "the depths
        // disagree" - it means rejection happens before the depth test is reached, i.e. the
        // REPROJECTION gate is the binding one. Sweeping one of two thresholds answers nothing.
        foreach (var (tol, px) in new[]
        {
            (0.05f, 2f), (0.40f, 2f),           // depth relaxed, reprojection tight
            (0.05f, 8f), (0.05f, 32f),          // reprojection relaxed, depth tight
            (0.40f, 32f), (0.40f, 128f),        // both wide open
        })
        {
            var st = await fusionGpu.ForwardBackAsync(
                metricPacked.View, confPacked.View, camBuf.View,
                w, h, n,
                subsample: Math.Max(4, subsample),   // coarser: this is a statistic, not output
                maxDepthError: tol,
                maxReprojPx: px,
                minViews: minViews,
                hasConf: false, confMin: 0f);
            float kr = st.Input > 0 ? (float)st.Kept / st.Input : 0f;
            Console.WriteLine($"[MultiView-GT] align_sweep depth={tol:P0} reproj={px:F0}px keep={kr:P2} " +
                $"({st.Kept}/{st.Input}, minview_rej={st.MinViewReject})");
        }

        if (isTemple && (keepRatio < 0.001f || keepRatio > 0.95f))
        {
            mvResult.Dispose();
            SetStatus($"Error: MVS keepRatio={keepRatio:P2} out of bounds.");
            Console.WriteLine($"[MultiView-GT] FAIL mvs_fuse keepRatio={keepRatio:P2}");
            return null;
        }

        // Best visual so far: dense per-view metric unproject + consistency (~323k, columns visible).
        SetStatus($"Dense GPU unproject ({n} metric views)...");
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        var device = accelerator.NativeAccelerator.NativeDevice!;
        var queue = accelerator.NativeAccelerator.Queue!;

        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0;
        int fuseSub = Math.Max(1, subsample);

        // ── Pass A: materialise every view's metric depth ──
        // Done up front so pass B can screen ANY view against ANY other. Previously the depth
        // was sliced inside the unproject loop, so at i=0 no other view existed yet and view 0
        // simply skipped consistency filtering - its entire unfiltered cloud survived, border
        // ring and all. metricPacked already holds all n views; this is device-to-device
        // slicing, no readback.
        for (int i = 0; i < n; i++)
        {
            var depthGpu = accelerator.Allocate1D<float>(pixelsPerView);
            depthGpu.View.CopyFrom(metricPacked.View.SubView((long)i * pixelsPerView, pixelsPerView));

            var dr = mvResult.DepthResults[i];
            dr.RawDepthGpu?.Dispose();
            dr.RawDepthGpu = depthGpu;
            // KEEP the DAv3 confidence. Metricizing rescales depth VALUES; it does not change
            // which pixels the network was confident about, so the per-pixel confidence still
            // applies. Disposing and nulling it here meant SplatWorldParams arrived with
            // HasConfidence=0 and the ConfMin gate never ran on the live path at all.
            dr.MinDepth = ranges[i].min;
            dr.MaxDepth = ranges[i].max;
        }
        await accelerator.SynchronizeAsync();

        int confRetained = 0;
        for (int i = 0; i < n; i++) if (mvResult.DepthResults[i].ConfidenceGpu != null) confRetained++;
        Console.WriteLine($"[MultiView-GT] metric depths ready for {n} views, confidence retained on {confRetained}/{n}");

        // ── Pass B: unproject each view and screen it against a DIFFERENT view ──
        // onlyView isolates ONE view's cloud. Diagnostic, not a product path: rendering a
        // training view from its own depth map alone separates "the per-view depth is wrong"
        // from "the views disagree with each other". Joint depth still runs over all N views,
        // so only the unprojection is restricted.
        if (onlyView >= 0)
            Console.WriteLine($"[MultiView-GT] DIAGNOSTIC onlyView={onlyView} - unprojecting a single view, no cross-view merge");

        for (int i = 0; i < n; i++)
        {
            if (onlyView >= 0 && i != onlyView) continue;
            var src = mvResult.DepthResults[i];

            SetStatus($"Unprojecting metric view {i + 1}/{n}...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                src, images[i], cameras[i], fuseSub, edgeSharpness, depthScale: 1f);
            int rawCount = count;

            // View 0 is NOT screened, and that exemption is load-bearing rather than an
            // oversight. MEASURED 2026-09-20: screening it against view 1 cost templeR0001
            // 3.55 dB, because the views are not in a common metric frame (keepRatio 0.8%) -
            // so the filter discards view 0's GOOD data for disagreeing with a bad reference.
            // Revisit only once cross-view alignment is fixed; screening against a misaligned
            // reference is worse than not screening at all.
            int refIdx = 0;
            if (onlyView < 0 && i > 0 && count > 0 && n > 1
                && mvResult.DepthResults[refIdx].RawDepthGpu != null)
            {
                (buf, count) = await _gaussianKernel.FuseConsistencyVsRefAsync(
                    buf, count, mvResult.DepthResults[refIdx], cameras[refIdx],
                    depthScale: 1f, relThresh: 0.08f);
            }

            viewResults.Add((buf, count));
            totalSplats += count;
            Console.WriteLine($"[MultiView-GT] View {i}: {count:N0} dense splats " +
                $"(raw {rawCount:N0}, {(i > 0 ? $"screened vs view {refIdx}" : "reference, unscreened")})");
        }

        mvResult.Dispose();

        if (totalSplats == 0)
        {
            foreach (var (buf, _) in viewResults) buf.Dispose();
            SetStatus("Error: No views produced splats.");
            return null;
        }

        SetStatus($"Merging {totalSplats:N0} dense MVS splats...");
        var merged = accelerator.Allocate1D<float>(totalSplats * SplatFormat.Floats);
        long offsetBytes = 0;
        int actualTotal = 0;
        foreach (var (buf, count) in viewResults)
        {
            if (count <= 0) { buf.Dispose(); continue; }
            ulong byteCount = (ulong)count * SplatFormat.Floats * sizeof(float);
            using (var encoder = device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(buf.GetGPUBuffer()!, 0, merged.GetGPUBuffer()!, (ulong)offsetBytes, byteCount);
                queue.Submit(new[] { encoder.Finish() });
            }
            offsetBytes += (long)byteCount;
            actualTotal += count;
            buf.Dispose();
        }
        await accelerator.SynchronizeAsync();

        if (isTemple)
        {
            SetStatus("Culling background away from temple...");
            (merged, actualTotal) = await _gaussianKernel.CullOutsideSphereAsync(
                merged, actualTotal, lookAt, radius: 0.13f);
            if (actualTotal == 0)
            {
                SetStatus("Error: Sphere cull removed all splats.");
                return null;
            }
        }

        Console.WriteLine($"[MultiView-GT] mvs_voxel=dense-gpu-full count={actualTotal:N0}");
        SetStatus($"GT complete: {actualTotal:N0} splats pose=gt+mvs");
        Console.WriteLine($"[MultiView-GT] Total: {actualTotal:N0} pose=gt+mvs");
        return (merged, actualTotal);
    }

    /// <summary>Multi-anchor points for DN-Splatter affine fit (lookAt + offset ring).</summary>
    private static List<Vector3> BuildTempleAnchors(Vector3 lookAt, bool isTemple)
    {
        var anchors = new List<Vector3> { lookAt };
        float s = isTemple ? 0.04f : 0.1f;
        anchors.Add(lookAt + new Vector3(s, 0, 0));
        anchors.Add(lookAt + new Vector3(-s, 0, 0));
        anchors.Add(lookAt + new Vector3(0, s, 0));
        anchors.Add(lookAt + new Vector3(0, -s, 0));
        anchors.Add(lookAt + new Vector3(0, 0, s));
        anchors.Add(lookAt + new Vector3(0, 0, -s));
        anchors.Add(lookAt + new Vector3(s, s, 0));
        anchors.Add(lookAt + new Vector3(-s, s, 0));
        anchors.Add(lookAt + new Vector3(s, -s, 0));
        anchors.Add(lookAt + new Vector3(-s, -s, 0));
        return anchors;
    }

    /// <summary>Monocular depth + per-view lookAt scale + GT poses.</summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateWithGroundTruthMonocularFallbackAsync(
            IReadOnlyList<ImportedImage> images, IReadOnlyList<CameraParams> cameras,
            int subsample, float edgeSharpness)
    {
        bool isTemple = images.Any(im => im.FileName.StartsWith("templeR", StringComparison.OrdinalIgnoreCase));
        var lookAt = isTemple
            ? new Vector3(0.028f, 0.042f, -0.054f)
            : cameras.Aggregate(Vector3.Zero, (a, c) => a + c.Position) / cameras.Count;

        Console.WriteLine($"[MultiView-GT] lookAt=({lookAt.X:F4},{lookAt.Y:F4},{lookAt.Z:F4}) subsample={subsample} pose=gt (monocular fallback)");

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        var device = accelerator.NativeAccelerator.NativeDevice!;
        var queue = accelerator.NativeAccelerator.Queue!;

        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0;

        for (int i = 0; i < images.Count; i++)
        {
            SetStatus($"Depth estimation: {images[i].FileName} ({i + 1}/{images.Count})...");
            var depthResult = await _depthService.EstimateDepthAsync(images[i]);
            if (depthResult == null)
            {
                Console.WriteLine($"[MultiView-GT] Depth failed for {i}");
                continue;
            }

            var cam = cameras[i];
            float viewScale = 1f;
            if (WorldSpaceGeometry.Project(cam, lookAt, out float u, out float v, out float zCam) && zCam > 1e-4f)
            {
                try
                {
                    float[] host = await depthResult.RawDepthGpu!.CopyToHostAsync<float>(0, depthResult.RawDepthGpu.Length);
                    int ix = Math.Clamp((int)MathF.Round(u), 0, depthResult.Width - 1);
                    int iy = Math.Clamp((int)MathF.Round(v), 0, depthResult.Height - 1);
                    float raw = host[iy * depthResult.Width + ix];
                    if (raw > 1e-4f)
                    {
                        viewScale = zCam / raw;
                        Console.WriteLine($"[MultiView-GT] View {i} lookAt uv=({u:F1},{v:F1}) z={zCam:F4} raw={raw:F4} scale={viewScale:F4}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MultiView-GT] View {i} scale: {ex.Message}");
                }
            }

            SetStatus($"Generating splats: {images[i].FileName} ({i + 1}/{images.Count}, pose=gt)...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                depthResult, images[i], cam, subsample, edgeSharpness, viewScale);
            depthResult.Dispose();

            viewResults.Add((buf, count));
            totalSplats += count;
            Console.WriteLine($"[MultiView-GT] View {i}: {count:N0} splats scale={viewScale:F4}");
        }

        if (totalSplats == 0)
        {
            foreach (var (buf, _) in viewResults) buf.Dispose();
            SetStatus("Error: No views produced splats.");
            return null;
        }

        SetStatus($"Merging {totalSplats:N0} splats (pose=gt)...");
        var merged = accelerator.Allocate1D<float>(totalSplats * SplatFormat.Floats);
        long offsetBytes = 0;
        int actualTotal = 0;
        foreach (var (buf, count) in viewResults)
        {
            if (count <= 0) { buf.Dispose(); continue; }
            ulong byteCount = (ulong)count * SplatFormat.Floats * sizeof(float);
            using (var encoder = device.CreateCommandEncoder())
            {
                encoder.CopyBufferToBuffer(buf.GetGPUBuffer()!, 0, merged.GetGPUBuffer()!, (ulong)offsetBytes, byteCount);
                queue.Submit(new[] { encoder.Finish() });
            }
            offsetBytes += (long)byteCount;
            actualTotal += count;
            buf.Dispose();
        }
        await accelerator.SynchronizeAsync();

        SetStatus($"GT complete: {actualTotal:N0} splats pose=gt");
        Console.WriteLine($"[MultiView-GT] Total: {actualTotal:N0} pose=gt (monocular)");
        return (merged, actualTotal);
    }

    /// <summary>
    /// Parse a Middlebury-format camera parameter file (templeR_par.txt, dinoSR_par.txt).
    /// See <see cref="WorldSpaceGeometry.ParseMiddleburyParams"/>.
    /// </summary>
    public static List<(string filename, CameraParams camera)> ParseMiddleburyParams(string parFileContent, int imageWidth, int imageHeight)
        => WorldSpaceGeometry.ParseMiddleburyParams(parFileContent, imageWidth, imageHeight);

    private void SetStatus(string status)
    {
        Status = status;
        Console.WriteLine($"[MultiView] {status}");
        OnStatusChanged?.Invoke();
    }
}
