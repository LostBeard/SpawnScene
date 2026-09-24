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

    /// <summary>Where <see cref="LastCameras"/> came from: sfm, dav3, dav3-chunked or fallback.</summary>
    public string LastPoseSource { get; private set; } = "none";

    /// <summary>
    /// dav3-chunked only: the joint pass each of <see cref="LastCameras"/> was adopted from (-1 = unposed;
    /// 0 = the reference pass, which also holds the shared anchors). Empty for every other pose source.
    /// Lets a ground-truth report separate a pass's own error from the error its fold added.
    /// </summary>
    public int[] LastChunkOf { get; private set; } = [];

    /// <summary>
    /// Publish cameras that came from outside the pose cascade, so everything downstream sees
    /// them the same way it sees recovered ones.
    ///
    /// The point of this is measurement. Every Bathroom number conflates two error sources -
    /// the poses being wrong and the optimiser being wrong - because Bathroom has no ground
    /// truth. Deep Blending's drjohnson and playroom are real rooms WITH COLMAP poses, so
    /// feeding those in measures the optimiser on its own.
    /// </summary>
    public void UseExternalCameras(CameraParams?[] cameras, string source)
    {
        LastCameras = cameras;
        LastPoseSource = source;
        LastChunkOf = [];
    }

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
    /// prefers SfM and falls back.
    ///
    /// Default is dav3 (MEASURED 2026-09-23 on DrJohnson 2K): dav3-chunked reached supervised
    /// ~22 dB / held ~13 before a device loss; sfm finished supervised 14.9 / held 12.6 with
    /// held-out cross-match picking the WRONG target on every sampled view. Preferring SfM
    /// under "auto" put joint DAv3 depths and SfM cameras in different frames - the Bathroom
    /// soup case the comment above already named.
    /// </summary>
    public string PosePreference { get; set; } = "dav3";

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

    /// <summary>
    /// Choose the shared anchors by how much the views actually OVERLAP, rather than spreading
    /// them evenly over the capture. See <see cref="MultiViewChunkPlan.PickAnchorsByOverlap"/>:
    /// an even spread is right for an orbit and close to the worst choice for a walk-through.
    /// </summary>
    public bool OverlapAnchors { get; set; } = true;

    /// <summary>
    /// Refine the chunked cascade's cameras with bundle adjustment over the import's feature matches
    /// (<see cref="BundleAdjuster"/>). The cascade alone measured median 7.8% / p90 27% of the camera
    /// spread on Truck, and those poses shred a splat scene that COLMAP poses render cleanly.
    /// </summary>
    public bool BundleAdjust { get; set; } = true;

    /// <summary>Levenberg-Marquardt iterations per BA round.</summary>
    public int BundleAdjustIterations { get; set; } = 150;

    /// <summary>Pairs whose cascade cameras face further apart than this are not verified or used by BA.</summary>
    public float MaxPairAngleDeg { get; set; } = 45f;

    private void RefineWithBundleAdjustment(
        IReadOnlyList<ImportedImage> images, CameraParams?[] cameras, IReadOnlyList<int> posed)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pairs = _importService.MatchedPairs;
        if (pairs.Count == 0 || _importService.Images.Count != images.Count)
        {
            Console.WriteLine(
                $"[BA] SKIPPED: {pairs.Count} matched pairs for {_importService.Images.Count} imported vs " +
                $"{images.Count} images - no feature tracks to adjust against.");
            return;
        }

        // BA works on the posed views only; map global image index <-> BA camera index.
        var baIndex = new Dictionary<int, int>();
        var cams = new List<CameraParams>();
        foreach (int g in posed) { baIndex[g] = cams.Count; cams.Add(cameras[g]!); }

        // Verify every pair's matches geometrically before they may form tracks. MEASURED on Truck: ~34
        // chance matches on EVERY pair (adjacent frames: median 216) - unverified, they chained into
        // inconsistent tracks and left BA 5,197 points for 126 cameras. Pairs whose cascade cameras face
        // more than MaxPairAngleDeg apart cannot share much and are not worth a RANSAC.
        float cosMax = MathF.Cos(MaxPairAngleDeg * MathF.PI / 180f);
        var verified = new List<(int, int, int, int)>();
        int considered = 0, passed = 0;
        var tv = System.Diagnostics.Stopwatch.StartNew();
        foreach (var p in pairs)
        {
            if (!baIndex.ContainsKey(p.ImageIndexA) || !baIndex.ContainsKey(p.ImageIndexB)) continue;
            var ca = cameras[p.ImageIndexA]!; var cb = cameras[p.ImageIndexB]!;
            if (System.Numerics.Vector3.Dot(System.Numerics.Vector3.Normalize(ca.Forward),
                    System.Numerics.Vector3.Normalize(cb.Forward)) < cosMax) continue;
            if (p.Matches.Count < 15) continue;
            considered++;
            var fa = images[p.ImageIndexA].Features; var fb = images[p.ImageIndexB].Features;
            var xa = new float[p.Matches.Count * 2]; var xb = new float[p.Matches.Count * 2];
            for (int k = 0; k < p.Matches.Count; k++)
            {
                var m = p.Matches[k];
                xa[k * 2] = fa[m.IndexA].X; xa[k * 2 + 1] = fa[m.IndexA].Y;
                xb[k * 2] = fb[m.IndexB].X; xb[k * 2 + 1] = fb[m.IndexB].Y;
            }
            var r = EpipolarRansac.Estimate(xa, xb, thresholdPx: 2.0, minInliers: 15,
                seed: p.ImageIndexA * 7919 + p.ImageIndexB);
            if (r == null) continue;
            passed++;
            for (int k = 0; k < p.Matches.Count; k++)
                if (r.Inliers[k]) verified.Add((p.ImageIndexA, p.Matches[k].IndexA, p.ImageIndexB, p.Matches[k].IndexB));
        }
        Console.WriteLine(
            $"[BA] verification: {considered} pairs within {MaxPairAngleDeg} deg of each other, {passed} verified, " +
            $"{verified.Count} inlier matches ({tv.Elapsed.TotalSeconds:F1}s)");
        var tracks = BundleAdjuster.BuildTracks(verified);

        var points = new List<System.Numerics.Vector3>();
        var obs = new List<BundleAdjuster.Observation>();
        var trackObs = new List<(int Camera, float U, float V)>();
        foreach (var track in tracks)
        {
            trackObs.Clear();
            foreach (var (img, feat) in track)
            {
                var f = images[img].Features[feat];
                trackObs.Add((baIndex[img], f.X, f.Y));
            }
            if (!BundleAdjuster.Triangulate(cams, trackObs, out var x)) continue;
            int id = points.Count;
            points.Add(x);
            foreach (var (c, u, v) in trackObs) obs.Add(new BundleAdjuster.Observation(c, id, u, v));
        }
        if (points.Count < 50)
        {
            Console.WriteLine($"[BA] SKIPPED: only {points.Count} triangulated tracks from {tracks.Count}.");
            return;
        }

        // One image size = one camera (a phone video, a photo set from one device): solve ONE focal with the
        // poses instead of holding DAv3's per-frame guesses fixed. MEASURED on a synthetic rig with DAv3-like
        // focal error (+10% bias, +-8% scatter): fixed per-view focals leave the poses at 13.8% of spread,
        // a shared focal gets 0.033% and recovers f exactly.
        bool oneCamera = cams.Select(c => (c.Width, c.Height)).Distinct().Count() == 1;
        var focals = cams.Select(c => 0.5f * (c.FocalX + c.FocalY)).OrderBy(f => f).ToList();
        var ba = new BundleAdjuster(cams, points, obs, sharedFocal: oneCamera);
        var result = ba.Solve(new BundleAdjuster.Options
        {
            MaxIterations = BundleAdjustIterations,
            RoundLog = (round, iters, rms, kept) => Console.WriteLine(
                $"[BA]   round {round}: {iters} iterations, RMS {rms:F2} px, {kept} obs kept"),
        });
        for (int i = 0; i < cams.Count; i++) ba.WriteCamera(i, cams[i]);
        Console.WriteLine(
            $"[BA] focal: DAv3 per-view median {focals[focals.Count / 2]:F1} (p10 {focals[focals.Count / 10]:F1}, " +
            $"p90 {focals[focals.Count * 9 / 10]:F1})" +
            (oneCamera ? $" -> shared {ba.SharedFocal:F1}" : " (mixed image sizes: per-view focals held)"));

        Console.WriteLine(
            $"[BA] {cams.Count} cameras, {tracks.Count} tracks -> {points.Count} points, {result.Observations} obs " +
            $"({result.ObservationsKept} kept), reprojection RMS {result.InitialRmsPixels:F1} -> " +
            $"{result.FinalRmsPixels:F2} px in {result.Iterations} iterations; solve {result.Seconds:F1}s, " +
            $"total {sw.Elapsed.TotalSeconds:F1}s");
    }

    /// <summary>Anchor views the last chunked run used, and why they were picked.</summary>
    public string LastAnchorSource { get; private set; } = "none";

    /// <summary>
    /// Keep splats the screening reference camera cannot see.
    ///
    /// Default true for rooms (MEASURED 2026-09-23 on DrJohnson dav3-chunked): with false, a
    /// non-reference view kept 0 / 39,928 splats - reject reason "outside the reference view"
    /// for 39,915 of them. That is the object-centric frustum screen eating the other walls.
    /// Object-centric captures that need the old behaviour can set <c>?outside=0</c>.
    /// See <see cref="WorldSpaceGeometry.ClassifySplatVsRef"/>.
    /// </summary>
    public bool KeepOutsideReferenceView { get; set; } = true;

    /// <summary>
    /// Relative depth agreement the screen demands. 0.06 is what the object path uses; an
    /// indoor handheld capture may not deserve that precision, so it is a knob to measure.
    /// </summary>
    public float ConsistencyRelThreshold { get; set; } = 0.06f;

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
        LastChunkOf = [];

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

        if (BundleAdjust) RefineWithBundleAdjustment(images, poses.Cameras, posed);

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

        // The consistency screen compares a view's splats against a REFERENCE view's depth, and
        // that only means anything inside one joint pass, where the views genuinely share a
        // frame. Screening across passes measures the FOLD, not the geometry: a fold that lands
        // at 10% of the camera spread cannot pass a 6% screen, so those views lose everything
        // they have regardless of how good their depth is.
        //
        // MEASURED before this change, Bathroom at 14 posed views: 9 of 13 non-reference views
        // kept 0%, and the whole scene was one view's cloud with fragments attached.
        //
        // So each pass screens against its own first view, exactly as the single-pass path
        // screens against view 0, and that view goes in unscreened.
        var refOfChunk = new Dictionary<int, int>();
        foreach (int i in posed)
            if (!refOfChunk.ContainsKey(poses.ChunkOf[i])) refOfChunk[poses.ChunkOf[i]] = i;

        var viewResults = new List<(MemoryBuffer1D<float, Stride1D.Dense> buf, int count)>();
        int totalSplats = 0, nonRefIn = 0, nonRefKept = 0;

        foreach (int i in posed)
        {
            SetStatus($"Generating splats: {images[i].FileName} ({viewResults.Count + 1}/{posed.Count})...");
            var (buf, count) = await _gaussianKernel.GeneratePackedGpuBufferWorldSpaceAsync(
                poses.Depths[i]!, images[i], poses.Cameras[i]!, effectiveSub, edgeSharpness, scales[i]);

            int refView = refOfChunk[poses.ChunkOf[i]];
            if (i != refView && count > 0)
            {
                nonRefIn += count;
                (buf, count) = await _gaussianKernel.FuseConsistencyVsRefAsync(
                    buf, count, poses.Depths[refView]!, poses.Cameras[refView]!,
                    scales[refView], relThresh: ConsistencyRelThreshold,
                    keepOutsideView: KeepOutsideReferenceView);
                nonRefKept += count;
            }

            viewResults.Add((buf, count));
            totalSplats += count;
        }

        Console.WriteLine(
            $"[MultiView] screened within {refOfChunk.Count} pass(es); each pass's first view is " +
            "its own reference and goes in unscreened");

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

        await AlignToGravityAsync(merged, actualTotal, posed.Select(i => poses.Cameras[i]!));

        LastCameras = poses.Cameras;
        LastPoseSource = "dav3-chunked";
        LastChunkOf = poses.ChunkOf.ToArray();

        Console.WriteLine(
            $"[MultiView] Total: {actualTotal:N0} splats from {posed.Count}/{images.Count} views " +
            $"pose=dav3-chunked N={poses.ChunkSize} subsample={effectiveSub}");
        SetStatus($"Multi-view complete: {actualTotal:N0} splats from {posed.Count} views.");
        return (merged, actualTotal);
    }

    /// <summary>
    /// Target splats for one initialisation, derived from what the TRAINER can actually train
    /// rather than picked by eye.
    ///
    /// The binding limit is the real constraint. Key-indexed training buffers are bounded by
    /// <c>maxStorageBufferBindingSize</c> - a guaranteed 128 MiB - not by VRAM, and the widest
    /// key-indexed binding is 12 bytes, so there are about 11.2M keys to go round. Ask for more
    /// splats than that supports and <c>SplatTrainerGpu</c> clamps keysPerSplat, and then frames
    /// silently come out INCOMPLETE.
    ///
    /// MEASURED on Bathroom at a 3M budget: 2,015,285 splats, keysPerSplat clamped 8 -> 5, and
    /// frames needing 12.9M keys against a 10.1M capacity - "KEY OVERFLOW ... this frame is
    /// incomplete" on nearly every iteration, ending with the device lost outright ("A valid
    /// external Instance reference no longer exists").
    ///
    /// So the default is the count that lets keysPerSplat stay at its own default of 8 without
    /// clamping. Emitting a splat per pixel per view is not a better start, it is a bigger one;
    /// a 3DGS optimiser densifies from an initialisation, and coarser sampling across MORE views
    /// beats dense sampling of a few, because the extra views are the new information.
    /// </summary>
    public int SplatBudget { get; set; } = TrainableSplatBudget;

    /// <summary>
    /// Splats the trainer can carry at its default keysPerSplat without clamping.
    /// 128 MiB guaranteed binding / 12 bytes per key / 8 keys per splat.
    /// </summary>
    public const int TrainableSplatBudget = (128 * 1024 * 1024) / 12 / 8;

    private int ChooseSubsample(int requested, IReadOnlyList<DepthResult> depths)
        => ChooseSubsample(requested, depths.Select(d => (d.Width, d.Height)).ToList());

    /// <summary>
    /// Coarsen the per-view sampling until the whole set fits the budget.
    ///
    /// Every path that emits a splat per pixel per view needs this, not just the one it was
    /// written for. MEASURED: the ground-truth path had no budget at all, so 88 views of
    /// drjohnson at subsample 2 emitted 14,011,008 splats, and the merge - 784 MB in a single
    /// buffer - lost the device outright with "A valid external Instance reference no longer
    /// exists". A limit that exists in one code path is not a limit.
    /// </summary>
    private int ChooseSubsample(int requested, IReadOnlyList<(int Width, int Height)> views)
    {
        int sub = Math.Max(1, requested);
        long PixelsAt(int s) => views.Sum(v => (long)(v.Width / s) * (v.Height / s));

        long at = PixelsAt(sub);
        if (at <= SplatBudget) return sub;

        int chosen = sub;
        while (chosen < 16 && PixelsAt(chosen) > SplatBudget) chosen++;
        Console.WriteLine(
            $"[MultiView] subsample {sub} -> {chosen}: {views.Count} views would emit " +
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

        /// <summary>
        /// Which joint pass each view's depth came out of, or -1 if unposed.
        ///
        /// Needed because the cross-view consistency screen is only meaningful WITHIN a pass.
        /// Two views from one forward genuinely share a frame; two views from different passes
        /// share it only as well as the fold between them, and a fold that lands at 10% of the
        /// camera spread cannot pass a screen calibrated at 6%.
        /// </summary>
        public required int[] ChunkOf { get; init; }

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
            ChunkOf = Enumerable.Repeat(-1, images.Count).ToArray(),
        };

        IReadOnlyList<MultiViewChunkPlan.MultiViewShapeGroup>? groups = null;
        DepthEstimationService.MultiViewDepthResult? firstRun = null;

        // One GPU upload per image for the whole run. Anchors are in EVERY chunk, so without
        // this each of them is re-copied on the managed heap and re-uploaded once per chunk.
        using var uploads = new DepthEstimationService.MultiViewUploadCache(_gpu);

        // Anchor choice is independent of N, so it is decided once, before any backoff.
        Func<IReadOnlyList<int>, int, int[]>? pickAnchors = null;
        LastAnchorSource = "spread";
        if (OverlapAnchors && images.Count > chunkSize)
        {
            var overlap = await BuildOverlapMatrixAsync(images);
            if (overlap != null)
            {
                LastAnchorSource = "overlap";
                pickAnchors = (global, count) =>
                {
                    var local = MultiViewChunkPlan.PickAnchorsByOverlap(
                        global.Count, (a, b) => overlap[global[a], global[b]], count);
                    return local.Select(i => global[i]).ToArray();
                };
            }
        }

        while (chunkSize > anchors)
        {
            groups = MultiViewChunkPlan.PlanByShape(shapes, chunkSize, anchors, pickAnchors);
            SetStatus($"Joint depth pass 1 of {groups[0].Chunks.Count} (N={chunkSize})...");
            firstRun = await TryRunChunkAsync(images, groups[0].Chunks[0], chunkSize, uploads);

            // Depths without EXTRINSICS is also a failure of this N, not a chunk to reject. The
            // whole point of the joint pass is the shared frame, and nothing can be folded
            // without cameras - so it backs off rather than limping on with one posed chunk.
            if (firstRun != null && firstRun.Extrinsics == null)
            {
                Console.WriteLine(
                    $"[MultiView] N={chunkSize} returned depth but no extrinsics - no shared frame " +
                    "to fold into.");
                firstRun.Dispose();
                firstRun = null;
            }
            if (firstRun != null) break;

            // Multiplicative, not one at a time. MEASURED on an RTX 40-series: N=8 runs and N=10
            // dies inside the graph executor on GPU memory ("A valid external Instance reference
            // no longer exists"), so the ceiling is real and a failed forward is expensive.
            // Walking 10,9,8 would buy the same answer for two extra full passes.
            int next = Math.Max(anchors + 1, chunkSize * 3 / 4);
            if (next >= chunkSize) next = chunkSize - 1;
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
            $"{anchors} shared anchors [{string.Join(",", reference.Chunks[0].Anchors.ToArray())}] " +
            $"chosen by {LastAnchorSource}");

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

            var run = await TryRunChunkAsync(images, chunk, chunkSize, uploads);
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
                chunk, cams, placed, out var sim, out float rms, out int used, out float spread,
                out int inliers, out int[] inlierSlots);

            // Always report residual AGAINST the spread. The threshold is otherwise a judgement
            // call nobody can check, and at MinAnchors the fit is over-determined by only two -
            // so a non-zero residual is not rounding, it says the model gave a differently SHAPED
            // anchor triangle in this pass than in the reference one.
            // When no subset found support there is no fit, and rms is still the sentinel:
            // printing it gives 3.4e38 and reads as a broken number rather than as "no fit".
            string residual = inliers >= MultiViewChunkPlan.MinFoldAnchors
                ? $"residual {rms:F4} on a spread of {spread:F4} " +
                  $"({(spread > 0 ? rms / spread : float.NaN):P1} of it, limit " +
                  $"{MultiViewChunkPlan.MaxAnchorRmsFraction:P0})"
                : $"no pair of them agreed in both position and orientation (spread {spread:F4}, " +
                  $"tolerance {MultiViewChunkPlan.InlierAnchorFraction:P0} / " +
                  $"{MultiViewChunkPlan.MaxAnchorRotationRadians * 180f / MathF.PI:F0} deg)";
            // Name the anchor the fold threw out. On DrJohnson that is the whole story of a pass.
            var excluded = Enumerable.Range(0, chunk.AnchorCount)
                .Where(s => cams[s] != null && placed.ContainsKey(chunk.Views[s]) && !inlierSlots.Contains(s))
                .Select(s => chunk.Views[s]).ToArray();
            string fitLine =
                $"anchors {used}/{chunk.AnchorCount} recovered, {inliers} agreeing" +
                (fitted && excluded.Length > 0 ? $" (anchor {string.Join(",", excluded)} excluded)" : "") +
                $", {residual}";

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
            AdoptChunk(result, chunk, cams, run, sim, adoptAnchors: false, chunkIndex: ci);
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
            $"[MultiView] frame uploads: {uploads.Uploads} new, {uploads.Reuses} reused " +
            "(an anchor is in every chunk; each upload is a full RGBA frame)");
        Console.WriteLine(
            $"[MultiView] chunked poses done: {result.PosedCount}/{images.Count} views posed in one " +
            $"frame, {result.ChunksRejected} chunk(s) rejected");
        // Depth is finished for this generate; training is next on the same GPU. Without this the
        // session keeps its entire activation arena (MEASURED 3.9 GB after 14 DAv3 N=6 passes) and
        // the trainer's first densify resize is what finally tips Chrome over.
        _depthService.ReleaseWorkingMemory();
        return result;
    }

    /// <summary>
    /// How much each pair of views actually saw of each other, as geometrically verified feature
    /// matches.
    ///
    /// This runs the feature matcher on a path that deliberately SKIPS SfM, so to be explicit
    /// about why that is not a contradiction: commit eeedfff skipped SfM because taking DEPTHS
    /// from DAv3 and CAMERAS from SfM puts geometry and cameras in different frames at different
    /// scales. Nothing here takes any geometry. The only question asked is which frames have
    /// pixels in common, so that the anchors shared between passes are views the depth model can
    /// actually relate to each other. Every pose still comes from DAv3, in one frame.
    ///
    /// Returns null when matching produces nothing, and the even spread is used instead.
    /// </summary>
    private async Task<int[,]?> BuildOverlapMatrixAsync(IReadOnlyList<ImportedImage> images)
    {
        try
        {
            var t0 = DateTime.UtcNow;
            SetStatus($"Measuring view overlap ({images.Count} images)...");
            _importService.Clear();
            await _importService.ImportFromImagesAsync(images);

            var pairs = _importService.MatchedPairs;
            if (pairs.Count == 0)
            {
                Console.WriteLine(
                    "[MultiView] no matched pairs - anchors fall back to an even spread across " +
                    "the capture.");
                return null;
            }

            var m = new int[images.Count, images.Count];
            foreach (var pair in pairs)
            {
                int a = pair.ImageIndexA, b = pair.ImageIndexB;
                if (a < 0 || b < 0 || a >= images.Count || b >= images.Count) continue;
                // Inliers where the pair was verified; raw matches are the fallback, since a pair
                // that was never verified still says something about overlap.
                int weight = pair.InlierCount > 0 ? pair.InlierCount : pair.Matches.Count;
                m[a, b] = weight;
                m[b, a] = weight;
            }

            Console.WriteLine(
                $"[MultiView] overlap from {pairs.Count} matched pairs in " +
                $"{(DateTime.UtcNow - t0).TotalSeconds:F1}s");
            return m;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MultiView] overlap measurement failed ({ex.Message}); using an even spread.");
            return null;
        }
    }

    /// <summary>
    /// Stand a reconstruction up, if its cameras agree on which way that is.
    ///
    /// DAv3 and COLMAP both recover geometry up to an ARBITRARY rotation - their world +Y is
    /// whatever the solver landed on. Everything downstream assumes +Y is up:
    /// <c>CameraController.UpdateCamera</c> rebuilds the camera's up as <c>Vector3.UnitY</c> on
    /// EVERY frame, so a capture pose's roll is discarded the moment anyone moves, and the
    /// yaw/pitch model is defined about world +Y too. MEASURED on Bathroom: the room rendered
    /// rotated about 90 degrees with the floor up the side of the screen, and tumbled when the
    /// camera moved. TJ found it by looking; no number in the run showed it, because the trainer
    /// renders from real CameraParams with the correct up and only the VIEWER forces +Y.
    ///
    /// A rigid rotation of the splats and their cameras TOGETHER, so it changes nothing about
    /// what any camera sees - gated by
    /// <c>SceneUpTests.AligningChangesNothingAboutWhatACameraSees</c>.
    ///
    /// ⚠ The cameras are mutated in place, deliberately: the training views hold the same
    /// objects, and a scene rotated away from its cameras would be worse than one left alone.
    /// </summary>
    /// <summary>
    /// Build the scene from a sparse SfM point cloud instead of per-view depth.
    ///
    /// The cloud arrives in the SAME world frame as the ground-truth cameras - both come out of
    /// one COLMAP reconstruction - so there is nothing to register, which is the point. Gravity
    /// alignment runs exactly as it does for the depth path, and moves the cameras with the
    /// splats.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)?>
        GenerateFromPointCloudAsync(float[] packed, int splatCount, IEnumerable<CameraParams> cameras)
    {
        if (splatCount <= 0) return null;
        var accelerator = _gpu.Accelerator!;

        SetStatus($"Uploading {splatCount:N0} splats from the sparse cloud...");
        var buf = accelerator.Allocate1D<float>((long)splatCount * SplatFormat.Floats);
        buf.CopyFromCPU(packed);
        await accelerator.SynchronizeAsync();

        await AlignToGravityAsync(buf, splatCount, cameras);

        LastPoseSource = "colmap";
        SetStatus($"Sparse-cloud init complete: {splatCount:N0} splats");
        return (buf, splatCount);
    }

    private async Task AlignToGravityAsync(
        MemoryBuffer1D<float, Stride1D.Dense> packed, int splatCount,
        IEnumerable<CameraParams> cameras)
    {
        var list = cameras.ToList();
        if (splatCount <= 0 || list.Count == 0) return;

        if (!WorldSpaceGeometry.TryEstimateSceneUp(list, out var sceneUp, out float agreement))
        {
            Console.WriteLine(
                $"[MultiView] gravity: the cameras agree only {agreement:P1} on an up direction " +
                $"(needs {WorldSpaceGeometry.MinUpAgreement:P0}), so the scene is left in the " +
                "solver's frame. A rig that rolls the camera has no meaningful up - TempleRing " +
                "sits at 49.1% - and rotating onto it would tip the scene over.");
            return;
        }

        var align = new Similarity3(
            1f, WorldSpaceGeometry.RotationBringingUpToY(sceneUp), Vector3.Zero);
        await _gaussianKernel.ApplySimilarityTransformAsync(
            packed, splatCount, align.Scale, align.Rotation, align.Translation);
        foreach (var cam in list) align.ApplyToCamera(cam);

        Console.WriteLine(
            $"[MultiView] gravity: scene up was ({sceneUp.X:F3},{sceneUp.Y:F3},{sceneUp.Z:F3}), " +
            $"agreement {agreement:P1} across {list.Count} cameras - rotated onto +Y");
    }

    /// <summary>One joint forward, returning null rather than throwing when the device refuses it.</summary>
    private async Task<DepthEstimationService.MultiViewDepthResult?> TryRunChunkAsync(
        IReadOnlyList<ImportedImage> images, MultiViewChunk chunk, int chunkSize,
        DepthEstimationService.MultiViewUploadCache? uploads = null)
    {
        var views = chunk.Views.Select(v => images[v]).ToList();
        try
        {
            var run = await _depthService.EstimateDepthMultiViewAsync(
                views, maxViews: chunkSize, uploads: uploads);
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
        DepthEstimationService.MultiViewDepthResult run, Similarity3 sim, bool adoptAnchors,
        int chunkIndex = 0)
    {
        int from = adoptAnchors ? 0 : chunk.AnchorCount;
        for (int slot = from; slot < chunk.Views.Length; slot++)
        {
            int g = chunk.Views[slot];
            if (cams[slot] == null || into.Cameras[g] != null) continue;

            sim.ApplyToCamera(cams[slot]!);
            into.Cameras[g] = cams[slot];
            into.FrameScales[g] = sim.Scale;
            into.ChunkOf[g] = chunkIndex;

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

        // Median and median-absolute-deviation, not (max-min)/mean: that statistic is decided by
        // the single worst pair and labelled TempleRing's tightly clustered ratios a 25.7%
        // "shape difference" on data whose fold was exact to 0.4%. Report the spread and let the
        // numbers speak; the verdict was a guess wearing a measurement's clothes.
        var sorted = ratios.OrderBy(r => r).ToList();
        float median = sorted[sorted.Count / 2];
        var dev = sorted.Select(r => MathF.Abs(r - median)).OrderBy(d => d).ToList();
        float relMad = median > 1e-6f ? dev[dev.Count / 2] / median : float.NaN;
        float worst = MathF.Max(MathF.Abs(sorted[^1] - median), MathF.Abs(sorted[0] - median))
                      / MathF.Max(median, 1e-6f);
        Console.WriteLine(
            $"[MultiView]   chunk {chunkIndex} anchor triangle: {string.Join("  ", parts)} " +
            $"-> median {median:F3}, typical deviation {relMad:P1}, worst pair {worst:P1}");

        // Each anchor's own estimate of the frame rotation, and how far the anchors disagree
        // pairwise. This is the number that calibrates MaxAnchorRotationRadians: a pair the
        // model placed consistently agrees to within noise, the anchor it guessed at does not.
        var rot = new List<(int view, Quaternion q)>();
        for (int a = 0; a < chunk.AnchorCount; a++)
        {
            if (cams[a] == null || !reference.TryGetValue(chunk.Views[a], out var refA)) continue;
            if (MultiViewChunkPlan.TryRotationBetweenCameras(cams[a]!, refA, out var q))
                rot.Add((chunk.Views[a], q));
        }
        if (rot.Count < 2) return;
        var rotParts = new List<string>();
        for (int a = 0; a < rot.Count; a++)
            for (int b = a + 1; b < rot.Count; b++)
                rotParts.Add(
                    $"{rot[a].view}-{rot[b].view} " +
                    $"{MultiViewChunkPlan.AngleBetween(rot[a].q, rot[b].q) * 180f / MathF.PI:F1}deg");
        Console.WriteLine(
            $"[MultiView]   chunk {chunkIndex} anchor rotation disagreement: {string.Join("  ", rotParts)} " +
            $"(limit {MultiViewChunkPlan.MaxAnchorRotationRadians * 180f / MathF.PI:F0}deg)");

        // What the fold actually chose between: each pair's inliers and its score, position and
        // orientation terms separately (each in units of its tolerance). This is the line that
        // says WHY an anchor was excluded; the disagreement line above only says by how much.
        var pairs = MultiViewChunkPlan.ScoreAnchorPairs(chunk, cams, reference);
        if (pairs.Count == 0) return;
        var pairParts = pairs.Select(p => p.InlierViews.Length == 0
            ? $"{p.ViewA}-{p.ViewB} gated ({p.PairRotationRadians * 180f / MathF.PI:F1}deg)"
            : $"{p.ViewA}-{p.ViewB} in[{string.Join(",", p.InlierViews)}] " +
              $"pos {p.PositionError:F2} + rot {p.RotationError:F2} = {p.PositionError + p.RotationError:F2}");
        Console.WriteLine($"[MultiView]   chunk {chunkIndex} pair scores: {string.Join("  ", pairParts)}");
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
        int fuseSub = ChooseSubsample(
            subsample, images.Select(im => (im.Width, im.Height)).ToList());

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

        // Same treatment for a ground-truth frame: COLMAP's world orientation is arbitrary too,
        // and drjohnson's cameras come back pointing along +/-Y. It refuses itself on a rig that
        // rolls the camera, so TempleRing - whose cameras sit 90 degrees off at 49.1% agreement -
        // is left exactly as it was.
        await AlignToGravityAsync(merged, actualTotal, cameras);

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

        // The budget applies here too. Without it 88 views of drjohnson emitted 14,011,008
        // splats and the merge lost the device.
        subsample = ChooseSubsample(subsample, images.Select(im => (im.Width, im.Height)).ToList());

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
                    // ONE pixel, not the whole map.
                    //
                    // This read the entire depth buffer back to the managed heap to sample a
                    // single value: 1024x673 floats is 2.75 MB per view, and at 88 views that is
                    // about 242 MB of churn inside a 2 GB WASM heap, for 88 floats' worth of
                    // information. The index is known before the copy, so copy from it.
                    int ix = Math.Clamp((int)MathF.Round(u), 0, depthResult.Width - 1);
                    int iy = Math.Clamp((int)MathF.Round(v), 0, depthResult.Height - 1);
                    long offset = (long)iy * depthResult.Width + ix;
                    float[] one = await depthResult.RawDepthGpu!.CopyToHostAsync<float>(offset, 1);
                    float raw = one[0];
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

        // The monocular fallback is an EARLY RETURN out of GenerateWithGroundTruthAsync, so the
        // alignment further down that method never runs for it. Same treatment here or a posed
        // room still arrives on its side.
        await AlignToGravityAsync(merged, actualTotal, cameras);

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
