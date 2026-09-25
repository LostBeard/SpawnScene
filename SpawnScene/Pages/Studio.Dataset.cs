using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Run the whole pipeline on an UNPOSED capture, which is what the product is actually for.
///
/// TempleRing ships a calibration file. Nothing else does - not Bathroom, not Skull, not
/// SouthBuilding, not SmallPlastic, not DinoSparseRing - so until now the optimiser could only
/// ever touch the one dataset that is least like the target. Bathroom is the honest test: 35
/// handheld phone frames of a room, 13 megapixels each, 34 portrait and one landscape, no EXIF
/// orientation, no poses.
///
/// This drives: load dataset -> pose cascade (SfM, then DAv3 extrinsics) -> depth init ->
/// photometric optimisation -> report. It reports rather than asserts, because the question it
/// exists to answer is "are the recovered poses good enough to optimise against", and a
/// pass/fail threshold would just be a guess dressed up as a gate.
///
/// Entry: <c>/studio?autotest=dataset&amp;name=Bathroom&amp;train=1600&amp;geom=1</c>
/// </summary>
public partial class Studio
{
    /// <summary>
    /// Load a dataset's sparse SfM cloud and turn it into packed splats, or null when it has none.
    ///
    /// Reports the scale distribution because that is the one thing a wrong unit system shows up
    /// in immediately: a cloud in different units than the cameras still loads, still renders,
    /// and is silently unusable.
    /// </summary>
    private async Task<float[]?> LoadSparseCloudAsync(string datasetName)
    {
        var manifest = await _importService.TryLoadManifestAsync(datasetName);
        if (manifest == null || string.IsNullOrEmpty(manifest.Points))
        {
            Console.WriteLine(
                $"[Dataset] {datasetName} has no sparse cloud in its manifest - falling back to " +
                "per-view depth init. Regenerate it with tools/colmap_to_dataset.py.");
            return null;
        }

        var bytes = await _importService.TryLoadPointCloudAsync(datasetName, manifest.Points);
        if (bytes == null)
        {
            Console.WriteLine($"[Dataset] FAIL: manifest lists {manifest.Points} but it did not fetch");
            return null;
        }

        var t = DateTime.UtcNow;
        var cloud = SparsePointCloudInit.Parse(bytes);
        var packed = SparsePointCloudInit.BuildPacked(cloud);
        int n = cloud.Count;

        var scales = new float[n];
        for (int i = 0; i < n; i++) scales[i] = packed[i * SplatFormat.Floats + SplatFormat.OffScale];
        Array.Sort(scales);
        Console.WriteLine(
            $"[Dataset] sparse cloud: {n:N0} points in {(DateTime.UtcNow - t).TotalSeconds:F1}s, " +
            $"splat scale p10 {scales[n / 10]:F4} median {scales[n / 2]:F4} p90 {scales[n * 9 / 10]:F4}, " +
            $"opacity {SparsePointCloudInit.InitialOpacity}");
        return packed;
    }

    /// <summary>
    /// Testing UI entry: run the selected dataset with train settings from the Testing panel.
    /// Skips harness capture delays so generate+train can be driven from the UI.
    /// </summary>
    private async Task OnRunDatasetFromUiAsync()
    {
        if (_pipelineBusy) return;
        _pipelineBusy = true;
        SetUiStatus($"Running {_uiDataset}…");
        BuildTestingUI();
        try
        {
            await RunDatasetAutotestAsync(
                _uiDataset, _uiTrainIters, _uiTrainGeom, maxTrainDimension: 1024,
                useGroundTruthPoses: _uiUseGtPoses, initFromPointCloud: _uiInitFromCloud,
                forUi: true);
        }
        finally
        {
            _pipelineBusy = false;
            if (_state == StudioState.SceneViewer)
                BuildViewerHudUI();
            else
            {
                if (string.IsNullOrEmpty(_statusMessage))
                    SetUiStatus("Ready");
                if (_state == StudioState.Testing)
                    BuildTestingUI();
            }
        }
    }

    private async Task RunDatasetAutotestAsync(
        string datasetName, int trainIters, bool optimiseGeometry, int maxTrainDimension,
        string posePreference = "dav3", int depthPatchesPerSide = DepthEstimationService.SafeMultiViewPatches,
        bool useGroundTruthPoses = false, bool initFromPointCloud = false, bool forUi = false)
    {
        Console.WriteLine(
            $"[Dataset] starting name={datasetName} train={trainIters} geom={optimiseGeometry} " +
            $"maxDim={maxTrainDimension} poses={posePreference} patches={depthPatchesPerSide}");
        if (forUi)
            SetUiStatus($"Loading {datasetName}…");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();

            // -- 1. Load the photographs --
            var t0 = DateTime.UtcNow;
            // Pairwise matching feeds SfM pose recovery. With ground-truth poses nothing reads
            // it, and at 132 images it is 8,646 GPU matches of pure wall clock.
            _importService.SkipPairMatching = useGroundTruthPoses;
            await _importService.LoadSampleDatasetAsync(datasetName);
            var images = _importService.Images.ToList();
            if (images.Count < 2)
            {
                Console.WriteLine($"[Dataset] FAIL: {datasetName} gave {images.Count} image(s)");
                if (forUi) SetUiStatus($"Error: {datasetName} gave {images.Count} image(s)");
                return;
            }
            Console.WriteLine(
                $"[Dataset] {images.Count} images in {(DateTime.UtcNow - t0).TotalSeconds:F1}s, " +
                $"first {images[0].Width}x{images[0].Height} ({images[0].SourceUrl})");
            if (forUi)
                SetUiStatus($"{datasetName}: {images.Count} images — posing…");

            // Bind a SQUARE; its side is the letterbox size and NativeAspect's long side.
            //
            // CORRECTED 2026-09-24: the 2026-09-20 "aspect-matched 32x43 grid measured 4 dB worse
            // (Bathroom held out 8.57 -> 4.34)" never fed the model a non-square tensor. The ML
            // pipeline read a non-square binding as its WIDTH and letterboxed into 448x448 - a
            // SMALLER square with fewer picture patches. The position-embedding theory built on it
            // was never tested. The pipeline now refuses a non-square binding, and the real
            // aspect-preserving path is DepthEstimationService.ResizeMode = NativeAspect.
            DepthEstimationService.SetSquareInput(depthPatchesPerSide);

            // -- 2. Poses + depth init, through the ordinary cascade --
            void OnStatus()
            {
                Console.WriteLine($"[Dataset] {_multiViewService.Status}");
                if (forUi) SetUiStatus(_multiViewService.Status);
            }
            _multiViewService.OnStatusChanged += OnStatus;
            (ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense> buf, int count)? result;
            try
            {
                t0 = DateTime.UtcNow;
                _multiViewService.PosePreference = posePreference;

                // Ground-truth poses, when the dataset ships them.
                //
                // This is the measurement Bathroom cannot give. Bathroom has no poses, so every
                // number it produces mixes "our poses are wrong" with "our optimiser is wrong"
                // and there is no way to tell them apart. A COLMAP-posed room lets the optimiser
                // be measured on its own, against a scene of the kind this project is FOR rather
                // than against an object on a turntable.
                var gtCameras = useGroundTruthPoses
                    ? await LoadGroundTruthCamerasAsync(datasetName, images)
                    : null;

                if (gtCameras != null)
                {
                    Console.WriteLine(
                        $"[Dataset] using ground-truth poses for all {gtCameras.Count} views " +
                        "- the pose cascade is skipped entirely");
                    _multiViewService.UseExternalCameras(
                        gtCameras.Select(c => (CameraParams?)c).ToArray(), "colmap");

                    // Sparse-cloud init, which is what 3DGS actually does.
                    //
                    // Unprojecting a monocular depth map per view gives one private shell per
                    // camera: measured on drjohnson, 91.9% of those splats were constrained by
                    // at most one view and 49.5% by none, so supervised loss fell while held-out
                    // loss rose and no optimiser setting could fix it. Every point in an SfM
                    // cloud is triangulated from two or more images by construction.
                    float[]? cloud = initFromPointCloud
                        ? await LoadSparseCloudAsync(datasetName)
                        : null;

                    result = cloud != null
                        ? await _multiViewService.GenerateFromPointCloudAsync(
                            cloud, cloud.Length / SplatFormat.Floats, gtCameras)
                        : await _multiViewService.GenerateWithGroundTruthAsync(
                            images, gtCameras, subsample: 2, edgeSharpness: 0.3f);
                }
                else
                {
                    result = await _multiViewService.GenerateAsync(images, subsample: 2, edgeSharpness: 0.3f);
                }
            }
            finally
            {
                _multiViewService.OnStatusChanged -= OnStatus;
            }

            if (result == null)
            {
                Console.WriteLine($"[Dataset] FAIL: generation returned nothing - {_multiViewService.Status}");
                if (forUi) SetUiStatus($"Error: {_multiViewService.Status}");
                return;
            }
            var (packedBuf, splatCount) = result.Value;
            Console.WriteLine(
                $"[Dataset] {splatCount:N0} splats in {(DateTime.UtcNow - t0).TotalSeconds:F1}s, " +
                $"pose source = {_multiViewService.LastPoseSource}");
            if (forUi)
                SetUiStatus($"Uploading {splatCount:N0} splats…");

            await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

            var scene = new GaussianScene
            {
                GpuSplatCount = splatCount,
                SourceName = "multi-view",
            };
            // The dataset loader fetched these over HTTP, so they are re-fetchable by URL and
            // do not need the project store.
            RecordTrainingViews(scene, images, fromProjectStore: false);

            // When COLMAP poses are on disk but we recovered our own, print how far off we are.
            // Held-out cross-match on DrJohnson dav3-chunked picked the wrong target on every
            // sampled view; GT with the same bookkeeping picked its own. This is that gap as a
            // number, not a theory.
            if (!string.Equals(_multiViewService.LastPoseSource, "colmap", StringComparison.Ordinal))
            {
                var gtForCompare = await LoadGroundTruthCamerasAsync(datasetName, images);
                if (gtForCompare != null)
                {
                    ReportPoseAccuracyVsGroundTruth(_multiViewService.LastCameras, gtForCompare);
                    ReportChunkAccuracyVsGroundTruth(
                        _multiViewService.LastCameras, _multiViewService.LastChunkOf, gtForCompare);
                }
            }

            _renderService.SetActiveSceneGpuLoaded(scene);
            _sceneManager.ActiveScene = scene;
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
            _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
            _state = StudioState.SceneViewer;
            if (forUi)
                BuildViewerHudUI();

            // Stand the viewer where one of the photographs was actually taken.
            //
            // Two reasons not to use FitToScene here. It leaves the camera at the origin looking
            // down -Z when there is nothing to fit to, and an SfM reconstruction can be anywhere
            // - Bathroom lands at z 0.49 to 1.44, entirely behind that - so the viewer renders
            // half a million splats at 58 fps and shows an empty frame. And its multi-view
            // branch aims at a HARDCODED TempleRing point (0.028, 0.042, -0.054), which has
            // nothing to do with a bathroom.
            //
            // A capture pose needs no heuristic: it is a viewpoint that definitely saw the
            // subject, because a photograph was taken from it.
            if (scene.TrainingCameras.Count > 0 && _cameraController != null)
            {
                var seat = scene.TrainingCameras[0];
                _cameraController.SetPose(seat.Position, seat.Forward, seat.Up);
                Console.WriteLine(
                    $"[Dataset] viewer seated at capture pose 0: " +
                    $"pos=({seat.Position.X:F3},{seat.Position.Y:F3},{seat.Position.Z:F3}) " +
                    $"fwd=({seat.Forward.X:F3},{seat.Forward.Y:F3},{seat.Forward.Z:F3})");
            }
            else
            {
                _cameraController?.FitToScene();
            }

            if (scene.TrainingViews.Count == 0)
            {
                // Still capture a frame. This branch has a full initialisation on screen - the
                // run that hit it had 242,440 splats - and a finding with no picture beside it is
                // how three runs got scored and committed while the viewer was blank. The
                // question about any reconstruction is what it LOOKS like, especially this one.
                if (!forUi)
                {
                    _hideUiOverlay = true;
                    await Task.Delay(1500);
                    Console.WriteLine("[Dataset] READY-FOR-CAPTURE");
                    await Task.Delay(2500);
                }
                Console.WriteLine(
                    "[Dataset] DONE (no optimisation): the cascade produced no usable poses. " +
                    "That is the finding, not a failure of this test.");
                if (forUi)
                    SetUiStatus("Done — no usable poses for training. Explore the init scene.");
                return;
            }

            // -- 3. Optimise --
            if (trainIters > 0)
            {
                if (forUi) SetUiStatus($"Training {trainIters} iters…");
                await TrainOnTrainingViewsAsync(
                    trainIters, optimiseGeometry: optimiseGeometry,
                    maxTrainDimension: maxTrainDimension);
            }

            if (!forUi)
            {
                // Announce before DONE so the harness can capture a frame of the finished scene.
                _hideUiOverlay = true;
                await Task.Delay(1500);
                Console.WriteLine("[Dataset] READY-FOR-CAPTURE");
                await Task.Delay(2500);

                // Then LOOK AROUND. A capture-pose render is close to a re-projection of the photo
                // it was taken from, so it flatters any reconstruction; the question a room has to
                // answer is what it looks like from somewhere nobody stood. TJ found the scene
                // rotated and tumbling this way while every number said it was fine.
                await CaptureFreeViewsAsync(scene);
            }

            Console.WriteLine("[Dataset] DONE");
            if (forUi)
                SetUiStatus($"Done — {splatCount:N0} splats. Click canvas to look around.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] FAIL: {ex}");
            if (forUi) SetUiStatus($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Render from poses nobody captured from: step away from the seat and look back.
    ///
    /// Deliberately modest offsets. The point is not a flythrough, it is to catch a
    /// reconstruction that only holds together from the exact pose it was seated at - which is
    /// what an unaligned world frame, a broken depth sort or a set of per-view shells all look
    /// like, and what a capture-pose screenshot cannot show.
    /// </summary>
    private async Task CaptureFreeViewsAsync(GaussianScene scene)
    {
        if (_cameraController == null || scene.TrainingCameras.Count == 0) return;

        var seat = scene.TrainingCameras[0];
        var fwd = Vector3.Normalize(seat.Forward);
        var up = Vector3.Normalize(seat.Up);
        var right = Vector3.Normalize(Vector3.Cross(fwd, up));

        // Scale the steps to the scene so this means the same thing on any capture.
        float span = 0.25f;
        if (scene.TrainingCameras.Count > 1)
        {
            var centroid = Vector3.Zero;
            foreach (var c in scene.TrainingCameras) centroid += c.Position;
            centroid /= scene.TrainingCameras.Count;
            float spread = 0f;
            foreach (var c in scene.TrainingCameras)
                spread = MathF.Max(spread, Vector3.Distance(c.Position, centroid));
            if (spread > 1e-3f) span = spread * 0.35f;
        }

        // Park at real CAPTURE poses first - intrinsics included - so a render can be put next to
        // the photograph taken from that exact pose. SUPERVISED views show what the trainer fitted;
        // HELD-OUT views were never fitted and are the honest test. TJ judges these by eye: a number
        // alone has declared "better" on runs that looked like mush.
        //
        // These used to print "READY-FOR-CAPTURE gtpose-N" with only SetPose (the viewer kept its own
        // FOV) and the harness regex only knew "free-", so not one of them was ever saved.
        var viewIdx = Enumerable.Range(0, scene.TrainingViews.Count).ToList();
        var picks = new List<(string Kind, int Index)>();
        foreach (var (kind, sel) in new[] { ("sup", true), ("held", false) })
        {
            var pool = viewIdx.Where(i => scene.TrainingViews[i].UsedForSupervision == sel).ToList();
            for (int k = 0; k < Math.Min(3, pool.Count); k++)
                picks.Add((kind, pool[(int)Math.Round(k * (pool.Count - 1) / (double)Math.Max(1, Math.Min(3, pool.Count) - 1))]));
        }
        foreach (var (kind, i) in picks.Distinct())
        {
            var tv = scene.TrainingViews[i];
            await StashTrainerRenderAsync($"view-{kind}-{i}", tv.Camera);
            await StashVideoPhotoAsync($"view-{kind}-{i}", tv.ImageName);
            await ParkOnGroundTruthPoseAsync($"{kind}-{i}", tv.Camera);
            Console.WriteLine(
                $"[Dataset] READY-FOR-CAPTURE view-{kind}-{i} {tv.ImageName} {tv.Camera.Width}x{tv.Camera.Height} " +
                $"turns={tv.QuarterTurns}");
            await Task.Delay(1800);
        }

        var moves = new (string Name, Vector3 Offset, float Yaw)[]
        {
            ("left",  -right * span, 0f),
            ("right",  right * span, 0f),
            ("back",  -fwd * span,   0f),
            ("up",     up * span * 0.5f, 0f),
            ("turned", Vector3.Zero, 0.35f),
        };

        foreach (var (name, offset, yaw) in moves)
        {
            var look = fwd;
            if (yaw != 0f)
            {
                var q = Quaternion.CreateFromAxisAngle(up, yaw);
                look = Vector3.Normalize(Vector3.Transform(fwd, q));
            }
            _cameraController.SetPose(seat.Position + offset, look, up);
            await Task.Delay(1200);
            Console.WriteLine($"[Dataset] READY-FOR-CAPTURE free-{name}");
            await Task.Delay(1800);
        }
    }

    /// <summary>
    /// Cameras from the dataset's own <c>poses.par</c>, ordered to match the loaded images.
    ///
    /// Reuses <see cref="WorldSpaceGeometry.ParseMiddleburyParams"/> rather than adding a second
    /// pose parser: COLMAP stores the same quantities in the same convention (world-to-camera R
    /// and t, OpenCV axes), so tools/colmap_to_dataset.py writes that format and this reads it.
    /// That converter proves its own conversion by reprojection - 0.586 px mean over 7,896
    /// observations on drjohnson - rather than trusting a quaternion ordering.
    ///
    /// Returns null when the dataset has no poses, which is the normal case for a handheld
    /// capture and not an error.
    /// </summary>
    private async Task<List<CameraParams>?> LoadGroundTruthCamerasAsync(
        string datasetName, IReadOnlyList<ImportedImage> images)
    {
        var manifest = await _importService.TryLoadManifestAsync(datasetName);
        if (manifest == null || string.IsNullOrEmpty(manifest.Poses))
        {
            Console.WriteLine($"[Dataset] {datasetName} ships no poses; recovering them instead.");
            return null;
        }

        try
        {
            var text = await _http.GetStringAsync($"datasets/{datasetName}/{manifest.Poses}");
            var parsed = WorldSpaceGeometry.ParseMiddleburyParams(
                text, manifest.Width, manifest.Height);
            var byName = parsed.ToDictionary(
                e => e.filename, e => e.camera, StringComparer.OrdinalIgnoreCase);

            var ordered = new List<CameraParams>(images.Count);
            foreach (var im in images)
            {
                if (!byName.TryGetValue(im.FileName, out var cam))
                {
                    Console.WriteLine(
                        $"[Dataset] {im.FileName} has no pose in {manifest.Poses} - falling back " +
                        "to the pose cascade rather than posing part of the capture.");
                    return null;
                }

                // The images are decoded at a capped size; the intrinsics must come with them or
                // every splat projects to the wrong place. ScaledTo carries the principal point
                // as well as the focal length, which is the classic half of this to get wrong.
                ordered.Add(cam.Width == im.Width && cam.Height == im.Height
                    ? cam
                    : cam.ScaledTo(im.Width, im.Height));
            }
            return ordered;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] could not read {manifest.Poses}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// dav3-chunked: per joint pass, two ground-truth fits. ALONE = that pass's views by themselves (its own
    /// accuracy; a similarity absorbs the fold). WITH REF = that pass plus the reference pass 0 in the world
    /// frame (the pass AND its fold). Alone good + with-ref bad names the fold; alone bad names the pass.
    /// </summary>
    static void ReportChunkAccuracyVsGroundTruth(
        CameraParams?[] estimated, int[] chunkOf, IReadOnlyList<CameraParams> groundTruth)
    {
        if (chunkOf.Length != estimated.Length || chunkOf.Length != groundTruth.Count) return;
        int chunks = chunkOf.Length == 0 ? 0 : chunkOf.Max() + 1;

        string Fit(Func<int, bool> member)
        {
            var est = new CameraParams?[estimated.Length];
            var gt = new CameraParams?[estimated.Length];
            int n = 0;
            for (int i = 0; i < estimated.Length; i++)
            {
                if (!member(i) || estimated[i] == null) continue;
                est[i] = estimated[i]; gt[i] = groundTruth[i]; n++;
            }
            if (n < 3 || !WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, gt, out var acc, out _, out _))
                return $"n={n} (no fit)";
            return $"n={n} {(acc.Spread > 0 ? acc.PositionRms / acc.Spread : float.NaN):P1} of spread, " +
                   $"fwd {acc.MedianForwardDeg:F1}deg";
        }

        for (int k = 0; k < chunks; k++)
        {
            int kk = k;
            Console.WriteLine(
                $"[Dataset]   pose-vs-GT chunk {k} " +
                $"[{string.Join(",", Enumerable.Range(0, chunkOf.Length).Where(i => chunkOf[i] == kk))}]: " +
                $"alone {Fit(i => chunkOf[i] == kk)}" +
                (k == 0 ? "" : $" | with ref {Fit(i => chunkOf[i] == kk || chunkOf[i] == 0)}"));
        }
    }

    /// <summary>
    /// Log estimated-vs-COLMAP camera accuracy after the best similarity alignment. Pair by
    /// image index. Does not change any camera.
    /// </summary>
    static void ReportPoseAccuracyVsGroundTruth(
        CameraParams?[] estimated, IReadOnlyList<CameraParams> groundTruth)
    {
        var gt = new CameraParams?[groundTruth.Count];
        for (int i = 0; i < groundTruth.Count; i++) gt[i] = groundTruth[i];
        if (!WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
                estimated, gt, out var acc, out var posFrac, out var fwdDeg))
        {
            Console.WriteLine(
                "[Dataset] pose-vs-GT: could not align (need >=3 paired cameras with a non-degenerate spread)");
            return;
        }

        Console.WriteLine(
            $"[Dataset] pose-vs-GT ({acc.Compared} cams, source aligned to COLMAP): " +
            $"scale {acc.Scale:F4}, position RMS {acc.PositionRms:F4} on spread {acc.Spread:F4} " +
            $"({(acc.Spread > 0 ? acc.PositionRms / acc.Spread : float.NaN):P1} of it); " +
            $"pos frac median {acc.MedianPosFrac:P1} p90 {acc.P90PosFrac:P1}; " +
            $"forward err median {acc.MedianForwardDeg:F1}deg p90 {acc.P90ForwardDeg:F1}deg");

        // Error along the capture order in 12 bins: smooth growth = drift (an under-constrained chain),
        // isolated spikes = individual bad views.
        if (posFrac.Length >= 12)
        {
            int bins = 12, per = (posFrac.Length + bins - 1) / bins;
            var prof = Enumerable.Range(0, bins)
                .Select(b => posFrac.Skip(b * per).Take(per).DefaultIfEmpty(float.NaN).Average())
                .Select(v => $"{v:P0}");
            Console.WriteLine($"[Dataset]   pose-vs-GT along capture order ({per} views/bin): {string.Join(" ", prof)}");
        }

        // Name the worst few so a bad fold / bad view is findable in the log.
        var order = Enumerable.Range(0, posFrac.Length).OrderByDescending(i => posFrac[i]).Take(5);
        foreach (int i in order)
            Console.WriteLine(
                $"[Dataset]   pose-vs-GT worst: view {i} pos {posFrac[i]:P1} of spread, " +
                $"forward {fwdDeg[i]:F1}deg");
    }
}
