using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// Project/scene CRUD, scene generation, file handling, thumbnails
public partial class Studio
{
    /// <summary>
    /// The reference's held-out rule (<c>llffhold</c>): hold out every posed view whose index % N == 0. 0 (default) keeps
    /// SpawnScene's every-fourth split. <c>&amp;llffhold=8</c> is the published 3DGS protocol.
    /// </summary>
    public static int LlffHold { get; set; }

    private async void OnNewProjectClicked()
    {
        Console.WriteLine("[Studio] New Project button clicked");
        try
        {
            int count = (_projects?.Count ?? 0) + 1;
            string name = $"Project {count}";
            Console.WriteLine($"[Studio] Creating project '{name}'...");
            var project = await _projectService.CreateProjectAsync(name);
            Console.WriteLine($"[Studio] Project created in OPFS, refreshing list...");
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = project;
            _state = StudioState.ProjectDetail;
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Created project '{project.Name}' ({project.Id}), {_projects.Count} total");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Error creating project: {ex}");
        }
    }

    private void OnOpenProject(Project project)
    {
        _activeProject = project;
        _state = StudioState.ProjectDetail;
        _statusMessage = null;
        BuildProjectDetailUI();
        Console.WriteLine($"[Studio] Opened project '{project.Name}'");
    }

    private async Task OnRemoveSource(ProjectSource source)
    {
        if (_activeProject == null) return;
        try
        {
            await _projectService.RemoveSourceAsync(_activeProject.Id, source.FileName);

            // Remove cached thumbnail
            string srcKey = SourceThumbKey(_activeProject.Id, source.FileName);
            if (_thumbnailCache.TryGetValue(srcKey, out var cached))
            {
                cached.view.Dispose();
                cached.tex.Destroy();
                cached.tex.Dispose();
                _thumbnailCache.Remove(srcKey);
            }

            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.FirstOrDefault(p => p.Id == _activeProject.Id) ?? _activeProject;
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Removed source: {source.FileName}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Error removing source: {ex.Message}");
        }
    }

    private async Task OnDeleteScene(ProjectScene scene)
    {
        if (_activeProject == null) return;
        try
        {
            await _projectService.DeleteSceneAsync(_activeProject.Id, scene.Id);
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.FirstOrDefault(p => p.Id == _activeProject.Id) ?? _activeProject;
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Deleted scene ({scene.SplatCount:N0} splats)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Error deleting scene: {ex.Message}");
        }
    }

    private async Task OnDeleteProject(Project project)
    {
        try
        {
            await _projectService.DeleteProjectAsync(project.Id);
            _projects = await _projectService.ListProjectsAsync();
            if (_activeProject?.Id == project.Id)
                _activeProject = null;
            BuildProjectBrowserUI();
            Console.WriteLine($"[Studio] Deleted project '{project.Name}'");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Error deleting project: {ex.Message}");
        }
    }

    // ─── Scene Generation ───

    private async void OnGenerateSceneClicked()
    {
        if (_activeProject == null || _activeProject.Sources.Count == 0) return;

        // Multi-view path: 2+ images → SfM + per-view depth fusion
        if (_activeProject.Sources.Count >= 2)
        {
            await GenerateMultiViewScene();
            return;
        }

        // Single-image path (existing)
        _statusMessage = "Loading depth model...";
        BuildProjectDetailUI();

        try
        {
            // Load depth model from project settings (or default). Unknown / retired ids fall back.
            var targetModel = _activeProject.Settings.DepthModel ?? DepthEstimationService.DefaultModelId;
            if (!DepthEstimationService.AvailableModels.Any(m => m.Id == targetModel))
                targetModel = DepthEstimationService.DefaultModelId;
            if (!_depthService.IsReady || _depthService.LoadedModelId != targetModel)
            {
                await _depthService.LoadModelAsync(targetModel);
                if (!_depthService.IsReady)
                {
                    _statusMessage = $"Error: {_depthService.Status}";
                    BuildProjectDetailUI();
                    return;
                }
            }

            // Use the first source image
            var source = _activeProject.Sources[0];
            _statusMessage = $"Loading {source.FileName}...";
            BuildProjectDetailUI();

            var imageBytes = await _projectService.GetSourceAsync(_activeProject.Id, source.FileName);
            if (imageBytes == null) { _statusMessage = "Error: could not read source image"; BuildProjectDetailUI(); return; }

            // Extract EXIF focal length before decoding (JPEG headers only)
            var exifFocal = ExifReader.ExtractFocalLength(imageBytes);

            // Decode image
            _statusMessage = "Decoding image...";
            BuildProjectDetailUI();

            using var blob = new Blob(new byte[][] { imageBytes }, new BlobOptions { Type = "image/jpeg" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
            int w = (int)bitmap.Width;
            int h = (int)bitmap.Height;

            // Rasterize to RGBA via OffscreenCanvas — the pixels stay in the JS heap.
            using var osc = new OffscreenCanvas(w, h);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var dataArray = imageData.Data; // JS Uint8ClampedArray — RGBA pixels, JS-side

            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();

            // The photo on the GPU: the camera estimate and the unprojection both read it in place.
            var rgbaGpuBuf = _gpuService.WebGPUAccelerator.Allocate1D<int>(w * h);
            SpawnDev.ILGPU.ML.Preprocessing.MediaInterop.UploadToDevice(dataArray, rgbaGpuBuf);

            // Super-resolution (Studio.SuperRes): a small photo tripled before anything reads it, so the camera estimate,
            // depth and the splat grid all see the larger image.
            bool upscaled = false;
            var srMode = SuperResOverride ?? _activeProject.Settings.SuperResolution;
            if (ShouldSuperResolve(srMode, w, h))
            {
                _statusMessage = "Super-resolution (x3)...";
                BuildProjectDetailUI();
                if (await SuperResolveAsync(rgbaGpuBuf, w, h) is { } up)
                {
                    rgbaGpuBuf.Dispose();
                    (rgbaGpuBuf, w, h) = up;
                    upscaled = true;
                }
            }

            // Build camera params from EXIF (or fall back to heuristic)
            var camera = CameraParams.CreateFromExif(w, h, exifFocal);
            var focalSource = exifFocal?.FocalLength35mm is > 0 ? "EXIF 35mm"
                : exifFocal?.FocalLengthMm is > 0 and < 10f ? $"phone estimate ({exifFocal.FocalLengthMm:F1}mm * 7x)"
                : "heuristic 1.2x";
            // Without a real focal length (no EXIF 35mm equivalent: generated images, stripped metadata) ask DAv3 for
            // the camera. The 1.2x-the-long-side guess is a ~45 degree view, far narrower than real photos: measured
            // against COLMAP ground truth it was +102% on Truck (979 px, f=582) and +54% on DrJohnson (1332 px,
            // f=1035) - a room built that narrow for its depth looked 1.5-2x too deep (TJ, 2026-10-03: "single photo
            // generated splat scenes seem to have exaggerated depth"). DAv3 on one photo: +10..+29%, median ~+14%.
            if (exifFocal?.FocalLength35mm is not > 0 && await EstimateSingleViewIntrinsicsAsync(rgbaGpuBuf, w, h, source.FileName) is { } k)
            {
                camera.FocalX = k[0]; camera.FocalY = k[4];
                camera.CenterX = k[2]; camera.CenterY = k[5];
                focalSource = "DAv3 camera estimate";
            }
            Console.WriteLine($"[EXIF] {source.FileName}: fx={camera.FocalX:F1}px ({focalSource})");

            // Depth: JS TypedArray → EstimateGpuRawAsync(TypedArray) via the service (no managed Read<int>).
            // Gaussian path below: UploadToDevice keeps RGBA on GPU for unprojection.
            _statusMessage = "Estimating depth...";
            BuildProjectDetailUI();
            // Gaussian path: the RGBA uploaded above (JS TypedArray → GPU directly, no .NET heap), upscaled if SR ran.
            using var gpuImage = new GpuImage
            {
                PackedRgba = rgbaGpuBuf,
                Width = w,
                Height = h,
                FileName = source.FileName,
            };
            var depthResult = upscaled
                ? await _depthService.EstimateDepthAsync(gpuImage)
                : await _depthService.EstimateDepthFromJsRgbaAsync(dataArray, w, h);
            if (depthResult == null) { _statusMessage = "Error: depth estimation failed"; BuildProjectDetailUI(); return; }

            // Capture depth map for visualization (before Gaussian kernel consumes the buffer)
            await CaptureDepthMapAsync(depthResult);

            // Generate Gaussians
            _statusMessage = "Generating Gaussians...";
            BuildProjectDetailUI();

            int subsample = _activeProject.Settings.Subsample;
            float edgeSharpness = _activeProject.Settings.EdgeSharpness;
            var (packedBuf, splatCount) = await _gaussianKernel.GeneratePackedGpuBufferAsync(
                depthResult, gpuImage, subsample, edgeSharpness, camera);

            // Upload to renderer
            _statusMessage = $"Uploading {splatCount:N0} splats...";
            BuildProjectDetailUI();

            _gpuRenderer.UseRgbColours();
            await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

            var scene = new GaussianScene
            {
                GpuSplatCount = splatCount,
                SourceName = "depth-splat", // signals FitCameraToScene to use depth-splat positioning
            };
            // Mark as GPU-loaded BEFORE setting ActiveScene (prevents redundant CPU upload)
            _renderService.SetActiveSceneGpuLoaded(scene);
            _sceneManager.ActiveScene = scene;

            // Save scene data to OPFS: GPU → CPU readback (justified: file I/O)
            _statusMessage = $"Saving {splatCount:N0} splats to storage...";
            BuildProjectDetailUI();

            // GPU → JS Uint8Array → OPFS. The packed splats stay in the JS heap and never enter
            // the .NET/WASM managed heap (a 5K image is ~14.7M splats ≈ 588 MB — marshalling that
            // into a byte[] OOMs the managed heap).
            using var packedU8 = await _gpuRenderer.ReadPackedUint8ArrayAsync(splatCount);
            if (packedU8 != null)
            {
                var projectScene = new ProjectScene
                {
                    SplatCount = splatCount,
                    FloatsPerSplat = SplatFormat.Floats,
                    QualityPreset = _activeProject.Settings.QualityPreset,
                };
                await _projectService.SaveSceneAsync(_activeProject.Id, projectScene, packedU8);
                _viewedProjectScene = projectScene;
                Console.WriteLine($"[Studio] Scene saved to OPFS: {packedU8.Length / (1024 * 1024):F1} MB");
            }
            else
            {
                // Save metadata only if readback failed
                var projectScene = new ProjectScene
                {
                    SplatCount = splatCount,
                    FloatsPerSplat = SplatFormat.Floats,
                    QualityPreset = _activeProject.Settings.QualityPreset,
                    SizeBytes = (long)splatCount * SplatFormat.Floats * sizeof(float),
                };
                _activeProject.Scenes.Add(projectScene);
                await _projectService.UpdateProjectAsync(_activeProject);
                Console.WriteLine("[Studio] Warning: GPU readback failed, scene not saved to OPFS");
            }

            // Schedule thumbnail capture after scene has converged (~30 frames at 60fps = 0.5s)
            var lastScene = _activeProject.Scenes.LastOrDefault();
            if (lastScene != null)
            {
                _pendingThumbnailProjectId = _activeProject.Id;
                _pendingThumbnailSceneId = lastScene.Id;
                _thumbnailDelayFrames = 30;
            }

            _statusMessage = null;
            _state = StudioState.SceneViewer;
            _cameraController?.FitToScene();
            BuildViewerHudUI();

            Console.WriteLine($"[Studio] Generated scene: {splatCount:N0} splats");
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Generation error: {ex}");
        }
    }

    // ─── Dataset Testing ───

    /// <summary>
    /// Turn whatever the pose cascade produced into training views.
    ///
    /// Until this existed only the TempleRing path - which ships a calibration file - could be
    /// optimised, so none of the real captures could be. Orientation comes from the POSE and
    /// not from EXIF: Bathroom carries no Orientation tag at all, and Skull and SouthBuilding
    /// report "normal" while a third of the frames are not.
    ///
    /// Fallback poses are deliberately refused. They are a placeholder, not a measurement, and
    /// fitting a scene to them produces a confident reconstruction of a fiction.
    /// </summary>
    private void RecordTrainingViews(
        GaussianScene scene, IReadOnlyList<ImportedImage> images, bool fromProjectStore, bool holdOut = true)
    {
        // Orientation correction needs a gravity-aligned world frame, and only a calibration
        // file gives one. SfM recovers geometry up to an arbitrary rotation and DAv3 extrinsics
        // are relative, so "which way is up" in either frame is noise - on Bathroom it asked
        // for four different quarter turns across six cameras of the same room, and the
        // resulting mix of portrait and landscape cameras cannot share a trainer viewport.
        //
        // It is also unnecessary here: a phone writes its frames the right way up.
        const bool poseFrameHasGravity = false;
        var poses = _multiViewService.LastCameras;
        string poseSource = _multiViewService.LastPoseSource;

        // Reject the known-bad source, do not whitelist the known-good ones.
        //
        // This WAS a whitelist, and it silently dropped "dav3-chunked" - the same DAv3
        // extrinsics in the same world frame, from several passes instead of one - because it
        // was not spelled like the two entries that already existed. I added that string to the
        // list and wrote a comment here saying a whitelist drops any new one. Then I added
        // "colmap" and it happened again, same place, same day, and a 397-second run produced no
        // training at all. Adding an entry was never the fix.
        //
        // The semantics are a rejection, not an admission: a FALLBACK pose is a placeholder
        // invented from image dimensions and nothing else is. Any source that recovered real
        // poses or was handed them is trainable, including sources that do not exist yet.
        bool posesArePlaceholders = poseSource is "fallback" or "none" or "";
        if (posesArePlaceholders || poses.Length == 0)
        {
            Console.WriteLine(
                $"[Studio] pose source '{poseSource}' gives nothing to train against - the " +
                "optimiser needs real poses, and a fallback pose is a placeholder");
            return;
        }

        int held = 0, posed = 0, unposed = 0;
        for (int i = 0; i < images.Count && i < poses.Length; i++)
        {
            var cam = poses[i];
            if (cam == null) { unposed++; continue; }

            string name = fromProjectStore ? images[i].FileName : images[i].SourceUrl;
            if (string.IsNullOrEmpty(name)) { unposed++; continue; }

            int turns = poseFrameHasGravity ? ImageOrientation.QuarterTurnsToUpright(cam) : 0;

            // Hold every fourth posed view out, so the run reports a novel-view number rather
            // than a reconstruction of its own input. &llffhold=N instead uses the reference's split
            // (dataset_readers: test = image index % llffhold == 0, images sorted by name), for numbers
            // comparable with the published ones.
            // holdOut: false (a user's own project) trains on EVERY photo - the held-out split is for measuring.
            // By IMAGE index (i), not by posed count: with our own SfM a camera can go unplaced, and counting posed views
            // shifted every later held-out photo by one - bicycle's own-SfM run and its COLMAP-pose run then shared only
            // 2 of 25 test photos, and the comparison read as a 5 dB pose gap (2026-10-04). An unplaced test photo is
            // simply not scored.
            bool supervise = !holdOut || (LlffHold > 0 ? i % LlffHold != 0 : posed % 4 != 3);
            if (!supervise) held++;
            posed++;

            scene.TrainingViews.Add(new TrainingView
            {
                Camera = turns != 0 ? ImageOrientation.Rotate(cam, turns) : cam,
                ImageName = name,
                FromProjectStore = fromProjectStore,
                UsedForInit = true,
                QuarterTurns = turns,
                UsedForSupervision = supervise,
                SourceLongestSide = Math.Max(images[i].SourceWidth, images[i].SourceHeight),
            });
            scene.TrainingCameras.Add(cam);
        }

        var turnCounts = scene.TrainingViews
            .GroupBy(v => v.QuarterTurns)
            .OrderBy(g => g.Key)
            .Select(g => $"{g.Key}x{g.Count()}");
        Console.WriteLine(
            $"[Studio] supervision from {poseSource}: {posed} of {images.Count} views posed " +
            $"({unposed} not), {held} held out, quarter turns [{string.Join(", ", turnCounts)}]");

        // MEASURED 2026-09-23 DrJohnson dav3-chunked: 20/44 posed (8 chunks rejected). Training
        // still runs, but held-out cannot climb while half the capture never entered the frame.
        // Do not refuse here - a thin pose set is still better than inventing cameras - but the
        // product path must surface it so the cascade gap is not mistaken for an optimiser miss.
        if (images.Count >= 4 && posed * 2 < images.Count)
        {
            Console.WriteLine(
                $"[Studio] WARNING: only {posed}/{images.Count} views posed in one frame - " +
                "held-out quality will be cascade-limited, not optimiser-limited. " +
                "See Research/handoff-pose-cascade-drjohnson-2026-09-23.md");
        }
    }

    private async Task GenerateFromTempleRingAsync(int onlyView = -1, bool globalScale = false,
        bool upright = false)
    {
        if (_activeProject == null) return;

        _statusMessage = "Loading TempleRing dataset...";
        BuildProjectDetailUI();

        try
        {
            // Load the parameter file
            var parText = await _http.GetStringAsync("datasets/TempleRing/templeR_par.txt");

            // Parse camera params — TempleRing images are 640x480
            var gtCameras = MultiViewGenerationService.ParseMiddleburyParams(parText, 640, 480);
            Console.WriteLine($"[Studio] TempleRing: {gtCameras.Count} cameras in par file");

            // Discover which par entries exist on disk, then pick 4 views spaced around the ring.
            // (First 4 consecutive frames are near-identical viewpoints → floating near-duplicates.)
            // DAv3 joint multi-view: 4 spaced ring views (6+ with minViews=3 over-culls relative MDE).
            const int maxImages = 4;
            var available = new List<(string filename, CameraParams cam)>();
            foreach (var (filename, cam) in gtCameras)
            {
                try
                {
                    // GET (not HEAD): some static hosts lie on HEAD. Reject SPA HTML fallbacks.
                    using var resp = await _http.GetAsync($"datasets/TempleRing/{filename}",
                        HttpCompletionOption.ResponseHeadersRead);
                    if (!resp.IsSuccessStatusCode) continue;
                    var ctype = resp.Content.Headers.ContentType?.MediaType ?? "";
                    var len = resp.Content.Headers.ContentLength ?? -1;
                    if (!ctype.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) continue;
                    if (len >= 0 && len < 1024) continue; // empty / stub
                    available.Add((filename, cam));
                }
                catch { /* missing */ }
            }
            Console.WriteLine($"[Studio] TempleRing: {available.Count} images on disk");
            if (available.Count < 2)
            {
                _statusMessage = "TempleRing: need at least 2 images on disk.";
                BuildProjectDetailUI();
                return;
            }

            var pickIdx = WorldSpaceGeometry.PickFarthestCameras(
                available.Select(a => a.cam).ToList(), Math.Min(maxImages, available.Count));
            Console.WriteLine($"[Studio] TempleRing farthest picks: [{string.Join(",", pickIdx)}]");

            var images = new List<ImportedImage>();
            var cameras = new List<CameraParams>();

            foreach (int i in pickIdx)
            {
                var (filename, cam) = available[i];
                _statusMessage = $"Loading {filename} ({images.Count + 1}/{pickIdx.Count})...";
                BuildProjectDetailUI();

                byte[] bytes = await _http.GetByteArrayAsync($"datasets/TempleRing/{filename}");
                using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/png" });
                using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
                int w = (int)bitmap.Width;
                int h = (int)bitmap.Height;

                using var osc = new OffscreenCanvas(w, h);
                using var ctx = osc.Get2DContext();
                ctx.DrawImage(bitmap, 0, 0);
                using var imageData = ctx.GetImageData(0, 0, w, h);
                using var dataArray = imageData.Data;
                var rgba = dataArray.ReadBytes();

                cam.Width = w;
                cam.Height = h;

                // Stand the photograph up before the depth model sees it. Every TempleRing image
                // is a quarter turn off level, and monocular depth networks are trained on
                // upright photographs. The direction is NOT uniform across the ring - 31 of the
                // 47 entries need one turn and 16 need three - so it is decided per camera. The
                // camera is turned with the pixels, so nothing downstream changes meaning.
                // See ImageOrientation.
                int turns = upright ? ImageOrientation.QuarterTurnsToUpright(cam) : 0;
                if (turns != 0)
                {
                    rgba = ImageOrientation.RotateRgba(rgba, w, h, turns);
                    cam = ImageOrientation.Rotate(cam, turns);
                    (w, h) = (cam.Width, cam.Height);
                }

                images.Add(new ImportedImage { FileName = filename, Width = w, Height = h, RgbaPixels = rgba });
                cameras.Add(cam);
                if (turns != 0 && images.Count == 1)
                    Console.WriteLine($"[Studio] TempleRing upright: {turns} quarter turn(s), now {w}x{h}");
                Console.WriteLine($"[Studio] TempleRing pick[{images.Count - 1}]={filename} pos=({cam.Position.X:F3},{cam.Position.Y:F3},{cam.Position.Z:F3})");
            }

            // Full-res GT regression — TempleRing is only 4×640×480; prefer subsample 1.
            int subsample = 1;
            float edgeSharpness = _activeProject.Settings.EdgeSharpness;

            void OnStatus() { _statusMessage = _multiViewService.Status; BuildProjectDetailUI(); }
            _multiViewService.OnStatusChanged += OnStatus;

            try
            {
                var result = await _multiViewService.GenerateWithGroundTruthAsync(
                    images, cameras, subsample, edgeSharpness, onlyView, globalScale);

                if (result == null) { _statusMessage = _multiViewService.Status; BuildProjectDetailUI(); return; }

                var (packedBuf, splatCount) = result.Value;
                _gpuRenderer.UseRgbColours();
                await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

                var scene = new GaussianScene
                {
                    GpuSplatCount = splatCount,
                    SourceName = "multi-view",
                };
                foreach (var cam in cameras)
                    scene.TrainingCameras.Add(cam);

                // Every posed photo on disk becomes photometric supervision, not just the ones
                // that seeded geometry. The 4-view cap comes from the joint-depth model; training
                // only needs an image plus a pose, and TempleRing ships 16 of those. Nothing
                // consumes this yet - the optimiser does - so it cannot change current output.
                var initNames = pickIdx.Select(i => available[i].filename).ToHashSet(StringComparer.OrdinalIgnoreCase);

                // Hold every third non-init view out of SUPERVISION entirely. It is still posed,
                // still loaded and still scored - its pixels just never reach the loss. Training
                // on all sixteen and then reporting the twelve that did not seed depth as
                // "held out" measures reconstruction of the training set, which is a much
                // larger number than novel-view quality and not the question being asked.
                // Init views are never held out: their geometry is already baked in, so they
                // could not be novel to anything.
                int nonInitSeen = 0;
                var heldOut = new List<string>();
                foreach (var (filename, cam) in available)
                {
                    bool isInit = initNames.Contains(filename);
                    bool supervise = true;
                    if (!isInit)
                    {
                        supervise = nonInitSeen % 3 != 0;
                        nonInitSeen++;
                    }
                    if (!supervise) heldOut.Add(filename);

                    int turns = upright ? ImageOrientation.QuarterTurnsToUpright(cam) : 0;
                    scene.TrainingViews.Add(new TrainingView
                    {
                        Camera = turns != 0 ? ImageOrientation.Rotate(cam, turns) : cam,
                        ImageName = $"datasets/TempleRing/{filename}",
                        FromProjectStore = false,
                        UsedForInit = isInit,
                        QuarterTurns = turns,
                        UsedForSupervision = supervise,
                    });
                }
                // The harness reads this line to decide what to score. Emitted even when
                // training is off, so a run can never disagree with the scorer about it.
                Console.WriteLine($"[Studio] heldout: {string.Join(", ", heldOut)}");
                Console.WriteLine(
                    $"[Studio] TempleRing supervision: " +
                    $"{scene.TrainingViews.Count(v => v.UsedForSupervision)} of {scene.TrainingViews.Count} " +
                    $"posed views ({initNames.Count} also used for depth init), " +
                    $"{scene.TrainingViews.Count(v => !v.UsedForSupervision)} held out");

                _renderService.SetActiveSceneGpuLoaded(scene);
                _sceneManager.ActiveScene = scene;

                // Sharper demo presentation: sorted alpha for dense MVS (stochastic looks holey at <200k).
                _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
                _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
                _gpuRenderer.StochasticSPP = 4;

                _statusMessage = null;
                _state = StudioState.SceneViewer;
                _cameraController?.FitToScene();
                BuildViewerHudUI();

                Console.WriteLine($"[Studio] TempleRing GT scene: {splatCount:N0} splats ({images.Count}-view mvs fuse)");
            }
            finally
            {
                _multiViewService.OnStatusChanged -= OnStatus;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] TempleRing error: {ex}");
        }
    }

    // ─── Multi-View Generation ───

    private async Task GenerateMultiViewScene()
    {
        if (_activeProject == null) return;
        // The depth input the measured runs used (the dataset path sets it per run; nothing else may leak in).
        DepthEstimationService.SetSquareInput(DepthEstimationService.SafeMultiViewPatches);

        _statusMessage = "Preparing multi-view pipeline...";
        BuildProjectDetailUI();

        List<ImportedImage>? gpuImages = null;
        if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();
        try
        {
            // Load every source photo STRAIGHT TO THE GPU: OPFS File -> browser decode -> canvas resize -> device buffer
            // (MediaInterop.DecodeToDeviceAsync). Neither the JPEG nor its pixels enter the .NET heap.
            //
            // 🔴 This used to read each photo's bytes into a byte[], decode it at FULL resolution and ReadBytes() the
            // RGBA onto the managed heap: 35 phone photos (~12 MP) is ~1.7 GB against the 2 GB WASM ceiling, and TJ's
            // 35-photo Bathroom project on gh-pages died with "Garbage collector could not allocate 16384u bytes of
            // memory for major heap section" (2026-09-28). TJ: data in .NET WASM that does not need to be there.
            // 1024 on the longest edge (ImageImportService.MaxImportDimension): features detect at 1024, depth resizes
            // to its own input, and training reloads its targets from the stored photos at its own size.
            var images = new List<ImportedImage>();
            gpuImages = images;
            var accel = _gpuService.WebGPUAccelerator;
            foreach (var source in _activeProject.Sources)
            {
                _statusMessage = $"Loading {source.FileName}...";
                BuildProjectDetailUI();

                // The OPFS File IS the image's Source from here on (disposed with the image), so no using.
                var file = await _projectService.GetSourceFileAsync(_activeProject.Id, source.FileName);
                if (file == null) continue;

                // EXIF lives in the first APP1 segment (<= 64 KB); only a prefix crosses into .NET for it.
                ExifReader.ExifFocalLength? exifFocal = null;
                try
                {
                    using var head = file.Slice(0, 256 * 1024);
                    using var headBuf = await head.ArrayBuffer();
                    using var headBytes = new Uint8Array(headBuf);
                    exifFocal = ExifReader.ExtractFocalLength(headBytes.ReadBytes());
                }
                catch (Exception ex) { Console.WriteLine($"[Studio] {source.FileName}: no EXIF ({ex.Message})"); }

                MemoryBuffer1D<int, Stride1D.Dense> rgba;
                int w, h, srcW, srcH;
                try
                {
                    (rgba, w, h, srcW, srcH) = await SpawnDev.ILGPU.ML.Preprocessing.MediaInterop.DecodeToDeviceAsync(
                        file, accel, ImageImportService.MaxImportDimension);
                }
                catch (Exception ex) { Console.WriteLine($"[Studio] could not decode {source.FileName} - skipped ({ex.Message})"); file.Dispose(); continue; }
                // Only the SIZE is needed now: release the pixels. Every consumer decodes from the Source on demand and
                // releases after, so the GPU never holds every photo at once either.
                rgba.Dispose();

                // EXIF focal is converted to pixels AT this size (CreateFromExif scales by width/height).
                var camera = CameraParams.CreateFromExif(w, h, exifFocal);

                images.Add(new ImportedImage
                {
                    FileName = source.FileName,
                    Width = w,
                    Height = h,
                    Source = file,
                    DecodeMaxEdge = ImageImportService.MaxImportDimension,
                    SourceWidth = srcW,
                    SourceHeight = srcH,
                    EstimatedCamera = camera,
                });
            }

            if (images.Count < 2)
            {
                _statusMessage = "Need at least 2 images for multi-view generation.";
                BuildProjectDetailUI();
                return;
            }

            // Subscribe to status updates
            void OnStatus() { _statusMessage = _multiViewService.Status; BuildProjectDetailUI(); }
            _multiViewService.OnStatusChanged += OnStatus;

            try
            {
                // Note: UploadSceneFromGpuBuffer → EnsureSplatBuffer destroys old buffers automatically

                int subsample = _activeProject.Settings.Subsample;
                float edgeSharpness = _activeProject.Settings.EdgeSharpness;

                // The project's keypoint budget for this run (a static the dataset harness sets from &lgk), restored after.
                int savedKeypoints = LearnedFeatureMatcher.KeypointBudget;
                LearnedFeatureMatcher.KeypointBudget = _activeProject.Settings.LearnedKeypoints;
                (MemoryBuffer1D<float, Stride1D.Dense> Buffer, int Count)? result;
                try { result = await _multiViewService.GenerateAsync(images, subsample, edgeSharpness); }
                finally { LearnedFeatureMatcher.KeypointBudget = savedKeypoints; }
                if (result == null)
                {
                    _statusMessage = _multiViewService.Status;
                    BuildProjectDetailUI();
                    return;
                }

                var (packedBuf, splatCount) = result.Value;

                // Upload to renderer
                _statusMessage = $"Uploading {splatCount:N0} splats...";
                BuildProjectDetailUI();

                _gpuRenderer.UseRgbColours();
                await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

                var scene = new GaussianScene
                {
                    GpuSplatCount = splatCount,
                    // Hybrid path may lack TrainingCameras; FitToScene uses depth-splat origin then.
                    SourceName = "depth-splat",
                };

                // Record what the pose cascade settled on, so the optimiser can run on captures
                // that have no calibration file. A user's own project trains on EVERY photo
                // (holdOut: false) - the held-out split exists to measure, not to build a scene.
                //
                // Fallback poses are not recorded on purpose: they are a placeholder, not a
                // measurement, and training against them would fit the scene to a fiction.
                // A measurement run (&llffhold=N) holds out every Nth photo here too, so the scene can be scored on
                // photos it never saw; a user's project never sets it.
                RecordTrainingViews(scene, images, fromProjectStore: true, holdOut: LlffHold > 0);

                _renderService.SetActiveSceneGpuLoaded(scene);
                _sceneManager.ActiveScene = scene;

                // Straight to the viewer, standing where a photo was taken (FitToScene can leave an
                // SfM reconstruction behind the camera - see the dataset path), so training can be watched.
                _statusMessage = null;
                _state = StudioState.SceneViewer;
                SeatViewerAtCapturePose(scene);
                BuildViewerHudUI();

                int trainIters = _activeProject.Settings.TrainIterations;
                int trainedIters = 0;
                if (trainIters > 0 && scene.TrainingViews.Count > 0)
                    trainedIters = TrainingBlocks.Columns * TrainingBlocks.Rows > 1
                        ? await TrainPartitionedAsync(
                            trainIters, _activeProject.Settings.TrainMaxSplats, _activeProject.Settings.TrainMaxDimension)
                        : await TrainProjectSceneAsync(
                            trainIters, _activeProject.Settings.TrainMaxSplats, _activeProject.Settings.TrainMaxDimension);
                else if (trainIters > 0)
                    Console.WriteLine("[Studio] not training: the pose recovery produced no usable camera poses");

                // A partitioned run that saved itself as a streamed scene (Studio.Partition) is saved already.
                if (_partitionSavedStreamed) _partitionSavedStreamed = false;
                else await SaveViewedSceneToProjectAsync(trainedIters);

                // Schedule thumbnail
                var lastScene = _activeProject.Scenes.LastOrDefault();
                if (lastScene != null)
                {
                    _pendingThumbnailProjectId = _activeProject.Id;
                    _pendingThumbnailSceneId = lastScene.Id;
                    _thumbnailDelayFrames = 30;
                }

                Console.WriteLine($"[Studio] Multi-view scene: {splatCount:N0} splats from {images.Count} images");
            }
            finally
            {
                _multiViewService.OnStatusChanged -= OnStatus;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Multi-view generation error: {ex}");
        }
        finally
        {
            // The device photos are only read by generation (training reloads its targets from OPFS). Every
            // dispatch that read them has completed by here - generation and training both awaited their results.
            if (gpuImages != null)
                foreach (var im in gpuImages) im.DisposeSource();
        }
    }

    /// <summary>Stand the viewer at the first photo's pose (else fit to the scene).</summary>
    private void SeatViewerAtCapturePose(GaussianScene scene)
    {
        if (scene.TrainingCameras.Count > 0 && _cameraController != null)
        {
            var seat = scene.TrainingCameras[0];
            _cameraController.SetPose(seat.Position, seat.Forward, seat.Up);
        }
        else
        {
            _cameraController?.FitToScene();
        }
    }

    /// <summary>
    /// Train the viewed scene with the settings that produced the measured TruckFull result (b24, 2026-09-28:
    /// 24.75 dB held out at 30K) rather than the trainer's bare defaults, which the dataset harness always
    /// overrides from the URL (opacity reset off, a 450K densify cap, a 256 MB target budget). The static knobs
    /// are restored afterwards so a later dataset run still gets its own URL settings. Returns the iterations
    /// done (0 = did not train).
    /// </summary>
    private async Task<int> TrainProjectSceneAsync(int iterations, int maxSplats = 3_000_000, int maxDimension = 1024)
    {
        var saved = (DensifyEveryIters, OpacityResetEveryIters, SplatDensityControl.GrowthSelectFraction,
            MaxDensifiedSplats, MaxTargetStackBytes, HeldOutEveryCycles, _unloadDepthBeforeTraining);
        bool savedRefine = RefinePoses;
        // Camera refinement on for the user's own scenes. MEASURED 2026-10-04 (7K, own SfM): bicycle training-photo
        // fit 22.32 -> 23.16 dB / SSIM 0.692 -> 0.734, held-out +0.07 dB / +0.009 SSIM over test-view alignment alone;
        // TruckFull held-out +0.02 / +0.005. Small on new viewpoints, never worse, and the scene agrees with its photos.
        // Not in a partitioned block: the views' cameras are shared and refined in place, so each block re-fitted all
        // of them to its own region and the next block started from poses its coarse model was never fitted to
        // (TruckFull 2x2: block 1 began at 21.91 dB against the coarse model's 22.16). The coarse run refines them once.
        RefinePoses = _frozenOutside == null;
        long savedMaxKeys = SplatTrainerGpu.MaxTotalKeys;
        DensifyEveryIters = 100;
        OpacityResetEveryIters = 3000;
        SplatDensityControl.GrowthSelectFraction = 1f;
        // From the device's GPU memory budget (Device settings) and its binding limit, not constants: the photo stack
        // was a fixed 640 MB and the splat cap was the preset's whatever the GPU (GpuMemoryBudget).
        var (targetBytes, splatCap) = GpuMemoryBudget.Derive(GpuMemoryGB, DeviceBindingLimitBytes, maxSplats);
        MaxDensifiedSplats = splatCap;
        MaxTargetStackBytes = targetBytes;
        SplatTrainerGpu.MaxTotalKeys = GpuMemoryBudget.MaxTotalKeys(GpuMemoryGB, DeviceBindingLimitBytes);
        Console.WriteLine(
            $"[Train] GPU memory budget {(GpuMemoryGB > 0 ? $"{GpuMemoryGB} GB" : $"Auto ({GpuMemoryBudget.AutoGB} GB)")}: " +
            $"photos up to {targetBytes >> 20} MB, up to {splatCap:N0} splats and {SplatTrainerGpu.MaxTotalKeys:N0} keys (preset {(maxSplats == ReconstructionPresets.DeviceMaxSplats ? "device max" : maxSplats.ToString("N0"))}; device binding " +
            $"limit {DeviceBindingLimitBytes >> 20} MB)");
        HeldOutEveryCycles = 0;           // no held-out views to score, so no mid-run evaluation passes
        _unloadDepthBeforeTraining = true; // give the trainer the depth model's GPU memory

        _trainingActive = true;
        _trainHudText = $"{_trainHudPrefix}Preparing training ({iterations:N0} iterations)…";
        if (_state == StudioState.SceneViewer) BuildViewerHudUI();
        int done = 0;
        try
        {
            bool ok = await TrainOnTrainingViewsAsync(
                iterations, optimiseGeometry: true, maxTrainDimension: maxDimension,
                onProgress: p =>
                {
                    done = p.Done;
                    double rate = p.Done / Math.Max(p.Seconds, 1e-6);
                    var left = TimeSpan.FromSeconds((p.Total - p.Done) / Math.Max(rate, 1e-6));
                    _trainHudText = _trainStopRequested
                        ? "Stopping - finishing up and saving…"
                        : $"{_trainHudPrefix}Training {p.Done:N0} / {p.Total:N0} · {p.Splats:N0} splats · {rate:F1} it/s · " +
                          $"~{(int)left.TotalMinutes}m {left.Seconds:D2}s left";
                    if (_hudTrainLabel != null) _hudTrainLabel.Text = _trainHudText;
                },
                livePreview: true);
            return ok ? done : 0;
        }
        finally
        {
            (DensifyEveryIters, OpacityResetEveryIters, SplatDensityControl.GrowthSelectFraction,
                MaxDensifiedSplats, MaxTargetStackBytes, HeldOutEveryCycles, _unloadDepthBeforeTraining) = saved;
            RefinePoses = savedRefine;
            SplatTrainerGpu.MaxTotalKeys = savedMaxKeys;
            _trainingActive = false;
            if (_state == StudioState.SceneViewer) BuildViewerHudUI();
        }
    }

    /// <summary>
    /// Save the scene on screen to the active project: packed splats, plus (trained) the colour model and the SH
    /// bands. GPU -> JS -> OPFS, never the .NET heap.
    /// </summary>
    /// <summary>
    /// The visible splats (opacity &gt; 0) of the scene on screen, gathered on the GPU and read back as JS bytes - packed
    /// rows and, with SH, each part's rows - with their count. Null bytes when nothing was deleted (the caller reads the
    /// buffers whole).
    /// </summary>
    async Task<(int Count, Uint8Array? Packed, Uint8Array[]? Sh)> ReadVisibleRowsAsync(
        ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense> packed, int n)
    {
        var a = _gpuService.WebGPUAccelerator;
        var all = SplatEditor.Volume.Rows(0, n);
        int visible = await _splatEditor.CountAsync(a, packed, n, all);
        if (visible <= 0 || visible >= n) return (n, null, null);
        using var indices = await SplatRows.SelectIndicesAsync(a, packed, n, all, visible);
        using var kept = SplatRows.GatherRows(a, packed, indices, visible, SplatFormat.Floats);
        await a.SynchronizeAsync();
        var packedU8 = await kept.CopyToHostUint8ArrayAsync(0, (long)visible * SplatFormat.Floats * sizeof(float));
        Uint8Array[]? sh = null;
        if (_gpuRenderer.ShDegree > 0 && _gpuRenderer.ShRestBuffers is { } parts)
        {
            sh = new Uint8Array[parts.Length];
            for (int p = 0; p < parts.Length; p++)
            {
                using var whole = _gpuRenderer.CopyShPartToIlgpu(a, p, n);
                using var rows = SplatRows.GatherRows(a, whole, indices, visible, SphericalHarmonics.PartFloatsPerSplat);
                await a.SynchronizeAsync();
                sh[p] = await rows.CopyToHostUint8ArrayAsync(0, (long)visible * SphericalHarmonics.PartFloatsPerSplat * sizeof(float));
            }
        }
        Console.WriteLine($"[Studio] save drops {n - visible:N0} deleted splats: {visible:N0} of {n:N0} kept");
        return (visible, packedU8, sh);
    }

    /// <summary>
    /// The view a scene saved now should open at: the first capture camera when the scene on screen still has its
    /// cameras (just generated), else what the viewer is looking at.
    /// </summary>
    float[]? CurrentHomeView()
    {
        var cams = _sceneManager.ActiveScene?.TrainingCameras;
        if (cams is { Count: > 0 }) return Pose(cams[0].Position, cams[0].Forward, cams[0].Up);
        var c = _sceneManager.Camera;
        return Pose(c.Position, c.Forward, c.Up);
        static float[] Pose(System.Numerics.Vector3 p, System.Numerics.Vector3 f, System.Numerics.Vector3 u)
            => new[] { p.X, p.Y, p.Z, f.X, f.Y, f.Z, u.X, u.Y, u.Z };
    }

    /// <summary>
    /// Open a loaded scene at its home view; a trained scene saved before home views existed starts outside its robust
    /// bounds looking at their middle (the single-photo default - origin, looking +Z - is arbitrary for SfM).
    /// </summary>
    async Task SeatAtHomeViewAsync(ProjectScene scene)
    {
        if (_cameraController == null) return;
        if (scene.HomeView is { Length: 9 } h)
        {
            _cameraController.SetPose(new(h[0], h[1], h[2]), new(h[3], h[4], h[5]), new(h[6], h[7], h[8]));
            return;
        }
        if (scene.TrainedIterations <= 0 || _gpuRenderer.PackedSplatBuffer is not { } packed) return;
        var box = await SplatBounds.ComputeRobustAsync(_gpuService.WebGPUAccelerator, packed, _gpuRenderer.SplatCount);
        if (box is not { } b) return;
        var centre = new System.Numerics.Vector3(b.CentreX, b.CentreY, b.CentreZ);
        float radius = 0.5f * b.Diagonal;
        // A little above, from the -Z side (SfM scenes are levelled to world up).
        var from = centre + new System.Numerics.Vector3(0, 0.35f * radius, -1.6f * radius);
        _cameraController.SetPose(from, System.Numerics.Vector3.Normalize(centre - from), System.Numerics.Vector3.UnitY);
        Console.WriteLine($"[Studio] no home view saved: seated outside the scene's bounds (radius {radius:F2})");
    }

    /// <summary>
    /// DAv3's camera for a single photo (a one-view joint pass): its 3x3 intrinsics in the photo's pixels, row-major, or
    /// null. Reads the photo where it already is on the GPU.
    /// </summary>
    async Task<float[]?> EstimateSingleViewIntrinsicsAsync(
        ILGPU.Runtime.MemoryBuffer1D<int, ILGPU.Stride1D.Dense> rgba, int w, int h, string name)
    {
        try
        {
            _statusMessage = "Estimating the camera...";
            BuildProjectDetailUI();
            var image = new ImportedImage { FileName = name, Width = w, Height = h, GpuRgba = rgba };
            using var mv = await _depthService.EstimateDepthMultiViewAsync(new[] { image }, maxViews: 1);
            if (mv?.Intrinsics is { Length: > 0 } ks && ks[0] is { Length: >= 9 } k && k[0] > 0 && k[4] > 0)
                return k;
            Console.WriteLine("[Studio] DAv3 gave no camera for this photo; keeping the fallback focal length");
        }
        catch (Exception ex) { Console.WriteLine($"[Studio] camera estimate failed: {ex.Message}"); }
        return null;
    }

    // The project scene on screen (loaded or just saved), so an edited copy keeps its training metadata.
    ProjectScene? _viewedProjectScene;

    private async Task SaveViewedSceneToProjectAsync(int trainedIters, string? editedFrom = null, bool dropDeleted = false)
    {
        if (_activeProject == null) return;
        int count = _gpuRenderer.SplatCount;
        Uint8Array? packedU8 = null;
        Uint8Array[]? shRest = null;
        // An edited scene drops its deleted splats (opacity 0) on the way out: the saved file and every later load
        // carry only what is visible. The viewer keeps its rows (and its undo history).
        if (dropDeleted && _gpuRenderer.PackedSplatBuffer is { } packed)
            (count, packedU8, shRest) = await ReadVisibleRowsAsync(packed, count);
        packedU8 ??= await _gpuRenderer.ReadPackedUint8ArrayAsync(count);
        using var packedHold = packedU8;
        if (packedU8 == null) { Console.WriteLine("[Studio] save skipped: no packed splat data"); return; }

        // The SH bands drawn on screen, from the viewer: it gets an exact copy of the trainer's after training, and only
        // the viewer's follow edits. The trainer's were read here before - after a paste grew the scene (508,091 ->
        // 512,736 splats) that copy overran the trainer's buffers and saved garbage bands (2026-10-03).
        if (shRest == null && _gpuRenderer.ShDegree > 0)
            shRest = await _gpuRenderer.ReadShRestUint8ArraysAsync();
        try
        {
            var projectScene = new ProjectScene
            {
                SplatCount = count,
                FloatsPerSplat = SplatFormat.Floats,
                QualityPreset = _activeProject.Settings.QualityPreset,
                ColoursAreShDc = _gpuRenderer.ColoursAreShDc,
                ShDegree = shRest != null ? _gpuRenderer.ShDegree : 0,
                TrainedIterations = trainedIters,
                EditedFrom = editedFrom,
                HomeView = CurrentHomeView(),
            };
            await _projectService.SaveSceneAsync(_activeProject.Id, projectScene, packedU8);
            _viewedProjectScene = projectScene;
            if (shRest != null)
                await _projectService.SaveSceneShRestAsync(_activeProject.Id, projectScene, shRest);
            Console.WriteLine(
                $"[Studio] scene saved: {count:N0} splats, {packedU8.Length / (1024 * 1024)} MB" +
                (shRest != null ? $" + SH degree {projectScene.ShDegree} bands {shRest.Sum(p => (long)p.Length) / (1024 * 1024)} MB" : "") +
                (trainedIters > 0 ? $", trained {trainedIters:N0} iterations" : ", untrained"));
        }
        finally
        {
            if (shRest != null) foreach (var part in shRest) part.Dispose();
        }
    }

    // ─── Scene Viewing ───

    private async void OnViewScene(ProjectScene scene) => await LoadProjectSceneAsync(scene);

    /// <summary>Open a saved scene in the viewer (what clicking it in the project does).</summary>
    private async Task LoadProjectSceneAsync(ProjectScene scene)
    {
        if (_activeProject == null) return;
        _lodPager?.Dispose(); _lodPager = null;   // a streamed v3 file was open
        if (scene.Format == ProjectScene.FormatLod)
        {
            // A streamed scene (a partitioned run larger than one GPU): open its file streamed, by slices.
            var file = await _projectService.GetStreamedSceneFileAsync(_activeProject.Id, scene.Id);
            if (file == null) { _statusMessage = "Error: scene data not found in storage"; BuildProjectDetailUI(); return; }
            await OpenSceneBlobAsync(file, $"{scene.Id}.spawnscene");
            _viewedProjectScene = scene;
            Console.WriteLine($"[Studio] Opened streamed scene {scene.Id}: {scene.SplatCount:N0} splats");
            return;
        }

        _statusMessage = $"Loading {scene.SplatCount:N0} splats from storage...";
        BuildProjectDetailUI();

        try
        {
            // Stream OPFS → GPU. The packed splats flow JS-side chunk-by-chunk straight into the GPU
            // buffer and never enter the .NET managed heap (a 5K scene ≈ 560 MB — reading it into a
            // byte[]+float[] OOMs WASM). BlobStream is an IJSReadStream, so CopyFromStreamAsync takes the
            // JS-side streaming path automatically.
            using var sceneStream = await _projectService.OpenSceneStreamAsync(_activeProject.Id, scene.Id);
            if (sceneStream == null)
            {
                _statusMessage = "Error: scene data not found in storage";
                BuildProjectDetailUI();
                return;
            }

            _statusMessage = $"Streaming {scene.SplatCount:N0} splats to GPU...";
            BuildProjectDetailUI();

            // A trained scene stores SH DC in the colour slots and its SH bands beside it.
            _gpuRenderer.UseRgbColours();
            _gpuRenderer.ColoursAreShDc = scene.ColoursAreShDc;
            await _gpuRenderer.UploadSceneFromStream(sceneStream, scene.SplatCount, scene.EffectiveFloatsPerSplat);
            if (scene.ShDegree > 0)
            {
                bool loaded = false;
                if (scene.ShParts == SphericalHarmonics.Parts)
                {
                    var parts = await _projectService.ReadSceneShRestPartsAsync(_activeProject.Id, scene.Id, scene.ShParts);
                    if (parts != null)
                    {
                        _gpuRenderer.LoadShRest(parts, scene.ShDegree);
                        foreach (var part in parts) part.Dispose();
                        loaded = true;
                    }
                }
                else
                {
                    // Saved before the part split: one row-major file, split on the GPU.
                    using var rows = await _projectService.ReadSceneShRestAsync(_activeProject.Id, scene.Id);
                    if (rows != null)
                    {
                        _gpuRenderer.LoadShRestRows(rows, scene.ShDegree);
                        loaded = true;
                    }
                }
                if (loaded) _gpuRenderer.RepackForDisplay();
                else Console.WriteLine($"[Studio] scene {scene.Id}: SH bands missing - drawing base colour only");
            }
            // &lodtau=N: draw the scene through its LOD tree's cut (Studio.Lod).
            if (LodTauOption > 0f) await InstallLodTreeAsync();

            var gaussianScene = new GaussianScene
            {
                GpuSplatCount = scene.SplatCount,
                SourceName = "depth-splat",
            };
            // As the generation paths show a scene: sorted alpha at full resolution. A reopened scene fell back to the
            // stochastic renderer - 2 samples a pixel while moving, so every saved scene looked grainy and broken until
            // the camera stopped, while the same scene had looked clean when it was generated (2026-10-04).
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
            _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
            _renderService.SetActiveSceneGpuLoaded(gaussianScene);
            _sceneManager.ActiveScene = gaussianScene;

            _statusMessage = null;
            _state = StudioState.SceneViewer;
            _cameraController?.FitToScene();
            BuildViewerHudUI();

            Console.WriteLine($"[Studio] Loaded scene from OPFS: {scene.SplatCount:N0} splats");
            _viewedProjectScene = scene;
            await SeatAtHomeViewAsync(scene);
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error loading scene: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Error loading scene: {ex}");
        }
    }

    // ─── File/Image Handling ───

    private void OnAddImagesClicked()
    {
        try
        {
            using var el = _fileInput!.Element!.Value.As<HTMLElement>();
            el.Click();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Error triggering file picker: {ex.Message}");
        }
    }

    private async void OnFileSelected(InputFileChangeEventArgs e)
    {
        if (_activeProject == null) return;

        _statusMessage = $"Loading {e.FileCount} image(s)...";
        BuildProjectDetailUI();

        try
        {
            foreach (var file in e.GetMultipleFiles(2000))
            {
                var name = file.Name;
                if (file.ContentType.StartsWith("video/", StringComparison.OrdinalIgnoreCase))
                {
                    await AddVideoSourcesAsync(file.Name);
                    continue;
                }
                _statusMessage = $"Loading {name}...";
                BuildProjectDetailUI();

                // Read file bytes
                using var stream = file.OpenReadStream(maxAllowedSize: 50 * 1024 * 1024);
                using var ms = new MemoryStream();
                await stream.CopyToAsync(ms);
                var bytes = ms.ToArray();

                // Decode image to get dimensions
                int width = 0, height = 0;
                try
                {
                    using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = file.ContentType });
                    using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
                    width = (int)bitmap.Width;
                    height = (int)bitmap.Height;
                    Console.WriteLine($"[Studio] Image decoded: {name} ({width}x{height})");
                }
                catch (Exception ex2)
                {
                    Console.WriteLine($"[Studio] Image decode failed: {ex2.Message}");
                }

                // Save to OPFS
                await _projectService.AddSourceAsync(_activeProject.Id, name, bytes, width, height);
                Console.WriteLine($"[Studio] Saved source: {name} ({bytes.Length / 1024}KB)");
            }

            _statusMessage = null;
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.FirstOrDefault(p => p.Id == _activeProject.Id) ?? _activeProject;
            BuildProjectDetailUI();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Error loading images: {ex}");
        }
    }

    /// <summary>Frames taken from a picked video (<c>&amp;videoframes=N</c>, default 120).</summary>
    public static int VideoFrameCount { get; set; } = 120;

    /// <summary>
    /// A picked video becomes <see cref="VideoFrameCount"/> sharp, evenly spaced frames, each saved as an ordinary
    /// project source - from there a video is a photo set to generation, training and the viewer.
    /// </summary>
    private async Task AddVideoSourcesAsync(string fileName)
    {
        if (_activeProject == null) return;
        string url = await _videoExtractor.UrlForPickedFileAsync(fileName);
        if (string.IsNullOrEmpty(url))
        {
            Console.WriteLine($"[Studio] {fileName}: picked video not found on the page");
            return;
        }
        try
        {
            _statusMessage = $"Extracting {VideoFrameCount} frames from {fileName}...";
            BuildProjectDetailUI();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var frames = await _videoExtractor.ExtractAsync(url, VideoFrameCount, progress: (i, n) =>
            {
                _statusMessage = $"Extracting frames from {fileName}: {i}/{n}";
                BuildProjectDetailUI();
            });
            string stem = System.IO.Path.GetFileNameWithoutExtension(fileName);
            foreach (var f in frames)
                await _projectService.AddSourceAsync(_activeProject.Id, $"{stem}_{f.Name}", f.Jpeg, f.Width, f.Height);
            Console.WriteLine($"[Studio] {fileName}: {frames.Count} frames saved as sources in {sw.Elapsed.TotalSeconds:F1}s");
        }
        finally
        {
            await _videoExtractor.RevokeAsync(url);
        }
    }

    SampleCatalog? _sampleCatalog;
    bool _sampleCatalogLoading;

    /// <summary>Fetch samples/catalog.json once; the project page rebuilds when it lands. A failure leaves no list.</summary>
    private async Task EnsureSampleCatalogAsync()
    {
        if (_sampleCatalog != null || _sampleCatalogLoading) return;
        _sampleCatalogLoading = true;
        try
        {
            _sampleCatalog = await System.Net.Http.Json.HttpClientJsonExtensions.GetFromJsonAsync<SampleCatalog>(
                _http, "samples/catalog.json", new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? new SampleCatalog();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] sample catalog unavailable: {ex.Message}");
            _sampleCatalog = new SampleCatalog();
        }
        if (_state == StudioState.ProjectDetail) BuildProjectDetailUI();
    }

    /// <summary>
    /// Download a catalog sample into the open project, photo by photo, exactly as picked files are stored, and keep
    /// its credit on the project (the licenses ask for attribution wherever the photos are used).
    /// </summary>
    private async Task LoadSampleAsync(SampleCatalog catalog, SampleEntry sample)
    {
        if (_activeProject == null || _pipelineBusy) return;
        var project = _activeProject;
        try
        {
            string folder = catalog.Base.TrimEnd('/') + "/" + sample.Folder.Trim('/') + "/";
            // Six downloads in flight (the browser's per-host limit): through the hub, one at a time took 39.5 s for
            // Hamamni's 59 photos on a cold cache. Stored in order as each completes.
            const int InFlight = 6;
            var pending = new Queue<Task<byte[]>>();
            int next = 0;
            void Fill()
            {
                while (pending.Count < InFlight && next < sample.Images.Count)
                    pending.Enqueue(_http.GetByteArrayAsync(folder + Uri.EscapeDataString(sample.Images[next++])));
            }
            Fill();
            for (int i = 0; i < sample.Images.Count; i++)
            {
                string file = sample.Images[i];
                _statusMessage = $"Downloading {sample.Name}: photo {i + 1} of {sample.Images.Count}...";
                BuildProjectDetailUI();
                var bytes = await pending.Dequeue();
                Fill();
                int width = 0, height = 0;
                try
                {
                    string type = file.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? "image/png" : "image/jpeg";
                    using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = type });
                    using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
                    width = (int)bitmap.Width;
                    height = (int)bitmap.Height;
                }
                catch (Exception decode)
                {
                    Console.WriteLine($"[Studio] sample {file}: decode failed ({decode.Message}), stored anyway");
                }
                // Prefixed with the sample's folder so two samples in one project cannot collide.
                await _projectService.AddSourceAsync(project.Id, $"{sample.Folder.Trim('/')}_{file}", bytes, width, height);
            }

            _projects = await _projectService.ListProjectsAsync();
            var updated = _projects.FirstOrDefault(p => p.Id == project.Id) ?? project;
            string credit = $"{sample.Name}: {sample.Credit}, {sample.License}" + (string.IsNullOrEmpty(sample.Source) ? "" : $" ({sample.Source})");
            updated.Credit = string.IsNullOrEmpty(updated.Credit) ? credit : $"{updated.Credit}; {credit}";
            await _projectService.UpdateProjectAsync(updated);
            _activeProject = updated;
            _statusMessage = null;
            Console.WriteLine($"[Studio] Loaded sample '{sample.Name}': {sample.Images.Count} photos");
        }
        catch (Exception ex)
        {
            _statusMessage = $"Could not download the sample: {ex.Message}";
            Console.WriteLine($"[Studio] Error loading sample {sample.Name}: {ex}");
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.FirstOrDefault(p => p.Id == project.Id) ?? project;
        }
        BuildProjectDetailUI();
    }

    /// <summary>
    /// <c>autotest=samples&amp;name=folder</c>: the catalog sample through <see cref="LoadSampleAsync"/> into a fresh project
    /// - downloaded from the catalog's host (cross-origin), stored, credited. "[Dataset] DONE" / "[Dataset] FAIL".
    /// </summary>
    private async Task RunSampleAutotestAsync(string folder)
    {
        await EnsureSampleCatalogAsync();
        var entry = _sampleCatalog?.Samples.FirstOrDefault(sm => sm.Folder == folder);
        if (_sampleCatalog == null || entry == null) { Console.WriteLine($"[Dataset] FAIL: sample '{folder}' not in samples/catalog.json"); return; }
        var project = await _projectService.CreateProjectAsync($"sample {folder} {DateTime.Now:MMdd-HHmmss}");
        _projects = await _projectService.ListProjectsAsync();
        _activeProject = project;
        _state = StudioState.ProjectDetail;
        _hideUiOverlay = false;
        BuildProjectDetailUI();   // the empty project: the catalog's buttons
        await Task.Delay(1000);
        Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-sample_empty");
        await Task.Delay(2000);
        var t0 = DateTime.UtcNow;
        await LoadSampleAsync(_sampleCatalog, entry);
        var stored = _activeProject?.Sources ?? new();
        bool sized = stored.Count > 0 && stored.All(src => src.Width > 0 && src.Height > 0);
        Console.WriteLine($"[Dataset] sample {folder}: {stored.Count}/{entry.Images.Count} photos stored in " +
            $"{(DateTime.UtcNow - t0).TotalSeconds:F1}s ({stored.Sum(src => src.SizeBytes) / 1e6:F1} MB, all decoded: {sized}); " +
            $"credit: {_activeProject?.Credit}");
        Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-sample_loaded");
        await Task.Delay(1500);
        if (stored.Count != entry.Images.Count || !sized || string.IsNullOrEmpty(_activeProject?.Credit))
            Console.WriteLine("[Dataset] FAIL: sample not stored as the catalog lists it");
        else
            Console.WriteLine("[Dataset] DONE");
    }

    /// <summary>The harness's <c>&amp;sample=</c> path: one image from the app's own samples/ folder.</summary>
    private async Task LoadSampleImage(string name, string path)
    {
        if (_activeProject == null) return;
        _statusMessage = $"Loading sample: {name}...";
        BuildProjectDetailUI();

        try
        {
            var bytes = await _http.GetByteArrayAsync(path);
            int width = 0, height = 0;
            try
            {
                using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/png" });
                using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
                width = (int)bitmap.Width;
                height = (int)bitmap.Height;
            }
            catch { }

            await _projectService.AddSourceAsync(_activeProject.Id, $"{name}.png", bytes, width, height);
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.FirstOrDefault(p => p.Id == _activeProject.Id) ?? _activeProject;
            _statusMessage = null;
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Loaded sample '{name}' ({bytes.Length / 1024}KB, {width}x{height})");
        }
        catch (Exception ex)
        {
            _statusMessage = $"Error loading sample: {ex.Message}";
            BuildProjectDetailUI();
            Console.WriteLine($"[Studio] Error loading sample: {ex}");
        }
    }

    // ─── Thumbnails ───

    private async void CaptureSceneThumbnail(string projectId, string sceneId)
    {
        try
        {
            const int thumbW = 320, thumbH = 200;
            using var canvas = _canvasRef.As<HTMLCanvasElement>();

            using var bitmap = await _js.CallAsync<HTMLCanvasElement, ImageBitmap>("createImageBitmap", canvas);

            using var osc = new OffscreenCanvas(thumbW, thumbH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, thumbW, thumbH);
            using var imageData = ctx.GetImageData(0, 0, thumbW, thumbH);
            using var dataArray = imageData.Data; // Uint8ClampedArray — stay JS-side

            await _projectService.SaveSceneThumbnailAsync(projectId, sceneId, dataArray);
            UploadThumbnailToCache($"scene:{sceneId}", dataArray, thumbW, thumbH);

            Console.WriteLine($"[Studio] Scene thumbnail captured for {sceneId}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Thumbnail capture error: {ex.Message}");
        }
    }

    private async void LoadSceneThumbnailAsync(string projectId, string sceneId)
    {
        string key = $"scene:{sceneId}";
        if (_thumbnailCache.ContainsKey(key) || _device == null || _queue == null) return;
        try
        {
            var pixels = await _projectService.GetSceneThumbnailAsync(projectId, sceneId);
            if (pixels == null || pixels.Length == 0) return;

            // OPFS → byte[] is the file-I/O boundary; upload that straight to the GPU texture.
            UploadThumbnailToCache(key, pixels, 320, 200);

            // In place when the page shows a tile for it (the browser's cards), else rebuild that page.
            if (_thumbTiles.TryGetValue(key, out var tile) && _thumbnailCache.TryGetValue(key, out var cachedThumb))
                tile.TextureView = cachedThumb.view;
            else if (_state == StudioState.ProjectBrowser)
                BuildProjectBrowserUI();
            else if (_state == StudioState.ProjectDetail && _activeProject?.Id == projectId)
                BuildProjectDetailUI();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Scene thumbnail load error: {ex.Message}");
        }
    }

    private void UploadThumbnailToCache(string key, TypedArray pixels, int width, int height)
    {
        if (_device == null || _queue == null) return;
        var (tex, view) = CreateThumbnailTexture(key, width, height);
        _queue.WriteTexture(
            new GPUTexelCopyTextureInfo { Texture = tex },
            pixels,
            new GPUTexelCopyBufferLayout { Offset = 0, BytesPerRow = (uint)(width * 4), RowsPerImage = (uint)height },
            new uint[] { (uint)width, (uint)height });
        _thumbnailCache[key] = (tex, view);
    }

    private void UploadThumbnailToCache(string key, byte[] pixels, int width, int height)
    {
        if (_device == null || _queue == null) return;
        var (tex, view) = CreateThumbnailTexture(key, width, height);
        _queue.WriteTexture(
            new GPUTexelCopyTextureInfo { Texture = tex },
            pixels,
            new GPUTexelCopyBufferLayout { Offset = 0, BytesPerRow = (uint)(width * 4), RowsPerImage = (uint)height },
            new uint[] { (uint)width, (uint)height });
        _thumbnailCache[key] = (tex, view);
    }

    private (GPUTexture tex, GPUTextureView view) CreateThumbnailTexture(string key, int width, int height)
    {
        if (_thumbnailCache.TryGetValue(key, out var old))
        {
            old.view.Dispose();
            old.tex.Destroy();
            old.tex.Dispose();
        }

        var tex = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { width, height },
            Format = "rgba8unorm",
            Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
        });
        return (tex, tex.CreateView());
    }

    private async void LoadThumbnailAsync(string projectId, string fileName, int thumbW = 256, int thumbH = 256)
    {
        string key = SourceThumbKey(projectId, fileName, thumbW, thumbH);
        if (_thumbnailCache.ContainsKey(key) || _device == null || _queue == null) return;
        try
        {
            // The OPFS File straight to the browser decoder: the photo (several MB) never enters the .NET heap.
            using var file = await _projectService.GetSourceFileAsync(projectId, fileName);
            if (file == null) return;
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", file);

            // A centre "cover" crop to the tile's own aspect (square tiles on the project page, 16:10 cards in the browser):
            // stretching the whole photo into a fixed 240x160 squashed every portrait photo.
            int bw = (int)bitmap.Width, bh = (int)bitmap.Height;
            double tileAspect = (double)thumbW / thumbH;
            double cropW = Math.Min(bw, bh * tileAspect), cropH = cropW / tileAspect;
            using var osc = new OffscreenCanvas(thumbW, thumbH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, (bw - cropW) / 2.0, (bh - cropH) / 2.0, cropW, cropH, 0, 0, thumbW, thumbH);
            using var imageData = ctx.GetImageData(0, 0, thumbW, thumbH);
            using var dataArray = imageData.Data; // Uint8ClampedArray — writeTexture directly, no ReadBytes

            UploadThumbnailToCache(key, dataArray, thumbW, thumbH);

            // Update the tile in place (the project page or the browser); rebuild only when the page has no tile for it.
            if (_thumbTiles.TryGetValue(key, out var tile) && _thumbnailCache.TryGetValue(key, out var cachedThumb))
                tile.TextureView = cachedThumb.view;
            else if (_state == StudioState.ProjectDetail && _activeProject?.Id == projectId)
                BuildProjectDetailUI();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Thumbnail load error for {fileName}: {ex.Message}");
        }
    }
}
