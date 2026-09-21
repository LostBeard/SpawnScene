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
            string srcKey = $"source:{source.FileName}";
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

            // Build camera params from EXIF (or fall back to heuristic)
            var camera = CameraParams.CreateFromExif(w, h, exifFocal);
            var focalSource = exifFocal?.FocalLength35mm is > 0 ? "EXIF 35mm"
                : exifFocal?.FocalLengthMm is > 0 and < 10f ? $"phone estimate ({exifFocal.FocalLengthMm:F1}mm * 7x)"
                : "heuristic 1.2x";
            Console.WriteLine($"[EXIF] {source.FileName}: fx={camera.FocalX:F1}px ({focalSource})");

            // Depth: JS TypedArray → EstimateGpuRawAsync(TypedArray) via the service (no managed Read<int>).
            // Gaussian path below: UploadToDevice keeps RGBA on GPU for unprojection.
            _statusMessage = "Estimating depth...";
            BuildProjectDetailUI();
            var depthResult = await _depthService.EstimateDepthFromJsRgbaAsync(dataArray, w, h);
            if (depthResult == null) { _statusMessage = "Error: depth estimation failed"; BuildProjectDetailUI(); return; }

            // Gaussian path: JS TypedArray → GPU directly (no .NET heap).
            var rgbaGpuBuf = _gpuService.WebGPUAccelerator.Allocate1D<int>(w * h);
            SpawnDev.ILGPU.ML.Preprocessing.MediaInterop.UploadToDevice(dataArray, rgbaGpuBuf);
            using var gpuImage = new GpuImage
            {
                PackedRgba = rgbaGpuBuf,
                Width = w,
                Height = h,
                FileName = source.FileName,
            };

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
                // is a quarter turn off level (world-up projects to image-right in all 47
                // calibration entries), and monocular depth networks are trained on upright
                // photographs. The camera is turned with the pixels, so nothing downstream
                // changes meaning - see ImageOrientation.
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
                foreach (var (filename, cam) in available)
                {
                    int turns = upright ? ImageOrientation.QuarterTurnsToUpright(cam) : 0;
                    scene.TrainingViews.Add(new TrainingView
                    {
                        Camera = turns != 0 ? ImageOrientation.Rotate(cam, turns) : cam,
                        ImageName = $"datasets/TempleRing/{filename}",
                        UsedForInit = initNames.Contains(filename),
                        QuarterTurns = turns,
                    });
                }
                Console.WriteLine($"[Studio] TempleRing supervision: {scene.TrainingViews.Count} posed views " +
                    $"({initNames.Count} also used for depth init)");

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

        _statusMessage = "Preparing multi-view pipeline...";
        BuildProjectDetailUI();

        try
        {
            // Load all source images as ImportedImage objects
            var images = new List<ImportedImage>();
            foreach (var source in _activeProject.Sources)
            {
                _statusMessage = $"Loading {source.FileName}...";
                BuildProjectDetailUI();

                var imageBytes = await _projectService.GetSourceAsync(_activeProject.Id, source.FileName);
                if (imageBytes == null) continue;

                // Extract EXIF focal length before decoding
                var exifFocal = ExifReader.ExtractFocalLength(imageBytes);

                // Decode image
                using var blob = new Blob(new byte[][] { imageBytes }, new BlobOptions { Type = "image/jpeg" });
                using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
                int w = (int)bitmap.Width;
                int h = (int)bitmap.Height;

                using var osc = new OffscreenCanvas(w, h);
                using var ctx = osc.Get2DContext();
                ctx.DrawImage(bitmap, 0, 0);
                using var imageData = ctx.GetImageData(0, 0, w, h);
                using var dataArray = imageData.Data;
                var rgbaPixels = dataArray.ReadBytes();

                var camera = CameraParams.CreateFromExif(w, h, exifFocal);

                images.Add(new ImportedImage
                {
                    FileName = source.FileName,
                    Width = w,
                    Height = h,
                    RgbaPixels = rgbaPixels,
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

                var result = await _multiViewService.GenerateAsync(images, subsample, edgeSharpness);
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

                await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

                var scene = new GaussianScene
                {
                    GpuSplatCount = splatCount,
                    // Hybrid path may lack TrainingCameras; FitToScene uses depth-splat origin then.
                    SourceName = "depth-splat",
                };

                _renderService.SetActiveSceneGpuLoaded(scene);
                _sceneManager.ActiveScene = scene;

                // Save to OPFS
                _statusMessage = $"Saving {splatCount:N0} splats to storage...";
                BuildProjectDetailUI();

                // GPU → JS Uint8Array → OPFS (zero .NET managed-heap copies — see single-image path).
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
                    Console.WriteLine($"[Studio] Multi-view scene saved: {packedU8.Length / (1024 * 1024):F1} MB");
                }

                // Schedule thumbnail
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
    }

    // ─── Scene Viewing ───

    private async void OnViewScene(ProjectScene scene)
    {
        if (_activeProject == null) return;

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

            await _gpuRenderer.UploadSceneFromStream(sceneStream, scene.SplatCount, scene.EffectiveFloatsPerSplat);

            var gaussianScene = new GaussianScene
            {
                GpuSplatCount = scene.SplatCount,
                SourceName = "depth-splat",
            };
            _renderService.SetActiveSceneGpuLoaded(gaussianScene);
            _sceneManager.ActiveScene = gaussianScene;

            _statusMessage = null;
            _state = StudioState.SceneViewer;
            _cameraController?.FitToScene();
            BuildViewerHudUI();

            Console.WriteLine($"[Studio] Loaded scene from OPFS: {scene.SplatCount:N0} splats");
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
            foreach (var file in e.GetMultipleFiles(20))
            {
                var name = file.Name;
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

            if (_state == StudioState.ProjectBrowser)
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

    private async void LoadThumbnailAsync(string projectId, string fileName)
    {
        string key = $"source:{fileName}";
        if (_thumbnailCache.ContainsKey(key) || _device == null || _queue == null) return;
        try
        {
            var bytes = await _projectService.GetSourceAsync(projectId, fileName);
            if (bytes == null) return;

            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/jpeg" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);

            const int thumbW = 240, thumbH = 160;
            using var osc = new OffscreenCanvas(thumbW, thumbH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, thumbW, thumbH);
            using var imageData = ctx.GetImageData(0, 0, thumbW, thumbH);
            using var dataArray = imageData.Data; // Uint8ClampedArray — writeTexture directly, no ReadBytes

            UploadThumbnailToCache(key, dataArray, thumbW, thumbH);

            if (_state == StudioState.ProjectDetail && _activeProject?.Id == projectId)
                BuildProjectDetailUI();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Thumbnail load error for {fileName}: {ex.Message}");
        }
    }
}
