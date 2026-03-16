using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
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
            BuildProjectBrowserUI();
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

        _statusMessage = "Loading depth model...";
        BuildProjectDetailUI();

        try
        {
            // Ensure depth model is loaded (use model ID, not display name)
            if (!_depthService.IsReady)
            {
                await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);
            }

            // Use the first source image
            var source = _activeProject.Sources[0];
            _statusMessage = $"Loading {source.FileName}...";
            BuildProjectDetailUI();

            var imageBytes = await _projectService.GetSourceAsync(_activeProject.Id, source.FileName);
            if (imageBytes == null) { _statusMessage = "Error: could not read source image"; BuildProjectDetailUI(); return; }

            // Decode image
            _statusMessage = "Decoding image...";
            BuildProjectDetailUI();

            using var blob = new Blob(new byte[][] { imageBytes }, new BlobOptions { Type = "image/jpeg" });
            using var bitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", blob);
            int w = (int)bitmap.Width;
            int h = (int)bitmap.Height;

            // Get RGBA pixels via OffscreenCanvas
            using var osc = new OffscreenCanvas(w, h);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var dataArray = imageData.Data;
            var rgbaPixels = dataArray.ReadBytes();

            var importedImage = new ImportedImage
            {
                FileName = source.FileName,
                Width = w,
                Height = h,
                RgbaPixels = rgbaPixels,
            };

            // Estimate depth
            _statusMessage = "Estimating depth...";
            BuildProjectDetailUI();

            var depthResult = await _depthService.EstimateDepthAsync(importedImage);
            if (depthResult == null) { _statusMessage = "Error: depth estimation failed"; BuildProjectDetailUI(); return; }

            // Generate Gaussians
            _statusMessage = "Generating Gaussians...";
            BuildProjectDetailUI();

            int subsample = _activeProject.Settings.Subsample;
            float edgeSharpness = _activeProject.Settings.EdgeSharpness;
            var (packedBuf, splatCount) = await _gaussianKernel.GeneratePackedGpuBufferAsync(
                depthResult, importedImage, subsample, edgeSharpness);

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

            var packedFloats = await _gpuRenderer.ReadPackedDataAsync(splatCount);
            if (packedFloats != null)
            {
                var projectScene = new ProjectScene
                {
                    SplatCount = splatCount,
                    QualityPreset = _activeProject.Settings.QualityPreset,
                };
                // Convert float[] to byte[] for OPFS storage — free floats immediately
                var sceneBytes = new byte[packedFloats.Length * sizeof(float)];
                Buffer.BlockCopy(packedFloats, 0, sceneBytes, 0, sceneBytes.Length);
                packedFloats = null; // free GPU readback array

                await _projectService.SaveSceneAsync(_activeProject.Id, projectScene, sceneBytes);
                Console.WriteLine($"[Studio] Scene saved to OPFS: {sceneBytes.Length / (1024 * 1024):F1} MB");
            }
            else
            {
                // Save metadata only if readback failed
                var projectScene = new ProjectScene
                {
                    SplatCount = splatCount,
                    QualityPreset = _activeProject.Settings.QualityPreset,
                    SizeBytes = (long)splatCount * 10 * sizeof(float),
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

    // ─── Scene Viewing ───

    private async void OnViewScene(ProjectScene scene)
    {
        if (_activeProject == null) return;

        _statusMessage = $"Loading {scene.SplatCount:N0} splats from storage...";
        BuildProjectDetailUI();

        try
        {
            var sceneBytes = await _projectService.GetSceneDataAsync(_activeProject.Id, scene.Id);
            if (sceneBytes == null)
            {
                _statusMessage = "Error: scene data not found in storage";
                BuildProjectDetailUI();
                return;
            }

            _statusMessage = $"Uploading {scene.SplatCount:N0} splats to GPU...";
            BuildProjectDetailUI();

            // Reinterpret byte[] as float[] — null out sceneBytes immediately to reduce peak memory.
            var packedFloats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(sceneBytes.AsSpan()).ToArray();
            sceneBytes = null!; // free ~560MB before GPU allocation

            // Upload to GPU (releases old scene buffers internally via DisposeBuffers)
            var accelerator = _gpuService.WebGPUAccelerator;
            var packedBuf = accelerator.Allocate1D(packedFloats);
            packedFloats = null!; // free ~560MB before vertex buffer allocation
            await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, scene.SplatCount);

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
            using var el = new HTMLElement(_fileInput!.Element!.Value);
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
                    using var bitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", blob);
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
                using var bitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", blob);
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
            using var canvas = new HTMLCanvasElement(_canvasRef);

            using var bitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", canvas);

            using var osc = new OffscreenCanvas(thumbW, thumbH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, thumbW, thumbH);
            using var imageData = ctx.GetImageData(0, 0, thumbW, thumbH);
            using var dataArray = imageData.Data;
            var pixels = dataArray.ReadBytes();

            await _projectService.SaveSceneThumbnailAsync(projectId, sceneId, pixels);
            UploadThumbnailToCache($"scene:{sceneId}", pixels, thumbW, thumbH);

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

    private void UploadThumbnailToCache(string key, byte[] pixels, int width, int height)
    {
        if (_device == null || _queue == null) return;

        if (_thumbnailCache.TryGetValue(key, out var old))
        {
            old.view.Dispose();
            old.tex.Destroy();
            old.tex.Dispose();
        }

        var tex = _device.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { width, height },
            Format = "rgba8unorm",
            Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
        });
        var view = tex.CreateView();

        _queue.WriteTexture(
            new GPUTexelCopyTextureInfo { Texture = tex },
            pixels,
            new GPUTexelCopyBufferLayout { Offset = 0, BytesPerRow = (uint)(width * 4), RowsPerImage = (uint)height },
            new uint[] { (uint)width, (uint)height }
        );

        _thumbnailCache[key] = (tex, view);
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
            using var bitmap = await BlazorJSRuntime.JS.CallAsync<ImageBitmap>("createImageBitmap", blob);

            const int thumbW = 240, thumbH = 160;
            using var osc = new OffscreenCanvas(thumbW, thumbH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0, thumbW, thumbH);
            using var imageData = ctx.GetImageData(0, 0, thumbW, thumbH);
            using var dataArray = imageData.Data;
            var pixels = dataArray.ReadBytes();

            UploadThumbnailToCache(key, pixels, thumbW, thumbH);

            if (_state == StudioState.ProjectDetail && _activeProject?.Id == projectId)
                BuildProjectDetailUI();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Thumbnail load error for {fileName}: {ex.Message}");
        }
    }
}
