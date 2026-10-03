using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// ?autotest=project - the USER path end to end, not the dataset harness's own path.
public partial class Studio
{
    /// <summary>
    /// Build a project the way a user does (photos stored as project sources), press Generate (the real
    /// <see cref="GenerateMultiViewScene"/>: poses, init, TRAINING, save), then open the saved scene the way
    /// clicking it does, and capture the viewer at the first photo's pose before and after the reload.
    /// </summary>
    /// <remarks>
    /// The captures are the check: "project_live" (the trained scene still in memory) and "project_reloaded"
    /// (the same scene read back from OPFS) must match. Two red-check captures load the SAME saved scene the ways
    /// a broken load would - "project_misread_rgb" (SH DC colours drawn as RGB, what every load did before the
    /// colour flag) and "project_no_sh" (the SH bands missing) - so the comparison is shown to be able to fail.
    /// Speaks the dataset harness's markers ([Dataset] READY-FOR-CAPTURE free-..., DONE, FAIL):
    /// <c>AUTOTEST=project node tools/_cdp_dataset.js NAME ITERS</c>, with COUNT / STRIDE picking the photos.
    /// </remarks>
    private async Task RunProjectAutotestAsync(string datasetName, int trainIters, int count, int stride,
        int maxSplats = 3_000_000, int trainRes = 1024, bool pageOnly = false)
    {
        Console.WriteLine(
            $"[Dataset] project autotest name={datasetName} train={trainIters} count={count} stride={stride}");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();

            var manifest = await _importService.TryLoadManifestAsync(datasetName);
            List<string> allNames;
            string imageDir;
            if (manifest != null && manifest.Images.Count >= 2 && string.IsNullOrEmpty(manifest.Video))
            {
                allNames = manifest.Images;
                imageDir = $"datasets/{datasetName}/{manifest.ImageDir}";
            }
            else if (datasetName == "Bathroom")
            {
                // TJ's phone capture (no manifest) - the gh-pages crash repro, 2026-09-28.
                allNames = ImageImportService.BathroomImages.ToList();
                imageDir = $"datasets/{datasetName}";
            }
            else
            {
                Console.WriteLine($"[Dataset] FAIL: {datasetName} has no photo manifest with 2+ images");
                return;
            }
            var names = allNames.Where((_, i) => i % Math.Max(1, stride) == 0).Take(count).ToList();

            // -- A project, with the photos stored exactly as the file picker stores them --
            var project = await _projectService.CreateProjectAsync(
                $"autotest {datasetName} {DateTime.Now:MMdd-HHmmss}",
                new ProjectSettings { TrainIterations = trainIters, TrainMaxSplats = maxSplats, TrainMaxDimension = trainRes });
            var t0 = DateTime.UtcNow;
            foreach (var name in names)
            {
                var bytes = await _http.GetByteArrayAsync($"{imageDir}/{name}");
                // Real size, as the file picker records it; the bitmap's pixels are never read here.
                using var sizeBlob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/jpeg" });
                using var sizeBitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", sizeBlob);
                await _projectService.AddSourceAsync(project.Id, name, bytes, (int)sizeBitmap.Width, (int)sizeBitmap.Height);
            }
            Console.WriteLine(
                $"[Dataset] project {project.Id}: {names.Count} photos stored in " +
                $"{(DateTime.UtcNow - t0).TotalSeconds:F1}s ({names.First()} .. {names.Last()})");

            _projects = await _projectService.ListProjectsAsync();
            // The project browser as a user lands on it, with this project in the list.
            _hideUiOverlay = false;
            _state = StudioState.ProjectBrowser;
            BuildProjectBrowserUI();
            await Task.Delay(1500);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_browser");
            await Task.Delay(2500);
            _activeProject = _projects.First(p => p.Id == project.Id);
            _state = StudioState.ProjectDetail;
            // The project page as a user sees it, settings included.
            _hideUiOverlay = false;
            BuildProjectDetailUI();
            await Task.Delay(1500);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_page");
            await Task.Delay(2500);
            // ...and scrolled to the bottom: the Training settings and Generate sit under the photo list.
            _projectDetailScroll?.ScrollToBottom();
            await Task.Delay(1500);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_page_settings");
            await Task.Delay(2500);
            if (pageOnly) { Console.WriteLine("[Dataset] DONE"); return; }
            // Deterministic captures: sorted mode, full resolution (as the dataset harness renders).
            _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;

            // -- Generate, exactly as the button does --
            t0 = DateTime.UtcNow;
            await GenerateMultiViewScene();
            var liveScene = _sceneManager.ActiveScene;
            var saved = _activeProject?.Scenes.LastOrDefault();
            if (saved == null || liveScene == null)
            {
                Console.WriteLine($"[Dataset] FAIL: Generate produced no saved scene ({_statusMessage})");
                return;
            }
            Console.WriteLine(
                $"[Dataset] project scene {saved.Id} saved after {(DateTime.UtcNow - t0).TotalSeconds:F0}s: " +
                $"{saved.SplatCount:N0} splats, trained {saved.TrainedIterations:N0} iters, " +
                $"shDc={saved.ColoursAreShDc}, shDegree={saved.ShDegree}, {saved.SizeBytes / (1024 * 1024)} MB, " +
                $"{liveScene.TrainingViews.Count} training views");
            if (saved.SplatCount > maxSplats)
            {
                Console.WriteLine($"[Dataset] FAIL: {saved.SplatCount:N0} splats exceeds the project's cap {maxSplats:N0}");
                return;
            }
            if (trainIters > 0 && saved.TrainedIterations != trainIters)
            {
                Console.WriteLine($"[Dataset] FAIL: asked for {trainIters} iterations, the saved scene got {saved.TrainedIterations}");
                return;
            }
            // SH degree follows the reference schedule (one band per 1,000 iterations), so a 1,000-iteration scene
            // legitimately has none; expect exactly what the schedule reached on the last iteration.
            int expectDegree = trainIters > 0 ? SphericalHarmonics.DegreeForIteration(trainIters - 1) : 0;
            if (trainIters > 0 && (!saved.ColoursAreShDc || saved.ShDegree != expectDegree))
            {
                Console.WriteLine(
                    $"[Dataset] FAIL: trained scene saved with shDc={saved.ColoursAreShDc} shDegree={saved.ShDegree}, " +
                    $"expected shDc=True shDegree={expectDegree}");
                return;
            }

            // -- Before and after the round trip through OPFS, from the same pose --
            var seat = liveScene.TrainingCameras.Count > 0 ? liveScene.TrainingCameras[0] : null;
            await CaptureProjectViewAsync("live", seat);

            await LoadProjectSceneAsync(saved);
            await CaptureProjectViewAsync("reloaded", seat);

            // SH storage: new saves are SphericalHarmonics.Parts files, and a scene saved before the split (one
            // row-major file) must load to the SAME part buffers, bit for bit, through the GPU split.
            if (saved.ShDegree > 0)
            {
                if (saved.ShParts != SphericalHarmonics.Parts)
                {
                    Console.WriteLine($"[Dataset] FAIL: scene saved SH as {saved.ShParts} parts, expected {SphericalHarmonics.Parts}");
                    return;
                }
                var fromParts = await _gpuRenderer.ReadShRestPartsAsync();
                if (fromParts == null) { Console.WriteLine("[Dataset] FAIL: the reloaded scene has no SH on the GPU"); return; }
                var rows = SphericalHarmonics.JoinParts(fromParts);
                using (var legacy = new Uint8Array(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rows.AsSpan()).ToArray()))
                    await _projectService.RewriteSceneShAsLegacyRowsAsync(project.Id, saved.Id, legacy);
                _projects = await _projectService.ListProjectsAsync();
                _activeProject = _projects.First(p => p.Id == project.Id);
                var legacyScene = _activeProject.Scenes.First(sc => sc.Id == saved.Id);
                await LoadProjectSceneAsync(legacyScene);
                var fromRows = await _gpuRenderer.ReadShRestPartsAsync();
                int bad = fromRows == null ? -1 : Enumerable.Range(0, fromParts.Length)
                    .Sum(part => fromParts[part].Length != fromRows[part].Length ? int.MaxValue / 4
                        : fromParts[part].Where((v, i) => BitConverter.SingleToInt32Bits(v) != BitConverter.SingleToInt32Bits(fromRows[part][i])).Count());
                Console.WriteLine(bad == 0
                    ? $"[Dataset] legacy SH load PASS: {rows.Length:N0} row-major floats split on the GPU to the same {SphericalHarmonics.Parts} part buffers, bit for bit"
                    : $"[Dataset] FAIL: legacy SH load differs from the part load ({bad} floats, -1 = no SH)");
                if (bad != 0) return;
            }

            // The viewer as a user sees it, its HUD (back, view buttons, stats) on.
            _state = StudioState.SceneViewer;
            _hideUiOverlay = false;
            BuildViewerHudUI();
            await Task.Delay(1500);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_viewer_hud");
            await Task.Delay(2500);
            // ...and with its render settings open.
            _showSettings = true;
            BuildViewerHudUI();
            await Task.Delay(1000);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_viewer_settings");
            await Task.Delay(2500);
            _showSettings = false;
            BuildViewerHudUI();

            // Red checks on the same loaded scene.
            if (saved.ColoursAreShDc)
            {
                _gpuRenderer.ColoursAreShDc = false;
                _gpuRenderer.RepackForDisplay();
                await CaptureProjectViewAsync("misread_rgb", seat);
                _gpuRenderer.ColoursAreShDc = true;
                _gpuRenderer.RepackForDisplay();
            }
            if (saved.ShDegree > 0)
            {
                _gpuRenderer.SetShRest(null, 0);
                _gpuRenderer.RepackForDisplay();
                await CaptureProjectViewAsync("no_sh", seat);
            }
            // The project page again, now with its generated scene: scenes sit above the photo grid.
            _state = StudioState.ProjectDetail;
            _hideUiOverlay = false;
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = _projects.First(p => p.Id == project.Id);
            _projectTab = "Scenes"; // as Back from the viewer opens it
            BuildProjectDetailUI();
            await Task.Delay(2000);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE free-project_page_with_scene");
            await Task.Delay(2500);
            Console.WriteLine("[Dataset] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] FAIL: {ex}");
        }
    }

    private async Task CaptureProjectViewAsync(string name, CameraParams? seat)
    {
        if (seat != null) _cameraController?.SetPose(seat.Position, seat.Forward, seat.Up);
        _hideUiOverlay = true;
        await Task.Delay(1500);
        Console.WriteLine($"[Dataset] READY-FOR-CAPTURE free-project_{name}");
        await Task.Delay(2500);
    }
}
