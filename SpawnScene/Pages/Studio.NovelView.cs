using System.Numerics;
using SpawnScene.Models;
using System.IO;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Novel-view fidelity harness.
///
/// TempleRing reconstructs from 4 of its 16 photos. The other 12 have ground-truth poses AND
/// ground-truth photographs, so rendering from one of them and comparing to the real picture
/// turns "does it look correct" into a number — and a pose the reconstruction never saw is
/// exactly what looking around produces, so it measures both halves of the goal at once.
///
/// The comparison itself is deliberately NOT done in here. The app only parks the camera and
/// says it is ready; the harness screenshots and scores offline. That keeps a 1.2 MB-per-view
/// pixel readback out of the WASM heap entirely (Rule 4) and keeps the metric in the harness
/// where a change to it cannot quietly change the product.
///
/// Entry: <c>/studio?autotest=novel-view&amp;view=templeR0016</c>
/// Logs:  <c>[NovelView] READY view=... splats=... pose=...</c> or <c>[NovelView] FAIL: ...</c>
/// </summary>
public partial class Studio
{
    /// <summary>Suppresses the UI overlay so a captured frame is pure render, no HUD.</summary>
    private bool _hideUiOverlay;

    /// <summary>Project the harness reuses, so the scene is generated once and then loaded.</summary>
    private const string NovelViewProjectName = "NovelView TempleRing";

    private async Task RunNovelViewAutotestAsync(string viewName, int onlyView = -1, bool globalScale = false)
    {
        Console.WriteLine($"[NovelView] starting view={viewName}");
        try
        {
            // ── 0. Resolve the pose FIRST ──
            // A bad view name must cost seconds, not a three-minute generate followed by a
            // lookup failure.
            var parText = await _http.GetStringAsync("datasets/TempleRing/templeR_par.txt");
            var gtCameras = WorldSpaceGeometry.ParseMiddleburyParams(parText, 640, 480);
            // The par file stores names WITH the extension; accept either spelling.
            var match = gtCameras.FirstOrDefault(c =>
                string.Equals(c.filename, viewName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Path.GetFileNameWithoutExtension(c.filename), viewName,
                    StringComparison.OrdinalIgnoreCase));
            if (match.filename == null)
                throw new InvalidOperationException(
                    $"view '{viewName}' not in templeR_par.txt ({gtCameras.Count} poses, " +
                    $"e.g. {string.Join(", ", gtCameras.Take(3).Select(c => c.filename))})");
            var gt = match.camera;
            Console.WriteLine($"[NovelView] resolved pose {match.filename}");

            // ── 1. Reuse the saved scene if we already built one ──
            // Generation is minutes; loading from OPFS is seconds. The harness walks a dozen
            // views, so regenerating per view would make the measurement unusable. This also
            // exercises the saved-scene load path on every run rather than only the fresh one.
            _projects = await _projectService.ListProjectsAsync();
            var project = _projects.FirstOrDefault(p => p.Name == NovelViewProjectName);
            bool mustGenerate = project == null || project.Scenes.Count == 0;

            if (project == null)
                project = await _projectService.CreateProjectAsync(NovelViewProjectName);

            _activeProject = project;
            OnOpenProject(project);

            var sceneReady = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            void OnDone()
            {
                // HasScene is only "ActiveScene != null" - it goes true for an empty placeholder
                // scene, which made this complete instantly and skip the whole generate.
                var sc = _sceneManager.ActiveScene;
                int n = sc == null ? 0 : Math.Max(sc.GpuSplatCount, sc.Count);
                if (_state == StudioState.SceneViewer && n > 0)
                {
                    Console.WriteLine($"[NovelView] scene ready: {n:N0} splats");
                    sceneReady.TrySetResult();
                }
            }
            _sceneManager.OnSceneChanged += OnDone;
            try
            {
                if (mustGenerate)
                {
                    Console.WriteLine($"[NovelView] no saved scene — generating from TempleRing (slow path, onlyView={onlyView}, globalScale={globalScale})");
                    _ = GenerateFromTempleRingAsync(onlyView, globalScale);
                }
                else
                {
                    Console.WriteLine($"[NovelView] loading saved scene {project.Scenes[0].Id} " +
                        $"({project.Scenes[0].SplatCount:N0} splats, stride {project.Scenes[0].EffectiveFloatsPerSplat})");
                    OnViewScene(project.Scenes[0]);
                }

                var deadline = DateTime.UtcNow.AddMinutes(mustGenerate ? 12 : 3);
                while (DateTime.UtcNow < deadline)
                {
                    if (sceneReady.Task.IsCompleted) break;
                    if (_statusMessage != null
                        && (_statusMessage.StartsWith("Error", StringComparison.OrdinalIgnoreCase)
                            || _statusMessage.Contains('❌')))
                        throw new InvalidOperationException(_statusMessage);
                    await Task.Delay(250);
                }
                if (!sceneReady.Task.IsCompleted)
                    throw new TimeoutException("scene never became ready");
            }
            finally
            {
                _sceneManager.OnSceneChanged -= OnDone;
            }

            // ── 2. Deterministic render settings ──
            // Sorted, not stochastic: stochastic converges over many frames via temporal
            // accumulation, so a screenshot of it measures how long we waited as much as how
            // good the scene is. ForceFull defeats the adaptive half-res path.
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
            _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
            // Middlebury shoots the temple against black and 61% of a GT frame is near-black.
            // Rendering it on the app's dark-blue clear would make the background mismatch
            // dominate the score and mask every change to the temple itself.
            _gpuRenderer.BackgroundColor = (0.0, 0.0, 0.0);

            _state = StudioState.SceneViewer;
            _hideUiOverlay = true;

            await ParkOnGroundTruthPoseAsync(match.filename, gt);

            // ── 4. Serve further poses from the URL HASH, on this same page load ──
            // Re-navigating per view meant re-deciding "is there a saved scene?" from an OPFS
            // listing that is not always settled, so views intermittently regenerated from
            // scratch (and two of six then failed outright). A hash change does not reload the
            // page, so the scene stays resident and each extra view costs a camera move.
            var hashDeadline = DateTime.UtcNow.AddMinutes(30);
            string lastHash = CurrentHash();
            Console.WriteLine($"[NovelView] hash-driver active (hash='{lastHash}')");
            while (DateTime.UtcNow < hashDeadline)
            {
                await Task.Delay(100);
                string hash = CurrentHash();
                if (hash == lastHash) continue;
                lastHash = hash;

                string want = hash.TrimStart('#');
                if (want.StartsWith("view=", StringComparison.OrdinalIgnoreCase))
                    want = want.Substring(5);
                if (want.Length == 0) continue;
                if (string.Equals(want, "done", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("[NovelView] DONE");
                    return;
                }

                var next = gtCameras.FirstOrDefault(c =>
                    string.Equals(c.filename, want, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(Path.GetFileNameWithoutExtension(c.filename), want,
                        StringComparison.OrdinalIgnoreCase));
                if (next.filename == null)
                {
                    Console.WriteLine($"[NovelView] FAIL: unknown view '{want}'");
                    continue;
                }
                await ParkOnGroundTruthPoseAsync(next.filename, next.camera);
            }
            Console.WriteLine("[NovelView] DONE (deadline)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NovelView] FAIL: {ex}");
        }
    }

    /// <summary>Current <c>location.hash</c>, or empty if unavailable.</summary>
    private bool _hashReadFailed;
    private string CurrentHash()
    {
        try { return _js.Get<string>("location.hash") ?? ""; }
        catch (Exception ex)
        {
            // Announce ONCE. A silently-empty hash makes every later pose request look like a
            // harness problem (timeouts with no error) when the real cause is interop.
            if (!_hashReadFailed)
            {
                _hashReadFailed = true;
                Console.WriteLine($"[NovelView] FAIL: cannot read location.hash: {ex.Message}");
            }
            return "";
        }
    }

    /// <summary>
    /// Point the camera at a ground-truth pose, intrinsics and all, and announce it.
    /// Announces the VIEW NAME so a harness reusing one page can wait for the pose it asked
    /// for rather than matching a stale marker from the previous one.
    /// </summary>
    private async Task ParkOnGroundTruthPoseAsync(string viewName, CameraParams gt)
    {
        var cam = _sceneManager.Camera;
        cam.Width = gt.Width;
        cam.Height = gt.Height;
        cam.FocalX = gt.FocalX;
        cam.FocalY = gt.FocalY;
        cam.CenterX = gt.CenterX;
        cam.CenterY = gt.CenterY;
        // The temple sits ~0.5 world units from the ring; the 0.1 default near plane is
        // uncomfortably close to that, so tighten it for this measurement.
        cam.Near = 0.01f;
        cam.Far = 100f;
        _sceneManager.Camera = cam;

        // SetPose writes Forward/Up exactly - a dataset pose is generally rolled, which a
        // yaw/pitch-only camera cannot represent.
        _cameraController?.SetPose(gt.Position, gt.Forward, gt.Up);

        // The sorted path runs its radix sort asynchronously and self-throttles to ~50ms, so the
        // first frames after a camera jump are drawn against a stale ordering.
        await Task.Delay(1200);

        var sc = _sceneManager.ActiveScene;
        int splats = sc == null ? 0 : Math.Max(sc.GpuSplatCount, sc.Count);
        Console.WriteLine(
            $"[NovelView] READY view={viewName} splats={splats} " +
            $"pose=({gt.Position.X:F4},{gt.Position.Y:F4},{gt.Position.Z:F4}) " +
            $"fwd=({gt.Forward.X:F4},{gt.Forward.Y:F4},{gt.Forward.Z:F4}) " +
            $"K=({gt.FocalX:F1},{gt.FocalY:F1},{gt.CenterX:F1},{gt.CenterY:F1}) " +
            $"canvas={_canvasWidth}x{_canvasHeight}");
    }
}
