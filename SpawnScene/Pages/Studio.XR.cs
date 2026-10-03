using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// WebXR VR/AR session management
public partial class Studio
{
    private async Task EnterXRAsync(string mode)
    {
        try
        {
            Console.WriteLine($"[Studio] Entering {mode}...");
            ReleasePointerLock();

            _xrService.OnXRFrame += OnXRFrame;
            _xrService.OnSessionEnded += OnXRSessionEnded;
            _xrSceneFromRoom = null; // placed on the session's first frame (XRSceneAlignment)
            _xrBoundsReady = false;  // set once the AR bounds are measured (immediately for VR, below)
            _xrLocomotion.Reset();
            _xrWorldGrab.Reset();
            _xrPlacing = false;
            _xrTriggerWasDown = false;
            _xrHitLogged = false;
            _xrExitWasDown = true;   // a B/Y still held from before the session does not end it at once
            _xrMenuWasDown = true;
            _xrMenu?.Close();
            _xrLastHit = null;
            _xrStickLog = true;
            _xrLastTime = -1;

            // Pause canvas RAF loop — XR has its own render loop
            _xrActive = true;

            await _xrService.EnterSessionAsync(mode);
            // Passthrough: draw only the splats; the real world shows wherever the scene has nothing.
            _gpuRenderer.XRTransparent = mode == "immersive-ar" && (_xrService.EnvironmentBlendMode != "opaque" || _xrForceAlpha);
            // The scene's robust bounds size the start: AR shows it as a miniature in front of the viewer, VR puts its
            // middle a room's width away (XRSceneAlignment.ComfortScale). Measured after requestSession (an await
            // before it could spend the click's user activation); frames until then draw nothing.
            _xrSceneBox = null;
            _xrPlaceMiniature = _gpuRenderer.XRTransparent;
            var packed = _gpuRenderer.PackedSplatBuffer;
            int n = _gpuRenderer.SplatCount;
            try
            {
                _xrSceneBox = packed != null && n > 0
                    ? await SplatBounds.ComputeRobustAsync(_gpuService.WebGPUAccelerator, packed, n) : null;
            }
            catch (Exception bex) { Console.WriteLine($"[XR] scene bounds failed ({bex.Message}); VR start at scale 1"); }
            if (_xrSceneBox is { } b)
                Console.WriteLine($"[XR] scene bounds (1-99%) {b.MinX:F2},{b.MinY:F2},{b.MinZ:F2} .. {b.MaxX:F2},{b.MaxY:F2},{b.MaxZ:F2}");
            else
                _xrPlaceMiniature = false;   // no bounds: the VR start at scale 1 (head at the camera)
            _xrBoundsReady = true;

            // Initialize WebGL blit helper for WebGL XR fallback
            if (_xrService.IsWebGLFallback && _xrService.GLContext != null)
            {
                _xrBlit = new WebGLXRBlit();
                _xrBlit.Initialize(_xrService.GLContext);
            }

            Console.WriteLine($"[Studio] {mode} session active (WebGL fallback: {_xrService.IsWebGLFallback})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Failed to enter {mode}: {ex.Message}");
            _xrActive = false;
            _xrBlit?.Dispose();
            _xrBlit = null;
            _xrService.OnXRFrame -= OnXRFrame;
            _xrService.OnSessionEnded -= OnXRSessionEnded;
            // Resume canvas rendering
            RequestFrame();
        }
    }

    // XR frame cost, logged every XRStatsFrames frames: what the per-eye render + copy costs on this device.
    const int XRStatsFrames = 90;
    readonly System.Diagnostics.Stopwatch _xrFrameClock = new();
    double _xrFrameMsSum;
    int _xrFrameCount;
    long _xrStatsStart;

    private void OnXRFrame(XRFrameData frameData)
    {
        _xrFrameClock.Restart();
        if (_xrFrameCount == 0) _xrStatsStart = System.Diagnostics.Stopwatch.GetTimestamp();
        try { RenderXRFrame(frameData); }
        finally
        {
            _xrFrameMsSum += _xrFrameClock.Elapsed.TotalMilliseconds;
            if (++_xrFrameCount == XRStatsFrames)
            {
                double wall = System.Diagnostics.Stopwatch.GetElapsedTime(_xrStatsStart).TotalMilliseconds;
                var vp = frameData.Views.Length > 0 ? frameData.Views[0].Viewport : null;
                Console.WriteLine($"[XR] {XRStatsFrames} frames: {_xrFrameMsSum / XRStatsFrames:F1} ms submit per frame " +
                    $"({frameData.Views.Length} views of {vp?.Width}x{vp?.Height}), {1000.0 * XRStatsFrames / wall:F0} fps, " +
                    $"{(frameData.IsWebGLFallback ? "WebGL copy path" : "WebGPU layer")}, head at {_xrHeadInScene:F2} in the scene");
                _xrFrameCount = 0; _xrFrameMsSum = 0;
            }
        }
    }

    // Room (local-floor) -> scene: the head starts where the desktop camera was, facing its way (XRSceneAlignment);
    // in AR, the scene starts as a miniature in front (_xrPlaceMiniature, sized from _xrSceneBox).
    System.Numerics.Matrix4x4? _xrSceneFromRoom;
    bool _xrPlaceMiniature, _xrBoundsReady = true;
    bool _xrForceAlpha; // &xralpha=1 (diagnostics)
    // AR placement: while placing, the miniature's bottom centre (_xrAnchorScene) follows where the right controller
    // points on a real surface (hit-test); a trigger press drops it there, another picks it up again.
    bool _xrPlacing, _xrTriggerWasDown, _xrHitLogged, _xrExitWasDown;
    System.Numerics.Vector3? _xrLastHit;
    System.Numerics.Vector3 _xrAnchorScene;
    SplatBounds.Aabb? _xrSceneBox;
    // Thumbstick move / snap-turn / rise, applied to _xrSceneFromRoom each frame.
    readonly XRLocomotion _xrLocomotion = new();
    // Grips: one drags the scene, both scale and turn it (XRWorldGrab).
    readonly XRWorldGrab _xrWorldGrab = new();
    double _xrLastTime = -1;
    System.Numerics.Vector3 _xrHeadInScene;
    bool _xrStickLog; // log the first controller input of a session (shows the sticks are read)

    private void RenderXRFrame(XRFrameData frameData)
    {
        if (_xrSceneFromRoom == null)
        {
            if (!_xrBoundsReady) return;   // AR bounds still being measured: passthrough only
            var cam = _sceneManager.Camera;
            if (_xrPlaceMiniature && _xrSceneBox is { } box)
            {
                _xrSceneFromRoom = XRSceneAlignment.SceneFromRoomMiniature(frameData.HeadPosition, frameData.HeadOrientation, cam.Forward, box);
                _xrAnchorScene = new System.Numerics.Vector3(box.CentreX, box.MinY, box.CentreZ);
                _xrPlacing = true;
                Console.WriteLine($"[XR] scene placed as a miniature: {XRWorldGrab.Scale(_xrSceneFromRoom.Value):G3} scene units per metre");
            }
            else
            {
                float scale = _xrSceneBox is { } b ? XRSceneAlignment.ComfortScale(b, cam.Position, cam.Forward) : 1f;
                _xrSceneFromRoom = XRSceneAlignment.SceneFromRoom(frameData.HeadPosition, frameData.HeadOrientation, cam.Position, cam.Forward, scale);
                Console.WriteLine($"[XR] room placed in the scene: head {frameData.HeadPosition} -> camera {cam.Position}, facing {cam.Forward}, {scale:G3} scene units per metre");
            }
        }
        // dt clamped: a stalled frame must not throw the viewer across the scene.
        float dt = _xrLastTime < 0 ? 0f : (float)Math.Clamp((frameData.Time - _xrLastTime) / 1000.0, 0.0, 0.1);
        _xrLastTime = frameData.Time;
        bool wasGrabbing = _xrWorldGrab.Active;
        _xrSceneFromRoom = _xrWorldGrab.Step(_xrSceneFromRoom.Value, frameData.LeftGrip, frameData.LeftGripPosition,
            frameData.RightGrip, frameData.RightGripPosition);
        if (wasGrabbing && !_xrWorldGrab.Active)
            Console.WriteLine($"[XR] world grab released: scale {XRWorldGrab.Scale(_xrSceneFromRoom.Value):G3} scene units per metre");
        if (frameData.ExitButton && !_xrExitWasDown)
        {
            Console.WriteLine("[XR] B/Y pressed: leaving the session");
            _xrService.RequestEnd();
        }
        _xrExitWasDown = frameData.ExitButton;
        if (frameData.MenuButton && !_xrMenuWasDown)
        {
            EnsureXRMenu().Toggle(frameData.HeadPosition, frameData.HeadOrientation);
            Console.WriteLine(_xrMenu!.IsOpen ? "[XR] menu opened" : "[XR] menu closed");
        }
        _xrMenuWasDown = frameData.MenuButton;
        // While the menu is open the trigger belongs to it (clicks), not to the scene (AR placement).
        bool menuOpen = _xrMenu?.IsOpen == true;
        if (_xrMenu != null) _xrMenu.Step(frameData.RightRayOrigin, frameData.RightRayDirection, frameData.RightTrigger, dt);
        // The box tool (menu: Select box) owns the right trigger while it is on and the menu is closed.
        bool boxTool = _xrBox.Active && !menuOpen;
        if (_xrBox.Step(boxTool ? frameData.RightRayOrigin : null, boxTool && frameData.RightTrigger, _xrSceneFromRoom.Value))
        {
            _selection = SplatEditor.Volume.Box(_xrBox.BoxToScene!.Value);
            _ = CountSelectionAsync().ContinueWith(_ => RefreshXRMenu());
            Console.WriteLine("[XR] selection box drawn");
        }
        bool trigger = !menuOpen && !_xrBox.Active && (frameData.LeftTrigger || frameData.RightTrigger);
        if (trigger && !_xrTriggerWasDown && _xrPlaceMiniature)
        {
            _xrPlacing = !_xrPlacing;
            Console.WriteLine(_xrPlacing ? "[XR] AR placement: picked up (follows the controller ray)"
                : $"[XR] AR placement: dropped{(_xrLastHit is { } at ? $" at {at:F2}" : " (no surface hit yet)")}");
        }
        _xrTriggerWasDown = trigger;
        if (frameData.HitPosition is { } hit)
        {
            _xrLastHit = hit;
            if (!_xrHitLogged) { _xrHitLogged = true; Console.WriteLine($"[XR] AR hit-test: ray from {frameData.RightRayOrigin:F2} meets a surface at {hit:F2}"); }
            if (_xrPlacing && !_xrWorldGrab.Active)
                _xrSceneFromRoom = XRSceneAlignment.MoveAnchorTo(_xrSceneFromRoom.Value, _xrAnchorScene, hit);
        }
        // Sticks while no grip holds the scene (a grab restarts from its own start transform). Speed follows the
        // scale, so a scene grown by the grips is not crossed faster in room terms.
        if (!_xrWorldGrab.Active)
            _xrSceneFromRoom = _xrLocomotion.Step(_xrSceneFromRoom.Value, frameData.HeadPosition, frameData.HeadOrientation,
                frameData.LeftStick, frameData.RightStick, dt,
                (_cameraController?.MoveSpeed ?? 1f) * _xrSpeedScale * XRWorldGrab.Scale(_xrSceneFromRoom.Value));
        if (_xrStickLog && (frameData.LeftStick != default || frameData.RightStick != default))
        {
            _xrStickLog = false;
            Console.WriteLine($"[XR] controller input: left {frameData.LeftStick}, right {frameData.RightStick}");
        }
        _xrHeadInScene = System.Numerics.Vector3.Transform(frameData.HeadPosition, _xrSceneFromRoom.Value);
        foreach (var v in frameData.Views)
        {
            v.RoomViewMatrix = v.ViewMatrix;
            v.ViewMatrix = XRSceneAlignment.SceneView(v.ViewMatrix, _xrSceneFromRoom.Value);
        }
        if (frameData.IsWebGLFallback && _gpuRenderer.XRSorted)
        {
            // One sort per frame from the head (scene space), a frustum wide enough for both eyes, then each eye.
            var m = _xrSceneFromRoom.Value;
            var head = new CameraParams
            {
                Width = 1000, Height = 1000, FocalX = 182, FocalY = 182, CenterX = 500, CenterY = 500, // ~140 deg: both eyes
                Near = _sceneManager.Camera.Near, Far = _sceneManager.Camera.Far,
                Position = System.Numerics.Vector3.Transform(frameData.HeadPosition, m),
                Forward = System.Numerics.Vector3.Normalize(System.Numerics.Vector3.TransformNormal(
                    System.Numerics.Vector3.Transform(-System.Numerics.Vector3.UnitZ, frameData.HeadOrientation), m)),
                Up = System.Numerics.Vector3.Normalize(System.Numerics.Vector3.TransformNormal(
                    System.Numerics.Vector3.Transform(System.Numerics.Vector3.UnitY, frameData.HeadOrientation), m)),
            };
            var cullProj = CameraParams.CreateWebGpuProjection(head.FocalX, head.FocalY, head.CenterX, head.CenterY,
                head.Width, head.Height, head.Near, head.Far);
            _gpuRenderer.BeginXRFrameSorted(head, head.ViewMatrix * cullProj);
            var layer = _xrService.WebGLLayer!;
            foreach (var view in frameData.Views)
            {
                var vp = view.Viewport;
                var roomViewProj = view.RoomViewMatrix * view.ProjectionMatrix;
                _gpuRenderer.RenderXRViewSortedToCanvas(view.ViewMatrix, view.ProjectionMatrix, (int)vp.Width, (int)vp.Height, _xrCasEnabled,
                    _xrMenu?.IsOpen == true ? (enc, color, depth) => _xrMenu.Draw(_gameUI.Renderer, roomViewProj, enc, color, depth) : null);
                if (_xrBox.BoxToRoom(_xrSceneFromRoom.Value) is { } boxRoom)
                    _gpuRenderer.RenderXROverlayToCanvas((enc, color, depth) =>
                        DrawXRBox(enc, color, depth, roomViewProj, boxRoom, frameData.HeadPosition, _xrBox.Dragging));
                _xrBlit!.Blit(_gpuRenderer.XRBridgeCanvas!, layer.Framebuffer, vp);
            }
            return;
        }
        if (frameData.IsWebGLFallback)
        {
            // WebGL fallback: render each eye to WebGPU bridge canvas, blit to XR framebuffer
            var xrLayer = _xrService.WebGLLayer!;
            foreach (var view in frameData.Views)
            {
                var vp = view.Viewport;
                _gpuRenderer.RenderXRViewToCanvas(
                    view.ViewMatrix, view.ProjectionMatrix,
                    (int)vp.Width, (int)vp.Height, _xrCasEnabled);
                _xrBlit!.Blit(
                    _gpuRenderer.XRBridgeCanvas!,
                    xrLayer.Framebuffer, vp);
            }
        }
        else
        {
            // WebGPU native XR (future — once browsers support XRGPUBinding)
            foreach (var view in frameData.Views)
            {
                _gpuRenderer.RenderXRView(
                    view.ViewMatrix, view.ProjectionMatrix,
                    view.ColorTexture!, view.DepthStencilTexture,
                    (int)view.Viewport.X, (int)view.Viewport.Y,
                    (int)view.Viewport.Width, (int)view.Viewport.Height);
            }
        }
    }

    private void OnXRSessionEnded()
    {
        _xrService.OnXRFrame -= OnXRFrame;
        _xrService.OnSessionEnded -= OnXRSessionEnded;

        // Clean up XR resources
        _xrBlit?.Dispose();
        _xrBlit = null;
        _gpuRenderer.DisposeXRBridge();
        _gpuRenderer.XRTransparent = false;

        // Resume canvas RAF loop
        _xrActive = false;
        Console.WriteLine("[Studio] XR session ended, resuming canvas rendering");
        RequestFrame();
    }

    // ── In-headset menu (X/A opens it; right ray + trigger click) ─────────────────────────────────────────────
    XRMenu? _xrMenu;
    bool _xrMenuWasDown;
    float _xrSpeedScale = 1f;
    UIButton? _xrTurnButton, _xrSpeedButton, _xrPlaceButton, _xrSelectButton;
    UILabel? _xrEditLabel;
    readonly XRBoxTool _xrBox = new();

    async Task XREditAsync(SplatEditor.Mode mode)
    {
        await ApplyEditAsync(mode);
        _xrBox.Clear();
        RefreshXRMenu();
    }

    /// <summary>The selection box's 12 edges as thin quads turned toward the head (room space, metres).</summary>
    void DrawXRBox(SpawnDev.SpawnJS.JSObjects.GPUCommandEncoder enc, SpawnDev.SpawnJS.JSObjects.GPUTextureView color,
        SpawnDev.SpawnJS.JSObjects.GPUTextureView depth, System.Numerics.Matrix4x4 roomViewProj,
        System.Numerics.Matrix4x4 boxRoom, System.Numerics.Vector3 head, bool dragging)
    {
        var r = _gameUI.Renderer;
        var c = XRBoxTool.Corners(boxRoom);
        float cr = dragging ? 1f : 0.45f, cg = dragging ? 0.85f : 0.86f, cb = dragging ? 0.3f : 1f;
        foreach (var (ia, ib) in XRBoxTool.Edges)
        {
            var a = c[ia]; var b = c[ib];
            var side = System.Numerics.Vector3.Cross(b - a, (a + b) * 0.5f - head);
            if (side.LengthSquared() < 1e-12f) continue;
            side = System.Numerics.Vector3.Normalize(side) * 0.003f;
            r.DrawWorldRayQuad(a - side, a + side, b - side, b + side, cr, cg, cb, 0.95f);
        }
        r.EndWorldSpace(enc, color, depth, roomViewProj);
    }

    XRMenu EnsureXRMenu()
    {
        if (_xrMenu != null) { RefreshXRMenu(); return _xrMenu; }
        _xrMenu = new XRMenu();
        var p = _xrMenu.Panel;
        p.AddChild(new UILabel
        {
            X = 24, Y = 18, Text = "SpawnScene", FontSize = FontSize.Heading,
            Color = UITheme.Current.TextPrimary,
        });
        p.AddChild(new UILabel
        {
            X = 24, Y = 58, Text = "Point with the right controller, trigger to choose. A/X closes, B/Y exits.",
            FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
        });
        const float bw = 244, bh = 50, x0 = 24, x1 = 292;
        UIButton Button(float x, float y, string text, Action click)
            => p.AddChild(new UIButton
            {
                X = x, Y = y, Width = bw, Height = bh, Text = text, FontSize = FontSize.Body, OnClick = click,
            });
        Button(x0, 96, "Reset view", () =>
        {
            _xrSceneFromRoom = null;   // re-placed on the next frame (comfort scale / AR miniature)
            _xrBox.Clear();
            _xrMenu!.Close();
            Console.WriteLine("[XR] menu: reset view");
        });
        _xrTurnButton = Button(x1, 96, "", () =>
        {
            _xrLocomotion.SnapTurnDegrees = _xrLocomotion.SnapTurnDegrees >= 45f ? 30f : 45f;
            RefreshXRMenu();
        });
        _xrSpeedButton = Button(x0, 156, "", () =>
        {
            _xrSpeedScale = _xrSpeedScale >= 2f ? 0.5f : _xrSpeedScale * 2f;
            RefreshXRMenu();
        });
        _xrPlaceButton = Button(x1, 156, "", () =>
        {
            _xrPlacing = !_xrPlacing;
            _xrMenu!.Close();
            Console.WriteLine(_xrPlacing ? "[XR] menu: place on a surface" : "[XR] menu: placement off");
        });
        // Editing: draw a box with the trigger, then delete it / keep only it / undo (SplatEditor, as on the desktop).
        _xrSelectButton = Button(x0, 230, "", () =>
        {
            _xrBox.Active = !_xrBox.Active;
            if (_xrBox.Active) _xrMenu!.Close();
            RefreshXRMenu();
            Console.WriteLine(_xrBox.Active ? "[XR] menu: select box on" : "[XR] menu: select box off");
        });
        Button(x1, 230, "Clear box", () => { _xrBox.Clear(); _selection = null; _selectedCount = 0; RefreshXRMenu(); });
        Button(x0, 290, "Delete", () => _ = XREditAsync(SplatEditor.Mode.DeleteInside));
        Button(x1, 290, "Keep only", () => _ = XREditAsync(SplatEditor.Mode.KeepInside));
        Button(x0, 350, "Undo", () => _ = UndoEditAsync().ContinueWith(_ => RefreshXRMenu()));
        Button(x1, 350, "Save as new scene", () => _ = SaveEditedSceneAsync().ContinueWith(_ => RefreshXRMenu()));
        Button(x1, 410, "Exit", () => { _xrMenu!.Close(); _xrService.RequestEnd(); });
        _xrEditLabel = p.AddChild(new UILabel
        {
            X = 24, Y = 476, Text = "", FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
        });
        RefreshXRMenu();
        return _xrMenu;
    }

    void RefreshXRMenu()
    {
        if (_xrTurnButton != null) _xrTurnButton.Text = $"Snap turn: {_xrLocomotion.SnapTurnDegrees:0} deg";
        if (_xrSpeedButton != null) _xrSpeedButton.Text = $"Move speed: x{_xrSpeedScale:0.#}";
        if (_xrSelectButton != null) _xrSelectButton.Text = _xrBox.Active ? "Select box: on" : "Select box";
        if (_xrEditLabel != null)
            _xrEditLabel.Text = _editBusy ? "Working..." : _editNote != null ? _editNote : _selection != null ? $"{_selectedCount:N0} splats in the box"
                : _xrBox.Active ? "Hold the trigger and drag to draw a box" : "Edit: choose Select box, draw it, then Delete or Keep only";
        if (_xrPlaceButton != null)
        {
            _xrPlaceButton.Visible = _xrPlaceMiniature;   // AR only
            _xrPlaceButton.Text = _xrPlacing ? "Stop placing" : "Place on surface";
        }
    }
}
