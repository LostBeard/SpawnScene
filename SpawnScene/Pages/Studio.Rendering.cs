using System.Numerics;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// Render loop, canvas sizing, pointer lock, scene events
public partial class Studio
{
    private void StartRenderLoop()
    {
        if (_renderLoopRunning) return;
        _renderLoopRunning = true;
        _lastFrameTime = 0;
        _rafCallback ??= new ActionCallback<double>(OnAnimationFrame);
        RequestFrame();
    }

    private void RequestFrame()
    {
        if (!_renderLoopRunning || _rafCallback == null || _window == null) return;
        _window.RequestAnimationFrame(_rafCallback);
    }

    readonly TouchNavigator _touchNavigator = new();
    readonly List<Vector2> _touchPositions = new();
    int _canvasCssHeight = 640;

    private void OnAnimationFrame(double timestamp)
    {
        if (!_renderLoopRunning || _xrActive) return;

        float dt = _lastFrameTime > 0 ? (float)((timestamp - _lastFrameTime) / 1000.0) : 1f / 60f;
        _lastFrameTime = timestamp;
        dt = Math.Min(dt, 0.1f);

        if (!_gameUI.IsInitialized)
        {
            RequestFrame();
            return;
        }

        // Poll unified input (mouse/keyboard/touch/XR)
        _gameUI.Input.Poll();
        var input = _gameUI.Input;

        // Bridge keyboard to CameraController
        if (_cameraController != null && _state == StudioState.SceneViewer)
        {
            foreach (var key in input.Keyboard.KeysPressed)
                _cameraController.OnKeyDown(key);
            foreach (var key in _prevKeysDown)
            {
                if (!input.Keyboard.IsKeyDown(key))
                    _cameraController.OnKeyUp(key);
            }
            _prevKeysDown.Clear();
            foreach (var key in input.Keyboard.KeysDown)
                _prevKeysDown.Add(key);

            var primary = input.PrimaryPointer;
            // Select mode (Edit toolbar): a drag over the scene draws a selection instead of grabbing the mouse.
            bool selecting = HandleSelectPointer(primary);
            if (!selecting && !_selectMode && primary != null && primary.WasPressed && !_isPointerLocked
                && primary.Type != SpawnDev.GameUI.Input.PointerType.Touch)   // touch navigates without a lock
            {
                var pos = primary.ScreenPosition ?? Vector2.Zero;
                var hit = _uiRoot.HitTest(pos);
                if (hit == null)
                {
                    using var canvas = _canvasRef.As<HTMLCanvasElement>();
                    canvas.RequestPointerLock();
                }
            }

            if (_isPointerLocked && primary != null && MathF.Abs(primary.ScrollDelta) > 0.1f)
                _cameraController.OnWheel(primary.ScrollDelta);

            // Touch (no pointer lock on phones/tablets): one finger looks, two pan and pinch (TouchNavigator).
            _touchPositions.Clear();
            bool touchStartsOnUi = false;
            foreach (var p in input.Pointers)
            {
                if (p.Type != SpawnDev.GameUI.Input.PointerType.Touch || !p.IsPressed || p.ScreenPosition is not { } tp) continue;
                _touchPositions.Add(tp);
                if (p.WasPressed && _uiRoot.HitTest(tp) != null) touchStartsOnUi = true;
            }
            var gesture = _touchNavigator.Step(_selectMode ? (IReadOnlyList<Vector2>)System.Array.Empty<Vector2>() : _touchPositions, touchStartsOnUi);
            if (gesture.LookPixels != Vector2.Zero)
            {
                // The scene stays under the finger: one CSS pixel is 1 / (focal length in CSS pixels) radians.
                var cam = _sceneManager.Camera;
                float focalCss = cam.FocalY * _canvasCssHeight / Math.Max(1, cam.Height);
                _cameraController.TouchLook(gesture.LookPixels, 1f / Math.Max(focalCss, 1f));
            }
            if (gesture.PanPixels != Vector2.Zero || gesture.PinchLog != 0f)
                _cameraController.TouchPanPinch(gesture.PanPixels, gesture.PinchLog);

            // Gamepad: left stick moves, right stick looks, LB/RB sink/rise, RT fast.
            var pad = input.Gamepad;
            if (pad.Connected)
                _cameraController.TickGamepad(pad.LeftStick, pad.RightStick, pad.IsButtonDown(4), pad.IsButtonDown(5), pad.IsButtonDown(7), dt);
        }

        // Update UI when not pointer-locked (clicks go to UI, not camera)
        if (!_isPointerLocked)
            _uiRoot.Update(input, dt);

        if (_state == StudioState.SceneViewer && _isPointerLocked)
            _cameraController?.Tick(dt);

        if (_state == StudioState.SceneViewer)
            UpdateViewerHud();

        if (_state == StudioState.SceneViewer && _sceneManager.HasScene)
        {
            _renderService.RenderFrame();

            if (_pendingThumbnailSceneId != null)
            {
                // A streamed scene: wait for the view's chunks (at most ~20 s), or the card shows chunk 0's coarse copy.
                if (_lodPager is { Busy: true } && _thumbnailHoldFrames++ < 1200) _thumbnailDelayFrames = Math.Max(_thumbnailDelayFrames, 10);
                _thumbnailDelayFrames--;
                if (_thumbnailDelayFrames <= 0)
                {
                    var sceneId = _pendingThumbnailSceneId;
                    var projId = _pendingThumbnailProjectId;
                    _pendingThumbnailSceneId = null;
                    _pendingThumbnailProjectId = null;
                    _thumbnailHoldFrames = 0;
                    CaptureSceneThumbnail(projId!, sceneId);
                }
            }
        }

        RenderUIOverlay();
        RequestFrame();
    }

    private void RenderUIOverlay()
    {
        if (!_gameUI.IsInitialized || _context == null || _device == null) return;
        // Novel-view measurement captures the canvas; the HUD would be scored as scene content.
        if (_hideUiOverlay) return;

        _gameUI.BeginRender(_canvasWidth, _canvasHeight);
        _uiRoot.Draw(_gameUI.Renderer);

        using var colorTexture = _context.GetCurrentTexture();
        using var colorView = colorTexture.CreateView();
        using var encoder = _device.CreateCommandEncoder();

        // If no scene is rendering, clear the canvas first
        if (_state == StudioState.ProjectBrowser || !_sceneManager.HasScene)
        {
            var clearAttach = new GPURenderPassColorAttachment
            {
                View = colorView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = new GPUColorDict { R = 0.05, G = 0.06, B = 0.08, A = 1.0 },
            };
            using var clearPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { clearAttach },
            });
            clearPass.End();
        }

        _gameUI.EndRender(encoder, colorView);

        using var cmdBuf = encoder.Finish();
        RawSubmit.Submit(_gpuService.IsInitialized ? _gpuService.WebGPUAccelerator : null, _queue!, new[] { cmdBuf });
    }

    // ─── Canvas Sizing ───

    private void UpdateCanvasSize()
    {
        using var container = _containerRef.As<HTMLElement>();
        int cssWidth = container.ClientWidth;
        int cssHeight = container.ClientHeight;
        if (cssWidth <= 0 || cssHeight <= 0) { cssWidth = 960; cssHeight = 640; }

        float dpr = _js.Get<float>("devicePixelRatio");
        if (dpr < 1f) dpr = 1f;
        _canvasWidth = Math.Max(1, (int)(cssWidth * dpr));
        _canvasHeight = Math.Max(1, (int)(cssHeight * dpr));
        _canvasCssHeight = cssHeight;

        using var canvas = _canvasRef.As<HTMLCanvasElement>();
        canvas.Width = _canvasWidth;
        canvas.Height = _canvasHeight;
        canvas.Style.SetProperty("width", $"{cssWidth}px");
        canvas.Style.SetProperty("height", $"{cssHeight}px");

        if (_gameUI.IsInitialized)
            _gameUI.SetViewport(_canvasWidth, _canvasHeight);
    }

    private async void OnWindowResize(UIEvent e)
    {
        try
        {
            int prevW = _canvasWidth, prevH = _canvasHeight;
            UpdateCanvasSize();
            if (_canvasWidth == prevW && _canvasHeight == prevH) return;

            _renderService.HandleResize(_canvasWidth, _canvasHeight);
            RebuildCurrentUI();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] resize failed: {ex.Message}");
        }
        await Task.CompletedTask;
    }

    private void RebuildCurrentUI()
    {
        switch (_state)
        {
            case StudioState.ProjectBrowser: BuildProjectBrowserUI(); break;
            case StudioState.ProjectDetail: BuildProjectDetailUI(); break;
            case StudioState.SceneViewer: BuildViewerHudUI(); break;
            case StudioState.Testing: BuildTestingUI(); break;
        }
    }

    private void OnPointerLockChange()
    {
        _isPointerLocked = _document?.PointerLockElement != null;
    }

    private void OnNativeMouseMove(MouseEvent e)
    {
        if (!_isPointerLocked || _cameraController == null || _state != StudioState.SceneViewer) return;
        _cameraController.OnMouseMove(e.MovementX, e.MovementY, isPointerLocked: true);
    }

    private void ReleasePointerLock()
    {
        if (_isPointerLocked)
            _document?.ExitPointerLock();
    }

    private void OnSceneChanged()
    {
        // Edits and their undo snapshots belong to the scene they were made on.
        _splatEditor.ClearUndo();
        _selection = null; _selectedCount = 0; _dragStart = _dragEnd = null; _editNote = null;
        _state = StudioState.SceneViewer;
        _cameraController?.FitToScene();
        BuildViewerHudUI();
    }
}
