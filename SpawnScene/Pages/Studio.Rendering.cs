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
            if (primary != null && primary.WasPressed && !_isPointerLocked)
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
                _thumbnailDelayFrames--;
                if (_thumbnailDelayFrames <= 0)
                {
                    var sceneId = _pendingThumbnailSceneId;
                    var projId = _pendingThumbnailProjectId;
                    _pendingThumbnailSceneId = null;
                    _pendingThumbnailProjectId = null;
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
        _queue!.Submit(new[] { cmdBuf });
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
        _state = StudioState.SceneViewer;
        _cameraController?.FitToScene();
        BuildViewerHudUI();
    }
}
