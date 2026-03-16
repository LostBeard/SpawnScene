using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
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

        // Poll input
        _inputManager?.Poll();

        // Bridge InputManager keyboard to CameraController (it tracks its own key set)
        if (_inputManager != null && _cameraController != null && _state == StudioState.SceneViewer)
        {
            foreach (var key in _inputManager.FrameKeysPressed)
                _cameraController.OnKeyDown(key);
            foreach (var key in _inputManager.FrameKeysReleased)
                _cameraController.OnKeyUp(key);

            // Request pointer lock on left click — but only if no UI element was hit
            if (_inputManager.WasMousePressed(0) && !_isPointerLocked)
            {
                var hit = _uiRoot.HitTest(_inputManager.MousePosition);
                if (hit == null)
                {
                    using var canvas = new HTMLCanvasElement(_canvasRef);
                    canvas.RequestPointerLock();
                }
            }

            // Scroll → zoom
            if (_isPointerLocked && MathF.Abs(_inputManager.ScrollDelta) > 0.1f)
                _cameraController.OnWheel(_inputManager.ScrollDelta);
        }

        // Update UI (only when not pointer-locked, so clicks go to UI not camera)
        if (_inputManager != null && !_isPointerLocked)
            _uiRoot.Update(_inputManager, dt);

        // Camera movement (only in viewer state with pointer lock)
        if (_state == StudioState.SceneViewer && _isPointerLocked)
            _cameraController?.Tick(dt);

        // Update dynamic HUD labels
        if (_state == StudioState.SceneViewer)
            UpdateViewerHud();

        // Render 3D scene (if active)
        if (_state == StudioState.SceneViewer && _sceneManager.HasScene)
        {
            _renderService.RenderFrame();

            // Capture scene thumbnail after delay (allows scene to converge)
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

        // Render UI overlay on top of scene (or as full-screen UI)
        RenderUIOverlay();

        RequestFrame();
    }

    private void RenderUIOverlay()
    {
        if (_uiRenderer == null || _context == null || _device == null) return;

        _uiRenderer.Begin(_canvasWidth, _canvasHeight);
        _uiRoot.Draw(_uiRenderer);

        // Get swapchain texture for UI overlay
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
                ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.08, A = 1.0 },
            };
            using var clearPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { clearAttach },
            });
            clearPass.End();
        }

        _uiRenderer.End(encoder, colorView);

        using var cmdBuf = encoder.Finish();
        _queue!.Submit(new[] { cmdBuf });
    }

    // ─── Canvas Sizing ───

    private void UpdateCanvasSize()
    {
        using var container = new HTMLElement(_containerRef);
        int cssWidth = container.ClientWidth;
        int cssHeight = container.ClientHeight;
        if (cssWidth <= 0 || cssHeight <= 0) { cssWidth = 960; cssHeight = 640; }

        float dpr = BlazorJSRuntime.JS.Get<float>("devicePixelRatio");
        if (dpr < 1f) dpr = 1f;
        if (dpr > 2f) dpr = 2f;

        _canvasWidth = (int)(cssWidth * dpr);
        _canvasHeight = (int)(cssHeight * dpr);
        _lastResizeWidth = cssWidth;
        _lastResizeHeight = cssHeight;
    }

    private async void OnWindowResize(UIEvent e)
    {
        try
        {
            using var container = new HTMLElement(_containerRef);
            int cssWidth = container.ClientWidth;
            int cssHeight = container.ClientHeight;
            if (cssWidth == _lastResizeWidth && cssHeight == _lastResizeHeight) return;
            _lastResizeWidth = cssWidth;
            _lastResizeHeight = cssHeight;

            float dpr = BlazorJSRuntime.JS.Get<float>("devicePixelRatio");
            if (dpr < 1f) dpr = 1f;
            if (dpr > 2f) dpr = 2f;

            int newWidth = (int)(cssWidth * dpr);
            int newHeight = (int)(cssHeight * dpr);
            if (newWidth == _canvasWidth && newHeight == _canvasHeight) return;

            _canvasWidth = newWidth;
            _canvasHeight = newHeight;

            _renderService.HandleResize(_canvasWidth, _canvasHeight);

            // Rebuild UI for new dimensions
            if (_state == StudioState.ProjectBrowser)
                BuildProjectBrowserUI();
            else
                BuildViewerHudUI();

            await InvokeAsync(StateHasChanged);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Resize error: {ex.Message}");
        }
    }

    // ─── Pointer Lock (for FPS camera in viewer mode) ───

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

    // ─── Scene Events ───

    private void OnSceneChanged()
    {
        _state = StudioState.SceneViewer;
        _cameraController?.FitToScene();
        BuildViewerHudUI();
    }
}
