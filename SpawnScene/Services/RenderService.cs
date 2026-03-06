using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS.JSObjects;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Handles rendering Gaussian Splat scenes using the GPU-only renderer.
/// The renderer blits directly to the canvas via WebGPU — no CPU image copies.
/// IMPORTANT: The canvas is used exclusively for WebGPU — never get a 2D context
/// from it, as that would lock the canvas to 2D mode and prevent WebGPU access.
/// </summary>
public class RenderService : IDisposable
{
    private readonly GpuService _gpuService;
    private readonly SceneManager _sceneManager;
    private readonly GpuGaussianRenderer _gpuRenderer;
    private HTMLCanvasElement? _canvas;
    private int _frameCount;
    private DateTime _lastFpsTime = DateTime.UtcNow;
    private float _fps;
    private bool _rendererAttached;
    private GaussianScene? _uploadedScene; // Track which scene is in GPU buffer

    /// <summary>Current frames per second.</summary>
    public float Fps => _fps;

    /// <summary>Fired when FPS is updated (approximately once per second).</summary>
    public event Action<float>? OnFpsUpdated;

    /// <summary>Fired when a render completes.</summary>
    public event Action? OnRenderComplete;

    public RenderService(GpuService gpuService, SceneManager sceneManager, GpuGaussianRenderer gpuRenderer)
    {
        _gpuService = gpuService;
        _sceneManager = sceneManager;
        _gpuRenderer = gpuRenderer;
        _sceneManager.OnSceneChanged += OnSceneChanged;
        _sceneManager.OnCameraChanged += () => { };
    }

    private async void OnSceneChanged()
    {
        if (_rendererAttached && _sceneManager.ActiveScene != null)
        {
            var scene = _sceneManager.ActiveScene;
            if (_uploadedScene != scene)
            {
                // Only CPU-upload when scene has actual Gaussian data.
                // GPU fast-path scenes (GpuSplatCount > 0, empty Gaussians array) are
                // pre-loaded via UploadSceneFromGpuBuffer — skip the CPU upload.
                if (scene.Gaussians?.Length > 0)
                    await _gpuRenderer.UploadScene(scene);
                _uploadedScene = scene;
            }
        }
    }

    /// <summary>
    /// Mark a GPU-preloaded scene so that when it becomes the ActiveScene
    /// the render service does not redundantly call UploadScene.
    /// Call this BEFORE setting SceneManager.ActiveScene.
    /// </summary>
    public void SetActiveSceneGpuLoaded(SpawnScene.Models.GaussianScene scene)
    {
        _uploadedScene = scene;
    }

    /// <summary>
    /// Attach the renderer to a canvas element reference.
    /// The canvas is claimed exclusively for WebGPU rendering.
    /// </summary>
    public async Task AttachCanvasAsync(ElementReference canvasRef)
    {
        _canvas?.Dispose();
        _canvas = new HTMLCanvasElement(canvasRef);
        _sceneManager.ResizeViewport(_canvas.Width, _canvas.Height);

        // Initialize GPU if needed
        if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();

        // Attach canvas to GPU renderer for direct WebGPU blitting
        _gpuRenderer.AttachCanvas(_canvas);
        _rendererAttached = true;
    }

    /// <summary>Mark the scene as needing a redraw.</summary>
    public void Invalidate() { }

    /// <summary>
    /// Render a single frame of the current scene using the GPU-only renderer.
    /// </summary>
    public async Task RenderFrameAsync()
    {
        if (_canvas == null || !_rendererAttached) return;

        var scene = _sceneManager.ActiveScene;
        var camera = _sceneManager.Camera;

        // For GPU fast-path scenes (GpuSplatCount > 0, empty Gaussians array),
        // data is already in the renderer — use HasGpuData to confirm readiness.
        var hasGpuData = _gpuRenderer.HasGpuData;
        if (scene == null || (scene.Count == 0 && !hasGpuData)) return;

        // Ensure scene is uploaded (skipped for GPU fast-path via SetActiveSceneGpuLoaded)
        if (_uploadedScene != scene)
        {
            if (scene.Gaussians?.Length > 0)
                await _gpuRenderer.UploadScene(scene);
            _uploadedScene = scene;
        }

        // Render using native WebGPU (single Draw() call — no async needed)
        await _gpuRenderer.RenderAsync(scene, camera);

        // Track FPS
        _frameCount++;
        var now = DateTime.UtcNow;
        var elapsed = (now - _lastFpsTime).TotalSeconds;
        if (elapsed >= 1.0)
        {
            _fps = (float)(_frameCount / elapsed);
            _frameCount = 0;
            _lastFpsTime = now;
            OnFpsUpdated?.Invoke(_fps);
        }

        OnRenderComplete?.Invoke();
    }

    /// <summary>Handle canvas resize: update viewport and renderer textures.</summary>
    public void HandleResize(int newWidth, int newHeight)
    {
        _sceneManager.ResizeViewport(newWidth, newHeight);
        _gpuRenderer.ResizeCanvas(newWidth, newHeight);
        Console.WriteLine($"[RenderService] Resized to {newWidth}×{newHeight}");
    }

    public void Dispose()
    {
        _canvas?.Dispose();
        GC.SuppressFinalize(this);
    }
}
