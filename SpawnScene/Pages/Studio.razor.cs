using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using SpawnScene.Services;
using SpawnScene.UI;
using SpawnScene.UI.Elements;

namespace SpawnScene.Pages;

public partial class Studio : IAsyncDisposable
{
    [Inject] private GpuService _gpuService { get; set; } = default!;
    [Inject] private SceneManager _sceneManager { get; set; } = default!;
    [Inject] private RenderService _renderService { get; set; } = default!;
    [Inject] private ProjectService _projectService { get; set; } = default!;
    [Inject] private DepthEstimationService _depthService { get; set; } = default!;
    [Inject] private DepthToGaussianKernel _gaussianKernel { get; set; } = default!;
    [Inject] private GpuGaussianRenderer _gpuRenderer { get; set; } = default!;
    [Inject] private HttpClient _http { get; set; } = default!;
    [Inject] private NavigationManager _nav { get; set; } = default!;
    [Inject] private XRService _xrService { get; set; } = default!;
    [Inject] private MultiViewGenerationService _multiViewService { get; set; } = default!;
    [Inject] private BlazorJSRuntime _js { get; set; } = default!;
    // SpawnDev.ILGPU.ML — created on-demand after GPU init (not injected)

    private ElementReference _canvasRef;
    private ElementReference _containerRef;
    private InputFile? _fileInput;
    private CameraController? _cameraController;

    // Canvas sizing
    private int _canvasWidth = 960;
    private int _canvasHeight = 640;
    private int _lastResizeWidth;
    private int _lastResizeHeight;

    // Render loop
    private bool _renderLoopRunning;
    private ActionCallback<double>? _rafCallback;
    private Window? _window;
    private double _lastFrameTime;

    // Pointer lock
    private Document? _document;
    private bool _isPointerLocked;

    // XR (VR/AR)
    private WebGLXRBlit? _xrBlit;
    private bool _xrCasEnabled; // CAS sharpening in XR — off by default
    private bool _xrActive; // true while XR session is active (pauses canvas RAF)

    // WebGPU UI system
    private FontAtlas? _fontAtlas;
    private UIRenderer? _uiRenderer;
    private InputManager? _inputManager;
    private UIElement _uiRoot = new();

    // App state
    private enum StudioState { ProjectBrowser, ProjectDetail, SceneViewer }
    private StudioState _state = StudioState.ProjectBrowser;
    private List<Project>? _projects;
    private Project? _activeProject;
    private string? _statusMessage;

    // Thumbnail cache: key → (GPUTexture, GPUTextureView)
    private readonly Dictionary<string, (GPUTexture tex, GPUTextureView view)> _thumbnailCache = new();
    private string? _pendingThumbnailSceneId;
    private string? _pendingThumbnailProjectId;
    private int _thumbnailDelayFrames;

    // GPU resources for UI overlay
    private GPUDevice? _device;
    private GPUQueue? _queue;
    private GPUCanvasContext? _context;
    private string _canvasFormat = "bgra8unorm";

    // Dynamic HUD labels (updated each frame)
    private UILabel? _hudSplatLabel;
    private UILabel? _hudFpsLabel;
    private UIPanel? _settingsPanel;
    private bool _showSettings;

    protected override void OnInitialized()
    {
        _cameraController = new CameraController(_sceneManager);
        _sceneManager.OnSceneChanged += OnSceneChanged;
    }

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (!firstRender) return;

        UpdateCanvasSize();
        StateHasChanged();
        await Task.Yield();

        // Attach canvas to the existing render service (for 3D scene rendering)
        await _renderService.AttachCanvasAsync(_canvasRef);

        // Get WebGPU device for UI rendering
        var webGpuAccel = _gpuService.WebGPUAccelerator;
        var nativeAccel = webGpuAccel.NativeAccelerator;
        _device = nativeAccel.NativeDevice;
        _queue = nativeAccel.Queue;

        using var canvas = new HTMLCanvasElement(_canvasRef);
        _context = canvas.GetContext<GPUCanvasContext>("webgpu");

        using var navigator = _js.Get<Navigator>("navigator");
        using var gpu = navigator.Gpu;
        if (gpu is not null)
            _canvasFormat = gpu.GetPreferredCanvasFormat();

        // Initialize WebGPU UI system
        _fontAtlas = new FontAtlas();
        _fontAtlas.Init(_device, _queue);

        _uiRenderer = new UIRenderer();
        _uiRenderer.Init(_device, _queue, _fontAtlas, _canvasFormat);

        _inputManager = new InputManager();
        _inputManager.Attach(_canvasRef);

        // Load projects from OPFS and build initial UI
        _projects = await _projectService.ListProjectsAsync();
        BuildProjectBrowserUI();

        // If a scene is already loaded, switch to viewer
        if (_sceneManager.HasScene)
        {
            _state = StudioState.SceneViewer;
            _cameraController?.FitToScene();
            BuildViewerHudUI();
        }

        // Start render loop
        _window = _js.Get<Window>("window");
        _window.OnResize += OnWindowResize;

        _document = _js.Get<Document>("document");
        _document.OnPointerLockChange += OnPointerLockChange;
        _document.OnMouseMove += OnNativeMouseMove;

        StartRenderLoop();

        Console.WriteLine($"[Studio] Initialized: {_canvasWidth}×{_canvasHeight}, UI ready");

        // Validate MatMul kernel (temporary — remove after confirmed working)
        _ = Task.Run(async () =>
        {
            try
            {
                // Build marker — update on every code change to detect stale DLLs
                Console.WriteLine("[NeuralOps] Build: 2026-03-17 head-arch-inspect");

                var accelerator = _gpuService.WebGPUAccelerator;
                var weightLoader = new SpawnDev.ILGPU.ML.WeightLoader(accelerator, _http);
                await weightLoader.LoadAsync();

                // Print all head.* weight names with shapes to understand DPT head architecture
                var headWeights = weightLoader.Shapes
                    .Where(kv => kv.Key.StartsWith("head."))
                    .OrderBy(kv => kv.Key)
                    .ToList();
                Console.WriteLine($"[DPT] {headWeights.Count} head.* tensors:");
                foreach (var (name, shape) in headWeights)
                    Console.WriteLine($"  {name}: [{string.Join(",", shape)}] = {shape.Aggregate(1, (a, b) => a * b):N0}");

                Console.WriteLine("[NeuralOps] All tests complete!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MatMul] Validation error: {ex.Message}");
            }
        });
    }

    public async ValueTask DisposeAsync()
    {
        _renderLoopRunning = false;
        _sceneManager.OnSceneChanged -= OnSceneChanged;

        if (_window != null)
        {
            _window.OnResize -= OnWindowResize;
            _window.Dispose();
        }
        if (_document != null)
        {
            _document.OnPointerLockChange -= OnPointerLockChange;
            _document.OnMouseMove -= OnNativeMouseMove;
            _document.Dispose();
        }

        if (_xrService.IsSessionActive)
            await _xrService.ExitSessionAsync();

        _xrBlit?.Dispose();
        _xrBlit = null;

        _inputManager?.Dispose();
        _uiRenderer?.Dispose();
        _fontAtlas?.Dispose();
        _rafCallback?.Dispose();

        // Clean up thumbnail textures
        foreach (var (tex, view) in _thumbnailCache.Values)
        {
            view.Dispose();
            tex.Destroy();
            tex.Dispose();
        }
        _thumbnailCache.Clear();
    }
}
