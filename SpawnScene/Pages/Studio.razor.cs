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

    // Depth map visualization
    private GPUTexture? _depthMapTex;
    private GPUTextureView? _depthMapView;
    private int _depthMapW, _depthMapH;
    private bool _showDepthMap;

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

        // Clean up depth map visualization
        _depthMapView?.Dispose();
        _depthMapTex?.Destroy();
        _depthMapTex?.Dispose();
        _depthMapView = null;
        _depthMapTex = null;
    }

    /// <summary>
    /// Reads back depth result to CPU and uploads as a colorized RGBA GPUTexture for visualization.
    /// CPU readback is justified: display-only, not in the compute pipeline.
    /// </summary>
    private async Task CaptureDepthMapAsync(DepthResult depthResult)
    {
        if (_device == null || _queue == null || depthResult.RawDepthGpu == null) return;

        // Dispose old depth map
        _depthMapView?.Dispose(); _depthMapView = null;
        _depthMapTex?.Destroy(); _depthMapTex?.Dispose(); _depthMapTex = null;

        int w = depthResult.Width;
        int h = depthResult.Height;

        // CPU transfer: display only
        var depth = await depthResult.RawDepthGpu.CopyToHostAsync<float>(0, depthResult.RawDepthGpu.Length);

        // Use p2/p98 percentiles for visualization range (robust against outliers).
        // Sample every Nth pixel to keep sort fast in WASM (~150K elements max).
        int n = depth.Length;
        int stride = Math.Max(1, n / 150_000);
        int sampleCount = (n + stride - 1) / stride;
        var sample = new float[sampleCount];
        for (int i = 0, j = 0; i < n && j < sampleCount; i += stride, j++)
            sample[j] = depth[i];
        System.Array.Sort(sample);
        float vizMin = sample[(int)(sampleCount * 0.02f)];
        float vizMax = sample[Math.Min((int)(sampleCount * 0.98f), sampleCount - 1)];
        float range = vizMax > vizMin ? vizMax - vizMin : 1f;

        var rgba = new byte[w * h * 4];
        for (int i = 0; i < depth.Length; i++)
        {
            float t = Math.Clamp((depth[i] - vizMin) / range, 0f, 1f);
            DepthColormap(t, out rgba[i * 4], out rgba[i * 4 + 1], out rgba[i * 4 + 2]);
            rgba[i * 4 + 3] = 255;
        }

        var tex = _device.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { w, h },
            Format = "rgba8unorm",
            Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
        });
        _queue.WriteTexture(
            new GPUTexelCopyTextureInfo { Texture = tex },
            rgba,
            new GPUTexelCopyBufferLayout { Offset = 0, BytesPerRow = (uint)(w * 4), RowsPerImage = (uint)h },
            new uint[] { (uint)w, (uint)h }
        );

        _depthMapTex = tex;
        _depthMapView = tex.CreateView();
        _depthMapW = w;
        _depthMapH = h;
    }

    /// <summary>Plasma-like colormap: t=0 (min depth) → dark purple, t=1 (max depth) → bright yellow.</summary>
    private static void DepthColormap(float t, out byte r, out byte g, out byte b)
    {
        // 5-stop plasma colormap (dark purple → blue-purple → magenta → orange → yellow)
        ReadOnlySpan<float> kr = stackalloc float[] { 0.05f, 0.46f, 0.80f, 0.97f, 0.94f };
        ReadOnlySpan<float> kg = stackalloc float[] { 0.03f, 0.07f, 0.14f, 0.51f, 0.98f };
        ReadOnlySpan<float> kb = stackalloc float[] { 0.53f, 0.67f, 0.37f, 0.09f, 0.13f };
        float seg = t * 4f;
        int lo = Math.Clamp((int)seg, 0, 3);
        float s = seg - lo;
        r = (byte)((kr[lo] + s * (kr[lo + 1] - kr[lo])) * 255f);
        g = (byte)((kg[lo] + s * (kg[lo + 1] - kg[lo])) * 255f);
        b = (byte)((kb[lo] + s * (kb[lo + 1] - kb[lo])) * 255f);
    }
}
