using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
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
    // The dataset autotest loads unposed captures through the ordinary import path.
    [Inject] private ImageImportService _importService { get; set; } = default!;
    [Inject] private SpawnJSRuntime _js { get; set; } = default!;
    [Inject] private GpuDepthColorizer _depthColorizer { get; set; } = default!;
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

        using var canvas = _canvasRef.As<HTMLCanvasElement>();
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

        // Optional automated gate: /studio?autotest=generate-room
        await RunAutotestIfRequestedAsync();
    }

    /// <summary>
    /// Browser automation entry: create a project, load the Room sample, generate a scene.
    /// Triggered by query <c>?autotest=generate-room</c>. Logs <c>[Autotest] PASS</c> / <c>FAIL</c>.
    /// </summary>
    private async Task RunAutotestIfRequestedAsync()
    {
        var uri = new Uri(_nav.Uri);
        // Manual query parse — avoid Microsoft.AspNetCore.WebUtilities package dep in WASM.
        var query = uri.Query.TrimStart('?').Split('&', StringSplitOptions.RemoveEmptyEntries)
            .Select(p => p.Split('=', 2))
            .Where(p => p.Length == 2)
            .ToDictionary(p => Uri.UnescapeDataString(p[0]), p => Uri.UnescapeDataString(p[1]),
                StringComparer.OrdinalIgnoreCase);
        if (!query.TryGetValue("autotest", out var mode))
            return;

        if (mode == "trainer-gate")
        {
            await RunTrainerGateAsync();
            return;
        }

        if (mode == "novel-view")
        {
            query.TryGetValue("view", out var viewName);
            // ?onlyview=N restricts the dense unproject to a single view (diagnostic).
            int onlyView = query.TryGetValue("onlyview", out var ov) && int.TryParse(ov, out var ovi) ? ovi : -1;
            bool globalScale = query.TryGetValue("globalscale", out var gs) && gs is "1" or "true";
            // ?train=N runs N photometric optimisation steps before the first pose is parked.
            int trainIters = query.TryGetValue("train", out var tr) && int.TryParse(tr, out var tri) ? tri : 0;
            // ?upright=1 stands each source photograph up before depth inference.
            bool upright = query.TryGetValue("upright", out var ur) && ur is "1" or "true";
            // ?geom=1 also optimises position, scale and rotation, not just colour and opacity.
            bool geom = query.TryGetValue("geom", out var gm) && gm is "1" or "true";
            await RunNovelViewAutotestAsync(
                viewName ?? "templeR0016", onlyView, globalScale, trainIters, upright, geom);
        }
        else if (mode == "dataset")
        {
            // The unposed path: a real capture with no calibration file.
            string name = query.TryGetValue("name", out var dn) ? dn : "Bathroom";
            int iters = query.TryGetValue("train", out var dt) && int.TryParse(dt, out var dti) ? dti : 0;
            bool dgeom = query.TryGetValue("geom", out var dg) && dg is "1" or "true";
            int maxDim = query.TryGetValue("maxdim", out var md) && int.TryParse(md, out var mdi) ? mdi : 720;
            // ?poses=dav3 keeps depth and cameras in one frame by skipping SfM.
            string poses = query.TryGetValue("poses", out var pp) ? pp : "auto";
            await RunDatasetAutotestAsync(name, iters, dgeom, maxDim, poses);
            return;
        }

        if (mode != "generate-room")
            return;

        Console.WriteLine($"[Autotest] starting mode={mode}");
        try
        {
            var project = await _projectService.CreateProjectAsync($"Autotest {DateTime.UtcNow:HHmmss}");
            _projects = await _projectService.ListProjectsAsync();
            _activeProject = project;
            OnOpenProject(project);

            await LoadSampleImage("Room", "samples/room.png");
            if (_activeProject.Sources.Count == 0)
                throw new InvalidOperationException("Room sample did not load");

            // Drive the same path as the Generate Scene button (single-image).
            var generateDone = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            void OnDone()
            {
                if (_state == StudioState.SceneViewer && _sceneManager.HasScene)
                    generateDone.TrySetResult();
            }
            _sceneManager.OnSceneChanged += OnDone;
            try
            {
                OnGenerateSceneClicked();
                // Poll for success or surfaced error status (OnGenerateSceneClicked is async void).
                var pollDeadline = DateTime.UtcNow.AddMinutes(10);
                while (DateTime.UtcNow < pollDeadline)
                {
                    if (generateDone.Task.IsCompleted) break;
                    if (_statusMessage != null
                        && (_statusMessage.StartsWith("Error", StringComparison.OrdinalIgnoreCase)
                            || _statusMessage.Contains('❌')))
                        throw new InvalidOperationException(_statusMessage);
                    await Task.Delay(250);
                }
                if (!generateDone.Task.IsCompleted)
                    throw new TimeoutException("Generate did not finish within 10 minutes");
            }
            finally
            {
                _sceneManager.OnSceneChanged -= OnDone;
            }

            Console.WriteLine($"[Autotest] PASS — scene with {_sceneManager.ActiveScene?.Count ?? 0} splats");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Autotest] FAIL: {ex}");
        }
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
    /// Colorize depth on the GPU and upload as a UI texture.
    /// Uses MinDepth/MaxDepth already computed on GPU — no full-map .NET readback/colorize loop.
    /// </summary>
    private async Task CaptureDepthMapAsync(DepthResult depthResult)
    {
        if (_device == null || _queue == null || depthResult.RawDepthGpu == null) return;

        _depthMapView?.Dispose(); _depthMapView = null;
        _depthMapTex?.Destroy(); _depthMapTex?.Dispose(); _depthMapTex = null;

        int w = depthResult.Width;
        int h = depthResult.Height;

        var result = await _depthColorizer.ColorizeToTextureAsync(depthResult, _device, _queue, w, h);
        if (result == null) return;

        _depthMapTex = result.Value.tex;
        _depthMapView = result.Value.view;
        _depthMapW = w;
        _depthMapH = h;
    }

    /// <summary>Plasma-like colormap kept for any remaining CPU viz callers.</summary>
    private static void DepthColormap(float t, out byte r, out byte g, out byte b)
    {
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
