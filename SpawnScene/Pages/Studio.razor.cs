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

        if (mode == "dav3-pose")
        {
            // How good ARE the recovered cameras, measured against poses we know?
            int poseN = query.TryGetValue("n", out var pn) && int.TryParse(pn, out var pni) ? pni : 6;
            int posePatches = query.TryGetValue("patches", out var pq) && int.TryParse(pq, out var pqi)
                ? pqi : DepthEstimationService.SafeMultiViewPatches;
            await RunDav3PoseGateAsync(poseN, posePatches);
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
            // ?patches=N binds a square NxN ViT patch grid for depth. 37 = the familiar 518.
            int nvPatches = query.TryGetValue("patches", out var np) && int.TryParse(np, out var npi)
                ? npi : DepthEstimationService.SafeMultiViewPatches;
            await RunNovelViewAutotestAsync(
                viewName ?? "templeR0016", onlyView, globalScale, trainIters, upright, geom, nvPatches);
        }
        else if (mode == "depthmap")
        {
            // Dump one depth map for visual comparison at a chosen patch budget.
            string img = query.TryGetValue("img", out var im) ? im : "samples/living-room-hd-2.jpg";
            int dmPatches = query.TryGetValue("patches", out var dp) && int.TryParse(dp, out var dpi)
                ? dpi : 37;
            bool disp = query.TryGetValue("disparity", out var ds) && ds is "1" or "true";
            await RunDepthMapAutotestAsync(img, dmPatches, disp);
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
            // ?patches=N binds a square NxN ViT patch grid. 37 = the familiar 518. Same
            // meaning as the novel-view harness; it used to mean a TOTAL here, which made
            // PATCHES=64 bind a 7x9 grid.
            int patches = query.TryGetValue("patches", out var pb) && int.TryParse(pb, out var pbi)
                ? pbi : DepthEstimationService.SafeMultiViewPatches;
            // ?anchors=N views shared by every chunked pass. Three is the minimum that determines
            // a similarity, but a minimal sample always fits itself, so four is the fewest that
            // can DETECT a bad anchor and five the fewest that can identify which.
            if (query.TryGetValue("anchors", out var an) && int.TryParse(an, out var ani))
                _multiViewService.ChunkAnchorCount = ani;
            // ?n=N views per joint forward. The default is a starting point, not a measured
            // limit; the first pass proves whatever is asked for on this device and backs off.
            if (query.TryGetValue("n", out var nn) && int.TryParse(nn, out var nni))
                DepthEstimationService.MaxMultiViewImages = nni;
            // ?outside=1 keeps splats the screening reference cannot see - the other walls of a
            // room. ?relthresh=N is how closely depths must agree to survive the screen.
            if (query.TryGetValue("outside", out var ov2))
                _multiViewService.KeepOutsideReferenceView = ov2 is "1" or "true";
            if (query.TryGetValue("relthresh", out var rt) && float.TryParse(rt, out var rtf))
                _multiViewService.ConsistencyRelThreshold = rtf;
            // ?budget=N splats for the initialisation. More views at a FIXED budget is sparser
            // coverage per view, not richer coverage - 34 views at 1.5M chose subsample 5 and
            // gave each view ~31k splats where a 6-view run gave ~40k each.
            if (query.TryGetValue("budget", out var bg) && int.TryParse(bg, out var bgi))
                _multiViewService.SplatBudget = bgi;
            // ?maxscale=N caps a splat at N * scene diagonal; ?poslr=N scales the position rate.
            if (query.TryGetValue("maxscale", out var ms) && float.TryParse(ms, out var msf))
                MaxScaleFraction = msf;
            if (query.TryGetValue("poslr", out var pl) && float.TryParse(pl, out var plf))
                PositionLrScale = plf;
            // ?heldevery=N evaluates held-out PSNR every N cycles (0 = only at the ends).
            if (query.TryGetValue("heldevery", out var he) && int.TryParse(he, out var hei))
                HeldOutEveryCycles = hei;
            await RunDatasetAutotestAsync(name, iters, dgeom, maxDim, poses, patches);
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
