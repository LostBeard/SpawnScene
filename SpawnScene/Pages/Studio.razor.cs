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
using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnDev.GameUI.Input;
using SpawnDev.GameUI.Rendering;

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
    [Inject] private GameUIService _gameUI { get; set; } = default!;
    [Inject] private SpawnJSRuntime _js { get; set; } = default!;
    [Inject] private VideoFrameExtractor _videoExtractor { get; set; } = default!;
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

    // GameUI root (screen-space overlay). Rebuilt per StudioState.
    private UIElement _uiRoot = new();
    private readonly HashSet<string> _prevKeysDown = new();

    // App state
    private enum StudioState { ProjectBrowser, ProjectDetail, SceneViewer, Testing }

    private StudioState _state = StudioState.ProjectBrowser;
    private List<Project>? _projects;
    private Project? _activeProject;
    private string? _statusMessage;
    private bool _pipelineBusy;
    private int _uiTrainIters = 500;
    private bool _uiTrainGeom = true;
    private string _uiDataset = "Bathroom";
    // Testing pose path: false = DAv3 cascade (product default). true = COLMAP/poses.par when
    // the dataset ships them. Bathroom has none - GT is DrJohnson/Truck/TempleRing only.
    private bool _uiUseGtPoses;
    private bool _uiInitFromCloud;

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
    // Project training (TrainProjectSceneAsync): progress line + Stop button in the viewer HUD.
    private bool _trainingActive;
    private string _trainHudText = "";
    private UILabel? _hudTrainLabel;
    private UIScrollView? _projectDetailScroll; // the project page body (autotest scrolls it to capture the settings)
    private UILabel? _statusLabel;
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

        // Initialize SpawnDev.GameUI (WebGPU overlay, SDF fonts, unified input)
        _gameUI.Init(_device, _queue, _canvasFormat, _canvasRef, _canvasWidth, _canvasHeight);
        UITheme.Current = UITheme.Dark;

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

        // &resize=native|letterbox - A/B the depth preprocessing on one build.
        if (query.TryGetValue("resize", out var resize))
        {
            DepthEstimationService.ResizeMode = resize.ToLowerInvariant() switch
            {
                "native" or "nativeaspect" => SpawnDev.ILGPU.ML.Pipelines.DepthResizeMode.NativeAspect,
                "letterbox" => SpawnDev.ILGPU.ML.Pipelines.DepthResizeMode.Letterbox,
                _ => throw new ArgumentException($"resize={resize}: expected native or letterbox"),
            };
        }
        Console.WriteLine($"[Autotest] depth resize mode {DepthEstimationService.ResizeMode}");

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
            query.TryGetValue("dataset", out var poseDataset);
            int[]? poseViews = query.TryGetValue("views", out var pv)
                ? pv.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray()
                : null;
            await RunDav3PoseGateAsync(poseN, posePatches, poseDataset ?? "TempleRing", poseViews);
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
        else if (mode == "project")
        {
            // The USER path: a project from a dataset's photos, Generate (train + save), reopen. See Studio.ProjectAutotest.cs.
            string pname = query.TryGetValue("name", out var pn2) ? pn2 : "TruckFull";
            int piters = query.TryGetValue("train", out var pt2) && int.TryParse(pt2, out var pti2) ? pti2 : 1000;
            int pcount = query.TryGetValue("count", out var pc2) && int.TryParse(pc2, out var pci2) ? pci2 : 24;
            int pstride = query.TryGetValue("stride", out var ps2) && int.TryParse(ps2, out var psi2) ? psi2 : 1;
            int pmax = query.TryGetValue("maxsplats", out var pm2) && int.TryParse(pm2, out var pmi2) ? pmi2 : 3_000_000;
            int pres = query.TryGetValue("trainres", out var pr2) && int.TryParse(pr2, out var pri2) ? pri2 : 1024;
            bool ppage = query.TryGetValue("pageonly", out var pp2) && pp2 is "1" or "true";
            // &importmax=N: longest edge photos are decoded at (0 = full size - the red check for the managed-heap
            // blow-up TJ hit on gh-pages with 35 Bathroom photos; default 1024).
            if (query.TryGetValue("importmax", out var imq) && int.TryParse(imq, out var imv))
                ImageImportService.MaxImportDimension = Math.Max(0, imv);
            await RunProjectAutotestAsync(pname, piters, pcount, pstride, pmax, pres, ppage);
        }
        else if (mode == "dataset")
        {
            // The unposed path: a real capture with no calibration file.
            string name = query.TryGetValue("name", out var dn) ? dn : "Bathroom";
            int iters = query.TryGetValue("train", out var dt) && int.TryParse(dt, out var dti) ? dti : 0;
            bool dgeom = query.TryGetValue("geom", out var dg) && dg is "1" or "true";
            int maxDim = query.TryGetValue("maxdim", out var md) && int.TryParse(md, out var mdi) ? mdi : 720;
            // ?poses=dav3 keeps depth and cameras in one frame (product default after
            // DrJohnson 2026-09-23). ?poses=sfm / auto remain for A/B.
            string poses = query.TryGetValue("poses", out var pp) ? pp : "dav3";
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
            // &ba=0 skips bundle adjustment; &baiters=N sets LM iterations per round.
            if (query.TryGetValue("ba", out var bav))
                _multiViewService.BundleAdjust = bav is not ("0" or "false");
            if (query.TryGetValue("baiters", out var bai) && int.TryParse(bai, out var baii))
                _multiViewService.BundleAdjustIterations = baii;
            // &gtintrinsics=1 (dataset runs, diagnosis): BA holds the COLMAP intrinsics fixed.
            if (query.TryGetValue("gtintrinsics", out var gtiq))
                DiagnoseWithGroundTruthIntrinsics = gtiq is "1" or "true";
            // &bagtinit=1 (dataset runs, diagnosis): BA starts from the COLMAP poses.
            // 1 = full GT poses, 2 = GT rotations only, 3 = GT positions only.
            if (query.TryGetValue("bagtinit", out var bgtq))
                DiagnoseFromGroundTruthPoses = bgtq is "true" ? 1 : int.TryParse(bgtq, out var bgti) ? bgti : 0;
            // &globalinit=0: bundle adjustment starts from the depth cascade instead of the global SfM solution (the default).
            if (query.TryGetValue("globalfocal", out var gsf) && double.TryParse(gsf, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var gsfd))
                _multiViewService.GlobalSfmFocalOverride = gsfd;
            if (query.TryGetValue("globalgtrot", out var ggr))
                DiagnoseGlobalWithGroundTruthRotations = ggr is "1" or "true";
            // &denseinit=0: with the learned front end, no dense FAST/BRIEF triangulation for the initial cloud (A/B).
            if (query.TryGetValue("denseinit", out var dinq))
                _multiViewService.DenseInitFromBriefFeatures = dinq is not ("0" or "false");
            if (query.TryGetValue("densepx", out var dpq) && double.TryParse(dpq, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var dpv))
                _multiViewService.DenseReprojectionPixels = dpv;
            if (query.TryGetValue("denseangle", out var dangq) && double.TryParse(dangq, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var dangv))
                _multiViewService.DenseMinParallaxDeg = dangv;
            // &core=0: keep every placed camera instead of the strongly connected core; &coremin=N its link strength (shared points).
            if (query.TryGetValue("core", out var coq))
                _multiViewService.KeepStrongCore = coq is not ("0" or "false");
            if (query.TryGetValue("coremin", out var cmq) && int.TryParse(cmq, out var cmi))
                _multiViewService.CoreSharedPoints = Math.Max(1, cmi);
            // &relpose=f: relative poses decomposed from the verification's F instead of a calibrated five-point E-RANSAC (A/B).
            if (query.TryGetValue("relpose", out var rpq))
                _multiViewService.CalibratedRelativePoses = rpq is not ("f" or "F" or "0");
            // &relposefilter=0: tracks from every verified pair, not only the rotation-consistent ones (A/B).
            if (query.TryGetValue("relposefilter", out var rpf))
                _multiViewService.FilterPairsByRotation = rpf is not ("0" or "false");
            if (query.TryGetValue("globalinit", out var gsi))
                _multiViewService.UseGlobalSfmInit = gsi is "1" or "true";
            // &pairangle=N: BA pair candidates within N degrees by cascade pose (default 45; 180 = every pair with 15+ matches).
            if (query.TryGetValue("pairangle", out var pa) && float.TryParse(pa, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var paf))
                _multiViewService.MaxPairAngleDeg = paf;
            if (query.TryGetValue("bacg", out var bacg) && int.TryParse(bacg, out var bacgi))
                _multiViewService.BundleAdjustCgIterations = bacgi;
            if (query.TryGetValue("barel", out var barel) && double.TryParse(barel, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var bareld))
                GpuBundleAdjuster.RelativeStopOverride = bareld;
            // &gpugp=0: the global init's managed positioning instead of GpuGlobalPositioner (A/B); &gptrace=1 logs its attempts.
            if (query.TryGetValue("gpugp", out var ggq))
                _multiViewService.UseGpuGlobalPositioning = ggq is not ("0" or "false");
            if (query.TryGetValue("gptrace", out var gtq) && gtq is "1" or "true")
                GpuGlobalPositioner.Trace = m => Console.WriteLine($"[GlobalPos] {m}");
            // &gpuba=0: the managed bundle adjuster instead of GpuBundleAdjuster (A/B).
            if (query.TryGetValue("gpuba", out var gbq))
                _multiViewService.UseGpuBundleAdjust = gbq is not ("0" or "false");
            // &bacapture=0: GPU BA without WebGPU dispatch capture/replay of its CG batches (A/B).
            if (query.TryGetValue("bacapture", out var bcq))
                GpuBundleAdjuster.UseDispatchCapture = bcq is not ("0" or "false");
            // &batrace=1: log every GPU BA attempt that is not accepted, and the device state behind it.
            if (query.TryGetValue("batrace", out var btq))
                GpuBundleAdjuster.TraceAttempts = btq is "1" or "true";
            // &tracecarry=1: log each densify-carry operation before it runs.
            if (query.TryGetValue("tracecarry", out var tcq))
                SplatTrainerGpu.TraceCarrySteps = tcq is "1" or "true";
            // &pooltrace=1: the ML pool's misses by rent name, per [GPU] chunk report. &pooltrace=2 adds the pool's
            // ownership trace, which logs EVERY violation as it happens - on DAv3 ~500 ALIEN-RETURN lines per pass
            // (40,709 by chunk 80 in b23), so it is opt-in separately.
            if (query.TryGetValue("pooltrace", out var ptq) && ptq is "1" or "2" or "true")
            {
                SpawnDev.ILGPU.ML.Tensors.BufferPool.TraceFreshAllocNames = true;
                SpawnDev.ILGPU.ML.Tensors.BufferPool.TracePoolOwnership = ptq == "2";
            }
            // &bufsites=1: record who allocates every WebGPU buffer; the [GPU] reports then list the top creation sites.
            if (query.TryGetValue("bufsites", out var bsq))
            {
                SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.CaptureCreationSites = bsq is "1" or "true";
                SpawnDev.ILGPU.WebGPU.WebGPUBufferAccounting.CreationSiteDepth = 5;   // pool frames hide the real caller at 3
            }
            // &unloaddepth=1: unload the depth model before training (A/B for the densify-carry device loss).
            if (query.TryGetValue("unloaddepth", out var udq))
                _unloadDepthBeforeTraining = udq is "1" or "true";
            // Features: RaCo-ALIKED + LightGlue+ (LearnedFeatureMatcher) by default; &features=brief for FAST/BRIEF features and pair
            // matches; &lgk=1024|3072 its keypoint budget, &lgpairs=N pairs per matcher run.
            if (query.TryGetValue("features", out var ftq))
                _importService.UseLearnedFeatures = ftq is not ("brief" or "fast");
            if (query.TryGetValue("lgk", out var lgk) && int.TryParse(lgk, out var lgki))
                LearnedFeatureMatcher.KeypointBudget = lgki;
            if (query.TryGetValue("lgretrieval", out var lgr) && int.TryParse(lgr, out var lgri))
                _importService.LearnedAllPairsUpTo = Math.Max(0, lgri);
            if (query.TryGetValue("lgtopk", out var lgt) && int.TryParse(lgt, out var lgti))
                _importService.LearnedRetrievalTopK = Math.Max(1, lgti);
            if (query.TryGetValue("lgprofile", out var lgpr))
                LearnedFeatureMatcher.ProfileSecondRun = lgpr is "1" or "true";
            if (query.TryGetValue("lgpairs", out var lgp) && int.TryParse(lgp, out var lgpi))
                LearnedFeatureMatcher.PairsPerRun = Math.Max(1, lgpi);
            // &verify=cpu: verify matched pairs with the CPU estimator instead of GpuEpipolarRansac (A/B).
            if (query.TryGetValue("verify", out var vq))
                _multiViewService.GpuPairVerification = vq != "cpu";
            // &bainit=0: per-view depth-shell init even when BA produced a sparse cloud (A/B).
            // &targetmb=N: the resident training-target budget (MiB). 256 shrank Truck's 126 views to 734 px.
            // &trainprofile=1: wait after each training-step phase and log the per-phase ms (diagnostic).
            // &maxradpx=N: densify prunes splats whose 3-sigma screen radius exceeded N px (post-reset). Off by default.
            if (query.TryGetValue("maxradpx", out var mrq) && float.TryParse(mrq,
                    System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var mrv) && mrv > 0)
                SplatDensityControl.MaxScreenRadiusPx = mrv;
            // &shdeg=N: cap the viewer's SH degree after training (the trainer-render dump uses the same).
            if (query.TryGetValue("shdeg", out var shq) && int.TryParse(shq, out var shv))
                ViewerShDegreeCap = Math.Max(0, shv);
            // &cas=N: the viewer's CAS sharpening strength 0..1 (0 = none; default 0.5).
            if (query.TryGetValue("cas", out var casq) && float.TryParse(casq,
                    System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var casv))
                _gpuRenderer.SharpeningStrength = casv;
            // &lodpx=N: the viewer's screen-space LOD cull threshold in pixels (0 = draw every splat).
            if (query.TryGetValue("lodpx", out var lpx) && float.TryParse(lpx,
                    System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var lpxv))
                _gpuRenderer.LodCullPixels = Math.Max(0f, lpxv);
            if (query.TryGetValue("trainprofile", out var tpv))
                ProfileTrainPhases = tpv is "1" or "true";
            if (query.TryGetValue("llffhold", out var lhq) && int.TryParse(lhq, out var lhv)) LlffHold = Math.Max(0, lhv);
            if (query.TryGetValue("targetmb", out var tmb) && int.TryParse(tmb, out var tmbi) && tmbi > 0)
                MaxTargetStackBytes = (long)tmbi * 1024 * 1024;
            if (query.TryGetValue("badump", out var bdv))
                _multiViewService.DumpFailedResections = bdv is "1" or "true";
            if (query.TryGetValue("bainit", out var bin))
                _multiViewService.InitFromBundlePoints = bin is not ("0" or "false");
            // ?n=N views per joint forward. The default is a starting point, not a measured
            // limit; the first pass proves whatever is asked for on this device and backs off.
            if (query.TryGetValue("n", out var nn) && int.TryParse(nn, out var nni))
                DepthEstimationService.MaxMultiViewImages = nni;
            // ?outside=1 keeps splats the screening reference cannot see (default ON after
            // DrJohnson 2026-09-23: outside=false zeroed non-ref views). ?outside=0 restores the
            // object-centric screen. ?relthresh=N is how closely depths must agree.
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
            // ?densify=N runs adaptive density control every N ITERATIONS (reference: 100).
            if (query.TryGetValue("densify", out var dnq) && int.TryParse(dnq, out var dni))
                DensifyEveryIters = dni;
            // ?densifygrad=X is the densify bar on the reference quantity: mean over views of
            // |dL/dmean2D| in NDC units (default 2e-4, Kerbl's densify_grad_threshold).
            if (query.TryGetValue("densifygrad", out var dgq) && float.TryParse(dgq,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var dgf))
                SplatDensityControl.GradientThreshold = dgf;
            // ?densifyuntil=N stops densify/opacity-reset after iteration N (Kerbl: 15000).
            if (query.TryGetValue("densifyuntil", out var duq) && int.TryParse(duq, out var dui))
                DensifyUntilIter = dui;
            // ?densifyfrac=X keeps only the top fraction of above-threshold candidates (Brush: 0.2).
            if (query.TryGetValue("densifyfrac", out var dfq) && float.TryParse(dfq,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var dff))
                SplatDensityControl.GrowthSelectFraction = dff;
            // ?densifynoop=1 runs the whole densify APPLY path (host round trip, Resize, Adam/SH
            // remap, renderer re-upload) on an EMPTY plan every period. Bisects "the apply path
            // damages the model" from "the clones/splits/prunes do".
            if (query.TryGetValue("densifynoop", out var dnoq) && dnoq == "1")
                DensifyNoOp = true;
            // ?maxdensify=N caps splat count during densify (default 450k; growhost OOM'd ~780k).
            if (query.TryGetValue("maxdensify", out var mdnq) && int.TryParse(mdnq, out var mdni))
                MaxDensifiedSplats = mdni;
            // ?opacityreset=N caps every opacity every N ITERATIONS (reference: 3000), which
            // also unlocks the size prunes - without it nothing removes a bloated splat.
            if (query.TryGetValue("opacityreset", out var orq) && int.TryParse(orq, out var ori))
                OpacityResetEveryIters = ori;
            // ?poslrdecay=N is how far the position rate falls over PositionLrMaxSteps; 1 = off.
            if (query.TryGetValue("poslrdecay", out var pdq) && float.TryParse(pdq, out var pdf))
                PositionLrDecay = pdf;
            if (query.TryGetValue("capturetwice", out var ctq)) CaptureTwice = ctq == "1";
            if (query.TryGetValue("capturesubsets", out var csq)) CaptureSubsets = csq == "1";
            // Training D-SSIM is per RGB channel (the reference); ?ssimrgb=0 computes it on luma.
            if (query.TryGetValue("ssimrgb", out var srq)) SplatTrainerGpu.SsimPerChannel = !(srq == "0" || srq == "false");
            // Geometry Adam steps splats with no gradient too (torch Adam, the reference); ?denseadam=0 skips them.
            if (query.TryGetValue("denseadam", out var daq)) SplatTrainerGpu.DenseGeometryAdam = !(daq == "0" || daq == "false");
            // The densify gradient averages over every step a splat projected on screen (the reference's radii > 0);
            // ?densifydenom=contrib averages only over the steps it received a gradient.
            if (query.TryGetValue("densifydenom", out var ddq)) SplatTrainerGpu.DensifyDenominatorFrustum = ddq != "contrib";
            // Each epoch's views train in a fresh random order (the reference); ?shuffleviews=0 = file order; ?shuffleseed=N.
            if (query.TryGetValue("shuffleviews", out var svq)) ShuffleViews = !(svq == "0" || svq == "false");
            if (query.TryGetValue("shuffleseed", out var ssq) && int.TryParse(ssq, out var ssi)) ShuffleViewsSeed = ssi;
            // ?fitone=N trains against view N alone - a ceiling on what the rasteriser and its
            // gradients can express, independent of supervision or scheduling.
            if (query.TryGetValue("fitone", out var foq) && int.TryParse(foq, out var foi))
                FitSingleViewIndex = foi;
            // ?pruneunseen=0 keeps splats no supervised view constrained (default: drop them
            // after the first cycle - measured as ~half a depth-shell scene).
            if (query.TryGetValue("pruneunseen", out var pu))
                PruneUnconstrainedAfterCycle = pu is not ("0" or "false");
            // ?skipzerograd=1 stops colour/opacity Adam stepping splats with no gradient, the
            // guard adam_geometry has always had for position. Off by default: its effect on the
            // held-out curve is the measurement.
            if (query.TryGetValue("skipzerograd", out var sz))
                SkipZeroGradientSteps = sz is "1" or "true";
            // ?gtposes=1 uses the dataset's own COLMAP poses and skips pose recovery, so the
            // optimiser can be measured without the pose error folded in.
            bool gtPoses = query.TryGetValue("gtposes", out var gp) && gp is "1" or "true";
            // ?init=points initialises from the dataset's sparse SfM cloud instead of
            // unprojecting a depth map per view - the 3DGS reference initialisation.
            bool cloudInit = query.TryGetValue("init", out var ini) && ini is "points" or "cloud";
            await RunDatasetAutotestAsync(name, iters, dgeom, maxDim, poses, patches, gtPoses, cloudInit);
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

        _gameUI.Dispose();
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
