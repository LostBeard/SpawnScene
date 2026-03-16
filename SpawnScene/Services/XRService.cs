using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Manages WebXR sessions for VR and AR rendering.
/// Detects capabilities, creates sessions, and provides per-frame XR pose/view data.
/// Tries WebGPU XR (XRGPUBinding) first, falls back to WebGL XR (XRWebGLLayer).
/// </summary>
public class XRService : IDisposable
{
    private readonly BlazorJSRuntime _js;
    private readonly GpuService _gpu;

    // XR state
    private XRSession? _session;
    private XRReferenceSpace? _refSpace;
    private ActionCallback<double, XRFrame?>? _xrRafCallback;

    // WebGPU XR path
    private XRGPUBinding? _gpuBinding;
    private XRProjectionLayer? _projectionLayer;

    // WebGL XR fallback path
    private HTMLCanvasElement? _glCanvas;
    private WebGL2RenderingContext? _glContext;
    private XRWebGLLayer? _webglLayer;

    // Capability flags (cached after first check)
    private bool? _vrSupported;
    private bool? _arSupported;

    public bool IsSessionActive => _session != null;
    public XRSession? Session => _session;
    public string? SessionMode { get; private set; }

    /// <summary>True when using WebGL XR fallback (WebGPU XR binding unavailable).</summary>
    public bool IsWebGLFallback { get; private set; }

    /// <summary>The WebGL context used for XR (only valid when IsWebGLFallback).</summary>
    public WebGL2RenderingContext? GLContext => _glContext;

    /// <summary>The XR WebGL layer (only valid when IsWebGLFallback).</summary>
    public XRWebGLLayer? WebGLLayer => _webglLayer;

    /// <summary>Fired each XR frame with pose and view data.</summary>
    public event Action<XRFrameData>? OnXRFrame;

    /// <summary>Fired when the XR session ends.</summary>
    public event Action? OnSessionEnded;

    public XRService(BlazorJSRuntime js, GpuService gpu)
    {
        _js = js;
        _gpu = gpu;
    }

    /// <summary>Check if immersive-vr is supported.</summary>
    public async Task<bool> IsVRSupportedAsync()
    {
        if (_vrSupported.HasValue) return _vrSupported.Value;
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var xr = navigator.XR;
            if (xr == null) { _vrSupported = false; return false; }
            _vrSupported = await xr.IsSessionSupported("immersive-vr");
        }
        catch { _vrSupported = false; }
        return _vrSupported.Value;
    }

    /// <summary>Check if immersive-ar is supported.</summary>
    public async Task<bool> IsARSupportedAsync()
    {
        if (_arSupported.HasValue) return _arSupported.Value;
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            using var xr = navigator.XR;
            if (xr == null) { _arSupported = false; return false; }
            _arSupported = await xr.IsSessionSupported("immersive-ar");
        }
        catch { _arSupported = false; }
        return _arSupported.Value;
    }

    /// <summary>Enter an immersive VR or AR session.</summary>
    public async Task EnterSessionAsync(string mode = "immersive-vr")
    {
        if (_session != null) return;

        using var navigator = _js.Get<Navigator>("navigator");
        using var xr = navigator.XR;
        if (xr == null) throw new InvalidOperationException("WebXR not available");

        _session = await xr.RequestSession(mode, new XRSessionInit
        {
            RequiredFeatures = new[] { "local-floor" },
            OptionalFeatures = mode == "immersive-ar" ? new[] { "hit-test" } : null,
        });

        SessionMode = mode;
        _session.OnEnd += OnSessionEnd;

        // Get reference space
        _refSpace = await _session.RequestReferenceSpace("local-floor");

        // Try WebGPU XR first, fall back to WebGL XR
        var webGpuAccel = _gpu.WebGPUAccelerator;
        var device = webGpuAccel.NativeAccelerator.NativeDevice!;

        try
        {
            _gpuBinding = new XRGPUBinding(_session, device);
            _projectionLayer = _gpuBinding.CreateProjectionLayer();

            _session.UpdateRenderState(new XRRenderStateInit
            {
                Layers = new XRLayer[] { _projectionLayer },
            });

            IsWebGLFallback = false;
            Console.WriteLine($"[XRService] WebGPU XR session started ({mode})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[XRService] WebGPU XR binding failed: {ex.Message}, using WebGL fallback");

            // Clean up failed WebGPU binding attempt
            _projectionLayer?.Dispose();
            _projectionLayer = null;
            _gpuBinding?.Dispose();
            _gpuBinding = null;

            // WebGL XR fallback — use HTMLCanvasElement (XRWebGLLayer requires it in most browsers)
            using var doc = _js.Get<Document>("document");
            _glCanvas = doc.CreateElement<HTMLCanvasElement>("canvas");
            _glContext = _glCanvas.GetContext<WebGL2RenderingContext>("webgl2", new WebGLContextAttributes { XrCompatible = true });
            if (_glContext == null)
                throw new InvalidOperationException("Failed to create WebGL2 xr-compatible context");
            _webglLayer = new XRWebGLLayer(_session, _glContext);

            _session.UpdateRenderState(new XRRenderStateInit
            {
                BaseLayer = _webglLayer,
            });

            IsWebGLFallback = true;
            Console.WriteLine($"[XRService] WebGL XR fallback session started ({mode})");
        }

        // Start XR render loop
        _xrRafCallback = new ActionCallback<double, XRFrame?>(OnXRAnimationFrame);
        _session.RequestAnimationFrame(_xrRafCallback);
        Console.WriteLine("[XRService] XR render loop started");
    }

    /// <summary>Exit the current XR session.</summary>
    public async Task ExitSessionAsync()
    {
        if (_session != null)
        {
            try { await _session.End(); } catch { }
        }
        CleanupSession();
    }

    private void OnSessionEnd(XRSessionEvent e)
    {
        Console.WriteLine("[XRService] XR session ended");
        CleanupSession();
        OnSessionEnded?.Invoke();
    }

    private void CleanupSession()
    {
        _xrRafCallback?.Dispose();
        _xrRafCallback = null;

        // WebGPU XR resources
        _projectionLayer?.Dispose();
        _projectionLayer = null;
        _gpuBinding?.Dispose();
        _gpuBinding = null;

        // WebGL XR resources
        _webglLayer?.Dispose();
        _webglLayer = null;
        _glContext?.Dispose();
        _glContext = null;
        _glCanvas?.Dispose();
        _glCanvas = null;

        _refSpace?.Dispose();
        _refSpace = null;
        if (_session != null)
        {
            _session.OnEnd -= OnSessionEnd;
            _session.Dispose();
            _session = null;
        }
        SessionMode = null;
        IsWebGLFallback = false;
        _xrFrameCount = 0;
    }

    private int _xrFrameCount;

    private void OnXRAnimationFrame(double time, XRFrame? frame)
    {
        if (_session == null || _refSpace == null || frame == null) return;

        // Request next frame first (ensures continuous loop)
        _session.RequestAnimationFrame(_xrRafCallback!);

        _xrFrameCount++;
        if (_xrFrameCount <= 3)
            Console.WriteLine($"[XRService] XR frame #{_xrFrameCount} (t={time:F1}, fallback={IsWebGLFallback})");

        try
        {
            using var pose = frame.GetViewerPose(_refSpace);
            if (pose == null) return;

            var views = pose.Views;
            var frameData = new XRFrameData
            {
                Frame = frame,
                Pose = pose,
                Views = new XRViewData[views.Length],
                IsWebGLFallback = IsWebGLFallback,
            };

            if (IsWebGLFallback)
            {
                // WebGL path: viewports come from XRWebGLLayer, no GPU textures
                for (int i = 0; i < views.Length; i++)
                {
                    var view = views[i];
                    using var projArr = view.ProjectionMatrix;
                    var projFloats = (float[])projArr;
                    frameData.Views[i] = new XRViewData
                    {
                        Eye = view.Eye,
                        ProjectionMatrix = JsFloatArrayToMatrix(projFloats),
                        ViewMatrix = RigidTransformToViewMatrix(view.Transform),
                        Viewport = _webglLayer!.GetViewport(view),
                    };
                }
            }
            else
            {
                // WebGPU path: textures from XRGPUBinding
                for (int i = 0; i < views.Length; i++)
                {
                    var view = views[i];
                    using var subImage = _gpuBinding!.GetViewSubImage(_projectionLayer!, view);

                    using var projArr = view.ProjectionMatrix;
                    var projFloats = (float[])projArr;
                    frameData.Views[i] = new XRViewData
                    {
                        Eye = view.Eye,
                        ProjectionMatrix = JsFloatArrayToMatrix(projFloats),
                        ViewMatrix = RigidTransformToViewMatrix(view.Transform),
                        ColorTexture = subImage.ColorTexture,
                        DepthStencilTexture = subImage.DepthStencilTexture,
                        Viewport = subImage.Viewport,
                    };
                }
            }

            OnXRFrame?.Invoke(frameData);

            // Dispose view references
            foreach (var v in views) v.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[XRService] XR frame error: {ex.Message}");
        }
    }

    /// <summary>Convert a JS Float32Array (16 elements, column-major) to System.Numerics.Matrix4x4.</summary>
    private static Matrix4x4 JsFloatArrayToMatrix(float[] m)
    {
        // WebXR matrices are column-major; System.Numerics is row-major
        return new Matrix4x4(
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15]
        );
    }

    /// <summary>Convert XRRigidTransform to a view matrix.</summary>
    private static Matrix4x4 RigidTransformToViewMatrix(XRRigidTransform transform)
    {
        var pos = transform.Position;
        var ori = transform.Orientation;

        var position = new Vector3((float)pos.X, (float)pos.Y, (float)pos.Z);
        var rotation = new Quaternion((float)ori.X, (float)ori.Y, (float)ori.Z, (float)ori.W);

        // View matrix = inverse of the pose (world → camera)
        var rotMatrix = Matrix4x4.CreateFromQuaternion(Quaternion.Conjugate(rotation));
        var transMatrix = Matrix4x4.CreateTranslation(-position);
        return transMatrix * rotMatrix;
    }

    public void Dispose()
    {
        CleanupSession();
    }
}

/// <summary>Per-frame XR data passed to the render callback.</summary>
public class XRFrameData
{
    public XRFrame Frame { get; set; } = null!;
    public XRViewerPose Pose { get; set; } = null!;
    public XRViewData[] Views { get; set; } = System.Array.Empty<XRViewData>();
    public bool IsWebGLFallback { get; set; }
}

/// <summary>Per-eye view data for XR rendering.</summary>
public class XRViewData
{
    public string Eye { get; set; } = "none";
    public Matrix4x4 ProjectionMatrix { get; set; }
    public Matrix4x4 ViewMatrix { get; set; }
    /// <summary>GPU texture for the eye (WebGPU XR path only, null for WebGL fallback).</summary>
    public GPUTexture? ColorTexture { get; set; }
    public GPUTexture? DepthStencilTexture { get; set; }
    public XRViewport Viewport { get; set; } = null!;
}
