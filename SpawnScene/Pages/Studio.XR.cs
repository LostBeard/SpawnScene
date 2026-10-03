using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// WebXR VR/AR session management
public partial class Studio
{
    private async Task EnterXRAsync(string mode)
    {
        try
        {
            Console.WriteLine($"[Studio] Entering {mode}...");
            ReleasePointerLock();

            _xrService.OnXRFrame += OnXRFrame;
            _xrService.OnSessionEnded += OnXRSessionEnded;
            _xrSceneFromRoom = null; // placed on the session's first frame (XRSceneAlignment)
            _xrLocomotion.Reset();
            _xrStickLog = true;
            _xrLastTime = -1;

            // Pause canvas RAF loop — XR has its own render loop
            _xrActive = true;

            await _xrService.EnterSessionAsync(mode);

            // Initialize WebGL blit helper for WebGL XR fallback
            if (_xrService.IsWebGLFallback && _xrService.GLContext != null)
            {
                _xrBlit = new WebGLXRBlit();
                _xrBlit.Initialize(_xrService.GLContext);
            }

            Console.WriteLine($"[Studio] {mode} session active (WebGL fallback: {_xrService.IsWebGLFallback})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Studio] Failed to enter {mode}: {ex.Message}");
            _xrActive = false;
            _xrBlit?.Dispose();
            _xrBlit = null;
            _xrService.OnXRFrame -= OnXRFrame;
            _xrService.OnSessionEnded -= OnXRSessionEnded;
            // Resume canvas rendering
            RequestFrame();
        }
    }

    // XR frame cost, logged every XRStatsFrames frames: what the per-eye render + copy costs on this device.
    const int XRStatsFrames = 90;
    readonly System.Diagnostics.Stopwatch _xrFrameClock = new();
    double _xrFrameMsSum;
    int _xrFrameCount;
    long _xrStatsStart;

    private void OnXRFrame(XRFrameData frameData)
    {
        _xrFrameClock.Restart();
        if (_xrFrameCount == 0) _xrStatsStart = System.Diagnostics.Stopwatch.GetTimestamp();
        try { RenderXRFrame(frameData); }
        finally
        {
            _xrFrameMsSum += _xrFrameClock.Elapsed.TotalMilliseconds;
            if (++_xrFrameCount == XRStatsFrames)
            {
                double wall = System.Diagnostics.Stopwatch.GetElapsedTime(_xrStatsStart).TotalMilliseconds;
                var vp = frameData.Views.Length > 0 ? frameData.Views[0].Viewport : null;
                Console.WriteLine($"[XR] {XRStatsFrames} frames: {_xrFrameMsSum / XRStatsFrames:F1} ms submit per frame " +
                    $"({frameData.Views.Length} views of {vp?.Width}x{vp?.Height}), {1000.0 * XRStatsFrames / wall:F0} fps, " +
                    $"{(frameData.IsWebGLFallback ? "WebGL copy path" : "WebGPU layer")}, head at {_xrHeadInScene:F2} in the scene");
                _xrFrameCount = 0; _xrFrameMsSum = 0;
            }
        }
    }

    // Room (local-floor) -> scene: the head starts where the desktop camera was, facing its way (XRSceneAlignment).
    System.Numerics.Matrix4x4? _xrSceneFromRoom;
    // Thumbstick move / snap-turn / rise, applied to _xrSceneFromRoom each frame.
    readonly XRLocomotion _xrLocomotion = new();
    double _xrLastTime = -1;
    System.Numerics.Vector3 _xrHeadInScene;
    bool _xrStickLog; // log the first controller input of a session (shows the sticks are read)

    private void RenderXRFrame(XRFrameData frameData)
    {
        if (_xrSceneFromRoom == null)
        {
            var cam = _sceneManager.Camera;
            _xrSceneFromRoom = XRSceneAlignment.SceneFromRoom(frameData.HeadPosition, frameData.HeadOrientation, cam.Position, cam.Forward);
            Console.WriteLine($"[XR] room placed in the scene: head {frameData.HeadPosition} -> camera {cam.Position}, facing {cam.Forward}");
        }
        // dt clamped: a stalled frame must not throw the viewer across the scene.
        float dt = _xrLastTime < 0 ? 0f : (float)Math.Clamp((frameData.Time - _xrLastTime) / 1000.0, 0.0, 0.1);
        _xrLastTime = frameData.Time;
        _xrSceneFromRoom = _xrLocomotion.Step(_xrSceneFromRoom.Value, frameData.HeadPosition, frameData.HeadOrientation,
            frameData.LeftStick, frameData.RightStick, dt, _cameraController?.MoveSpeed ?? 1f);
        if (_xrStickLog && (frameData.LeftStick != default || frameData.RightStick != default))
        {
            _xrStickLog = false;
            Console.WriteLine($"[XR] controller input: left {frameData.LeftStick}, right {frameData.RightStick}");
        }
        _xrHeadInScene = System.Numerics.Vector3.Transform(frameData.HeadPosition, _xrSceneFromRoom.Value);
        foreach (var v in frameData.Views) v.ViewMatrix = XRSceneAlignment.SceneView(v.ViewMatrix, _xrSceneFromRoom.Value);
        if (frameData.IsWebGLFallback && _gpuRenderer.XRSorted)
        {
            // One sort per frame from the head (scene space), a frustum wide enough for both eyes, then each eye.
            var m = _xrSceneFromRoom.Value;
            var head = new CameraParams
            {
                Width = 1000, Height = 1000, FocalX = 182, FocalY = 182, CenterX = 500, CenterY = 500, // ~140 deg: both eyes
                Near = _sceneManager.Camera.Near, Far = _sceneManager.Camera.Far,
                Position = System.Numerics.Vector3.Transform(frameData.HeadPosition, m),
                Forward = System.Numerics.Vector3.Normalize(System.Numerics.Vector3.TransformNormal(
                    System.Numerics.Vector3.Transform(-System.Numerics.Vector3.UnitZ, frameData.HeadOrientation), m)),
                Up = System.Numerics.Vector3.Normalize(System.Numerics.Vector3.TransformNormal(
                    System.Numerics.Vector3.Transform(System.Numerics.Vector3.UnitY, frameData.HeadOrientation), m)),
            };
            var cullProj = CameraParams.CreateWebGpuProjection(head.FocalX, head.FocalY, head.CenterX, head.CenterY,
                head.Width, head.Height, head.Near, head.Far);
            _gpuRenderer.BeginXRFrameSorted(head, head.ViewMatrix * cullProj);
            var layer = _xrService.WebGLLayer!;
            foreach (var view in frameData.Views)
            {
                var vp = view.Viewport;
                _gpuRenderer.RenderXRViewSortedToCanvas(view.ViewMatrix, view.ProjectionMatrix, (int)vp.Width, (int)vp.Height, _xrCasEnabled);
                _xrBlit!.Blit(_gpuRenderer.XRBridgeCanvas!, layer.Framebuffer, vp);
            }
            return;
        }
        if (frameData.IsWebGLFallback)
        {
            // WebGL fallback: render each eye to WebGPU bridge canvas, blit to XR framebuffer
            var xrLayer = _xrService.WebGLLayer!;
            foreach (var view in frameData.Views)
            {
                var vp = view.Viewport;
                _gpuRenderer.RenderXRViewToCanvas(
                    view.ViewMatrix, view.ProjectionMatrix,
                    (int)vp.Width, (int)vp.Height, _xrCasEnabled);
                _xrBlit!.Blit(
                    _gpuRenderer.XRBridgeCanvas!,
                    xrLayer.Framebuffer, vp);
            }
        }
        else
        {
            // WebGPU native XR (future — once browsers support XRGPUBinding)
            foreach (var view in frameData.Views)
            {
                _gpuRenderer.RenderXRView(
                    view.ViewMatrix, view.ProjectionMatrix,
                    view.ColorTexture!, view.DepthStencilTexture,
                    (int)view.Viewport.X, (int)view.Viewport.Y,
                    (int)view.Viewport.Width, (int)view.Viewport.Height);
            }
        }
    }

    private void OnXRSessionEnded()
    {
        _xrService.OnXRFrame -= OnXRFrame;
        _xrService.OnSessionEnded -= OnXRSessionEnded;

        // Clean up XR resources
        _xrBlit?.Dispose();
        _xrBlit = null;
        _gpuRenderer.DisposeXRBridge();

        // Resume canvas RAF loop
        _xrActive = false;
        Console.WriteLine("[Studio] XR session ended, resuming canvas rendering");
        RequestFrame();
    }
}
