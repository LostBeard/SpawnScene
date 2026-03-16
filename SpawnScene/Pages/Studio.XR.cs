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

    private void OnXRFrame(XRFrameData frameData)
    {
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
