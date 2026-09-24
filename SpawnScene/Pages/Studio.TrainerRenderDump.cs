using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// Render <paramref name="cam"/> with the TRAINER's rasteriser on the scene the viewer is showing, and put it
    /// on a hidden canvas the dataset harness saves as <c>&lt;view&gt;-trainer.png</c>, next to the viewer's
    /// capture of the same pose. The trainer's per-view PSNR and the viewer's capture of that view disagreed by
    /// 0.7-3.6 dB on Truck 7K; with both pictures the difference can be looked at instead of guessed at.
    /// </summary>
    async Task StashTrainerRenderAsync(string key, CameraParams cam)
    {
        try
        {
            var packed = _gpuRenderer.PackedSplatBuffer;
            int n = _gpuRenderer.SplatCount;
            if (_trainer == null || packed == null || n <= 0) return;
            var (w, h) = _trainer.Size;
            var c = cam.ScaledTo(w, h);
            var box = await SplatBounds.ComputeAsync(_gpuService.WebGPUAccelerator, packed, n);
            if (box == null) return;
            var (near, far) = SplatBounds.DepthRangeFor(box.Value, c);
            // CPU transfer: one diagnostic frame per captured view. The viewer's SH degree, so a capped
            // A/B (&shdeg=) compares like with like.
            int trainerDegree = _trainer.ActiveShDegree;
            _trainer.ActiveShDegree = Math.Min(trainerDegree, _gpuRenderer.ShDegree);
            float[] rgb;
            try { rgb = await _trainer.RenderForwardAsync(packed, n, c, near, far); }
            finally { _trainer.ActiveShDegree = trainerDegree; }
            var px = new byte[w * h * 4];
            for (int i = 0; i < w * h; i++)
            {
                px[i * 4 + 0] = (byte)MathF.Round(Math.Clamp(rgb[i * 3 + 0], 0f, 1f) * 255f);
                px[i * 4 + 1] = (byte)MathF.Round(Math.Clamp(rgb[i * 3 + 1], 0f, 1f) * 255f);
                px[i * 4 + 2] = (byte)MathF.Round(Math.Clamp(rgb[i * 3 + 2], 0f, 1f) * 255f);
                px[i * 4 + 3] = 255;
            }
            _js.Set("__trW", w);
            _js.Set("__trH", h);
            _js.Set("__trPx", px);
            await _js.CallVoidAsync("eval", @"
                (function(){
                  var c = document.getElementById('trainerdump');
                  if (!c) { c = document.createElement('canvas'); c.id='trainerdump';
                            c.style.position='fixed'; c.style.left='-99999px';
                            document.body.appendChild(c); }
                  c.width = window.__trW; c.height = window.__trH;
                  var ctx = c.getContext('2d');
                  var img = ctx.createImageData(c.width, c.height);
                  img.data.set(new Uint8ClampedArray(window.__trPx));
                  ctx.putImageData(img, 0, 0);
                })();");
            Console.WriteLine($"[Dataset] TRAINER-RENDER {key} {w}x{h}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] trainer render for {key} failed: {ex.Message}");
        }
    }
}
