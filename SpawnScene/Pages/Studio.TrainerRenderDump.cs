using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// For a view whose photograph is a video frame (in memory, not at a URL), put the frame on a hidden canvas the
    /// harness saves as <c>&lt;view&gt;-photo.png</c>, so score_views / compose_views can compare against it.
    /// </summary>
    async Task StashVideoPhotoAsync(string key, string imageName)
    {
        if (!imageName.StartsWith(VideoFrameStore.Prefix, StringComparison.Ordinal)
            || !VideoFrameStore.TryGet(imageName, out var jpeg)) return;
        try
        {
            using var blob = new Blob(new byte[][] { jpeg }, new BlobOptions { Type = "image/jpeg" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
            _js.Set("__phBitmap", bitmap);
            await _js.CallVoidAsync("eval", @"
                (function(){
                  var b = window.__phBitmap;
                  var c = document.getElementById('photodump');
                  if (!c) { c = document.createElement('canvas'); c.id='photodump';
                            c.style.position='fixed'; c.style.left='-99999px';
                            document.body.appendChild(c); }
                  c.width = b.width; c.height = b.height;
                  c.getContext('2d').drawImage(b, 0, 0);
                  window.__phBitmap = null;
                })();");
            Console.WriteLine($"[Dataset] PHOTO-DUMP {key}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] photo dump for {key} failed: {ex.Message}");
        }
    }

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
