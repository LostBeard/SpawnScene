using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using SpawnDev.ILGPU;
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
    public struct DepthMaskParams
    {
        public float Px, Py, Pz, Fx, Fy, Fz, Split;
        public int KeepNear;
        // Mode 1: split by the reference's EWA clamp region instead of depth; KeepNear = keep the ON-axis half.
        public int Mode;
        public float Rx, Ry, Rz, Ux, Uy, Uz, LimX, LimY;
    }

    /// <summary>
    /// Diagnostic subset mask, IN PLACE on the shared splat buffer so the trainer dump and the viewer both see it:
    /// copy every row to <paramref name="backup"/>, then zero the opacity of splats on the other side of
    /// <c>Split</c> (camera-space depth) from the kept half.
    /// </summary>
    static void DepthMaskKernel(Index1D i, ArrayView<float> packed, ArrayView<float> backup, DepthMaskParams p)
    {
        long o = (long)i.X * SplatFormat.Floats;
        for (int k = 0; k < SplatFormat.Floats; k++) backup[o + k] = packed[o + k];
        float dx = packed[o] - p.Px, dy = packed[o + 1] - p.Py, dz = packed[o + 2] - p.Pz;
        float tz = dx * p.Fx + dy * p.Fy + dz * p.Fz;
        bool near = tz < p.Split;
        if (p.Mode == 1)
        {
            float tx = dx * p.Rx + dy * p.Ry + dz * p.Rz;
            float ty = dx * p.Ux + dy * p.Uy + dz * p.Uz;
            near = !(tz > 0.2f && (XMath.Abs(tx / tz) > p.LimX || XMath.Abs(ty / tz) > p.LimY));   // "near" = on-axis
        }
        if (near != (p.KeepNear != 0)) packed[o + 9] = 0f;
    }

    static void RestoreKernel(Index1D i, ArrayView<float> backup, ArrayView<float> packed)
    {
        long o = (long)i.X * SplatFormat.Floats;
        for (int k = 0; k < SplatFormat.Floats; k++) packed[o + k] = backup[o + k];
    }

    /// <summary>
    /// Diagnostic (&amp;capturesubsets=1): at the capture pose, render the NEAR half and then the FAR half of the scene
    /// (split at the depth of <paramref name="splitPoint"/>) through BOTH renderers - trainer dump + viewer capture
    /// as view-&lt;kind&gt;near-&lt;i&gt; / view-&lt;kind&gt;far-&lt;i&gt; - so a trainer/viewer difference can be pinned to a subset
    /// of one scene instead of compared across scenes.
    /// </summary>
    async Task CaptureDepthSubsetsAsync(string kind, int index, CameraParams cam, string imageName, int quarterTurns,
        System.Numerics.Vector3 splitPoint)
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (packed == null || n <= 0) return;
        var accel = _gpuService.WebGPUAccelerator;
        float split = System.Numerics.Vector3.Dot(splitPoint - cam.Position, cam.Forward);
        using var backup = accel.Allocate1D<float>((long)n * SplatFormat.Floats);
        var mask = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, DepthMaskParams>(DepthMaskKernel);
        var restore = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(RestoreKernel);
        var right = System.Numerics.Vector3.Normalize(System.Numerics.Vector3.Cross(cam.Forward, cam.Up));
        var up = System.Numerics.Vector3.Cross(right, cam.Forward);
        foreach (var (tag, keepNear, mode) in new[] { ("near", 1, 0), ("far", 0, 0), ("onaxis", 1, 1), ("offaxis", 0, 1) })
        {
            mask((Index1D)n, packed.View, backup.View, new DepthMaskParams
            {
                Px = cam.Position.X, Py = cam.Position.Y, Pz = cam.Position.Z,
                Fx = cam.Forward.X, Fy = cam.Forward.Y, Fz = cam.Forward.Z, Split = split, KeepNear = keepNear,
                Mode = mode, Rx = right.X, Ry = right.Y, Rz = right.Z, Ux = up.X, Uy = up.Y, Uz = up.Z,
                LimX = 1.3f * cam.Width / (2f * cam.FocalX), LimY = 1.3f * cam.Height / (2f * cam.FocalY),
            });
            accel.FlushPendingCommands();
            _gpuRenderer.RepackForDisplay(cam.Position);
            string key = $"{kind}{tag}-{index}";
            await StashTrainerRenderAsync($"view-{key}", cam);
            await ParkOnGroundTruthPoseAsync(key, cam);
            Console.WriteLine($"[Dataset] subset {key}: split at depth {split:F2}");
            Console.WriteLine($"[Dataset] READY-FOR-CAPTURE view-{key} {imageName} {cam.Width}x{cam.Height} turns={quarterTurns}");
            await Task.Delay(1800);
            restore((Index1D)n, backup.View, packed.View);
            accel.FlushPendingCommands();
            _gpuRenderer.RepackForDisplay(cam.Position);
        }
        await accel.SynchronizeAsync();
    }

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
