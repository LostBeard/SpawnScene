using SpawnDev.ILGPU;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Dump one depth map to a plain canvas so it can be looked at, and compared like for like.
///
/// Comparing depth models from screenshots does not work: the crops do not align, JPEG smears
/// the colour map, and inverting a colour map back to values quantises it into visible bands.
/// This renders from the RAW float depth of a chosen image at a chosen patch budget, with the
/// same turbo mapping the app uses, so two runs differ only in the thing being tested.
///
/// Entry: <c>/studio?autotest=depthmap&amp;img=samples/living-room-hd-2.jpg&amp;patches=37</c>
/// Logs:  <c>[DepthMap] READY WxH ...</c>, then the harness reads canvas #depthdump.
/// </summary>
public partial class Studio
{
    private async Task RunDepthMapAutotestAsync(string imageUrl, int patchesPerSide, bool asDisparity)
    {
        Console.WriteLine(
            $"[DepthMap] starting img={imageUrl} patches={patchesPerSide}x{patchesPerSide} " +
            $"disparity={asDisparity}");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();
            DepthEstimationService.SetSquareInput(patchesPerSide);

            // -- Decode --
            byte[] bytes = await _http.GetByteArrayAsync(imageUrl);
            string mime = imageUrl.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                ? "image/png" : "image/jpeg";
            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = mime });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);

            // Cap the decode the same way import does - the point is the model's detail, not
            // how many pixels we upsample it onto.
            int srcW = (int)bitmap.Width, srcH = (int)bitmap.Height;
            int cap = ImageImportService.MaxImportDimension;
            int w = srcW, h = srcH;
            if (Math.Max(srcW, srcH) > cap)
            {
                float s = (float)cap / Math.Max(srcW, srcH);
                w = Math.Max(1, (int)MathF.Round(srcW * s));
                h = Math.Max(1, (int)MathF.Round(srcH * s));
            }
            using var osc = new OffscreenCanvas(w, h);
            using var octx = osc.Get2DContext();
            octx.DrawImage(bitmap, 0, 0, w, h);
            using var srcData = octx.GetImageData(0, 0, w, h);
            using var srcBytes = srcData.Data;
            var rgba = srcBytes.ReadBytes();
            Console.WriteLine($"[DepthMap] {srcW}x{srcH} decoded to {w}x{h}");

            // -- Depth --
            if (!_depthService.IsReady)
                await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);

            var img = new ImportedImage { FileName = imageUrl, Width = w, Height = h, RgbaPixels = rgba };
            var depth = await _depthService.EstimateDepthAsync(img);
            if (depth?.RawDepthGpu == null)
            {
                Console.WriteLine("[DepthMap] FAIL: no depth returned");
                return;
            }

            // CPU transfer: a diagnostic dump, once, of one image.
            int dw = depth.Width, dh = depth.Height;
            float[] raw = await depth.RawDepthGpu.CopyToHostAsync<float>(0, (long)dw * dh);
            Console.WriteLine(
                $"[DepthMap] depth {dw}x{dh} range [{depth.MinDepth:F4}, {depth.MaxDepth:F4}]");

            // -- Colourise on the CPU with the app's own turbo mapping --
            var px = new byte[dw * dh * 4];
            float lo = depth.MinDepth, hi = depth.MaxDepth;
            float range = MathF.Max(hi - lo, 1e-6f);
            for (int i = 0; i < dw * dh; i++)
            {
                float v = raw[i];
                float t;
                if (asDisparity)
                {
                    // Show DEPTH as DISPARITY, which is what DAv2 emits natively. Same data,
                    // different value distribution - this is the comparison that tells whether
                    // "DAv2 looks more detailed" is about the data or about the normalisation.
                    float z = MathF.Max(v, 1e-6f);
                    float dispLo = 1f / MathF.Max(hi, 1e-6f);
                    float dispHi = 1f / MathF.Max(lo, 1e-6f);
                    t = (1f / z - dispLo) / MathF.Max(dispHi - dispLo, 1e-9f);
                }
                else
                {
                    t = (v - lo) / range;
                }
                t = 1f - Math.Clamp(t, 0f, 1f);   // matches GpuDepthColorizer

                float r = 0.13572138f + t * (4.61539260f + t * (-42.66032258f + t * (132.13108234f + t * (-152.94239396f + t * 59.28637943f))));
                float g = 0.09140261f + t * (2.19418839f + t * (4.84296658f + t * (-14.18503333f + t * (4.27729857f + t * 2.82956604f))));
                float b = 0.10667330f + t * (12.64194608f + t * (-60.58204836f + t * (110.36276771f + t * (-89.90310912f + t * 27.34824973f))));

                px[i * 4 + 0] = (byte)(Math.Clamp(r, 0f, 1f) * 255f);
                px[i * 4 + 1] = (byte)(Math.Clamp(g, 0f, 1f) * 255f);
                px[i * 4 + 2] = (byte)(Math.Clamp(b, 0f, 1f) * 255f);
                px[i * 4 + 3] = 255;
            }

            // -- Put it on a plain canvas the harness can read as a PNG --
            _js.Set("__depthW", dw);
            _js.Set("__depthH", dh);
            _js.Set("__depthPx", px);
            await _js.CallVoidAsync("eval", @"
                (function(){
                  var c = document.getElementById('depthdump');
                  if (!c) { c = document.createElement('canvas'); c.id='depthdump';
                            c.style.position='fixed'; c.style.left='-99999px';
                            document.body.appendChild(c); }
                  c.width = window.__depthW; c.height = window.__depthH;
                  var ctx = c.getContext('2d');
                  var img = ctx.createImageData(c.width, c.height);
                  img.data.set(new Uint8ClampedArray(window.__depthPx));
                  ctx.putImageData(img, 0, 0);
                })();");

            Console.WriteLine($"[DepthMap] READY {dw}x{dh}");
            await Task.Delay(1500);
            Console.WriteLine("[DepthMap] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DepthMap] FAIL: {ex}");
        }
    }
}
