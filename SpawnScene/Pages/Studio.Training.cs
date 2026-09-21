using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Photometric optimisation of the live scene against its posed source photographs.
///
/// Everything up to this point is an INITIALISATION: monocular depth, unprojected and fused into
/// a coloured point cloud. Measurement showed that merge is destructive - a single view renders
/// its own photo at 27.7 dB, four merged views render the same photo at 19.2 dB - and the
/// literature is unanimous that depth fusion is only ever a starting point. This is the step that
/// was missing.
///
/// Only colour and opacity move here; position, scale and rotation are frozen. That is the
/// cheapest gradient that can exist (dL/dcolour = alpha*T, dL/dopacity = G*dL/dalpha, no
/// covariance chain) and it answers the question that has to be answered first: does a gradient
/// reach the parameters of a real scene at all, and does the picture get better when it does.
/// </summary>
public partial class Studio
{
    SplatTrainerGpu? _trainer;
    bool _trainerInitialized;

    /// <summary>
    /// Fit the active scene's colour and opacity to its training photographs.
    /// Logs <c>[Train] ...</c> throughout; never throws (a failed optimisation must still leave
    /// a renderable scene behind for the measurement that follows).
    /// </summary>
    private async Task TrainOnTrainingViewsAsync(int iterations, int keysPerSplat = 8)
    {
        try
        {
            var scene = _sceneManager.ActiveScene;
            if (scene == null || scene.TrainingViews.Count == 0)
            {
                Console.WriteLine("[Train] FAIL: no training views on the active scene");
                return;
            }

            var packed = _gpuRenderer.PackedSplatBuffer;
            int n = _gpuRenderer.SplatCount;
            if (packed == null || n <= 0)
            {
                Console.WriteLine("[Train] FAIL: no packed splat buffer uploaded");
                return;
            }

            var accel = _gpuService.WebGPUAccelerator;

            // -- Scene extent, from the splats themselves --
            // The sort key packs depth into 18 bits, so it needs a real range to quantise
            // against. Deriving it from the scene means a room works the same way an object does.
            var aabb = await SplatBounds.ComputeAsync(accel, packed, n);
            if (aabb == null)
            {
                Console.WriteLine("[Train] FAIL: scene has no splats with opacity");
                return;
            }
            var box = aabb.Value;
            Console.WriteLine(
                $"[Train] scene aabb ({box.MinX:F3},{box.MinY:F3},{box.MinZ:F3})-" +
                $"({box.MaxX:F3},{box.MaxY:F3},{box.MaxZ:F3}) diag={box.Diagonal:F3}");

            // -- Viewport: the training photographs' own resolution --
            var views = scene.TrainingViews;
            int w = views[0].Camera.Width, h = views[0].Camera.Height;
            foreach (var v in views)
            {
                if (v.Camera.Width != w || v.Camera.Height != h)
                {
                    Console.WriteLine(
                        $"[Train] FAIL: mixed resolutions ({w}x{h} vs " +
                        $"{v.Camera.Width}x{v.Camera.Height}); one trainer viewport cannot serve both");
                    return;
                }
            }

            _trainer ??= new SplatTrainerGpu(_gpuService);
            if (!_trainerInitialized) { _trainer.Initialize(); _trainerInitialized = true; }
            _trainer.Resize(w, h, n, keysPerSplat);

            // -- Upload every target photograph once --
            // Per-iteration upload would be 3.7 MB of traffic per step and would dominate the
            // measurement. One scratch frame is reused on the host so the WASM heap never holds
            // more than a single image.
            int frameFloats = w * h * 3;
            using var targets = accel.Allocate1D<float>((long)views.Count * frameFloats);
            var scratch = new float[frameFloats];

            var loadStart = DateTime.UtcNow;
            for (int i = 0; i < views.Count; i++)
            {
                bool ok = await LoadTargetAsync(views[i].ImageName, views[i].QuarterTurns, w, h, scratch);
                if (!ok)
                {
                    Console.WriteLine($"[Train] FAIL: could not load target {views[i].ImageName}");
                    return;
                }
                targets.View.SubView((long)i * frameFloats, frameFloats).CopyFromCPU(scratch);
            }
            await accel.SynchronizeAsync();
            Console.WriteLine(
                $"[Train] {views.Count} targets at {w}x{h} uploaded in " +
                $"{(DateTime.UtcNow - loadStart).TotalSeconds:F1}s");

            // -- Baseline: how well does the untrained scene already explain each photo? --
            var (baseInit, baseHeld) = await EvaluateAsync(_trainer, packed, n, views, targets, box);

            _trainer.InitOptimizerState(packed, n);

            // -- Optimise --
            var start = DateTime.UtcNow;
            int overflowed = 0;
            // Loss is per VIEW, and views differ in how much of the frame the temple covers, so
            // comparing iteration 0 against iteration N compares two different pictures. Average
            // over a full cycle of views instead - that is the only comparable quantity here.
            double cycleSum = 0;
            int cycleN = 0;
            double firstCycle = double.NaN, lastCycle = double.NaN;
            for (int it = 0; it < iterations; it++)
            {
                int vi = it % views.Count;
                var cam = views[vi].Camera;
                var (near, far) = SplatBounds.DepthRangeFor(box, cam);

                _trainer.SetTargetFrom(targets, vi);
                float loss = await _trainer.TrainStepAsync(packed, n, cam, near, far);
                if (_trainer.LastOverflowed) overflowed++;

                cycleSum += loss;
                cycleN++;
                if (vi == views.Count - 1)
                {
                    double mean = cycleSum / cycleN;
                    if (double.IsNaN(firstCycle)) firstCycle = mean;
                    lastCycle = mean;
                    int cycle = (it + 1) / views.Count;
                    if (cycle % 5 == 1 || cycle == 1)
                    {
                        double secs = (DateTime.UtcNow - start).TotalSeconds;
                        Console.WriteLine(
                            $"[Train] cycle {cycle,4} mean loss {mean:F6} " +
                            $"({secs:F1}s, {(it + 1) / Math.Max(secs, 1e-6):F1} it/s)");
                    }
                    cycleSum = 0; cycleN = 0;
                }

                // Yield to the browser. A tight await loop still starves rAF on some drivers,
                // and a starved page is a lost device.
                if (it % 4 == 3) await Task.Delay(1);
            }
            double total = (DateTime.UtcNow - start).TotalSeconds;
            Console.WriteLine(
                $"[Train] {iterations} iters in {total:F1}s ({iterations / Math.Max(total, 1e-6):F1} it/s), " +
                $"mean loss/cycle {firstCycle:F6} -> {lastCycle:F6}");
            if (overflowed > 0)
                Console.WriteLine(
                    $"[Train] WARNING: {overflowed}/{iterations} iterations overflowed the key " +
                    $"buffer (keysPerSplat={keysPerSplat}) - those gradients are incomplete");

            var (fitInit, fitHeld) = await EvaluateAsync(_trainer, packed, n, views, targets, box);
            Console.WriteLine($"[Train] trainer PSNR init-views {baseInit:F2} -> {fitInit:F2} dB");
            Console.WriteLine($"[Train] trainer PSNR held-out   {baseHeld:F2} -> {fitHeld:F2} dB");

            // The display renderer reads a packed vertex buffer built at upload time; training
            // wrote straight through to the splat data behind it.
            _gpuRenderer.RepackForDisplay();
            Console.WriteLine("[Train] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Train] FAIL: {ex}");
        }
    }

    /// <summary>
    /// Mean PSNR of the trainer's own forward render against the targets, split by whether the
    /// view seeded the geometry. Held-out is the number that matters; init-views is the control -
    /// if those do not improve, no gradient is reaching the parameters at all.
    /// </summary>
    static async Task<(float Init, float Held)> EvaluateAsync(
        SplatTrainerGpu trainer,
        MemoryBuffer1D<float, Stride1D.Dense> packed, int n,
        IReadOnlyList<TrainingView> views,
        MemoryBuffer1D<float, Stride1D.Dense> targets,
        SplatBounds.Aabb box)
    {
        var (w, h) = trainer.Size;
        int frameFloats = w * h * 3;
        double sumInit = 0, sumHeld = 0;
        int nInit = 0, nHeld = 0;

        for (int i = 0; i < views.Count; i++)
        {
            var cam = views[i].Camera;
            var (near, far) = SplatBounds.DepthRangeFor(box, cam);
            float[] rendered = await trainer.RenderForwardAsync(packed, n, cam, near, far);
            // CPU transfer: evaluation only, a handful of times per run.
            float[] target = await targets.CopyToHostAsync<float>((long)i * frameFloats, frameFloats);

            double se = 0;
            for (int k = 0; k < frameFloats; k++)
            {
                double d = rendered[k] - target[k];
                se += d * d;
            }
            double mse = se / frameFloats;
            double psnr = mse <= 1e-12 ? 99.0 : 10.0 * Math.Log10(1.0 / mse);

            if (views[i].UsedForInit) { sumInit += psnr; nInit++; }
            else { sumHeld += psnr; nHeld++; }
        }
        return (
            nInit > 0 ? (float)(sumInit / nInit) : 0f,
            nHeld > 0 ? (float)(sumHeld / nHeld) : 0f);
    }

    /// <summary>
    /// Decode one training photograph into <paramref name="dest"/> as RGB in [0,1].
    /// Matches how splat colour was created at import (byte/255), so the loss compares
    /// like with like.
    /// </summary>
    private async Task<bool> LoadTargetAsync(string url, int quarterTurns, int w, int h, float[] dest)
    {
        try
        {
            byte[] bytes = await _http.GetByteArrayAsync(url);
            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/png" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);

            // The turn was applied to the camera at generation time; the picture has to get the
            // same one or the loss compares a render of one framing against pixels from another.
            int srcW = (quarterTurns % 2 == 0) ? w : h;
            int srcH = (quarterTurns % 2 == 0) ? h : w;
            if ((int)bitmap.Width != srcW || (int)bitmap.Height != srcH)
            {
                Console.WriteLine(
                    $"[Train] {url} is {bitmap.Width}x{bitmap.Height}, expected {srcW}x{srcH} " +
                    $"for a {w}x{h} trainer with {quarterTurns} quarter turn(s)");
                return false;
            }
            using var osc = new OffscreenCanvas(srcW, srcH);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0);
            using var imageData = ctx.GetImageData(0, 0, srcW, srcH);
            using var dataArray = imageData.Data;
            var rgba = ImageOrientation.RotateRgba(dataArray.ReadBytes(), srcW, srcH, quarterTurns);

            const float inv = 1f / 255f;
            for (int p = 0; p < w * h; p++)
            {
                dest[p * 3 + 0] = rgba[p * 4 + 0] * inv;
                dest[p * 3 + 1] = rgba[p * 4 + 1] * inv;
                dest[p * 3 + 2] = rgba[p * 4 + 2] * inv;
            }
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Train] target load failed for {url}: {ex.Message}");
            return false;
        }
    }
}
