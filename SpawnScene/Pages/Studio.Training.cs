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
    private async Task TrainOnTrainingViewsAsync(
        int iterations, int keysPerSplat = 8, bool optimiseGeometry = false)
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

            // -- Geometric learning rates --
            // Position is quoted in the reference relative to the scene extent, and the extent
            // it means is the camera rig, not the object: a rate in world units is meaningless
            // without it. Scale and rotation are scale-free parameterisations (log and unit
            // quaternion), so their published rates are absolute.
            SplatTrainerGpu.GeometryStep? geo = null;
            if (optimiseGeometry)
            {
                var centroid = Vector3.Zero;
                foreach (var v in views) centroid += v.Camera.Position;
                centroid /= views.Count;
                float rigRadius = 0f;
                foreach (var v in views)
                    rigRadius = MathF.Max(rigRadius, Vector3.Distance(v.Camera.Position, centroid));
                if (rigRadius <= 0f) rigRadius = MathF.Max(box.Diagonal, 1e-3f);

                geo = new SplatTrainerGpu.GeometryStep(
                    PositionLr: 1.6e-4f * rigRadius,
                    LogScaleLr: 0.005f,
                    RotationLr: 0.001f,
                    // Without density control nothing prunes, so a splat that stops being
                    // constrained has to be bounded instead of pruned. The upper bound is the
                    // reference's own prune threshold; the lower one just keeps it positive.
                    MinScale: 1e-6f,
                    MaxScale: 0.1f * MathF.Max(box.Diagonal, 1e-3f));

                Console.WriteLine(
                    $"[Train] geometry ON: rig radius {rigRadius:F3}, " +
                    $"posLr {geo.Value.PositionLr:G3}, scaleLr {geo.Value.LogScaleLr:G3}, " +
                    $"rotLr {geo.Value.RotationLr:G3}, scale in " +
                    $"[{geo.Value.MinScale:G3}, {geo.Value.MaxScale:G3}]");
            }

            // Only supervised views drive the loss. Held-out ones are loaded and evaluated so
            // the run reports a number that means something, and never fitted.
            var supervised = new List<int>();
            for (int i = 0; i < views.Count; i++)
                if (views[i].UsedForSupervision) supervised.Add(i);
            if (supervised.Count == 0)
            {
                Console.WriteLine("[Train] FAIL: every view is held out - nothing to fit to");
                return;
            }
            Console.WriteLine(
                $"[Train] supervising on {supervised.Count} views, " +
                $"{views.Count - supervised.Count} held out");

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
                int vi = supervised[it % supervised.Count];
                var cam = views[vi].Camera;
                var (near, far) = SplatBounds.DepthRangeFor(box, cam);

                _trainer.SetTargetFrom(targets, vi);
                float loss = await _trainer.TrainStepAsync(packed, n, cam, near, far, geometry: geo);

                // Once, early: how much of the geometry gradient survives the fixed-point
                // atomic? Gradients cross it as integers scaled by 2^20, and dL/d(pixel) is
                // 1/(3*W*H) - about 1e-6 at this resolution - so a small splat's position
                // gradient can be only a few QUANTA. If most splats quantise to zero the
                // geometry cannot move and the run would look like a bad learning rate.
                if (geo != null && it == supervised.Count * 2) await ReportGradientHealthAsync(n);
                if (_trainer.LastOverflowed) overflowed++;

                cycleSum += loss;
                cycleN++;
                if (it % supervised.Count == supervised.Count - 1)
                {
                    double mean = cycleSum / cycleN;
                    if (double.IsNaN(firstCycle)) firstCycle = mean;
                    lastCycle = mean;
                    int cycle = (it + 1) / supervised.Count;
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
            Console.WriteLine($"[Train] trainer PSNR supervised {baseInit:F2} -> {fitInit:F2} dB");
            Console.WriteLine($"[Train] trainer PSNR HELD OUT   {baseHeld:F2} -> {fitHeld:F2} dB");

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
    /// How much of the gradient survives quantisation. Reported once per run, from one readback.
    /// </summary>
    private async Task ReportGradientHealthAsync(int n)
    {
        float[] g = await _trainer!.ReadGradientsAsync(n);
        const float quantum = 1f / 1048576f;

        int stride = SplatTrainerGpu.GradsPerSplat;
        int liveColour = 0, liveCentre = 0, liveConic = 0;
        double sumCentre = 0;
        float maxCentre = 0, maxConic = 0;
        for (int i = 0; i < n; i++)
        {
            int o = i * stride;
            if (g[o] != 0f || g[o + 1] != 0f || g[o + 2] != 0f) liveColour++;

            float cx = MathF.Abs(g[o + 4]), cy = MathF.Abs(g[o + 5]);
            float cen = MathF.Max(cx, cy);
            if (cen > 0f) { liveCentre++; sumCentre += cen; }
            if (cen > maxCentre) maxCentre = cen;

            float con = MathF.Max(MathF.Abs(g[o + 6]), MathF.Max(MathF.Abs(g[o + 7]), MathF.Abs(g[o + 8])));
            if (con > 0f) liveConic++;
            if (con > maxConic) maxConic = con;
        }
        double meanCentre = liveCentre > 0 ? sumCentre / liveCentre : 0;

        Console.WriteLine(
            $"[Train] gradient health: colour {liveColour * 100.0 / n:F1}% nonzero, " +
            $"centre {liveCentre * 100.0 / n:F1}%, conic {liveConic * 100.0 / n:F1}%");
        Console.WriteLine(
            $"[Train] centre |grad| mean {meanCentre:G3} ({meanCentre / quantum:F1} quanta), " +
            $"max {maxCentre:G3}; conic max {maxConic:G3} " +
            $"({maxConic / quantum:G3} quanta, i32 overflows at {int.MaxValue * (double)quantum:G3})");
    }

    /// <summary>
    /// Mean PSNR of the trainer's own forward render against the targets, split by whether the
    /// optimiser was allowed to fit to the view. Held-out is the number that matters; the
    /// supervised set is the control - if THAT does not improve, no gradient is reaching the
    /// parameters at all, and any held-out movement is noise.
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

            if (views[i].UsedForSupervision) { sumInit += psnr; nInit++; }
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
