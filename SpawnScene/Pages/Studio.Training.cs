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
    /// <summary>
    /// Splat size ceiling as a fraction of the scene diagonal. 0.1 is the reference's PRUNE
    /// threshold being used as a bound because nothing prunes yet; tightening it is the cheap
    /// test of whether bloated splats are what melts the render.
    /// </summary>
    public static float MaxScaleFraction { get; set; } = 0.1f;

    /// <summary>Multiplier on the position learning rate, for measuring rather than guessing.</summary>
    public static float PositionLrScale { get; set; } = 1f;

    private async Task TrainOnTrainingViewsAsync(
        int iterations, int keysPerSplat = 8, bool optimiseGeometry = false,
        int maxTrainDimension = 720)
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

            // -- Viewport --
            // The cameras carry capture resolution, which for a phone is 13 megapixels; 35 of
            // those as float RGB targets is 5.5 GB. Training runs on a downscaled copy and the
            // intrinsics come with it (CameraParams.ScaledTo).
            // One trainer viewport serves one shape, so take the majority shape and drop the
            // odd ones rather than refusing the run. Real captures are not uniform: Bathroom is
            // 34 portrait frames and one landscape, and losing that single view is obviously
            // better than losing the other 21.
            var allViews = scene.TrainingViews;
            var byShape = allViews
                .GroupBy(v => (v.Camera.Width, v.Camera.Height))
                .OrderByDescending(g => g.Count())
                .ToList();
            var views = byShape[0].ToList();
            int w = views[0].Camera.Width, h = views[0].Camera.Height;
            if (byShape.Count > 1)
            {
                string others = string.Join(", ",
                    byShape.Skip(1).Select(g => $"{g.Count()}x({g.Key.Width}x{g.Key.Height})"));
                Console.WriteLine(
                    $"[Train] training on the {views.Count} views that are {w}x{h}; " +
                    $"skipping {allViews.Count - views.Count} of another shape [{others}]");
            }

            var (tw, th) = views[0].Camera.FitWithin(maxTrainDimension);
            if (tw != w || th != h)
                Console.WriteLine(
                    $"[Train] training at {tw}x{th} instead of {w}x{h} " +
                    $"({(long)w * h / 1_000_000.0:F1} MP per view is too much target memory)");
            w = tw; h = th;

            _trainer ??= new SplatTrainerGpu(_gpuService);
            if (!_trainerInitialized) { _trainer.Initialize(); _trainerInitialized = true; }
            _trainer.Resize(w, h, n, keysPerSplat);

            // -- Upload every target photograph once --
            // Per-iteration upload would be 3.7 MB of traffic per step and would dominate the
            // measurement. One scratch frame is reused on the host so the WASM heap never holds
            // more than a single image.
            int frameFloats = w * h * 3;
            using var targets = accel.Allocate1D<float>((long)views.Count * frameFloats);

            var loadStart = DateTime.UtcNow;
            for (int i = 0; i < views.Count; i++)
            {
                bool ok = await LoadTargetAsync(
                    views[i].ImageName, views[i].FromProjectStore, views[i].QuarterTurns,
                    w, h, targets, i);
                if (!ok)
                {
                    Console.WriteLine($"[Train] FAIL: could not load target {views[i].ImageName}");
                    return;
                }
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
                    PositionLr: PositionLrScale * 1.6e-4f * rigRadius,
                    LogScaleLr: 0.005f,
                    RotationLr: 0.001f,
                    // Without density control nothing prunes, so a splat that stops being
                    // constrained has to be BOUNDED rather than removed - and a bound is not a
                    // substitute for a prune. The reference uses 0.1 * scene extent to DELETE a
                    // bloated Gaussian; used as a ceiling instead, a splat grows to it and stays
                    // there, so one splat can span a tenth of the room. MEASURED on Bathroom with
                    // a 12.41 dB initialisation: training took held-out to 10.81 while supervised
                    // climbed, and the render melted into vertical drips. That is what a
                    // population of splats sitting on the ceiling looks like.
                    MinScale: 1e-6f,
                    MaxScale: MaxScaleFraction * MathF.Max(box.Diagonal, 1e-3f));

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

            // -- Size the key budget from a measurement, not a guess --
            // How many tiles a splat covers is a property of the scene: TempleRing wants about
            // 3, a room wants about 13. Probe one view, then resize once if the budget was
            // short. This must happen before InitOptimizerState, because Resize reallocates.
            {
                // EVERY supervised view, not one. Demand varies a lot between views - on
                // Bathroom the first view fits and 6% of the rest do not, so a single probe
                // reports success and the run then loses gradients on 94 frames.
                int peak = 0;
                foreach (int si in supervised)
                {
                    var probeCam = views[si].Camera.ScaledTo(w, h);
                    var (pn, pf) = SplatBounds.DepthRangeFor(box, probeCam);
                    await _trainer.RenderForwardAsync(packed, n, probeCam, pn, pf, readback: false);
                    peak = Math.Max(peak, _trainer.LastKeyDemand);
                }
                if (peak > n * keysPerSplat)
                {
                    int want = (int)Math.Ceiling(peak * 1.25 / n);
                    Console.WriteLine(
                        $"[Train] peak demand {peak:N0} keys for {n:N0} splats " +
                        $"({peak / (double)n:F1} per splat) across {supervised.Count} views; " +
                        $"re-sizing to {want} per splat with 25% headroom");
                    // The target STACK is a separate allocation and stays filled; Resize only
                    // reallocates the trainer's own per-frame buffers.
                    _trainer.Resize(w, h, n, want);
                }
            }

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
                var cam = views[vi].Camera.ScaledTo(w, h);
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
                    $"buffer (keysPerSplat={_trainer.KeysPerSplat}) - those gradients are incomplete");

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
        float centreQuantum = 1f / SplatTrainerGpu.FixedScaleFor(4);
        float conicQuantum = 1f / SplatTrainerGpu.FixedScaleFor(6);

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

        if (liveColour == 0 && liveCentre == 0 && liveConic == 0)
        {
            // Do not report this as "0% of splats have a gradient". The accumulator is cleared
            // at the start of each step and read after it, and on larger scenes that readback
            // has come back empty while the loss was demonstrably falling - so an all-zero
            // result means the probe saw nothing, not that nothing has a gradient. Saying the
            // latter would send the next reader after an optimiser bug that is not there.
            Console.WriteLine(
                "[Train] gradient health: readback saw an empty accumulator - INCONCLUSIVE, " +
                "not a measurement of zero. Trust the loss curve instead for this run.");
            return;
        }

        Console.WriteLine(
            $"[Train] gradient health: colour {liveColour * 100.0 / n:F1}% nonzero, " +
            $"centre {liveCentre * 100.0 / n:F1}%, conic {liveConic * 100.0 / n:F1}%");
        double centreCeiling = int.MaxValue * (double)centreQuantum;
        double conicCeiling = int.MaxValue * (double)conicQuantum;
        Console.WriteLine(
            $"[Train] centre |grad| mean {meanCentre:G3} ({meanCentre / centreQuantum:F0} quanta, " +
            $"saturates at {centreCeiling:G3}); " +
            $"conic max {maxConic:G3} ({maxConic / conicQuantum:G3} quanta, " +
            $"saturates at {conicCeiling:G3})");

        // The conic gradient grows with a splat's pixel AREA, so it is the one that can run out
        // of range rather than out of precision - and an i32 atomic wraps silently rather than
        // clamping, which would read as a wrong gradient, not as an error. Densification will
        // create larger splats than exist today, so this needs to be watched, not assumed.
        if (maxConic > 0.1 * conicCeiling || maxCentre > 0.1 * centreCeiling)
            Console.WriteLine(
                "[Train] WARNING: a fixed-point gradient is within 10% of saturating. The atomic " +
                "WRAPS rather than clamping, so gradients past this point are wrong, not merely " +
                "coarse. Lower the scale for that slot in SplatTrainerShaders.");
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
        double sumInit = 0, sumHeld = 0;
        int nInit = 0, nHeld = 0;

        for (int i = 0; i < views.Count; i++)
        {
            var cam = views[i].Camera.ScaledTo(w, h);
            var (near, far) = SplatBounds.DepthRangeFor(box, cam);

            // readback:false - the comparison happens on the GPU. Copying the render and the
            // target back to difference them in a loop was about 650 MB per pass on a 35-view
            // capture, to produce one number per view.
            await trainer.RenderForwardAsync(packed, n, cam, near, far, readback: false);
            double psnr = await trainer.PsnrAgainstAsync(targets, i);

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
    private async Task<bool> LoadTargetAsync(
        string url, bool fromProjectStore, int quarterTurns, int w, int h,
        MemoryBuffer1D<float, Stride1D.Dense> stack, int viewIndex)
    {
        try
        {
            byte[]? bytes;
            if (fromProjectStore)
            {
                if (_activeProject == null)
                {
                    Console.WriteLine($"[Train] {url} is a project source but no project is open");
                    return false;
                }
                bytes = await _projectService.GetSourceAsync(_activeProject.Id, url);
                if (bytes == null)
                {
                    Console.WriteLine($"[Train] project source {url} is missing");
                    return false;
                }
            }
            else
            {
                bytes = await _http.GetByteArrayAsync(url);
            }
            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/png" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);

            // The turn was applied to the camera at generation time; the picture has to get the
            // same one or the loss compares a render of one framing against pixels from another.
            // The rotation happens at the TRAINER's resolution, not the capture's - turning a
            // 13 megapixel frame byte by byte in WASM before throwing 99% of it away is pure
            // waste, and DrawImage resamples on the GPU for free.
            int srcW = (quarterTurns % 2 == 0) ? w : h;
            int srcH = (quarterTurns % 2 == 0) ? h : w;

            float srcAspect = (float)bitmap.Width / (int)bitmap.Height;
            float wantAspect = (float)srcW / srcH;
            if (MathF.Abs(srcAspect - wantAspect) > 0.02f)
            {
                Console.WriteLine(
                    $"[Train] {url} is {bitmap.Width}x{bitmap.Height} (aspect {srcAspect:F3}), " +
                    $"but the camera says {srcW}x{srcH} (aspect {wantAspect:F3}) after " +
                    $"{quarterTurns} quarter turn(s) - resizing would distort it");
                return false;
            }

            // Rotate on the CANVAS, not in .NET. A quarter turn is a transform the 2D context
            // applies while resampling, so the upright image is produced in one draw and the
            // pixels never need to be touched a second time.
            using var osc = new OffscreenCanvas(w, h);
            using var ctx = osc.Get2DContext();
            if (quarterTurns != 0)
            {
                // Counter-clockwise, about the centre of the DESTINATION frame.
                ctx.Translate(w / 2.0, h / 2.0);
                ctx.Rotate(-quarterTurns * Math.PI / 2.0);
                ctx.Translate(-srcW / 2.0, -srcH / 2.0);
            }
            ctx.DrawImage(bitmap, 0, 0, srcW, srcH);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var dataArray = imageData.Data;

            // Straight from the canvas to the GPU: the pixels never enter the managed heap,
            // and a kernel expands them into the float stack.
            using var pixels = new Uint8Array(
                dataArray.Buffer, dataArray.ByteOffset, dataArray.Length);
            _trainer!.UploadTargetFrom(stack, viewIndex, pixels);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Train] target load failed for {url}: {ex.Message}");
            return false;
        }
    }
}
