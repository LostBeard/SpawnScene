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

    /// <summary>
    /// Skip the colour/opacity Adam step for splats with no gradient this iteration. Off by
    /// default; see <c>SplatTrainerGpu.SkipZeroGradientSteps</c>.
    /// </summary>
    public static bool SkipZeroGradientSteps { get; set; }

    /// <summary>
    /// Fit the fixed-point gradient scales to the scene from the measured gradients. On by
    /// default; a knob so a run can be compared against the old fixed constants.
    /// </summary>
    public static bool AdaptGradientScales { get; set; } = true;

    /// <summary>
    /// Ceiling on the resident target stack, which holds every view as float RGB.
    ///
    /// 256 MiB leaves room for the splats, the Adam moments, the key-indexed gradient buffers
    /// and the render targets alongside it. It is a budget rather than a guess at what a card
    /// has, for the same reason the splat budget is derived from the binding limit: the failure
    /// is not a clean out-of-memory, it is the device disappearing mid-sort.
    /// </summary>
    public static long MaxTargetStackBytes { get; set; } = 256L * 1024 * 1024;

    /// <summary>
    /// Evaluate held-out PSNR every N cycles, 0 to disable. A full evaluation renders every
    /// view, so this trades run time for the shape of the curve - worth it whenever the two
    /// endpoints disagree about what is happening in between.
    /// </summary>
    public static int HeldOutEveryCycles { get; set; } = 10;

    /// <summary>
    /// Run adaptive density control every N cycles, 0 to disable.
    ///
    /// This is the step that CREATES geometry. Optimising a fixed set of Gaussians can only
    /// redistribute what the initialisation put there, and an SfM cloud is deliberately sparse
    /// - 79,922 points for a room the reference would finish with one to three million. The
    /// blur in an under-densified render is not a tuning problem, it is missing Gaussians.
    /// </summary>
    /// <summary>
    /// Run adaptive density control every N ITERATIONS, 0 to disable. The reference uses 100.
    ///
    /// Iterations, not cycles. A cycle is one pass over the supervised views, so on a 99-view
    /// capture "every 5 cycles" is every 495 iterations - and with densification stopping
    /// halfway that gave EIGHT densification steps against the reference's 145. The growth this
    /// step exists to produce simply never had the chances to happen.
    /// </summary>
    public static int DensifyEveryIters { get; set; }

    /// <summary>
    /// Warm-up before densifying, in iterations. The reference waits 500: the screen-space
    /// gradient is the signal, and it means nothing until the splats have been fitted at all.
    /// </summary>
    public static int DensifyFromIter { get; set; } = 500;

    /// <summary>
    /// How far the position learning rate falls across a run. The reference goes from 1.6e-4 to
    /// 1.6e-6, a factor of 100.
    /// </summary>
    public static float PositionLrDecay { get; set; } = 0.01f;

    /// <summary>
    /// Steps the position decay is measured against. The reference's
    /// <c>position_lr_max_steps</c>, which is a FIXED 30,000 - not the length of the run.
    ///
    /// I tied it to the run length, which is wrong and measurably so: an 8,000-iteration run
    /// then travels the whole 100x decay, so geometry is frozen by iteration 4,000 while the
    /// scene is still coarse. Held-out PSNR fell 1.5 dB against the same configuration without
    /// decay. Against a fixed 30,000 an 8,000-step run ends around 0.46x its starting rate,
    /// which is the schedule the published numbers come from.
    /// </summary>
    public static int PositionLrMaxSteps { get; set; } = 30_000;

    /// <summary>
    /// Train against ONE view only, -1 to disable. A capacity probe, not a reconstruction:
    /// see the comment at its use site.
    /// </summary>
    public static int FitSingleViewIndex { get; set; } = -1;

    /// <summary>
    /// Ceiling on the splat count during densification.
    ///
    /// Growth is unbounded by nature and a browser tab is not. Derived from the trainer key
    /// binding limit the same way the generator budget is, rather than picked.
    /// </summary>
    public static int MaxDensifiedSplats { get; set; } = 1_200_000;

    /// <summary>
    /// Cap every opacity every N cycles, 0 to disable. The reference does this every 3,000
    /// iterations.
    ///
    /// It forces the optimiser to re-earn every Gaussian: the ones that matter recover within a
    /// few hundred iterations and the ones that were only filling space stay faint and are
    /// pruned. It is also what UNLOCKS the size prunes - the reference holds those back until
    /// the first reset, because a large Gaussian may simply not have had the chance to shrink.
    /// Without it nothing removes an over-elongated splat, and the render fills with needles.
    /// </summary>
    public static int OpacityResetEveryIters { get; set; }

    /// <summary>
    /// Stop densifying after this fraction of the run, as the reference does at 15,000 of
    /// 30,000. Splats added at the very end never get trained; they only add cost and noise.
    /// </summary>
    public const float DensifyUntilFraction = 0.5f;

    bool _hadOpacityReset;

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
            var box = aabb.Value;   // reassigned after densification changes the scene
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

            // Bound the TOTAL target memory, not the per-view resolution.
            //
            // maxTrainDimension caps one view and says nothing about how many there are, which
            // is fine until the view count grows. MEASURED: 88 views of drjohnson came down to
            // 720x474 by that rule and still needed 88 x 720 x 474 x 3 floats = 360 MB for the
            // target stack alone, before a single trainer buffer existed, and the device was
            // lost during the first sort. That is the third limit today that watched one
            // dimension while the scaling one was somewhere else - the splat budget lived in a
            // single code path, keysPerSplat was a caller default, and now this.
            //
            // The targets are float RGB and every supervised AND held-out view is resident, so
            // the bound is views x pixels x 3 x 4 bytes.
            var (tw, th) = views[0].Camera.FitWithin(maxTrainDimension);
            long TargetBytes(int pw, int ph) => (long)views.Count * pw * ph * sizeof(uint);
            if (TargetBytes(tw, th) > MaxTargetStackBytes)
            {
                int shrunk = maxTrainDimension;
                while (shrunk > 128 && TargetBytes(tw, th) > MaxTargetStackBytes)
                {
                    shrunk = shrunk * 3 / 4;
                    (tw, th) = views[0].Camera.FitWithin(shrunk);
                }
                Console.WriteLine(
                    $"[Train] {views.Count} views would need " +
                    $"{TargetBytes(views[0].Camera.FitWithin(maxTrainDimension).Width, views[0].Camera.FitWithin(maxTrainDimension).Height) / (1024 * 1024)} MiB " +
                    $"of target stack at {maxTrainDimension}px; training at {tw}x{th} to fit " +
                    $"{MaxTargetStackBytes / (1024 * 1024)} MiB");
            }
            else if (tw != w || th != h)
            {
                Console.WriteLine(
                    $"[Train] training at {tw}x{th} instead of {w}x{h} " +
                    $"({(long)w * h / 1_000_000.0:F1} MP per view is too much target memory)");
            }
            w = tw; h = th;

            _trainer ??= new SplatTrainerGpu(_gpuService);
            _trainer.SkipZeroGradientSteps = SkipZeroGradientSteps;
            if (!_trainerInitialized) { _trainer.Initialize(); _trainerInitialized = true; }
            _trainer.Resize(w, h, n, keysPerSplat);

            // -- Upload every target photograph once --
            // Per-iteration upload would be 3.7 MB of traffic per step and would dominate the
            // measurement. One scratch frame is reused on the host so the WASM heap never holds
            // more than a single image.
            // Targets are stored PACKED - one RGBA8 word per pixel, not three floats. Four
            // bytes against twelve, so the same budget supervises three times as many views,
            // and views are the scarce resource: the reference trains drjohnson on roughly 230
            // images while we were using 33.
            using var targets = accel.Allocate1D<uint>((long)views.Count * w * h);

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
            // The camera-rig radius, the scene extent the reference scales both the position
            // learning rate and the clone/split size threshold by. Needed whether or not
            // geometry is being optimised, because density control uses it too.
            var rigCentroid = Vector3.Zero;
            foreach (var v in views) rigCentroid += v.Camera.Position;
            rigCentroid /= views.Count;
            float rigRadius = 0f;
            foreach (var v in views)
                rigRadius = MathF.Max(rigRadius, Vector3.Distance(v.Camera.Position, rigCentroid));
            if (rigRadius <= 0f) rigRadius = MathF.Max(box.Diagonal, 1e-3f);

            SplatTrainerGpu.GeometryStep? geo = null;
            float positionLrInit = 0f;
            if (optimiseGeometry)
            {

                positionLrInit = PositionLrScale * 1.6e-4f * rigRadius;
                geo = new SplatTrainerGpu.GeometryStep(
                    PositionLr: positionLrInit,
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

            // Fit ONE view, as a capacity test.
            //
            // Supervised PSNR sits at 15-18 dB on views the model trains on directly, where a
            // working 3DGS reaches 30+. That is not a generalisation failure, it is a failure
            // to fit data we are handing it - and the two want completely different fixes. With
            // a single view and enough iterations the model can simply memorise the image, so
            // whatever PSNR it plateaus at is the CEILING of the forward and backward passes.
            // If that ceiling is ~18 dB the defect is in the renderer or the gradients, and no
            // amount of supervision, density or scheduling will move it.
            if (FitSingleViewIndex >= 0 && FitSingleViewIndex < views.Count)
            {
                supervised = new List<int> { FitSingleViewIndex };
                Console.WriteLine(
                    $"[Train] CAPACITY TEST: fitting view {FitSingleViewIndex} " +
                    $"({views[FitSingleViewIndex].ImageName}) ALONE. Supervised PSNR is now a " +
                    "ceiling on what this rasteriser and its gradients can express, not a " +
                    "reconstruction quality. Held-out numbers are meaningless here.");
            }
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
                // EVERY view, not one, and not only the supervised ones. Demand varies a lot
                // between views - on Bathroom the first view fits and 6% of the rest do not, so
                // a single probe reports success and the run then loses gradients on 94 frames.
                //
                // Held-out views are included because they are RENDERED during evaluation. One
                // that overflows loses splats and scores an incomplete image, which reads as
                // "that pose is bad" rather than "that frame was truncated" - and with geometry
                // enabled a view can cross the threshold part-way through a run, so the same
                // view scores differently for a reason that has nothing to do with quality.
                int peak = 0;
                foreach (int si in Enumerable.Range(0, views.Count))
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
                        $"({peak / (double)n:F1} per splat) across {views.Count} views " +
                        $"({supervised.Count} supervised, {views.Count - supervised.Count} held out); " +
                        $"re-sizing to {want} per splat with 25% headroom");
                    // The target STACK is a separate allocation and stays filled; Resize only
                    // reallocates the trainer's own per-frame buffers.
                    _trainer.Resize(w, h, n, want);
                }
            }

            // -- Baseline: how well does the untrained scene already explain each photo? --
            var baseline = await EvaluateAsync(_trainer, packed, n, views, targets, box);
            WarnOnEvalOverflow(baseline, views.Count);

            _trainer.InitOptimizerState(packed, n);

            // -- Optimise --
            var start = DateTime.UtcNow;
            int overflowed = 0;
            // Loss is per VIEW, and views differ in how much of the frame the temple covers, so
            // comparing iteration 0 against iteration N compares two different pictures. Average
            // over a full cycle of views instead - that is the only comparable quantity here.
            double cycleSum = 0;
            int cycleN = 0;
            var curve = new List<EvalScores>();
            var probeIterations = TrainingSchedule.ProbeIterations(iterations, supervised.Count);
            int totalCycles = iterations / Math.Max(1, supervised.Count);

            // Count, over the FIRST full cycle, how many views ever move each splat. Measured at
            // the start because it is a property of the INITIALISATION - a stack of per-view
            // depth shells - not of anything the optimiser subsequently does.
            _trainer.ResetViewSupport(n);
            double firstCycle = double.NaN, lastCycle = double.NaN;
            for (int it = 0; it < iterations; it++)
            {
                // Decay the position rate as the reference does, 100x across the run. Rebuilt
                // per iteration because it is the only rate that changes; the others are
                // scale-free parameterisations (log scale, unit quaternion) and stay put.
                if (geo is { } g0)
                    geo = g0 with
                    {
                        PositionLr = TrainingSchedule.ExponentialLr(
                            positionLrInit, positionLrInit * PositionLrDecay,
                            it, PositionLrMaxSteps),
                    };

                int vi = supervised[it % supervised.Count];
                var cam = views[vi].Camera.ScaledTo(w, h);
                var (near, far) = SplatBounds.DepthRangeFor(box, cam);

                _trainer.SetTargetFrom(targets, vi);
                float loss = await _trainer.TrainStepAsync(packed, n, cam, near, far, geometry: geo);

                // How much of the gradient survives the fixed-point atomic? Gradients cross it
                // as scaled integers, and dL/d(pixel) is 1/(3*W*H) - about 1e-6 at this
                // resolution - so a small splat's position gradient can be only a few QUANTA.
                // If most splats quantise to zero the geometry cannot move and the run would
                // look like a bad learning rate. Not gated on geometry: the colour/opacity
                // stale-step fraction matters either way, and at 6 KB a call this is cheap.
                if (it < supervised.Count) _trainer.AccumulateViewSupport(n);
                if (it == supervised.Count - 1) await ReportViewSupportAsync(n);
                if (DensifyEveryIters > 0) _trainer.AccumulateDensifyStats(n);

                // Density control on an ITERATION schedule, like the reference: every 100
                // iterations from 500 until half way, then the model is left to settle.
                if (it >= DensifyFromIter)
                {
                    bool stillGrowing = it < iterations * DensifyUntilFraction;
                    // Each schedule is checked on its OWN period. Nesting the reset inside the
                    // densify period would silently disable it whenever the two are not
                    // multiples of one another - a whitelist of one, in arithmetic form.
                    bool densifying = DensifyEveryIters > 0 && stillGrowing
                        && (it + 1) % DensifyEveryIters == 0;
                    bool resetOpacity = OpacityResetEveryIters > 0 && stillGrowing
                        && (it + 1) % OpacityResetEveryIters == 0;
                    if (densifying || resetOpacity)
                    {
                        var grown = await DensifyAsync(
                            packed, n, rigRadius, densifying, resetOpacity);
                        if (grown != null)
                        {
                            (packed, n) = grown.Value;
                            var refreshed = await SplatBounds.ComputeAsync(accel, packed, n);
                            if (refreshed != null) box = refreshed.Value;
                        }
                    }
                }

                if (probeIterations.Contains(it))
                    await ReportGradientHealthAsync(n, vi);
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

                    // Held-out PSNR DURING the run, not only at the ends.
                    //
                    // Bathroom starts at 12.41 dB held out and finishes at 10.81 while its
                    // supervised number climbs, and two endpoints cannot tell the difference
                    // between "peaks early then declines" and "degrades from the first step".
                    // Those want opposite responses - the first is early stopping, the second
                    // is a learning rate or a regulariser - so the curve is the measurement,
                    // and guessing between them without it is how a day gets spent on the
                    // wrong one.
                    // Densify BEFORE evaluating, so the reported number is of the scene that
                    // will keep training rather than the one that just stopped existing.
                    if (HeldOutEveryCycles > 0 && cycle % HeldOutEveryCycles == 0)
                    {
                        var sample = await EvaluateAsync(_trainer, packed, n, views, targets, box);
                        curve.Add(sample);
                        Console.WriteLine($"[Train] cycle {cycle,4} {sample.Describe()}");
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

            var fitted = await EvaluateAsync(_trainer, packed, n, views, targets, box);
            WarnOnEvalOverflow(fitted, views.Count);

            Console.WriteLine(
                $"[Train] trainer supervised PSNR {baseline.SupPsnr:F2} -> {fitted.SupPsnr:F2} dB, " +
                $"SSIM {baseline.SupSsim:F4} -> {fitted.SupSsim:F4}");
            Console.WriteLine(
                $"[Train] trainer HELD OUT   PSNR {baseline.HeldPsnr:F2} -> {fitted.HeldPsnr:F2} dB, " +
                $"SSIM {baseline.HeldSsim:F4} -> {fitted.HeldSsim:F4}");

            // The comparable number. Endpoints are single samples of an oscillation; this is
            // what an A/B should be read off.
            if (curve.Count >= 2)
                Console.WriteLine(
                    $"[Train] COMPARE: held-out mean over {curve.Count} samples - " +
                    $"PSNR {curve.Average(c => c.HeldPsnr):F3} dB, " +
                    $"SSIM {curve.Average(c => c.HeldSsim):F4} " +
                    $"(baseline before training: PSNR {baseline.HeldPsnr:F3}, SSIM {baseline.HeldSsim:F4})");

            ReportCurve(curve);

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
    /// The oscillation is the point of the curve, so report its SPREAD rather than leaving six
    /// log lines to be compared by eye. MEASURED on Bathroom: held-out PSNR swings 2.74 dB
    /// within a run and 1.40 dB between two runs of the SAME configuration, which is larger
    /// than most of the differences anyone would want to read off the endpoints.
    /// </summary>
    private static void ReportCurve(IReadOnlyList<EvalScores> curve)
    {
        if (curve.Count < 2) return;

        // The MEAN is the comparison statistic, not the last value.
        //
        // Held-out oscillates - measured at 1.88 to 1.99 dB of spread within a single run - so
        // the final number is one sample of that oscillation, and two runs of the SAME
        // configuration differed by 1.52 dB. Comparing endpoints cannot resolve anything smaller
        // than the swing, and averaging whole extra runs to beat it costs GPU hours. The mean
        // over the samples already collected has roughly half the standard error of one of them,
        // for free.
        static string Line(string metric, IReadOnlyList<float> v) =>
            $"{metric} mean {v.Average():F4} (min {v.Min():F4} max {v.Max():F4} " +
            $"spread {v.Max() - v.Min():F4} last {v[^1]:F4})";

        var psnr = curve.Select(c => c.HeldPsnr).ToList();
        var ssim = curve.Select(c => c.HeldSsim).ToList();
        Console.WriteLine($"[Train] held-out curve over {curve.Count} samples: {Line("PSNR", psnr)}");
        Console.WriteLine($"[Train] held-out curve over {curve.Count} samples: {Line("SSIM", ssim)}");
    }

    /// <summary>
    /// A view that overflowed the key buffer during EVALUATION scored an incomplete render. The
    /// run summary only counts overflow during training iterations, so without this the two
    /// cases are indistinguishable in the log.
    /// </summary>
    private static void WarnOnEvalOverflow(EvalScores scores, int viewCount)
    {
        if (scores.Overflowed == 0) return;
        Console.WriteLine(
            $"[Train] WARNING: {scores.Overflowed} of {viewCount} views overflowed the key buffer " +
            "during evaluation - those scores are of an incomplete render, not of the scene");
    }

    /// <summary>
    /// How much of the gradient survives quantisation, and how many splats have none at all.
    ///
    /// Over EVERY splat, via a GPU reduction. It used to copy the whole accumulator to the host
    /// - 725k splats is 26 MB - and when that was cut to a 65,536-splat prefix it went blind:
    /// the merged splat buffer is view-major, so a prefix is the top of view 0's depth map and
    /// can legitimately be all zero. Every Bathroom run today printed "INCONCLUSIVE" because of
    /// it. A prefix of a view-major buffer is not a sample.
    /// </summary>
    /// <summary>
    /// One adaptive density control step: clone, split, prune, and rebuild the splat buffer.
    ///
    /// Returns the new buffer and count, or null when nothing changed.
    ///
    /// CPU transfer: the whole splat buffer, both ways. It is deliberate rather than a lapse.
    /// The decision changes the splat COUNT, so every trainer buffer has to be reallocated and
    /// the optimiser state reseeded regardless; at 80k splats the round trip is 4.5 MB against
    /// a step that already reallocates tens of megabytes, and it runs once every hundred
    /// iterations. A GPU clone/split needs a prefix sum and a compaction pass, and that is
    /// worth writing when the splat count makes it worth writing, not before it works at all.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packed, int n)?> DensifyAsync(
        MemoryBuffer1D<float, Stride1D.Dense> packed, int n, float sceneExtent,
        bool densify, bool resetOpacity)
    {
        var accel = _gpuService.WebGPUAccelerator;
        var stats = await _trainer!.ReadDensifyStatsAsync(n);
        float[] raw = await packed.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);

        var splats = new SplatDensityControl.Splat[n];
        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            splats[i] = new SplatDensityControl.Splat
            {
                PosX = raw[o], PosY = raw[o + 1], PosZ = raw[o + 2],
                ColR = raw[o + 3], ColG = raw[o + 4], ColB = raw[o + 5],
                ScaleX = raw[o + 6], ScaleY = raw[o + 7], ScaleZ = raw[o + 8],
                Opacity = raw[o + 9],
                QuatX = raw[o + 10], QuatY = raw[o + 11], QuatZ = raw[o + 12], QuatW = raw[o + 13],
            };
        }

        // Split children are drawn from their parent's own ellipsoid, so this needs normal
        // deviates. Seeded per step so a run reproduces; Box-Muller because there is no
        // Gaussian in the BCL and an approximation here biases where geometry appears.
        var rng = new Random(1234 + n);
        float NextNormal()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // Never grow past what the trainer can actually hold. Resize clamps keysPerSplat down
        // to fit the binding, and a clamped budget overflows - so splats beyond this point do
        // not buy detail, they buy frames rendered from an incomplete key list.
        int budget = Math.Min(MaxDensifiedSplats, _trainer.MaxTrainableSplats(_trainer.KeysPerSplat));

        // The size prunes are unlocked by the first opacity reset, exactly as in the reference.
        // Before it, a large Gaussian may simply not have been given the chance to shrink; after
        // it, one that is still large and still faint is not going to earn its place.
        var plan = densify
            ? SplatDensityControl.Decide(
                splats, stats, sceneExtent, _hadOpacityReset, NextNormal, budget)
            : new SplatDensityControl.Plan();

        // Always report, including - especially including - when the answer is "nothing".
        //
        // The first run of this decided nothing NINE times and printed not one line, so there
        // was no way to tell "the scene does not need densifying" from "the signal never
        // arrived". A step that declines to do the main work has to say why, and the why here
        // is a distribution against a threshold, not a boolean.
        var avg = new float[n];
        int visible = 0;
        for (int i = 0; i < n; i++)
        {
            avg[i] = stats[i].AverageGradient;
            if (stats[i].VisibleCount > 0) visible++;
        }
        System.Array.Sort(avg);
        float sizeSplit = SplatDensityControl.PercentDense * sceneExtent;
        int big = 0;
        foreach (var sp in splats) if (sp.MaxScale > sizeSplit) big++;

        Console.WriteLine(
            $"[Densify] signal: {visible * 100.0 / n:F1}% of {n:N0} splats visible, " +
            $"avg |grad| median {avg[n / 2]:G3} p90 {avg[n * 9 / 10]:G3} max {avg[n - 1]:G3} " +
            $"vs threshold {SplatDensityControl.GradientThresholdNdc:G3}; " +
            $"{big:N0} splats above the {sizeSplit:G3} split size; plan: {plan}");

        if (plan.Add.Count == 0 && plan.Remove.Count == 0 && !resetOpacity)
        {
            _trainer.ResetDensifyStats();
            return null;
        }

        var priorAdam = await _trainer.ReadAdamStateAsync(n);
        var grown = SplatDensityControl.Apply(splats, plan, out var survivors);

        if (resetOpacity)
        {
            var arr = grown.ToArray();
            SplatDensityControl.ResetOpacity(arr);
            grown = arr.ToList();
            _hadOpacityReset = true;
            Console.WriteLine(
                $"[Densify] opacity reset to at most {SplatDensityControl.OpacityResetTo} " +
                "- every Gaussian now has to re-earn its place, and the size prunes are live");
        }
        int m = grown.Count;
        if (m <= 0)
        {
            Console.WriteLine("[Densify] plan would remove every splat - ignored");
            _trainer.ResetDensifyStats();
            return null;
        }

        var outRaw = new float[(long)m * SplatFormat.Floats];
        for (int i = 0; i < m; i++)
        {
            var g = grown[i];
            int o = i * SplatFormat.Floats;
            outRaw[o] = g.PosX; outRaw[o + 1] = g.PosY; outRaw[o + 2] = g.PosZ;
            outRaw[o + 3] = g.ColR; outRaw[o + 4] = g.ColG; outRaw[o + 5] = g.ColB;
            outRaw[o + 6] = g.ScaleX; outRaw[o + 7] = g.ScaleY; outRaw[o + 8] = g.ScaleZ;
            outRaw[o + 9] = g.Opacity;
            outRaw[o + 10] = g.QuatX; outRaw[o + 11] = g.QuatY;
            outRaw[o + 12] = g.QuatZ; outRaw[o + 13] = g.QuatW;
        }

        var next = accel.Allocate1D<float>((long)m * SplatFormat.Floats);
        next.CopyFromCPU(outRaw);
        await accel.SynchronizeAsync();

        // Through the renderer, so the displayed scene and the trained scene stay the same
        // object. Reading the buffer back out of it afterwards is what keeps that true.
        await _gpuRenderer.UploadSceneFromGpuBuffer(next, m);
        var live = _gpuRenderer.PackedSplatBuffer;
        if (live == null)
        {
            Console.WriteLine("[Densify] FAIL: renderer did not take the grown buffer");
            return null;
        }
        if (_sceneManager.ActiveScene != null) _sceneManager.ActiveScene.GpuSplatCount = m;

        // Re-size with the trainer's CURRENT key budget, not the caller's original guess.
        //
        // The loop measures peak key demand after the first cycle and re-sizes with headroom -
        // drjohnson settles around 22 keys per splat, not the default 8. Passing the original
        // argument back in threw that away and every frame after the first densification
        // overflowed: "KEY OVERFLOW: 1,775,260 needed, capacity 641,256". A measurement the
        // system already made is not something a later caller gets to discard.
        // Re-size on MEASURED demand, not on the last sizing decision.
        //
        // The loop measures peak demand once, early, and adds 25% headroom. Densification then
        // adds splats for hundreds of frames and eats it: the log filled with overflows missing
        // by half a percent - "4,622,178 needed, capacity 4,601,320" - each one a frame trained
        // on an incomplete render. The peak over the whole window is the evidence; use it, with
        // the same headroom, rather than carrying forward a number that was true of a smaller
        // scene.
        var (w, h) = _trainer.Size;
        int keys = _trainer.KeysPerSplat;
        if (_trainer.PeakKeyDemand > 0)
        {
            int needed = (int)Math.Ceiling(_trainer.PeakKeyDemand * 1.25 / Math.Max(1, n));
            if (needed > keys)
            {
                Console.WriteLine(
                    $"[Densify] keysPerSplat {keys} -> {needed}: peak demand was " +
                    $"{_trainer.PeakKeyDemand:N0} keys for {n:N0} splats over this window");
                keys = needed;
            }
        }
        _trainer.Resize(w, h, m, keys);
        _trainer.ResetPeakKeyDemand();

        // Reseed the derived state - opacity logits and log scales come from the packed buffer,
        // which is correct for clones and split children alike - and then put the Adam moments
        // back where they belong. InitOptimizerState zeroes them, so the order matters.
        _trainer.InitOptimizerState(live, m);

        // Carrying opacity momentum through a reset would simply undo it within a few steps,
        // which is why the reference zeroes it alongside. Slot 9 is opacity in the packed
        // layout, and the Adam moments follow that layout one for one.
        _trainer.RestoreAdamState(priorAdam, survivors,
            zeroSlot: resetOpacity ? SplatFormat.OffOpacity : -1);
        _trainer.ResetDensifyStats();

        Console.WriteLine($"[Densify] {n:N0} -> {m:N0} splats: {plan}");
        return (live, m);
    }

    /// <summary>
    /// How many views actually constrain each splat, after one full cycle.
    ///
    /// The number that decides whether any optimiser experiment on this project is worth
    /// running. Initialisation unprojects a monocular depth map per view, so N views produce N
    /// private shells; a splat that only ever receives a gradient from its own view is free to
    /// explain that view and nothing else. Supervised loss falls, held-out loss rises, and the
    /// cause is the parameterisation rather than the learning rate.
    /// </summary>
    private async Task ReportViewSupportAsync(int n)
    {
        var vs = await _trainer!.ReadViewSupportAsync(n);

        Console.WriteLine(
            $"[Train] view support over {vs.Views} views ({vs.Splats:N0} splats): " +
            $"never {vs.Unconstrained * 100.0 / n:F1}%, " +
            $"1 view {vs.OneView * 100.0 / n:F1}%, " +
            $"2 {vs.TwoViews * 100.0 / n:F1}%, " +
            $"3 {vs.ThreeViews * 100.0 / n:F1}%, " +
            $"4+ {vs.FourOrMore * 100.0 / n:F1}%  (mean {vs.MeanViews:F2} views/splat)");

        // Say what it MEANS, once, rather than leaving the reader to do the arithmetic at the
        // bottom of a thousand-line log. The threshold is a reporting choice, not a gate - it
        // decides which sentence prints, never what the run does.
        if (vs.SingleViewFraction > 0.9)
            Console.WriteLine(
                $"[Train] ⚠ {vs.SingleViewFraction:P1} of splats are constrained by AT MOST ONE " +
                "view. This reconstruction is a stack of per-view shells, not a shared scene: " +
                "every splat can fit its own view without ever being contradicted. Expect " +
                "supervised PSNR to rise and held-out PSNR to fall, and expect no optimiser " +
                "setting to change that - the parameterisation is the problem, not the step.");
        else
            Console.WriteLine(
                $"[Train] {1 - vs.SingleViewFraction:P1} of splats are seen by two or more views " +
                "and can therefore be contradicted - that is what makes generalisation possible.");
    }

    private async Task ReportGradientHealthAsync(int n, int viewIndex)
    {
        var st = await _trainer!.ReadGradientStatsAsync(n);

        float centreQuantum = 1f / _trainer!.FixedScaleFor(4);
        float conicQuantum = 1f / _trainer!.FixedScaleFor(6);

        if (st.ColourLive == 0 && st.CentreLive == 0 && st.ConicLive == 0)
        {
            // Every splat is counted now, so this is a measurement rather than a shrug. It used
            // to be reported as INCONCLUSIVE because the probe could not tell "nothing has a
            // gradient" from "I looked in the wrong place".
            Console.WriteLine(
                $"[Train] gradient health (view {viewIndex}): NO splat received a gradient " +
                "this step. The backward pass produced nothing for THIS view - which is a " +
                "property of the view, not of the run, unless every probe says the same.");
            return;
        }

        Console.WriteLine(
            $"[Train] gradient health (view {viewIndex}, all {n:N0} splats): " +
            $"colour {st.ColourLive * 100.0 / n:F1}% nonzero, " +
            $"centre {st.CentreLive * 100.0 / n:F1}%, conic {st.ConicLive * 100.0 / n:F1}%");

        // The premise of the stale-momentum question, as a number rather than an inference from
        // the round-robin structure. adam_geometry refuses to step a splat with no gradient;
        // adam_step does not, on the stated judgement that it is "harmless for colour".
        Console.WriteLine(
            $"[Train] {st.StaleColourFraction:P1} of splats will take a colour/opacity Adam step " +
            "on a gradient of exactly zero this iteration");

        double meanCentre = st.MeanCentreQuanta * centreQuantum;
        double maxConic = st.MaxConicQuanta * conicQuantum;
        double centreCeiling = int.MaxValue * (double)centreQuantum;
        double conicCeiling = int.MaxValue * (double)conicQuantum;
        Console.WriteLine(
            $"[Train] centre |grad| mean {meanCentre:G3} ({st.MeanCentreQuanta:F0} quanta, " +
            $"saturates at {centreCeiling:G3}); " +
            $"conic max {maxConic:G3} ({st.MaxConicQuanta:G3} quanta, " +
            $"saturates at {conicCeiling:G3})");

        // Fit the scales to what this scene actually produces.
        //
        // Watching was not enough. Both scales were consts and on drjohnson they were wrong in
        // OPPOSITE directions at the same time - the centre gradient carried one quantum of 32
        // while the conic sat at 96% of the i32 wrap - so a warning alone just reported a
        // reconstruction that could not work.
        if (AdaptGradientScales && _trainer.RecalibrateGradientScales(st))
        {
            // The scales changed, so the numbers just printed were measured at the old ones.
            Console.WriteLine(
                "[Train] the readings above were taken at the previous scales; the next report " +
                "is the one to compare against.");
        }

        // The conic gradient grows with a splat's pixel AREA, so it is the one that can run out
        // of range rather than out of precision - and an i32 atomic wraps silently rather than
        // clamping, which would read as a wrong gradient, not as an error. Densification will
        // create larger splats than exist today, so this needs to be watched, not assumed.
        if (st.MaxConicQuanta > 0.1 * int.MaxValue || st.MaxCentreQuanta > 0.1 * int.MaxValue)
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
    /// <summary>
    /// One evaluation pass: mean PSNR and mean SSIM, split by whether the optimiser was allowed
    /// to fit the view, plus how many views rendered incomplete.
    /// </summary>
    internal readonly record struct EvalScores(
        float SupPsnr, float SupSsim, float HeldPsnr, float HeldSsim, int Overflowed)
    {
        public string Describe() =>
            $"held out PSNR {HeldPsnr:F2} dB SSIM {HeldSsim:F4} | " +
            $"supervised PSNR {SupPsnr:F2} dB SSIM {SupSsim:F4}";
    }

    static async Task<EvalScores> EvaluateAsync(
        SplatTrainerGpu trainer,
        MemoryBuffer1D<float, Stride1D.Dense> packed, int n,
        IReadOnlyList<TrainingView> views,
        MemoryBuffer1D<uint, Stride1D.Dense> targets,
        SplatBounds.Aabb box)
    {
        var (w, h) = trainer.Size;
        double supPsnr = 0, supSsim = 0, heldPsnr = 0, heldSsim = 0;
        int nSup = 0, nHeld = 0, overflowed = 0;

        for (int i = 0; i < views.Count; i++)
        {
            var cam = views[i].Camera.ScaledTo(w, h);
            var (near, far) = SplatBounds.DepthRangeFor(box, cam);

            // readback:false - the comparison happens on the GPU. Copying the render and the
            // target back to difference them in a loop was about 650 MB per pass on a 35-view
            // capture, to produce one number per view.
            await trainer.RenderForwardAsync(packed, n, cam, near, far, readback: false);
            var (psnr, ssim) = await trainer.ScoreAgainstAsync(targets, i);

            // A view that overflowed the key buffer rendered with splats missing, so its score
            // is of an incomplete image. Counting it here is what stops that reading as "that
            // pose is bad" rather than "that frame was truncated".
            if (trainer.LastOverflowed) overflowed++;

            if (views[i].UsedForSupervision) { supPsnr += psnr; supSsim += ssim; nSup++; }
            else { heldPsnr += psnr; heldSsim += ssim; nHeld++; }
        }

        return new EvalScores(
            nSup > 0 ? (float)(supPsnr / nSup) : 0f,
            nSup > 0 ? (float)(supSsim / nSup) : 0f,
            nHeld > 0 ? (float)(heldPsnr / nHeld) : 0f,
            nHeld > 0 ? (float)(heldSsim / nHeld) : 0f,
            overflowed);
    }

    /// <summary>
    /// Decode one training photograph into <paramref name="dest"/> as RGB in [0,1].
    /// Matches how splat colour was created at import (byte/255), so the loss compares
    /// like with like.
    /// </summary>
    private async Task<bool> LoadTargetAsync(
        string url, bool fromProjectStore, int quarterTurns, int w, int h,
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex)
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
