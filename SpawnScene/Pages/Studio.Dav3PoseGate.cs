using System.Numerics;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// How accurate are DAv3's recovered camera poses, against an answer we actually know?
///
/// Every pose decision in this pipeline currently rests on a claim with no measurement behind
/// it: that joint DAv3 "puts every view's depth and camera in ONE shared frame". Commit eeedfff
/// preferred DAv3 over SfM on the strength of it. But on Bathroom the cross-view consistency
/// screen keeps 7% of the splats from a SINGLE joint pass - two of six views keep 0% - and every
/// chunked fold was rejected with a residual of 22-31% of the camera spread. Both of those are
/// consistent with the model's relative camera geometry simply not being that good, and neither
/// proves it, because Bathroom has no ground truth.
///
/// TempleRing does. It ships 47 calibrated poses, so DAv3 can be asked for cameras whose right
/// answer is on disk. This measures three things, in order of how much they decide:
///
/// 1. **Absolute accuracy** - fit the recovered camera centres to the GT centres with a
///    similarity (the frame is arbitrary, so only shape is comparable) and report the residual
///    as a FRACTION of the camera spread. That is the same number the chunk fold rejects on.
/// 2. **Shape versus scale** - pairwise distance ratios. A similarity absorbs any difference in
///    size, position or orientation, and none in proportions, so ratios that agree with each
///    other mean a pure rescale and ratios that disagree mean geometry no transform can fix.
/// 3. **Batch dependence** - the question chunking actually turns on. Run two passes that SHARE
///    views but differ in the others, and compare what each says about the shared ones. If the
///    model gives a different answer for the same physical cameras depending on their company,
///    then folding passes together is limited by that and not by the fold.
///
/// Reports; does not gate. A pass/fail threshold here would be the guess this is replacing.
///
/// <c>?autotest=dav3-pose</c>, optionally <c>&amp;n=6&amp;patches=48</c>.
/// </summary>
public partial class Studio
{
    private async Task RunDav3PoseGateAsync(int n, int patchesPerSide, string dataset)
    {
        DepthEstimationService.SetSquareInput(patchesPerSide);
        Console.WriteLine(
            $"[Dav3Pose] starting dataset={dataset} n={n} patches={patchesPerSide}x{patchesPerSide}");
        try
        {
            var available = await LoadDav3PoseAvailableAsync(dataset);
            Console.WriteLine($"[Dav3Pose] {available.Count} images with ground-truth poses on disk");
            if (available.Count < n + 2)
            {
                Console.WriteLine($"[Dav3Pose] FAIL: need at least {n + 2} images");
                return;
            }

            if (!_depthService.IsReady)
                await _depthService.LoadModelAsync(DepthEstimationService.DefaultModelId);

            // Two batches sharing the first three views. Batch A is the reference; batch B keeps
            // the shared three and replaces the rest, which is exactly the chunk arrangement.
            var shared = MultiViewChunkPlan.SpreadPick(available.Count, 3);
            var others = Enumerable.Range(0, available.Count).Where(i => !shared.Contains(i)).ToArray();
            int extra = n - 3;

            var batchA = shared.Concat(others.Take(extra)).ToArray();
            var batchB = shared.Concat(others.Skip(extra).Take(extra)).ToArray();

            Console.WriteLine(
                $"[Dav3Pose] shared=[{string.Join(",", shared)}] " +
                $"A=[{string.Join(",", batchA)}] B=[{string.Join(",", batchB)}]");

            var runA = await RunDav3OnAsync(available, batchA);
            if (runA == null) { Console.WriteLine("[Dav3Pose] FAIL: batch A produced no poses"); return; }
            ReportAgainstGroundTruth("A", available, batchA, runA);

            var runB = await RunDav3OnAsync(available, batchB);
            if (runB == null) { Console.WriteLine("[Dav3Pose] FAIL: batch B produced no poses"); return; }
            ReportAgainstGroundTruth("B", available, batchB, runB);

            // 3. Batch dependence on the SHARED views, which is the chunking question.
            var aShared = shared.Select((_, k) => runA[k]).ToList();
            var bShared = shared.Select((_, k) => runB[k]).ToList();
            if (aShared.All(c => c != null) && bShared.All(c => c != null))
            {
                var src = bShared.Select(c => c!.Position).ToList();
                var dst = aShared.Select(c => c!.Position).ToList();
                if (WorldSpaceGeometry.TryUmeyamaSimilarity(
                        src, dst, out float s, out _, out _, out float rms))
                {
                    float spread = MeanSpread(dst);
                    Console.WriteLine(
                        $"[Dav3Pose] BATCH DEPENDENCE on the 3 shared views: B->A scale {s:F4}, " +
                        $"residual {rms:F4} on a spread of {spread:F4} " +
                        $"({(spread > 0 ? rms / spread : float.NaN):P1}). " +
                        "This is what a chunk fold has to live with.");
                }
                ReportTriangle("B vs A (shared views)", src, dst);
            }

            Console.WriteLine("[Dav3Pose] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dav3Pose] FAIL: {ex}");
        }
    }

    /// <summary>
    /// Images that both exist on disk and have a ground-truth pose. TempleRing uses its Middlebury
    /// par file; every other dataset uses the COLMAP-converted <c>poses.par</c> via the manifest.
    /// </summary>
    private async Task<List<(string url, string filename, CameraParams camera)>> LoadDav3PoseAvailableAsync(
        string dataset)
    {
        if (string.Equals(dataset, "TempleRing", StringComparison.OrdinalIgnoreCase))
        {
            var parText = await _http.GetStringAsync("datasets/TempleRing/templeR_par.txt");
            var gtAll = WorldSpaceGeometry.ParseMiddleburyParams(parText, 640, 480);
            Console.WriteLine($"[Dav3Pose] {gtAll.Count} ground-truth poses in templeR_par.txt");
            var available = new List<(string, string, CameraParams)>();
            foreach (var entry in gtAll)
            {
                string url = $"datasets/TempleRing/{entry.filename}";
                using var resp = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                if (resp.IsSuccessStatusCode)
                    available.Add((url, entry.filename, entry.camera));
            }
            return available;
        }

        var manifest = await _importService.TryLoadManifestAsync(dataset)
            ?? throw new InvalidOperationException($"dataset '{dataset}' has no manifest");
        if (string.IsNullOrEmpty(manifest.Poses))
            throw new InvalidOperationException($"dataset '{dataset}' ships no poses");
        var text = await _http.GetStringAsync($"datasets/{dataset}/{manifest.Poses}");
        var parsed = WorldSpaceGeometry.ParseMiddleburyParams(text, manifest.Width, manifest.Height);
        Console.WriteLine($"[Dav3Pose] {parsed.Count} ground-truth poses in {manifest.Poses}");
        var byName = parsed.ToDictionary(e => e.filename, e => e.camera, StringComparer.OrdinalIgnoreCase);
        var list = new List<(string, string, CameraParams)>();
        foreach (var name in manifest.Images)
        {
            if (!byName.TryGetValue(name, out var cam)) continue;
            string url = $"datasets/{dataset}/{manifest.ImageDir}/{name}";
            using var resp = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            if (!resp.IsSuccessStatusCode) continue;
            list.Add((url, name, cam));
        }
        return list;
    }

    /// <summary>One joint pass over the given indices; returns a camera per slot, or null.</summary>
    private async Task<CameraParams?[]?> RunDav3OnAsync(
        IReadOnlyList<(string url, string filename, CameraParams camera)> available, int[] pick)
    {
        var images = new List<ImportedImage>();
        var gtScaled = new List<CameraParams>();
        foreach (int i in pick)
        {
            var (url, filename, gtCam) = available[i];
            byte[] bytes = await _http.GetByteArrayAsync(url);
            string mime = filename.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                ? "image/png" : "image/jpeg";
            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = mime });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
            int w = (int)bitmap.Width, h = (int)bitmap.Height;
            using var osc = new OffscreenCanvas(w, h);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var dataArray = imageData.Data;
            var rgba = dataArray.ReadBytes();

            // Upright only when the calibration frame has gravity (TempleRing). DrJohnson COLMAP
            // poses are already in the photograph's orientation; production keeps them that way
            // (RecordTrainingViews poseFrameHasGravity=false). Rotating the photo here while
            // reporting against the un-rotated GT was a bookkeeping bug waiting to happen, and
            // on a walk-through the "up" from COLMAP is not a reliable gravity signal anyway.
            // GT intrinsics in the DECODED pixel grid (the manifest may describe the full capture). This
            // used to overwrite Width/Height in place first, which made the ScaledTo check below dead
            // code and left GT focal in manifest pixels - invisible while the gate compared positions only.
            var cam = available[i].camera;
            if (cam.Width != w || cam.Height != h)
                cam = cam.ScaledTo(w, h);
            bool upright = filename.StartsWith("templeR", StringComparison.OrdinalIgnoreCase);
            int turns = upright ? ImageOrientation.QuarterTurnsToUpright(cam) : 0;
            if (turns != 0)
            {
                rgba = ImageOrientation.RotateRgba(rgba, w, h, turns);
                cam = ImageOrientation.Rotate(cam, turns);
                (w, h) = (cam.Width, cam.Height);
            }
            gtScaled.Add(cam);
            images.Add(new ImportedImage { FileName = filename, Width = w, Height = h, RgbaPixels = rgba });
        }

        using var mv = await _depthService.EstimateDepthMultiViewAsync(images, maxViews: pick.Length);
        if (mv?.Extrinsics == null) return null;

        // Intrinsics against GT. Production copies K straight into the camera it unprojects depth with,
        // so a K in the wrong units is a wrong field of view for every splat of that view.
        if (mv.Intrinsics != null)
        {
            var errs = new List<float>();
            for (int slot = 0; slot < pick.Length && slot < mv.Intrinsics.Length; slot++)
            {
                var k = mv.Intrinsics[slot];
                var gt = gtScaled[slot];
                if (k == null || k.Length < 9 || gt.FocalX <= 0) continue;
                float fErr = (k[0] + k[4]) / (gt.FocalX + gt.FocalY) - 1f;
                errs.Add(MathF.Abs(fErr));
                Console.WriteLine(
                    $"[Dav3Pose]   K[{slot}] {images[slot].FileName} {images[slot].Width}x{images[slot].Height}: " +
                    $"f={k[0]:F1},{k[4]:F1} c={k[2]:F1},{k[5]:F1} vs GT f={gt.FocalX:F1},{gt.FocalY:F1} " +
                    $"c={gt.CenterX:F1},{gt.CenterY:F1} (focal {fErr:+0.0%;-0.0%})");
            }
            if (errs.Count > 0)
            {
                errs.Sort();
                Console.WriteLine(
                    $"[Dav3Pose] focal vs GROUND TRUTH ({DepthEstimationService.ResizeMode}): median |err| " +
                    $"{errs[errs.Count / 2]:P1}, worst {errs[^1]:P1} over {errs.Count} views");
            }
        }

        var cams = new CameraParams?[pick.Length];
        for (int slot = 0; slot < pick.Length && slot < mv.Extrinsics.Length; slot++)
        {
            var ext = mv.Extrinsics[slot];
            if (ext == null || ext.Length < 12) continue;
            var c = CameraParams.CreateDefault(images[slot].Width, images[slot].Height);
            // Same conversion the production path uses, so this measures what ships.
            c.Forward = new Vector3(ext[8], ext[9], ext[10]);
            c.Up = new Vector3(-ext[4], -ext[5], -ext[6]);
            c.Position = new Vector3(
                -(ext[0] * ext[3] + ext[4] * ext[7] + ext[8] * ext[11]),
                -(ext[1] * ext[3] + ext[5] * ext[7] + ext[9] * ext[11]),
                -(ext[2] * ext[3] + ext[6] * ext[7] + ext[10] * ext[11]));
            cams[slot] = c;
        }
        return cams;
    }

    private void ReportAgainstGroundTruth(
        string label, IReadOnlyList<(string url, string filename, CameraParams camera)> available,
        int[] pick, CameraParams?[] got)
    {
        var est = new CameraParams?[pick.Length];
        var reference = new CameraParams?[pick.Length];
        for (int slot = 0; slot < pick.Length; slot++)
        {
            est[slot] = got[slot];
            reference[slot] = available[pick[slot]].camera;
        }
        Console.WriteLine(
            $"[Dav3Pose] batch {label}: {est.Count(c => c != null)}/{pick.Length} views returned a pose");
        if (!WorldSpaceGeometry.TryMeasureCameraSetAccuracy(
                est, reference, out var acc, out var posFrac, out var fwdDeg))
        {
            Console.WriteLine($"[Dav3Pose] batch {label}: similarity fit failed outright");
            return;
        }

        Console.WriteLine(
            $"[Dav3Pose] batch {label} vs GROUND TRUTH: scale {acc.Scale:F4}, " +
            $"residual {acc.PositionRms:F4} on a spread of {acc.Spread:F4} " +
            $"({(acc.Spread > 0 ? acc.PositionRms / acc.Spread : float.NaN):P1} of it); " +
            $"forward median {acc.MedianForwardDeg:F1}deg p90 {acc.P90ForwardDeg:F1}deg");

        for (int i = 0; i < posFrac.Length; i++)
        {
            Console.WriteLine(
                $"[Dav3Pose]   {label}[{i}] {available[pick[i]].filename}: " +
                $"off by {posFrac[i] * acc.Spread:F4} ({posFrac[i]:P1} of spread), " +
                $"forward {fwdDeg[i]:F1}deg");
        }

        var src = new List<Vector3>();
        var dst = new List<Vector3>();
        for (int i = 0; i < pick.Length; i++)
        {
            if (got[i] == null) continue;
            src.Add(got[i]!.Position);
            dst.Add(available[pick[i]].camera.Position);
        }
        ReportTriangle($"batch {label} vs GT", src, dst);
    }

    /// <summary>
    /// Pairwise distance ratios. A similarity preserves proportions exactly, so ratios that
    /// agree mean the two point sets differ only by a rescale (which folds away perfectly) and
    /// ratios that disagree mean a shape difference that no transform can absorb.
    /// </summary>
    private static void ReportTriangle(string label, IReadOnlyList<Vector3> a, IReadOnlyList<Vector3> b)
    {
        var ratios = new List<float>();
        var parts = new List<string>();
        for (int i = 0; i < a.Count; i++)
            for (int j = i + 1; j < a.Count; j++)
            {
                float da = Vector3.Distance(a[i], a[j]);
                float db = Vector3.Distance(b[i], b[j]);
                if (!(db > 1e-6f)) continue;
                float r = da / db;
                ratios.Add(r);
                if (parts.Count < 10) parts.Add($"{i}-{j}={r:F3}");
            }
        if (ratios.Count == 0) return;
        var sorted = ratios.OrderBy(x => x).ToList();
        float median = sorted[sorted.Count / 2];
        var dev = sorted.Select(r => MathF.Abs(r - median)).OrderBy(d => d).ToList();
        float mad = median > 1e-6f ? dev[dev.Count / 2] / median : float.NaN;
        float worst = MathF.Max(MathF.Abs(sorted[^1] - median), MathF.Abs(sorted[0] - median))
                      / MathF.Max(median, 1e-6f);
        Console.WriteLine(
            $"[Dav3Pose] {label} distance ratios: {string.Join(" ", parts)} " +
            $"({ratios.Count} pairs; median {median:F3}, typical deviation {mad:P1}, " +
            $"worst pair {worst:P1})");
    }

    private static float MeanSpread(IReadOnlyList<Vector3> pts)
    {
        if (pts.Count == 0) return 0;
        var c = Vector3.Zero;
        foreach (var p in pts) c += p;
        c /= pts.Count;
        float s = 0;
        foreach (var p in pts) s += Vector3.Distance(p, c);
        return s / pts.Count;
    }
}
