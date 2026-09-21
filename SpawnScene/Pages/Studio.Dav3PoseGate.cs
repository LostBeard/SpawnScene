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
    private async Task RunDav3PoseGateAsync(int n, int patchesPerSide)
    {
        DepthEstimationService.SetSquareInput(patchesPerSide);
        Console.WriteLine($"[Dav3Pose] starting n={n} patches={patchesPerSide}x{patchesPerSide}");
        try
        {
            var parText = await _http.GetStringAsync("datasets/TempleRing/templeR_par.txt");
            var gtAll = WorldSpaceGeometry.ParseMiddleburyParams(parText, 640, 480);
            Console.WriteLine($"[Dav3Pose] {gtAll.Count} ground-truth poses on disk");

            // Only the images actually present.
            var available = new List<(string filename, CameraParams camera)>();
            foreach (var entry in gtAll)
            {
                using var resp = await _http.GetAsync($"datasets/TempleRing/{entry.filename}",
                    HttpCompletionOption.ResponseHeadersRead);
                if (resp.IsSuccessStatusCode) available.Add(entry);
            }
            Console.WriteLine($"[Dav3Pose] {available.Count} images on disk");
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

    /// <summary>One joint pass over the given indices; returns a camera per slot, or null.</summary>
    private async Task<CameraParams?[]?> RunDav3OnAsync(
        IReadOnlyList<(string filename, CameraParams camera)> available, int[] pick)
    {
        var images = new List<ImportedImage>();
        foreach (int i in pick)
        {
            var (filename, gtCam) = available[i];
            byte[] bytes = await _http.GetByteArrayAsync($"datasets/TempleRing/{filename}");
            using var blob = new Blob(new byte[][] { bytes }, new BlobOptions { Type = "image/png" });
            using var bitmap = await _js.CallAsync<Blob, ImageBitmap>("createImageBitmap", blob);
            int w = (int)bitmap.Width, h = (int)bitmap.Height;
            using var osc = new OffscreenCanvas(w, h);
            using var ctx = osc.Get2DContext();
            ctx.DrawImage(bitmap, 0, 0);
            using var imageData = ctx.GetImageData(0, 0, w, h);
            using var dataArray = imageData.Data;
            var rgba = dataArray.ReadBytes();

            // Upright, as the real path does: the depth model wants standing photographs, and
            // TempleRing ships gravity in its calibration so the turn is well defined here.
            var cam = available[i].camera;
            cam.Width = w; cam.Height = h;
            int turns = ImageOrientation.QuarterTurnsToUpright(cam);
            if (turns != 0)
            {
                rgba = ImageOrientation.RotateRgba(rgba, w, h, turns);
                cam = ImageOrientation.Rotate(cam, turns);
                (w, h) = (cam.Width, cam.Height);
            }
            images.Add(new ImportedImage { FileName = filename, Width = w, Height = h, RgbaPixels = rgba });
        }

        using var mv = await _depthService.EstimateDepthMultiViewAsync(images, maxViews: pick.Length);
        if (mv?.Extrinsics == null) return null;

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
        string label, IReadOnlyList<(string filename, CameraParams camera)> available,
        int[] pick, CameraParams?[] got)
    {
        var src = new List<Vector3>();
        var dst = new List<Vector3>();
        for (int slot = 0; slot < pick.Length; slot++)
        {
            if (got[slot] == null) continue;
            src.Add(got[slot]!.Position);
            dst.Add(available[pick[slot]].camera.Position);
        }
        Console.WriteLine($"[Dav3Pose] batch {label}: {src.Count}/{pick.Length} views returned a pose");
        if (src.Count < 3) return;

        if (!WorldSpaceGeometry.TryUmeyamaSimilarity(
                src, dst, out float s, out var R, out var t, out float rms))
        {
            Console.WriteLine($"[Dav3Pose] batch {label}: similarity fit failed outright");
            return;
        }

        float spread = MeanSpread(dst);
        Console.WriteLine(
            $"[Dav3Pose] batch {label} vs GROUND TRUTH: scale {s:F4}, residual {rms:F4} on a " +
            $"spread of {spread:F4} ({(spread > 0 ? rms / spread : float.NaN):P1} of it)");

        // Per-camera error after the best possible alignment: which views are wrong, not just
        // how wrong the set is on average.
        for (int i = 0; i < src.Count; i++)
        {
            var p = WorldSpaceGeometry.ApplySimilarity(src[i], s, R, t);
            Console.WriteLine(
                $"[Dav3Pose]   {label}[{i}] {available[pick[i]].filename}: " +
                $"off by {Vector3.Distance(p, dst[i]):F4} " +
                $"({(spread > 0 ? Vector3.Distance(p, dst[i]) / spread : float.NaN):P1} of spread)");
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
                ratios.Add(da / db);
                if (parts.Count < 10) parts.Add($"{i}-{j}={da / db:F3}");
            }
        if (ratios.Count < 2) return;

        // Summarise ROBUSTLY. (max-min)/mean is decided entirely by the single worst pair, which
        // on six views means one pair out of fifteen: it called TempleRing's ratios - clustered
        // between 1.59 and 1.74 - a 25.7% disagreement, and "no similarity can absorb this",
        // when the batch-to-batch fold on the same data was in fact exact to 0.4%. A statistic
        // that a lone outlier can swing is not one to put a verdict on.
        var sorted = ratios.OrderBy(r => r).ToList();
        float median = sorted[sorted.Count / 2];
        var deviations = sorted.Select(r => Math.Abs(r - median)).OrderBy(d => d).ToList();
        float mad = deviations[deviations.Count / 2];          // median absolute deviation
        float relMad = median > 1e-6f ? mad / median : float.NaN;
        float worst = Math.Max(
            Math.Abs(sorted[^1] - median), Math.Abs(sorted[0] - median)) / Math.Max(median, 1e-6f);

        Console.WriteLine(
            $"[Dav3Pose] {label} distance ratios: {string.Join(" ", parts)} " +
            $"({ratios.Count} pairs; median {median:F3}, typical deviation {relMad:P1}, " +
            $"worst pair {worst:P1})");
    }

    private static float MeanSpread(IReadOnlyList<Vector3> pts)
    {
        var c = Vector3.Zero;
        foreach (var p in pts) c += p;
        c /= pts.Count;
        float s = 0;
        foreach (var p in pts) s += Vector3.Distance(p, c);
        return s / pts.Count;
    }
}
