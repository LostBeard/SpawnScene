using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The global SfM init on DrJohnson's REAL learned matches, on the desktop (2026-10-01). Browser runs take ~25 min each;
/// this replays the app's steps - F verification, five-point relative poses, loop filter, rotation averaging, the
/// relative-pose filter, tracks from the kept pairs, global positioning - in seconds, against COLMAP's poses.
/// Data: _scratch/djmatch/k1024.bin (the app's front end - Kornia RaCo-ALIKED + LightGlue+ k1024 at 1024x672 - cached by
/// onnxruntime over all 946 pairs: keypoints at 1024x673 and each pair's matched indices) and names.txt (image order =
/// the dataset manifest's). Skipped without it.
/// </summary>
public class DrJohnsonGlobalInitTests
{
    const int W = 1024, H = 673;

    sealed record Data(int N, int K, float[] Kp, List<(int A, int B, int[] IA, int[] IB)> Pairs, List<CameraParams> Truth);

    static Data? Load()
    {
        var d = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
        while (d != null && !File.Exists(Path.Combine(d.FullName, "_scratch", "djmatch", "k1024.bin"))) d = d.Parent;
        if (d == null) return null;
        string dir = Path.Combine(d.FullName, "_scratch", "djmatch");
        var names = File.ReadAllLines(Path.Combine(dir, "names.txt"));
        using var br = new BinaryReader(File.OpenRead(Path.Combine(dir, "k1024.bin")));
        int n = br.ReadInt32(), k = br.ReadInt32();
        var kp = new float[n * k * 2];
        for (int i = 0; i < kp.Length; i++) kp[i] = br.ReadSingle();
        int np = br.ReadInt32();
        var pairs = new List<(int, int, int[], int[])>(np);
        for (int p = 0; p < np; p++)
        {
            int a = br.ReadInt32(), b = br.ReadInt32(), m = br.ReadInt32();
            var ia = new int[m]; var ib = new int[m];
            for (int i = 0; i < m; i++) ia[i] = br.ReadInt32();
            for (int i = 0; i < m; i++) ib[i] = br.ReadInt32();
            pairs.Add((a, b, ia, ib));
        }
        string posesPath = Path.Combine(d.FullName, "SpawnScene", "wwwroot", "datasets", "DrJohnson", "poses.par");
        var byName = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(posesPath), 1332, 876)
            .ToDictionary(e => e.filename, e => e.camera.ScaledTo(W, H));
        var truth = names.Select(nm => byName[nm]).ToList();
        return new Data(n, k, kp, pairs, truth);
    }

    static CameraParams Copy(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY, CenterX = c.CenterX, CenterY = c.CenterY,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    // gtRotations: position with COLMAP's rotations (diagnosis: separates rotation error from positioning).
    // trueTracks: tracks only from kept pairs within 5 deg of COLMAP's relative rotation (diagnosis: poisoned tracks?).
    // exactObs: every observation replaced by the reprojection, into the COLMAP camera, of its track triangulated from the
    // COLMAP cameras (diagnosis: observation noise/outliers vs the positioning itself).
    [TestCase(false, false, false)]
    [TestCase(true, false, false)]
    [TestCase(true, true, false)]
    [TestCase(false, true, false)]
    [TestCase(true, true, true)]
    public async Task GlobalInit_RealMatches_PlacesCamerasNearColmap(bool gtRotations, bool trueTracks, bool exactObs = false)
    {
        var data = Load();
        if (data == null) Assert.Ignore("_scratch/djmatch not present");
        var (n, k, kp, pairs, truth) = (data.N, data.K, data.Kp, data.Pairs, data.Truth);
        double focal = truth.Average(c => 0.5 * (c.FocalX + c.FocalY));
        double cx = truth[0].CenterX, cy = truth[0].CenterY;

        // 1. Verification (F-RANSAC, 2 px, >= 15) and five-point relative poses, as the app.
        var rel = new List<GlobalSfmInit.RelativePose>();
        var relMatches = new Dictionary<GlobalSfmInit.RelativePose, (int A, int B, int[] IA, int[] IB, bool[] Inl)>(ReferenceEqualityComparer.Instance);
        int verified = 0;
        foreach (var (a, b, ia, ib) in pairs)
        {
            if (ia.Length < 15) continue;
            var xa = new float[ia.Length * 2]; var xb = new float[ia.Length * 2];
            for (int i = 0; i < ia.Length; i++)
            {
                xa[i * 2] = kp[(a * k + ia[i]) * 2]; xa[i * 2 + 1] = kp[(a * k + ia[i]) * 2 + 1];
                xb[i * 2] = kp[(b * k + ib[i]) * 2]; xb[i * 2 + 1] = kp[(b * k + ib[i]) * 2 + 1];
            }
            var f = EpipolarRansac.Estimate(xa, xb, thresholdPx: 2.0, minInliers: 15, seed: a * 7919 + b);
            if (f == null) continue;
            verified++;
            var pose = GlobalSfmInit.FromMatchesCalibrated(a, b, xa, xb, focal, cx, cy, cx, cy, seed: a * 7919 + b);
            if (pose == null) continue;
            rel.Add(pose);
            relMatches[pose] = (a, b, ia, ib, f.Inliers);
        }

        // 2. Rotations and the relative-pose filter; the start poses only set the output frame (trusted = all).
        var start = truth.Select(Copy).ToList();
        var solved = GlobalSfmInit.SolveRotations(start, rel, null);
        var kept = GlobalSfmInit.FilterByRotations(rel, solved.Rot, solved.Connected);
        int trueRot = 0;
        foreach (var e in kept)
        {
            var ra = GlobalSfmInit.RotationOf(truth[e.A]); var rb = GlobalSfmInit.RotationOf(truth[e.B]);
            if (GlobalSfmInit.AngleDeg(e.R, GlobalSfmInit.Mul(rb, GlobalSfmInit.Transpose(ra))) < 5) trueRot++;
        }

        // 3. Tracks from the kept pairs' verified matches, as the app (keypoint (image, index) keys).
        var matches = new List<(int, int, int, int)>();
        foreach (var e in kept)
        {
            if (trueTracks)
            {
                var r0 = GlobalSfmInit.RotationOf(truth[e.A]); var r1 = GlobalSfmInit.RotationOf(truth[e.B]);
                if (GlobalSfmInit.AngleDeg(e.R, GlobalSfmInit.Mul(r1, GlobalSfmInit.Transpose(r0))) >= 5) continue;
            }
            var (a, b, ia, ib, inl) = relMatches[e];
            for (int i = 0; i < ia.Length; i++) if (inl[i]) matches.Add((a, ia[i], b, ib[i]));
        }
        var tracks = BundleAdjuster.BuildTracks(matches);
        var obs = new List<BundleAdjuster.Observation>();
        int points = 0;
        foreach (var t in tracks.OrderByDescending(t => t.Count))
        {
            if (points >= 15000) break;
            var seen = new HashSet<int>();
            var mine = new List<BundleAdjuster.Observation>();
            foreach (var (img, feat) in t)
                if (seen.Add(img)) mine.Add(new BundleAdjuster.Observation(img, points, kp[(img * k + feat) * 2], kp[(img * k + feat) * 2 + 1]));
            if (mine.Count < 2) continue;
            if (exactObs)
            {
                var gtObs = mine.Select(o => (o.Camera, o.U, o.V)).ToList();
                if (!BundleAdjuster.Triangulate(truth, gtObs, out var x)) continue;
                var exact = new List<BundleAdjuster.Observation>();
                foreach (var o in mine)
                    if (WorldSpaceGeometry.Project(truth[o.Camera], x, out var u, out var v, out var zc) && zc > 0)
                        exact.Add(new BundleAdjuster.Observation(o.Camera, points, u, v));
                if (exact.Count < 2) continue;
                mine = exact;
            }
            obs.AddRange(mine);
            points++;
        }

        // Diagnosis: how far the real observations are from their tracks triangulated from the COLMAP cameras.
        if (!exactObs)
        {
            var res = new List<double>(); var dx = new List<double>(); var dy = new List<double>();
            var trackBad = 0; var trackN = 0;
            foreach (var g in obs.GroupBy(o => o.Point))
            {
                var list = g.ToList();
                if (list.Count < 3) continue;
                if (!BundleAdjuster.Triangulate(truth, list.Select(o => (o.Camera, o.U, o.V)).ToList(), out var x)) continue;
                double worst = 0;
                foreach (var o in list)
                {
                    if (!WorldSpaceGeometry.Project(truth[o.Camera], x, out var u, out var v, out var zc) || zc <= 0) { worst = 1e9; continue; }
                    double d = Math.Sqrt((u - o.U) * (u - o.U) + (v - o.V) * (v - o.V));
                    res.Add(d); dx.Add(o.U - u); dy.Add(o.V - v); worst = Math.Max(worst, d);
                }
                trackN++; if (worst > 8) trackBad++;
            }
            res.Sort(); dx.Sort(); dy.Sort();
            TestContext.Out.WriteLine($"real obs vs GT-triangulated reprojection (3+ view tracks): median {res[res.Count / 2]:F2} p75 {res[res.Count * 3 / 4]:F2} " +
                $"p90 {res[res.Count * 9 / 10]:F2} px; signed dx median {dx[dx.Count / 2]:F2}, dy median {dy[dy.Count / 2]:F2}; tracks with an obs > 8 px: {trackBad}/{trackN}");
        }

        // 4. Positioning (managed; the GPU positioner equals it - GpuGlobalPositionerTests).
        foreach (var c in start) { c.FocalX = (float)focal; c.FocalY = (float)focal; }
        var rot = gtRotations ? truth.Select(GlobalSfmInit.RotationOf).ToArray() : solved.Rot;
        if (gtRotations && trueTracks && !exactObs)
            foreach (string variant in new[] { "real", "real, obs > 4 px from GT removed", "exact + 1 px noise" })
            {
                double hub = 0.003;
                var rngN = new Random(1);
                var vobs = new List<BundleAdjuster.Observation>();
                foreach (var g in obs.GroupBy(o => o.Point))
                {
                    var list = g.ToList();
                    if (variant == "real") { vobs.AddRange(list); continue; }
                    if (!BundleAdjuster.Triangulate(truth, list.Select(o => (o.Camera, o.U, o.V)).ToList(), out var x)) continue;
                    foreach (var o in list)
                    {
                        if (!WorldSpaceGeometry.Project(truth[o.Camera], x, out var u, out var v, out var zc) || zc <= 0) continue;
                        if (variant.StartsWith("exact"))
                        {
                            double G() => Math.Sqrt(-2 * Math.Log(Math.Max(rngN.NextDouble(), 1e-12))) * Math.Cos(2 * Math.PI * rngN.NextDouble());
                            vobs.Add(new BundleAdjuster.Observation(o.Camera, o.Point, u + (float)G(), v + (float)G()));
                        }
                        else if (Math.Sqrt((u - o.U) * (u - o.U) + (v - o.V) * (v - o.V)) <= 4) vobs.Add(o);
                    }
                }
                var gp = GlobalSfmInit.GlobalPositioningRobust(truth.Select(Copy).ToList(), rot, solved.Connected, vobs, points, focal, huber: hub);
                var e2 = new CameraParams?[n]; var g2 = new CameraParams?[n];
                for (int i = 0; i < n; i++) if (solved.Connected[i]) { var cc = Copy(truth[i]); cc.Position = gp.Centres[i]; e2[i] = cc; g2[i] = truth[i]; }
                WorldSpaceGeometry.TryMeasureCameraSetAccuracy(e2, g2, out var a2, out _, out _);
                TestContext.Out.WriteLine($"  {variant} ({vobs.Count} obs): median {a2.MedianPosFrac:P2}; {gp.Summary.Substring(0, Math.Min(200, gp.Summary.Length))}");
            }
        var (summary, connected) = await GlobalSfmInit.ApplyAsync(start, kept, obs, points, focal, rot, null, null, solved.Connected);
        var est = new CameraParams?[n]; var gt = new CameraParams?[n];
        for (int i = 0; i < n; i++) if (connected[i]) { est[i] = start[i]; gt[i] = truth[i]; }
        bool ok = WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, gt, out var acc, out var perView, out var perFwd);
        TestContext.Out.WriteLine($"{verified} verified, {rel.Count} relative poses, {solved.Consistent.Count} loop-consistent, " +
            $"{kept.Count} rotation-consistent ({trueRot} within 5 deg of COLMAP); {tracks.Count} tracks, {points} positioning points");
        TestContext.Out.WriteLine(summary);
        TestContext.Out.WriteLine(ok
            ? $"global init vs COLMAP: {connected.Count(c => c)} cams, median {acc.MedianPosFrac:P2}, fwd median {acc.MedianForwardDeg:F2} deg; " +
              "per view: " + string.Join(" ", Enumerable.Range(0, n).Where(i => connected[i]).Select((i, j) => $"{i}:{(perView.Length == n ? perView[i] : perView[j]):P0}"))
            : "global init vs COLMAP: could not align");
        Assert.That(ok, Is.True);
    }

    /// <summary>
    /// Track hygiene before positioning (2026-10-01). With COLMAP rotations the positioning of DrJohnson's real tracks
    /// landed 64-77% off; removing the ~6% of observations more than 4 px from their COLMAP reprojection (oracle) gave
    /// 0.29%, exact observations + 1 px noise 0.15% - a few percent of bad observations wreck it. Two non-oracle sources:
    /// union-find CONFLICTS (a track holding two features of one image; the first was kept) and tracks built from the F
    /// verification's loose 2 px inliers instead of the calibrated E-RANSAC's.
    /// </summary>
    [TestCase(false, false)]
    [TestCase(true, false)]
    [TestCase(false, true)]
    [TestCase(true, true)]
    public async Task TrackHygiene_ConflictsAndEssentialInliers(bool dropConflicts, bool essentialInliers)
    {
        var data = Load();
        if (data == null) Assert.Ignore("_scratch/djmatch not present");
        await InitAsync(data, dropConflicts, essentialInliers, log: true);
    }

    sealed record Init(List<CameraParams> Cams, bool[] Connected, List<BundleAdjuster.Observation> Obs, int Points);

    static async Task<Init> InitAsync(Data data, bool dropConflicts, bool essentialInliers, bool log)
    {
        var (n, k, kp, pairs, truth) = (data.N, data.K, data.Kp, data.Pairs, data.Truth);
        double focal = truth.Average(c => 0.5 * (c.FocalX + c.FocalY));
        double cx = truth[0].CenterX, cy = truth[0].CenterY;
        var rel = new List<GlobalSfmInit.RelativePose>();
        var relMatches = new Dictionary<GlobalSfmInit.RelativePose, (int A, int B, int[] IA, int[] IB, bool[] Inl)>(ReferenceEqualityComparer.Instance);
        foreach (var (a, b, ia, ib) in pairs)
        {
            if (ia.Length < 15) continue;
            var xa = new float[ia.Length * 2]; var xb = new float[ia.Length * 2];
            for (int i = 0; i < ia.Length; i++)
            {
                xa[i * 2] = kp[(a * k + ia[i]) * 2]; xa[i * 2 + 1] = kp[(a * k + ia[i]) * 2 + 1];
                xb[i * 2] = kp[(b * k + ib[i]) * 2]; xb[i * 2 + 1] = kp[(b * k + ib[i]) * 2 + 1];
            }
            var f = EpipolarRansac.Estimate(xa, xb, thresholdPx: 2.0, minInliers: 15, seed: a * 7919 + b);
            if (f == null) continue;
            var pose = GlobalSfmInit.FromMatchesCalibrated(a, b, xa, xb, focal, cx, cy, cx, cy, seed: a * 7919 + b);
            if (pose == null) continue;
            var inl = f.Inliers;
            if (essentialInliers)
            {
                var t = pose.T;
                var tx = new[] { 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0 };
                var e = GlobalSfmInit.Mul(tx, pose.R);
                inl = new bool[ia.Length];
                for (int i = 0; i < ia.Length; i++)
                    inl[i] = FivePoint.SampsonPx(e, ((xa[i * 2] - cx) / focal, (xa[i * 2 + 1] - cy) / focal),
                        ((xb[i * 2] - cx) / focal, (xb[i * 2 + 1] - cy) / focal), focal) <= 2.0;
            }
            rel.Add(pose);
            relMatches[pose] = (a, b, ia, ib, inl);
        }
        var start = truth.Select(Copy).ToList();
        var solved = GlobalSfmInit.SolveRotations(start, rel, null);
        var kept = GlobalSfmInit.FilterByRotations(rel, solved.Rot, solved.Connected);
        var matches = new List<(int, int, int, int)>();
        foreach (var e in kept)
        {
            var (a, b, ia, ib, inl) = relMatches[e];
            for (int i = 0; i < ia.Length; i++) if (inl[i]) matches.Add((a, ia[i], b, ib[i]));
        }
        var tracks = BundleAdjuster.BuildTracks(matches);
        int conflicts = 0;
        var obs = new List<BundleAdjuster.Observation>();
        int points = 0;
        foreach (var t in tracks.OrderByDescending(t => t.Count))
        {
            if (points >= 15000) break;
            bool conflict = t.Select(o => o.Image).Distinct().Count() != t.Count;
            if (conflict) { conflicts++; if (dropConflicts) continue; }
            var seen = new HashSet<int>();
            var mine = new List<BundleAdjuster.Observation>();
            foreach (var (img, feat) in t)
                if (seen.Add(img)) mine.Add(new BundleAdjuster.Observation(img, points, kp[(img * k + feat) * 2], kp[(img * k + feat) * 2 + 1]));
            if (mine.Count < 2) continue;
            obs.AddRange(mine);
            points++;
        }
        // Observation quality against COLMAP (diagnosis only).
        int bad = 0;
        foreach (var g in obs.GroupBy(o => o.Point))
        {
            var list = g.ToList();
            if (!BundleAdjuster.Triangulate(truth, list.Select(o => (o.Camera, o.U, o.V)).ToList(), out var x)) continue;
            foreach (var o in list)
                if (!WorldSpaceGeometry.Project(truth[o.Camera], x, out var u, out var v, out var zc) || zc <= 0
                    || Math.Sqrt((u - o.U) * (u - o.U) + (v - o.V) * (v - o.V)) > 4) bad++;
        }
        foreach (var c in start) { c.FocalX = (float)focal; c.FocalY = (float)focal; }
        var (summary, connected) = await GlobalSfmInit.ApplyAsync(start, kept, obs, points, focal, solved.Rot, null, null, solved.Connected);
        var est = new CameraParams?[n]; var gt = new CameraParams?[n];
        for (int i = 0; i < n; i++) if (connected[i]) { est[i] = start[i]; gt[i] = truth[i]; }
        bool ok = WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, gt, out var acc, out _, out _);
        if (log) TestContext.Out.WriteLine($"conflicts {(dropConflicts ? "dropped" : "kept")}, {(essentialInliers ? "E" : "F")} inliers: {tracks.Count} tracks " +
            $"({conflicts} conflicting), {obs.Count} obs ({bad} > 4 px from COLMAP = {(double)bad / obs.Count:P1}) -> " +
            (ok ? $"{connected.Count(c => c)} cams, median {acc.MedianPosFrac:P2}, fwd {acc.MedianForwardDeg:F2} deg" : "could not align"));
        return new Init(start, connected, obs, points);
    }

    /// <summary>The app's next step on the real data: triangulate from the global init and bundle-adjust (production
    /// settings, managed solver), then per-camera error against COLMAP - which cameras BA fails to fix, and why.</summary>
    [Test]
    public async Task GlobalInitThenBundleAdjust_RealMatches()
    {
        var data = Load();
        if (data == null) Assert.Ignore("_scratch/djmatch not present");
        var init = await InitAsync(data, true, true, log: true);
        int n = data.N;
        var idx = Enumerable.Range(0, n).Where(i => init.Connected[i]).ToList();
        var map = idx.Select((g, j) => (g, j)).ToDictionary(t => t.g, t => t.j);
        var cams = idx.Select(i => Copy(init.Cams[i])).ToList();
        var points = new List<Vector3>(); var baObs = new List<BundleAdjuster.Observation>();
        foreach (var g in init.Obs.GroupBy(o => o.Point))
        {
            var track = g.Where(o => map.ContainsKey(o.Camera)).Select(o => (map[o.Camera], o.U, o.V)).ToList();
            if (track.Count < 2 || !BundleAdjuster.Triangulate(cams, track, out var x)) continue;
            int id = points.Count; points.Add(x);
            foreach (var (c, u, v) in track) baObs.Add(new BundleAdjuster.Observation(c, id, u, v));
        }
        var (ba, result) = BundleAdjusterTruckScaleTests.SolveLikeProduction(cams, points, baObs);
        for (int j = 0; j < cams.Count; j++) ba.WriteCamera(j, cams[j]);
        var est = new CameraParams?[n]; var gt = new CameraParams?[n];
        for (int j = 0; j < idx.Count; j++) { est[idx[j]] = cams[j]; gt[idx[j]] = data.Truth[idx[j]]; }
        WorldSpaceGeometry.TryMeasureCameraSetAccuracy(est, gt, out var acc, out var perView, out var perFwd);
        var stats = ba.CameraStats();
        TestContext.Out.WriteLine($"BA: {points.Count} points, RMS {result.InitialRmsPixels:F1} -> {result.FinalRmsPixels:F2} px; " +
            $"{idx.Count} cams median {acc.MedianPosFrac:P2}, fwd {acc.MedianForwardDeg:F2} deg");
        for (int j = 0; j < idx.Count; j++)
        {
            double pf = perView.Length == n ? perView[idx[j]] : perView[j];
            double ff = perFwd.Length == n ? perFwd[idx[j]] : perFwd[j];
            if (pf > 0.05) TestContext.Out.WriteLine($"  view {idx[j]}: {pf:P0} off, fwd {ff:F1} deg; BA obs {stats[j].Kept}/{stats[j].Total} kept, median {stats[j].MedianError:F1} px");
        }

        // The camera graph of the solution: points two cameras both observe within 4 px.
        int m = idx.Count;
        var shared = new int[m, m];
        foreach (var g in baObs.GroupBy(o => o.Point))
        {
            var x = ba.PointAt(g.Key);
            var good = new List<int>();
            foreach (var o in g)
                if (WorldSpaceGeometry.Project(cams[o.Camera], x, out var u, out var v, out var zc) && zc > 0
                    && Math.Sqrt((u - o.U) * (u - o.U) + (v - o.V) * (v - o.V)) < 4) good.Add(o.Camera);
            for (int a = 0; a < good.Count; a++) for (int b = a + 1; b < good.Count; b++) { shared[good[a], good[b]]++; shared[good[b], good[a]]++; }
        }
        double Err(int j) => perView.Length == n ? perView[idx[j]] : perView[j];
        for (int j = 0; j < m; j++)
        {
            var links = Enumerable.Range(0, m).Where(q => q != j && shared[j, q] > 0).OrderByDescending(q => shared[j, q]).Take(4)
                .Select(q => $"{idx[q]}{(Err(q) > 0.1 ? "*" : "")}:{shared[j, q]}");
            TestContext.Out.WriteLine($"  cam {idx[j]}{(Err(j) > 0.1 ? " WRONG" : "")} ({Err(j):P0}): links {string.Join(" ", links)}");
        }
        foreach (int thr in new[] { 10, 30, 60, 100 })
        {
            var parent = Enumerable.Range(0, m).ToArray();
            int Find(int a) { while (parent[a] != a) a = parent[a] = parent[parent[a]]; return a; }
            for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) if (shared[a, b] >= thr) parent[Find(a)] = Find(b);
            var comp = Enumerable.Range(0, m).GroupBy(Find).OrderByDescending(g => g.Count()).First().ToList();
            var e3 = new CameraParams?[n]; var g3 = new CameraParams?[n];
            foreach (int j in comp) { e3[idx[j]] = cams[j]; g3[idx[j]] = data.Truth[idx[j]]; }
            bool ok3 = WorldSpaceGeometry.TryMeasureCameraSetAccuracy(e3, g3, out var a3, out var pv3, out _);
            int wrongIn = comp.Count(j => Err(j) > 0.1);
            TestContext.Out.WriteLine($"  core at >= {thr} shared points: {comp.Count} cams ({wrongIn} of them wrong in the full alignment) -> " +
                (ok3 ? $"median {a3.MedianPosFrac:P2}, p90 {pv3.Where(x => x > 0).OrderBy(x => x).ElementAtOrDefault((int)(pv3.Count(x => x > 0) * 0.9)):P1}" : "n/a"));
        }
    }
}
