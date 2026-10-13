using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Floater census gate (SplatTrainerGpu.Floaters) on a scene with a known answer: an opaque wall filling the frame, an
/// opaque box in front of part of it, and one faint splat hanging in front of the wall. The wall is the surface
/// everywhere it is seen (front share 0), the faint splat is in front of it at every pixel it touches (share 1), the box
/// is the surface at its core and in front only at its soft rim (share 0.4995, analytic). The carve must remove the
/// faint splat and nothing else. A thin splat alone in a hole in the wall (no pixel of it turns opaque, so there is no
/// surface there) must read 0 - and survive the carve. Red check, in the same run: at a 0.8 margin (in front = closer than a fifth of the
/// surface depth) the faint splat at a third of it no longer counts, so the same assertion must FAIL.
/// <para>
/// Photo depth (2026-10-13): an OPAQUE floater hangs in front of the wall too. Against the render's own surface it reads
/// ~0.5 like the box (it IS the surface at its core) and a 0.9 carve keeps it - TJ's floaters. Against a depth map that
/// holds the wall and the box (what the photo saw), it must read ~1 while wall, box and the thin splat stay ~0; the
/// agreement bins must put the wall's pixels at 0.95-1.05. Red check: the same census with the map's scale 0 (no map)
/// must give the opaque floater back its ~0.5.
/// </para>
/// <para>
/// Free space (SplatTrainerGpu.FreeSpace) on the same scene and map, colour target = the scene's own render (the colour
/// terms sit at their minimum): the opacity gradient of each floater must match central finite differences of the loss
/// as the census measures it (its front weight summed per pixel is 1 - T_f), be positive (opacity goes down), and wall,
/// box and thin splat must get none of it. Red check: at a 0.8 margin both floaters are behind the threshold and their
/// gradient must vanish. It trains: opacity alone moving, the opaque floater fades and the wall stays.
/// </para>
/// </summary>
public partial class Studio
{
    async Task<bool> FreeSpaceGateAsync(CameraParams cam, float[] packed, int n, MemoryBuffer1D<float, Stride1D.Dense> mapBuf,
        int wallCount, int box, int floater, int opaqueFloater, int thin)
    {
        var accel = _gpuService.WebGPUAccelerator;
        bool was = SplatTrainerGpu.FreeSpace;
        SplatTrainerGpu.FreeSpace = true;
        using var trainer = new SplatTrainerGpu(_gpuService);
        try { trainer.Initialize(); }
        finally { SplatTrainerGpu.FreeSpace = was; }
        await trainer.ResizeAsync(cam.Width, cam.Height, n, keysPerSplat: 64);
        using var buf = accel.Allocate1D<float>(packed.Length);
        buf.CopyFromCPU(packed);
        await accel.SynchronizeAsync();
        trainer.SeedLogits(buf, n);
        trainer.SetDepthTarget(mapBuf, 0, cam.Width, cam.Height, 1f);
        trainer.FreeSpaceRadius = 1;
        trainer.FreeSpaceMargin = 0.15f;
        trainer.FreeSpaceWeight = 0f;
        var frozen = new SplatTrainerGpu.GeometryStep(0f, 0f, 0f, 1e-9f, 1e9f);
        const float lambda = 1f;
        int pixels = cam.Width * cam.Height, F = SplatFormat.Floats, op = SplatFormat.OffOpacity;
        try
        {
            // Warm-up (the first step rewrites the rows from the trainer's own buffers), then the scene's own render as target.
            await trainer.TrainStepAsync(buf, n, cam, 0.5f, 20f, 0f, 0f, frozen, readLoss: true);
            var img = await trainer.RenderForwardAsync(buf, n, cam, 0.5f, 20f, readback: true);
            trainer.SetTarget(img);

            async Task<float[]> OpacityGradsAsync(float margin)
            {
                trainer.FreeSpaceMargin = margin;
                trainer.FreeSpaceWeight = lambda;
                await trainer.TrainStepAsync(buf, n, cam, 0.5f, 20f, 0f, 0f, frozen, readLoss: true);
                trainer.FreeSpaceWeight = 0f;
                var gr = await trainer.ReadGradientsAsync(n);
                var o = new float[n];
                for (int i = 0; i < n; i++) o[i] = gr[i * SplatTrainerGpu.GradsPerSplat + 3];
                return o;
            }
            // The loss as the census measures it: lambda x mean over pixels of the weight in front of the photo's depth.
            async Task<double> LossAsync()
            {
                trainer.ResetFloaterCensus(n);
                await trainer.AccumulateFloaterCensusAsync(buf, n, cam, 0.5f, 20f, 0.15f,
                    new SplatTrainerGpu.CensusDepth(mapBuf, 0, cam.Width, cam.Height, 1f, 1));
                var t = await trainer.ReadFloaterTotalsAsync(n);
                double front = 0;
                for (int i = 0; i < n; i++) front += t[2 * i + 1];
                return lambda * front / pixels;
            }

            var g = await OpacityGradsAsync(0.15f);
            var rows = await buf.CopyToHostAsync<float>(0, packed.Length);
            var fd = new Dictionary<int, double>();
            foreach (int i in new[] { floater, opaqueFloater })
            {
                float o0 = rows[i * F + op], eps = 0.004f;
                buf.View.SubView(i * F + op, 1).CopyFromCPU(new[] { o0 + eps });
                double lp = await LossAsync();
                buf.View.SubView(i * F + op, 1).CopyFromCPU(new[] { o0 - eps });
                double lm = await LossAsync();
                buf.View.SubView(i * F + op, 1).CopyFromCPU(new[] { o0 });
                fd[i] = (lp - lm) / (2 * eps);
            }
            await accel.SynchronizeAsync();
            float others = 0f;
            for (int i = 0; i < wallCount; i++) others = MathF.Max(others, MathF.Abs(g[i]));
            others = MathF.Max(others, MathF.Max(MathF.Abs(g[box]), MathF.Abs(g[thin])));
            float gMax = MathF.Max(g[floater], g[opaqueFloater]);
            bool close(int i) => g[i] > 0f && Math.Abs(g[i] - fd[i]) <= 0.25 * Math.Max(Math.Abs(g[i]), Math.Abs(fd[i]));
            Console.WriteLine($"[TrainerGate] free space dL/d(opacity), analytic vs finite difference of the census: faint floater " +
                $"{g[floater]:G4}/{fd[floater]:G4}, opaque floater {g[opaqueFloater]:G4}/{fd[opaqueFloater]:G4}; wall/box/thin max |g| {others:G3}");
            if (!close(floater) || !close(opaqueFloater) || !(others < 0.02f * gMax))
            {
                Console.WriteLine("[TrainerGate] FAIL: free space gradient");
                return false;
            }

            // Red check: both floaters behind the threshold (0.2 x the wall's depth = 1.2, nearer than either).
            var red = await OpacityGradsAsync(0.8f);
            Console.WriteLine($"[TrainerGate] free space red check at a 0.8 margin: faint {red[floater]:G3}, opaque {red[opaqueFloater]:G3}");
            if (MathF.Abs(red[floater]) > 0.02f * gMax || MathF.Abs(red[opaqueFloater]) > 0.02f * gMax)
            {
                Console.WriteLine("[TrainerGate] FAIL: free space red check - floaters behind the threshold still pushed");
                return false;
            }

            // It trains: opacity alone, 100 steps.
            trainer.FreeSpaceMargin = 0.15f;
            trainer.FreeSpaceWeight = lambda;
            for (int k = 0; k < 100; k++)
                await trainer.TrainStepAsync(buf, n, cam, 0.5f, 20f, 0f, 0.1f, frozen, readLoss: k == 99);
            var after = await buf.CopyToHostAsync<float>(0, packed.Length);
            float wallMin = 1f;
            for (int i = 0; i < wallCount; i++) wallMin = MathF.Min(wallMin, after[i * F + op]);
            Console.WriteLine($"[TrainerGate] free space training, opacity only, 100 steps: opaque floater {rows[opaqueFloater * F + op]:F3} -> " +
                $"{after[opaqueFloater * F + op]:F3}, faint {rows[floater * F + op]:F3} -> {after[floater * F + op]:F3}, wall min {wallMin:F3}");
            if (!(after[opaqueFloater * F + op] < 0.5f) || !(after[floater * F + op] < rows[floater * F + op]) || !(wallMin > 0.9f))
            {
                Console.WriteLine("[TrainerGate] FAIL: free space did not fade the floaters alone");
                return false;
            }
            // Cover mode: with the wall behind them the floaters' gradients are plain mode's (B ~ 1); with the wall gone
            // (nothing behind them, B = 0) cover mode must leave them alone while plain mode still pushes - B does the work.
            buf.CopyFromCPU(packed);
            await accel.SynchronizeAsync();
            trainer.SeedLogits(buf, n);
            trainer.FreeSpaceWeight = 0f;
            await trainer.TrainStepAsync(buf, n, cam, 0.5f, 20f, 0f, 0f, frozen, readLoss: true);
            trainer.SetTarget(await trainer.RenderForwardAsync(buf, n, cam, 0.5f, 20f, readback: true));
            trainer.FreeSpaceCover = false;
            var plainWall = await OpacityGradsAsync(0.15f);
            trainer.FreeSpaceCover = true;
            var coverWall = await OpacityGradsAsync(0.15f);
            var noWall = (float[])packed.Clone();
            for (int i = 0; i < wallCount; i++) noWall[i * F + op] = 0f;
            buf.CopyFromCPU(noWall);
            await accel.SynchronizeAsync();
            trainer.SeedLogits(buf, n);
            await trainer.TrainStepAsync(buf, n, cam, 0.5f, 20f, 0f, 0f, frozen, readLoss: true);
            trainer.SetTarget(await trainer.RenderForwardAsync(buf, n, cam, 0.5f, 20f, readback: true));
            var coverNoWall = await OpacityGradsAsync(0.15f);
            trainer.FreeSpaceCover = false;
            var plainNoWall = await OpacityGradsAsync(0.15f);
            bool near(float a, float b) => MathF.Abs(a - b) <= 0.05f * MathF.Max(MathF.Abs(a), MathF.Abs(b));
            Console.WriteLine($"[TrainerGate] free space cover: wall behind - faint {plainWall[floater]:G4}/{coverWall[floater]:G4}, opaque " +
                $"{plainWall[opaqueFloater]:G4}/{coverWall[opaqueFloater]:G4} (plain/cover); nothing behind - faint {plainNoWall[floater]:G4}/" +
                $"{coverNoWall[floater]:G4}, opaque {plainNoWall[opaqueFloater]:G4}/{coverNoWall[opaqueFloater]:G4}");
            if (!near(plainWall[floater], coverWall[floater]) || !near(plainWall[opaqueFloater], coverWall[opaqueFloater])
                || !(plainNoWall[floater] > 0.5f * plainWall[floater]) || !(plainNoWall[opaqueFloater] > 0.5f * plainWall[opaqueFloater])
                || MathF.Abs(coverNoWall[floater]) > 0.02f * plainWall[floater] || MathF.Abs(coverNoWall[opaqueFloater]) > 0.02f * plainWall[opaqueFloater])
            {
                Console.WriteLine("[TrainerGate] FAIL: free space cover mode");
                return false;
            }
            Console.WriteLine("[TrainerGate] free space PASS");
            return true;
        }
        finally
        {
            trainer.FreeSpaceWeight = 0f;
            trainer.FreeSpaceCover = false;
            trainer.SetDepthTarget(null);
        }
    }

    async Task<bool> FloaterCensusGateAsync(CameraParams cam)
    {
        var accel = _gpuService.WebGPUAccelerator;
        var right = cam.Right; var up = Vector3.Normalize(cam.Up); var fwd = Vector3.Normalize(cam.Forward);
        float halfW(float d) => d * 0.5f * cam.Width / cam.FocalX;
        var rows = new List<float>();
        void Add(Vector3 p, float sx, float sy, float sz, float opacity)
        {
            // Splat-local axes = world axes (identity quaternion): give the camera-facing extent to the axes that are
            // not the view axis, whichever world axis that is.
            var s = new Vector3(sx, sy, sz);
            var a = Vector3.Abs(fwd);
            if (a.X >= a.Y && a.X >= a.Z) s = new Vector3(sz, sy, sx);
            else if (a.Y >= a.X && a.Y >= a.Z) s = new Vector3(sx, sz, sy);
            rows.AddRange(new[] { p.X, p.Y, p.Z, 0.5f, 0.5f, 0.5f, s.X, s.Y, s.Z, opacity, 0f, 0f, 0f, 1f });
        }
        const float wallDepth = 6f, boxDepth = 3f, floaterDepth = 2f;
        float ww = halfW(wallDepth), wh = ww * cam.Height / cam.Width;
        const int gx = 16, gy = 12;
        // A HOLE in the wall (tiles 11-15 x 0-5, lower right of the frame, clear of the floater), filled only by one
        // thin splat: pixels that never turn opaque have no surface, so nothing on them is "in front" of anything. A
        // tile's sigma is a whole tile and its tail reaches 3 tiles, so the thin splat sits 3 tiles inside every wall
        // edge - a 3x3 hole was filled by the neighbours' tails, turned opaque, and could not fail (MEASURED, gate g3:
        // 0.0000 on the broken shader).
        static bool InHole(int i, int j) => i >= 11 && j <= 5;
        int wallCount = 0;
        for (int j = 0; j < gy; j++)
            for (int i = 0; i < gx; i++)
            {
                if (InHole(i, j)) continue;
                float x = -ww * 1.2f + 2.4f * ww * (i + 0.5f) / gx, y = -wh * 1.2f + 2.4f * wh * (j + 0.5f) / gy;
                Add(cam.Position + fwd * wallDepth + right * x + up * y, 2.4f * ww / gx, 2.4f * wh / gy, 0.01f, 0.99f);
                wallCount++;
            }
        int box = wallCount, floater = wallCount + 1;
        Add(cam.Position + fwd * boxDepth + up * (0.35f * halfW(boxDepth) * cam.Height / cam.Width), 0.12f * halfW(boxDepth),
            0.12f * halfW(boxDepth), 0.12f * halfW(boxDepth), 0.99f);
        Add(cam.Position + fwd * floaterDepth - up * (0.3f * halfW(floaterDepth) * cam.Height / cam.Width),
            0.12f * halfW(floaterDepth), 0.12f * halfW(floaterDepth), 0.12f * halfW(floaterDepth), 0.15f);
        // Hidden behind the wall: under a pixel of weight in the photo, so "unseen" - kept by the floater carve, removed
        // by the unseen carve.
        int hidden = wallCount + 2;
        Add(cam.Position + fwd * 10f, 0.01f * halfW(10f), 0.01f * halfW(10f), 0.01f * halfW(10f), 0.9f);
        // The thin splat in the hole: a surface still filling in (opacity 0.3, the photo sees through it). Its pixels never
        // reach transmittance 0.5, so it must read front share 0. 🔴 It read 1.0 until 2026-10-07: those pixels kept the
        // 3e38 "no surface" depth and every splat on them counted as in front of it - the carve then deleted every
        // still-thin wall of TJ's Bathroom (h1 vs &carve=0: supervised 29.65 vs 33.47 dB, single photos down 13 dB).
        int thin = wallCount + 3;
        float tileW = 2.4f * ww / gx, tileH = 2.4f * wh / gy;
        Add(cam.Position + fwd * wallDepth + right * (-ww * 1.2f + tileW * 13.5f) + up * (-wh * 1.2f + tileH * 2.5f),
            0.5f * tileW, 0.5f * tileH, 0.01f, 0.3f);
        // The opaque floater (photo-depth case): left of centre, nearer than the faint one, clear of box, faint splat and hole.
        int opaqueFloater = wallCount + 4;
        const float opaqueDepth = 1.5f;
        Add(cam.Position + fwd * opaqueDepth - right * (0.5f * halfW(opaqueDepth)), 0.08f * halfW(opaqueDepth),
            0.08f * halfW(opaqueDepth), 0.08f * halfW(opaqueDepth), 0.99f);
        int n = rows.Count / SplatFormat.Floats;
        var packed = rows.ToArray();

        using var trainer = new SplatTrainerGpu(_gpuService);
        trainer.Initialize();
        await trainer.ResizeAsync(cam.Width, cam.Height, n, keysPerSplat: 64);
        using var buf = accel.Allocate1D<float>(packed.Length);
        buf.CopyFromCPU(packed);
        await accel.SynchronizeAsync();

        async Task<float[]> SharesAsync(float margin)
        {
            trainer.ResetFloaterCensus(n);
            await trainer.AccumulateFloaterCensusAsync(buf, n, cam, 0.5f, 20f, margin);
            // Per-splat shares straight from the census totals (gate scale, a few hundred floats).
            var t = await trainer.ReadFloaterTotalsAsync(n);
            var share = new float[n];
            for (int i = 0; i < n; i++) share[i] = t[2 * i] > 0f ? t[2 * i + 1] / t[2 * i] : -1f;
            return share;
        }

        var s = await SharesAsync(0.1f);
        if (trainer.LastOverflowed) { Console.WriteLine("[TrainerGate] FAIL: floater census: key overflow"); return false; }
        float wallMax = 0f; int wallSeen = 0;
        for (int i = 0; i < wallCount; i++) if (s[i] >= 0f) { wallSeen++; wallMax = MathF.Max(wallMax, s[i]); }
        // The box: its rim, where its own alpha is under 0.5, lies in front of the wall's surface. For one isotropic Gaussian of peak
        // alpha 0.99 that rim (r^2 > 2 ln 1.98 sigma^2, out to the 3-sigma cutoff) carries (e^-0.683 - e^-4.5) / (1 - e^-4.5)
        // = 0.4995 of its weight (MEASURED 0.5073: pixel sampling). So an isolated opaque splat reads ~0.5 by construction,
        // and a carve must sit well above that.
        bool ok = wallSeen > wallCount / 2 && wallMax < 0.01f && s[floater] > 0.99f && MathF.Abs(s[box] - 0.4995f) < 0.05f
            && s[thin] >= 0f && s[thin] < 0.01f;
        Console.WriteLine($"[TrainerGate] floater census at 10%: wall {wallSeen}/{wallCount} seen, max front share {wallMax:F4}; " +
            $"box {s[box]:F4}; faint floater {s[floater]:F4}; thin splat with no surface behind it {s[thin]:F4}");
        if (!ok) { Console.WriteLine("[TrainerGate] FAIL: floater census shares"); return false; }

        // The carve: the faint splat's opacity to 0, every other row untouched.
        var report = await trainer.ClassifyFloatersAsync(buf, n, frontShare: 0.9f, minWeight: 1f, carve: true);
        var after = await buf.CopyToHostAsync<float>(0, packed.Length);
        int changed = 0;
        for (int i = 0; i < packed.Length; i++)
            if (i != floater * SplatFormat.Floats + SplatFormat.OffOpacity && after[i] != packed[i]) changed++;
        float floaterOpacity = after[floater * SplatFormat.Floats + SplatFormat.OffOpacity];
        Console.WriteLine($"[TrainerGate] floater carve: {report.Floaters} floater(s), its opacity {floaterOpacity}, {changed} other floats changed");
        if (report.Floaters != 1 || floaterOpacity != 0f || changed != 0)
        {
            Console.WriteLine("[TrainerGate] FAIL: floater carve");
            return false;
        }

        // The unseen carve: from the same census, the hidden splat goes too, and still nothing else but the floater.
        buf.CopyFromCPU(packed);
        await accel.SynchronizeAsync();
        var both = await trainer.ClassifyFloatersAsync(buf, n, frontShare: 0.9f, minWeight: 1f, carve: true, carveUnseen: true);
        var after2 = await buf.CopyToHostAsync<float>(0, packed.Length);
        // Expected removals from the census itself: the floater and every splat under 1 px of weight - the hidden one and
        // the wall tiles the grid runs past the frame corners (MEASURED: 2 of them).
        int op = SplatFormat.OffOpacity, F = SplatFormat.Floats, changed2 = 0;
        var totals = await trainer.ReadFloaterTotalsAsync(n);
        var expectZero = new HashSet<int> { floater };
        for (int i = 0; i < n; i++) if (totals[2 * i] < 1f) expectZero.Add(i);
        for (int i = 0; i < packed.Length; i++)
        {
            bool expected = i % F == op && expectZero.Contains(i / F);
            if (expected ? after2[i] != 0f : after2[i] != packed[i]) changed2++;
        }
        Console.WriteLine($"[TrainerGate] unseen carve: {both.Unseen} unseen, hidden opacity {after2[hidden * F + op]}, " +
            $"floater opacity {after2[floater * F + op]}, {changed2} floats not as the census says");
        if (!expectZero.Contains(hidden) || both.Unseen != expectZero.Count - 1 || changed2 != 0)
        {
            Console.WriteLine("[TrainerGate] FAIL: unseen carve");
            return false;
        }

        // Photo depth: the map the photo would have - the wall plane's depth at every pixel, the box's over its footprint
        // (3 sigma plus 2 px), no floaters. In the SHADER's camera basis (from the view matrix, which orthogonalises Up):
        // the scene above is laid out along the camera's raw Up, so the wall plane is tilted in camera space and its depth
        // varies across the frame (a constant 6 put the gate's own map 5-10% off - MEASURED, first run).
        buf.CopyFromCPU(packed);
        await accel.SynchronizeAsync();
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var sRight, out var sUp, out var sFwd, out var sPos);
        var wallNormal = Vector3.Normalize(Vector3.Cross(right, up));
        var wallPoint = cam.Position + fwd * wallDepth;
        var boxCentre = cam.Position + fwd * boxDepth + up * (0.35f * halfW(boxDepth) * cam.Height / cam.Width);
        var rel = boxCentre - sPos;
        float bz = Vector3.Dot(rel, sFwd);
        float bu = cam.CenterX + cam.FocalX * Vector3.Dot(rel, sRight) / bz, bv = cam.CenterY - cam.FocalY * Vector3.Dot(rel, sUp) / bz;
        float boxRadiusPx = 3f * cam.FocalX * 0.12f * halfW(boxDepth) / bz + 2f;
        var map = new float[cam.Width * cam.Height];
        for (int y = 0; y < cam.Height; y++)
            for (int x = 0; x < cam.Width; x++)
            {
                float dx = x + 0.5f - bu, dy = y + 0.5f - bv;
                if (dx * dx + dy * dy <= boxRadiusPx * boxRadiusPx) { map[y * cam.Width + x] = bz; continue; }
                // Camera-space ray with z = 1, into the world; its hit on the wall plane, as camera depth.
                var dir = sRight * ((x + 0.5f - cam.CenterX) / cam.FocalX) + sUp * ((cam.CenterY - (y + 0.5f)) / cam.FocalY) + sFwd;
                float tHit = Vector3.Dot(wallPoint - sPos, wallNormal) / Vector3.Dot(dir, wallNormal);
                map[y * cam.Width + x] = tHit;   // dir has unit camera z, so the parameter IS the depth
            }
        using var mapBuf = accel.Allocate1D<float>(map.Length);
        mapBuf.CopyFromCPU(map);
        await accel.SynchronizeAsync();
        async Task<float[]> DepthSharesAsync(float scale)
        {
            trainer.ResetFloaterCensus(n);
            await trainer.AccumulateFloaterCensusAsync(buf, n, cam, 0.5f, 20f, 0.15f,
                new SplatTrainerGpu.CensusDepth(mapBuf, 0, cam.Width, cam.Height, scale, 1));
            var t = await trainer.ReadFloaterTotalsAsync(n);
            var share = new float[n];
            for (int i = 0; i < n; i++) share[i] = t[2 * i] > 0f ? t[2 * i + 1] / t[2 * i] : -1f;
            return share;
        }
        var own = s[opaqueFloater];   // the render's own surface, from the first census above
        var d = await DepthSharesAsync(1f);
        var agreement = await trainer.ReadDepthAgreementAsync();
        float dWallMax = 0f;
        for (int i = 0; i < wallCount; i++) if (d[i] >= 0f) dWallMax = MathF.Max(dWallMax, d[i]);
        double agreeAll = agreement.Bins.Sum(x => (double)x), agreeOne = agreement.Bins[5] / Math.Max(1.0, agreeAll);
        Console.WriteLine($"[TrainerGate] photo depth: opaque floater {d[opaqueFloater]:F4} (render's own surface {own:F4}); " +
            $"faint floater {d[floater]:F4}; wall max {dWallMax:F4}; box {d[box]:F4}; thin splat {d[thin]:F4}; agreement {agreement}");
        if (!(d[opaqueFloater] > 0.99f && own < 0.9f && d[floater] > 0.99f && dWallMax < 0.01f && d[box] < 0.01f
              && d[thin] >= 0f && d[thin] < 0.01f && agreement.Views == 1 && agreeOne > 0.5))
        {
            Console.WriteLine("[TrainerGate] FAIL: floater census against photo depth");
            return false;
        }
        // Red check: no map (scale 0) - the opaque floater must read the render's ~0.5 again.
        var noMap = await DepthSharesAsync(0f);
        if (noMap[opaqueFloater] > 0.9f)
        {
            Console.WriteLine($"[TrainerGate] FAIL: photo depth red check - with no map the opaque floater still reads {noMap[opaqueFloater]:F4}");
            return false;
        }
        Console.WriteLine($"[TrainerGate] photo depth PASS (red check: no map, opaque floater {noMap[opaqueFloater]:F4})");
        if (!await FreeSpaceGateAsync(cam, packed, n, mapBuf, wallCount, box, floater, opaqueFloater, thin)) return false;

        // Red check: with the faint splat no longer "in front" by the margin, the same assertion must fail.
        buf.CopyFromCPU(packed);
        await accel.SynchronizeAsync();
        var red = await SharesAsync(0.8f);
        if (red[floater] > 0.99f)
        {
            Console.WriteLine($"[TrainerGate] FAIL: floater census red check - at a 0.8 margin the faint splat still reads {red[floater]:F4}");
            return false;
        }
        Console.WriteLine($"[TrainerGate] floater census PASS (red check: at a 0.8 margin the faint splat reads {red[floater]:F4})");
        return true;
    }
}
