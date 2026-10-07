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
/// faint splat and nothing else. Red check, in the same run: at a 0.8 margin (in front = closer than a fifth of the
/// surface depth) the faint splat at a third of it no longer counts, so the same assertion must FAIL.
/// </summary>
public partial class Studio
{
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
        for (int j = 0; j < gy; j++)
            for (int i = 0; i < gx; i++)
            {
                float x = -ww * 1.2f + 2.4f * ww * (i + 0.5f) / gx, y = -wh * 1.2f + 2.4f * wh * (j + 0.5f) / gy;
                Add(cam.Position + fwd * wallDepth + right * x + up * y, 2.4f * ww / gx, 2.4f * wh / gy, 0.01f, 0.99f);
            }
        int wallCount = gx * gy;
        int box = wallCount, floater = wallCount + 1;
        Add(cam.Position + fwd * boxDepth + up * (0.35f * halfW(boxDepth) * cam.Height / cam.Width), 0.12f * halfW(boxDepth),
            0.12f * halfW(boxDepth), 0.12f * halfW(boxDepth), 0.99f);
        Add(cam.Position + fwd * floaterDepth - up * (0.3f * halfW(floaterDepth) * cam.Height / cam.Width),
            0.12f * halfW(floaterDepth), 0.12f * halfW(floaterDepth), 0.12f * halfW(floaterDepth), 0.15f);
        // Hidden behind the wall: under a pixel of weight in the photo, so "unseen" - kept by the floater carve, removed
        // by the unseen carve.
        int hidden = wallCount + 2;
        Add(cam.Position + fwd * 10f, 0.01f * halfW(10f), 0.01f * halfW(10f), 0.01f * halfW(10f), 0.9f);
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
        bool ok = wallSeen > wallCount / 2 && wallMax < 0.01f && s[floater] > 0.99f && MathF.Abs(s[box] - 0.4995f) < 0.05f;
        Console.WriteLine($"[TrainerGate] floater census at 10%: wall {wallSeen}/{wallCount} seen, max front share {wallMax:F4}; " +
            $"box {s[box]:F4}; faint floater {s[floater]:F4}");
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
