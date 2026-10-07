using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;
using SpawnDev.ILGPU;

namespace SpawnScene.Pages;

// Off-path views of a trained scene: where a user goes the moment they move the camera.
public partial class Studio
{
    /// <summary>
    /// Park the viewer at poses NO photo was taken from - moved in toward the subject, raised, lowered, pulled back, and
    /// between two photos - and announce each for capture. Every quality check before this parked at a photo's own
    /// pose (or one held out between two of them), which is exactly where a 3DGS scene looks right: TJ's Bicycle at
    /// the home view "looks fine", and the moment the camera moves "there are floaters EVERYWHERE" (2026-10-07).
    /// </summary>
    /// <remarks>
    /// Each pose is logged as a TURN-POSE line (position, forward, up, intrinsics, size) so a reference trainer can
    /// render the identical view (gsplat render_turns.py), with three photo poses parked first as "view=sup-N" lines -
    /// N the photo's index by sorted name - to fit SpawnScene's frame to the reference's. The subject is the point
    /// nearest every camera's optical axis (least squares), which is the orbit centre of an inward capture; for a
    /// capture whose axes do not meet in front of the cameras it falls back to the median ray depth along each axis.
    /// Intrinsics stay the first photo's.
    /// </remarks>
    private async Task CaptureWanderViewsAsync(GaussianScene scene)
    {
        var cams = scene.TrainingCameras;
        if (cams.Count < 4) { Console.WriteLine("[Wander] skipped: fewer than 4 cameras"); return; }

        // Frame-fit anchors: photo poses with their sorted-name index.
        var names = scene.TrainingViews.Select(v => v.ImageName).ToList();
        var sorted = names.OrderBy(n => n, StringComparer.Ordinal).ToList();
        foreach (int i in new[] { 0, cams.Count / 3, 2 * cams.Count / 3 })
        {
            var tv = scene.TrainingViews[i];
            await ParkOnGroundTruthPoseAsync($"sup-{sorted.IndexOf(tv.ImageName)}", tv.Camera);
        }

        // -- The subject: least-squares point nearest every optical axis --
        var A = new double[3, 3]; var b = new double[3];
        var upSum = Vector3.Zero;
        foreach (var c in cams)
        {
            var d = Vector3.Normalize(c.Forward);
            upSum += Vector3.Normalize(c.Up);
            // (I - d d^T) p = (I - d d^T) o, summed over cameras.
            for (int r = 0; r < 3; r++)
            {
                for (int k = 0; k < 3; k++)
                    A[r, k] += (r == k ? 1.0 : 0.0) - Comp(d, r) * Comp(d, k);
                double proj = 0;
                for (int k = 0; k < 3; k++) proj += ((r == k ? 1.0 : 0.0) - Comp(d, r) * Comp(d, k)) * Comp(c.Position, k);
                b[r] += proj;
            }
        }
        var up = Vector3.Normalize(upSum);
        Vector3? lsq = Solve3(A, b);
        int inFront = 0;
        if (lsq is Vector3 p0)
            foreach (var c in cams) if (Vector3.Dot(p0 - c.Position, c.Forward) > 0) inFront++;
        Vector3 subject;
        string how;
        if (lsq is Vector3 p && inFront >= cams.Count * 3 / 4)
        {
            subject = p; how = $"axes meet ({inFront}/{cams.Count} cameras face it)";
        }
        else
        {
            // Forward-facing capture: the median camera-to-centroid distance along the first camera's axis.
            var centroid = Vector3.Zero;
            foreach (var c in cams) centroid += c.Position;
            centroid /= cams.Count;
            var dists = cams.Select(c => Vector3.Distance(c.Position, centroid)).OrderBy(x => x).ToList();
            float reach = MathF.Max(dists[dists.Count / 2] * 2f, 1e-3f);
            subject = cams[0].Position + Vector3.Normalize(cams[0].Forward) * reach;
            how = $"axes do not meet in front ({inFront}/{cams.Count}); {reach:F3} along photo 1's axis";
        }
        Console.WriteLine($"[Wander] subject ({subject.X:F3},{subject.Y:F3},{subject.Z:F3}): {how}; up ({up.X:F3},{up.Y:F3},{up.Z:F3})");

        var seat = cams[0];
        var poses = new List<(string Name, Vector3 Pos)>();
        // Four photos spread along the capture path.
        for (int q = 0; q < 4; q++)
        {
            var c = cams[q * cams.Count / 4].Position;
            var toSubject = subject - c;
            float dist = toSubject.Length();
            poses.Add(($"in-{q}", c + toSubject * 0.5f));                 // half way to the subject
            poses.Add(($"up-{q}", c + up * (0.5f * dist)));               // raised, looking down at it
            poses.Add(($"low-{q}", c - up * (0.2f * dist)));              // lowered
            poses.Add(($"out-{q}", subject - toSubject * 1.5f));          // pulled back
            var next = cams[(q * cams.Count / 4 + Math.Max(1, cams.Count / 16)) % cams.Count].Position;
            poses.Add(($"mid-{q}", (c + next) * 0.5f));                   // between two photos
        }

        // Standing among the cameras and turning around: eight headings in the horizontal plane from the rig centroid.
        // This is what a user does inside a ROOM capture (TJ's Bathroom, 2026-10-07: "TONS of holes"), and the subject
        // poses above never look away from the subject. Each heading also logs how many photos look that way.
        var rigCentre = Vector3.Zero;
        foreach (var c in cams) rigCentre += c.Position;
        rigCentre /= cams.Count;
        var flat0 = Vector3.Normalize(cams[0].Forward) - up * Vector3.Dot(Vector3.Normalize(cams[0].Forward), up);
        if (flat0.LengthSquared() < 1e-6f) flat0 = Vector3.Cross(up, Vector3.UnitX);
        flat0 = Vector3.Normalize(flat0);
        var panDirs = new Dictionary<string, Vector3>();
        for (int k = 0; k < 8; k++)
        {
            var dir = Vector3.Transform(flat0, Quaternion.CreateFromAxisAngle(up, k * MathF.PI / 4f));
            panDirs[$"pan-{k}"] = dir;
            int facing = cams.Count(c => Vector3.Dot(Vector3.Normalize(c.Forward), dir) > MathF.Cos(MathF.PI / 6f));
            Console.WriteLine($"[Wander] pan-{k}: heading {k * 45} deg from photo 1's, {facing}/{cams.Count} photos look within 30 deg of it");
            poses.Add(($"pan-{k}", rigCentre));
        }
        // And from outside it, looking back at the rig: the whole reconstruction's shape and its coverage at once.
        float reachOut = MathF.Max(Vector3.Distance(subject, cams[0].Position), 1e-3f);
        poses.Add(("over-0", rigCentre + up * (2.5f * reachOut) - flat0 * (1.5f * reachOut)));
        poses.Add(("over-1", rigCentre - flat0 * (3f * reachOut) + up * reachOut));

        static string F(params float[] v) => string.Join(" ",
            v.Select(x => x.ToString("R", System.Globalization.CultureInfo.InvariantCulture)));
        foreach (var (name, pos) in poses)
        {
            var fwd = panDirs.TryGetValue(name, out var pd) ? pd
                : Vector3.Normalize((name.StartsWith("over-") ? rigCentre : subject) - pos);
            var right = Vector3.Cross(fwd, up);
            if (right.LengthSquared() < 1e-8f) continue;   // looking straight along up: no roll-free frame
            var camUp = Vector3.Normalize(Vector3.Cross(Vector3.Normalize(right), fwd));
            var view = seat.ScaledTo(seat.Width, seat.Height);
            view.Position = pos; view.Forward = fwd; view.Up = camUp;
            string tag = $"wander{name}";   // "wanderin-0": the harness matches view-(\w+-\d+)
            await ParkOnGroundTruthPoseAsync(tag, view);
            Console.WriteLine(
                $"[Dataset] TURN-POSE {tag} - pos {F(pos.X, pos.Y, pos.Z)} fwd {F(fwd.X, fwd.Y, fwd.Z)} " +
                $"up {F(camUp.X, camUp.Y, camUp.Z)} K {F(view.FocalX, view.FocalY, view.CenterX, view.CenterY)} " +
                $"size {view.Width} {view.Height}");
            Console.WriteLine($"[Dataset] READY-FOR-CAPTURE view-{tag} - {view.Width}x{view.Height} turns=0");
            await Task.Delay(1800);
            if (WanderPicks.TryGetValue(tag, out var pts)) await PickWanderPixelsAsync(tag, view, pts, cams);
        }
    }

    /// <summary>&amp;pick=view@x,y;view@x,y: wander pixels to explain (which splats paint them). Diagnostic.</summary>
    internal static Dictionary<string, List<(int X, int Y)>> WanderPicks { get; } = new();

    internal static void ParseWanderPicks(string q)
    {
        foreach (var item in q.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var at = item.Split('@');
            var xy = at.Length == 2 ? at[1].Split(',') : Array.Empty<string>();
            if (xy.Length != 2 || !int.TryParse(xy[0], out int x) || !int.TryParse(xy[1], out int y)) continue;
            if (!WanderPicks.TryGetValue(at[0], out var l)) WanderPicks[at[0]] = l = new();
            l.Add((x, y));
        }
    }

    /// <summary>
    /// Print the splats that paint each picked pixel of a wander view, with what decides whether they are floaters:
    /// opacity, scales, distance to the nearest photo camera (and how that compares with the trainer's 0.2 near plane),
    /// and the census (share of weight in front of the photos' surfaces, total weight over the photos). CPU transfer: a
    /// row and two census floats per picked splat - a diagnostic, a few dozen splats a run.
    /// </summary>
    private async Task PickWanderPixelsAsync(string tag, CameraParams view, List<(int X, int Y)> pts, IReadOnlyList<CameraParams> cams)
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (_trainer == null || packed == null || n <= 0) { Console.WriteLine($"[Pick] {tag}: no trainer/scene"); return; }
        var aabb = await SplatBounds.ComputeAsync(_gpuService.WebGPUAccelerator, packed, n);
        if (aabb == null) return;
        var (near, far) = SplatBounds.DepthRangeFor(aabb.Value, view);
        var I = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var (x, y) in pts)
        {
            var hits = await _trainer.PickPixelAsync(packed, n, view, near, far, x, y);
            Console.WriteLine($"[Pick] {tag} ({x},{y}): {hits.Length} splats with weight >= 0.02");
            foreach (var h in hits)
            {
                var row = await packed.CopyToHostAsync<float>((long)h.Index * SplatFormat.Floats, SplatFormat.Floats);
                var (total, front) = await _trainer.ReadFloaterTotalsOfAsync(h.Index);
                var pos = new Vector3(row[0], row[1], row[2]);
                float nearest = cams.Min(c => Vector3.Distance(c.Position, pos));
                // How many photos had this splat IN FRAME (centre projects inside the image, past the 0.2 near plane),
                // and how deep inside the nearest frame edge it sat at best (fraction of the half-size: 0 = on the edge).
                int inFrame = 0; float bestInside = -1f;
                foreach (var c in cams)
                {
                    var f = Vector3.Normalize(c.Forward); var u = Vector3.Normalize(c.Up); var r = Vector3.Cross(f, u);
                    var rel = pos - c.Position; float z = Vector3.Dot(rel, f);
                    if (z <= 0.2f) continue;
                    float px = c.FocalX * Vector3.Dot(rel, r) / z + c.CenterX, py = c.CenterY - c.FocalY * Vector3.Dot(rel, u) / z;
                    if (px < 0 || py < 0 || px >= c.Width || py >= c.Height) continue;
                    inFrame++;
                    float inside = MathF.Min(MathF.Min(px, c.Width - px) / (0.5f * c.Width), MathF.Min(py, c.Height - py) / (0.5f * c.Height));
                    bestInside = MathF.Max(bestInside, inside);
                }
                Console.WriteLine(string.Format(I,
                    "[Pick]   #{0} w {1:F3} a {2:F3} depth {3:F3} | opacity {4:F3} scale {5:G3},{6:G3},{7:G3} pos ({8:F3},{9:F3},{10:F3}) " +
                    "nearest cam {11:F3} in frame of {12} photos, best {13:P0} inside the edge | census total {14:F2} px front {15:P0}",
                    h.Index, h.Weight, h.Alpha, h.Depth, row[9], row[6], row[7], row[8], pos.X, pos.Y, pos.Z,
                    nearest, inFrame, bestInside, total, total > 0 ? front / total : float.NaN));
            }
        }
    }

    static double Comp(Vector3 v, int i) => i == 0 ? v.X : i == 1 ? v.Y : v.Z;

    /// <summary>Solve a 3x3 system by Cramer's rule; null when it is (near) singular.</summary>
    static Vector3? Solve3(double[,] m, double[] r)
    {
        double Det(double[,] a) =>
            a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
            - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
            + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]);
        double det = Det(m);
        double scale = 0; foreach (var x in m) scale = Math.Max(scale, Math.Abs(x));
        if (Math.Abs(det) < 1e-9 * scale * scale * scale) return null;
        var res = new double[3];
        for (int c = 0; c < 3; c++)
        {
            var t = (double[,])m.Clone();
            for (int row = 0; row < 3; row++) t[row, c] = r[row];
            res[c] = Det(t) / det;
        }
        return new Vector3((float)res[0], (float)res[1], (float)res[2]);
    }
}
