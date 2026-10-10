using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// &amp;blobpick=1 (diagnostic): after training, held-out views more than 4 dB under the held-out median get a 6x4 grid of
    /// pixels picked (SplatTrainerGpu.PickPixelAsync) and the heaviest splats at each logged - opacity, size, colour,
    /// distance to the nearest supervised camera and how many supervised photos had them in frame. Parity 2026-10-09:
    /// Kitchen's held-out views lose 7-16 dB to a dark blob in front of the camera on some seeds, and the camera bubble
    /// (splats AT a camera) did not remove it - this says what the blob is. CPU transfer: a row per picked splat.
    /// </summary>
    public static bool BlobPickOption { get; set; }

    private async Task BlobPickAsync(MemoryBuffer1D<float, Stride1D.Dense> packed, int n, IReadOnlyList<TrainingView> views,
        IReadOnlyList<int> supervised, MemoryBuffer1D<uint, Stride1D.Dense> targets, SplatBounds.Aabb box)
    {
        var (w, h) = _trainer!.Size;
        var held = new List<(int View, float Psnr)>();
        for (int i = 0; i < views.Count; i++)
        {
            if (views[i].UsedForSupervision) continue;
            var cam = views[i].Camera.ScaledTo(w, h);
            var (near, far) = SplatBounds.DepthRangeFor(box, cam);
            await _trainer.RenderForwardAsync(packed, n, cam, near, far, readback: false);
            var (psnr, _) = await _trainer.ScoreAgainstAsync(targets, i);
            held.Add((i, (float)psnr));
        }
        if (held.Count == 0) return;
        var sorted = held.Select(x => x.Psnr).OrderBy(x => x).ToArray();
        float median = sorted[sorted.Length / 2];
        var bad = held.Where(x => x.Psnr < median - 4f).OrderBy(x => x.Psnr).Take(3).ToList();
        Console.WriteLine($"[Blob] held-out median {median:F2} dB; {bad.Count} view(s) more than 4 dB under it: " +
            string.Join(", ", bad.Select(b => $"{System.IO.Path.GetFileNameWithoutExtension(views[b.View].ImageName)} {b.Psnr:F2}")));
        var sup = supervised.Select(vi => views[vi].Camera.ScaledTo(w, h)).ToList();
        var centres = sup.Select(c => c.Position).ToArray();
        var mean = centres.Aggregate(Vector3.Zero, (a, b) => a + b) / centres.Length;
        float spread = MathF.Sqrt(centres.Average(c => (c - mean).LengthSquared()));
        var I = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var (vi, psnr) in bad)
        {
            var cam = views[vi].Camera.ScaledTo(w, h);
            var (near, far) = SplatBounds.DepthRangeFor(box, cam);
            string tag = System.IO.Path.GetFileNameWithoutExtension(views[vi].ImageName);
            for (int gy = 0; gy < 4; gy++)
                for (int gx = 0; gx < 6; gx++)
                {
                    int x = (int)((gx + 0.5f) * w / 6f), y = (int)((gy + 0.5f) * h / 4f);
                    var hits = await _trainer.PickPixelAsync(packed, n, cam, near, far, x, y, minWeight: 0.1f);
                    foreach (var hit in hits.Take(2))
                    {
                        var row = await packed.CopyToHostAsync<float>((long)hit.Index * SplatFormat.Floats, SplatFormat.Floats);
                        var pos = new Vector3(row[0], row[1], row[2]);
                        float nearest = centres.Min(c => Vector3.Distance(c, pos)) / spread;
                        int inFrame = 0;
                        foreach (var c in sup)
                        {
                            var f = Vector3.Normalize(c.Forward); var u = Vector3.Normalize(c.Up); var r = Vector3.Cross(f, u);
                            var rel = pos - c.Position; float z = Vector3.Dot(rel, f);
                            if (z <= SplatTrainerGpu.NearPlane) continue;
                            float px = c.FocalX * Vector3.Dot(rel, r) / z + c.CenterX, py = c.CenterY - c.FocalY * Vector3.Dot(rel, u) / z;
                            if (px >= 0 && py >= 0 && px < c.Width && py < c.Height) inFrame++;
                        }
                        float size = MathF.Max(row[6], MathF.Max(row[7], row[8])) / spread;
                        Console.WriteLine(string.Format(I,
                            "[Blob] {0} {1:F1} dB ({2},{3}) #{4} w {5:F2} a {6:F2} depth {7:F3} | opacity {8:F3} size {9:G3} spreads " +
                            "colour ({10:F2},{11:F2},{12:F2}) | nearest supervised camera {13:F3} spreads, in frame of {14}/{15}",
                            tag, psnr, x, y, hit.Index, hit.Weight, hit.Alpha, hit.Depth, row[SplatFormat.OffOpacity], size,
                            row[3], row[4], row[5], nearest, inFrame, sup.Count));
                    }
                }
        }
    }
}
