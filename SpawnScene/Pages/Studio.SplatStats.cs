using System.Numerics;
using SpawnScene.Models;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>&amp;splatstats=1 (harness): after Generate, statistics of the trained splats, to compare with another
    /// trainer's PLY (tools/splat_stats.py; parity, Hamamni dark floaters 2026-10-08).</summary>
    public static bool SplatStatsOption { get; set; }

    /// <summary>
    /// Opacity, size (largest axis) and distance to the nearest training camera (both in units of the cameras' spread),
    /// and the dark opaque splats, as percentiles - the same numbers tools/splat_stats.py computes from a 3DGS PLY.
    /// CPU transfer: a diagnostic readback of the whole packed buffer, once.
    /// </summary>
    async Task LogSplatStatsAsync(GaussianScene scene)
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        int n = _gpuRenderer.SplatCount;
        if (packed == null || n <= 0 || scene.TrainingCameras.Count == 0) return;
        var rows = await packed.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
        var cams = scene.TrainingCameras.Select(c => c.Position).ToArray();
        var mean = cams.Aggregate(Vector3.Zero, (a, b) => a + b) / cams.Length;
        float spread = MathF.Sqrt(cams.Average(c => (c - mean).LengthSquared()));
        bool shDc = _gpuRenderer.ColoursAreShDc;
        var opac = new float[n]; var size = new float[n]; var aniso = new float[n]; var dist = new float[n]; var lum = new float[n];
        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var p = new Vector3(rows[o], rows[o + 1], rows[o + 2]);
            float r = rows[o + 3], g = rows[o + 4], b = rows[o + 5];
            if (shDc) { r = 0.5f + 0.28209479f * r; g = 0.5f + 0.28209479f * g; b = 0.5f + 0.28209479f * b; }
            lum[i] = 0.299f * r + 0.587f * g + 0.114f * b;
            opac[i] = rows[o + SplatFormat.OffOpacity];
            size[i] = MathF.Max(rows[o + 6], MathF.Max(rows[o + 7], rows[o + 8])) / spread;
            // Largest / smallest axis: needles smear thin repeated structure (DrJohnson's radiator, parity 2026-10-09).
            aniso[i] = MathF.Max(rows[o + 6], MathF.Max(rows[o + 7], rows[o + 8]))
                / MathF.Max(1e-12f, MathF.Min(rows[o + 6], MathF.Min(rows[o + 7], rows[o + 8])));
            float best = float.MaxValue;
            foreach (var c in cams) best = MathF.Min(best, (p - c).LengthSquared());
            dist[i] = MathF.Sqrt(best) / spread;
        }
        string P(float[] a, params double[] q)
        {
            var s = (float[])a.Clone(); Array.Sort(s);
            return string.Join(" ", q.Select(x => $"p{x * 100:0}={s[(int)Math.Min(s.Length - 1, x * (s.Length - 1))]:G3}"));
        }
        int dark = 0, darkNear = 0;
        for (int i = 0; i < n; i++)
            if (lum[i] < 0.2f && opac[i] > 0.5f) { dark++; if (dist[i] < 0.5f) darkNear++; }
        Console.WriteLine($"[Stats] {n:N0} splats, camera spread {spread:G4} (sizes and distances below are in spreads)");
        Console.WriteLine($"[Stats] opacity {P(opac, 0.1, 0.5, 0.9)}; > 0.5: {opac.Count(v => v > 0.5f) / (float)n:P1}");
        Console.WriteLine($"[Stats] size (largest axis) {P(size, 0.5, 0.9, 0.99, 0.999)}");
        Console.WriteLine($"[Stats] anisotropy (largest / smallest axis) {P(aniso, 0.5, 0.9, 0.99)}; > 10: {aniso.Count(v => v > 10f) / (float)n:P1}");
        Console.WriteLine($"[Stats] distance to nearest camera {P(dist, 0.01, 0.05, 0.5)}; < 0.25: {dist.Count(v => v < 0.25f) / (float)n:P2}, < 0.5: {dist.Count(v => v < 0.5f) / (float)n:P2}");
        Console.WriteLine($"[Stats] dark opaque (luma < 0.2, opacity > 0.5): {dark / (float)n:P2}, of them within 0.5 spreads of a camera: {darkNear:N0}");
    }
}
