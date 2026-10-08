using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnScene.Services;

/// <summary>
/// Snap a single photo's depth edges to its colour edges before unprojection (TJ 2026-10-07: "single photo scenes have
/// a lot of tearing ... the depth map edges could be better aligned with the color image").
/// <para>
/// The depth model sees the photo at ~518 px and its depth is resized to the photo with a smooth filter, so every
/// object edge becomes a RAMP several photo pixels wide. The unproject kernel refuses to tilt a splat toward a neighbour
/// more than 5% deeper, but along a ramp every step is under 5%: the splats orient along it and stretch up to
/// MaxCellStretch footprints - a rubber sheet from foreground to background, the tearing a moved camera sees.
/// </para>
/// <para>
/// Here each pixel whose neighbourhood spans more than <see cref="Params.MinRelRange"/> of its depth takes the
/// COLOUR-WEIGHTED MEDIAN of the depths on a (2R+1)^2 grid around it, spaced by the upsampling factor: a ramp pixel
/// lands on the plateau whose colour it shares (joint-bilateral in spirit, but a median, so it picks a side instead of
/// averaging the two into a new ramp). Smooth regions are untouched; what opens behind the snapped edge is what
/// OcclusionFill's hidden background layer is for.
/// </para>
/// </summary>
public static class DepthEdgeSnap
{
    public struct Params
    {
        public int Width, Height, Step, Radius;
        /// <summary>1 / (2 sigma^2) for the RGB distance (channels in 0..1).</summary>
        public float ColourFalloff;
        /// <summary>1 / (2 sigma^2) for the grid distance (in steps).</summary>
        public float SpatialFalloff;
        /// <summary>Neighbourhood depth spread, relative to the pixel's depth, below which it is left alone.</summary>
        public float MinRelRange;
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _snap;
    static Accelerator? _loadedFor;

    /// <summary>The model's input long side the depth was upsampled from (Depth Anything's 518).</summary>
    public const int ModelLongSide = 518;

    /// <summary>&amp;snapr=N: grid half-size in steps ((2N+1)^2 candidates).</summary>
    public static int GridRadius { get; set; } = 2;
    /// <summary>&amp;snapstep=X: candidate spacing as a multiple of the upsampling factor.</summary>
    public static float StepScale { get; set; } = 1f;

    /// <summary>Snap <paramref name="depth"/> (W x H) into <paramref name="output"/> using the photo's packed RGBA.</summary>
    public static void Run(Accelerator a, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output, int width, int height, float colourSigma = 0.08f)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _snap = null; _loadedFor = a; }
        _snap ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, Params>(SnapKernel);
        int step = Math.Max(1, (int)MathF.Round(StepScale * Math.Max(width, height) / ModelLongSide));
        int radius = Math.Clamp(GridRadius, 1, 4);
        _snap((int)((long)width * height), depth, rgba, output, new Params
        {
            Width = width, Height = height, Step = step, Radius = radius,
            ColourFalloff = 1f / (2f * colourSigma * colourSigma),
            SpatialFalloff = 1f / (2f * (radius * 0.75f) * (radius * 0.75f)),
            MinRelRange = 0.03f,
        });
    }

    static float Weight(int c0, int c1, int dx, int dy, Params p)
    {
        float dr = ((c0 & 0xFF) - (c1 & 0xFF)) / 255f;
        float dg = (((c0 >> 8) & 0xFF) - ((c1 >> 8) & 0xFF)) / 255f;
        float db = (((c0 >> 16) & 0xFF) - ((c1 >> 16) & 0xFF)) / 255f;
        return XMath.Exp(-(dr * dr + dg * dg + db * db) * p.ColourFalloff - (dx * dx + dy * dy) * p.SpatialFalloff);
    }

    static void SnapKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output, Params p)
    {
        int x = i % p.Width, y = i / p.Width;
        float d0 = depth[i];
        int r = p.Radius;
        float lo = d0, hi = d0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                int sx = XMath.Clamp(x + dx * p.Step, 0, p.Width - 1), sy = XMath.Clamp(y + dy * p.Step, 0, p.Height - 1);
                float d = depth[sy * p.Width + sx];
                lo = XMath.Min(lo, d); hi = XMath.Max(hi, d);
            }
        if (!(d0 > 0f) || hi - lo <= p.MinRelRange * d0) { output[i] = d0; return; }

        int c0 = rgba[i];
        float total = 0f;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                int sx = XMath.Clamp(x + dx * p.Step, 0, p.Width - 1), sy = XMath.Clamp(y + dy * p.Step, 0, p.Height - 1);
                total += Weight(c0, rgba[sy * p.Width + sx], dx, dy, p);
            }

        // Weighted median: the candidate whose weighted rank sits closest to half the total.
        float best = d0, bestErr = float.MaxValue;
        for (int jy = -r; jy <= r; jy++)
            for (int jx = -r; jx <= r; jx++)
            {
                int ax = XMath.Clamp(x + jx * p.Step, 0, p.Width - 1), ay = XMath.Clamp(y + jy * p.Step, 0, p.Height - 1);
                float dj = depth[ay * p.Width + ax];
                float below = 0f;
                for (int ky = -r; ky <= r; ky++)
                    for (int kx = -r; kx <= r; kx++)
                    {
                        int bx = XMath.Clamp(x + kx * p.Step, 0, p.Width - 1), by = XMath.Clamp(y + ky * p.Step, 0, p.Height - 1);
                        float dk = depth[by * p.Width + bx];
                        float wk = Weight(c0, rgba[by * p.Width + bx], kx, ky, p);
                        below += dk < dj ? wk : dk == dj ? 0.5f * wk : 0f;
                    }
                float err = XMath.Abs(below - 0.5f * total);
                if (err < bestErr) { bestErr = err; best = dj; }
            }
        output[i] = best;
    }
}
