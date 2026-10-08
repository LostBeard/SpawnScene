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
/// plateau depth of the side its COLOUR belongs to: the (2R+1)^2 grid around it (spaced by the upsampling factor) is split
/// at the midpoint depth into a near and a far side, and the pixel takes lo or hi by which side's mean colour it is
/// closer to - never a value from the ramp. If both sides have the same colour it is a slope, not an edge, and is left
/// alone. (A colour-weighted MEDIAN came first: ramp samples share the pixel's colour too, so it picked the ramp -
/// DepthEdgeSnapTests.) What opens behind the snapped edge is what OcclusionFill's hidden background layer is for.
/// </para>
/// </summary>
public static class DepthEdgeSnap
{
    public struct Params
    {
        public int Width, Height, Step, Radius;
        /// <summary>Squared RGB distance (channels 0..1) the two sides' mean colours need to count as an edge.</summary>
        public float MinColourSeparation;
        /// <summary>1 / (2 sigma^2) for the grid distance (in steps).</summary>
        public float SpatialFalloff;
        /// <summary>Neighbourhood depth spread, relative to the pixel's depth, below which it is left alone.</summary>
        public float MinRelRange;
        /// <summary>Most of the grid allowed in the middle half of [lo, hi] for an EDGE: an edge's depths bunch at two
        /// plateaus (only the ramp lies between), a slope's spread evenly (about half in the middle).</summary>
        public float MaxMidFraction;
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _snap;
    static Accelerator? _loadedFor;

    /// <summary>The model's input long side the depth was upsampled from (Depth Anything's 518).</summary>
    public const int ModelLongSide = 518;

    /// <summary>&amp;snapr=N: grid half-size in steps ((2N+1)^2 candidates).</summary>
    public static int GridRadius { get; set; } = 3;
    /// <summary>&amp;snapstep=X: candidate spacing as a multiple of the upsampling factor.</summary>
    public static float StepScale { get; set; } = 1f;

    /// <summary>&amp;snapmid=X: <see cref="Params.MaxMidFraction"/>. 0.3 measured 2026-10-07 on the four single-photo
    /// samples (flying pixels at depth steps, numpy port): kitchen 0.39% raw / 0.28 colour-only / 0.25 guarded, living
    /// room 0.31 / 0.25 / 0.22, castle room 0.59 / 0.32 / 0.32, garden path 2.30 / 3.71 / 2.49 (the colour-only snap
    /// terraced the textured path). 1 = no guard.</summary>
    public static float MaxMidFraction { get; set; } = 0.3f;

    /// <summary>Snap <paramref name="depth"/> (W x H) into <paramref name="output"/> using the photo's packed RGBA.</summary>
    public static void Run(Accelerator a, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output, int width, int height, float colourSigma = 0.08f)
    {
        if (!ReferenceEquals(_loadedFor, a)) { _snap = null; _loadedFor = a; }
        _snap ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, Params>(SnapKernel);
        int step = Math.Max(1, (int)MathF.Round(StepScale * Math.Max(width, height) / ModelLongSide));
        int radius = Math.Clamp(GridRadius, 1, 6);
        _snap((int)((long)width * height), depth, rgba, output, new Params
        {
            Width = width, Height = height, Step = step, Radius = radius,
            MinColourSeparation = colourSigma * colourSigma,
            SpatialFalloff = 1f / (2f * (radius * 0.75f) * (radius * 0.75f)),
            MinRelRange = 0.03f,
            MaxMidFraction = MaxMidFraction,
        });
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

        // Two sides of the edge: candidates nearer than the midpoint depth, and farther. Their mean colours (spatially
        // weighted) say which side this pixel's colour belongs to; it takes that side's plateau depth (lo or hi), never
        // a value from the ramp between. Both sides the same colour = a slope, not an edge: left alone.
        float mid = 0.5f * (lo + hi);
        float q1 = lo + 0.25f * (hi - lo), q3 = lo + 0.75f * (hi - lo);
        float nr = 0f, ng = 0f, nb = 0f, nw = 0f, fr = 0f, fg = 0f, fb = 0f, fw = 0f;
        int inMid = 0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
            {
                int sx = XMath.Clamp(x + dx * p.Step, 0, p.Width - 1), sy = XMath.Clamp(y + dy * p.Step, 0, p.Height - 1);
                int o = sy * p.Width + sx;
                int c = rgba[o];
                float w = XMath.Exp(-(dx * dx + dy * dy) * p.SpatialFalloff);
                float cr = (c & 0xFF) / 255f, cg = ((c >> 8) & 0xFF) / 255f, cb = ((c >> 16) & 0xFF) / 255f;
                float ds = depth[o];
                if (ds > q1 && ds < q3) inMid++;
                if (ds < mid) { nr += w * cr; ng += w * cg; nb += w * cb; nw += w; }
                else { fr += w * cr; fg += w * cg; fb += w * cb; fw += w; }
            }
        if (nw <= 0f || fw <= 0f) { output[i] = d0; return; }
        // Depths spread evenly through the window: a slope (textured ground passes the colour test by chance).
        if (inMid > p.MaxMidFraction * (2 * r + 1) * (2 * r + 1)) { output[i] = d0; return; }
        nr /= nw; ng /= nw; nb /= nw; fr /= fw; fg /= fw; fb /= fw;
        float sep = (nr - fr) * (nr - fr) + (ng - fg) * (ng - fg) + (nb - fb) * (nb - fb);
        if (sep < p.MinColourSeparation) { output[i] = d0; return; }
        int c0 = rgba[i];
        float pr = (c0 & 0xFF) / 255f, pg = ((c0 >> 8) & 0xFF) / 255f, pb = ((c0 >> 16) & 0xFF) / 255f;
        float toNear = (pr - nr) * (pr - nr) + (pg - ng) * (pg - ng) + (pb - nb) * (pb - nb);
        float toFar = (pr - fr) * (pr - fr) + (pg - fg) * (pg - fg) + (pb - fb) * (pb - fb);
        output[i] = toNear <= toFar ? lo : hi;
    }
}
