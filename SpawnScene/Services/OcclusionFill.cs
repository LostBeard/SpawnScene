using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// What a single photo never saw, filled so a moving camera finds scene instead of holes (MEASURED 2026-10-06, the Room
/// sample: 25-39% of a view moved 10-30% of the scene depth was empty). Two hidden layers, the way layered-depth "3D
/// photo" methods build them:
/// <list type="number">
/// <item><b>Behind depth edges.</b> The farthest surface near each grid cell comes from a separable max filter of depth
/// (<see cref="Params.Radius"/> cells each way, about the parallax a 20% camera move opens). Cells that ARE that far
/// surface seed a push-pull pyramid that spreads their colour and depth over the grid (coarse-to-fine, bilinear, so a gap
/// takes the background around it, not a smear of one edge pixel). Where that background lies well behind a cell's own
/// depth, one more splat goes there - behind the foreground, so the photo's own view is unchanged.</item>
/// <item><b>Past the frame.</b> The grid carries a margin of <see cref="Params.Margin"/> cells on every side; a second
/// pyramid, seeded by every cell, continues the photo's edges into it (soft, like a blurred outpaint), so turning the
/// camera shows scene, not the clear colour.</item>
/// </list>
/// Everything stays on the device; the splats are appended through the caller's compaction counter.
/// </summary>
public static class OcclusionFill
{
    const int Floats = SplatFormat.Floats;
    const int Ch = 5;   // r, g, b, depth, weight per pyramid cell

    public struct Params
    {
        public int Width, Height, Subsample, GridW, GridH, Radius, Margin, Capacity;
        public float FocalX, FocalY, CenterX, CenterY, DepthScale, Tau, BackgroundBand, SizeFactor, Opacity;
        /// <summary>0: seed the pyramid with background cells only (behind edges); 1: with every cell (past the frame).</summary>
        public int SeedAll;
        /// <summary>Past the frame, one splat per this many cells each way (that many times bigger): the continuation is a
        /// blur of the edges anyway, and at full density the margin cost more splats than the photo.</summary>
        public int BorderStride;
    }

    public struct LevelParams
    {
        public int FineW, FineH, CoarseW, CoarseH;
        /// <summary>1: a coarse cell keeps only its children near the farthest of them (the behind-edges pyramid), so
        /// coarse levels hold the background rather than a blend of layers; 0: plain averages (past the frame).</summary>
        public int FarPriority;
        public float Band;
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _rowMax;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _colMax;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _rowMin;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _colMin;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>? _level0;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LevelParams>? _down;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int>? _normalize;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LevelParams>? _up;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>? _emitBehind;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, Params, int>? _emitBorder;
    static Accelerator? _loadedFor;

    static void Load(Accelerator a)
    {
        if (!ReferenceEquals(_loadedFor, a))
        {
            _rowMax = null; _colMax = null; _rowMin = null; _colMin = null; _level0 = null; _down = null; _normalize = null; _up = null; _emitBehind = null; _emitBorder = null;
        }
        _loadedFor = a;
        _rowMax ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(RowMaxKernel);
        _colMax ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(ColMaxKernel);
        _rowMin ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(RowMinKernel);
        _colMin ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(ColMinKernel);
        _level0 ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, Params>(Level0Kernel);
        _down ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LevelParams>(DownKernel);
        _normalize ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int>(NormalizeKernel);
        _up ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, LevelParams>(UpKernel);
        _emitBehind ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>(EmitBehindKernel);
        _emitBorder ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params, int>(EmitBorderKernel);
    }

    /// <summary>Splats <see cref="Append"/> may add at most, beyond the surface layer: one per cell behind edges, one per margin cell.</summary>
    public static int ExtraCapacity(int gridW, int gridH, int margin, int borderStride)
        => gridW * gridH + 2 * (((gridW + 2 * margin) * (gridH + 2 * margin) - gridW * gridH) / Math.Max(1, borderStride * borderStride)
           + 4 * (gridW + gridH + 4 * margin));   // two border layers; stride rows / columns that straddle the frame edge

    /// <summary>
    /// Append both hidden layers to <paramref name="outPacked"/> after the <paramref name="counter"/> splats already there,
    /// up to <see cref="Params.Capacity"/> in all. The counter is the new total.
    /// </summary>
    public static void Append(Accelerator a, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> outPacked, ArrayView1D<int, Stride1D.Dense> counter, Params p,
        List<MemoryBuffer1D<float, Stride1D.Dense>> scratch)
    {
        Load(a);
        int cells = p.GridW * p.GridH;
        var rowMax = a.Allocate1D<float>(cells); scratch.Add(rowMax);
        var bgMax = a.Allocate1D<float>(cells); scratch.Add(bgMax);
        _rowMax!(cells, depth, rowMax.View, p);
        _colMax!(cells, rowMax.View, bgMax.View, p);
        var rowMin = a.Allocate1D<float>(cells); scratch.Add(rowMin);
        var bgMin = a.Allocate1D<float>(cells); scratch.Add(bgMin);
        _rowMin!(cells, depth, rowMin.View, p);
        _colMin!(cells, rowMin.View, bgMin.View, p);

        var behind = p; behind.SeedAll = 0;
        var l0Behind = Pyramid(a, depth, rgba, bgMax, bgMin, behind, scratch);
        _emitBehind!(cells, depth, bgMax.View, l0Behind.View, outPacked, counter, behind);
        if (p.Margin > 0)
        {
            var border = p; border.SeedAll = 1;
            var l0Border = Pyramid(a, depth, rgba, bgMax, bgMin, border, scratch);
            int padded = (p.GridW + 2 * p.Margin) * (p.GridH + 2 * p.Margin);
            _emitBorder!(padded, l0Border.View, l0Behind.View, outPacked, counter, border, 0);
            // And the background past the frame, where it lies behind that continuation (a plant in the corner of the
            // photo is continued at its own depth; turning past it needs the wall behind it too).
            _emitBorder!(padded, l0Border.View, l0Behind.View, outPacked, counter, border, 1);
        }
    }

    /// <summary>The filled level 0 (padded grid) of a push-pull pyramid seeded as <paramref name="p"/>.SeedAll says.</summary>
    static MemoryBuffer1D<float, Stride1D.Dense> Pyramid(Accelerator a, ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba, MemoryBuffer1D<float, Stride1D.Dense> bgMax, MemoryBuffer1D<float, Stride1D.Dense> bgMin,
        Params p, List<MemoryBuffer1D<float, Stride1D.Dense>> scratch)
    {
        int lw = p.GridW + 2 * p.Margin, lh = p.GridH + 2 * p.Margin;
        var levels = new List<(MemoryBuffer1D<float, Stride1D.Dense> Buf, int W, int H)>();
        var l0 = a.Allocate1D<float>((long)lw * lh * Ch); scratch.Add(l0);
        levels.Add((l0, lw, lh));
        _level0!(lw * lh, depth, rgba, bgMax.View, bgMin.View, l0.View, p);
        while (lw > 1 || lh > 1)
        {
            int cw = (lw + 1) / 2, chh = (lh + 1) / 2;
            var c = a.Allocate1D<float>((long)cw * chh * Ch); scratch.Add(c);
            _down!(cw * chh, levels[^1].Buf.View, c.View, new LevelParams
            {
                FineW = lw, FineH = lh, CoarseW = cw, CoarseH = chh, FarPriority = p.SeedAll == 0 ? 1 : 0, Band = p.BackgroundBand,
            });
            levels.Add((c, cw, chh));
            lw = cw; lh = chh;
        }
        _normalize!(levels[^1].W * levels[^1].H, levels[^1].Buf.View, levels[^1].W * levels[^1].H);
        for (int l = levels.Count - 2; l >= 0; l--)
        {
            var lp = new LevelParams { FineW = levels[l].W, FineH = levels[l].H, CoarseW = levels[l + 1].W, CoarseH = levels[l + 1].H };
            _up!(levels[l].W * levels[l].H, levels[l].Buf.View, levels[l + 1].Buf.View, lp);
        }
        return l0;
    }

    // ---------------------------------------------------------------- kernels

    static float DepthAt(ArrayView1D<float, Stride1D.Dense> depth, Params p, int gx, int gy)
    {
        float d = depth[gy * p.Subsample * p.Width + gx * p.Subsample] * p.DepthScale;
        return d > 0.01f ? d : 0f;
    }

    static void RowMaxKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<float, Stride1D.Dense> rowMax, Params p)
    {
        if (i >= p.GridW * p.GridH) return;
        int gx = i % p.GridW, gy = i / p.GridW;
        int x0 = gx - p.Radius < 0 ? 0 : gx - p.Radius;
        int x1 = gx + p.Radius >= p.GridW ? p.GridW - 1 : gx + p.Radius;
        float m = 0f;
        for (int x = x0; x <= x1; x++) m = XMath.Max(m, DepthAt(depth, p, x, gy));
        rowMax[i] = m;
    }

    static void RowMinKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<float, Stride1D.Dense> rowMin, Params p)
    {
        if (i >= p.GridW * p.GridH) return;
        int gx = i % p.GridW, gy = i / p.GridW;
        int x0 = gx - p.Radius < 0 ? 0 : gx - p.Radius;
        int x1 = gx + p.Radius >= p.GridW ? p.GridW - 1 : gx + p.Radius;
        float m = 3.0e38f;
        for (int x = x0; x <= x1; x++)
        {
            float d = DepthAt(depth, p, x, gy);
            if (d > 0f) m = XMath.Min(m, d);
        }
        rowMin[i] = m;
    }

    static void ColMinKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> rowMin, ArrayView1D<float, Stride1D.Dense> bgMin, Params p)
    {
        if (i >= p.GridW * p.GridH) return;
        int gx = i % p.GridW, gy = i / p.GridW;
        int y0 = gy - p.Radius < 0 ? 0 : gy - p.Radius;
        int y1 = gy + p.Radius >= p.GridH ? p.GridH - 1 : gy + p.Radius;
        float m = 3.0e38f;
        for (int y = y0; y <= y1; y++) m = XMath.Min(m, rowMin[y * p.GridW + gx]);
        bgMin[i] = m;
    }

    static void ColMaxKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> rowMax, ArrayView1D<float, Stride1D.Dense> bgMax, Params p)
    {
        if (i >= p.GridW * p.GridH) return;
        int gx = i % p.GridW, gy = i / p.GridW;
        int y0 = gy - p.Radius < 0 ? 0 : gy - p.Radius;
        int y1 = gy + p.Radius >= p.GridH ? p.GridH - 1 : gy + p.Radius;
        float m = 0f;
        for (int y = y0; y <= y1; y++) m = XMath.Max(m, rowMax[y * p.GridW + gx]);
        bgMax[i] = m;
    }

    /// <summary>
    /// The padded grid's seeds: inside the photo, every valid cell (SeedAll) or only the background ones (within
    /// <see cref="Params.BackgroundBand"/> of the farthest depth near them, with something nearer in reach); the margin is empty.
    /// </summary>
    static void Level0Kernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> bgMax, ArrayView1D<float, Stride1D.Dense> bgMin, ArrayView1D<float, Stride1D.Dense> l0, Params p)
    {
        int pw = p.GridW + 2 * p.Margin;
        if (i >= pw * (p.GridH + 2 * p.Margin)) return;
        int gx = i % pw - p.Margin, gy = i / pw - p.Margin;
        long o = (long)i.X * Ch;
        float w = 0f, d = 0f;
        int c = 0;
        if (gx >= 0 && gx < p.GridW && gy >= 0 && gy < p.GridH)
        {
            d = DepthAt(depth, p, gx, gy);
            // Behind-edges seeds are the FAR side of an edge: about the farthest surface near them, with a nearer one in
            // reach. The middle of a big foreground object is the farthest thing in its own window, and as a seed it bled
            // foreground colour into the fill behind its edges (MEASURED: the OcclusionFill test's red square).
            int gi = gy * p.GridW + gx;
            w = d > 0f && (p.SeedAll != 0 || (d >= p.BackgroundBand * bgMax[gi] && bgMin[gi] * (1f + p.Tau) < d)) ? 1f : 0f;
            c = rgba[gy * p.Subsample * p.Width + gx * p.Subsample];
        }
        l0[o + 0] = (c & 0xFF) / 255f * w;
        l0[o + 1] = ((c >> 8) & 0xFF) / 255f * w;
        l0[o + 2] = ((c >> 16) & 0xFF) / 255f * w;
        l0[o + 3] = d * w;
        l0[o + 4] = w;
    }

    static void DownKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> fine, ArrayView1D<float, Stride1D.Dense> coarse, LevelParams lp)
    {
        if (i >= lp.CoarseW * lp.CoarseH) return;
        int cx = i % lp.CoarseW, cy = i / lp.CoarseW;
        float far = 0f;
        if (lp.FarPriority != 0)
            for (int k = 0; k < 4; k++)
            {
                int fx = cx * 2 + (k & 1), fy = cy * 2 + (k >> 1);
                if (fx >= lp.FineW || fy >= lp.FineH) continue;
                long o = (long)(fy * lp.FineW + fx) * Ch;
                if (fine[o + 4] > 0f) far = XMath.Max(far, fine[o + 3] / fine[o + 4]);
            }
        float r = 0f, g = 0f, b = 0f, d = 0f, w = 0f;
        for (int k = 0; k < 4; k++)
        {
            int fx = cx * 2 + (k & 1), fy = cy * 2 + (k >> 1);
            if (fx >= lp.FineW || fy >= lp.FineH) continue;
            long o = (long)(fy * lp.FineW + fx) * Ch;
            float fw = fine[o + 4];
            if (fw <= 0f || (lp.FarPriority != 0 && fine[o + 3] / fw < lp.Band * far)) continue;
            r += fine[o]; g += fine[o + 1]; b += fine[o + 2]; d += fine[o + 3]; w += fw;
        }
        long co = (long)i.X * Ch;
        coarse[co] = r; coarse[co + 1] = g; coarse[co + 2] = b; coarse[co + 3] = d; coarse[co + 4] = w;
    }

    /// <summary>Sums to averages (weight 1) where anything landed; empty cells stay weight 0.</summary>
    static void NormalizeKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> level, int count)
    {
        if (i >= count) return;
        long o = (long)i.X * Ch;
        float w = level[o + 4];
        if (w <= 0f) return;
        for (int c = 0; c < 4; c++) level[o + c] /= w;
        level[o + 4] = 1f;
    }

    /// <summary>A fine cell keeps its own average; an empty one takes the bilinear blend of its (filled) coarse neighbours.</summary>
    static void UpKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> fine, ArrayView1D<float, Stride1D.Dense> coarse, LevelParams lp)
    {
        if (i >= lp.FineW * lp.FineH) return;
        long o = (long)i.X * Ch;
        float w = fine[o + 4];
        if (w > 0f)
        {
            for (int c = 0; c < 4; c++) fine[o + c] /= w;
            fine[o + 4] = 1f;
            return;
        }
        int fx = i % lp.FineW, fy = i / lp.FineW;
        float sx = (fx + 0.5f) * 0.5f - 0.5f, sy = (fy + 0.5f) * 0.5f - 0.5f;
        int x0 = (int)XMath.Floor(sx), y0 = (int)XMath.Floor(sy);
        float tx = sx - x0, ty = sy - y0;
        float r = 0f, g = 0f, b = 0f, d = 0f, tw = 0f;
        for (int k = 0; k < 4; k++)
        {
            int cx = x0 + (k & 1), cy = y0 + (k >> 1);
            cx = cx < 0 ? 0 : cx >= lp.CoarseW ? lp.CoarseW - 1 : cx;
            cy = cy < 0 ? 0 : cy >= lp.CoarseH ? lp.CoarseH - 1 : cy;
            long po = (long)(cy * lp.CoarseW + cx) * Ch;
            if (coarse[po + 4] <= 0f) continue;
            float wk = ((k & 1) != 0 ? tx : 1f - tx) * ((k >> 1) != 0 ? ty : 1f - ty);
            r += coarse[po] * wk; g += coarse[po + 1] * wk; b += coarse[po + 2] * wk; d += coarse[po + 3] * wk; tw += wk;
        }
        if (tw <= 1e-6f) return;
        fine[o] = r / tw; fine[o + 1] = g / tw; fine[o + 2] = b / tw; fine[o + 3] = d / tw;
        fine[o + 4] = 1f;
    }

    static void WriteSplat(ArrayView1D<float, Stride1D.Dense> outPacked, int slot, Params p, float u, float v, float fd,
        float r, float g, float b)
    {
        float px = -((u - p.CenterX) * fd / p.FocalX), py = -((v - p.CenterY) * fd / p.FocalY), pz = fd;
        float s = fd * p.Subsample / p.FocalX * p.SizeFactor;
        var q = SplatCovariance.QuatFromNormal(-px, -py, -pz);
        long so = (long)slot * Floats;
        outPacked[so + 0] = px; outPacked[so + 1] = py; outPacked[so + 2] = pz;
        outPacked[so + 3] = r; outPacked[so + 4] = g; outPacked[so + 5] = b;
        outPacked[so + 6] = s; outPacked[so + 7] = s; outPacked[so + 8] = s * 0.15f;
        outPacked[so + 9] = p.Opacity;
        outPacked[so + 10] = q.X; outPacked[so + 11] = q.Y; outPacked[so + 12] = q.Z; outPacked[so + 13] = q.W;
    }

    static void EmitBehindKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<float, Stride1D.Dense> bgMax,
        ArrayView1D<float, Stride1D.Dense> l0, ArrayView1D<float, Stride1D.Dense> outPacked, ArrayView1D<int, Stride1D.Dense> counter, Params p)
    {
        if (i >= p.GridW * p.GridH) return;
        int gx = i % p.GridW, gy = i / p.GridW;
        float d = DepthAt(depth, p, gx, gy);
        if (d <= 0f || bgMax[i] <= d * (1f + p.Tau)) return;   // no farther surface within reach: nothing hidden here
        long o = (long)((gy + p.Margin) * (p.GridW + 2 * p.Margin) + gx + p.Margin) * Ch;
        if (l0[o + 4] <= 0f) return;
        float fd = l0[o + 3];
        if (fd <= d * (1f + p.Tau)) return;                    // the background here is not behind this cell
        int slot = Atomic.Add(ref counter[0], 1);
        if (slot >= p.Capacity) return;
        WriteSplat(outPacked, slot, p, gx * p.Subsample, gy * p.Subsample, fd, l0[o], l0[o + 1], l0[o + 2]);
    }

    /// <summary>Layer 0: the continuation of every edge (<paramref name="l0"/>); layer 1: the background continuation
    /// (<paramref name="behind"/>) where it lies behind layer 0.</summary>
    static void EmitBorderKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> l0, ArrayView1D<float, Stride1D.Dense> behind,
        ArrayView1D<float, Stride1D.Dense> outPacked, ArrayView1D<int, Stride1D.Dense> counter, Params p, int layer)
    {
        int pw = p.GridW + 2 * p.Margin;
        if (i >= pw * (p.GridH + 2 * p.Margin)) return;
        int gx = i % pw - p.Margin, gy = i / pw - p.Margin;
        if (gx >= 0 && gx < p.GridW && gy >= 0 && gy < p.GridH) return;   // inside the photo: the surface layer has it
        int stride = p.BorderStride < 1 ? 1 : p.BorderStride;
        if ((gx + p.Margin) % stride != 0 || (gy + p.Margin) % stride != 0) return;
        long o = (long)i.X * Ch;
        if (l0[o + 4] <= 0f || l0[o + 3] <= 0f) return;
        // Values, not a choice of view: WGSL cannot pick between two storage bindings at run time (a "src = cond ? a : b"
        // view failed every dispatch, 2026-10-06).
        float d = l0[o + 3], r = l0[o], g = l0[o + 1], b = l0[o + 2];
        if (layer != 0)
        {
            float bd = behind[o + 3];
            if (behind[o + 4] <= 0f || bd <= d * (1f + p.Tau)) return;
            d = bd; r = behind[o]; g = behind[o + 1]; b = behind[o + 2];
        }
        int slot = Atomic.Add(ref counter[0], 1);
        if (slot >= p.Capacity) return;
        var q = p; q.SizeFactor = p.SizeFactor * stride;
        WriteSplat(outPacked, slot, q, gx * p.Subsample, gy * p.Subsample, d, r, g, b);
    }
}
