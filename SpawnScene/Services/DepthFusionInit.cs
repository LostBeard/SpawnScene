using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Dense initial splats from the photos' DAv3 depth, kept only where the photos AGREE (&amp;depthinit=1, opt-in).
/// <para>
/// The SfM cloud seeds splats only where features matched: TJ's Bathroom starts a whole room from 7,793 points, and a wall
/// with no seed fills from its neighbours or not at all - the smears and empty patches off the photo path. Every photo
/// already has a DAv3 depth map from the pose pass. Unprojecting them was tried in September and failed for two reasons
/// (SparsePointCloudInit's notes): each view's depth was a PRIVATE shell no other view constrained (DrJohnson: 91.9% of
/// splats seen by at most one view), and at the chunk's scale, not the bundle-adjusted cameras'. Here:
/// </para>
/// <list type="number">
/// <item>Each view's depth is scaled to the bundle-adjusted SfM points it sees: median of z_ba / d over the points that
/// project into it, then again over those within 15% of that (points behind a wall project in too and disagree).</item>
/// <item>A grid sample is kept only when another view's depth agrees with it (within <see cref="Params.RelTol"/> of its
/// camera depth there) - the multi-view consistency test of depth-map fusion. A private shell agrees with nobody.</item>
/// <item>Emitted once: by the FIRST view (by index) that agrees, so a surface six photos see is seeded once, not six times.</item>
/// </list>
/// Sized like SparsePointCloudInit's points (isotropic, the sample spacing), opacity 0.1, the photo's colour when the
/// imported pixels are still resident (else grey - the colour rate fixes it in a few hundred steps).
/// </summary>
public static class DepthFusionInit
{
    const int Floats = SplatFormat.Floats;
    /// <summary>Per view in the camera table: position, right, down, forward (12), fx fy cx cy, width height, depth scale, has-rgba.</summary>
    public const int CamFloats = 20;

    public struct Params
    {
        public int View, Views, Stride, Capacity, MinAgree;
        public float RelTol, Opacity, MaxScale;
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>? _fuse;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _gather;
    static Accelerator? _loadedFor;

    /// <summary>One view's depth on the device (DepthResult.RawDepthGpu and its size): camera depth in the view's own unit.</summary>
    public readonly record struct DepthMap(MemoryBuffer1D<float, Stride1D.Dense> Depth, int Width, int Height);

    /// <summary>What the fusion did, for the log.</summary>
    public sealed record Report(int Views, int ScaledViews, long Candidates, int Emitted, string Scales);

    /// <summary>
    /// Fuse <paramref name="views"/> (indices into <paramref name="cameras"/> / <paramref name="depths"/>) into packed splats.
    /// Returns the GPU buffer (caller owns) and its splat count, or null when fewer than two views could be scaled.
    /// </summary>
    public static async Task<(MemoryBuffer1D<float, Stride1D.Dense> Packed, int Count, Report Report)?> FuseAsync(
        Accelerator a, IReadOnlyList<CameraParams?> cameras, IReadOnlyList<DepthMap?> depths, IReadOnlyList<int> views,
        IReadOnlyList<Vector3> sparsePoints, IReadOnlyList<byte[]?>? rgba, int stride, float relTol, float maxScale,
        bool snapEdges = false)
    {
        // Per accelerator: a kernel cached from another (disposed) accelerator runs against freed state - MEASURED, the
        // second of two unit tests on fresh CPU accelerators failed only when run after the first.
        if (!ReferenceEquals(_loadedFor, a)) { _fuse = null; _gather = null; _loadedFor = a; }
        _fuse ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, Params>(FuseKernel);
        _gather ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(GatherKernel);

        var used = views.Where(v => cameras[v] != null && depths[v] != null).ToList();
        if (used.Count < 2) return null;

        // -- One buffer of every view's depth (and colour), each at its own offset --
        var offsets = new int[used.Count];
        long total = 0;
        for (int k = 0; k < used.Count; k++)
        {
            offsets[k] = (int)total;
            total += (long)depths[used[k]]!.Value.Width * depths[used[k]]!.Value.Height;
        }
        if (total > int.MaxValue) return null;
        using var allDepth = a.Allocate1D<float>(Math.Max(1, total));
        using var allRgba = a.Allocate1D<int>(Math.Max(1, total));
        var hasRgba = new bool[used.Count];
        for (int k = 0; k < used.Count; k++)
        {
            var d = depths[used[k]]!.Value;
            long px = (long)d.Width * d.Height;
            allDepth.View.SubView(offsets[k], px).CopyFrom(d.Depth.View.SubView(0, px));
            var pixels = rgba != null && used[k] < rgba.Count ? rgba[used[k]] : null;
            if (pixels != null && pixels.Length == px * 4)
            {
                // CPU transfer: the photo's own pixels, already decoded on the host by the import - colour for the seeds.
                var words = new int[px];
                Buffer.BlockCopy(pixels, 0, words, 0, pixels.Length);
                allRgba.View.SubView(offsets[k], px).CopyFromCPU(words);
                hasRgba[k] = true;
            }
        }

        // TJ 2026-10-07: the multi-photo seeds come from the same depth maps as a single photo - snap their edges to the
        // colours too (DepthEdgeSnap), so a resize ramp between an object and the wall seeds no splats in mid-air.
        int snappedViews = 0;
        if (snapEdges)
            for (int k = 0; k < used.Count; k++)
            {
                if (!hasRgba[k]) continue;
                var d = depths[used[k]]!.Value;
                long px = (long)d.Width * d.Height;
                using var tmp = a.Allocate1D<float>(px);
                DepthEdgeSnap.Run(a, allDepth.View.SubView(offsets[k], px), allRgba.View.SubView(offsets[k], px), tmp.View, d.Width, d.Height);
                allDepth.View.SubView(offsets[k], px).CopyFrom(tmp.View);
                snappedViews++;
            }

        // -- Per-view depth scale from the SfM points (CPU transfer: one depth value per projected point) --
        var scales = new float[used.Count];
        var camsAtDepth = new CameraParams[used.Count];
        for (int k = 0; k < used.Count; k++)
        {
            var d = depths[used[k]]!.Value;
            camsAtDepth[k] = cameras[used[k]]!.ScaledTo(d.Width, d.Height);
            var idx = new List<int>(); var z = new List<float>();
            foreach (var p in sparsePoints)
            {
                if (!WorldSpaceGeometry.Project(camsAtDepth[k], p, out float u, out float v, out float zc) || zc <= 0) continue;
                int x = (int)u, y = (int)v;
                if (x < 0 || y < 0 || x >= d.Width || y >= d.Height) continue;
                idx.Add(offsets[k] + y * d.Width + x); z.Add(zc);
            }
            if (idx.Count < 20) continue;
            using var idxBuf = a.Allocate1D(idx.ToArray());
            using var vals = a.Allocate1D<float>(idx.Count);
            _gather!(idx.Count, allDepth.View, idxBuf.View, vals.View);
            var dv = await vals.CopyToHostAsync<float>(0, idx.Count);
            var ratios = new List<float>();
            for (int i = 0; i < dv.Length; i++) if (dv[i] > 1e-6f && float.IsFinite(dv[i])) ratios.Add(z[i] / dv[i]);
            if (ratios.Count < 20) continue;
            ratios.Sort();
            float s = ratios[ratios.Count / 2];
            var near = ratios.Where(r => MathF.Abs(MathF.Log(r / s)) < 0.15f).ToList();
            if (near.Count < 10) continue;
            scales[k] = near[near.Count / 2];
        }
        int scaled = scales.Count(s => s > 0f);
        string scaleText = string.Join(" ", scales.Select((s, k) => $"{used[k]}:{s:G3}"));
        if (scaled < 2) return null;

        // -- Camera table --
        var table = new float[used.Count * CamFloats];
        for (int k = 0; k < used.Count; k++)
        {
            var c = camsAtDepth[k];
            WorldSpaceGeometry.GetOpenCvAxes(c, out var right, out var down, out var fwd);
            int o = k * CamFloats;
            table[o] = c.Position.X; table[o + 1] = c.Position.Y; table[o + 2] = c.Position.Z;
            table[o + 3] = right.X; table[o + 4] = right.Y; table[o + 5] = right.Z;
            table[o + 6] = down.X; table[o + 7] = down.Y; table[o + 8] = down.Z;
            table[o + 9] = fwd.X; table[o + 10] = fwd.Y; table[o + 11] = fwd.Z;
            table[o + 12] = c.FocalX; table[o + 13] = c.FocalY; table[o + 14] = c.CenterX; table[o + 15] = c.CenterY;
            table[o + 16] = c.Width; table[o + 17] = c.Height; table[o + 18] = scales[k]; table[o + 19] = hasRgba[k] ? 1f : 0f;
        }
        using var camBuf = a.Allocate1D(table);
        using var offBuf = a.Allocate1D(offsets);

        long candidates = 0;
        for (int k = 0; k < used.Count; k++)
            if (scales[k] > 0f)
                candidates += (long)((camsAtDepth[k].Width + stride - 1) / stride) * ((camsAtDepth[k].Height + stride - 1) / stride);
        int capacity = (int)Math.Min(candidates, 4_000_000);
        var outPacked = a.Allocate1D<float>((long)Math.Max(1, capacity) * Floats);
        using var counter = a.Allocate1D<int>(1);
        counter.MemSetToZero();
        for (int k = 0; k < used.Count; k++)
        {
            if (scales[k] <= 0f) continue;
            int gw = (camsAtDepth[k].Width + stride - 1) / stride, gh = (camsAtDepth[k].Height + stride - 1) / stride;
            _fuse!(gw * gh, allDepth.View, offBuf.View, allRgba.View, camBuf.View, outPacked.View, counter.View, new Params
            {
                View = k, Views = used.Count, Stride = stride, Capacity = capacity, MinAgree = 1,
                RelTol = relTol, Opacity = SparsePointCloudInit.InitialOpacity, MaxScale = maxScale,
            });
        }
        // CPU transfer: the emitted count.
        int emitted = Math.Min((await counter.CopyToHostAsync<int>(0, 1))[0], capacity);
        return (outPacked, emitted, new Report(used.Count, scaled, candidates, emitted,
            (snapEdges ? $"(edges snapped in {snappedViews} views) " : "") + scaleText));
    }

    static void GatherKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> idx,
        ArrayView1D<float, Stride1D.Dense> dst) => dst[i] = src[idx[i]];

    /// <summary>One grid sample of view p.View: unproject, test against every other view, emit if agreed and first.</summary>
    static void FuseKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> offsets,
        ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<float, Stride1D.Dense> cams, ArrayView1D<float, Stride1D.Dense> outPacked,
        ArrayView1D<int, Stride1D.Dense> counter, Params p)
    {
        int c = p.View * CamFloats;
        int w = (int)cams[c + 16], h = (int)cams[c + 17];
        int gw = (w + p.Stride - 1) / p.Stride;
        int sx = i % gw, sy = i / gw;
        int px = XMath.Min(sx * p.Stride + p.Stride / 2, w - 1), py = XMath.Min(sy * p.Stride + p.Stride / 2, h - 1);
        float d = depth[offsets[p.View] + py * w + px] * cams[c + 18];
        if (!(d > 1e-6f) || d > 1e20f) return;
        float fx = cams[c + 12], fy = cams[c + 13];
        float xc = (px + 0.5f - cams[c + 14]) * d / fx, yc = (py + 0.5f - cams[c + 15]) * d / fy;
        float wx = cams[c] + cams[c + 3] * xc + cams[c + 6] * yc + cams[c + 9] * d;
        float wy = cams[c + 1] + cams[c + 4] * xc + cams[c + 7] * yc + cams[c + 10] * d;
        float wz = cams[c + 2] + cams[c + 5] * xc + cams[c + 8] * yc + cams[c + 11] * d;

        int agree = 0;
        bool earlier = false;
        for (int j = 0; j < p.Views; j++)
        {
            if (j == p.View) continue;
            int o = j * CamFloats;
            float s = cams[o + 18];
            if (s <= 0f) continue;
            float rx = wx - cams[o], ry = wy - cams[o + 1], rz = wz - cams[o + 2];
            float zc = cams[o + 9] * rx + cams[o + 10] * ry + cams[o + 11] * rz;
            if (zc <= 1e-4f) continue;
            float u = cams[o + 12] * (cams[o + 3] * rx + cams[o + 4] * ry + cams[o + 5] * rz) / zc + cams[o + 14];
            float v = cams[o + 13] * (cams[o + 6] * rx + cams[o + 7] * ry + cams[o + 8] * rz) / zc + cams[o + 15];
            int wj = (int)cams[o + 16], hj = (int)cams[o + 17];
            if (u < 0f || v < 0f || u >= wj || v >= hj) continue;
            float dj = depth[offsets[j] + (int)v * wj + (int)u] * s;
            if (XMath.Abs(dj - zc) < p.RelTol * zc)
            {
                agree++;
                if (j < p.View) earlier = true;
            }
        }
        if (agree < p.MinAgree || earlier) return;

        int slot = Atomic.Add(ref counter[0], 1);
        if (slot >= p.Capacity) return;
        long q = (long)slot * Floats;
        outPacked[q] = wx; outPacked[q + 1] = wy; outPacked[q + 2] = wz;
        float r = 0.5f, g = 0.5f, b = 0.5f;
        if (cams[c + 19] > 0.5f)
        {
            int px4 = rgba[offsets[p.View] + py * w + px];
            r = (px4 & 0xFF) / 255f; g = ((px4 >> 8) & 0xFF) / 255f; b = ((px4 >> 16) & 0xFF) / 255f;
        }
        outPacked[q + 3] = r; outPacked[q + 4] = g; outPacked[q + 5] = b;
        // The sample spacing at this depth: a seed as big as the gap to the next one, as SparsePointCloudInit sizes points.
        float scale = XMath.Min(d * p.Stride / fx, p.MaxScale);
        outPacked[q + 6] = scale; outPacked[q + 7] = scale; outPacked[q + 8] = scale;
        outPacked[q + 9] = p.Opacity;
        outPacked[q + 10] = 0f; outPacked[q + 11] = 0f; outPacked[q + 12] = 0f; outPacked[q + 13] = 1f;
    }
}
