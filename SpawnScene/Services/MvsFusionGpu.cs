using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// GPU twin of <see cref="MvsGeometricFusion"/>. The depth maps arrive GPU-resident from the joint
/// DAv3 forward and stay there: metricization, forward-back consistency and the scale probe all run
/// as kernels, and the only thing that crosses to the host is a handful of ints and floats.
///
/// This exists because the CPU path was a GPU -> .NET -> GPU round trip. At 640x480 with 4 views it
/// downloaded 4.9 MB, ran ~1.8M forward-back tests in WASM (the probe's result was discarded except
/// for a log line and a keep-ratio gate), then uploaded the same 4.9 MB back for the unproject.
/// That is what hung the browser lane.
///
/// <see cref="MvsGeometricFusion"/> stays as the CPU oracle. These kernels must match it.
/// Gate: SpawnScene.Tests MvsFusionGpuTests (GPU vs CPU oracle on the synth + TempleRing fixtures).
/// </summary>
public sealed class MvsFusionGpu : IDisposable
{
    /// <summary>Threads that entered the last forward-back dispatch. 0 means it never ran.</summary>
    public int LastFbThreads { get; private set; }

    /// <summary>Floats per camera in the packed camera buffer.</summary>
    public const int CamStride = 16;

    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // rawDepths   [n * w * h]
        ArrayView1D<float, Stride1D.Dense>,   // scaleA      [n]
        ArrayView1D<float, Stride1D.Dense>,   // scaleB      [n]
        ArrayView1D<float, Stride1D.Dense>,   // outMetric   [n * w * h]
        MetricizeParams>? _metricizeKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // metric      [n * w * h]
        ArrayView1D<float, Stride1D.Dense>,   // conf        [n * w * h] or dummy
        ArrayView1D<float, Stride1D.Dense>,   // cams        [n * CamStride]
        ArrayView1D<float, Stride1D.Dense>,   // outClean    [n * w * h]
        ArrayView1D<int, Stride1D.Dense>,     // stats       [5]
        FbParams>? _fbKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // metric      [n * w * h]
        ArrayView1D<float, Stride1D.Dense>,   // cams        [n * CamStride]
        ArrayView1D<int, Stride1D.Dense>,     // counts      [scaleSteps]
        ScaleProbeParams>? _scaleProbeKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // metric      [n * w * h]
        ArrayView1D<float, Stride1D.Dense>,   // scales      [n]
        ScaleApplyParams>? _scaleApplyKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // maps        [n * w * h]
        ArrayView1D<float, Stride1D.Dense>,   // cams        [n * CamStride]
        ArrayView1D<float, Stride1D.Dense>,   // anchors     [anchorCount * 3]
        ArrayView1D<float, Stride1D.Dense>,   // out         [n * anchorCount * 2] (raw, zCam)
        AnchorGatherParams>? _anchorGatherKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // cleaned     [n * w * h] (snapshot, read only)
        ArrayView1D<float, Stride1D.Dense>,   // metric      [n * w * h] (agreement gate)
        ArrayView1D<float, Stride1D.Dense>,   // cams        [n * CamStride]
        ArrayView1D<int, Stride1D.Dense>,     // warpBits    [n * w * h] float bits, min-reduced
        WarpFillParams>? _warpScatterKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // cleaned     [n * w * h] (written)
        ArrayView1D<int, Stride1D.Dense>,     // warpBits    [n * w * h]
        ArrayView1D<int, Stride1D.Dense>,     // counter     [1]
        WarpFillParams>? _warpMergeKernel;

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, int>? _fillIntKernel;

    // Small per-view parameter buffers, kept alive for the object's lifetime. See Metricize.
    private MemoryBuffer1D<float, Stride1D.Dense>? _scaleABuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _scaleBBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _scalesBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dummyBuf;

    /// <summary>
    /// A distinct 1-element buffer for kernel slots that are declared but unused on this call.
    /// Never alias another bound buffer into the slot: WebGPU rejects the dispatch.
    /// </summary>
    private ArrayView1D<float, Stride1D.Dense> DummyFloat()
    {
        _dummyBuf ??= _accelerator.Allocate1D<float>(1);
        return _dummyBuf.View;
    }

    /// <summary>
    /// Upload a small float[] into a cached device buffer, reallocating only when the length
    /// changes. The buffer is NOT disposed at the call site: a WebGPU dispatch is submitted after
    /// the managed call returns, so a param buffer freed there is destroyed before its own submit.
    /// </summary>
    private ArrayView1D<float, Stride1D.Dense> ParamBuffer(
        ref MemoryBuffer1D<float, Stride1D.Dense>? slot, float[] values)
    {
        if (slot == null || slot.Length != values.Length)
        {
            slot?.Dispose();
            slot = _accelerator.Allocate1D<float>(values.Length);
        }
        slot.CopyFromCPU(values);
        return slot.View;
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // maps     [n * w * h]
        ArrayView1D<int, Stride1D.Dense>,     // minBits  [n]
        ArrayView1D<int, Stride1D.Dense>,     // maxBits  [n]
        MetricizeParams>? _minMaxKernel;

    public MvsFusionGpu(Accelerator accelerator) => _accelerator = accelerator;

    // ─── Parameter structs (ILGPU decomposes these into scalar bindings) ───

    public struct MetricizeParams
    {
        public int PixelsPerView;
        public int ViewCount;
    }

    public struct FbParams
    {
        public int Width, Height, ViewCount;
        public int Subsample;
        public float MaxDepthError;
        public float MaxReprojPx;
        public int MinViews;
        /// <summary>1 = conf buffer is a real [n*w*h] map; 0 = dummy, ignore.</summary>
        public int HasConf;
        public float ConfMin;
        /// <summary>1 = write the median-cleaned depth to outClean; 0 = probe only (stats).</summary>
        public int WriteClean;
    }

    public struct ScaleProbeParams
    {
        public int Width, Height, ViewCount;
        public int TargetView;
        public int Subsample;
        public float MaxDepthError;
        public float MaxReprojPx;
        public int ScaleSteps;
        public float ScaleStart;
        public float ScaleStep;
    }

    public struct ScaleApplyParams
    {
        public int PixelsPerView;
        public int ViewCount;
    }

    public struct AnchorGatherParams
    {
        public int Width, Height, ViewCount, AnchorCount;
    }

    public struct WarpFillParams
    {
        public int Width, Height, ViewCount;
        /// <summary>Relative depth agreement required against the destination's own metric depth.</summary>
        public float AgreeDepthError;
    }

    // ─── Camera packing (host side, ViewCount * 16 floats) ───

    /// <summary>
    /// Pack cameras into the flat layout the kernels index: pos(3), right(3), down(3), forward(3),
    /// fx, fy, cx, cy. The OpenCV axes are derived once here rather than per thread, and must match
    /// <see cref="WorldSpaceGeometry.GetOpenCvAxes"/>.
    /// </summary>
    public static float[] PackCameras(IReadOnlyList<CameraParams> cams)
    {
        var packed = new float[cams.Count * CamStride];
        for (int i = 0; i < cams.Count; i++)
        {
            WorldSpaceGeometry.GetOpenCvAxes(cams[i], out var right, out var down, out var forward);
            int o = i * CamStride;
            packed[o + 0] = cams[i].Position.X;
            packed[o + 1] = cams[i].Position.Y;
            packed[o + 2] = cams[i].Position.Z;
            packed[o + 3] = right.X; packed[o + 4] = right.Y; packed[o + 5] = right.Z;
            packed[o + 6] = down.X; packed[o + 7] = down.Y; packed[o + 8] = down.Z;
            packed[o + 9] = forward.X; packed[o + 10] = forward.Y; packed[o + 11] = forward.Z;
            packed[o + 12] = cams[i].FocalX;
            packed[o + 13] = cams[i].FocalY;
            packed[o + 14] = cams[i].CenterX;
            packed[o + 15] = cams[i].CenterY;
        }
        return packed;
    }

    // ─── Device-side geometry (must match WorldSpaceGeometry bit for bit) ───

    private static void Unproject(
        ArrayView1D<float, Stride1D.Dense> cams, int cam,
        float u, float v, float z,
        out float wx, out float wy, out float wz)
    {
        int o = cam * CamStride;
        float x = (u - cams[o + 14]) * z / cams[o + 12];
        float y = (v - cams[o + 15]) * z / cams[o + 13];
        wx = cams[o + 0] + cams[o + 3] * x + cams[o + 6] * y + cams[o + 9] * z;
        wy = cams[o + 1] + cams[o + 4] * x + cams[o + 7] * y + cams[o + 10] * z;
        wz = cams[o + 2] + cams[o + 5] * x + cams[o + 8] * y + cams[o + 11] * z;
    }

    private static bool Project(
        ArrayView1D<float, Stride1D.Dense> cams, int cam,
        float wx, float wy, float wz,
        out float u, out float v, out float zCam)
    {
        int o = cam * CamStride;
        float dx = wx - cams[o + 0];
        float dy = wy - cams[o + 1];
        float dz = wz - cams[o + 2];
        float xc = cams[o + 3] * dx + cams[o + 4] * dy + cams[o + 5] * dz;
        float yc = cams[o + 6] * dx + cams[o + 7] * dy + cams[o + 8] * dz;
        zCam = cams[o + 9] * dx + cams[o + 10] * dy + cams[o + 11] * dz;
        if (zCam <= 1e-6f) { u = 0f; v = 0f; return false; }
        u = cams[o + 12] * xc / zCam + cams[o + 14];
        v = cams[o + 13] * yc / zCam + cams[o + 15];
        return true;
    }

    /// <summary>
    /// Forward-back consistency of a src pixel against neighbor view <paramref name="nbr"/>.
    /// Mirrors <see cref="MvsGeometricFusion.IsForwardBackConsistent"/>. When it passes,
    /// <paramref name="zBackOut"/> carries the neighbor's depth reprojected into src, which the
    /// clean pass medians.
    /// </summary>
    private static bool FbConsistent(
        ArrayView1D<float, Stride1D.Dense> cams,
        ArrayView1D<float, Stride1D.Dense> maps,
        int src, int nbr, int width, int height, int pixelsPerView,
        float uSrc, float vSrc, float srcDepth,
        float maxDepthError, float maxReprojPx,
        out float zBackOut)
    {
        zBackOut = 0f;
        if (!(srcDepth > 1e-6f)) return false;

        Unproject(cams, src, uSrc, vSrc, srcDepth, out float wx, out float wy, out float wz);
        if (!Project(cams, nbr, wx, wy, wz, out float uN, out float vN, out float zProj)) return false;
        if (zProj <= 1e-6f) return false;

        int iu = (int)MathF.Round(uN);
        int iv = (int)MathF.Round(vN);
        if (iu < 0 || iu >= width || iv < 0 || iv >= height) return false;

        float dNbr = maps[nbr * pixelsPerView + iv * width + iu];
        if (!(dNbr > 1e-6f)) return false;

        float denom = MathF.Max(zProj, dNbr);
        float rel = MathF.Abs(zProj - dNbr) / denom;
        if (rel > maxDepthError) return false;

        Unproject(cams, nbr, iu + 0.5f, iv + 0.5f, dNbr, out float w2x, out float w2y, out float w2z);
        if (!Project(cams, src, w2x, w2y, w2z, out float uB, out float vB, out float zBack)) return false;
        if (zBack <= 1e-6f) return false;

        float du = uB - uSrc;
        float dv = vB - vSrc;
        if (MathF.Sqrt(du * du + dv * dv) > maxReprojPx) return false;

        zBackOut = zBack;
        return true;
    }

    // ─── Kernels ───

    /// <summary>metric = a*raw + b per view, zero where raw is invalid.</summary>
    private static void MetricizeKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> raw,
        ArrayView1D<float, Stride1D.Dense> scaleA,
        ArrayView1D<float, Stride1D.Dense> scaleB,
        ArrayView1D<float, Stride1D.Dense> outMetric,
        MetricizeParams p)
    {
        int i = index;
        if (i >= p.PixelsPerView * p.ViewCount) return;
        int view = i / p.PixelsPerView;
        float r = raw[i];
        outMetric[i] = r > 1e-6f ? scaleA[view] * r + scaleB[view] : 0f;
    }

    /// <summary>
    /// One thread per (view, pixel). Counts agreeing views by forward-back, writes the median of
    /// source + reprojected neighbor depths into outClean when WriteClean=1, and accumulates
    /// stats[0]=input stats[1]=kept stats[2]=minViewReject stats[3]=confReject.
    /// </summary>
    private static void FbKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> conf,
        ArrayView1D<float, Stride1D.Dense> cams,
        ArrayView1D<float, Stride1D.Dense> outClean,
        ArrayView1D<int, Stride1D.Dense> stats,
        FbParams p)
    {
        int pixelsPerView = p.Width * p.Height;
        int total = pixelsPerView * p.ViewCount;
        int i = index;
        if (i >= total) return;

        // stats[4] counts every thread that entered, so a zero result can be told apart from a
        // dispatch that never ran.
        Atomic.Add(ref stats[4], 1);

        int view = i / pixelsPerView;
        int pix = i - view * pixelsPerView;
        int x = pix % p.Width;
        int y = pix / p.Width;

        // 🔴 DO NOT INLINE THIS PREDICATE INTO THE `if`. On the WebGPU backend, written as
        //     if (p.Subsample > 1 && ((x % p.Subsample) != 0 || (y % p.Subsample) != 0)) return;
        // it rejected EVERY pixel of 1,228,800 - including x=0,y=0, which provably passes.
        // Hoisting the scalar into a local was NOT enough on its own; the predicate has to be
        // evaluated into a bool local and the branch taken on that. Verified both ways on real
        // WebGPU (307,200 survivors with this spelling, 0 with the inline one).
        // NOT fully characterised as an ILGPU codegen bug yet - see the note on the board.
        int sub = p.Subsample;
        bool skip = sub > 1 && (((x / sub) * sub) != x || ((y / sub) * sub) != y);
        if (skip) return;

        float d = metric[i];
        if (!(d > 1e-6f)) return;
        Atomic.Add(ref stats[0], 1);

        if (p.HasConf == 1 && conf[i] < p.ConfMin)
        {
            Atomic.Add(ref stats[3], 1);
            return;
        }

        float uSrc = x + 0.5f;
        float vSrc = y + 0.5f;

        // Up to 8 candidate depths for the median: self plus one per neighbor view.
        var zs = new ZBuf8();
        zs.Set(0, d);
        int nz = 1;
        int agree = 1;

        for (int j = 0; j < p.ViewCount; j++)
        {
            if (j == view) continue;
            if (!FbConsistent(cams, metric, view, j, p.Width, p.Height, pixelsPerView,
                    uSrc, vSrc, d, p.MaxDepthError, p.MaxReprojPx, out float zBack))
                continue;
            agree++;
            if (nz < 8) { zs.Set(nz, zBack); nz++; }
        }

        if (agree < p.MinViews)
        {
            Atomic.Add(ref stats[2], 1);
            return;
        }

        Atomic.Add(ref stats[1], 1);
        if (p.WriteClean == 1)
            outClean[i] = zs.Median(nz);
    }

    /// <summary>
    /// Grid-search probe: thread = (scaleStep, probe pixel) over TargetView only. Counts pixels
    /// reaching 2 agreeing views at each candidate scale. Mirrors
    /// <see cref="MvsGeometricFusion.OptimizeScaleFactor"/>; the host picks the argmax.
    /// </summary>
    private static void ScaleProbeKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> cams,
        ArrayView1D<int, Stride1D.Dense> counts,
        ScaleProbeParams p)
    {
        int pixelsPerView = p.Width * p.Height;
        int probeW = (p.Width + p.Subsample - 1) / p.Subsample;
        int probeH = (p.Height + p.Subsample - 1) / p.Subsample;
        int probesPerScale = probeW * probeH;
        int total = probesPerScale * p.ScaleSteps;
        int i = index;
        if (i >= total) return;

        int step = i / probesPerScale;
        int probe = i - step * probesPerScale;
        int x = (probe % probeW) * p.Subsample;
        int y = (probe / probeW) * p.Subsample;
        if (x >= p.Width || y >= p.Height) return;

        float f = p.ScaleStart + step * p.ScaleStep;
        float d0 = metric[p.TargetView * pixelsPerView + y * p.Width + x];
        if (!(d0 > 1e-6f)) return;
        float d = d0 * f;

        // The neighbor maps are unscaled; only the target view's sample carries the trial scale,
        // which is what the CPU oracle does (it substitutes a scaled copy of the target map only).
        int agree = 1;
        for (int j = 0; j < p.ViewCount; j++)
        {
            if (j == p.TargetView) continue;
            if (FbConsistent(cams, metric, p.TargetView, j, p.Width, p.Height, pixelsPerView,
                    x + 0.5f, y + 0.5f, d, p.MaxDepthError, p.MaxReprojPx, out _))
                agree++;
        }
        if (agree >= 2) Atomic.Add(ref counts[step], 1);
    }

    /// <summary>Multiply each view's metric map by its refined scale, in place.</summary>
    private static void ScaleApplyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> scales,
        ScaleApplyParams p)
    {
        int i = index;
        if (i >= p.PixelsPerView * p.ViewCount) return;
        float d = metric[i];
        if (d > 1e-6f) metric[i] = d * scales[i / p.PixelsPerView];
    }

    /// <summary>
    /// Gather the (raw depth, camera z) pair at each anchor's projection in each view. This is the
    /// one legitimate host crossing in the path: ViewCount * AnchorCount * 2 floats, a few dozen
    /// values, which is what the scalar exemption is for. Emits -1 raw for a miss.
    /// </summary>
    private static void AnchorGatherKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> maps,
        ArrayView1D<float, Stride1D.Dense> cams,
        ArrayView1D<float, Stride1D.Dense> anchors,
        ArrayView1D<float, Stride1D.Dense> outPairs,
        AnchorGatherParams p)
    {
        int i = index;
        if (i >= p.ViewCount * p.AnchorCount) return;
        int view = i / p.AnchorCount;
        int a = i - view * p.AnchorCount;

        outPairs[i * 2 + 0] = -1f;
        outPairs[i * 2 + 1] = 0f;

        if (!Project(cams, view, anchors[a * 3 + 0], anchors[a * 3 + 1], anchors[a * 3 + 2],
                out float u, out float v, out float zCam))
            return;
        if (zCam <= 1e-4f) return;

        int ix = (int)MathF.Round(u);
        int iy = (int)MathF.Round(v);
        if (ix < 0 || ix >= p.Width || iy < 0 || iy >= p.Height) return;

        float raw = maps[view * p.Width * p.Height + iy * p.Width + ix];
        if (!(raw > 1e-4f)) return;

        outPairs[i * 2 + 0] = raw;
        outPairs[i * 2 + 1] = zCam;
    }

    /// <summary>
    /// Cross-view warp fill, scatter half. Every cleaned source pixel is unprojected and projected
    /// into every other view; where the destination has no cleaned depth and its own metric depth
    /// agrees (or is absent), the warped z competes for the pixel.
    ///
    /// ⚠️ NEAREST WINS, by atomic min on the float bit pattern. The CPU oracle originally took the
    /// FIRST writer in source-then-raster order, which is not reproducible on a GPU and is wrong on
    /// its own terms: when a near and a far surface land on the same destination pixel, first-wins
    /// can paint the occluded one into the depth map. A warp fill is a z-buffer. The oracle was
    /// changed to match this, not the other way round.
    /// Positive floats order identically to their int bit patterns, and depth is always > 0 here.
    /// </summary>
    private static void WarpScatterKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> cleaned,
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> cams,
        ArrayView1D<int, Stride1D.Dense> warpBits,
        WarpFillParams p)
    {
        int pixelsPerView = p.Width * p.Height;
        int i = index;
        if (i >= pixelsPerView * p.ViewCount) return;

        int src = i / pixelsPerView;
        int pix = i - src * pixelsPerView;
        float d = cleaned[i];
        if (!(d > 1e-6f)) return;

        int x = pix % p.Width;
        int y = pix / p.Width;
        Unproject(cams, src, x + 0.5f, y + 0.5f, d, out float wx, out float wy, out float wz);

        for (int dst = 0; dst < p.ViewCount; dst++)
        {
            if (dst == src) continue;
            if (!Project(cams, dst, wx, wy, wz, out float u, out float v, out float z)) continue;
            if (z <= 1e-6f) continue;

            int iu = (int)MathF.Round(u);
            int iv = (int)MathF.Round(v);
            if (iu < 0 || iu >= p.Width || iv < 0 || iv >= p.Height) continue;

            int dp = dst * pixelsPerView + iv * p.Width + iu;
            if (cleaned[dp] > 1e-6f) continue; // destination already has a cleaned core

            float dstSrc = metric[dp];
            if (dstSrc > 1e-6f)
            {
                float rel = MathF.Abs(dstSrc - z) / MathF.Max(dstSrc, z);
                if (rel > p.AgreeDepthError) continue;
            }

            Atomic.Min(ref warpBits[dp], BitConverter.SingleToInt32Bits(z));
        }
    }

    /// <summary>
    /// Per-view min/max of the positive depths, as atomic min/max on the float bit pattern
    /// (valid because depth is always > 0 here). Lets the caller fill DepthResult's UI metadata
    /// without scanning the map on the host.
    /// </summary>
    private static void MinMaxKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> maps,
        ArrayView1D<int, Stride1D.Dense> minBits,
        ArrayView1D<int, Stride1D.Dense> maxBits,
        MetricizeParams p)
    {
        int i = index;
        if (i >= p.PixelsPerView * p.ViewCount) return;
        float d = maps[i];
        if (!(d > 1e-6f)) return;
        int view = i / p.PixelsPerView;
        int bits = BitConverter.SingleToInt32Bits(d);
        Atomic.Min(ref minBits[view], bits);
        Atomic.Max(ref maxBits[view], bits);
    }

    /// <summary>Seed a device int buffer with a constant. Avoids staging a fill array on the host.</summary>
    private static void FillIntKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> buffer,
        int value)
    {
        if (index < buffer.Length) buffer[index] = value;
    }

    /// <summary>Warp fill, merge half: fold the min-reduced warp depths into the cleaned maps.</summary>
    private static void WarpMergeKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> cleaned,
        ArrayView1D<int, Stride1D.Dense> warpBits,
        ArrayView1D<int, Stride1D.Dense> counter,
        WarpFillParams p)
    {
        int i = index;
        if (i >= p.Width * p.Height * p.ViewCount) return;
        if (cleaned[i] > 1e-6f) return;
        int bits = warpBits[i];
        if (bits == int.MaxValue) return;
        cleaned[i] = BitConverter.Int32BitsToSingle(bits);
        Atomic.Add(ref counter[0], 1);
    }

    /// <summary>
    /// Fixed 8-slot register buffer with an insertion-sorted median. ViewCount is capped at 6 by the
    /// DAv3 joint cap, so 8 slots covers self plus every neighbor with room to spare.
    /// </summary>
    private struct ZBuf8
    {
        private float z0, z1, z2, z3, z4, z5, z6, z7;

        public void Set(int i, float v)
        {
            if (i == 0) z0 = v; else if (i == 1) z1 = v; else if (i == 2) z2 = v;
            else if (i == 3) z3 = v; else if (i == 4) z4 = v; else if (i == 5) z5 = v;
            else if (i == 6) z6 = v; else z7 = v;
        }

        public float Get(int i)
        {
            if (i == 0) return z0; if (i == 1) return z1; if (i == 2) return z2;
            if (i == 3) return z3; if (i == 4) return z4; if (i == 5) return z5;
            if (i == 6) return z6; return z7;
        }

        /// <summary>Median of the first <paramref name="n"/> slots, matching List.Sort()[n/2].</summary>
        public float Median(int n)
        {
            // Selection sort in registers; n <= 8.
            for (int a = 0; a < n - 1; a++)
            {
                int m = a;
                for (int b = a + 1; b < n; b++)
                    if (Get(b) < Get(m)) m = b;
                if (m != a)
                {
                    float t = Get(a);
                    Set(a, Get(m));
                    Set(m, t);
                }
            }
            return Get(n / 2);
        }
    }

    // ─── Kernel loading ───

    private void EnsureKernels()
    {
        _metricizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            MetricizeParams>(MetricizeKernel);

        _fbKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            FbParams>(FbKernel);

        _scaleProbeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ScaleProbeParams>(ScaleProbeKernel);

        _scaleApplyKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ScaleApplyParams>(ScaleApplyKernel);

        _anchorGatherKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            AnchorGatherParams>(AnchorGatherKernel);

        _warpScatterKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            WarpFillParams>(WarpScatterKernel);

        _warpMergeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            WarpFillParams>(WarpMergeKernel);

        _fillIntKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<int, Stride1D.Dense>, int>(FillIntKernel);

        _minMaxKernel ??= _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            MetricizeParams>(MinMaxKernel);
    }

    /// <summary>
    /// Per-view (min, max) over positive depths. 2 * ViewCount floats cross to the host, which is
    /// the UI metadata the DepthResult carries - not a map readback.
    /// </summary>
    public async Task<(float min, float max)[]> MinMaxPerViewAsync(
        ArrayView1D<float, Stride1D.Dense> maps, int pixelsPerView, int viewCount)
    {
        EnsureKernels();
        using var minBits = _accelerator.Allocate1D<int>(viewCount);
        using var maxBits = _accelerator.Allocate1D<int>(viewCount);
        _fillIntKernel!(viewCount, minBits.View, int.MaxValue);
        _fillIntKernel!(viewCount, maxBits.View, int.MinValue);

        _minMaxKernel!(pixelsPerView * viewCount, maps, minBits.View, maxBits.View,
            new MetricizeParams { PixelsPerView = pixelsPerView, ViewCount = viewCount });
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var lo = await minBits.CopyToHostAsync<int>(0, viewCount).ConfigureAwait(false);
        var hi = await maxBits.CopyToHostAsync<int>(0, viewCount).ConfigureAwait(false);

        var result = new (float, float)[viewCount];
        for (int v = 0; v < viewCount; v++)
        {
            result[v] = lo[v] == int.MaxValue
                ? (0f, 0f)
                : (BitConverter.Int32BitsToSingle(lo[v]), BitConverter.Int32BitsToSingle(hi[v]));
        }
        return result;
    }

    /// <summary>
    /// Cross-view warp fill over the cleaned maps, in place. Returns the number of pixels filled
    /// (one int crosses to the host, for the log line the CPU path already prints).
    /// </summary>
    public async Task<int> WarpFillAsync(
        ArrayView1D<float, Stride1D.Dense> cleaned,
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> cams,
        int width, int height, int viewCount,
        float agreeDepthError = 0.05f)
    {
        EnsureKernels();
        long total = (long)width * height * viewCount;
        var prm = new WarpFillParams
        {
            Width = width,
            Height = height,
            ViewCount = viewCount,
            AgreeDepthError = agreeDepthError,
        };

        using var warpBits = _accelerator.Allocate1D<int>(total);
        // Zero is +0.0f as a bit pattern and would win every min, so the sentinel has to be
        // int.MaxValue. Seeded by a kernel, NOT by staging an int[n*w*h] through the managed heap.
        _fillIntKernel!((int)total, warpBits.View, int.MaxValue);

        using var counter = _accelerator.Allocate1D<int>(1);
        counter.CopyFromCPU(new int[] { 0 });

        _warpScatterKernel!((int)total, cleaned, metric, cams, warpBits.View, prm);
        _warpMergeKernel!((int)total, cleaned, warpBits.View, counter.View, prm);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var host = await counter.CopyToHostAsync<int>(0, 1).ConfigureAwait(false);
        return host[0];
    }

    // ─── Public host API. Everything here keeps bulk data on the device. ───

    /// <summary>
    /// Concatenate N GPU-resident per-view depth maps into one [n * w * h] device buffer.
    /// Device to device, no host crossing. Caller owns the result.
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> PackViews(
        IReadOnlyList<MemoryBuffer1D<float, Stride1D.Dense>> views, int pixelsPerView)
    {
        var packed = _accelerator.Allocate1D<float>((long)views.Count * pixelsPerView);
        for (int i = 0; i < views.Count; i++)
            packed.View.SubView(i * (long)pixelsPerView, pixelsPerView).CopyFrom(views[i].View);
        return packed;
    }

    /// <summary>Allocate and upload the packed camera buffer (n * 16 floats). Caller owns it.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> UploadCameras(IReadOnlyList<CameraParams> cams)
        => _accelerator.Allocate1D(PackCameras(cams));

    /// <summary>
    /// metric = a*raw + b into an EXISTING destination view. Avoids allocating a second full
    /// [n*w*h] buffer plus a device-to-device copy when re-metricizing after a re-anchor.
    /// </summary>
    public void MetricizeInto(
        ArrayView1D<float, Stride1D.Dense> raw,
        float[] scaleA, float[] scaleB, int pixelsPerView,
        ArrayView1D<float, Stride1D.Dense> dest)
    {
        EnsureKernels();
        int n = scaleA.Length;
        var a = ParamBuffer(ref _scaleABuf, scaleA);
        var b = ParamBuffer(ref _scaleBBuf, scaleB);
        _metricizeKernel!(n * pixelsPerView, raw, a, b, dest,
            new MetricizeParams { PixelsPerView = pixelsPerView, ViewCount = n });
    }

    /// <summary>metric = a*raw + b, on device. Caller owns the returned buffer.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Metricize(
        ArrayView1D<float, Stride1D.Dense> raw,
        float[] scaleA, float[] scaleB, int pixelsPerView)
    {
        EnsureKernels();
        int n = scaleA.Length;
        var outMetric = _accelerator.Allocate1D<float>((long)n * pixelsPerView);
        // ⚠️ These param buffers must OUTLIVE this call. WebGPU batches the dispatch and submits
        // later, so a `using` here destroys them before the submit and Chrome reports
        // '[Buffer "Storage#...:16B"] used in submit while destroyed'. The error then surfaces at
        // the NEXT Synchronize, in an unrelated method, which is what made it hard to place.
        // They live for the object's lifetime instead. See ref-a-shared-param-arena-freed.
        var a = ParamBuffer(ref _scaleABuf, scaleA);
        var b = ParamBuffer(ref _scaleBBuf, scaleB);
        _metricizeKernel!(n * pixelsPerView, raw, a, b, outMetric.View,
            new MetricizeParams { PixelsPerView = pixelsPerView, ViewCount = n });
        return outMetric;
    }

    /// <summary>Multiply each view's map by its refined scale, in place on device.</summary>
    public void ApplyScales(ArrayView1D<float, Stride1D.Dense> metric, float[] scales, int pixelsPerView)
    {
        EnsureKernels();
        // Same lifetime rule as Metricize: do not dispose before the batched submit.
        var s = ParamBuffer(ref _scalesBuf, scales);
        _scaleApplyKernel!(scales.Length * pixelsPerView, metric, s,
            new ScaleApplyParams { PixelsPerView = pixelsPerView, ViewCount = scales.Length });
    }

    /// <summary>
    /// Forward-back pass. Returns the fuse stats; when <paramref name="outClean"/> is supplied it
    /// also receives the median-cleaned metric depth (0 where rejected). Only the 5-int stats
    /// buffer crosses to the host.
    /// </summary>
    public async Task<MvsGeometricFusion.FuseStats> ForwardBackAsync(
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> conf,
        ArrayView1D<float, Stride1D.Dense> cams,
        int width, int height, int viewCount,
        int subsample, float maxDepthError, float maxReprojPx, int minViews,
        bool hasConf, float confMin,
        ArrayView1D<float, Stride1D.Dense>? outClean = null)
    {
        EnsureKernels();
        int pixelsPerView = width * height;
        using var stats = _accelerator.Allocate1D<int>(8);
        // ⚠️ CopyFromCPU of a tiny zero array, NOT MemSetToZero: the WebGPU backend errored on
        // MemSetToZero (2 GPU errors during dispatch). Five sites in DepthToGaussianKernel already
        // use this idiom. A handful of ints is inside the scalar exemption.
        stats.CopyFromCPU(new int[8]);

        // ⚠️ The unused slot needs a SEPARATE buffer, not `metric` again. WebGPU forbids binding one
        // buffer to two read_write storage slots and SpawnDev.ILGPU fails the dispatch with
        // "Storage buffer aliasing detected ... binding 0 and binding 3". The kernel never indexes
        // outClean when WriteClean=0, so a 1-element dummy is enough.
        var cleanView = outClean ?? DummyFloat();

        _fbKernel!(viewCount * pixelsPerView, metric, conf, cams, cleanView, stats.View,
            new FbParams
            {
                Width = width,
                Height = height,
                ViewCount = viewCount,
                Subsample = subsample,
                MaxDepthError = maxDepthError,
                MaxReprojPx = maxReprojPx,
                MinViews = minViews,
                HasConf = hasConf ? 1 : 0,
                ConfMin = confMin,
                WriteClean = outClean.HasValue ? 1 : 0,
            });
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        // 5 ints. This is the scalar exemption, not a bulk readback.
        var host = await stats.CopyToHostAsync<int>(0, 8).ConfigureAwait(false);
        LastFbThreads = host[4];
        

        return new MvsGeometricFusion.FuseStats
        {
            Input = host[0],
            Kept = host[1],
            MinViewReject = host[2],
            ConfReject = host[3],
        };
    }

    /// <summary>
    /// Grid-search the multiplicative scale for one view that maximizes FB agreements.
    /// Reads back one int per scale step (16 by default), never a map.
    /// </summary>
    public async Task<float> OptimizeScaleFactorAsync(
        ArrayView1D<float, Stride1D.Dense> metric,
        ArrayView1D<float, Stride1D.Dense> cams,
        int view, int width, int height, int viewCount,
        float maxDepthError = 0.05f, float maxReprojPx = 2f, int probeSubsample = 8,
        float scaleStart = 0.85f, float scaleStep = 0.02f, int scaleSteps = 16)
    {
        EnsureKernels();
        int probeW = (width + probeSubsample - 1) / probeSubsample;
        int probeH = (height + probeSubsample - 1) / probeSubsample;
        using var counts = _accelerator.Allocate1D<int>(scaleSteps);
        counts.CopyFromCPU(new int[scaleSteps]);   // see the note in ForwardBackAsync

        _scaleProbeKernel!(probeW * probeH * scaleSteps, metric, cams, counts.View,
            new ScaleProbeParams
            {
                Width = width,
                Height = height,
                ViewCount = viewCount,
                TargetView = view,
                Subsample = probeSubsample,
                MaxDepthError = maxDepthError,
                MaxReprojPx = maxReprojPx,
                ScaleSteps = scaleSteps,
                ScaleStart = scaleStart,
                ScaleStep = scaleStep,
            });
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var host = await counts.CopyToHostAsync<int>(0, scaleSteps).ConfigureAwait(false);
        int best = 0;
        for (int s = 1; s < scaleSteps; s++)
            if (host[s] > host[best]) best = s;
        return scaleStart + best * scaleStep;
    }

    /// <summary>
    /// Sample (raw, zCam) at every anchor in every view. Returns [view][anchor] pairs; raw &lt; 0
    /// means the anchor missed that view. ViewCount * AnchorCount * 2 floats cross the boundary.
    /// </summary>
    public async Task<(float raw, float zCam)[][]> GatherAnchorsAsync(
        ArrayView1D<float, Stride1D.Dense> maps,
        ArrayView1D<float, Stride1D.Dense> cams,
        IReadOnlyList<Vector3> anchors,
        int width, int height, int viewCount)
    {
        EnsureKernels();
        var flat = new float[anchors.Count * 3];
        for (int a = 0; a < anchors.Count; a++)
        {
            flat[a * 3 + 0] = anchors[a].X;
            flat[a * 3 + 1] = anchors[a].Y;
            flat[a * 3 + 2] = anchors[a].Z;
        }

        using var anchorBuf = _accelerator.Allocate1D(flat);
        using var outBuf = _accelerator.Allocate1D<float>((long)viewCount * anchors.Count * 2);

        _anchorGatherKernel!(viewCount * anchors.Count, maps, cams, anchorBuf.View, outBuf.View,
            new AnchorGatherParams
            {
                Width = width,
                Height = height,
                ViewCount = viewCount,
                AnchorCount = anchors.Count,
            });
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        var host = await outBuf.CopyToHostAsync<float>(0, viewCount * anchors.Count * 2)
            .ConfigureAwait(false);

        var result = new (float, float)[viewCount][];
        for (int v = 0; v < viewCount; v++)
        {
            result[v] = new (float, float)[anchors.Count];
            for (int a = 0; a < anchors.Count; a++)
            {
                int k = (v * anchors.Count + a) * 2;
                result[v][a] = (host[k], host[k + 1]);
            }
        }
        return result;
    }

    public void Dispose()
    {
        _metricizeKernel = null;
        _fbKernel = null;
        _scaleProbeKernel = null;
        _scaleApplyKernel = null;
        _anchorGatherKernel = null;
        _warpScatterKernel = null;
        _warpMergeKernel = null;
        _fillIntKernel = null;
        _minMaxKernel = null;
        _scaleABuf?.Dispose(); _scaleABuf = null;
        _scaleBBuf?.Dispose(); _scaleBBuf = null;
        _scalesBuf?.Dispose(); _scalesBuf = null;
        _dummyBuf?.Dispose(); _dummyBuf = null;
    }
}
