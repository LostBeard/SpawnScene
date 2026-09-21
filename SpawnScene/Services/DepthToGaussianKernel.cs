using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnScene.Services;

/// <summary>
/// ILGPU kernel for GPU-only depth-to-Gaussian conversion with atomic compaction.
///
/// Pipeline:
///   1. Unproject depth + RGBA → one packed splat (see SplatFormat) per valid pixel.
///   2. Invalid pixels (bad depth range) are skipped entirely via Atomic.Add compaction.
///   3. 4-byte counter readback → actual valid splat count (no wasted slots in output buffer).
///   4. Optional edge-sharpening: depth gradient magnitude shrinks splat scale at edges.
/// </summary>
public class DepthToGaussianKernel
{
    private readonly GpuService _gpu;

    /// <summary>
    /// Kernel parameters for single-image / offset unprojection.
    /// ILGPU decomposes struct fields into scalar bindings — no GPU buffer allocation needed.
    /// </summary>
    public struct SplatParams
    {
        public int Width, Height;
        public float FocalX, FocalY;
        public float CenterX, CenterY;
        public int Subsample;
        public float MinDepth, MaxDepth;
        public float EdgeSharpness;
        // Exclusion rect: skip pixels inside this region (multi-view overlap avoidance)
        public int ExclX0, ExclY0, ExclX1, ExclY1;
        public float DepthScaleCorrection;
    }

    /// <summary>
    /// Kernel parameters for world-space unprojection using SfM-recovered camera pose.
    /// </summary>
    public struct SplatWorldParams
    {
        public int Width, Height;
        public float FocalX, FocalY;
        public float CenterX, CenterY;
        public int Subsample;
        public float MinDepth, MaxDepth;
        public float EdgeSharpness;
        // Rotation matrix R (3x3, row-major, world→camera)
        public float R00, R01, R02;
        public float R10, R11, R12;
        public float R20, R21, R22;
        // Camera world position
        public float PosX, PosY, PosZ;
        // Depth scale (maps relative MDE depth → world/SfM units): d = rawDepth * DepthScale
        public float DepthScale;
        /// <summary>1 = confidence buffer is valid W×H; 0 = ignore (dummy buffer).</summary>
        public int HasConfidence;
        /// <summary>Early-out when confidence below this (when HasConfidence=1).</summary>
        public float ConfMin;
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // depthValues
        ArrayView1D<int, Stride1D.Dense>,    // packedRGBA
        ArrayView1D<float, Stride1D.Dense>,  // outPacked (compacted)
        ArrayView1D<int, Stride1D.Dense>,    // counter [0] = valid splat count
        SplatParams>?                         // params struct
        _unprojectAndPackKernel;

    public DepthToGaussianKernel(GpuService gpu) => _gpu = gpu;

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernels
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: unproject depth + RGBA → compacted packed splat buffer.
    /// Only valid pixels write output (Atomic.Add compaction — no zero-opacity dummy splats).
    /// </summary>
    private static void UnprojectAndPackKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> depthValues,
        ArrayView1D<int, Stride1D.Dense> packedRGBA,
        ArrayView1D<float, Stride1D.Dense> outPacked,
        ArrayView1D<int, Stride1D.Dense> counter,
        SplatParams p)
    {
        int width = p.Width;
        int height = p.Height;
        float fx = p.FocalX; float fy = p.FocalY;
        float cx = p.CenterX; float cy = p.CenterY;
        int subsample = p.Subsample;
        float minDepth = p.MinDepth;
        float maxDepth = p.MaxDepth;
        float edgeSharpness = p.EdgeSharpness;

        int exclX0 = p.ExclX0; int exclY0 = p.ExclY0;
        int exclX1 = p.ExclX1; int exclY1 = p.ExclY1;
        float depthScaleCorr = p.DepthScaleCorrection;

        int globalIndex = index;

        int sampledW = width / subsample;
        int sx = globalIndex % sampledW;
        int sy = globalIndex / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;

        // Exclusion check: if this pixel maps inside the reference image's coverage, skip it.
        if (exclX1 > exclX0 && exclY1 > exclY0)
        {
            if (imgX >= exclX0 && imgX < exclX1 && imgY >= exclY0 && imgY < exclY1)
                return;
        }

        int imgIdx = imgY * width + imgX;

        float rawDepth = depthValues[imgIdx];

        // DAv3: direct relative depth (high = far). d = raw * scale (no disparity invert).
        float range = maxDepth - minDepth;
        float d = rawDepth * depthScaleCorr;

        // Validity check: skip non-positive / extreme depths
        if (rawDepth <= 1e-6f || d <= 0.01f || d >= 100f * MathF.Max(depthScaleCorr, 1e-3f)) return;

        // Per-splat scale: world size of one pixel at this depth
        float pixelScale = d * subsample / fx;
        float splatScale = pixelScale > 1e-6f ? pixelScale : 1e-6f;

        // Depth gradient for edge-adaptive scale + opacity + flying pixel removal
        float gradMag = 0f;
        if (edgeSharpness > 0f && range > 1e-6f)
        {
            int x0 = (imgX > 0) ? imgX - subsample : imgX;
            int x1 = (imgX + subsample < width) ? imgX + subsample : imgX;
            int y0 = (imgY > 0) ? imgY - subsample : imgY;
            int y1 = (imgY + subsample < height) ? imgY + subsample : imgY;

            float gx = (depthValues[imgY * width + x1] - depthValues[imgY * width + x0]) / range;
            float gy = (depthValues[y1 * width + imgX] - depthValues[y0 * width + imgX]) / range;
            gradMag = MathF.Sqrt(gx * gx + gy * gy);

            // Edge-adaptive scale: shrink splats at depth discontinuities
            splatScale /= (1f + gradMag * edgeSharpness);
        }

        // Edge-aware opacity: full opacity in smooth regions, slightly reduced at depth edges
        float alpha = 0.9f;
        if (gradMag > 0.05f)
            alpha = 0.9f - (gradMag - 0.05f) * 0.5f;
        if (alpha < 0.3f) alpha = 0.3f;

        int packed = packedRGBA[imgIdx];
        float r = (packed & 0xFF) / 255f;
        float g = ((packed >> 8) & 0xFF) / 255f;
        float b = ((packed >> 16) & 0xFF) / 255f;

        float posX = -((imgX - cx) * d / fx);
        float posY = -((imgY - cy) * d / fy);
        float posZ = d;

        // ── Surface orientation ──
        // Unproject the +x and +y neighbours in the SAME frame as posX/posY/posZ and take the
        // triangle normal. The splat is then a disk lying ON the surface rather than a disk
        // facing the camera, which is the whole point of carrying a rotation: at a grazing angle
        // an oriented disk foreshortens and a camera-facing one does not.
        int nX = (imgX + subsample < width) ? imgX + subsample : imgX;
        int nY = (imgY + subsample < height) ? imgY + subsample : imgY;
        float dnx = depthValues[imgY * width + nX] * depthScaleCorr;
        float dny = depthValues[nY * width + imgX] * depthScaleCorr;

        // A neighbour across a depth discontinuity would tilt the normal into the void. Fall back
        // to the centre depth there, which yields the camera-facing disk drawn before.
        float relX = MathF.Abs(dnx - d) / MathF.Max(d, 1e-6f);
        float relY = MathF.Abs(dny - d) / MathF.Max(d, 1e-6f);
        bool badX = relX > 0.05f;
        bool badY = relY > 0.05f;
        if (badX) dnx = d;
        if (badY) dny = d;

        float qPosX = -((nX - cx) * dnx / fx);
        float qPosY = -((imgY - cy) * dnx / fy);
        float rPosX = -((imgX - cx) * dny / fx);
        float rPosY = -((nY - cy) * dny / fy);

        var quat = SplatCovariance.NormalQuatFromNeighbors(
            posX, posY, posZ,
            qPosX, qPosY, dnx,
            rPosX, rPosY, dny);

        int slot = Atomic.Add(ref counter[0], 1);
        int outOff = slot * SplatFormat.Floats;

        outPacked[outOff + 0] = posX;
        outPacked[outOff + 1] = posY;
        outPacked[outOff + 2] = posZ;
        outPacked[outOff + 3] = r;
        outPacked[outOff + 4] = g;
        outPacked[outOff + 5] = b;
        outPacked[outOff + 6] = splatScale;
        outPacked[outOff + 7] = splatScale;
        outPacked[outOff + 8] = splatScale * SurfaceFlatten;
        outPacked[outOff + 9] = alpha;
        outPacked[outOff + 10] = quat.X;
        outPacked[outOff + 11] = quat.Y;
        outPacked[outOff + 12] = quat.Z;
        outPacked[outOff + 13] = quat.W;
    }

    /// <summary>
    /// Thickness of a depth-initialised splat along its surface normal, as a fraction of its
    /// in-plane radius. A depth sample measures a surface, not a blob: 1.0 (a sphere) is what
    /// produced the "giant voxel" look, and 0 is a zero-volume sheet that aliases. 0.15 is the
    /// DN-Splatter initialisation ratio.
    /// </summary>
    private const float SurfaceFlatten = 0.15f;

    /// <summary>
    /// Splats within this many pixels of the frame edge are dropped outright. Edge geometry is
    /// grazing, unsupported by neighbouring views, and the least reliable part of any depth map.
    /// </summary>
    private const float BorderCullPx = 16f;

    /// <summary>
    /// Minimum opacity worth keeping. Anything fainter composites as translucent haze over the
    /// background rather than reading as a surface, which is what the dark sheets were made of.
    /// </summary>
    private const float MinSplatAlpha = 0.25f;

    /// <summary>
    /// GPU kernel: unproject depth → camera space → world space using SfM/GT camera params.
    /// Name bumped (...V5) after confidence modulation + sharper radius (no full-subsample inflate).
    /// </summary>
    private static void UnprojectWorldSpaceKernelV5(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> depthValues,
        ArrayView1D<int, Stride1D.Dense> packedRGBA,
        ArrayView1D<float, Stride1D.Dense> confidence,
        ArrayView1D<float, Stride1D.Dense> outPacked,
        ArrayView1D<int, Stride1D.Dense> counter,
        SplatWorldParams p)
    {
        int width = p.Width;
        int height = p.Height;
        float fx = p.FocalX; float fy = p.FocalY;
        float cx = p.CenterX; float cy = p.CenterY;
        int subsample = p.Subsample;
        float edgeSharpness = p.EdgeSharpness;
        float minDepth = p.MinDepth;
        float maxDepth = p.MaxDepth;

        float r00 = p.R00; float r01 = p.R01; float r02 = p.R02;
        float r10 = p.R10; float r11 = p.R11; float r12 = p.R12;
        float r20 = p.R20; float r21 = p.R21; float r22 = p.R22;
        float tx = p.PosX; float ty = p.PosY; float tz = p.PosZ;
        float depthScale = p.DepthScale;

        int sampledW = width / subsample;
        int sampledH = height / subsample;
        int globalIndex = index;
        if (globalIndex >= sampledW * sampledH) return;

        int sx = globalIndex % sampledW;
        int sy = globalIndex / sampledW;
        int imgX = sx * subsample;
        int imgY = sy * subsample;
        if (imgX >= width || imgY >= height) return;
        int imgIdx = imgY * width + imgX;

        float conf = 1f;
        if (p.HasConfidence != 0)
        {
            conf = confidence[imgIdx];
            if (conf < p.ConfMin) return;
        }

        float rawDepth = depthValues[imgIdx];
        // Direct depth (DAv3): d = raw * scale. Min/Max = gradient range only (no cull band).
        float d = rawDepth * depthScale;
        if (!(d > 0.001f) || d >= 1000f) return;

        float range = maxDepth - minDepth;
        // Radius = 1× pixel footprint (coverage from density, not fattening).
        float pixelScale = d * MathF.Sqrt((float)subsample) / fx;
        float splatScale = pixelScale > 1e-6f ? pixelScale : 1e-6f;

        // ── Border: CULL, do not fade ──
        // A faded splat is not a fainter surface, it is a TRANSLUCENT one, and a ring of them
        // per view composites into exactly the dark "wing" sheets flanking the object. There is
        // no such thing as a half-existing surface: near the frame edge the geometry is
        // unsupported by any other view, so drop it. Other views cover that surface where it
        // is real. The old code faded over 16 px but only hard-culled within ~1.9 px, so the
        // whole ring survived at low alpha.
        float border = MathF.Min(
            MathF.Min((float)imgX, (float)(width - 1 - imgX)),
            MathF.Min((float)imgY, (float)(height - 1 - imgY)));
        if (border < BorderCullPx) return;

        float gradMag = 0f;
        if (edgeSharpness > 0f && range > 1e-6f)
        {
            int x0 = (imgX > 0) ? imgX - subsample : imgX;
            int x1 = (imgX + subsample < width) ? imgX + subsample : imgX;
            int y0 = (imgY > 0) ? imgY - subsample : imgY;
            int y1 = (imgY + subsample < height) ? imgY + subsample : imgY;

            float gx = (depthValues[imgY * width + x1] - depthValues[imgY * width + x0]) / range;
            float gy = (depthValues[y1 * width + imgX] - depthValues[y0 * width + imgX]) / range;
            gradMag = MathF.Sqrt(gx * gx + gy * gy);
            splatScale /= (1f + gradMag * edgeSharpness * 1.25f);
        }

        // Opacity from depth-edge confidence only. The old form multiplied by borderFade and
        // then, on the low branch, multiplied by it a SECOND time - squaring the suppression of
        // exactly the near-edge splats that then read as haze instead of disappearing.
        float alpha = 0.9f * conf;
        if (gradMag > 0.05f)
            alpha = (0.9f - (gradMag - 0.05f) * 0.5f) * conf;
        // Below this a splat contributes haze rather than surface: drop it instead of dimming it.
        if (alpha < MinSplatAlpha) return;
        int packed = packedRGBA[imgIdx];
        float r = (packed & 0xFF) / 255f;
        float g = ((packed >> 8) & 0xFF) / 255f;
        float b = ((packed >> 16) & 0xFF) / 255f;

        // OpenCV: X right, Y down, Z forward. c2w = [right|down|forward].
        float camX = (imgX - cx) * d / fx;
        float camY = (imgY - cy) * d / fy;
        float camZ = d;

        float worldX = r00 * camX + r10 * camY + r20 * camZ + tx;
        float worldY = r01 * camX + r11 * camY + r21 * camZ + ty;
        float worldZ = r02 * camX + r12 * camY + r22 * camZ + tz;

        // ── Surface orientation (camera space, then rotated into world) ──
        // Same recipe as the camera-space kernel: unproject the +x / +y neighbours, take the
        // triangle normal, fall back to the centre depth across a discontinuity.
        int nX = (imgX + subsample < width) ? imgX + subsample : imgX;
        int nY = (imgY + subsample < height) ? imgY + subsample : imgY;
        float dnx = depthValues[imgY * width + nX] * depthScale;
        float dny = depthValues[nY * width + imgX] * depthScale;

        float relX = MathF.Abs(dnx - d) / MathF.Max(d, 1e-6f);
        float relY = MathF.Abs(dny - d) / MathF.Max(d, 1e-6f);
        bool badX = relX > 0.05f;
        bool badY = relY > 0.05f;
        if (badX) dnx = d;
        if (badY) dny = d;

        var camQuat = SplatCovariance.NormalQuatFromNeighbors(
            camX, camY, camZ,
            (nX - cx) * dnx / fx, (imgY - cy) * dnx / fy, dnx,
            (imgX - cx) * dny / fx, (nY - cy) * dny / fy, dny);

        // p.R** are the world-to-camera ROWS (row0 = right, row1 = down, row2 = forward).
        var quat = SplatCovariance.RotateQuatToWorld(camQuat,
            r00, r01, r02,
            r10, r11, r12,
            r20, r21, r22);

        int slot = Atomic.Add(ref counter[0], 1);
        int outOff = slot * SplatFormat.Floats;
        outPacked[outOff + 0] = worldX;
        outPacked[outOff + 1] = worldY;
        outPacked[outOff + 2] = worldZ;
        outPacked[outOff + 3] = r;
        outPacked[outOff + 4] = g;
        outPacked[outOff + 5] = b;
        outPacked[outOff + 6] = splatScale;
        outPacked[outOff + 7] = splatScale;
        outPacked[outOff + 8] = splatScale * SurfaceFlatten;
        outPacked[outOff + 9] = alpha;
        outPacked[outOff + 10] = quat.X;
        outPacked[outOff + 11] = quat.Y;
        outPacked[outOff + 12] = quat.Z;
        outPacked[outOff + 13] = quat.W;
    }

    /// <summary>
    /// Widen a pre-rotation (10-float) packed buffer to the current layout.
    /// A legacy scene carries no surface information, so the honest widening is an ISOTROPIC
    /// Gaussian - sz forced up to sx and an identity rotation - which projects to exactly the
    /// circle the old billboard renderer drew. Inventing a normal here would fabricate geometry
    /// that was never measured.
    /// </summary>
    public struct WidenParams
    {
        public int Count;
        public int SrcStride;
    }

    private static void WidenPackedKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        WidenParams p)
    {
        int i = index;
        if (i >= p.Count) return;

        int so = i * p.SrcStride;
        int d = i * SplatFormat.Floats;

        dst[d + 0] = src[so + 0];
        dst[d + 1] = src[so + 1];
        dst[d + 2] = src[so + 2];
        dst[d + 3] = src[so + 3];
        dst[d + 4] = src[so + 4];
        dst[d + 5] = src[so + 5];

        float sx = src[so + 6];
        float sy = src[so + 7];
        float sMax = sx > sy ? sx : sy;
        dst[d + 6] = sx;
        dst[d + 7] = sy;
        dst[d + 8] = sMax;   // isotropic: no normal was ever recorded for this splat
        dst[d + 9] = src[so + 9];

        dst[d + 10] = 0f;
        dst[d + 11] = 0f;
        dst[d + 12] = 0f;
        dst[d + 13] = 1f;    // identity quaternion
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        WidenParams>? _widenPackedKernel;

    /// <summary>Apply p' = scale * R * p + t to packed splat xyz; scales radii by |scale|.</summary>
    public struct SimilarityParams
    {
        public float Scale;
        public float R00, R01, R02, R10, R11, R12, R20, R21, R22;
        public float TX, TY, TZ;
        public int Count;
    }

    private static void TransformPackedSimilarityKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packed,
        SimilarityParams p)
    {
        int i = index;
        if (i >= p.Count) return;
        int o = i * SplatFormat.Floats;
        float x = packed[o], y = packed[o + 1], z = packed[o + 2];
        float nx = p.Scale * (p.R00 * x + p.R01 * y + p.R02 * z) + p.TX;
        float ny = p.Scale * (p.R10 * x + p.R11 * y + p.R12 * z) + p.TY;
        float nz = p.Scale * (p.R20 * x + p.R21 * y + p.R22 * z) + p.TZ;
        packed[o] = nx; packed[o + 1] = ny; packed[o + 2] = nz;
        float absS = p.Scale >= 0 ? p.Scale : -p.Scale;
        packed[o + 6] *= absS;
        packed[o + 7] *= absS;
        packed[o + 8] *= absS;

        // Orientations ride along with the positions. Leaving them alone would keep every splat
        // facing the direction it had in the SOURCE frame, so an aligned cloud would render with
        // the surface normals of the frame it came from.
        var q = new SplatCovariance.Quat
        {
            X = packed[o + 10],
            Y = packed[o + 11],
            Z = packed[o + 12],
            W = packed[o + 13],
        };
        var rq = SplatCovariance.RotateQuatByMatrix(q,
            p.R00, p.R01, p.R02,
            p.R10, p.R11, p.R12,
            p.R20, p.R21, p.R22);
        packed[o + 10] = rq.X;
        packed[o + 11] = rq.Y;
        packed[o + 12] = rq.Z;
        packed[o + 13] = rq.W;
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        SplatWorldParams>? _unprojectWorldSpaceKernel;

    /// <summary>
    /// Project packed world-space splats into a reference camera; keep if depth agrees with ref
    /// (or splat is novel / higher-confidence than the ref pixel). Compacts survivors.
    /// </summary>
    public struct ConsistencyFuseParams
    {
        public int Width, Height;
        public float FocalX, FocalY, CenterX, CenterY;
        // OpenCV axes (right, down, forward) for world→camera: Xc = dot(axis, Xw-C)
        public float Rx, Ry, Rz;
        public float Dx, Dy, Dz;
        public float Fx, Fy, Fz;
        public float PosX, PosY, PosZ;
        public float DepthScale;
        public float RelThresh;
        public int SplatCount;
        public int HasConf;
        /// <summary>1 keeps splats the reference camera cannot see. See the oracle for why.</summary>
        public int KeepOutsideView;
    }

    /// <summary>
    /// CPU-oracle twin of the GPU consistency fuse — delegates to
    /// <see cref="WorldSpaceGeometry.ShouldKeepSplatVsRef"/>.
    ///
    /// 🔴 <b>The GPU kernel below does NOT call this.</b> It reimplements the same rule inline,
    /// so the two can drift and the tests only cover this one. Changing the keep/reject policy
    /// here alone is a no-op on every real scene - verified the hard way.
    /// </summary>
    public static bool ShouldKeepSplatVsRef(
        float zCam, float refDepthRaw, float splatConf, float refConf,
        float depthScale, float relThresh, bool inBounds, bool hasConf)
        => WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam, refDepthRaw, splatConf, refConf, depthScale, relThresh, inBounds, hasConf);

    private static void ConsistencyFuseKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packedIn,
        ArrayView1D<float, Stride1D.Dense> refDepth,
        ArrayView1D<float, Stride1D.Dense> refConf,
        ArrayView1D<float, Stride1D.Dense> packedOut,
        ArrayView1D<int, Stride1D.Dense> counter,
        ConsistencyFuseParams p)
    {
        int i = index;
        if (i >= p.SplatCount) return;
        int o = i * SplatFormat.Floats;
        float wx = packedIn[o];
        float wy = packedIn[o + 1];
        float wz = packedIn[o + 2];

        float dx = wx - p.PosX;
        float dy = wy - p.PosY;
        float dz = wz - p.PosZ;
        float xc = p.Rx * dx + p.Ry * dy + p.Rz * dz;
        float yc = p.Dx * dx + p.Dy * dy + p.Dz * dz;
        float zCam = p.Fx * dx + p.Fy * dy + p.Fz * dz;

        // Counters are the only way to see why a screen rejected something: it kept 3% on
        // Bathroom and nothing said whether the reference DISAGREED or simply could not see it.
        // Slots: 1 behind, 2 outside the reference view, 3 reference had no depth, 4 disagreed.
        bool keep;
        if (zCam <= 1e-6f)
        {
            keep = false; // behind ref camera — not visible overlap
            Atomic.Add(ref counter[1], 1);
        }
        else
        {
            float u = p.FocalX * xc / zCam + p.CenterX;
            float v = p.FocalY * yc / zCam + p.CenterY;
            int iu = (int)(u + 0.5f);
            int iv = (int)(v + 0.5f);
            bool inBounds = iu >= 0 && iu < p.Width && iv >= 0 && iv < p.Height;
            if (!inBounds)
            {
                // A splat the reference cannot see is UNVERIFIED, not wrong. Dropping it suits an
                // object every view looks at; for a room it discards the other walls, which are
                // the whole point of the extra views. Policy, measured per dataset.
                keep = p.KeepOutsideView != 0;
                if (!keep) Atomic.Add(ref counter[2], 1);
            }
            else
            {
                int pix = iv * p.Width + iu;
                float raw = refDepth[pix];
                float refZ = raw * p.DepthScale;
                if (!(refZ > 1e-4f))
                {
                    keep = false;
                    Atomic.Add(ref counter[3], 1);
                }
                else
                {
                    float denom = MathF.Max(refZ, zCam);
                    float rel = MathF.Abs(zCam - refZ) / denom;
                    keep = rel <= p.RelThresh;
                    if (!keep) Atomic.Add(ref counter[4], 1);
                }
            }
        }

        if (!keep) return;
        int slot = Atomic.Add(ref counter[0], 1);
        int d = slot * SplatFormat.Floats;
        for (int k = 0; k < SplatFormat.Floats; k++)
            packedOut[d + k] = packedIn[o + k];
    }

    /// <summary>Keep packed splats with ||xyz - center|| <= radius (atomic compact).</summary>
    public struct SphereCullParams
    {
        public float CX, CY, CZ, RadiusSq;
        public int Count;
    }

    private static void CullOutsideSphereKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packedIn,
        ArrayView1D<float, Stride1D.Dense> packedOut,
        ArrayView1D<int, Stride1D.Dense> counter,
        SphereCullParams p)
    {
        int i = index;
        if (i >= p.Count) return;
        int o = i * SplatFormat.Floats;
        float x = packedIn[o] - p.CX;
        float y = packedIn[o + 1] - p.CY;
        float z = packedIn[o + 2] - p.CZ;
        if (x * x + y * y + z * z > p.RadiusSq) return;
        int slot = Atomic.Add(ref counter[0], 1);
        int d = slot * SplatFormat.Floats;
        for (int k = 0; k < SplatFormat.Floats; k++)
            packedOut[d + k] = packedIn[o + k];
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, SimilarityParams>? _transformSimilarityKernel;
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        SphereCullParams>? _cullSphereKernel;
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ConsistencyFuseParams>? _consistencyFuseKernel;

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer from GPU-resident depth + CPU RGBA.
    /// Returns (packedBuf, validSplatCount) — ownership of packedBuf transfers to caller.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, ImportedImage image, int subsample = 2,
            float edgeSharpness = 0.3f, CameraParams? camera = null)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        // Upload RGBA to GPU — justified: image data from file/picker (CPU source boundary).
        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        return await RunUnprojectAsync(accelerator, depth, rgbaBuf.View, subsample, edgeSharpness, camera);
    }

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer from GPU-resident depth + GPU-resident RGBA.
    /// SR fast path — skips CPU→GPU upload.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferAsync(DepthResult depth, GpuImage gpuImage, int subsample = 2,
            float edgeSharpness = 0.3f, CameraParams? camera = null)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        return await RunUnprojectAsync(accelerator, depth, gpuImage.PackedRgba.View, subsample, edgeSharpness, camera);
    }

    /// <summary>
    /// Generate a compacted GPU-packed splat buffer in WORLD space using SfM-recovered camera parameters.
    /// Each splat position is transformed from camera-local to world coordinates via (R, t).
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferWorldSpaceAsync(DepthResult depth, ImportedImage image,
            CameraParams camera, int subsample = 2, float edgeSharpness = 0.3f,
            float depthScale = 1.0f)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        // Upload RGBA to GPU
        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        // Build c2w columns matching WorldSpaceGeometry.CameraToWorld (OpenCV axes).
        // Gate: SpawnScene.Tests TempleRingWorldSpaceTests.
        WorldSpaceGeometry.GetOpenCvAxes(camera, out var right, out var down, out var fwd);

        var worldParams = new SplatWorldParams
        {
            Width = w, Height = h,
            FocalX = camera.FocalX, FocalY = camera.FocalY,
            CenterX = camera.CenterX, CenterY = camera.CenterY,
            Subsample = subsample,
            MinDepth = depth.MinDepth,
            MaxDepth = depth.MaxDepth,
            EdgeSharpness = edgeSharpness,
            R00 = right.X, R01 = right.Y, R02 = right.Z,
            R10 = down.X, R11 = down.Y, R12 = down.Z,
            R20 = fwd.X, R21 = fwd.Y, R22 = fwd.Z,
            PosX = camera.Position.X, PosY = camera.Position.Y, PosZ = camera.Position.Z,
            DepthScale = depthScale,
            HasConfidence = depth.ConfidenceGpu != null ? 1 : 0,
            ConfMin = 0.12f,
        };

        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * SplatFormat.Floats);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null — GPU path requires GPU-resident depth.");

        // Dummy 1-element confidence when absent (kernel ignores it when HasConfidence=0).
        using var dummyConf = depth.ConfidenceGpu == null ? accelerator.Allocate1D<float>(1) : null;
        var confView = depth.ConfidenceGpu?.View ?? dummyConf!.View;

        _unprojectWorldSpaceKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            confView,
            outPackedBuf.View,
            counterBuf.View,
            worldParams);

        // Drain before counter readback (WebGPU batches; counter may still be 0 without this).
        await accelerator.SynchronizeAsync();

        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] World-space V5: {validCount:N0} splats, pos={camera.Position}, scale={depthScale:F4}, conf={worldParams.HasConfidence}");
        Console.WriteLine($"[DepthGPU]   R=[{right.X:F3},{right.Y:F3},{right.Z:F3} | {down.X:F3},{down.Y:F3},{down.Z:F3} | {fwd.X:F3},{fwd.Y:F3},{fwd.Z:F3}]");

        return (outPackedBuf, validCount);
    }

    /// <summary>
    /// Generate splats using the single-image pipeline but with a pixel offset applied.
    /// Used by multi-view fusion: the offset shifts each view's splats into the reference frame.
    /// The offset (dx, dy) represents the pixel displacement of this image relative to the reference.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        GeneratePackedGpuBufferWithOffsetAsync(DepthResult depth, ImportedImage image,
            float pixelOffsetX, float pixelOffsetY,
            int subsample = 2, float edgeSharpness = 0.3f,
            int refWidth = 0, int refHeight = 0,
            float depthScaleCorrection = 1.0f,
            CameraParams? camera = null,
            bool isDirectDepth = false)
    {
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        var packedRgba = MemoryMarshal.Cast<byte, int>(image.RgbaPixels.AsSpan()).ToArray();
        using var rgbaBuf = accelerator.Allocate1D(packedRgba);

        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        float fx = camera?.FocalX ?? MathF.Max(w, h) * 1.2f;
        float fy = camera?.FocalY ?? fx;
        float cx = (camera?.CenterX ?? w / 2f) + pixelOffsetX;
        float cy = (camera?.CenterY ?? h / 2f) + pixelOffsetY;

        // Compute exclusion rectangle: the region in THIS image that overlaps with the reference.
        // Pixel (x, y) in this image maps to reference pixel (x + dx, y + dy).
        // Overlap = where (x + dx) ∈ [0, refW) AND (y + dy) ∈ [0, refH)
        // → x ∈ [-dx, refW - dx) clamped to [0, w)
        int exclX0 = 0, exclY0 = 0, exclX1 = 0, exclY1 = 0;
        if (refWidth > 0 && refHeight > 0 && (pixelOffsetX != 0 || pixelOffsetY != 0))
        {
            exclX0 = Math.Clamp((int)(-pixelOffsetX), 0, w);
            exclY0 = Math.Clamp((int)(-pixelOffsetY), 0, h);
            exclX1 = Math.Clamp((int)(refWidth - pixelOffsetX), 0, w);
            exclY1 = Math.Clamp((int)(refHeight - pixelOffsetY), 0, h);
        }

        var splatParams = new SplatParams
        {
            Width = w, Height = h,
            FocalX = fx, FocalY = fy,
            CenterX = cx, CenterY = cy,
            Subsample = subsample,
            MinDepth = depth.MinDepth,
            MaxDepth = depth.MaxDepth,
            EdgeSharpness = edgeSharpness,
            ExclX0 = exclX0, ExclY0 = exclY0,
            ExclX1 = exclX1, ExclY1 = exclY1,
            DepthScaleCorrection = depthScaleCorrection,
        };

        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * SplatFormat.Floats);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null.");

        _unprojectAndPackKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaBuf.View,
            outPackedBuf.View,
            counterBuf.View,
            splatParams);

        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] Offset: {validCount:N0} splats (offset=({pixelOffsetX:F1},{pixelOffsetY:F1}), excl=[{exclX0},{exclY0}→{exclX1},{exclY1}], img={w}x{h})");

        return (outPackedBuf, validCount);
    }

    private void EnsureKernelLoaded(WebGPUAccelerator accelerator)
    {
        _unprojectAndPackKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            SplatParams>(UnprojectAndPackKernel);

        _unprojectWorldSpaceKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            SplatWorldParams>(UnprojectWorldSpaceKernelV5);

        _transformSimilarityKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            SimilarityParams>(TransformPackedSimilarityKernel);

        _cullSphereKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            SphereCullParams>(CullOutsideSphereKernel);

        _consistencyFuseKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ConsistencyFuseParams>(ConsistencyFuseKernel);

        _widenPackedKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            WidenParams>(WidenPackedKernel);
    }

    /// <summary>
    /// Convert a packed buffer written at an older stride to the current layout, on the GPU.
    /// Takes ownership of <paramref name="src"/> and disposes it. No managed-heap round trip:
    /// a 5K scene is hundreds of MB and reading it into a float[] OOMs WASM.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> WidenPackedAsync(
        MemoryBuffer1D<float, Stride1D.Dense> src, int splatCount, int srcStride)
    {
        if (srcStride == SplatFormat.Floats) return src;
        if (srcStride != ProjectScene.LegacyFloatsPerSplat)
            throw new InvalidOperationException(
                $"Unknown packed splat stride {srcStride}; expected {ProjectScene.LegacyFloatsPerSplat} or {SplatFormat.Floats}.");

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        var dst = accelerator.Allocate1D<float>((long)splatCount * SplatFormat.Floats);
        _widenPackedKernel!(splatCount, src.View, dst.View,
            new WidenParams { Count = splatCount, SrcStride = srcStride });
        await accelerator.SynchronizeAsync();
        src.Dispose();

        Console.WriteLine($"[DepthGPU] Widened {splatCount:N0} legacy splats {srcStride}→{SplatFormat.Floats} floats " +
            "(isotropic, identity rotation — no normals were recorded)");
        return dst;
    }

    /// <summary>
    /// Depth-consistency fuse vs a reference view: keep agreeing / novel / higher-conf challengers.
    /// Disposes <paramref name="packedIn"/>. Returns compacted survivors.
    ///
    /// <paramref name="keepOutsideView"/> keeps splats the reference camera cannot see. Dropping
    /// them suits an object every camera looks at; for a room it throws away the other walls.
    /// See <see cref="WorldSpaceGeometry.ClassifySplatVsRef"/> for the measurement behind that.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packed, int count)>
        FuseConsistencyVsRefAsync(
            MemoryBuffer1D<float, Stride1D.Dense> packedIn, int splatCount,
            DepthResult refDepth, CameraParams refCam, float depthScale,
            float relThresh = 0.12f, bool keepOutsideView = false)
    {
        if (splatCount <= 0) { packedIn.Dispose(); return (packedIn, 0); }
        if (refDepth.RawDepthGpu == null)
            throw new InvalidOperationException("Ref depth GPU buffer required for consistency fuse.");
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        WorldSpaceGeometry.GetOpenCvAxes(refCam, out var right, out var down, out var fwd);
        // Slot 0 is the compaction cursor; 1-4 count WHY a splat was rejected. Without them a
        // 3% keep rate says nothing about whether the reference disagreed or simply could not
        // see it, and those two want opposite treatment.
        using var counterBuf = accelerator.Allocate1D<int>(5);
        counterBuf.CopyFromCPU(new int[] { 0, 0, 0, 0, 0 });
        var packedOut = accelerator.Allocate1D<float>(splatCount * SplatFormat.Floats);

        using var dummyConf = refDepth.ConfidenceGpu == null ? accelerator.Allocate1D<float>(1) : null;
        var confView = refDepth.ConfidenceGpu?.View ?? dummyConf!.View;

        var fp = new ConsistencyFuseParams
        {
            Width = refDepth.Width, Height = refDepth.Height,
            FocalX = refCam.FocalX, FocalY = refCam.FocalY,
            CenterX = refCam.CenterX, CenterY = refCam.CenterY,
            Rx = right.X, Ry = right.Y, Rz = right.Z,
            Dx = down.X, Dy = down.Y, Dz = down.Z,
            Fx = fwd.X, Fy = fwd.Y, Fz = fwd.Z,
            PosX = refCam.Position.X, PosY = refCam.Position.Y, PosZ = refCam.Position.Z,
            DepthScale = depthScale,
            RelThresh = relThresh,
            SplatCount = splatCount,
            HasConf = refDepth.ConfidenceGpu != null ? 1 : 0,
            KeepOutsideView = keepOutsideView ? 1 : 0,
        };

        _consistencyFuseKernel!(splatCount,
            packedIn.View, refDepth.RawDepthGpu.View, confView,
            packedOut.View, counterBuf.View, fp);
        await accelerator.SynchronizeAsync();
        int[] c = await counterBuf.CopyToHostAsync<int>(0, 5);
        int kept = Math.Clamp(c[0], 0, splatCount);
        packedIn.Dispose();
        float frac = splatCount > 0 ? (float)kept / splatCount : 0f;
        Console.WriteLine(
            $"[DepthGPU] Consistency fuse: {kept:N0} / {splatCount:N0} kept ({frac:P0}, " +
            $"thresh={relThresh:F2}, conf={fp.HasConf}, outsideKept={keepOutsideView}) " +
            $"- rejected: {c[1]:N0} behind, {c[2]:N0} outside the reference view, " +
            $"{c[3]:N0} no reference depth, {c[4]:N0} depths disagree");
        return (packedOut, kept);
    }

    /// <summary>
    /// Compact splats inside a sphere (drops background sheets far from the object).
    /// Returns a new buffer; caller owns it. Disposes <paramref name="packedIn"/>.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> packed, int count)>
        CullOutsideSphereAsync(MemoryBuffer1D<float, Stride1D.Dense> packedIn, int splatCount,
            Vector3 center, float radius)
    {
        if (splatCount <= 0) { packedIn.Dispose(); return (packedIn, 0); }
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });
        var packedOut = accelerator.Allocate1D<float>(splatCount * SplatFormat.Floats);
        var sp = new SphereCullParams
        {
            CX = center.X, CY = center.Y, CZ = center.Z,
            RadiusSq = radius * radius,
            Count = splatCount,
        };
        _cullSphereKernel!(splatCount, packedIn.View, packedOut.View, counterBuf.View, sp);
        await accelerator.SynchronizeAsync();
        int[] c = await counterBuf.CopyToHostAsync<int>(0, 1);
        int kept = Math.Clamp(c[0], 0, splatCount);
        packedIn.Dispose();
        Console.WriteLine($"[DepthGPU] Sphere cull: {kept:N0} / {splatCount:N0} kept (r={radius:F3} @ {center})");
        return (packedOut, kept);
    }

    /// <summary>
    /// In-place similarity transform of packed splat buffer (xyz + radii).
    /// </summary>
    public async Task ApplySimilarityTransformAsync(
        MemoryBuffer1D<float, Stride1D.Dense> packed, int splatCount,
        float scale, Matrix4x4 rotation, Vector3 translation)
    {
        if (splatCount <= 0) return;
        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;
        EnsureKernelLoaded(accelerator);

        // rotation is packed for Vector3.Transform (row-vector); kernel uses column R*v.
        // Extract R from Numerics matrix so kernel gets r00=M11, r01=M21, r02=M31, ...
        var sp = new SimilarityParams
        {
            Scale = scale,
            R00 = rotation.M11, R01 = rotation.M21, R02 = rotation.M31,
            R10 = rotation.M12, R11 = rotation.M22, R12 = rotation.M32,
            R20 = rotation.M13, R21 = rotation.M23, R22 = rotation.M33,
            TX = translation.X, TY = translation.Y, TZ = translation.Z,
            Count = splatCount,
        };
        _transformSimilarityKernel!(splatCount, packed.View, sp);
        await accelerator.SynchronizeAsync();
    }

    /// <summary>
    /// Shared unprojection pipeline: GPU-resident depth + GPU-resident packed RGBA → compacted splat buffer.
    /// </summary>
    private async Task<(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)>
        RunUnprojectAsync(WebGPUAccelerator accelerator, DepthResult depth,
            ArrayView1D<int, Stride1D.Dense> rgbaView, int subsample, float edgeSharpness,
            CameraParams? camera = null)
    {
        int w = depth.Width;
        int h = depth.Height;
        int sampledW = w / subsample;
        int sampledH = h / subsample;
        int numPoints = sampledW * sampledH;

        float fx = camera?.FocalX ?? MathF.Max(w, h) * 1.2f;
        float fy = camera?.FocalY ?? fx;
        float cx = camera?.CenterX ?? w / 2f;
        float cy = camera?.CenterY ?? h / 2f;

        var splatParams = new SplatParams
        {
            Width = w, Height = h,
            FocalX = fx, FocalY = fy,
            CenterX = cx, CenterY = cy,
            Subsample = subsample,
            MinDepth = depth.MinDepth,
            MaxDepth = depth.MaxDepth,
            EdgeSharpness = edgeSharpness,
            DepthScaleCorrection = 1.0f,
        };

        // Atomic compaction counter
        using var counterBuf = accelerator.Allocate1D<int>(1);
        counterBuf.CopyFromCPU(new int[] { 0 });

        // Output buffer: worst case all pixels are valid (over-allocated, compacted on GPU).
        // Ownership transfers to caller → GpuSplatSorter.
        var outPackedBuf = accelerator.Allocate1D<float>(numPoints * SplatFormat.Floats);

        if (depth.RawDepthGpu == null)
            throw new InvalidOperationException("DepthResult.RawDepthGpu is null — GPU path requires GPU-resident depth.");

        _unprojectAndPackKernel!(numPoints,
            depth.RawDepthGpu.View,
            rgbaView,
            outPackedBuf.View,
            counterBuf.View,
            splatParams);

        // Readback valid splat count only (4 bytes)
        int[] counterResult = await counterBuf.CopyToHostAsync<int>(0, 1);
        int validCount = Math.Clamp(counterResult[0], 0, numPoints);

        Console.WriteLine($"[DepthGPU] Compacted: {validCount:N0} valid / {numPoints:N0} candidate splats " +
            $"(subsample={subsample}, edgeSharpness={edgeSharpness:F2})");

        return (outPackedBuf, validCount);
    }
}
