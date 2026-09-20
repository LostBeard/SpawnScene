using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// COLMAP / MVSNet-style multi-view geometric fusion (CPU oracle).
/// Defaults match COLMAP StereoFusion + MVSNet forward–back:
///   max relative depth error = 1%, max FB reproj = 2 px, min agreeing views = 2.
/// Gate: SpawnScene.Tests MvsGeometricFusionTests.
/// </summary>
public static class MvsGeometricFusion
{
    /// <summary>COLMAP StereoFusion.max_depth_error</summary>
    public const float DefaultMaxDepthError = 0.01f;

    /// <summary>COLMAP / MVSNet-style forward–back pixel budget (slightly looser than MVSNet's 1px).</summary>
    public const float DefaultMaxReprojPx = 2.0f;

    /// <summary>Minimum agreeing views including the source (COLMAP filter_min_num_consistent ≈ 2).</summary>
    public const int DefaultMinViews = 2;

    public struct FuseStats
    {
        public int Input;
        public int Kept;
        public int DepthReject;
        public int FbReject;
        public int MinViewReject;
        public int ConfReject;
    }

    /// <summary>
    /// Forward–back geometric consistency between source view <paramref name="src"/> and
    /// neighbor <paramref name="nbr"/> for a pixel in src with metric depth <paramref name="srcDepth"/>.
    /// </summary>
    public static bool IsForwardBackConsistent(
        CameraParams src, CameraParams nbr,
        float uSrc, float vSrc, float srcDepth,
        float[] nbrDepthMap, int nbrW, int nbrH,
        float maxDepthError = DefaultMaxDepthError,
        float maxReprojPx = DefaultMaxReprojPx)
    {
        if (!(srcDepth > 1e-6f)) return false;

        var world = WorldSpaceGeometry.UnprojectPixel(src, uSrc, vSrc, srcDepth);
        if (!WorldSpaceGeometry.Project(nbr, world, out float uNbr, out float vNbr, out float zProj))
            return false;
        if (zProj <= 1e-6f) return false;

        int iu = (int)MathF.Round(uNbr);
        int iv = (int)MathF.Round(vNbr);
        if (iu < 0 || iu >= nbrW || iv < 0 || iv >= nbrH) return false;

        float dNbr = nbrDepthMap[iv * nbrW + iu];
        if (!(dNbr > 1e-6f)) return false;

        float denom = MathF.Max(zProj, dNbr);
        float rel = MathF.Abs(zProj - dNbr) / denom;
        if (rel > maxDepthError) return false;

        // Forward–back: unproject nbr sample → reproject into src.
        var world2 = WorldSpaceGeometry.UnprojectPixel(nbr, iu + 0.5f, iv + 0.5f, dNbr);
        if (!WorldSpaceGeometry.Project(src, world2, out float uBack, out float vBack, out float zBack))
            return false;
        if (zBack <= 1e-6f) return false;

        float du = uBack - uSrc;
        float dv = vBack - vSrc;
        float pixErr = MathF.Sqrt(du * du + dv * dv);
        return pixErr <= maxReprojPx;
    }

    /// <summary>
    /// Count how many views (including src) agree with the src pixel via forward–back.
    /// </summary>
    public static int CountAgreeingViews(
        int srcView,
        float uSrc, float vSrc, float srcDepth,
        IReadOnlyList<CameraParams> cams,
        IReadOnlyList<float[]> depthMaps,
        int width, int height,
        float maxDepthError = DefaultMaxDepthError,
        float maxReprojPx = DefaultMaxReprojPx)
    {
        int agree = 1; // self
        for (int j = 0; j < cams.Count; j++)
        {
            if (j == srcView) continue;
            if (IsForwardBackConsistent(
                    cams[srcView], cams[j], uSrc, vSrc, srcDepth,
                    depthMaps[j], width, height, maxDepthError, maxReprojPx))
                agree++;
        }
        return agree;
    }

    /// <summary>
    /// Fuse metric depth maps under GT cameras: keep pixels that pass FB consistency with ≥ minViews.
    /// Returns world-space points (one per kept src pixel from all views; no dedupe pass yet).
    /// </summary>
    public static List<Vector3> FuseDepthMaps(
        IReadOnlyList<CameraParams> cams,
        IReadOnlyList<float[]> depthMaps,
        int width, int height,
        out FuseStats stats,
        int subsample = 4,
        float maxDepthError = DefaultMaxDepthError,
        float maxReprojPx = DefaultMaxReprojPx,
        int minViews = DefaultMinViews,
        IReadOnlyList<float[]>? confMaps = null,
        float confMin = 0f)
    {
        stats = default;
        var kept = new List<Vector3>();
        if (cams.Count == 0 || depthMaps.Count != cams.Count) return kept;

        for (int i = 0; i < cams.Count; i++)
        {
            var depth = depthMaps[i];
            var conf = confMaps != null && i < confMaps.Count ? confMaps[i] : null;
            for (int y = 0; y < height; y += subsample)
            for (int x = 0; x < width; x += subsample)
            {
                int pix = y * width + x;
                float d = depth[pix];
                if (!(d > 1e-6f)) continue;
                stats.Input++;

                if (conf != null && conf[pix] < confMin)
                {
                    stats.ConfReject++;
                    continue;
                }

                int agree = CountAgreeingViews(
                    i, x + 0.5f, y + 0.5f, d, cams, depthMaps, width, height,
                    maxDepthError, maxReprojPx);

                if (agree < minViews)
                {
                    stats.MinViewReject++;
                    continue;
                }

                stats.Kept++;
                kept.Add(WorldSpaceGeometry.UnprojectPixel(cams[i], x + 0.5f, y + 0.5f, d));
            }
        }

        return kept;
    }

    /// <summary>
    /// COLMAP-style depth-map cleaning: keep metric depth only where FB agrees with ≥ minViews;
    /// replace kept pixels with the median of source + reprojected neighbor depths.
    /// Non-agreeing pixels → 0. Optionally densify by filling holes near kept cores when
    /// source depth is within <paramref name="densifyDepthError"/> of a kept neighbor.
    /// </summary>
    public static float[][] CleanDepthMaps(
        IReadOnlyList<CameraParams> cams,
        IReadOnlyList<float[]> depthMaps,
        int width, int height,
        out FuseStats stats,
        float maxDepthError = DefaultMaxDepthError,
        float maxReprojPx = DefaultMaxReprojPx,
        int minViews = DefaultMinViews,
        int densifyRadius = 2,
        float densifyDepthError = 0.05f)
    {
        stats = default;
        int n = cams.Count;
        var cleaned = new float[n][];
        for (int i = 0; i < n; i++)
            cleaned[i] = new float[width * height];

        var zsBuf = new List<float>(n);
        for (int i = 0; i < n; i++)
        {
            var depth = depthMaps[i];
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int pix = y * width + x;
                float d = depth[pix];
                if (!(d > 1e-6f)) continue;
                stats.Input++;

                zsBuf.Clear();
                zsBuf.Add(d);
                int agree = 1;
                var worldSrc = WorldSpaceGeometry.UnprojectPixel(cams[i], x + 0.5f, y + 0.5f, d);
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    if (!IsForwardBackConsistent(
                            cams[i], cams[j], x + 0.5f, y + 0.5f, d,
                            depthMaps[j], width, height, maxDepthError, maxReprojPx))
                        continue;
                    agree++;
                    if (!WorldSpaceGeometry.Project(cams[j], worldSrc, out float uN, out float vN, out _)
                        || !(uN >= 0) || !(vN >= 0))
                        continue;
                    int iu = (int)MathF.Round(uN), iv = (int)MathF.Round(vN);
                    if (iu < 0 || iu >= width || iv < 0 || iv >= height) continue;
                    float dN = depthMaps[j][iv * width + iu];
                    if (!(dN > 1e-6f)) continue;
                    var worldN = WorldSpaceGeometry.UnprojectPixel(cams[j], iu + 0.5f, iv + 0.5f, dN);
                    if (WorldSpaceGeometry.Project(cams[i], worldN, out _, out _, out float zBack) && zBack > 1e-6f)
                        zsBuf.Add(zBack);
                }

                if (agree < minViews)
                {
                    stats.MinViewReject++;
                    continue;
                }
                stats.Kept++;
                zsBuf.Sort();
                cleaned[i][pix] = zsBuf[zsBuf.Count / 2];
            }
        }

        if (densifyRadius > 0)
        {
            int filled = 0;
            for (int i = 0; i < n; i++)
            {
                var depth = depthMaps[i];
                var outMap = cleaned[i];
                var next = (float[])outMap.Clone();
                for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int pix = y * width + x;
                    if (outMap[pix] > 1e-6f) continue;
                    float src = depth[pix];
                    if (!(src > 1e-6f)) continue;

                    bool ok = false;
                    for (int dy = -densifyRadius; dy <= densifyRadius && !ok; dy++)
                    for (int dx = -densifyRadius; dx <= densifyRadius; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        float nd = outMap[ny * width + nx];
                        if (!(nd > 1e-6f)) continue;
                        float rel = MathF.Abs(src - nd) / MathF.Max(src, nd);
                        if (rel <= densifyDepthError) { ok = true; break; }
                    }
                    if (ok) { next[pix] = src; filled++; }
                }
                cleaned[i] = next;
            }
            Console.WriteLine($"[MvsFusion] densify filled={filled} (r={densifyRadius}, err={densifyDepthError:P0})");
        }

        // Pass 3: cross-view warp fill — project cleaned cores into empty pixels of other views
        {
            int warped = 0;
            var nextMaps = cleaned.Select(m => (float[])m.Clone()).ToArray();
            for (int src = 0; src < n; src++)
            {
                for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    float d = cleaned[src][y * width + x];
                    if (!(d > 1e-6f)) continue;
                    var world = WorldSpaceGeometry.UnprojectPixel(cams[src], x + 0.5f, y + 0.5f, d);
                    for (int dst = 0; dst < n; dst++)
                    {
                        if (dst == src) continue;
                        if (!WorldSpaceGeometry.Project(cams[dst], world, out float u, out float v, out float z)
                            || z <= 1e-6f) continue;
                        int iu = (int)MathF.Round(u), iv = (int)MathF.Round(v);
                        if (iu < 0 || iu >= width || iv < 0 || iv >= height) continue;
                        int dp = iv * width + iu;
                        if (cleaned[dst][dp] > 1e-6f) continue; // destination already has a cleaned core
                        // Only fill if source metric depth at dst (if any) agrees, OR dst was empty of signal
                        float dstSrc = depthMaps[dst][dp];
                        if (dstSrc > 1e-6f)
                        {
                            float rel = MathF.Abs(dstSrc - z) / MathF.Max(dstSrc, z);
                            if (rel > densifyDepthError) continue;
                        }
                        // ⚠️ NEAREST WINS. This took the FIRST writer in source-then-raster order until
                        // 2026-09-20, which no GPU can reproduce and which is wrong anyway: when a near
                        // and a far surface land on the same destination pixel, first-wins can paint the
                        // OCCLUDED one into the depth map. A warp fill is a z-buffer.
                        // MvsFusionGpu.WarpScatterKernel does the same thing by atomic min on the float
                        // bits; gate is MvsFusionGpuTests.CleanedDepth_MatchesCpuOracle.
                        float prev = nextMaps[dst][dp];
                        if (prev > 1e-6f && prev <= z) continue;
                        if (!(prev > 1e-6f)) warped++;
                        nextMaps[dst][dp] = z;
                    }
                }
            }
            cleaned = nextMaps;
            Console.WriteLine($"[MvsFusion] cross-view warp filled={warped}");
        }

        return cleaned;
    }

    /// <summary>
    /// Grid-search multiplicative scale on view <paramref name="view"/>'s metric map
    /// that maximizes FB agreements vs other views. Returns refined scale factor
    /// to multiply into the existing metric depths (and into a0 if tracking scales).
    /// </summary>
    public static float OptimizeScaleFactor(
        int view,
        IReadOnlyList<CameraParams> cams,
        IReadOnlyList<float[]> metricMaps,
        int width, int height,
        float maxDepthError = 0.05f,
        float maxReprojPx = DefaultMaxReprojPx,
        int probeSubsample = 8)
    {
        int bestCount = -1;
        float bestF = 1f;
        var scaled = new float[width * height];
        var maps = new float[cams.Count][];
        for (int i = 0; i < cams.Count; i++)
            maps[i] = metricMaps[i];

        for (float f = 0.85f; f <= 1.15f + 1e-6f; f += 0.02f)
        {
            var src = metricMaps[view];
            for (int i = 0; i < src.Length; i++)
                scaled[i] = src[i] > 1e-6f ? src[i] * f : 0f;
            maps[view] = scaled;

            int count = 0;
            for (int y = 0; y < height; y += probeSubsample)
            for (int x = 0; x < width; x += probeSubsample)
            {
                float d = scaled[y * width + x];
                if (!(d > 1e-6f)) continue;
                if (CountAgreeingViews(view, x + 0.5f, y + 0.5f, d, cams, maps, width, height,
                        maxDepthError, maxReprojPx) >= 2)
                    count++;
            }
            if (count > bestCount) { bestCount = count; bestF = f; }
        }
        maps[view] = metricMaps[view];
        return bestF;
    }

    // ─── DN-Splatter affine scale ─────────────────────────────────

    /// <summary>
    /// Closed-form least-squares fit of <c>z ≈ a·r + b</c> (DN-Splatter / MiDaS SSI).
    /// Returns false if underdetermined.
    /// </summary>
    public static bool TryFitAffineScaleShift(
        IReadOnlyList<float> rawDepths, IReadOnlyList<float> metricDepths,
        out float a, out float b)
    {
        a = 1f; b = 0f;
        int n = Math.Min(rawDepths.Count, metricDepths.Count);
        if (n < 2) return false;

        double sumR = 0, sumZ = 0, sumRR = 0, sumRZ = 0;
        int used = 0;
        for (int i = 0; i < n; i++)
        {
            float r = rawDepths[i], z = metricDepths[i];
            if (!(r > 1e-6f) || !(z > 1e-6f)) continue;
            sumR += r; sumZ += z; sumRR += r * (double)r; sumRZ += r * (double)z;
            used++;
        }
        if (used < 2) return false;

        double det = used * sumRR - sumR * sumR;
        if (Math.Abs(det) < 1e-12) return false;
        a = (float)((used * sumRZ - sumR * sumZ) / det);
        b = (float)((sumZ * sumRR - sumR * sumRZ) / det);
        if (!(a > 1e-8f)) return false;
        return true;
    }

    /// <summary>Scale-only fit <c>z ≈ a·r</c> (b=0) — median of z/r.</summary>
    public static bool TryFitScaleOnly(
        IReadOnlyList<float> rawDepths, IReadOnlyList<float> metricDepths, out float a)
    {
        a = 1f;
        var ratios = new List<float>();
        int n = Math.Min(rawDepths.Count, metricDepths.Count);
        for (int i = 0; i < n; i++)
        {
            float r = rawDepths[i], z = metricDepths[i];
            if (r > 1e-6f && z > 1e-6f) ratios.Add(z / r);
        }
        if (ratios.Count == 0) return false;
        ratios.Sort();
        a = ratios[ratios.Count / 2];
        return a > 1e-8f;
    }

    /// <summary>
    /// COLMAP-style voxel average: merge points that fall in the same voxel (reduces multi-view doubles).
    /// </summary>
    public static List<Vector3> VoxelAverage(IReadOnlyList<Vector3> points, float voxelSize)
    {
        if (points.Count == 0 || voxelSize <= 1e-8f) return points.ToList();
        var buckets = new Dictionary<long, (Vector3 sum, int n)>();
        float inv = 1f / voxelSize;
        foreach (var p in points)
        {
            int ix = (int)MathF.Floor(p.X * inv);
            int iy = (int)MathF.Floor(p.Y * inv);
            int iz = (int)MathF.Floor(p.Z * inv);
            // Pack signed coords into a long key
            long key = ((long)(ix + 500000) & 0xFFFFF)
                     | (((long)(iy + 500000) & 0xFFFFF) << 20)
                     | (((long)(iz + 500000) & 0xFFFFF) << 40);
            if (buckets.TryGetValue(key, out var acc))
                buckets[key] = (acc.sum + p, acc.n + 1);
            else
                buckets[key] = (p, 1);
        }
        var result = new List<Vector3>(buckets.Count);
        foreach (var (_, acc) in buckets)
            result.Add(acc.sum / acc.n);
        return result;
    }

    // ─── Synthetic helpers (Stage 0) ───────────────────────────────

    public static List<Vector3> SampleSphereCloud(Vector3 center, float radius, int count)
    {
        var pts = new List<Vector3>(count);
        // Fibonacci sphere
        const float golden = 2.399963f; // ≈ π(3-√5)
        for (int i = 0; i < count; i++)
        {
            float y = 1f - (i / (float)(count - 1)) * 2f;
            float r = MathF.Sqrt(MathF.Max(0f, 1f - y * y));
            float theta = golden * i;
            pts.Add(center + new Vector3(MathF.Cos(theta) * r, y, MathF.Sin(theta) * r) * radius);
        }
        return pts;
    }

    /// <summary>
    /// Soft z-buffer depth map from a point cloud (metric Z in camera space).
    /// Invalid = 0. Uses nearest splat footprint of 1 px.
    /// </summary>
    public static float[] RenderDepthMap(CameraParams cam, IReadOnlyList<Vector3> cloud, int width, int height)
    {
        var depth = new float[width * height];
        // leave zeros = invalid
        for (int i = 0; i < cloud.Count; i++)
        {
            if (!WorldSpaceGeometry.Project(cam, cloud[i], out float u, out float v, out float z))
                continue;
            if (z <= 1e-6f) continue;
            int x = (int)MathF.Round(u);
            int y = (int)MathF.Round(v);
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            int pix = y * width + x;
            if (depth[pix] <= 0f || z < depth[pix])
                depth[pix] = z;
        }
        return depth;
    }

    /// <summary>Mean nearest-neighbor distance from <paramref name="a"/> to <paramref name="b"/> (one-sided Hausdorff proxy).</summary>
    public static float MeanNearestNeighbor(IReadOnlyList<Vector3> a, IReadOnlyList<Vector3> b)
    {
        if (a.Count == 0 || b.Count == 0) return float.PositiveInfinity;
        double sum = 0;
        for (int i = 0; i < a.Count; i++)
        {
            float best = float.MaxValue;
            for (int j = 0; j < b.Count; j++)
            {
                float d = Vector3.Distance(a[i], b[j]);
                if (d < best) best = d;
            }
            sum += best;
        }
        return (float)(sum / a.Count);
    }

    /// <summary>Symmetric mean NN (approx Hausdorff mean).</summary>
    public static float SymmetricMeanNn(IReadOnlyList<Vector3> a, IReadOnlyList<Vector3> b)
        => 0.5f * (MeanNearestNeighbor(a, b) + MeanNearestNeighbor(b, a));
}
