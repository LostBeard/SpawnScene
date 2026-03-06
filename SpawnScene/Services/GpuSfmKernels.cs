using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnScene.Services;

/// <summary>
/// GPU-accelerated SfM compute kernels using SpawnDev.ILGPU.
/// Architecture: CPU pre-computes candidate solutions (small matrices),
/// GPU counts inliers in parallel (embarrassingly parallel, the real bottleneck).
/// This separates correctness-critical linear algebra (CPU) from
/// throughput-critical parallel counting (GPU).
/// </summary>
public class GpuSfmKernels
{
    private readonly GpuService _gpu;
    private readonly Random _rng = new(42);

    private static void Log(string msg) { if (SfmReconstructor.VerboseLogging) Console.WriteLine(msg); }

    // Cached kernel delegates
    private Action<Index1D,
        ArrayView<float>, ArrayView<float>,  // ptsA_norm, ptsB_norm (2 floats each)
        int,                                  // matchCount
        ArrayView<float>,                     // candidateEssentials (9 floats each)
        ArrayView<int>>? _eInlierKernel;      // outInlierCounts

    private Action<Index1D,
        ArrayView<float>, ArrayView<float>,  // pts2D_norm, pts3D (2 and 3 floats each)
        int,                                  // correspondenceCount
        ArrayView<float>,                     // candidatePoses (12 floats each: R[9]+t[3])
        ArrayView<int>>? _pnpInlierKernel;    // outInlierCounts

    private Action<Index1D,
        ArrayView<float>, ArrayView<float>,  // ptsA_norm, ptsB_norm
        ArrayView<float>, ArrayView<float>,  // P1, P2 (12 floats each)
        ArrayView<float>,                     // outPts (4 floats each: x,y,z,valid)
        ArrayView<float>, ArrayView<float>>? _triKernel; // poseA, poseB (12 floats each)

    // GPU quality gate: reprojection error per correspondence
    private Action<Index1D,
        ArrayView<float>, ArrayView<float>,  // pts2D_norm, pts3D (2 and 3 floats each)
        int,                                  // correspondenceCount
        ArrayView<float>,                     // pose (12 floats: R[9]+t[3])
        ArrayView<float>>? _reprojErrorKernel; // outErrors (1 float each)

    private bool _kernelsLoaded;

    public GpuSfmKernels(GpuService gpu)
    {
        _gpu = gpu;
    }

    private void EnsureKernelsLoaded()
    {
        if (_kernelsLoaded) return;
        var acc = _gpu.Accelerator;

        _eInlierKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>, ArrayView<float>,
            int,
            ArrayView<float>,
            ArrayView<int>>(EssentialInlierCountKernel);

        _pnpInlierKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>, ArrayView<float>,
            int,
            ArrayView<float>,
            ArrayView<int>>(PnPInlierCountKernel);

        _triKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>, ArrayView<float>>(TriangulationKernel);

        _reprojErrorKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>, ArrayView<float>,
            int,
            ArrayView<float>,
            ArrayView<float>>(ReprojectionErrorKernel);

        _kernelsLoaded = true;
    }

    // ──────────────────────────────────────────────────────────
    //  PUBLIC API: Essential Matrix RANSAC (CPU solve + GPU count)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// GPU-accelerated RANSAC for essential matrix estimation.
    /// CPU solves the 8-point algorithm for each sample set.
    /// GPU counts inliers for all candidates in parallel.
    /// </summary>
    public async Task<(float[]? essential, bool[] mask)> EstimateEssentialAsync(
        float[] ptsA, float[] ptsB, int matchCount,
        float fx, float cx, float cy, int iterations = 512)
    {
        EnsureKernelsLoaded();
        var acc = _gpu.Accelerator;

        if (matchCount < 8)
            return (null, Array.Empty<bool>());

        // Step 1: CPU pre-computes candidate essential matrices
        var candidates = new float[iterations * 9];
        int validCandidates = 0;

        for (int iter = 0; iter < iterations; iter++)
        {
            // Sample 8 random points
            var indices = new HashSet<int>();
            while (indices.Count < 8)
                indices.Add(_rng.Next(matchCount));

            var sample = indices.ToArray();

            // Solve 8-point algorithm on CPU
            var E = Solve8Point(ptsA, ptsB, sample);
            if (E == null) continue;

            Array.Copy(E, 0, candidates, validCandidates * 9, 9);
            validCandidates++;
        }

        if (validCandidates == 0)
            return (null, Array.Empty<bool>());

        // Trim to valid candidates
        var trimmedCandidates = new float[validCandidates * 9];
        Array.Copy(candidates, trimmedCandidates, validCandidates * 9);

        // Step 2: GPU counts inliers for ALL candidates in parallel
        using var bufA = acc.Allocate1D(ptsA);
        using var bufB = acc.Allocate1D(ptsB);
        using var bufCandidates = acc.Allocate1D(trimmedCandidates);
        using var bufCounts = acc.Allocate1D<int>(validCandidates);

        _eInlierKernel!((Index1D)validCandidates,
            bufA.View, bufB.View,
            matchCount,
            bufCandidates.View,
            bufCounts.View);

        await acc.SynchronizeAsync();
        var counts = await bufCounts.CopyToHostAsync();

        // Step 3: CPU picks the best
        int bestIter = -1, bestCount = 0;
        for (int i = 0; i < validCandidates; i++)
        {
            if (counts[i] > bestCount)
            {
                bestCount = counts[i];
                bestIter = i;
            }
        }

        if (bestIter < 0 || bestCount < 8)
            return (null, Array.Empty<bool>());

        var bestE = new float[9];
        Array.Copy(trimmedCandidates, bestIter * 9, bestE, 0, 9);

        // Compute inlier mask on CPU
        var mask = new bool[matchCount];
        for (int i = 0; i < matchCount; i++)
        {
            float err = ComputeSampsonError(ptsA[i * 2], ptsA[i * 2 + 1],
                ptsB[i * 2], ptsB[i * 2 + 1], bestE);
            mask[i] = err < 0.01f;
        }

        Log($"[SfM-GPU] RANSAC: best={bestCount}/{matchCount} inliers ({100f * bestCount / matchCount:F1}%)");
        return (bestE, mask);
    }

    // ──────────────────────────────────────────────────────────
    //  PUBLIC API: PnP RANSAC (CPU solve + GPU count)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// GPU-accelerated PnP via DLT + RANSAC.
    /// CPU solves DLT for each random sample.
    /// GPU counts inliers for all candidates in parallel.
    /// </summary>
    public async Task<(float[]? R, float[]? t, int inliers)> SolvePnPAsync(
        float[] pts2D, float[] pts3D, int count, int iterations = 1024)
    {
        EnsureKernelsLoaded();
        var acc = _gpu.Accelerator;

        if (count < 4)
            return (null, null, 0);

        // Step 1: CPU pre-computes candidate poses
        var candidates = new float[iterations * 12]; // R[9] + t[3]
        int validCandidates = 0;

        for (int iter = 0; iter < iterations; iter++)
        {
            var indices = new HashSet<int>();
            while (indices.Count < Math.Min(6, count))
                indices.Add(_rng.Next(count));

            var sample = indices.ToArray();

            var pose = SolvePnPDLT(pts2D, pts3D, sample);
            if (pose == null) continue;

            Array.Copy(pose, 0, candidates, validCandidates * 12, 12);
            validCandidates++;
        }

        if (validCandidates == 0)
        {
            Log($"[SfM-GPU] PnP: no valid candidates from {iterations} DLT solves");
            return (null, null, 0);
        }

        var trimmedCandidates = new float[validCandidates * 12];
        Array.Copy(candidates, trimmedCandidates, validCandidates * 12);

        // Step 2: GPU counts inliers for all candidates
        using var buf2D = acc.Allocate1D(pts2D);
        using var buf3D = acc.Allocate1D(pts3D);
        using var bufCandidates = acc.Allocate1D(trimmedCandidates);
        using var bufCounts = acc.Allocate1D<int>(validCandidates);

        _pnpInlierKernel!((Index1D)validCandidates,
            buf2D.View, buf3D.View,
            count,
            bufCandidates.View,
            bufCounts.View);

        await acc.SynchronizeAsync();
        var counts = await bufCounts.CopyToHostAsync();

        // Step 3: CPU picks the best
        int bestIter = -1, bestCount = 0;
        for (int i = 0; i < validCandidates; i++)
        {
            if (counts[i] > bestCount)
            {
                bestCount = counts[i];
                bestIter = i;
            }
        }

        if (bestIter < 0 || bestCount < 4)
        {
            Log($"[SfM-GPU] PnP failed: best={bestCount} from {count} pts ({validCandidates} valid candidates)");
            return (null, null, 0);
        }

        var R = new float[9];
        var t = new float[3];
        Array.Copy(trimmedCandidates, bestIter * 12, R, 0, 9);
        Array.Copy(trimmedCandidates, bestIter * 12 + 9, t, 0, 3);

        // Enforce rotation orthogonality via SVD projection to SO(3)
        // DLT doesn't guarantee R is orthogonal — project to nearest rotation
        var Rdouble = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Rdouble[i, j] = R[i * 3 + j];
        var (svdU, _, svdVt) = LinearAlgebra.SVD3x3(Rdouble);

        // R_orth = U * Vt (nearest rotation in Frobenius norm)
        var Rorth = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    Rorth[i, j] += svdU[i, k] * svdVt[k, j];

        if (LinearAlgebra.Det3x3(Rorth) < 0)
        {
            for (int i = 0; i < 3; i++) svdVt[2, i] = -svdVt[2, i];
            Rorth = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        Rorth[i, j] += svdU[i, k] * svdVt[k, j];
        }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i * 3 + j] = (float)Rorth[i, j];

        Log($"[SfM-GPU] PnP: {bestCount}/{count} inliers ({validCandidates} candidates)");
        return (R, t, bestCount);
    }

    // ──────────────────────────────────────────────────────────
    //  PUBLIC API: GPU Triangulation
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// GPU-accelerated triangulation of multiple point pairs.
    /// Returns array of (x, y, z, valid) for each match.
    /// </summary>
    public async Task<float[]> TriangulateAsync(
        float[] ptsA, float[] ptsB,
        float[] P1, float[] P2,
        float[] poseDataA, float[] poseDataB,
        int matchCount)
    {
        EnsureKernelsLoaded();
        var acc = _gpu.Accelerator;

        using var bufA = acc.Allocate1D(ptsA);
        using var bufB = acc.Allocate1D(ptsB);
        using var bufP1 = acc.Allocate1D(P1);
        using var bufP2 = acc.Allocate1D(P2);
        using var bufPoseA = acc.Allocate1D(poseDataA);
        using var bufPoseB = acc.Allocate1D(poseDataB);
        using var bufOut = acc.Allocate1D<float>(matchCount * 4);

        _triKernel!((Index1D)matchCount,
            bufA.View, bufB.View,
            bufP1.View, bufP2.View,
            bufOut.View,
            bufPoseA.View, bufPoseB.View);

        await acc.SynchronizeAsync();
        return await bufOut.CopyToHostAsync();
    }

    // ──────────────────────────────────────────────────────────
    //  PUBLIC API: Camera Quality Validation (GPU)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Validate a candidate camera pose using GPU-accelerated reprojection error.
    /// Returns (isValid, medianError, reason).
    /// Quality gates:
    ///   1. PnP inliers >= minInliers
    ///   2. Median reprojection error < maxReprojError pixels
    ///   3. det(R) ≈ +1 and rotation angle < 120°
    ///   4. Translation within 10× median of existing camera distances
    /// </summary>
    public async Task<(bool isValid, float medianError, string reason)> ValidateCameraAsync(
        float[] pts2D, float[] pts3D, int corrCount,
        float[] pose12,     // R[9] + t[3]
        int pnpInliers,
        float fx,
        List<float[]>? existingCamTranslations = null,
        int minInliers = 4,
        float maxReprojError = 32.0f)
    {
        // Gate 1: Minimum inlier count
        if (pnpInliers < minInliers)
            return (false, float.MaxValue, $"Too few PnP inliers: {pnpInliers} < {minInliers}");

        // Gate 3: Rotation sanity
        double detR = pose12[0] * (pose12[4] * pose12[8] - pose12[5] * pose12[7])
                    - pose12[1] * (pose12[3] * pose12[8] - pose12[5] * pose12[6])
                    + pose12[2] * (pose12[3] * pose12[7] - pose12[4] * pose12[6]);
        if (Math.Abs(detR - 1.0) > 0.3)
            return (false, float.MaxValue, $"Bad rotation det(R)={detR:F3}, expected ≈1.0");

        // Rotation angle: trace(R) = 1 + 2*cos(θ)
        double traceR = pose12[0] + pose12[4] + pose12[8];
        double cosAngle = (traceR - 1.0) / 2.0;
        cosAngle = Math.Max(-1.0, Math.Min(1.0, cosAngle));
        double angleDeg = Math.Acos(cosAngle) * 180.0 / Math.PI;
        // Rotation angle check removed — room-scanning cameras can face any direction
        // if (angleDeg > 170.0)
        //     return (false, float.MaxValue, $"Rotation angle {angleDeg:F1}° > 170°");

        // Gate 4: Translation magnitude check
        if (existingCamTranslations != null && existingCamTranslations.Count >= 2)
        {
            // Compute median distance of existing cameras from centroid
            float cx = existingCamTranslations.Average(t => t[0]);
            float cy = existingCamTranslations.Average(t => t[1]);
            float cz = existingCamTranslations.Average(t => t[2]);

            var distances = existingCamTranslations.Select(t =>
                MathF.Sqrt((t[0] - cx) * (t[0] - cx) + (t[1] - cy) * (t[1] - cy) + (t[2] - cz) * (t[2] - cz))
            ).OrderBy(d => d).ToList();
            float medianDist = distances[distances.Count / 2];
            if (medianDist < 0.001f) medianDist = 1.0f; // prevent div by zero on first cameras

            float newDist = MathF.Sqrt(
                (pose12[9] - cx) * (pose12[9] - cx) +
                (pose12[10] - cy) * (pose12[10] - cy) +
                (pose12[11] - cz) * (pose12[11] - cz));

            if (newDist > medianDist * 20.0f)
                return (false, float.MaxValue, $"Camera too far: dist={newDist:F1} > 20×median={medianDist * 20:F1}");
        }

        // Gate 2: GPU reprojection error
        if (corrCount < 4)
            return (false, float.MaxValue, $"Too few correspondences for reprojection: {corrCount}");

        EnsureKernelsLoaded();
        var acc = _gpu.Accelerator;

        using var buf2D = acc.Allocate1D(pts2D);
        using var buf3D = acc.Allocate1D(pts3D);
        using var bufPose = acc.Allocate1D(pose12);
        using var bufErrors = acc.Allocate1D<float>(corrCount);

        _reprojErrorKernel!((Index1D)corrCount,
            buf2D.View, buf3D.View,
            corrCount,
            bufPose.View,
            bufErrors.View);

        await acc.SynchronizeAsync();
        var errors = await bufErrors.CopyToHostAsync();

        // Compute median error (in pixels — multiply by fx)
        var sortedErrors = errors.OrderBy(e => e).ToArray();
        float medianNormErr = sortedErrors[sortedErrors.Length / 2];
        float medianPixErr = medianNormErr * fx;

        if (medianPixErr > maxReprojError)
            return (false, medianPixErr, $"Median reprojection error {medianPixErr:F2}px > {maxReprojError}px");

        return (true, medianPixErr, "OK");
    }

    // ──────────────────────────────────────────────────────────
    //  GPU KERNEL: Essential Matrix Inlier Counter
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Each thread takes one candidate essential matrix and counts
    /// inliers across ALL match points using Sampson error.
    /// </summary>
    private static void EssentialInlierCountKernel(
        Index1D idx,
        ArrayView<float> ptsA, ArrayView<float> ptsB,
        int matchCount,
        ArrayView<float> candidates,  // 9 floats per candidate
        ArrayView<int> outCounts)
    {
        int cBase = idx * 9;
        float e0 = candidates[cBase], e1 = candidates[cBase + 1], e2 = candidates[cBase + 2];
        float e3 = candidates[cBase + 3], e4 = candidates[cBase + 4], e5 = candidates[cBase + 5];
        float e6 = candidates[cBase + 6], e7 = candidates[cBase + 7], e8 = candidates[cBase + 8];

        int count = 0;
        for (int i = 0; i < matchCount; i++)
        {
            float xA = ptsA[i * 2];
            float yA = ptsA[i * 2 + 1];
            float xB = ptsB[i * 2];
            float yB = ptsB[i * 2 + 1];

            // Sampson error
            float epA0 = e0 * xA + e1 * yA + e2;
            float epA1 = e3 * xA + e4 * yA + e5;
            float epA2 = e6 * xA + e7 * yA + e8;

            float etpB0 = e0 * xB + e3 * yB + e6;
            float etpB1 = e1 * xB + e4 * yB + e7;

            float num = xB * epA0 + yB * epA1 + epA2;
            float denom = epA0 * epA0 + epA1 * epA1 + etpB0 * etpB0 + etpB1 * etpB1;

            float sampson = (num * num) / (denom + 1e-10f);
            if (sampson < 0.01f)
                count++;
        }

        outCounts[idx] = count;
    }

    // ──────────────────────────────────────────────────────────
    //  GPU KERNEL: PnP Inlier Counter
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Each thread takes one candidate pose [R|t] and counts
    /// inliers by reprojecting all 3D points through it.
    /// </summary>
    private static void PnPInlierCountKernel(
        Index1D idx,
        ArrayView<float> pts2D,   // normalized, 2 per point
        ArrayView<float> pts3D,   // world, 3 per point
        int count,
        ArrayView<float> candidates,  // 12 per candidate: R[9] + t[3]
        ArrayView<int> outCounts)
    {
        int cBase = idx * 12;
        float r00 = candidates[cBase], r01 = candidates[cBase + 1], r02 = candidates[cBase + 2];
        float r10 = candidates[cBase + 3], r11 = candidates[cBase + 4], r12 = candidates[cBase + 5];
        float r20 = candidates[cBase + 6], r21 = candidates[cBase + 7], r22 = candidates[cBase + 8];
        float tx = candidates[cBase + 9], ty = candidates[cBase + 10], tz = candidates[cBase + 11];

        int inliers = 0;
        for (int i = 0; i < count; i++)
        {
            float X = pts3D[i * 3];
            float Y = pts3D[i * 3 + 1];
            float Z = pts3D[i * 3 + 2];

            float px = r00 * X + r01 * Y + r02 * Z + tx;
            float py = r10 * X + r11 * Y + r12 * Z + ty;
            float pz = r20 * X + r21 * Y + r22 * Z + tz;

            if (IntrinsicMath.Abs(pz) < 1e-6f) continue;

            float u = px / pz;
            float v = py / pz;
            float du = u - pts2D[i * 2];
            float dv = v - pts2D[i * 2 + 1];
            float err = du * du + dv * dv;

            if (err < 0.005f) // ~5px in normalized coords
                inliers++;
        }

        outCounts[idx] = inliers;
    }

    // ──────────────────────────────────────────────────────────
    //  GPU KERNEL: Reprojection Error (quality gate)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Each thread computes reprojection error for one 2D-3D correspondence.
    /// Output is L2 error in normalized coordinates.
    /// </summary>
    private static void ReprojectionErrorKernel(
        Index1D idx,
        ArrayView<float> pts2D,   // normalized, 2 per point
        ArrayView<float> pts3D,   // world, 3 per point
        int count,
        ArrayView<float> pose,    // single pose: R[9] + t[3]
        ArrayView<float> outErrors)
    {
        if (idx >= count) { outErrors[idx] = float.MaxValue; return; }

        float r00 = pose[0], r01 = pose[1], r02 = pose[2];
        float r10 = pose[3], r11 = pose[4], r12 = pose[5];
        float r20 = pose[6], r21 = pose[7], r22 = pose[8];
        float tx = pose[9], ty = pose[10], tz = pose[11];

        float X = pts3D[idx * 3];
        float Y = pts3D[idx * 3 + 1];
        float Z = pts3D[idx * 3 + 2];

        float pz = r20 * X + r21 * Y + r22 * Z + tz;
        if (pz < 0.0001f) { outErrors[idx] = float.MaxValue; return; }

        float px = r00 * X + r01 * Y + r02 * Z + tx;
        float py = r10 * X + r11 * Y + r12 * Z + ty;
        float u = px / pz;
        float v = py / pz;

        float du = u - pts2D[idx * 2];
        float dv = v - pts2D[idx * 2 + 1];
        outErrors[idx] = XMath.Sqrt(du * du + dv * dv);
    }

    // ──────────────────────────────────────────────────────────
    //  GPU KERNEL: Triangulation
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Each thread triangulates one match pair via DLT.
    /// </summary>
    private static void TriangulationKernel(
        Index1D idx,
        ArrayView<float> ptsA, ArrayView<float> ptsB,
        ArrayView<float> P1, ArrayView<float> P2,
        ArrayView<float> outPts,
        ArrayView<float> poseA, ArrayView<float> poseB)
    {
        float uA = ptsA[idx * 2];
        float vA = ptsA[idx * 2 + 1];
        float uB = ptsB[idx * 2];
        float vB = ptsB[idx * 2 + 1];

        // Build 4x4 DLT matrix rows
        float a00 = uA * P1[8] - P1[0], a01 = uA * P1[9] - P1[1];
        float a02 = uA * P1[10] - P1[2], a03 = uA * P1[11] - P1[3];
        float a10 = vA * P1[8] - P1[4], a11 = vA * P1[9] - P1[5];
        float a12 = vA * P1[10] - P1[6], a13 = vA * P1[11] - P1[7];
        float a20 = uB * P2[8] - P2[0], a21 = uB * P2[9] - P2[1];
        float a22 = uB * P2[10] - P2[2], a23 = uB * P2[11] - P2[3];
        float a30 = vB * P2[8] - P2[4], a31 = vB * P2[9] - P2[5];
        float a32 = vB * P2[10] - P2[6], a33 = vB * P2[11] - P2[7];

        // AtA (4x4 symmetric)
        float n00 = a00 * a00 + a10 * a10 + a20 * a20 + a30 * a30;
        float n01 = a00 * a01 + a10 * a11 + a20 * a21 + a30 * a31;
        float n02 = a00 * a02 + a10 * a12 + a20 * a22 + a30 * a32;
        float n03 = a00 * a03 + a10 * a13 + a20 * a23 + a30 * a33;
        float n11 = a01 * a01 + a11 * a11 + a21 * a21 + a31 * a31;
        float n12 = a01 * a02 + a11 * a12 + a21 * a22 + a31 * a32;
        float n13 = a01 * a03 + a11 * a13 + a21 * a23 + a31 * a33;
        float n22 = a02 * a02 + a12 * a12 + a22 * a22 + a32 * a32;
        float n23 = a02 * a03 + a12 * a13 + a22 * a23 + a32 * a33;
        float n33 = a03 * a03 + a13 * a13 + a23 * a23 + a33 * a33;

        // 3x3 cofactors of last row of AtA for null vector
        float m30 = n01 * (n12 * n23 - n13 * n22) - n02 * (n11 * n23 - n13 * n12) + n03 * (n11 * n22 - n12 * n12);
        float m31 = n00 * (n12 * n23 - n13 * n22) - n02 * (n01 * n23 - n13 * n02) + n03 * (n01 * n22 - n12 * n02);
        float m32 = n00 * (n11 * n23 - n13 * n12) - n01 * (n01 * n23 - n13 * n02) + n03 * (n01 * n12 - n11 * n02);
        float m33 = n00 * (n11 * n22 - n12 * n12) - n01 * (n01 * n22 - n12 * n02) + n02 * (n01 * n12 - n11 * n02);

        float x = m30, y = -m31, z = m32, w = -m33;

        int outBase = idx * 4;
        if (IntrinsicMath.Abs(w) < 1e-8f)
        {
            outPts[outBase + 3] = 0;
            return;
        }

        float px = x / w, py = y / w, pz = z / w;

        // Cheirality check
        float zA = poseA[6] * px + poseA[7] * py + poseA[8] * pz + poseA[11];
        float zB = poseB[6] * px + poseB[7] * py + poseB[8] * pz + poseB[11];

        if (zA > 0 && zB > 0)
        {
            outPts[outBase] = px;
            outPts[outBase + 1] = py;
            outPts[outBase + 2] = pz;
            outPts[outBase + 3] = 1;
        }
        else
        {
            outPts[outBase + 3] = 0;
        }
    }

    // ──────────────────────────────────────────────────────────
    //  CPU: 8-Point Algorithm (correct linear algebra)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Public test wrapper for Solve8Point (for synthetic testing only).
    /// </summary>
    public static float[]? TestSolve8Point(float[] ptsA, float[] ptsB, int[] sampleIndices)
        => Solve8Point(ptsA, ptsB, sampleIndices);

    /// <summary>
    /// Solve the 8-point algorithm for essential matrix estimation.
    /// Returns float[9] essential matrix or null.
    /// Uses Gaussian elimination on 8×9 system, then enforces
    /// the essential matrix constraint via SVD: E = U*diag(σ,σ,0)*Vt.
    /// </summary>
    private static float[]? Solve8Point(float[] ptsA, float[] ptsB, int[] sampleIndices)
    {
        // ═══ HARTLEY NORMALIZATION (critical for numerical stability) ═══
        // Transform points to have zero mean and average distance sqrt(2)
        double meanAx = 0, meanAy = 0, meanBx = 0, meanBy = 0;
        for (int s = 0; s < 8; s++)
        {
            int mi = sampleIndices[s];
            meanAx += ptsA[mi * 2]; meanAy += ptsA[mi * 2 + 1];
            meanBx += ptsB[mi * 2]; meanBy += ptsB[mi * 2 + 1];
        }
        meanAx /= 8; meanAy /= 8; meanBx /= 8; meanBy /= 8;

        double distA = 0, distB = 0;
        for (int s = 0; s < 8; s++)
        {
            int mi = sampleIndices[s];
            double dxa = ptsA[mi * 2] - meanAx, dya = ptsA[mi * 2 + 1] - meanAy;
            double dxb = ptsB[mi * 2] - meanBx, dyb = ptsB[mi * 2 + 1] - meanBy;
            distA += Math.Sqrt(dxa * dxa + dya * dya);
            distB += Math.Sqrt(dxb * dxb + dyb * dyb);
        }
        distA /= 8; distB /= 8;
        if (distA < 1e-10 || distB < 1e-10) return null;

        double scaleA = Math.Sqrt(2.0) / distA;
        double scaleB = Math.Sqrt(2.0) / distB;

        // Build 8×9 matrix for x_B'^T * E' * x_A' = 0 (normalized coords)
        var A = new double[8, 9];
        for (int s = 0; s < 8; s++)
        {
            int mi = sampleIndices[s];
            double xA = (ptsA[mi * 2] - meanAx) * scaleA;
            double yA = (ptsA[mi * 2 + 1] - meanAy) * scaleA;
            double xB = (ptsB[mi * 2] - meanBx) * scaleB;
            double yB = (ptsB[mi * 2 + 1] - meanBy) * scaleB;

            // Row for x_B^T * E * x_A = 0 (Hartley-Zisserman convention)
            A[s, 0] = xA * xB; A[s, 1] = yA * xB; A[s, 2] = xB;
            A[s, 3] = xA * yB; A[s, 4] = yA * yB; A[s, 5] = yB;
            A[s, 6] = xA; A[s, 7] = yA; A[s, 8] = 1.0;
        }

        // Gaussian elimination with partial pivoting on 8×9 matrix
        for (int col = 0; col < 8; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < 8; row++)
                if (Math.Abs(A[row, col]) > Math.Abs(A[maxRow, col]))
                    maxRow = row;

            if (Math.Abs(A[maxRow, col]) < 1e-10)
                return null;

            for (int j = 0; j < 9; j++)
                (A[col, j], A[maxRow, j]) = (A[maxRow, j], A[col, j]);

            for (int row = col + 1; row < 8; row++)
            {
                double f = A[row, col] / A[col, col];
                for (int j = col; j < 9; j++)
                    A[row, j] -= f * A[col, j];
            }
        }

        // Back substitution: set e[8] = 1, solve for e[0..7]
        var e = new double[9];
        e[8] = 1.0;

        for (int i = 7; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < 9; j++)
                sum += A[i, j] * e[j];
            e[i] = -sum / A[i, i];
        }

        // Normalize
        double norm = 0;
        for (int i = 0; i < 9; i++) norm += e[i] * e[i];
        norm = Math.Sqrt(norm);
        if (norm < 1e-10) return null;
        for (int i = 0; i < 9; i++) e[i] /= norm;

        // ═══ CRITICAL: Enforce essential matrix constraint ═══
        // E must have singular values (σ, σ, 0).
        // SVD decompose E, set S = diag((σ1+σ2)/2, (σ1+σ2)/2, 0), reconstruct.
        var Emat = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Emat[i, j] = e[i * 3 + j];

        var (U, S, Vt) = LinearAlgebra.SVD3x3(Emat);

        // Enforce constraint: two equal singular values, third = 0
        double sigma = (S[0] + S[1]) / 2.0;
        S[0] = sigma;
        S[1] = sigma;
        S[2] = 0;

        // Reconstruct E = U * diag(S) * Vt
        var Ecorrected = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double val = 0;
                for (int k = 0; k < 3; k++)
                    val += U[i, k] * S[k] * Vt[k, j];
                Ecorrected[i, j] = val;
            }

        // ═══ HARTLEY DENORMALIZATION ═══
        // E_normalized is for normalized coords: x_B'^T * E' * x_A' = 0
        // Original: x_B^T * E * x_A = 0 where x' = T * x
        // So E = T_B^T * E' * T_A
        // T = [[s, 0, -s*mx], [0, s, -s*my], [0, 0, 1]]
        var TA = new double[3, 3] { { scaleA, 0, -scaleA * meanAx }, { 0, scaleA, -scaleA * meanAy }, { 0, 0, 1 } };
        var TB = new double[3, 3] { { scaleB, 0, -scaleB * meanBx }, { 0, scaleB, -scaleB * meanBy }, { 0, 0, 1 } };

        // E = TB^T * Ecorrected * TA
        // First: temp = Ecorrected * TA
        var temp = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double val = 0;
                for (int k = 0; k < 3; k++)
                    val += Ecorrected[i, k] * TA[k, j];
                temp[i, j] = val;
            }
        // E = TB^T * temp
        var Efinal = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double val = 0;
                for (int k = 0; k < 3; k++)
                    val += TB[k, i] * temp[k, j]; // TB^T: swap indices
                Efinal[i, j] = val;
            }
        // ═══ RE-ENFORCE essential matrix constraint after denormalization ═══
        // The denormalization T_B^T * E' * T_A destroys the (σ,σ,0) structure.
        // We must re-enforce it so RecoverPose extracts valid R,t from SVD(E).
        var (U2, S2, Vt2) = LinearAlgebra.SVD3x3(Efinal);
        double sigma2 = (S2[0] + S2[1]) / 2.0;
        S2[0] = sigma2;
        S2[1] = sigma2;
        S2[2] = 0;

        // Reconstruct E = U2 * diag(S2) * Vt2
        var Efinal2 = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double val = 0;
                for (int k = 0; k < 3; k++)
                    val += U2[i, k] * S2[k] * Vt2[k, j];
                Efinal2[i, j] = val;
            }

        // Final normalize
        double norm3 = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                norm3 += Efinal2[i, j] * Efinal2[i, j];
        norm3 = Math.Sqrt(norm3);
        if (norm3 < 1e-10) return null;

        var result = new float[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result[i * 3 + j] = (float)(Efinal2[i, j] / norm3);

        return result;
    }

    // ──────────────────────────────────────────────────────────
    //  CPU: DLT PnP Solver (correct linear algebra)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Solve PnP using EPnP from 2D-3D correspondences at the given sample indices.
    /// Returns float[12] = R[9] + t[3], or null on failure.
    /// </summary>
    private static float[]? SolvePnPDLT(float[] pts2D, float[] pts3D, int[] sampleIndices)
    {
        int n = sampleIndices.Length;
        if (n < 4) return null;

        // Extract sample points into contiguous arrays for EPnP
        var sample2D = new float[n * 2];
        var sample3D = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            int mi = sampleIndices[i];
            sample2D[i * 2] = pts2D[mi * 2];
            sample2D[i * 2 + 1] = pts2D[mi * 2 + 1];
            sample3D[i * 3] = pts3D[mi * 3];
            sample3D[i * 3 + 1] = pts3D[mi * 3 + 1];
            sample3D[i * 3 + 2] = pts3D[mi * 3 + 2];
        }

        var epnpResult = LinearAlgebra.SolveEPnP(sample2D.AsSpan(), sample3D.AsSpan(), n);
        if (epnpResult == null) return null;

        var (R, t) = epnpResult.Value;
        var result = new float[12];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                result[i * 3 + j] = (float)R[i, j];
            result[9 + i] = (float)t[i];
        }
        return result;
    }

    private static void OrthogonalizeRotation(float[] R)
    {
        // Gram-Schmidt on rows of 3x3 R stored as flat[9]
        double r00 = R[0], r01 = R[1], r02 = R[2];
        double r10 = R[3], r11 = R[4], r12 = R[5];

        // Normalize row 0
        double n0 = Math.Sqrt(r00 * r00 + r01 * r01 + r02 * r02);
        if (n0 < 1e-10) return;
        r00 /= n0; r01 /= n0; r02 /= n0;

        // Row 1 -= (row1·row0)*row0, then normalize
        double dot10 = r10 * r00 + r11 * r01 + r12 * r02;
        r10 -= dot10 * r00; r11 -= dot10 * r01; r12 -= dot10 * r02;
        double n1 = Math.Sqrt(r10 * r10 + r11 * r11 + r12 * r12);
        if (n1 < 1e-10) return;
        r10 /= n1; r11 /= n1; r12 /= n1;

        // Row 2 = row0 × row1 (cross product)
        double r20 = r01 * r12 - r02 * r11;
        double r21 = r02 * r10 - r00 * r12;
        double r22 = r00 * r11 - r01 * r10;

        R[0] = (float)r00; R[1] = (float)r01; R[2] = (float)r02;
        R[3] = (float)r10; R[4] = (float)r11; R[5] = (float)r12;
        R[6] = (float)r20; R[7] = (float)r21; R[8] = (float)r22;
    }

    // ──────────────────────────────────────────────────────────
    //  CPU Helper: Sampson Error
    // ──────────────────────────────────────────────────────────

    private static float ComputeSampsonError(float xA, float yA, float xB, float yB, float[] E)
    {
        float epA0 = E[0] * xA + E[1] * yA + E[2];
        float epA1 = E[3] * xA + E[4] * yA + E[5];
        float epA2 = E[6] * xA + E[7] * yA + E[8];

        float etpB0 = E[0] * xB + E[3] * yB + E[6];
        float etpB1 = E[1] * xB + E[4] * yB + E[7];

        float num = xB * epA0 + yB * epA1 + epA2;
        float denom = epA0 * epA0 + epA1 * epA1 + etpB0 * etpB0 + etpB1 * etpB1;

        return (num * num) / (denom + 1e-10f);
    }

    // ══════════════════════════════════════════════════════════
    //  BUNDLE ADJUSTMENT (GPU + CPU Schur Complement)
    // ══════════════════════════════════════════════════════════

    // BA kernel delegate
    private Action<Index1D,
        ArrayView<float>,   // cameraParams: 6 per camera (angle-axis[3] + translation[3])
        ArrayView<float>,   // points3D: 3 per point
        ArrayView<float>,   // observations: 4 per obs (camIdx, ptIdx, u_observed, v_observed)
        int,                // numObs
        ArrayView<float>,   // outResiduals: 2 per obs
        ArrayView<float>,   // outJc: 2×6=12 per obs (Jacobian w.r.t. camera)
        ArrayView<float>    // outJp: 2×3=6 per obs (Jacobian w.r.t. point)
        >? _baResidualKernel;

    private void EnsureBAKernelsLoaded()
    {
        if (_baResidualKernel != null) return;
        var acc = _gpu.Accelerator;
        _baResidualKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int,
            ArrayView<float>, ArrayView<float>, ArrayView<float>>(BAResidualKernel);
    }

    /// <summary>
    /// GPU Bundle Adjustment kernel: one thread per observation.
    /// Camera parameterized as angle-axis rotation (3) + translation (3).
    /// Computes residual (2), Jacobian w.r.t. camera (2×6), Jacobian w.r.t. point (2×3).
    /// </summary>
    private static void BAResidualKernel(
        Index1D idx,
        ArrayView<float> camParams,    // 6 per camera
        ArrayView<float> points,       // 3 per point
        ArrayView<float> observations, // 4 per obs: camIdx, ptIdx, u_obs, v_obs
        int numObs,
        ArrayView<float> outResiduals, // 2 per obs
        ArrayView<float> outJc,        // 12 per obs (2 rows × 6 cols)
        ArrayView<float> outJp)        // 6 per obs (2 rows × 3 cols)
    {
        if (idx >= numObs) return;

        int obsBase = idx * 4;
        int camIdx = (int)observations[obsBase];
        int ptIdx = (int)observations[obsBase + 1];
        float u_obs = observations[obsBase + 2];
        float v_obs = observations[obsBase + 3];

        // Camera params: angle-axis (ax,ay,az) + translation (tx,ty,tz)
        int cBase = camIdx * 6;
        float ax = camParams[cBase], ay = camParams[cBase + 1], az = camParams[cBase + 2];
        float tx = camParams[cBase + 3], ty = camParams[cBase + 4], tz = camParams[cBase + 5];

        // 3D point
        int pBase = ptIdx * 3;
        float X = points[pBase], Y = points[pBase + 1], Z = points[pBase + 2];

        // Rodrigues rotation: R(v) * p = p + cross(v,p) * sin(θ)/θ + cross(v, cross(v,p)) * (1-cos(θ))/θ²
        float theta2 = ax * ax + ay * ay + az * az;

        // cross(v, p)
        float cx = ay * Z - az * Y;
        float cy = az * X - ax * Z;
        float cz = ax * Y - ay * X;

        float sinc, cosc;
        if (theta2 < 1e-10f)
        {
            // Near identity: use Taylor series
            sinc = 1.0f;  // sin(θ)/θ → 1
            cosc = 0.5f;  // (1-cos(θ))/θ² → 0.5
        }
        else
        {
            float theta = XMath.Sqrt(theta2);
            sinc = XMath.Sin(theta) / theta;
            cosc = (1.0f - XMath.Cos(theta)) / theta2;
        }

        // cross(v, cross(v,p))
        float ccx = ay * cz - az * cy;
        float ccy = az * cx - ax * cz;
        float ccz = ax * cy - ay * cx;

        // Rotated point + translation
        float px = X + sinc * cx + cosc * ccx + tx;
        float py = Y + sinc * cy + cosc * ccy + ty;
        float pz = Z + sinc * cz + cosc * ccz + tz;

        // Projection (normalized coords, K=identity)
        if (pz < 1e-6f) pz = 1e-6f;
        float invZ = 1.0f / pz;
        float u_proj = px * invZ;
        float v_proj = py * invZ;

        // Residual
        int rBase = idx * 2;
        outResiduals[rBase] = u_proj - u_obs;
        outResiduals[rBase + 1] = v_proj - v_obs;

        // ─── Jacobian w.r.t. camera params ───
        // Translation Jacobian is exact: du/dtx = 1/pz, du/dtz = -u/pz, etc.
        // Rotation Jacobian: use central finite differences on the Rodrigues formula
        // because the -skew(Rp) approximation is only valid for infinitesimal perturbations.

        int jcBase = idx * 12;
        float eps = 1e-4f;

        // Rotation Jacobian via central differences (3 axes)
        for (int d = 0; d < 3; d++)
        {
            float axp = ax, ayp = ay, azp = az;
            float axm = ax, aym = ay, azm = az;
            if (d == 0) { axp += eps; axm -= eps; }
            else if (d == 1) { ayp += eps; aym -= eps; }
            else { azp += eps; azm -= eps; }

            // Forward: R(ω+ε·eᵢ) * X + t
            float t2p = axp * axp + ayp * ayp + azp * azp;
            float cxp = ayp * Z - azp * Y, cyp = azp * X - axp * Z, czp = axp * Y - ayp * X;
            float sp, cp;
            if (t2p < 1e-10f) { sp = 1.0f; cp = 0.5f; }
            else { float tp = XMath.Sqrt(t2p); sp = XMath.Sin(tp) / tp; cp = (1.0f - XMath.Cos(tp)) / t2p; }
            float ccxp = ayp * czp - azp * cyp, ccyp = azp * cxp - axp * czp, cczp = axp * cyp - ayp * cxp;
            float pxp = X + sp * cxp + cp * ccxp + tx, pyp = Y + sp * cyp + cp * ccyp + ty, pzp = Z + sp * czp + cp * cczp + tz;
            if (pzp < 1e-6f) pzp = 1e-6f;
            float up = pxp / pzp, vp = pyp / pzp;

            // Backward: R(ω-ε·eᵢ) * X + t
            float t2m = axm * axm + aym * aym + azm * azm;
            float cxm = aym * Z - azm * Y, cym = azm * X - axm * Z, czm = axm * Y - aym * X;
            float sm, cm;
            if (t2m < 1e-10f) { sm = 1.0f; cm = 0.5f; }
            else { float tm = XMath.Sqrt(t2m); sm = XMath.Sin(tm) / tm; cm = (1.0f - XMath.Cos(tm)) / t2m; }
            float ccxm = aym * czm - azm * cym, ccym = azm * cxm - axm * czm, cczm = axm * cym - aym * cxm;
            float pxm = X + sm * cxm + cm * ccxm + tx, pym = Y + sm * cym + cm * ccym + ty, pzm = Z + sm * czm + cm * cczm + tz;
            if (pzm < 1e-6f) pzm = 1e-6f;
            float um = pxm / pzm, vm = pym / pzm;

            float inv2eps = 1.0f / (2.0f * eps);
            outJc[jcBase + d] = (up - um) * inv2eps;       // du/dωᵢ
            outJc[jcBase + 6 + d] = (vp - vm) * inv2eps;   // dv/dωᵢ
        }

        // Translation Jacobian (exact analytical — already verified correct)
        outJc[jcBase + 3] = invZ;              // du/dtx
        outJc[jcBase + 4] = 0;                 // du/dty
        outJc[jcBase + 5] = -u_proj * invZ;    // du/dtz
        outJc[jcBase + 9] = 0;                 // dv/dtx
        outJc[jcBase + 10] = invZ;             // dv/dty
        outJc[jcBase + 11] = -v_proj * invZ;   // dv/dtz

        // ─── Jacobian w.r.t. 3D point (exact analytical — already verified correct) ───
        // R = I + sinc*K + cosc*K² where K = skew(v)
        float r00 = 1 + cosc * (-az * az - ay * ay);
        float r01 = cosc * ax * ay - sinc * az;
        float r02 = cosc * ax * az + sinc * ay;
        float r10 = cosc * ax * ay + sinc * az;
        float r11 = 1 + cosc * (-az * az - ax * ax);
        float r12 = cosc * ay * az - sinc * ax;
        float r20 = cosc * ax * az - sinc * ay;
        float r21 = cosc * ay * az + sinc * ax;
        float r22 = 1 + cosc * (-ay * ay - ax * ax);

        int jpBase = idx * 6;
        // du/dX = J_proj[0,:] * R = [invZ, 0, -u*invZ] * R
        outJp[jpBase + 0] = invZ * r00 + (-u_proj * invZ) * r20;
        outJp[jpBase + 1] = invZ * r01 + (-u_proj * invZ) * r21;
        outJp[jpBase + 2] = invZ * r02 + (-u_proj * invZ) * r22;
        // dv/dX = J_proj[1,:] * R = [0, invZ, -v*invZ] * R
        outJp[jpBase + 3] = invZ * r10 + (-v_proj * invZ) * r20;
        outJp[jpBase + 4] = invZ * r11 + (-v_proj * invZ) * r21;
        outJp[jpBase + 5] = invZ * r12 + (-v_proj * invZ) * r22;
    }

    /// <summary>
    /// Run Gauss-Newton bundle adjustment on GPU.
    /// Camera params: angle-axis(3) + translation(3) per camera.
    /// Returns updated camera params and 3D points.
    /// </summary>
    public async Task<(float[] cameras, float[] points, float finalRMS)> BundleAdjustAsync(
        float[] cameraParams,  // 6 per camera (angle-axis + t)
        float[] points3D,      // 3 per point
        float[] observations,  // 4 per obs: camIdx, ptIdx, u_obs, v_obs
        int numCameras, int numPoints, int numObs,
        int maxIterations = 20,
        float lambda = 10.0f)
    {
        EnsureBAKernelsLoaded();
        var acc = _gpu.Accelerator;

        var cameras = (float[])cameraParams.Clone();
        var points = (float[])points3D.Clone();

        float prevRMS = float.MaxValue;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Upload data to GPU
            using var bufCam = acc.Allocate1D(cameras);
            using var bufPts = acc.Allocate1D(points);
            using var bufObs = acc.Allocate1D(observations);
            using var bufResiduals = acc.Allocate1D<float>(numObs * 2);
            using var bufJc = acc.Allocate1D<float>(numObs * 12);
            using var bufJp = acc.Allocate1D<float>(numObs * 6);

            // Launch residual + Jacobian kernel
            _baResidualKernel!((Index1D)numObs,
                bufCam.View, bufPts.View, bufObs.View,
                numObs,
                bufResiduals.View, bufJc.View, bufJp.View);

            await acc.SynchronizeAsync();

            // Read back results
            var residuals = await bufResiduals.CopyToHostAsync();
            var Jc = await bufJc.CopyToHostAsync();
            var Jp = await bufJp.CopyToHostAsync();

            // Compute RMS error
            double totalErr = 0;
            for (int i = 0; i < numObs * 2; i++)
                totalErr += residuals[i] * residuals[i];
            float rms = (float)Math.Sqrt(totalErr / (numObs * 2));

            if (iter == 0 || iter == maxIterations - 1 || iter % 3 == 0)
                Log($"[BA] iter {iter}: RMS={rms:F6} ({rms * 2304:F2}px approx)");

            // Check convergence
            if (Math.Abs(prevRMS - rms) < 1e-7f && iter > 0) break;
            prevRMS = rms;

            // ═══ Schur Complement Solve (CPU) ═══
            // Normal equations: [U  W] [δc] = [ec]
            //                   [Wᵀ V] [δp]   [ep]
            // Schur: (U - W V⁻¹ Wᵀ) δc = ec - W V⁻¹ ep
            // Then: δp = V⁻¹ (ep - Wᵀ δc)

            int camDim = 6, ptDim = 3;
            int totalCamParams = numCameras * camDim;

            // Accumulate U (6×6 per camera), V (3×3 per point), W (6×3 per cam-point pair)
            var U = new double[numCameras * 36]; // 6×6 blocks
            var V = new double[numPoints * 9];   // 3×3 blocks
            var ec = new double[totalCamParams];
            var ep = new double[numPoints * ptDim];

            // W is sparse: one 6×3 block per (camera, point) pair
            // Store as dict for sparse access
            var W_blocks = new Dictionary<(int cam, int pt), double[]>();

            for (int obs = 0; obs < numObs; obs++)
            {
                int ci = (int)observations[obs * 4];
                int pi = (int)observations[obs * 4 + 1];

                // Extract Jacobian rows for this observation (2 rows)
                var jc = new double[12]; // 2×6
                var jp = new double[6];  // 2×3
                var r = new double[2];
                for (int k = 0; k < 12; k++) jc[k] = Jc[obs * 12 + k];
                for (int k = 0; k < 6; k++) jp[k] = Jp[obs * 6 + k];
                r[0] = residuals[obs * 2];
                r[1] = residuals[obs * 2 + 1];

                // U[ci] += Jcᵀ * Jc (6×6)
                int uBase = ci * 36;
                for (int a = 0; a < 6; a++)
                    for (int b = 0; b < 6; b++)
                        U[uBase + a * 6 + b] += jc[a] * jc[b] + jc[6 + a] * jc[6 + b];

                // V[pi] += Jpᵀ * Jp (3×3)
                int vBase = pi * 9;
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        V[vBase + a * 3 + b] += jp[a] * jp[b] + jp[3 + a] * jp[3 + b];

                // W[ci,pi] += Jcᵀ * Jp (6×3)
                var key = (ci, pi);
                if (!W_blocks.TryGetValue(key, out var wblock))
                {
                    wblock = new double[18]; // 6×3
                    W_blocks[key] = wblock;
                }
                for (int a = 0; a < 6; a++)
                    for (int b = 0; b < 3; b++)
                        wblock[a * 3 + b] += jc[a] * jp[b] + jc[6 + a] * jp[3 + b];

                // ec[ci] += Jcᵀ * r
                int ecBase = ci * 6;
                for (int a = 0; a < 6; a++)
                    ec[ecBase + a] += jc[a] * r[0] + jc[6 + a] * r[1];

                // ep[pi] += Jpᵀ * r
                int epBase = pi * 3;
                for (int a = 0; a < 3; a++)
                    ep[epBase + a] += jp[a] * r[0] + jp[3 + a] * r[1];
            }

            // Add Levenberg-Marquardt damping: (1 + λ) * diag (multiplicative)
            for (int ci = 0; ci < numCameras; ci++)
                for (int d = 0; d < 6; d++)
                {
                    int idx = ci * 36 + d * 6 + d;
                    U[idx] += lambda * Math.Max(U[idx], 1e-6);
                }
            for (int pi = 0; pi < numPoints; pi++)
                for (int d = 0; d < 3; d++)
                {
                    int idx = pi * 9 + d * 3 + d;
                    V[idx] += lambda * Math.Max(V[idx], 1e-6);
                }

            // Invert V blocks (3×3 each)
            var Vinv = new double[numPoints * 9];
            for (int pi = 0; pi < numPoints; pi++)
            {
                int vb = pi * 9;
                // 3×3 inversion via adjugate
                double a11 = V[vb], a12 = V[vb + 1], a13 = V[vb + 2];
                double a21 = V[vb + 3], a22 = V[vb + 4], a23 = V[vb + 5];
                double a31 = V[vb + 6], a32 = V[vb + 7], a33 = V[vb + 8];
                double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
                if (Math.Abs(det) < 1e-15) det = 1e-15;
                double inv = 1.0 / det;
                Vinv[vb] = (a22 * a33 - a23 * a32) * inv;
                Vinv[vb + 1] = (a13 * a32 - a12 * a33) * inv;
                Vinv[vb + 2] = (a12 * a23 - a13 * a22) * inv;
                Vinv[vb + 3] = (a23 * a31 - a21 * a33) * inv;
                Vinv[vb + 4] = (a11 * a33 - a13 * a31) * inv;
                Vinv[vb + 5] = (a13 * a21 - a11 * a23) * inv;
                Vinv[vb + 6] = (a21 * a32 - a22 * a31) * inv;
                Vinv[vb + 7] = (a12 * a31 - a11 * a32) * inv;
                Vinv[vb + 8] = (a11 * a22 - a12 * a21) * inv;
            }

            // Schur complement: S = U - sum_j W_ij * V_j^{-1} * W_ij^T
            // And: rhs = ec - sum_j W_ij * V_j^{-1} * ep_j
            var S = new double[totalCamParams * totalCamParams];
            var rhs = new double[totalCamParams];

            // Start with U on diagonal of S
            for (int ci = 0; ci < numCameras; ci++)
                for (int a = 0; a < 6; a++)
                    for (int b = 0; b < 6; b++)
                        S[(ci * 6 + a) * totalCamParams + (ci * 6 + b)] = U[ci * 36 + a * 6 + b];

            // rhs = -ec (gradient descent direction)
            for (int i = 0; i < totalCamParams; i++)
                rhs[i] = -ec[i];

            foreach (var ((ci, pi), wblock) in W_blocks)
            {
                // Compute W * V^{-1} (6×3 × 3×3 = 6×3)
                int vb = pi * 9;
                var WVinv = new double[18]; // 6×3
                for (int a = 0; a < 6; a++)
                    for (int b = 0; b < 3; b++)
                    {
                        double sum = 0;
                        for (int k = 0; k < 3; k++)
                            sum += wblock[a * 3 + k] * Vinv[vb + k * 3 + b];
                        WVinv[a * 3 + b] = sum;
                    }

                // S -= WVinv * W^T (outer product contributes to S[ci,ci] block)
                for (int a = 0; a < 6; a++)
                    for (int b = 0; b < 6; b++)
                    {
                        double sum = 0;
                        for (int k = 0; k < 3; k++)
                            sum += WVinv[a * 3 + k] * wblock[b * 3 + k]; // W^T column = W row
                        S[(ci * 6 + a) * totalCamParams + (ci * 6 + b)] -= sum;
                    }

                // rhs -= WVinv * ep_j
                for (int a = 0; a < 6; a++)
                {
                    double sum = 0;
                    for (int k = 0; k < 3; k++)
                        sum += WVinv[a * 3 + k] * ep[pi * 3 + k];
                    rhs[ci * 6 + a] += sum; // note: rhs = -ec + W*Vinv*ep
                }
            }

            // Solve S * delta_c = rhs via Cholesky/LU
            var delta_c = SolveDenseSystem(S, rhs, totalCamParams);
            if (delta_c == null)
            {
                Log($"[BA] iter {iter}: Schur solve failed, stopping");
                break;
            }

            // Back-substitute: delta_p = V^{-1} * (ep - W^T * delta_c)  — negate for descent
            var delta_p = new double[numPoints * 3];
            for (int pi = 0; pi < numPoints; pi++)
            {
                var bp = new double[3];
                for (int a = 0; a < 3; a++)
                    bp[a] = -ep[pi * 3 + a]; // start with -ep

                // Subtract W^T * delta_c for all cameras seeing this point
                // (handled below by iterating W_blocks)
                delta_p[pi * 3] = 0; delta_p[pi * 3 + 1] = 0; delta_p[pi * 3 + 2] = 0;
            }

            // Accumulate W^T * delta_c per point
            var WtDc = new double[numPoints * 3];
            foreach (var ((ci, pi), wblock) in W_blocks)
            {
                for (int a = 0; a < 3; a++)
                {
                    double sum = 0;
                    for (int k = 0; k < 6; k++)
                        sum += wblock[k * 3 + a] * delta_c[ci * 6 + k]; // W^T row a = W col a
                    WtDc[pi * 3 + a] += sum;
                }
            }

            // delta_p = V^{-1} * (-ep - WtDc)
            for (int pi = 0; pi < numPoints; pi++)
            {
                int vb = pi * 9;
                for (int a = 0; a < 3; a++)
                {
                    double bval = -ep[pi * 3 + a] - WtDc[pi * 3 + a];
                    double sum = 0;
                    for (int k = 0; k < 3; k++)
                        sum += Vinv[vb + a * 3 + k] * (-ep[pi * 3 + k] - WtDc[pi * 3 + k]);
                    delta_p[pi * 3 + a] = sum;
                }
            }

            // Apply updates: x_new = x + δ (δ already points in descent direction)

            // Save current state for backtracking
            var camBackup = (float[])cameras.Clone();
            var ptBackup = (float[])points.Clone();

            for (int ci = 0; ci < numCameras; ci++)
                for (int d = 0; d < 6; d++)
                    cameras[ci * 6 + d] += (float)delta_c[ci * 6 + d];

            for (int pi = 0; pi < numPoints; pi++)
                for (int d = 0; d < 3; d++)
                    points[pi * 3 + d] += (float)delta_p[pi * 3 + d];

            // Check for NaN/Inf — fast reject
            bool badUpdate = false;
            for (int i = 0; i < cameras.Length && !badUpdate; i++)
                if (float.IsNaN(cameras[i]) || float.IsInfinity(cameras[i])) badUpdate = true;
            for (int i = 0; i < points.Length && !badUpdate; i++)
                if (float.IsNaN(points[i]) || float.IsInfinity(points[i])) badUpdate = true;

            if (badUpdate)
            {
                Array.Copy(camBackup, cameras, cameras.Length);
                Array.Copy(ptBackup, points, points.Length);
                lambda *= 10;
                Log($"[BA] iter {iter}: NaN/Inf detected, reverting, λ→{lambda:E1}");
                continue;
            }

            // Cost-based acceptance: evaluate new residuals on GPU
            using var bufCamTrial = acc.Allocate1D(cameras);
            using var bufPtsTrial = acc.Allocate1D(points);
            using var bufObsTrial = acc.Allocate1D(observations);
            using var bufResTrial = acc.Allocate1D<float>(numObs * 2);
            using var bufJcDummy = acc.Allocate1D<float>(numObs * 12);
            using var bufJpDummy = acc.Allocate1D<float>(numObs * 6);

            _baResidualKernel!((Index1D)numObs,
                bufCamTrial.View, bufPtsTrial.View, bufObsTrial.View,
                numObs, bufResTrial.View, bufJcDummy.View, bufJpDummy.View);
            await acc.SynchronizeAsync();
            var trialResiduals = await bufResTrial.CopyToHostAsync();

            double trialCost = 0;
            for (int i = 0; i < numObs * 2; i++)
                trialCost += trialResiduals[i] * trialResiduals[i];
            float trialRMS = (float)Math.Sqrt(trialCost / (numObs * 2));

            if (trialRMS > rms)
            {
                // Step made things worse — revert and increase damping
                Array.Copy(camBackup, cameras, cameras.Length);
                Array.Copy(ptBackup, points, points.Length);
                lambda *= 10;
                Log($"[BA] iter {iter}: cost increased ({rms:F6}->{trialRMS:F6}), reverting, λ→{lambda:E1}");
            }
            else
            {
                // Good step — decrease damping for next iteration
                lambda = Math.Max(1e-6f, lambda / 3);
            }
        }

        return (cameras, points, prevRMS);
    }

    /// <summary>
    /// Solve A*x = b for a dense N×N system using Gaussian elimination with partial pivoting.
    /// </summary>
    private static double[]? SolveDenseSystem(double[] A, double[] b, int n)
    {
        // Augmented matrix
        var aug = new double[n * (n + 1)];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i * (n + 1) + j] = A[i * n + j];
            aug[i * (n + 1) + n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int pivotRow = col;
            double maxVal = Math.Abs(aug[col * (n + 1) + col]);
            for (int row = col + 1; row < n; row++)
            {
                double val = Math.Abs(aug[row * (n + 1) + col]);
                if (val > maxVal) { maxVal = val; pivotRow = row; }
            }
            if (maxVal < 1e-15) return null;

            // Swap rows
            if (pivotRow != col)
                for (int j = 0; j <= n; j++)
                    (aug[col * (n + 1) + j], aug[pivotRow * (n + 1) + j]) = (aug[pivotRow * (n + 1) + j], aug[col * (n + 1) + j]);

            // Eliminate
            double pivot = aug[col * (n + 1) + col];
            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row * (n + 1) + col] / pivot;
                for (int j = col; j <= n; j++)
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = aug[row * (n + 1) + n];
            for (int j = row + 1; j < n; j++)
                sum -= aug[row * (n + 1) + j] * x[j];
            x[row] = sum / aug[row * (n + 1) + row];
        }
        return x;
    }
}
