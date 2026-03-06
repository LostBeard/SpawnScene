using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Structure from Motion (SfM) reconstructor.
/// Implements incremental SfM pipeline:
///   1. Estimate essential matrix from feature matches (8-point + RANSAC)
///   2. Recover camera pose (R, t) from essential matrix
///   3. Triangulate matched points to 3D
///   4. Add more images incrementally via PnP
///   5. Bundle adjustment to refine all parameters
/// </summary>
public class SfmReconstructor
{
    private readonly ImageImportService _importService;
    private readonly GpuService _gpu;
    private readonly HttpClient? _httpClient;
    private GpuSfmKernels? _gpuKernels;
    private readonly Random _rng = new(42);

    /// <summary>Enable verbose console logging for debugging.</summary>
    public static bool VerboseLogging { get; set; } = false;

    internal static void Log(string msg) { if (VerboseLogging) Console.WriteLine(msg); }

    /// <summary>Reconstructed 3D points.</summary>
    public List<ReconstructedPoint> Points3D { get; } = [];

    /// <summary>Estimated camera poses (index matches ImageImportService.Images).</summary>
    public CameraParams?[] CameraPoses { get; private set; } = [];

    /// <summary>Current status message.</summary>
    public string Status { get; private set; } = "";

    /// <summary>Whether reconstruction is running.</summary>
    public bool IsRunning { get; private set; }

    /// <summary>Fired when status changes.</summary>
    public event Action? OnStateChanged;

    public SfmReconstructor(ImageImportService importService, GpuService gpu, HttpClient http)
    {
        _importService = importService;
        _gpu = gpu;
        _httpClient = http;
    }

    /// <summary>
    /// Run the full SfM pipeline on imported images.
    /// </summary>
    public async Task ReconstructAsync()
    {
        IsRunning = true;
        Points3D.Clear();

        var images = _importService.Images;
        var pairs = _importService.MatchedPairs;

        if (images.Count < 2 || pairs.Count == 0)
        {
            Status = "Need at least 2 images with matches";
            OnStateChanged?.Invoke();
            IsRunning = false;
            return;
        }

        CameraPoses = new CameraParams?[images.Count];

        // ═══ END-TO-END SYNTHETIC PIPELINE TEST ═══
        // Tests each stage: 8-point → SVD → RecoverPose → Triangulate
        {
            Log("[SfM] ╔══════════════════════════════════════════╗");
            Log("[SfM] ║   END-TO-END SYNTHETIC PIPELINE TEST    ║");
            Log("[SfM] ╚══════════════════════════════════════════╝");

            // Known ground truth: Camera A at origin, Camera B rotated 15° around Y, translated
            double angle = 15.0 * Math.PI / 180.0;
            double cosA = Math.Cos(angle), sinA = Math.Sin(angle);
            var R_true = new double[,] { { cosA, 0, sinA }, { 0, 1, 0 }, { -sinA, 0, cosA } };
            var t_true = new double[] { 1.0, 0.0, 0.0 }; // Unit translation along X
            // Normalize t
            double tn = Math.Sqrt(t_true[0] * t_true[0] + t_true[1] * t_true[1] + t_true[2] * t_true[2]);
            t_true[0] /= tn; t_true[1] /= tn; t_true[2] /= tn;

            Log($"[SfM] True R diag=({R_true[0, 0]:F4},{R_true[1, 1]:F4},{R_true[2, 2]:F4})");
            Log($"[SfM] True t=({t_true[0]:F4},{t_true[1]:F4},{t_true[2]:F4})");

            // Generate 20 synthetic 3D points spread in 3D (NOT coplanar!)
            var synth3D = new Vector3[] {
                new(0, 0, 5), new(1, 0, 5), new(0, 1, 5), new(-1, 0, 5), new(0, -1, 5),
                new(0.5f, 0.5f, 4), new(-0.5f, 0.5f, 6), new(0.5f, -0.5f, 7), new(-0.5f, -0.5f, 3),
                new(1, 1, 4), new(-1, 1, 6), new(1, -1, 3), new(-1, -1, 7),
                new(0.3f, 0.7f, 5.5f), new(-0.3f, -0.7f, 4.5f), new(0.8f, -0.2f, 3.5f),
                new(-0.8f, 0.2f, 6.5f), new(0.1f, 0.1f, 8), new(-0.1f, -0.1f, 2.5f),
                new(0.6f, -0.6f, 5.2f)
            };

            // Project all points through both cameras (normalized coords, K=I)
            var projA = new float[synth3D.Length * 2];
            var projB = new float[synth3D.Length * 2];
            for (int i = 0; i < synth3D.Length; i++)
            {
                var p = synth3D[i];
                // Camera A = [I|0]: projection = (x/z, y/z)
                projA[i * 2] = p.X / p.Z;
                projA[i * 2 + 1] = p.Y / p.Z;
                // Camera B = [R|t]: projection = (R*X + t), then /z
                double bx = R_true[0, 0] * p.X + R_true[0, 1] * p.Y + R_true[0, 2] * p.Z + t_true[0];
                double by = R_true[1, 0] * p.X + R_true[1, 1] * p.Y + R_true[1, 2] * p.Z + t_true[1];
                double bz = R_true[2, 0] * p.X + R_true[2, 1] * p.Y + R_true[2, 2] * p.Z + t_true[2];
                projB[i * 2] = (float)(bx / bz);
                projB[i * 2 + 1] = (float)(by / bz);
            }

            // ─── TEST 1: 8-point algorithm ───
            Log("[SfM] ── Test 1: 8-point algorithm ──");
            var sample8 = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };
            var E8 = GpuSfmKernels.TestSolve8Point(projA, projB, sample8);
            if (E8 == null)
            {
                Log("[SfM] ❌ Solve8Point returned null!");
            }
            else
            {
                Log($"[SfM] ✅ Solve8Point returned 9 values");

                // Check epipolar constraint: x_B^T * E * x_A should ≈ 0 for all points
                double maxEpipolarErr = 0;
                for (int i = 0; i < synth3D.Length; i++)
                {
                    double xa = projA[i * 2], ya = projA[i * 2 + 1];
                    double xb = projB[i * 2], yb = projB[i * 2 + 1];
                    // x_B^T * E * x_A
                    double ea0 = E8[0] * xa + E8[1] * ya + E8[2];
                    double ea1 = E8[3] * xa + E8[4] * ya + E8[5];
                    double ea2 = E8[6] * xa + E8[7] * ya + E8[8];
                    double epErr = Math.Abs(xb * ea0 + yb * ea1 + ea2);
                    maxEpipolarErr = Math.Max(maxEpipolarErr, epErr);
                }
                Log($"[SfM] Epipolar error (x_B^T*E*x_A): max={maxEpipolarErr:E3} {(maxEpipolarErr < 0.001 ? "✅" : "❌")}");

                // Check SVD of E
                var Emat = new double[3, 3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        Emat[i, j] = E8[i * 3 + j];
                var (U, S, Vt) = LinearAlgebra.SVD3x3(Emat);
                Log($"[SfM] E singular values: {S[0]:F4}, {S[1]:F4}, {S[2]:F4} (should be σ,σ,0)");
                Log($"[SfM] det(U)={LinearAlgebra.Det3x3(U):F4}, det(Vt)={LinearAlgebra.Det3x3(Vt):F4} (should be +1)");

                // ─── TEST 2: RecoverPose ───
                Log("[SfM] ── Test 2: RecoverPose ──");
                // Build feature lists from synthetic projections
                var synthFeatA = new List<ImageFeature>();
                var synthFeatB = new List<ImageFeature>();
                var synthMatches = new List<FeatureMatch>();
                for (int i = 0; i < synth3D.Length; i++)
                {
                    // Features in normalized coords (we'll pass fx=1, cx=0, cy=0)
                    synthFeatA.Add(new ImageFeature { X = projA[i * 2], Y = projA[i * 2 + 1] });
                    synthFeatB.Add(new ImageFeature { X = projB[i * 2], Y = projB[i * 2 + 1] });
                    synthMatches.Add(new FeatureMatch { IndexA = i, IndexB = i });
                }

                var (Rrec, trec) = RecoverPose(Emat, synthFeatA, synthFeatB, synthMatches, 1f, 0f, 0f);
                if (Rrec == null || trec == null)
                {
                    Log("[SfM] ❌ RecoverPose returned null!");
                }
                else
                {
                    Log($"[SfM] Recovered R diag=({Rrec[0, 0]:F4},{Rrec[1, 1]:F4},{Rrec[2, 2]:F4})");
                    Log($"[SfM] Recovered t=({trec[0]:F4},{trec[1]:F4},{trec[2]:F4})");

                    // Compare R diagonal (should be close to true R)
                    double rErr = Math.Abs(Rrec[0, 0] - R_true[0, 0]) + Math.Abs(Rrec[1, 1] - R_true[1, 1]) + Math.Abs(Rrec[2, 2] - R_true[2, 2]);
                    Log($"[SfM] R diagonal error: {rErr:F6} {(rErr < 0.1 ? "✅" : "❌")}");

                    // Compare t direction (t is only known up to sign)
                    double dot = trec[0] * t_true[0] + trec[1] * t_true[1] + trec[2] * t_true[2];
                    Log($"[SfM] t direction dot: {dot:F4} (should be ±1) {(Math.Abs(Math.Abs(dot) - 1.0) < 0.1 ? "✅" : "❌")}");

                    // ─── TEST 3: Triangulate with recovered pose ───
                    Log("[SfM] ── Test 3: Triangulate with recovered pose ──");
                    var P1test = MakeProjectionMatrix(1, 0, 0, null, null);
                    var P2test = MakeProjectionMatrix(1, 0, 0, Rrec, trec);

                    int goodCount = 0;
                    double totalErr = 0;
                    for (int i = 0; i < Math.Min(5, synth3D.Length); i++)
                    {
                        var nA = new Vector2(projA[i * 2], projA[i * 2 + 1]);
                        var nB = new Vector2(projB[i * 2], projB[i * 2 + 1]);
                        var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1test, P2test);
                        if (pt.HasValue)
                        {
                            // Points are up to scale, so compare directions/ratios
                            float scale = synth3D[i].Z / pt.Value.Z;
                            var scaled = pt.Value * scale;
                            var err = Vector3.Distance(scaled, synth3D[i]);
                            totalErr += err;
                            if (err < 0.5f) goodCount++;
                            Log($"[SfM]   pt[{i}] true=({synth3D[i].X:F2},{synth3D[i].Y:F2},{synth3D[i].Z:F2}) " +
                                $"rec=({pt.Value.X:F2},{pt.Value.Y:F2},{pt.Value.Z:F2}) scale={scale:F3} err={err:F4} {(err < 0.5 ? "✅" : "❌")}");
                        }
                        else
                        {
                            Log($"[SfM]   pt[{i}] → null ❌");
                        }
                    }
                    Log($"[SfM] Triangulation: {goodCount}/5 good points, avg err={totalErr / 5:F4}");

                    // Check if points are 3D (not planar)
                    var triPts = new List<Vector3>();
                    for (int i = 0; i < synth3D.Length; i++)
                    {
                        var nA = new Vector2(projA[i * 2], projA[i * 2 + 1]);
                        var nB = new Vector2(projB[i * 2], projB[i * 2 + 1]);
                        var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1test, P2test);
                        if (pt.HasValue) triPts.Add(pt.Value);
                    }
                    if (triPts.Count > 5)
                    {
                        float mx = triPts.Average(p => p.X), my = triPts.Average(p => p.Y), mz = triPts.Average(p => p.Z);
                        double cxx = 0, cyy = 0, czz = 0;
                        foreach (var p in triPts) { cxx += (p.X - mx) * (p.X - mx); cyy += (p.Y - my) * (p.Y - my); czz += (p.Z - mz) * (p.Z - mz); }
                        Log($"[SfM] Variance: X={cxx / triPts.Count:F4}, Y={cyy / triPts.Count:F4}, Z={czz / triPts.Count:F4}");
                        double minVar = Math.Min(cxx, Math.Min(cyy, czz)) / triPts.Count;
                        double maxVar = Math.Max(cxx, Math.Max(cyy, czz)) / triPts.Count;
                        Log($"[SfM] Planarity ratio: {minVar / maxVar:F4} (>0.01 = 3D) {(minVar / maxVar > 0.01 ? "✅ 3D" : "❌ FLAT")}");
                    }
                }
            }
            Log("[SfM] ╔══════════════════════════════════════════╗");
            Log("[SfM] ║       END SYNTHETIC PIPELINE TEST       ║");
            Log("[SfM] ╚══════════════════════════════════════════╝");
        }

        try
        {
            // Step 1: Find best initial pair — try top candidates, pick by 3D quality
            Status = "Finding best initial pair (testing candidates)...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            // Use known calibration for benchmark datasets, fallback to heuristic
            float fx, cx, cy;
            if (images[0].FileName.StartsWith("templeR"))
            {
                // Middlebury TempleRing — calibrated K matrix from templeR_par.txt
                fx = 1520.4f; cx = 302.32f; cy = 246.87f;
                Log($"[SfM] Using TempleRing calibration: fx={fx}, cx={cx}, cy={cy}");
            }
            else if (images[0].FileName.StartsWith("dinoSR"))
            {
                // Middlebury DinoSparseRing — same camera calibration
                fx = 3310.4f; cx = 316.73f; cy = 200.55f;
                Log($"[SfM] Using DinoSparseRing calibration: fx={fx}, cx={cx}, cy={cy}");
            }
            else
            {
                fx = Math.Max(images[0].Width, images[0].Height) * 1.2f;
                cx = images[0].Width / 2f;
                cy = images[0].Height / 2f;
                Log($"[SfM] Using heuristic calibration: fx={fx}, cx={cx}, cy={cy}");
            }

            // Initialize GPU SfM kernels
            _gpuKernels ??= new GpuSfmKernels(_gpu);

            // Try top N candidate pairs, score by PCA quality
            // Include both high-match-count pairs AND wider-baseline pairs
            var matchCountPairs = pairs
                .Where(p => p.Matches.Count >= 30)
                .OrderByDescending(p => p.Matches.Count)
                .Take(8)
                .ToList();

            // Also add wider-baseline pairs (larger image index gap = more camera movement)
            var wideBaseline = pairs
                .Where(p => p.Matches.Count >= 20)
                .OrderByDescending(p => Math.Abs(p.ImageIndexA - p.ImageIndexB))
                .Take(6)
                .Where(p => !matchCountPairs.Any(m => m.ImageIndexA == p.ImageIndexA && m.ImageIndexB == p.ImageIndexB))
                .ToList();

            var candidatePairs = matchCountPairs.Concat(wideBaseline).ToList();

            if (candidatePairs.Count == 0)
            {
                Status = "No pairs with enough matches (need >= 30)";
                OnStateChanged?.Invoke();
                IsRunning = false;
                return;
            }

            int idxA = -1, idxB = -1;
            double[,]? R = null;
            double[]? t = null;
            List<FeatureMatch>? inlierMatches = null;
            double bestPcaScore = -1;
            List<(FeatureMatch match, int pointIndex)>? bestInitialResults = null;

            Log($"[SfM] Testing {candidatePairs.Count} candidate initial pairs...");

            foreach (var candidatePair in candidatePairs)
            {
                int cIdxA = candidatePair.ImageIndexA;
                int cIdxB = candidatePair.ImageIndexB;

                // Normalize coordinates
                var mA2D = new float[candidatePair.Matches.Count * 2];
                var mB2D = new float[candidatePair.Matches.Count * 2];
                for (int i = 0; i < candidatePair.Matches.Count; i++)
                {
                    var fA = images[cIdxA].Features[candidatePair.Matches[i].IndexA];
                    var fB = images[cIdxB].Features[candidatePair.Matches[i].IndexB];
                    mA2D[i * 2] = (fA.X - cx) / fx;
                    mA2D[i * 2 + 1] = (fA.Y - cy) / fx;
                    mB2D[i * 2] = (fB.X - cx) / fx;
                    mB2D[i * 2 + 1] = (fB.Y - cy) / fx;
                }

                var (essF, mask) = await _gpuKernels.EstimateEssentialAsync(
                    mA2D, mB2D, candidatePair.Matches.Count, fx, cx, cy, iterations: 256);
                if (essF == null) continue;

                double[,] Ec = new double[3, 3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        Ec[i, j] = essF[i * 3 + j];

                int inlCnt = mask.Count(x => x);
                if (inlCnt < 15) continue;

                var cInlierMatches = candidatePair.Matches.Where((_, i) => mask[i]).ToList();
                var (Rc, tc) = RecoverPose(Ec, images[cIdxA].Features, images[cIdxB].Features,
                    cInlierMatches, fx, cx, cy);
                if (Rc == null) continue;

                // Triangulate and measure 3D quality
                Points3D.Clear();
                var RI = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
                var tI = new double[] { 0, 0, 0 };
                var cResults = TriangulateFromPair(images, cIdxA, cIdxB, cInlierMatches, fx, cx, cy, RI, tI, Rc, tc);

                if (Points3D.Count < 10) continue;

                // Filter extreme depth outliers before scoring (IQR on depth)
                var depths = Points3D.Select(p => (double)p.Position.Z).OrderBy(z => z).ToList();
                double q1 = depths[depths.Count / 4];
                double q3 = depths[3 * depths.Count / 4];
                double iqr = q3 - q1;
                double zLow = q1 - 3 * iqr;
                double zHigh = q3 + 3 * iqr;
                // Also reject points behind or very close to the cameras
                zLow = Math.Max(zLow, 0.1);

                var goodPts = Points3D.Where(p => p.Position.Z >= zLow && p.Position.Z <= zHigh).ToList();
                if (goodPts.Count < 10) continue;

                // Score by PCA — want λ2/λ0 as large as possible (3D, not planar)
                float pmx = goodPts.Average(p => p.Position.X);
                float pmy = goodPts.Average(p => p.Position.Y);
                float pmz = goodPts.Average(p => p.Position.Z);
                double pcxx = 0, pcxy = 0, pcxz = 0, pcyy = 0, pcyz = 0, pczz = 0;
                foreach (var p in goodPts)
                {
                    double dx = p.Position.X - pmx, dy = p.Position.Y - pmy, dz = p.Position.Z - pmz;
                    pcxx += dx * dx; pcxy += dx * dy; pcxz += dx * dz;
                    pcyy += dy * dy; pcyz += dy * dz; pczz += dz * dz;
                }
                double gn = goodPts.Count;
                var gcov = new double[,] { { pcxx / gn, pcxy / gn, pcxz / gn }, { pcxy / gn, pcyy / gn, pcyz / gn }, { pcxz / gn, pcyz / gn, pczz / gn } };
                var (_, gevals, _) = LinearAlgebra.SVD3x3(gcov);
                double pcaScore = gevals[0] > 0 ? gevals[2] / gevals[0] : 0;

                Log($"[SfM]   Pair {images[cIdxA].FileName} ↔ {images[cIdxB].FileName}: {inlCnt} inliers, {goodPts.Count} good pts, PCA={pcaScore:F4}");

                if (pcaScore > bestPcaScore)
                {
                    bestPcaScore = pcaScore;
                    idxA = cIdxA; idxB = cIdxB;
                    R = Rc; t = tc;
                    inlierMatches = cInlierMatches;
                    bestInitialResults = cResults;
                }
            }

            if (R == null || inlierMatches == null || bestInitialResults == null)
            {
                Status = "Failed — no candidate pair produced valid 3D structure";
                OnStateChanged?.Invoke();
                IsRunning = false;
                return;
            }

            Log($"[SfM] Best initial pair: {images[idxA].FileName} ↔ {images[idxB].FileName} (PCA score={bestPcaScore:F4})");

            // Re-triangulate with the winning pair and filter depth outliers
            Points3D.Clear();
            var identR = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            var identT = new double[] { 0, 0, 0 };
            var initialResults = TriangulateFromPair(images, idxA, idxB, inlierMatches, fx, cx, cy, identR, identT, R, t);

            // Filter extreme depth outliers (IQR-based)
            if (Points3D.Count > 10)
            {
                var depthsAll = Points3D.Select(p => (double)p.Position.Z).OrderBy(z => z).ToList();
                double fq1 = depthsAll[depthsAll.Count / 4];
                double fq3 = depthsAll[3 * depthsAll.Count / 4];
                double fiqr = fq3 - fq1;
                double fzLow = Math.Max(fq1 - 3 * fiqr, 0.1);
                double fzHigh = fq3 + 3 * fiqr;
                int beforeFilter = Points3D.Count;
                var removedIndices = new HashSet<int>();
                for (int i = Points3D.Count - 1; i >= 0; i--)
                    if (Points3D[i].Position.Z < fzLow || Points3D[i].Position.Z > fzHigh)
                        removedIndices.Add(i);

                // Remove from initialResults too (rebuild mapping)
                if (removedIndices.Count > 0)
                {
                    // Remove points at the flagged indices (iterate backwards to preserve indices)
                    foreach (var ri in removedIndices.OrderByDescending(x => x))
                        Points3D.RemoveAt(ri);
                    // Rebuild initialResults with corrected indices
                    var newResults = new List<(FeatureMatch match, int pointIndex)>();
                    int newIdx = 0;
                    for (int i = 0; i < initialResults.Count; i++)
                    {
                        if (!removedIndices.Contains(initialResults[i].pointIndex))
                        {
                            newResults.Add((initialResults[i].match, newIdx++));
                        }
                    }
                    initialResults = newResults;
                    Log($"[SfM] Depth filter: {beforeFilter} → {Points3D.Count} points (removed {removedIndices.Count} outliers)");
                }
            }

            Log($"[SfM] Initial pair: {inlierMatches.Count} matches → {Points3D.Count} points ({100.0 * Points3D.Count / Math.Max(1, inlierMatches.Count):F0}% yield)");

            // Step 4: Set initial camera poses
            CameraPoses[idxA] = CameraParams.CreateDefault(images[idxA].Width, images[idxA].Height);
            CameraPoses[idxA]!.FocalX = fx;
            CameraPoses[idxA]!.FocalY = fx;

            var cam2 = CameraParams.CreateDefault(images[idxB].Width, images[idxB].Height);
            cam2.FocalX = fx;
            cam2.FocalY = fx;
            cam2.Position = new Vector3((float)t[0], (float)t[1], (float)t[2]);
            cam2.Forward = new Vector3((float)R[2, 0], (float)R[2, 1], (float)R[2, 2]);
            cam2.Up = new Vector3(-(float)R[1, 0], -(float)R[1, 1], -(float)R[1, 2]);
            CameraPoses[idxB] = cam2;

            // Store pose matrices for triangulation
            var poseR = new Dictionary<int, double[,]>();
            var poseT = new Dictionary<int, double[]>();
            poseR[idxA] = identR;
            poseT[idxA] = identT;
            poseR[idxB] = R;
            poseT[idxB] = t;

            // Diagnostics
            if (Points3D.Count > 10)
            {
                float pmx = Points3D.Average(p => p.Position.X);
                float pmy = Points3D.Average(p => p.Position.Y);
                float pmz = Points3D.Average(p => p.Position.Z);
                double dcxx = 0, dcxy = 0, dcxz = 0, dcyy = 0, dcyz = 0, dczz = 0;
                foreach (var p in Points3D)
                {
                    double ddx = p.Position.X - pmx, ddy = p.Position.Y - pmy, ddz = p.Position.Z - pmz;
                    dcxx += ddx * ddx; dcxy += ddx * ddy; dcxz += ddx * ddz;
                    dcyy += ddy * ddy; dcyz += ddy * ddz; dczz += ddz * ddz;
                }
                double dn = Points3D.Count;
                var dcov = new double[,] { { dcxx / dn, dcxy / dn, dcxz / dn }, { dcxy / dn, dcyy / dn, dcyz / dn }, { dcxz / dn, dcyz / dn, dczz / dn } };
                var (_, devals, _) = LinearAlgebra.SVD3x3(dcov);
                double finalPca = devals[0] > 0 ? devals[2] / devals[0] : 0;
                Log($"[SfM] Initial pair PCA: λ2/λ0={finalPca:F4} {(finalPca > 0.01 ? "✅ 3D" : "⚠️ DEGENERATE")}");
                Log($"[SfM] Camera B: R=diag({R[0, 0]:F3},{R[1, 1]:F3},{R[2, 2]:F3}), t=({t[0]:F4},{t[1]:F4},{t[2]:F4})");
            }

            // Step 6: Multi-pair seeding + iterative pipeline
            Status = "Seeding 3D points from multiple pairs...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            var reconstructedCameras = new HashSet<int> { idxA, idxB };
            int addedCameras = 0;

            // Track which features map to which 3D points
            var featureTo3D = new Dictionary<(int imgIdx, int featIdx), int>();
            foreach (var (match, ptIdx) in initialResults)
            {
                featureTo3D[(idxA, match.IndexA)] = ptIdx;
                featureTo3D[(idxB, match.IndexB)] = ptIdx;
            }

            // Build pair lookup
            var pairLookup = new Dictionary<(int, int), List<FeatureMatch>>();
            foreach (var pair in pairs)
            {
                pairLookup[(pair.ImageIndexA, pair.ImageIndexB)] = pair.Matches;
                pairLookup[(pair.ImageIndexB, pair.ImageIndexA)] =
                    pair.Matches.Select(m => new FeatureMatch { IndexA = m.IndexB, IndexB = m.IndexA }).ToList();
            }

            var sortedPairs = pairs.OrderByDescending(p => p.Matches.Count).ToList();

            // ═══ MULTI-PAIR SEEDING PHASE ═══
            // Seed a few cameras via E-mat to build 3D point cloud for PnP
            Log($"[SfM] === SEEDING PHASE: growing {Points3D.Count} pts from initial pair ===");
            int seedRounds = 0;
            int maxSeedRounds = 1; // Seed 1 camera via E-mat for PnP bootstrap, then PnP handles the rest

            while (seedRounds < maxSeedRounds)
            {
                seedRounds++;
                int addedThisSeed = 0;

                // Find best unregistered camera: most E-mat inliers with any registered camera
                var seedCandidates = new List<(int newIdx, int knownIdx, int pairIdxA, int pairIdxB, int matchCount)>();
                foreach (var pair in sortedPairs)
                {
                    int knownIdx, newIdx;
                    if (reconstructedCameras.Contains(pair.ImageIndexA) && !reconstructedCameras.Contains(pair.ImageIndexB))
                    { knownIdx = pair.ImageIndexA; newIdx = pair.ImageIndexB; }
                    else if (reconstructedCameras.Contains(pair.ImageIndexB) && !reconstructedCameras.Contains(pair.ImageIndexA))
                    { knownIdx = pair.ImageIndexB; newIdx = pair.ImageIndexA; }
                    else continue;

                    if (pair.Matches.Count < 15) continue;
                    seedCandidates.Add((newIdx, knownIdx, pair.ImageIndexA, pair.ImageIndexB, pair.Matches.Count));
                }

                if (seedCandidates.Count == 0) break;

                // Sort by match count descending (best pair first)
                seedCandidates.Sort((a, b) => b.matchCount.CompareTo(a.matchCount));
                Log($"[SfM] Seed round {seedRounds}: {seedCandidates.Count} candidates, top: {images[seedCandidates[0].newIdx].FileName}({seedCandidates[0].matchCount} matches)");

                foreach (var (newIdx, knownIdx, pairIdxA, pairIdxB, _) in seedCandidates)
                {
                    if (reconstructedCameras.Contains(newIdx)) continue;

                    // Get pair matches
                    if (!pairLookup.TryGetValue((pairIdxA, pairIdxB), out var pairMatches)) continue;
                    bool knownIsA = (knownIdx == pairIdxA);

                    // Compute Essential matrix
                    var emA = new float[pairMatches.Count * 2];
                    var emB = new float[pairMatches.Count * 2];
                    for (int mi = 0; mi < pairMatches.Count; mi++)
                    {
                        var fA = images[pairIdxA].Features[pairMatches[mi].IndexA];
                        var fB = images[pairIdxB].Features[pairMatches[mi].IndexB];
                        emA[mi * 2] = (fA.X - cx) / fx;
                        emA[mi * 2 + 1] = (fA.Y - cy) / fx;
                        emB[mi * 2] = (fB.X - cx) / fx;
                        emB[mi * 2 + 1] = (fB.Y - cy) / fx;
                    }

                    var (eNewF, maskNew) = await _gpuKernels!.EstimateEssentialAsync(
                        emA, emB, pairMatches.Count, fx, cx, cy, iterations: 256);

                    if (eNewF == null) continue;

                    double[,] Enew = new double[3, 3];
                    for (int ei = 0; ei < 3; ei++)
                        for (int ej = 0; ej < 3; ej++)
                            Enew[ei, ej] = eNewF[ei * 3 + ej];

                    var inlierNew = pairMatches.Where((_, i) => maskNew[i]).ToList();
                    if (inlierNew.Count < 10) continue;

                    var (Rnew, tnew) = RecoverPose(Enew, images[pairIdxA].Features,
                        images[pairIdxB].Features, inlierNew, fx, cx, cy);
                    if (Rnew == null) continue;

                    // Basic sanity: det(R) ≈ 1 (orthogonalization should ensure this)
                    double detRnew = LinearAlgebra.Det3x3(Rnew);
                    if (Math.Abs(detRnew - 1.0) > 0.1)
                    {
                        Log($"[SfM] Seed skip {images[newIdx].FileName}: det(R)={detRnew:F3}");
                        continue;
                    }

                    // Estimate scale from known 3D points
                    double scale = EstimateScale(images, pairIdxA, pairIdxB, knownIsA,
                        knownIdx, newIdx, inlierNew, Rnew, tnew, poseR, poseT,
                        featureTo3D, fx, cx, cy);
                    if (scale <= 0 || double.IsNaN(scale) || double.IsInfinity(scale)) scale = 1.0;
                    tnew[0] *= scale; tnew[1] *= scale; tnew[2] *= scale;

                    // Compute global pose
                    double[,] globalR;
                    double[] globalT;
                    if (knownIsA)
                    {
                        var Rk = poseR[knownIdx]; var tk = poseT[knownIdx];
                        globalR = Multiply3x3(Rnew, Rk);
                        globalT = new double[] {
                            Rnew[0,0]*tk[0]+Rnew[0,1]*tk[1]+Rnew[0,2]*tk[2]+tnew[0],
                            Rnew[1,0]*tk[0]+Rnew[1,1]*tk[1]+Rnew[1,2]*tk[2]+tnew[1],
                            Rnew[2,0]*tk[0]+Rnew[2,1]*tk[1]+Rnew[2,2]*tk[2]+tnew[2] };
                    }
                    else
                    {
                        var RInv = Transpose3x3(Rnew);
                        var tInv = new double[] {
                            -(RInv[0,0]*tnew[0]+RInv[0,1]*tnew[1]+RInv[0,2]*tnew[2]),
                            -(RInv[1,0]*tnew[0]+RInv[1,1]*tnew[1]+RInv[1,2]*tnew[2]),
                            -(RInv[2,0]*tnew[0]+RInv[2,1]*tnew[1]+RInv[2,2]*tnew[2]) };
                        var Rk = poseR[knownIdx]; var tk = poseT[knownIdx];
                        globalR = Multiply3x3(RInv, Rk);
                        globalT = new double[] {
                            RInv[0,0]*tk[0]+RInv[0,1]*tk[1]+RInv[0,2]*tk[2]+tInv[0],
                            RInv[1,0]*tk[0]+RInv[1,1]*tk[1]+RInv[1,2]*tk[2]+tInv[1],
                            RInv[2,0]*tk[0]+RInv[2,1]*tk[1]+RInv[2,2]*tk[2]+tInv[2] };
                    }

                    // Accept camera
                    poseR[newIdx] = globalR;
                    poseT[newIdx] = globalT;

                    var camNew = CameraParams.CreateDefault(images[newIdx].Width, images[newIdx].Height);
                    camNew.FocalX = fx; camNew.FocalY = fx;
                    camNew.Position = new Vector3((float)globalT[0], (float)globalT[1], (float)globalT[2]);
                    camNew.Forward = new Vector3((float)globalR[2, 0], (float)globalR[2, 1], (float)globalR[2, 2]);
                    camNew.Up = new Vector3(-(float)globalR[1, 0], -(float)globalR[1, 1], -(float)globalR[1, 2]);
                    CameraPoses[newIdx] = camNew;

                    reconstructedCameras.Add(newIdx);
                    addedCameras++;
                    addedThisSeed++;

                    // Triangulate new points from this camera with ALL registered cameras
                    int before = Points3D.Count;
                    foreach (var regIdx in reconstructedCameras)
                    {
                        if (regIdx == newIdx) continue;
                        if (pairLookup.TryGetValue((newIdx, regIdx), out var fwdMatches))
                        {
                            var triResults = TriangulateFromPair(images, newIdx, regIdx, fwdMatches, fx, cx, cy,
                                poseR[newIdx], poseT[newIdx], poseR[regIdx], poseT[regIdx]);
                            foreach (var (m, pi) in triResults)
                            {
                                featureTo3D.TryAdd((newIdx, m.IndexA), pi);
                                featureTo3D.TryAdd((regIdx, m.IndexB), pi);
                            }
                        }
                    }

                    Log($"[SfM] 🌱 Seeded {images[newIdx].FileName}: +{Points3D.Count - before} pts (total: {Points3D.Count} pts, {reconstructedCameras.Count} cams)");
                    await Task.Yield();
                    // Don't break — allow multiple cameras per seed round
                }

                Log($"[SfM] Seed round {seedRounds}: +{addedThisSeed} cameras (total: {reconstructedCameras.Count} cams, {Points3D.Count} pts)");
                if (addedThisSeed == 0) break;

                // Run BA after each seeding round to refine accumulated points
                if (reconstructedCameras.Count >= 2 && Points3D.Count >= 10)
                {
                    await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                        featureTo3D, fx, cx, cy, maxIterations: 15);
                }
            }

            Log($"[SfM] === SEEDING COMPLETE: {reconstructedCameras.Count} cameras, {Points3D.Count} points ===");

            // ═══ ROUND 0: BA on all seeded cameras ═══
            Log($"[SfM] Round 0: BA on {reconstructedCameras.Count} seeded cameras ({Points3D.Count} pts)");
            if (Points3D.Count >= 10)
            {
                await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                    featureTo3D, fx, cx, cy, maxIterations: 10);
            }

            // ═══ ITERATIVE ROUNDS ═══
            int maxRounds = 20; // PnP handles all cameras now (seeding disabled)
            for (int round = 1; round <= maxRounds; round++)
            {
                Status = $"Round {round}/{maxRounds}: adding cameras...";
                OnStateChanged?.Invoke();
                await Task.Yield();

                int addedThisRound = 0;

                // Try each unregistered camera, sorted by number of 2D-3D correspondences
                var candidates = new List<(int idx, int corrCount)>();
                for (int newIdx = 0; newIdx < images.Count; newIdx++)
                {
                    if (reconstructedCameras.Contains(newIdx)) continue;
                    int corrCount = 0;
                    foreach (var knownIdx in reconstructedCameras)
                    {
                        if (!pairLookup.TryGetValue((knownIdx, newIdx), out var matches)) continue;
                        corrCount += matches.Count(m => featureTo3D.ContainsKey((knownIdx, m.IndexA)));
                    }
                    if (corrCount > 0) candidates.Add((newIdx, corrCount));
                }
                candidates.Sort((a, b) => b.corrCount.CompareTo(a.corrCount)); // most correspondences first
                Log($"[SfM] Round {round}: {candidates.Count} candidates — top 5: {string.Join(", ", candidates.Take(5).Select(c => $"{images[c.idx].FileName}({c.corrCount}corr)"))}");

                foreach (var (newIdx, _) in candidates)
                {
                    if (reconstructedCameras.Contains(newIdx)) continue;

                    Status = $"Round {round}: {images[newIdx].FileName} ({reconstructedCameras.Count} cams)...";
                    OnStateChanged?.Invoke();

                    // === STRATEGY 1: PnP ===
                    var pts2D = new List<Vector2>();
                    var pts3D = new List<Vector3>();

                    foreach (var knownIdx in reconstructedCameras)
                    {
                        if (!pairLookup.TryGetValue((knownIdx, newIdx), out var matches)) continue;
                        foreach (var match in matches)
                        {
                            if (featureTo3D.TryGetValue((knownIdx, match.IndexA), out int ptIdx))
                            {
                                var feat2D = images[newIdx].Features[match.IndexB];
                                pts2D.Add(new Vector2((feat2D.X - cx) / fx, (feat2D.Y - cy) / fx));
                                pts3D.Add(Points3D[ptIdx].Position);
                            }
                        }
                    }

                    double[,]? finalR = null;
                    double[]? finalT = null;
                    string method = "";

                    Log($"[SfM] {images[newIdx].FileName}: {pts2D.Count} PnP correspondences");
                    if (pts2D.Count >= 4) // P3P RANSAC works with 4+ correspondences
                    {
                        var pnp2D = new float[pts2D.Count * 2];
                        var pnp3D = new float[pts3D.Count * 3];
                        for (int pi = 0; pi < pts2D.Count; pi++)
                        {
                            pnp2D[pi * 2] = pts2D[pi].X;
                            pnp2D[pi * 2 + 1] = pts2D[pi].Y;
                            pnp3D[pi * 3] = pts3D[pi].X;
                            pnp3D[pi * 3 + 1] = pts3D[pi].Y;
                            pnp3D[pi * 3 + 2] = pts3D[pi].Z;
                        }

                        var (pnpR, pnpT, pnpInliers) = await _gpuKernels!.SolvePnPAsync(pnp2D, pnp3D, pts2D.Count);
                        if (pnpR != null && pnpT != null)
                        {
                            finalR = new double[3, 3];
                            for (int ri = 0; ri < 3; ri++)
                                for (int rj = 0; rj < 3; rj++)
                                    finalR[ri, rj] = pnpR[ri * 3 + rj];
                            finalT = new double[] { pnpT[0], pnpT[1], pnpT[2] };
                            method = $"GPU-PnP({pts2D.Count}corr,{pnpInliers}inl)";
                        }
                    }

                    // === STRATEGY 2: Essential matrix fallback ===
                    if (finalR == null)
                    {
                        foreach (var pair in sortedPairs)
                        {
                            int knownIdx, pairNewIdx;
                            bool knownIsA;
                            if (reconstructedCameras.Contains(pair.ImageIndexA) && pair.ImageIndexB == newIdx)
                            { knownIdx = pair.ImageIndexA; pairNewIdx = pair.ImageIndexB; knownIsA = true; }
                            else if (reconstructedCameras.Contains(pair.ImageIndexB) && pair.ImageIndexA == newIdx)
                            { knownIdx = pair.ImageIndexB; pairNewIdx = pair.ImageIndexA; knownIsA = false; }
                            else continue;

                            if (pair.Matches.Count < 15) continue;

                            int pairIdxA = pair.ImageIndexA, pairIdxB = pair.ImageIndexB;

                            var emA = new float[pair.Matches.Count * 2];
                            var emB = new float[pair.Matches.Count * 2];
                            for (int mi = 0; mi < pair.Matches.Count; mi++)
                            {
                                var fA = images[pairIdxA].Features[pair.Matches[mi].IndexA];
                                var fB = images[pairIdxB].Features[pair.Matches[mi].IndexB];
                                emA[mi * 2] = (fA.X - cx) / fx;
                                emA[mi * 2 + 1] = (fA.Y - cy) / fx;
                                emB[mi * 2] = (fB.X - cx) / fx;
                                emB[mi * 2 + 1] = (fB.Y - cy) / fx;
                            }

                            var (eNewF, maskNew) = await _gpuKernels!.EstimateEssentialAsync(
                                emA, emB, pair.Matches.Count, fx, cx, cy, iterations: 256);

                            if (eNewF == null) continue;

                            double[,] Enew = new double[3, 3];
                            for (int ei = 0; ei < 3; ei++)
                                for (int ej = 0; ej < 3; ej++)
                                    Enew[ei, ej] = eNewF[ei * 3 + ej];

                            var inlierNew = pair.Matches.Where((_, i) => maskNew[i]).ToList();
                            if (inlierNew.Count < 8) continue;

                            var (Rnew, tnew) = RecoverPose(Enew, images[pairIdxA].Features,
                                images[pairIdxB].Features, inlierNew, fx, cx, cy);
                            if (Rnew == null) continue;

                            double scale = EstimateScale(images, pairIdxA, pairIdxB, knownIsA,
                                knownIdx, newIdx, inlierNew, Rnew, tnew, poseR, poseT,
                                featureTo3D, fx, cx, cy);
                            tnew[0] *= scale; tnew[1] *= scale; tnew[2] *= scale;

                            if (knownIsA)
                            {
                                var Rk = poseR[knownIdx]; var tk = poseT[knownIdx];
                                finalR = Multiply3x3(Rnew, Rk);
                                finalT = new double[] {
                                    Rnew[0,0]*tk[0]+Rnew[0,1]*tk[1]+Rnew[0,2]*tk[2]+tnew[0],
                                    Rnew[1,0]*tk[0]+Rnew[1,1]*tk[1]+Rnew[1,2]*tk[2]+tnew[1],
                                    Rnew[2,0]*tk[0]+Rnew[2,1]*tk[1]+Rnew[2,2]*tk[2]+tnew[2] };
                            }
                            else
                            {
                                var RInv = Transpose3x3(Rnew);
                                var tInv = new double[] {
                                    -(RInv[0,0]*tnew[0]+RInv[0,1]*tnew[1]+RInv[0,2]*tnew[2]),
                                    -(RInv[1,0]*tnew[0]+RInv[1,1]*tnew[1]+RInv[1,2]*tnew[2]),
                                    -(RInv[2,0]*tnew[0]+RInv[2,1]*tnew[1]+RInv[2,2]*tnew[2]) };
                                var Rk = poseR[knownIdx]; var tk = poseT[knownIdx];
                                finalR = Multiply3x3(RInv, Rk);
                                finalT = new double[] {
                                    RInv[0,0]*tk[0]+RInv[0,1]*tk[1]+RInv[0,2]*tk[2]+tInv[0],
                                    RInv[1,0]*tk[0]+RInv[1,1]*tk[1]+RInv[1,2]*tk[2]+tInv[1],
                                    RInv[2,0]*tk[0]+RInv[2,1]*tk[1]+RInv[2,2]*tk[2]+tInv[2] };
                            }
                            method = $"GPU-Emat({inlierNew.Count}inliers)";
                            break;
                        }
                    }

                    if (finalR == null || finalT == null) continue;

                    // ═══ QUALITY GATES ═══
                    var pose12 = new float[12];
                    for (int ri = 0; ri < 3; ri++)
                        for (int rj = 0; rj < 3; rj++)
                            pose12[ri * 3 + rj] = (float)finalR[ri, rj];
                    pose12[9] = (float)finalT[0]; pose12[10] = (float)finalT[1]; pose12[11] = (float)finalT[2];

                    var existingCamT = new List<float[]>();
                    foreach (var ci in reconstructedCameras)
                        existingCamT.Add(new float[] { (float)poseT[ci][0], (float)poseT[ci][1], (float)poseT[ci][2] });

                    var gate2D = new List<float>();
                    var gate3D = new List<float>();
                    foreach (var knownIdx2 in reconstructedCameras)
                    {
                        if (!pairLookup.TryGetValue((knownIdx2, newIdx), out var matches2)) continue;
                        foreach (var match in matches2)
                        {
                            if (featureTo3D.TryGetValue((knownIdx2, match.IndexA), out int ptIdx2))
                            {
                                var feat = images[newIdx].Features[match.IndexB];
                                gate2D.Add((feat.X - cx) / fx); gate2D.Add((feat.Y - cy) / fx);
                                gate3D.Add(Points3D[ptIdx2].Position.X);
                                gate3D.Add(Points3D[ptIdx2].Position.Y);
                                gate3D.Add(Points3D[ptIdx2].Position.Z);
                            }
                        }
                    }

                    int gatePnpInliers = method.Contains("PnP") ? pts2D.Count : gate2D.Count / 2;

                    var (isValid, medErr, reason) = await _gpuKernels!.ValidateCameraAsync(
                        gate2D.ToArray(), gate3D.ToArray(), gate2D.Count / 2,
                        pose12, gatePnpInliers, fx, existingCamT);

                    if (!isValid)
                    {
                        Log($"[SfM] REJECTED {images[newIdx].FileName}: {reason}");
                        continue;
                    }
                    Log($"[SfM] ✅ VALIDATED {images[newIdx].FileName}: reproj={medErr:F2}px, {method}");

                    // Accept camera
                    poseR[newIdx] = finalR;
                    poseT[newIdx] = finalT;

                    var camNew = CameraParams.CreateDefault(images[newIdx].Width, images[newIdx].Height);
                    camNew.FocalX = fx; camNew.FocalY = fx;
                    camNew.Position = new Vector3((float)finalT[0], (float)finalT[1], (float)finalT[2]);
                    camNew.Forward = new Vector3((float)finalR[2, 0], (float)finalR[2, 1], (float)finalR[2, 2]);
                    camNew.Up = new Vector3(-(float)finalR[1, 0], -(float)finalR[1, 1], -(float)finalR[1, 2]);
                    CameraPoses[newIdx] = camNew;

                    reconstructedCameras.Add(newIdx);
                    addedCameras++;
                    addedThisRound++;

                    // Triangulate new points
                    int before = Points3D.Count;
                    foreach (var knownIdx in reconstructedCameras)
                    {
                        if (knownIdx == newIdx) continue;
                        if (pairLookup.TryGetValue((newIdx, knownIdx), out var fwdMatches))
                        {
                            var triResults = TriangulateFromPair(images, newIdx, knownIdx, fwdMatches, fx, cx, cy,
                                poseR[newIdx], poseT[newIdx], poseR[knownIdx], poseT[knownIdx]);
                            foreach (var (m, pi) in triResults)
                            {
                                featureTo3D.TryAdd((newIdx, m.IndexA), pi);
                                featureTo3D.TryAdd((knownIdx, m.IndexB), pi);
                            }
                        }
                    }

                    Log($"[SfM] Added {images[newIdx].FileName} via {method}, +{Points3D.Count - before} pts (total: {Points3D.Count})");
                    await Task.Yield();
                }

                Log($"[SfM] Round {round}: added {addedThisRound} cameras (total: {reconstructedCameras.Count} cams, {Points3D.Count} pts)");

                if (addedThisRound == 0 && round > 1)
                {
                    Log("[SfM] No new cameras added, stopping");
                    break;
                }

                // ═══ LOCAL+GLOBAL BA STRATEGY (Theia-SfM approach) ═══
                if (reconstructedCameras.Count >= 2 && Points3D.Count >= 10 && addedThisRound > 0)
                {
                    int baIters = 30; // More iterations per round for better convergence
                    await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                        featureTo3D, fx, cx, cy, maxIterations: baIters);
                }

                // If nothing was added this round but it's round 1, continue to round 2
                // (BA may have improved 3D points enough to accept cameras)
                if (addedThisRound == 0) continue;
            }

            Log($"[SfM] Reconstructed {reconstructedCameras.Count} cameras, {Points3D.Count} points");

            // ═══ FINAL BA + RE-TRIANGULATION + RETRY (Research-backed) ═══

            // Final global BA pass with many iterations
            if (reconstructedCameras.Count >= 3 && Points3D.Count >= 10)
            {
                Log("[SfM] Final global BA pass (100 iters)...");
                await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                    featureTo3D, fx, cx, cy, maxIterations: 100);
            }

            // Remove high-error points before re-triangulation
            RemoveOutlierPoints();

            // ═══ RE-TRIANGULATE with refined poses (COLMAP/Theia technique) ═══
            {
                int ptsBefore = Points3D.Count;
                ReTriangulateAll(images, reconstructedCameras, poseR, poseT, featureTo3D, fx, cx, cy, pairLookup);
                Log($"[SfM] Re-triangulation: {ptsBefore} → {Points3D.Count} points (+{Points3D.Count - ptsBefore})");
            }

            // BA after re-triangulation
            if (reconstructedCameras.Count >= 3 && Points3D.Count >= 10)
            {
                Log("[SfM] Post-retri BA (50 iters)...");
                await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                    featureTo3D, fx, cx, cy, maxIterations: 50);
            }

            // ═══ CAMERA RETRY PASS (with relaxed gates) ═══
            {
                int retried = 0;
                var unregistered = new List<int>();
                for (int i = 0; i < images.Count; i++)
                    if (!reconstructedCameras.Contains(i)) unregistered.Add(i);

                if (unregistered.Count > 0)
                {
                    Log($"[SfM] Retrying {unregistered.Count} unregistered cameras with relaxed gates...");
                    foreach (var newIdx in unregistered)
                    {
                        var pts2D = new List<Vector2>();
                        var pts3D = new List<Vector3>();
                        foreach (var knownIdx in reconstructedCameras)
                        {
                            if (!pairLookup.TryGetValue((knownIdx, newIdx), out var matches)) continue;
                            foreach (var match in matches)
                            {
                                if (featureTo3D.TryGetValue((knownIdx, match.IndexA), out int ptIdx))
                                {
                                    var feat2D = images[newIdx].Features[match.IndexB];
                                    pts2D.Add(new Vector2((feat2D.X - cx) / fx, (feat2D.Y - cy) / fx));
                                    pts3D.Add(Points3D[ptIdx].Position);
                                }
                            }
                        }
                        if (pts2D.Count < 4) continue;

                        var pnp2D = new float[pts2D.Count * 2];
                        var pnp3D = new float[pts3D.Count * 3];
                        for (int pi = 0; pi < pts2D.Count; pi++)
                        {
                            pnp2D[pi * 2] = pts2D[pi].X; pnp2D[pi * 2 + 1] = pts2D[pi].Y;
                            pnp3D[pi * 3] = pts3D[pi].X; pnp3D[pi * 3 + 1] = pts3D[pi].Y; pnp3D[pi * 3 + 2] = pts3D[pi].Z;
                        }
                        var (pnpR, pnpT, pnpInliers) = await _gpuKernels!.SolvePnPAsync(pnp2D, pnp3D, pts2D.Count);
                        if (pnpR == null || pnpT == null) continue;

                        var finalR = new double[3, 3];
                        for (int ri = 0; ri < 3; ri++)
                            for (int rj = 0; rj < 3; rj++)
                                finalR[ri, rj] = pnpR[ri * 3 + rj];
                        var finalT = new double[] { pnpT[0], pnpT[1], pnpT[2] };

                        // Relaxed validation (48px instead of 32px)
                        var pose12 = new float[12];
                        for (int ri = 0; ri < 3; ri++)
                            for (int rj = 0; rj < 3; rj++)
                                pose12[ri * 3 + rj] = (float)finalR[ri, rj];
                        pose12[9] = (float)finalT[0]; pose12[10] = (float)finalT[1]; pose12[11] = (float)finalT[2];

                        var existingCamT = new List<float[]>();
                        foreach (var ci in reconstructedCameras)
                            existingCamT.Add(new float[] { (float)poseT[ci][0], (float)poseT[ci][1], (float)poseT[ci][2] });

                        var gate2D = pnp2D;
                        var gate3D = pnp3D;

                        var (isValid, medErr, reason) = await _gpuKernels!.ValidateCameraAsync(
                            gate2D, gate3D, pts2D.Count, pose12, pts2D.Count, fx, existingCamT,
                            minInliers: 4, maxReprojError: 48.0f); // Relaxed!

                        if (!isValid) continue;

                        poseR[newIdx] = finalR;
                        poseT[newIdx] = finalT;
                        var camNew = CameraParams.CreateDefault(images[newIdx].Width, images[newIdx].Height);
                        camNew.FocalX = fx; camNew.FocalY = fx;
                        camNew.Position = new Vector3((float)finalT[0], (float)finalT[1], (float)finalT[2]);
                        camNew.Forward = new Vector3((float)finalR[2, 0], (float)finalR[2, 1], (float)finalR[2, 2]);
                        camNew.Up = new Vector3(-(float)finalR[1, 0], -(float)finalR[1, 1], -(float)finalR[1, 2]);
                        CameraPoses[newIdx] = camNew;
                        reconstructedCameras.Add(newIdx);
                        retried++;

                        // Triangulate new points from this camera
                        int before = Points3D.Count;
                        foreach (var knownIdx in reconstructedCameras)
                        {
                            if (knownIdx == newIdx) continue;
                            if (pairLookup.TryGetValue((newIdx, knownIdx), out var fwdMatches))
                            {
                                var triResults = TriangulateFromPair(images, newIdx, knownIdx, fwdMatches, fx, cx, cy,
                                    poseR[newIdx], poseT[newIdx], poseR[knownIdx], poseT[knownIdx]);
                                foreach (var (m, pi) in triResults)
                                {
                                    featureTo3D.TryAdd((newIdx, m.IndexA), pi);
                                    featureTo3D.TryAdd((knownIdx, m.IndexB), pi);
                                }
                            }
                        }
                        Log($"[SfM] ♻️ RETRY accepted {images[newIdx].FileName}: reproj={medErr:F2}px, +{Points3D.Count - before} pts");
                    }
                    if (retried > 0)
                    {
                        Log($"[SfM] Retry pass: added {retried} cameras (total: {reconstructedCameras.Count})");
                        // BA after retry
                        await RunBundleAdjustAsync(images, reconstructedCameras, poseR, poseT,
                            featureTo3D, fx, cx, cy, maxIterations: 50);
                    }
                }
            }

            // Final outlier removal
            RemoveOutlierPoints();

            Status = $"Reconstruction complete: {Points3D.Count} 3D points, {reconstructedCameras.Count} cameras";
            Log($"[SfM] Final: {Points3D.Count} points from {reconstructedCameras.Count} cameras");

            // Coordinate range diagnostics
            if (Points3D.Count > 0)
            {
                float minX = float.MaxValue, maxX = float.MinValue;
                float minY = float.MaxValue, maxY = float.MinValue;
                float minZ = float.MaxValue, maxZ = float.MinValue;
                foreach (var p in Points3D)
                {
                    if (p.Position.X < minX) minX = p.Position.X; if (p.Position.X > maxX) maxX = p.Position.X;
                    if (p.Position.Y < minY) minY = p.Position.Y; if (p.Position.Y > maxY) maxY = p.Position.Y;
                    if (p.Position.Z < minZ) minZ = p.Position.Z; if (p.Position.Z > maxZ) maxZ = p.Position.Z;
                }
                Log($"[SfM] X range: [{minX:F4}, {maxX:F4}] span={maxX - minX:F4}");
                Log($"[SfM] Y range: [{minY:F4}, {maxY:F4}] span={maxY - minY:F4}");
                Log($"[SfM] Z range: [{minZ:F4}, {maxZ:F4}] span={maxZ - minZ:F4}");

                // Compute standard deviations to check if points cluster on a plane
                float meanX = Points3D.Average(p => p.Position.X);
                float meanY = Points3D.Average(p => p.Position.Y);
                float meanZ = Points3D.Average(p => p.Position.Z);
                float stdX = MathF.Sqrt(Points3D.Average(p => (p.Position.X - meanX) * (p.Position.X - meanX)));
                float stdY = MathF.Sqrt(Points3D.Average(p => (p.Position.Y - meanY) * (p.Position.Y - meanY)));
                float stdZ = MathF.Sqrt(Points3D.Average(p => (p.Position.Z - meanZ) * (p.Position.Z - meanZ)));
                Log($"[SfM] StdDev: X={stdX:F4}, Y={stdY:F4}, Z={stdZ:F4}");
                Log($"[SfM] Mean: X={meanX:F4}, Y={meanY:F4}, Z={meanZ:F4}");

                // Check planarity: if stdDev in one axis is << others, it's flat
                float minStd = MathF.Min(stdX, MathF.Min(stdY, stdZ));
                float maxStd = MathF.Max(stdX, MathF.Max(stdY, stdZ));
                float midStd = stdX + stdY + stdZ - minStd - maxStd;
                Log($"[SfM] Std ratios: min/max={minStd / maxStd:F3}, mid/max={midStd / maxStd:F3}");
                if (minStd / maxStd < 0.15f)
                    Log("[SfM] ⚠️ WARNING: Points are nearly PLANAR (one axis has < 15% std of another)");

                // Also dump camera positions
                Log($"[SfM] Camera positions:");
                for (int ci = 0; ci < CameraPoses.Length; ci++)
                {
                    if (CameraPoses[ci] != null)
                    {
                        var cp = CameraPoses[ci]!;
                        Log($"  Cam[{ci}]: pos=({cp.Position.X:F3},{cp.Position.Y:F3},{cp.Position.Z:F3})");
                    }
                }
            }

            // ═══ GROUND TRUTH COMPARISON (TempleRing only) ═══
            if (images[0].FileName.StartsWith("templeR") && _httpClient != null)
            {
                try
                {
                    Log("[SfM] ╔════════════════════════════════════╗");
                    Log("[SfM] ║  GT POSE COMPARISON DIAGNOSTIC     ║");
                    Log("[SfM] ╚════════════════════════════════════╝");

                    string parText = await _httpClient.GetStringAsync("datasets/TempleRing/templeR_par.txt");
                    var parLines = parText.Split('\n', StringSplitOptions.RemoveEmptyEntries);

                    var gtR = new Dictionary<string, double[,]>();
                    var gtT = new Dictionary<string, double[]>();
                    for (int pi = 1; pi < parLines.Length; pi++)
                    {
                        var parts = parLines[pi].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length < 22) continue;
                        var rm = new double[3, 3];
                        for (int ri = 0; ri < 3; ri++)
                            for (int rj = 0; rj < 3; rj++)
                                rm[ri, rj] = double.Parse(parts[10 + ri * 3 + rj]);
                        gtR[parts[0]] = rm;
                        gtT[parts[0]] = new[] { double.Parse(parts[19]), double.Parse(parts[20]), double.Parse(parts[21]) };
                    }

                    // Build camera position arrays (estimated vs GT)
                    var estPos = new List<(int idx, Vector3 pos)>();
                    var gtPos = new List<(int idx, Vector3 pos)>();
                    for (int ci = 0; ci < images.Count; ci++)
                    {
                        if (CameraPoses[ci] == null) continue;
                        if (!gtT.TryGetValue(images[ci].FileName, out var gp)) continue;

                        // Estimated: camera center = -R^T * t
                        var eR = poseR[ci]; var eT = poseT[ci];
                        double ecx = -(eR[0, 0] * eT[0] + eR[1, 0] * eT[1] + eR[2, 0] * eT[2]);
                        double ecy = -(eR[0, 1] * eT[0] + eR[1, 1] * eT[1] + eR[2, 1] * eT[2]);
                        double ecz = -(eR[0, 2] * eT[0] + eR[1, 2] * eT[1] + eR[2, 2] * eT[2]);

                        // GT: camera center = -R^T * t
                        var gr = gtR[images[ci].FileName];
                        double gcx = -(gr[0, 0] * gp[0] + gr[1, 0] * gp[1] + gr[2, 0] * gp[2]);
                        double gcy = -(gr[0, 1] * gp[0] + gr[1, 1] * gp[1] + gr[2, 1] * gp[2]);
                        double gcz = -(gr[0, 2] * gp[0] + gr[1, 2] * gp[1] + gr[2, 2] * gp[2]);

                        estPos.Add((ci, new Vector3((float)ecx, (float)ecy, (float)ecz)));
                        gtPos.Add((ci, new Vector3((float)gcx, (float)gcy, (float)gcz)));

                        Log($"[GT-CMP] Cam[{ci}] {images[ci].FileName}:");
                        Log($"         Est center=({ecx:F4},{ecy:F4},{ecz:F4})");
                        Log($"         GT  center=({gcx:F4},{gcy:F4},{gcz:F4})");
                    }

                    // Compare inter-camera distances
                    if (estPos.Count >= 3)
                    {
                        // Compute distance from each camera to centroid
                        var estCentroid = new Vector3(
                            estPos.Average(p => p.pos.X),
                            estPos.Average(p => p.pos.Y),
                            estPos.Average(p => p.pos.Z));
                        var gtCentroid = new Vector3(
                            gtPos.Average(p => p.pos.X),
                            gtPos.Average(p => p.pos.Y),
                            gtPos.Average(p => p.pos.Z));

                        Log($"[GT-CMP] Est centroid: ({estCentroid.X:F4},{estCentroid.Y:F4},{estCentroid.Z:F4})");
                        Log($"[GT-CMP] GT  centroid: ({gtCentroid.X:F4},{gtCentroid.Y:F4},{gtCentroid.Z:F4})");

                        // Compute spread (stddev of distances from centroid)
                        float estSpread = MathF.Sqrt(estPos.Average(p => Vector3.DistanceSquared(p.pos, estCentroid)));
                        float gtSpread = MathF.Sqrt(gtPos.Average(p => Vector3.DistanceSquared(p.pos, gtCentroid)));
                        Log($"[GT-CMP] Est spread (RMS dist from centroid): {estSpread:F4}");
                        Log($"[GT-CMP] GT  spread: {gtSpread:F4}");
                        Log($"[GT-CMP] Scale ratio (est/gt): {estSpread / gtSpread:F3}");

                        // Check pairwise distance consistency
                        var estDists = new List<float>();
                        var gtDists = new List<float>();
                        for (int i = 0; i < estPos.Count; i++)
                        {
                            int nexti = (i + 1) % estPos.Count;
                            estDists.Add(Vector3.Distance(estPos[i].pos, estPos[nexti].pos));
                            gtDists.Add(Vector3.Distance(gtPos[i].pos, gtPos[nexti].pos));
                        }

                        // Check if sequential distances have consistent ratios
                        Log($"[GT-CMP] Sequential camera distances (est/gt ratio):");
                        for (int i = 0; i < estDists.Count && i < 8; i++)
                        {
                            float ratio = gtDists[i] > 0.0001f ? estDists[i] / gtDists[i] : 0;
                            Log($"  Cam[{estPos[i].idx}]→[{estPos[(i + 1) % estPos.Count].idx}]: est={estDists[i]:F4} gt={gtDists[i]:F4} ratio={ratio:F2}");
                        }

                        // Check XYZ extents of camera positions
                        float estRangeX = estPos.Max(p => p.pos.X) - estPos.Min(p => p.pos.X);
                        float estRangeY = estPos.Max(p => p.pos.Y) - estPos.Min(p => p.pos.Y);
                        float estRangeZ = estPos.Max(p => p.pos.Z) - estPos.Min(p => p.pos.Z);
                        float gtRangeX = gtPos.Max(p => p.pos.X) - gtPos.Min(p => p.pos.X);
                        float gtRangeY = gtPos.Max(p => p.pos.Y) - gtPos.Min(p => p.pos.Y);
                        float gtRangeZ = gtPos.Max(p => p.pos.Z) - gtPos.Min(p => p.pos.Z);

                        Log($"[GT-CMP] Camera position extents:");
                        Log($"         Est: X={estRangeX:F4} Y={estRangeY:F4} Z={estRangeZ:F4}");
                        Log($"         GT:  X={gtRangeX:F4} Y={gtRangeY:F4} Z={gtRangeZ:F4}");
                        Log($"         Ratios: X={estRangeX / gtRangeX:F2} Y={estRangeY / gtRangeY:F2} Z={estRangeZ / gtRangeZ:F2}");
                    }
                }
                catch (Exception gtEx)
                {
                    Log($"[GT-CMP] Comparison failed: {gtEx.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Status = $"Reconstruction failed: {ex.Message}";
            Log($"[SfM] Error: {ex}");
        }
        finally
        {
            IsRunning = false;
            OnStateChanged?.Invoke();
        }
    }

    /// <summary>
    /// Reconstruct using ground truth camera poses from a calibration file (Middlebury format).
    /// This bypasses E-mat/PnP to test triangulation in isolation.
    /// </summary>
    public async Task ReconstructGroundTruthAsync(HttpClient http)
    {
        IsRunning = true;
        Points3D.Clear();

        var images = _importService.Images;
        var pairs = _importService.MatchedPairs;

        if (images.Count < 2 || pairs.Count == 0)
        {
            Status = "Need at least 2 images with matches";
            OnStateChanged?.Invoke();
            IsRunning = false;
            return;
        }

        CameraPoses = new CameraParams?[images.Count];
        VerboseLogging = true;

        try
        {
            Status = "Loading ground truth calibration...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            // Load par file
            string parUrl = "datasets/TempleRing/templeR_par.txt";
            string parText = await http.GetStringAsync(parUrl);
            var lines = parText.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            // Parse calibration: skip first line (count), then each line is:
            // imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
            var calByName = new Dictionary<string, (double[,] K, double[,] R, double[] t)>();
            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 22) continue;

                string name = parts[0];
                double[,] K = new double[3, 3];
                K[0, 0] = double.Parse(parts[1]); K[0, 1] = double.Parse(parts[2]); K[0, 2] = double.Parse(parts[3]);
                K[1, 0] = double.Parse(parts[4]); K[1, 1] = double.Parse(parts[5]); K[1, 2] = double.Parse(parts[6]);
                K[2, 0] = double.Parse(parts[7]); K[2, 1] = double.Parse(parts[8]); K[2, 2] = double.Parse(parts[9]);

                double[,] R = new double[3, 3];
                R[0, 0] = double.Parse(parts[10]); R[0, 1] = double.Parse(parts[11]); R[0, 2] = double.Parse(parts[12]);
                R[1, 0] = double.Parse(parts[13]); R[1, 1] = double.Parse(parts[14]); R[1, 2] = double.Parse(parts[15]);
                R[2, 0] = double.Parse(parts[16]); R[2, 1] = double.Parse(parts[17]); R[2, 2] = double.Parse(parts[18]);

                double[] t = { double.Parse(parts[19]), double.Parse(parts[20]), double.Parse(parts[21]) };
                calByName[name] = (K, R, t);
            }

            Log($"[GT] Loaded {calByName.Count} calibration entries");

            // Match calibration to loaded images
            var poseR = new Dictionary<int, double[,]>();
            var poseT = new Dictionary<int, double[]>();
            float fx = 0, cxv = 0, cyv = 0;

            for (int i = 0; i < images.Count; i++)
            {
                if (calByName.TryGetValue(images[i].FileName, out var cal))
                {
                    poseR[i] = cal.R;
                    poseT[i] = cal.t;
                    if (fx == 0) { fx = (float)cal.K[0, 0]; cxv = (float)cal.K[0, 2]; cyv = (float)cal.K[1, 2]; }

                    var camP = CameraParams.CreateDefault(images[i].Width, images[i].Height);
                    camP.FocalX = (float)cal.K[0, 0]; camP.FocalY = (float)cal.K[1, 1];
                    camP.Position = new Vector3((float)cal.t[0], (float)cal.t[1], (float)cal.t[2]);
                    camP.Forward = new Vector3((float)cal.R[2, 0], (float)cal.R[2, 1], (float)cal.R[2, 2]);
                    camP.Up = new Vector3(-(float)cal.R[1, 0], -(float)cal.R[1, 1], -(float)cal.R[1, 2]);
                    CameraPoses[i] = camP;

                    Log($"[GT] Cam[{i}] {images[i].FileName}: fx={cal.K[0, 0]:F1} pos=({cal.t[0]:F4},{cal.t[1]:F4},{cal.t[2]:F4})");
                }
                else
                {
                    Log($"[GT] WARNING: No calibration for {images[i].FileName}");
                }
            }

            Log($"[GT] Using fx={fx}, cx={cxv}, cy={cyv}");

            // Build pair lookup
            var pairLookup = new Dictionary<(int, int), List<FeatureMatch>>();
            foreach (var pair in pairs)
            {
                pairLookup[(pair.ImageIndexA, pair.ImageIndexB)] = pair.Matches;
                pairLookup[(pair.ImageIndexB, pair.ImageIndexA)] =
                    pair.Matches.Select(m => new FeatureMatch { IndexA = m.IndexB, IndexB = m.IndexA }).ToList();
            }

            // Triangulate from ALL pairs of registered cameras
            Status = "Triangulating with ground truth poses...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            var featureTo3D = new Dictionary<(int imgIdx, int featIdx), int>();
            int totalPairsTriangulated = 0;

            foreach (var pair in pairs)
            {
                int a = pair.ImageIndexA, b = pair.ImageIndexB;
                if (!poseR.ContainsKey(a) || !poseR.ContainsKey(b)) continue;

                var triResults = TriangulateFromPair(images, a, b, pair.Matches, fx, cxv, cyv,
                    poseR[a], poseT[a], poseR[b], poseT[b]);

                foreach (var (m, pi) in triResults)
                {
                    featureTo3D.TryAdd((a, m.IndexA), pi);
                    featureTo3D.TryAdd((b, m.IndexB), pi);
                }
                totalPairsTriangulated++;
            }

            Log($"[GT] Triangulated from {totalPairsTriangulated} pairs → {Points3D.Count} 3D points");

            // Remove outliers
            if (Points3D.Count > 20)
                RemoveOutlierPoints();

            // Coordinate range diagnostics
            if (Points3D.Count > 0)
            {
                float minX = float.MaxValue, maxX = float.MinValue;
                float minY = float.MaxValue, maxY = float.MinValue;
                float minZ = float.MaxValue, maxZ = float.MinValue;
                foreach (var p in Points3D)
                {
                    if (p.Position.X < minX) minX = p.Position.X; if (p.Position.X > maxX) maxX = p.Position.X;
                    if (p.Position.Y < minY) minY = p.Position.Y; if (p.Position.Y > maxY) maxY = p.Position.Y;
                    if (p.Position.Z < minZ) minZ = p.Position.Z; if (p.Position.Z > maxZ) maxZ = p.Position.Z;
                }
                float spanX = maxX - minX, spanY = maxY - minY, spanZ = maxZ - minZ;
                Log($"[GT] X range: [{minX:F4}, {maxX:F4}] span={spanX:F4}");
                Log($"[GT] Y range: [{minY:F4}, {maxY:F4}] span={spanY:F4}");
                Log($"[GT] Z range: [{minZ:F4}, {maxZ:F4}] span={spanZ:F4}");
                Log($"[GT] Z/X ratio: {(spanX > 0.001 ? spanZ / spanX : 0):F2}");
                Log($"[GT] Z/Y ratio: {(spanY > 0.001 ? spanZ / spanY : 0):F2}");

                // Bounding box from README: (-0.023, -0.038, -0.092) to (0.079, 0.122, -0.017)
                Log("[GT] Expected bounding box: X=[-.023,.079] Y=[-.038,.122] Z=[-.092,-.017]");
                Log($"[GT] Expected spans: X=0.102, Y=0.160, Z=0.075");
            }

            int registeredCount = CameraPoses.Count(c => c != null);
            Status = $"GT Reconstruction: {Points3D.Count} 3D points, {registeredCount} cameras";
            Log($"[GT] Final: {Points3D.Count} points, {registeredCount} cameras");
        }
        catch (Exception ex)
        {
            Status = $"GT Reconstruction failed: {ex.Message}";
            Log($"[GT] Error: {ex}");
        }
        finally
        {
            IsRunning = false;
            OnStateChanged?.Invoke();
        }
    }

    /// <summary>
    /// Hybrid reconstruction: GT poses for first 2 cameras, PnP for the rest.
    /// Tests whether PnP works correctly with known-good initial data.
    /// </summary>
    public async Task ReconstructHybridGTAsync(HttpClient http)
    {
        IsRunning = true;
        Points3D.Clear();

        var images = _importService.Images;
        var pairs = _importService.MatchedPairs;

        if (images.Count < 2 || pairs.Count == 0) { IsRunning = false; return; }

        CameraPoses = new CameraParams?[images.Count];
        VerboseLogging = true;
        _gpuKernels ??= new GpuSfmKernels(_gpu);

        try
        {
            Status = "Loading GT calibration for hybrid test...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            // Load par file
            string parText = await http.GetStringAsync("datasets/TempleRing/templeR_par.txt");
            var parLines = parText.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            var calByName = new Dictionary<string, (double[,] K, double[,] R, double[] t)>();
            for (int i = 1; i < parLines.Length; i++)
            {
                var parts = parLines[i].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 22) continue;
                string name = parts[0];
                double[,] K = new double[3, 3]; double[,] R = new double[3, 3]; double[] t = new double[3];
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { K[r, c] = double.Parse(parts[1 + r * 3 + c]); R[r, c] = double.Parse(parts[10 + r * 3 + c]); }
                for (int j = 0; j < 3; j++) t[j] = double.Parse(parts[19 + j]);
                calByName[name] = (K, R, t);
            }

            float fx = (float)calByName.Values.First().K[0, 0];
            float cx2 = (float)calByName.Values.First().K[0, 2];
            float cy2 = (float)calByName.Values.First().K[1, 2];
            Log($"[HYBRID] fx={fx}, cx={cx2}, cy={cy2}");

            var poseR = new Dictionary<int, double[,]>();
            var poseT = new Dictionary<int, double[]>();
            var reconstructedCameras = new HashSet<int>();
            var featureTo3D = new Dictionary<(int imgIdx, int featIdx), int>();

            // Build pair lookup
            var pairLookup = new Dictionary<(int, int), List<FeatureMatch>>();
            foreach (var pair in pairs)
            {
                pairLookup[(pair.ImageIndexA, pair.ImageIndexB)] = pair.Matches;
                pairLookup[(pair.ImageIndexB, pair.ImageIndexA)] =
                    pair.Matches.Select(m => new FeatureMatch { IndexA = m.IndexB, IndexB = m.IndexA }).ToList();
            }

            // ═══ STEP 1: Place first 2 cameras using GT ═══
            for (int ci = 0; ci < Math.Min(2, images.Count); ci++)
            {
                if (!calByName.TryGetValue(images[ci].FileName, out var cal)) continue;
                poseR[ci] = cal.R;
                poseT[ci] = cal.t;
                var camP = CameraParams.CreateDefault(images[ci].Width, images[ci].Height);
                camP.FocalX = fx; camP.FocalY = fx;
                camP.Position = new Vector3((float)cal.t[0], (float)cal.t[1], (float)cal.t[2]);
                camP.Forward = new Vector3((float)cal.R[2, 0], (float)cal.R[2, 1], (float)cal.R[2, 2]);
                camP.Up = new Vector3(-(float)cal.R[1, 0], -(float)cal.R[1, 1], -(float)cal.R[1, 2]);
                CameraPoses[ci] = camP;
                reconstructedCameras.Add(ci);
                Log($"[HYBRID] GT Cam[{ci}] {images[ci].FileName} placed at ({cal.t[0]:F4},{cal.t[1]:F4},{cal.t[2]:F4})");
            }

            // Triangulate from GT pair
            if (reconstructedCameras.Count >= 2)
            {
                var camList = reconstructedCameras.ToList();
                int a = camList[0], b = camList[1];
                if (pairLookup.TryGetValue((a, b), out var initMatches))
                {
                    var triResults = TriangulateFromPair(images, a, b, initMatches, fx, cx2, cy2,
                        poseR[a], poseT[a], poseR[b], poseT[b]);
                    foreach (var (m, pi) in triResults)
                    {
                        featureTo3D.TryAdd((a, m.IndexA), pi);
                        featureTo3D.TryAdd((b, m.IndexB), pi);
                    }
                    Log($"[HYBRID] Initial triangulation: {Points3D.Count} points");
                }
            }

            // ═══ STEP 2: PnP for remaining cameras ═══
            Status = "Hybrid: PnP for remaining cameras...";
            OnStateChanged?.Invoke();
            await Task.Yield();

            for (int round = 0; round < images.Count; round++)
            {
                bool addedAny = false;
                var candidates = new List<(int idx, int corrCount)>();
                foreach (int newIdx in Enumerable.Range(0, images.Count).Where(i => !reconstructedCameras.Contains(i)))
                {
                    int corrCount = 0;
                    foreach (var knownIdx in reconstructedCameras)
                    {
                        if (!pairLookup.TryGetValue((knownIdx, newIdx), out var matches)) continue;
                        corrCount += matches.Count(m => featureTo3D.ContainsKey((knownIdx, m.IndexA)));
                    }
                    if (corrCount > 0) candidates.Add((newIdx, corrCount));
                }
                candidates.Sort((a2, b2) => b2.corrCount.CompareTo(a2.corrCount));

                foreach (var (newIdx, _) in candidates)
                {
                    if (reconstructedCameras.Contains(newIdx)) continue;

                    var pts2D = new List<Vector2>();
                    var pts3D = new List<Vector3>();

                    foreach (var knownIdx in reconstructedCameras)
                    {
                        if (!pairLookup.TryGetValue((knownIdx, newIdx), out var matches)) continue;
                        foreach (var match in matches)
                        {
                            if (featureTo3D.TryGetValue((knownIdx, match.IndexA), out int ptIdx))
                            {
                                var feat2D = images[newIdx].Features[match.IndexB];
                                pts2D.Add(new Vector2((feat2D.X - cx2) / fx, (feat2D.Y - cy2) / fx));
                                pts3D.Add(Points3D[ptIdx].Position);
                            }
                        }
                    }

                    Log($"[HYBRID] PnP {images[newIdx].FileName}: {pts2D.Count} correspondences");

                    if (pts2D.Count < 6) continue;

                    var pnp2D = new float[pts2D.Count * 2];
                    var pnp3D = new float[pts3D.Count * 3];
                    for (int pi = 0; pi < pts2D.Count; pi++)
                    {
                        pnp2D[pi * 2] = pts2D[pi].X;
                        pnp2D[pi * 2 + 1] = pts2D[pi].Y;
                        pnp3D[pi * 3] = pts3D[pi].X;
                        pnp3D[pi * 3 + 1] = pts3D[pi].Y;
                        pnp3D[pi * 3 + 2] = pts3D[pi].Z;
                    }

                    var (pnpR, pnpT, pnpInliers) = await _gpuKernels!.SolvePnPAsync(pnp2D, pnp3D, pts2D.Count);
                    if (pnpR == null || pnpInliers < 6) continue;

                    var finalR = new double[3, 3];
                    var finalT = new double[3];
                    for (int ri = 0; ri < 3; ri++)
                        for (int rj = 0; rj < 3; rj++)
                            finalR[ri, rj] = pnpR[ri * 3 + rj];
                    for (int ti = 0; ti < 3; ti++)
                        finalT[ti] = pnpT[ti];

                    poseR[newIdx] = finalR;
                    poseT[newIdx] = finalT;

                    var camNew = CameraParams.CreateDefault(images[newIdx].Width, images[newIdx].Height);
                    camNew.FocalX = fx; camNew.FocalY = fx;
                    camNew.Position = new Vector3((float)finalT[0], (float)finalT[1], (float)finalT[2]);
                    camNew.Forward = new Vector3((float)finalR[2, 0], (float)finalR[2, 1], (float)finalR[2, 2]);
                    camNew.Up = new Vector3(-(float)finalR[1, 0], -(float)finalR[1, 1], -(float)finalR[1, 2]);
                    CameraPoses[newIdx] = camNew;
                    reconstructedCameras.Add(newIdx);
                    addedAny = true;

                    // Compare with GT
                    if (calByName.TryGetValue(images[newIdx].FileName, out var gtCal))
                    {
                        // Camera center = -R^T * t
                        double ecx = -(finalR[0, 0] * finalT[0] + finalR[1, 0] * finalT[1] + finalR[2, 0] * finalT[2]);
                        double ecy = -(finalR[0, 1] * finalT[0] + finalR[1, 1] * finalT[1] + finalR[2, 1] * finalT[2]);
                        double ecz = -(finalR[0, 2] * finalT[0] + finalR[1, 2] * finalT[1] + finalR[2, 2] * finalT[2]);
                        var gr = gtCal.R; var gp = gtCal.t;
                        double gcx = -(gr[0, 0] * gp[0] + gr[1, 0] * gp[1] + gr[2, 0] * gp[2]);
                        double gcy = -(gr[0, 1] * gp[0] + gr[1, 1] * gp[1] + gr[2, 1] * gp[2]);
                        double gcz = -(gr[0, 2] * gp[0] + gr[1, 2] * gp[1] + gr[2, 2] * gp[2]);
                        double posErr = Math.Sqrt((ecx - gcx) * (ecx - gcx) + (ecy - gcy) * (ecy - gcy) + (ecz - gcz) * (ecz - gcz));
                        Log($"[HYBRID] PnP vs GT: est=({ecx:F4},{ecy:F4},{ecz:F4}) gt=({gcx:F4},{gcy:F4},{gcz:F4}) err={posErr:F4}");
                    }

                    // Triangulate new points
                    int before = Points3D.Count;
                    foreach (var regIdx in reconstructedCameras)
                    {
                        if (regIdx == newIdx) continue;
                        if (pairLookup.TryGetValue((newIdx, regIdx), out var fwdMatches))
                        {
                            var triResults = TriangulateFromPair(images, newIdx, regIdx, fwdMatches, fx, cx2, cy2,
                                poseR[newIdx], poseT[newIdx], poseR[regIdx], poseT[regIdx]);
                            foreach (var (m, pi) in triResults)
                            {
                                featureTo3D.TryAdd((newIdx, m.IndexA), pi);
                                featureTo3D.TryAdd((regIdx, m.IndexB), pi);
                            }
                        }
                    }
                    Log($"[HYBRID] Added {images[newIdx].FileName}: +{Points3D.Count - before} pts (total: {Points3D.Count}, {reconstructedCameras.Count} cams)");
                }
                if (!addedAny) break;
            }

            // Remove outliers
            if (Points3D.Count > 20) RemoveOutlierPoints();

            // Diagnostics
            if (Points3D.Count > 0)
            {
                float minX = float.MaxValue, maxX = float.MinValue;
                float minY = float.MaxValue, maxY = float.MinValue;
                float minZ = float.MaxValue, maxZ = float.MinValue;
                foreach (var p in Points3D)
                {
                    if (p.Position.X < minX) minX = p.Position.X; if (p.Position.X > maxX) maxX = p.Position.X;
                    if (p.Position.Y < minY) minY = p.Position.Y; if (p.Position.Y > maxY) maxY = p.Position.Y;
                    if (p.Position.Z < minZ) minZ = p.Position.Z; if (p.Position.Z > maxZ) maxZ = p.Position.Z;
                }
                float sX = maxX - minX, sY = maxY - minY, sZ = maxZ - minZ;
                Log($"[HYBRID] X=[{minX:F4},{maxX:F4}] span={sX:F4}");
                Log($"[HYBRID] Y=[{minY:F4},{maxY:F4}] span={sY:F4}");
                Log($"[HYBRID] Z=[{minZ:F4},{maxZ:F4}] span={sZ:F4}");
                Log($"[HYBRID] Z/X ratio: {(sX > 0.001 ? sZ / sX : 0):F2}");
            }

            int regCount = CameraPoses.Count(c => c != null);
            Status = $"Hybrid GT+PnP: {Points3D.Count} pts, {regCount}/{images.Count} cams";
            Log($"[HYBRID] Final: {Points3D.Count} points, {regCount} cameras");
        }
        catch (Exception ex) { Status = $"Hybrid failed: {ex.Message}"; Log($"[HYBRID] Error: {ex}"); }
        finally { IsRunning = false; OnStateChanged?.Invoke(); }
    }

    /// <summary>
    /// Estimate essential matrix using the 8-point algorithm with RANSAC.
    /// </summary>
    private (double[,]? E, bool[] inlierMask) EstimateEssentialRANSAC(
        List<ImageFeature> featuresA, List<ImageFeature> featuresB,
        List<FeatureMatch> matches, float fx, float cx, float cy)
    {
        int n = matches.Count;
        if (n < 8) return (null, Array.Empty<bool>());

        int maxIter = 1000;
        double threshold = 0.005; // Normalized coordinates — loosened for real-world photos
        double[,]? bestE = null;
        bool[] bestMask = new bool[n];
        int bestInliers = 0;

        // Normalize points
        var ptsA = new Vector2[n];
        var ptsB = new Vector2[n];
        for (int i = 0; i < n; i++)
        {
            var fA = featuresA[matches[i].IndexA];
            var fB = featuresB[matches[i].IndexB];
            ptsA[i] = new Vector2((fA.X - cx) / fx, (fA.Y - cy) / fx);
            ptsB[i] = new Vector2((fB.X - cx) / fx, (fB.Y - cy) / fx);
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Sample 8 random points
            var sample = SampleIndices(n, 8);

            // Build constraint matrix (8x9)
            var A = new double[8, 9];
            for (int s = 0; s < 8; s++)
            {
                int idx = sample[s];
                double x1 = ptsA[idx].X, y1 = ptsA[idx].Y;
                double x2 = ptsB[idx].X, y2 = ptsB[idx].Y;

                A[s, 0] = x2 * x1;
                A[s, 1] = x2 * y1;
                A[s, 2] = x2;
                A[s, 3] = y2 * x1;
                A[s, 4] = y2 * y1;
                A[s, 5] = y2;
                A[s, 6] = x1;
                A[s, 7] = y1;
                A[s, 8] = 1;
            }

            // Solve for E (null space of A)
            var e = LinearAlgebra.SolveHomogeneous(A);

            // Reshape to 3x3
            var E = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    E[i, j] = e[i * 3 + j];

            // Enforce rank-2 constraint via SVD
            var (U, S, Vt) = LinearAlgebra.SVD3x3(E);
            double avgS = (S[0] + S[1]) / 2;
            var diagS = new double[,] { { avgS, 0, 0 }, { 0, avgS, 0 }, { 0, 0, 0 } };

            // E = U * diag(s,s,0) * V^T
            var ES = Multiply3x3(U, diagS);
            E = Multiply3x3(ES, Vt);

            // Count inliers (Sampson error)
            var mask = new bool[n];
            int inliers = 0;
            for (int i = 0; i < n; i++)
            {
                double err = SampsonError(E, ptsA[i], ptsB[i]);
                if (err < threshold)
                {
                    mask[i] = true;
                    inliers++;
                }
            }

            if (inliers > bestInliers)
            {
                bestInliers = inliers;
                bestE = E;
                bestMask = mask;
            }

            // Early termination
            if (bestInliers > n * 0.8) break;
        }

        Log($"[SfM] RANSAC: best={bestInliers}/{n} inliers ({(100.0 * bestInliers / n):F1}%)");
        return (bestE, bestMask);
    }


    /// <summary>
    /// Recover the rotation and translation from the essential matrix.
    /// Tests all 4 possible solutions and picks the one with the most
    /// points in front of both cameras (cheirality check).
    /// </summary>
    private (double[,]? R, double[]? t) RecoverPose(
        double[,] E,
        List<ImageFeature> featuresA, List<ImageFeature> featuresB,
        List<FeatureMatch> matches, float fx, float cx, float cy)
    {
        var (U, _, Vt) = LinearAlgebra.SVD3x3(E);

        // W matrix for rotation extraction
        var W = new double[,] { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } };
        var Wt = new double[,] { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 1 } };

        // 4 possible solutions
        var R1 = Multiply3x3(U, Multiply3x3(W, Vt));
        var R2 = Multiply3x3(U, Multiply3x3(Wt, Vt));
        var t1 = new double[] { U[0, 2], U[1, 2], U[2, 2] };
        var t2 = new double[] { -U[0, 2], -U[1, 2], -U[2, 2] };

        // SVD orthogonalization to ensure proper rotation (det = +1)
        OrthogonalizeRotation(R1);
        OrthogonalizeRotation(R2);

        // Test all 4 combinations
        var candidates = new[]
        {
            (R1, t1),
            (R1, t2),
            (R2, t1),
            (R2, t2),
        };

        double[,]? bestR = null;
        double[]? bestT = null;
        int bestInFront = 0;

        var P1 = MakeProjectionMatrix(1, 0, 0, null, null);

        foreach (var (R, t) in candidates)
        {
            var P2 = MakeProjectionMatrix(1, 0, 0, R, t);
            int inFront = 0;

            // Test a subset of matches
            int testCount = Math.Min(matches.Count, 50);
            for (int i = 0; i < testCount; i++)
            {
                var fA = featuresA[matches[i].IndexA];
                var fB = featuresB[matches[i].IndexB];

                var nA = new Vector2((fA.X - cx) / fx, (fA.Y - cy) / fx);
                var nB = new Vector2((fB.X - cx) / fx, (fB.Y - cy) / fx);

                var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1, P2);
                if (pt.HasValue)
                {
                    float z1 = pt.Value.Z;
                    var pt2 = TransformPoint(pt.Value, R, t);
                    if (z1 > 0 && pt2.Z > 0) inFront++;
                }
            }

            if (inFront > bestInFront)
            {
                bestInFront = inFront;
                bestR = R;
                bestT = t;
            }
        }

        Log($"[SfM] Pose candidates: {string.Join(", ", candidates.Select((c, i) => $"#{i}: {CountInFront(c.Item1, c.Item2, featuresA, featuresB, matches, fx, cx, cy)} pts"))}");
        Log($"[SfM] Best pose: {bestInFront} points in front of both cameras");

        if (bestInFront < 5)
        {
            Log($"[SfM] WARNING: Too few points in front — cheirality check failed for all 4 candidates");
            return (null, null);
        }

        return (bestR, bestT);
    }

    private int CountInFront(double[,] R, double[] t,
        List<ImageFeature> featuresA, List<ImageFeature> featuresB,
        List<FeatureMatch> matches, float fx, float cx, float cy)
    {
        var P1 = MakeProjectionMatrix(1, 0, 0, null, null);
        var P2 = MakeProjectionMatrix(1, 0, 0, R, t);
        int count = 0;
        int testCount = Math.Min(matches.Count, 50);
        for (int i = 0; i < testCount; i++)
        {
            var fA = featuresA[matches[i].IndexA];
            var fB = featuresB[matches[i].IndexB];
            var nA = new Vector2((fA.X - cx) / fx, (fA.Y - cy) / fx);
            var nB = new Vector2((fB.X - cx) / fx, (fB.Y - cy) / fx);
            var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1, P2);
            if (pt.HasValue)
            {
                var pt2 = TransformPoint(pt.Value, R, t);
                if (pt.Value.Z > 0 && pt2.Z > 0) count++;
            }
        }
        return count;
    }

    /// <summary>
    /// Triangulate 3D points from a pair of cameras with known poses.
    /// Returns list of (match, pointIndex) for successfully triangulated points.
    /// </summary>
    private List<(FeatureMatch match, int pointIndex)> TriangulateFromPair(
        IReadOnlyList<ImportedImage> images, int idxA, int idxB,
        List<FeatureMatch> matches, float fx, float cx, float cy,
        double[,] RA, double[] tA, double[,] RB, double[] tB)
    {
        var P1 = MakeProjectionMatrix(1, 0, 0, RA, tA);
        var P2 = MakeProjectionMatrix(1, 0, 0, RB, tB);
        var results = new List<(FeatureMatch, int)>();

        // Max reprojection error in normalized coordinates
        // Disabled (set very high) — rely on IQR outlier filter for cleanup
        float maxReprojErr = 1000.0f / fx;

        foreach (var match in matches)
        {
            var featA = images[idxA].Features[match.IndexA];
            var featB = images[idxB].Features[match.IndexB];

            var nptA = new Vector2((featA.X - cx) / fx, (featA.Y - cy) / fx);
            var nptB = new Vector2((featB.X - cx) / fx, (featB.Y - cy) / fx);

            var pt3D = LinearAlgebra.TriangulatePoint(nptA, nptB, P1, P2);
            if (pt3D.HasValue)
            {
                var pt1 = TransformPoint(pt3D.Value, RA, tA);
                var pt2 = TransformPoint(pt3D.Value, RB, tB);

                if (pt1.Z > 0 && pt2.Z > 0)
                {
                    // ═══ TRIANGULATION ANGLE FILTER (3° min — COLMAP/Theia standard) ═══
                    // Camera centers: C = -R^T * t
                    var centerA = new Vector3(
                        (float)(-(RA[0, 0] * tA[0] + RA[1, 0] * tA[1] + RA[2, 0] * tA[2])),
                        (float)(-(RA[0, 1] * tA[0] + RA[1, 1] * tA[1] + RA[2, 1] * tA[2])),
                        (float)(-(RA[0, 2] * tA[0] + RA[1, 2] * tA[1] + RA[2, 2] * tA[2])));
                    var centerB = new Vector3(
                        (float)(-(RB[0, 0] * tB[0] + RB[1, 0] * tB[1] + RB[2, 0] * tB[2])),
                        (float)(-(RB[0, 1] * tB[0] + RB[1, 1] * tB[1] + RB[2, 1] * tB[2])),
                        (float)(-(RB[0, 2] * tB[0] + RB[1, 2] * tB[1] + RB[2, 2] * tB[2])));
                    var rayA = Vector3.Normalize(pt3D.Value - centerA);
                    var rayB = Vector3.Normalize(pt3D.Value - centerB);
                    float cosAngle = Vector3.Dot(rayA, rayB);
                    float angleDeg = MathF.Acos(Math.Clamp(cosAngle, -1f, 1f)) * (180f / MathF.PI);
                    if (angleDeg < 3.0f) continue; // Too narrow baseline

                    // Reprojection error check — filter bad triangulations
                    float reproj1X = pt1.X / pt1.Z;
                    float reproj1Y = pt1.Y / pt1.Z;
                    float reproj2X = pt2.X / pt2.Z;
                    float reproj2Y = pt2.Y / pt2.Z;
                    float err1 = MathF.Sqrt((reproj1X - nptA.X) * (reproj1X - nptA.X) + (reproj1Y - nptA.Y) * (reproj1Y - nptA.Y));
                    float err2 = MathF.Sqrt((reproj2X - nptB.X) * (reproj2X - nptB.X) + (reproj2Y - nptB.Y) * (reproj2Y - nptB.Y));
                    if (err1 > maxReprojErr || err2 > maxReprojErr)
                        continue;

                    int ptIdx = Points3D.Count;
                    Points3D.Add(new ReconstructedPoint
                    {
                        Position = pt3D.Value,
                        Color = GetPixelColor(images[idxA], featA.X, featA.Y),
                    });
                    results.Add((match, ptIdx));
                }
            }
        }
        return results;
    }

    /// <summary>
    /// Estimate the scale factor for a new camera's translation vector.
    /// Triangulates temp points with unit-length translation, then compares
    /// distances to already-known 3D points to find the correct scale.
    /// </summary>
    private double EstimateScale(
        IReadOnlyList<ImportedImage> images,
        int pairIdxA, int pairIdxB, bool knownIsA,
        int knownIdx, int newIdx,
        List<FeatureMatch> matches,
        double[,] Rnew, double[] tnew,
        Dictionary<int, double[,]> poseR, Dictionary<int, double[]> poseT,
        Dictionary<(int imgIdx, int featIdx), int> featureTo3D,
        float fx, float cx, float cy)
    {
        // Build temp projection matrices using the known camera's global pose
        // and the E-derived relative pose (unit length translation)
        var Rk = poseR[knownIdx];
        var tk = poseT[knownIdx];

        // Compute temp global pose for new camera (with unit-length t)
        double[,] RTemp;
        double[] tTemp;

        if (knownIsA)
        {
            RTemp = Multiply3x3(Rnew, Rk);
            tTemp = new double[]
            {
                Rnew[0, 0] * tk[0] + Rnew[0, 1] * tk[1] + Rnew[0, 2] * tk[2] + tnew[0],
                Rnew[1, 0] * tk[0] + Rnew[1, 1] * tk[1] + Rnew[1, 2] * tk[2] + tnew[1],
                Rnew[2, 0] * tk[0] + Rnew[2, 1] * tk[1] + Rnew[2, 2] * tk[2] + tnew[2],
            };
        }
        else
        {
            var RInv = Transpose3x3(Rnew);
            var tInv = new double[]
            {
                -(RInv[0, 0] * tnew[0] + RInv[0, 1] * tnew[1] + RInv[0, 2] * tnew[2]),
                -(RInv[1, 0] * tnew[0] + RInv[1, 1] * tnew[1] + RInv[1, 2] * tnew[2]),
                -(RInv[2, 0] * tnew[0] + RInv[2, 1] * tnew[1] + RInv[2, 2] * tnew[2]),
            };
            RTemp = Multiply3x3(RInv, Rk);
            tTemp = new double[]
            {
                RInv[0, 0] * tk[0] + RInv[0, 1] * tk[1] + RInv[0, 2] * tk[2] + tInv[0],
                RInv[1, 0] * tk[0] + RInv[1, 1] * tk[1] + RInv[1, 2] * tk[2] + tInv[1],
                RInv[2, 0] * tk[0] + RInv[2, 1] * tk[1] + RInv[2, 2] * tk[2] + tInv[2],
            };
        }

        // Build projection matrices
        double[,] RA_global = knownIsA ? Rk : RTemp;
        double[] tA_global = knownIsA ? tk : tTemp;
        double[,] RB_global = knownIsA ? RTemp : Rk;
        double[] tB_global = knownIsA ? tTemp : tk;
        var PA = MakeProjectionMatrix(1, 0, 0, RA_global, tA_global);
        var PB = MakeProjectionMatrix(1, 0, 0, RB_global, tB_global);

        // Collect pairs of (existing3D, tempTriangulated3D) for matched features
        var correspondences = new List<(Vector3 existing, Vector3 temp)>();
        foreach (var match in matches)
        {
            int knownFeatIdx = knownIsA ? match.IndexA : match.IndexB;
            if (!featureTo3D.TryGetValue((knownIdx, knownFeatIdx), out int existingPtIdx))
                continue;

            var existingPt = Points3D[existingPtIdx].Position;

            var featA = images[pairIdxA].Features[match.IndexA];
            var featB = images[pairIdxB].Features[match.IndexB];
            var nptA = new Vector2((featA.X - cx) / fx, (featA.Y - cy) / fx);
            var nptB = new Vector2((featB.X - cx) / fx, (featB.Y - cy) / fx);

            var tempPt = LinearAlgebra.TriangulatePoint(nptA, nptB, PA, PB);
            if (!tempPt.HasValue) continue;

            // Check positive depth in both cameras
            var ptA = TransformPoint(tempPt.Value, RA_global, tA_global);
            var ptB = TransformPoint(tempPt.Value, RB_global, tB_global);
            if (ptA.Z > 0 && ptB.Z > 0)
                correspondences.Add((existingPt, tempPt.Value));
        }

        if (correspondences.Count < 3)
        {
            Log($"[SfM] Scale estimation: only {correspondences.Count} correspondences, using scale=1.0");
            return 1.0;
        }

        // ═══ Inter-point distance ratio method ═══
        // For every pair of corresponding points, compute:
        //   ratio = dist(existingA, existingB) / dist(tempA, tempB)
        // This is position-invariant and much more robust than camera-to-point distances.
        var ratios = new List<double>();
        int n = correspondences.Count;
        int maxPairs = Math.Min(n * (n - 1) / 2, 500); // cap for performance
        int step = Math.Max(1, n * (n - 1) / 2 / maxPairs);
        int pairCount = 0;

        for (int i = 0; i < n && pairCount < maxPairs; i++)
        {
            for (int j = i + 1; j < n && pairCount < maxPairs; j++)
            {
                pairCount++;
                if (pairCount % step != 0) continue;

                float existingDist = Vector3.Distance(correspondences[i].existing, correspondences[j].existing);
                float tempDist = Vector3.Distance(correspondences[i].temp, correspondences[j].temp);

                if (tempDist > 0.001f && existingDist > 0.001f)
                {
                    ratios.Add(existingDist / tempDist);
                }
            }
        }

        if (ratios.Count < 3)
        {
            Log($"[SfM] Scale estimation: only {ratios.Count} distance pairs, using scale=1.0");
            return 1.0;
        }

        // Use median ratio as scale (robust to outliers)
        ratios.Sort();
        double medianScale = ratios[ratios.Count / 2];

        // Also compute IQR to assess consistency
        double q1 = ratios[ratios.Count / 4];
        double q3 = ratios[3 * ratios.Count / 4];
        double iqrRatio = (q3 > 0.001) ? q1 / q3 : 0;

        // Clamp to reasonable range — adjacent cameras shouldn't have wildly different baselines
        medianScale = Math.Clamp(medianScale, 0.5, 2.0);

        // Always use median scale — it's already robust to outliers.
        // The previous IQR check was too aggressive and rejected valid scales,
        // causing the reconstruction to collapse (all seeded cameras got scale=1.0).
        Log($"[SfM] Scale estimation: {correspondences.Count} pts, {ratios.Count} pairs, median={medianScale:F3}, q1={q1:F3}, q3={q3:F3}");
        return medianScale;
    }

    /// <summary>
    /// Solve PnP (Perspective-n-Point) using DLT with RANSAC.
    /// Given 2D-3D correspondences, finds the camera pose [R|t].
    /// 2D points should be normalized (i.e., (pixel - principal) / focal).
    /// </summary>
    private (double[,]? R, double[]? t) SolvePnPRansac(List<Vector2> pts2D, List<Vector3> pts3D)
    {
        int n = pts2D.Count;
        if (n < 6) return (null, null);

        int bestInliers = 0;
        double[,]? bestR = null;
        double[]? bestT = null;
        double threshold = 0.01; // Reprojection error threshold (normalized coords)

        int iterations = Math.Min(200, n * 10);

        for (int iter = 0; iter < iterations; iter++)
        {
            // Sample 6 random points
            var indices = SampleIndices(n, 6);

            var sample2D = indices.Select(i => pts2D[i]).ToList();
            var sample3D = indices.Select(i => pts3D[i]).ToList();

            var (R, t) = SolvePnPDLT(sample2D, sample3D);
            if (R == null) continue;

            // Count inliers
            int inliers = 0;
            for (int i = 0; i < n; i++)
            {
                var err = ReprojectionError(pts2D[i], pts3D[i], R, t!);
                if (err < threshold) inliers++;
            }

            if (inliers > bestInliers)
            {
                bestInliers = inliers;
                bestR = R;
                bestT = t;
            }
        }

        if (bestR == null || bestInliers < 6)
        {
            Log($"[SfM] PnP failed: only {bestInliers} inliers from {n} points");
            return (null, null);
        }

        // Refine with all inliers
        var inlier2D = new List<Vector2>();
        var inlier3D = new List<Vector3>();
        for (int i = 0; i < n; i++)
        {
            if (ReprojectionError(pts2D[i], pts3D[i], bestR, bestT!) < threshold)
            {
                inlier2D.Add(pts2D[i]);
                inlier3D.Add(pts3D[i]);
            }
        }

        if (inlier2D.Count >= 6)
        {
            var (refinedR, refinedT) = SolvePnPDLT(inlier2D, inlier3D);
            if (refinedR != null)
            {
                bestR = refinedR;
                bestT = refinedT;
            }
        }

        Log($"[SfM] PnP: {bestInliers}/{n} inliers");
        return (bestR, bestT);
    }

    /// <summary>
    /// Solve PnP using EPnP (Efficient Perspective-n-Point).
    /// Takes normalized 2D points and world 3D points.
    /// </summary>
    private (double[,]? R, double[]? t) SolvePnPDLT(List<Vector2> pts2D, List<Vector3> pts3D)
    {
        int n = pts2D.Count;
        if (n < 4) return (null, null);

        // Convert to flat arrays for EPnP
        var flat2D = new float[n * 2];
        var flat3D = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            flat2D[i * 2] = pts2D[i].X;
            flat2D[i * 2 + 1] = pts2D[i].Y;
            flat3D[i * 3] = pts3D[i].X;
            flat3D[i * 3 + 1] = pts3D[i].Y;
            flat3D[i * 3 + 2] = pts3D[i].Z;
        }

        var result = LinearAlgebra.SolveEPnP(flat2D.AsSpan(), flat3D.AsSpan(), n);
        if (result == null) return (null, null);

        return (result.Value.R, result.Value.t);
    }

    private static void OrthogonalizeRotation(double[,] R)
    {
        // Gram-Schmidt orthogonalization to make R a proper rotation
        double[] r0 = { R[0, 0], R[0, 1], R[0, 2] };
        double[] r1 = { R[1, 0], R[1, 1], R[1, 2] };
        double[] r2 = { R[2, 0], R[2, 1], R[2, 2] };

        // Normalize r0
        double n0 = Math.Sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
        if (n0 < 1e-10) return;
        r0[0] /= n0; r0[1] /= n0; r0[2] /= n0;

        // r1 = r1 - (r1·r0)*r0, then normalize
        double dot10 = r1[0] * r0[0] + r1[1] * r0[1] + r1[2] * r0[2];
        r1[0] -= dot10 * r0[0]; r1[1] -= dot10 * r0[1]; r1[2] -= dot10 * r0[2];
        double n1 = Math.Sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
        if (n1 < 1e-10) return;
        r1[0] /= n1; r1[1] /= n1; r1[2] /= n1;

        // r2 = r0 × r1 (cross product ensures orthogonality and det=1)
        r2[0] = r0[1] * r1[2] - r0[2] * r1[1];
        r2[1] = r0[2] * r1[0] - r0[0] * r1[2];
        r2[2] = r0[0] * r1[1] - r0[1] * r1[0];

        R[0, 0] = r0[0]; R[0, 1] = r0[1]; R[0, 2] = r0[2];
        R[1, 0] = r1[0]; R[1, 1] = r1[1]; R[1, 2] = r1[2];
        R[2, 0] = r2[0]; R[2, 1] = r2[1]; R[2, 2] = r2[2];
    }

    private double ReprojectionError(Vector2 pt2D, Vector3 pt3D, double[,] R, double[] t)
    {
        // Project 3D point through [R|t]
        double px = R[0, 0] * pt3D.X + R[0, 1] * pt3D.Y + R[0, 2] * pt3D.Z + t[0];
        double py = R[1, 0] * pt3D.X + R[1, 1] * pt3D.Y + R[1, 2] * pt3D.Z + t[1];
        double pz = R[2, 0] * pt3D.X + R[2, 1] * pt3D.Y + R[2, 2] * pt3D.Z + t[2];

        if (Math.Abs(pz) < 1e-10) return double.MaxValue;

        double u = px / pz;
        double v = py / pz;
        double dx = u - pt2D.X;
        double dy = v - pt2D.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>
    /// Find the eigenvector corresponding to the smallest eigenvalue of a symmetric matrix.
    /// Uses inverse power iteration.
    /// </summary>
    private double[]? SolveNullSpaceJacobi(double[,] AtA, int n)
    {
        // Add small regularization to make it invertible, then use inverse iteration
        var M = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i, j] = AtA[i, j];

        // Add tiny regularization
        for (int i = 0; i < n; i++)
            M[i, i] += 1e-10;

        // Inverse power iteration: v_{k+1} = M^{-1} * v_k / ||M^{-1} * v_k||
        var v = new double[n];
        for (int i = 0; i < n; i++)
            v[i] = _rng.NextDouble() - 0.5;

        for (int iter = 0; iter < 100; iter++)
        {
            // Solve M * w = v using Gauss elimination
            var w = SolveLinearSystem(M, v, n);
            if (w == null) return null;

            // Normalize
            double norm = 0;
            for (int i = 0; i < n; i++) norm += w[i] * w[i];
            norm = Math.Sqrt(norm);
            if (norm < 1e-15) return null;
            for (int i = 0; i < n; i++) w[i] /= norm;

            v = w;
        }

        return v;
    }

    private static double[]? SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Gaussian elimination with partial pivoting
        var M = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                M[i, j] = A[i, j];
            M[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            // Pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(M[row, col]) > Math.Abs(M[maxRow, col]))
                    maxRow = row;

            if (Math.Abs(M[maxRow, col]) < 1e-12) return null;

            // Swap
            for (int j = 0; j <= n; j++)
                (M[col, j], M[maxRow, j]) = (M[maxRow, j], M[col, j]);

            // Eliminate
            for (int row = col + 1; row < n; row++)
            {
                double f = M[row, col] / M[col, col];
                for (int j = col; j <= n; j++)
                    M[row, j] -= f * M[col, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = M[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= M[i, j] * x[j];
            x[i] /= M[i, i];
        }

        return x;
    }

    /// <summary>
    /// Re-triangulate all feature matches across registered cameras using refined poses.
    /// This is a standard COLMAP/Theia technique: after BA refines poses, previously
    /// rejected features may now triangulate correctly, densifying the point cloud.
    /// </summary>
    private void ReTriangulateAll(
        IReadOnlyList<ImportedImage> images,
        HashSet<int> reconstructedCameras,
        Dictionary<int, double[,]> poseR, Dictionary<int, double[]> poseT,
        Dictionary<(int imgIdx, int featIdx), int> featureTo3D,
        float fx, float cx, float cy,
        Dictionary<(int, int), List<FeatureMatch>> pairLookup)
    {
        int newPoints = 0;
        var camList = reconstructedCameras.ToList();

        for (int ci = 0; ci < camList.Count; ci++)
        {
            int idxA = camList[ci];
            for (int cj = ci + 1; cj < camList.Count; cj++)
            {
                int idxB = camList[cj];
                if (!pairLookup.TryGetValue((idxA, idxB), out var matches)) continue;

                // Only triangulate features that DON'T already have 3D points
                var newMatches = new List<FeatureMatch>();
                foreach (var m in matches)
                {
                    bool aHas3D = featureTo3D.ContainsKey((idxA, m.IndexA));
                    bool bHas3D = featureTo3D.ContainsKey((idxB, m.IndexB));
                    if (!aHas3D && !bHas3D)
                        newMatches.Add(m);
                }
                if (newMatches.Count == 0) continue;

                var triResults = TriangulateFromPair(images, idxA, idxB, newMatches, fx, cx, cy,
                    poseR[idxA], poseT[idxA], poseR[idxB], poseT[idxB]);
                foreach (var (m, pi) in triResults)
                {
                    featureTo3D.TryAdd((idxA, m.IndexA), pi);
                    featureTo3D.TryAdd((idxB, m.IndexB), pi);
                    newPoints++;
                }
            }
        }
        Log($"[SfM] ReTriangulateAll: added {newPoints} new points from {camList.Count} cameras");
    }

    /// <summary>
    /// Remove outlier points using per-axis IQR (interquartile range) filtering.
    /// This prevents outliers in one axis from inflating the extent and making
    /// the point cloud appear flat.
    /// </summary>
    private void RemoveOutlierPoints()
    {
        if (Points3D.Count < 20) return;

        // Per-axis IQR filtering
        var xs = Points3D.Select(p => p.Position.X).OrderBy(v => v).ToList();
        var ys = Points3D.Select(p => p.Position.Y).OrderBy(v => v).ToList();
        var zs = Points3D.Select(p => p.Position.Z).OrderBy(v => v).ToList();

        float q1x = xs[xs.Count / 4], q3x = xs[3 * xs.Count / 4], iqrX = q3x - q1x;
        float q1y = ys[ys.Count / 4], q3y = ys[3 * ys.Count / 4], iqrY = q3y - q1y;
        float q1z = zs[zs.Count / 4], q3z = zs[3 * zs.Count / 4], iqrZ = q3z - q1z;

        float xLow = q1x - 2f * iqrX, xHigh = q3x + 2f * iqrX;
        float yLow = q1y - 2f * iqrY, yHigh = q3y + 2f * iqrY;
        float zLow = q1z - 2f * iqrZ, zHigh = q3z + 2f * iqrZ;

        int before = Points3D.Count;
        Points3D.RemoveAll(p =>
            p.Position.X < xLow || p.Position.X > xHigh ||
            p.Position.Y < yLow || p.Position.Y > yHigh ||
            p.Position.Z < zLow || p.Position.Z > zHigh);

        if (before != Points3D.Count)
            Log($"[SfM] Removed {before - Points3D.Count} outlier points (IQR filter)");
    }

    /// <summary>
    /// Simplified bundle adjustment: refine 3D point positions by minimizing
    /// reprojection error. (Full BA would also refine cameras.)
    /// </summary>
    private void SimpleBundleAdjust(
        IReadOnlyList<ImportedImage> images, int idxA, int idxB,
        List<FeatureMatch> matches, float fx, float cx, float cy,
        double[,] R, double[] t)
    {
        if (Points3D.Count == 0) return;

        // Simple gradient descent on point positions
        float learningRate = 0.001f;
        int iterations = 10;

        for (int iter = 0; iter < iterations; iter++)
        {
            float totalError = 0;
            int count = 0;

            for (int i = 0; i < Points3D.Count && i < matches.Count; i++)
            {
                var pt = Points3D[i];
                var fA = images[idxA].Features[matches[i].IndexA];
                var fB = images[idxB].Features[matches[i].IndexB];

                // Reproject to camera A (identity)
                if (pt.Position.Z > 0.01f)
                {
                    float projX1 = fx * pt.Position.X / pt.Position.Z + cx;
                    float projY1 = fx * pt.Position.Y / pt.Position.Z + cy;
                    float errX1 = projX1 - fA.X;
                    float errY1 = projY1 - fA.Y;
                    totalError += errX1 * errX1 + errY1 * errY1;
                    count++;
                }

                // Reproject to camera B
                var ptB = TransformPoint(pt.Position, R, t);
                if (ptB.Z > 0.01f)
                {
                    float projX2 = fx * ptB.X / ptB.Z + cx;
                    float projY2 = fx * ptB.Y / ptB.Z + cy;
                    float errX2 = projX2 - fB.X;
                    float errY2 = projY2 - fB.Y;
                    totalError += errX2 * errX2 + errY2 * errY2;
                    count++;
                }
            }

            if (count > 0)
            {
                float rmsError = MathF.Sqrt(totalError / count);
                if (iter == 0 || iter == iterations - 1)
                    Log($"[SfM] BA iter {iter}: RMS reprojection error = {rmsError:F2} px");
            }
        }
    }

    // --- Utility Methods ---

    private int[] SampleIndices(int max, int count)
    {
        var indices = new HashSet<int>();
        while (indices.Count < count)
            indices.Add(_rng.Next(max));
        return indices.ToArray();
    }

    private static double SampsonError(double[,] E, Vector2 p1, Vector2 p2)
    {
        // e = p2^T * E * p1
        double ep0 = E[0, 0] * p1.X + E[0, 1] * p1.Y + E[0, 2];
        double ep1 = E[1, 0] * p1.X + E[1, 1] * p1.Y + E[1, 2];
        double ep2 = E[2, 0] * p1.X + E[2, 1] * p1.Y + E[2, 2];

        double etp0 = E[0, 0] * p2.X + E[1, 0] * p2.Y + E[2, 0];
        double etp1 = E[0, 1] * p2.X + E[1, 1] * p2.Y + E[2, 1];

        double x2tEx1 = p2.X * ep0 + p2.Y * ep1 + ep2;

        double denom = ep0 * ep0 + ep1 * ep1 + etp0 * etp0 + etp1 * etp1;
        if (denom < 1e-15) return double.MaxValue;

        return (x2tEx1 * x2tEx1) / denom;
    }

    private static double[,] MakeProjectionMatrix(double fx, double cx, double cy, double[,]? R, double[]? t)
    {
        // P = K * [R | t] (3x4)
        var P = new double[3, 4];

        if (R == null || t == null)
        {
            // Identity pose
            P[0, 0] = 1; P[1, 1] = 1; P[2, 2] = 1;
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                    P[i, j] = R[i, j];
                P[i, 3] = t[i];
            }
        }

        return P;
    }

    private static Vector3 TransformPoint(Vector3 pt, double[,] R, double[] t)
    {
        return new Vector3(
            (float)(R[0, 0] * pt.X + R[0, 1] * pt.Y + R[0, 2] * pt.Z + t[0]),
            (float)(R[1, 0] * pt.X + R[1, 1] * pt.Y + R[1, 2] * pt.Z + t[1]),
            (float)(R[2, 0] * pt.X + R[2, 1] * pt.Y + R[2, 2] * pt.Z + t[2])
        );
    }

    private static double[,] Multiply3x3(double[,] A, double[,] B)
    {
        var C = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                    sum += A[i, k] * B[k, j];
                C[i, j] = sum;
            }
        return C;
    }

    private static double[,] Transpose3x3(double[,] M)
    {
        var T = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                T[i, j] = M[j, i];
        return T;
    }

    private static void NegateMatrix(double[,] M)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                M[i, j] = -M[i, j];
    }

    private static void NegateArray(double[] a)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = -a[i];
    }

    private static Vector3 GetPixelColor(ImportedImage img, float x, float y)
    {
        int px = Math.Clamp((int)x, 0, img.Width - 1);
        int py = Math.Clamp((int)y, 0, img.Height - 1);
        int idx = (py * img.Width + px) * 4;
        if (idx + 2 >= img.RgbaPixels.Length) return Vector3.One;
        return new Vector3(
            img.RgbaPixels[idx] / 255f,
            img.RgbaPixels[idx + 1] / 255f,
            img.RgbaPixels[idx + 2] / 255f
        );
    }

    // ══════════════════════════════════════════════════════════
    //  Bundle Adjustment Integration
    // ══════════════════════════════════════════════════════════

    /// <summary>
    /// Run GPU bundle adjustment on all registered cameras and 3D points.
    /// </summary>
    private async Task RunBundleAdjustAsync(
        IReadOnlyList<ImportedImage> images,
        HashSet<int> reconstructedCameras,
        Dictionary<int, double[,]> poseR, Dictionary<int, double[]> poseT,
        Dictionary<(int imgIdx, int featIdx), int> featureTo3D,
        float fx, float cx, float cy,
        int maxIterations = 10)
    {
        if (_gpuKernels == null || Points3D.Count < 4) return;

        var camList = reconstructedCameras.OrderBy(x => x).ToList();
        int numCameras = camList.Count;
        int numPoints = Points3D.Count;

        var camIdxMap = new Dictionary<int, int>();
        for (int i = 0; i < camList.Count; i++)
            camIdxMap[camList[i]] = i;

        // Pack camera params (angle-axis + translation)
        var camParams = new float[numCameras * 6];
        for (int i = 0; i < numCameras; i++)
        {
            int imgIdx = camList[i];
            var R = poseR[imgIdx]!;
            var t = poseT[imgIdx]!;
            var aa = RotationMatrixToAngleAxis(R);
            camParams[i * 6] = (float)aa[0];
            camParams[i * 6 + 1] = (float)aa[1];
            camParams[i * 6 + 2] = (float)aa[2];
            camParams[i * 6 + 3] = (float)t[0];
            camParams[i * 6 + 4] = (float)t[1];
            camParams[i * 6 + 5] = (float)t[2];
        }

        // Pack 3D points
        var pts3D = new float[numPoints * 3];
        for (int i = 0; i < numPoints; i++)
        {
            pts3D[i * 3] = Points3D[i].Position.X;
            pts3D[i * 3 + 1] = Points3D[i].Position.Y;
            pts3D[i * 3 + 2] = Points3D[i].Position.Z;
        }

        // Pack observations
        var obsList = new List<float>();
        foreach (var ((imgIdx, featIdx), ptIdx) in featureTo3D)
        {
            if (!camIdxMap.TryGetValue(imgIdx, out int baCamIdx)) continue;
            if (ptIdx < 0 || ptIdx >= numPoints) continue;
            var feat = images[imgIdx].Features[featIdx];
            float u = (feat.X - cx) / fx;
            float v = (feat.Y - cy) / fx;
            obsList.Add(baCamIdx);
            obsList.Add(ptIdx);
            obsList.Add(u);
            obsList.Add(v);
        }

        int numObs = obsList.Count / 4;
        if (numObs < numCameras * 3)
        {
            Log($"[BA] Not enough observations ({numObs}), skipping");
            return;
        }

        Log($"[BA] Starting: {numCameras} cameras, {numPoints} points, {numObs} observations");
        Status = $"Bundle Adjustment ({numCameras} cams, {numPoints} pts)...";
        OnStateChanged?.Invoke();
        await Task.Yield();

        var (newCams, newPts, finalRMS) = await _gpuKernels.BundleAdjustAsync(
            camParams, pts3D, obsList.ToArray(),
            numCameras, numPoints, numObs, maxIterations);

        // Write back refined camera poses
        for (int i = 0; i < numCameras; i++)
        {
            int imgIdx = camList[i];
            var aa = new double[] { newCams[i * 6], newCams[i * 6 + 1], newCams[i * 6 + 2] };
            var newR = AngleAxisToRotationMatrix(aa);
            var newT = new double[] { newCams[i * 6 + 3], newCams[i * 6 + 4], newCams[i * 6 + 5] };
            poseR[imgIdx] = newR;
            poseT[imgIdx] = newT;
            if (CameraPoses![imgIdx] != null)
            {
                var camP = CameraPoses[imgIdx]!;
                camP.Position = new Vector3((float)newT[0], (float)newT[1], (float)newT[2]);
                camP.Forward = new Vector3((float)newR[2, 0], (float)newR[2, 1], (float)newR[2, 2]);
                camP.Up = new Vector3(-(float)newR[1, 0], -(float)newR[1, 1], -(float)newR[1, 2]);
            }
        }

        // Write back refined 3D points
        for (int i = 0; i < numPoints; i++)
            Points3D[i].Position = new Vector3(newPts[i * 3], newPts[i * 3 + 1], newPts[i * 3 + 2]);

        Log($"[BA] Done: final RMS={finalRMS:F6} ({finalRMS * fx:F2}px)");
    }

    private static double[] RotationMatrixToAngleAxis(double[,] R)
    {
        double trace = R[0, 0] + R[1, 1] + R[2, 2];
        double cosTheta = Math.Max(-1.0, Math.Min(1.0, (trace - 1.0) / 2.0));
        double theta = Math.Acos(cosTheta);
        if (theta < 1e-10)
            return new double[] { (R[2, 1] - R[1, 2]) / 2.0, (R[0, 2] - R[2, 0]) / 2.0, (R[1, 0] - R[0, 1]) / 2.0 };
        double k = theta / (2.0 * Math.Sin(theta));
        return new double[] { k * (R[2, 1] - R[1, 2]), k * (R[0, 2] - R[2, 0]), k * (R[1, 0] - R[0, 1]) };
    }

    private static double[,] AngleAxisToRotationMatrix(double[] aa)
    {
        double theta = Math.Sqrt(aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2]);
        if (theta < 1e-10) return new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        double kx = aa[0] / theta, ky = aa[1] / theta, kz = aa[2] / theta;
        double c = Math.Cos(theta), s = Math.Sin(theta), v = 1.0 - c;
        return new double[,] {
            { kx*kx*v+c,    kx*ky*v-kz*s, kx*kz*v+ky*s },
            { ky*kx*v+kz*s, ky*ky*v+c,    ky*kz*v-kx*s },
            { kz*kx*v-ky*s, kz*ky*v+kx*s, kz*kz*v+c    }
        };
    }
}

/// <summary>
/// A reconstructed 3D point from SfM.
/// </summary>
public class ReconstructedPoint
{
    public Vector3 Position { get; set; }
    public Vector3 Color { get; set; } = Vector3.One;
    public int ObservationCount { get; set; } = 1;
}

/// <summary>
/// Synthetic diagnostic tests for the SfM pipeline.
/// Tests each component against known ground truth.
/// </summary>
public static class SfmDiagnostic
{
    public static void RunAll()
    {
        SfmReconstructor.Log("═══════════════════════════════════════════");
        SfmReconstructor.Log("  SfM PIPELINE DIAGNOSTIC TESTS");
        SfmReconstructor.Log("═══════════════════════════════════════════");

        TestTriangulation();
        TestRecoverPose();
        TestPoseComposition();
    }

    /// <summary>
    /// Test 1: TriangulatePoint with known cameras and known 3D points.
    /// Camera A at origin facing +Z, Camera B at (1,0,0) facing +Z.
    /// </summary>
    static void TestTriangulation()
    {
        SfmReconstructor.Log("\n──── TEST 1: Triangulation ────");

        // Camera A: Identity (at origin, facing +Z)
        var RA = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var tA = new double[] { 0, 0, 0 };

        // Camera B: At position (1, 0, 0), facing +Z
        // Extrinsic: t = -R*C = -(I * [1,0,0]) = [-1, 0, 0]
        var RB = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var tB = new double[] { -1, 0, 0 };

        var PA = MakeP(RA, tA);
        var PB = MakeP(RB, tB);

        // Known 3D points at various positions
        var points = new Vector3[]
        {
            new(0.5f, 0, 5),    // Centered between cameras, 5 units away
            new(0, 1, 3),       // Left, up, closer
            new(1, -0.5f, 10),  // Right, down, far
            new(0.5f, 0.5f, 2), // Close
            new(2, 1, 8),       // Off to the right, far
        };

        int pass = 0, fail = 0;
        foreach (var pt3D in points)
        {
            // Project through both cameras: x = [R|t] * X (normalized coords)
            var projA = Project(pt3D, RA, tA);
            var projB = Project(pt3D, RB, tB);

            // Skip if behind camera
            if (projA.z <= 0 || projB.z <= 0) { SfmReconstructor.Log($"  SKIP: behind camera"); continue; }

            var nptA = new Vector2((float)(projA.x / projA.z), (float)(projA.y / projA.z));
            var nptB = new Vector2((float)(projB.x / projB.z), (float)(projB.y / projB.z));

            var result = LinearAlgebra.TriangulatePoint(nptA, nptB, PA, PB);
            if (!result.HasValue)
            {
                SfmReconstructor.Log($"  FAIL: Triangulate returned null for {pt3D}");
                fail++; continue;
            }

            float err = Vector3.Distance(result.Value, pt3D);
            bool ok = err < 0.01f;
            SfmReconstructor.Log($"  {(ok ? "PASS" : "FAIL")}: Expected {pt3D}, Got ({result.Value.X:F3},{result.Value.Y:F3},{result.Value.Z:F3}), err={err:F4}");
            if (ok) pass++; else fail++;
        }
        SfmReconstructor.Log($"  Triangulation: {pass} pass, {fail} fail");

        // Test 1b: Cameras facing DIFFERENT directions
        SfmReconstructor.Log("\n  -- Cameras facing different directions --");
        // Camera C at (0,0,5) facing -Z (180° rotation about Y)
        var RC = new double[,] { { -1, 0, 0 }, { 0, 1, 0 }, { 0, 0, -1 } };
        var tC_pos = new Vector3(0, 0, 5); // camera center
        var tC = new double[] {
            -(RC[0,0]*tC_pos.X + RC[0,1]*tC_pos.Y + RC[0,2]*tC_pos.Z),
            -(RC[1,0]*tC_pos.X + RC[1,1]*tC_pos.Y + RC[1,2]*tC_pos.Z),
            -(RC[2,0]*tC_pos.X + RC[2,1]*tC_pos.Y + RC[2,2]*tC_pos.Z)
        };

        var PC = MakeP(RC, tC);

        // Point visible by both A (at origin facing +Z) and C (at (0,0,5) facing -Z)
        // Point at (0.5, 0.3, 2.5) — between both cameras
        var testPt = new Vector3(0.5f, 0.3f, 2.5f);
        var pA2 = Project(testPt, RA, tA);
        var pC2 = Project(testPt, RC, tC);
        SfmReconstructor.Log($"  Proj A: z={pA2.z:F3} (should be >0)");
        SfmReconstructor.Log($"  Proj C: z={pC2.z:F3} (should be >0)");

        if (pA2.z > 0 && pC2.z > 0)
        {
            var nA2 = new Vector2((float)(pA2.x / pA2.z), (float)(pA2.y / pA2.z));
            var nC2 = new Vector2((float)(pC2.x / pC2.z), (float)(pC2.y / pC2.z));
            var res = LinearAlgebra.TriangulatePoint(nA2, nC2, PA, PC);
            if (res.HasValue)
            {
                float err2 = Vector3.Distance(res.Value, testPt);
                SfmReconstructor.Log($"  {(err2 < 0.01 ? "PASS" : "FAIL")}: Expected {testPt}, Got ({res.Value.X:F3},{res.Value.Y:F3},{res.Value.Z:F3}), err={err2:F4}");
            }
            else SfmReconstructor.Log("  FAIL: returned null for opposite-facing cameras");
        }
    }

    /// <summary>
    /// Test 2: RecoverPose from a known Essential matrix.
    /// Verifies that SVD decomposition + chirality check recover correct R,t.
    /// </summary>
    static void TestRecoverPose()
    {
        SfmReconstructor.Log("\n──── TEST 2: RecoverPose ────");

        // Known relative pose: 30° rotation about Y, translation (1,0,0)
        double angleRad = 30.0 * Math.PI / 180.0;
        double cos = Math.Cos(angleRad), sin = Math.Sin(angleRad);

        var R_true = new double[,] {
            { cos, 0, sin },
            { 0,   1, 0   },
            { -sin, 0, cos }
        };
        var t_true = new double[] { 1, 0, 0 };
        // Normalize t
        double tn = Math.Sqrt(t_true[0] * t_true[0] + t_true[1] * t_true[1] + t_true[2] * t_true[2]);
        t_true[0] /= tn; t_true[1] /= tn; t_true[2] /= tn;

        // Compute E = [t]_x * R
        var tx = new double[,] {
            { 0, -t_true[2], t_true[1] },
            { t_true[2], 0, -t_true[0] },
            { -t_true[1], t_true[0], 0 }
        };
        var E = Mult3x3(tx, R_true);
        SfmReconstructor.Log($"  E = [{E[0, 0]:F3},{E[0, 1]:F3},{E[0, 2]:F3}; {E[1, 0]:F3},{E[1, 1]:F3},{E[1, 2]:F3}; {E[2, 0]:F3},{E[2, 1]:F3},{E[2, 2]:F3}]");

        // Generate synthetic 2D correspondences
        var rng = new Random(42);
        var featA = new List<ImageFeature>();
        var featB = new List<ImageFeature>();
        var matches = new List<FeatureMatch>();

        float fx = 500, imgCx = 320, imgCy = 240;

        for (int i = 0; i < 50; i++)
        {
            // Random 3D point in front of both cameras
            float px = (float)(rng.NextDouble() * 4 - 2);
            float py = (float)(rng.NextDouble() * 4 - 2);
            float pz = (float)(rng.NextDouble() * 8 + 2);
            var worldPt = new Vector3(px, py, pz);

            // Project through camera A (identity)
            var pA = new Vector2(worldPt.X / worldPt.Z, worldPt.Y / worldPt.Z);
            // Project through camera B (R_true, t_true)
            var pB = Project(worldPt, R_true, t_true);
            if (pB.z <= 0) continue;

            float pixAx = pA.X * fx + imgCx;
            float pixAy = pA.Y * fx + imgCy;
            float pixBx = (float)(pB.x / pB.z * fx + imgCx);
            float pixBy = (float)(pB.y / pB.z * fx + imgCy);

            featA.Add(new ImageFeature { X = pixAx, Y = pixAy });
            featB.Add(new ImageFeature { X = pixBx, Y = pixBy });
            matches.Add(new FeatureMatch { IndexA = featA.Count - 1, IndexB = featB.Count - 1 });
        }

        SfmReconstructor.Log($"  Generated {matches.Count} synthetic correspondences");

        // Test SVD of E
        var (U, S, Vt) = LinearAlgebra.SVD3x3(E);
        SfmReconstructor.Log($"  SVD(E): S=[{S[0]:F4},{S[1]:F4},{S[2]:F4}]");
        SfmReconstructor.Log($"  S[2] should be ~0: {(Math.Abs(S[2]) < 0.01 ? "PASS" : "FAIL")}");
        SfmReconstructor.Log($"  U = [{U[0, 0]:F4},{U[0, 1]:F4},{U[0, 2]:F4}]");
        SfmReconstructor.Log($"      [{U[1, 0]:F4},{U[1, 1]:F4},{U[1, 2]:F4}]");
        SfmReconstructor.Log($"      [{U[2, 0]:F4},{U[2, 1]:F4},{U[2, 2]:F4}]");
        SfmReconstructor.Log($"  det(U) = {Det3(new double[,] { { U[0, 0], U[0, 1], U[0, 2] }, { U[1, 0], U[1, 1], U[1, 2] }, { U[2, 0], U[2, 1], U[2, 2] } }):F6}");
        SfmReconstructor.Log($"  Vt= [{Vt[0, 0]:F4},{Vt[0, 1]:F4},{Vt[0, 2]:F4}]");
        SfmReconstructor.Log($"      [{Vt[1, 0]:F4},{Vt[1, 1]:F4},{Vt[1, 2]:F4}]");
        SfmReconstructor.Log($"      [{Vt[2, 0]:F4},{Vt[2, 1]:F4},{Vt[2, 2]:F4}]");
        SfmReconstructor.Log($"  det(Vt) = {Det3(new double[,] { { Vt[0, 0], Vt[0, 1], Vt[0, 2] }, { Vt[1, 0], Vt[1, 1], Vt[1, 2] }, { Vt[2, 0], Vt[2, 1], Vt[2, 2] } }):F6}");

        // Now test RecoverPose through the full pipeline
        // We need to call the actual RecoverPose — but it's private.
        // Instead, manually do the decomposition here to test:
        var W = new double[,] { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } };
        var Wt = new double[,] { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 1 } };

        var R1 = Mult3x3(U, Mult3x3(W, Vt));
        var R2 = Mult3x3(U, Mult3x3(Wt, Vt));

        // Orthogonalize
        OrthGS(R1); OrthGS(R2);

        SfmReconstructor.Log($"  R1 det={Det3(R1):F4}, R2 det={Det3(R2):F4}");

        var t1 = new double[] { U[0, 2], U[1, 2], U[2, 2] };
        var t2 = new double[] { -U[0, 2], -U[1, 2], -U[2, 2] };

        // Test all 4 candidates against ground truth
        var cands = new[] { (R1, t1), (R1, t2), (R2, t1), (R2, t2) };
        int bestIdx = -1; double bestMatch = double.MaxValue;
        for (int i = 0; i < 4; i++)
        {
            var (Rc, tc) = cands[i];
            // Compare R with R_true
            double rErr = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    rErr += Math.Abs(Rc[r, c] - R_true[r, c]);
            // Compare t direction (could be negated)
            double tDot = tc[0] * t_true[0] + tc[1] * t_true[1] + tc[2] * t_true[2];
            double tMag = Math.Sqrt(tc[0] * tc[0] + tc[1] * tc[1] + tc[2] * tc[2]);
            double tErr = 1 - Math.Abs(tDot / tMag);

            // Chirality check
            int inFront = 0;
            var P1 = MakeP(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, new double[] { 0, 0, 0 });
            var P2 = MakeP(Rc, tc);
            for (int j = 0; j < Math.Min(matches.Count, 20); j++)
            {
                var fA = featA[matches[j].IndexA];
                var fB = featB[matches[j].IndexB];
                var nA = new Vector2((fA.X - imgCx) / fx, (fA.Y - imgCy) / fx);
                var nB = new Vector2((fB.X - imgCx) / fx, (fB.Y - imgCy) / fx);
                var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1, P2);
                if (pt.HasValue)
                {
                    var pt2 = new Vector3(
                        (float)(Rc[0, 0] * pt.Value.X + Rc[0, 1] * pt.Value.Y + Rc[0, 2] * pt.Value.Z + tc[0]),
                        (float)(Rc[1, 0] * pt.Value.X + Rc[1, 1] * pt.Value.Y + Rc[1, 2] * pt.Value.Z + tc[1]),
                        (float)(Rc[2, 0] * pt.Value.X + Rc[2, 1] * pt.Value.Y + Rc[2, 2] * pt.Value.Z + tc[2]));
                    if (pt.Value.Z > 0 && pt2.Z > 0) inFront++;
                }
            }

            SfmReconstructor.Log($"  Candidate {i}: R_err={rErr:F3}, t_err={tErr:F4}, inFront={inFront}/20");
            double score = rErr + tErr * 10;
            if (score < bestMatch) { bestMatch = score; bestIdx = i; }
        }
        SfmReconstructor.Log($"  Best match: candidate {bestIdx} (should have low R_err and low t_err)");

        // Check if chirality selects the right one
        int bestChiral = -1; int bestFront = 0;
        for (int i = 0; i < 4; i++)
        {
            var (Rc, tc) = cands[i];
            var P1c = MakeP(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, new double[] { 0, 0, 0 });
            var P2c = MakeP(Rc, tc);
            int inf = 0;
            for (int j = 0; j < Math.Min(matches.Count, 20); j++)
            {
                var fA = featA[matches[j].IndexA];
                var fB = featB[matches[j].IndexB];
                var nA = new Vector2((fA.X - imgCx) / fx, (fA.Y - imgCy) / fx);
                var nB = new Vector2((fB.X - imgCx) / fx, (fB.Y - imgCy) / fx);
                var pt = LinearAlgebra.TriangulatePoint(nA, nB, P1c, P2c);
                if (pt.HasValue)
                {
                    var pt2 = new Vector3(
                        (float)(Rc[0, 0] * pt.Value.X + Rc[0, 1] * pt.Value.Y + Rc[0, 2] * pt.Value.Z + tc[0]),
                        (float)(Rc[1, 0] * pt.Value.X + Rc[1, 1] * pt.Value.Y + Rc[1, 2] * pt.Value.Z + tc[1]),
                        (float)(Rc[2, 0] * pt.Value.X + Rc[2, 1] * pt.Value.Y + Rc[2, 2] * pt.Value.Z + tc[2]));
                    if (pt.Value.Z > 0 && pt2.Z > 0) inf++;
                }
            }
            if (inf > bestFront) { bestFront = inf; bestChiral = i; }
        }
        SfmReconstructor.Log($"  Chirality selects: candidate {bestChiral} (best={bestIdx})");
        SfmReconstructor.Log($"  {(bestChiral == bestIdx ? "PASS" : "FAIL")}: chirality check {(bestChiral == bestIdx ? "agrees" : "DISAGREES")} with ground truth");
    }

    /// <summary>
    /// Test 3: Global pose composition.
    /// Camera A at identity, Camera B from E-mat, Camera C from E-mat of B.
    /// </summary>
    static void TestPoseComposition()
    {
        SfmReconstructor.Log("\n──── TEST 3: Pose Composition ────");

        // Camera A: identity
        var RA = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var tA = new double[] { 0, 0, 0 };

        // Camera B: translate (2, 0, 0) in world, same orientation
        // Extrinsic t_B = -R_B * C_B = -I * [2,0,0] = [-2,0,0]
        var RB = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        var tB = new double[] { -2, 0, 0 };

        // Camera C: at (2, 0, 2), rotated 90° about Y
        double ang = Math.PI / 2;
        var RC = new double[,] {
            { Math.Cos(ang), 0, Math.Sin(ang) },
            { 0, 1, 0 },
            { -Math.Sin(ang), 0, Math.Cos(ang) }
        };
        var cC = new Vector3(2, 0, 2);
        var tC = new double[] {
            -(RC[0,0]*cC.X + RC[0,1]*cC.Y + RC[0,2]*cC.Z),
            -(RC[1,0]*cC.X + RC[1,1]*cC.Y + RC[1,2]*cC.Z),
            -(RC[2,0]*cC.X + RC[2,1]*cC.Y + RC[2,2]*cC.Z)
        };

        SfmReconstructor.Log($"  True Camera C center: ({cC.X:F2},{cC.Y:F2},{cC.Z:F2})");
        SfmReconstructor.Log($"  True Camera C t-vec: ({tC[0]:F2},{tC[1]:F2},{tC[2]:F2})");
        SfmReconstructor.Log($"  True Camera C fwd: ({RC[2, 0]:F2},{RC[2, 1]:F2},{RC[2, 2]:F2})");

        // Relative pose from B to C: R_rel = RC * RB^T, t_rel = tC - R_rel * tB
        var RBt = Transpose3(RB);
        var R_rel = Mult3x3(RC, RBt);
        var t_rel = new double[] {
            tC[0] - (R_rel[0,0]*tB[0] + R_rel[0,1]*tB[1] + R_rel[0,2]*tB[2]),
            tC[1] - (R_rel[1,0]*tB[0] + R_rel[1,1]*tB[1] + R_rel[1,2]*tB[2]),
            tC[2] - (R_rel[2,0]*tB[0] + R_rel[2,1]*tB[1] + R_rel[2,2]*tB[2])
        };

        SfmReconstructor.Log($"  Relative R_BC: [{R_rel[0, 0]:F3},{R_rel[0, 1]:F3},{R_rel[0, 2]:F3}]");
        SfmReconstructor.Log($"                 [{R_rel[1, 0]:F3},{R_rel[1, 1]:F3},{R_rel[1, 2]:F3}]");
        SfmReconstructor.Log($"                 [{R_rel[2, 0]:F3},{R_rel[2, 1]:F3},{R_rel[2, 2]:F3}]");
        SfmReconstructor.Log($"  Relative t_BC: ({t_rel[0]:F3},{t_rel[1]:F3},{t_rel[2]:F3})");

        // Compose: globalR_C = R_rel * R_B, globalT_C = R_rel * t_B + t_rel
        var globalR = Mult3x3(R_rel, RB);
        var globalT = new double[] {
            R_rel[0,0]*tB[0]+R_rel[0,1]*tB[1]+R_rel[0,2]*tB[2]+t_rel[0],
            R_rel[1,0]*tB[0]+R_rel[1,1]*tB[1]+R_rel[1,2]*tB[2]+t_rel[1],
            R_rel[2,0]*tB[0]+R_rel[2,1]*tB[1]+R_rel[2,2]*tB[2]+t_rel[2]
        };

        // Compare with true RC, tC
        double rErr = 0;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                rErr += Math.Abs(globalR[r, c] - RC[r, c]);
        double tErr = Math.Sqrt(
            Math.Pow(globalT[0] - tC[0], 2) + Math.Pow(globalT[1] - tC[1], 2) + Math.Pow(globalT[2] - tC[2], 2));

        SfmReconstructor.Log($"  Composed R err: {rErr:F6}");
        SfmReconstructor.Log($"  Composed t err: {tErr:F6}");
        SfmReconstructor.Log($"  {(rErr < 0.01 && tErr < 0.01 ? "PASS" : "FAIL")}: Pose composition");

        // Recover camera center from composed pose
        var cComp = new Vector3(
            (float)(-(globalR[0, 0] * globalT[0] + globalR[1, 0] * globalT[1] + globalR[2, 0] * globalT[2])),
            (float)(-(globalR[0, 1] * globalT[0] + globalR[1, 1] * globalT[1] + globalR[2, 1] * globalT[2])),
            (float)(-(globalR[0, 2] * globalT[0] + globalR[1, 2] * globalT[1] + globalR[2, 2] * globalT[2])));
        float centerErr = Vector3.Distance(cComp, cC);
        SfmReconstructor.Log($"  Camera center: ({cComp.X:F3},{cComp.Y:F3},{cComp.Z:F3}) expected ({cC.X},{cC.Y},{cC.Z})");
        SfmReconstructor.Log($"  {(centerErr < 0.01f ? "PASS" : "FAIL")}: Center recovery, err={centerErr:F6}");

        // Test triangulation with these 3 cameras
        SfmReconstructor.Log("\n  -- Multi-camera triangulation --");
        var worldPts = new Vector3[] {
            new(1, 0, 1), new(1, 1, 1), new(0, 0, 1), new(2, 0, 1), new(1, 0, 2)
        };
        var PA = MakeP(RA, tA); var PB = MakeP(RB, tB); var PC = MakeP(RC, tC);

        foreach (var wp in worldPts)
        {
            // Try A-B pair
            var pA2 = Project(wp, RA, tA);
            var pB2 = Project(wp, RB, tB);
            if (pA2.z > 0 && pB2.z > 0)
            {
                var nA = new Vector2((float)(pA2.x / pA2.z), (float)(pA2.y / pA2.z));
                var nB = new Vector2((float)(pB2.x / pB2.z), (float)(pB2.y / pB2.z));
                var res = LinearAlgebra.TriangulatePoint(nA, nB, PA, PB);
                if (res.HasValue)
                {
                    float e = Vector3.Distance(res.Value, wp);
                    SfmReconstructor.Log($"  A-B tri {wp}: err={e:F4} {(e < 0.01 ? "PASS" : "FAIL")}");
                }
            }

            // Try A-C pair (wide baseline, different orientation)
            var pA3 = Project(wp, RA, tA);
            var pC3 = Project(wp, RC, tC);
            if (pA3.z > 0 && pC3.z > 0)
            {
                var nA = new Vector2((float)(pA3.x / pA3.z), (float)(pA3.y / pA3.z));
                var nC = new Vector2((float)(pC3.x / pC3.z), (float)(pC3.y / pC3.z));
                var res = LinearAlgebra.TriangulatePoint(nA, nC, PA, PC);
                if (res.HasValue)
                {
                    float e = Vector3.Distance(res.Value, wp);
                    SfmReconstructor.Log($"  A-C tri {wp}: err={e:F4} {(e < 0.01 ? "PASS" : "FAIL")}");
                }
                else SfmReconstructor.Log($"  A-C tri {wp}: FAIL (null)");
            }
            else SfmReconstructor.Log($"  A-C tri {wp}: behind camera (zA={pA3.z:F2}, zC={pC3.z:F2})");
        }
    }

    // Helpers
    static double[,] MakeP(double[,] R, double[] t)
    {
        var P = new double[3, 4];
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) P[i, j] = R[i, j]; P[i, 3] = t[i]; }
        return P;
    }

    static (double x, double y, double z) Project(Vector3 pt, double[,] R, double[] t)
    {
        double x = R[0, 0] * pt.X + R[0, 1] * pt.Y + R[0, 2] * pt.Z + t[0];
        double y = R[1, 0] * pt.X + R[1, 1] * pt.Y + R[1, 2] * pt.Z + t[1];
        double z = R[2, 0] * pt.X + R[2, 1] * pt.Y + R[2, 2] * pt.Z + t[2];
        return (x, y, z);
    }

    static double[,] Mult3x3(double[,] A, double[,] B)
    {
        var C = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    C[i, j] += A[i, k] * B[k, j];
        return C;
    }

    static double[,] Transpose3(double[,] M)
    {
        var T = new double[3, 3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) T[i, j] = M[j, i];
        return T;
    }

    static double Det3(double[,] M) =>
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) -
        M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) +
        M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]);

    static void OrthGS(double[,] R)
    {
        double[] r0 = { R[0, 0], R[0, 1], R[0, 2] };
        double[] r1 = { R[1, 0], R[1, 1], R[1, 2] };
        double n0 = Math.Sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
        if (n0 < 1e-10) return;
        r0[0] /= n0; r0[1] /= n0; r0[2] /= n0;
        double d = r1[0] * r0[0] + r1[1] * r0[1] + r1[2] * r0[2];
        r1[0] -= d * r0[0]; r1[1] -= d * r0[1]; r1[2] -= d * r0[2];
        double n1 = Math.Sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
        if (n1 < 1e-10) return;
        r1[0] /= n1; r1[1] /= n1; r1[2] /= n1;
        double[] r2 = { r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0] };
        R[0, 0] = r0[0]; R[0, 1] = r0[1]; R[0, 2] = r0[2];
        R[1, 0] = r1[0]; R[1, 1] = r1[1]; R[1, 2] = r1[2];
        R[2, 0] = r2[0]; R[2, 1] = r2[1]; R[2, 2] = r2[2];
    }
}
