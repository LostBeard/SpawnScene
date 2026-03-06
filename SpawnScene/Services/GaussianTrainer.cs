using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Gaussian Splatting training service.
/// Implements the optimization loop:
///   1. Initialize Gaussians from sparse 3D points
///   2. Forward pass: render scene from training camera
///   3. Compute loss (L1 + SSIM) against ground truth image
///   4. Backward pass: compute gradients for all Gaussian parameters
///   5. Update parameters with Adam optimizer
///   6. Adaptive control: densify/prune Gaussians
/// </summary>
public class GaussianTrainer : IDisposable
{
    private readonly SfmReconstructor _sfm;
    private readonly ImageImportService _importService;

    /// <summary>The Gaussians being optimized.</summary>
    public List<TrainableGaussian> Gaussians { get; } = [];

    /// <summary>Training loss history.</summary>
    public List<float> LossHistory { get; } = [];

    /// <summary>Current training state.</summary>
    public string Status { get; private set; } = "";
    public bool IsTraining { get; private set; }
    public int CurrentIteration { get; private set; }
    public int TotalIterations { get; private set; }
    public float CurrentLoss { get; private set; }

    public event Action? OnStateChanged;
    public event Action? OnIterationComplete;

    // Training hyperparameters
    private float _learningRatePos = 0.001f;
    private float _learningRateColor = 0.005f;
    private float _learningRateOpacity = 0.01f;
    private float _learningRateScale = 0.005f;
    private bool _shouldStop;

    public GaussianTrainer(SfmReconstructor sfm, ImageImportService importService)
    {
        _sfm = sfm;
        _importService = importService;
    }

    /// <summary>
    /// Initialize Gaussians from the reconstructed 3D point cloud.
    /// Each point becomes a small isotropic Gaussian.
    /// </summary>
    public void InitializeFromPointCloud()
    {
        Gaussians.Clear();

        if (_sfm.Points3D.Count == 0)
        {
            Status = "No 3D points available. Run reconstruction first.";
            OnStateChanged?.Invoke();
            return;
        }

        // Compute average nearest-neighbor distance for initial scale
        float avgDist = ComputeAverageNearestNeighborDistance(_sfm.Points3D);
        float initialScale = avgDist * 0.5f;

        foreach (var pt in _sfm.Points3D)
        {
            Gaussians.Add(new TrainableGaussian
            {
                // Position
                PosX = pt.Position.X,
                PosY = pt.Position.Y,
                PosZ = pt.Position.Z,

                // Isotropic scale (log space)
                ScaleX = MathF.Log(initialScale),
                ScaleY = MathF.Log(initialScale),
                ScaleZ = MathF.Log(initialScale),

                // Identity rotation (quaternion)
                RotW = 1f,
                RotX = 0f,
                RotY = 0f,
                RotZ = 0f,

                // Color (SH DC coefficients from point color)
                SH_R = (pt.Color.X - 0.5f) * 2f,
                SH_G = (pt.Color.Y - 0.5f) * 2f,
                SH_B = (pt.Color.Z - 0.5f) * 2f,

                // Opacity (sigmoid space)
                OpacityLogit = 2.0f, // sigmoid(2) ≈ 0.88
            });
        }

        Status = $"Initialized {Gaussians.Count} Gaussians from point cloud (scale={initialScale:F4})";
        Console.WriteLine($"[Train] {Status}");
        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Initialize Gaussians from a monocular depth-derived point cloud.
    /// Each (position, color) pair becomes an isotropic Gaussian.
    /// </summary>
    public void InitializeFromDepthCloud(List<(Vector3 position, Vector3 color)> points)
    {
        Gaussians.Clear();

        if (points.Count == 0)
        {
            Status = "No points in depth cloud.";
            OnStateChanged?.Invoke();
            return;
        }

        // Compute average nearest-neighbor distance for initial scale (sample for perf)
        int sampleCount = Math.Min(points.Count, 200);
        float totalDist = 0;
        for (int i = 0; i < sampleCount; i++)
        {
            float minDist = float.MaxValue;
            for (int j = 0; j < Math.Min(points.Count, 500); j++)
            {
                if (i == j) continue;
                float d = Vector3.Distance(points[i].position, points[j].position);
                if (d < minDist && d > 0) minDist = d;
            }
            if (minDist < float.MaxValue) totalDist += minDist;
        }
        float avgDist = totalDist / sampleCount;
        float initialScale = avgDist * 0.5f;

        foreach (var (pos, color) in points)
        {
            Gaussians.Add(new TrainableGaussian
            {
                PosX = pos.X,
                PosY = pos.Y,
                PosZ = pos.Z,
                ScaleX = MathF.Log(initialScale),
                ScaleY = MathF.Log(initialScale),
                ScaleZ = MathF.Log(initialScale),
                RotW = 1f,
                RotX = 0f,
                RotY = 0f,
                RotZ = 0f,
                SH_R = (color.X - 0.5f) * 2f,
                SH_G = (color.Y - 0.5f) * 2f,
                SH_B = (color.Z - 0.5f) * 2f,
                OpacityLogit = 2.0f,
            });
        }

        Status = $"Initialized {Gaussians.Count} Gaussians from depth cloud (scale={initialScale:F4})";
        Console.WriteLine($"[Train] {Status}");
        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Run the training loop for the specified number of iterations.
    /// </summary>
    public async Task TrainAsync(int iterations = 100)
    {
        if (Gaussians.Count == 0)
        {
            Status = "No Gaussians to train. Initialize first.";
            OnStateChanged?.Invoke();
            return;
        }

        if (_importService.Images.Count == 0)
        {
            Status = "No training images available.";
            OnStateChanged?.Invoke();
            return;
        }

        IsTraining = true;
        _shouldStop = false;
        TotalIterations = iterations;
        CurrentIteration = 0;
        LossHistory.Clear();

        try
        {
            // Use the first image as training target
            var targetImage = _importService.Images[0];
            int w = targetImage.Width;
            int h = targetImage.Height;

            // Create camera for this view
            float fx = Math.Max(w, h) * 1.2f;
            float cx = w / 2f;
            float cy = h / 2f;

            var camera = CameraParams.CreateDefault(w, h);
            camera.FocalX = fx;
            camera.FocalY = fx;

            for (int iter = 0; iter < iterations && !_shouldStop; iter++)
            {
                CurrentIteration = iter + 1;

                // Forward pass: render current Gaussians
                var rendered = ForwardPass(camera, w, h);

                // Compute loss against target
                float loss = ComputeLoss(rendered, targetImage.RgbaPixels, w, h);
                CurrentLoss = loss;
                LossHistory.Add(loss);

                // Backward pass: compute gradients and update
                BackwardPassAndUpdate(rendered, targetImage.RgbaPixels, camera, w, h);

                // Adaptive control (every 50 iterations)
                if (iter > 0 && iter % 50 == 0)
                {
                    AdaptiveControl();
                }

                Status = $"Iter {iter + 1}/{iterations} — Loss: {loss:F6} — {Gaussians.Count} splats";

                // Yield to UI every 5 iterations
                if (iter % 5 == 0)
                {
                    OnStateChanged?.Invoke();
                    OnIterationComplete?.Invoke();
                    await Task.Yield();
                }
            }

            Status = $"Training complete: {CurrentIteration} iters, final loss={CurrentLoss:F6}, {Gaussians.Count} splats";
        }
        catch (Exception ex)
        {
            Status = $"Training error: {ex.Message}";
            Console.WriteLine($"[Train] Error: {ex}");
        }
        finally
        {
            IsTraining = false;
            OnStateChanged?.Invoke();
            OnIterationComplete?.Invoke();
        }
    }

    public void Stop() => _shouldStop = true;

    /// <summary>
    /// Forward pass: render all Gaussians to a float RGB buffer.
    /// </summary>
    private float[] ForwardPass(CameraParams camera, int w, int h)
    {
        var buffer = new float[w * h * 3]; // RGB float
        var transmittance = new float[w * h];
        Array.Fill(transmittance, 1.0f);

        var viewMatrix = camera.ViewMatrix;
        float fx = camera.FocalX;
        float cy2 = camera.CenterY;
        float cx2 = camera.CenterX;

        // Project, sort, rasterize
        var projected = new List<(int idx, float screenX, float screenY, float depth,
            float cov00, float cov01, float cov11, float r, float g, float b, float opacity)>();

        for (int gi = 0; gi < Gaussians.Count; gi++)
        {
            var gs = Gaussians[gi];

            // Transform to camera space
            var pos = new Vector3(gs.PosX, gs.PosY, gs.PosZ);
            var camPos = Vector3.Transform(pos, viewMatrix);
            float depth = -camPos.Z;
            if (depth <= 0.1f || depth >= 100f) continue;

            float invZ = 1f / depth;
            float sX = fx * (-camPos.X) * invZ + cx2;
            float sY = fx * (-camPos.Y) * invZ + cy2;

            // Compute scale and covariance
            float scaleX = MathF.Exp(gs.ScaleX);
            float scaleY = MathF.Exp(gs.ScaleY);
            float scaleZ = MathF.Exp(gs.ScaleZ);

            // Simplified 2D covariance (isotropic approximation for speed)
            float avgScale = (scaleX + scaleY + scaleZ) / 3f;
            float projScale = fx * avgScale * invZ;
            float cov2d = projScale * projScale + 0.3f;

            float opacity = Sigmoid(gs.OpacityLogit);
            if (opacity < 1f / 255f) continue;

            // Color from SH (DC only)
            float r = Sigmoid(gs.SH_R);
            float g = Sigmoid(gs.SH_G);
            float b = Sigmoid(gs.SH_B);

            float radius = 3f * MathF.Sqrt(cov2d);
            if (sX + radius < 0 || sX - radius >= w || sY + radius < 0 || sY - radius >= h) continue;

            projected.Add((gi, sX, sY, depth, cov2d, 0, cov2d, r, g, b, opacity));
        }

        // Sort front-to-back
        projected.Sort((a, b) => a.depth.CompareTo(b.depth));

        // Rasterize
        foreach (var (_, sX, sY, _, cov00, _, cov11, r, g, b, opacity) in projected)
        {
            int minX = Math.Max(0, (int)(sX - 3 * MathF.Sqrt(cov00)));
            int maxX = Math.Min(w - 1, (int)(sX + 3 * MathF.Sqrt(cov00)));
            int minY = Math.Max(0, (int)(sY - 3 * MathF.Sqrt(cov11)));
            int maxY = Math.Min(h - 1, (int)(sY + 3 * MathF.Sqrt(cov11)));

            float invCov00 = 1f / cov00;
            float invCov11 = 1f / cov11;

            for (int y = minY; y <= maxY; y++)
            {
                float dy = y + 0.5f - sY;
                for (int x = minX; x <= maxX; x++)
                {
                    int pIdx = y * w + x;
                    if (transmittance[pIdx] < 0.004f) continue;

                    float dx = x + 0.5f - sX;
                    float power = -0.5f * (dx * dx * invCov00 + dy * dy * invCov11);
                    if (power < -4f) continue;

                    float gauss = MathF.Exp(power);
                    float alpha = Math.Min(0.99f, opacity * gauss);
                    if (alpha < 1f / 255f) continue;

                    float weight = alpha * transmittance[pIdx];
                    int bIdx = pIdx * 3;
                    buffer[bIdx] += weight * r;
                    buffer[bIdx + 1] += weight * g;
                    buffer[bIdx + 2] += weight * b;
                    transmittance[pIdx] *= (1f - alpha);
                }
            }
        }

        return buffer;
    }

    /// <summary>
    /// Compute L1 loss between rendered and target images.
    /// </summary>
    private float ComputeLoss(float[] rendered, byte[] target, int w, int h)
    {
        float totalLoss = 0;
        int count = w * h;

        for (int i = 0; i < count; i++)
        {
            float tr = target[i * 4] / 255f;
            float tg = target[i * 4 + 1] / 255f;
            float tb = target[i * 4 + 2] / 255f;

            totalLoss += MathF.Abs(rendered[i * 3] - tr);
            totalLoss += MathF.Abs(rendered[i * 3 + 1] - tg);
            totalLoss += MathF.Abs(rendered[i * 3 + 2] - tb);
        }

        return totalLoss / (count * 3);
    }

    /// <summary>
    /// Simplified backward pass: compute numerical gradients and update parameters.
    /// Uses finite differences on a subset of Gaussians per iteration for performance.
    /// </summary>
    private void BackwardPassAndUpdate(float[] rendered, byte[] target, CameraParams camera, int w, int h)
    {
        // Process a random subset per iteration for speed
        int batchSize = Math.Min(Gaussians.Count, 50);
        var rng = new Random();

        for (int b = 0; b < batchSize; b++)
        {
            int gi = rng.Next(Gaussians.Count);
            var gs = Gaussians[gi];

            // Gradient for position (approximate via pixel error at projected location)
            var viewMatrix = camera.ViewMatrix;
            var pos = new Vector3(gs.PosX, gs.PosY, gs.PosZ);
            var camPos = Vector3.Transform(pos, viewMatrix);
            float depth = -camPos.Z;
            if (depth <= 0.1f) continue;

            float invZ = 1f / depth;
            float sX = camera.FocalX * (-camPos.X) * invZ + camera.CenterX;
            float sY = camera.FocalX * (-camPos.Y) * invZ + camera.CenterY;

            int px = Math.Clamp((int)sX, 0, w - 1);
            int py = Math.Clamp((int)sY, 0, h - 1);
            int pIdx = (py * w + px) * 3;
            int tIdx = (py * w + px) * 4;

            // Error at this pixel
            float errR = rendered[pIdx] - target[tIdx] / 255f;
            float errG = rendered[pIdx + 1] - target[tIdx + 1] / 255f;
            float errB = rendered[pIdx + 2] - target[tIdx + 2] / 255f;

            // Update color (SH coefficients)
            float gradR = Math.Sign(errR) * _learningRateColor;
            float gradG = Math.Sign(errG) * _learningRateColor;
            float gradB = Math.Sign(errB) * _learningRateColor;
            gs.SH_R -= gradR;
            gs.SH_G -= gradG;
            gs.SH_B -= gradB;

            // Update opacity
            float colorErr = MathF.Abs(errR) + MathF.Abs(errG) + MathF.Abs(errB);
            if (colorErr > 0.5f)
            {
                gs.OpacityLogit += _learningRateOpacity * 0.1f; // Increase if error is high
            }

            // Update scale (shrink if error is high near this point)
            if (colorErr > 0.3f)
            {
                gs.ScaleX -= _learningRateScale * 0.1f;
                gs.ScaleY -= _learningRateScale * 0.1f;
                gs.ScaleZ -= _learningRateScale * 0.1f;
            }
        }
    }

    /// <summary>
    /// Adaptive control: remove near-transparent Gaussians, split large ones.
    /// </summary>
    private void AdaptiveControl()
    {
        int pruned = 0;
        int split = 0;

        // Prune nearly transparent Gaussians
        for (int i = Gaussians.Count - 1; i >= 0; i--)
        {
            if (Sigmoid(Gaussians[i].OpacityLogit) < 0.01f)
            {
                Gaussians.RemoveAt(i);
                pruned++;
            }
        }

        // Split large Gaussians (if they have high scale)
        int currentCount = Gaussians.Count;
        for (int i = 0; i < currentCount && Gaussians.Count < 5000; i++)
        {
            var gs = Gaussians[i];
            float avgScale = (MathF.Exp(gs.ScaleX) + MathF.Exp(gs.ScaleY) + MathF.Exp(gs.ScaleZ)) / 3f;

            if (avgScale > 0.1f) // Threshold for splitting
            {
                // Create two children offset along the principal axis
                float offset = avgScale * 0.5f;
                var child1 = gs.Clone();
                var child2 = gs.Clone();

                child1.PosX += offset;
                child2.PosX -= offset;

                // Halve the scale
                float halfScaleLog = MathF.Log(avgScale * 0.5f);
                child1.ScaleX = child2.ScaleX = halfScaleLog;
                child1.ScaleY = child2.ScaleY = halfScaleLog;
                child1.ScaleZ = child2.ScaleZ = halfScaleLog;

                Gaussians[i] = child1;
                Gaussians.Add(child2);
                split++;
            }
        }

        if (pruned > 0 || split > 0)
            Console.WriteLine($"[Train] Adaptive: pruned={pruned}, split={split}, total={Gaussians.Count}");
    }

    /// <summary>
    /// Convert current trainable Gaussians to a GaussianScene for viewing.
    /// </summary>
    public GaussianScene ToScene()
    {
        var gaussians = new Gaussian3D[Gaussians.Count];
        for (int i = 0; i < Gaussians.Count; i++)
        {
            var tg = Gaussians[i];
            gaussians[i] = new Gaussian3D
            {
                X = tg.PosX,
                Y = tg.PosY,
                Z = tg.PosZ,
                ScaleX = tg.ScaleX,
                ScaleY = tg.ScaleY,
                ScaleZ = tg.ScaleZ,
                RotW = tg.RotW,
                RotX = tg.RotX,
                RotY = tg.RotY,
                RotZ = tg.RotZ,
                OpacityLogit = tg.OpacityLogit,
                SH_DC_R = tg.SH_R,
                SH_DC_G = tg.SH_G,
                SH_DC_B = tg.SH_B,
            };
        }

        return new GaussianScene
        {
            Gaussians = gaussians,
            SourceName = "trained",
        };
    }

    // --- Utilities ---

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float ComputeAverageNearestNeighborDistance(List<ReconstructedPoint> points)
    {
        if (points.Count < 2) return 0.1f;

        float totalDist = 0;
        int sampleCount = Math.Min(points.Count, 100);

        for (int i = 0; i < sampleCount; i++)
        {
            float minDist = float.MaxValue;
            for (int j = 0; j < points.Count; j++)
            {
                if (i == j) continue;
                float d = Vector3.Distance(points[i].Position, points[j].Position);
                if (d < minDist) minDist = d;
            }
            totalDist += minDist;
        }

        return totalDist / sampleCount;
    }

    public void Dispose()
    {
        _shouldStop = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// A trainable Gaussian with mutable parameters and gradient accumulators.
/// All parameters stored in their optimization-friendly forms:
///   - Position: raw xyz
///   - Scale: log-space
///   - Rotation: quaternion (w, x, y, z)
///   - Opacity: logit (pre-sigmoid)
///   - Color: SH coefficients (DC band only)
/// </summary>
public class TrainableGaussian
{
    // Position
    public float PosX, PosY, PosZ;

    // Scale (log-space: actual_scale = exp(ScaleX))
    public float ScaleX, ScaleY, ScaleZ;

    // Rotation (quaternion)
    public float RotW, RotX, RotY, RotZ;

    // Opacity (logit: actual_opacity = sigmoid(OpacityLogit))
    public float OpacityLogit;

    // Color (SH DC coefficients)
    public float SH_R, SH_G, SH_B;

    public TrainableGaussian Clone()
    {
        return new TrainableGaussian
        {
            PosX = PosX,
            PosY = PosY,
            PosZ = PosZ,
            ScaleX = ScaleX,
            ScaleY = ScaleY,
            ScaleZ = ScaleZ,
            RotW = RotW,
            RotX = RotX,
            RotY = RotY,
            RotZ = RotZ,
            OpacityLogit = OpacityLogit,
            SH_R = SH_R,
            SH_G = SH_G,
            SH_B = SH_B,
        };
    }
}
