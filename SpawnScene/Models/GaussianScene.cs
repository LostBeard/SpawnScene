using System.Numerics;

namespace SpawnScene.Models;

/// <summary>
/// A complete Gaussian Splat scene containing all Gaussians and metadata.
/// This is the primary data model that gets rendered and can be loaded/saved.
/// </summary>
public class GaussianScene
{
    /// <summary>All Gaussians in the scene.</summary>
    public Gaussian3D[] Gaussians { get; set; } = [];

    /// <summary>
    /// Splat count when scene data lives entirely in a GPU buffer (GPU fast path).
    /// When set, Count returns this value instead of Gaussians.Length so that the
    /// renderer and UI see the correct count without a CPU-side Gaussian array.
    /// </summary>
    public int GpuSplatCount { get; set; } = 0;

    /// <summary>Number of Gaussians in the scene (GPU-resident or CPU).</summary>
    public int Count => GpuSplatCount > 0 ? GpuSplatCount : Gaussians.Length;

    /// <summary>The SH degree used (0 = DC only, 1 = first order, etc.).</summary>
    public int ShDegree { get; set; } = 0;

    /// <summary>Scene origin/center for camera positioning.</summary>
    public Vector3 Center { get; set; } = Vector3.Zero;

    /// <summary>Scene extent (half-diagonal of bounding box).</summary>
    public float Extent { get; set; } = 1.0f;

    /// <summary>Optional: camera parameters from training views.</summary>
    public List<CameraParams> TrainingCameras { get; set; } = [];

    /// <summary>
    /// Every posed image available as photometric supervision, which is NOT the same set as the
    /// views used to initialise geometry.
    ///
    /// Depth initialisation is capped by the joint-depth model
    /// (<c>DepthEstimationService.MaxMultiViewImages</c>); training supervision only needs an
    /// image and a pose, so it should use everything available. TempleRing ships 16 posed photos
    /// and the pipeline was initialising - and therefore supervising - from 4.
    ///
    /// Images are referenced by name rather than held as pixels: 16 x 640 x 480 RGBA is ~20 MB
    /// and belongs on the GPU at training time, not in the scene model.
    /// </summary>
    public List<TrainingView> TrainingViews { get; set; } = [];

    /// <summary>Source file path or name (for display purposes).</summary>
    public string? SourceName { get; set; }

    /// <summary>
    /// Compute scene bounds from the Gaussians and update Center/Extent.
    /// </summary>
    public void ComputeBounds()
    {
        if (Count == 0) return;

        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);

        for (int i = 0; i < Count; i++)
        {
            var pos = Gaussians[i].Position;
            min = Vector3.Min(min, pos);
            max = Vector3.Max(max, pos);
        }

        Center = (min + max) * 0.5f;
        Extent = Vector3.Distance(min, max) * 0.5f;

        if (Extent < 0.001f) Extent = 1.0f; // Prevent degenerate scenes
    }

    /// <summary>
    /// Create a scene from a point cloud with default Gaussian initialization.
    /// </summary>
    public static GaussianScene FromPointCloud(PointCloud pointCloud, float gaussianScale = 0.01f)
    {
        var scene = new GaussianScene
        {
            Gaussians = pointCloud.ToGaussians(gaussianScale),
            ShDegree = 0,
        };
        scene.ComputeBounds();
        return scene;
    }
}

/// <summary>A posed image usable as photometric supervision during optimisation.</summary>
public sealed class TrainingView
{
    /// <summary>Pose and intrinsics this image was taken with.</summary>
    public required CameraParams Camera { get; init; }

    /// <summary>Resolvable source name, e.g. a dataset filename.</summary>
    public required string ImageName { get; init; }

    /// <summary>True when this view also seeded geometry (vs supervision only).</summary>
    public bool UsedForInit { get; init; }

    /// <summary>
    /// Counter-clockwise quarter turns already applied to <see cref="Camera"/>, which the
    /// target image must be given too. Storing it on the view rather than as a global setting
    /// is what keeps a mixed-orientation capture (a phone turned mid-video) coherent.
    /// </summary>
    public int QuarterTurns { get; init; }

    /// <summary>
    /// Whether the optimiser is allowed to fit to this photograph.
    ///
    /// False marks a genuine HOLD-OUT: the view is still posed, still loaded and still scored,
    /// but its pixels never reach the loss. Without this the only thing a novel-view score
    /// measures is how well the model reproduces its own training images, which is not the
    /// question - and is a much larger number.
    /// </summary>
    public bool UsedForSupervision { get; init; } = true;
}
