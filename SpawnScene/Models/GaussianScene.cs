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
