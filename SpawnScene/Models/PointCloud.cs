using System.Numerics;

namespace SpawnScene.Models;

/// <summary>
/// A collection of 3D points with colors, typically generated from
/// Structure-from-Motion or depth estimation.
/// </summary>
public class PointCloud
{
    /// <summary>3D positions of each point.</summary>
    public Vector3[] Positions { get; set; } = [];

    /// <summary>RGB colors (0-1 range) of each point.</summary>
    public Vector3[] Colors { get; set; } = [];

    /// <summary>Number of points in the cloud.</summary>
    public int Count => Positions.Length;

    /// <summary>
    /// Convert this point cloud to an initial set of Gaussians.
    /// Each point becomes a small isotropic Gaussian.
    /// </summary>
    public Gaussian3D[] ToGaussians(float initialScale = 0.01f, float initialOpacity = 0.8f)
    {
        var gaussians = new Gaussian3D[Count];
        var scale = new Vector3(initialScale);
        var rotation = Quaternion.Identity;

        for (int i = 0; i < Count; i++)
        {
            var color = i < Colors.Length ? Colors[i] : new Vector3(0.5f);
            gaussians[i] = Gaussian3D.Create(Positions[i], scale, rotation, initialOpacity, color);
        }

        return gaussians;
    }

    /// <summary>
    /// Compute the axis-aligned bounding box of the point cloud.
    /// </summary>
    public (Vector3 min, Vector3 max) GetBounds()
    {
        if (Count == 0) return (Vector3.Zero, Vector3.Zero);

        var min = Positions[0];
        var max = Positions[0];

        for (int i = 1; i < Count; i++)
        {
            min = Vector3.Min(min, Positions[i]);
            max = Vector3.Max(max, Positions[i]);
        }

        return (min, max);
    }
}
