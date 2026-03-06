using System.Runtime.InteropServices;

namespace SpawnScene.Models;

/// <summary>
/// A projected 2D Gaussian ready for rasterization.
/// Created by the projection step from a 3D Gaussian + camera.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ProjectedGaussian
{
    /// <summary>Screen-space center X.</summary>
    public float ScreenX;

    /// <summary>Screen-space center Y.</summary>
    public float ScreenY;

    /// <summary>Depth (distance from camera, for sorting).</summary>
    public float Depth;

    /// <summary>2D covariance matrix element [0,0].</summary>
    public float Cov2D_00;

    /// <summary>2D covariance matrix element [0,1] = [1,0].</summary>
    public float Cov2D_01;

    /// <summary>2D covariance matrix element [1,1].</summary>
    public float Cov2D_11;

    /// <summary>Red color component (0-1).</summary>
    public float R;

    /// <summary>Green color component (0-1).</summary>
    public float G;

    /// <summary>Blue color component (0-1).</summary>
    public float B;

    /// <summary>Activated opacity (0-1).</summary>
    public float Opacity;

    /// <summary>Index back to the original Gaussian (for gradient tracking).</summary>
    public int OriginalIndex;

    /// <summary>
    /// Compute the screen-space bounding rectangle for this Gaussian.
    /// Uses 3-sigma extent of the 2D covariance ellipse.
    /// </summary>
    public readonly (int minX, int minY, int maxX, int maxY) GetScreenBounds(int screenWidth, int screenHeight)
    {
        // Eigenvalue-based radius (3-sigma for 99.7% coverage)
        float det = Cov2D_00 * Cov2D_11 - Cov2D_01 * Cov2D_01;
        float trace = Cov2D_00 + Cov2D_11;
        float mid = 0.5f * trace;
        float disc = MathF.Max(0.1f, mid * mid - det);
        float lambda1 = mid + MathF.Sqrt(disc);
        float radius = 3.0f * MathF.Sqrt(lambda1);

        int minX = Math.Max(0, (int)(ScreenX - radius));
        int minY = Math.Max(0, (int)(ScreenY - radius));
        int maxX = Math.Min(screenWidth - 1, (int)(ScreenX + radius));
        int maxY = Math.Min(screenHeight - 1, (int)(ScreenY + radius));

        return (minX, minY, maxX, maxY);
    }
}
