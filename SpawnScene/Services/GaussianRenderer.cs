using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// CPU-based Gaussian Splat renderer that projects, sorts, and rasterizes
/// 3D Gaussians to a pixel framebuffer.
/// 
/// Pipeline:
///   1. Project 3D Gaussians → 2D screen space (covariance + color + opacity)
///   2. Sort by depth (back-to-front)
///   3. Rasterize: for each Gaussian, alpha-blend its contribution to covered pixels
///   4. Output: byte[] RGBA framebuffer ready for canvas putImageData
///
/// This is a reference implementation. GPU-accelerated version via ILGPU
/// will eventually replace the inner loops for real-time performance.
/// </summary>
public class GaussianRenderer
{
    /// <summary>
    /// Render a scene from the given camera, producing an RGBA framebuffer.
    /// </summary>
    public byte[] Render(GaussianScene scene, CameraParams camera)
    {
        int width = camera.Width;
        int height = camera.Height;
        var framebuffer = new byte[width * height * 4];

        if (scene.Count == 0) return framebuffer;

        // Step 1: Project all Gaussians to 2D
        var projected = ProjectGaussians(scene, camera);

        // Step 2: Sort by depth (front-to-back for early termination)
        Array.Sort(projected, (a, b) => a.Depth.CompareTo(b.Depth));

        // Step 3: Rasterize with alpha blending (front-to-back)
        RasterizeFrontToBack(projected, framebuffer, width, height);

        return framebuffer;
    }

    /// <summary>
    /// Project 3D Gaussians to 2D screen space.
    /// Returns only visible Gaussians (in front of camera, on screen).
    /// </summary>
    private ProjectedGaussian[] ProjectGaussians(GaussianScene scene, CameraParams camera)
    {
        var viewMatrix = camera.ViewMatrix;
        var projected = new List<ProjectedGaussian>(scene.Count);

        float fx = camera.FocalX;
        float fy = camera.FocalY;
        float cx = camera.CenterX;
        float cy = camera.CenterY;

        for (int i = 0; i < scene.Count; i++)
        {
            ref readonly var g = ref scene.Gaussians[i];

            // Transform to camera space
            var pos = g.Position;
            var camPos = Vector3.Transform(pos, viewMatrix);

            // In .NET's right-handed CreateLookAt, objects in front of the camera
            // have NEGATIVE Z. We negate it for our calculations.
            float depth = -camPos.Z;

            // Cull: behind camera or too close/far
            if (depth <= camera.Near || depth >= camera.Far) continue;

            float invZ = 1.0f / depth;

            // Project to screen (pinhole model)
            float screenX = fx * (-camPos.X) * invZ + cx;
            float screenY = fy * (-camPos.Y) * invZ + cy;

            // Compute 3D covariance
            var cov3D = g.CovarianceMatrix;

            // Project covariance to 2D using the Jacobian of the projection
            float j00 = fx * invZ;
            float j02 = -fx * (-camPos.X) * invZ * invZ;
            float j11 = fy * invZ;
            float j12 = -fy * (-camPos.Y) * invZ * invZ;

            // Extract view rotation
            float r00 = viewMatrix.M11, r01 = viewMatrix.M12, r02 = viewMatrix.M13;
            float r10 = viewMatrix.M21, r11 = viewMatrix.M22, r12 = viewMatrix.M23;
            float r20 = viewMatrix.M31, r21 = viewMatrix.M32, r22 = viewMatrix.M33;

            // Cov3D elements (symmetric)
            float s00 = cov3D.M11, s01 = cov3D.M12, s02 = cov3D.M13;
            float s11 = cov3D.M22, s12 = cov3D.M23;
            float s22 = cov3D.M33;

            // W = R * S * R^T
            float rs00 = r00 * s00 + r01 * s01 + r02 * s02;
            float rs01 = r00 * s01 + r01 * s11 + r02 * s12;
            float rs02 = r00 * s02 + r01 * s12 + r02 * s22;
            float rs10 = r10 * s00 + r11 * s01 + r12 * s02;
            float rs11 = r10 * s01 + r11 * s11 + r12 * s12;
            float rs12 = r10 * s02 + r11 * s12 + r12 * s22;
            float rs20 = r20 * s00 + r21 * s01 + r22 * s02;
            float rs21 = r20 * s01 + r21 * s11 + r22 * s12;
            float rs22 = r20 * s02 + r21 * s12 + r22 * s22;

            float w00 = rs00 * r00 + rs01 * r01 + rs02 * r02;
            float w01 = rs00 * r10 + rs01 * r11 + rs02 * r12;
            float w02 = rs00 * r20 + rs01 * r21 + rs02 * r22;
            float w11 = rs10 * r10 + rs11 * r11 + rs12 * r12;
            float w12 = rs10 * r20 + rs11 * r21 + rs12 * r22;
            float w22 = rs20 * r20 + rs21 * r21 + rs22 * r22;

            // Cov2D = J * W * J^T (2x2 symmetric)
            float cov00 = j00 * j00 * w00 + 2 * j00 * j02 * w02 + j02 * j02 * w22;
            float cov01 = j00 * j11 * w01 + j00 * j12 * w02 + j02 * j11 * w12 + j02 * j12 * w22;
            float cov11 = j11 * j11 * w11 + 2 * j11 * j12 * w12 + j12 * j12 * w22;

            // Numerical stability
            cov00 += 0.3f;
            cov11 += 0.3f;

            // Get color and opacity
            var color = g.BaseColor;
            float opacity = g.Opacity;

            if (opacity < 1.0f / 255.0f) continue;

            float det = cov00 * cov11 - cov01 * cov01;
            if (det <= 0) continue;

            float trace = cov00 + cov11;
            float mid = 0.5f * trace;
            float disc = MathF.Max(0.1f, mid * mid - det);
            float lambda1 = mid + MathF.Sqrt(disc);
            float radius = 3.0f * MathF.Sqrt(lambda1);

            if (screenX + radius < 0 || screenX - radius >= camera.Width ||
                screenY + radius < 0 || screenY - radius >= camera.Height) continue;

            projected.Add(new ProjectedGaussian
            {
                ScreenX = screenX,
                ScreenY = screenY,
                Depth = depth,
                Cov2D_00 = cov00,
                Cov2D_01 = cov01,
                Cov2D_11 = cov11,
                R = Math.Clamp(color.X, 0f, 1f),
                G = Math.Clamp(color.Y, 0f, 1f),
                B = Math.Clamp(color.Z, 0f, 1f),
                Opacity = opacity,
                OriginalIndex = i,
            });
        }

        Console.WriteLine($"[Render] Projected {projected.Count}/{scene.Count} Gaussians. Camera at {camera.Position}, looking {camera.Forward}");
        if (projected.Count > 0)
        {
            var first = projected[0];
            Console.WriteLine($"[Render] First projected: screen=({first.ScreenX:F1},{first.ScreenY:F1}), depth={first.Depth:F2}, color=({first.R:F2},{first.G:F2},{first.B:F2}), opacity={first.Opacity:F2}");
        }

        return projected.ToArray();
    }

    /// <summary>
    /// Rasterize projected Gaussians using front-to-back alpha blending.
    /// Uses per-pixel alpha accumulation for early termination.
    /// </summary>
    private void RasterizeFrontToBack(ProjectedGaussian[] sorted, byte[] framebuffer, int width, int height)
    {
        // Per-pixel accumulated transmittance (1 = fully transparent, 0 = fully opaque)
        var transmittance = new float[width * height];
        Array.Fill(transmittance, 1.0f);

        // RGB accumulators
        var accR = new float[width * height];
        var accG = new float[width * height];
        var accB = new float[width * height];

        for (int gi = 0; gi < sorted.Length; gi++)
        {
            ref readonly var g = ref sorted[gi];

            var (minX, minY, maxX, maxY) = g.GetScreenBounds(width, height);
            if (minX > maxX || minY > maxY) continue;

            // Precompute inverse covariance for Gaussian evaluation
            float det = g.Cov2D_00 * g.Cov2D_11 - g.Cov2D_01 * g.Cov2D_01;
            if (det <= 0) continue;

            float invDet = 1.0f / det;
            float inv00 = g.Cov2D_11 * invDet;
            float inv01 = -g.Cov2D_01 * invDet;
            float inv11 = g.Cov2D_00 * invDet;

            for (int y = minY; y <= maxY; y++)
            {
                float dy = y + 0.5f - g.ScreenY;
                for (int x = minX; x <= maxX; x++)
                {
                    int pIdx = y * width + x;

                    // Early out: pixel already saturated
                    if (transmittance[pIdx] < 0.004f) continue;

                    float dx = x + 0.5f - g.ScreenX;

                    // Evaluate 2D Gaussian: exp(-0.5 * [dx,dy] * Cov^{-1} * [dx,dy]^T)
                    float power = -0.5f * (inv00 * dx * dx + 2 * inv01 * dx * dy + inv11 * dy * dy);
                    if (power > 0 || power < -4.0f) continue; // clamp for perf

                    float gauss = MathF.Exp(power);
                    float alpha = Math.Min(0.99f, g.Opacity * gauss);
                    if (alpha < 1.0f / 255.0f) continue;

                    // Front-to-back blending
                    float weight = alpha * transmittance[pIdx];
                    accR[pIdx] += weight * g.R;
                    accG[pIdx] += weight * g.G;
                    accB[pIdx] += weight * g.B;
                    transmittance[pIdx] *= (1.0f - alpha);
                }
            }
        }

        // Convert accumulators to RGBA bytes
        for (int i = 0; i < width * height; i++)
        {
            int fbIdx = i * 4;
            framebuffer[fbIdx + 0] = (byte)Math.Clamp((int)(accR[i] * 255), 0, 255);
            framebuffer[fbIdx + 1] = (byte)Math.Clamp((int)(accG[i] * 255), 0, 255);
            framebuffer[fbIdx + 2] = (byte)Math.Clamp((int)(accB[i] * 255), 0, 255);
            framebuffer[fbIdx + 3] = (byte)Math.Clamp((int)((1.0f - transmittance[i]) * 255), 0, 255);
        }
    }
}
