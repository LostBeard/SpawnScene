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
/// NOT the shipped path and currently referenced by nothing: <see cref="GpuGaussianRenderer"/>
/// renders every scene. This is kept as a readable CPU reference, and it shares the covariance
/// projection in <see cref="SplatCovariance"/> with the WGSL vertex stage so it cannot drift into
/// being a second, differently-wrong answer.
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
        var projected = new List<ProjectedGaussian>(scene.Count);

        float fx = camera.FocalX;
        float fy = camera.FocalY;
        float cx = camera.CenterX;
        float cy = camera.CenterY;

        // One implementation of the covariance projection, shared with the WGSL vertex stage and
        // unit-tested in SplatCovarianceTests. This used to be a second, inline copy of the Kerbl
        // math that transposed the view rotation - Sigma_cam came out as A^T Sigma A.
        WorldSpaceGeometry.ViewMatrixToCameraBasis(
            camera.ViewMatrix, out var right, out var up, out var forward, out var eye);

        for (int i = 0; i < scene.Count; i++)
        {
            ref readonly var g = ref scene.Gaussians[i];

            // Camera space: x right, y UP, z forward and positive in front.
            var rel = g.Position - eye;
            float camX = Vector3.Dot(right, rel);
            float camY = Vector3.Dot(up, rel);
            float depth = Vector3.Dot(forward, rel);

            // Cull: behind camera or too close/far
            if (depth <= camera.Near || depth >= camera.Far) continue;

            float invZ = 1.0f / depth;

            // Project to screen (pinhole). The framebuffer's y grows DOWNWARD, so the up-measured
            // camera y is negated here - and the covariance's off-diagonal term with it.
            float screenX = fx * camX * invZ + cx;
            float screenY = -fy * camY * invZ + cy;

            var q = g.Rotation;
            var cov3 = SplatCovariance.Cov3DFromScaleQuat(
                g.Scale.X, g.Scale.Y, g.Scale.Z,
                new SplatCovariance.Quat { X = q.X, Y = q.Y, Z = q.Z, W = q.W });

            var camCov = SplatCovariance.RotateToCamera(cov3,
                right.X, right.Y, right.Z,
                up.X, up.Y, up.Z,
                forward.X, forward.Y, forward.Z);

            var cov2 = SplatCovariance.ProjectCov2D(camCov, camX, camY, depth, fx, fy);

            float cov00 = cov2.A;
            float cov01 = -cov2.B;   // y flip: up-positive covariance into a y-down framebuffer
            float cov11 = cov2.C;

            // Get color and opacity
            var color = g.BaseColor;
            float opacity = g.Opacity;

            if (opacity < 1.0f / 255.0f) continue;

            var ellipse = SplatCovariance.EigenAxes(cov2, 3.0f);
            if (!ellipse.Valid) continue;

            float radius = MathF.Sqrt(ellipse.Ax * ellipse.Ax + ellipse.Ay * ellipse.Ay);

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
