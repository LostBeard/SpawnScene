using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// EquirectProjection (360 frames as pinhole views): the convention is COLMAP's EQUIRECTANGULAR model, so a sign slip here would
/// not show up as a crash - it would show up as poses that disagree with every other tool. Each test derives its expectation a
/// different way from the production code (explicit pitch-then-yaw rotation, a panorama that stores its own directions).
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter EquirectProjection</c>
/// </summary>
public class EquirectProjectionTests
{
    const int W = 720, H = 360;

    [Test]
    public void DirectionAndPixelAreInverses()
    {
        var rng = new Random(11);
        for (int n = 0; n < 2000; n++)
        {
            double u = rng.NextDouble() * W, v = 0.5 + rng.NextDouble() * (H - 1);
            var d = EquirectProjection.DirectionForPixel(u, v, W, H);
            var (u2, v2) = EquirectProjection.PixelForDirection(d.X * 3.7, d.Y * 3.7, d.Z * 3.7, W, H);   // length must not matter
            double du = Math.Abs(u - u2); du = Math.Min(du, W - du);                                         // the seam is one point
            Assert.That(du, Is.LessThan(1e-6).Or.GreaterThan(W - 1e-6), $"u {u} -> {u2}");
            Assert.That(v2, Is.EqualTo(v).Within(1e-6));
        }
    }

    [Test]
    public void AxesLandWhereColmapPutsThem()
    {
        // x right, y DOWN, z forward: ahead is the image centre, +x is three quarters across, up (-y) is the top row.
        var ahead = EquirectProjection.PixelForDirection(0, 0, 1, W, H);
        Assert.That(ahead.U, Is.EqualTo(W / 2.0).Within(1e-9)); Assert.That(ahead.V, Is.EqualTo(H / 2.0).Within(1e-9));
        var right = EquirectProjection.PixelForDirection(1, 0, 0, W, H);
        Assert.That(right.U, Is.EqualTo(0.75 * W).Within(1e-9)); Assert.That(right.V, Is.EqualTo(H / 2.0).Within(1e-9));
        var left = EquirectProjection.PixelForDirection(-1, 0, 0, W, H);
        Assert.That(left.U, Is.EqualTo(0.25 * W).Within(1e-9));
        Assert.That(EquirectProjection.PixelForDirection(0, -1, 0, W, H).V, Is.EqualTo(0).Within(1e-9), "straight up is the top row");
        Assert.That(EquirectProjection.PixelForDirection(0, 1, 0, W, H).V, Is.EqualTo(H).Within(1e-9), "straight down is the bottom row");
    }

    /// <summary>The independent derivation: tilt the view-frame ray up by pitch about x, then yaw by heading about y.</summary>
    static (double X, double Y, double Z) PanoDirection(double headingDeg, double pitchDeg, double rx, double ry, double rz)
    {
        double h = headingDeg * Math.PI / 180, p = pitchDeg * Math.PI / 180;
        double y1 = ry * Math.Cos(p) - rz * Math.Sin(p), z1 = ry * Math.Sin(p) + rz * Math.Cos(p);
        double x2 = rx * Math.Cos(h) + z1 * Math.Sin(h), z2 = -rx * Math.Sin(h) + z1 * Math.Cos(h);
        return (x2, y1, z2);
    }

    [Test]
    public void ViewRotationMatchesPitchThenYaw()
    {
        var rng = new Random(5);
        foreach (var view in EquirectProjection.Ring10.Concat(new[] { new EquirectProjection.ViewSpec("odd", 37, 23), new EquirectProjection.ViewSpec("odd2", 301, -58) }))
        {
            var R = EquirectProjection.ViewRotation(view.HeadingDeg, view.PitchDeg);
            for (int n = 0; n < 50; n++)
            {
                double rx = rng.NextDouble() * 2 - 1, ry = rng.NextDouble() * 2 - 1, rz = 1;
                var e = PanoDirection(view.HeadingDeg, view.PitchDeg, rx, ry, rz);
                // pano_dir = R^T * ray
                double dx = R[0] * rx + R[3] * ry + R[6] * rz, dy = R[1] * rx + R[4] * ry + R[7] * rz, dz = R[2] * rx + R[5] * ry + R[8] * rz;
                Assert.That(dx, Is.EqualTo(e.X).Within(1e-9), view.Name); Assert.That(dy, Is.EqualTo(e.Y).Within(1e-9), view.Name);
                Assert.That(dz, Is.EqualTo(e.Z).Within(1e-9), view.Name);
            }
        }
    }

    /// <summary>A panorama whose every pixel stores its own unit direction as (d + 1) / 2: continuous across the seam, so a sampler
    /// that does not wrap shows up as wrong directions on views that straddle it.</summary>
    static float[] DirectionPanorama()
    {
        var rgb = new float[W * H * 3];
        for (int j = 0; j < H; j++)
            for (int i = 0; i < W; i++)
            {
                var d = EquirectProjection.DirectionForPixel(i + 0.5, j + 0.5, W, H);
                int o = (j * W + i) * 3;
                rgb[o] = (float)((d.X + 1) / 2); rgb[o + 1] = (float)((d.Y + 1) / 2); rgb[o + 2] = (float)((d.Z + 1) / 2);
            }
        return rgb;
    }

    [Test]
    public void RenderedViewsSeeTheDirectionsTheyShould()
    {
        const int size = 64; const double fov = 90;
        var pano = DirectionPanorama();
        double f = EquirectProjection.FocalForFov(size, fov);
        foreach (var view in EquirectProjection.Ring10)
        {
            var img = EquirectProjection.RenderView(pano, W, H, EquirectProjection.ViewRotation(view.HeadingDeg, view.PitchDeg), size, fov);
            double worst = 0;
            for (int j = 0; j < size; j++)
                for (int i = 0; i < size; i++)
                {
                    double rx = (i + 0.5 - 0.5 * size) / f, ry = (j + 0.5 - 0.5 * size) / f;
                    var e = PanoDirection(view.HeadingDeg, view.PitchDeg, rx, ry, 1);
                    double n = Math.Sqrt(e.X * e.X + e.Y * e.Y + e.Z * e.Z);
                    int o = (j * size + i) * 3;
                    worst = Math.Max(worst, Math.Abs(img[o] * 2 - 1 - e.X / n));
                    worst = Math.Max(worst, Math.Abs(img[o + 1] * 2 - 1 - e.Y / n));
                    worst = Math.Max(worst, Math.Abs(img[o + 2] * 2 - 1 - e.Z / n));
                }
            // The panorama is 0.5 degrees a pixel and a view pixel is ~1.4: bilinear error stays far under this; a wrong sign, a
            // transposed axis or a seam that does not wrap is off by 0.1 to 2.
            Assert.That(worst, Is.LessThan(0.02), $"view {view.Name} (heading {view.HeadingDeg}, pitch {view.PitchDeg})");
        }
    }

    /// <summary>
    /// The seam: rolling a panorama by half its width turns every direction 180 degrees about the vertical, so the view at heading h of the
    /// rolled panorama must equal the view at heading h + 180 of the original. White noise (no smoothness across the seam) means a sampler
    /// (ODD view size: a pixel centre then sits exactly on the centre ray, which for heading 180 IS the seam - with an even size every centre is
    /// a pixel or more away from it and a clamp never differs from a wrap) that clamps instead of wrapping gives the seam views different values - the direction panorama above is continuous across the seam, so
    /// a clamp is only a pixel of angle wrong there and slips under its tolerance.
    /// </summary>
    [Test]
    public void RollingThePanoramaIsTheSameAsTurningTheView()
    {
        var rng = new Random(3);
        var pano = new float[W * H * 3];
        for (int i = 0; i < pano.Length; i++) pano[i] = (float)rng.NextDouble();
        var rolled = new float[pano.Length];
        for (int j = 0; j < H; j++)
            for (int i = 0; i < W; i++)
                Array.Copy(pano, (j * W + (i + W / 2) % W) * 3, rolled, (j * W + i) * 3, 3);
        foreach (var view in EquirectProjection.Ring10)
        {
            var a = EquirectProjection.RenderView(pano, W, H, EquirectProjection.ViewRotation(view.HeadingDeg, view.PitchDeg), 49, 90);
            var b = EquirectProjection.RenderView(rolled, W, H, EquirectProjection.ViewRotation(view.HeadingDeg - 180, view.PitchDeg), 49, 90);
            double worst = 0;
            for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
            Assert.That(worst, Is.LessThan(1e-4), $"view {view.Name}");
        }
    }
}
