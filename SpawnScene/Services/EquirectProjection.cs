namespace SpawnScene.Services;

/// <summary>
/// 360-degree (equirectangular) frames as pinhole views (TJ 2026-10-15: "video source, standard and 360 degree"). A 360 clip or
/// photo is one panorama per frame; the rest of the pipeline (matching, SfM, training) wants pinhole cameras, so each panorama is
/// resampled into a ring of overlapping pinhole views that all share the panorama's centre.
/// <para>
/// CONVENTION - COLMAP's EQUIRECTANGULAR camera model (the Big Sur reference reconstruction, cameras.bin model 17, no parameters):
/// the panorama's frame is x right, y DOWN, z forward; a direction d maps to longitude <c>atan2(x, z)</c> and latitude
/// <c>asin(y / |d|)</c>, and to the pixel <c>u = W (0.5 + lon / 2pi)</c>, <c>v = H (0.5 + lat / pi)</c> (pixel centres at +0.5).
/// So the image centre is straight ahead, u = 0.75 W is +x (right), v = 0 is straight UP (-y), v = H straight down. The view
/// rotation is <c>view_from_pano</c> (rows = the view's x right, y down, z forward axes in the panorama's frame).
/// </para>
/// </summary>
public static class EquirectProjection
{
    /// <summary>One pinhole view of a panorama: where it looks, in degrees (heading: 0 = ahead, +90 = to the right; pitch: +90 = up).</summary>
    public readonly record struct ViewSpec(string Name, double HeadingDeg, double PitchDeg);

    /// <summary>Six 90-degree faces: four around the horizon, up and down (the minimum that covers the sphere).</summary>
    public static IReadOnlyList<ViewSpec> Cube6 { get; } = new[]
    {
        new ViewSpec("h000", 0, 0), new ViewSpec("h090", 90, 0), new ViewSpec("h180", 180, 0), new ViewSpec("h270", 270, 0),
        new ViewSpec("up", 0, 90), new ViewSpec("down", 0, -90),
    };

    /// <summary>
    /// Ten views: eight around the horizon 45 degrees apart (a 90-degree field overlaps its neighbours by half, which is what
    /// feature matching between views of one frame needs) plus up and down.
    /// </summary>
    public static IReadOnlyList<ViewSpec> Ring10 { get; } = BuildRing10();

    static IReadOnlyList<ViewSpec> BuildRing10()
    {
        var v = new List<ViewSpec>();
        for (int k = 0; k < 8; k++) v.Add(new ViewSpec($"h{k * 45:000}", k * 45, 0));
        v.Add(new ViewSpec("up", 0, 90));
        v.Add(new ViewSpec("down", 0, -90));
        return v;
    }

    /// <summary>The panorama pixel (centre-based coordinates: the first pixel's centre is 0.5) a direction in the panorama's frame lands on.</summary>
    public static (double U, double V) PixelForDirection(double x, double y, double z, int width, int height)
    {
        double r = Math.Sqrt(x * x + y * y + z * z);
        double lon = Math.Atan2(x, z);
        double lat = Math.Asin(Math.Clamp(y / r, -1.0, 1.0));
        return (width * (0.5 + lon / (2 * Math.PI)), height * (0.5 + lat / Math.PI));
    }

    /// <summary>The unit direction (panorama frame) of the panorama pixel (u, v) - the inverse of <see cref="PixelForDirection"/>.</summary>
    public static (double X, double Y, double Z) DirectionForPixel(double u, double v, int width, int height)
    {
        double lon = (u / width - 0.5) * 2 * Math.PI;
        double lat = (v / height - 0.5) * Math.PI;
        double c = Math.Cos(lat);
        return (c * Math.Sin(lon), Math.Sin(lat), c * Math.Cos(lon));
    }

    /// <summary>
    /// <c>view_from_pano</c> for a view looking toward (heading, pitch), as row-major 3x3: row 0 = the view's right axis (horizontal,
    /// whatever the pitch, so a pole view needs no special case), row 1 = down, row 2 = forward - all in the panorama's frame.
    /// </summary>
    public static double[] ViewRotation(double headingDeg, double pitchDeg)
    {
        double h = headingDeg * Math.PI / 180.0, p = pitchDeg * Math.PI / 180.0;
        double[] fwd = { Math.Cos(p) * Math.Sin(h), -Math.Sin(p), Math.Cos(p) * Math.Cos(h) };
        double[] right = { Math.Cos(h), 0, -Math.Sin(h) };
        // down = forward x right
        double[] down =
        {
            fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0],
        };
        return new[] { right[0], right[1], right[2], down[0], down[1], down[2], fwd[0], fwd[1], fwd[2] };
    }

    /// <summary>The pinhole focal length (pixels) of a square view of <paramref name="size"/> pixels with the given field of view across.</summary>
    public static double FocalForFov(int size, double fovDeg) => 0.5 * size / Math.Tan(0.5 * fovDeg * Math.PI / 180.0);

    /// <summary>
    /// CPU reference: the view (row-major RGB floats, size x size x 3) of an interleaved-RGB panorama, bilinear, wrapping across the
    /// longitude seam and clamping at the poles. The GPU version must match this.
    /// </summary>
    public static float[] RenderView(float[] rgb, int width, int height, double[] viewFromPano, int size, double fovDeg)
    {
        double f = FocalForFov(size, fovDeg);
        var outPx = new float[size * size * 3];
        for (int j = 0; j < size; j++)
            for (int i = 0; i < size; i++)
            {
                double rx = (i + 0.5 - 0.5 * size) / f, ry = (j + 0.5 - 0.5 * size) / f;   // ray (rx, ry, 1) in the view's frame
                // pano_dir = view_from_pano^T * ray
                double dx = viewFromPano[0] * rx + viewFromPano[3] * ry + viewFromPano[6];
                double dy = viewFromPano[1] * rx + viewFromPano[4] * ry + viewFromPano[7];
                double dz = viewFromPano[2] * rx + viewFromPano[5] * ry + viewFromPano[8];
                var (u, v) = PixelForDirection(dx, dy, dz, width, height);
                Sample(rgb, width, height, u - 0.5, v - 0.5, outPx, (j * size + i) * 3);
            }
        return outPx;
    }

    /// <summary>Bilinear sample at the continuous pixel coordinate (x, y) (pixel centres at integers): x wraps, y clamps.</summary>
    static void Sample(float[] rgb, int w, int h, double x, double y, float[] dst, int o)
    {
        double fx = Math.Floor(x), fy = Math.Floor(y);
        double ax = x - fx, ay = y - fy;
        int x0 = (int)fx, y0 = (int)fy;
        int xa = ((x0 % w) + w) % w, xb = (((x0 + 1) % w) + w) % w;
        int ya = Math.Clamp(y0, 0, h - 1), yb = Math.Clamp(y0 + 1, 0, h - 1);
        for (int c = 0; c < 3; c++)
        {
            double top = rgb[(ya * w + xa) * 3 + c] * (1 - ax) + rgb[(ya * w + xb) * 3 + c] * ax;
            double bottom = rgb[(yb * w + xa) * 3 + c] * (1 - ax) + rgb[(yb * w + xb) * 3 + c] * ax;
            dst[o + c] = (float)(top * (1 - ay) + bottom * ay);
        }
    }
}
