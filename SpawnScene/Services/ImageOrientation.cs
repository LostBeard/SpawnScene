using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Turn a posed photograph upright before it reaches the depth model.
///
/// Monocular depth networks carry a strong gravity prior - they are trained almost entirely on
/// upright photographs, and a scene lying on its side is out of distribution. TempleRing is the
/// case that forced this: in all 47 of its calibration entries world-up projects to within about
/// two degrees of image-RIGHT, so every one of those photos shows the temple on its side. The
/// calibration is perfectly self-consistent with the rotated pixels, so nothing downstream is
/// wrong - the projection, the sort and the score all agree - but the depth that everything is
/// initialised from was estimated from a sideways picture.
///
/// The correction is a quarter turn of the pixels plus the matching change of camera
/// parameterisation, so the rotated image and the rotated camera describe the same rays. A world
/// point that landed on a given piece of the subject still lands on that same piece.
///
/// Derivation for one counter-clockwise quarter turn, pixel-centre coordinates:
///
///     (u', v') = (v, (W-1) - u)                        pixels move right-edge to top
///     u  = fx (x/z) + cx,  v = fy (-y/z) + cy          the original pinhole
///     right' = -up,  up' = right,  forward' = forward  the camera turns about its own axis
///     fx' = fy,  fy' = fx,  cx' = cy,  cy' = (W-1) - cx
///     W'  = H,   H'  = W
///
/// Substituting the new basis into the new pinhole reproduces (u', v') exactly, which is what
/// <c>ImageOrientationTests</c> checks by projecting world points through both and comparing.
///
/// <para>
/// "Upright" here means world +Y. That is this application's convention everywhere else, and for
/// a dataset it comes from the calibration. For a scene whose poses were recovered from video the
/// pose source defines up, so the same assumption holds - but if a pose source ever emits a
/// different gravity axis this is the one place that has to learn about it.
/// </para>
/// </summary>
public static class ImageOrientation
{
    /// <summary>World up. See the note on the class about where this assumption comes from.</summary>
    public static readonly Vector3 WorldUp = Vector3.UnitY;

    /// <summary>
    /// How many counter-clockwise quarter turns bring <paramref name="cam"/>'s image upright,
    /// 0 to 3. Zero when the camera is already within 45 degrees of level, including when it is
    /// looking straight up or down (where "upright" is meaningless and turning the picture would
    /// be arbitrary).
    /// </summary>
    public static int QuarterTurnsToUpright(CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out _, out _);

        // Where world-up points in image space. Image x is right, image y is DOWN, so the
        // vertical component flips sign relative to the camera's up vector.
        float ix = Vector3.Dot(right, WorldUp);
        float iy = -Vector3.Dot(up, WorldUp);

        // Looking near-vertically: world-up projects to almost nothing and its direction is
        // noise. Leave the picture alone rather than turning it on the strength of rounding.
        if (ix * ix + iy * iy < 1e-4f) return 0;

        // Upright means world-up points UP the image, i.e. iy negative, ix about zero.
        // Measure the angle from image-up and snap to the nearest quarter turn.
        float angleFromUp = MathF.Atan2(ix, -iy);           // 0 when already upright
        int turns = (int)MathF.Round(angleFromUp / (MathF.PI / 2f));
        return ((turns % 4) + 4) % 4;
    }

    /// <summary>
    /// Rotate RGBA pixels by <paramref name="turns"/> counter-clockwise quarter turns.
    /// Returns the source array unchanged when there is nothing to do.
    /// </summary>
    public static byte[] RotateRgba(byte[] src, int width, int height, int turns)
    {
        turns = ((turns % 4) + 4) % 4;
        if (turns == 0) return src;

        // One turn at a time, swapping the dimensions each time. Three more index derivations
        // for the 180 and 270 cases would be three more chances to get one wrong.
        byte[] cur = src;
        int w = width, h = height;
        for (int t = 0; t < turns; t++)
        {
            var dst = new byte[cur.Length];
            RotateOnceRgba(cur, dst, w, h);
            cur = dst;
            (w, h) = (h, w);
        }
        return cur;
    }

    /// <summary>One counter-clockwise quarter turn: dst is <paramref name="h"/> x <paramref name="w"/>.</summary>
    static void RotateOnceRgba(byte[] src, byte[] dst, int w, int h)
    {
        int dstW = h;
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                int di = j;
                int dj = w - 1 - i;
                int s = (j * w + i) * 4;
                int d = (dj * dstW + di) * 4;
                dst[d + 0] = src[s + 0];
                dst[d + 1] = src[s + 1];
                dst[d + 2] = src[s + 2];
                dst[d + 3] = src[s + 3];
            }
        }
    }

    /// <summary>
    /// The camera that describes the same rays as <paramref name="cam"/> once its image has been
    /// turned <paramref name="turns"/> counter-clockwise quarter turns.
    /// </summary>
    public static CameraParams Rotate(CameraParams cam, int turns)
    {
        turns = ((turns % 4) + 4) % 4;
        var result = Clone(cam);
        for (int t = 0; t < turns; t++) result = RotateOnce(result);
        return result;
    }

    static CameraParams RotateOnce(CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out _, out var fwd, out _);
        var next = Clone(cam);

        // right' = -up, up' = right. CameraParams derives Right from Forward and Up, and
        // cross(forward, right) == -up for an orthonormal basis, so setting Up alone is enough.
        next.Up = right;
        next.Forward = fwd;

        next.Width = cam.Height;
        next.Height = cam.Width;
        next.FocalX = cam.FocalY;
        next.FocalY = cam.FocalX;
        next.CenterX = cam.CenterY;
        next.CenterY = (cam.Width - 1) - cam.CenterX;
        return next;
    }

    static CameraParams Clone(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height,
        FocalX = c.FocalX, FocalY = c.FocalY,
        CenterX = c.CenterX, CenterY = c.CenterY,
        Near = c.Near, Far = c.Far,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };
}
