using System.Numerics;

namespace SpawnScene.Models;

/// <summary>
/// Camera parameters for rendering and reconstruction.
/// Combines intrinsic (lens) and extrinsic (pose) parameters.
/// </summary>
public class CameraParams
{
    // --- Intrinsics ---

    /// <summary>Focal length in pixels (x-axis).</summary>
    public float FocalX { get; set; }

    /// <summary>Focal length in pixels (y-axis).</summary>
    public float FocalY { get; set; }

    /// <summary>Principal point X (typically image width / 2).</summary>
    public float CenterX { get; set; }

    /// <summary>Principal point Y (typically image height / 2).</summary>
    public float CenterY { get; set; }

    /// <summary>Image width in pixels.</summary>
    public int Width { get; set; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; set; }

    /// <summary>Near clipping plane.</summary>
    public float Near { get; set; } = 0.1f;

    /// <summary>Far clipping plane.</summary>
    public float Far { get; set; } = 1000.0f;

    // --- Extrinsics ---

    /// <summary>Camera position in world space.</summary>
    public Vector3 Position { get; set; } = Vector3.Zero;

    /// <summary>Camera forward direction (normalized).</summary>
    public Vector3 Forward { get; set; } = -Vector3.UnitZ;

    /// <summary>Camera up direction (normalized).</summary>
    public Vector3 Up { get; set; } = Vector3.UnitY;

    /// <summary>Camera right direction (derived).</summary>
    public Vector3 Right => Vector3.Normalize(Vector3.Cross(Forward, Up));

    /// <summary>
    /// The same camera looking at a resized version of its image.
    ///
    /// Training cannot run at capture resolution - a 3120x4160 phone photo is 13 megapixels, and
    /// 35 of them as float RGB targets is 5.5 GB - so the optimiser works on a downscaled copy.
    /// The intrinsics have to come with it or every splat projects to the wrong place.
    ///
    /// Pixel coordinates here are continuous (pixel p spans [p, p+1)), so a resize by factor s
    /// maps u to u*s, which scales the focal length AND the principal point by s. Scaling the
    /// focal but not the principal point is the classic version of this bug: it looks right at
    /// the image centre and drifts toward the edges.
    ///
    /// The axes scale independently, matching a stretch resize. Callers should preserve aspect
    /// ratio (see <see cref="FitWithin"/>) unless they genuinely intend to stretch the pixels.
    /// </summary>
    public CameraParams ScaledTo(int width, int height)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), $"{width}x{height} is not a viewport");

        float sx = (float)width / Width;
        float sy = (float)height / Height;
        return new CameraParams
        {
            Width = width,
            Height = height,
            FocalX = FocalX * sx,
            FocalY = FocalY * sy,
            CenterX = CenterX * sx,
            CenterY = CenterY * sy,
            Near = Near,
            Far = Far,
            Position = Position,
            Forward = Forward,
            Up = Up,
        };
    }

    /// <summary>
    /// The largest size with this camera's aspect ratio that fits inside
    /// <paramref name="maxDimension"/>, rounded to even numbers so a 16px tile grid divides it
    /// predictably. Never upscales.
    /// </summary>
    public (int Width, int Height) FitWithin(int maxDimension)
    {
        int longest = Math.Max(Width, Height);
        if (longest <= maxDimension) return (Width, Height);

        float s = (float)maxDimension / longest;
        int w = Math.Max(2, (int)MathF.Round(Width * s / 2f) * 2);
        int h = Math.Max(2, (int)MathF.Round(Height * s / 2f) * 2);
        return (w, h);
    }

    /// <summary>Aspect ratio.</summary>
    public float AspectRatio => (float)Width / Height;

    /// <summary>
    /// Build the 4x4 view matrix (world → camera space).
    /// </summary>
    public Matrix4x4 ViewMatrix => Matrix4x4.CreateLookAt(Position, Position + Forward, Up);

    /// <summary>
    /// Build the 4x4 projection matrix (camera → clip space) from the real intrinsics.
    ///
    /// Row-vector convention (<c>clip = v * M</c>) to match <see cref="ViewMatrix"/> and
    /// System.Numerics, and WebGPU clip depth (z in [0,1], not OpenGL's [-1,1]).
    ///
    /// Unlike a symmetric fov/aspect projection this carries the PRINCIPAL POINT and allows
    /// fx != fy, so it agrees with the pinhole model the splat covariance Jacobian uses.
    /// TempleRing is the case that needs it: fx=1520.4, fy=1525.9, and a principal point of
    /// (302.32, 246.87) against a 640x480 image - 18px off centre.
    /// </summary>
    public Matrix4x4 ProjectionMatrix
        => CreateWebGpuProjection(FocalX, FocalY, CenterX, CenterY, Width, Height, Near, Far);

    /// <summary>
    /// Off-axis pinhole projection for WebGPU, row-vector convention.
    ///
    /// Reduces EXACTLY to the symmetric fov/aspect form when the principal point is centred
    /// and fx == fy - that equivalence is pinned by
    /// <c>CameraProjectionTests.CentredIntrinsics_MatchTheSymmetricPerspectiveItReplaces</c>.
    /// </summary>
    public static Matrix4x4 CreateWebGpuProjection(
        float focalX, float focalY, float centerX, float centerY,
        int width, int height, float near, float far)
    {
        // Eye space is right-handed looking down -Z, so depth = -z_eye and clip.w = -z_eye.
        //   pixel_x = fx * (x_eye / depth) + cx            ndc_x = 2*pixel_x/W - 1
        //   pixel_y = fy * (-y_eye / depth) + cy           ndc_y = 1 - 2*pixel_y/H
        // Multiplying through by depth gives the clip-space rows below.
        float m11 = 2f * focalX / width;
        float m31 = 1f - 2f * centerX / width;

        float m22 = 2f * focalY / height;
        float m32 = 2f * centerY / height - 1f;

        float rangeInv = 1f / (near - far); // negative
        return new Matrix4x4(
            m11, 0, 0, 0,
            0, m22, 0, 0,
            m31, m32, far * rangeInv, -1,   // -1 for right-handed → w = -z_eye
            0, 0, near * far * rangeInv, 0
        );
    }

    /// <summary>
    /// Recover pinhole intrinsics from a projection matrix in the convention
    /// <see cref="CreateWebGpuProjection"/> produces. Exact inverse of it.
    ///
    /// This is what makes XR correct. A VR headset's per-eye frustum is ASYMMETRIC by design -
    /// the principal point sits off centre so the two eyes converge - so reading a focal length
    /// as <c>|M11| * width / 2</c> and assuming a centred principal point silently discards the
    /// stereo offset. The splat covariance Jacobian needs the real fx/fy/cx/cy of whatever
    /// frustum WebXR hands us, per eye, or every splat ellipse is computed for the wrong point.
    /// </summary>
    public static void ExtractIntrinsics(
        Matrix4x4 proj, int width, int height,
        out float focalX, out float focalY, out float centerX, out float centerY)
    {
        focalX = proj.M11 * width * 0.5f;
        focalY = proj.M22 * height * 0.5f;
        centerX = width * (1f - proj.M31) * 0.5f;
        centerY = height * (1f + proj.M32) * 0.5f;
    }

    /// <summary>
    /// Create default camera parameters for a given image size.
    /// Uses a reasonable default focal length (equivalent to ~50mm lens).
    /// </summary>
    public static CameraParams CreateDefault(int width, int height)
    {
        float focalLength = MathF.Max(width, height) * 1.2f;
        return new CameraParams
        {
            Width = width,
            Height = height,
            FocalX = focalLength,
            FocalY = focalLength,
            CenterX = width / 2.0f,
            CenterY = height / 2.0f,
            Position = new Vector3(0, 0, 3),
            Forward = -Vector3.UnitZ,
            Up = Vector3.UnitY,
        };
    }

    /// <summary>
    /// Create camera parameters from EXIF focal length data.
    /// Priority: FocalLength35mm (exact) → phone camera estimate → default heuristic.
    /// </summary>
    public static CameraParams CreateFromExif(int width, int height, ExifReader.ExifFocalLength? exif)
    {
        float focalLength;

        if (exif?.FocalLength35mm is > 0)
        {
            // Best path: 35mm equivalent → pixel focal length
            focalLength = ExifReader.FocalLength35mmToPixels(exif.FocalLength35mm.Value, width, height);
        }
        else if (exif?.FocalLengthMm is > 0 and < 10f)
        {
            // Phone camera: focal length < 10mm is almost certainly a smartphone sensor.
            // Typical phone sensors are 4-6mm wide → crop factor ~6-9x, median ~7x.
            // Estimate 35mm equiv: f_mm * 7, then convert to pixels.
            float estimated35mm = exif.FocalLengthMm.Value * 7f;
            focalLength = ExifReader.FocalLength35mmToPixels(estimated35mm, width, height);
        }
        else
        {
            // No useful EXIF: fall back to default heuristic (~43mm equiv)
            focalLength = MathF.Max(width, height) * 1.2f;
        }

        return new CameraParams
        {
            Width = width,
            Height = height,
            FocalX = focalLength,
            FocalY = focalLength,
            CenterX = width / 2.0f,
            CenterY = height / 2.0f,
            Position = new Vector3(0, 0, 3),
            Forward = -Vector3.UnitZ,
            Up = Vector3.UnitY,
        };
    }
}
