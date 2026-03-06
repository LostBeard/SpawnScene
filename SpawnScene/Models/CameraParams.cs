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

    /// <summary>Aspect ratio.</summary>
    public float AspectRatio => (float)Width / Height;

    /// <summary>
    /// Build the 4x4 view matrix (world → camera space).
    /// </summary>
    public Matrix4x4 ViewMatrix => Matrix4x4.CreateLookAt(Position, Position + Forward, Up);

    /// <summary>
    /// Build the 4x4 projection matrix (camera → clip space).
    /// Uses a pinhole camera model matching the intrinsics.
    /// </summary>
    public Matrix4x4 ProjectionMatrix
    {
        get
        {
            // OpenGL-style projection from intrinsics
            float l = -CenterX * Near / FocalX;
            float r = (Width - CenterX) * Near / FocalX;
            float b = -(Height - CenterY) * Near / FocalY;
            float t = CenterY * Near / FocalY;

            return new Matrix4x4(
                2 * Near / (r - l), 0, (r + l) / (r - l), 0,
                0, 2 * Near / (t - b), (t + b) / (t - b), 0,
                0, 0, -(Far + Near) / (Far - Near), -2 * Far * Near / (Far - Near),
                0, 0, -1, 0
            );
        }
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
}
