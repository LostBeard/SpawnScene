using System.Numerics;
using System.Runtime.InteropServices;

namespace SpawnScene.Models;

/// <summary>
/// A single 3D Gaussian splat with all learnable parameters.
/// Uses SH degree 0 (DC only) initially for simplicity.
/// Layout is optimized for GPU buffer transfers.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct Gaussian3D
{
    // Position (mean) in world space
    public float X;
    public float Y;
    public float Z;

    // Scale (log-space, exponentiated before use)
    public float ScaleX;
    public float ScaleY;
    public float ScaleZ;

    // Rotation as quaternion (normalized)
    public float RotW;
    public float RotX;
    public float RotY;
    public float RotZ;

    // Opacity (sigmoid-space, sigmoid applied before use)
    public float OpacityLogit;

    // Spherical harmonics DC component (degree 0) — base RGB color
    public float SH_DC_R;
    public float SH_DC_G;
    public float SH_DC_B;

    /// <summary>Size in floats for GPU buffer sizing.</summary>
    public const int FloatCount = 14;

    /// <summary>Get the world-space position.</summary>
    public readonly Vector3 Position => new(X, Y, Z);

    /// <summary>Get the activated opacity (sigmoid).</summary>
    public readonly float Opacity => 1.0f / (1.0f + MathF.Exp(-OpacityLogit));

    /// <summary>Get the activated scale (exp).</summary>
    public readonly Vector3 Scale => new(MathF.Exp(ScaleX), MathF.Exp(ScaleY), MathF.Exp(ScaleZ));

    /// <summary>Get the base color from SH DC coefficients (C0 = 0.28209479f).</summary>
    public readonly Vector3 BaseColor
    {
        get
        {
            const float C0 = 0.28209479f; // 0.5 * sqrt(1/pi)
            return new Vector3(
                0.5f + C0 * SH_DC_R,
                0.5f + C0 * SH_DC_G,
                0.5f + C0 * SH_DC_B
            );
        }
    }

    /// <summary>Get the normalized rotation quaternion.</summary>
    public readonly Quaternion Rotation
    {
        get
        {
            var q = new Quaternion(RotX, RotY, RotZ, RotW);
            return Quaternion.Normalize(q);
        }
    }

    /// <summary>
    /// Compute the 3x3 covariance matrix from scale and rotation.
    /// Cov = R * S * S^T * R^T
    /// </summary>
    public readonly Matrix4x4 CovarianceMatrix
    {
        get
        {
            var s = Scale;
            var q = Rotation;
            var rotMatrix = Matrix4x4.CreateFromQuaternion(q);

            // S is diagonal scale matrix
            // M = R * S
            var m00 = rotMatrix.M11 * s.X;
            var m01 = rotMatrix.M12 * s.Y;
            var m02 = rotMatrix.M13 * s.Z;
            var m10 = rotMatrix.M21 * s.X;
            var m11 = rotMatrix.M22 * s.Y;
            var m12 = rotMatrix.M23 * s.Z;
            var m20 = rotMatrix.M31 * s.X;
            var m21 = rotMatrix.M32 * s.Y;
            var m22 = rotMatrix.M33 * s.Z;

            // Cov = M * M^T (symmetric 3x3, stored in 4x4)
            return new Matrix4x4(
                m00 * m00 + m01 * m01 + m02 * m02,
                m00 * m10 + m01 * m11 + m02 * m12,
                m00 * m20 + m01 * m21 + m02 * m22,
                0,
                m10 * m00 + m11 * m01 + m12 * m02,
                m10 * m10 + m11 * m11 + m12 * m12,
                m10 * m20 + m11 * m21 + m12 * m22,
                0,
                m20 * m00 + m21 * m01 + m22 * m02,
                m20 * m10 + m21 * m11 + m22 * m12,
                m20 * m20 + m21 * m21 + m22 * m22,
                0,
                0, 0, 0, 1
            );
        }
    }

    /// <summary>Create a Gaussian from explicit parameters.</summary>
    public static Gaussian3D Create(Vector3 position, Vector3 scale, Quaternion rotation, float opacity, Vector3 color)
    {
        const float C0 = 0.28209479f;
        return new Gaussian3D
        {
            X = position.X,
            Y = position.Y,
            Z = position.Z,
            ScaleX = MathF.Log(scale.X),
            ScaleY = MathF.Log(scale.Y),
            ScaleZ = MathF.Log(scale.Z),
            RotW = rotation.W,
            RotX = rotation.X,
            RotY = rotation.Y,
            RotZ = rotation.Z,
            OpacityLogit = MathF.Log(opacity / (1.0f - opacity)),
            SH_DC_R = (color.X - 0.5f) / C0,
            SH_DC_G = (color.Y - 0.5f) / C0,
            SH_DC_B = (color.Z - 0.5f) / C0,
        };
    }
}
