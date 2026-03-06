using SpawnScene.Models;

namespace SpawnScene.Formats;

/// <summary>
/// Parses .splat files — a compact binary format for Gaussian Splat data.
/// 
/// Format: 32 bytes per Gaussian (no header):
///   [0..11]  float32 x3  — position (x, y, z)
///   [12..14] uint8 x3    — scale (quantized log-scale)
///   [15]     uint8        — opacity (quantized)
///   [16..19] int8 x4     — rotation quaternion (quantized, w,x,y,z)
///   [20..23] uint8 x4    — RGBA color
///   [24..31] reserved/padding (varies by implementation)
///
/// Some implementations use a simpler 32-byte layout:
///   [0..11]  float32 x3  — position
///   [12..23] float32 x3  — scale (log-space)
///   [24..27] uint8 x4    — color RGBA
///   [28..31] uint8 x4    — rotation (quantized)
///
/// We detect the format heuristically.
/// </summary>
public static class SplatParser
{
    /// <summary>
    /// Parse a .splat file from raw bytes.
    /// Uses the common 32-byte-per-splat format.
    /// </summary>
    public static GaussianScene Parse(byte[] data)
    {
        // Standard .splat format: 32 bytes per splat
        const int bytesPerSplat = 32;

        if (data.Length < bytesPerSplat)
            throw new FormatException("File too small to contain any splats");

        int count = data.Length / bytesPerSplat;
        var gaussians = new Gaussian3D[count];

        for (int i = 0; i < count; i++)
        {
            int offset = i * bytesPerSplat;

            // Position: float32 x3 at bytes [0..11]
            float x = BitConverter.ToSingle(data, offset + 0);
            float y = BitConverter.ToSingle(data, offset + 4);
            float z = BitConverter.ToSingle(data, offset + 8);

            // Scale: float32 x3 at bytes [12..23] (log-space)
            float sx = BitConverter.ToSingle(data, offset + 12);
            float sy = BitConverter.ToSingle(data, offset + 16);
            float sz = BitConverter.ToSingle(data, offset + 20);

            // Color: uint8 RGBA at bytes [24..27]
            float r = data[offset + 24] / 255.0f;
            float g = data[offset + 25] / 255.0f;
            float b = data[offset + 26] / 255.0f;
            float a = data[offset + 27] / 255.0f;

            // Rotation: int8 quaternion at bytes [28..31] (quantized to -128..127 → -1..1)
            float rw = ((sbyte)data[offset + 28]) / 128.0f;
            float rx = ((sbyte)data[offset + 29]) / 128.0f;
            float ry = ((sbyte)data[offset + 30]) / 128.0f;
            float rz = ((sbyte)data[offset + 31]) / 128.0f;

            // Normalize quaternion
            float qLen = MathF.Sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
            if (qLen > 0.0001f) { rw /= qLen; rx /= qLen; ry /= qLen; rz /= qLen; }
            else { rw = 1; rx = 0; ry = 0; rz = 0; }

            // Convert color from linear RGB to SH DC coefficients
            const float C0 = 0.28209479f;
            gaussians[i] = new Gaussian3D
            {
                X = x,
                Y = y,
                Z = z,
                ScaleX = sx,
                ScaleY = sy,
                ScaleZ = sz, // Already in log-space
                RotW = rw,
                RotX = rx,
                RotY = ry,
                RotZ = rz,
                OpacityLogit = a > 0.001f && a < 0.999f
                    ? MathF.Log(a / (1.0f - a))
                    : (a >= 0.999f ? 5.0f : -5.0f),
                SH_DC_R = (r - 0.5f) / C0,
                SH_DC_G = (g - 0.5f) / C0,
                SH_DC_B = (b - 0.5f) / C0,
            };
        }

        var scene = new GaussianScene
        {
            Gaussians = gaussians,
            ShDegree = 0,
        };
        scene.ComputeBounds();
        return scene;
    }
}
