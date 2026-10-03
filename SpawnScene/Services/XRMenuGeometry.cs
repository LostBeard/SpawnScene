using System.Numerics;

namespace SpawnScene.Services;

/// <summary>Geometry of the in-headset menu (<see cref="XRMenu"/>), kept free of GameUI so it can be tested.</summary>
public static class XRMenuGeometry
{
    public const float PanelWidth = 560, PanelHeight = 580;
    /// <summary>Metres per panel pixel: 0.56 x 0.58 m.</summary>
    public const float WorldScale = 0.001f;

    /// <summary>
    /// The panel's world transform (row vectors) 0.65 m ahead of the head, level, 0.15 m below eye height, facing it:
    /// local X = the viewer's right, local Y = up.
    /// </summary>
    public static Matrix4x4 PlaceInFront(Vector3 headPosition, Quaternion headOrientation)
    {
        var f = Vector3.Transform(-Vector3.UnitZ, headOrientation);
        f.Y = 0;
        f = f.LengthSquared() < 1e-8f ? -Vector3.UnitZ : Vector3.Normalize(f);
        var right = Vector3.Cross(f, Vector3.UnitY);
        var back = -f;   // local +Z points at the viewer
        var pos = headPosition + f * 0.65f - Vector3.UnitY * 0.15f;
        return new Matrix4x4(
            right.X, right.Y, right.Z, 0,
            0, 1, 0, 0,
            back.X, back.Y, back.Z, 0,
            pos.X, pos.Y, pos.Z, 1);
    }

    /// <summary>Where a ray meets the panel, in panel pixels (x right, y down), or null if it misses.</summary>
    public static Vector2? RayToPanelPixel(Matrix4x4 model, Vector3 rayOrigin, Vector3 rayDirection)
    {
        var drawn = Matrix4x4.CreateScale(PanelWidth * WorldScale, PanelHeight * WorldScale, 1f) * model;
        if (!Matrix4x4.Invert(drawn, out var inv)) return null;
        var o = Vector3.Transform(rayOrigin, inv);
        var d = Vector3.TransformNormal(rayDirection, inv);
        if (MathF.Abs(d.Z) < 1e-8f) return null;
        float t = -o.Z / d.Z;
        if (t <= 0) return null;
        var hit = o + d * t;   // local: x, y in [-0.5, 0.5] on the panel
        if (hit.X < -0.5f || hit.X > 0.5f || hit.Y < -0.5f || hit.Y > 0.5f) return null;
        return new Vector2((hit.X + 0.5f) * PanelWidth, (0.5f - hit.Y) * PanelHeight);
    }
}
