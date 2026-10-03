using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Thumbstick locomotion for a VR session, applied to the room-to-scene transform from <see cref="XRSceneAlignment"/>
/// (row-vector convention: pScene = pRoom * sceneFromRoom). The comfort defaults of most VR apps: the LEFT stick moves
/// smoothly along the head's horizontal facing (strafing sideways), the RIGHT stick snap-turns in fixed steps about the
/// head (no smooth yaw, the main source of VR sickness) and, pushed forward or back, rises or sinks - splat scenes have
/// no floor to walk on, so up and down is the only way to reach a ceiling or look down on a scene.
/// </summary>
public sealed class XRLocomotion
{
    /// <summary>Stick deflection below this is ignored (sticks rest a little off centre).</summary>
    public const float Deadzone = 0.15f;
    /// <summary>A snap-turn fires when the right stick passes this sideways ...</summary>
    public const float SnapFire = 0.7f;
    /// <summary>... and re-arms once it is back under this.</summary>
    public const float SnapRearm = 0.3f;

    /// <summary>Degrees per snap-turn.</summary>
    public float SnapTurnDegrees { get; set; } = 30f;

    bool _snapArmed = true;

    /// <summary>Forget a held snap-turn (a new session).</summary>
    public void Reset() => _snapArmed = true;

    /// <summary>
    /// One frame of locomotion. <paramref name="left"/> and <paramref name="right"/> are xr-standard thumbsticks (x right,
    /// y DOWN: pushed forward reads -1), <paramref name="dt"/> in seconds, <paramref name="speed"/> in scene units per
    /// second at full deflection. Returns the updated room-to-scene transform.
    /// </summary>
    public Matrix4x4 Step(Matrix4x4 sceneFromRoom, Vector3 headPosition, Quaternion headOrientation,
        Vector2 left, Vector2 right, float dt, float speed)
    {
        var m = sceneFromRoom;
        var forward = Vector3.TransformNormal(Vector3.Transform(-Vector3.UnitZ, headOrientation), m);
        forward.Y = 0;
        if (forward.LengthSquared() < 1e-8f) forward = Vector3.TransformNormal(-Vector3.UnitZ, m); // looking straight up/down
        forward = Vector3.Normalize(forward);
        var rightDir = Vector3.Cross(forward, Vector3.UnitY);

        var l = ApplyDeadzone(left);
        var move = (forward * -l.Y + rightDir * l.X) * speed * dt;

        // Right stick: whichever axis dominates, so a slightly diagonal push does one thing.
        var r = ApplyDeadzone(right);
        bool turnAxis = MathF.Abs(right.X) >= MathF.Abs(right.Y);
        if (!turnAxis) move.Y += -r.Y * speed * dt;
        if (move != Vector3.Zero) m *= Matrix4x4.CreateTranslation(move);

        if (MathF.Abs(right.X) < SnapRearm) _snapArmed = true;
        else if (_snapArmed && turnAxis && MathF.Abs(right.X) >= SnapFire)
        {
            _snapArmed = false;
            var head = Vector3.Transform(headPosition, m);
            // CreateRotationY with a positive angle turns -Z toward -X (left), so a right push turns by a negative angle.
            float angle = -MathF.Sign(right.X) * SnapTurnDegrees * MathF.PI / 180f;
            m *= Matrix4x4.CreateTranslation(-head) * Matrix4x4.CreateRotationY(angle) * Matrix4x4.CreateTranslation(head);
        }
        return m;
    }

    /// <summary>Radial deadzone, rescaled so the response starts at 0 just past it and still reaches 1.</summary>
    public static Vector2 ApplyDeadzone(Vector2 v)
    {
        float len = v.Length();
        if (len <= Deadzone) return Vector2.Zero;
        return v / len * (MathF.Min(len, 1f) - Deadzone) / (1f - Deadzone);
    }
}
