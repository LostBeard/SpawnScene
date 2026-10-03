using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Grip "world grab" for a VR session, the way VR splat and map viewers let people size a scene: hold one grip and the
/// scene moves with that hand (pull yourself through it); hold both and the scene scales with the distance between the
/// hands and turns with the line between them, about their midpoint. Scenes come in arbitrary units (monocular depth,
/// SfM), so no fixed scale is right for every scene - the viewer sets it by hand. Works on the room-to-scene transform
/// (row vectors: pScene = pRoom * sceneFromRoom); hand positions are in the room (local-floor space).
/// </summary>
public sealed class XRWorldGrab
{
    /// <summary>Scene units per room metre stays inside this range, so a fumble cannot shrink the scene to nothing.</summary>
    public const float MinScale = 1e-3f, MaxScale = 1e3f;

    bool _left, _right;
    Vector3 _left0, _right0;
    Matrix4x4 _start;

    /// <summary>True while either grip holds the scene.</summary>
    public bool Active => _left || _right;

    /// <summary>Release both grips (a new session).</summary>
    public void Reset() => _left = _right = false;

    /// <summary>
    /// One frame. <paramref name="leftHeld"/>/<paramref name="rightHeld"/> are the grips' states, the positions the grip
    /// poses (ignored when not held). A change in which grips are held restarts the grab from the current transform, so
    /// going from one hand to two (or back) never jumps.
    /// </summary>
    public Matrix4x4 Step(Matrix4x4 sceneFromRoom, bool leftHeld, Vector3 left, bool rightHeld, Vector3 right)
    {
        if (leftHeld != _left || rightHeld != _right)
        {
            _left = leftHeld; _right = rightHeld;
            _left0 = left; _right0 = right;
            _start = sceneFromRoom;
            return sceneFromRoom;
        }
        if (!_left && !_right) return sceneFromRoom;

        // G takes where the hands ARE to where they WERE (room space); the scene point that was under a hand at the
        // start is then under it now: pRoomNow * G * start = pRoomStart * start.
        Matrix4x4 g;
        if (_left && _right)
        {
            var d0 = Flat(_right0 - _left0);
            var d = Flat(right - left);
            float len0 = d0.Length(), len = d.Length();
            if (len0 < 1e-3f || len < 1e-3f) return sceneFromRoom;   // hands together: no defined scale or turn
            float s = len0 / len;
            // Clamp the total scale (scene units per room metre).
            float total = Scale(_start) * s;
            s *= Math.Clamp(total, MinScale, MaxScale) / total;
            // CreateRotationY(a) turns a vector's heading by -a in Heading's sense (see XRLocomotion).
            float a = Heading(d) - Heading(d0);
            var mid0 = (_left0 + _right0) * 0.5f;
            var mid = (left + right) * 0.5f;
            g = Matrix4x4.CreateTranslation(-mid) * Matrix4x4.CreateScale(s) * Matrix4x4.CreateRotationY(a)
                * Matrix4x4.CreateTranslation(mid0);
        }
        else
        {
            var p0 = _left ? _left0 : _right0;
            var p = _left ? left : right;
            g = Matrix4x4.CreateTranslation(p0 - p);
        }
        return g * _start;
    }

    /// <summary>Scene units per room metre of a room-to-scene transform (uniform scale).</summary>
    public static float Scale(Matrix4x4 sceneFromRoom) => new Vector3(sceneFromRoom.M11, sceneFromRoom.M12, sceneFromRoom.M13).Length();

    static Vector3 Flat(Vector3 v) => new(v.X, 0, v.Z);
    static float Heading(Vector3 f) => MathF.Atan2(f.X, -f.Z);   // 0 = facing -Z, +pi/2 = facing +X
}
