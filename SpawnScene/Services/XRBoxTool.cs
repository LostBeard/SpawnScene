using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// The in-headset selection box: with the tool on, hold the trigger at one corner and let go at the opposite one - the
/// box is drawn in the air between the controller tip's two positions (level with the room). On release it is fixed in
/// the SCENE (<see cref="BoxToScene"/>), so it stays on the content however the viewer then moves, turns or rescales.
/// Its volume (<see cref="SplatEditor.Volume.Box"/>) feeds the same Delete / Keep only / Undo as the desktop tools.
/// </summary>
public sealed class XRBoxTool
{
    /// <summary>Smallest half-size (metres) so a click without a drag still makes a usable box.</summary>
    public const float MinHalfSize = 0.01f;

    public bool Active { get; set; }
    public bool Dragging { get; private set; }
    Vector3 _a, _b;
    bool _wasDown;

    /// <summary>The finished box: the cube -1..1 mapped into the scene (null until one is drawn).</summary>
    public Matrix4x4? BoxToScene { get; private set; }

    public void Clear() { BoxToScene = null; Dragging = false; }

    /// <summary>
    /// One frame. <paramref name="tip"/> is the controller tip in the room; returns true on the frame a box is
    /// finished (trigger released).
    /// </summary>
    public bool Step(Vector3? tip, bool triggerDown, Matrix4x4 sceneFromRoom)
    {
        bool finished = false;
        if (Active && tip is { } t)
        {
            if (triggerDown && !_wasDown) { _a = _b = t; Dragging = true; BoxToScene = null; }
            else if (triggerDown && Dragging) _b = t;
            else if (!triggerDown && _wasDown && Dragging)
            {
                _b = t;
                Dragging = false;
                BoxToScene = RoomBox(_a, _b) * sceneFromRoom;
                finished = true;
            }
        }
        _wasDown = triggerDown;
        return finished;
    }

    /// <summary>The box to draw this frame, as the cube -1..1 mapped into the ROOM: the one being dragged, or the
    /// finished one brought back from the scene.</summary>
    public Matrix4x4? BoxToRoom(Matrix4x4 sceneFromRoom)
    {
        if (Dragging) return RoomBox(_a, _b);
        if (BoxToScene is not { } s || !Matrix4x4.Invert(sceneFromRoom, out var roomFromScene)) return null;
        return s * roomFromScene;
    }

    /// <summary>The cube -1..1 mapped onto the room-aligned box with corners <paramref name="a"/> and <paramref name="b"/>.</summary>
    public static Matrix4x4 RoomBox(Vector3 a, Vector3 b)
    {
        var half = Vector3.Max(Vector3.Abs(b - a) * 0.5f, new Vector3(MinHalfSize));
        return Matrix4x4.CreateScale(half) * Matrix4x4.CreateTranslation((a + b) * 0.5f);
    }

    /// <summary>The 8 corners of the cube -1..1 under <paramref name="cubeTo"/> (bit 0 = x, 1 = y, 2 = z).</summary>
    public static Vector3[] Corners(Matrix4x4 cubeTo)
    {
        var c = new Vector3[8];
        for (int i = 0; i < 8; i++)
            c[i] = Vector3.Transform(new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1), cubeTo);
        return c;
    }

    /// <summary>The 12 edges, as corner index pairs.</summary>
    public static readonly (int A, int B)[] Edges =
    {
        (0, 1), (2, 3), (4, 5), (6, 7),   // along x
        (0, 2), (1, 3), (4, 6), (5, 7),   // along y
        (0, 4), (1, 5), (2, 6), (3, 7),   // along z
    };
}
