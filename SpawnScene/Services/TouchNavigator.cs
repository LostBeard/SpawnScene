using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Touch gestures for the desktop viewer on phones and tablets, where there is no pointer lock and so no mouse-look.
/// One finger drags the view (the scene follows the finger); two or more fingers pan with their centre and dolly with
/// their spread (pinch out = move in). Works from the finger count, centroid and mean spread each frame, so it needs no
/// touch ids (GameUI touch pointers carry none); a change in the number of fingers restarts the gesture, so lifting or
/// adding a finger never jumps the view. A gesture that began on a UI element is left to the UI until every finger lifts.
/// </summary>
public sealed class TouchNavigator
{
    int _count;
    Vector2 _centroid;
    float _spread;
    bool _onUi;

    /// <summary>What one frame of touches asks the camera to do (all zero when nothing applies).</summary>
    public readonly record struct Gesture(Vector2 LookPixels, Vector2 PanPixels, float PinchLog);

    /// <summary>
    /// One frame. <paramref name="touches"/> are the screen positions of the fingers down now;
    /// <paramref name="startsOnUi"/> says whether a finger that touched down this frame landed on a UI element.
    /// </summary>
    public Gesture Step(IReadOnlyList<Vector2> touches, bool startsOnUi)
    {
        int n = touches.Count;
        if (n == 0) { _count = 0; _onUi = false; return default; }
        if (_count == 0) _onUi = startsOnUi;   // a new gesture: whose is it? (fingers added later do not change it)

        var c = Vector2.Zero;
        foreach (var t in touches) c += t;
        c /= n;
        float s = 0;
        foreach (var t in touches) s += Vector2.Distance(t, c);
        s /= n;

        Gesture g = default;
        if (!_onUi && n == _count)
        {
            var d = c - _centroid;
            if (n == 1) g = new Gesture(d, Vector2.Zero, 0f);
            else g = new Gesture(Vector2.Zero, d, _spread > 1f && s > 1f ? MathF.Log(s / _spread) : 0f);
        }
        _count = n; _centroid = c; _spread = s;
        return g;
    }
}
