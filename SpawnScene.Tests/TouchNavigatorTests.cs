using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Viewer touch gestures (TouchNavigator, 2026-10-03).</summary>
public class TouchNavigatorTests
{
    static Vector2[] P(params float[] xy) => Enumerable.Range(0, xy.Length / 2).Select(i => new Vector2(xy[2 * i], xy[2 * i + 1])).ToArray();

    [Test]
    public void OneFinger_Looks_ByItsMovement()
    {
        var t = new TouchNavigator();
        Assert.That(t.Step(P(100, 100), false), Is.EqualTo(default(TouchNavigator.Gesture)), "touch-down moves nothing");
        var g = t.Step(P(130, 90), false);
        Assert.That(g.LookPixels, Is.EqualTo(new Vector2(30, -10)));
        Assert.That(g.PanPixels, Is.EqualTo(Vector2.Zero));
    }

    [Test]
    public void TwoFingers_PanWithTheCentre_PinchWithTheSpread()
    {
        var t = new TouchNavigator();
        t.Step(P(100, 100, 200, 100), false);
        var g = t.Step(P(80, 110, 220, 110), false);   // centre +0,+10; spread 50 -> 70
        Assert.That(g.PanPixels.X, Is.EqualTo(0f).Within(1e-4f));
        Assert.That(g.PanPixels.Y, Is.EqualTo(10f).Within(1e-4f));
        Assert.That(g.PinchLog, Is.EqualTo(MathF.Log(70f / 50f)).Within(1e-5f));
        Assert.That(g.LookPixels, Is.EqualTo(Vector2.Zero));
    }

    [Test]
    public void ChangingFingerCount_DoesNotJump()
    {
        var t = new TouchNavigator();
        t.Step(P(100, 100), false);
        t.Step(P(110, 100), false);
        // A second finger far away: the centroid leaps, but the gesture restarts.
        Assert.That(t.Step(P(110, 100, 400, 300), false), Is.EqualTo(default(TouchNavigator.Gesture)));
        // Lifting it again: restarts once more.
        Assert.That(t.Step(P(112, 100), false), Is.EqualTo(default(TouchNavigator.Gesture)));
        Assert.That(t.Step(P(120, 100), false).LookPixels, Is.EqualTo(new Vector2(8, 0)));
    }

    [Test]
    public void GestureStartedOnUi_IsLeftToTheUi_UntilAllFingersLift()
    {
        var t = new TouchNavigator();
        t.Step(P(10, 10), true);
        Assert.That(t.Step(P(50, 10), false), Is.EqualTo(default(TouchNavigator.Gesture)), "slider drag is not a look");
        t.Step(P(), false);
        t.Step(P(10, 10), false);
        Assert.That(t.Step(P(20, 10), false).LookPixels, Is.EqualTo(new Vector2(10, 0)));
    }
}
