using NUnit.Framework;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The float-as-ordered-int trick and the per-view depth bracket.
///
/// The ordering map is the part that can be wrong silently: get it subtly backwards for
/// negatives and the GPU reduction still returns a plausible box, just the wrong one, and the
/// only symptom is a scene whose sort keys quantise badly. So it is tested exhaustively over
/// the awkward values and by property over a wide random sample.
/// </summary>
public class SplatBoundsTests
{
    static readonly float[] Awkward =
    {
        0f, -0f, float.Epsilon, -float.Epsilon,
        1e-30f, -1e-30f, 1e-8f, -1e-8f,
        0.5f, -0.5f, 1f, -1f, 2f, -2f,
        1000f, -1000f, 3.4e38f, -3.4e38f,
        float.MaxValue, float.MinValue,
        float.PositiveInfinity, float.NegativeInfinity,
    };

    [Test]
    public void Ordered_RoundTrips()
    {
        foreach (float f in Awkward)
        {
            float back = SplatBounds.Unordered(SplatBounds.Ordered(f));
            // -0.0 round-trips to -0.0; compare bits so that is not mistaken for a failure.
            Assert.That(BitConverter.SingleToInt32Bits(back),
                Is.EqualTo(BitConverter.SingleToInt32Bits(f)), $"round trip of {f}");
        }
    }

    [Test]
    public void Ordered_PreservesOrderOnAwkwardValues()
    {
        // -0.0 and 0.0 are equal as floats but map to different ints (-1 and 0). That is fine
        // for a min/max reduction - it only has to never invert a STRICT ordering - so the
        // check is on strict pairs.
        var vals = Awkward.OrderBy(f => f).ToArray();
        for (int i = 0; i < vals.Length; i++)
            for (int j = 0; j < vals.Length; j++)
            {
                if (!(vals[i] < vals[j])) continue;
                Assert.That(SplatBounds.Ordered(vals[i]), Is.LessThan(SplatBounds.Ordered(vals[j])),
                    $"{vals[i]} < {vals[j]} but the ordered ints disagree");
            }
    }

    [Test]
    public void Ordered_PreservesOrderOnRandomSample()
    {
        var rng = new Random(4242);
        for (int t = 0; t < 20000; t++)
        {
            float a = (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-20, 20)));
            float b = (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-20, 20)));
            if (a == b) continue;
            Assert.That(SplatBounds.Ordered(a) < SplatBounds.Ordered(b), Is.EqualTo(a < b),
                $"a={a} b={b}");
        }
    }

    [Test]
    public void Ordered_SeedsBracketEverything()
    {
        // The reduction seeds min with int.MaxValue and max with int.MinValue. If any finite
        // value mapped onto a seed the reduction could not tell "untouched" from "real".
        foreach (float f in Awkward)
        {
            if (float.IsInfinity(f)) continue;
            int o = SplatBounds.Ordered(f);
            Assert.That(o, Is.LessThan(int.MaxValue), $"{f} maps to the min seed");
            Assert.That(o, Is.GreaterThan(int.MinValue), $"{f} maps to the max seed");
        }
    }

    static CameraParams CamAt(Vector3 pos, Vector3 lookAt) => new()
    {
        Width = 640,
        Height = 480,
        FocalX = 1520f,
        FocalY = 1525f,
        CenterX = 320f,
        CenterY = 240f,
        Near = 0.01f,
        Far = 100f,
        Position = pos,
        Forward = Vector3.Normalize(lookAt - pos),
        Up = Vector3.UnitY,
    };

    static readonly SplatBounds.Aabb UnitBox = new(-1f, -1f, -1f, 1f, 1f, 1f);

    [Test]
    public void DepthRange_BracketsEveryCorner()
    {
        var cam = CamAt(new Vector3(0f, 0f, 5f), Vector3.Zero);
        var (near, far) = SplatBounds.DepthRangeFor(UnitBox, cam);

        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out _, out _, out var fwd, out var pos);
        for (int c = 0; c < 8; c++)
        {
            var corner = new Vector3(
                (c & 1) == 0 ? UnitBox.MinX : UnitBox.MaxX,
                (c & 2) == 0 ? UnitBox.MinY : UnitBox.MaxY,
                (c & 4) == 0 ? UnitBox.MinZ : UnitBox.MaxZ);
            float d = Vector3.Dot(fwd, corner - pos);
            Assert.That(d, Is.InRange(near - 1e-4f, far + 1e-4f), $"corner {c} at depth {d}");
        }
        // A cube of side 2 seen head-on from 5 away spans depth 4..6.
        Assert.That(near, Is.InRange(3.9f, 4.1f));
        Assert.That(far, Is.InRange(5.9f, 6.1f));
    }

    [Test]
    public void DepthRange_StaysPositiveWhenTheCameraIsInsideTheBox()
    {
        // A room: the camera stands in the middle and looks at a wall, so half the corners are
        // behind it. A negative or zero near would make the depth normalisation divide through
        // a nonsense span.
        var cam = CamAt(Vector3.Zero, new Vector3(0f, 0f, -1f));
        var (near, far) = SplatBounds.DepthRangeFor(UnitBox, cam);
        Assert.That(near, Is.GreaterThan(0f), $"near={near}");
        Assert.That(far, Is.GreaterThan(near), $"near={near} far={far}");
    }

    [Test]
    public void DepthRange_IsNonDegenerateForAFlatScene()
    {
        // Every splat at one depth: the span must still be usable, not zero-width, or every
        // key quantises identically and the front-to-back sort stops ordering anything.
        var flat = new SplatBounds.Aabb(-1f, -1f, 0f, 1f, 1f, 0f);
        var cam = CamAt(new Vector3(0f, 0f, 3f), Vector3.Zero);
        var (near, far) = SplatBounds.DepthRangeFor(flat, cam);
        Assert.That(far, Is.GreaterThan(near), $"near={near} far={far}");
    }

    [Test]
    public void Aabb_ReportsCentreAndDiagonal()
    {
        var box = new SplatBounds.Aabb(-2f, 0f, 1f, 4f, 6f, 1f);
        Assert.That(box.CentreX, Is.EqualTo(1f).Within(1e-5f));
        Assert.That(box.CentreY, Is.EqualTo(3f).Within(1e-5f));
        Assert.That(box.CentreZ, Is.EqualTo(1f).Within(1e-5f));
        Assert.That(box.Diagonal, Is.EqualTo(MathF.Sqrt(72f)).Within(1e-4f));
    }
}
