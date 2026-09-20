using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for the yaw/pitch camera model shared by <c>CameraController</c> and
/// <c>CameraController.SetPose</c>.
///
/// Why this exists: parking the camera at an exact pose has to sync the controller's yaw/pitch,
/// or the first time the user looks around the view jumps. I wrote the inverse as
/// <c>Atan2(x, z)</c> when forward.z is <c>-cos(yaw)cos(pitch)</c>, which is 180 degrees out -
/// invisible until someone drags the mouse.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter CameraPose</c>
/// </summary>
public class CameraPoseTests
{
    [Test]
    public void ForwardFromYawPitch_MatchesTheDocumentedConvention()
    {
        // yaw 0, pitch 0 looks down world -Z.
        var f = WorldSpaceGeometry.ForwardFromYawPitch(0f, 0f);
        Assert.That(f.X, Is.EqualTo(0f).Within(1e-6f));
        Assert.That(f.Y, Is.EqualTo(0f).Within(1e-6f));
        Assert.That(f.Z, Is.EqualTo(-1f).Within(1e-6f), "yaw 0 must look down -Z");

        // +yaw turns toward +X.
        var right = WorldSpaceGeometry.ForwardFromYawPitch(MathF.PI / 2f, 0f);
        Assert.That(right.X, Is.EqualTo(1f).Within(1e-6f), "+yaw must turn toward +X");

        // +pitch looks up.
        var up = WorldSpaceGeometry.ForwardFromYawPitch(0f, MathF.PI / 4f);
        Assert.That(up.Y, Is.GreaterThan(0.7f), "+pitch must look up");
    }

    [Test]
    public void YawPitch_RoundTripsThroughForward()
    {
        // The load-bearing test. Atan2(x, z) instead of Atan2(x, -z) passes a yaw-0 spot check
        // and fails here on every other angle.
        for (int yi = -7; yi <= 7; yi++)
        {
            for (int pi = -4; pi <= 4; pi++)
            {
                float yaw = yi * (MathF.PI / 8f);
                float pitch = pi * (MathF.PI / 10f);   // stays clear of the poles

                var f = WorldSpaceGeometry.ForwardFromYawPitch(yaw, pitch);
                WorldSpaceGeometry.YawPitchFromForward(f, out float yaw2, out float pitch2);
                var f2 = WorldSpaceGeometry.ForwardFromYawPitch(yaw2, pitch2);

                Assert.That(Vector3.Distance(f, f2), Is.LessThan(1e-5f),
                    $"forward must survive the round trip at yaw={yaw:F3} pitch={pitch:F3}: {f} -> {f2}");
            }
        }
    }

    [Test]
    public void YawPitchFromForward_RecoversAnArbitraryLookDirection()
    {
        foreach (var dir in new[]
        {
            new Vector3(0.3f, -0.2f, 0.9f),     // behind the camera in Z - the case the sign bug broke
            new Vector3(-0.5f, 0.1f, 0.85f),
            new Vector3(0.7f, 0.6f, -0.4f),
            new Vector3(-0.2f, -0.8f, -0.55f),
        })
        {
            var f = Vector3.Normalize(dir);
            WorldSpaceGeometry.YawPitchFromForward(f, out float yaw, out float pitch);
            var back = WorldSpaceGeometry.ForwardFromYawPitch(yaw, pitch);

            Assert.That(Vector3.Dot(f, back), Is.EqualTo(1f).Within(1e-5f),
                $"direction {f} came back as {back}");
        }
    }

    [Test]
    public void LookingBackwardsAlongPositiveZ_DoesNotFlipTheView()
    {
        // Concrete regression for the bug: a camera looking toward +Z. With Atan2(x, z) this
        // comes back pointing at -Z, i.e. the user's first mouse move spins them around.
        var f = Vector3.Normalize(new Vector3(0.05f, 0f, 1f));
        WorldSpaceGeometry.YawPitchFromForward(f, out float yaw, out float pitch);
        var back = WorldSpaceGeometry.ForwardFromYawPitch(yaw, pitch);

        Assert.That(back.Z, Is.GreaterThan(0f), "must still be looking toward +Z, not flipped to -Z");
        Assert.That(Vector3.Dot(f, back), Is.EqualTo(1f).Within(1e-5f));
    }
}
