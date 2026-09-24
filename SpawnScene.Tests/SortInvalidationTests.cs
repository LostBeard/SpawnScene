using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The viewer must re-cull when the PROJECTION changes, not only when the camera moves. MEASURED
/// 2026-09-24: parking at Truck view 0 (same pose as the previous shot, photo intrinsics) kept the old
/// narrow frustum's cull and rendered the periphery black - 8.4 dB in the viewer vs the trainer's 24.55.
/// </summary>
public class SortInvalidationTests
{
    static Matrix4x4 Mvp(CameraParams cam, int w, int h)
    {
        var proj = CameraParams.CreateWebGpuProjection(cam.FocalX, cam.FocalY, cam.CenterX, cam.CenterY, w, h, 0.1f, 100f);
        return cam.ViewMatrix * proj;
    }

    static CameraParams Cam(float f) => new()
    {
        Width = 979, Height = 546, FocalX = f, FocalY = f, CenterX = 489.5f, CenterY = 273f,
        Position = new Vector3(-1.03f, 0.02f, 4.03f), Forward = Vector3.Normalize(new Vector3(0.52f, -0.12f, -0.84f)), Up = Vector3.UnitY,
    };

    [Test]
    public void SamePose_NewIntrinsics_Resorts()
    {
        var narrow = Mvp(Cam(1400f), 979, 546);  // the viewer's default, zoomed-in
        var photo = Mvp(Cam(700f), 979, 546);    // the photo's intrinsics, wider
        Assert.That(SortInvalidation.NeedsResort(0f, narrow, photo), Is.True,
            "a wider field of view at the same pose must re-cull, or the periphery stays black");
    }

    [Test]
    public void SamePose_NewViewport_Resorts()
    {
        var a = Mvp(Cam(900f), 979, 546);
        var b = Mvp(Cam(900f), 1920, 1080);   // fullscreen / resize / VR eye buffer
        Assert.That(SortInvalidation.NeedsResort(0f, a, b), Is.True);
    }

    [Test]
    public void Unchanged_DoesNotResort()
    {
        var a = Mvp(Cam(900f), 979, 546);
        Assert.That(SortInvalidation.NeedsResort(0f, a, a), Is.False, "a still camera must not re-sort every frame");
    }

    [Test]
    public void FirstFrame_AndMotion_Resort()
    {
        var a = Mvp(Cam(900f), 979, 546);
        var never = new Matrix4x4(float.NaN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Assert.That(SortInvalidation.NeedsResort(0f, never, a), Is.True);
        Assert.That(SortInvalidation.NeedsResort(1e-3f, a, a), Is.True);
    }
}
