using System.Numerics;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

public sealed class SphericalHarmonicsTests
{
    [Test]
    public void Degree0_round_trips_dc_rgb()
    {
        var rgb = new Vector3(0.2f, 0.5f, 0.9f);
        var dc = SphericalHarmonics.RgbToDc(rgb);
        Span<float> dcSpan = stackalloc float[3];
        dcSpan[0] = dc.X; dcSpan[1] = dc.Y; dcSpan[2] = dc.Z;
        Span<float> rest = stackalloc float[SphericalHarmonics.RestFloatsPerSplat];
        var dir = Vector3.Normalize(new Vector3(1f, 0.2f, -0.3f));
        var back = SphericalHarmonics.EvalRgb(0, dcSpan, rest, dir);
        Assert.That(back.X, Is.EqualTo(rgb.X).Within(1e-5f));
        Assert.That(back.Y, Is.EqualTo(rgb.Y).Within(1e-5f));
        Assert.That(back.Z, Is.EqualTo(rgb.Z).Within(1e-5f));
    }

    [Test]
    public void Degree1_rest_coeff_finite_difference()
    {
        var dcV = SphericalHarmonics.RgbToDc(new Vector3(0.4f, 0.4f, 0.4f));
        Span<float> dc = stackalloc float[3];
        dc[0] = dcV.X; dc[1] = dcV.Y; dc[2] = dcV.Z;
        Span<float> rest = stackalloc float[SphericalHarmonics.RestFloatsPerSplat];
        var dir = Vector3.Normalize(new Vector3(-0.2f, 0.7f, 0.5f));

        const int band = 0; // first rest vec3 (sh1)
        const float eps = 1e-3f;
        rest[band] = eps;
        float plus = SphericalHarmonics.EvalRgb(1, dc, rest, dir).X;
        rest[band] = -eps;
        float minus = SphericalHarmonics.EvalRgb(1, dc, rest, dir).X;
        float numeric = (plus - minus) / (2f * eps);

        rest[band] = 0f;
        float x = dir.X, y = dir.Y, z = dir.Z;
        float basis = SphericalHarmonics.C1 * (-y);
        float analytic = basis;

        Assert.That(numeric, Is.EqualTo(analytic).Within(1e-3f));
    }
}
