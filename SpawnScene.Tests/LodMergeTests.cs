using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// The level-of-detail merge (LodMerge, Plans/lod-streaming.md): a parent splat must carry its children's weighted
/// mean and full covariance (their spread AND their own extents), keep a single child unchanged, and conserve opacity x
/// area. The eigen step must invert Cov3DFromScaleQuat.
/// </summary>
public class LodMergeTests
{
    const int F = SplatFormat.Floats;

    static float[] Row(Random rng, float spread = 1f)
    {
        var r = new float[F];
        r[0] = (float)(rng.NextDouble() * 2 - 1) * spread; r[1] = (float)(rng.NextDouble() * 2 - 1) * spread; r[2] = (float)(rng.NextDouble() * 2 - 1) * spread;
        r[3] = (float)rng.NextDouble(); r[4] = (float)rng.NextDouble(); r[5] = (float)rng.NextDouble();
        r[6] = 0.01f + (float)rng.NextDouble() * 0.1f; r[7] = 0.01f + (float)rng.NextDouble() * 0.1f; r[8] = 0.001f + (float)rng.NextDouble() * 0.05f;
        r[9] = 0.2f + (float)rng.NextDouble() * 0.8f;
        double qx = rng.NextDouble() * 2 - 1, qy = rng.NextDouble() * 2 - 1, qz = rng.NextDouble() * 2 - 1, qw = rng.NextDouble() * 2 - 1;
        double n = Math.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        r[10] = (float)(qx / n); r[11] = (float)(qy / n); r[12] = (float)(qz / n); r[13] = (float)(qw / n);
        return r;
    }

    static SplatCovariance.Cov3 CovOf(float[] r) => SplatCovariance.Cov3DFromScaleQuat(r[6], r[7], r[8],
        new SplatCovariance.Quat { X = r[10], Y = r[11], Z = r[12], W = r[13] });

    static void SameCov(SplatCovariance.Cov3 a, SplatCovariance.Cov3 b, float tol, string what)
    {
        Assert.That(a.M00, Is.EqualTo(b.M00).Within(tol), what + " xx");
        Assert.That(a.M01, Is.EqualTo(b.M01).Within(tol), what + " xy");
        Assert.That(a.M02, Is.EqualTo(b.M02).Within(tol), what + " xz");
        Assert.That(a.M11, Is.EqualTo(b.M11).Within(tol), what + " yy");
        Assert.That(a.M12, Is.EqualTo(b.M12).Within(tol), what + " yz");
        Assert.That(a.M22, Is.EqualTo(b.M22).Within(tol), what + " zz");
    }

    [Test]
    public void Eigen_InvertsCov3DFromScaleQuat()
    {
        var rng = new Random(1);
        for (int k = 0; k < 500; k++)
        {
            var r = Row(rng);
            var c = CovOf(r);
            LodMerge.ScaleQuatFromCov(c, out float sx, out float sy, out float sz, out var q);
            var back = SplatCovariance.Cov3DFromScaleQuat(sx, sy, sz, q);
            SameCov(back, c, 1e-7f, $"case {k}");
            // The rotation is a proper rotation (unit quaternion).
            Assert.That(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W, Is.EqualTo(1f).Within(1e-5f));
        }
    }

    [Test]
    public void SingleChild_IsUnchanged()
    {
        var rng = new Random(2);
        for (int k = 0; k < 50; k++)
        {
            var r = Row(rng);
            var p = new float[F];
            LodMerge.Merge(r, 1, p);
            for (int i = 0; i < 6; i++) Assert.That(p[i], Is.EqualTo(r[i]).Within(1e-5f), $"case {k} float {i}");
            SameCov(CovOf(p), CovOf(r), 1e-7f, $"case {k}");
            Assert.That(p[9], Is.EqualTo(r[9]).Within(1e-4f), $"case {k} opacity");
        }
    }

    [Test]
    public void Parent_CarriesTheChildrensMomentsAndOpacityArea()
    {
        var rng = new Random(3);
        for (int k = 0; k < 50; k++)
        {
            int n = 2 + rng.Next(9);
            var rows = new float[n * F];
            for (int i = 0; i < n; i++) Array.Copy(Row(rng, 0.2f), 0, rows, i * F, F);
            var p = new float[F];
            LodMerge.Merge(rows, n, p);

            // Oracle in doubles, written out independently: w = opacity x area.
            double W = 0, mx = 0, my = 0, mz = 0, cr = 0, aa = 0;
            var w = new double[n];
            for (int i = 0; i < n; i++)
            {
                int o = i * F;
                w[i] = rows[o + 9] * LodMerge.Area(rows[o + 6], rows[o + 7], rows[o + 8]);
                W += w[i]; mx += w[i] * rows[o]; my += w[i] * rows[o + 1]; mz += w[i] * rows[o + 2]; cr += w[i] * rows[o + 3];
                aa += rows[o + 9] * LodMerge.Area(rows[o + 6], rows[o + 7], rows[o + 8]);
            }
            mx /= W; my /= W; mz /= W; cr /= W;
            double s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
            for (int i = 0; i < n; i++)
            {
                int o = i * F;
                var c = CovOf(rows[o..(o + F)]);
                double dx = rows[o] - mx, dy = rows[o + 1] - my, dz = rows[o + 2] - mz;
                s00 += w[i] * (c.M00 + dx * dx); s01 += w[i] * (c.M01 + dx * dy); s02 += w[i] * (c.M02 + dx * dz);
                s11 += w[i] * (c.M11 + dy * dy); s12 += w[i] * (c.M12 + dy * dz); s22 += w[i] * (c.M22 + dz * dz);
            }
            var expected = new SplatCovariance.Cov3
            {
                M00 = (float)(s00 / W), M01 = (float)(s01 / W), M02 = (float)(s02 / W),
                M11 = (float)(s11 / W), M12 = (float)(s12 / W), M22 = (float)(s22 / W),
            };
            Assert.That(p[0], Is.EqualTo(mx).Within(1e-5), $"case {k} mean x");
            Assert.That(p[1], Is.EqualTo(my).Within(1e-5), $"case {k} mean y");
            Assert.That(p[2], Is.EqualTo(mz).Within(1e-5), $"case {k} mean z");
            Assert.That(p[3], Is.EqualTo(cr).Within(1e-5), $"case {k} colour");
            SameCov(CovOf(p), expected, 2e-6f, $"case {k}");
            double area = LodMerge.Area(p[6], p[7], p[8]);
            Assert.That(p[9], Is.EqualTo(Math.Min(1.0, aa / area)).Within(1e-4), $"case {k} opacity x area conserved");
        }
    }

    [Test]
    public void TwoCoincidentChildren_DoubleTheCoverage()
    {
        var rng = new Random(4);
        var r = Row(rng);
        r[9] = 0.3f;
        var rows = new float[2 * F];
        Array.Copy(r, 0, rows, 0, F); Array.Copy(r, 0, rows, F, F);
        var p = new float[F];
        LodMerge.Merge(rows, 2, p);
        SameCov(CovOf(p), CovOf(r), 1e-7f, "same extent");
        Assert.That(p[9], Is.EqualTo(0.6f).Within(1e-4f), "opacity x area of two coincident splats");
    }
}
