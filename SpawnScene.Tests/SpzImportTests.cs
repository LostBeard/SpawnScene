using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// SPZ decode (SpzImport kernel, CPU accelerator) against files packed with Niantic's own rules (nianticlabs/spz
/// load-spz.cc packGaussians / packQuaternionSmallestThree, ported line by line below): every attribute must come back
/// within its quantisation step, v2 (first-three quaternions) and v3 (smallest three), with SH degree 3, and degree 4
/// (its fourth band skipped, the first three intact).
/// </summary>
public class SpzImportTests
{
    const int N = 5, Frac = 12;

    record G(Vector3 Pos, Vector3 LogScale, Quaternion Rot, float Alpha, Vector3 Dc, float[] Sh);

    static List<G> Cloud(int shDim)
    {
        var rng = new Random(7);
        float R(float a, float b) => a + (float)rng.NextDouble() * (b - a);
        var list = new List<G>();
        for (int i = 0; i < N; i++)
        {
            var q = Quaternion.Normalize(new Quaternion(R(-1, 1), R(-1, 1), R(-1, 1), R(-1, 1)));
            list.Add(new G(new Vector3(R(-5, 5), R(-2, 2), R(-5, 5)), new Vector3(R(-6, 0), R(-6, 0), R(-6, 0)), q,
                R(-3, 3), new Vector3(R(-1.5f, 1.5f), R(-1.5f, 1.5f), R(-1.5f, 1.5f)),
                Enumerable.Range(0, shDim * 3).Select(_ => R(-0.9f, 0.9f)).ToArray()));
        }
        return list;
    }

    static byte ToU8(float x) => (byte)Math.Clamp(MathF.Round(x), 0, 255);

    /// <summary>Niantic's packQuaternionSmallestThree (x y z w).</summary>
    static void PackSmallestThree(byte[] r, int at, Quaternion qq)
    {
        var q = new[] { qq.X, qq.Y, qq.Z, qq.W };
        float n = MathF.Sqrt(q.Sum(v => v * v));
        for (int i = 0; i < 4; i++) q[i] /= n;
        int largest = 0;
        for (int i = 1; i < 4; i++) if (MathF.Abs(q[i]) > MathF.Abs(q[largest])) largest = i;
        uint negate = q[largest] < 0 ? 1u : 0u;
        uint comp = (uint)largest;
        for (int i = 0; i < 4; i++)
        {
            if (i == largest) continue;
            uint negbit = (q[i] < 0 ? 1u : 0u) ^ negate;
            uint mag = (uint)(511f * (MathF.Abs(q[i]) / 0.70710678f) + 0.5f);
            comp = (comp << 10) | (negbit << 9) | mag;
        }
        r[at] = (byte)comp; r[at + 1] = (byte)(comp >> 8); r[at + 2] = (byte)(comp >> 16); r[at + 3] = (byte)(comp >> 24);
    }

    static byte[] Pack(List<G> g, int version, int shDegree, int shDim)
    {
        var h = new SpzImport.Header(version, N, shDegree, Frac, false);
        var d = new byte[h.Bytes];
        BitConverter.GetBytes(SpzImport.Magic).CopyTo(d, 0);
        BitConverter.GetBytes(version).CopyTo(d, 4);
        BitConverter.GetBytes(N).CopyTo(d, 8);
        d[12] = (byte)shDegree; d[13] = Frac;
        for (int i = 0; i < N; i++)
        {
            var p = g[i].Pos;
            float[] pv = { p.X, p.Y, p.Z };
            for (int a = 0; a < 3; a++)
            {
                int fixed32 = (int)MathF.Round(pv[a] * (1 << Frac));
                int at = (int)h.PosOff + (i * 3 + a) * 3;
                d[at] = (byte)fixed32; d[at + 1] = (byte)(fixed32 >> 8); d[at + 2] = (byte)(fixed32 >> 16);
            }
            d[h.AlphaOff + i] = ToU8(255f / (1f + MathF.Exp(-g[i].Alpha)));
            float[] dc = { g[i].Dc.X, g[i].Dc.Y, g[i].Dc.Z }, ls = { g[i].LogScale.X, g[i].LogScale.Y, g[i].LogScale.Z };
            for (int c = 0; c < 3; c++)
            {
                d[h.ColourOff + i * 3 + c] = ToU8(dc[c] * (0.15f * 255f) + 0.5f * 255f);
                d[h.ScaleOff + i * 3 + c] = ToU8((ls[c] + 10f) * 16f);
            }
            if (version >= 3) PackSmallestThree(d, (int)h.RotOff + i * 4, g[i].Rot);
            else
            {
                // v2: first three, w made non-negative (q and -q are the same rotation).
                var q = Quaternion.Normalize(g[i].Rot);
                float s = q.W < 0 ? -1f : 1f;
                int at = (int)h.RotOff + i * 3;
                d[at] = ToU8((q.X * s + 1f) * 127.5f); d[at + 1] = ToU8((q.Y * s + 1f) * 127.5f); d[at + 2] = ToU8((q.Z * s + 1f) * 127.5f);
            }
            for (int k = 0; k < shDim * 3; k++) d[h.ShOff + i * shDim * 3 + k] = ToU8(g[i].Sh[k] * 128f + 128f);
        }
        return d;
    }

    static (float[] Packed, float[] ShRest) Decode(byte[] d, bool flip = false)
    {
        var h = SpzImport.ParseHeader(d);
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        var padded = new byte[(d.Length + 3) / 4 * 4 + 8];
        d.CopyTo(padded, 0);
        var words = new uint[padded.Length / 4];
        Buffer.BlockCopy(padded, 0, words, 0, padded.Length);
        using var w = a.Allocate1D(words);
        using var packed = a.Allocate1D<float>(N * SplatFormat.Floats);
        var sh = Enumerable.Range(0, 3).Select(_ => a.Allocate1D<float>(N * SphericalHarmonics.PartFloatsPerSplat)).ToArray();
        SpzImport.Run(a, w.View, h, packed.View, sh[0].View, sh[1].View, sh[2].View, flip);
        a.Synchronize();
        var rest = SphericalHarmonics.JoinParts(sh.Select(s => s.GetAsArray1D()).ToArray());
        foreach (var s in sh) s.Dispose();
        return (packed.GetAsArray1D(), rest);
    }

    [TestCase(2, 3)]
    [TestCase(3, 3)]
    [TestCase(3, 4)]
    public void DecodesWithinTheQuantisationStep(int version, int shDegree)
    {
        int shDim = shDegree switch { 3 => 15, 4 => 24, _ => 0 };
        var g = Cloud(shDim);
        var (p, rest) = Decode(Pack(g, version, shDegree, shDim));
        for (int i = 0; i < N; i++)
        {
            int o = i * SplatFormat.Floats;
            Assert.That(new Vector3(p[o], p[o + 1], p[o + 2]), Is.EqualTo(g[i].Pos).Using<Vector3>((x, y) => Vector3.Distance(x, y) < 1e-3f), $"#{i} position");
            Assert.That(p[o + 3], Is.EqualTo(g[i].Dc.X).Within(0.5f / (0.15f * 255f) + 1e-4f), $"#{i} dc");
            Assert.That(MathF.Log(p[o + 6]), Is.EqualTo(g[i].LogScale.X).Within(0.5f / 16f + 1e-4f), $"#{i} scale");
            Assert.That(p[o + 9], Is.EqualTo(1f / (1f + MathF.Exp(-g[i].Alpha))).Within(0.5f / 255f + 1e-4f), $"#{i} opacity");
            var q = new Quaternion(p[o + 10], p[o + 11], p[o + 12], p[o + 13]);
            float dot = MathF.Abs(Quaternion.Dot(q, Quaternion.Normalize(g[i].Rot)));
            Assert.That(dot, Is.GreaterThan(version >= 3 ? 0.9995f : 0.995f), $"#{i} rotation (|dot| {dot})");
            // The first 15 coefficients (bands 1-3) in our band-major order; SPZ is coefficient-major too.
            for (int k = 0; k < 45; k++)
                Assert.That(rest[i * 45 + k], Is.EqualTo(g[i].Sh[k]).Within(0.5f / 128f + 1e-4f), $"#{i} sh {k}");
        }
    }

    [Test]
    public void OtherVersionsAreRefusedWithAReason()
    {
        var d = Pack(Cloud(0), 3, 0, 0);
        BitConverter.GetBytes(4).CopyTo(d, 4);
        Assert.That(Assert.Throws<FormatException>(() => SpzImport.ParseHeader(d))!.Message, Does.Contain("version 4"));
        Assert.Throws<FormatException>(() => SpzImport.ParseHeader(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }));
    }

    /// <summary>The y-up turn (files in circulation are y down): positions and rotations turned 180 degrees about X and
    /// the SH colour seen from F d equal to the original from d, as GaussianPlyImportTests checks the PLY path.</summary>
    [Test]
    public void TheYUpTurnTurnsPositionsRotationsAndShTogether()
    {
        var d = Pack(Cloud(15), 3, 3, 15);
        var (p0, r0) = Decode(d);
        var (p1, r1) = Decode(d, flip: true);
        var F = new Matrix4x4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
        var dirs = new[] { new Vector3(0.3f, 0.5f, 0.81f), new Vector3(-0.7f, 0.1f, 0.7f), new Vector3(0.2f, -0.9f, 0.4f) };
        for (int i = 0; i < N; i++)
        {
            int o = i * SplatFormat.Floats;
            Assert.That(new[] { p1[o], p1[o + 1], p1[o + 2] }, Is.EqualTo(new[] { p0[o], -p0[o + 1], -p0[o + 2] }));
            var m0 = Matrix4x4.CreateFromQuaternion(new Quaternion(p0[o + 10], p0[o + 11], p0[o + 12], p0[o + 13]));
            var m1 = Matrix4x4.CreateFromQuaternion(new Quaternion(p1[o + 10], p1[o + 11], p1[o + 12], p1[o + 13]));
            var want = m0 * F;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    Assert.That(m1[a, b], Is.EqualTo(want[a, b]).Within(1e-5f), $"#{i} rotation [{a},{b}]");
            var dc = p0.AsSpan(o + 3, 3).ToArray();
            foreach (var d0 in dirs)
            {
                var dir = Vector3.Normalize(d0);
                var seen0 = SphericalHarmonics.EvalRgb(3, dc, r0.AsSpan(i * 45, 45), dir);
                var seen1 = SphericalHarmonics.EvalRgb(3, dc, r1.AsSpan(i * 45, 45), new Vector3(dir.X, -dir.Y, -dir.Z));
                Assert.That(Vector3.Distance(seen0, seen1), Is.LessThan(1e-5f), $"#{i} dir {dir}");
            }
        }
    }
}
