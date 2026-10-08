using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// antimatter15's .splat decode (SplatFileImport, CPU accelerator) against bytes written by convert.py's exact rules:
/// position, exp(scale), rgb = 0.5 + C0 f_dc and a = sigmoid(opacity) as bytes, the unit quaternion w x y z as
/// q * 128 + 128. Each attribute back within its byte's step; the y-up turn as the PLY import's.
/// </summary>
public class SplatFileImportTests
{
    const int N = 4;
    const float C0 = 0.28209479177387814f;

    static byte U8(float x) => (byte)Math.Clamp((int)x, 0, 255);   // numpy astype(uint8) after clip: truncation

    static (byte[] File, Vector3[] Pos, Vector3[] LogScale, Quaternion[] Rot, float[] Logit, Vector3[] Dc) Make()
    {
        var rng = new Random(3);
        float R(float a, float b) => a + (float)rng.NextDouble() * (b - a);
        var pos = new Vector3[N]; var ls = new Vector3[N]; var rot = new Quaternion[N]; var logit = new float[N]; var dc = new Vector3[N];
        var file = new byte[N * 32];
        for (int i = 0; i < N; i++)
        {
            pos[i] = new(R(-4, 4), R(-4, 4), R(-4, 4)); ls[i] = new(R(-5, 0), R(-5, 0), R(-5, 0));
            rot[i] = Quaternion.Normalize(new Quaternion(R(-1, 1), R(-1, 1), R(-1, 1), R(-1, 1)));
            logit[i] = R(-3, 3); dc[i] = new(R(-1.5f, 1.5f), R(-1.5f, 1.5f), R(-1.5f, 1.5f));
            int o = i * 32;
            BitConverter.GetBytes(pos[i].X).CopyTo(file, o); BitConverter.GetBytes(pos[i].Y).CopyTo(file, o + 4); BitConverter.GetBytes(pos[i].Z).CopyTo(file, o + 8);
            BitConverter.GetBytes(MathF.Exp(ls[i].X)).CopyTo(file, o + 12); BitConverter.GetBytes(MathF.Exp(ls[i].Y)).CopyTo(file, o + 16);
            BitConverter.GetBytes(MathF.Exp(ls[i].Z)).CopyTo(file, o + 20);
            file[o + 24] = U8((0.5f + C0 * dc[i].X) * 255); file[o + 25] = U8((0.5f + C0 * dc[i].Y) * 255);
            file[o + 26] = U8((0.5f + C0 * dc[i].Z) * 255); file[o + 27] = U8(255f / (1f + MathF.Exp(-logit[i])));
            var q = rot[i];
            file[o + 28] = U8(q.W * 128 + 128); file[o + 29] = U8(q.X * 128 + 128); file[o + 30] = U8(q.Y * 128 + 128); file[o + 31] = U8(q.Z * 128 + 128);
        }
        return (file, pos, ls, rot, logit, dc);
    }

    static float[] Decode(byte[] file, bool flip)
    {
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        var words = new uint[file.Length / 4];
        Buffer.BlockCopy(file, 0, words, 0, file.Length);
        using var w = a.Allocate1D(words);
        using var packed = a.Allocate1D<float>(N * SplatFormat.Floats);
        SplatFileImport.Run(a, w.View, N, flip, packed.View);
        a.Synchronize();
        return packed.GetAsArray1D();
    }

    [Test]
    public void DecodesConvertPyBytes()
    {
        var (file, pos, ls, rot, logit, dc) = Make();
        var p = Decode(file, flip: false);
        for (int i = 0; i < N; i++)
        {
            int o = i * SplatFormat.Floats;
            Assert.That(new Vector3(p[o], p[o + 1], p[o + 2]), Is.EqualTo(pos[i]));
            Assert.That(MathF.Log(p[o + 6]), Is.EqualTo(ls[i].X).Within(1e-5f));
            Assert.That(p[o + 3], Is.EqualTo(dc[i].X).Within(1f / (255 * C0) + 1e-4f));
            Assert.That(p[o + 9], Is.EqualTo(1f / (1f + MathF.Exp(-logit[i]))).Within(1f / 255 + 1e-4f));
            var q = new Quaternion(p[o + 10], p[o + 11], p[o + 12], p[o + 13]);
            Assert.That(MathF.Abs(Quaternion.Dot(q, rot[i])), Is.GreaterThan(0.999f), $"#{i} rotation");
        }
    }

    [Test]
    public void TheYUpTurnTurnsPositionsAndRotations()
    {
        var (file, _, _, _, _, _) = Make();
        var p0 = Decode(file, false); var p1 = Decode(file, true);
        var F = new Matrix4x4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
        for (int i = 0; i < N; i++)
        {
            int o = i * SplatFormat.Floats;
            Assert.That(new[] { p1[o], p1[o + 1], p1[o + 2] }, Is.EqualTo(new[] { p0[o], -p0[o + 1], -p0[o + 2] }));
            var m0 = Matrix4x4.CreateFromQuaternion(new Quaternion(p0[o + 10], p0[o + 11], p0[o + 12], p0[o + 13]));
            var m1 = Matrix4x4.CreateFromQuaternion(new Quaternion(p1[o + 10], p1[o + 11], p1[o + 12], p1[o + 13]));
            var want = m0 * F;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    Assert.That(m1[a, b], Is.EqualTo(want[a, b]).Within(1e-5f));
        }
    }
}
