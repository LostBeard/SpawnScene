using System.Numerics;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Formats;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// 3DGS .ply import (GaussianPly header + GaussianPlyImport kernel, CPU accelerator) on a hand-made file in the
/// reference trainer's exact layout, with a header whose length is NOT a multiple of 4 (vertex fields unaligned):
/// every field lands in its SplatFormat slot (exp of the log scale, sigmoid of the logit, quaternion w x y z -> x y z w,
/// f_rest channel-major -> band-major parts). With the y-up turn: positions and rotations turned 180 degrees about X,
/// and the SH colour seen from F d equals the original seen from d - the sign table checked against the physics.
/// </summary>
public class GaussianPlyImportTests
{
    const int N = 3;

    static float Rest(int v, int j) => 0.01f * j - 0.2f + 0.05f * v;

    static (byte[] File, float[][] Q) MakePly(int shRestPerChannel = 15)
    {
        var props = new List<string> { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" };
        for (int j = 0; j < shRestPerChannel * 3; j++) props.Add($"f_rest_{j}");
        props.AddRange(new[] { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" });
        var head = new StringBuilder("ply\nformat binary_little_endian 1.0\ncomment made by a test\n");
        head.Append($"element vertex {N}\n");
        foreach (var p in props) head.Append($"property float {p}\n");
        head.Append("end_header\n");
        // Make the header length 1 mod 4: every float in the vertices is unaligned.
        while (head.Length % 4 != 1) head.Insert(head.ToString().IndexOf("\nelement", StringComparison.Ordinal), "!");   // lengthen the comment
        var quats = new[] { new[] { 2f, 0f, 0f, 0f }, new[] { 0.5f, 0.5f, 0.5f, 0.5f }, new[] { 0.9f, -0.1f, 0.3f, 0.2f } };
        using var ms = new MemoryStream();
        ms.Write(Encoding.ASCII.GetBytes(head.ToString()));
        using var bw = new BinaryWriter(ms);
        for (int v = 0; v < N; v++)
        {
            bw.Write(1f + v); bw.Write(2f + v); bw.Write(3f + v);          // x y z
            bw.Write(0f); bw.Write(0f); bw.Write(0f);                    // normals
            bw.Write(0.1f * (v + 1)); bw.Write(0.2f); bw.Write(-0.3f);   // f_dc
            for (int j = 0; j < shRestPerChannel * 3; j++) bw.Write(Rest(v, j));
            bw.Write(v == 0 ? 0f : 2f);                                  // opacity logit
            bw.Write(MathF.Log(0.5f)); bw.Write(MathF.Log(0.25f)); bw.Write(MathF.Log(2f));
            foreach (var q in quats[v]) bw.Write(q);                     // w x y z
        }
        bw.Flush();
        return (ms.ToArray(), quats);
    }

    static (float[] Packed, float[][] Sh) Convert(byte[] file, bool flip)
    {
        var L = GaussianPly.Parse(file.AsSpan(0, Math.Min(file.Length, 65536)));
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        long start = L.HeaderBytes, baseByte = start & ~3L;
        var padded = new byte[(file.Length - baseByte + 3) / 4 * 4 + 8];
        Array.Copy(file, baseByte, padded, 0, file.Length - baseByte);
        var words = new uint[padded.Length / 4];
        Buffer.BlockCopy(padded, 0, words, 0, padded.Length);
        using var w = a.Allocate1D(words);
        using var packed = a.Allocate1D<float>(N * SplatFormat.Floats);
        var sh = Enumerable.Range(0, 3).Select(_ => a.Allocate1D<float>(N * SphericalHarmonics.PartFloatsPerSplat)).ToArray();
        GaussianPlyImport.RunChunk(a, w.View, (int)(start - baseByte), 0, N, L, flip, packed.View, sh[0].View, sh[1].View, sh[2].View);
        a.Synchronize();
        var result = (packed.GetAsArray1D(), sh.Select(s => s.GetAsArray1D()).ToArray());
        foreach (var s in sh) s.Dispose();
        return result;
    }

    [Test]
    public void HeaderIsParsedWithOffsetsAndDegree()
    {
        var (file, _) = MakePly();
        var L = GaussianPly.Parse(file);
        Assert.That(L.Count, Is.EqualTo(N));
        Assert.That(L.StrideBytes, Is.EqualTo((9 + 45 + 8) * 4));
        Assert.That(L.ShDegree, Is.EqualTo(3));
        Assert.That(L.HeaderBytes % 4, Is.EqualTo(1));
        Assert.That(L.Opacity, Is.EqualTo((9 + 45) * 4));
        Assert.That(GaussianPly.Parse(MakePly(3).File).ShDegree, Is.EqualTo(1));
        Assert.That(GaussianPly.Parse(MakePly(0).File).ShDegree, Is.EqualTo(0));
    }

    [Test]
    public void APointCloudPlyIsRefusedWithAReason()
    {
        var ply = Encoding.ASCII.GetBytes("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\n" +
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
        var ex = Assert.Throws<FormatException>(() => GaussianPly.Parse(ply));
        Assert.That(ex!.Message, Does.Contain("not a Gaussian splat file"));
    }

    [Test]
    public void FieldsLandInTheirSlots()
    {
        var (file, quats) = MakePly();
        var (p, sh) = Convert(file, flip: false);
        for (int v = 0; v < N; v++)
        {
            int o = v * SplatFormat.Floats;
            Assert.That(p.AsSpan(o, 3).ToArray(), Is.EqualTo(new[] { 1f + v, 2f + v, 3f + v }));
            Assert.That(p[o + 3], Is.EqualTo(0.1f * (v + 1)).Within(1e-6f));
            Assert.That(p[o + 6], Is.EqualTo(0.5f).Within(1e-5f));
            Assert.That(p[o + 7], Is.EqualTo(0.25f).Within(1e-5f));
            Assert.That(p[o + 8], Is.EqualTo(2f).Within(1e-5f));
            Assert.That(p[o + 9], Is.EqualTo(v == 0 ? 0.5f : 1f / (1f + MathF.Exp(-2f))).Within(1e-5f));
            var q = quats[v]; float len = MathF.Sqrt(q.Sum(c => c * c));
            Assert.That(p.AsSpan(o + 10, 4).ToArray(), Is.EqualTo(new[] { q[1] / len, q[2] / len, q[3] / len, q[0] / len }).Within(1e-6f));
            var rows = SphericalHarmonics.JoinParts(sh);
            for (int k = 1; k <= 15; k++)
                for (int c = 0; c < 3; c++)
                    Assert.That(rows[v * 45 + (k - 1) * 3 + c], Is.EqualTo(Rest(v, c * 15 + k - 1)).Within(1e-6f), $"v{v} band {k} ch {c}");
        }
    }

    [Test]
    public void TheYUpTurnTurnsPositionsRotationsAndShTogether()
    {
        var (file, _) = MakePly();
        var (p0, sh0) = Convert(file, flip: false);
        var (p1, sh1) = Convert(file, flip: true);
        var F = new Matrix4x4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
        var rest0 = SphericalHarmonics.JoinParts(sh0);
        var rest1 = SphericalHarmonics.JoinParts(sh1);
        var dirs = new[] { new Vector3(0.3f, 0.5f, 0.81f), new Vector3(-0.7f, 0.1f, 0.7f), new Vector3(0.2f, -0.9f, 0.4f) };
        for (int v = 0; v < N; v++)
        {
            int o = v * SplatFormat.Floats;
            Assert.That(new[] { p1[o], p1[o + 1], p1[o + 2] }, Is.EqualTo(new[] { p0[o], -p0[o + 1], -p0[o + 2] }));
            // Row-vector matrices (System.Numerics): the turned rotation is R0 * F (= (F R0^T)^T... i.e. F applied after R).
            var r0 = Matrix4x4.CreateFromQuaternion(new Quaternion(p0[o + 10], p0[o + 11], p0[o + 12], p0[o + 13]));
            var r1 = Matrix4x4.CreateFromQuaternion(new Quaternion(p1[o + 10], p1[o + 11], p1[o + 12], p1[o + 13]));
            var want = r0 * F;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.That(r1[i, j], Is.EqualTo(want[i, j]).Within(1e-5f), $"v{v} rotation [{i},{j}]");
            var dc = p0.AsSpan(o + 3, 3).ToArray();
            foreach (var d0 in dirs)
            {
                var d = Vector3.Normalize(d0);
                var seen0 = SphericalHarmonics.EvalRgb(3, dc, rest0.AsSpan(v * 45, 45), d);
                var seen1 = SphericalHarmonics.EvalRgb(3, dc, rest1.AsSpan(v * 45, 45), new Vector3(d.X, -d.Y, -d.Z));
                Assert.That(Vector3.Distance(seen0, seen1), Is.LessThan(1e-5f), $"v{v} dir {d}: {seen0} vs {seen1}");
            }
        }
    }
}
