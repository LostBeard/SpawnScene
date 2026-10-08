using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Formats;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// PlayCanvas compressed PLY (GaussianPly.ParseCompressed + GaussianPlyImport.RunCompressed, CPU accelerator) against
/// a C# port of playcanvas/engine's SplatCompressedIterator.read + decompress() as the oracle, on random packed words,
/// 300 splats (two chunks), both chunk layouts (12 and 18 floats), SH degree 3, a header of odd length. And a real
/// SuperSplat export's header, when the sample is on this machine.
/// </summary>
public class CompressedPlyImportTests
{
    const int N = 300;

    static float Unorm(uint v, int bits) { uint m = (1u << bits) - 1u; return (v & m) / (float)m; }
    static float Lerp(float a, float b, float t) => a * (1 - t) + b * t;

    static byte[] Make(int chunkProps, int shPerChannel, out float[] chunks, out uint[] verts, out byte[] sh)
    {
        var rng = new Random(11);
        int nChunks = (N + 255) / 256;
        chunks = new float[nChunks * chunkProps];
        for (int c = 0; c < nChunks; c++)
            for (int k = 0; k < chunkProps; k++)
            {
                bool isMax = (k % 6) >= 3;
                float lo = k < 6 ? -3f : k < 12 ? -6f : 0f;
                chunks[c * chunkProps + k] = lo + (float)rng.NextDouble() * 2f + (isMax ? 2f : 0f);
            }
        verts = new uint[N * 4];
        for (int i = 0; i < verts.Length; i++) verts[i] = (uint)rng.NextInt64(0, 1L << 32);
        sh = new byte[N * 3 * shPerChannel];
        rng.NextBytes(sh);
        var head = new StringBuilder("ply\nformat binary_little_endian 1.0\ncomment odd\n");
        head.Append($"element chunk {nChunks}\n");
        string[] names = { "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "min_scale_x", "min_scale_y", "min_scale_z",
            "max_scale_x", "max_scale_y", "max_scale_z", "min_r", "min_g", "min_b", "max_r", "max_g", "max_b" };
        for (int k = 0; k < chunkProps; k++) head.Append($"property float {names[k]}\n");
        head.Append($"element vertex {N}\nproperty uint packed_position\nproperty uint packed_rotation\nproperty uint packed_scale\nproperty uint packed_color\n");
        if (shPerChannel > 0)
        {
            head.Append($"element sh {N}\n");
            for (int k = 0; k < shPerChannel * 3; k++) head.Append($"property uchar f_rest_{k}\n");
        }
        head.Append("end_header\n");
        while (head.Length % 4 != 3) head.Insert(head.ToString().IndexOf("\nelement", StringComparison.Ordinal), "!");
        using var ms = new MemoryStream();
        ms.Write(Encoding.ASCII.GetBytes(head.ToString()));
        foreach (var f in chunks) ms.Write(BitConverter.GetBytes(f));
        foreach (var v in verts) ms.Write(BitConverter.GetBytes(v));
        ms.Write(sh);
        return ms.ToArray();
    }

    /// <summary>The oracle: SplatCompressedIterator.read, then decompress()'s conversions, then our row layout.</summary>
    static (float[] Row, float[] ShRest) Oracle(int i, float[] ch, int props, uint[] vt, byte[] sh, int spc)
    {
        int ci = (i / 256) * props;
        uint pp = vt[i * 4], rr = vt[i * 4 + 1], ss = vt[i * 4 + 2], cc = vt[i * 4 + 3];
        var row = new float[14];
        row[0] = Lerp(ch[ci], ch[ci + 3], Unorm(pp >> 21, 11));
        row[1] = Lerp(ch[ci + 1], ch[ci + 4], Unorm(pp >> 11, 10));
        row[2] = Lerp(ch[ci + 2], ch[ci + 5], Unorm(pp, 11));
        float sq = MathF.Sqrt(2f);
        float a = (Unorm(rr >> 20, 10) - 0.5f) * sq, b = (Unorm(rr >> 10, 10) - 0.5f) * sq, c = (Unorm(rr, 10) - 0.5f) * sq;
        float m = MathF.Sqrt(1f - (a * a + b * b + c * c));   // the JS has no clamp; a NaN there would be NaN here too
        (float x, float y, float z, float w) q = (rr >> 30) switch { 0 => (a, b, c, m), 1 => (m, b, c, a), 2 => (b, m, c, a), _ => (b, c, m, a) };
        row[6] = MathF.Exp(Lerp(ch[ci + 6], ch[ci + 9], Unorm(ss >> 21, 11)));
        row[7] = MathF.Exp(Lerp(ch[ci + 7], ch[ci + 10], Unorm(ss >> 11, 10)));
        row[8] = MathF.Exp(Lerp(ch[ci + 8], ch[ci + 11], Unorm(ss, 11)));
        float r = Unorm(cc >> 24, 8), g = Unorm(cc >> 16, 8), bl = Unorm(cc >> 8, 8), al = Unorm(cc, 8);
        if (props > 12) { r = Lerp(ch[ci + 12], ch[ci + 15], r); g = Lerp(ch[ci + 13], ch[ci + 16], g); bl = Lerp(ch[ci + 14], ch[ci + 17], bl); }
        const float C0 = 0.28209479177387814f;
        row[3] = (r - 0.5f) / C0; row[4] = (g - 0.5f) / C0; row[5] = (bl - 0.5f) / C0;
        row[9] = al;
        row[10] = q.x; row[11] = q.y; row[12] = q.z; row[13] = q.w;
        var rest = new float[45];
        for (int k = 1; k <= 15; k++)
            for (int ch3 = 0; ch3 < 3; ch3++)
                rest[(k - 1) * 3 + ch3] = k <= spc ? sh[i * 3 * spc + ch3 * spc + k - 1] * (8f / 255f) - 4f : 0f;
        return (row, rest);
    }

    [TestCase(18, 15)]
    [TestCase(12, 0)]
    [TestCase(18, 3)]
    public void MatchesPlayCanvasDecode(int chunkProps, int shPerChannel)
    {
        var file = Make(chunkProps, shPerChannel, out var chunks, out var verts, out var sh);
        var L = GaussianPly.ParseCompressed(file.AsSpan(0, Math.Min(file.Length, 65536)));
        Assert.That(L, Is.Not.Null);
        Assert.That(L!.Count, Is.EqualTo(N));
        Assert.That(L.HeaderBytes % 4, Is.EqualTo(3));
        Assert.That(L.Bytes, Is.EqualTo(file.Length));
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        uint[] Words(long start, long len, out int first)
        {
            long bs = start & ~3L;
            var buf = new byte[(start + len - bs + 3) / 4 * 4 + 8];
            Array.Copy(file, bs, buf, 0, Math.Min(start + len, file.Length) - bs);
            var w = new uint[buf.Length / 4];
            Buffer.BlockCopy(buf, 0, w, 0, buf.Length);
            first = (int)(start - bs);
            return w;
        }
        using var cw = a.Allocate1D(Words(L.ChunkOffset, 4L * L.Chunks * L.ChunkProps, out int cFirst));
        using var vw = a.Allocate1D(Words(L.VertexOffset, 16L * N, out int vFirst));
        using var sw = a.Allocate1D(Words(L.ShOffset, Math.Max(1, 3L * shPerChannel * N), out int sFirst));
        using var packed = a.Allocate1D<float>(N * SplatFormat.Floats);
        var shb = Enumerable.Range(0, 3).Select(_ => a.Allocate1D<float>(N * SphericalHarmonics.PartFloatsPerSplat)).ToArray();
        GaussianPlyImport.RunCompressed(a, cw.View, cFirst, vw.View, vFirst, sw.View, sFirst, 0, N, L, false, packed.View, shb[0].View, shb[1].View, shb[2].View);
        a.Synchronize();
        var p = packed.GetAsArray1D();
        var rest = SphericalHarmonics.JoinParts(shb.Select(x => x.GetAsArray1D()).ToArray());
        foreach (var x in shb) x.Dispose();
        int checkedRows = 0;
        for (int i = 0; i < N; i++)
        {
            var (row, wantRest) = Oracle(i, chunks, chunkProps, verts, sh, shPerChannel);
            if (row.Any(float.IsNaN)) continue;   // random words can be off the unit sphere; the JS returns NaN there
            checkedRows++;
            for (int k = 0; k < 14; k++)
                Assert.That(p[i * 14 + k], Is.EqualTo(row[k]).Within(1e-4f * MathF.Max(1f, MathF.Abs(row[k]))), $"#{i} slot {k}");
            if (shPerChannel > 0)
                for (int k = 0; k < 45; k++) Assert.That(rest[i * 45 + k], Is.EqualTo(wantRest[k]).Within(1e-5f), $"#{i} sh {k}");
        }
        Assert.That(checkedRows, Is.GreaterThan(N / 4));
    }

    [Test]
    public void ARealSuperSplatHeaderIsRecognised()
    {
        string path = Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..", "_models_scratch", "pc", "biker.compressed.ply");
        if (!File.Exists(path)) Assert.Ignore("sample not on this machine (playcanvas/engine examples/assets/splats)");
        var head = File.ReadAllBytes(path)[..4096];
        var L = GaussianPly.ParseCompressed(head);
        Assert.That(L, Is.Not.Null);
        Assert.That(L!.Count, Is.EqualTo(152746));
        Assert.That(L.ChunkProps, Is.EqualTo(18));
        Assert.That(L.Bytes, Is.EqualTo(new FileInfo(path).Length));
        Assert.Throws<FormatException>(() => GaussianPly.Parse(head));   // the plain parser refuses it with a reason
    }
}
