using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnScene.Formats;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// SOG decode (SogImport kernel, CPU accelerator) against a C# port of playcanvas/engine GSplatSogIterator.read +
/// decompress() on random texels, version 2 (codebooks) and version 1 (ranges), SH degree 3, textures wider than the
/// splats need (rows padded as copyTextureToBuffer pads them). And the meta.json parser on both shapes.
/// </summary>
public class SogImportTests
{
    const int N = 150, W = 13;   // 13 x 12 texels hold 150 splats; rows padded to 64 words

    static float Lerp(float a, float b, float t) => a * (1 - t) + b * t;

    static SogMeta Meta(int version, Random rng)
    {
        float[] R(int n, float lo, float hi) => Enumerable.Range(0, n).Select(_ => lo + (float)rng.NextDouble() * (hi - lo)).ToArray();
        var cb = R(256, -2f, 2f); Array.Sort(cb);
        return new SogMeta
        {
            Version = version, Count = N, MeansMin = R(3, -2f, -1f), MeansMax = R(3, 1f, 2f), MeansFiles = new[] { "a", "b" },
            ScalesCodebook = version == 2 ? R(256, -8f, 0f) : null, ScalesMin = R(3, -8f, -5f), ScalesMax = R(3, -3f, 0f), ScalesFiles = new[] { "s" },
            QuatsFiles = new[] { "q" }, Sh0Codebook = version == 2 ? cb : null, Sh0Min = R(4, -2f, -1f), Sh0Max = R(4, 1f, 2f), Sh0Files = new[] { "c" },
            ShNCodebook = version == 2 ? R(256, -1f, 1f) : null, ShNMin = -0.7f, ShNMax = 0.9f, ShNFiles = new[] { "cent", "lab" },
        };
    }

    [TestCase(2)]
    [TestCase(1)]
    public void MatchesPlayCanvasDecode(int version)
    {
        var rng = new Random(5 + version);
        var m = Meta(version, rng);
        const int Row = 64, Rows = (N + W - 1) / W, Coeffs = 15, CentW = 64 * Coeffs, CentRows = 4, CentRow = 960;
        uint Rand() => (uint)rng.NextInt64(0, 1L << 32);
        var tex = new uint[Row * Rows * 6 + CentRow * CentRows];
        int Off(int k) => k * Row * Rows;
        for (int k = 0; k < 6; k++)
            for (int i = 0; i < N; i++)
            {
                uint v = Rand();
                if (k == 2) v = (v & 0x00FFFFFFu) | ((uint)(252 + rng.Next(4)) << 24);   // quats: alpha = 252 + mode
                if (k == 5) v = (uint)rng.Next(CentRows * 64) | (v & 0xFFFF0000u);         // labels: a palette index
                tex[Off(k) + (i / W) * Row + i % W] = v;
            }
        int centOff = Off(6);
        for (int j = 0; j < CentRow * CentRows; j++) tex[centOff + j] = Rand();
        SogImport.Tex T(int k) => new() { Off = Off(k), Row = Row, Width = W };
        var prm = SogImport.MakeParams(m, T(0), T(1), T(2), T(3), T(4), T(5), new SogImport.Tex { Off = centOff, Row = CentRow, Width = CentW }, Coeffs, false);

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var a = context.CreateCPUAccelerator(0);
        using var t = a.Allocate1D(tex);
        using var books = a.Allocate1D(SogImport.Codebooks(m));
        using var packed = a.Allocate1D<float>(N * SplatFormat.Floats);
        var sh = Enumerable.Range(0, 3).Select(_ => a.Allocate1D<float>(N * SphericalHarmonics.PartFloatsPerSplat)).ToArray();
        SogImport.Run(a, t.View, books.View, prm, packed.View, sh[0].View, sh[1].View, sh[2].View);
        a.Synchronize();
        var p = packed.GetAsArray1D();
        var rest = SphericalHarmonics.JoinParts(sh.Select(x => x.GetAsArray1D()).ToArray());
        foreach (var x in sh) x.Dispose();

        uint Px(int k, int i) => tex[Off(k) + (i / W) * Row + i % W];
        static int B(uint v, int c) => (int)((v >> (8 * c)) & 0xFF);
        for (int i = 0; i < N; i++)
        {
            // The oracle - GSplatSogIterator.read, then decompress()'s conversions and our row layout.
            var want = new float[14];
            for (int c = 0; c < 3; c++)
            {
                float n = Lerp(m.MeansMin[c], m.MeansMax[c], ((B(Px(1, i), c) << 8) + B(Px(0, i), c)) / 65535f);
                want[c] = MathF.Sign(n) * (MathF.Exp(MathF.Abs(n)) - 1f);
                want[6 + c] = MathF.Exp(version == 2 ? m.ScalesCodebook![B(Px(3, i), c)] : Lerp(m.ScalesMin[c], m.ScalesMax[c], B(Px(3, i), c) / 255f));
                want[3 + c] = version == 2 ? m.Sh0Codebook![B(Px(4, i), c)] : Lerp(m.Sh0Min[c], m.Sh0Max[c], B(Px(4, i), c) / 255f);
            }
            want[9] = version == 2 ? B(Px(4, i), 3) / 255f : 1f / (1f + MathF.Exp(-Lerp(m.Sh0Min[3], m.Sh0Max[3], B(Px(4, i), 3) / 255f)));
            float sq = MathF.Sqrt(2f);
            float qa = (B(Px(2, i), 0) / 255f - 0.5f) * sq, qb = (B(Px(2, i), 1) / 255f - 0.5f) * sq, qc = (B(Px(2, i), 2) / 255f - 0.5f) * sq;
            float qd = MathF.Sqrt(MathF.Max(0, 1 - (qa * qa + qb * qb + qc * qc)));
            (float x, float y, float z, float w) q = (B(Px(2, i), 3) - 252) switch { 0 => (qa, qb, qc, qd), 1 => (qd, qb, qc, qa), 2 => (qb, qd, qc, qa), _ => (qb, qc, qd, qa) };
            float len = MathF.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            want[10] = q.x / len; want[11] = q.y / len; want[12] = q.z / len; want[13] = q.w / len;
            for (int k = 0; k < 14; k++)
                Assert.That(p[i * 14 + k], Is.EqualTo(want[k]).Within(2e-5f * MathF.Max(1f, MathF.Abs(want[k]))), $"#{i} slot {k}");
            int lab = B(Px(5, i), 0) + (B(Px(5, i), 1) << 8);
            int u = (lab % 64) * Coeffs, v = lab / 64;
            for (int k = 0; k < Coeffs; k++)
                for (int c = 0; c < 3; c++)
                {
                    int by = B(tex[centOff + v * CentRow + u + k], c);
                    float w = version == 2 ? m.ShNCodebook![by] : Lerp(m.ShNMin, m.ShNMax, by / 255f);
                    Assert.That(rest[i * 45 + k * 3 + c], Is.EqualTo(w).Within(1e-6f), $"#{i} sh {k}/{c}");
                }
        }
    }

    [Test]
    public void MetaJsonParsesBothShapes()
    {
        string v2 = "{\"version\":2,\"count\":5,\"means\":{\"mins\":[0,0,0],\"maxs\":[1,1,1],\"files\":[\"means_l.webp\",\"means_u.webp\"]}," +
            "\"scales\":{\"codebook\":[" + string.Join(",", Enumerable.Range(0, 256)) + "],\"files\":[\"scales.webp\"]}," +
            "\"quats\":{\"files\":[\"quats.webp\"]},\"sh0\":{\"codebook\":[" + string.Join(",", Enumerable.Range(0, 256)) + "],\"files\":[\"sh0.webp\"]}}";
        var m2 = SogMeta.Parse(v2);
        Assert.That(m2.Version, Is.EqualTo(2)); Assert.That(m2.Count, Is.EqualTo(5)); Assert.That(m2.ShNFiles, Is.Empty);
        Assert.That(m2.ScalesCodebook![255], Is.EqualTo(255f));
        string v1 = "{\"means\":{\"shape\":[7,3],\"mins\":[0,0,0],\"maxs\":[1,1,1],\"files\":[\"means_l.webp\",\"means_u.webp\"]}," +
            "\"scales\":{\"mins\":[-5,-5,-5],\"maxs\":[0,0,0],\"files\":[\"scales.webp\"]},\"quats\":{\"files\":[\"quats.webp\"]}," +
            "\"sh0\":{\"mins\":[-1,-1,-1,-4],\"maxs\":[1,1,1,4],\"files\":[\"sh0.webp\"]},\"shN\":{\"mins\":-0.5,\"maxs\":0.5,\"files\":[\"c.webp\",\"l.webp\"]}}";
        var m1 = SogMeta.Parse(v1);
        Assert.That(m1.Version, Is.EqualTo(1)); Assert.That(m1.Count, Is.EqualTo(7)); Assert.That(m1.Sh0Max[3], Is.EqualTo(4f));
        Assert.That(m1.ShNMin, Is.EqualTo(-0.5f)); Assert.That(m1.ShNFiles, Is.EqualTo(new[] { "c.webp", "l.webp" }));
    }
}
