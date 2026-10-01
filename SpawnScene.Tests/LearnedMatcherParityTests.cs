using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnScene.Tests;

/// <summary>
/// RaCo-ALIKED + LightGlue+ (chosen front end, Research/sfm-front-end-drjohnson-2026-09-30.md) through SpawnDev.ILGPU.ML
/// must reproduce onnxruntime. The fused LightGlue-ONNX pipeline is split into a per-image EXTRACTOR and a per-pair MATCHER
/// (the halves reproduce the fused model bit for bit under onnxruntime). Data: _scratch/lg (models + ref/ from
/// onnxruntime, ORT_DISABLE_ALL); skipped without it.
/// </summary>
public class LearnedMatcherParityTests
{
    static string? Dir()
    {
        var d = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
        while (d != null && !File.Exists(Path.Combine(d.FullName, "_scratch", "lg", "raco_aliked_k2048_extractor.onnx"))) d = d.Parent;
        return d == null ? null : Path.Combine(d.FullName, "_scratch", "lg");
    }

    static float[] ReadF32(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    [Test]
    public async Task Extractor_MatchesOnnxRuntime()
    {
        var dir = Dir();
        if (dir == null) Assert.Ignore("_scratch/lg not present");
        var images = ReadF32(Path.Combine(dir, "ref", "images_2x3x416x640.f32"));
        var refKp = ReadF32(Path.Combine(dir, "ref", "keypoints_2x2048x2.f32"));
        var refDesc = ReadF32(Path.Combine(dir, "ref", "desc_2x2048x128.f32"));

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var session = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(Path.Combine(dir, "raco_aliked_k2048_extractor.onnx")),
            null, new Dictionary<string, int[]> { ["images"] = new[] { 2, 3, 416, 640 } });
        TestContext.Out.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s; inputs [{string.Join(", ", session.InputNames)}] outputs [{string.Join(", ", session.OutputNames)}]");
        using var inBuf = accel.Allocate1D(images);
        sw.Restart();
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["images"] = new Tensor(inBuf.View, new[] { 2, 3, 416, 640 }) });
        await accel.SynchronizeAsync();
        TestContext.Out.WriteLine($"ran in {sw.Elapsed.TotalSeconds:F1}s");
        var kp = outs["keypoints"].Data.GetAsArray1D();
        var desc = outs["div_87"].Data.GetAsArray1D();
        Assert.That(kp.Length, Is.EqualTo(refKp.Length), "keypoints size");
        Assert.That(desc.Length, Is.EqualTo(refDesc.Length), "descriptor size");

        // Compare by POSITION, not row: the first keypoint TopK is sorted=0 and its k=2304 boundary has ~900-2200
        // candidates per row TIED at the boundary score. ONNX does not define which tied candidates sorted=0 keeps;
        // onnxruntime's CPU kernel (nth_element) keeps an arbitrary subset, ILGPU.ML the lowest indices. Measured
        // 2026-10-01: 4060 of 4096 keypoints coincide (<= 6.1e-5 px), and their descriptors agree to 1.7e-5.
        const int K = 2048, D = 128;
        int matched = 0; double posMax = 0, descMax = 0;
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < K; i++)
            {
                float x = kp[(b * K + i) * 2], y = kp[(b * K + i) * 2 + 1];
                int best = -1; double bestD = double.MaxValue;
                for (int j = 0; j < K; j++)
                {
                    double d = Math.Max(Math.Abs(refKp[(b * K + j) * 2] - x), Math.Abs(refKp[(b * K + j) * 2 + 1] - y));
                    if (d < bestD) { bestD = d; best = j; }
                }
                if (bestD > 0.01) continue;
                matched++;
                posMax = Math.Max(posMax, bestD);
                for (int c = 0; c < D; c++)
                    descMax = Math.Max(descMax, Math.Abs(desc[(b * K + i) * D + c] - refDesc[(b * K + best) * D + c]));
            }
        TestContext.Out.WriteLine($"keypoints coinciding with onnxruntime: {matched} / {2 * K} (max {posMax:G3} px); their descriptors max |diff| {descMax:G3}");
        Assert.That(matched, Is.GreaterThanOrEqualTo(2 * K * 98 / 100), "keypoints must coincide with onnxruntime (tied-score selection aside)");
        Assert.That(posMax, Is.LessThan(1e-3), "coinciding keypoints must agree to sub-millipixel");
        Assert.That(descMax, Is.LessThan(1e-4), "descriptors of coinciding keypoints must match onnxruntime");
    }
}
