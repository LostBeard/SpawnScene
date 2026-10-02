using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnScene.Tests;

/// <summary>
/// The SHIPPABLE front end: Kornia's published RaCo-ALIKED extractor + LightGlue+ matcher halves
/// (huggingface.co/kornia/raco-aliked + kornia/lightglue, served through the hub's /hf proxy like every other model),
/// through SpawnDev.ILGPU.ML, against onnxruntime. Same split as LearnedMatcherParityTests' hand-cut k2048 files, but
/// published, at K = 1024 / 3072, and the matcher is cut BEFORE LightGlue's NonZero: static matches0 [P,K] (index into
/// image 1, or -1) and mscores0 [P,K]. Data: _scratch/kornia (models via the hub + ref/ from onnxruntime with
/// ORT_DISABLE_ALL, same DrJohnson pair as _scratch/lg/ref); skipped without it.
/// </summary>
public class KorniaRacoLightGlueParityTests
{
    static string? Dir()
    {
        var d = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
        while (d != null && !File.Exists(Path.Combine(d.FullName, "_scratch", "kornia", "raco_aliked_extractor_k1024.onnx"))) d = d.Parent;
        return d == null ? null : Path.Combine(d.FullName, "_scratch");
    }

    static float[] ReadF32(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    static int[] ReadI32(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new int[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    // stream: through InferenceSession.CreateFromStreamAsync, the loader the app's hub path (IModelSource) uses. It
    // threw at load on optimizer-folded constants (ML 5.3.1-local.9) while the byte-array loader worked.
    [TestCase(1024, false)]
    [TestCase(3072, false)]
    [TestCase(1024, true)]
    public async Task Extractor_MatchesOnnxRuntime(int K, bool stream)
    {
        var dir = Dir();
        if (dir == null) Assert.Ignore("_scratch/kornia not present");
        var images = ReadF32(Path.Combine(dir, "lg", "ref", "images_2x3x416x640.f32"));
        var refKp = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_kp.f32"));
        var refNkp = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_nkp.f32"));
        var refDesc = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_desc.f32"));

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        var shapes = new Dictionary<string, int[]> { ["images"] = new[] { 2, 3, 416, 640 } };
        var modelPath = Path.Combine(dir, "kornia", $"raco_aliked_extractor_k{K}.onnx");
        InferenceSession session;
        if (stream)
        {
            await using var fs = File.OpenRead(modelPath);
            session = await InferenceSession.CreateFromStreamAsync(accel, fs, inputShapes: shapes);
        }
        else session = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(modelPath), null, shapes);
        using var _ = session;
        using var inBuf = accel.Allocate1D(images);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["images"] = new Tensor(inBuf.View, new[] { 2, 3, 416, 640 }) });
        await accel.SynchronizeAsync();
        TestContext.Out.WriteLine($"k{K} extractor ran in {sw.Elapsed.TotalSeconds:F1}s");
        var kp = outs["keypoints"].Data.SubView(0, 2 * K * 2).GetAsArray1D();
        var nkp = outs["normalized_keypoints"].Data.SubView(0, 2 * K * 2).GetAsArray1D();
        var desc = outs["descriptors"].Data.SubView(0, 2 * K * 128).GetAsArray1D();

        // By POSITION (top-K ties at the selection boundary are not ordered by ONNX - see LearnedMatcherParityTests).
        int matched = 0; double posMax = 0, nkpMax = 0, descMax = 0;
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
                for (int c = 0; c < 2; c++)
                    nkpMax = Math.Max(nkpMax, Math.Abs(nkp[(b * K + i) * 2 + c] - refNkp[(b * K + best) * 2 + c]));
                for (int c = 0; c < 128; c++)
                    descMax = Math.Max(descMax, Math.Abs(desc[(b * K + i) * 128 + c] - refDesc[(b * K + best) * 128 + c]));
            }
        TestContext.Out.WriteLine($"k{K}: keypoints coinciding with onnxruntime {matched} / {2 * K} (max {posMax:G3} px); normalized max |diff| {nkpMax:G3}; descriptors max |diff| {descMax:G3}");
        Assert.That(matched, Is.GreaterThanOrEqualTo(2 * K * 98 / 100), "keypoints must coincide with onnxruntime (tied-score selection aside)");
        Assert.That(posMax, Is.LessThan(1e-3), "coinciding keypoints must agree to sub-millipixel");
        Assert.That(nkpMax, Is.LessThan(1e-5), "normalized keypoints must agree");
        Assert.That(descMax, Is.LessThan(1e-4), "descriptors of coinciding keypoints must match onnxruntime");
    }

    [TestCase(1024)]
    [TestCase(3072)]
    public async Task Matcher_MatchesOnnxRuntime(int K)
    {
        var dir = Dir();
        if (dir == null) Assert.Ignore("_scratch/kornia not present");
        var nkp = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_nkp.f32"));
        var desc = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_desc.f32"));
        var refM = ReadI32(Path.Combine(dir, "kornia", "ref", $"k{K}_matches0.i32"));
        var refS = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_mscores0.f32"));

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var session = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(Path.Combine(dir, "kornia", $"lightglue_matcher_k{K}.onnx")),
            null, new Dictionary<string, int[]> { ["normalized_keypoints"] = new[] { 2, 1, K, 2 }, ["descriptors"] = new[] { 2, 1, K, 128 } });
        using var kB = accel.Allocate1D(nkp);
        using var dB = accel.Allocate1D(desc);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["normalized_keypoints"] = new Tensor(kB.View, new[] { 2, 1, K, 2 }),
            ["descriptors"] = new Tensor(dB.View, new[] { 2, 1, K, 128 }),
        });
        await accel.SynchronizeAsync();
        TestContext.Out.WriteLine($"k{K} matcher ran in {sw.Elapsed.TotalSeconds:F1}s");
        var m = outs["matches0"]; var s = outs["mscores0"];
        Assert.That(m.Shape, Is.EqualTo(new[] { 1, K }), "matches0 shape");
        var mv = m.Data.SubView(0, K).GetAsArray1D();
        var sv = s.Data.SubView(0, K).GetAsArray1D();
        int diffMatches = 0, ours = 0, theirs = 0; double sMax = 0;
        for (int i = 0; i < K; i++)
        {
            if (mv[i] >= 0) ours++;
            if (refM[i] >= 0) theirs++;
            if ((int)mv[i] != refM[i]) diffMatches++;
            sMax = Math.Max(sMax, Math.Abs(sv[i] - refS[i]));
        }
        TestContext.Out.WriteLine($"k{K}: matched ours {ours}, onnxruntime {theirs}; differing matches0 entries {diffMatches}; mscores0 max |diff| {sMax:G3}");
        // On onnxruntime's own extractor output the matcher is deterministic: identical matches0, close scores.
        Assert.That(diffMatches, Is.EqualTo(0), "matches0 must be identical");
        Assert.That(sMax, Is.LessThan(1e-4), "mscores0 must agree");
    }

    /// <summary>The model files from _scratch/kornia, as the hub would serve them.</summary>
    sealed class ScratchModels(string dir) : SpawnDev.ILGPU.ML.Hub.IModelSource
    {
        public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
            => Task.FromResult<Stream>(File.OpenRead(Path.Combine(dir, filePath)));
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
            => File.ReadAllBytesAsync(Path.Combine(dir, filePath), cancellationToken);
    }

    /// <summary>
    /// The app's batched matching (LearnedFeatureMatcher.MatchPairsAsync, 2 pairs per run) on onnxruntime's extractor
    /// output: pair (0, 1) must give onnxruntime's matches0 exactly, and pair (0, 0) - image 0 against itself - must
    /// match keypoints to themselves. Pairing the 2P rows the wrong way (interleaved vs halves) swaps those two answers.
    /// </summary>
    [Test]
    public async Task Matcher_PairLayout_BatchedPairsMatchOnnxRuntime()
    {
        const int K = 1024;
        var dir = Dir();
        if (dir == null) Assert.Ignore("_scratch/kornia not present");
        var nkp = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_nkp.f32"));
        var desc = ReadF32(Path.Combine(dir, "kornia", "ref", $"k{K}_desc.f32"));
        var refM = ReadI32(Path.Combine(dir, "kornia", "ref", $"k{K}_matches0.i32"));

        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var matcher = new SpawnScene.Services.LearnedFeatureMatcher(() => accel, new ScratchModels(Path.Combine(dir, "kornia")));
        int oldK = SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget, oldP = SpawnScene.Services.LearnedFeatureMatcher.PairsPerRun;
        SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget = K;
        SpawnScene.Services.LearnedFeatureMatcher.PairsPerRun = 2;
        var images = new List<SpawnScene.Models.ImportedImage>();
        try
        {
            for (int b = 0; b < 2; b++)
                images.Add(new SpawnScene.Models.ImportedImage
                {
                    FileName = $"ref{b}",
                    LearnedDescriptors = new SpawnScene.Services.LearnedFeatureMatcher.ImageDescriptors
                    {
                        K = K,
                        Normalized = accel.Allocate1D(nkp[(b * K * 2)..((b + 1) * K * 2)]),
                        Descriptors = accel.Allocate1D(desc[(b * K * 128)..((b + 1) * K * 128)]),
                    },
                });
            var results = new Dictionary<int, List<SpawnScene.Models.FeatureMatch>>();
            await matcher.MatchPairsAsync(images, new[] { (0, 1), (0, 0) }, (p, list) => results[p] = list);

            var m01 = new int[K];
            Array.Fill(m01, -1);
            foreach (var fm in results[0]) m01[fm.IndexA] = fm.IndexB;
            int diff = 0;
            for (int i = 0; i < K; i++) if (m01[i] != refM[i]) diff++;
            int self = results[1].Count(fm => fm.IndexA == fm.IndexB);
            TestContext.Out.WriteLine($"pair (0,1): {results[0].Count} matches, {diff} differ from onnxruntime; pair (0,0): {results[1].Count} matches, {self} to themselves");
            Assert.That(diff, Is.EqualTo(0), "pair (0, 1) must be onnxruntime's matches0");
            Assert.That(self, Is.GreaterThanOrEqualTo(K * 9 / 10), "image 0 against itself must match keypoints to themselves");
        }
        finally
        {
            SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget = oldK;
            SpawnScene.Services.LearnedFeatureMatcher.PairsPerRun = oldP;
            foreach (var im in images) im.DisposeSource();
        }
    }

    /// <summary>
    /// One extractor session at a time (2026-10-02): a capture mixing portrait and landscape photos needs two input shapes,
    /// and two resident sessions (~4 GB of WebGPU pool each at 1024 px) lost the device on the Bathroom set. Extract a
    /// landscape and then a portrait image (random pixels): one session may remain, and both extractions must work.
    /// </summary>
    [Test]
    public async Task Extract_TwoInputShapes_KeepsOneSessionResident()
    {
        var dir = Dir();
        if (dir == null) Assert.Ignore("_scratch/kornia not present");
        using var context = Context.Create(b => b.CPU().EnableAlgorithms());
        using var accel = context.CreateCPUAccelerator(0);
        using var matcher = new SpawnScene.Services.LearnedFeatureMatcher(() => accel, new ScratchModels(Path.Combine(dir, "kornia")));
        int old = SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget;
        SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget = 1024;
        var rng = new Random(1);
        var images = new List<SpawnScene.Models.ImportedImage>();
        try
        {
            foreach (var (w, h) in new[] { (320, 224), (224, 320) })
            {
                var px = new int[w * h];
                for (int i = 0; i < px.Length; i++) px[i] = rng.Next() | unchecked((int)0xFF000000);
                using var rgba = accel.Allocate1D(px);
                var img = new SpawnScene.Models.ImportedImage { FileName = $"{w}x{h}", Width = w, Height = h };
                images.Add(img);
                var feats = await matcher.ExtractAsync(img, rgba, w, h, w, h);
                Assert.That(feats.Count, Is.EqualTo(1024), $"{w}x{h} keypoints");
                Assert.That(matcher.ExtractorSessionsResident, Is.EqualTo(1), $"after {w}x{h}: one extractor session resident");
            }
        }
        finally
        {
            SpawnScene.Services.LearnedFeatureMatcher.KeypointBudget = old;
            foreach (var im in images) im.DisposeSource();
        }
    }
}
