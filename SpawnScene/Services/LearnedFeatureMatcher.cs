using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The learned SfM front end: RaCo-ALIKED keypoints + descriptors once per image, LightGlue+ per image pair, both through
/// SpawnDev.ILGPU.ML on the device. The models are Kornia's split of fabio-sim's fused export (huggingface.co/kornia/raco-aliked
/// and kornia/lightglue, fetched through the hub like every other model), proven equal to onnxruntime by
/// SpawnScene.Tests.KorniaRacoLightGlueParityTests. Chosen 2026-09-30 on DrJohnson (Research/sfm-front-end-drjohnson-2026-09-30.md):
/// 43 of 44 cameras at a 1.9 deg median from rotation averaging, against SIFT's 33 deg.
///
/// The output is the same as the FAST/BRIEF front end's - <see cref="ImportedImage.Features"/> and per-pair
/// <see cref="FeatureMatch"/> lists - so verification (GpuEpipolarRansac), the global init and BA are unchanged.
/// Descriptors stay on the device (<see cref="ImageDescriptors"/>); only keypoints and the match lists come back.
///
/// Licences: RaCo, LightGlue and the export are Apache-2.0; the ALIKED descriptor weights are BSD-3-Clause, whose notice
/// ships in wwwroot/licenses/THIRD-PARTY-NOTICES.md (required for redistribution in binary form).
/// </summary>
public sealed class LearnedFeatureMatcher : IDisposable
{
    public const string ExtractorRepo = "kornia/raco-aliked";
    public const string MatcherRepo = "kornia/lightglue";

    /// <summary>Keypoints per image: 1024 or 3072, the two budgets Kornia publishes (K is baked into both graphs, and at
    /// 3072 RaCo's ranker is bypassed). The matcher costs O(K^2) per pair. &amp;lgk=N.</summary>
    public static int KeypointBudget { get; set; } = 1024;

    /// <summary>Image pairs per matcher run (the graph takes [2P, 1, K, ...]). A short final batch is padded.</summary>
    public static int PairsPerRun { get; set; } = 8;

    /// <summary>
    /// How the matcher's 2P batch rows form pairs: true = interleaved (rows 2p, 2p+1 are pair p - fabio-sim's
    /// <c>desc[0::2], desc[1::2]</c>), false = halves (rows p, P+p). Proven by
    /// SpawnScene.Tests.KorniaRacoLightGlueParityTests.Matcher_PairLayout.
    /// </summary>
    public const bool InterleavedPairs = true;

    /// <summary>Extractor input sides are multiples of this.</summary>
    public const int InputMultiple = 32;

    /// <summary>One image's matcher inputs, on the device: normalized keypoints [K, 2] and descriptors [K, 128].</summary>
    public sealed class ImageDescriptors : IDisposable
    {
        public required int K { get; init; }
        public required MemoryBuffer1D<float, Stride1D.Dense> Normalized { get; init; }
        public required MemoryBuffer1D<float, Stride1D.Dense> Descriptors { get; init; }
        public void Dispose() { Normalized.Dispose(); Descriptors.Dispose(); }
    }

    private readonly Func<Accelerator> _accel;
    private readonly IModelSource _models;
    private readonly Dictionary<(int K, int W, int H), InferenceSession> _extractors = new();
    private InferenceSession? _matcher;
    private (int K, int P) _matcherShape;
    private ImagePreprocessKernel? _preprocess;

    /// <summary>Progress / model-load lines (the import service routes them to its status and the console).</summary>
    public Action<string>? OnStatus { get; set; }

    /// <summary>On the accelerator <paramref name="accelerator"/> returns when first needed (the app: GpuService's
    /// WebGPU accelerator, see Program.cs; the tests: the ILGPU CPU accelerator, against onnxruntime).</summary>
    public LearnedFeatureMatcher(Func<Accelerator> accelerator, IModelSource models)
    {
        _accel = accelerator;
        _models = models;
    }

    private Accelerator Accel => _accel();

    /// <summary>The extractor input for an image detected at <paramref name="w"/> x <paramref name="h"/>: each side
    /// rounded to the nearest multiple of <see cref="InputMultiple"/>.</summary>
    public static (int W, int H) InputSize(int w, int h)
    {
        static int R(int v) => Math.Max(InputMultiple, (int)MathF.Round(v / (float)InputMultiple) * InputMultiple);
        return (R(w), R(h));
    }

    private async Task<InferenceSession> LoadAsync(string repo, string file, Dictionary<string, int[]> shapes)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        OnStatus?.Invoke($"Loading {repo}/{file}...");
        await using var stream = await _models.OpenAsync(repo, file);
        var session = await InferenceSession.CreateFromStreamAsync(Accel, stream, inputShapes: shapes);
        OnStatus?.Invoke($"Loaded {repo}/{file} in {sw.Elapsed.TotalSeconds:F1}s");
        return session;
    }

    private async Task<InferenceSession> ExtractorAsync(int k, int w, int h)
    {
        if (_extractors.TryGetValue((k, w, h), out var s)) return s;
        s = await LoadAsync(ExtractorRepo, $"raco_aliked_extractor_k{k}.onnx",
            new Dictionary<string, int[]> { ["images"] = new[] { 1, 3, h, w } });
        _extractors[(k, w, h)] = s;
        return s;
    }

    private async Task<InferenceSession> MatcherAsync(int k, int p)
    {
        if (_matcher != null && _matcherShape == (k, p)) return _matcher;
        _matcher?.Dispose();
        _matcher = null;
        _matcher = await LoadAsync(MatcherRepo, $"lightglue_matcher_k{k}.onnx", new Dictionary<string, int[]>
        {
            ["normalized_keypoints"] = new[] { 2 * p, 1, k, 2 },
            ["descriptors"] = new[] { 2 * p, 1, k, 128 },
        });
        _matcherShape = (k, p);
        return _matcher;
    }

    /// <summary>
    /// Keypoints for <paramref name="rgba"/> (packed RGBA, <paramref name="width"/> x <paramref name="height"/>, on the
    /// device) at FEATURE resolution (<paramref name="featureWidth"/> x <paramref name="featureHeight"/>, the frame the
    /// FAST/BRIEF detector would have seen - callers scale both front ends' features the same way). The image is resized
    /// on the device to <see cref="InputSize"/>; keypoints map back through the same half-pixel convention as that resize.
    /// Replaces <paramref name="img"/>'s <see cref="ImportedImage.LearnedDescriptors"/>.
    /// </summary>
    public async Task<List<ImageFeature>> ExtractAsync(ImportedImage img, MemoryBuffer1D<int, Stride1D.Dense> rgba,
        int width, int height, int featureWidth, int featureHeight)
    {
        int k = KeypointBudget;
        var (inW, inH) = InputSize(featureWidth, featureHeight);
        var session = await ExtractorAsync(k, inW, inH);
        _preprocess ??= new ImagePreprocessKernel(Accel);
        using var input = Accel.Allocate1D<float>(3L * inH * inW);
        _preprocess.ForwardNormalized01(rgba.View, input.View, width, height, inW, inH);
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["images"] = new Tensor(input.View, new[] { 1, 3, inH, inW }),
        });

        // The session's outputs are its own (reused by the next run): keep this image's matcher inputs in our buffers.
        var normalized = Accel.Allocate1D<float>(k * 2);
        var descriptors = Accel.Allocate1D<float>(k * 128);
        await normalized.View.CopyFromAsync(outs["normalized_keypoints"].Data.SubView(0, k * 2));
        await descriptors.View.CopyFromAsync(outs["descriptors"].Data.SubView(0, k * 128));
        using var kpHost = Accel.Allocate1D<float>(k * 2);
        await kpHost.View.CopyFromAsync(outs["keypoints"].Data.SubView(0, k * 2));
        await Accel.SynchronizeAsync();
        // CPU transfer: keypoint positions (K x 2 floats) - SfM's tracks, triangulation and BA are host-side.
        var kp = await kpHost.View.CopyToHostAsync();

        img.LearnedDescriptors?.Dispose();
        img.LearnedDescriptors = new ImageDescriptors { K = k, Normalized = normalized, Descriptors = descriptors };

        float sx = featureWidth / (float)inW, sy = featureHeight / (float)inH;
        var features = new List<ImageFeature>(k);
        for (int i = 0; i < k; i++)
            features.Add(new ImageFeature
            {
                X = (kp[i * 2] + 0.5f) * sx - 0.5f,
                Y = (kp[i * 2 + 1] + 0.5f) * sy - 0.5f,
                Score = 1f,
            });
        return features;
    }

    /// <summary>
    /// LightGlue+ on every pair in <paramref name="pairs"/> (indices into <paramref name="images"/>, each extracted by
    /// <see cref="ExtractAsync"/>), <see cref="PairsPerRun"/> per graph run. <paramref name="onPair"/> receives each pair's
    /// matches (mutual, above LightGlue's own filter threshold - both are inside the graph); Distance is
    /// round((1 - score) * 1000), so lower is better as for BRIEF's Hamming distance.
    /// </summary>
    public async Task MatchPairsAsync(IReadOnlyList<ImportedImage> images, IReadOnlyList<(int A, int B)> pairs,
        Action<int, List<FeatureMatch>> onPair, Func<int, Task>? onProgress = null)
    {
        if (pairs.Count == 0) return;
        int k = images[pairs[0].A].LearnedDescriptors?.K ?? throw new InvalidOperationException(
            $"{images[pairs[0].A].FileName} has no learned descriptors - run ExtractAsync first");
        int p = Math.Max(1, PairsPerRun);
        var session = await MatcherAsync(k, p);
        using var nkpIn = Accel.Allocate1D<float>(2L * p * k * 2);
        using var descIn = Accel.Allocate1D<float>(2L * p * k * 128);
        using var matchesOut = Accel.Allocate1D<float>((long)p * k);
        using var scoresOut = Accel.Allocate1D<float>((long)p * k);
        var nkpShape = new[] { 2 * p, 1, k, 2 };
        var descShape = new[] { 2 * p, 1, k, 128 };

        for (int start = 0; start < pairs.Count; start += p)
        {
            int count = Math.Min(p, pairs.Count - start);
            for (int s = 0; s < p; s++)
            {
                // A short final batch repeats its last pair; those rows are never read.
                var (a, b) = pairs[start + Math.Min(s, count - 1)];
                for (int side = 0; side < 2; side++)
                {
                    var img = images[side == 0 ? a : b];
                    var d = img.LearnedDescriptors ?? throw new InvalidOperationException($"{img.FileName} has no learned descriptors");
                    if (d.K != k) throw new InvalidOperationException($"{img.FileName}: K={d.K}, the batch is K={k}");
                    int row = InterleavedPairs ? 2 * s + side : side * p + s;
                    await nkpIn.View.SubView((long)row * k * 2, k * 2).CopyFromAsync(d.Normalized.View);
                    await descIn.View.SubView((long)row * k * 128, k * 128).CopyFromAsync(d.Descriptors.View);
                }
            }
            var outs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                ["normalized_keypoints"] = new Tensor(nkpIn.View, nkpShape),
                ["descriptors"] = new Tensor(descIn.View, descShape),
            });
            await matchesOut.View.CopyFromAsync(outs["matches0"].Data.SubView(0, p * k));
            await scoresOut.View.CopyFromAsync(outs["mscores0"].Data.SubView(0, p * k));
            await Accel.SynchronizeAsync();
            // CPU transfer: the match lists (P x K index + score) - SfM consumes them on the host.
            var m = await matchesOut.View.SubView(0, count * k).CopyToHostAsync();
            var sc = await scoresOut.View.SubView(0, count * k).CopyToHostAsync();
            for (int s = 0; s < count; s++)
            {
                var list = new List<FeatureMatch>();
                for (int i = 0; i < k; i++)
                {
                    int j = (int)m[s * k + i];
                    if (j < 0 || j >= k) continue;
                    list.Add(new FeatureMatch
                    {
                        IndexA = i,
                        IndexB = j,
                        Distance = (int)MathF.Round((1f - Math.Clamp(sc[s * k + i], 0f, 1f)) * 1000f),
                    });
                }
                onPair(start + s, list);
            }
            if (onProgress != null) await onProgress(start + count);
        }
    }

    public void Dispose()
    {
        foreach (var s in _extractors.Values) s.Dispose();
        _extractors.Clear();
        _matcher?.Dispose();
        _matcher = null;
        _preprocess?.Dispose();
        _preprocess = null;
    }
}
