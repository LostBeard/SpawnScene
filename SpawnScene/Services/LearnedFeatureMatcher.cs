using ILGPU;
using ILGPU.Algorithms;
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

    /// <summary>
    /// DIAGNOSIS (&amp;lgprofile=1): time every node of the SECOND matcher run (the first compiles shaders) with a GPU sync
    /// after each (SpawnDev.ILGPU.ML GraphExecutor.PerOpSync + CapturedNodeTimingsMs), then log the cost by op type and the
    /// slowest nodes. The synced total is larger than an unsynced run; the split is what it is for.
    /// </summary>
    public static bool ProfileSecondRun { get; set; }

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
            bool profile = ProfileSecondRun && start == p;
            if (profile)
            {
                await Accel.SynchronizeAsync();
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
            }
            var swRun = System.Diagnostics.Stopwatch.StartNew();
            var outs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                ["normalized_keypoints"] = new Tensor(nkpIn.View, nkpShape),
                ["descriptors"] = new Tensor(descIn.View, descShape),
            });
            if (profile)
            {
                await Accel.SynchronizeAsync();
                var t = SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs!;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = false;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null;
                string OpOf(string key) { var parts = key.Split('_', 3); return parts.Length > 1 ? parts[1] : key; }
                double total = t.Values.Sum();
                Console.WriteLine($"[LightGlue profile] run {swRun.Elapsed.TotalMilliseconds:F0} ms wall with a sync per op; {t.Count} nodes, {total:F0} ms in nodes");
                foreach (var g in t.GroupBy(kv => OpOf(kv.Key)).OrderByDescending(g => g.Sum(kv => kv.Value)).Take(12))
                    Console.WriteLine($"[LightGlue profile]   {g.Key,-22} {g.Sum(kv => kv.Value),8:F1} ms ({g.Sum(kv => kv.Value) / total:P0}) over {g.Count()} nodes");
                foreach (var kv in t.OrderByDescending(kv => kv.Value).Take(12))
                    Console.WriteLine($"[LightGlue profile]   slowest {kv.Key}: {kv.Value:F2} ms");
            }
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

    /// <summary>
    /// Free the extractor sessions and their buffer pools (reloaded on the next <see cref="ExtractAsync"/>). The images'
    /// <see cref="ImageDescriptors"/> are separate and stay. MEASURED 2026-10-01 (DrJohnson, K=1024, 1 pair per run): the
    /// extractor + matcher sessions held 4.6 GB of WebGPU buffers after matching, and DAv3's first multi-view pass then
    /// lost the device. Call once every ExtractAsync has completed (each one ends synchronized).
    /// </summary>
    public void ReleaseExtractors()
    {
        foreach (var s in _extractors.Values) s.Dispose();
        _extractors.Clear();
        _preprocess?.Dispose();
        _preprocess = null;
    }

    /// <summary>Free the matcher session and its buffer pool (reloaded on the next <see cref="MatchPairsAsync"/>, which
    /// ends synchronized).</summary>
    public void ReleaseMatcher()
    {
        _matcher?.Dispose();
        _matcher = null;
    }

    // ---- Pair retrieval -------------------------------------------------------------------------------------------------

    /// <summary>Lowe ratio for <see cref="PairScoresAsync"/>: a mutual nearest neighbour counts when its distance is below
    /// this fraction of the second-nearest's.</summary>
    public const float RetrievalRatio = 0.9f;

    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _nearest;
    Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int, float, ArrayView1D<int, Stride1D.Dense>>? _countMutual;

    /// <summary>For descriptor i of image pa[p] (t = p * k + i): its nearest and second-nearest descriptor of image pb[p] by
    /// dot product (descriptors are unit length, so the largest dot is the nearest), and the nearest's index.</summary>
    static void NearestKernel(Index1D t, ArrayView1D<float, Stride1D.Dense> desc, ArrayView1D<int, Stride1D.Dense> pa,
        ArrayView1D<int, Stride1D.Dense> pb, int k, ArrayView1D<int, Stride1D.Dense> bestIdx, ArrayView1D<float, Stride1D.Dense> best,
        ArrayView1D<float, Stride1D.Dense> second)
    {
        int p = t / k, i = t - p * k;
        int ai = (pa[p] * k + i) * 128, b0 = pb[p] * k * 128;
        float s1 = -2f, s2 = -2f; int arg = -1;
        for (int j = 0; j < k; j++)
        {
            int bj = b0 + j * 128;
            float dot = 0f;
            for (int d = 0; d < 128; d++) dot += desc[ai + d] * desc[bj + d];
            if (dot > s1) { s2 = s1; s1 = dot; arg = j; }
            else if (dot > s2) s2 = dot;
        }
        bestIdx[t] = arg; best[t] = s1; second[t] = s2;
    }

    /// <summary>Per pair p: descriptors whose nearest neighbour in the other image points back (mutual) and passes the ratio
    /// test on Euclidean distance (|a - b| = sqrt(2 - 2 a.b) for unit vectors).</summary>
    static void CountMutualKernel(Index1D p, ArrayView1D<int, Stride1D.Dense> bestAB, ArrayView1D<float, Stride1D.Dense> best,
        ArrayView1D<float, Stride1D.Dense> second, ArrayView1D<int, Stride1D.Dense> bestBA, int k, float ratio,
        ArrayView1D<int, Stride1D.Dense> count)
    {
        int c = 0;
        for (int i = 0; i < k; i++)
        {
            int t = p * k + i;
            int j = bestAB[t];
            if (j < 0 || bestBA[p * k + j] != i) continue;
            float d1 = XMath.Sqrt(XMath.Max(0f, 2f - 2f * best[t]));
            float d2 = XMath.Sqrt(XMath.Max(0f, 2f - 2f * second[t]));
            if (d1 < ratio * d2) c++;
        }
        count[p] = c;
    }

    /// <summary>
    /// How strongly each pair of images is likely to overlap, for choosing which pairs LightGlue matches: per pair, the
    /// ALIKED descriptors that are MUTUAL nearest neighbours and pass a <see cref="RetrievalRatio"/> ratio test, on the
    /// device (descriptors stay there). Symmetric n x n; the diagonal is 0. MEASURED offline 2026-10-01 (TruckFull, 251
    /// images, COLMAP truth): each image's top 20 partners by this score are ~98% true pairs (3,104 pairs of 31,375) - the
    /// coverage SfM needs at a tenth of LightGlue's all-pairs cost; on DrJohnson's wide baselines it is weaker (top 15:
    /// 72% of true pairs), which is why small sets still match every pair.
    /// </summary>
    public async Task<int[,]> PairScoresAsync(IReadOnlyList<ImportedImage> images, int batchPairs = 256)
    {
        int n = images.Count;
        var scores = new int[n, n];
        if (n < 2) return scores;
        int k = images[0].LearnedDescriptors?.K ?? throw new InvalidOperationException("no learned descriptors");
        _nearest ??= Accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(NearestKernel);
        _countMutual ??= Accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, float, ArrayView1D<int, Stride1D.Dense>>(CountMutualKernel);
        using var desc = Accel.Allocate1D<float>((long)n * k * 128);
        for (int i = 0; i < n; i++)
        {
            var d = images[i].LearnedDescriptors ?? throw new InvalidOperationException($"{images[i].FileName} has no learned descriptors");
            if (d.K != k) throw new InvalidOperationException($"{images[i].FileName}: K={d.K}, expected {k}");
            await desc.View.SubView((long)i * k * 128, k * 128).CopyFromAsync(d.Descriptors.View);
        }
        var pairs = new List<(int A, int B)>();
        for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) pairs.Add((a, b));
        int batch = Math.Max(1, batchPairs);
        using var pa = Accel.Allocate1D<int>(batch); using var pb = Accel.Allocate1D<int>(batch);
        using var bestAB = Accel.Allocate1D<int>((long)batch * k); using var simAB = Accel.Allocate1D<float>((long)batch * k);
        using var secAB = Accel.Allocate1D<float>((long)batch * k);
        using var bestBA = Accel.Allocate1D<int>((long)batch * k); using var simBA = Accel.Allocate1D<float>((long)batch * k);
        using var secBA = Accel.Allocate1D<float>((long)batch * k);
        using var count = Accel.Allocate1D<int>(batch);
        var ha = new int[batch]; var hb = new int[batch];
        for (int start = 0; start < pairs.Count; start += batch)
        {
            int m = Math.Min(batch, pairs.Count - start);
            for (int q = 0; q < batch; q++) { var (a, b) = pairs[start + Math.Min(q, m - 1)]; ha[q] = a; hb[q] = b; }
            pa.View.CopyFromCPU(ha); pb.View.CopyFromCPU(hb);
            _nearest(batch * k, desc.View, pa.View, pb.View, k, bestAB.View, simAB.View, secAB.View);
            _nearest(batch * k, desc.View, pb.View, pa.View, k, bestBA.View, simBA.View, secBA.View);
            _countMutual(batch, bestAB.View, simAB.View, secAB.View, bestBA.View, k, RetrievalRatio, count.View);
            await Accel.SynchronizeAsync();
            // CPU transfer: one int per pair - the pair choice is made on the host.
            var c = await count.View.SubView(0, m).CopyToHostAsync();
            for (int q = 0; q < m; q++) { var (a, b) = pairs[start + q]; scores[a, b] = scores[b, a] = c[q]; }
        }
        return scores;
    }

    /// <summary>The union of each image's <paramref name="topK"/> highest-scoring partners (<see cref="PairScoresAsync"/>),
    /// as (a, b) with a &lt; b, in ascending order.</summary>
    public static List<(int A, int B)> TopPartnerPairs(int[,] scores, int topK)
    {
        int n = scores.GetLength(0);
        var sel = new SortedSet<(int, int)>();
        for (int a = 0; a < n; a++)
        {
            foreach (int b in Enumerable.Range(0, n).Where(b => b != a).OrderByDescending(b => scores[a, b]).ThenBy(b => b).Take(topK))
                sel.Add((Math.Min(a, b), Math.Max(a, b)));
        }
        return sel.ToList();
    }

    public void Dispose()
    {
        ReleaseExtractors();
        ReleaseMatcher();
    }
}
