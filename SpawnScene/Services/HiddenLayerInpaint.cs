using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnScene.Services;

/// <summary>
/// MI-GAN (Picsart AI Research, ICCV 2023, MIT) paints what a single photo cannot see: OcclusionFill masks the cells
/// that get a hidden background splat behind a depth edge, and this fills them with plausible background, replacing the
/// push-pull blur that showed as grey streaky sheets from a moved camera (TJ 2026-10-07).
/// <para>
/// The model is MI-GAN's plain 512 net with float I/O (tools/migan_float_io.py; LostBeard/spawnscene-models, through the
/// hub). It matches onnxruntime on SpawnDev.ILGPU.ML (autotest=inpaint-parity). Inputs: image [1,3,512,512] 0..255,
/// mask [1,1,512,512] (255 = known, 0 = paint); output the same image with the hole painted.
/// </para>
/// <para>
/// <c>&amp;inpaintmodel=lama</c>: big-LaMa (Suvorov et al., WACV 2022, Apache-2.0; Carve/LaMa-ONNX's lama_fp32.onnx, 208 MB,
/// through the hub's /hf) instead. Measured 2026-10-08 on the kitchen (CPU onnxruntime, the same holes): where MI-GAN
/// invents another chair or leaves the lamp's pole, LaMa continues the cabinet and floor - it is trained for large
/// object-removal masks, which is what a depth edge opens. 7x the download, ~2.6x the time. Its inputs are image 0..1 and
/// mask 1 = paint (converted here), output 0..255.
/// </para>
/// </summary>
public sealed class HiddenLayerInpaint : IDisposable
{
    public const int Size = 512;
    const string MiganRepo = "LostBeard/spawnscene-models", MiganFile = "migan_float.onnx";
    const string LamaRepo = "Carve/LaMa-ONNX", LamaFile = "lama_fp32.onnx";

    /// <summary>"migan" (default) or "lama" (&amp;inpaintmodel=).</summary>
    public static string Model { get; set; } = "migan";
    bool Lama => Model == "lama";
    string? _loadedModel;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _toLama;
    static Accelerator? _toLamaFor;

    /// <summary>MI-GAN's inputs (image 0..255, mask 255 = known) as LaMa's (image 0..1, mask 1 = paint).</summary>
    static void ToLamaKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> image, ArrayView1D<float, Stride1D.Dense> mask,
        ArrayView1D<float, Stride1D.Dense> outImage, ArrayView1D<float, Stride1D.Dense> outMask)
    {
        const int Px = Size * Size;
        if (i >= Px) return;
        outImage[i] = image[i] / 255f;
        outImage[Px + i] = image[Px + i] / 255f;
        outImage[2 * Px + i] = image[2 * Px + i] / 255f;
        outMask[i] = mask[i] > 127f ? 0f : 1f;
    }

    readonly IModelSource _models;
    InferenceSession? _session;
    Accelerator? _for;
    bool _failed;

    public HiddenLayerInpaint(IModelSource models) => _models = models;

    /// <summary>
    /// Paint the masked pixels; a new [3, 512, 512] buffer the caller owns, or null when the model cannot load or run
    /// (the caller keeps its fallback colours).
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>?> RunAsync(Accelerator a, ArrayView1D<float, Stride1D.Dense> image,
        ArrayView1D<float, Stride1D.Dense> mask)
    {
        if (_failed) return null;
        try
        {
            if (_session == null || !ReferenceEquals(_for, a) || _loadedModel != Model)
            {
                _session?.Dispose();
                var t0 = DateTime.UtcNow;
                await using var stream = await _models.OpenAsync(Lama ? LamaRepo : MiganRepo, Lama ? LamaFile : MiganFile);
                _session = await InferenceSession.CreateFromStreamAsync(a, stream, inputShapes: new Dictionary<string, int[]>
                {
                    ["image"] = new[] { 1, 3, Size, Size },
                    ["mask"] = new[] { 1, 1, Size, Size },
                });
                _for = a;
                _loadedModel = Model;
                Console.WriteLine($"[Inpaint] {(Lama ? "LaMa" : "MI-GAN")} loaded in {(DateTime.UtcNow - t0).TotalSeconds:F1}s");
            }
            var t1 = DateTime.UtcNow;
            using var lamaImage = Lama ? a.Allocate1D<float>(3L * Size * Size) : null;
            using var lamaMask = Lama ? a.Allocate1D<float>((long)Size * Size) : null;
            if (Lama)
            {
                if (!ReferenceEquals(_toLamaFor, a)) { _toLama = null; _toLamaFor = a; }
                _toLama ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ToLamaKernel);
                _toLama(Size * Size, image, mask, lamaImage!.View, lamaMask!.View);
            }
            var outs = await _session.RunAsync(new Dictionary<string, Tensor>
            {
                ["image"] = new Tensor(Lama ? lamaImage!.View : image, new[] { 1, 3, Size, Size }),
                ["mask"] = new Tensor(Lama ? lamaMask!.View : mask, new[] { 1, 1, Size, Size }),
            });
            // The session's outputs are its own (reused by the next run). Both models give 0..255.
            var result = a.Allocate1D<float>(3L * Size * Size);
            await result.View.CopyFromAsync(outs[Lama ? "output" : "result"].Data.SubView(0, 3L * Size * Size));
            await a.SynchronizeAsync();
            Console.WriteLine($"[Inpaint] hidden layer painted in {(DateTime.UtcNow - t1).TotalMilliseconds:F0} ms");
            return result;
        }
        catch (Exception ex)
        {
            _failed = true;
            Console.WriteLine($"[Inpaint] {(Lama ? "LaMa" : "MI-GAN")} unavailable ({ex.GetType().Name}: {ex.Message}) - the hidden layer keeps the blur");
            return null;
        }
    }

    public void Dispose() { _session?.Dispose(); _session = null; }
}
