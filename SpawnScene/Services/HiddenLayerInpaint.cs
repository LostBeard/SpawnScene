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
/// </summary>
public sealed class HiddenLayerInpaint : IDisposable
{
    public const int Size = 512;
    const string Repo = "LostBeard/spawnscene-models", File = "migan_float.onnx";

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
            if (_session == null || !ReferenceEquals(_for, a))
            {
                _session?.Dispose();
                var t0 = DateTime.UtcNow;
                await using var stream = await _models.OpenAsync(Repo, File);
                _session = await InferenceSession.CreateFromStreamAsync(a, stream, inputShapes: new Dictionary<string, int[]>
                {
                    ["image"] = new[] { 1, 3, Size, Size },
                    ["mask"] = new[] { 1, 1, Size, Size },
                });
                _for = a;
                Console.WriteLine($"[Inpaint] MI-GAN loaded in {(DateTime.UtcNow - t0).TotalSeconds:F1}s");
            }
            var t1 = DateTime.UtcNow;
            var outs = await _session.RunAsync(new Dictionary<string, Tensor>
            {
                ["image"] = new Tensor(image, new[] { 1, 3, Size, Size }),
                ["mask"] = new Tensor(mask, new[] { 1, 1, Size, Size }),
            });
            // The session's outputs are its own (reused by the next run).
            var result = a.Allocate1D<float>(3L * Size * Size);
            await result.View.CopyFromAsync(outs["result"].Data.SubView(0, 3L * Size * Size));
            await a.SynchronizeAsync();
            Console.WriteLine($"[Inpaint] hidden layer painted in {(DateTime.UtcNow - t1).TotalMilliseconds:F0} ms");
            return result;
        }
        catch (Exception ex)
        {
            _failed = true;
            Console.WriteLine($"[Inpaint] MI-GAN unavailable ({ex.GetType().Name}: {ex.Message}) - the hidden layer keeps the blur");
            return null;
        }
    }

    public void Dispose() { _session?.Dispose(); _session = null; }
}
