using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnScene.Pages;

/// <summary>
/// <c>autotest=inpaint-parity</c>: MI-GAN (float-I/O variant, LostBeard/spawnscene-models through the hub) on
/// SpawnDev.ILGPU.ML against onnxruntime's output for the same input - a 512 crop of the kitchen sample with a 256 px hole
/// (test/migan_in_image.bin, test/migan_in_mask.bin, test/migan_ref.bin: float32, made by tools/migan_float_io.py's
/// recipe; harness-only files, not part of the app). The question before any integration: does our engine run the graph
/// and agree with the reference (TJ 2026-10-07: inpaint what a moved camera sees behind objects).
/// </summary>
public partial class Studio
{
    async Task RunInpaintParityAsync()
    {
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();
            var a = _gpuService.WebGPUAccelerator;
            var t0 = DateTime.UtcNow;
            using var stream = await _modelSource.OpenAsync("LostBeard/spawnscene-models", "migan_float.onnx");
            using var session = await InferenceSession.CreateFromStreamAsync(a, stream, inputShapes: new Dictionary<string, int[]>
            {
                ["image"] = new[] { 1, 3, 512, 512 },
                ["mask"] = new[] { 1, 1, 512, 512 },
            });
            Console.WriteLine($"[Inpaint] MI-GAN loaded in {(DateTime.UtcNow - t0).TotalSeconds:F1}s");

            // CPU transfer: the test's fixed inputs and reference (harness files, a few MB).
            static float[] Floats(byte[] b) { var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
            var img = Floats(await _http.GetByteArrayAsync("test/migan_in_image.bin"));
            var mask = Floats(await _http.GetByteArrayAsync("test/migan_in_mask.bin"));
            var reference = Floats(await _http.GetByteArrayAsync("test/migan_ref.bin"));
            using var imgBuf = a.Allocate1D(img);
            using var maskBuf = a.Allocate1D(mask);

            var t1 = DateTime.UtcNow;
            var outs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                ["image"] = new Tensor(imgBuf.View, new[] { 1, 3, 512, 512 }),
                ["mask"] = new Tensor(maskBuf.View, new[] { 1, 1, 512, 512 }),
            });
            await a.SynchronizeAsync();
            double ms = (DateTime.UtcNow - t1).TotalMilliseconds;
            var result = outs["result"];
            using var host = a.Allocate1D<float>(reference.Length);
            await host.View.CopyFromAsync(result.Data.SubView(0, reference.Length));
            var got = await host.View.CopyToHostAsync();

            double sum = 0, worst = 0, holeSum = 0; int holeN = 0;
            for (int i = 0; i < reference.Length; i++)
            {
                double d = Math.Abs(got[i] - reference[i]);
                sum += d; worst = Math.Max(worst, d);
                if (mask[i % (512 * 512)] < 128f) { holeSum += d; holeN++; }
            }
            double mean = sum / reference.Length, holeMean = holeSum / Math.Max(1, holeN);
            // Red check: in the hole the output must NOT be the input image (a pass-through would match nothing there).
            double vsInput = 0; int hp = -1;
            for (int i = 0; i < reference.Length; i++)
                if (mask[i % (512 * 512)] < 128f) { vsInput += Math.Abs(got[i] - img[i]); if (hp < 0) hp = i; }
            vsInput /= Math.Max(1, holeN);
            int mid = 256 * 512 + 256;   // the hole's centre, red channel
            Console.WriteLine($"[Inpaint] hole vs the input image: mean |diff| {vsInput:F1}; centre pixel got {got[mid]:F3} ref {reference[mid]:F3} " +
                $"input {img[mid]:F3}; result tensor shape [{string.Join(",", result.Shape)}]");
            if (vsInput < 5.0) { Console.WriteLine("[Dataset] FAIL: the hole was not painted (output = input)"); return; }
            Console.WriteLine($"[Inpaint] MI-GAN on ILGPU.ML vs onnxruntime: mean |diff| {mean:F3} (0..255), in the hole {holeMean:F3}, " +
                $"max {worst:F1}; first run {ms:F0} ms");
            Console.WriteLine(mean < 2.0 && holeMean < 4.0 ? "[Dataset] DONE" : "[Dataset] FAIL: MI-GAN output differs from onnxruntime");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] FAIL: inpaint parity threw {ex.GetType().Name}: {ex.Message}");
            Console.WriteLine(ex.StackTrace?.Split('\n').Take(8).Aggregate("", (s, l) => s + " | " + l.Trim()));
        }
    }
}
