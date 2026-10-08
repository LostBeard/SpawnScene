using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Per-photo exposure gate (SplatTrainerGpu.Exposure): the target is the scene's own render through a KNOWN colour
/// transform (per-channel gains 0.8 / 0.9 / 1.1, offsets +0.05 / -0.03 / +0.02); with the scene frozen (zero colour and
/// opacity rates, no geometry) only the exposure can explain the difference, so it must converge to that transform -
/// the apply pass, the exposure gradient and its Adam all have to be right. A flipped sign walks away from it, a wrong
/// partial sum lands somewhere else. Then the null case: against the plain render the exposure must stay identity.
/// </summary>
public partial class Studio
{
    async Task<bool> ExposureGateAsync(SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        var img = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar, readback: true);
        float[] gain = { 0.8f, 0.9f, 1.1f }, offset = { 0.05f, -0.03f, 0.02f };
        var target = new float[img.Length];
        for (int i = 0; i < img.Length; i++) target[i] = gain[i % 3] * img[i] + offset[i % 3];

        async Task<float[]> FitAsync(float[] t, int steps)
        {
            trainer.SetTarget(t);
            trainer.ResetExposure(1);
            for (int k = 0; k < steps; k++)
            {
                trainer.ExposureLr = TrainingSchedule.ExponentialLr(0.01f, 0.001f, k, steps);
                await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0f,
                    readLoss: false, exposureSlot: 0);
            }
            return await trainer.ReadExposureAsync(1);
        }

        var e = await FitAsync(target, 600);
        string Rows(float[] x) => string.Join(" | ", Enumerable.Range(0, 3).Select(r =>
            string.Join(" ", Enumerable.Range(0, 4).Select(k => x[r * 4 + k].ToString("F3")))));
        float worst = 0f;
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 4; k++)
            {
                float want = k == 3 ? offset[r] : (k == r ? gain[r] : 0f);
                worst = MathF.Max(worst, MathF.Abs(e[r * 4 + k] - want));
            }
        Console.WriteLine($"[TrainerGate] exposure fit to gains 0.8/0.9/1.1, offsets +0.05/-0.03/+0.02: {Rows(e)} (worst error {worst:F4})");
        if (worst > 0.03f) { Console.WriteLine("[TrainerGate] FAIL exposure: did not recover the transform"); return false; }

        // The fold: a fitted GAINS-only transform moved into the scene's base colours must make the plain render (no
        // exposure) equal the transformed render - exactly, at any coverage (an offset would scale with each pixel's
        // opacity: this gate scene is half transparent, so an offset fold read 0.015 here, not 0). Base colour only (the
        // SH bands are left by the fold) and the rows are restored afterwards.
        var saved = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
        int savedDeg = trainer.ActiveShDegree;
        try
        {
            trainer.ActiveShDegree = 0;
            var img0 = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar, readback: true);
            var gainsOnly = new float[img0.Length];
            for (int i = 0; i < img0.Length; i++) gainsOnly[i] = gain[i % 3] * img0[i];
            await FitAsync(gainsOnly, 600);
            var fold = await trainer.FoldMeanExposureAsync(splatBuf, n, new[] { 0 });
            trainer.ResetExposure(1);
            var plain = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar, readback: true);
            double err = 0, before = 0;
            for (int i = 0; i < plain.Length; i++) { err += Math.Abs(plain[i] - gainsOnly[i]); before += Math.Abs(img0[i] - gainsOnly[i]); }
            err /= plain.Length; before /= plain.Length;
            Console.WriteLine($"[TrainerGate] exposure fold (gains 0.8/0.9/1.1): plain render vs the transformed render, mean |error| " +
                $"{before:F5} before -> {err:F5} after");
            if (fold == null || err > 0.002 || err > 0.1 * before)
            {
                Console.WriteLine("[TrainerGate] FAIL exposure: the folded scene does not render the photo's exposure");
                return false;
            }
        }
        finally
        {
            trainer.ActiveShDegree = savedDeg;
            splatBuf.CopyFromCPU(saved);
            await _gpuService.WebGPUAccelerator.SynchronizeAsync();
        }

        var id = await FitAsync(img, 200);
        float drift = 0f;
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 4; k++) drift = MathF.Max(drift, MathF.Abs(id[r * 4 + k] - (k == r ? 1f : 0f)));
        Console.WriteLine($"[TrainerGate] exposure against the plain render: {Rows(id)} (largest move {drift:F4})");
        if (drift > 0.02f) { Console.WriteLine("[TrainerGate] FAIL exposure: drifted off identity with nothing to explain"); return false; }
        Console.WriteLine("[TrainerGate] exposure PASS");
        return true;
    }
}
