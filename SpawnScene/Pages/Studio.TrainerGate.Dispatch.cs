using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Linear dispatch gate: WebGPU allows 65535 workgroups a dimension, so a per-splat pass of 64-thread groups tops out
/// at 4,194,240 splats in one. The c27 bicycle run (6 GB budget) grew to 4,200,325 splats and died on "Dispatch
/// workgroup count X (65631) exceeds max compute workgroups per dimension (65535)" (2026-10-05). Every per-splat pass
/// now wraps into Y (SplatTrainerGpu.DispatchLinear) and its shader rebuilds the flat index; this runs one of them
/// over 4.3M splats and checks the rows past the old limit came out like the first.
/// </summary>
public partial class Studio
{
    async Task<bool> LinearDispatchGateAsync(SplatTrainerGpu trainer)
    {
        const int F = SplatFormat.Floats;
        const int n = 4_300_000;   // > 65535 x 64
        var a = _gpuService.WebGPUAccelerator;
        using var buf = a.Allocate1D<float>((long)n * F);   // 240 MB, zeroed on the GPU (never through .NET)
        buf.MemSetToZero();
        await a.SynchronizeAsync();
        trainer.ConvertRgbToShDc(buf, n);
        await a.SynchronizeAsync();
        var first = await buf.CopyToHostAsync<float>(0, F);
        int[] probe = { 4_194_239, 4_194_240, 4_250_000, n - 1 };
        int bad = 0;
        foreach (int i in probe)
        {
            var row = await buf.CopyToHostAsync<float>((long)i * F, F);
            for (int c = 3; c < 6; c++) if (row[c] != first[c]) { bad++; break; }
        }
        Console.WriteLine($"[TrainerGate] linear dispatch: {n:N0} splats, colour slot 3 of row 0 {first[3]:G6}; " +
            $"{probe.Length - bad} of {probe.Length} rows past 4,194,239 converted alike");
        if (first[3] == 0f || bad > 0)
        {
            Console.WriteLine("[TrainerGate] FAIL linear dispatch: " + (first[3] == 0f ? "row 0 was not converted" : "rows past the 65535-group limit were missed"));
            return false;
        }
        Console.WriteLine("[TrainerGate] linear dispatch PASS");
        return true;
    }
}
