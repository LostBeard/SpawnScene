using ILGPU;
using ILGPU.Runtime;
using NUnit.Framework;
using SpawnDev.ILGPU;

namespace SpawnScene.Tests;

/// <summary>Minimal shape of FAST's 9-contiguous loop: an if/else inside a counted loop, the if-side with a break.
/// b29's WGSL left the else path without the loop increment (infinite loop -> DXGI_ERROR_DEVICE_HUNG).</summary>
public class WgslLatchRepro
{
    static void RadixSortLatchRepro(Index1D i, ArrayView1D<int, Stride1D.Dense> data, ArrayView1D<int, Stride1D.Dense> outp)
    {
        int maxRun = 0, run = 0;
        for (int k = 0; k < 32; k++)
        {
            if (data[(i + k) % 16] > 5)
            {
                run++;
                if (run > maxRun) maxRun = run;
                if (maxRun >= 9) break;
            }
            else run = 0;
        }
        outp[i] = maxRun;
    }

    [Test, Explicit("diagnostic dump")]
    public void Dump()
    {
        var m = typeof(WgslLatchRepro).GetMethod(nameof(RadixSortLatchRepro), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.VerboseLogging = true;
        var src = ShaderCompiler.Generate(m, CapabilityProfiles.WebGPUBaseline).Source;
        SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.VerboseLogging = false;
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "latch_repro.wgsl"), src);
    }
}
