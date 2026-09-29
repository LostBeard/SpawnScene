using ILGPU;
using ILGPU.Runtime;
using NUnit.Framework;
using SpawnDev.ILGPU;

namespace SpawnScene.Tests;

/// <summary>Minimal shape of GpuBundleAdjuster.PointSolveKernel (b34, 2026-09-28): one view stored in two separate
/// branches (one returns early). The WGSL post-processor hoisted each branch's `let v_N = &amp;paramX;` pointer alias to
/// function scope and emitted it twice: "redeclaration of 'v_421'" in Chrome.</summary>
public class WgslAliasRepro
{
    static void TwoBranchStoreRepro(Index1D i, ArrayView1D<double, Stride1D.Dense> v, ArrayView1D<double, Stride1D.Dense> flags,
        ArrayView1D<double, Stride1D.Dense> outp)
    {
        double x = v[i];
        if (!(x > -1.0 && x < 2.0))
        {
            flags[0] = 1.0;
            outp[i] = 0.0;
            return;
        }
        if (!(x > 1e-14))
        {
            x = 1e-14;
            flags[1] = 1.0;
        }
        outp[i] = 1.0 / x;
    }

    static void TwoBranchStoreReproF32(Index1D i, ArrayView1D<float, Stride1D.Dense> v, ArrayView1D<float, Stride1D.Dense> flags,
        ArrayView1D<float, Stride1D.Dense> outp)
    {
        float x = v[i];
        if (!(x > -1f && x < 2f))
        {
            flags[0] = 1f;
            outp[i] = 0f;
            return;
        }
        if (!(x > 1e-14f))
        {
            x = 1e-14f;
            flags[1] = 1f;
        }
        outp[i] = 1f / x;
    }

    [Test, Explicit("diagnostic dump")]
    public void Dump()
    {
        foreach (var name in new[] { nameof(TwoBranchStoreRepro), nameof(TwoBranchStoreReproF32) })
        {
            var m = typeof(WgslAliasRepro).GetMethod(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
            var src = ShaderCompiler.Generate(m, CapabilityProfiles.WebGPUBaseline).Source;
            File.WriteAllText(Path.Combine(Path.GetTempPath(), $"alias_{name}.wgsl"), src);
            var glsl = ShaderCompiler.Generate(m, CapabilityProfiles.WebGL2Baseline).Source;
            File.WriteAllText(Path.Combine(Path.GetTempPath(), $"alias_{name}.glsl"), glsl);
        }
    }
}
