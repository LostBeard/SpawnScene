using System.Reflection;
using NUnit.Framework;
using SpawnDev.ILGPU;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Writes the WGSL each GpuBundleAdjuster kernel compiles to (no browser, no GPU), for validating with naga
/// before a browser run - the CG kernels put barriers in a single workgroup, which WGSL only accepts in uniform
/// control flow.</summary>
public class GpuBundleAdjusterWgslDump
{
    [Test, Explicit("diagnostic dump")]
    public void DumpWgsl()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gba_wgsl");
        Directory.CreateDirectory(dir);
        foreach (var m in typeof(GpuBundleAdjuster).GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                     .Where(m => m.Name.EndsWith("Kernel", StringComparison.Ordinal)))
        {
            var src = ShaderCompiler.Generate(m, CapabilityProfiles.WebGPUBaseline).Source;
            var path = Path.Combine(dir, $"{m.Name}.wgsl");
            File.WriteAllText(path, src);
            TestContext.Out.WriteLine($"{m.Name}: {src.Length} chars -> {path}");
        }
    }
}
