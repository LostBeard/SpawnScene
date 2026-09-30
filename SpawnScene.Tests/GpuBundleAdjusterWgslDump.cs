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
    public void DumpWgsl() => Dump(typeof(GpuBundleAdjuster), "gba_wgsl");

    /// <summary>GpuGlobalPositioner's kernels (2026-09-29): validate every one with naga before a browser run.</summary>
    [Test, Explicit("diagnostic dump")]
    public void DumpPositionerWgsl() => Dump(typeof(GpuGlobalPositioner), "ggp_wgsl");

    static void Dump(Type type, string folder)
    {
        var dir = Path.Combine(Path.GetTempPath(), folder);
        Directory.CreateDirectory(dir);
        foreach (var m in type.GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                     .Where(m => m.Name.EndsWith("Kernel", StringComparison.Ordinal)))
        {
            var src = ShaderCompiler.Generate(m, CapabilityProfiles.WebGPUBaseline).Source;
            var path = Path.Combine(dir, $"{m.Name}.wgsl");
            File.WriteAllText(path, src);
            TestContext.Out.WriteLine($"{m.Name}: {src.Length} chars -> {path}");
        }
    }
}
