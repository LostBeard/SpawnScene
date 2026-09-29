using System.Reflection;
using NUnit.Framework;
using SpawnDev.ILGPU;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>Writes the WGSL each GpuFeatureDetector kernel compiles to (no browser, no GPU) - inspect loop exits
/// after a device hang (b29, 2026-09-28: DXGI_ERROR_DEVICE_HUNG during the first on-device detection).</summary>
public class GpuFeatureDetectorWgslDump
{
    [Test, Explicit("diagnostic dump")]
    public void DumpWgsl()
    {
        foreach (var name in new[] { "FastKernel", "CellMaxKernel", "BlurRowsKernel", "BlurColsKernel", "BriefKernel" })
        {
            var m = typeof(GpuFeatureDetector).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!;
            var src = ShaderCompiler.Generate(m, CapabilityProfiles.WebGPUBaseline).Source;
            var path = Path.Combine(Path.GetTempPath(), $"gfd_{name}.wgsl");
            File.WriteAllText(path, src);
            TestContext.Out.WriteLine($"{name}: {src.Length} chars -> {path}");
        }
    }
}
