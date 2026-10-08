using System.Diagnostics;
using System.Reflection;
using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Every trainer WGSL shader, and the depth-supervised build of each (SplatTrainerShaders.DepthVariant), must pass naga's
/// validator. Valid is not executed (a valid shader can still compute the wrong thing - the TrainerGate checks that on
/// the GPU), but an invalid one fails pipeline creation in the browser after an hour's AOT build. Skipped where naga
/// (cargo install naga-cli) is not on the machine.
/// </summary>
public class TrainerShaderValidationTests
{
    static IEnumerable<TestCaseData> Shaders()
    {
        foreach (var f in typeof(SplatTrainerShaders).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.FieldType != typeof(string) || !f.IsLiteral && !f.IsInitOnly) continue;
            var src = (string)f.GetValue(null)!;
            if (!src.Contains("@compute")) continue;
            yield return new TestCaseData(f.Name, src).SetName($"Valid_{f.Name}");
            if (src.Contains("//DEPTH: "))
                yield return new TestCaseData(f.Name + "+depth", SplatTrainerShaders.DepthVariant(src)).SetName($"Valid_{f.Name}_Depth");
        }
    }

    [TestCaseSource(nameof(Shaders))]
    public void NagaValidates(string name, string wgsl)
    {
        string naga = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cargo", "bin", "naga.exe");
        if (!File.Exists(naga)) Assert.Ignore("naga not installed");
        string file = Path.Combine(Path.GetTempPath(), $"spawnscene-{name.Replace('+', '_')}.wgsl");
        File.WriteAllText(file, wgsl);
        var psi = new ProcessStartInfo(naga, $"\"{file}\"") { RedirectStandardError = true, RedirectStandardOutput = true };
        using var proc = Process.Start(psi)!;
        string err = proc.StandardError.ReadToEnd() + proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();
        File.Delete(file);
        Assert.That(proc.ExitCode, Is.Zero, $"{name}: {err}");
    }
}
