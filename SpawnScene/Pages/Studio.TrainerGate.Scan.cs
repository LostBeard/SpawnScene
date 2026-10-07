using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnScene.Pages;

/// <summary>
/// Exclusive scan gate: ILGPU.Algorithms' scan (GpuDensify's compaction: keep/add destinations) against a CPU prefix
/// sum, at the sizes training reaches, three scans in a row sharing one temp buffer as GpuDensify does - and the same
/// with a fresh temp buffer each time, to tell a reuse fault from a scan fault. Densify was losing kept rows in
/// contiguous runs starting at 69,632 = 17 x 4096 on a 615K-splat TruckFull block (2026-10-05).
/// </summary>
public partial class Studio
{
    async Task<bool> ScanGateAsync()
    {
        var accel = _gpuService.WebGPUAccelerator;
        // AOTSTEP markers: on the AOT build this gate died with "RuntimeError: function signature mismatch" (2026-10-07,
        // also on live spawnscene.com) while the interpreted build passes. The last marker names the call.
        bool trace = true;
        void Step(string s) { if (trace) Console.WriteLine($"[TrainerGate] AOTSTEP {s}"); }
        Step("create scan");
        var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, AddInt32>(ScanKind.Exclusive);
        var rng = new Random(77);
        bool ok = true;
        foreach (int n in new[] { 4095, 4097, 70_000, 615_764, 1_600_003 })
        {
            foreach (bool shareTemp in new[] { true, false })
            {
                Step($"inputs n={n}");
                var inputs = Enumerable.Range(0, 3).Select(k => Enumerable.Range(0, n)
                    .Select(_ => k == 0 ? (rng.NextDouble() < 0.02 ? 1 : 0) : (rng.NextDouble() < 0.97 ? 1 : 0)).ToArray()).ToArray();
                Step("temp size");
                long tempLen = Math.Max(1L, accel.ComputeScanTempStorageSize<int>(n));
                Step("alloc shared");
                using var shared = accel.Allocate1D<int>(tempLen);
                for (int k = 0; k < 3; k++)
                {
                    Step("alloc src from array");
                    using var src = accel.Allocate1D(inputs[k]);
                    Step("alloc dst/own");
                    using var dst = accel.Allocate1D<int>(n);
                    using var own = shareTemp ? null : accel.Allocate1D<int>(tempLen);
                    Step("scan");
                    scan(accel.DefaultStream, src.View, dst.View, (own ?? shared).View);
                    Step("sync");
                    await accel.SynchronizeAsync();
                    Step("readback");
                    var got = await dst.CopyToHostAsync<int>(0, n);
                    Step("compare");
                    trace = false;
                    int run = 0, bad = -1, badCount = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if (got[i] != run) { badCount++; if (bad < 0) bad = i; }
                        run += inputs[k][i];
                    }
                    if (bad >= 0)
                    {
                        ok = false;
                        Console.WriteLine($"[TrainerGate] scan FAIL: n={n:N0} scan {k + 1} of 3, {(shareTemp ? "shared" : "fresh")} temp " +
                            $"({tempLen} ints): {badCount:N0} wrong, first at {bad:N0} (got {got[bad]:N0})");
                    }
                }
            }
        }
        Console.WriteLine(ok ? "[TrainerGate] scan PASS: exclusive scans match the CPU at 4,095 to 1.6M, shared and fresh temp"
                             : "[TrainerGate] scan: see failures above");
        return ok;
    }
}
