using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnScene.Pages;

// ?autotest=aotprobe&probe=N - one host->device upload shape per page load, to find which one the AOT build traps on
// ("RuntimeError: function signature mismatch" in ArrayViewExtensions.CopyFromCPU<int>(ArrayView1D<int, Dense>,
// AcceleratorStream, int[]), reached from the trainer gate's scan stage, 2026-10-07). A wasm trap cannot be caught,
// so each probe runs alone and announces itself first.
public partial class Studio
{
    private async Task RunAotProbeAsync(int probe)
    {
        var accel = _gpuService.WebGPUAccelerator;
        Console.WriteLine($"[AotProbe] START {probe}");
        try
        {
            switch (probe)
            {
                case 0: // the gate's shape: Allocate1D(int[]) from a LINQ-built array, 4095 elements
                {
                    var a = Enumerable.Range(0, 4095).Select(i => i & 1).ToArray();
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 1: // a plain new int[4095]
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 2: // a small new int[16]
                {
                    var a = new int[16];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 3: // float[4095]
                {
                    var a = new float[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 4: // Allocate1D<int>(n) then View.CopyFromCPU(int[]) (no stream overload)
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D<int>(a.Length);
                    b.View.CopyFromCPU(a);
                    break;
                }
                case 5: // the base view from a span, as CopyFromCPU does inside
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D<int>(a.Length);
                    b.View.BaseView.CopyFromCPU(accel.DefaultStream, new ReadOnlySpan<int>(a));
                    break;
                }
                case 6: // the same call through the Accelerator static type (as GpuFeatureDetector)
                {
                    Accelerator acc = accel;
                    var a = new int[4095];
                    using var b = acc.Allocate1D(a);
                    break;
                }
                case 8: // ORDER: uint[] first (as the radix sort gate does), then int[] in the same page
                {
                    using (var u = accel.Allocate1D(new uint[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 8: uint[] done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 9: // ORDER: int[] first, then uint[]
                {
                    using (var i0 = accel.Allocate1D(new int[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 9: int[] done, now uint[]");
                    using var b = accel.Allocate1D(new uint[4095]);
                    break;
                }
                case 10: // ORDER: float[] first, then int[]
                {
                    using (var f0 = accel.Allocate1D(new float[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 10: float[] done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 11: // ORDER: the trainer gate's radix sort stage, then the scan stage's int[] upload
                {
                    bool ok = await RadixSortGateAsync();
                    Console.WriteLine($"[AotProbe] 11: radix sort gate {(ok ? "passed" : "failed")}, now int[]");
                    using var b = accel.Allocate1D(Enumerable.Range(0, 4095).Select(i => i & 1).ToArray());
                    break;
                }
                case 12: // ORDER: GpuRadixSort alone (one sort, uint keys/values), then int[]
                {
                    var keys = new uint[1025]; var vals = new uint[1025];
                    for (int i = 0; i < keys.Length; i++) { keys[i] = (uint)((i * 2654435761u) >> 2); vals[i] = (uint)i; }
                    using var kBuf = accel.Allocate1D(keys);
                    using var vBuf = accel.Allocate1D(vals);
                    using var sorter = new Services.GpuRadixSort(accel.NativeAccelerator.NativeDevice!, accel.NativeAccelerator.Queue!, accel);
                    sorter.EnsureCapacity(keys.Length);
                    sorter.Sort(kBuf.GetGPUBuffer()!, vBuf.GetGPUBuffer()!, keys.Length, 30);
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 12: one GpuRadixSort done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 13: // the scan stage's exact prelude: CreateScan, temp size, temp alloc, then Allocate1D(int[])
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    long tempLen = Math.Max(1L, accel.ComputeScanTempStorageSize<int>(4095));
                    using var shared = accel.Allocate1D<int>(tempLen);
                    Console.WriteLine("[AotProbe] 13: scan prelude done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 14: // only CreateScan, then Allocate1D(int[])
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    Console.WriteLine("[AotProbe] 14: CreateScan done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 15: // only the temp size, then Allocate1D(int[])
                {
                    long tempLen = Math.Max(1L, accel.ComputeScanTempStorageSize<int>(4095));
                    Console.WriteLine($"[AotProbe] 15: temp size {tempLen}, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 7: // uint[4095]
                {
                    var a = new uint[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                default:
                    // A probe this build does not have must not read as a pass (a stale publish printed PASS 13-15).
                    Console.WriteLine($"[AotProbe] FAIL {probe}: no such probe in this build");
                    return;
            }
            await accel.SynchronizeAsync();
            Console.WriteLine($"[AotProbe] PASS {probe}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AotProbe] FAIL {probe}: {ex.GetType().Name}: {ex.Message}");
        }
    }
}
