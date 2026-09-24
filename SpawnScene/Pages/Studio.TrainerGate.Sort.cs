using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Radix sort gate: does <see cref="GpuRadixSort"/> produce exactly a CPU stable sort, and how long does it
/// take next to the ILGPU.Algorithms sort it replaces? Values are the original indices, so a stability bug
/// (equal keys out of input order) fails as surely as a mis-ordered key.
/// </summary>
public partial class Studio
{
    async Task<bool> RadixSortGateAsync()
    {
        var accel = _gpuService.WebGPUAccelerator;
        var device = accel.NativeAccelerator.NativeDevice!;
        var queue = accel.NativeAccelerator.Queue!;
        using var sorter = new GpuRadixSort(device, queue);
        var rng = new Random(1234);
        bool ok = true;

        foreach (var (n, bits, distinct) in new[]
        {
            (1, 30, 0), (2, 30, 0), (1023, 30, 0), (1025, 30, 7), (300_001, 30, 0),
            (300_001, 30, 2_000), (2_000_003, 30, 50_000), (2_000_003, 32, 0),
        })
        {
            // distinct > 0: draw from a small key set so long runs of equal keys test stability.
            var keySet = distinct > 0
                ? Enumerable.Range(0, distinct).Select(_ => NextKey(rng, bits)).ToArray()
                : null;
            var keys = new uint[n];
            for (int i = 0; i < n; i++) keys[i] = keySet != null ? keySet[rng.Next(distinct)] : NextKey(rng, bits);
            var values = Enumerable.Range(0, n).Select(i => (uint)i).ToArray();

            using var kBuf = accel.Allocate1D(keys);
            using var vBuf = accel.Allocate1D(values);
            await accel.SynchronizeAsync();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            sorter.Sort(kBuf.GetGPUBuffer()!, vBuf.GetGPUBuffer()!, n, bits);
            await accel.SynchronizeAsync();
            double ms = clock.Elapsed.TotalMilliseconds;
            var gotK = await kBuf.CopyToHostAsync<uint>(0, n);
            var gotV = await vBuf.CopyToHostAsync<uint>(0, n);

            // CPU stable sort: by key, ties in input order (values are the input indices).
            var order = Enumerable.Range(0, n).OrderBy(i => keys[i]).ThenBy(i => i).ToArray();
            int bad = -1;
            for (int i = 0; i < n && bad < 0; i++)
                if (gotK[i] != keys[order[i]] || gotV[i] != (uint)order[i]) bad = i;
            if (bad >= 0)
            {
                Console.WriteLine($"[TrainerGate] radix sort FAIL: n={n:N0} bits={bits} distinct={distinct}: first wrong at {bad}: " +
                    $"got ({gotK[bad]}, {gotV[bad]}), expected ({keys[order[bad]]}, {order[bad]})");
                ok = false;
            }
            else
                Console.WriteLine($"[TrainerGate] radix sort n={n:N0} bits={bits} distinct={distinct}: exact, {ms:F1} ms");
        }

        // The sort it replaces, same 2M keys, for the record.
        {
            const int n = 2_000_003;
            var keys = Enumerable.Range(0, n).Select(_ => NextKey(rng, 30)).ToArray();
            using var kBuf = accel.Allocate1D(keys);
            using var vBuf = accel.Allocate1D(Enumerable.Range(0, n).Select(i => (uint)i).ToArray());
            int temp = accel.ComputeRadixSortPairsTempStorageSize<uint, uint, AscendingUInt32>((Index1D)n);
            using var tBuf = accel.Allocate1D<int>(Math.Max(1, temp));
            var ilgpuSort = accel.CreateRadixSortPairs<uint, Stride1D.Dense, uint, Stride1D.Dense, AscendingUInt32>();
            ilgpuSort(accel.DefaultStream, kBuf.View, vBuf.View, tBuf.View); // warm-up: kernel compile
            await accel.SynchronizeAsync();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            ilgpuSort(accel.DefaultStream, kBuf.View, vBuf.View, tBuf.View);
            await accel.SynchronizeAsync();
            Console.WriteLine($"[TrainerGate] radix sort reference: ILGPU.Algorithms RadixSortPairs n={n:N0}: " +
                $"{clock.Elapsed.TotalMilliseconds:F1} ms");
        }

        Console.WriteLine(ok ? "[TrainerGate] radix sort PASS" : "[TrainerGate] FAIL: radix sort (see above)");
        return ok;

        static uint NextKey(Random r, int bits)
        {
            uint k = (uint)r.Next() ^ ((uint)r.Next() << 16);
            return bits >= 32 ? k : k & ((1u << bits) - 1u);
        }
    }
}
