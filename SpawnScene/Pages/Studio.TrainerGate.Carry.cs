using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Densify carry gate: does <see cref="SplatTrainerGpu.CarryOptimizerRowsAsync"/> move every per-splat
/// optimizer row to where the CPU oracle (<see cref="SplatDensityControl.RemapFloatRows"/>,
/// <see cref="SplatTrainerGpu.RestoreAdamState"/>) says it goes?
///
/// Densify runs every 100 iterations, and nothing looked at what it hands the next step: SH bands and
/// their moments go through a GPU gather, the Adam moments through the host. A carry that drops or
/// scrambles rows produces no error - the scene just trains as if those parameters had been reset,
/// which reads as "SH does nothing" or "densify hurts" and gets tuned around instead of fixed.
/// </summary>
public partial class Studio
{
    async Task<bool> CarryGateAsync(int n)
    {
        const int Sh = SphericalHarmonics.RestFloatsPerSplat;
        const int Adam = 14;
        const int ZeroSlot = 3; // the opacity slot, as a real opacity reset carries it

        using var trainer = new SplatTrainerGpu(_gpuService);
        trainer.Initialize();
        await trainer.ResizeAsync(GateWidth, GateHeight, n);

        // Distinct, non-zero values in every row, so a dropped, zeroed or shifted row cannot match.
        static float[] Pattern(int count, float seed)
        {
            var a = new float[count];
            for (int i = 0; i < count; i++) a[i] = seed + 0.001f * i + MathF.Sin(i * 0.37f + seed);
            return a;
        }
        var sh = Pattern(n * Sh, 1f);
        var shM = Pattern(n * Sh, 2f);
        var shV = Pattern(n * Sh, 3f);
        var adamM = Pattern(n * Adam, 4f);
        var adamV = Pattern(n * Adam, 5f);
        var identity = Enumerable.Range(0, n).ToArray();
        trainer.RestoreShRest(sh, identity);
        trainer.RestoreShAdamState(new SplatTrainerGpu.ShAdamState(shM, shV), identity);
        trainer.RestoreAdamState(new SplatTrainerGpu.AdamState(adamM, adamV, 17), identity);

        // A densify's shape: survivors in a new order with one pruned, then children that take their
        // parent's features but start their moments at zero.
        const int Children = 9;
        int m = n - 1 + Children;
        var survivors = new int[m];
        var features = new int[m];
        for (int i = 0; i < n - 1; i++) { survivors[i] = n - 1 - i; features[i] = n - 1 - i; } // index 0 pruned
        for (int c = 0; c < Children; c++) { survivors[n - 1 + c] = -1; features[n - 1 + c] = 5 + 7 * c; }

        await trainer.CarryOptimizerRowsAsync(n, m, survivors, features, zeroAdamSlot: ZeroSlot);
        await trainer.ResizeAsync(GateWidth, GateHeight, m);

        var gotSh = await trainer.ReadShRestAsync(m);
        var gotShAdam = await trainer.ReadShAdamStateAsync(m);
        var gotAdam = await trainer.ReadAdamStateAsync(m);

        var wantAdamM = SplatDensityControl.RemapFloatRows(adamM, survivors, Adam);
        var wantAdamV = SplatDensityControl.RemapFloatRows(adamV, survivors, Adam);
        for (int i = 0; i < m; i++) { wantAdamM[i * Adam + ZeroSlot] = 0f; wantAdamV[i * Adam + ZeroSlot] = 0f; }

        bool ok = true;
        ok &= Same("SH rest", gotSh, SplatDensityControl.RemapFloatRows(sh, features, Sh), Sh);
        ok &= Same("SH Adam m", gotShAdam.M, SplatDensityControl.RemapFloatRows(shM, survivors, Sh), Sh);
        ok &= Same("SH Adam v", gotShAdam.V, SplatDensityControl.RemapFloatRows(shV, survivors, Sh), Sh);
        ok &= Same("Adam m", gotAdam.M, wantAdamM, Adam);
        ok &= Same("Adam v", gotAdam.V, wantAdamV, Adam);
        if (gotAdam.StepCount != 17)
        {
            Console.WriteLine($"[TrainerGate] carry FAIL: Adam step count {gotAdam.StepCount}, expected 17");
            ok = false;
        }
        Console.WriteLine(ok
            ? $"[TrainerGate] carry {n} -> {m} splats PASS (SH rest, SH moments, Adam moments, step count)"
            : "[TrainerGate] FAIL: densify carry lost optimizer rows (see above)");
        return ok;

        static bool Same(string what, float[] got, float[] want, int stride)
        {
            if (got.Length != want.Length)
            {
                Console.WriteLine($"[TrainerGate] carry FAIL: {what} has {got.Length} floats, expected {want.Length}");
                return false;
            }
            int bad = 0, first = -1, zeroed = 0;
            for (int i = 0; i < got.Length; i++)
            {
                if (got[i] == want[i]) continue;
                bad++;
                if (first < 0) first = i;
                if (got[i] == 0f) zeroed++;
            }
            if (bad == 0) return true;
            Console.WriteLine(
                $"[TrainerGate] carry FAIL: {what} {bad}/{got.Length} floats differ ({zeroed} of them zero); " +
                $"first at row {first / stride} slot {first % stride}: got {got[first]}, expected {want[first]}");
            return false;
        }
    }
}
