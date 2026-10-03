namespace SpawnScene.Services;

/// <summary>
/// The device's GPU memory budget for training, as a user setting ("GPU memory": Auto / 2 / 4 / 8 / 16 GB), turned into
/// the two limits training needs: the resident target-photo stack and the splat cap.
/// </summary>
/// <remarks>
/// WebGPU does not report VRAM, so the budget is the user's (Auto = <see cref="AutoGB"/>). What WebGPU does report,
/// <c>maxStorageBufferBindingSize</c> (2047 MiB on an RTX 4070 in Chrome), caps the target stack, which is one binding.
/// Replaces two constants that ignored the device: a 640 MB target stack for projects (TruckFull's 251 photos at 1600
/// px would need 1.4 GB) and a splat cap taken from the preset alone.
/// MEASURED 2026-10-02 (Bathroom b112, live storage at every densify): ~250 MB fixed, then ~1,350 bytes per splat
/// (253 MB at 11,745 splats -> 817 MB at 449,523). Of that, 352 B was key buffers sized at a floor of 8 keys a splat;
/// with the floor gone and the buffers growing on demand (b127, TruckFull 1.58M splats) keys took 133 MB instead of
/// 532, and the whole was ~1,050 B: SH bank 720, other per-splat 120, Adam 112, keys ~90. With the SH Adam moments in
/// bfloat16 (b128) the SH bank is 543 and the whole ~870 (TruckFull 1.6M splats: keys 201, per-splat 183, Adam 171,
/// SH 828 MB); 900 keeps headroom for key demand, which varies by scene and view.
/// </remarks>
public static class GpuMemoryBudget
{
    /// <summary>The budget "Auto" stands for: a mid-range discrete GPU. Users with more (or less) pick it.</summary>
    public const int AutoGB = 4;

    /// <summary>The choices the settings panel offers; 0 = Auto.</summary>
    public static readonly int[] ChoicesGB = { 0, 2, 4, 6, 8, 12, 16, 24, 32, 48 };

    /// <summary>
    /// The trainer's widest per-splat buffer row: one SH part, 15 floats (SphericalHarmonics.Parts; the packed splats
    /// and the Adam moments are 14). One binding holds at most bindingLimit / this many splats: 35.8M at the RTX 4070's
    /// 2047 MiB. Before the SH bands were split into parts the row was all 45 floats and the ceiling 11.9M.
    /// </summary>
    public const long WidestSplatRowBytes = SphericalHarmonics.PartFloatsPerSplat * sizeof(float);

    /// <summary>Training GPU memory per splat beyond the fixed part (measured: b112; b127 keys on demand; b128 bf16 SH moments).</summary>
    public const long BytesPerSplat = 900;

    /// <summary>Training GPU memory that does not scale with splats or photos (frame buffers, sort scratch, pipelines).</summary>
    public const long FixedBytes = 256L * 1024 * 1024;

    /// <summary>The share of the budget the resident target photos may take.</summary>
    public const double TargetShare = 0.25;

    /// <summary>
    /// The target-stack byte budget and the splat cap for <paramref name="budgetGB"/> (0 = Auto) on a device whose
    /// storage-binding limit is <paramref name="bindingLimitBytes"/>; the cap never exceeds
    /// <paramref name="requestedMaxSplats"/> (the preset's) and never drops below 100,000.
    /// </summary>
    public static (long TargetStackBytes, int MaxSplats) Derive(int budgetGB, long bindingLimitBytes, int requestedMaxSplats)
    {
        long budget = (budgetGB > 0 ? budgetGB : AutoGB) * (1L << 30);
        long targets = Math.Min(Math.Max(bindingLimitBytes, 128L << 20), (long)(budget * TargetShare));
        long forSplats = Math.Max(0, budget - targets - FixedBytes);
        long cap = Math.Clamp(forSplats / BytesPerSplat, 100_000L, int.MaxValue);
        cap = Math.Min(cap, MaxSplatsPerBinding(bindingLimitBytes));
        return (targets, (int)Math.Min(requestedMaxSplats, cap));
    }

    /// <summary>
    /// GPU memory a training key costs across its buffers: key and value (8), three 12-byte gradient bindings (36), and
    /// sort scratch (~8). SplatTrainerGpu sizes key buffers to measured demand and grows them on overflow; this bounds
    /// how far they may grow.
    /// </summary>
    public const long BytesPerKey = 52;

    /// <summary>The share of the budget key buffers may grow into (demand spikes: large early splats, close views).</summary>
    public const double KeyShare = 0.25;

    /// <summary>
    /// The trainer's total key cap (SplatTrainerGpu.MaxTotalKeys) for <paramref name="budgetGB"/> (0 = Auto): a quarter
    /// of the budget at <see cref="BytesPerKey"/>, never past one 12-byte gradient binding, never under 4M. It was a
    /// fixed 40M (2 GB of keys), measured on a 12 GB card: more than a 2-4 GB budget holds, and a ceiling that 36M-splat
    /// scenes (~3 keys each) would hit on a 48 GB one.
    /// </summary>
    public static long MaxTotalKeys(int budgetGB, long bindingLimitBytes)
    {
        long budget = (budgetGB > 0 ? budgetGB : AutoGB) * (1L << 30);
        long byShare = (long)(budget * KeyShare) / BytesPerKey;
        return Math.Clamp(byShare, 4_000_000L, Math.Max(4_000_000L, bindingLimitBytes / (3 * sizeof(float))));
    }

    /// <summary>The most splats one trainer binding holds on a device with <paramref name="bindingLimitBytes"/>.</summary>
    public static long MaxSplatsPerBinding(long bindingLimitBytes) => bindingLimitBytes / WidestSplatRowBytes;

    /// <summary>
    /// The device's <c>maxStorageBufferBindingSize</c>, or the 128 MiB WebGPU guarantee when it reports nothing.
    /// </summary>
    public static long ReadMaxStorageBindingBytes(SpawnDev.SpawnJS.JSObjects.GPUDevice? device)
    {
        const long Guaranteed = 128L * 1024 * 1024;
        try
        {
            using var limits = device?.JSRef?.Get<SpawnDev.SpawnJS.SpawnJSObject>("limits");
            double? reported = limits?.JSRef?.Get<double?>("maxStorageBufferBindingSize");
            return reported is > 0 ? Math.Max(Guaranteed, (long)reported.Value) : Guaranteed;
        }
        catch
        {
            return Guaranteed;
        }
    }
}
