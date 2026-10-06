using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// MCMC gate (&amp;mcmc=1) on the real device, three parts:
/// <list type="number">
/// <item>GpuMcmc's relocation step at 100K splats (past where ILGPU's WebGPU scan once broke): every dead slot refilled,
/// growth as asked, and every touched row exactly its parent's share by the host formula - the CPU-accelerator tests
/// cannot see a WGSL miscompile of the nested binomial loop.</item>
/// <item>The noise pass against a host replica of its hash, Box-Muller and covariance product; opaque splats never move.</item>
/// <item>The regularisers' signs: from the same state and fresh Adam, a step whose opacity and scale regularisers dwarf
/// the image gradient must shrink EVERY stepped splat's opacity and scale; the plain step must not (it is the image's call).</item>
/// </list>
/// </summary>
public partial class Studio
{
    async Task<bool> McmcGateAsync(SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        var accel = _gpuService.WebGPUAccelerator;
        const int F = SplatFormat.Floats;

        // -- 1. relocation --
        {
            const int m = 100_000;
            var rng = new Random(5);
            var rows = new float[m * F];
            int deadIn = 0;
            for (int i = 0; i < m; i++)
            {
                int o = i * F;
                for (int c = 0; c < 6; c++) rows[o + c] = (float)rng.NextDouble();
                float s = 0.01f + (float)rng.NextDouble() * 0.04f;
                rows[o + 6] = s; rows[o + 7] = s * 0.6f; rows[o + 8] = s * 0.3f;
                bool dead = rng.NextDouble() < 0.1;
                if (dead) deadIn++;
                rows[o + 9] = dead ? (float)rng.NextDouble() * GpuMcmc.MinOpacity : 0.01f + (float)rng.NextDouble() * 0.99f;
                var q = System.Numerics.Quaternion.Normalize(new System.Numerics.Quaternion(
                    (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, 1f));
                rows[o + 10] = q.X; rows[o + 11] = q.Y; rows[o + 12] = q.Z; rows[o + 13] = q.W;
            }
            using var packed = accel.Allocate1D(rows);
            int cap = m + m / 50;
            var step = await new GpuMcmc(accel).RunAsync(packed.View, m, cap, grow: true, seed: 11);
            if (step == null) { Console.WriteLine("[TrainerGate] FAIL mcmc: relocation did nothing"); return false; }
            var (r, st) = step.Value;
            float[] outRows; int[] adam, feat;
            try
            {
                outRows = await r.Packed.CopyToHostAsync<float>(0, (long)r.Count * F);
                adam = await r.AdamSources.CopyToHostAsync<int>(0, r.Count);
                feat = await r.FeatureSources.CopyToHostAsync<int>(0, r.Count);
            }
            finally { r.Packed.Dispose(); r.AdamSources.Dispose(); r.FeatureSources.Dispose(); }

            var copies = new int[m];
            int badSource = 0;
            for (int d = 0; d < r.Count; d++)
            {
                int src = feat[d];
                if (src < 0 || src >= m) { badSource++; continue; }
                if (src != d) copies[src]++;
            }
            int deadLeft = 0, deadParent = 0, wrongRow = 0, wrongAdam = 0, firstWrong = -1;
            float worst = 0f;
            for (int d = 0; d < r.Count && badSource == 0; d++)
            {
                int src = feat[d];
                if (d < m && rows[d * F + 9] <= GpuMcmc.MinOpacity && src == d) deadLeft++;
                if (rows[src * F + 9] <= GpuMcmc.MinOpacity) deadParent++;
                bool touched = copies[src] > 0;
                if (adam[d] != (touched ? -1 : d)) wrongAdam++;
                int ratio = Math.Min(copies[src] + 1, GpuMcmc.MaxRatio);
                float pa = rows[src * F + 9];
                float na = touched ? GpuMcmc.ClampOpacity(GpuMcmc.NewOpacity(pa, ratio)) : pa;
                float k = touched ? GpuMcmc.ScaleFactor(pa, GpuMcmc.NewOpacity(pa, ratio), ratio) : 1f;
                for (int c = 0; c < F; c++)
                {
                    float want = c == 9 ? na : c is >= 6 and <= 8 ? rows[src * F + c] * k : rows[src * F + c];
                    float err = MathF.Abs(outRows[d * F + c] - want) / (MathF.Abs(want) + 1e-6f);
                    if (err > worst) worst = err;
                    if (err > 1e-3f) { wrongRow++; if (firstWrong < 0) firstWrong = d; break; }
                }
            }
            Console.WriteLine($"[TrainerGate] mcmc relocation: {m:N0} splats, {st}; {r.Count:N0} out (want {cap:N0}), " +
                $"{deadIn:N0} dead in, {deadLeft} dead slots left, {deadParent} dead parents, {badSource} bad sources, " +
                $"{wrongAdam} wrong Adam sources, {wrongRow} rows off the host share (worst rel. error {worst:G3}" +
                (firstWrong >= 0 ? $", first row {firstWrong}" : "") + ")");
            if (r.Count != cap || st.Relocated != deadIn || copies.Sum() != deadIn + (cap - m)
                || deadLeft + deadParent + badSource + wrongAdam + wrongRow > 0)
            {
                Console.WriteLine("[TrainerGate] FAIL mcmc relocation");
                return false;
            }
        }

        var p0 = await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
        try
        {
            // -- 2. noise: half the splats nearly transparent, half opaque --
            {
                var rows = (float[])p0.Clone();
                for (int i = 0; i < n; i++) rows[i * F + 9] = i % 2 == 0 ? 0.001f : 0.5f;
                splatBuf.CopyFromCPU(rows);
                const float scaler = 50f; const uint seed = 3;
                trainer.InjectMcmcNoise(splatBuf.GetGPUBuffer()!, n, scaler, seed);
                var after = await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
                int movedOpaque = 0, wrong = 0, moved = 0, first = -1;
                for (int i = 0; i < n; i++)
                {
                    var want = McmcNoiseReference(rows, i, scaler, seed);
                    for (int c = 0; c < 3; c++)
                    {
                        float got = after[i * F + c] - rows[i * F + c];
                        if (rows[i * F + 9] > 0.1f) { if (got != 0f) movedOpaque++; continue; }
                        if (got != 0f) moved++;
                        if (MathF.Abs(got - want[c]) > 1e-3f * MathF.Abs(want[c]) + 1e-6f) { wrong++; if (first < 0) first = i; }
                    }
                }
                Console.WriteLine($"[TrainerGate] mcmc noise: {moved} coordinates of the transparent half moved, {wrong} off the host " +
                    $"replica{(first >= 0 ? $" (first splat {first})" : "")}, {movedOpaque} of the opaque half moved");
                if (moved == 0 || wrong > 0 || movedOpaque > 0) { Console.WriteLine("[TrainerGate] FAIL mcmc noise"); return false; }
            }

            // -- 3. regulariser signs --
            {
                var geo = new SplatTrainerGpu.GeometryStep(PositionLr: 1e-4f, LogScaleLr: 5e-3f, RotationLr: 1e-3f, MinScale: 1e-7f, MaxScale: 1f);
                async Task<float[]> StepAsync(float reg)
                {
                    splatBuf.CopyFromCPU(p0);
                    trainer.InitOptimizerState(splatBuf, n);
                    trainer.McmcOpacityReg = reg * n;
                    trainer.McmcScaleReg = reg * 3 * n;
                    await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0.01f, geometry: geo);
                    return await splatBuf.CopyToHostAsync<float>(0, (long)n * F);
                }
                var plain = await StepAsync(0f);
                var reg = await StepAsync(1e3f);
                trainer.McmcOpacityReg = 0f; trainer.McmcScaleReg = 0f;
                int stepped = 0, opDownPlain = 0, opDownReg = 0, scSteppped = 0, scDownPlain = 0, scDownReg = 0;
                for (int i = 0; i < n; i++)
                {
                    int o = i * F;
                    if (plain[o + 9] != p0[o + 9])
                    {
                        stepped++;
                        if (plain[o + 9] < p0[o + 9]) opDownPlain++;
                        if (reg[o + 9] < p0[o + 9]) opDownReg++;
                    }
                    if (plain[o + 6] != p0[o + 6])
                    {
                        scSteppped++;
                        if (plain[o + 6] < p0[o + 6]) scDownPlain++;
                        if (reg[o + 6] < p0[o + 6]) scDownReg++;
                    }
                }
                Console.WriteLine($"[TrainerGate] mcmc regularisers: opacity fell for {opDownPlain}/{stepped} stepped splats plain, " +
                    $"{opDownReg}/{stepped} regularised; scale fell for {scDownPlain}/{scSteppped} plain, {scDownReg}/{scSteppped} regularised");
                if (stepped == 0 || scSteppped == 0 || opDownReg != stepped || scDownReg != scSteppped
                    || opDownPlain == stepped || scDownPlain == scSteppped)
                {
                    Console.WriteLine("[TrainerGate] FAIL mcmc regularisers");
                    return false;
                }
            }
        }
        finally
        {
            trainer.McmcOpacityReg = 0f; trainer.McmcScaleReg = 0f;
            splatBuf.CopyFromCPU(p0);
            trainer.InitOptimizerState(splatBuf, n);
        }
        Console.WriteLine("[TrainerGate] mcmc PASS");
        return true;
    }

    /// <summary>Host replica of SplatTrainerShaders.McmcNoise for one splat: its displacement.</summary>
    static float[] McmcNoiseReference(float[] rows, int i, float scaler, uint seed)
    {
        int o = i * SplatFormat.Floats;
        float a = rows[o + 9];
        float gate = 1f / (1f + MathF.Exp(-100f * ((1f - a) - 0.995f)));
        if (gate < 1e-12f) return new float[3];
        float Normal(uint slot)
        {
            uint h1 = GpuMcmc.Hash(seed ^ GpuMcmc.Hash((uint)i * 8u + slot * 2u));
            uint h2 = GpuMcmc.Hash(seed ^ GpuMcmc.Hash((uint)i * 8u + slot * 2u + 1u) ^ 0x9e3779b9u);
            float u1 = ((float)h1 + 1f) * (1f / 4294967296f);
            float u2 = (float)h2 * (1f / 4294967296f);
            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(6.28318530718f * u2);
        }
        var q = System.Numerics.Quaternion.Normalize(new System.Numerics.Quaternion(rows[o + 10], rows[o + 11], rows[o + 12], rows[o + 13]));
        var R = System.Numerics.Matrix4x4.CreateFromQuaternion(q);   // row-vector convention: v' = v R
        var e = new System.Numerics.Vector3(Normal(0), Normal(1), Normal(2)) * (gate * scaler);
        // cov e = R S^2 R^T e with column-vector R; in row-vector form R^T e is Transform(e, R^T) = e . R-columns.
        var rt = System.Numerics.Vector3.Transform(e, System.Numerics.Matrix4x4.Transpose(R));
        var s2 = new System.Numerics.Vector3(rows[o + 6] * rows[o + 6], rows[o + 7] * rows[o + 7], rows[o + 8] * rows[o + 8]);
        var d = System.Numerics.Vector3.Transform(rt * s2, R);
        return new[] { d.X, d.Y, d.Z };
    }
}
