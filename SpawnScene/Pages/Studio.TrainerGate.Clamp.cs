using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// EWA Jacobian clamp gate: the GPU gradients against the CPU oracles on a scene that HAS splats beyond the
/// reference's 1.3 x half-FOV clamp. The ordinary gate scene has none (MEASURED 2026-09-25: with the GPU clamp
/// removed from the geometry backward, the whole gate still PASSED), so it could not see the clamp at all.
/// </summary>
public partial class Studio
{
    async Task<bool> JacobianClampGateAsync(
        SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, float[] scenePacked, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var eye);
        float limX = SplatCovariance.JacobianClampLimit(cam.Width, cam.FocalX);
        float limY = SplatCovariance.JacobianClampLimit(cam.Height, cam.FocalY);

        // Gate scale (a few hundred splats): the host copy is the test's input. Every 12th splat moves outside the
        // clamp on one or both axes (alternating sides), at its own depth, and grows so its clamped footprint still
        // reaches the frame - a splat off screen with no pixels would have no gradient to compare.
        var packed = (float[])scenePacked.Clone();
        int moved = 0;
        for (int i = 0; i < n; i += 12, moved++)
        {
            int o = i * SplatFormat.Floats;
            var p = new Vector3(packed[o], packed[o + 1], packed[o + 2]);
            float tz = MathF.Max(Vector3.Dot(p - eye, fwd), 0.5f);
            float sx = (moved % 2 == 0 ? 1f : -1f) * (moved % 3 == 2 ? 0f : 1.6f * limX);
            float sy = (moved % 4 < 2 ? 1f : -1f) * (moved % 3 == 1 ? 0f : 1.4f * limY);
            var q = eye + fwd * tz + right * (sx * tz) + up * (sy * tz);
            packed[o] = q.X; packed[o + 1] = q.Y; packed[o + 2] = q.Z;
            float s = 0.5f * tz * cam.Width / cam.FocalX;
            packed[o + 6] = s; packed[o + 7] = 0.6f * s; packed[o + 8] = 0.3f * s;
            packed[o + 9] = 0.3f;
        }

        using var buf = splatBuf.Accelerator.Allocate1D<float>(packed.Length);
        buf.CopyFromCPU(packed);
        // The opacity step rewrites splats[9] from the trainer's own logits; seed them from this copy, restore after.
        trainer.SeedLogits(buf, n);
        try
        {
            if (!await GradientGateAsync(trainer, buf, packed, n, cam, depthNear, depthFar)) return false;
        }
        finally { trainer.SeedLogits(splatBuf, n); }
        Console.WriteLine($"[TrainerGate] Jacobian clamp PASS: gradients match the CPU oracle with {moved} splats beyond the " +
            $"1.3x half-FOV clamp (limits {limX:F3}, {limY:F3})");
        return true;
    }
}
