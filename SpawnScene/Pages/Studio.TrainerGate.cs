using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Gate for the WGSL tile rasteriser: does it reproduce the CPU model?
///
/// The comparison is deliberately NOT against the display renderer. That would conflate two
/// differences at once - a different algorithm (hardware blending vs compute compositing) and
/// a possible shader bug. Instead this runs the GPU kernels and
/// <see cref="SplatTileRasterizer"/> over byte-identical splat data and compares pixels. The
/// CPU model is already verified against the finite-difference-checked
/// <see cref="SplatRasterizer"/>, so any disagreement here is the shader.
///
/// A small synthetic scene is used on purpose: few enough splats to read back and compare
/// exactly, but with overlap, anisotropy and rotation so the covariance path is exercised.
///
/// Entry: <c>/studio?autotest=trainer-gate</c>
/// Logs:  <c>[TrainerGate] PASS|FAIL ...</c>
/// </summary>
public partial class Studio
{
    const int GateWidth = 128;
    const int GateHeight = 96;
    const int GateSplats = 240;

    async Task RunTrainerGateAsync()
    {
        Console.WriteLine("[TrainerGate] starting");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();
            var accel = _gpuService.WebGPUAccelerator;

            // ── A synthetic scene with known parameters ──
            var rng = new Random(20260920);
            int n = GateSplats;
            var packed = new float[n * SplatFormat.Floats];
            var centre = new Vector3(0f, 0f, 0f);
            for (int i = 0; i < n; i++)
            {
                int o = i * SplatFormat.Floats;
                // A loose cloud in front of the camera.
                packed[o + 0] = (float)(rng.NextDouble() - 0.5) * 0.6f;
                packed[o + 1] = (float)(rng.NextDouble() - 0.5) * 0.4f;
                packed[o + 2] = (float)(rng.NextDouble() - 0.5) * 0.6f;
                packed[o + 3] = 0.15f + (float)rng.NextDouble() * 0.8f;
                packed[o + 4] = 0.15f + (float)rng.NextDouble() * 0.8f;
                packed[o + 5] = 0.15f + (float)rng.NextDouble() * 0.8f;
                // Anisotropic, so the covariance path matters rather than cancelling out.
                packed[o + 6] = 0.010f + (float)rng.NextDouble() * 0.020f;
                packed[o + 7] = 0.010f + (float)rng.NextDouble() * 0.020f;
                packed[o + 8] = 0.004f + (float)rng.NextDouble() * 0.008f;
                packed[o + 9] = 0.25f + (float)rng.NextDouble() * 0.6f;
                // A real rotation, normalised.
                var q = Quaternion.Normalize(new Quaternion(
                    (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f,
                    (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f));
                packed[o + 10] = q.X; packed[o + 11] = q.Y; packed[o + 12] = q.Z; packed[o + 13] = q.W;
            }

            using var splatBuf = accel.Allocate1D<float>(packed.Length);
            splatBuf.CopyFromCPU(packed);
            await accel.SynchronizeAsync();

            var cam = new CameraParams
            {
                Width = GateWidth,
                Height = GateHeight,
                FocalX = 140f,
                FocalY = 140f,
                CenterX = GateWidth / 2f,
                CenterY = GateHeight / 2f,
                Near = 0.01f,
                Far = 100f,
                Position = new Vector3(0.35f, 0.25f, 1.1f),
            };
            cam.Forward = Vector3.Normalize(centre - cam.Position);
            cam.Up = Vector3.UnitY;

            // Depth range for the sort key. Conservative bounds around the cloud.
            float dist = Vector3.Distance(cam.Position, centre);
            float depthNear = MathF.Max(dist - 0.8f, 0.01f);
            float depthFar = dist + 0.8f;

            // ── GPU ──
            using var trainer = new SplatTrainerGpu(_gpuService);
            trainer.Initialize();
            trainer.Resize(GateWidth, GateHeight, n);
            var gpuColour = await trainer.RenderForwardAsync(splatBuf, n, cam, depthNear, depthFar);

            if (trainer.LastOverflowed)
            {
                Console.WriteLine("[TrainerGate] FAIL: key buffer overflowed");
                return;
            }
            Console.WriteLine($"[TrainerGate] gpu keys={trainer.LastKeyCount:N0}");

            // ── CPU model over the same data ──
            var cpuSplats = ProjectForCpu(packed, n, cam);
            var bin = SplatTileRasterizer.Bin(cpuSplats, GateWidth, GateHeight);
            var (cpuColour, _, _) = SplatTileRasterizer.Forward(cpuSplats, bin);
            Console.WriteLine($"[TrainerGate] cpu keys={bin.Keys.Length:N0}");

            // ── Compare ──
            double sumAbs = 0;
            float maxAbs = 0;
            int lit = 0;
            for (int i = 0; i < gpuColour.Length; i++)
            {
                float d = MathF.Abs(gpuColour[i] - cpuColour[i]);
                sumAbs += d;
                if (d > maxAbs) maxAbs = d;
                if (cpuColour[i] > 0.01f) lit++;
            }
            float meanAbs = (float)(sumAbs / gpuColour.Length);

            float gpuMean = gpuColour.Average();
            float cpuMean = cpuColour.Average();
            Console.WriteLine(
                $"[TrainerGate] lit={lit} gpuMean={gpuMean:F5} cpuMean={cpuMean:F5} " +
                $"meanAbs={meanAbs:F6} maxAbs={maxAbs:F6}");

            // A blank GPU image would trivially "agree" with a blank CPU image, so require the
            // comparison to have actually had content before believing a small difference.
            bool hasContent = lit > gpuColour.Length / 50 && cpuMean > 0.01f;
            bool close = meanAbs < 2e-3f && maxAbs < 5e-2f;

            if (!hasContent)
                Console.WriteLine("[TrainerGate] FAIL: nothing rendered - comparison is vacuous");
            else if (!close)
                Console.WriteLine("[TrainerGate] FAIL: GPU and CPU disagree");
            else
                Console.WriteLine("[TrainerGate] PASS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TrainerGate] FAIL: {ex}");
        }
    }

    /// <summary>
    /// Project packed splats to screen space the way the shader does, so the CPU model sees
    /// the same inputs. Mirrors <c>project()</c> in SplatTrainerShaders.Common.
    /// </summary>
    static List<SplatRasterizer.Splat2D> ProjectForCpu(float[] packed, int n, CameraParams cam)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);
        var list = new List<SplatRasterizer.Splat2D>(n);

        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            var p = new Vector3(packed[o + 0], packed[o + 1], packed[o + 2]);
            var rel = p - pos;
            float cx = Vector3.Dot(right, rel);
            float cy = Vector3.Dot(up, rel);
            float cz = Vector3.Dot(fwd, rel);
            if (cz <= 1e-6f) continue;

            var q = new SplatCovariance.Quat
            {
                X = packed[o + 10], Y = packed[o + 11], Z = packed[o + 12], W = packed[o + 13],
            };
            float len = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
            q = new SplatCovariance.Quat { X = q.X / len, Y = q.Y / len, Z = q.Z / len, W = q.W / len };

            var cov3 = SplatCovariance.Cov3DFromScaleQuat(
                MathF.Max(packed[o + 6], 1e-9f), MathF.Max(packed[o + 7], 1e-9f), MathF.Max(packed[o + 8], 1e-9f), q);
            var camCov = SplatCovariance.RotateToCamera(cov3,
                right.X, right.Y, right.Z, up.X, up.Y, up.Z, fwd.X, fwd.Y, fwd.Z);
            var cov2 = SplatCovariance.ProjectCov2D(camCov, cx, cy, cz, cam.FocalX, cam.FocalY);

            float det = cov2.A * cov2.C - cov2.B * cov2.B;
            if (!(det > 1e-20f)) continue;
            float invDet = 1f / det;

            list.Add(new SplatRasterizer.Splat2D
            {
                // Screen y grows DOWN, camera y grows UP.
                Px = cam.FocalX * cx / cz + cam.CenterX,
                Py = cam.CenterY - cam.FocalY * cy / cz,
                ConicA = cov2.C * invDet,
                ConicB = -cov2.B * invDet,
                ConicC = cov2.A * invDet,
                R = packed[o + 3], G = packed[o + 4], B = packed[o + 5],
                Opacity = packed[o + 9],
                Depth = cz,
            });
        }
        return list;
    }
}
