using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Dense geometry Adam gate (SplatTrainerGpu.DenseGeometryAdam): a splat with no gradient this step must take
/// exactly torch Adam's zero-gradient step - m and v decay, the parameter moves by the momentum it carries -
/// and with the flag off it must not move at all. After warm-up steps give every splat real momentum, the camera
/// moves forward to the scene's median depth: the splats behind it go silent while the rest still render, which
/// is the situation training produces. (A view with NO keys returns before any Adam dispatch, so turning the
/// camera fully around tests nothing - MEASURED: the first version of this gate did that and moved nothing.)
/// Silent = the raw 2D centre and conic gradients are all zero, the shader's own test.
/// </summary>
public partial class Studio
{
    async Task<bool> DenseAdamGateAsync(
        SplatTrainerGpu trainer, MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int n,
        CameraParams cam, float depthNear, float depthFar)
    {
        var geo = new SplatTrainerGpu.GeometryStep(
            PositionLr: 1e-3f, LogScaleLr: 5e-3f, RotationLr: 1e-3f, MinScale: 1e-7f, MaxScale: 1f);

        // Warm-up on the real camera: real gradients, so every splat carries momentum into the silent step.
        for (int k = 0; k < 3; k++)
            await trainer.TrainStepAsync(splatBuf, n, cam, depthNear, depthFar, colourLr: 0f, opacityLr: 0f, geometry: geo);

        var warm = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
        var depths = new float[n];
        for (int i = 0; i < n; i++)
        {
            int o = i * SplatFormat.Floats;
            depths[i] = System.Numerics.Vector3.Dot(
                new System.Numerics.Vector3(warm[o], warm[o + 1], warm[o + 2]) - cam.Position, cam.Forward);
        }
        var sortedDepths = (float[])depths.Clone();
        Array.Sort(sortedDepths);
        var away = new CameraParams
        {
            FocalX = cam.FocalX, FocalY = cam.FocalY, CenterX = cam.CenterX, CenterY = cam.CenterY,
            Width = cam.Width, Height = cam.Height, Near = cam.Near, Far = cam.Far,
            Position = cam.Position + cam.Forward * sortedDepths[n / 2], Forward = cam.Forward, Up = cam.Up,
        };

        async Task<bool[]> SilentAsync()
        {
            var g2 = await trainer.ReadGradientsAsync(n);
            var silent = new bool[n];
            for (int i = 0; i < n; i++)
            {
                int b = i * SplatTrainerGpu.GradsPerSplat;
                silent[i] = g2[b + 4] == 0f && g2[b + 5] == 0f && g2[b + 6] == 0f && g2[b + 7] == 0f && g2[b + 8] == 0f;
            }
            return silent;
        }

        bool was = SplatTrainerGpu.DenseGeometryAdam;
        try
        {
            // -- flag off: a silent splat is not stepped at all --
            SplatTrainerGpu.DenseGeometryAdam = false;
            var p0 = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
            var a0 = await trainer.ReadAdamStateAsync(n);
            await trainer.TrainStepAsync(splatBuf, n, away, depthNear, depthFar, colourLr: 0f, opacityLr: 0f, geometry: geo);
            var silentOff = await SilentAsync();
            var p1 = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
            var a1 = await trainer.ReadAdamStateAsync(n);
            int moved = 0, silentCountOff = 0;
            for (int i = 0; i < n; i++) if (silentOff[i]) silentCountOff++;
            for (int i = 0; i < n; i++)
                if (silentOff[i]) for (int k = 0; k < 10; k++)
                {
                    int slot = 4 + k;   // ADAM_POS 4, ADAM_SCALE 7, ADAM_QUAT 10
                    if (a1.M[i * SplatTrainerGpu.AdamSlots + slot] != a0.M[i * SplatTrainerGpu.AdamSlots + slot]
                        || a1.V[i * SplatTrainerGpu.AdamSlots + slot] != a0.V[i * SplatTrainerGpu.AdamSlots + slot]) moved++;
                }
            for (int i = 0; i < n; i++)
            {
                if (!silentOff[i]) continue;
                int o = i * SplatFormat.Floats;
                for (int f = 0; f < 3; f++) if (p1[o + f] != p0[o + f]) moved++;
                for (int f = 6; f < 9; f++) if (p1[o + f] != p0[o + f]) moved++;
                for (int f = 10; f < 14; f++) if (p1[o + f] != p0[o + f]) moved++;
            }
            if (moved > 0 || silentCountOff == 0 || silentCountOff == n)
            {
                Console.WriteLine($"[TrainerGate] FAIL: dense Adam off: {moved} geometry values of {silentCountOff} silent " +
                    $"splats changed (need some silent and some live of {n})");
                return false;
            }

            // -- flag on: torch Adam with a zero gradient --
            SplatTrainerGpu.DenseGeometryAdam = true;
            await trainer.TrainStepAsync(splatBuf, n, away, depthNear, depthFar, colourLr: 0f, opacityLr: 0f, geometry: geo);
            var silent = await SilentAsync();
            var p2 = await splatBuf.CopyToHostAsync<float>(0, (long)n * SplatFormat.Floats);
            var a2 = await trainer.ReadAdamStateAsync(n);
            int silentCount = 0;
            for (int i = 0; i < n; i++) if (silent[i]) silentCount++;

            int t = a2.StepCount;
            double bc1 = 1 - Math.Pow(0.9, t), bc2 = 1 - Math.Pow(0.999, t);
            int bad = 0, withMomentum = 0, first = -1;
            string firstWhat = "";
            double worst = 0;

            // One Adam step with zero gradient from (m, v); returns the new value and checks the moments.
            double Step(int i, int slot, double value, double lr)
            {
                long ab = (long)i * SplatTrainerGpu.AdamSlots + slot;
                double m = 0.9 * a1.M[ab], v = 0.999 * a1.V[ab];
                double mRel = Math.Abs(a2.M[ab] - m) / Math.Max(Math.Abs(m), 1e-30);
                double vRel = Math.Abs(a2.V[ab] - v) / Math.Max(Math.Abs(v), 1e-30);
                if ((m != 0 && mRel > 1e-5) || (v != 0 && vRel > 1e-5) || (m == 0 && a2.M[ab] != 0) || (v == 0 && a2.V[ab] != 0))
                {
                    bad++;
                    if (first < 0) { first = i; firstWhat = $"moment slot {slot}: m {a2.M[ab]:G6} want {m:G6}, v {a2.V[ab]:G6} want {v:G6}"; }
                }
                if (m != 0) withMomentum++;
                return value - lr * (m / bc1) / (Math.Sqrt(v / bc2) + 1e-15);
            }

            void Check(int i, string what, double got, double want)
            {
                double err = Math.Abs(got - want) / Math.Max(1.0, Math.Abs(want));
                worst = Math.Max(worst, err);
                if (!(err < 2e-5)) { bad++; if (first < 0) { first = i; firstWhat = $"{what}: got {got:G8} want {want:G8}"; } }
            }

            double lnMin = Math.Log(geo.MinScale), lnMax = Math.Log(geo.MaxScale);
            for (int i = 0; i < n; i++)
            {
                if (!silent[i]) continue;
                int o = i * SplatFormat.Floats;
                for (int c = 0; c < 3; c++)
                    Check(i, $"pos[{c}]", p2[o + c], Step(i, 4 + c, p1[o + c], geo.PositionLr));
                for (int c = 0; c < 3; c++)
                {
                    double ls = Math.Clamp(Step(i, 7 + c, Math.Log(p1[o + 6 + c]), geo.LogScaleLr), lnMin, lnMax);
                    Check(i, $"scale[{c}]", p2[o + 6 + c], Math.Exp(ls));
                }
                var q = new double[4];
                for (int c = 0; c < 4; c++) q[c] = Step(i, 10 + c, p1[o + 10 + c], geo.RotationLr);
                double ql = Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                for (int c = 0; c < 4; c++) Check(i, $"quat[{c}]", p2[o + 10 + c], q[c] / ql);
            }

            // A zero-momentum scene would pass trivially: nothing would move either way.
            int moving = 0;
            for (int i = 0; i < n; i++)
            {
                if (!silent[i]) continue;
                int o = i * SplatFormat.Floats;
                if (p2[o] != p1[o] || p2[o + 1] != p1[o + 1] || p2[o + 2] != p1[o + 2]) moving++;
            }
            if (bad > 0 || moving == 0 || silentCount == 0)
            {
                Console.WriteLine($"[TrainerGate] FAIL: dense Adam: {bad} wrong of {silentCount} silent splats x 10 params, " +
                    $"{moving} positions moved" + (first >= 0 ? $", first splat {first} {firstWhat}" : ""));
                return false;
            }
            Console.WriteLine($"[TrainerGate] dense Adam PASS: {silentCount}/{n} silent splats took torch Adam's " +
                $"zero-gradient step ({moving} moved, {withMomentum} moments carried, max rel err {worst:G3}); " +
                $"flag off moved none of {silentCountOff}");
            return true;
        }
        finally { SplatTrainerGpu.DenseGeometryAdam = was; }
    }
}
