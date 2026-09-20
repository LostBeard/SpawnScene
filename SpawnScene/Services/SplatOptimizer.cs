namespace SpawnScene.Services;

/// <summary>
/// Adam optimiser and the colour/opacity training loop, on CPU.
///
/// Reference implementation for the WebGPU trainer, and the thing that proves the loop
/// converges at all before any WGSL exists. If this cannot fit a synthetic target, nothing on
/// the GPU will either, and debugging it there costs GPU runs instead of milliseconds.
///
/// Parameterisation matches the reference implementation
/// (graphdeco-inria/gaussian-splatting, see <c>Research/README.md</c>), because the published
/// learning rates are meaningless otherwise:
///   - opacity is optimised as a LOGIT, read through a sigmoid
///   - colour is optimised directly (SH degree 0 = the DC term)
/// Adam uses eps = 1e-15, as the reference does; the default 1e-8 is large relative to the
/// tiny gradients late in training.
/// </summary>
public sealed class SplatOptimizer
{
    /// <summary>Reference default for the SH DC / base colour term.</summary>
    public const float DefaultColourLr = 0.0025f;

    /// <summary>Reference default (current upstream; the 2023 release shipped 0.05).</summary>
    public const float DefaultOpacityLr = 0.025f;

    const float Beta1 = 0.9f, Beta2 = 0.999f, Eps = 1e-15f;

    readonly int _n;
    readonly float[] _mColour, _vColour;   // 3 per splat
    readonly float[] _mOpacity, _vOpacity; // 1 per splat
    int _step;

    public float ColourLr { get; set; } = DefaultColourLr;
    public float OpacityLr { get; set; } = DefaultOpacityLr;

    public SplatOptimizer(int splatCount)
    {
        _n = splatCount;
        _mColour = new float[splatCount * 3];
        _vColour = new float[splatCount * 3];
        _mOpacity = new float[splatCount];
        _vOpacity = new float[splatCount];
    }

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Inverse sigmoid. The reference stores opacity in this space.</summary>
    public static float Logit(float p)
    {
        p = Math.Clamp(p, 1e-6f, 1f - 1e-6f);
        return MathF.Log(p / (1f - p));
    }

    /// <summary>One Adam step on a single parameter, returning the updated value.</summary>
    static float AdamStep(float value, float grad, float lr, int step, ref float m, ref float v)
    {
        m = Beta1 * m + (1f - Beta1) * grad;
        v = Beta2 * v + (1f - Beta2) * grad * grad;
        // Bias correction: without it the first steps are far too small, which reads as
        // "the learning rate is wrong" rather than "the optimiser is warming up".
        float mHat = m / (1f - MathF.Pow(Beta1, step));
        float vHat = v / (1f - MathF.Pow(Beta2, step));
        return value - lr * mHat / (MathF.Sqrt(vHat) + Eps);
    }

    /// <summary>
    /// Apply one optimiser step to colour and opacity.
    /// <paramref name="opacityLogits"/> is the optimised representation; the caller writes the
    /// activated value back into the splats.
    /// </summary>
    public void Step(
        List<SplatRasterizer.Splat2D> splats,
        float[] opacityLogits,
        float[] dLdColour,
        float[] dLdOpacity)
    {
        _step++;
        for (int i = 0; i < _n; i++)
        {
            var s = splats[i];

            s.R = AdamStep(s.R, dLdColour[i * 3 + 0], ColourLr, _step, ref _mColour[i * 3 + 0], ref _vColour[i * 3 + 0]);
            s.G = AdamStep(s.G, dLdColour[i * 3 + 1], ColourLr, _step, ref _mColour[i * 3 + 1], ref _vColour[i * 3 + 1]);
            s.B = AdamStep(s.B, dLdColour[i * 3 + 2], ColourLr, _step, ref _mColour[i * 3 + 2], ref _vColour[i * 3 + 2]);

            // Colour is a radiance coefficient, not a display value, but clamping to a sane
            // range keeps a bad early step from parking a splat somewhere it cannot recover from.
            s.R = Math.Clamp(s.R, 0f, 1f);
            s.G = Math.Clamp(s.G, 0f, 1f);
            s.B = Math.Clamp(s.B, 0f, 1f);

            // Chain the opacity gradient through the sigmoid: d(sigmoid)/d(logit) = a(1-a).
            float a = s.Opacity;
            float dLdLogit = dLdOpacity[i] * a * (1f - a);
            opacityLogits[i] = AdamStep(opacityLogits[i], dLdLogit, OpacityLr, _step,
                ref _mOpacity[i], ref _vOpacity[i]);
            s.Opacity = Sigmoid(opacityLogits[i]);

            splats[i] = s;
        }
    }

    /// <summary>
    /// Fit colour and opacity to one target image, geometry frozen. Returns the loss per
    /// iteration so a caller (or a test) can assert it actually goes down.
    /// </summary>
    public static float[] FitColourOpacity(
        List<SplatRasterizer.Splat2D> splats, float[] target, int width, int height,
        int iterations, SplatOptimizer? opt = null)
    {
        opt ??= new SplatOptimizer(splats.Count);

        var logits = new float[splats.Count];
        for (int i = 0; i < splats.Count; i++) logits[i] = Logit(splats[i].Opacity);

        // Geometry is frozen, so depth order never changes: sort once.
        var order = SplatRasterizer.DepthOrder(splats);

        var history = new float[iterations];
        var dCol = new float[splats.Count * 3];
        var dOpa = new float[splats.Count];

        for (int it = 0; it < iterations; it++)
        {
            var fwd = SplatRasterizer.Render(splats, width, height, order);
            history[it] = SplatRasterizer.L1(fwd.Colour, target);

            Array.Clear(dCol);
            Array.Clear(dOpa);
            var dPix = SplatRasterizer.L1Gradient(fwd.Colour, target);
            SplatRasterizer.BackwardColorOpacity(splats, fwd, dPix, dCol, dOpa);

            opt.Step(splats, logits, dCol, dOpa);
        }

        return history;
    }
}
