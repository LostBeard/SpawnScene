namespace SpawnScene.Services;

/// <summary>
/// PSNR and SSIM, as the single source of truth for both the CPU oracle and the GPU shader.
///
/// Why this class exists rather than a second SSIM next to the shader: the consistency screen in
/// <c>DepthToGaussianKernel</c> had a CPU "oracle twin" that its GPU kernel never actually
/// called, so the two drifted and the tests only ever covered the copy nobody ran. A WGSL kernel
/// cannot call C#, so the next best thing is that it holds no CONSTANTS of its own - every
/// weight, stabiliser and luma coefficient below is uploaded to the shader in a uniform, and the
/// only value duplicated in WGSL is the window size, which sizes a loop and is asserted against
/// <see cref="WindowSize"/> at allocation time.
///
/// The definition is transcribed from <c>tools/score_novel_view.py</c>, which scores the
/// TempleRing novel-view runs. Those two must agree or the same reconstruction gets two
/// different scores depending on which path measured it, so
/// <c>ImageQualityTests.MatchesPythonOracle</c> pins this against a number produced by running
/// that script.
///
/// Computed in double. This is the DEFINITION; the GPU is the implementation, and is compared
/// against it with a tolerance that allows for f32. An oracle that imitates the shader's
/// precision would be checking that two approximations agree rather than that one is right.
/// </summary>
public static class ImageQuality
{
    /// <summary>
    /// SSIM window, in pixels. Wang et al.'s 11x11, matching <c>_gaussian_kernel(size=11)</c>.
    ///
    /// The WGSL sizes a loop with a literal 11 because a shader cannot read this; the allocation
    /// path asserts the two agree rather than trusting them to.
    /// </summary>
    public const int WindowSize = 11;

    /// <summary>Gaussian sigma, matching <c>_gaussian_kernel(sigma=1.5)</c>.</summary>
    public const double WindowSigma = 1.5;

    /// <summary>SSIM luminance stabiliser, (0.01 * L)^2 with L = 1 for images in [0,1].</summary>
    public const double C1 = 0.01 * 0.01;

    /// <summary>SSIM contrast stabiliser, (0.03 * L)^2.</summary>
    public const double C2 = 0.03 * 0.03;

    // Rec.601 luma, matching luma() in tools/score_novel_view.py. SSIM is measured on luma
    // rather than per channel, so a colour shift that leaves structure intact does not read as
    // structural damage.
    public const double LumaR = 0.299, LumaG = 0.587, LumaB = 0.114;

    /// <summary>
    /// The separable half of the SSIM window: an 11-tap Gaussian normalised to sum 1.
    ///
    /// The Python builds its 2-D kernel as <c>np.outer(g, g)</c> AFTER normalising g, so the 2-D
    /// kernel is separable and already sums to 1. That is not a detail - it is the fact that
    /// lets the GPU do two 11-tap passes instead of one 121-tap pass, which at 720x540 is the
    /// difference between about 1.1 GB of load traffic per view and about 100 MB.
    /// </summary>
    public static double[] GaussianKernel1D()
    {
        var k = new double[WindowSize];
        double centre = (WindowSize - 1) / 2.0;
        double sum = 0;
        for (int i = 0; i < WindowSize; i++)
        {
            double d = i - centre;
            k[i] = Math.Exp(-(d * d) / (2.0 * WindowSigma * WindowSigma));
            sum += k[i];
        }
        for (int i = 0; i < WindowSize; i++) k[i] /= sum;
        return k;
    }

    /// <summary>Rec.601 luma plane from interleaved RGB (length <c>w * h * 3</c>).</summary>
    public static double[] LumaPlane(ReadOnlySpan<float> rgb, int width, int height)
    {
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException(
                $"{rgb.Length} floats is short of {(long)width * height * 3} for {width}x{height} RGB",
                nameof(rgb));

        var y = new double[width * height];
        for (int i = 0; i < y.Length; i++)
        {
            int o = i * 3;
            y[i] = rgb[o] * LumaR + rgb[o + 1] * LumaG + rgb[o + 2] * LumaB;
        }
        return y;
    }

    /// <summary>
    /// Mean SSIM on luma over the 'valid' windows, i.e. <c>(height-10) * (width-10)</c> of them.
    /// Inputs are interleaved RGB in [0,1].
    /// </summary>
    public static double MeanSsim(ReadOnlySpan<float> rgbA, ReadOnlySpan<float> rgbB,
        int width, int height)
        => MeanSsimLuma(LumaPlane(rgbA, width, height), LumaPlane(rgbB, width, height),
            width, height);

    /// <summary>
    /// Mean SSIM of two luma planes. Separated out so the browser gate can score planes it
    /// already holds without rebuilding them.
    /// </summary>
    public static double MeanSsimLuma(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, int width, int height)
    {
        if (width < WindowSize || height < WindowSize)
            throw new ArgumentException(
                $"{width}x{height} is smaller than the {WindowSize}x{WindowSize} SSIM window");

        var k = GaussianKernel1D();
        int wx = width - WindowSize + 1;
        int wy = height - WindowSize + 1;

        // Horizontal pass: five filtered channels per (x, y), exactly what the GPU's ssim_rows
        // writes. Keeping the same decomposition on both sides means a disagreement points at
        // one pass rather than at "SSIM".
        var rows = new double[wx * height * 5];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < wx; x++)
            {
                double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int t = 0; t < WindowSize; t++)
                {
                    double wt = k[t];
                    double va = a[y * width + x + t];
                    double vb = b[y * width + x + t];
                    sa += wt * va;
                    sb += wt * vb;
                    saa += wt * va * va;
                    sbb += wt * vb * vb;
                    sab += wt * va * vb;
                }
                int o = (y * wx + x) * 5;
                rows[o] = sa; rows[o + 1] = sb; rows[o + 2] = saa;
                rows[o + 3] = sbb; rows[o + 4] = sab;
            }
        }

        // Vertical pass and the SSIM formula, averaged over windows.
        double total = 0;
        for (int y = 0; y < wy; y++)
        {
            for (int x = 0; x < wx; x++)
            {
                double m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
                for (int t = 0; t < WindowSize; t++)
                {
                    double wt = k[t];
                    int o = ((y + t) * wx + x) * 5;
                    m0 += wt * rows[o];
                    m1 += wt * rows[o + 1];
                    m2 += wt * rows[o + 2];
                    m3 += wt * rows[o + 3];
                    m4 += wt * rows[o + 4];
                }

                double muA = m0, muB = m1;
                // The BIASED variance, as the reference computes it. On a flat window this can
                // come out very slightly negative through cancellation, and the Python does not
                // clamp it. Neither do we: a max(.., 0) "safety" clamp here would be a silent
                // disagreement with the definition, which is the drift this class exists to stop.
                double sa2 = m2 - muA * muA;
                double sb2 = m3 - muB * muB;
                double sab = m4 - muA * muB;

                double num = (2 * muA * muB + C1) * (2 * sab + C2);
                double den = (muA * muA + muB * muB + C1) * (sa2 + sb2 + C2);
                total += num / den;
            }
        }
        return total / (wx * wy);
    }

    /// <summary>PSNR of two interleaved RGB buffers in [0,1], matching the Python's psnr().</summary>
    public static double Psnr(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sse = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sse += d * d;
        }
        double mse = sse / a.Length;
        return mse <= 1e-12 ? 99.0 : 10.0 * Math.Log10(1.0 / mse);
    }

    /// <summary>
    /// A deterministic pair of images with mid-range SSIM, shared by the NUnit test and the
    /// browser gate.
    ///
    /// Analytic rather than random on purpose: a .NET <c>Random</c> sequence cannot be
    /// reproduced in NumPy, and the whole point of the fixture is that C#, WGSL and Python can
    /// all score the SAME pixels and be required to agree. B is A shifted by one pixel, gained
    /// and offset, plus a low-amplitude ripple - so SSIM is well away from both 1.0 (which a
    /// broken kernel reaches trivially) and 0.
    /// </summary>
    public static (float[] A, float[] B) AnalyticFixture(int width, int height)
    {
        var a = new float[width * height * 3];
        var b = new float[width * height * 3];

        static double Sample(int x, int y, int c) =>
            0.5 + 0.45 * Math.Sin(0.21 * x + 0.7 * c) * Math.Cos(0.17 * y);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int o = (y * width + x) * 3 + c;
                    a[o] = (float)Sample(x, y, c);
                    double shifted = 0.9 * Sample(x + 1, y, c) + 0.04
                                     + 0.06 * Math.Sin(0.05 * x * y);
                    b[o] = (float)Math.Clamp(shifted, 0.0, 1.0);
                }
            }
        }
        return (a, b);
    }
}
