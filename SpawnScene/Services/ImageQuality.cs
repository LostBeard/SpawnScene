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

    /// <summary>One colour channel (0 = R, 1 = G, 2 = B) of interleaved RGB as a plane.</summary>
    public static double[] ChannelPlane(ReadOnlySpan<float> rgb, int width, int height, int channel)
    {
        var y = new double[width * height];
        for (int i = 0; i < y.Length; i++) y[i] = rgb[i * 3 + channel];
        return y;
    }

    /// <summary>
    /// Mean SSIM per RGB channel, averaged over the three channels: the reference 3DGS training loss's SSIM
    /// (<c>utils/loss_utils.py</c> ssim: <c>conv2d(..., groups=channel)</c>, then <c>.mean()</c> over every channel
    /// and pixel). Same window, stabilisers and 'valid' windows as <see cref="MeanSsim"/>; only the luma
    /// projection differs. <see cref="MeanSsim"/> (luma) stays the SCORING metric so reports remain comparable.
    /// </summary>
    public static double MeanSsimRgb(ReadOnlySpan<float> rgbA, ReadOnlySpan<float> rgbB, int width, int height)
    {
        double sum = 0;
        for (int c = 0; c < 3; c++)
            sum += MeanSsimLuma(ChannelPlane(rgbA, width, height, c), ChannelPlane(rgbB, width, height, c), width, height);
        return sum / 3;
    }

    /// <summary>
    /// Gradient of <see cref="MeanSsimRgb"/> w.r.t. image A, times <paramref name="scale"/>, added into
    /// <paramref name="dRgbA"/>. Each channel carries its own full SSIM gradient (at 1/3), where the luma form
    /// hands blue 0.114 of one shared gradient.
    /// </summary>
    public static void AddMeanSsimRgbGradient(
        ReadOnlySpan<float> rgbA, ReadOnlySpan<float> rgbB,
        int width, int height, Span<float> dRgbA, double scale = 1.0)
    {
        if (width < WindowSize || height < WindowSize)
            throw new ArgumentException(
                $"{width}x{height} is smaller than the {WindowSize}x{WindowSize} SSIM window");
        int nPix = width * height;
        if (rgbA.Length < nPix * 3 || rgbB.Length < nPix * 3 || dRgbA.Length < nPix * 3)
            throw new ArgumentException("RGB buffers shorter than width*height*3");
        var d = new double[nPix];
        for (int c = 0; c < 3; c++)
        {
            Array.Clear(d);
            AddMeanSsimLumaGradient(ChannelPlane(rgbA, width, height, c), ChannelPlane(rgbB, width, height, c),
                width, height, d, scale / 3);
            for (int i = 0; i < nPix; i++) dRgbA[i * 3 + c] += (float)d[i];
        }
    }

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

    /// <summary>
    /// Reference training mix: <c>0.8 * L1 + 0.2 * (1 - SSIM)</c>. Kept here so the loss
    /// shader and the host cannot disagree about the weights.
    /// </summary>
    public const float LambdaDssim = 0.2f;
    public const float LambdaL1 = 1f - LambdaDssim;

    /// <summary>
    /// Gradient of <see cref="MeanSsim"/> w.r.t. interleaved RGB of image A, accumulated into
    /// <paramref name="dRgbA"/>. Image B is the (constant) reference.
    ///
    /// The training loss uses <c>lambda * (1 - SSIM)</c>, so callers wanting that gradient pass
    /// <paramref name="scale"/> = <c>-lambda</c>.
    /// </summary>
    public static void AddMeanSsimGradient(
        ReadOnlySpan<float> rgbA, ReadOnlySpan<float> rgbB,
        int width, int height, Span<float> dRgbA, double scale = 1.0)
    {
        if (width < WindowSize || height < WindowSize)
            throw new ArgumentException(
                $"{width}x{height} is smaller than the {WindowSize}x{WindowSize} SSIM window");
        int nPix = width * height;
        if (rgbA.Length < nPix * 3 || rgbB.Length < nPix * 3 || dRgbA.Length < nPix * 3)
            throw new ArgumentException("RGB buffers shorter than width*height*3");

        var a = LumaPlane(rgbA, width, height);
        var b = LumaPlane(rgbB, width, height);
        var dLuma = new double[nPix];
        AddMeanSsimLumaGradient(a, b, width, height, dLuma, scale);

        for (int i = 0; i < nPix; i++)
        {
            int o = i * 3;
            double g = dLuma[i];
            dRgbA[o] += (float)(g * LumaR);
            dRgbA[o + 1] += (float)(g * LumaG);
            dRgbA[o + 2] += (float)(g * LumaB);
        }
    }

    /// <summary>
    /// Gradient of mean SSIM on luma planes w.r.t. plane A. Separated so the GPU path and the
    /// finite-difference gate can share one definition.
    /// </summary>
    public static void AddMeanSsimLumaGradient(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b,
        int width, int height, Span<double> dA, double scale = 1.0)
    {
        var k = GaussianKernel1D();
        int wx = width - WindowSize + 1;
        int wy = height - WindowSize + 1;
        int nWin = wx * wy;
        double invN = scale / nWin;

        // Horizontal pass - same layout as MeanSsimLuma / ssim_rows.
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

        // dL/d(rows) from every window's vertical filter adjoint.
        var dRows = new double[rows.Length];
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

                double mu1 = m0, mu2 = m1;
                double s1 = m2 - mu1 * mu1;
                double s2 = m3 - mu2 * mu2;
                double s12 = m4 - mu1 * mu2;

                double A = mu1 * mu1 + mu2 * mu2 + C1;
                double B = s1 + s2 + C2;
                double C = 2 * mu1 * mu2 + C1;
                double D = 2 * s12 + C2;
                double den = A * B;
                // dL/dS = invN for L = scale * mean(S); training uses scale = -lambda for (1-S).
                double dLdS = invN;
                // S = C*D/den. Differentiate w.r.t. M0..M4 through mu/s.
                // Holding the other M fixed:
                //   dmu1/dM0=1, ds1/dM0=-2*mu1, ds12/dM0=-mu2
                //   dA/dM0=2*mu1, dB/dM0=-2*mu1, dC/dM0=2*mu2, dD/dM0=-2*mu2
                double dCdM0 = 2 * mu2, dDdM0 = -2 * mu2, dAdM0 = 2 * mu1, dBdM0 = -2 * mu1;
                double dSdM0 = ((dCdM0 * D + C * dDdM0) * den - C * D * (dAdM0 * B + A * dBdM0))
                    / (den * den);

                // M1 = mu2: ds2/dM1=-2*mu2, ds12/dM1=-mu1
                double dCdM1 = 2 * mu1, dDdM1 = -2 * mu1, dAdM1 = 2 * mu2, dBdM1 = -2 * mu2;
                double dSdM1 = ((dCdM1 * D + C * dDdM1) * den - C * D * (dAdM1 * B + A * dBdM1))
                    / (den * den);

                // M2 = E[a^2]: ds1=1, dB=1 → dS = -C*D*A / den^2 = -S/B
                double dSdM2 = -C * D * A / (den * den);
                // M3 = E[b^2]: same through B
                double dSdM3 = -C * D * A / (den * den);
                // M4 = E[ab]: ds12=1, dD=2 → dS = C*2 / den
                double dSdM4 = 2 * C / den;

                double g0 = dLdS * dSdM0;
                double g1 = dLdS * dSdM1;
                double g2 = dLdS * dSdM2;
                double g3 = dLdS * dSdM3;
                double g4 = dLdS * dSdM4;

                for (int t = 0; t < WindowSize; t++)
                {
                    double wt = k[t];
                    int o = ((y + t) * wx + x) * 5;
                    dRows[o] += wt * g0;
                    dRows[o + 1] += wt * g1;
                    dRows[o + 2] += wt * g2;
                    dRows[o + 3] += wt * g3;
                    dRows[o + 4] += wt * g4;
                }
            }
        }

        // Horizontal adjoint: rows depend on luma a (and b, which we do not differentiate).
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < wx; x++)
            {
                int o = (y * wx + x) * 5;
                double dSa = dRows[o];
                double dSaa = dRows[o + 2];
                double dSab = dRows[o + 4];
                // d(sa)/d(va)=wt, d(saa)/d(va)=2*wt*va, d(sab)/d(va)=wt*vb
                for (int t = 0; t < WindowSize; t++)
                {
                    double wt = k[t];
                    int p = y * width + x + t;
                    double va = a[p];
                    double vb = b[p];
                    dA[p] += wt * (dSa + 2 * dSaa * va + dSab * vb);
                }
            }
        }
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
