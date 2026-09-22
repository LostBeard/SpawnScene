using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for <see cref="ImageQuality"/>, the SSIM/PSNR definition shared by the CPU oracle and
/// the GPU shader.
///
/// Why this matters more than it looks: today the reconstruction's PSNR went 12.50 -> 12.21 dB
/// across a training run - essentially flat - while the render visibly melted from a
/// recognisable room into fog. PSNR on a sparse reconstruction is dominated by large smooth
/// regions, so smoothing structure away barely moves it. SSIM is the metric that sees that, and
/// it is about to become the number the optimiser work is judged by. A metric nobody has pinned
/// is not a measurement.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter ImageQuality</c>
/// </summary>
public class ImageQualityTests
{
    [Test]
    public void KernelIsNormalisedAndSymmetric()
    {
        var k = ImageQuality.GaussianKernel1D();
        Assert.That(k, Has.Length.EqualTo(ImageQuality.WindowSize));

        double sum = 0;
        foreach (double v in k) sum += v;
        Assert.That(sum, Is.EqualTo(1.0).Within(1e-12),
            "the 1-D kernel is normalised BEFORE the outer product, which is what makes the " +
            "2-D window sum to 1 and lets the GPU run it as two separable passes");

        for (int i = 0; i < k.Length; i++)
            Assert.That(k[i], Is.EqualTo(k[k.Length - 1 - i]).Within(1e-12));
    }

    [Test]
    public void IdenticalImagesScoreOne()
    {
        var (a, _) = ImageQuality.AnalyticFixture(40, 32);
        Assert.That(ImageQuality.MeanSsim(a, a, 40, 32), Is.EqualTo(1.0).Within(1e-9));
    }

    [Test]
    public void SymmetricInItsArguments()
    {
        var (a, b) = ImageQuality.AnalyticFixture(40, 32);
        Assert.That(ImageQuality.MeanSsim(a, b, 40, 32),
            Is.EqualTo(ImageQuality.MeanSsim(b, a, 40, 32)).Within(1e-12));
    }

    /// <summary>
    /// The load-bearing test: this implementation must agree with the scorer that produces the
    /// TempleRing novel-view numbers. Two scorers that disagree mean the same reconstruction
    /// gets two different scores depending on which path measured it.
    ///
    /// Regenerate the constant with:
    /// <code>
    /// python -c "
    /// import sys, numpy as np; sys.path.insert(0,'tools'); import score_novel_view as s
    /// W,H=64,48
    /// x=np.arange(W)[None,:,None]; y=np.arange(H)[:,None,None]; c=np.arange(3)[None,None,:]
    /// f=lambda xx: 0.5+0.45*np.sin(0.21*xx+0.7*c)*np.cos(0.17*y)
    /// A=f(x); B=np.clip(0.9*f(x+1)+0.04+0.06*np.sin(0.05*x*y),0,1)
    /// A=A.astype(np.float32).astype(np.float64); B=B.astype(np.float32).astype(np.float64)
    /// print('%.17g' % s.ssim(s.luma(A), s.luma(B)))"
    /// </code>
    ///
    /// If this fails after a change to tools/score_novel_view.py, the TOOL moved and this number
    /// must move with it. Do not adjust the C# to make it pass.
    /// </summary>
    [Test]
    public void MatchesPythonOracle()
    {
        const double PythonSsim = 0.82332333134893054;
        const double PythonPsnr = 23.851795289471923;

        var (a, b) = ImageQuality.AnalyticFixture(64, 48);

        Assert.That(ImageQuality.MeanSsim(a, b, 64, 48), Is.EqualTo(PythonSsim).Within(1e-6));
        Assert.That(ImageQuality.Psnr(a, b), Is.EqualTo(PythonPsnr).Within(1e-6));
    }

    /// <summary>
    /// The fixture has to be able to FAIL, or agreeing on it proves nothing. Measured against
    /// the Python for the same three cases: a one-pixel misalignment moves SSIM by 0.026, which
    /// is 260x the tolerance the GPU gate uses, and flipping an axis takes it to -0.03.
    /// </summary>
    [Test]
    public void FixtureIsSensitiveToMisalignmentAndAxisSwap()
    {
        const int W = 64, H = 48;
        var (a, b) = ImageQuality.AnalyticFixture(W, H);
        double baseline = ImageQuality.MeanSsim(a, b, W, H);

        // Flip b vertically: an implementation that transposed or mis-strode its rows would
        // score something like this instead of the baseline.
        var flipped = new float[b.Length];
        for (int y = 0; y < H; y++)
            System.Array.Copy(b, (H - 1 - y) * W * 3, flipped, y * W * 3, W * 3);

        double flippedSsim = ImageQuality.MeanSsim(a, flipped, W, H);
        Assert.That(System.Math.Abs(baseline - flippedSsim), Is.GreaterThan(0.5),
            "a vertical flip must be obvious, or the fixture cannot catch an axis mistake");
        Assert.That(baseline, Is.LessThan(0.98).And.GreaterThan(0.05),
            "the fixture must sit away from both ends, or agreement on it is vacuous");
    }

    [Test]
    public void RefusesAnImageSmallerThanTheWindow()
    {
        var (a, b) = ImageQuality.AnalyticFixture(8, 8);
        Assert.Throws<System.ArgumentException>(() => ImageQuality.MeanSsim(a, b, 8, 8),
            "the Python oracle raises here too; silently returning a number would be worse");
    }

    /// <summary>
    /// The D-SSIM term is about to drive densification. A wrong sign or a missing chain-rule
    /// factor would push geometry the wrong way and look like "densify still does nothing".
    /// Central finite difference on the analytic fixture is the gate.
    /// </summary>
    [Test]
    public void MeanSsimGradient_MatchesCentralFiniteDifference()
    {
        const int W = 32, H = 28;
        var (a, b) = ImageQuality.AnalyticFixture(W, H);
        var grad = new float[a.Length];
        ImageQuality.AddMeanSsimGradient(a, b, W, H, grad);

        // Sparse sample: every 17th channel across the interior so the test stays fast but
        // still covers pixels that sit in many overlapping windows and ones near the crop.
        const float Eps = 1e-3f;
        int checkedN = 0;
        double maxRel = 0, maxAbs = 0;
        for (int i = 0; i < a.Length; i += 17)
        {
            float save = a[i];
            a[i] = save + Eps;
            double plus = ImageQuality.MeanSsim(a, b, W, H);
            a[i] = save - Eps;
            double minus = ImageQuality.MeanSsim(a, b, W, H);
            a[i] = save;

            double numeric = (plus - minus) / (2 * Eps);
            double analytic = grad[i];
            double abs = System.Math.Abs(analytic - numeric);
            if (abs > maxAbs) maxAbs = abs;
            double floor = System.Math.Max(System.Math.Abs(numeric), 1e-8);
            double rel = abs / floor;
            if (rel > maxRel) maxRel = rel;
            checkedN++;

            // Near-zero grads: absolute bound. Meaningful grads: relative.
            if (System.Math.Abs(numeric) < 1e-6)
                Assert.That(abs, Is.LessThan(1e-8),
                    $"channel {i}: analytic {analytic:G6} vs numeric {numeric:G6}");
            else
                Assert.That(rel, Is.LessThan(0.02),
                    $"channel {i}: analytic {analytic:G6} vs numeric {numeric:G6} (rel {rel:G3})");
        }
        Assert.That(checkedN, Is.GreaterThan(20));
        Assert.That(maxAbs, Is.LessThan(2e-4),
            $"largest absolute FD disagreement {maxAbs:G4}");
        Assert.That(maxRel, Is.LessThan(0.05),
            $"largest relative FD disagreement {maxRel:G3}");
    }

    [Test]
    public void DSsimLossGradient_IsNegatedMeanSsimGradient()
    {
        // loss = lambda * (1 - SSIM) ⇒ dL/d(pixel) = -lambda * dSSIM/d(pixel).
        const int W = 24, H = 20;
        var (a, b) = ImageQuality.AnalyticFixture(W, H);
        var dSsim = new float[a.Length];
        var dLoss = new float[a.Length];
        ImageQuality.AddMeanSsimGradient(a, b, W, H, dSsim, scale: 1.0);
        ImageQuality.AddMeanSsimGradient(a, b, W, H, dLoss, scale: -ImageQuality.LambdaDssim);
        for (int i = 0; i < a.Length; i++)
            Assert.That(dLoss[i], Is.EqualTo(-ImageQuality.LambdaDssim * dSsim[i]).Within(1e-6f));
    }
}
