namespace SpawnScene.Services;

/// <summary>
/// CPU reference for the differentiable splat rasteriser: alpha compositing forward, and the
/// gradients the optimiser needs back out of it.
///
/// This is the ORACLE, not the shipped renderer. The WGSL compute rasteriser must reproduce it.
/// It deliberately has no tiles - tiling is a GPU work-distribution strategy, not part of the
/// maths - so this file stays small enough to reason about and to finite-difference check.
///
/// Conventions match <see cref="SplatCovariance"/>: screen-space covariance in PIXELS squared,
/// front-to-back compositing, and the same 1/255 and 0.99 cutoffs the reference CUDA
/// implementation uses (graphdeco-inria/diff-gaussian-rasterization). Those cutoffs are not
/// cosmetic - they change the gradient, because a clamped alpha has ZERO derivative.
///
/// Gate: <c>SpawnScene.Tests/SplatRasterizerTests.cs</c>, which finite-differences every
/// analytic gradient. See <c>Research/README.md</c> for the derivations.
/// </summary>
public static class SplatRasterizer
{
    /// <summary>Below this a splat contributes nothing and is skipped (reference: 1/255).</summary>
    public const float MinAlpha = 1f / 255f;

    /// <summary>Alpha is clamped here. A CLAMPED alpha has zero gradient - see BackwardColorOpacity.</summary>
    public const float MaxAlpha = 0.99f;

    /// <summary>Once transmittance drops below this the pixel is saturated and we stop.</summary>
    public const float MinTransmittance = 1e-4f;

    /// <summary>A splat already projected to screen space, ready to composite.</summary>
    public struct Splat2D
    {
        public float Px, Py;            // screen-space centre, pixels
        public float ConicA, ConicB, ConicC;  // INVERSE 2D covariance (a, b; b, c)
        public float R, G, B;           // linear colour
        public float Opacity;           // 0..1, already through any sigmoid
        public float Depth;             // for sorting only
    }

    /// <summary>
    /// Everything the backward pass needs from the forward pass.
    /// Storing per-pixel transmittance for EVERY splat would be enormous, so - exactly as the
    /// reference CUDA kernel does - we keep only the final transmittance and the index of the
    /// last splat that contributed. Backward then walks the list in reverse and recovers each
    /// T by dividing out that splat's alpha.
    /// </summary>
    public sealed class Forward
    {
        public required int Width { get; init; }
        public required int Height { get; init; }
        /// <summary>RGB per pixel, row-major, length W*H*3.</summary>
        public required float[] Colour { get; init; }
        /// <summary>Transmittance remaining after the last contributor, length W*H.</summary>
        public required float[] FinalT { get; init; }
        /// <summary>
        /// EXCLUSIVE upper bound, per pixel, into <see cref="Order"/>: the forward stopped here.
        /// Backward must start from this index, not from the end of the list. The forward breaks
        /// early once transmittance saturates, so splats past this point never contributed - and
        /// feeding them to the reverse recurrence both invents gradient for them and corrupts the
        /// T recovery for everything nearer.
        /// </summary>
        public required int[] OrderEnd { get; init; }
        /// <summary>Depth-sorted splat order used, front to back.</summary>
        public required int[] Order { get; init; }
    }

    /// <summary>Front-to-back order. Sorting by depth ascending puts the nearest splat first.</summary>
    public static int[] DepthOrder(IReadOnlyList<Splat2D> splats)
    {
        var order = new int[splats.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => splats[a].Depth.CompareTo(splats[b].Depth));
        return order;
    }

    /// <summary>Gaussian weight of splat <paramref name="s"/> at a pixel centre, or 0 if behind the peak.</summary>
    public static float Weight(in Splat2D s, float pixelX, float pixelY)
    {
        float dx = pixelX - s.Px;
        float dy = pixelY - s.Py;
        float power = -0.5f * (s.ConicA * dx * dx + s.ConicC * dy * dy) - s.ConicB * dx * dy;
        if (power > 0f) return 0f;          // numerically behind the peak; reference skips these
        return MathF.Exp(power);
    }

    /// <summary>Composite front to back. Background is black, matching the measurement harness.</summary>
    public static Forward Render(IReadOnlyList<Splat2D> splats, int width, int height, int[]? order = null)
    {
        order ??= DepthOrder(splats);
        var colour = new float[width * height * 3];
        var finalT = new float[width * height];
        var orderEnd = new int[width * height];

        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                int p = py * width + px;
                float cx = px + 0.5f, cy = py + 0.5f;
                float t = 1f;
                float accR = 0f, accG = 0f, accB = 0f;
                int stop = order.Length;

                for (int k = 0; k < order.Length; k++)
                {
                    if (t < MinTransmittance) { stop = k; break; }
                    var s = splats[order[k]];

                    float g = Weight(in s, cx, cy);
                    if (g <= 0f) continue;
                    float alpha = MathF.Min(MaxAlpha, s.Opacity * g);
                    if (alpha < MinAlpha) continue;

                    accR += s.R * alpha * t;
                    accG += s.G * alpha * t;
                    accB += s.B * alpha * t;
                    t *= (1f - alpha);
                }

                colour[p * 3 + 0] = accR;
                colour[p * 3 + 1] = accG;
                colour[p * 3 + 2] = accB;
                finalT[p] = t;
                orderEnd[p] = stop;
            }
        }

        return new Forward
        {
            Width = width, Height = height,
            Colour = colour, FinalT = finalT, OrderEnd = orderEnd, Order = order,
        };
    }

    /// <summary>
    /// Gradients of the loss w.r.t. per-splat COLOUR and OPACITY only, with geometry frozen.
    ///
    /// This is the first optimiser milestone: it needs none of the 2D-mean / covariance /
    /// quaternion chain, which is where the algebra and the atomic traffic live.
    ///
    /// <paramref name="dLdPixel"/> is dL/d(rendered RGB), length W*H*3.
    /// Accumulates into <paramref name="dLdColour"/> (3 per splat) and
    /// <paramref name="dLdOpacity"/> (1 per splat).
    /// </summary>
    public static void BackwardColorOpacity(
        IReadOnlyList<Splat2D> splats, Forward fwd, float[] dLdPixel,
        float[] dLdColour, float[] dLdOpacity)
    {
        var order = fwd.Order;
        int width = fwd.Width, height = fwd.Height;

        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                int p = py * width + px;
                float cx = px + 0.5f, cy = py + 0.5f;

                float dLdR = dLdPixel[p * 3 + 0];
                float dLdG = dLdPixel[p * 3 + 1];
                float dLdB = dLdPixel[p * 3 + 2];

                // Walk BACK to front. Forward accumulates transmittance as a forward recurrence,
                // so differentiating it is inherently a reverse recurrence: start from the final
                // transmittance and divide out each splat's alpha to recover the T it saw.
                float t = fwd.FinalT[p];
                // Running colour contributed by everything BEHIND the current splat.
                float recR = 0f, recG = 0f, recB = 0f;

                for (int k = fwd.OrderEnd[p] - 1; k >= 0; k--)
                {
                    int i = order[k];
                    var s = splats[i];

                    float g = Weight(in s, cx, cy);
                    if (g <= 0f) continue;
                    float rawAlpha = s.Opacity * g;
                    float alpha = MathF.Min(MaxAlpha, rawAlpha);
                    if (alpha < MinAlpha) continue;

                    // Undo this splat's contribution to recover the transmittance it rendered with.
                    t /= (1f - alpha);

                    float w = alpha * t;

                    // Colour gradient: the pixel saw this splat's colour weighted by alpha*T.
                    dLdColour[i * 3 + 0] += w * dLdR;
                    dLdColour[i * 3 + 1] += w * dLdG;
                    dLdColour[i * 3 + 2] += w * dLdB;

                    // Alpha gradient: raising alpha adds this splat's colour and removes, by
                    // (1-alpha), everything behind it.
                    float dLdAlpha =
                        (s.R - recR) * t * dLdR +
                        (s.G - recG) * t * dLdG +
                        (s.B - recB) * t * dLdB;

                    // A CLAMPED alpha is constant w.r.t. opacity, so its derivative is zero.
                    // Omitting this makes the analytic gradient disagree with a finite difference
                    // on exactly the splats that matter most - the opaque ones.
                    if (rawAlpha < MaxAlpha)
                        dLdOpacity[i] += g * dLdAlpha;

                    // Fold this splat into "everything behind" for the next (nearer) splat.
                    recR = alpha * s.R + (1f - alpha) * recR;
                    recG = alpha * s.G + (1f - alpha) * recG;
                    recB = alpha * s.B + (1f - alpha) * recB;
                }
            }
        }
    }

    /// <summary>Screen-space gradients for one splat, before chaining back to 3D.</summary>
    public struct Grad2D
    {
        public float Px, Py;                 // d L / d screen-space centre
        public float ConicA, ConicB, ConicC; // d L / d inverse-covariance terms
        public float Opacity;
        public float R, G, B;
    }

    /// <summary>
    /// Full screen-space backward: colour, opacity, 2D mean AND 2D conic.
    ///
    /// Everything geometric flows through these last two. Once <c>dL/dmean2D</c> and
    /// <c>dL/dconic</c> are correct, reaching 3D position / scale / rotation is a change of
    /// variables through the projection Jacobian and <c>Sigma = R S S^T R^T</c> - no new
    /// rasterisation reasoning required.
    ///
    /// Derivation (d = pixel - mean, so dd/dmean = -1):
    ///   power = -0.5*(A dx^2 + C dy^2) - B dx dy
    ///   G     = exp(power)
    ///   dpower/ddx = -(A dx + B dy)   =&gt;  dL/dPx = dL/dG * G * (A dx + B dy)
    ///   dpower/dA  = -0.5 dx^2,  dpower/dB = -dx dy,  dpower/dC = -0.5 dy^2
    ///
    /// <c>dL/dmean2D</c> is also the quantity adaptive density control thresholds on, so it has
    /// to be right for densification to place geometry sensibly, not just for gradient descent.
    /// </summary>
    public static void Backward(
        IReadOnlyList<Splat2D> splats, Forward fwd, float[] dLdPixel, Grad2D[] grads)
    {
        var order = fwd.Order;
        int width = fwd.Width, height = fwd.Height;

        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                int p = py * width + px;
                float cx = px + 0.5f, cy = py + 0.5f;

                float dLdR = dLdPixel[p * 3 + 0];
                float dLdG_ = dLdPixel[p * 3 + 1];
                float dLdB = dLdPixel[p * 3 + 2];

                float t = fwd.FinalT[p];
                float recR = 0f, recG = 0f, recB = 0f;

                for (int k = fwd.OrderEnd[p] - 1; k >= 0; k--)
                {
                    int i = order[k];
                    var s = splats[i];

                    float dx = cx - s.Px;
                    float dy = cy - s.Py;
                    float power = -0.5f * (s.ConicA * dx * dx + s.ConicC * dy * dy) - s.ConicB * dx * dy;
                    if (power > 0f) continue;
                    float g = MathF.Exp(power);

                    float rawAlpha = s.Opacity * g;
                    float alpha = MathF.Min(MaxAlpha, rawAlpha);
                    if (alpha < MinAlpha) continue;

                    t /= (1f - alpha);
                    float w = alpha * t;

                    grads[i].R += w * dLdR;
                    grads[i].G += w * dLdG_;
                    grads[i].B += w * dLdB;

                    float dLdAlpha =
                        (s.R - recR) * t * dLdR +
                        (s.G - recG) * t * dLdG_ +
                        (s.B - recB) * t * dLdB;

                    // Past the clamp, alpha is constant: nothing upstream of it has any gradient.
                    if (rawAlpha < MaxAlpha)
                    {
                        grads[i].Opacity += g * dLdAlpha;

                        float dLdG_w = s.Opacity * dLdAlpha;   // dL/dG
                        float gdG = dLdG_w * g;                // dL/dpower

                        grads[i].Px += gdG * (s.ConicA * dx + s.ConicB * dy);
                        grads[i].Py += gdG * (s.ConicC * dy + s.ConicB * dx);

                        grads[i].ConicA += gdG * (-0.5f * dx * dx);
                        grads[i].ConicB += gdG * (-dx * dy);
                        grads[i].ConicC += gdG * (-0.5f * dy * dy);
                    }

                    recR = alpha * s.R + (1f - alpha) * recR;
                    recG = alpha * s.G + (1f - alpha) * recG;
                    recB = alpha * s.B + (1f - alpha) * recB;
                }
            }
        }
    }

    /// <summary>
    /// dL/d(rendered) for L = mean over pixels and channels of |rendered - target|.
    /// L1 is what the reference weights at 0.8; SSIM supplies the other 0.2.
    /// </summary>
    public static float[] L1Gradient(float[] rendered, float[] target)
    {
        var g = new float[rendered.Length];
        float inv = 1f / rendered.Length;
        for (int i = 0; i < rendered.Length; i++)
            g[i] = MathF.Sign(rendered[i] - target[i]) * inv;
        return g;
    }

    /// <summary>Mean absolute error, the forward half of <see cref="L1Gradient"/>.</summary>
    public static float L1(float[] rendered, float[] target)
    {
        double s = 0;
        for (int i = 0; i < rendered.Length; i++) s += Math.Abs(rendered[i] - target[i]);
        return (float)(s / rendered.Length);
    }
}
