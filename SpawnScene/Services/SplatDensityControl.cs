namespace SpawnScene.Services;

/// <summary>
/// Adaptive density control: the step that actually creates geometry.
///
/// Optimising colour, opacity and the pose of a fixed set of Gaussians can only redistribute
/// what the initialisation already put there. Holes stay holes. Every surveyed method attributes
/// the bulk of 3DGS quality to cloning and splitting Gaussians where the reconstruction is still
/// wrong, and pruning the ones that stopped earning their place.
///
/// The signal is the SCREEN-SPACE POSITION GRADIENT. A Gaussian that the loss keeps trying to
/// drag across the image is one that is being asked to explain more than one thing, and the
/// answer is to give it help rather than to keep moving it:
///
///   - large gradient, SMALL footprint  -> under-reconstruction. CLONE it; the copy is free to
///                                         drift to the part its parent cannot cover.
///   - large gradient, LARGE footprint  -> over-reconstruction. SPLIT it into two smaller
///                                         children drawn from its own distribution.
///   - negligible opacity               -> PRUNE.
///
/// This class is the decision layer only, and it is deliberately a pure function of the
/// accumulated statistics: it says which splats to clone, split and drop, and what the children
/// look like. Applying that to a GPU buffer is a separate compaction pass, and keeping the two
/// apart is what makes the rules testable at all.
///
/// Thresholds follow Kerbl et al. so the published numbers mean something. The one unit
/// conversion that matters is called out on <see cref="GradientThresholdNdc"/>.
/// </summary>
public static class SplatDensityControl
{
    /// <summary>
    /// Gradient magnitude above which a Gaussian is considered under-reconstructed.
    ///
    /// The reference quotes 2e-4 in NDC, where the frame spans [-1, 1]. This codebase works in
    /// PIXELS, and ndc = 2 * pixel / width, so d/d(pixel) = (2/width) * d/d(ndc): a pixel-space
    /// gradient has to be multiplied by width/2 before it can be compared to this number.
    /// Comparing raw pixel gradients against 2e-4 would set the bar about 320x too high at
    /// 640 wide, and nothing would ever densify.
    /// </summary>
    public const float GradientThresholdNdc = 2e-4f;

    /// <summary>
    /// A Gaussian is "large" when its biggest axis exceeds this fraction of the scene extent.
    /// Large ones split, small ones clone.
    /// </summary>
    public const float PercentDense = 0.01f;

    /// <summary>Children of a split are this much smaller than the parent.</summary>
    public const float SplitScaleDivisor = 1.6f;

    /// <summary>Opacity below which a Gaussian contributes nothing worth keeping.</summary>
    public const float MinOpacity = 0.005f;

    /// <summary>Opacity every Gaussian is capped to at a reset.</summary>
    public const float OpacityResetTo = 0.01f;

    /// <summary>
    /// World-space size, as a fraction of the scene extent, above which a Gaussian is pruned
    /// as a bloated catch-all. Only applied after the first opacity reset, because before that
    /// a large Gaussian may simply not have been given the chance to shrink yet.
    /// </summary>
    public const float MaxWorldSizeFraction = 0.1f;

    /// <summary>Screen radius in pixels above which a Gaussian is pruned. Also post-reset only.</summary>
    public const float MaxScreenRadiusPx = 20f;

    /// <summary>
    /// What the training loop accumulates per Gaussian between densification steps.
    ///
    /// The average is over the iterations in which the Gaussian was actually VISIBLE, not over
    /// all iterations. A Gaussian seen in one view out of sixteen would otherwise look sixteen
    /// times less urgent than an identical one seen in all of them, purely because of where the
    /// cameras are.
    /// </summary>
    public struct Accumulator
    {
        /// <summary>Sum over iterations of |dL/d(screen position)|, in NDC units.</summary>
        public float GradientSum;

        /// <summary>Iterations in which this Gaussian contributed to any pixel.</summary>
        public int VisibleCount;

        /// <summary>Largest screen radius seen, in pixels. Used only by the post-reset prune.</summary>
        public float MaxScreenRadiusPx;

        public readonly float AverageGradient =>
            VisibleCount > 0 ? GradientSum / VisibleCount : 0f;
    }

    /// <summary>One Gaussian, as the decision layer needs to see it.</summary>
    public struct Splat
    {
        public float PosX, PosY, PosZ;
        public float ScaleX, ScaleY, ScaleZ;
        public float QuatX, QuatY, QuatZ, QuatW;
        public float Opacity;

        /// <summary>
        /// Linear RGB. Carried, never decided on: a clone is <c>var child = parent</c> and a
        /// split child starts from the parent too, so colour propagates for free - but only if
        /// it lives in this struct. Left out, every new Gaussian would be born black.
        /// </summary>
        public float ColR, ColG, ColB;

        public readonly float MaxScale => MathF.Max(ScaleX, MathF.Max(ScaleY, ScaleZ));
    }

    /// <summary>What to do to the splat set. Indices refer to the input array.</summary>
    public sealed class Plan
    {
        /// <summary>Indices to remove: everything pruned, plus every parent that was split.</summary>
        public List<int> Remove { get; } = new();

        /// <summary>New Gaussians to append, from clones and from splits.</summary>
        public List<Splat> Add { get; } = new();

        public int Cloned { get; set; }
        public int Split { get; set; }
        public int PrunedOpacity { get; set; }
        public int PrunedTooBig { get; set; }

        public override string ToString() =>
            $"clone {Cloned}, split {Split}, prune {PrunedOpacity} faint + {PrunedTooBig} bloated, " +
            $"net {Add.Count - Remove.Count:+#;-#;0}";
    }

    /// <summary>
    /// Decide what to clone, split and prune.
    ///
    /// <paramref name="sceneExtent"/> is the camera-rig radius, the same quantity the position
    /// learning rate is scaled by. <paramref name="afterFirstOpacityReset"/> enables the two
    /// size-based prunes, which the reference holds back until then.
    ///
    /// <paramref name="sampleUnitNormal"/> supplies standard normal deviates for placing split
    /// children; it is injected so the tests can make placement deterministic and check that
    /// children really are drawn from the parent's own distribution.
    /// </summary>
    public static Plan Decide(
        IReadOnlyList<Splat> splats,
        IReadOnlyList<Accumulator> stats,
        float sceneExtent,
        bool afterFirstOpacityReset,
        Func<float> sampleUnitNormal,
        int maxSplats = int.MaxValue)
    {
        if (splats.Count != stats.Count)
            throw new ArgumentException(
                $"{splats.Count} splats but {stats.Count} accumulators", nameof(stats));

        var plan = new Plan();
        float sizeSplit = PercentDense * sceneExtent;
        float maxWorldSize = MaxWorldSizeFraction * sceneExtent;

        // Budget: densification is unbounded by nature and a browser tab is not. Growth stops
        // at the cap rather than the tab dying, and pruning still runs so the set can recover.
        int budget = Math.Max(0, maxSplats - splats.Count);

        for (int i = 0; i < splats.Count; i++)
        {
            var s = splats[i];

            // -- Prune first. A Gaussian being removed must not also be densified. --
            if (s.Opacity < MinOpacity)
            {
                plan.Remove.Add(i);
                plan.PrunedOpacity++;
                continue;
            }
            if (afterFirstOpacityReset &&
                (s.MaxScale > maxWorldSize || stats[i].MaxScreenRadiusPx > MaxScreenRadiusPx))
            {
                plan.Remove.Add(i);
                plan.PrunedTooBig++;
                continue;
            }

            if (stats[i].AverageGradient < GradientThresholdNdc) continue;

            if (s.MaxScale <= sizeSplit)
            {
                // Under-reconstructed: clone. The copy starts exactly where its parent is and
                // the optimiser separates them - placing it by hand would be guessing at the
                // direction the loss is already telling us about.
                if (plan.Add.Count >= budget) continue;
                plan.Add.Add(s);
                plan.Cloned++;
            }
            else
            {
                // Over-reconstructed: split into two smaller children, positioned by sampling
                // the parent's OWN distribution. Offsetting along a fixed axis instead would
                // bias every split in the scene the same way.
                if (plan.Add.Count + 2 > budget) continue;
                plan.Add.Add(Child(s, sampleUnitNormal));
                plan.Add.Add(Child(s, sampleUnitNormal));
                plan.Remove.Add(i);
                plan.Split++;
            }
        }
        return plan;
    }

    /// <summary>
    /// One child of a split: displaced by a draw from the parent's ellipsoid, rotated into
    /// world space, and shrunk.
    /// </summary>
    static Splat Child(in Splat parent, Func<float> sampleUnitNormal)
    {
        float lx = sampleUnitNormal() * parent.ScaleX;
        float ly = sampleUnitNormal() * parent.ScaleY;
        float lz = sampleUnitNormal() * parent.ScaleZ;

        var q = new SplatCovariance.Quat
        {
            X = parent.QuatX, Y = parent.QuatY, Z = parent.QuatZ, W = parent.QuatW,
        };
        float len = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
        if (len > 1e-20f)
            q = new SplatCovariance.Quat { X = q.X / len, Y = q.Y / len, Z = q.Z / len, W = q.W / len };
        else
            q = SplatCovariance.Quat.Identity;

        // Local offset into world space by the parent's rotation.
        float xx = q.X * q.X, yy = q.Y * q.Y, zz = q.Z * q.Z;
        float xy = q.X * q.Y, xz = q.X * q.Z, yz = q.Y * q.Z;
        float wx = q.W * q.X, wy = q.W * q.Y, wz = q.W * q.Z;

        float r00 = 1f - 2f * (yy + zz), r01 = 2f * (xy - wz), r02 = 2f * (xz + wy);
        float r10 = 2f * (xy + wz), r11 = 1f - 2f * (xx + zz), r12 = 2f * (yz - wx);
        float r20 = 2f * (xz - wy), r21 = 2f * (yz + wx), r22 = 1f - 2f * (xx + yy);

        float dx = r00 * lx + r01 * ly + r02 * lz;
        float dy = r10 * lx + r11 * ly + r12 * lz;
        float dz = r20 * lx + r21 * ly + r22 * lz;

        var child = parent;
        child.PosX = parent.PosX + dx;
        child.PosY = parent.PosY + dy;
        child.PosZ = parent.PosZ + dz;
        child.ScaleX = parent.ScaleX / SplitScaleDivisor;
        child.ScaleY = parent.ScaleY / SplitScaleDivisor;
        child.ScaleZ = parent.ScaleZ / SplitScaleDivisor;
        return child;
    }

    /// <summary>
    /// Apply a plan, returning the new splat set. Removals are applied to the ORIGINAL indices
    /// and additions appended, so a plan is never invalidated by its own earlier entries.
    /// </summary>
    public static List<Splat> Apply(IReadOnlyList<Splat> splats, Plan plan)
    {
        var drop = new HashSet<int>(plan.Remove);
        var result = new List<Splat>(splats.Count - drop.Count + plan.Add.Count);
        for (int i = 0; i < splats.Count; i++)
            if (!drop.Contains(i)) result.Add(splats[i]);
        result.AddRange(plan.Add);
        return result;
    }

    /// <summary>
    /// Cap every opacity, the periodic reset from the reference. It forces the optimiser to
    /// re-earn every Gaussian: the ones that matter recover their opacity within a few hundred
    /// iterations and the ones that were only filling space stay faint and get pruned.
    /// The Adam moments for opacity must be zeroed alongside this, or momentum simply undoes it.
    /// </summary>
    public static void ResetOpacity(Splat[] splats)
    {
        for (int i = 0; i < splats.Length; i++)
            splats[i].Opacity = MathF.Min(splats[i].Opacity, OpacityResetTo);
    }

    /// <summary>
    /// Convert a screen-space position gradient from PIXELS to NDC, so it can be compared with
    /// <see cref="GradientThresholdNdc"/>. See that field for why this exists.
    /// </summary>
    public static float PixelGradientToNdc(float gradPxX, float gradPxY, int width, int height)
    {
        float nx = gradPxX * width * 0.5f;
        float ny = gradPxY * height * 0.5f;
        return MathF.Sqrt(nx * nx + ny * ny);
    }
}
