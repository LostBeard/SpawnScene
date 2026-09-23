using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// A similarity transform p' = scale * R * p + t, the exact freedom a monocular multi-view depth
/// model leaves unresolved between two independent forward passes.
///
/// <see cref="Rotation"/> is packed for the row-vector convention System.Numerics uses
/// (<c>Vector3.Transform(v, M)</c> computes R*v), matching
/// <see cref="WorldSpaceGeometry.TryUmeyamaSimilarity"/>, which produces it.
/// </summary>
public readonly record struct Similarity3(float Scale, Matrix4x4 Rotation, Vector3 Translation)
{
    public static Similarity3 Identity => new(1f, Matrix4x4.Identity, Vector3.Zero);

    /// <summary>Transform a POINT: scale, rotate, translate.</summary>
    public Vector3 Apply(Vector3 p) => Scale * Vector3.Transform(p, Rotation) + Translation;

    /// <summary>
    /// Transform a DIRECTION: rotation only. A camera's forward and up are directions, so they
    /// take neither the scale nor the translation; applying <see cref="Apply"/> to them is the
    /// classic way to get a camera that is in the right place and looking at nothing.
    /// </summary>
    public Vector3 ApplyDirection(Vector3 d) => Vector3.Normalize(Vector3.Transform(d, Rotation));

    /// <summary>
    /// Move a camera into the target frame, in place. Intrinsics are untouched: a similarity
    /// acts on the world, not on the lens, so focal length and principal point stay in pixels.
    /// </summary>
    public void ApplyToCamera(CameraParams cam)
    {
        cam.Position = Apply(cam.Position);
        cam.Forward = ApplyDirection(cam.Forward);
        cam.Up = ApplyDirection(cam.Up);
    }
}

/// <summary>
/// One joint depth-model forward pass: which images go in, and which of them are the shared
/// anchors that put its output back into the reference frame.
///
/// <see cref="Views"/> lists global image indices with the anchors FIRST, so slot i &lt;
/// <see cref="AnchorCount"/> is an anchor and the rest are views this chunk is responsible for.
/// </summary>
public sealed record MultiViewChunk(int[] Views, int AnchorCount)
{
    /// <summary>Global image indices of the shared anchor views.</summary>
    public ReadOnlySpan<int> Anchors => Views.AsSpan(0, AnchorCount);

    /// <summary>Global image indices this chunk is the first and only one to pose.</summary>
    public ReadOnlySpan<int> NewViews => Views.AsSpan(AnchorCount);
}

/// <summary>
/// Pose EVERY view in ONE frame using a joint depth model that can only see a handful at a time.
///
/// The problem this solves, measured on Bathroom (35 handheld frames, commit eeedfff): joint DAv3
/// puts all of its views' depths and cameras in one shared frame, which is worth about 5 dB of
/// initialisation over taking depths from DAv3 and cameras from SfM. But it only accepts
/// <see cref="DepthEstimationService.MaxMultiViewImages"/> views per forward, so preferring it
/// dropped supervision from 17 views to 5. Better geometry, less supervision - a fork with no
/// good side.
///
/// There is no fork. Run the model several times, with the SAME anchor views present in every
/// pass, and each pass reports those anchors in its own arbitrary frame. Two sets of camera
/// centres for the same physical cameras determine the similarity between the frames exactly
/// (<see cref="WorldSpaceGeometry.TryUmeyamaSimilarity"/>), so every chunk folds into chunk 0's
/// frame and every view ends up posed in one world - depth maps included, since the fitted scale
/// is precisely the factor that view's depths were off by.
///
/// Anchors are shared GLOBALLY rather than chained chunk-to-chunk on purpose: a chain accumulates
/// its alignment error over the sequence, while a common anchor set gives every chunk a direct,
/// independent fit to the reference frame.
/// </summary>
public static class MultiViewChunkPlan
{
    /// <summary>
    /// Fewest anchors that determine a similarity robustly. Two points fix scale and one axis but
    /// leave a free roll about the line joining them, so <see cref="WorldSpaceGeometry.TryUmeyamaSimilarity"/>
    /// would return a transform that fits the anchors and rolls everything else.
    /// </summary>
    public const int MinAnchors = 3;

    /// <summary>
    /// Anchor fit residual, as a fraction of how far the anchors spread from their own centroid,
    /// above which a chunk is not trusted into the world frame. Relative because the whole frame
    /// is only defined up to scale - an absolute metre threshold means nothing here.
    /// </summary>
    public const float MaxAnchorRmsFraction = 0.15f;

    /// <summary>
    /// How far one anchor may sit from where a candidate transform puts it, as a fraction of the
    /// anchor spread, and still count as agreeing with it. Same units and same reasoning as
    /// <see cref="MaxAnchorRmsFraction"/>; separate because one is a per-anchor test during the
    /// search and the other is the whole fit's acceptance afterwards.
    /// </summary>
    public const float InlierAnchorFraction = 0.15f;

    /// <summary>
    /// Fewest anchors a POSE-aware fold needs. A camera is a full rigid pose, not a point: its
    /// orientation alone fixes the frame rotation, so two cameras give 12 constraints for the 7
    /// a similarity has. Planning still carries <see cref="MinAnchors"/> per chunk - the third
    /// anchor is what makes a bad one identifiable, not what makes the fold solvable.
    /// </summary>
    public const int MinFoldAnchors = 2;

    /// <summary>
    /// How far one anchor's orientation may disagree with the frame rotation the fold settles on
    /// and still count as agreeing, in radians.
    ///
    /// 10 degrees. MEASURED on DrJohnson 2000 (dav3, 13 chunks, anchors 3/12/37, 2026-09-23) from
    /// the <c>[MultiView] chunk N anchor rotation disagreement</c> lines:
    ///   - honest pair 3-12: 1.0 to 3.4 degrees on every chunk (the noise floor this must sit above);
    ///   - anchor 37 (the one whose depth scale swung 0.52x to 1.43x): 52-54 (chunks 2, 7, 12),
    ///     26-27 (8), 13.7/12.9 (11), 10.5/9.0 (9), 8.9/10.0 (5), 7.4/9.7 (10), 6.7/6.1 (13),
    ///     4.3/4.0 (1, 3), 3.3/1.4 (6); chunk 4, where 37 was honest, 2.9/3.2.
    /// So orientation alone names 37 on 7 of its 12 bad passes; the position vote catches the rest
    /// (chunks 1, 3, 5, 6, 10, 13), and no honest anchor is ever rejected. Bathroom 2000
    /// (dav3, anchors 31/32/33): every disagreement 0.1-0.5 deg, so the same gate never rejects
    /// an honest Bathroom anchor either. Loosening this past the point
    /// where 37 counts as agreeing defeats the purpose; tightening below ~4 degrees rejects honest
    /// anchors.
    /// </summary>
    public const float MaxAnchorRotationRadians = 10f * MathF.PI / 180f;

    /// <summary>
    /// Split <paramref name="viewCount"/> images into chunks of at most <paramref name="chunkSize"/>,
    /// each carrying the same <paramref name="anchorCount"/> anchor views.
    ///
    /// Anchors are spread evenly across the sequence, and so are each chunk's new views: handheld
    /// frames that are adjacent in time are near-parallel, which is the worst case for a
    /// multi-view depth model, so a chunk is dealt its new views round-robin rather than as a
    /// consecutive run. Every non-anchor view appears in exactly one chunk.
    /// </summary>
    public static IReadOnlyList<MultiViewChunk> Plan(int viewCount, int chunkSize, int anchorCount)
        => Plan(viewCount, chunkSize, anchorCount, null);

    /// <summary>
    /// As above, with <paramref name="anchorsOverride"/> naming the anchor views explicitly
    /// (local indices). Passing null keeps the even spread, which suits a capture that orbits a
    /// subject; a walk-through wants anchors that actually saw each other - see
    /// <see cref="PickAnchorsByOverlap"/>.
    /// </summary>
    public static IReadOnlyList<MultiViewChunk> Plan(
        int viewCount, int chunkSize, int anchorCount, int[]? anchorsOverride)
    {
        if (viewCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(viewCount), viewCount, "No views to plan.");
        if (chunkSize < MinAnchors + 1)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), chunkSize,
                $"A chunk must hold {MinAnchors} anchors plus at least one new view.");
        if (anchorCount < MinAnchors)
            throw new ArgumentOutOfRangeException(nameof(anchorCount), anchorCount,
                $"A similarity needs at least {MinAnchors} non-collinear anchors.");
        if (anchorCount >= chunkSize)
            throw new ArgumentOutOfRangeException(nameof(anchorCount), anchorCount,
                "Anchors would fill the chunk, leaving no room for a new view.");

        // Everything fits in one pass: no second frame exists, so no anchors are needed.
        if (viewCount <= chunkSize)
            return new[] { new MultiViewChunk(Enumerable.Range(0, viewCount).ToArray(), viewCount) };

        var anchors = anchorsOverride is { Length: > 0 }
            ? anchorsOverride.Where(i => i >= 0 && i < viewCount).Distinct().OrderBy(i => i).ToArray()
            : SpreadPick(viewCount, anchorCount);
        if (anchors.Length != anchorCount)
            anchors = SpreadPick(viewCount, anchorCount);
        var anchorSet = anchors.ToHashSet();
        var rest = Enumerable.Range(0, viewCount).Where(i => !anchorSet.Contains(i)).ToArray();

        int perChunk = chunkSize - anchorCount;
        int chunkCount = (rest.Length + perChunk - 1) / perChunk;

        var buckets = new List<int>[chunkCount];
        for (int c = 0; c < chunkCount; c++) buckets[c] = new List<int>();
        // Round-robin, so each chunk's new views span the whole capture rather than a local run.
        for (int r = 0; r < rest.Length; r++) buckets[r % chunkCount].Add(rest[r]);

        var chunks = new List<MultiViewChunk>(chunkCount);
        foreach (var bucket in buckets)
        {
            if (bucket.Count == 0) continue;
            var views = new int[anchorCount + bucket.Count];
            Array.Copy(anchors, views, anchorCount);
            bucket.CopyTo(views, anchorCount);
            chunks.Add(new MultiViewChunk(views, anchorCount));
        }
        return chunks;
    }

    /// <summary>
    /// Chunks for one image shape. The largest group defines the world frame; a group of another
    /// shape is posed in a frame of its own and shares no anchors with it, so
    /// <see cref="IsReferenceGroup"/> says which one the scene is actually built in.
    /// </summary>
    public sealed record MultiViewShapeGroup(
        int Width, int Height, IReadOnlyList<MultiViewChunk> Chunks, bool IsReferenceGroup)
    {
        public int ViewCount => Chunks.Count == 0
            ? 0
            : Chunks[0].AnchorCount + Chunks.Sum(c => c.NewViews.Length);
    }

    /// <summary>
    /// Plan chunks per image SHAPE, largest group first.
    ///
    /// A quarter turn swaps width and height, and <c>QuarterTurnsToUpright</c> is decided per
    /// frame - Bathroom asked for four different answers across six cameras of one room, and it
    /// is 34 portrait frames plus one landscape before any turning. That matters here because the
    /// joint depth pass emits EVERY view at its first view's resolution
    /// (<c>DepthEstimationService.EstimateDepthMultiViewAsync</c>: <c>outW = widths[0]</c>), so a
    /// portrait frame sharing a pass with landscape ones comes back at the wrong shape and
    /// unprojects into a stretched shell with no error anywhere.
    ///
    /// Mixing rotation DIRECTIONS is harmless - clockwise and counter-clockwise both land on one
    /// of four shapes, and the similarity fit reads camera centres, which a roll does not move
    /// (<c>MultiViewChunkPlanTests.MixedPerViewImageRotations_DoNotDisturbTheFold</c>). It is the
    /// shape, not the direction, that has to be kept apart.
    /// </summary>
    public static IReadOnlyList<MultiViewShapeGroup> PlanByShape(
        IReadOnlyList<(int Width, int Height)> shapes, int chunkSize, int anchorCount)
        => PlanByShape(shapes, chunkSize, anchorCount, null);

    /// <summary>
    /// As above, with <paramref name="pickAnchors"/> choosing each group's anchors. It is handed
    /// that group's GLOBAL image indices and the count wanted, and returns GLOBAL indices;
    /// mapping back into the group is done here so a caller never has to think in local indices.
    /// </summary>
    public static IReadOnlyList<MultiViewShapeGroup> PlanByShape(
        IReadOnlyList<(int Width, int Height)> shapes, int chunkSize, int anchorCount,
        Func<IReadOnlyList<int>, int, int[]>? pickAnchors)
    {
        if (shapes.Count == 0)
            throw new ArgumentOutOfRangeException(nameof(shapes), "No views to plan.");

        var groups = new List<MultiViewShapeGroup>();
        bool first = true;
        foreach (var byShape in shapes
            .Select((s, i) => (Shape: s, Index: i))
            .GroupBy(x => x.Shape)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key.Width)
            .ThenBy(g => g.Key.Height))
        {
            var global = byShape.Select(x => x.Index).ToArray();

            int[]? localAnchors = null;
            if (pickAnchors != null && global.Length > chunkSize)
            {
                var chosen = pickAnchors(global, anchorCount);
                var toLocal = new Dictionary<int, int>();
                for (int i = 0; i < global.Length; i++) toLocal[global[i]] = i;
                localAnchors = chosen
                    .Where(toLocal.ContainsKey)
                    .Select(g => toLocal[g])
                    .Distinct()
                    .ToArray();
            }

            var local = Plan(global.Length, chunkSize, anchorCount, localAnchors);

            var mapped = local
                .Select(c => new MultiViewChunk(
                    c.Views.Select(v => global[v]).ToArray(), c.AnchorCount))
                .ToArray();

            groups.Add(new MultiViewShapeGroup(
                byShape.Key.Width, byShape.Key.Height, mapped, IsReferenceGroup: first));
            first = false;
        }
        return groups;
    }

    /// <summary>
    /// Anchors chosen for mutual OVERLAP rather than for maximal spread.
    ///
    /// <see cref="SpreadPick"/> takes the first, middle and last frame, which is right for a
    /// turntable - the object is in every shot, so spreading maximises baseline at no cost. It is
    /// close to the worst possible choice for someone WALKING through a room, because the frames
    /// furthest apart in time are the ones least likely to have seen the same wall, and a
    /// multi-view model cannot place a camera whose view it cannot tie to the others.
    ///
    /// That is what Bathroom shows. With anchors 0, 17 and 34, the 0-34 distance ratio held at
    /// 1.00 across all ten passes (1.057, 1.018, 0.908, 1.031, 0.984) while every pair involving
    /// 17 swung between 0.43 and 3.46: two anchors the model could relate, and one it could only
    /// guess at. TempleRing, where every frame sees the temple, folds to 0.4% with spread anchors.
    ///
    /// <paramref name="overlap"/> scores a pair - geometrically verified feature matches
    /// (<c>ImagePair.InlierCount</c>) are the natural measure. The first anchor is the view with
    /// the most overlap overall, and each next one MAXIMISES THE MINIMUM overlap with the anchors
    /// already chosen, so the set is mutually connected rather than merely popular. That is
    /// farthest-point sampling run backwards, which is the point.
    /// </summary>
    public static int[] PickAnchorsByOverlap(int viewCount, Func<int, int, int> overlap, int count)
    {
        if (count >= viewCount) return Enumerable.Range(0, viewCount).ToArray();
        if (count <= 0) return Array.Empty<int>();

        var totals = new long[viewCount];
        for (int i = 0; i < viewCount; i++)
            for (int j = 0; j < viewCount; j++)
                if (i != j) totals[i] += Math.Max(0, overlap(i, j));

        // Greedy from ONE seed is not enough, and the test that found this is worth keeping in
        // mind: seeded at the most-connected view, greedy took the 9000-weight edge first and was
        // then cornered into a third anchor with ZERO overlap, because its own second choice had
        // eliminated every connected candidate. A locally best first step can leave no valid
        // second one. So run it from every seed and keep the set with the best BOTTLENECK - the
        // weakest link in an anchor set is the whole strength of it.
        int[]? bestSet = null;
        long bestBottleneck = -1, bestTotal = -1;

        for (int seed = 0; seed < viewCount; seed++)
        {
            var picked = new List<int> { seed };
            while (picked.Count < count)
            {
                int best = -1, bestMin = -1;
                for (int i = 0; i < viewCount; i++)
                {
                    if (picked.Contains(i)) continue;
                    int worst = int.MaxValue;
                    foreach (int p in picked) worst = Math.Min(worst, Math.Max(0, overlap(i, p)));
                    // Ties broken by total overlap, then by index, so this is deterministic.
                    if (worst > bestMin || (worst == bestMin && best >= 0 && totals[i] > totals[best]))
                    {
                        bestMin = worst;
                        best = i;
                    }
                }
                if (best < 0) break;
                picked.Add(best);
            }
            if (picked.Count < count) continue;

            long bottleneck = long.MaxValue, total = 0;
            for (int i = 0; i < picked.Count; i++)
                for (int j = i + 1; j < picked.Count; j++)
                {
                    int w = Math.Max(0, overlap(picked[i], picked[j]));
                    bottleneck = Math.Min(bottleneck, w);
                    total += w;
                }

            if (bottleneck > bestBottleneck
                || (bottleneck == bestBottleneck && total > bestTotal))
            {
                bestBottleneck = bottleneck;
                bestTotal = total;
                picked.Sort();
                bestSet = picked.ToArray();
            }
        }

        return bestSet ?? SpreadPick(viewCount, count);
    }

    /// <summary>
    /// <paramref name="count"/> indices spread evenly over <c>[0, total)</c>, endpoints included.
    /// </summary>
    public static int[] SpreadPick(int total, int count)
    {
        if (count >= total) return Enumerable.Range(0, total).ToArray();
        var picked = new int[count];
        for (int i = 0; i < count; i++)
            picked[i] = count == 1 ? 0 : (int)Math.Round(i * (total - 1) / (double)(count - 1));
        return picked;
    }

    /// <summary>
    /// Fit the similarity carrying this chunk's frame into the reference frame, using the anchor
    /// cameras it shares with chunk 0.
    ///
    /// <paramref name="chunkCameras"/> is indexed by SLOT (parallel to <see cref="MultiViewChunk.Views"/>)
    /// and may hold nulls where the model produced nothing usable. <paramref name="reference"/>
    /// maps a global image index to its already-placed camera.
    ///
    /// Returns false - and poses nothing - when too few anchors survived, when the anchors are
    /// degenerate, or when the residual is too large a fraction of the anchor spread. A chunk
    /// that cannot be placed must not be placed approximately: its splats would land in the
    /// middle of the scene as measured-looking geometry.
    /// </summary>
    public static bool TryFitChunkToReference(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out Similarity3 similarity, out float rms, out int anchorsUsed)
        => TryFitChunkToReference(chunk, chunkCameras, reference,
            out similarity, out rms, out anchorsUsed, out _);

    /// <summary>
    /// As above, also reporting the anchor SPREAD (mean distance from their centroid in the
    /// reference frame). The residual only means anything against it, and a caller that logs
    /// both turns <see cref="MaxAnchorRmsFraction"/> from a judgement call into an observation.
    ///
    /// Worth knowing when reading those numbers: at exactly <see cref="MinAnchors"/> anchors the
    /// fit is over-determined by two, so a non-zero residual is not rounding - it says the model
    /// reported a differently SHAPED anchor triangle in this pass than in the reference one.
    /// </summary>
    public static bool TryFitChunkToReference(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out Similarity3 similarity, out float rms, out int anchorsUsed, out float anchorSpread)
        => TryFitChunkToReference(chunk, chunkCameras, reference,
            out similarity, out rms, out anchorsUsed, out anchorSpread, out _);

    /// <summary>
    /// As above, fitted ROBUSTLY, and reporting how many anchors the transform actually agrees
    /// with (<paramref name="inlierCount"/>).
    ///
    /// Least squares assumes every anchor is equally trustworthy. MEASURED on Bathroom, they are
    /// not: across ten passes the distance ratio between anchors 0 and 34 stayed at 1.00 (1.057,
    /// 1.018, 0.908, 1.031, 0.984) while every pair involving anchor 17 swung between 0.43 and
    /// 3.46. One camera was being placed somewhere different each time the model saw it in
    /// different company, and least squares answers that by spreading its error evenly over the
    /// anchors that were RIGHT - which is how one bad view drags a whole chunk out of position
    /// while the residual still looks survivable.
    ///
    /// MEASURED again on DrJohnson (dj2k-dav3-pose, anchors 3/12/37): pair 3-12 held at 1.00
    /// across all 13 passes (0.88-1.12, which is per-pass scale and is absorbed) while every pair
    /// involving 37 swung 0.52-1.43. The position-only fit rejected 8 of 13 chunks, because three
    /// POINTS over-determine a similarity by two and cannot say which of them is the liar - so
    /// the residual crossed 15% and the chunk was lost, along with 24 of 44 views.
    /// </summary>
    public static bool TryFitChunkToReference(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out Similarity3 similarity, out float rms, out int anchorsUsed, out float anchorSpread,
        out int inlierCount)
        => TryFitChunkToReference(chunk, chunkCameras, reference,
            out similarity, out rms, out anchorsUsed, out anchorSpread, out inlierCount, out _);

    /// <summary>
    /// The fold itself, also naming the anchor SLOTS the transform was fitted on
    /// (<paramref name="inlierSlots"/>), so the caller can print which anchor was thrown out.
    ///
    /// A camera is a POSE, not a point. Its orientation alone fixes the frame rotation: for anchor
    /// i the rotation carrying this pass's frame into the reference frame is simply
    /// R_i = basis_chunk(i)^T · basis_ref(i), one estimate per anchor with no fitting at all. Two
    /// anchors then give 12 constraints for a 7-parameter similarity, and three give 18 - enough
    /// redundancy to detect a bad anchor with two and to name it with three, where positions alone
    /// needed four and five.
    ///
    /// So: every anchor PAIR whose two rotation estimates agree proposes a similarity (rotation =
    /// their mean, scale = their distance ratio, translation from their midpoint). Each proposal is
    /// scored by how many of ALL the anchors agree with it in BOTH position
    /// (<see cref="InlierAnchorFraction"/> of the spread) and orientation
    /// (<see cref="MaxAnchorRotationRadians"/>). The winner is refitted on its inliers. A pair
    /// fits its own positions exactly, so the independent evidence behind a two-anchor fold is
    /// that the two cameras' RELATIVE rotation matches the reference - which the model cannot get
    /// right by accident for two views it has placed inconsistently.
    ///
    /// Deterministic and exhaustive: C(n,2) proposals, no randomness, so a diagnostic reproduces.
    /// </summary>
    public static bool TryFitChunkToReference(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out Similarity3 similarity, out float rms, out int anchorsUsed, out float anchorSpread,
        out int inlierCount, out int[] inlierSlots)
    {
        similarity = Similarity3.Identity;
        rms = float.MaxValue;
        anchorsUsed = 0;
        anchorSpread = 0f;
        inlierCount = 0;
        inlierSlots = Array.Empty<int>();

        if (!TryCollectAnchors(chunk, chunkCameras, reference,
                out var slots, out var source, out var target, out var perAnchorRotation, out float spread))
        {
            anchorsUsed = source.Count;
            anchorSpread = spread;
            return false;
        }
        anchorsUsed = source.Count;
        anchorSpread = spread;

        var candidates = ScoreAnchorPairs(source, target, perAnchorRotation, spread);
        AnchorPairCandidate? best = null;
        foreach (var c in candidates)
            if (best == null || c.InlierCount > best.Value.InlierCount
                || (c.InlierCount == best.Value.InlierCount && c.Error < best.Value.Error))
                best = c;

        // A pair always fits itself, so two agreeing anchors out of six is not support, it is a
        // coincidence with company. The fold needs a strict MAJORITY of the recovered anchors:
        // 2 of 2, 2 of 3, 3 of 4, 4 of 6. That is what lets DrJohnson fold on 3+12 with 37 out,
        // and refuses a pass where nothing agrees with anything.
        if (best == null || best.Value.InlierCount < MinFoldAnchors) return false;
        var bestInliers = best.Value.Inliers;
        if (bestInliers.Length * 2 <= anchorsUsed) return false;

        // Refit on the agreeing set. Rotation is the mean of the inliers' own estimates - each
        // one is a direct measurement, not something recovered from a point triangle that a
        // single bad point can tilt. Scale and translation are least squares over their positions.
        var meanRotation = MeanRotation(bestInliers.Select(i => perAnchorRotation[i]));
        var fitRotation = Matrix4x4.CreateFromQuaternion(meanRotation);

        var hereCentroid = Vector3.Zero;
        var thereCentroid = Vector3.Zero;
        foreach (int i in bestInliers) { hereCentroid += source[i]; thereCentroid += target[i]; }
        hereCentroid /= bestInliers.Length;
        thereCentroid /= bestInliers.Length;

        float numer = 0f, denom = 0f;
        foreach (int i in bestInliers)
        {
            var h = Vector3.Transform(source[i] - hereCentroid, fitRotation);
            numer += Vector3.Dot(h, target[i] - thereCentroid);
            denom += h.LengthSquared();
        }
        if (!(denom > 1e-12f) || !(numer > 0f)) return false;
        float fitScale = numer / denom;
        var fitTranslation = thereCentroid - fitScale * Vector3.Transform(hereCentroid, fitRotation);

        float sq = 0f;
        foreach (int i in bestInliers)
        {
            float d = Vector3.Distance(
                WorldSpaceGeometry.ApplySimilarity(source[i], fitScale, fitRotation, fitTranslation), target[i]);
            sq += d * d;
        }
        rms = MathF.Sqrt(sq / bestInliers.Length);
        if (rms > MaxAnchorRmsFraction * spread) return false;

        inlierCount = bestInliers.Length;
        inlierSlots = bestInliers.Select(i => slots[i]).ToArray();
        similarity = new Similarity3(fitScale, fitRotation, fitTranslation);
        return true;
    }

    /// <summary>
    /// One anchor pair's proposal for the fold and how the recovered anchors judged it.
    /// Indices are into the RECOVERED anchor list (see <see cref="ScoreAnchorPairs"/>), which the
    /// public overload maps back to chunk slots. <see cref="PositionError"/> and
    /// <see cref="RotationError"/> are the inliers' summed misfits, each in units of its own
    /// tolerance, so a value of 1.0 means "one inlier sat exactly on the limit".
    /// </summary>
    public readonly record struct AnchorPairCandidate(
        int A, int B, int[] Inliers, float PositionError, float RotationError, float PairRotationRadians)
    {
        public int InlierCount => Inliers.Length;
        public float Error => PositionError + RotationError;
    }

    /// <summary>
    /// Every pair proposal for a chunk, with slots mapped to chunk view indices - what the fold
    /// chooses between, for the log. Empty when fewer than <see cref="MinFoldAnchors"/> anchors
    /// were recovered.
    /// </summary>
    public static IReadOnlyList<(int ViewA, int ViewB, int[] InlierViews, float PositionError, float RotationError, float PairRotationRadians)>
        ScoreAnchorPairs(
            MultiViewChunk chunk,
            IReadOnlyList<CameraParams?> chunkCameras,
            IReadOnlyDictionary<int, CameraParams> reference)
    {
        if (!TryCollectAnchors(chunk, chunkCameras, reference,
                out var slots, out var source, out var target, out var rotations, out float spread))
            return Array.Empty<(int, int, int[], float, float, float)>();
        return ScoreAnchorPairs(source, target, rotations, spread)
            .Select(c => (
                chunk.Views[slots[c.A]], chunk.Views[slots[c.B]],
                c.Inliers.Select(i => chunk.Views[slots[i]]).ToArray(),
                c.PositionError, c.RotationError, c.PairRotationRadians))
            .ToList();
    }

    /// <summary>
    /// The anchors a fold can use: those recovered in this pass, present in the reference, and
    /// with a non-degenerate orientation. <paramref name="spread"/> is the mean distance of the
    /// reference anchors from their centroid - what every residual is judged against.
    /// False when fewer than <see cref="MinFoldAnchors"/> qualify or they all sit at one point.
    /// </summary>
    static bool TryCollectAnchors(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out List<int> slots, out List<Vector3> source, out List<Vector3> target,
        out List<Quaternion> perAnchorRotation, out float spread)
    {
        slots = new List<int>();
        source = new List<Vector3>();
        target = new List<Vector3>();
        perAnchorRotation = new List<Quaternion>();
        spread = 0f;
        for (int slot = 0; slot < chunk.AnchorCount && slot < chunkCameras.Count; slot++)
        {
            var here = chunkCameras[slot];
            if (here == null) continue;
            if (!reference.TryGetValue(chunk.Views[slot], out var there)) continue;
            if (!TryRotationBetweenCameras(here, there, out var q)) continue;
            slots.Add(slot);
            source.Add(here.Position);
            target.Add(there.Position);
            perAnchorRotation.Add(q);
        }
        if (source.Count < MinFoldAnchors) return false;

        // Residual is only meaningful against how far apart the anchors are.
        var centroid = Vector3.Zero;
        foreach (var t in target) centroid += t;
        centroid /= target.Count;
        foreach (var t in target) spread += Vector3.Distance(t, centroid);
        spread /= target.Count;
        return spread > 1e-6f;
    }

    /// <summary>
    /// Every anchor pair whose two rotation estimates agree proposes a similarity (rotation =
    /// their mean, scale = their distance ratio, translation from their midpoint), and every
    /// recovered anchor votes on it in BOTH position (<see cref="InlierAnchorFraction"/> of the
    /// spread) and orientation (<see cref="MaxAnchorRotationRadians"/>).
    ///
    /// Score = inlier count, then the inliers' combined misfit in units of tolerance. A pair's
    /// OWN members fix scale and translation exactly, so the only way they can misfit is in
    /// direction: the here-baseline carried by the pair's rotation versus the reference
    /// baseline. That is an angle, and it is scored as one (against
    /// <see cref="MaxAnchorRotationRadians"/>), not as a distance. Every other anchor misfits
    /// in position for real and is scored as a distance against the spread tolerance.
    ///
    /// MEASURED on DrJohnson chunk 10 (dj2k-dav3-pose, 2026-09-23), scoring member misfit as a
    /// distance: 3-12 pos 1.47 + rot 0.33 = 1.80, 3-37 pos 0.63 + rot 0.74 = 1.36, so the fold
    /// kept 37 and threw out 12 - the anchor that agreed with 3 to within 3.4 deg in all 13
    /// passes. Back-computed, the direction misfit was 5.3 deg on BOTH pairs; the 1.47 versus
    /// 0.63 was nothing but 3-12's baseline being 2.3x longer (1.04 against 0.45). Position
    /// distance for a pair's own members is baseline bias, not evidence. As angles: 3-12 1.07 +
    /// 0.33 = 1.40, 3-37 1.06 + 0.74 = 1.80 - the orientation evidence decides, and every other
    /// chunk in that run keeps the decision it already had.
    /// </summary>
    static List<AnchorPairCandidate> ScoreAnchorPairs(
        IReadOnlyList<Vector3> source, IReadOnlyList<Vector3> target,
        IReadOnlyList<Quaternion> perAnchorRotation, float spread)
    {
        int n = source.Count;
        float tolerance = InlierAnchorFraction * spread;
        var candidates = new List<AnchorPairCandidate>();
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
            {
                float pairAngle = AngleBetween(perAnchorRotation[a], perAnchorRotation[b]);
                // Two anchors that disagree about the frame rotation cannot both be right, so
                // they do not get to propose together. Reported with no inliers so the log
                // still shows the pair and why it sat out.
                if (pairAngle > MaxAnchorRotationRadians)
                {
                    candidates.Add(new AnchorPairCandidate(a, b, Array.Empty<int>(), 0f, 0f, pairAngle));
                    continue;
                }

                float hereDist = Vector3.Distance(source[a], source[b]);
                if (!(hereDist > 1e-6f)) continue;
                float scale = Vector3.Distance(target[a], target[b]) / hereDist;
                var rotation = Matrix4x4.CreateFromQuaternion(
                    Quaternion.Normalize(Quaternion.Slerp(perAnchorRotation[a], perAnchorRotation[b], 0.5f)));
                var midHere = (source[a] + source[b]) * 0.5f;
                var midThere = (target[a] + target[b]) * 0.5f;
                var translation = midThere - scale * Vector3.Transform(midHere, rotation);
                var candidate = Quaternion.CreateFromRotationMatrix(rotation);

                // The pair's own direction misfit: the here-baseline under the pair's rotation
                // against the reference baseline. Symmetric, so one angle serves both members.
                var hereDir = Vector3.Normalize(Vector3.Transform(source[b] - source[a], rotation));
                var thereDir = Vector3.Normalize(target[b] - target[a]);
                float directionMisfit = MathF.Acos(Math.Clamp(Vector3.Dot(hereDir, thereDir), -1f, 1f));

                var inliers = new List<int>();
                float positionError = 0f, rotationError = 0f;
                for (int i = 0; i < n; i++)
                {
                    float d = Vector3.Distance(
                        WorldSpaceGeometry.ApplySimilarity(source[i], scale, rotation, translation), target[i]);
                    if (d > tolerance) continue;
                    float angle = AngleBetween(perAnchorRotation[i], candidate);
                    if (angle > MaxAnchorRotationRadians) continue;
                    inliers.Add(i);
                    positionError += i == a || i == b
                        ? directionMisfit / MaxAnchorRotationRadians
                        : d / tolerance;
                    rotationError += angle / MaxAnchorRotationRadians;
                }
                candidates.Add(new AnchorPairCandidate(
                    a, b, inliers.ToArray(), positionError, rotationError, pairAngle));
            }
        return candidates;
    }

    /// <summary>
    /// The rotation carrying <paramref name="here"/>'s frame onto <paramref name="there"/>'s,
    /// from the two cameras' orientations alone. Both describe the SAME physical camera, so the
    /// rotation that maps one camera basis onto the other is the rotation between the frames.
    /// False when either camera's forward and up are degenerate.
    /// </summary>
    public static bool TryRotationBetweenCameras(CameraParams here, CameraParams there, out Quaternion rotation)
    {
        rotation = Quaternion.Identity;
        if (!TryCameraBasis(here, out var hr, out var hu, out var hf)) return false;
        if (!TryCameraBasis(there, out var tr, out var tu, out var tf)) return false;

        // Row-vector convention (System.Numerics: v' = v * M). With B_here rows = (right, up,
        // forward) and B_there likewise, B_here * M = B_there, so M = B_here^T * B_there.
        var m = new Matrix4x4(
            hr.X * tr.X + hu.X * tu.X + hf.X * tf.X, hr.X * tr.Y + hu.X * tu.Y + hf.X * tf.Y, hr.X * tr.Z + hu.X * tu.Z + hf.X * tf.Z, 0f,
            hr.Y * tr.X + hu.Y * tu.X + hf.Y * tf.X, hr.Y * tr.Y + hu.Y * tu.Y + hf.Y * tf.Y, hr.Y * tr.Z + hu.Y * tu.Z + hf.Y * tf.Z, 0f,
            hr.Z * tr.X + hu.Z * tu.X + hf.Z * tf.X, hr.Z * tr.Y + hu.Z * tu.Y + hf.Z * tf.Y, hr.Z * tr.Z + hu.Z * tu.Z + hf.Z * tf.Z, 0f,
            0f, 0f, 0f, 1f);
        rotation = Quaternion.Normalize(Quaternion.CreateFromRotationMatrix(m));
        return !float.IsNaN(rotation.X);
    }

    /// <summary>Angle in radians between two rotations.</summary>
    public static float AngleBetween(Quaternion a, Quaternion b)
    {
        float dot = MathF.Abs(Quaternion.Dot(Quaternion.Normalize(a), Quaternion.Normalize(b)));
        return 2f * MathF.Acos(MathF.Min(1f, dot));
    }

    private static bool TryCameraBasis(CameraParams cam, out Vector3 right, out Vector3 up, out Vector3 forward)
    {
        forward = cam.Forward;
        right = Vector3.Cross(forward, cam.Up);
        up = Vector3.Zero;
        if (!(forward.LengthSquared() > 1e-12f) || !(right.LengthSquared() > 1e-12f)) return false;
        forward = Vector3.Normalize(forward);
        right = Vector3.Normalize(right);
        up = Vector3.Cross(right, forward);
        return true;
    }

    /// <summary>
    /// Mean of near-agreeing rotations: sign-aligned quaternion sum, normalised. Exact for two,
    /// and within the tolerances used here for any number that already agree.
    /// </summary>
    private static Quaternion MeanRotation(IEnumerable<Quaternion> rotations)
    {
        Quaternion? first = null;
        var sum = new Quaternion(0, 0, 0, 0);
        foreach (var q in rotations)
        {
            var n = Quaternion.Normalize(q);
            first ??= n;
            if (Quaternion.Dot(first.Value, n) < 0f) n = Quaternion.Negate(n);
            sum = Quaternion.Add(sum, n);
        }
        return Quaternion.Normalize(sum);
    }
}