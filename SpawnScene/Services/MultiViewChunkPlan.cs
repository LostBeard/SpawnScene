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
    /// So: fit on every three-anchor subset, score each candidate by how many of ALL the anchors
    /// it agrees with, and refit on the winner's inliers. The search is exhaustive rather than
    /// randomised - C(8,3) is 56 - so it is deterministic, which a diagnostic has to be.
    ///
    /// At exactly <see cref="MinAnchors"/> anchors this degrades to the plain fit, and that is
    /// not a limitation of the method: three points have no majority to disagree with. Detecting
    /// a bad anchor needs four, and identifying WHICH one needs five.
    /// </summary>
    public static bool TryFitChunkToReference(
        MultiViewChunk chunk,
        IReadOnlyList<CameraParams?> chunkCameras,
        IReadOnlyDictionary<int, CameraParams> reference,
        out Similarity3 similarity, out float rms, out int anchorsUsed, out float anchorSpread,
        out int inlierCount)
    {
        similarity = Similarity3.Identity;
        rms = float.MaxValue;
        anchorsUsed = 0;
        anchorSpread = 0f;
        inlierCount = 0;

        var source = new List<Vector3>();
        var target = new List<Vector3>();
        for (int slot = 0; slot < chunk.AnchorCount && slot < chunkCameras.Count; slot++)
        {
            var here = chunkCameras[slot];
            if (here == null) continue;
            if (!reference.TryGetValue(chunk.Views[slot], out var there)) continue;
            source.Add(here.Position);
            target.Add(there.Position);
        }

        anchorsUsed = source.Count;
        if (anchorsUsed < MinAnchors) return false;

        // Residual is only meaningful against how far apart the anchors are.
        var centroid = Vector3.Zero;
        foreach (var t in target) centroid += t;
        centroid /= target.Count;
        float spread = 0f;
        foreach (var t in target) spread += Vector3.Distance(t, centroid);
        spread /= target.Count;
        anchorSpread = spread;
        if (!(spread > 1e-6f)) return false;

        float tolerance = InlierAnchorFraction * spread;
        List<int>? bestInliers = null;

        if (anchorsUsed == MinAnchors)
        {
            bestInliers = Enumerable.Range(0, anchorsUsed).ToList();
        }
        else
        {
            float bestError = float.MaxValue;
            for (int a = 0; a < anchorsUsed; a++)
                for (int b = a + 1; b < anchorsUsed; b++)
                    for (int c = b + 1; c < anchorsUsed; c++)
                    {
                        var subS = new[] { source[a], source[b], source[c] };
                        var subT = new[] { target[a], target[b], target[c] };
                        if (!WorldSpaceGeometry.TryUmeyamaSimilarity(
                                subS, subT, out float cs, out var cr, out var ct, out _))
                            continue;

                        var inliers = new List<int>();
                        float error = 0f;
                        for (int i = 0; i < anchorsUsed; i++)
                        {
                            float d = Vector3.Distance(
                                WorldSpaceGeometry.ApplySimilarity(source[i], cs, cr, ct), target[i]);
                            if (d <= tolerance) { inliers.Add(i); error += d; }
                        }

                        if (inliers.Count > (bestInliers?.Count ?? 0)
                            || (inliers.Count == (bestInliers?.Count ?? 0) && error < bestError))
                        {
                            bestInliers = inliers;
                            bestError = error;
                        }
                    }
        }

        // A MINIMAL subset always fits itself, so three inliers from a three-point sample is not
        // evidence of anything - it is the sample agreeing with itself. Real support means at
        // least one anchor from OUTSIDE the sample agrees too. With exactly MinAnchors available
        // there is no outside, so that case keeps the plain fit and leans on the residual
        // threshold instead, which is meaningful because three points over-determine a
        // similarity by two.
        int required = anchorsUsed <= MinAnchors ? MinAnchors : MinAnchors + 1;
        if (bestInliers == null || bestInliers.Count < required) return false;
        inlierCount = bestInliers.Count;

        // Refit on the agreeing set: the subset that won is only three points, and the other
        // inliers carry information the final transform should use.
        var fitS = bestInliers.Select(i => source[i]).ToList();
        var fitT = bestInliers.Select(i => target[i]).ToList();
        if (!WorldSpaceGeometry.TryUmeyamaSimilarity(
                fitS, fitT, out var scale, out var rotation, out var translation, out rms))
            return false;

        if (rms > MaxAnchorRmsFraction * spread) return false;

        similarity = new Similarity3(scale, rotation, translation);
        return true;
    }
}
