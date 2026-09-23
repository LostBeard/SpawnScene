using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Gate for posing every view in ONE frame with a depth model that only sees a few at a time.
///
/// Why this exists: commit eeedfff measured joint DAv3 poses as ~5 dB better initialisation than
/// SfM on Bathroom, but DAv3 only poses the six views it is handed, so supervision fell from 17
/// views to 5. The way out is to run it repeatedly with shared anchor views and fold each pass
/// into the first pass's frame. The whole risk of that is the fold: a similarity fitted to the
/// wrong thing places a chunk's splats confidently in the wrong part of the room, and nothing
/// downstream can tell that from measured geometry.
///
/// <see cref="ChunkedPasses_InArbitraryFrames_ReassembleToTheGroundTruthRig"/> is the real gate -
/// it hands each chunk its cameras in a DIFFERENT random frame, exactly as independent forward
/// passes would, and demands the ground-truth rig back.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter MultiViewChunkPlan</c>
/// </summary>
public class MultiViewChunkPlanTests
{
    private const int ChunkSize = 6;   // DepthEstimationService.MaxMultiViewImages
    private const int Anchors = 3;

    // ---- Planning ----

    [Test]
    public void Plan_PosesEveryViewExactlyOnce()
    {
        var plan = MultiViewChunkPlan.Plan(viewCount: 35, ChunkSize, Anchors);

        var seen = new List<int>();
        foreach (var chunk in plan) seen.AddRange(chunk.NewViews.ToArray());
        var anchorSet = plan[0].Anchors.ToArray().ToHashSet();

        Assert.That(seen, Is.Unique, "a view posed by two chunks would be merged in twice");
        Assert.That(seen.Concat(anchorSet).OrderBy(i => i), Is.EqualTo(Enumerable.Range(0, 35)),
            "every one of the 35 frames must be posed - dropping views is the bug being fixed");
    }

    [Test]
    public void Plan_EveryChunkCarriesTheSameAnchorsAndFitsTheModelCap()
    {
        var plan = MultiViewChunkPlan.Plan(viewCount: 35, ChunkSize, Anchors);
        var anchors = plan[0].Anchors.ToArray();

        Assert.That(anchors, Has.Length.EqualTo(Anchors));
        foreach (var chunk in plan)
        {
            Assert.That(chunk.Views, Has.Length.LessThanOrEqualTo(ChunkSize),
                "a chunk larger than the cap is a forward pass the model will refuse");
            Assert.That(chunk.Anchors.ToArray(), Is.EqualTo(anchors),
                "anchors are shared globally so each chunk fits the reference DIRECTLY, " +
                "with no error chained through its neighbours");
            Assert.That(chunk.Views, Is.Unique);
        }
    }

    [Test]
    public void Plan_SpreadsAnchorsAcrossTheCapture()
    {
        var anchors = MultiViewChunkPlan.Plan(35, ChunkSize, Anchors)[0].Anchors.ToArray();
        // Adjacent handheld frames are near-parallel: the worst case for both the depth model
        // and the similarity fit. Endpoints included, evenly spaced.
        Assert.That(anchors, Is.EqualTo(new[] { 0, 17, 34 }));
    }

    [Test]
    public void Plan_SpreadsEachChunksNewViewsToo()
    {
        var plan = MultiViewChunkPlan.Plan(35, ChunkSize, Anchors);
        // 32 non-anchor views, 3 per chunk => 11 chunks, dealt round-robin. Chunk 0 must not be
        // three consecutive frames.
        Assert.That(plan, Has.Count.EqualTo(11));
        var first = plan[0].NewViews.ToArray();
        Assert.That(first.Max() - first.Min(), Is.GreaterThan(3),
            "a chunk of consecutive frames is the near-parallel case this deals around");
    }

    [Test]
    public void Plan_UsesOnePassWhenEverythingFits()
    {
        var plan = MultiViewChunkPlan.Plan(viewCount: 4, ChunkSize, Anchors);
        Assert.That(plan, Has.Count.EqualTo(1));
        Assert.That(plan[0].Views, Is.EqualTo(new[] { 0, 1, 2, 3 }));
        Assert.That(plan[0].NewViews.Length, Is.Zero, "one pass IS the reference frame");
    }

    [Test]
    public void Plan_RejectsAnchorCountsThatCannotDetermineASimilarity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MultiViewChunkPlan.Plan(35, ChunkSize, 2),
            "two anchors leave a free roll about the line joining them");
        Assert.Throws<ArgumentOutOfRangeException>(() => MultiViewChunkPlan.Plan(35, ChunkSize, 6),
            "anchors filling the chunk leave no view to pose");
        Assert.Throws<ArgumentOutOfRangeException>(() => MultiViewChunkPlan.Plan(35, 3, 3));
    }

    // ---- Anchor choice ----

    /// <summary>
    /// A walk-through capture: frame i sees frame j only when they are close in time. Spread
    /// anchors are then the WORST choice, because the ends of the sequence never saw the same
    /// wall - which is Bathroom, where anchor 17 swung between ratios of 0.43 and 3.46 while the
    /// pair the model could actually relate held at 1.00.
    /// </summary>
    [Test]
    public void PickAnchorsByOverlap_PrefersViewsThatSawEachOther()
    {
        const int Views = 35;
        int Overlap(int a, int b)
        {
            int gap = Math.Abs(a - b);
            return gap <= 6 ? 400 - gap * 60 : 0;      // nothing in common beyond six frames
        }

        var spread = MultiViewChunkPlan.SpreadPick(Views, 3);
        Assert.That(Overlap(spread[0], spread[2]), Is.Zero,
            "the spread pick's own endpoints share nothing - the case being fixed");

        var picked = MultiViewChunkPlan.PickAnchorsByOverlap(Views, Overlap, 3);
        Assert.That(picked, Has.Length.EqualTo(3));
        Assert.That(picked, Is.Unique);
        for (int i = 0; i < picked.Length; i++)
            for (int j = i + 1; j < picked.Length; j++)
                Assert.That(Overlap(picked[i], picked[j]), Is.GreaterThan(0),
                    $"anchors {picked[i]} and {picked[j]} must have seen each other");
    }

    /// <summary>
    /// Maximising the MINIMUM overlap, not the total. A view can be the most connected in the
    /// whole capture and still be useless as an anchor if it is blind to the others it would
    /// serve alongside; popularity alone would take it.
    ///
    /// Note the first draft of this test asserted the wrong answer. It named view 3 a "bridge to
    /// nowhere" for being blind to view 2, when {0,1,3} was in fact mutually connected AND had
    /// the better bottleneck - blindness to a view you do not pick costs nothing. The property
    /// worth asserting is the invariant, not a triple I picked by eye.
    /// </summary>
    [Test]
    public void PickAnchorsByOverlap_RequiresOverlapWithEveryAnchorNotJustMost()
    {
        // View 3 is hugely tied to view 0 and blind to everything else, so no valid anchor set
        // of three can contain it. It still has the highest TOTAL overlap by a wide margin.
        int Overlap(int a, int b)
        {
            if (a > b) (a, b) = (b, a);
            return (a, b) switch
            {
                (0, 3) => 9000,
                (1, 3) => 0, (2, 3) => 0, (3, 4) => 0,
                _ => 300 - Math.Abs(a - b) * 10,
            };
        }

        var picked = MultiViewChunkPlan.PickAnchorsByOverlap(5, Overlap, 3);

        Assert.That(picked, Does.Not.Contain(3),
            "view 3 has by far the most total overlap and must still lose: it cannot be tied to " +
            "two other anchors at once");
        for (int i = 0; i < picked.Length; i++)
            for (int j = i + 1; j < picked.Length; j++)
                Assert.That(Overlap(picked[i], picked[j]), Is.GreaterThan(0),
                    $"anchors {picked[i]} and {picked[j]} must have seen each other");
    }

    [Test]
    public void PickAnchorsByOverlap_IsDeterministicAndSorted()
    {
        int Overlap(int a, int b) => 100 - Math.Abs(a - b);
        var a = MultiViewChunkPlan.PickAnchorsByOverlap(20, Overlap, 4);
        var b = MultiViewChunkPlan.PickAnchorsByOverlap(20, Overlap, 4);
        Assert.That(a, Is.EqualTo(b), "a diagnostic that moves between runs cannot be read");
        Assert.That(a, Is.Ordered);
    }

    // ---- The fold ----

    [Test]
    public void Similarity_AppliesScaleAndTranslationToPointsButNotToDirections()
    {
        var sim = new Similarity3(3f, Matrix4x4.CreateRotationY(MathF.PI / 2f), new Vector3(10, 0, 0));

        var p = sim.Apply(new Vector3(1, 0, 0));
        Assert.That(p.Length(), Is.GreaterThan(1f), "a point takes the scale and the offset");

        var d = sim.ApplyDirection(new Vector3(1, 0, 0));
        Assert.That(d.Length(), Is.EqualTo(1f).Within(1e-5f),
            "a camera's forward is a direction: scaling or translating it aims the camera at nothing");
    }

    [Test]
    public void FitChunkToReference_RecoversAKnownTransform()
    {
        var truth = new Similarity3(
            2.5f, Matrix4x4.CreateFromYawPitchRoll(0.7f, -0.35f, 1.1f), new Vector3(-4f, 2f, 9f));

        var chunk = new MultiViewChunk(new[] { 0, 1, 2, 7 }, AnchorCount: 3);
        var reference = new Dictionary<int, CameraParams>
        {
            [0] = CamAt(new Vector3(0, 0, 0)),
            [1] = CamAt(new Vector3(1, 0.4f, 0)),
            [2] = CamAt(new Vector3(0, 1, 0.6f)),
        };
        // The chunk sees the same cameras through the INVERSE of truth, so fitting must return truth.
        // Whole poses, not points: a pass reports each camera's orientation in its frame too.
        var chunkCams = new CameraParams?[]
        {
            Seen(truth, reference[0]),
            Seen(truth, reference[1]),
            Seen(truth, reference[2]),
            CamAt(new Vector3(5, 5, 5)),
        };

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
            chunk, chunkCams, reference, out var sim, out var rms, out var used), Is.True);
        Assert.That(used, Is.EqualTo(3));
        Assert.That(rms, Is.LessThan(1e-3f));
        Assert.That(sim.Scale, Is.EqualTo(truth.Scale).Within(1e-3f));

        var probe = new Vector3(0.3f, -0.8f, 2f);
        var viaTruth = truth.Apply(probe);
        var viaFit = sim.Apply(probe);
        Assert.That(Vector3.Distance(viaTruth, viaFit), Is.LessThan(1e-3f));
    }

    /// <summary>
    /// The gate. Eleven independent forward passes, each reporting the shared anchors in its own
    /// arbitrary frame, must reassemble into the one rig the photographs were actually taken from.
    /// </summary>
    [Test]
    public void ChunkedPasses_InArbitraryFrames_ReassembleToTheGroundTruthRig()
    {
        const int Views = 35;
        var truthRig = BuildHandheldRig(Views);
        var plan = MultiViewChunkPlan.Plan(Views, ChunkSize, Anchors);

        var rng = new Random(20260921);
        var placed = new Dictionary<int, CameraParams>();
        var depthScale = new Dictionary<int, float>();

        for (int c = 0; c < plan.Count; c++)
        {
            var chunk = plan[c];
            // Chunk 0 IS the reference frame; every later pass gets a random frame of its own.
            var frame = c == 0 ? Similarity3.Identity : RandomSimilarity(rng);

            var asSeen = new CameraParams?[chunk.Views.Length];
            for (int slot = 0; slot < chunk.Views.Length; slot++)
            {
                var cam = Clone(truthRig[chunk.Views[slot]]);
                InverseSimilarity(frame).ApplyToCamera(cam);   // truth -> this pass's frame
                asSeen[slot] = cam;
            }

            if (c == 0)
            {
                for (int slot = 0; slot < chunk.Views.Length; slot++)
                {
                    placed[chunk.Views[slot]] = asSeen[slot]!;
                    depthScale[chunk.Views[slot]] = 1f;
                }
                continue;
            }

            Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                    chunk, asSeen, placed, out var sim, out var rms, out _), Is.True,
                $"chunk {c} must fold into the reference frame");
            Assert.That(rms, Is.LessThan(1e-3f), $"chunk {c} residual");

            foreach (var slot in Enumerable.Range(chunk.AnchorCount, chunk.NewViews.Length))
            {
                var cam = asSeen[slot]!;
                sim.ApplyToCamera(cam);
                placed[chunk.Views[slot]] = cam;
                // Depths came out of the pass in the pass's frame, so they take the same scale.
                depthScale[chunk.Views[slot]] = sim.Scale;
            }
        }

        Assert.That(placed, Has.Count.EqualTo(Views), "every frame posed, in one world");

        for (int i = 0; i < Views; i++)
        {
            var got = placed[i];
            var want = truthRig[i];
            Assert.That(Vector3.Distance(got.Position, want.Position), Is.LessThan(2e-3f),
                $"view {i} position");
            Assert.That(Vector3.Dot(got.Forward, want.Forward), Is.EqualTo(1f).Within(1e-3f),
                $"view {i} forward - a camera in the right place looking the wrong way renders nothing");
            Assert.That(Vector3.Dot(got.Up, want.Up), Is.EqualTo(1f).Within(1e-3f),
                $"view {i} up - a rolled camera unprojects its depth map into a twisted shell");
        }
    }

    /// <summary>
    /// Mixed per-view image rotations - some clockwise, some counter - must not disturb the fold.
    ///
    /// A real capture has them: <c>QuarterTurnsToUpright</c> is decided per frame, and Bathroom
    /// asked for four different answers across six cameras of one room. The reason it cannot
    /// matter here is worth stating, because "it should be fine" is how this kind of thing gets
    /// shipped: turning a photograph adds ROLL to the camera that reports it, and roll does not
    /// move the camera's centre. The fit reads centres and nothing else.
    ///
    /// What rotation DOES change is the image SHAPE, and that is handled where it bites - see
    /// <c>MultiViewChunkPlan.PlanByShape</c>.
    /// </summary>
    [Test]
    public void MixedPerViewImageRotations_DoNotDisturbTheFold()
    {
        const int Views = 35;
        var truthRig = BuildHandheldRig(Views);

        // Roll each camera about its own forward axis by a different amount and direction,
        // which is exactly what turning its photograph does.
        var rolls = new float[Views];
        for (int i = 0; i < Views; i++)
        {
            int turns = (i * 7) % 4;                       // 0, 1, 2, 3 - both directions
            rolls[i] = turns * MathF.PI / 2f * (i % 2 == 0 ? 1f : -1f);
            var q = Quaternion.CreateFromAxisAngle(truthRig[i].Forward, rolls[i]);
            truthRig[i].Up = Vector3.Normalize(Vector3.Transform(truthRig[i].Up, q));
        }

        var plan = MultiViewChunkPlan.Plan(Views, ChunkSize, Anchors);
        var rng = new Random(7);
        var placed = new Dictionary<int, CameraParams>();

        for (int c = 0; c < plan.Count; c++)
        {
            var chunk = plan[c];
            var frame = c == 0 ? Similarity3.Identity : RandomSimilarity(rng);
            var asSeen = new CameraParams?[chunk.Views.Length];
            for (int slot = 0; slot < chunk.Views.Length; slot++)
            {
                var cam = Clone(truthRig[chunk.Views[slot]]);
                InverseSimilarity(frame).ApplyToCamera(cam);
                asSeen[slot] = cam;
            }

            if (c == 0)
            {
                for (int slot = 0; slot < chunk.Views.Length; slot++)
                    placed[chunk.Views[slot]] = asSeen[slot]!;
                continue;
            }

            Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                    chunk, asSeen, placed, out var sim, out var rms, out _), Is.True,
                $"chunk {c} with mixed image rotations");
            Assert.That(rms, Is.LessThan(1e-3f));
            foreach (var slot in Enumerable.Range(chunk.AnchorCount, chunk.NewViews.Length))
            {
                var cam = asSeen[slot]!;
                sim.ApplyToCamera(cam);
                placed[chunk.Views[slot]] = cam;
            }
        }

        for (int i = 0; i < Views; i++)
        {
            Assert.That(Vector3.Distance(placed[i].Position, truthRig[i].Position), Is.LessThan(2e-3f),
                $"view {i} position");
            Assert.That(Vector3.Dot(placed[i].Up, truthRig[i].Up), Is.EqualTo(1f).Within(1e-3f),
                $"view {i} roll must be carried through, not straightened");
        }
    }

    /// <summary>
    /// Where image rotation DOES matter: a quarter turn swaps width and height, and the joint
    /// depth pass emits every view at its FIRST view's resolution. A portrait frame sharing a
    /// pass with landscape ones comes back at the wrong shape and unprojects into a stretched
    /// shell, silently. Chunks are therefore planned per shape.
    /// </summary>
    [Test]
    public void PlanByShape_NeverPutsTwoShapesInOnePass()
    {
        // 34 portrait frames and one landscape - Bathroom's actual mix.
        var shapes = new (int W, int H)[35];
        for (int i = 0; i < 35; i++) shapes[i] = (768, 1024);
        shapes[11] = (1024, 768);

        var groups = MultiViewChunkPlan.PlanByShape(shapes, ChunkSize, Anchors);

        foreach (var group in groups)
            foreach (var chunk in group.Chunks)
            {
                var distinct = chunk.Views.Select(v => shapes[v]).Distinct().ToList();
                Assert.That(distinct, Has.Count.EqualTo(1),
                    "one pass, one shape: the model emits every view at the first one's resolution");
            }

        var posed = groups.SelectMany(g => g.Chunks).SelectMany(c => c.Views).Distinct().ToList();
        Assert.That(posed, Has.Count.EqualTo(35), "the odd frame is still posed, in its own group");
    }

    /// <summary>
    /// Red check: the gate above must be able to FAIL. Forget to rotate the directions - the
    /// single most tempting simplification, since positions already line up - and it catches it.
    /// </summary>
    [Test]
    public void RedCheck_PlacingPositionsWithoutRotatingDirectionsIsDetected()
    {
        var rig = BuildHandheldRig(8);
        var frame = new Similarity3(1.7f, Matrix4x4.CreateFromYawPitchRoll(0.9f, 0.2f, -0.6f),
            new Vector3(3f, -1f, 2f));

        var cam = Clone(rig[5]);
        var inverse = InverseSimilarity(frame);
        inverse.ApplyToCamera(cam);

        // Position only, directions left in the pass's frame.
        var halfDone = Clone(cam);
        halfDone.Position = frame.Apply(cam.Position);

        Assert.That(Vector3.Distance(halfDone.Position, rig[5].Position), Is.LessThan(1e-4f),
            "positions agree, which is exactly why this is easy to miss");
        Assert.That(Vector3.Dot(halfDone.Forward, rig[5].Forward), Is.LessThan(0.99f),
            "and the camera is aimed somewhere else entirely");
    }

    /// <summary>
    /// One anchor placed somewhere else entirely must not move the other five.
    ///
    /// This is not hypothetical. MEASURED on Bathroom: across ten passes the distance ratio
    /// between anchors 0 and 34 held at 1.00 (1.057, 1.018, 0.908, 1.031, 0.984) while every
    /// pair involving anchor 17 swung between 0.43 and 3.46 - the model put ONE camera somewhere
    /// different each time it saw it in different company. Least squares answers that by
    /// spreading the bad anchor's error evenly over the good ones.
    /// </summary>
    [Test]
    public void RobustFit_IgnoresTheOneAnchorThatDisagrees()
    {
        var truth = new Similarity3(
            1.8f, Matrix4x4.CreateFromYawPitchRoll(0.4f, 0.9f, -0.7f), new Vector3(2f, -3f, 5f));

        var reference = new Dictionary<int, CameraParams>();
        var world = new[]
        {
            new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0),
            new Vector3(0, 0, 1), new Vector3(1, 1, 0), new Vector3(0.4f, -0.7f, 0.9f),
        };
        for (int i = 0; i < world.Length; i++) reference[i] = CamAt(world[i], aimedAt: i);

        var chunk = new MultiViewChunk(new[] { 0, 1, 2, 3, 4, 5 }, AnchorCount: 6);
        var cams = world.Select((_, i) => (CameraParams?)Seen(truth, reference[i])).ToArray();

        // Anchor 3 comes back somewhere else entirely, as view 17 did - aimed right, placed wrong.
        cams[3]!.Position += new Vector3(4.2f, -3.1f, 2.7f);

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                chunk, cams, reference,
                out var sim, out float rms, out int used, out float spread, out int inliers),
            Is.True, "five agreeing anchors are a clear majority");

        Assert.That(used, Is.EqualTo(6), "all six were available");
        Assert.That(inliers, Is.EqualTo(5), "exactly one anchor disagrees and must be dropped");
        Assert.That(sim.Scale, Is.EqualTo(truth.Scale).Within(1e-3f));

        var probe = new Vector3(0.8f, 0.2f, -0.5f);
        Assert.That(Vector3.Distance(sim.Apply(probe), truth.Apply(probe)), Is.LessThan(1e-3f),
            "the recovered transform must be the one the five good anchors describe");
        Assert.That(rms, Is.LessThan(0.02f * spread));
    }

    /// <summary>
    /// Red check for the test above: the plain least-squares fit on the SAME data is wrong.
    /// Without this, the robust fit could be doing nothing and the test would still pass.
    /// </summary>
    [Test]
    public void RedCheck_LeastSquaresOnTheSameDataIsDraggedOff()
    {
        var truth = new Similarity3(
            1.8f, Matrix4x4.CreateFromYawPitchRoll(0.4f, 0.9f, -0.7f), new Vector3(2f, -3f, 5f));
        var world = new[]
        {
            new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0),
            new Vector3(0, 0, 1), new Vector3(1, 1, 0), new Vector3(0.4f, -0.7f, 0.9f),
        };
        var inverse = InverseSimilarity(truth);
        var src = world.Select(w => inverse.Apply(w)).ToList();
        src[3] = inverse.Apply(world[3]) + new Vector3(4.2f, -3.1f, 2.7f);

        Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(
            src, world.ToList(), out float s, out _, out _, out float rms), Is.True);

        Assert.That(MathF.Abs(s - truth.Scale), Is.GreaterThan(0.05f),
            "one bad anchor in six visibly corrupts the least-squares scale");
        Assert.That(rms, Is.GreaterThan(0.1f),
            "and leaves a residual the robust fit does not have");
    }

    [Test]
    public void RobustFit_RefusesWhenNoThreeAnchorsAgreeWithTheRest()
    {
        var reference = new Dictionary<int, CameraParams>
        {
            [0] = CamAt(new Vector3(0, 0, 0)),
            [1] = CamAt(new Vector3(1, 0, 0)),
            [2] = CamAt(new Vector3(0, 1, 0)),
            [3] = CamAt(new Vector3(0, 0, 1)),
        };
        var chunk = new MultiViewChunk(new[] { 0, 1, 2, 3 }, AnchorCount: 4);

        // Scattered with no consistent similarity anywhere in them.
        var cams = new CameraParams?[]
        {
            CamAt(new Vector3(0, 0, 0)), CamAt(new Vector3(9f, 0.2f, -4f)),
            CamAt(new Vector3(-6f, 3f, 8f)), CamAt(new Vector3(0.1f, -11f, 2f)),
        };

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                chunk, cams, reference, out _, out _, out _, out _, out int inliers),
            Is.False, "no majority agrees, so there is nothing to trust");
        Assert.That(inliers, Is.LessThan(MultiViewChunkPlan.MinAnchors + 1));
    }

    /// <summary>
    /// A camera is a pose. Two surviving anchors whose ORIENTATIONS agree on the frame rotation
    /// pin the roll that two points alone could not, so the chunk folds. One cannot.
    /// </summary>
    [Test]
    public void FitChunkToReference_TwoPoseAnchorsFold_OneDoesNot()
    {
        var truth = new Similarity3(
            1.3f, Matrix4x4.CreateFromYawPitchRoll(-0.5f, 0.3f, 0.8f), new Vector3(1f, 2f, -3f));
        var chunk = new MultiViewChunk(new[] { 0, 1, 2, 7 }, AnchorCount: 3);
        var reference = new Dictionary<int, CameraParams>
        {
            [0] = CamAt(Vector3.Zero, aimedAt: 0),
            [1] = CamAt(new Vector3(1, 0, 0), aimedAt: 1),
            [2] = CamAt(new Vector3(0, 1, 0), aimedAt: 2),
        };
        var cams = new CameraParams?[] { Seen(truth, reference[0]), null, Seen(truth, reference[2]), CamAt(Vector3.One) };

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
            chunk, cams, reference, out var sim, out _, out var used), Is.True,
            "two full poses over-determine a similarity by five");
        Assert.That(used, Is.EqualTo(2));
        var probe = new Vector3(0.3f, -0.8f, 2f);
        Assert.That(Vector3.Distance(sim.Apply(probe), truth.Apply(probe)), Is.LessThan(1e-3f));

        cams[2] = null;
        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
            chunk, cams, reference, out _, out _, out used), Is.False, "one pose fixes no scale");
        Assert.That(used, Is.EqualTo(1));
    }

    /// <summary>
    /// Red check for the two-anchor fold: two anchors whose positions fit trivially but whose
    /// orientations DISAGREE about the frame rotation are not evidence, and are refused.
    /// </summary>
    [Test]
    public void RedCheck_TwoAnchorsWithDisagreeingOrientationsAreRefused()
    {
        var truth = new Similarity3(
            1.3f, Matrix4x4.CreateFromYawPitchRoll(-0.5f, 0.3f, 0.8f), new Vector3(1f, 2f, -3f));
        var chunk = new MultiViewChunk(new[] { 0, 1, 7 }, AnchorCount: 2);
        var reference = new Dictionary<int, CameraParams>
        {
            [0] = CamAt(Vector3.Zero, aimedAt: 0),
            [1] = CamAt(new Vector3(1, 0, 0), aimedAt: 1),
        };
        var cams = new CameraParams?[] { Seen(truth, reference[0]), Seen(truth, reference[1]), CamAt(Vector3.One) };
        // Anchor 1 is where it should be but aimed 40 degrees off.
        var twist = Matrix4x4.CreateFromAxisAngle(Vector3.Normalize(new Vector3(0.2f, 1f, 0.1f)), 40f * MathF.PI / 180f);
        cams[1]!.Forward = Vector3.Normalize(Vector3.Transform(cams[1]!.Forward, twist));
        cams[1]!.Up = Vector3.Normalize(Vector3.Transform(cams[1]!.Up, twist));

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
            chunk, cams, reference, out _, out _, out _), Is.False);
    }

    /// <summary>
    /// DrJohnson, dj2k-dav3-pose: anchors 3, 12, 37. In 13 passes the 3-12 distance ratio held at
    /// 1.00 (0.88-1.12, per-pass scale) while every pair with 37 swung 0.52-1.43 - the model
    /// placed that one camera differently each time. Points alone cannot say which of three is
    /// the liar; poses can. The chunk folds on 3 and 12 and names 37 as excluded.
    /// </summary>
    [Test]
    public void ThreePoseAnchors_OneMisplaced_FoldsOnTheOtherTwoAndNamesIt()
    {
        var truth = new Similarity3(
            0.95f, Matrix4x4.CreateFromYawPitchRoll(0.6f, -0.2f, 0.1f), new Vector3(-2f, 0.5f, 1f));
        var chunk = new MultiViewChunk(new[] { 3, 12, 37, 20, 21, 22 }, AnchorCount: 3);
        var reference = new Dictionary<int, CameraParams>
        {
            [3] = CamAt(new Vector3(0, 0, 0), aimedAt: 3),
            [12] = CamAt(new Vector3(1.04f, 0.1f, 0.2f), aimedAt: 12),
            [37] = CamAt(new Vector3(0.3f, -0.1f, 0.45f), aimedAt: 37),
        };
        var cams = new CameraParams?[6];
        for (int s = 0; s < 3; s++) cams[s] = Seen(truth, reference[chunk.Views[s]]);
        for (int s = 3; s < 6; s++) cams[s] = CamAt(new Vector3(s, 0, 0));

        // 37 comes back 35% of the anchor spread away from where it belongs, aimed 25 degrees off:
        // the model related it to the others differently in this company.
        var twist = Matrix4x4.CreateFromAxisAngle(Vector3.UnitY, 25f * MathF.PI / 180f);
        cams[2]!.Position += new Vector3(0.15f, 0.05f, -0.12f);
        cams[2]!.Forward = Vector3.Normalize(Vector3.Transform(cams[2]!.Forward, twist));
        cams[2]!.Up = Vector3.Normalize(Vector3.Transform(cams[2]!.Up, twist));

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                chunk, cams, reference, out var sim, out float rms, out int used, out float spread,
                out int inliers, out int[] inlierSlots),
            Is.True, "two of three anchors agree in position AND orientation; that is a majority");
        Assert.That(used, Is.EqualTo(3));
        Assert.That(inliers, Is.EqualTo(2));
        Assert.That(inlierSlots, Is.EquivalentTo(new[] { 0, 1 }), "37 (slot 2) is the one thrown out");
        Assert.That(rms, Is.LessThan(0.01f * spread));

        var probe = new Vector3(0.8f, 0.2f, -0.5f);
        Assert.That(Vector3.Distance(sim.Apply(probe), truth.Apply(probe)), Is.LessThan(2e-3f),
            "the fold is the one 3 and 12 describe, untouched by 37");

        // Red check: the position-only fit on the same three points is dragged off by 37.
        Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(
            cams.Take(3).Select(c => c!.Position).ToList(),
            chunk.Views.Take(3).Select(v => reference[v].Position).ToList(),
            out _, out _, out _, out float pointRms), Is.True);
        Assert.That(pointRms, Is.GreaterThan(MultiViewChunkPlan.MaxAnchorRmsFraction * spread),
            "points alone would have REJECTED this chunk, as DrJohnson did 8 times");
    }

    /// <summary>
    /// DrJohnson chunk 10 (dj2k-dav3-pose, 2026-09-23), from the logged pair scores. Anchor 37:
    /// 7.4 deg off 3 in orientation, 21% too far along a 0.45 baseline. Anchor 12: 3.3 deg off 3
    /// on a 1.04 baseline. Every pair proposes, every pair has exactly its own two inliers. And
    /// the model's positions disagree with its orientations by the SAME 5.3 deg on both pairs
    /// (back-computed from pos 1.47 on 1.04 and pos 0.63 on 0.45) - DAv3 noise that is common
    /// to every anchor. Scored as a distance, that shared misfit costs the long-baseline pair
    /// 2.3x more than the short one and the fold kept 37 and threw out 12. Scored as the angle
    /// it is, the two pairs tie on it and the orientation evidence names 37. Red check: scoring
    /// member misfit as a distance fails this test (verified by disabling the angle path).
    /// </summary>
    [Test]
    public void ThreePoseAnchors_SharedDirectionMisfit_ShortBaselineDoesNotWin()
    {
        var truth = new Similarity3(
            1.05f, Matrix4x4.CreateFromYawPitchRoll(-0.4f, 0.25f, -0.15f), new Vector3(1.5f, -0.3f, 0.8f));
        var chunk = new MultiViewChunk(new[] { 3, 12, 37, 20, 21, 22 }, AnchorCount: 3);
        var reference = new Dictionary<int, CameraParams>
        {
            [3] = CamAt(new Vector3(0, 0, 0), aimedAt: 3),
            [12] = CamAt(new Vector3(1.04f, 0.1f, 0.2f), aimedAt: 12),
            [37] = CamAt(new Vector3(0.36f, -0.08f, 0.26f), aimedAt: 37),
        };
        var cams = new CameraParams?[6];
        for (int s = 0; s < 3; s++) cams[s] = Seen(truth, reference[chunk.Views[s]]);
        for (int s = 3; s < 6; s++) cams[s] = CamAt(new Vector3(s, 0, 0));

        // The shared position-versus-orientation misfit: rotate every anchor POSITION 5.3 deg
        // about the anchor centroid, on an axis across both baselines, leaving orientations
        // alone. Both pairs now carry the same direction misfit, as the log showed.
        var b312 = cams[1]!.Position - cams[0]!.Position;
        var b337 = cams[2]!.Position - cams[0]!.Position;
        var axis = Vector3.Normalize(Vector3.Cross(b312, b337));
        var centroid = (cams[0]!.Position + cams[1]!.Position + cams[2]!.Position) / 3f;
        var shared = Matrix4x4.CreateFromAxisAngle(axis, 5.3f * MathF.PI / 180f);
        for (int s = 0; s < 3; s++)
            cams[s]!.Position = centroid + Vector3.Transform(cams[s]!.Position - centroid, shared);

        // 12: honest, 3.3 deg of orientation noise (3-12 ran 1.0-3.4 deg over 13 passes). The
        // twist is ABOUT the baseline so it changes the orientation evidence and nothing else -
        // the direction misfit stays the shared 5.3 deg on both pairs, as the log showed.
        Twist(cams[1]!, b312, -2.8f);

        // 37: 21% too far from 3 along its own baseline and 7 deg twisted the other way - so
        // 3-37 is ~7 deg and 12-37 ~9.7 deg apart as in the log, both under the 10 deg gate,
        // and nothing excludes 37 up front. All three pairs propose.
        cams[2]!.Position += Vector3.Normalize(cams[2]!.Position - cams[0]!.Position) * (b337.Length() * 0.21f);
        Twist(cams[2]!, b337, 7.0f);

        var pairs = MultiViewChunkPlan.ScoreAnchorPairs(chunk, cams, reference);
        string dump = string.Join("  ", pairs.Select(p =>
            $"{p.ViewA}-{p.ViewB} in[{string.Join(",", p.InlierViews)}] pos {p.PositionError:F2} rot {p.RotationError:F2} " +
            $"pair {p.PairRotationRadians * 180 / MathF.PI:F1}deg"));
        TestContext.Out.WriteLine(dump);
        var p312 = pairs.Single(p => p.ViewA == 3 && p.ViewB == 12);
        var p337 = pairs.Single(p => p.ViewA == 3 && p.ViewB == 37);
        Assert.That(p312.InlierViews, Is.EquivalentTo(new[] { 3, 12 }), "3-12 has only itself");
        Assert.That(p337.InlierViews, Is.EquivalentTo(new[] { 3, 37 }), "3-37 has only itself");
        Assert.That(pairs.All(p => p.PairRotationRadians < MultiViewChunkPlan.MaxAnchorRotationRadians), Is.True,
            "every pair must pass the rotation gate, or this is not the three-way tie-break case: " + dump);
        Assert.That(p312.PositionError, Is.EqualTo(p337.PositionError).Within(0.15f),
            "the shared direction misfit must cost both pairs the same - it is the same angle");
        Assert.That(p312.RotationError, Is.LessThan(p337.RotationError),
            "3 and 12 agree in orientation better than 3 and 37 - the evidence that must decide");

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
                chunk, cams, reference, out var sim, out _, out int used, out _,
                out int inliers, out int[] inlierSlots),
            Is.True);
        Assert.That(used, Is.EqualTo(3));
        Assert.That(inliers, Is.EqualTo(2));
        Assert.That(inlierSlots, Is.EquivalentTo(new[] { 0, 1 }), "37 (slot 2) is the one thrown out");
        Assert.That(sim.Scale, Is.EqualTo(truth.Scale).Within(0.03f),
            "the fold carries the honest pair's scale, not the 21%-off one 3+37 would give");
    }

    /// <summary>Rotate a camera's orientation in place by <paramref name="degrees"/> about <paramref name="axis"/>.</summary>
    private static void Twist(CameraParams cam, Vector3 axis, float degrees)
    {
        var m = Matrix4x4.CreateFromAxisAngle(Vector3.Normalize(axis), degrees * MathF.PI / 180f);
        cam.Forward = Vector3.Normalize(Vector3.Transform(cam.Forward, m));
        cam.Up = Vector3.Normalize(Vector3.Transform(cam.Up, m));
    }

    /// <summary>
    /// Three of four agree, one is somewhere else entirely: the three fold the chunk and the
    /// fourth is named. This is Bathroom's anchor 17 and it used to cost the whole pass.
    /// </summary>
    [Test]
    public void FitChunkToReference_DropsTheOneMisplacedAnchorOfFour()
    {
        var chunk = new MultiViewChunk(new[] { 0, 1, 2, 3, 7 }, AnchorCount: 4);
        var reference = new Dictionary<int, CameraParams>
        {
            [0] = CamAt(Vector3.Zero, aimedAt: 0),
            [1] = CamAt(new Vector3(1, 0, 0), aimedAt: 1),
            [2] = CamAt(new Vector3(0, 1, 0), aimedAt: 2),
            [3] = CamAt(new Vector3(0, 0, 1), aimedAt: 3),
        };
        var cams = new CameraParams?[]
        {
            Clone(reference[0]), Clone(reference[1]), Clone(reference[2]),
            CamAt(new Vector3(0, 0, -4f), aimedAt: 3),
            CamAt(Vector3.One),
        };

        Assert.That(MultiViewChunkPlan.TryFitChunkToReference(
            chunk, cams, reference, out var sim, out var rms, out _, out _, out int inliers, out int[] slots), Is.True);
        Assert.That(inliers, Is.EqualTo(3));
        Assert.That(slots, Is.EquivalentTo(new[] { 0, 1, 2 }));
        Assert.That(rms, Is.LessThan(1e-4f));
        Assert.That(sim.Scale, Is.EqualTo(1f).Within(1e-4f));
    }

    // ---- helpers ----

    private static CameraParams CamAt(Vector3 p) => new()
    {
        Width = 640, Height = 480, FocalX = 500, FocalY = 500, CenterX = 320, CenterY = 240,
        Position = p, Forward = -Vector3.UnitZ, Up = Vector3.UnitY,
    };

    /// <summary>A camera at <paramref name="p"/> with a distinct, non-degenerate orientation per index.</summary>
    private static CameraParams CamAt(Vector3 p, int aimedAt)
    {
        var cam = CamAt(p);
        var q = Quaternion.CreateFromYawPitchRoll(0.7f * aimedAt, 0.31f * aimedAt - 0.4f, 0.17f * aimedAt);
        cam.Forward = Vector3.Normalize(Vector3.Transform(-Vector3.UnitZ, q));
        cam.Up = Vector3.Normalize(Vector3.Transform(Vector3.UnitY, q));
        return cam;
    }

    /// <summary>The camera as a pass whose frame is the INVERSE of <paramref name="frame"/> reports it.</summary>
    private static CameraParams Seen(Similarity3 frame, CameraParams truth)
    {
        var cam = Clone(truth);
        InverseSimilarity(frame).ApplyToCamera(cam);
        return cam;
    }

    private static CameraParams Clone(CameraParams c) => new()
    {
        Width = c.Width, Height = c.Height, FocalX = c.FocalX, FocalY = c.FocalY,
        CenterX = c.CenterX, CenterY = c.CenterY, Near = c.Near, Far = c.Far,
        Position = c.Position, Forward = c.Forward, Up = c.Up,
    };

    /// <summary>Cameras walking an arc and looking inward, roughly what a handheld room capture is.</summary>
    private static CameraParams[] BuildHandheldRig(int n)
    {
        var rig = new CameraParams[n];
        for (int i = 0; i < n; i++)
        {
            float t = i / (float)(n - 1);
            float a = t * 2.2f;
            var pos = new Vector3(MathF.Cos(a) * 1.6f, 0.2f + 0.3f * MathF.Sin(t * 5f), MathF.Sin(a) * 1.6f);
            var fwd = Vector3.Normalize(new Vector3(0, 0.1f, 0) - pos);
            var right = Vector3.Normalize(Vector3.Cross(fwd, Vector3.UnitY));
            var up = Vector3.Normalize(Vector3.Cross(right, fwd));
            rig[i] = CamAt(pos);
            rig[i].Forward = fwd;
            rig[i].Up = up;
        }
        return rig;
    }

    private static Similarity3 RandomSimilarity(Random rng)
    {
        float F() => (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Similarity3(
            0.3f + (float)rng.NextDouble() * 3f,
            Matrix4x4.CreateFromYawPitchRoll(F() * 3f, F() * 1.4f, F() * 3f),
            new Vector3(F() * 8f, F() * 8f, F() * 8f));
    }

    private static Similarity3 InverseSimilarity(Similarity3 s)
    {
        Matrix4x4.Invert(s.Rotation, out var rInv);
        float invScale = 1f / s.Scale;
        // p = R^-1 (p' - t) / scale  =>  scale' = 1/scale, R' = R^-1, t' = -R^-1 t / scale
        return new Similarity3(invScale, rInv, -invScale * Vector3.Transform(s.Translation, rInv));
    }

}
