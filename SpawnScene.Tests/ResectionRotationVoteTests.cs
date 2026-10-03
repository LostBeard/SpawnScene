using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Re-registration's rotation vote (GlobalSfmInit.RotationVotes, 2026-10-03): a resected camera must agree with the
/// relative rotations of its own verified pairs. DrJohnson at K=3072 re-registered two cameras on the mirror twin of
/// their support (~126% of the spread off, forward ~100 deg) with most correspondences agreeing.
/// </summary>
public class ResectionRotationVoteTests
{
    static CameraParams Cam(Vector3 pos, float yawDeg)
    {
        float r = yawDeg * MathF.PI / 180f;
        return new CameraParams
        {
            Width = 640, Height = 480, FocalX = 500, FocalY = 500, CenterX = 320, CenterY = 240,
            Position = pos, Forward = new Vector3(MathF.Sin(r), 0, MathF.Cos(r)), Up = Vector3.UnitY,
        };
    }

    // A verified pair carrying the TRUE relative rotation (R_B = R * R_A) and `matches` matches.
    static GlobalSfmInit.PairMatch Pair(int a, int b, CameraParams ca, CameraParams cb, int matches) =>
        new(a, b, GlobalSfmInit.Mul(GlobalSfmInit.RotationOf(cb), GlobalSfmInit.Transpose(GlobalSfmInit.RotationOf(ca))),
            new (int, int)[matches], new (int, int)[matches]);

    [Test]
    public void TruePoseWins_MirrorTwinLoses_FromEitherSideOfThePairs()
    {
        var truth = new[] { Cam(new(0, 0, 0), 0), Cam(new(1, 0, 0), 15), Cam(new(2, 0, 0), 30) };
        // Camera 2 is the one being re-registered; it is CamB of one pair and CamA of the other.
        var pairs = new List<GlobalSfmInit.PairMatch>
        {
            Pair(0, 2, truth[0], truth[2], 300),
            Pair(2, 1, truth[2], truth[1], 200),
        };
        Func<int, bool> placed = i => i != 2;

        var (agree, disagree) = GlobalSfmInit.RotationVotes(2, truth[2], pairs, truth, placed);
        Assert.That((agree, disagree), Is.EqualTo((500, 0)));

        var twin = Cam(truth[2].Position, 30 + 95); // the resection's wrong solution: ~95 deg off
        (agree, disagree) = GlobalSfmInit.RotationVotes(2, twin, pairs, truth, placed);
        Assert.That((agree, disagree), Is.EqualTo((0, 500)));
    }

    [Test]
    public void MatchWeighted_OneWrongPairCannotOutvoteTheRest()
    {
        var truth = new[] { Cam(new(0, 0, 0), 0), Cam(new(1, 0, 0), 15), Cam(new(2, 0, 0), 30) };
        var bogus = Cam(new(0, 0, 0), 120); // a repeated-structure pair whose relative rotation is wrong
        var pairs = new List<GlobalSfmInit.PairMatch>
        {
            Pair(0, 2, truth[0], truth[2], 300),
            Pair(1, 2, bogus, truth[2], 40),
        };
        var (agree, disagree) = GlobalSfmInit.RotationVotes(2, truth[2], pairs, truth, i => i != 2);
        Assert.That(agree, Is.EqualTo(300));
        Assert.That(disagree, Is.EqualTo(40));
        Assert.That(agree > disagree, Is.True, "the true pose stays accepted");
    }
}
