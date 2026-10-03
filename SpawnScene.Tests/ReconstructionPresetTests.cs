using NUnit.Framework;
using SpawnScene.Models;

namespace SpawnScene.Tests;

/// <summary>
/// Multi-photo quality presets (2026-10-02): each one sets iterations, training resolution and the splat cap together,
/// a hand edit reads as "Custom", and the "Photo" resolution trains at the photos' own size.
/// </summary>
public class ReconstructionPresetTests
{
    [Test]
    public void EveryPreset_AppliesAndIsRecognised()
    {
        foreach (var p in ReconstructionPresets.All)
        {
            var s = new ProjectSettings();
            Assert.That(ReconstructionPresets.Apply(s, p.Name), Is.True, p.Name);
            Assert.That((s.TrainIterations, s.TrainMaxDimension, s.TrainMaxSplats, s.LearnedKeypoints),
                Is.EqualTo((p.Iterations, p.MaxDimension, p.MaxSplats, p.Keypoints)), p.Name);
            Assert.That(ReconstructionPresets.Match(s), Is.EqualTo(p.Name), p.Name);
        }
    }

    [Test]
    public void Defaults_AreStandard_AndAHandEditIsCustom()
    {
        var s = new ProjectSettings();
        Assert.That(ReconstructionPresets.Match(s), Is.EqualTo("Standard"), "new projects start on Standard");
        s.TrainIterations = 12345;
        Assert.That(ReconstructionPresets.Match(s), Is.EqualTo("Custom"));
        Assert.That(ReconstructionPresets.Apply(s, "Nope"), Is.False);
    }

    [Test]
    public void OnlyDraft_CapsSplats_TheRestTakeWhatTheDeviceFits()
    {
        // Very large scenes are a goal (TJ, 2026-10-02); the fixed 3M cap on every preset was not.
        foreach (var p in ReconstructionPresets.All)
            Assert.That(p.MaxSplats, p.Name == "Draft" ? Is.EqualTo(500_000) : Is.EqualTo(ReconstructionPresets.DeviceMaxSplats), p.Name);
    }

    [Test]
    public void HighAndMax_Use3072Keypoints_DraftAndStandard1024()
    {
        foreach (var p in ReconstructionPresets.All)
            Assert.That(p.Keypoints, Is.EqualTo(p.Name is "High" or "Max" ? 3072 : 1024), p.Name);
        var s = new ProjectSettings();
        ReconstructionPresets.Apply(s, "High");
        s.LearnedKeypoints = 1024;
        Assert.That(ReconstructionPresets.Match(s), Is.EqualTo("Custom"), "keypoints are part of the preset");
    }

    [Test]
    public void LegacyHigh_With1024Keypoints_Upgrades()
    {
        // Saved as High before High took 3072 (and with the old 3M cap): both move to High's current values.
        var s = new ProjectSettings { TrainIterations = 15000, TrainMaxDimension = 1600, TrainMaxSplats = 3_000_000, LearnedKeypoints = 1024, ReconstructionPreset = "High" };
        Assert.That(ReconstructionPresets.UpgradeLegacyPresetValues(s), Is.True);
        Assert.That((s.TrainMaxSplats, s.LearnedKeypoints), Is.EqualTo((ReconstructionPresets.DeviceMaxSplats, 3072)));
        Assert.That(ReconstructionPresets.Match(s), Is.EqualTo("High"));
        // Already current: nothing to do.
        Assert.That(ReconstructionPresets.UpgradeLegacyPresetValues(s), Is.False);
    }

    [Test]
    public void LegacyThreeMillionCap_UpgradesOnlyAnUneditedPreset()
    {
        // Saved as Standard with the old 3M cap: takes Standard's device-max cap and still reads as Standard.
        var s = new ProjectSettings { TrainIterations = 7000, TrainMaxDimension = 1024, TrainMaxSplats = 3_000_000, ReconstructionPreset = "Standard" };
        Assert.That(ReconstructionPresets.UpgradeLegacyPresetValues(s), Is.True);
        Assert.That(s.TrainMaxSplats, Is.EqualTo(ReconstructionPresets.DeviceMaxSplats));
        Assert.That(ReconstructionPresets.Match(s), Is.EqualTo("Standard"));
        // Custom (iterations edited by hand): the user's 3M stays.
        var custom = new ProjectSettings { TrainIterations = 9000, TrainMaxDimension = 1024, TrainMaxSplats = 3_000_000, ReconstructionPreset = "Custom" };
        Assert.That(ReconstructionPresets.UpgradeLegacyPresetValues(custom), Is.False);
        Assert.That(custom.TrainMaxSplats, Is.EqualTo(3_000_000));
        // Draft never had 3M: untouched.
        var draft = new ProjectSettings();
        ReconstructionPresets.Apply(draft, "Draft");
        Assert.That(ReconstructionPresets.UpgradeLegacyPresetValues(draft), Is.False);
        Assert.That(draft.TrainMaxSplats, Is.EqualTo(500_000));
    }

    [Test]
    public void PhotoResolution_TrainsAtThePhotosOwnSize()
    {
        var imported = new CameraParams { Width = 768, Height = 1024, FocalX = 790, FocalY = 790, CenterX = 384, CenterY = 512 };
        Assert.That(imported.TrainingSize(ReconstructionPresets.PhotoSize, 4160), Is.EqualTo((3120, 4160)));
    }
}
