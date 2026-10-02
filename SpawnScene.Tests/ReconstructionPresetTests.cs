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
            Assert.That((s.TrainIterations, s.TrainMaxDimension, s.TrainMaxSplats), Is.EqualTo((p.Iterations, p.MaxDimension, p.MaxSplats)), p.Name);
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
    public void PhotoResolution_TrainsAtThePhotosOwnSize()
    {
        var imported = new CameraParams { Width = 768, Height = 1024, FocalX = 790, FocalY = 790, CenterX = 384, CenterY = 512 };
        Assert.That(imported.TrainingSize(ReconstructionPresets.PhotoSize, 4160), Is.EqualTo((3120, 4160)));
    }
}
