using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>&amp;fillunseen=1: after training, paint what no photo saw around the capture position (UnseenFill).
    /// Opt-in (TJ 2026-10-08): invented content. The model is HiddenLayerInpaint's (&amp;inpaintmodel=lama for big-LaMa).</summary>
    public static bool FillUnseenOption { get; set; }

    HiddenLayerInpaint? _unseenInpaint;

    /// <summary>Fill the trained scene on screen from the training cameras' centre, before it is saved.</summary>
    async Task FillUnseenAsync(GaussianScene scene)
    {
        if (_trainer == null || scene.TrainingCameras.Count == 0) return;
        var centre = Vector3.Zero;
        var up = Vector3.Zero;
        foreach (var c in scene.TrainingCameras) { centre += c.Position; up += c.Up; }
        centre /= scene.TrainingCameras.Count;
        up = up.LengthSquared() > 1e-12f ? Vector3.Normalize(up) : Vector3.UnitY;
        _unseenInpaint ??= new HiddenLayerInpaint(_modelSource);
        _trainHudText = "Painting what no photo saw…";
        if (_hudTrainLabel != null) _hudTrainLabel.Text = _trainHudText;
        int before = _gpuRenderer.SplatCount;
        var t0 = DateTime.UtcNow;
        int added = await UnseenFill.FillAsync(_gpuService.WebGPUAccelerator, _trainer, _gpuRenderer, _unseenInpaint, centre, up);
        if (_sceneManager.ActiveScene != null) _sceneManager.ActiveScene.GpuSplatCount = _gpuRenderer.SplatCount;
        Console.WriteLine($"[Fill] unseen fill ({HiddenLayerInpaint.Model}) from {centre}: {added:N0} splats added " +
            $"({before:N0} -> {_gpuRenderer.SplatCount:N0}) in {(DateTime.UtcNow - t0).TotalSeconds:F1}s");
    }
}
