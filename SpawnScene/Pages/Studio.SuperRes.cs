using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnScene.Pages;

/// <summary>
/// Super-resolution before a single photo becomes splats: SpawnDev.ILGPU.ML's native SuperResolutionPipeline (ESPCN x3
/// on the luminance channel, tiled, GPU in and out - the ML demo's SR page), loaded through the same hub source as the
/// depth model. A small photo gives few, large splats: tripled, the colour detail is finer and the camera can come
/// closer before the scene turns to blobs. Depth is estimated from the upscaled photo too (its model resizes to its own
/// patch grid either way); only the splat grid and colours gain.
/// </summary>
public partial class Studio
{
    [Inject] private IModelSource _modelSource { get; set; } = default!;
    InferenceSession? _srSession;
    SuperResolutionPipeline? _srPipeline;

    /// <summary>Auto upscales photos whose longer side is under this many pixels: x3 makes ~9x the splats (the Room sample,
    /// 640 px: 225K -> 1.98M), so a bigger photo gains less and costs more. x3 in the settings does it for any photo.</summary>
    public const int SuperResAutoBelowPx = 800;

    /// <summary>&amp;superres=off|auto|on overrides the project's setting (harness A/B).</summary>
    public static SpawnScene.Models.SuperResolutionMode? SuperResOverride { get; set; }

    bool ShouldSuperResolve(SpawnScene.Models.SuperResolutionMode mode, int w, int h) => mode switch
    {
        SpawnScene.Models.SuperResolutionMode.On => true,
        SpawnScene.Models.SuperResolutionMode.Auto => Math.Max(w, h) < SuperResAutoBelowPx,
        _ => false,
    };

    /// <summary>
    /// Upscale a GPU-resident packed RGBA photo x3 (the caller disposes <paramref name="rgba"/>; the result is a new buffer
    /// it owns). Null if the model cannot load - the caller goes on at the photo's own size.
    /// </summary>
    async Task<(MemoryBuffer1D<int, Stride1D.Dense> Rgba, int Width, int Height)?> SuperResolveAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgba, int w, int h)
    {
        var a = _gpuService.WebGPUAccelerator;
        try
        {
            if (_srPipeline == null)
            {
                using var stream = await _modelSource.OpenAsync(ModelHub.KnownModels.SuperResolution, "super-resolution-10.onnx");
                _srSession = await InferenceSession.CreateFromOnnxStreamAsync(a, stream);
                _srPipeline = new SuperResolutionPipeline(_srSession, a);
            }
            var t0 = DateTime.UtcNow;
            var (up, uw, uh) = await _srPipeline.UpscaleGpuAsync(rgba.View, w, h);
            using (up)
            {
                var flat = a.Allocate1D<int>((long)uw * uh);
                // A device-to-device copy out of the 2D result's flat view (SpawnDev.ILGPU 5.3.4+: before it, CopyTo threw on
                // WebGPU and Wasm for a device target - found here, 2026-10-07).
                ArrayView1D<int, Stride1D.Dense> src = up.View.BaseView.SubView(0, (long)uw * uh);
                src.CopyTo(flat.View);
                await a.SynchronizeAsync();
                Console.WriteLine($"[SuperRes] {w}x{h} -> {uw}x{uh} (ESPCN x3) in {(DateTime.UtcNow - t0).TotalMilliseconds:F0} ms");
                return (flat, uw, uh);
            }
        }
        catch (Exception ex)
        {
            var where = string.Join(" | ", (ex.StackTrace ?? "").Split('\n').Take(6).Select(l => l.Trim()));
            Console.WriteLine($"[SuperRes] FAIL, continuing at {w}x{h}: {ex.Message} AT {where}");
            return null;
        }
    }
}
