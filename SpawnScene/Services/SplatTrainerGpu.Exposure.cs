using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Per-photo exposure compensation (SplatTrainerShaders.ExposureApply / ExposureBackward / ExposureAdam): the reference
/// 3DGS's appearance model (Inria, 2024: a 3x4 affine per training image on the rendered colour, identity at the start,
/// Adam lr 0.01 -> 0.001). A phone exposes, tone-maps and white-balances every shot on its own - TJ's Bathroom spans 6x
/// in shutter x ISO, with HDR merges mixed in - and a scene that must reproduce every photo's brightness exactly does it
/// with floaters in front of the cameras. With the affine, the scene keeps ONE appearance (what the viewer shows; the
/// exposures never leave training) and each photo's 12 numbers explain the rest.
/// </summary>
public sealed partial class SplatTrainerGpu
{
    GPUComputePipeline? _exposureApply, _exposureBackward, _exposureAdam;
    MemoryBuffer1D<float, Stride1D.Dense>? _exposure;          // 12 per view
    MemoryBuffer1D<float, Stride1D.Dense>? _exposureMoments;   // 24 per view: Adam m, v
    MemoryBuffer1D<float, Stride1D.Dense>? _exposureSteps;     // 1 per view
    MemoryBuffer1D<float, Stride1D.Dense>? _rawColour;         // 3 per pixel: the render before the exposure
    MemoryBuffer1D<float, Stride1D.Dense>? _exposurePartials;  // 12 per loss workgroup
    GPUBuffer? _exposureDimsBuf, _exposureCfgBuf;
    int _exposureViews;

    /// <summary>Learning rate of the next exposure steps (the caller decays it: the reference goes 0.01 -> 0.001).</summary>
    public float ExposureLr { get; set; } = 0.01f;

    /// <summary>Identity exposure for <paramref name="views"/> photos, fresh optimiser state.</summary>
    public void ResetExposure(int views)
    {
        var accel = _gpu.WebGPUAccelerator;
        _exposureApply ??= MakePipeline(SplatTrainerShaders.ExposureApply, "exposure_apply");
        _exposureBackward ??= MakePipeline(SplatTrainerShaders.ExposureBackward, "exposure_backward");
        _exposureAdam ??= MakePipeline(SplatTrainerShaders.ExposureAdam, "exposure_adam");
        _exposureDimsBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 16, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        _exposureCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 16, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        views = Math.Max(1, views);
        if (_exposure == null || _exposureViews < views)
        {
            _exposure?.Dispose(); _exposureMoments?.Dispose(); _exposureSteps?.Dispose();
            _exposure = accel.Allocate1D<float>(views * 12L);
            _exposureMoments = accel.Allocate1D<float>(views * 24L);
            _exposureSteps = accel.Allocate1D<float>(views);
            _exposureViews = views;
        }
        // CPU transfer: the identity rows, 48 bytes a view, once a run.
        var identity = new float[_exposureViews * 12];
        for (int v = 0; v < _exposureViews; v++) { identity[v * 12] = 1f; identity[v * 12 + 5] = 1f; identity[v * 12 + 10] = 1f; }
        _exposure.CopyFromCPU(identity);
        _exposureMoments!.MemSetToZero();
        _exposureSteps!.MemSetToZero();
        accel.FlushPendingCommands();
    }

    /// <summary>Every view's 12 exposure numbers (rows m[c] = gains on r, g, b and an offset). CPU transfer: a report.</summary>
    public async Task<float[]> ReadExposureAsync(int views)
    {
        if (_exposure == null) return System.Array.Empty<float>();
        return await _exposure.CopyToHostAsync<float>(0, Math.Min(views, _exposureViews) * 12L);
    }

    bool ExposureReady(int slot) => _exposure != null && slot >= 0 && slot < _exposureViews;

    /// <summary>Pass 1, after the forward: keep the raw render, put the exposed one where the loss reads.</summary>
    void DispatchExposureApply(int slot, int pixels, int groupsX, int groupsY, int groups)
    {
        long need = (long)pixels * 3;
        if (_rawColour == null || _rawColour.Length < need)
        {
            _rawColour?.Dispose();
            _rawColour = _gpu.WebGPUAccelerator.Allocate1D<float>(need);
        }
        if (_exposurePartials == null || _exposurePartials.Length < groups * 12L)
        {
            _exposurePartials?.Dispose();
            _exposurePartials = _gpu.WebGPUAccelerator.Allocate1D<float>(groups * 12L);
        }
        WriteU32x4(_exposureDimsBuf!, (uint)pixels, (uint)groupsX, (uint)slot, (uint)groups);
        Dispatch(_exposureApply!, groupsX, groupsY, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _rawColour.GetGPUBuffer()!),
            Buf(2, _exposure!.GetGPUBuffer()!), Buf(3, _exposureDimsBuf!),
        });
    }

    /// <summary>Pass 2 + 3, after the loss gradients: chain dL/dpixel back through the affine, step the exposure.</summary>
    void DispatchExposureBackward(int slot, int groupsX, int groupsY, int groups)
    {
        Dispatch(_exposureBackward!, groupsX, groupsY, new[]
        {
            Buf(0, _dLdPix!.GetGPUBuffer()!), Buf(1, _rawColour!.GetGPUBuffer()!),
            Buf(2, _exposure!.GetGPUBuffer()!), Buf(3, _exposurePartials!.GetGPUBuffer()!), Buf(4, _exposureDimsBuf!),
        });
        WriteVec4(_exposureCfgBuf!, groups, slot, ExposureLr, 0f);
        Dispatch(_exposureAdam!, 1, 1, new[]
        {
            Buf(0, _exposurePartials.GetGPUBuffer()!), Buf(1, _exposure.GetGPUBuffer()!),
            Buf(2, _exposureMoments!.GetGPUBuffer()!), Buf(3, _exposureSteps!.GetGPUBuffer()!), Buf(4, _exposureCfgBuf!),
        });
    }

    void DisposeExposure()
    {
        _exposure?.Dispose(); _exposure = null;
        _exposureMoments?.Dispose(); _exposureMoments = null;
        _exposureSteps?.Dispose(); _exposureSteps = null;
        _rawColour?.Dispose(); _rawColour = null;
        _exposurePartials?.Dispose(); _exposurePartials = null;
        _exposureViews = 0;
    }
}
