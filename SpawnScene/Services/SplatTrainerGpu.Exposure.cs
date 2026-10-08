using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;

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

    /// <summary>The mean affine of the supervised photos, folded into the scene (<see cref="FoldMeanExposureAsync"/>).</summary>
    public struct ExposureFold
    {
        public float M00, M01, M02, B0, M10, M11, M12, B1, M20, M21, M22, B2;
        public int Count;
    }

    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ExposureFold>? _foldKernel;
    static Accelerator? _foldFor;

    /// <summary>
    /// Fold the supervised photos' MEAN exposure into the scene's base colours, so what the viewer shows matches the
    /// average photo. Nothing pins the exposures' overall level during training: on Bathroom (f2, 2026-10-07) every gain
    /// drifted up (0.95..1.36, ~1.15 mean) while the scene drifted darker, and held-out photos - scored at identity -
    /// lost 1.3 dB. rgb = C0 dc + 0.5, so dc' = (M (C0 dc + 0.5) + b - 0.5) / C0 is exact for the base colour; the SH
    /// bands (view-dependent residuals) are left as they are. Returns the folded mean for the log, or null.
    /// </summary>
    public async Task<ExposureFold?> FoldMeanExposureAsync(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        IReadOnlyList<int> slots)
    {
        if (_exposure == null || slots.Count == 0 || splatCount <= 0) return null;
        // CPU transfer: 12 floats a view, once a run.
        var e = await _exposure.CopyToHostAsync<float>(0, _exposureViews * 12L);
        var m = new double[12];
        int used = 0;
        foreach (int v in slots)
        {
            if (v < 0 || v >= _exposureViews) continue;
            for (int k = 0; k < 12; k++) m[k] += e[v * 12 + k];
            used++;
        }
        if (used == 0) return null;
        for (int k = 0; k < 12; k++) m[k] /= used;
        var f = new ExposureFold
        {
            M00 = (float)m[0], M01 = (float)m[1], M02 = (float)m[2], B0 = (float)m[3],
            M10 = (float)m[4], M11 = (float)m[5], M12 = (float)m[6], B1 = (float)m[7],
            M20 = (float)m[8], M21 = (float)m[9], M22 = (float)m[10], B2 = (float)m[11],
            Count = splatCount,
        };
        var a = _gpu.WebGPUAccelerator;
        if (!ReferenceEquals(_foldFor, a)) { _foldKernel = null; _foldFor = a; }
        _foldKernel ??= a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ExposureFold>(FoldKernel);
        _foldKernel(splatCount, splatBuf.View, f);
        await a.SynchronizeAsync();
        return f;
    }

    static void FoldKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> splats, ExposureFold f)
    {
        if (i >= f.Count) return;
        long o = (long)i.X * SplatFormat.Floats + SplatFormat.OffColor;
        const float C0 = 0.28209479177387814f;
        float r = C0 * splats[o] + 0.5f, g = C0 * splats[o + 1] + 0.5f, b = C0 * splats[o + 2] + 0.5f;
        float r2 = f.M00 * r + f.M01 * g + f.M02 * b + f.B0;
        float g2 = f.M10 * r + f.M11 * g + f.M12 * b + f.B1;
        float b2 = f.M20 * r + f.M21 * g + f.M22 * b + f.B2;
        splats[o] = (r2 - 0.5f) / C0; splats[o + 1] = (g2 - 0.5f) / C0; splats[o + 2] = (b2 - 0.5f) / C0;
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
