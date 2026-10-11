using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// &amp;skymap=1: a learned equirectangular sky (SplatTrainerShaders.SkyComposite / SkyBackward / SkyAdam) composited
/// behind the splats wherever transmittance stays open - out = sum(T a c) + T_final x (background + sky(ray)). Parity
/// 2026-10-10: Train's held-out gap to Brush is bright blobs in the SKY of novel views, painted by an opaque far shell of
/// splats (6.7% past 16 camera spreads, grown only by densify) that few views constrain; a map at infinity has no depth
/// to get wrong. The map persists across densify resizes. Training-side only for now: held-out renders include it; the
/// saved scene and the viewer do not (yet).
/// </summary>
public sealed partial class SplatTrainerGpu
{
    /// <summary>&amp;skymap=1. Read when the pipelines are built (the raster backward's //SKY: lines).</summary>
    public static bool SkyMap { get; set; }

    /// <summary>&amp;skylr=X: Adam learning rate of the sky texels (raw RGB).</summary>
    public static float SkyLr { get; set; } = 0.005f;

    GPUComputePipeline? _skyComposite, _skyBackward, _skyAdam;
    MemoryBuffer1D<float, Stride1D.Dense>? _skyMapBuf, _skyM, _skyV, _skyPix, _skyDot;
    MemoryBuffer1D<int, Stride1D.Dense>? _skyGrad;
    GPUBuffer? _skyCfgBuf;
    int _skySteps, _skyPixels;

    void EnsureSky()
    {
        var accel = _gpu.WebGPUAccelerator;
        _skyComposite ??= MakePipeline(SplatTrainerShaders.SkyComposite, "sky_composite");
        _skyBackward ??= MakePipeline(SplatTrainerShaders.SkyBackward, "sky_backward");
        _skyAdam ??= MakePipeline(SplatTrainerShaders.SkyAdam, "sky_adam");
        int texels = SplatTrainerShaders.SkyW * SplatTrainerShaders.SkyH * 3;
        if (_skyMapBuf == null)
        {
            _skyMapBuf = accel.Allocate1D<float>(texels); _skyMapBuf.MemSetToZero();
            _skyM = accel.Allocate1D<float>(texels); _skyM.MemSetToZero();
            _skyV = accel.Allocate1D<float>(texels); _skyV.MemSetToZero();
            _skyGrad = accel.Allocate1D<int>(texels); _skyGrad.MemSetToZero();
            _skyCfgBuf = _device!.CreateBuffer(new GPUBufferDescriptor { Size = 32, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
            _skySteps = 0;
            Console.WriteLine($"[Trainer] sky map: {SplatTrainerShaders.SkyW}x{SplatTrainerShaders.SkyH} equirect behind the splats, lr {SkyLr:G3}");
        }
        int pixels = _width * _height;
        if (_skyPixels != pixels)
        {
            _skyPix?.Dispose(); _skyDot?.Dispose();
            _skyPix = accel.Allocate1D<float>((long)pixels * 3); _skyPix.MemSetToZero();
            _skyDot = accel.Allocate1D<float>(pixels); _skyDot.MemSetToZero();
            _skyPixels = pixels;
        }
    }

    /// <summary>Composite the sky behind this forward's splats (after raster_forward, before any exposure).</summary>
    void SkyCompositeDispatch()
    {
        EnsureSky();
        DispatchLinear(_skyComposite!, (long)_width * _height, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, _skyMapBuf!.GetGPUBuffer()!), Buf(2, _outColour!.GetGPUBuffer()!),
            Buf(3, _outFinalT!.GetGPUBuffer()!), Buf(4, _skyPix!.GetGPUBuffer()!),
        });
    }

    /// <summary>After the loss, before the raster backward: the splats' sky term and the sky map's gradient.</summary>
    void SkyBackwardDispatch()
    {
        DispatchLinear(_skyBackward!, (long)_width * _height, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, _dLdPix!.GetGPUBuffer()!), Buf(2, _outFinalT!.GetGPUBuffer()!),
            Buf(3, _skyPix!.GetGPUBuffer()!), Buf(4, _skyDot!.GetGPUBuffer()!), Buf(5, _skyGrad!.GetGPUBuffer()!),
        });
    }

    /// <summary>The raster backward's binding 16 (sky_dot) when the sky map is on.</summary>
    GPUBindGroupEntry SkyDotEntry() => Buf(16, _skyDot!.GetGPUBuffer()!);

    /// <summary>One Adam step on the sky texels.</summary>
    void SkyAdamDispatch()
    {
        _skySteps++;
        int texels = SplatTrainerShaders.SkyW * SplatTrainerShaders.SkyH * 3;
        var cfg = new byte[32];
        BitConverter.TryWriteBytes(cfg.AsSpan(0), SkyLr);
        BitConverter.TryWriteBytes(cfg.AsSpan(4), 0.9f);
        BitConverter.TryWriteBytes(cfg.AsSpan(8), 0.999f);
        BitConverter.TryWriteBytes(cfg.AsSpan(12), (float)_skySteps);
        BitConverter.TryWriteBytes(cfg.AsSpan(16), (uint)texels);
        _queue!.WriteBuffer(_skyCfgBuf!, 0, cfg);
        DispatchLinear(_skyAdam!, texels, new[]
        {
            Buf(0, _skyCfgBuf!), Buf(1, _skyMapBuf!.GetGPUBuffer()!), Buf(2, _skyM!.GetGPUBuffer()!),
            Buf(3, _skyV!.GetGPUBuffer()!), Buf(4, _skyGrad!.GetGPUBuffer()!),
        });
    }

    void DisposeSky()
    {
        _skyMapBuf?.Dispose(); _skyM?.Dispose(); _skyV?.Dispose(); _skyGrad?.Dispose(); _skyPix?.Dispose(); _skyDot?.Dispose();
        _skyMapBuf = _skyM = _skyV = _skyPix = _skyDot = null; _skyGrad = null; _skyPixels = 0;
        _skyCfgBuf?.Destroy(); _skyCfgBuf?.Dispose(); _skyCfgBuf = null;
    }
}
