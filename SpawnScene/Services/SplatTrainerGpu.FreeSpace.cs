using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Free space (TJ 2026-10-13: "thousands of floaters in the final results and none of them show up on the depth maps"):
/// every photo's depth map says the space between the camera and its surface is empty. The loss per pixel is the opacity
/// piled up in front of that surface, 1 - T_f (T_f = transmittance after the last splat closer than (1 - margin) x the
/// photo's depth), so a faint floater gets the same push as an opaque one - unlike the depth loss
/// (SplatTrainerGpu.Depth), whose L1 on the blended MEAN inverse depth barely moves for a splat carrying 5% of a pixel.
/// <para>
/// Build-time switch <see cref="FreeSpace"/> (the backward's <c>//FREE:</c> lines, binding 17); per step the view's map
/// comes from <see cref="SetDepthTarget"/> and <see cref="FreeSpaceWeight"/> scales it (0 = off: the prep writes zeros).
/// The prep pass (SplatTrainerShaders.FreeSpacePrep) maps pixels to the map exactly as the floater census does, so the
/// census measures this loss (TrainerGate photo-depth case).
/// </para>
/// </summary>
public sealed partial class SplatTrainerGpu
{
    /// <summary>Build the backward with the free-space term (read when the pipelines are built). <c>&amp;freespace=X</c>.</summary>
    public static bool FreeSpace { get; set; } = true;

    /// <summary>The term's weight (lambda; the loss is lambda x the mean over pixels of 1 - T_f). 0 = no term this step.</summary>
    public float FreeSpaceWeight { get; set; }

    /// <summary>In front = closer than (1 - margin) x the photo's depth.</summary>
    public float FreeSpaceMargin { get; set; } = 0.15f;

    /// <summary>
    /// Cover mode: scale each pixel's term by the opacity BEHIND the splats in front of the photo's surface
    /// (B = 1 - T_final / T_f). Floaters in front of something solid get the whole push; splats with nothing behind them
    /// keep their opacity, so a wrong far depth (a blown-out window) cannot open a see-through hole.
    /// </summary>
    public bool FreeSpaceCover { get; set; }

    /// <summary>The photo's surface at a pixel = the nearest map depth within this many map pixels (edges, thin parts).</summary>
    public int FreeSpaceRadius { get; set; } = 1;

    bool _freeBuilt;   // this trainer's pipelines carry the //FREE: lines (FreeSpace when they were built)
    GPUComputePipeline? _freeSpacePrep;
    MemoryBuffer1D<float, Stride1D.Dense>? _freePx;     // 2 per pixel: threshold 1/z, weight
    MemoryBuffer1D<float, Stride1D.Dense>? _freeNoMap;
    GPUBuffer? _freeCfgBuf;

    /// <summary>The free-space build needs one storage buffer more than the plain backward (12; 13 with the depth loss).</summary>
    void CheckFreeSpaceLimits()
    {
        if (!FreeSpace) return;
        int maxStorage = _gpu.WebGPUAccelerator.NativeAccelerator.MaxStorageBuffersPerShaderStage;
        if (maxStorage < 12)
        {
            FreeSpace = false;
            Console.WriteLine($"[Trainer] free space off: the device allows {maxStorage} storage buffers a shader, it needs 12");
        }
    }

    /// <summary>Per-pixel thresholds and weights for this step's view into <see cref="_freePx"/> (zeros without a map).</summary>
    void DispatchFreeSpacePrep(int pixels)
    {
        var accel = _gpu.WebGPUAccelerator;
        _freeSpacePrep ??= MakePipeline(SplatTrainerShaders.FreeSpacePrep, "free_space_prep");
        _freeCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 48, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        _freeNoMap ??= accel.Allocate1D<float>(4);
        if (_freePx == null || _freePx.Length < 2L * pixels)
        {
            _freePx?.Dispose();
            _freePx = accel.Allocate1D<float>(2L * pixels);
        }
        bool mapped = FreeSpaceWeight > 0f && _depthMaps != null;
        var f = new float[]
        {
            mapped ? _depthScale : 0f, mapped ? _depthMapW : 1f, mapped ? _depthMapH : 1f, Math.Max(0, FreeSpaceRadius),
            1f - Math.Clamp(FreeSpaceMargin, 0f, 0.9f), FreeSpaceWeight / Math.Max(1, pixels), FreeSpaceCover ? 1f : 0f, 0f,
        };
        var nu = new uint[] { (uint)_width, (uint)_height, mapped ? (uint)_depthMapOffset : 0u, 0u };
        var bytes = new byte[48];
        Buffer.BlockCopy(f, 0, bytes, 0, 32);
        Buffer.BlockCopy(nu, 0, bytes, 32, 16);
        _queue!.WriteBuffer(_freeCfgBuf, 0, bytes);
        DispatchLinear(_freeSpacePrep, pixels, groupSize: 256, entries: new[]
        {
            Buf(0, (mapped ? _depthMaps! : _freeNoMap).GetGPUBuffer()!), Buf(1, _freePx.GetGPUBuffer()!), Buf(2, _freeCfgBuf),
        });
    }

    /// <summary>The backward's binding 17 (call after <see cref="DispatchFreeSpacePrep"/>).</summary>
    GPUBindGroupEntry FreeSpaceEntry() => Buf(17, _freePx!.GetGPUBuffer()!);

    void DisposeFreeSpace()
    {
        _freePx?.Dispose(); _freePx = null;
        _freeNoMap?.Dispose(); _freeNoMap = null;
        _freeCfgBuf?.Destroy(); _freeCfgBuf = null;
        _freeSpacePrep = null;
    }
}
