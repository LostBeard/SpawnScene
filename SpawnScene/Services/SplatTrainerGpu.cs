using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Host side of the differentiable tile rasteriser: buffers, pipelines and dispatch for the
/// kernels in <see cref="SplatTrainerShaders"/>.
///
/// The algorithm these kernels implement is modelled and verified on CPU
/// (<see cref="SplatTileRasterizer"/>), which is in turn checked against the
/// finite-difference-verified <see cref="SplatRasterizer"/>. So the job here is plumbing, and
/// the first gate is simply: does the GPU forward reproduce the CPU model.
///
/// Buffers the sort touches are allocated through ILGPU so the existing radix sort works on
/// them directly, and bound to WGSL via <c>GetGPUBuffer()</c> - the same bridge the display
/// renderer already uses. Readback goes through ILGPU's <c>CopyToHostAsync</c> rather than
/// hand-rolled staging buffers.
/// </summary>
public sealed class SplatTrainerGpu : IDisposable
{
    readonly GpuService _gpu;

    GPUDevice? _device;
    GPUQueue? _queue;

    GPUComputePipeline? _emitKeys;
    GPUComputePipeline? _tileRanges;
    GPUComputePipeline? _rasterForward;
    GPUComputePipeline? _rasterBackward;
    GPUComputePipeline? _scatterGrad;
    GPUComputePipeline? _lossL1;
    GPUComputePipeline? _lossReduce;
    GPUComputePipeline? _adamStep;
    GPUComputePipeline? _initLogits;
    GPUComputePipeline? _adamGeometry;
    GPUComputePipeline? _evalSse;
    GPUComputePipeline? _ssimRowsPipe;
    GPUComputePipeline? _ssimReducePipe;
    GPUComputePipeline? _ssimWinGradPipe;
    GPUComputePipeline? _ssimRowsBwdPipe;
    GPUComputePipeline? _ssimPixBwdPipe;
    GPUComputePipeline? _gradStats;
    GPUComputePipeline? _maxMagnitude;
    GPUComputePipeline? _sampleStride;
    GPUComputePipeline? _densifyAccum;
    GPUComputePipeline? _remapFloatRows;
    GPUComputePipeline? _accumulateSupport;
    GPUComputePipeline? _supportHistogram;
    GPUComputePipeline? _unpackTarget;
    GPUComputePipeline? _initRgbToDc;
    GPUComputePipeline? _scatterShGrad;
    GPUComputePipeline? _adamShRest;

    GPUBuffer? _uniformBuf;     // TrainUniforms
    GPUBuffer? _capsBuf;        // vec4<u32>: key capacity
    GPUBuffer? _countBuf;       // vec4<u32>: key count, for the ranges pass
    byte[]? _uniformBytes;
    Uint32Array? _scratch4;

    // ILGPU-owned so the radix sort and CopyToHostAsync both work on them.
    MemoryBuffer1D<uint, Stride1D.Dense>? _keys;
    MemoryBuffer1D<uint, Stride1D.Dense>? _values;
    MemoryBuffer1D<int, Stride1D.Dense>? _counter;
    MemoryBuffer1D<uint, Stride1D.Dense>? _ranges;      // 2 per tile
    MemoryBuffer1D<float, Stride1D.Dense>? _outColour;  // 3 per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _outFinalT;  // 1 per pixel
    MemoryBuffer1D<uint, Stride1D.Dense>? _outEnd;      // 1 per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _target;      // 3 per pixel
    MemoryBuffer1D<float, Stride1D.Dense>? _dLdPix;      // 3 per pixel
    // Three bindings of three floats per key, not one of nine: maxStorageBufferBindingSize
    // applies per BINDING, so splitting triples how many keys fit. See Resize.
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyA;
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyB;
    MemoryBuffer1D<float, Stride1D.Dense>? _gradKeyC;
    MemoryBuffer1D<int, Stride1D.Dense>? _gradFixed;     // 9 per splat, f32 bit patterns (CAS-summed)
    MemoryBuffer1D<int, Stride1D.Dense>? _densifyAbs;    // 2 per splat: peak |dPx|,|dPy|

    MemoryBuffer1D<float, Stride1D.Dense>? _opacityLogit;
    MemoryBuffer1D<float, Stride1D.Dense>? _logScale;    // 3 per splat
    MemoryBuffer1D<float, Stride1D.Dense>? _geomOut;     // 10 per splat, diagnostics + gate
    MemoryBuffer1D<float, Stride1D.Dense>? _ssePartials; // 1 per 256-pixel workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimRows;    // 5 filtered channels per (window col, row)
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimPartials;// 1 per 256-window workgroup
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimWinGrad; // 5 per valid window (dL/dM)
    MemoryBuffer1D<float, Stride1D.Dense>? _ssimDRows;   // 5 per (window col, row) - row adjoint
    MemoryBuffer1D<float, Stride1D.Dense>? _gradStatsPartials; // 6 per workgroup, 256 workgroups
    MemoryBuffer1D<float, Stride1D.Dense>? _fitSample; // stride gather for the scale histogram
    const int FitSampleCount = 4096;
    MemoryBuffer1D<float, Stride1D.Dense>? _densifyStats;   // 2 per splat: pixel grad sum, visible count
    MemoryBuffer1D<float, Stride1D.Dense>? _screenRadius;   // 1 per splat: this view's 3-sigma radius, px
    MemoryBuffer1D<float, Stride1D.Dense>? _maxRadius;      // 1 per splat: max of that over the window
    MemoryBuffer1D<float, Stride1D.Dense>? _scaleFloor;     // 1 per splat: Mip 3D-filter scale floor (0 = none)
    int _scaleFloorFor = -1;                                // the splat count the floor was computed for
    int _scaleFloorStep;                                    // the Adam step it was computed at
    GPUBuffer? _mipCamsBuf, _mipCfgBuf;
    int _mipCamCount;
    GPUComputePipeline? _mipFloor;

    // ── Partitioned training: only splats inside a volume train (SplatTrainerShaders.FreezeOutside) ──
    GPUComputePipeline? _freezeOutside;
    GPUBuffer? _freezeCfgBuf;

    /// <summary>
    /// When set, only splats inside this volume learn: every other splat renders - so it explains the pixels it
    /// covers - but its gradients are zeroed before the Adam passes, so it never moves and densification never grows
    /// it (Studio.Partition's frozen context). GpuDensify takes the same volume so it never prunes or resets them.
    /// </summary>
    public SplatEditor.Volume? TrainableVolume { get; set; }

    /// <summary>When set, densification clones and splits only splats inside this volume (a partitioned block's own
    /// cell): the rest still train, but growth outside the cell would be dropped at the merge.</summary>
    public SplatEditor.Volume? GrowOnlyInside { get; set; }

    // ── Photometric camera refinement (SplatTrainerShaders.PoseGradPartial / PoseGradFinal) ──
    GPUComputePipeline? _posePartial, _poseFinal;
    MemoryBuffer1D<float, Stride1D.Dense>? _posePartials;   // 6 per workgroup of the last reduction
    MemoryBuffer1D<float, Stride1D.Dense>? _poseGrads;      // 6 per view: dL/d(translation), dL/d(rotation)
    GPUBuffer? _poseCfgBuf, _poseFinalCfgBuf;

    /// <summary>Room for <paramref name="views"/> per-view pose gradients, zeroed.</summary>
    public void EnsurePoseSlots(int views)
    {
        if (_poseGrads == null || _poseGrads.Length < views * 6L)
        {
            _poseGrads?.Dispose();
            _poseGrads = _gpu.WebGPUAccelerator.Allocate1D<float>(Math.Max(1, views) * 6L);
        }
        _poseGrads.MemSetToZero();
    }

    /// <summary>The pose gradients written since the last read (6 per view, zero for a view not stepped), then zero.</summary>
    public async Task<float[]> ReadPoseGradsAsync(int views)
    {
        if (_poseGrads == null) return new float[views * 6];
        // CPU transfer: 24 bytes a view, once a cycle - the host owns the camera poses.
        var g = await _poseGrads.CopyToHostAsync<float>(0, views * 6L);
        _poseGrads.MemSetToZero();
        return g;
    }

    void DispatchPoseGrad(GPUBuffer splatGpu, int splatCount, System.Numerics.Vector3 camCentre, int slot)
    {
        if (_poseGrads == null || _geomOut == null) return;
        int groups = (splatCount + 255) / 256;
        if (_posePartials == null || _posePartials.Length < groups * 6L)
        {
            _posePartials?.Dispose();
            _posePartials = _gpu.WebGPUAccelerator.Allocate1D<float>(groups * 6L);
        }
        _poseCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 16, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        _poseFinalCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor { Size = 16, Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst });
        WriteVec4(_poseCfgBuf, splatCount, camCentre.X, camCentre.Y, camCentre.Z);
        Dispatch(_posePartial!, groups, 1, new[]
        {
            Buf(0, splatGpu), Buf(1, _geomOut.GetGPUBuffer()!), Buf(2, _posePartials.GetGPUBuffer()!), Buf(3, _poseCfgBuf),
        });
        WriteVec4(_poseFinalCfgBuf, groups, slot, 0f, 0f);
        Dispatch(_poseFinal!, 1, 1, new[]
        {
            Buf(0, _posePartials.GetGPUBuffer()!), Buf(1, _poseGrads.GetGPUBuffer()!), Buf(2, _poseFinalCfgBuf),
        });
    }

    /// <summary>
    /// Mip-Splatting 3D filter size (0 = off): each splat's scale is floored at filter x depth / focal for the training
    /// camera that sees it most finely, so no splat is thinner than the photos resolve - the needles and streaks a
    /// viewer shows closer than the photos were taken (&amp;mipfilter=0.2, the paper's value).
    /// </summary>
    public static float MipFilter { get; set; }

    /// <summary>The training cameras the Mip floor is measured against (at the training resolution).</summary>
    public void SetMipCameras(IReadOnlyList<CameraParams> cams)
    {
        _mipCamCount = cams.Count;
        if (cams.Count == 0) return;
        var data = new float[cams.Count * 16];
        for (int c = 0; c < cams.Count; c++)
        {
            var cam = cams[c];
            WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);
            int o = c * 16;
            data[o + 0] = pos.X; data[o + 1] = pos.Y; data[o + 2] = pos.Z; data[o + 3] = cam.FocalX;
            data[o + 4] = right.X; data[o + 5] = right.Y; data[o + 6] = right.Z; data[o + 7] = cam.FocalY;
            data[o + 8] = up.X; data[o + 9] = up.Y; data[o + 10] = up.Z; data[o + 11] = cam.Width * 0.5f;
            data[o + 12] = fwd.X; data[o + 13] = fwd.Y; data[o + 14] = fwd.Z; data[o + 15] = cam.Height * 0.5f;
        }
        _mipCamsBuf?.Destroy();
        _mipCamsBuf = _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)(data.Length * sizeof(float)),
            Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
        });
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        _queue!.WriteBuffer(_mipCamsBuf, 0, bytes);
        _scaleFloorFor = -1;
    }

    /// <summary>Recompute the Mip floor when the splats changed (count) or every 500 steps (they move).</summary>
    void UpdateMipFloor(GPUBuffer splatGpu, int splatCount)
    {
        if (MipFilter <= 0f || _mipCamCount == 0 || _mipCamsBuf == null || _scaleFloor == null) return;
        if (_scaleFloorFor == splatCount && _adamStepCount - _scaleFloorStep < 500) return;
        _mipCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        WriteVec4(_mipCfgBuf, splatCount, _mipCamCount, MipFilter, 0f);
        DispatchLinear(_mipFloor!, splatCount, new[]
        {
            Buf(0, splatGpu), Buf(1, _mipCamsBuf), Buf(2, _scaleFloor.GetGPUBuffer()!), Buf(3, _mipCfgBuf),
        });
        _scaleFloorFor = splatCount;
        _scaleFloorStep = _adamStepCount;
    }
    MemoryBuffer1D<uint, Stride1D.Dense>? _viewSupport;        // views that ever moved each splat
    MemoryBuffer1D<float, Stride1D.Dense>? _supportPartials;   // 6 per workgroup, 256 workgroups
    MemoryBuffer1D<float, Stride1D.Dense>? _adamM;
    MemoryBuffer1D<float, Stride1D.Dense>? _adamV;
    // SH rest bands, their gradients and moments: one buffer per SphericalHarmonics part (5 bands each). A single
    // 45-float buffer capped one storage binding at 11.9M splats; a part holds 35.8M.
    MemoryBuffer1D<float, Stride1D.Dense>?[] _shRest = new MemoryBuffer1D<float, Stride1D.Dense>?[SphericalHarmonics.Parts];
    MemoryBuffer1D<int, Stride1D.Dense>?[] _gradShRest = new MemoryBuffer1D<int, Stride1D.Dense>?[SphericalHarmonics.Parts];
    // This view's display RGB per splat (3 floats), written by emit_keys and read by both raster passes.
    MemoryBuffer1D<float, Stride1D.Dense>? _splatColour;
    // SH Adam moments in bfloat16, two to a word (Bf16, SplatTrainerShaders.AdamShRest), per part: 8 words a splat for
    // 15 floats. They were 360 of the ~1,050 training bytes a splat.
    MemoryBuffer1D<uint, Stride1D.Dense>?[] _adamShM = new MemoryBuffer1D<uint, Stride1D.Dense>?[SphericalHarmonics.Parts];
    MemoryBuffer1D<uint, Stride1D.Dense>?[] _adamShV = new MemoryBuffer1D<uint, Stride1D.Dense>?[SphericalHarmonics.Parts];
    // Sum of per-step losses since the last read (float, LossReduce) and the per-workgroup partial sums of one step.
    MemoryBuffer1D<float, Stride1D.Dense>? _lossSum;
    MemoryBuffer1D<float, Stride1D.Dense>? _lossPartial;
    GPUBuffer? _lossDimsBuf;
    GPUBuffer? _dimsBuf;
    GPUBuffer? _ssimDimsBuf;
    GPUBuffer? _ssimCfgBuf;
    GPUBuffer? _lossWeightsBuf;
    GPUBuffer? _adamFlagsBuf;
    GPUBuffer? _adamCfgBuf;
    GPUBuffer? _geomCfgBuf;
    GPUBuffer? _targetBytes;   // one frame of packed RGBA, straight from the canvas
    int _adamStepCount;

    /// <summary>32-bit words of one splat's SH Adam moment row in ONE part (15 bfloat16 values, two a word).</summary>
    public static readonly int ShMomentWords = Bf16.WordsPerRow(SphericalHarmonics.PartFloatsPerSplat);

    /// <summary>Grow the key buffers when a frame needs more keys than they hold (see RenderForwardAsync). On by default.</summary>
    public bool GrowKeysOnOverflow { get; set; } = true;

    /// <summary>Key-buffer growths since the last <see cref="ResetPeakKeyDemand"/>.</summary>
    public int KeyGrowths { get; private set; }

    /// <summary>
    /// Reallocate the key-indexed buffers (keys, values, the three per-key gradient bindings) for
    /// <paramref name="needed"/> keys plus 25% headroom, within the per-binding and total key limits. Their contents
    /// are per frame, so nothing is carried. False when even <paramref name="needed"/> does not fit the limits.
    /// </summary>
    bool TryGrowKeys(int needed)
    {
        long maxKeys = Math.Min(MaxStorageBindingBytes() / (3 * sizeof(float)), MaxTotalKeys);
        if (needed > maxKeys) return false;
        int capacity = (int)Math.Min(maxKeys, (long)needed + needed / 4);
        var accel = _gpu.WebGPUAccelerator;
        _keys?.Dispose(); _values?.Dispose(); _gradKeyA?.Dispose(); _gradKeyB?.Dispose(); _gradKeyC?.Dispose();
        _keys = accel.Allocate1D<uint>(capacity);
        _values = accel.Allocate1D<uint>(capacity);
        _gradKeyA = accel.Allocate1D<float>((long)capacity * 3);
        _gradKeyB = accel.Allocate1D<float>((long)capacity * 3);
        _gradKeyC = accel.Allocate1D<float>((long)capacity * 3);
        Console.WriteLine($"[Trainer] key buffers grown {_keyCapacity:N0} -> {capacity:N0} for a frame that needed {needed:N0}");
        _keyCapacity = capacity;
        KeyGrowths++;
        return true;
    }

    /// <summary>
    /// The trainer's GPU memory by group, in MB: what each splat (and key, and pixel) actually costs, so a memory cut aims
    /// at the largest group (GpuMemoryBudget.BytesPerSplat is the total of the per-splat and per-key groups).
    /// </summary>
    public string MemoryBreakdown()
    {
        static long B(params MemoryBuffer?[] buffers) => buffers.Sum(b => b?.LengthInBytes ?? 0);
        long keys = B(_keys, _values, _gradKeyA, _gradKeyB, _gradKeyC);
        long pixels = B(_outColour, _outFinalT, _outEnd, _target, _dLdPix, _ssimRows, _ssimWinGrad, _ssimDRows, _lossPartial);
        long splat = B(_gradFixed, _densifyAbs, _opacityLogit, _logScale, _geomOut, _densifyStats, _screenRadius, _maxRadius, _scaleFloor,
            _viewSupport, _splatColour);
        long adam = B(_adamM, _adamV);
        long sh = B([.. _shRest, .. _gradShRest, .. _adamShM, .. _adamShV]);
        static string Mb(long b) => $"{b / 1048576.0:F0}";
        return $"keys {Mb(keys)} ({_keyCapacity:N0}), per-splat {Mb(splat)}, Adam {Mb(adam)}, " +
            $"SH {Mb(sh)}, per-pixel {Mb(pixels)} MB";
    }

    /// <summary>Gradient slots per splat. Must match GRADS_PER_SPLAT in the shaders.</summary>
    public const int GradsPerSplat = SplatTileRasterizer.GradsPerKey;

    /// <summary>Adam moment slots per splat: 3 colour, 1 opacity, 3 position, 3 scale, 4 quaternion.</summary>
    public const int AdamSlots = 14;

    /// <summary>Log each operation of a densify carry before it runs (&amp;tracecarry=1, device-loss diagnosis).</summary>
    public static bool TraceCarrySteps { get; set; }
    /// <summary>Adam layout: RGB(0..2), opacity(3), pos(4..6), log-scale(7..9), quat(10..13).</summary>
    const int AdamOpacitySlot = 3;

    /// <summary>Geometry gradients reported per splat: position xyz, scale xyz, quaternion xyzw.</summary>
    public const int GeomGradsPerSplat = 10;

    // The gradient accumulator (_gradFixed, 9 per splat) holds IEEE f32 bit patterns summed by
    // a compare-exchange loop in scatter_gradients. There is NO fixed-point scale any more.
    //
    // History, so nobody reintroduces one: the i32 fixed-point accumulator needed one scale per
    // slot, and no scale worked for the conic. Too coarse and it rounded to zero (no scale or
    // rotation learning - init-sized soft blobs). Too fine and ~200 keys per splat wrapped the
    // i32 (wrong-sign steps). The per-key +-1e5 clamp that stopped the wrap saturated typical
    // conic keys, which turns each conic component into a sign count and hands adam_geometry a
    // distorted a:b:c ratio to chain into scale and rotation. Twelve Truck 2K gates on
    // 2026-09-22 were spent fitting that scale; supervised PSNR never passed ~17 dB.

    /// <summary>
    /// Linear-scale p10/median/p90 from log-scale params (GPU stride sample). Geometry learning
    /// proof: median must shrink vs init (~0.017 on Truck points) once conic grads live.
    /// </summary>
    public async Task<(float P10, float Median, float P90)> ReadScaleHistogramAsync(int splatCount)
    {
        if (_logScale == null || splatCount <= 0 || _fitSample == null || _sampleStride == null)
            return (0, 0, 0);
        long count = (long)splatCount * 3;
        int n = (int)Math.Min(FitSampleCount, count);
        int stride = Math.Max(1, (int)(count / n));
        n = (int)Math.Min(FitSampleCount, (count + stride - 1) / stride);
        WriteU32x4(_dimsBuf!, (uint)n, (uint)stride, (uint)Math.Min(count, _logScale.Length), 0);
        DispatchLinear(_sampleStride, n, groupSize: 256, entries: new[]
        {
            Buf(0, _logScale.GetGPUBuffer()!), Buf(1, _fitSample.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // CPU transfer: scale histogram for train logs (gate: median vs init).
        float[] host = await _fitSample.CopyToHostAsync<float>(0, n);
        var lin = new float[n];
        for (int i = 0; i < n; i++) lin[i] = MathF.Exp(host[i]);
        System.Array.Sort(lin);
        float At(double p) => lin[(int)Math.Clamp(Math.Round((n - 1) * p), 0, n - 1)];
        return (At(0.10), At(0.50), At(0.90));
    }

    GpuRadixSort? _radixSort;

    int _width, _height, _tilesX, _tilesY, _keyCapacity;

    /// <summary>
    /// Keys per splat actually budgeted, which may be lower than requested: the ceiling is the
    /// storage BINDING size limit, not free memory. Reported so a log line cannot claim a
    /// budget that was not used.
    /// </summary>
    public int KeysPerSplat { get; private set; }

    /// <summary>Keys emitted by the last render. Exceeding capacity is reported, never silent.</summary>
    public int LastKeyCount { get; private set; }

    /// <summary>
    /// How many training steps the last non-NaN <see cref="TrainStepAsync"/> loss is the mean of. The loss
    /// accumulates on the GPU across steps that do not read it, so a caller summing a cycle's loss weights
    /// the returned mean by this.
    /// </summary>
    public int LastLossSteps { get; private set; }

    /// <summary>
    /// Diagnostic (off by default, <c>&amp;trainprofile=1</c>): wait for the GPU after each phase of a step and
    /// accumulate its wall time, so a slow step names the phase. The waits it adds make the step slower -
    /// the split is the measurement, not the total.
    /// </summary>
    public bool ProfilePhases { get; set; }
    readonly Dictionary<string, double> _phaseMs = new();
    int _phaseSteps;
    readonly System.Diagnostics.Stopwatch _phaseClock = new();

    async Task PhaseAsync(string name)
    {
        if (!ProfilePhases) return;
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        _phaseMs[name] = _phaseMs.GetValueOrDefault(name) + _phaseClock.Elapsed.TotalMilliseconds;
        _phaseClock.Restart();
    }

    /// <summary>Mean ms per step for each phase since the last call, then reset. Empty when not profiling.</summary>
    public string TakePhaseProfile()
    {
        if (_phaseSteps == 0) return "";
        double total = _phaseMs.Values.Sum();
        var parts = _phaseMs.Select(kv => $"{kv.Key} {kv.Value / _phaseSteps:F1}");
        string line = $"{total / _phaseSteps:F1} ms/step = " + string.Join(", ", parts);
        _phaseMs.Clear(); _phaseSteps = 0;
        return line;
    }

    // Steps added into _lossSum since it was last read. 0 = the buffer must be cleared before the next add.
    int _lossStepsPending;

    // The accumulator is a float sum of per-step losses (LossReduce). Reading at least this often keeps it a sum of
    // similar magnitudes.
    const int MaxLossStepsBetweenReads = 1024;

    /// <summary>True when the last render overflowed the key buffer and is therefore incomplete.</summary>
    public bool LastOverflowed { get; private set; }

    /// <summary>
    /// Keys the last render actually WANTED, which exceeds <see cref="LastKeyCount"/> when it
    /// overflowed. Lets a caller size the budget from a measurement instead of a guess: how
    /// many tiles a splat covers depends on the scene, and a room is not a turntable.
    /// </summary>
    public int LastKeyDemand { get; private set; }

    /// <summary>Active SH degree 0..3, sent in <c>TrainUniforms.sh_degree</c>.</summary>
    public int ActiveShDegree { get; set; }

    bool _rgbToDcDone;

    public SplatTrainerGpu(GpuService gpu) => _gpu = gpu;

    const int UniformFloats = 28;   // 112 bytes: 4 vec4 + 3 vec2 + 2 u32 + 2 f32 + u32 + pad

    public void Initialize()
    {
        var accel = _gpu.WebGPUAccelerator;
        var native = accel.NativeAccelerator;
        _device = native.NativeDevice ?? throw new InvalidOperationException("no WebGPU device");
        _queue = native.Queue ?? throw new InvalidOperationException("no WebGPU queue");

        _emitKeys = MakePipeline(SplatTrainerShaders.EmitKeys, "emit_keys");
        _tileRanges = MakePipeline(SplatTrainerShaders.TileRanges, "tile_ranges");
        _rasterForward = MakePipeline(SplatTrainerShaders.RasterForward, "raster_forward");
        _rasterBackward = MakePipeline(SplatTrainerShaders.RasterBackward, "raster_backward");
        _scatterGrad = MakePipeline(SplatTrainerShaders.ScatterGradients, "scatter_gradients");
        _lossL1 = MakePipeline(SplatTrainerShaders.LossL1, "loss_l1");
        _lossReduce = MakePipeline(SplatTrainerShaders.LossReduce, "loss_reduce");
        _adamStep = MakePipeline(SplatTrainerShaders.AdamStep, "adam_step");
        _initLogits = MakePipeline(SplatTrainerShaders.InitLogits, "init_logits");
        _adamGeometry = MakePipeline(SplatTrainerShaders.GeometryAdam, "adam_geometry");
        _mipFloor = MakePipeline(SplatTrainerShaders.MipScaleFloor, "mip_floor");
        _freezeOutside = MakePipeline(SplatTrainerShaders.FreezeOutside, "freeze_outside");
        _posePartial = MakePipeline(SplatTrainerShaders.PoseGradPartial, "pose_partial");
        _poseFinal = MakePipeline(SplatTrainerShaders.PoseGradFinal, "pose_final");
        _evalSse = MakePipeline(SplatTrainerShaders.EvalSse, "eval_sse");
        _ssimRowsPipe = MakePipeline(SplatTrainerShaders.SsimRows, "ssim_rows");
        _ssimReducePipe = MakePipeline(SplatTrainerShaders.SsimReduce, "ssim_reduce");
        _ssimWinGradPipe = MakePipeline(SplatTrainerShaders.SsimWinGrad, "ssim_win_grad");
        _ssimRowsBwdPipe = MakePipeline(SplatTrainerShaders.SsimRowsBwd, "ssim_rows_bwd");
        _ssimPixBwdPipe = MakePipeline(SplatTrainerShaders.SsimPixBwd, "ssim_pix_bwd");
        _gradStats = MakePipeline(SplatTrainerShaders.GradStats, "grad_stats");
        _maxMagnitude = MakePipeline(SplatTrainerShaders.MaxMagnitude, "max_magnitude");
        _sampleStride = MakePipeline(SplatTrainerShaders.SampleStride, "sample_stride");
        _densifyAccum = MakePipeline(SplatTrainerShaders.DensifyAccum, "densify_accum");
        _remapFloatRows = MakePipeline(SplatTrainerShaders.RemapFloatRows, "remap_float_rows");
        _accumulateSupport = MakePipeline(SplatTrainerShaders.AccumulateSupport, "accumulate_support");
        _supportHistogram = MakePipeline(SplatTrainerShaders.SupportHistogram, "support_histogram");
        _unpackTarget = MakePipeline(SplatTrainerShaders.UnpackTarget, "unpack_target");
        _initRgbToDc = MakePipeline(SplatTrainerShaders.InitRgbToDc, "init_rgb_to_dc");
        _scatterShGrad = MakePipeline(SplatTrainerShaders.ScatterShGrad, "scatter_sh_grad");
        _adamShRest = MakePipeline(SplatTrainerShaders.AdamShRest, "adam_sh_rest");

        _uniformBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = UniformFloats * sizeof(float),
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _capsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _countBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _uniformBytes = new byte[UniformFloats * sizeof(float)];
        _scratch4 = new Uint32Array(4);
        _dimsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _adamCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        // Separate from _adamCfgBuf rather than widening it: growing a uniform without updating
        // its allocation gives the driver's unhelpful "[Invalid CommandBuffer] ... previous
        // error", reported from whichever dispatch runs next rather than from the guilty one.
        _adamFlagsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _geomCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 32,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _ssimDimsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        // SsimCfg: vec4 luma + vec4 consts + array<vec4,3> weights = 80 bytes.
        _ssimCfgBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 80,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _lossWeightsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _lossDimsBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        Console.WriteLine("[Trainer] pipelines created: emit_keys, tile_ranges, raster_forward, " +
            "raster_backward, scatter_gradients, loss_l1, adam_step, init_logits, adam_geometry, " +
            "eval_sse, unpack_target, ssim_rows, ssim_reduce, ssim_win_grad, ssim_rows_bwd, " +
            "ssim_pix_bwd, grad_stats");
    }

    GPUComputePipeline MakePipeline(string wgsl, string entry)
    {
        using var module = _device!.CreateShaderModule(new GPUShaderModuleDescriptor { Code = wgsl });
        return _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage { Module = module, EntryPoint = entry },
        });
    }

    /// <summary>
    /// Size buffers for a viewport and splat count.
    /// <paramref name="keysPerSplat"/> is the worst-case tile overlap budget; the emit kernel
    /// refuses to write past capacity and the host reports the overflow rather than rendering
    /// a silently truncated scene.
    /// </summary>
    /// <summary>
    /// The most splats this trainer can hold at a given key budget.
    ///
    /// Densification must not grow past it. Resize clamps keysPerSplat downwards to fit, and a
    /// clamped budget overflows - so growing beyond this does not buy splats, it buys frames
    /// trained on incomplete renders.
    /// </summary>
    /// <summary>Total key capacity allowed across all key-indexed buffers (~52 bytes a key). See ResizeCoreAsync.</summary>
    public static long MaxTotalKeys { get; set; } = 40_000_000;

    public int MaxTrainableSplats(int keysPerSplat) =>
        (int)Math.Max(1, MaxStorageBindingBytes() / (3 * sizeof(float)) / Math.Max(1, keysPerSplat));

    long _maxBindingBytes;

    /// <summary>
    /// The device's real <c>maxStorageBufferBindingSize</c>, not the spec minimum.
    ///
    /// 128 MiB is what WebGPU GUARANTEES; it is not what hardware provides. Sizing to the
    /// guarantee is the right default for a shader that must behave identically everywhere, but
    /// here it is a hard ceiling on the splat count: at the ~50 keys per splat this scene
    /// actually demands, 128 MiB stops the reconstruction near 220,000 splats while the
    /// reference finishes a room in the millions. That is the guarantee deciding the quality.
    ///
    /// So ask. The guarantee stays the floor, so a device that reports something smaller or
    /// nothing at all behaves exactly as before.
    /// </summary>
    long MaxStorageBindingBytes()
    {
        const long Guaranteed = 128L * 1024 * 1024;
        if (_maxBindingBytes > 0) return _maxBindingBytes;
        // A device that will not answer keeps the guarantee (GpuMemoryBudget.ReadMaxStorageBindingBytes never throws).
        _maxBindingBytes = GpuMemoryBudget.ReadMaxStorageBindingBytes(_device);
        if (_maxBindingBytes > Guaranteed)
            Console.WriteLine(
                $"[Trainer] device maxStorageBufferBindingSize is " +
                $"{_maxBindingBytes / (1024 * 1024)} MiB, not the {Guaranteed / (1024 * 1024)} MiB " +
                $"guarantee - key capacity scales with it");
        return _maxBindingBytes;
    }

    /// <summary>
    /// Size every per-frame and per-splat buffer, waiting on the device after each group of
    /// allocations so a device loss is attributed to the group that caused it. MEASURED
    /// (dj2k-dav3-pose run 4, 2026-09-23): the second sizing at 913k splats lost the device with
    /// reason "unknown" and no validation error, somewhere between the first allocation and the
    /// sync after the last; the first sizing at 920k on a fresh trainer did not. Which group is
    /// the one that matters is what these stages answer.
    /// </summary>
    public async Task ResizeAsync(int width, int height, int splatCount, int keysPerSplat = 8)
    {
        var accel = _gpu.WebGPUAccelerator;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var stages = new List<string>();
        string current = "start";
        try
        {
            await ResizeCoreAsync(width, height, splatCount, keysPerSplat, async name =>
            {
                current = name;
                await accel.SynchronizeAsync();
                stages.Add($"{name} {clock.ElapsedMilliseconds}ms");
            });
        }
        catch (Exception ex)
        {
            Console.WriteLine(
                $"[Trainer] resize to {splatCount:N0} FAILED in stage '{current}' after " +
                $"[{string.Join(", ", stages)}]: {ex.Message}");
            throw;
        }
        Console.WriteLine($"[Trainer] sized {width}x{height} = {_tilesX}x{_tilesY} tiles, " +
            $"{splatCount:N0} splats, key capacity {_keyCapacity:N0} [{string.Join(", ", stages)}]");
    }

    async Task ResizeCoreAsync(int width, int height, int splatCount, int keysPerSplat, Func<string, Task> stage)
    {
        var accel = _gpu.WebGPUAccelerator;

        _width = width;
        _height = height;
        _tilesX = (width + SplatTrainerShaders.TileSize - 1) / SplatTrainerShaders.TileSize;
        _tilesY = (height + SplatTrainerShaders.TileSize - 1) / SplatTrainerShaders.TileSize;

        // The key packs the tile id above 18 depth bits, so the tile count has a hard ceiling.
        int tileCount = _tilesX * _tilesY;
        if (tileCount >= (1 << 14))
            throw new InvalidOperationException(
                $"{tileCount} tiles exceeds the {1 << 14} the 18-bit depth key leaves room for " +
                $"({width}x{height}). Reduce the training resolution or widen the key.");

        // The key budget is bounded by what a single storage BINDING may be, not by free VRAM.
        // The nine per-key gradients are split across three bindings of three floats, so the
        // widest key-indexed binding is 12 bytes rather than 36. As one buffer, 580k splats at
        // 8 keys each came to 159 MiB against a guaranteed 128 MiB, and the driver rejected the
        // bind group with "[Invalid CommandBuffer] is invalid due to a previous error" - naming
        // neither the buffer nor the limit, and surfacing from whichever dispatch ran next.
        //
        // 128 MiB is the WebGPU guaranteed minimum for maxStorageBufferBindingSize. Sizing to
        // the guarantee rather than querying means this behaves the same on every device.
        long MaxBindingBytes = MaxStorageBindingBytes();
        const long bytesPerKey = 3 * sizeof(float);   // the widest single key-indexed binding
        long maxKeys = MaxBindingBytes / bytesPerKey;

        // Clamp DOWN to what the binding allows, and no further.
        //
        // I changed this to also grow keysPerSplat up to the affordable maximum, on the theory
        // that leaving capacity unused was causing the overflow I saw. That was WRONG twice over.
        // The overflow self-corrects: the caller measures actual peak demand after the first
        // cycle and re-sizes with headroom ("peak demand 5,991,085 keys for 725,452 splats
        // (8.3 per splat); re-sizing to 11 per splat with 25% headroom"), so exactly one frame
        // is incomplete, not the run. And taking the maximum instead allocated 10.9M keys and
        // lost the device outright.
        //
        // Measured demand plus headroom is the right rule. "Use everything available" is not a
        // budget either - it is the same mistake as a fixed default, pointing the other way.
        int requested = keysPerSplat;
        if ((long)splatCount * keysPerSplat > maxKeys)
        {
            keysPerSplat = Math.Max(1, (int)(maxKeys / Math.Max(1, splatCount)));
            Console.WriteLine(
                $"[Trainer] keysPerSplat {requested} -> {keysPerSplat}: {splatCount:N0} splats " +
                $"would need {(long)splatCount * requested * bytesPerKey / (1024 * 1024)} MiB for " +
                $"one binding, over the {MaxBindingBytes / (1024 * 1024)} MiB this device allows. " +
                "A tighter budget can overflow, which is reported per frame, not hidden.");
        }

        // And a TOTAL budget. Each binding fitting is not the device fitting: key-indexed buffers are ~52 bytes a
        // key across keys, values, sort scratch and the three gradient bindings. MEASURED 2026-09-24: 673k splats
        // at 115 keys each (77.4M keys, ~4 GB) lost the device inside Resize; 2.0M splats at 17 (34M keys) ran.
        if ((long)splatCount * keysPerSplat > MaxTotalKeys)
        {
            int before = keysPerSplat;
            keysPerSplat = Math.Max(1, (int)(MaxTotalKeys / Math.Max(1, splatCount)));
            Console.WriteLine(
                $"[Trainer] keysPerSplat {before} -> {keysPerSplat}: {splatCount:N0} splats would need " +
                $"{(long)splatCount * before:N0} keys, over the {MaxTotalKeys:N0}-key device budget. Frames over it " +
                "overflow (reported); the next window re-measures demand.");
        }

        KeysPerSplat = keysPerSplat;
        _keyCapacity = Math.Max(1024, splatCount * keysPerSplat);
        if (_keyCapacity > maxKeys)
            throw new InvalidOperationException(
                $"{splatCount:N0} splats cannot be trained: even one key each needs " +
                $"{_keyCapacity * bytesPerKey / (1024 * 1024)} MiB for a single binding. " +
                "Reduce the splat count or the training resolution.");

        // Rows already carried to this count by CarryOptimizerRowsAsync stay; anything else
        // sized for the old count goes. A carry for a DIFFERENT count is a caller bug, not
        // something to paper over by reallocating - it would silently drop the momentum.
        bool carried = _carriedRowsFor == splatCount;
        if (_carriedRowsFor >= 0 && !carried)
            throw new InvalidOperationException(
                $"optimizer rows were carried to {_carriedRowsFor:N0} splats but Resize was " +
                $"called for {splatCount:N0}");
        _carriedRowsFor = -1;
        DisposeBuffers(keepOptimizerRows: carried);
        await stage("released");
        _keys = accel.Allocate1D<uint>(_keyCapacity);
        _values = accel.Allocate1D<uint>(_keyCapacity);
        _counter = accel.Allocate1D<int>(1);
        _ranges = accel.Allocate1D<uint>(tileCount * 2);
        _outColour = accel.Allocate1D<float>((long)width * height * 3);
        _outFinalT = accel.Allocate1D<float>((long)width * height);
        _outEnd = accel.Allocate1D<uint>((long)width * height);
        await stage("keys+frame");

        _target = accel.Allocate1D<float>((long)width * height * 3);
        _targetBytes?.Destroy(); _targetBytes?.Dispose();
        _targetBytes = _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)width * (ulong)height * 4UL,
            Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
        });
        _dLdPix = accel.Allocate1D<float>((long)width * height * 3);
        _lossPartial = accel.Allocate1D<float>(Math.Max(1L, ((long)width * height + 63) / 64));
        _gradKeyA = accel.Allocate1D<float>((long)_keyCapacity * 3);
        _gradKeyB = accel.Allocate1D<float>((long)_keyCapacity * 3);
        _gradKeyC = accel.Allocate1D<float>((long)_keyCapacity * 3);
        await stage("target+gradKeys");
        _gradFixed = accel.Allocate1D<int>((long)splatCount * GradsPerSplat);
        _densifyAbs = accel.Allocate1D<int>((long)splatCount * 2);
        _opacityLogit = accel.Allocate1D<float>(splatCount);
        _logScale = accel.Allocate1D<float>((long)splatCount * 3);
        _geomOut = accel.Allocate1D<float>((long)splatCount * GeomGradsPerSplat);
        _ssePartials = accel.Allocate1D<float>(SseWorkgroups);
        _gradStatsPartials = accel.Allocate1D<float>(GradStatsWorkgroups * GradStatsSlots);
        _fitSample = accel.Allocate1D<float>(FitSampleCount);
        _densifyStats = accel.Allocate1D<float>((long)splatCount * 2);
        _screenRadius = accel.Allocate1D<float>(splatCount);
        _splatColour = accel.Allocate1D<float>((long)splatCount * 3);
        _maxRadius = accel.Allocate1D<float>(splatCount);
        _maxRadius.MemSetToZero();
        _scaleFloor = accel.Allocate1D<float>(splatCount);
        _scaleFloor.MemSetToZero();
        _scaleFloorFor = -1;
        _viewSupport = accel.Allocate1D<uint>(splatCount);
        _supportPartials = accel.Allocate1D<float>(SupportWorkgroups * SupportSlots);
        await stage("per-splat");

        // SSIM works on 'valid' windows, so a viewport smaller than the window has none and the
        // metric is genuinely undefined there - the Python oracle raises rather than inventing a
        // number, and ScoreAgainstAsync reports NaN for the same reason.
        if (HasSsimWindows)
        {
            _ssimRows = accel.Allocate1D<float>((long)SsimWindowsX * _height * 5);
            _ssimPartials = accel.Allocate1D<float>(SsimWorkgroups);
            _ssimWinGrad = accel.Allocate1D<float>((long)SsimWindowsX * SsimWindowsY * 5);
            _ssimDRows = accel.Allocate1D<float>((long)SsimWindowsX * _height * 5);
            WriteSsimCfg();
        }
        await stage("ssim");
        for (int part = 0; part < SphericalHarmonics.Parts; part++)
            _gradShRest[part] = accel.Allocate1D<int>((long)splatCount * SphericalHarmonics.PartFloatsPerSplat);
        await stage("gradShRest");
        if (!carried)
        {
            _adamM = accel.Allocate1D<float>((long)splatCount * AdamSlots);
            _adamV = accel.Allocate1D<float>((long)splatCount * AdamSlots);
            for (int part = 0; part < SphericalHarmonics.Parts; part++)
            {
                _shRest[part] = accel.Allocate1D<float>((long)splatCount * SphericalHarmonics.PartFloatsPerSplat);
                _adamShM[part] = accel.Allocate1D<uint>((long)splatCount * ShMomentWords);
                _adamShV[part] = accel.Allocate1D<uint>((long)splatCount * ShMomentWords);
                _shRest[part]!.MemSetToZero();
                _adamShM[part]!.MemSetToZero();
                _adamShV[part]!.MemSetToZero();
            }
            _adamStepCount = 0;
            await stage("adam+sh");
        }
        _lossSum = accel.Allocate1D<float>(1);
        _lossStepsPending = 0;

        _radixSort ??= new GpuRadixSort(_device!, _queue!, _gpu.WebGPUAccelerator);
        _radixSort.EnsureCapacity(_keyCapacity);
        await stage($"sort scratch for {_keyCapacity:N0} keys");
    }

    /// <summary>
    /// Run the forward pass: emit keys, sort, find tile ranges, rasterise.
    /// Returns the rendered RGB (3 floats per pixel).
    /// </summary>
    public async Task<float[]> RenderForwardAsync(
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar, bool readback = true)
    {
        if (_device == null || _emitKeys == null) throw new InvalidOperationException("not initialized");

        var accel = _gpu.WebGPUAccelerator;
        var splatGpu = splatBuf.GetGPUBuffer() ?? throw new InvalidOperationException("splat buffer has no GPU handle");

        WriteUniforms(cam, depthNear, depthFar, splatCount);
        WriteU32(_capsBuf!, (uint)_keyCapacity);
        await PhaseAsync("prev");

        // Clear the counter and the tile ranges. Tiles with no keys are never written by the
        // ranges kernel, so stale values from a previous frame would be read as real spans.
        // Also clear colour / T / end_idx: a view that paints nothing must not leave the
        // previous view's image in the loss (MEASURED Truck STAGE PROBE: mean|rgb| live,
        // max|gradPerKey|==0 on a subset of views - stale colour + consumed=0 fits that split).
        // Clear keys/values too: overflow frames write only a prefix; the rest used to keep
        // the previous view's splat ids, and sort then fed mixed lists into raster.
        _counter!.MemSetToZero();
        _ranges!.MemSetToZero();
        _outColour!.MemSetToZero();
        _outFinalT!.MemSetToZero();
        _outEnd!.MemSetToZero();
        _keys!.MemSetToZero();
        _values!.MemSetToZero();
        // Submit, do not wait: the clears are recorded in ILGPU's pending encoder, the passes below go
        // straight to the queue, and a WebGPU queue runs submissions in order. Only a CPU read needs a wait.
        accel.FlushPendingCommands();
        await PhaseAsync("clear");

        // ── 1. Emit (tile, depth) keys ──
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_emitKeys);
            using var layout = _emitKeys.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuf! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = splatGpu } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _keys!.GetGPUBuffer()! } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _values!.GetGPUBuffer()! } },
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = _counter!.GetGPUBuffer()! } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _capsBuf! } },
                    new() { Binding = 6, Resource = new GPUBufferBinding { Buffer = _screenRadius!.GetGPUBuffer()! } },
                    new() { Binding = 7, Resource = new GPUBufferBinding { Buffer = _splatColour!.GetGPUBuffer()! } },
                    ShRestBindEntry(12, 0), ShRestBindEntry(13, 1), ShRestBindEntry(14, 2),
                },
            });
            pass.SetBindGroup(0, bg);
            var (ekX, ekY) = LinearGrid(splatCount);
            pass.DispatchWorkgroups((uint)ekX, (uint)ekY, 1);
            pass.End();
            using var cmd = enc.Finish();
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, new[] { cmd });
        }

        // 4 bytes back to learn how many keys exist. A scalar, not bulk data. The readback maps behind
        // everything already submitted, so it needs no separate wait.
        int[] counted = await _counter.CopyToHostAsync<int>(0, 1);
        await PhaseAsync("emit+count");
        int keyCount = counted[0];
        LastKeyDemand = keyCount;
        PeakKeyDemand = Math.Max(PeakKeyDemand, keyCount);
        // More keys than the buffers hold: grow them and emit again, so this frame's gradients are complete. The
        // capacity is sized from measured demand per densify window, and that window does not see every view
        // (TruckFull: ~100 of 219 per window), so a view busier than the window's peak lands here. Growing used to be
        // impossible, which is why capacity kept a floor of 8 keys a splat (~350 B/splat, 26% of training memory).
        if (keyCount > _keyCapacity && GrowKeysOnOverflow && TryGrowKeys(keyCount))
            return await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback);
        LastOverflowed = keyCount > _keyCapacity;
        LastKeyCount = Math.Min(keyCount, _keyCapacity);
        if (LastOverflowed)
        {
            Console.WriteLine($"[Trainer] KEY OVERFLOW: {keyCount:N0} needed, capacity {_keyCapacity:N0}. " +
                "Raise keysPerSplat; this frame is incomplete.");
        }
        if (LastKeyCount == 0)
        {
            Console.WriteLine("[Trainer] no keys emitted - nothing visible from this view");
            return readback ? new float[_width * _height * 3] : System.Array.Empty<float>();
        }

        // ── 2. Sort by key: groups by tile AND orders front-to-back within each tile ──
        // 8 bits a pass over only the bits the key uses (tile index above DEPTH_BITS of depth), all passes
        // in one submission. The ILGPU.Algorithms sort this replaces ran 16 two-bit passes, ~100 dispatches,
        // and was 60-67% of a training step (MEASURED, &trainprofile=1).
        int keyBits = 18 + Math.Max(1, System.Numerics.BitOperations.Log2((uint)Math.Max(1, _tilesX * _tilesY - 1)) + 1);
        _radixSort!.Sort(_keys!.GetGPUBuffer()!, _values!.GetGPUBuffer()!, LastKeyCount, Math.Min(32, keyBits));
        accel.FlushPendingCommands();
        await PhaseAsync("sort");

        // ── 3. Tile ranges ──
        WriteU32(_countBuf!, (uint)LastKeyCount);
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_tileRanges!);
            using var layout = _tileRanges!.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _keys!.GetGPUBuffer()! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = _ranges!.GetGPUBuffer()! } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _countBuf! } },
                },
            });
            pass.SetBindGroup(0, bg);
            var (trX, trY) = LinearGrid(LastKeyCount);
            pass.DispatchWorkgroups((uint)trX, (uint)trY, 1);
            pass.End();
            using var cmd = enc.Finish();
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, new[] { cmd });
        }

        // ── 4. Rasterise: one workgroup per tile ──
        using (var enc = _device.CreateCommandEncoder())
        {
            using var pass = enc.BeginComputePass();
            pass.SetPipeline(_rasterForward!);
            using var layout = _rasterForward!.GetBindGroupLayout(0);
            using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuf! } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = splatGpu } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _ranges!.GetGPUBuffer()! } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _values!.GetGPUBuffer()! } },
                    new() { Binding = 4, Resource = new GPUBufferBinding { Buffer = _outColour!.GetGPUBuffer()! } },
                    new() { Binding = 5, Resource = new GPUBufferBinding { Buffer = _outFinalT!.GetGPUBuffer()! } },
                    new() { Binding = 6, Resource = new GPUBufferBinding { Buffer = _outEnd!.GetGPUBuffer()! } },
                    new() { Binding = 11, Resource = new GPUBufferBinding { Buffer = _splatColour!.GetGPUBuffer()! } },
                },
            });
            pass.SetBindGroup(0, bg);
            pass.DispatchWorkgroups((uint)_tilesX, (uint)_tilesY, 1);
            pass.End();
            using var cmd = enc.Finish();
            RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, new[] { cmd });
        }
        await PhaseAsync("ranges+raster");

        // CPU transfer: gate comparison only. Training passes readback:false and the colour
        // stays on the GPU - at 640x480 this copy is 3.7 MB, which would dwarf the iteration.
        if (!readback) return System.Array.Empty<float>();
        return await _outColour!.CopyToHostAsync<float>(0, _width * _height * 3);
    }

    /// <summary>Upload the target image this view is being fitted to.</summary>
    public void SetTarget(float[] rgb)
    {
        if (rgb.Length != _width * _height * 3)
            throw new ArgumentException($"target is {rgb.Length}, expected {_width * _height * 3}");
        _target!.CopyFromCPU(rgb);
    }

    /// <summary>
    /// Point the loss at one image inside a GPU-resident stack of targets (view-major,
    /// width*height*3 floats each). Device-to-device, so a multi-view run pays the upload once.
    /// </summary>
    public void SetTargetFrom(MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex)
    {
        long pixels = (long)_width * _height;
        long off = (long)viewIndex * pixels;
        if (off + pixels > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} needs pixels [{off},{off + pixels}) of a {stack.Length}-pixel stack");

        // Unpack one frame out of the packed stack into the working float target.
        //
        // The stack holds RGBA8, four bytes a pixel, not three floats. That is a THIRD of the
        // memory, and target memory is what caps how many views can supervise a run: 256 MiB
        // held 62 views as floats and holds 187 packed. Views are the scarce resource here -
        // the reference trains drjohnson on about 230 images and we were using 33.
        WriteU32x4(_dimsBuf!, (uint)pixels, 0, (uint)off, 0);
        DispatchLinear(_unpackTarget!, pixels, new[]
        {
            Buf(0, stack.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
    }

    /// <summary>
    /// Gradient coverage over every splat. Magnitudes are the true float gradients.
    /// </summary>
    public readonly record struct GradientStats(
        long Splats, long ColourLive, long CentreLive, long ConicLive,
        double MeanCentreAbs, double MaxCentreAbs, double MaxConicAbs,
        double MaxColourAbs)
    {
        /// <summary>
        /// Splats that will take an Adam step on a gradient of exactly zero this iteration.
        ///
        /// With batch size 1 and a round robin over the supervised views, a splat visible in one
        /// view of 26 is invisible for the other 25 steps of the cycle. adam_geometry guards
        /// against stepping those; adam_step does not, on the stated judgement that it is
        /// "harmless for colour". This number is what makes that judgement checkable.
        /// </summary>
        public double StaleColourFraction =>
            Splats > 0 ? 1.0 - (double)ColourLive / Splats : 0.0;
    }

    const int GradStatsWorkgroups = 256;
    const int GradStatsSlots = 7;

    /// <summary>
    /// Reduce the gradient accumulator on the GPU and bring back 6 KB.
    ///
    /// Must be called AFTER TrainStepAsync returns and before the next one begins: the
    /// accumulator is cleared mid-step (between raster_backward and scatter_gradients), so it
    /// holds the completed step's totals until the following step reaches that point.
    /// </summary>
    public async Task<GradientStats> ReadGradientStatsAsync(int splatCount)
    {
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        Dispatch(_gradStats!, GradStatsWorkgroups, 1, new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _gradStatsPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();

        // CPU transfer: 6 floats per workgroup. Slots 0-3 are SUMS, slots 4-5 are MAXES - two
        // reduction operators in one array, so they must be combined differently here too.
        float[] p = await _gradStatsPartials!.CopyToHostAsync<float>(
            0, GradStatsWorkgroups * GradStatsSlots);

        double colour = 0, centre = 0, conic = 0, sumCentre = 0;
        double maxCentre = 0, maxConic = 0, maxColour = 0;
        for (int i = 0; i < GradStatsWorkgroups; i++)
        {
            int o = i * GradStatsSlots;
            colour += p[o];
            centre += p[o + 1];
            conic += p[o + 2];
            sumCentre += p[o + 3];
            maxCentre = Math.Max(maxCentre, p[o + 4]);
            maxConic = Math.Max(maxConic, p[o + 5]);
            maxColour = Math.Max(maxColour, p[o + 6]);
        }

        return new GradientStats(
            splatCount, (long)colour, (long)centre, (long)conic,
            centre > 0 ? sumCentre / centre : 0.0, maxCentre, maxConic, maxColour);
    }

    public void SetAdamStepCount(int step) => _adamStepCount = Math.Max(0, step);

    /// <summary>
    /// Splat count the optimizer rows were already carried to by
    /// <see cref="CarryOptimizerRowsAsync"/>, so the next <see cref="Resize"/> keeps them
    /// instead of reallocating. -1 when there is nothing carried.
    /// </summary>
    int _carriedRowsFor = -1;

    /// <summary>
    /// Carry the per-splat optimizer rows (Adam m/v, SH rest, SH Adam m/v) to a new splat set
    /// BEFORE <see cref="Resize"/>, one bank at a time, releasing the frame buffers first.
    ///
    /// The previous shape detached all five prior banks, ran Resize (which allocated all five
    /// again at the new count, plus every frame buffer), and only then remapped and disposed.
    /// At 912k splats the detached generation is 3 x 164 MB of SH plus 2 x 51 MB of Adam held
    /// alongside a complete new generation. MEASURED: the GT run densified to 450k through
    /// dozens of these with no trouble; Bathroom at 757k (bath2k-dav3-outside) and DrJohnson at
    /// 912k (dj2k-dav3-pose, after the pose-aware fold posed all 44 views) both lost the device
    /// at the SynchronizeAsync right after Resize, on the very first apply, with the buffers at
    /// the old count having run a full cycle without incident. The steady state fits; the
    /// doubled transient does not.
    ///
    /// Here the peak above steady state is ONE bank: frame buffers are released, then each
    /// bank is remapped into a fresh buffer and its prior disposed before the next is touched.
    /// Adam m/v go through the host (GPU remap of Adam killed opacity - MEASURED); SH banks are
    /// remapped on the GPU with a readback fence, because host SH OOM'd at ~780k (growhost).
    /// The trainer is unusable between this call and the Resize that must follow it.
    /// </summary>
    public async Task CarryOptimizerRowsAsync(
        int priorCount, int newCount, int[] adamSurvivors, int[] featureSources,
        int zeroAdamSlot = -1)
    {
        if (priorCount <= 0 || newCount <= 0 || adamSurvivors.Length != newCount
            || featureSources.Length != newCount)
            throw new ArgumentException(
                $"carry {priorCount:N0} -> {newCount:N0} needs one survivor and one feature " +
                $"source per new splat (got {adamSurvivors.Length:N0} / {featureSources.Length:N0})");
        var gpu = _gpu.WebGPUAccelerator;
        using var adamSrc = gpu.Allocate1D<int>(newCount);
        adamSrc.CopyFromCPU(adamSurvivors);
        using var featSrc = gpu.Allocate1D<int>(newCount);
        featSrc.CopyFromCPU(featureSources);
        await CarryOptimizerRowsAsync(priorCount, newCount, adamSrc, featSrc, zeroAdamSlot);
    }

    /// <summary>The per-splat densify accumulator (2 per splat: pixel-gradient sum, visible count), for <see cref="GpuDensify"/>.</summary>
    public ArrayView<float> DensifyStatsView => _densifyStats!.View;

    /// <summary>The per-splat max 3-sigma screen radius over the densify window, for <see cref="GpuDensify"/>.</summary>
    public ArrayView<float> MaxRadiusView => _maxRadius!.View;

    /// <summary>
    /// The same carry with the source maps already on the device (<see cref="GpuDensify.Result"/>): the maps
    /// never cross to the host. The caller keeps ownership of them.
    /// </summary>
    public async Task CarryOptimizerRowsAsync(
        int priorCount, int newCount,
        MemoryBuffer1D<int, Stride1D.Dense> adamSrc, MemoryBuffer1D<int, Stride1D.Dense> featSrc,
        int zeroAdamSlot = -1)
    {
        if (priorCount <= 0 || newCount <= 0 || adamSrc.Length < newCount || featSrc.Length < newCount)
            throw new ArgumentException(
                $"carry {priorCount:N0} -> {newCount:N0} needs one survivor and one feature source per new splat");

        var accel = _gpu.WebGPUAccelerator;
        await accel.SynchronizeAsync();
        int step = _adamStepCount;
        var clock = System.Diagnostics.Stopwatch.StartNew();

        // Everything sized for the old count that is NOT carried goes first, so the carry
        // runs against the smallest resident set the trainer can have.
        DisposeBuffers(keepOptimizerRows: true);
        await Stage("released frame buffers");

        {
            // All on the GPU, one bank at a time. The Adam moments used to go through the host because a
            // GPU remap "killed opacity" (MEASURED): that remap's clear ran AFTER its gather and zeroed the
            // moments, and zero moments under a large bias-corrected step count make Adam's next steps ~3x
            // the learning rate. Fixed in RemapGpuFencedAsync; CarryGateAsync checks every bank.
            _adamM = await CarryBankAsync(_adamM, adamSrc, priorCount, newCount, AdamSlots, zeroAdamSlot);
            await Stage("carried Adam m");
            _adamV = await CarryBankAsync(_adamV, adamSrc, priorCount, newCount, AdamSlots, zeroAdamSlot);
            await Stage("carried Adam v");
            for (int part = 0; part < SphericalHarmonics.Parts; part++)
            {
                _shRest[part] = await CarryBankAsync(_shRest[part], featSrc, priorCount, newCount, SphericalHarmonics.PartFloatsPerSplat);
                await Stage($"carried SH rest {part}");
                _adamShM[part] = await CarryBankAsync(_adamShM[part], adamSrc, priorCount, newCount, ShMomentWords);
                await Stage($"carried SH Adam m {part}");
                _adamShV[part] = await CarryBankAsync(_adamShV[part], adamSrc, priorCount, newCount, ShMomentWords);
                await Stage($"carried SH Adam v {part}");
            }
        }

        _adamStepCount = step;
        _carriedRowsFor = newCount;

        // Each stage waits for the device and says so. A device loss is reported by whichever
        // interop call happens to be awaiting, which names nothing; this names the stage.
        async Task Stage(string what)
        {
            await accel.SynchronizeAsync();
            Console.WriteLine(
                $"[Trainer] carry {priorCount:N0} -> {newCount:N0}: {what} ({clock.ElapsedMilliseconds} ms)");
        }
    }

    /// <summary>
    /// One per-splat bank (Adam or SH shaped): allocate at the new count, gather the prior rows into it on the GPU,
    /// fence, dispose the prior. Peak is prior + next for THIS bank only.
    /// </summary>
    async Task<MemoryBuffer1D<T, Stride1D.Dense>?> CarryBankAsync<T>(
        MemoryBuffer1D<T, Stride1D.Dense>? prior,
        MemoryBuffer1D<int, Stride1D.Dense> sources, int priorCount, int newCount,
        int stride, int zeroSlot = -1) where T : unmanaged
    {
        // Step markers BEFORE each operation (no sync): three TruckFull no-COLMAP runs lost the device inside the
        // first bank's carry and the post-stage log could not say which operation. Printed only when enabled.
        if (TraceCarrySteps) Console.WriteLine($"[Trainer]   carry bank x{stride}: allocate {(long)newCount * stride * 4 / 1048576.0:F0} MB ({GpuService.MemoryReport(4)})");
        var next = _gpu.WebGPUAccelerator.Allocate1D<T>((long)newCount * stride);
        if (prior == null)
        {
            // No prior bank (Resize never ran) - zeros, as Resize itself would hand out.
            next.MemSetToZero();
            return next;
        }
        await RemapGpuFencedAsync(prior, next, sources, priorCount, newCount, stride, zeroSlot);
        prior.Dispose();
        return next;
    }

    async Task RemapGpuFencedAsync<T>(
        MemoryBuffer1D<T, Stride1D.Dense>? prior,
        MemoryBuffer1D<T, Stride1D.Dense>? next,
        MemoryBuffer1D<int, Stride1D.Dense> sources,
        int priorCount,
        int newCount,
        int stride,
        int zeroSlot = -1) where T : unmanaged
    {
        if (prior == null || next == null || _remapFloatRows == null) return;
        if (TraceCarrySteps) Console.WriteLine("[Trainer]   carry bank: zero");
        next.MemSetToZero();
        // The clear sits in ILGPU's pending encoder and the gather below goes straight to the queue.
        // Unflushed, the clear was submitted by the fence's readback AFTER the gather and zeroed the whole
        // bank: every densify wiped all SH bands and their moments (CarryGateAsync, 11,160/11,160 floats).
        if (TraceCarrySteps) Console.WriteLine("[Trainer]   carry bank: flush");
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        uint z = zeroSlot >= 0 && zeroSlot < stride ? (uint)zeroSlot : uint.MaxValue;
        if (TraceCarrySteps) Console.WriteLine($"[Trainer]   carry bank: remap dispatch {(newCount + 63) / 64} groups");
        WriteU32x4(_dimsBuf!, (uint)newCount, (uint)stride, (uint)priorCount, z);
        DispatchLinear(_remapFloatRows!, newCount, new[]
        {
            Buf(0, prior.GetGPUBuffer()!), Buf(1, next.GetGPUBuffer()!),
            Buf(2, sources.GetGPUBuffer()!), Buf(3, _dimsBuf!),
        });
        if (TraceCarrySteps) Console.WriteLine("[Trainer]   carry bank: fence");
        // CPU transfer: 4-byte fence — drains WebGPU queue after Dispatch Submit.
        _ = await next.CopyToHostAsync<T>(0, 1);
    }

    /// <summary>Adam moments and the global step count, as one movable blob.</summary>
    public readonly record struct AdamState(float[] M, float[] V, int StepCount);

    /// <summary>
    /// Read the Adam moments so densification can carry them across a resize.
    ///
    /// CPU transfer: 14 moments x 2 per splat. Densification changes the splat count, which
    /// reallocates these buffers regardless, so the choice is between moving the state and
    /// throwing it away - not between moving it and leaving it alone.
    /// </summary>
    public async Task<AdamState> ReadAdamStateAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        long len = (long)splatCount * AdamSlots;
        return new AdamState(
            await _adamM!.CopyToHostAsync<float>(0, len),
            await _adamV!.CopyToHostAsync<float>(0, len),
            _adamStepCount);
    }

    /// <summary>
    /// Restore Adam moments after a resize, keeping the survivors' momentum.
    ///
    /// <paramref name="survivors"/> maps each NEW splat index to the OLD index it came from, or
    /// -1 for a splat that did not exist before. A clone and a split child both inherit their
    /// parent's momentum in the reference implementation only in the sense that they start at
    /// zero and the parent keeps its own; so new splats get zeros here, which is what the
    /// reference does when it concatenates zeroed rows.
    ///
    /// The step count is carried too. Bias correction divides by 1 - beta^t, and restarting t
    /// at zero makes the first step after every densification about ten times larger than it
    /// should be - a visible kick, nine times in a run.
    /// </summary>
    public void RestoreAdamState(AdamState prior, int[] survivors, int zeroSlot = -1)
    {
        int n = survivors.Length;
        var m = new float[(long)n * AdamSlots];
        var v = new float[(long)n * AdamSlots];
        int oldCount = prior.M.Length / AdamSlots;

        for (int i = 0; i < n; i++)
        {
            int src = survivors[i];
            if (src < 0 || src >= oldCount) continue;   // new splat: zeros, as allocated
            System.Array.Copy(prior.M, (long)src * AdamSlots, m, (long)i * AdamSlots, AdamSlots);
            System.Array.Copy(prior.V, (long)src * AdamSlots, v, (long)i * AdamSlots, AdamSlots);
        }

        // One parameter's momentum can be dropped while the rest is kept - used by the
        // opacity reset, which momentum would otherwise undo within a few steps.
        if (zeroSlot >= 0 && zeroSlot < AdamSlots)
            for (long i = 0; i < n; i++) { m[i * AdamSlots + zeroSlot] = 0f; v[i * AdamSlots + zeroSlot] = 0f; }

        _adamM!.CopyFromCPU(m);
        _adamV!.CopyFromCPU(v);
        _adamStepCount = prior.StepCount;
    }

    /// <summary>
    /// Peak key demand seen since it was last cleared, so a caller can re-size on evidence.
    ///
    /// <see cref="LastOverflowed"/> is per-frame and a densification window spans hundreds of
    /// frames, so a caller that only looked at the last one would miss every overflow but one.
    /// </summary>
    public int PeakKeyDemand { get; private set; }

    /// <summary>Clear the peak, at the start of a new densification window.</summary>
    public void ResetPeakKeyDemand() { PeakKeyDemand = 0; KeyGrowths = 0; }

    /// <summary>
    /// Where a view's gradient dies: magnitudes at each stage of the chain, plus whether the
    /// forward pass painted anything. keys + loss + zero per-key can mean "forward was black"
    /// or "backward dropped a live render" - those are different files.
    /// </summary>
    public async Task<(double DLdPix, double PerKey, double MeanColour, double MeanFinalT,
        int NanKeys, int NanT, double FracTOneOpaque)> ReadBackwardStagesAsync()
    {
        long pixels = (long)_width * _height;
        double dl = await MaxMagnitudeAsync(_dLdPix!, pixels * 3);
        double pk = await MaxMagnitudeAsync(_gradKeyA!, (long)LastKeyCount * 3);
        var (meanC, _, _) = await SampleStatsAsync(_outColour!, pixels * 3);
        // max_magnitude reports 0 for an all-NaN buffer (max(0, NaN) drops the NaN), so a NaN
        // backward looks identical to an empty one. Count NaNs explicitly. T == 1-MAX_ALPHA at
        // a pixel means exactly one splat was applied at the alpha cap and nothing after it;
        // that is what min(MAX_ALPHA, NaN) produces in the forward.
        var (_, nanKeys, _) = await SampleStatsAsync(_gradKeyA!, (long)LastKeyCount * 3);
        var (meanT, nanT, fracT) = await SampleStatsAsync(_outFinalT!, pixels, 0.0100000179f, 1e-6f);
        return (dl, pk, meanC, meanT, nanKeys, nanT, fracT);
    }

    /// <summary>
    /// Stride-sampled (mean |x|, NaN count, fraction within <paramref name="tol"/> of
    /// <paramref name="match"/>) over a GPU buffer. Stage-probe only: it runs on the handful
    /// of views that produced no gradient, never in the training loop.
    /// </summary>
    async Task<(double MeanAbs, int NanCount, double FracMatch)> SampleStatsAsync(
        MemoryBuffer1D<float, Stride1D.Dense> buf, long count, float match = float.NaN, float tol = 0f)
    {
        if (count <= 0 || _fitSample == null || _sampleStride == null) return (0, 0, 0);
        int n = (int)Math.Min(FitSampleCount, count);
        int stride = Math.Max(1, (int)(count / n));
        n = (int)Math.Min(FitSampleCount, (count + stride - 1) / stride);
        WriteU32x4(_dimsBuf!, (uint)n, (uint)stride, (uint)Math.Min(count, buf.Length), 0);
        DispatchLinear(_sampleStride, n, groupSize: 256, entries: new[]
        {
            Buf(0, buf.GetGPUBuffer()!), Buf(1, _fitSample.GetGPUBuffer()!), Buf(2, _dimsBuf!),
        });
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // CPU transfer: 4K-float stride sample for the dead-view stage probe.
        float[] host = await _fitSample.CopyToHostAsync<float>(0, n);
        double sum = 0; int nan = 0, hit = 0;
        for (int i = 0; i < host.Length; i++)
        {
            float v = host[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
            sum += Math.Abs(v);
            if (!float.IsNaN(match) && Math.Abs(v - match) <= tol) hit++;
        }
        int finite = host.Length - nan;
        return (finite > 0 ? sum / finite : 0, nan, host.Length > 0 ? (double)hit / host.Length : 0);
    }

    async Task<double> MaxMagnitudeAsync(MemoryBuffer1D<float, Stride1D.Dense> buf, long count)
    {
        if (count <= 0) return 0;
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)Math.Min(count, buf.Length), 0, 0, 0);
        Dispatch(_maxMagnitude!, GradStatsWorkgroups, 1, new[]
        {
            Buf(0, buf.GetGPUBuffer()!), Buf(1, _gradStatsPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();
        float[] p = await _gradStatsPartials!.CopyToHostAsync<float>(0, GradStatsWorkgroups);
        double m = 0;
        foreach (float v in p) m = Math.Max(m, v);
        return m;
    }

    /// <summary>
    /// Densify average denominator. false: steps where the splat got a centre gradient. true: the reference's
    /// <c>visibility_filter</c> (<c>radii &gt; 0</c>), every step it projected onto the screen, occluded or not,
    /// so a splat hidden half the time averages half as high. Default true; <c>&amp;densifydenom=contrib</c> for the old.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-25, Truck 7K / 3M cap / GT poses, same build (tuvok-b2-base vs -denom): held-out captures
    /// 22.52 vs 22.67 dB, SSIM .813 vs .811, sharpness .68 vs .67 (within run noise) with 2.17M -> 1.52M splats
    /// (-30%) and 13.8 -> 16.7 it/s (+21%).
    /// </remarks>
    public static bool DensifyDenominatorFrustum { get; set; } = true;

    /// <summary>
    /// Geometry Adam steps every splat every iteration, as torch Adam (the reference's default optimiser) does:
    /// a splat with no gradient this view still decays its moments and moves by the momentum it carries.
    /// Default false: such a splat is not stepped at all. <c>&amp;denseadam=1</c>.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-25, Truck 7K / 3M / GT poses, two batches on two builds: final held-out captures +0.18 dB
    /// (22.70 vs 22.52, tuvok-b2-dense) and +0.30 dB (22.62 vs 22.32, tuvok-b3-dense), SSIM up both times; run-to-run
    /// noise on that mean is 0.1-0.2 dB (tuvok-b3-base vs -seed2). ~9% more splats. Default on; &amp;denseadam=0.
    /// </remarks>
    public static bool DenseGeometryAdam { get; set; } = true;

    /// <summary>
    /// D-SSIM in the training loss per RGB channel, averaged (<see cref="ImageQuality.MeanSsimRgb"/>): the reference's
    /// loss_utils.ssim. false: SSIM on Rec.601 luma, which hands blue 0.114 of one shared structural gradient and
    /// none to an edge that differs only in colour. Scoring SSIM stays on luma either way. <c>&amp;ssimrgb=1</c>.
    /// </summary>
    /// <remarks>
    /// MEASURED 2026-09-25 (tuvok-b3-ssimrgb vs -base/-seed2): held-out captures 22.58 vs 22.32 / 22.36 dB, curve
    /// 18.96 vs 18.86 / 18.71 (best of the batch), SSIM .813 vs .811 / .807. Default on; &amp;ssimrgb=0 for luma.
    /// </remarks>
    public static bool SsimPerChannel { get; set; } = true;

    /// <summary>Start a fresh densification window. Call after each densify step.</summary>
    public void ResetDensifyStats()
    {
        _densifyStats!.MemSetToZero();
        _maxRadius!.MemSetToZero();
        // Submit now: the accumulate is a raw dispatch, and an unflushed clear would land after it.
        _gpu.WebGPUAccelerator.FlushPendingCommands();
    }

    /// <summary>
    /// Fold the step that just finished into the densification statistics. One dispatch, no sync.
    ///
    /// Same timing constraint as <see cref="ReadGradientStatsAsync"/>: the accumulator is cleared
    /// mid-step, so this must run after TrainStepAsync returns and before the next one begins.
    /// </summary>
    public void AccumulateDensifyStats(int splatCount)
    {
        WriteU32x4(_dimsBuf!, (uint)splatCount, (uint)_width, (uint)_height, DensifyDenominatorFrustum ? 1u : 0u);
        DispatchLinear(_densifyAccum!, splatCount, groupSize: 256, entries: new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _densifyStats!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!), Buf(3, _screenRadius!.GetGPUBuffer()!), Buf(4, _maxRadius!.GetGPUBuffer()!),
        });
    }

    /// <summary>
    /// Read the accumulated densification statistics.
    ///
    /// CPU transfer: 8 bytes per splat, and the decision it feeds changes the splat COUNT, which
    /// means reallocating every buffer anyway. At 80k splats this is 640 KB every hundred
    /// iterations. Doing the clone/split decision on the GPU would need a compaction pass and a
    /// prefix sum; that is worth writing when the count makes it worth writing.
    /// </summary>
    public async Task<SplatDensityControl.Accumulator[]> ReadDensifyStatsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        float[] raw = await _densifyStats!.CopyToHostAsync<float>(0, (long)splatCount * 2);
        // The largest screen radius each splat reached. Nothing used to fill this, so the reference's
        // max_screen_size prune (SplatDensityControl.MaxScreenRadiusPx) never fired: "0 bloated", every
        // densify, on every run - and a floater parked in front of a camera could never be removed.
        float[] radius = await _maxRadius!.CopyToHostAsync<float>(0, splatCount);

        var stats = new SplatDensityControl.Accumulator[splatCount];
        for (int i = 0; i < splatCount; i++)
            stats[i] = new SplatDensityControl.Accumulator
            {
                GradientSum = raw[i * 2],
                VisibleCount = (int)raw[i * 2 + 1],
                MaxScreenRadiusPx = radius[i],
            };
        return stats;
    }

    const int SupportWorkgroups = 256;
    const int SupportSlots = 6;

    /// <summary>
    /// How many distinct views constrain each splat, bucketed.
    ///
    /// <paramref name="Views"/> is how many views were accumulated, so the buckets can be read
    /// against the maximum a splat could possibly have reached.
    /// </summary>
    public readonly record struct ViewSupport(
        long Splats, int Views,
        long Unconstrained, long OneView, long TwoViews, long ThreeViews, long FourOrMore,
        double MeanViews)
    {
        /// <summary>
        /// Splats that no two views agree on - the ones a second view could contradict but
        /// never does. If this is most of the scene, the reconstruction is a stack of per-view
        /// shells and cannot generalise to a view nobody trained on, whatever the optimiser does.
        /// </summary>
        public double SingleViewFraction =>
            Splats > 0 ? (double)(Unconstrained + OneView) / Splats : 0.0;
    }

    /// <summary>Start a fresh support count. Call once, then accumulate over a full cycle.</summary>
    public void ResetViewSupport(int splatCount)
    {
        _viewSupport!.MemSetToZero();
        _gpu.WebGPUAccelerator.FlushPendingCommands(); // ahead of the raw accumulate dispatches
        _supportViewsAccumulated = 0;
    }

    int _supportViewsAccumulated;

    /// <summary>
    /// Fold the step that just finished into the support count. One dispatch, no sync.
    ///
    /// Same timing constraint as <see cref="ReadGradientStatsAsync"/>: the accumulator is
    /// cleared mid-step, so this must run after TrainStepAsync returns and before the next.
    /// </summary>
    public void AccumulateViewSupport(int splatCount)
    {
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        DispatchLinear(_accumulateSupport!, splatCount, groupSize: 256, entries: new[]
        {
            Buf(0, _gradFixed!.GetGPUBuffer()!), Buf(1, _viewSupport!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        _supportViewsAccumulated++;
    }

    /// <summary>Reduce the support counts to buckets and bring back 6 KB.</summary>
    public async Task<ViewSupport> ReadViewSupportAsync(int splatCount)
    {
        var accel = _gpu.WebGPUAccelerator;
        WriteU32x4(_dimsBuf!, (uint)splatCount, 0, 0, 0);
        Dispatch(_supportHistogram!, SupportWorkgroups, 1, new[]
        {
            Buf(0, _viewSupport!.GetGPUBuffer()!), Buf(1, _supportPartials!.GetGPUBuffer()!),
            Buf(2, _dimsBuf!),
        });
        await accel.SynchronizeAsync();

        // CPU transfer: 6 floats per workgroup. Every slot is a sum.
        float[] p = await _supportPartials!.CopyToHostAsync<float>(
            0, SupportWorkgroups * SupportSlots);

        double c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, total = 0;
        for (int i = 0; i < SupportWorkgroups; i++)
        {
            int o = i * SupportSlots;
            c0 += p[o]; c1 += p[o + 1]; c2 += p[o + 2];
            c3 += p[o + 3]; c4 += p[o + 4]; total += p[o + 5];
        }

        return new ViewSupport(
            splatCount, _supportViewsAccumulated,
            (long)c0, (long)c1, (long)c2, (long)c3, (long)c4,
            splatCount > 0 ? total / splatCount : 0.0);
    }

    /// <summary>
    /// Per-splat view-support counts after a full cycle. Used to compact out the unconstrained
    /// half before densification starts spending budget on them.
    ///
    /// CPU transfer: 4 bytes per splat. One read per run, and the alternative is leaving half
    /// the scene as dead weight that can only hurt held-out views.
    /// </summary>
    public async Task<uint[]> ReadViewSupportCountsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        return await _viewSupport!.CopyToHostAsync<uint>(0, splatCount);
    }

    /// <summary>
    /// Skip the colour/opacity Adam step for splats whose gradient is exactly zero.
    ///
    /// Off by default. adam_geometry has always done this for position; whether it helps colour
    /// and opacity at batch size 1 is the measurement, and a default that changes quietly would
    /// make every earlier run incomparable.
    /// </summary>
    public bool SkipZeroGradientSteps { get; set; }

    /// <summary>The viewport the trainer is currently sized for.</summary>
    public (int Width, int Height) Size => (_width, _height);

    int SseWorkgroups => (_width * _height + 255) / 256;

    /// <summary>Count of 'valid' SSIM window positions across and down.</summary>
    int SsimWindowsX => _width - ImageQuality.WindowSize + 1;
    int SsimWindowsY => _height - ImageQuality.WindowSize + 1;
    bool HasSsimWindows => _width >= ImageQuality.WindowSize && _height >= ImageQuality.WindowSize;
    int SsimWorkgroups => (SsimWindowsX * SsimWindowsY + 255) / 256;

    /// <summary>
    /// Upload every SSIM constant from <see cref="ImageQuality"/>, so the shader holds none.
    ///
    /// The one value WGSL must duplicate is the window size, because it sizes a loop and a
    /// shader cannot read a C# constant. Asserting it here rather than trusting it: if the
    /// oracle's window ever changes, this throws with the file to edit instead of silently
    /// scoring two different metrics.
    /// </summary>
    void WriteSsimCfg() => WriteSsimCfg((float)ImageQuality.LumaR, (float)ImageQuality.LumaG, (float)ImageQuality.LumaB);

    /// <summary>
    /// SSIM config with the given channel weights: luma for scoring, one-hot for one pass of the per-channel
    /// training loss (<see cref="SsimPerChannel"/>). Every value still comes from <see cref="ImageQuality"/>.
    /// </summary>
    void WriteSsimCfg(float wr, float wg, float wb)
    {
        if (ImageQuality.WindowSize != 11)
            throw new InvalidOperationException(
                $"ImageQuality.WindowSize is {ImageQuality.WindowSize} but the WGSL in " +
                "SplatTrainerShaders.SsimRows/SsimReduce declares 'const WINDOW : u32 = 11u'. " +
                "Change both or they measure different things.");

        var k = ImageQuality.GaussianKernel1D();
        var f = new float[20];                       // 80 bytes: luma, consts, 3 x vec4 of taps
        f[0] = wr;
        f[1] = wg;
        f[2] = wb;
        f[4] = (float)ImageQuality.C1;
        f[5] = (float)ImageQuality.C2;
        for (int i = 0; i < k.Length; i++) f[8 + i] = (float)k[i];

        var bytes = new byte[80];
        Buffer.BlockCopy(f, 0, bytes, 0, 80);
        _queue!.WriteBuffer(_ssimCfgBuf!, 0, bytes);
    }

    /// <summary>
    /// PSNR of the current render against one image in a GPU-resident target stack.
    ///
    /// The whole comparison happens on the GPU; only one partial sum per 256 pixels comes back,
    /// a few KB rather than the two full images this used to copy. On a 35-view capture that is
    /// the difference between about 650 MB of readback per evaluation pass and about 600 KB.
    /// </summary>
    public async Task<double> PsnrAgainstAsync(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex)
        => (await ScoreAgainstAsync(stack, viewIndex, withSsim: false)).Psnr;

    /// <summary>
    /// PSNR and SSIM of the current render against one image in the target stack.
    ///
    /// Both in one call because they share the render and can share the sync: three dispatches
    /// are queued, then ONE SynchronizeAsync, then both partial buffers come back. Scoring them
    /// separately would double the round trips per view for no benefit.
    ///
    /// SSIM matters because PSNR does not see the failure this trainer currently has. MEASURED
    /// on Bathroom: held out went 12.50 -> 12.21 dB across a run, essentially flat, while the
    /// render melted from a recognisable room into fog. PSNR on a sparse reconstruction is
    /// dominated by large smooth regions, so smoothing structure away barely moves it.
    ///
    /// Returns NaN for SSIM when the viewport is smaller than the window - there are no valid
    /// window positions and the metric is undefined, which the Python oracle signals by raising.
    /// </summary>
    public async Task<(double Psnr, double Ssim)> ScoreAgainstAsync(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex, bool withSsim = true)
    {
        var accel = _gpu.WebGPUAccelerator;

        // Unpack the view being scored into the working target, then compare against THAT.
        // Scoring straight out of the stack would mean teaching eval_sse and ssim_rows to
        // unpack as well, which is two more places for the pixel format to drift apart from
        // its oracle. One unpack, one format, one gate.
        SetTargetFrom(stack, viewIndex);
        uint targetOffset = 0;

        WriteU32x2(_dimsBuf!, (uint)(_width * _height), targetOffset);
        Dispatch(_evalSse!, SseWorkgroups, 1, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
            Buf(2, _ssePartials!.GetGPUBuffer()!), Buf(3, _dimsBuf!),
        });

        bool doSsim = withSsim && HasSsimWindows && _ssimRows != null;
        if (doSsim)
        {
            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY,
                (uint)_width, targetOffset);

            // Horizontal pass over every source ROW (not just window rows): the vertical pass
            // reads eleven rows above each window, so the rows above the last window position
            // are still needed.
            int rowThreads = SsimWindowsX * _height;
            DispatchLinear(_ssimRowsPipe!, rowThreads, new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimRows!.GetGPUBuffer()!), Buf(3, _ssimDimsBuf!), Buf(4, _ssimCfgBuf!),
            });
            Dispatch(_ssimReducePipe!, SsimWorkgroups, 1, new[]
            {
                Buf(0, _ssimRows!.GetGPUBuffer()!), Buf(1, _ssimPartials!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!),
            });
        }

        await accel.SynchronizeAsync();

        // CPU transfer: one float per 256 pixels / per 256 windows, summed in double because a
        // million f32 adds in sequence loses the tail.
        float[] partials = await _ssePartials!.CopyToHostAsync<float>(0, SseWorkgroups);
        double sse = 0;
        foreach (float v in partials) sse += v;
        double mse = sse / ((long)_width * _height * 3);
        double psnr = mse <= 1e-12 ? 99.0 : 10.0 * Math.Log10(1.0 / mse);

        double ssim = double.NaN;
        if (doSsim)
        {
            float[] sp = await _ssimPartials!.CopyToHostAsync<float>(0, SsimWorkgroups);
            double total = 0;
            foreach (float v in sp) total += v;
            ssim = total / ((double)SsimWindowsX * SsimWindowsY);
        }
        return (psnr, ssim);
    }

    /// <summary>
    /// Put one target photograph into a GPU-resident stack, straight from the canvas.
    ///
    /// <paramref name="rgba"/> is the JS typed array from <c>getImageData</c>; its bytes go to
    /// the GPU without passing through the managed heap, and a kernel expands them into floats.
    /// Reading them into .NET to convert in a loop was a million iterations and about 6 MB of
    /// managed allocation per view.
    /// </summary>
    public void UploadTargetFrom(
        MemoryBuffer1D<uint, Stride1D.Dense> stack, int viewIndex, Uint8Array rgba)
    {
        long pixels = (long)_width * _height;
        long off = (long)viewIndex * pixels;
        if (off + pixels > stack.Length)
            throw new ArgumentOutOfRangeException(nameof(viewIndex),
                $"view {viewIndex} does not fit a {stack.Length}-pixel stack");

        // Straight into the stack. The bytes are already the format the stack stores, so there
        // is nothing to expand at upload time - the unpack moved to SetTargetFrom, where it
        // runs on one frame instead of all of them.
        _queue!.WriteBuffer(stack.GetGPUBuffer()!, (ulong)(off * 4), rgba);
    }

    /// <summary>
    /// Seed opacity logits and log scales from the splats, and zero every moment. Call once
    /// before training. After a densify or prune use <see cref="SeedLogits"/> instead, so the
    /// moments <see cref="CarryOptimizerRowsAsync"/> just carried are not thrown away.
    /// </summary>
    public void InitOptimizerState(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        SeedLogits(splatBuf, splatCount);
        _adamM!.MemSetToZero();
        _adamV!.MemSetToZero();
        foreach (var m in _adamShM) m?.MemSetToZero();
        foreach (var v in _adamShV) v?.MemSetToZero();
        _gpu.WebGPUAccelerator.FlushPendingCommands(); // ahead of the raw Adam dispatches
        _adamStepCount = 0;
    }

    /// <summary>
    /// Opacity logits and log scales from the packed splats - the trainable parameters the
    /// packed format does not store in trainable form. Leaves Adam and SH state alone.
    /// </summary>
    public void SeedLogits(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        var splatGpu = splatBuf.GetGPUBuffer()!;
        WriteU32(_dimsBuf!, (uint)splatCount);
        DispatchLinear(_initLogits!, splatCount, new[]
        {
            Buf(0, splatGpu), Buf(1, _opacityLogit!.GetGPUBuffer()!), Buf(2, _dimsBuf!),
            Buf(3, _logScale!.GetGPUBuffer()!),
        });
    }

    /// <summary>Convert packed linear RGB to SH DC once before the first training step (once per TRAINER: the gates
    /// make one per check; the studio's trainer outlives its scenes and uses <see cref="ConvertRgbToShDc"/>).</summary>
    public void EnsureRgbConvertedToShDc(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        if (_rgbToDcDone) return;
        ConvertRgbToShDc(splatBuf, splatCount);
    }

    /// <summary>Convert packed linear RGB to SH DC, unconditionally: the caller knows the colours are RGB.</summary>
    public void ConvertRgbToShDc(MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        var splatGpu = splatBuf.GetGPUBuffer()!;
        WriteU32(_dimsBuf!, (uint)splatCount);
        DispatchLinear(_initRgbToDc!, splatCount, new[]
        {
            Buf(0, splatGpu), Buf(1, _dimsBuf!),
        });
        _rgbToDcDone = true;
    }

    /// <summary>
    /// Start the trainer's SH rest bands from a scene's (the display renderer's parts, PartFloatsPerSplat floats a
    /// splat), GPU to GPU, instead of zero: a saved trained scene trained further, or scored, keeps its view-dependent
    /// colour. Call after <see cref="ResizeAsync"/> for the same splat count.
    /// </summary>
    public void SeedShRestFrom(GPUBuffer[] parts, int splatCount)
    {
        if (_shRest[0] == null || _device == null || _queue == null || splatCount <= 0) return;
        if (parts.Length != SphericalHarmonics.Parts) throw new ArgumentException($"{parts.Length} SH parts, expected {SphericalHarmonics.Parts}");
        ulong bytes = (ulong)splatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var encoder = _device.CreateCommandEncoder();
        for (int part = 0; part < parts.Length; part++)
            encoder.CopyBufferToBuffer(parts[part], 0, _shRest[part]!.GetGPUBuffer()!, 0, bytes);
        using var cmd = encoder.Finish();
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, new[] { cmd });
    }

    /// <summary>
    /// GPU copies of the SH rest parts for the display renderer (caller owns them; SphericalHarmonics.Parts buffers of
    /// PartFloatsPerSplat floats a splat), or null when there are none. GPU to GPU - the coefficients never cross to
    /// the CPU. Pending trainer dispatches are submitted first so the copies see their writes.
    /// </summary>
    public GPUBuffer[]? CopyShRestForDisplay(int splatCount)
    {
        if (_shRest[0] == null || _device == null || _queue == null || splatCount <= 0) return null;
        ulong bytes = (ulong)splatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
        var parts = new GPUBuffer[SphericalHarmonics.Parts];
        _gpu.WebGPUAccelerator.FlushPendingCommands();
        using var encoder = _device.CreateCommandEncoder();
        for (int part = 0; part < parts.Length; part++)
        {
            parts[part] = _device.CreateBuffer(new GPUBufferDescriptor
            {
                Size = bytes,
                // The renderer owns these; CopySrc lets GpuGaussianRenderer.ReadShRestPartsAsync read them back.
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst | GPUBufferUsage.CopySrc,
            });
            encoder.CopyBufferToBuffer(_shRest[part]!.GetGPUBuffer()!, 0, parts[part], 0, bytes);
        }
        using var cmd = encoder.Finish();
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue, new[] { cmd });
        return parts;
    }

    /// <summary>
    /// The SH rest parts as JS Uint8Arrays (caller disposes), for saving to OPFS without passing through the .NET heap
    /// - one array per part, so no single array grows past ~60 bytes a splat (14M splats: 840 MB each, where one
    /// 45-float array would be 2.5 GB). Null when there are none.
    /// </summary>
    public async Task<Uint8Array[]?> ReadShRestUint8ArraysAsync(int splatCount)
    {
        if (_shRest[0] == null || splatCount <= 0) return null;
        long bytes = (long)splatCount * SphericalHarmonics.PartFloatsPerSplat * sizeof(float);
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        var parts = new Uint8Array[SphericalHarmonics.Parts];
        for (int part = 0; part < parts.Length; part++)
            parts[part] = await _shRest[part]!.CopyToHostUint8ArrayAsync(0, bytes);
        return parts;
    }

    /// <summary>The SH rest coefficients as rows of RestFloatsPerSplat (gate / CPU oracle only).</summary>
    public async Task<float[]> ReadShRestAsync(int splatCount)
    {
        if (_shRest[0] == null) return System.Array.Empty<float>();
        long len = (long)splatCount * SphericalHarmonics.PartFloatsPerSplat;
        var parts = new float[SphericalHarmonics.Parts][];
        for (int part = 0; part < parts.Length; part++)
            parts[part] = await _shRest[part]!.CopyToHostAsync<float>(0, len);
        return SphericalHarmonics.JoinParts(parts);
    }

    /// <summary>
    /// SH-rest Adam moments across a densify. Same survivor map as colour Adam: keep for
    /// survivors, zero for densified children. Resize zeros these; without a restore every
    /// densify (every 100 iters) throws away SH momentum while colour Adam is carefully kept.
    /// Rows of RestFloatsPerSplat floats; stored as bfloat16 pairs per part.
    /// </summary>
    public readonly record struct ShAdamState(float[] M, float[] V);

    public async Task<ShAdamState> ReadShAdamStateAsync(int splatCount)
    {
        if (_adamShM[0] == null || _adamShV[0] == null)
            return new ShAdamState(System.Array.Empty<float>(), System.Array.Empty<float>());
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // CPU transfer: gate / densify restore only.
        long len = (long)splatCount * ShMomentWords;
        int floats = SphericalHarmonics.PartFloatsPerSplat;
        var m = new float[SphericalHarmonics.Parts][];
        var v = new float[SphericalHarmonics.Parts][];
        for (int part = 0; part < m.Length; part++)
        {
            m[part] = Bf16.Unpack(await _adamShM[part]!.CopyToHostAsync<uint>(0, len), floats);
            v[part] = Bf16.Unpack(await _adamShV[part]!.CopyToHostAsync<uint>(0, len), floats);
        }
        return new ShAdamState(SphericalHarmonics.JoinParts(m), SphericalHarmonics.JoinParts(v));
    }

    public void RestoreShRest(ReadOnlySpan<float> prior, int[] featureSources)
    {
        if (_shRest[0] == null) return;
        var parts = SphericalHarmonics.SplitParts(
            SplatDensityControl.RemapFloatRows(prior, featureSources, SphericalHarmonics.RestFloatsPerSplat));
        for (int part = 0; part < parts.Length; part++) _shRest[part]!.CopyFromCPU(parts[part]);
    }

    public void RestoreShAdamState(ShAdamState prior, int[] adamSurvivors)
    {
        if (_adamShM[0] == null || _adamShV[0] == null) return;
        int stride = SphericalHarmonics.RestFloatsPerSplat, partFloats = SphericalHarmonics.PartFloatsPerSplat;
        var m = SphericalHarmonics.SplitParts(SplatDensityControl.RemapFloatRows(prior.M, adamSurvivors, stride));
        var v = SphericalHarmonics.SplitParts(SplatDensityControl.RemapFloatRows(prior.V, adamSurvivors, stride));
        for (int part = 0; part < m.Length; part++)
        {
            _adamShM[part]!.CopyFromCPU(Bf16.Pack(m[part], partFloats));
            _adamShV[part]!.CopyFromCPU(Bf16.Pack(v[part], partFloats));
        }
    }

    GPUBindGroupEntry ShRestBindEntry(int binding, int part) =>
        new()
        {
            Binding = (uint)binding,
            Resource = new GPUBufferBinding { Buffer = _shRest[part]!.GetGPUBuffer()! },
        };

    /// <summary>
    /// One training iteration on the current target: forward, loss, backward, scatter, Adam.
    /// Returns the L1 loss BEFORE the step, so a caller can watch it fall.
    ///
    /// Deliberately one iteration per call. A long-running dispatch chain can trip the driver
    /// watchdog and lose the device, so the host yields between iterations rather than queueing
    /// a whole training run.
    /// </summary>
    public async Task<float> TrainStepAsync(
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount,
        CameraParams cam, float depthNear, float depthFar,
        float colourLr = SplatOptimizer.DefaultColourLr,
        float opacityLr = SplatOptimizer.DefaultOpacityLr,
        GeometryStep? geometry = null,
        bool readLoss = true,
        int poseSlot = -1)
    {
        var accel = _gpu.WebGPUAccelerator;
        var splatGpu = splatBuf.GetGPUBuffer()!;
        if (ProfilePhases) { _phaseSteps++; _phaseClock.Restart(); }

        // The loss accumulates across unread steps; clear it only when a fresh sum starts. The forward's
        // flush submits this clear ahead of the loss pass.
        if (_lossStepsPending == 0) _lossSum!.MemSetToZero();

        // Forward also refreshes the tile binning for this view.
        await RenderForwardAsync(splatBuf, splatCount, cam, depthNear, depthFar, readback: false);
        if (LastKeyCount == 0)
        {
            // Clear before returning: the view-support census and densify accum read these
            // buffers after TrainStepAsync returns. Leaving the previous view's gradients here
            // made a zero-key view look live (Sonnet audit 2026-09-21).
            _gradFixed!.MemSetToZero();
            _densifyAbs!.MemSetToZero();
            accel.FlushPendingCommands();
            // Counts as a step of loss 0, as it always has: it adds nothing to the sum.
            return await FinishLossAsync(readLoss);
        }

        int pixels = _width * _height;

        // ── Loss and dL/d(pixel): 0.8 L1 + 0.2 D-SSIM, matching the reference ──
        WriteVec4(_lossWeightsBuf!, ImageQuality.LambdaL1, ImageQuality.LambdaDssim, 0f, 0f);
        // A 2D grid of 64-pixel workgroups: one dimension caps at 65,535 (4.2 MP), and the photos' own size is above it.
        int lossGroups = (pixels + 63) / 64;
        int lossGroupsX = Math.Min(lossGroups, 32768), lossGroupsY = (lossGroups + lossGroupsX - 1) / lossGroupsX;
        WriteU32x4(_lossDimsBuf!, (uint)pixels, (uint)lossGroupsX, (uint)lossGroups, 0);
        Dispatch(_lossL1!, lossGroupsX, lossGroupsY, new[]
        {
            Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
            Buf(2, _dLdPix!.GetGPUBuffer()!), Buf(3, _lossPartial!.GetGPUBuffer()!),
            Buf(4, _lossDimsBuf!), Buf(5, _lossWeightsBuf!),
        });
        Dispatch(_lossReduce!, 1, 1, new[]
        {
            Buf(0, _lossPartial!.GetGPUBuffer()!), Buf(1, _lossSum!.GetGPUBuffer()!),
            Buf(2, _lossDimsBuf!), Buf(3, _lossWeightsBuf!),
        });

        // Per channel: the same four passes three times, one-hot channel weights at lambda / 3 each, which sums to
        // exactly ImageQuality.AddMeanSsimRgbGradient (ssim_pix_bwd ADDS into dL/dpix). Each Dispatch submits, so
        // rewriting the cfg between passes is ordered. The luma cfg is restored for scoring afterwards.
        int ssimPasses = SsimPerChannel ? 3 : 1;
        for (int ssimPass = 0; ssimPass < ssimPasses; ssimPass++)
        if (HasSsimWindows && _ssimRows != null && _ssimWinGrad != null && _ssimDRows != null)
        {
            if (SsimPerChannel) WriteSsimCfg(ssimPass == 0 ? 1f : 0f, ssimPass == 1 ? 1f : 0f, ssimPass == 2 ? 1f : 0f);
            // Reuse the scoring forward's horizontal pass, then the three adjoint passes that
            // mirror ImageQuality.AddMeanSsimLumaGradient (FD-gated).
            // ssim_rows dims: wx, wy (=height-10), srcW, targetOffset. srcH = wy+10.
            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_width, 0);
            int rowThreads = SsimWindowsX * _height;
            DispatchLinear(_ssimRowsPipe!, rowThreads, new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimRows!.GetGPUBuffer()!), Buf(3, _ssimDimsBuf!), Buf(4, _ssimCfgBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_width, 0);
            WriteVec4(_lossWeightsBuf!, ImageQuality.LambdaDssim / ssimPasses, 0f, 0f, 0f);
            int winThreads = SsimWindowsX * SsimWindowsY;
            DispatchLinear(_ssimWinGradPipe!, winThreads, groupSize: 256, entries: new[]
            {
                Buf(0, _ssimRows!.GetGPUBuffer()!), Buf(1, _ssimWinGrad!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!), Buf(4, _lossWeightsBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)SsimWindowsY, (uint)_height, 0);
            DispatchLinear(_ssimRowsBwdPipe!, rowThreads, groupSize: 256, entries: new[]
            {
                Buf(0, _ssimWinGrad!.GetGPUBuffer()!), Buf(1, _ssimDRows!.GetGPUBuffer()!),
                Buf(2, _ssimDimsBuf!), Buf(3, _ssimCfgBuf!),
            });

            WriteU32x4(_ssimDimsBuf!, (uint)SsimWindowsX, (uint)_width, (uint)_height, 0);
            DispatchLinear(_ssimPixBwdPipe!, pixels, groupSize: 256, entries: new[]
            {
                Buf(0, _outColour!.GetGPUBuffer()!), Buf(1, _target!.GetGPUBuffer()!),
                Buf(2, _ssimDRows!.GetGPUBuffer()!), Buf(3, _dLdPix!.GetGPUBuffer()!),
                Buf(4, _ssimDimsBuf!), Buf(5, _ssimCfgBuf!),
            });
        }
        if (SsimPerChannel && HasSsimWindows) WriteSsimCfg();

        await PhaseAsync("loss+ssim");
        // ── Backward: one workgroup per tile, no atomics for signed grads ──
        // densify_abs is filled HERE with peak per-pixel |dCentre| (AbsGS). Clear first so a
        // previous view cannot leak into densify_accum after this step.
        _densifyAbs!.MemSetToZero();
        accel.FlushPendingCommands();
        // grad_per_key is written for every key this frame, so stale values cannot leak in.
        Dispatch(_rasterBackward!, _tilesX, _tilesY, new[]
        {
            Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _ranges!.GetGPUBuffer()!),
            Buf(3, _values!.GetGPUBuffer()!), Buf(4, _outFinalT!.GetGPUBuffer()!),
            Buf(5, _outEnd!.GetGPUBuffer()!), Buf(6, _dLdPix!.GetGPUBuffer()!),
            Buf(7, _gradKeyA!.GetGPUBuffer()!), Buf(8, _gradKeyB!.GetGPUBuffer()!),
            Buf(9, _gradKeyC!.GetGPUBuffer()!),
            Buf(10, _densifyAbs!.GetGPUBuffer()!),
            Buf(11, _splatColour!.GetGPUBuffer()!),
        });

        await PhaseAsync("backward");
        // Cleared here, not at the top of the step: the census and densify accum read the
        // completed step's totals after TrainStepAsync returns.
        _gradFixed!.MemSetToZero();
        accel.FlushPendingCommands();

        // ── Scatter per-key gradients into per-splat totals (f32 CAS add) ──
        WriteU32(_countBuf!, (uint)LastKeyCount);
        DispatchLinear(_scatterGrad!, LastKeyCount, new[]
        {
            Buf(0, _gradKeyA!.GetGPUBuffer()!), Buf(1, _gradKeyB!.GetGPUBuffer()!),
            Buf(2, _gradKeyC!.GetGPUBuffer()!), Buf(3, _values!.GetGPUBuffer()!),
            Buf(4, _gradFixed!.GetGPUBuffer()!), Buf(5, _countBuf!),
        });

        await PhaseAsync("scatter");
        if (TrainableVolume is { } trainable) FreezeOutside(splatGpu, splatCount, trainable);
        // ── Adam ──
        _adamStepCount++;
        WriteVec4(_adamCfgBuf!, colourLr, opacityLr, _adamStepCount, splatCount);
        WriteVec4(_adamFlagsBuf!, SkipZeroGradientSteps ? 1f : 0f, 0f, 0f, 0f);
        DispatchLinear(_adamStep!, splatCount, new[]
        {
            Buf(0, splatGpu), Buf(1, _gradFixed!.GetGPUBuffer()!),
            Buf(2, _opacityLogit!.GetGPUBuffer()!), Buf(3, _adamM!.GetGPUBuffer()!),
            Buf(4, _adamV!.GetGPUBuffer()!), Buf(5, _adamCfgBuf!), Buf(6, _adamFlagsBuf!),
        });

        await PhaseAsync("adam");
        if (ActiveShDegree >= 1 && _scatterShGrad != null && _adamShRest != null)
        {
            foreach (var g in _gradShRest) g!.MemSetToZero();
            accel.FlushPendingCommands();
            // Scatter cfg: .w = splat count (matches shader).
            WriteVec4(_adamCfgBuf!, 0f, 0f, 0f, splatCount);
            DispatchLinear(_scatterShGrad, splatCount, new[]
            {
                Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _gradFixed!.GetGPUBuffer()!),
                Buf(3, _gradShRest[0]!.GetGPUBuffer()!), Buf(4, _adamCfgBuf!),
                Buf(5, _gradShRest[1]!.GetGPUBuffer()!), Buf(6, _gradShRest[2]!.GetGPUBuffer()!),
            });
            // Rest bands use feature_lr / 20 (Kerbl). One dispatch per part (cfg.y = part: its own rounding noise);
            // Dispatch submits, so each write lands before its own dispatch.
            for (int part = 0; part < SphericalHarmonics.Parts; part++)
            {
                WriteVec4(_adamCfgBuf!, colourLr / 20f, part, _adamStepCount, splatCount);
                DispatchLinear(_adamShRest, splatCount, new[]
                {
                    Buf(0, _shRest[part]!.GetGPUBuffer()!), Buf(1, _gradShRest[part]!.GetGPUBuffer()!),
                    Buf(2, _adamShM[part]!.GetGPUBuffer()!), Buf(3, _adamShV[part]!.GetGPUBuffer()!),
                    Buf(4, _adamCfgBuf!),
                });
            }
        }

        // -- Geometry: the 2D gradients chained back to position, scale and rotation --
        // Separate dispatch, and optional, so a run can isolate whether a change came from
        // the colours or from the geometry moving.
        await PhaseAsync("sh");
        if (geometry is { } geo)
        {
            UpdateMipFloor(splatGpu, splatCount);
            WriteVec4x2(_geomCfgBuf!,
                geo.PositionLr, geo.LogScaleLr, geo.RotationLr, _adamStepCount,
                splatCount, geo.MaxScale, geo.MinScale, DenseGeometryAdam ? 1f : 0f);
            DispatchLinear(_adamGeometry!, splatCount, new[]
            {
                Buf(0, _uniformBuf!), Buf(1, splatGpu), Buf(2, _gradFixed!.GetGPUBuffer()!),
                Buf(3, _logScale!.GetGPUBuffer()!), Buf(4, _adamM!.GetGPUBuffer()!),
                Buf(5, _adamV!.GetGPUBuffer()!), Buf(6, _geomCfgBuf!),
                Buf(7, _geomOut!.GetGPUBuffer()!), Buf(8, _scaleFloor!.GetGPUBuffer()!),
            });
            // This view's camera-pose gradient, from the splat position gradients just written (poseSlot >= 0).
            if (poseSlot >= 0) DispatchPoseGrad(splatGpu, splatCount, cam.Position, poseSlot);
        }

        await PhaseAsync("geometry");
        float loss = await FinishLossAsync(readLoss);
        await PhaseAsync("loss read");
        return loss;
    }

    /// <summary>
    /// Count this step into the GPU loss sum and, when asked (or when the int32 sum is due), read it back:
    /// the mean loss per step since the last read, with <see cref="LastLossSteps"/> saying how many.
    /// Unread steps return NaN and cost no CPU-GPU round trip - the readback is the step's only wait.
    /// </summary>
    async Task<float> FinishLossAsync(bool readLoss)
    {
        _lossStepsPending++;
        if (!readLoss && _lossStepsPending < MaxLossStepsBetweenReads) return float.NaN;

        // CPU transfer: 4 bytes, the loss sum. Maps behind all the step's submitted work.
        float[] lossRaw = await _lossSum!.CopyToHostAsync<float>(0, 1);
        LastLossSteps = _lossStepsPending;
        _lossStepsPending = 0;
        return lossRaw[0] / LastLossSteps;
    }

    /// <summary>
    /// Learning rates for the geometric parameters. Position is scaled by the scene extent, as
    /// the reference does - a learning rate in world units means nothing without it. The scale
    /// bounds stop a splat that stops being constrained from collapsing to nothing or swelling
    /// to cover the frame; the reference prunes those instead, which needs density control.
    /// </summary>
    public readonly record struct GeometryStep(
        float PositionLr,
        float LogScaleLr,
        float RotationLr,
        float MinScale,
        float MaxScale);

    /// <summary>
    /// Per-splat accumulated gradients. Nine per splat, in the shader's order:
    /// colour RGB, opacity, screen centre x and y, conic a, b and c. For the GPU gate only.
    /// </summary>
    /// <remarks>
    /// Reads the WHOLE accumulator, so this is for the gate's few-hundred-splat scene, not for a
    /// real one. There used to be a maxSplats prefix option here for summary statistics; it is
    /// gone because a prefix of a VIEW-MAJOR splat buffer is not a sample - it is the top of
    /// view 0's depth map, which can be entirely zero while the buffer is full, and that is
    /// exactly how the health probe went blind. Use <see cref="ReadGradientStatsAsync"/>.
    /// </remarks>
    public async Task<float[]> ReadGradientsAsync(int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        // The buffer is f32 bit patterns in an int-typed allocation; reinterpret, do not convert.
        int[] raw = await _gradFixed!.CopyToHostAsync<int>(0, (long)splatCount * GradsPerSplat);
        var outp = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++) outp[i] = BitConverter.Int32BitsToSingle(raw[i]);
        return outp;
    }

    /// <summary>
    /// Geometry gradients from the last step: 10 per splat, position xyz then scale xyz then
    /// quaternion xyzw. What the GPU gate compares against the CPU oracle.
    /// </summary>
    public Task<float[]> ReadGeometryGradientsAsync(int splatCount) =>
        _geomOut!.CopyToHostAsync<float>(0, (long)splatCount * GeomGradsPerSplat);

    static GPUBindGroupEntry Buf(uint binding, GPUBuffer b) =>
        new() { Binding = binding, Resource = new GPUBufferBinding { Buffer = b } };

    /// <summary>
    /// Largest workgroup count WebGPU guarantees in any one dimension. Exceeding it does not
    /// clamp - the dispatch is rejected, and the driver reports it from whatever runs next.
    /// </summary>
    const int MaxWorkgroupsPerDim = 65535;

    /// <summary>
    /// Dispatch enough 64-thread workgroups to cover <paramref name="threads"/>, wrapping into
    /// a second dimension past the per-dimension limit. One dimension tops out at 4.19M threads
    /// and a room-scale scene emits more keys than that, so every key-indexed pass needs this.
    /// The shader recovers the flat index from workgroup_id and num_workgroups.
    /// </summary>
    void DispatchLinear(GPUComputePipeline pipeline, long threads, GPUBindGroupEntry[] entries, int groupSize = 64)
    {
        var (wgX, wgY) = LinearGrid(threads, groupSize);
        Dispatch(pipeline, wgX, wgY, entries);
    }

    /// <summary>Workgroup grid covering <paramref name="threads"/> at 64 per group.</summary>
    static (int X, int Y) LinearGrid(long threads, int groupSize = 64)
    {
        long groups = Math.Max(1, (threads + groupSize - 1) / groupSize);
        int x = (int)Math.Min(groups, MaxWorkgroupsPerDim);
        int y = (int)((groups + MaxWorkgroupsPerDim - 1) / MaxWorkgroupsPerDim);
        return (x, y);
    }

    void Dispatch(GPUComputePipeline pipeline, int wgX, int wgY, GPUBindGroupEntry[] entries)
    {
        using var enc = _device!.CreateCommandEncoder();
        using var pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        using var layout = pipeline.GetBindGroupLayout(0);
        using var bg = _device.CreateBindGroup(new GPUBindGroupDescriptor { Layout = layout, Entries = entries });
        pass.SetBindGroup(0, bg);
        pass.DispatchWorkgroups((uint)Math.Max(1, wgX), (uint)Math.Max(1, wgY), 1);
        pass.End();
        using var cmd = enc.Finish();
        RawSubmit.Submit(_gpu.WebGPUAccelerator, _queue!, new[] { cmd });
    }

    void WriteU32x4(GPUBuffer buf, uint x, uint y, uint z, uint w)
    {
        var v = new uint[] { x, y, z, w };
        var bytes = new byte[16];
        Buffer.BlockCopy(v, 0, bytes, 0, 16);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    void WriteVec4(GPUBuffer buf, float x, float y, float z, float w)
    {
        var f = new[] { x, y, z, w };
        var bytes = new byte[16];
        Buffer.BlockCopy(f, 0, bytes, 0, 16);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    /// <summary>Zero this step's gradient totals of the splats outside <paramref name="v"/> (see <see cref="TrainableVolume"/>).</summary>
    void FreezeOutside(GPUBuffer splatGpu, int splatCount, SplatEditor.Volume v)
    {
        _freezeCfgBuf ??= _device!.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 96,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        var f = new[]
        {
            v.M11, v.M21, v.M31, v.M41,   // dot with (x, y, z, 1) = SplatEditor.Inside's cx
            v.M12, v.M22, v.M32, v.M42,
            v.M13, v.M23, v.M33, v.M43,
            v.M14, v.M24, v.M34, v.M44,
            v.X0, v.X1, v.Y0, v.Y1,
            v.Z0, v.Z1,
        };
        // FreezeBox: 22 floats, then the count as a real u32 (never as float bits - see the shader).
        var bytes = new byte[96];
        Buffer.BlockCopy(f, 0, bytes, 0, f.Length * sizeof(float));
        BitConverter.TryWriteBytes(bytes.AsSpan(88), (uint)splatCount);
        _queue!.WriteBuffer(_freezeCfgBuf, 0, bytes);
        DispatchLinear(_freezeOutside!, splatCount, new[]
        {
            Buf(0, splatGpu), Buf(1, _gradFixed!.GetGPUBuffer()!), Buf(2, _freezeCfgBuf),
        });
    }

    void WriteVec4x2(GPUBuffer buf, float a, float b, float c, float d,
                     float e, float f, float g, float h)
    {
        var v = new[] { a, b, c, d, e, f, g, h };
        var bytes = new byte[32];
        Buffer.BlockCopy(v, 0, bytes, 0, 32);
        _queue!.WriteBuffer(buf, 0, bytes);
    }

    Vector3 _lastCamPos, _lastCamFwd;

    /// <summary>
    /// Dead-view forensics, probe only: how many splats each pixel applied (from end_idx minus
    /// the tile's range start) and who the first keys in the centre tile are. Answers "is one
    /// frame-covering splat in front of this camera" with data instead of theory.
    /// CPU transfer: end_idx + ranges (about 1.2 MB at 720x402) on the handful of dead views.
    /// </summary>
    public async Task<string> ReadDeadViewForensicsAsync(
        MemoryBuffer1D<float, Stride1D.Dense> splatBuf, int splatCount)
    {
        await _gpu.WebGPUAccelerator.SynchronizeAsync();
        long pixels = (long)_width * _height;
        uint[] end = await _outEnd!.CopyToHostAsync<uint>(0, pixels);
        uint[] ranges = await _ranges!.CopyToHostAsync<uint>(0, (long)_tilesX * _tilesY * 2);
        long c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int py = 0; py < _height; py++)
        for (int px = 0; px < _width; px++)
        {
            int tile = (py / 16) * _tilesX + (px / 16);
            long consumed = (long)end[py * _width + px] - ranges[tile * 2];
            if (consumed <= 0) c0++; else if (consumed == 1) c1++; else if (consumed == 2) c2++; else c3++;
        }
        var sb = new System.Text.StringBuilder();
        sb.Append($"consumed per pixel: 0:{c0 * 100.0 / pixels:F1}% 1:{c1 * 100.0 / pixels:F1}% ")
          .Append($"2:{c2 * 100.0 / pixels:F1}% 3+:{c3 * 100.0 / pixels:F1}%; ");

        int ct = (_tilesY / 2) * _tilesX + _tilesX / 2;
        uint r0 = ranges[ct * 2], r1 = ranges[ct * 2 + 1];
        sb.Append($"centre tile keys {r1 - r0}; first:");
        int take = (int)Math.Min(3, r1 - r0);
        if (take > 0)
        {
            uint[] ids = await _values!.CopyToHostAsync<uint>(r0, take);
            for (int i = 0; i < take; i++)
            {
                if (ids[i] >= splatCount) { sb.Append($" [#{ids[i]} OUT OF RANGE]"); continue; }
                float[] s = await splatBuf.CopyToHostAsync<float>((long)ids[i] * SplatFormat.Floats, SplatFormat.Floats);
                var rel = new Vector3(s[0], s[1], s[2]) - _lastCamPos;
                float depth = Vector3.Dot(_lastCamFwd, rel);
                sb.Append($" [#{ids[i]} depth {depth:G4} dist {rel.Length():G4} scale ({s[6]:G3},{s[7]:G3},{s[8]:G3}) opacity {s[9]:G4}]");
            }
        }
        return sb.ToString();
    }

    void WriteUniforms(CameraParams cam, float depthNear, float depthFar, int splatCount)
    {
        WorldSpaceGeometry.ViewMatrixToCameraBasis(cam.ViewMatrix, out var right, out var up, out var fwd, out var pos);
        _lastCamPos = pos; _lastCamFwd = fwd;

        var f = new float[UniformFloats];
        void V4(int o, Vector3 v, float w) { f[o] = v.X; f[o + 1] = v.Y; f[o + 2] = v.Z; f[o + 3] = w; }
        V4(0, right, 0f);
        V4(4, up, 0f);
        V4(8, fwd, 0f);
        V4(12, pos, 1f);
        f[16] = cam.FocalX; f[17] = cam.FocalY;
        f[18] = cam.CenterX; f[19] = cam.CenterY;
        f[20] = _width; f[21] = _height;
        f[22] = BitConverter.Int32BitsToSingle(_tilesX);
        f[23] = BitConverter.Int32BitsToSingle(_tilesY);
        f[24] = depthNear;
        f[25] = depthFar;
        f[26] = BitConverter.Int32BitsToSingle(splatCount);
        f[27] = BitConverter.Int32BitsToSingle(Math.Clamp(ActiveShDegree, 0, SphericalHarmonics.MaxDegree));

        Buffer.BlockCopy(f, 0, _uniformBytes!, 0, _uniformBytes!.Length);
        _queue!.WriteBuffer(_uniformBuf!, 0, _uniformBytes);
    }

    void WriteU32x2(GPUBuffer buf, uint x, uint y)
    {
        _scratch4![0] = x;
        _scratch4[1] = y;
        _scratch4[2] = 0; _scratch4[3] = 0;
        _queue!.WriteBuffer(buf, 0, _scratch4);
    }

    void WriteU32(GPUBuffer buf, uint value)
    {
        _scratch4![0] = value;
        _scratch4[1] = 0; _scratch4[2] = 0; _scratch4[3] = 0;
        _queue!.WriteBuffer(buf, 0, _scratch4);
    }

    /// <param name="keepOptimizerRows">
    /// Leave Adam m/v, SH rest and SH Adam m/v alone - they were carried to the new count by
    /// <see cref="CarryOptimizerRowsAsync"/> and the Resize that follows must not lose them.
    /// </param>
    void DisposeBuffers(bool keepOptimizerRows = false)
    {
        _keys?.Dispose(); _keys = null;
        _values?.Dispose(); _values = null;
        _counter?.Dispose(); _counter = null;
        _ranges?.Dispose(); _ranges = null;
        _outColour?.Dispose(); _outColour = null;
        _outFinalT?.Dispose(); _outFinalT = null;
        _outEnd?.Dispose(); _outEnd = null;
        _radixSort?.ReleaseScratch();
        _target?.Dispose(); _target = null;
        _dLdPix?.Dispose(); _dLdPix = null;
        _lossPartial?.Dispose(); _lossPartial = null;
        _gradKeyA?.Dispose(); _gradKeyA = null;
        _gradKeyB?.Dispose(); _gradKeyB = null;
        _gradKeyC?.Dispose(); _gradKeyC = null;
        _gradFixed?.Dispose(); _gradFixed = null;
        _densifyAbs?.Dispose(); _densifyAbs = null;
        _opacityLogit?.Dispose(); _opacityLogit = null;
        _logScale?.Dispose(); _logScale = null;
        _geomOut?.Dispose(); _geomOut = null;
        _ssePartials?.Dispose(); _ssePartials = null;
        _ssimRows?.Dispose(); _ssimRows = null;
        _ssimPartials?.Dispose(); _ssimPartials = null;
        _ssimWinGrad?.Dispose(); _ssimWinGrad = null;
        _ssimDRows?.Dispose(); _ssimDRows = null;
        _gradStatsPartials?.Dispose(); _gradStatsPartials = null;
        _fitSample?.Dispose(); _fitSample = null;
        _densifyStats?.Dispose(); _densifyStats = null;
        _screenRadius?.Dispose(); _screenRadius = null;
        _maxRadius?.Dispose(); _maxRadius = null;
        _scaleFloor?.Dispose(); _scaleFloor = null;
        _viewSupport?.Dispose(); _viewSupport = null;
        _supportPartials?.Dispose(); _supportPartials = null;
        for (int part = 0; part < SphericalHarmonics.Parts; part++) { _gradShRest[part]?.Dispose(); _gradShRest[part] = null; }
        _splatColour?.Dispose(); _splatColour = null;
        _lossSum?.Dispose(); _lossSum = null;
        if (keepOptimizerRows) return;
        _adamM?.Dispose(); _adamM = null;
        _adamV?.Dispose(); _adamV = null;
        for (int part = 0; part < SphericalHarmonics.Parts; part++)
        {
            _shRest[part]?.Dispose(); _shRest[part] = null;
            _adamShM[part]?.Dispose(); _adamShM[part] = null;
            _adamShV[part]?.Dispose(); _adamShV[part] = null;
        }
    }

    public void Dispose()
    {
        DisposeBuffers();
        // Per VIEW, not per splat: they live through every resize. In DisposeBuffers (which every densify resize runs)
        // they were freed after the first densify, and camera refinement silently stepped no camera (c19, 2026-10-04).
        _posePartials?.Dispose(); _posePartials = null;
        _poseGrads?.Dispose(); _poseGrads = null;
        _radixSort?.Dispose(); _radixSort = null;
        _uniformBuf?.Destroy(); _uniformBuf?.Dispose();
        _capsBuf?.Destroy(); _capsBuf?.Dispose();
        _geomCfgBuf?.Destroy(); _geomCfgBuf?.Dispose();
        _targetBytes?.Destroy(); _targetBytes?.Dispose();
        _countBuf?.Destroy(); _countBuf?.Dispose();
        _dimsBuf?.Destroy(); _dimsBuf?.Dispose();
        _adamCfgBuf?.Destroy(); _adamCfgBuf?.Dispose();
        _adamFlagsBuf?.Destroy(); _adamFlagsBuf?.Dispose();
        _ssimDimsBuf?.Destroy(); _ssimDimsBuf?.Dispose();
        _ssimCfgBuf?.Destroy(); _ssimCfgBuf?.Dispose();
        _lossWeightsBuf?.Destroy(); _lossWeightsBuf?.Dispose();
        _lossDimsBuf?.Destroy(); _lossDimsBuf?.Dispose();
        _scratch4?.Dispose();
        _emitKeys?.Dispose();
        _tileRanges?.Dispose();
        _rasterForward?.Dispose();
        _rasterBackward?.Dispose();
        _scatterGrad?.Dispose();
        _lossL1?.Dispose();
        _lossReduce?.Dispose();
        _ssimRowsPipe?.Dispose();
        _ssimReducePipe?.Dispose();
        _ssimWinGradPipe?.Dispose();
        _ssimRowsBwdPipe?.Dispose();
        _ssimPixBwdPipe?.Dispose();
        _adamStep?.Dispose();
        _initLogits?.Dispose();
    }
}
