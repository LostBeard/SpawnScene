using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SpawnScene.Services;

// ═══════════════════════════════════════════════════════════
//  16-bit RadixSort operation — half as many passes as DescendingInt32.
//  Uses the lower 16 bits of the DescendingInt32 transform.
//  Requires depth keys in [0..65534]; int.MinValue sentinel works.
//  Halves sort GPU time for Standard/Fast quality modes.
// ═══════════════════════════════════════════════════════════
public readonly struct DescendingInt16As32 : IRadixSortOperation<int>
{
    public int NumBits => 16;  // Half the passes of NumBits=32
    public int DefaultValue => 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int ExtractRadixBits(int value, int shift, int bitMask)
    {
        // Identical to DescendingInt32 — only the lower 16 bits are examined (NumBits=16).
        // Correct for depths in [0..65534]; int.MinValue (culled) sorts last. ✓
        AscendingInt32 operation = default;
        return (~operation.ExtractRadixBits(value, shift, bitMask)) & bitMask;
    }
}

/// <summary>
/// GPU-based radix sort for Gaussian splats using ILGPU.Algorithms.
/// Zero CPU readback per frame — all data stays on GPU.
///
/// Pipeline per sort frame:
///   1. CullAndDistanceKernel: writes int32 depth key to each of N slots.
///      Visible splats → quantized depth + original index.
///      Culled splats  → int.MinValue + -1 (sentinel, sorts LAST).
///   2. RadixSort on ALL N pairs — visible (≥0) before culled (int.MinValue).
///      High quality: DescendingInt32 (8 passes, 32-bit precision).
///      Standard/Fast: DescendingInt16As32 (4 passes, 16-bit precision).
///   3. Pack shader checks idx &lt; 0 → writes transparent vertex, fragment shader discards.
/// </summary>
public class GpuSplatSorter : IDisposable
{
    private readonly GpuService _gpu;
    private const int FloatsPerSplat = 10; // pos3 + color3 + scale3 + opacity1

    // ILGPU buffers — persistent across frames
    private MemoryBuffer1D<float, Stride1D.Dense>? _packedDataBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _distanceBuf;   // int32 depth keys (int.MinValue = culled)
    private MemoryBuffer1D<int, Stride1D.Dense>? _indicesBuf;    // splat indices (-1 = culled sentinel)
    private MemoryBuffer1D<int, Stride1D.Dense>? _tempBuf;       // radix sort temp storage

    private int _splatCount;
    private int _lastSortVisibleCount; // count from last sort (for non-sort frames)

    // Kernel delegate — frustum cull + distance in one pass
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // packedData
        ArrayView1D<int, Stride1D.Dense>,    // outDistances
        ArrayView1D<int, Stride1D.Dense>,    // outIndices
        CullParams>? _cullDistanceKernel;

    // Radix sort delegates — 32-bit (8 passes, High quality) and 16-bit (4 passes, Standard/Fast)
    private RadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense>? _radixSortPairs32;
    private RadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense>? _radixSortPairs16;

    /// <summary>
    /// When true, uses 16-bit depth sort (4 GPU passes instead of 8).
    /// Precision is sufficient for Gaussian splatting alpha blending.
    /// Set true for Standard/Fast presets, false for High.
    /// </summary>
    public bool Use16BitSort { get; set; } = true;

    // Camera tracking
    // _prevFrameCameraPos/Fwd: updated every Sort() call → accurate per-frame velocity for _smoothedVelocity
    private Vector3 _prevFrameCameraPos;
    private Vector3 _prevFrameCameraFwd;

    // Async sort tracking — prevents GPU queue backpressure
    // _syncTask: non-null while GPU sort is in flight; polled via IsCompleted (non-blocking)
    // _lastSortTicks: Stopwatch timestamp of last sort submission (50ms min between sorts)
    private Task? _syncTask;
    private long _lastSortTicks;
    private bool _sortPending;
    private float _smoothedVelocity;
    private const float VelocitySmoothing = 0.3f;

    public int SplatCount => _splatCount;
    public float SmoothedVelocity => _smoothedVelocity;

    public GpuSplatSorter(GpuService gpu) => _gpu = gpu;

    // ═══════════════════════════════════════════════════════════
    //  CullParams — all kernel scalars packed into one struct.
    //  ILGPU decomposes struct fields into the _scalar_params
    //  storage binding, keeping ArrayView binding count low.
    // ═══════════════════════════════════════════════════════════

    public struct CullParams
    {
        public float CamPosX, CamPosY, CamPosZ;
        public float CamFwdX, CamFwdY, CamFwdZ;
        // 6 frustum planes in world space: a*x + b*y + c*z + d >= 0 for inside points
        public float P0x, P0y, P0z, P0d; // Left
        public float P1x, P1y, P1z, P1d; // Right
        public float P2x, P2y, P2z, P2d; // Bottom
        public float P3x, P3y, P3z, P3d; // Top
        public float P4x, P4y, P4z, P4d; // Near
        public float P5x, P5y, P5z, P5d; // Far
        public int SplatCount;
        // Depth quantization: 10000f / int.MaxValue for 32-bit, 500f / 65534 for 16-bit
        public float DistScale;
        public int DistMax;
    }

    // ═══════════════════════════════════════════════════════════
    //  GPU Kernels
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// GPU kernel: frustum-cull splats and assign int32 depth keys.
    /// Writes to ALL N slots (indexed by splat index, no compaction):
    ///   Visible: outDistances[i] = quantized depth (≥ 0), outIndices[i] = i
    ///   Culled:  outDistances[i] = int.MinValue (-1 sentinel), outIndices[i] = -1
    /// After DescendingInt32 sort: visible splats (depth ≥ 0) come first, culled (int.MinValue) last.
    /// The pack shader checks outIndices[i] &lt; 0 to skip culled slots.
    /// No Atomic.Add, no counter readback, no CPU→GPU sync mid-frame.
    /// </summary>
    private static void CullAndDistanceKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packedData,
        ArrayView1D<int, Stride1D.Dense> outDistances,
        ArrayView1D<int, Stride1D.Dense> outIndices,
        CullParams p)
    {
        int i = index;
        if (i >= p.SplatCount) return;

        int o = i * 10;
        float x = packedData[o];
        float y = packedData[o + 1];
        float z = packedData[o + 2];
        float opacity = packedData[o + 9];

        // Frustum cull with per-splat radius margin + zero-opacity rejection.
        float margin = packedData[o + 6] * 3f;
        bool visible = opacity > 0f
            && x * p.P0x + y * p.P0y + z * p.P0z + p.P0d >= -margin  // Left
            && x * p.P1x + y * p.P1y + z * p.P1z + p.P1d >= -margin  // Right
            && x * p.P2x + y * p.P2y + z * p.P2z + p.P2d >= -margin  // Bottom
            && x * p.P3x + y * p.P3y + z * p.P3z + p.P3d >= -margin  // Top
            && x * p.P4x + y * p.P4y + z * p.P4z + p.P4d >= -margin  // Near
            && x * p.P5x + y * p.P5y + z * p.P5z + p.P5d >= -margin; // Far

        if (visible)
        {
            float dx = x - p.CamPosX;
            float dy = y - p.CamPosY;
            float dz = z - p.CamPosZ;
            float dist = dx * p.CamFwdX + dy * p.CamFwdY + dz * p.CamFwdZ;
            // Quantize depth; clamp to DistMax so 16-bit sort wraps are avoided.
            int qDist = dist > 0f ? (int)(dist * p.DistScale) : 0;
            outDistances[i] = qDist > p.DistMax ? p.DistMax : qDist;
            outIndices[i] = i;
        }
        else
        {
            // Sentinel: int.MinValue sorts LAST in DescendingInt32.
            // Pack shader skips idx < 0 → no wasted vertex/fragment work.
            outDistances[i] = int.MinValue;
            outIndices[i] = -1;
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Frustum Plane Extraction — Gribb-Hartmann method
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Extracts 6 normalized world-space frustum planes from the MVP matrix.
    /// Uses row-vector convention (C# System.Numerics): clip = worldPos * MVP.
    /// Plane equation: a*x + b*y + c*z + d >= 0 for points inside the frustum.
    /// WebGPU clip-space Z=[0,1]: near=row3, far=row4-row3, others use row4±rowN.
    /// </summary>
    internal static CullParams BuildCullParams(Matrix4x4 mvp, Vector3 camPos, Vector3 camFwd, int splatCount,
        float distScale = 10000f, int distMax = int.MaxValue)
    {
        // GpuMatrix4x4.FromMatrix4x4 transposes .NET's row-major (v*M) to GPU column-major (M*v).
        // GPU row i = .NET column i, giving clip = gm * worldPos (column-vector convention).
        // Gribb-Hartmann uses the ROWS of this GPU matrix:
        //   gm.row0 → clip.x,  gm.row1 → clip.y,  gm.row2 → clip.z,  gm.row3 → clip.w
        var gm = GpuMatrix4x4.FromMatrix4x4(mvp);

        // Read GPU rows (each row encodes one clip-space component)
        float r0x = gm.R00, r0y = gm.R01, r0z = gm.R02, r0w = gm.R03; // clip.x axis
        float r1x = gm.R10, r1y = gm.R11, r1z = gm.R12, r1w = gm.R13; // clip.y axis
        float r2x = gm.R20, r2y = gm.R21, r2z = gm.R22, r2w = gm.R23; // clip.z axis
        float r3x = gm.R30, r3y = gm.R31, r3z = gm.R32, r3w = gm.R33; // clip.w axis

        var p = new CullParams
        {
            CamPosX = camPos.X, CamPosY = camPos.Y, CamPosZ = camPos.Z,
            CamFwdX = camFwd.X, CamFwdY = camFwd.Y, CamFwdZ = camFwd.Z,
            SplatCount = splatCount,
            DistScale = distScale,
            DistMax = distMax,
        };

        static void SetPlane(ref float px, ref float py, ref float pz, ref float pd,
            float ax, float ay, float az, float aw)
        {
            float len = MathF.Sqrt(ax * ax + ay * ay + az * az);
            if (len > 1e-8f) { ax /= len; ay /= len; az /= len; aw /= len; }
            px = ax; py = ay; pz = az; pd = aw;
        }

        // Gribb-Hartmann planes (column-vector convention, WebGPU Z=[0,1])
        // plane test: a*x + b*y + c*z + d >= 0 for points inside the frustum
        SetPlane(ref p.P0x, ref p.P0y, ref p.P0z, ref p.P0d, r3x + r0x, r3y + r0y, r3z + r0z, r3w + r0w); // Left   (clip.x + clip.w >= 0)
        SetPlane(ref p.P1x, ref p.P1y, ref p.P1z, ref p.P1d, r3x - r0x, r3y - r0y, r3z - r0z, r3w - r0w); // Right  (-clip.x + clip.w >= 0)
        SetPlane(ref p.P2x, ref p.P2y, ref p.P2z, ref p.P2d, r3x + r1x, r3y + r1y, r3z + r1z, r3w + r1w); // Bottom (clip.y + clip.w >= 0)
        SetPlane(ref p.P3x, ref p.P3y, ref p.P3z, ref p.P3d, r3x - r1x, r3y - r1y, r3z - r1z, r3w - r1w); // Top    (-clip.y + clip.w >= 0)
        SetPlane(ref p.P4x, ref p.P4y, ref p.P4z, ref p.P4d, r2x,        r2y,        r2z,        r2w);        // Near   (clip.z >= 0)
        SetPlane(ref p.P5x, ref p.P5y, ref p.P5z, ref p.P5d, r3x - r2x, r3y - r2y, r3z - r2z, r3w - r2w); // Far    (clip.w - clip.z >= 0)
        return p;
    }

    // ═══════════════════════════════════════════════════════════
    //  Upload
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Upload packed splat data from a GPU-resident buffer (transfer ownership, no CPU copy).
    /// </summary>
    public async Task UploadFromGpuBufferAsync(MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)
    {
        DisposeBuffers();

        _splatCount = splatCount;
        if (_splatCount == 0) return;

        var accelerator = _gpu.WebGPUAccelerator;

        _packedDataBuf = packedBuf;
        _indicesBuf = accelerator.Allocate1D<int>(_splatCount);
        _distanceBuf = accelerator.Allocate1D<int>(_splatCount);

        var tempSize32 = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, DescendingInt32>((Index1D)_splatCount);
        var tempSize16 = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, DescendingInt16As32>((Index1D)_splatCount);
        _tempBuf = accelerator.Allocate1D<int>(Math.Max(tempSize32, tempSize16));

        _radixSortPairs32 = accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, DescendingInt32>();
        _radixSortPairs16 = accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, DescendingInt16As32>();

        await accelerator.SynchronizeAsync();

        _lastSortVisibleCount = splatCount;
        ResetSortState();
        Console.WriteLine($"[GpuSorter] GPU-resident upload: {_splatCount:N0} splats (ownership transferred, zero copies)");
    }

    /// <summary>Upload scene data from CPU to GPU buffers.</summary>
    public async Task UploadAsync(GaussianScene scene)
    {
        DisposeBuffers();

        _splatCount = scene.Count;
        if (_splatCount == 0) return;

        var accelerator = _gpu.WebGPUAccelerator;

        var packedData = new float[_splatCount * FloatsPerSplat];
        for (int i = 0; i < _splatCount; i++)
        {
            ref var g = ref scene.Gaussians[i];
            var color = g.BaseColor;
            var scale = g.Scale;
            int o = i * FloatsPerSplat;
            packedData[o + 0] = g.Position.X;
            packedData[o + 1] = g.Position.Y;
            packedData[o + 2] = g.Position.Z;
            packedData[o + 3] = Math.Clamp(color.X, 0f, 1f);
            packedData[o + 4] = Math.Clamp(color.Y, 0f, 1f);
            packedData[o + 5] = Math.Clamp(color.Z, 0f, 1f);
            packedData[o + 6] = scale.X;
            packedData[o + 7] = scale.Y;
            packedData[o + 8] = scale.Z;
            packedData[o + 9] = g.Opacity;
        }

        _packedDataBuf = accelerator.Allocate1D(packedData);
        _indicesBuf = accelerator.Allocate1D<int>(_splatCount);
        _distanceBuf = accelerator.Allocate1D<int>(_splatCount);

        var tempSize32 = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, DescendingInt32>((Index1D)_splatCount);
        var tempSize16 = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, DescendingInt16As32>((Index1D)_splatCount);
        int tempSize = Math.Max(tempSize32, tempSize16);
        _tempBuf = accelerator.Allocate1D<int>(tempSize);

        _radixSortPairs32 = accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, DescendingInt32>();
        _radixSortPairs16 = accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, DescendingInt16As32>();

        await accelerator.SynchronizeAsync();

        _lastSortVisibleCount = _splatCount;
        ResetSortState();
        Console.WriteLine($"[GpuSorter] Uploaded {_splatCount:N0} splats, {tempSize * 4 / 1024}KB radix temp");
    }

    private void ResetSortState()
    {
        _prevFrameCameraPos = new Vector3(float.NaN);
        _prevFrameCameraFwd = new Vector3(float.NaN);
        _syncTask = null;    // Abandon any in-flight sort (buffers being disposed)
        _lastSortTicks = 0;  // Allow immediate first sort (now - 0 >> 50ms)
        _sortPending = true;
        _smoothedVelocity = 0f;
    }

    // ═══════════════════════════════════════════════════════════
    //  Sort
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Frustum-cull and radix-sort splats by camera depth. Fully synchronous — no GPU readback.
    /// Returns (packedDataBuf, sortedIndicesBuf, sortRan, visibleCount).
    /// sortRan=true means the vertex buffer must be repacked this frame.
    /// sortRan=false means the caller can skip pack (vertex buffer is still valid from last sort).
    /// visibleCount = _splatCount (all slots, culled sentinels are discarded by pack/vert shaders).
    /// Culled splats get sentinel keys (int.MinValue / idx=-1) and sort LAST in DescendingInt32.
    /// </summary>
    public (MemoryBuffer1D<float, Stride1D.Dense>?, MemoryBuffer1D<int, Stride1D.Dense>?, bool sortRan, int visibleCount)
        Sort(CameraParams camera, Matrix4x4 mvp)
    {
        if (_splatCount == 0 || _packedDataBuf == null || _indicesBuf == null)
            return (_packedDataBuf, _indicesBuf, false, _lastSortVisibleCount);

        var camPos = camera.Position;
        var camFwd = camera.Forward;

        // Per-frame velocity: measured against previous frame for accurate _smoothedVelocity.
        float currentVelocity = 0f;
        if (!float.IsNaN(_prevFrameCameraPos.X))
        {
            float posDelta = Vector3.DistanceSquared(camPos, _prevFrameCameraPos);
            float fwdDelta = Vector3.DistanceSquared(camFwd, _prevFrameCameraFwd);
            currentVelocity = posDelta + fwdDelta;
        }
        _prevFrameCameraPos = camPos;
        _prevFrameCameraFwd = camFwd;

        // ── In-flight GPU sort check ──
        // Only one sort is submitted at a time. While the GPU is sorting, we render with
        // the stale vertex buffer — no queue backpressure, no blocking CPU wait.
        bool sortJustDone = false;
        if (_syncTask != null)
        {
            if (!_syncTask.IsCompleted)
            {
                // Sort still running — update velocity for adaptive-res, return stale results.
                _smoothedVelocity = _smoothedVelocity * (1f - VelocitySmoothing) + currentVelocity * VelocitySmoothing;
                return (_packedDataBuf, _indicesBuf, false, _lastSortVisibleCount);
            }
            // Sort completed this frame — signal caller to repack with new indices.
            // Do NOT start a new sort this same frame: the pack pass (submitted after Sort()
            // returns) and a new sort (which would write _indicesBuf) would conflict in the queue.
            _syncTask = null;
            sortJustDone = true;
        }

        _smoothedVelocity = _smoothedVelocity * (1f - VelocitySmoothing) + currentVelocity * VelocitySmoothing;

        if (currentVelocity > 1e-8f)
            _sortPending = true;

        // On the frame sort just completed: return sortRan=true so pack runs, skip new sort.
        if (sortJustDone)
            return (_packedDataBuf, _indicesBuf, true, _lastSortVisibleCount);

        // ── Rate gate: 50ms minimum between sort submissions ──
        // Prevents thrashing on fast GPUs. On slow GPUs, _syncTask completion is the natural limiter.
        long now = Stopwatch.GetTimestamp();
        if (!_sortPending || (now - _lastSortTicks) < Stopwatch.Frequency / 20)
            return (_packedDataBuf, _indicesBuf, false, _lastSortVisibleCount);

        _sortPending = false;
        _lastSortTicks = now;

        var accelerator = _gpu.WebGPUAccelerator;

        _cullDistanceKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            CullParams>(CullAndDistanceKernel);

        // ── Step 1: Frustum cull + depth quantize (all N slots, sentinels for culled splats) ──
        float distScale = Use16BitSort ? 500f : 10000f;
        int distMax = Use16BitSort ? 65534 : int.MaxValue;
        var cullParams = BuildCullParams(mvp, camPos, camFwd, _splatCount, distScale, distMax);
        _cullDistanceKernel(
            _splatCount,
            _packedDataBuf.View,
            _distanceBuf!.View,
            _indicesBuf.View,
            cullParams);

        // ── Step 2: Radix sort ALL N pairs ──
        // High: DescendingInt32 (8 passes) — full 32-bit depth precision.
        // Standard/Fast: DescendingInt16As32 (4 passes) — 16-bit precision, 2x faster.
        // Visible depths (≥ 0) sort before int.MinValue sentinels → back-to-front, culled last.
        if (Use16BitSort)
            _radixSortPairs16!(accelerator.DefaultStream, _distanceBuf.View, _indicesBuf.View, _tempBuf!.View);
        else
            _radixSortPairs32!(accelerator.DefaultStream, _distanceBuf.View, _indicesBuf.View, _tempBuf!.View);

        // Non-blocking async wait — RAF loop continues at full rate while GPU sorts.
        // _syncTask.IsCompleted is polled each frame; sortRan=true fires on completion frame.
        _syncTask = accelerator.DefaultStream.SynchronizeAsync();

        _lastSortVisibleCount = _splatCount;
        return (_packedDataBuf, _indicesBuf, false, _splatCount);
    }

    private void DisposeBuffers()
    {
        _packedDataBuf?.Dispose(); _packedDataBuf = null;
        _distanceBuf?.Dispose(); _distanceBuf = null;
        _indicesBuf?.Dispose(); _indicesBuf = null;
        _tempBuf?.Dispose(); _tempBuf = null;
        _radixSortPairs32 = null;
        _radixSortPairs16 = null;
        _lastSortVisibleCount = 0;
    }

    public void Dispose() => DisposeBuffers();
}
