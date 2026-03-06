using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// GPU-based radix sort for Gaussian splats using ILGPU.Algorithms.
/// All data stays on GPU — no CPU round-trips.
/// 
/// Uses ILGPU's built-in RadixSortPairs:
///   - O(n) complexity instead of O(n·log²n) for bitonic sort
///   - No power-of-2 padding needed — sorts exact splat count
///   - Handles millions of splats efficiently
///   - Back-to-front ordering via negated distances (ascending sort of negated = descending)
///   
/// Adaptive frame-skipping based on camera velocity reduces unnecessary sorts.
/// </summary>
public class GpuSplatSorter
{
    private readonly GpuService _gpu;
    private const int FloatsPerSplat = 10; // pos3 + color3 + scale3 + opacity1

    // ILGPU buffers — persistent across frames
    private MemoryBuffer1D<float, Stride1D.Dense>? _packedDataBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _distanceBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _indicesBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _sortedDataBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _tempBuf;

    private int _splatCount;

    // Kernel delegates
    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>>? _initIndicesKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        float, float, float, float, float, float,
        int>? _distanceKernel;

    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int>? _reorderKernel;

    // Radix sort delegate (cached after first use)
    private RadixSortPairs<float, Stride1D.Dense, int, Stride1D.Dense>? _radixSortPairs;

    // Camera tracking for dirty detection
    private Vector3 _lastCameraPos;
    private Vector3 _lastCameraFwd;
    private const float DirtyThreshold = 0.01f;

    // Adaptive sort interval: scales with camera velocity
    private int _framesSinceSort;
    private bool _sortPending;
    private float _smoothedVelocity;
    private const float VelocitySmoothing = 0.3f;

    public int SplatCount => _splatCount;

    public GpuSplatSorter(GpuService gpu) => _gpu = gpu;

    // ═══════════════════════════════════════════════════════════
    //  GPU Kernels
    // ═══════════════════════════════════════════════════════════

    /// <summary>GPU kernel: initialize indices [0, 1, 2, ...].</summary>
    private static void InitIndicesKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> indices)
    {
        indices[index] = index;
    }

    /// <summary>
    /// Compute negated distance from each splat to camera plane.
    /// Negated because we use AscendingFloat radix sort — ascending sort
    /// of negated distances gives back-to-front order for alpha blending.
    /// </summary>
    private static void ComputeDistancesKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> packedData,
        ArrayView1D<float, Stride1D.Dense> distances,
        float camPosX, float camPosY, float camPosZ,
        float camFwdX, float camFwdY, float camFwdZ,
        int splatCount)
    {
        int i = index;
        if (i >= splatCount)
        {
            // No padding needed — but guard just in case
            distances[i] = float.MaxValue;
            return;
        }
        int o = i * 10;
        float dx = packedData[o] - camPosX;
        float dy = packedData[o + 1] - camPosY;
        float dz = packedData[o + 2] - camPosZ;
        // Negate: ascending sort of negated distances = back-to-front
        distances[i] = -(dx * camFwdX + dy * camFwdY + dz * camFwdZ);
    }

    /// <summary>Reorder packed splat data using sorted indices.</summary>
    private static void ReorderKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> sortedIndices,
        ArrayView1D<float, Stride1D.Dense> srcData,
        ArrayView1D<float, Stride1D.Dense> dstData,
        int count)
    {
        int i = index;
        if (i >= count) return;
        int srcIdx = sortedIndices[i];
        int srcOff = srcIdx * 10;
        int dstOff = i * 10;
        dstData[dstOff + 0] = srcData[srcOff + 0];
        dstData[dstOff + 1] = srcData[srcOff + 1];
        dstData[dstOff + 2] = srcData[srcOff + 2];
        dstData[dstOff + 3] = srcData[srcOff + 3];
        dstData[dstOff + 4] = srcData[srcOff + 4];
        dstData[dstOff + 5] = srcData[srcOff + 5];
        dstData[dstOff + 6] = srcData[srcOff + 6];
        dstData[dstOff + 7] = srcData[srcOff + 7];
        dstData[dstOff + 8] = srcData[srcOff + 8];
        dstData[dstOff + 9] = srcData[srcOff + 9];
    }

    // ═══════════════════════════════════════════════════════════
    //  Upload + Sort
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Upload packed splat data from a GPU-resident buffer (transfer ownership, no CPU copy).
    /// The provided buffer becomes _packedDataBuf and will be disposed with this sorter.
    /// Called from the GPU fast path: DepthToGaussianKernel → GpuSplatSorter → sort → render.
    /// </summary>
    public async Task UploadFromGpuBufferAsync(
        MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)
    {
        DisposeBuffers();

        _splatCount = splatCount;
        if (_splatCount == 0) return;

        var accelerator = _gpu.WebGPUAccelerator;

        // Transfer ownership: the kernel's output buffer becomes the sorter's input buffer.
        // No GPU-to-GPU copy needed — the buffer is used in-place for distance computation
        // and the ReorderKernel reads from it to produce _sortedDataBuf.
        _packedDataBuf = packedBuf;
        _indicesBuf = accelerator.Allocate1D<int>(_splatCount);
        _distanceBuf = accelerator.Allocate1D<float>(_splatCount);
        _sortedDataBuf = accelerator.Allocate1D<float>(_splatCount * FloatsPerSplat);

        var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<
            float, int, AscendingFloat>((Index1D)_splatCount);
        _tempBuf = accelerator.Allocate1D<int>(tempSize);

        _radixSortPairs = accelerator.CreateRadixSortPairs<
            float, Stride1D.Dense,
            int, Stride1D.Dense,
            AscendingFloat>();

        await accelerator.SynchronizeAsync();

        _lastCameraPos = new Vector3(float.NaN);
        _lastCameraFwd = new Vector3(float.NaN);
        _framesSinceSort = 10;
        _sortPending = true;
        _smoothedVelocity = 0f;

        Console.WriteLine($"[GpuSorter] GPU-resident upload: {_splatCount:N0} splats (ownership transferred, zero copies)");
    }

    /// <summary>Upload scene data to GPU buffers.</summary>
    public async Task UploadAsync(GaussianScene scene)
    {
        DisposeBuffers();

        _splatCount = scene.Count;
        if (_splatCount == 0) return;

        var accelerator = _gpu.WebGPUAccelerator;

        // Pack scene data into flat float array (no power-of-2 padding!)
        var packedData = new float[_splatCount * FloatsPerSplat];
        for (int i = 0; i < _splatCount; i++)
        {
            ref var g = ref scene.Gaussians[i];
            var color = g.BaseColor;
            var scale = g.Scale;
            int o = i * FloatsPerSplat;
            packedData[o + 0] = g.Position.X; // pos X
            packedData[o + 1] = g.Position.Y; // pos Y
            packedData[o + 2] = g.Position.Z; // pos Z
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
        _distanceBuf = accelerator.Allocate1D<float>(_splatCount);
        _sortedDataBuf = accelerator.Allocate1D<float>(_splatCount * FloatsPerSplat);

        // Allocate temp buffer for radix sort
        var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<
            float, int, AscendingFloat>((Index1D)_splatCount);
        _tempBuf = accelerator.Allocate1D<int>(tempSize);

        // Create radix sort delegate (cached)
        _radixSortPairs = accelerator.CreateRadixSortPairs<
            float, Stride1D.Dense,
            int, Stride1D.Dense,
            AscendingFloat>();

        await accelerator.SynchronizeAsync();

        _lastCameraPos = new Vector3(float.NaN);
        _lastCameraFwd = new Vector3(float.NaN);
        _framesSinceSort = 10; // Force immediate sort on first frame
        _sortPending = true;
        _smoothedVelocity = 0f;

        Console.WriteLine($"[GpuSorter] Uploaded {_splatCount:N0} splats (no padding!), " +
            $"{packedData.Length * 4 / 1024}KB data, {tempSize * 4 / 1024}KB temp for radix sort");
    }

    /// <summary>Sort splats by camera distance on GPU using radix sort. Returns sorted buffer.</summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>?> SortAsync(CameraParams camera)
    {
        if (_splatCount == 0 || _packedDataBuf == null || _indicesBuf == null)
            return _sortedDataBuf;

        // Dirty check + velocity tracking
        var camPos = camera.Position;
        var camFwd = camera.Forward;
        float currentVelocity = 0f;

        if (!float.IsNaN(_lastCameraPos.X))
        {
            float posDelta = Vector3.DistanceSquared(camPos, _lastCameraPos);
            float fwdDelta = Vector3.DistanceSquared(camFwd, _lastCameraFwd);
            currentVelocity = posDelta + fwdDelta;

            if (currentVelocity >= DirtyThreshold)
                _sortPending = true;
        }
        else
        {
            _sortPending = true;
        }

        // Exponential moving average to smooth velocity
        _smoothedVelocity = _smoothedVelocity * (1f - VelocitySmoothing) + currentVelocity * VelocitySmoothing;

        // Adaptive sort interval: fast movement → sort every frame, slow → every 5 frames
        int sortInterval;
        if (_smoothedVelocity > 0.1f) sortInterval = 1;       // fast movement
        else if (_smoothedVelocity > 0.01f) sortInterval = 2;  // moderate
        else sortInterval = 5;                                  // slow/still

        // Frame-skip with adaptive interval
        _framesSinceSort++;
        if (!_sortPending || _framesSinceSort < sortInterval)
            return _sortedDataBuf;

        _framesSinceSort = 0;
        _sortPending = false;
        _lastCameraPos = camPos;
        _lastCameraFwd = camFwd;

        var accelerator = _gpu.WebGPUAccelerator;

        // Load kernels (cached after first call)
        _initIndicesKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>>(InitIndicesKernel);

        _distanceKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            float, float, float, float, float, float,
            int>(ComputeDistancesKernel);

        _reorderKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(ReorderKernel);

        // ── Step 1: Initialize indices [0..N-1]
        _initIndicesKernel(_splatCount, _indicesBuf.View);

        // ── Step 2: Compute negated distances (ascending sort of negated = back-to-front)
        _distanceKernel(_splatCount,
            _packedDataBuf.View, _distanceBuf!.View,
            camPos.X, camPos.Y, camPos.Z,
            camFwd.X, camFwd.Y, camFwd.Z,
            _splatCount);

        // ── Step 3: Radix sort pairs (distances as keys, indices as values)
        // This is O(n) and sorts the exact count — no power-of-2 padding!
        _radixSortPairs!(
            accelerator.DefaultStream,
            _distanceBuf!.View,
            _indicesBuf.View,
            _tempBuf!.View);

        // ── Step 4: Reorder data using sorted indices
        _reorderKernel(_splatCount,
            _indicesBuf.View,
            _packedDataBuf.View,
            _sortedDataBuf!.View,
            _splatCount);

        // Single sync at end
        await accelerator.SynchronizeAsync();

        return _sortedDataBuf;
    }

    private void DisposeBuffers()
    {
        _packedDataBuf?.Dispose(); _packedDataBuf = null;
        _distanceBuf?.Dispose(); _distanceBuf = null;
        _indicesBuf?.Dispose(); _indicesBuf = null;
        _sortedDataBuf?.Dispose(); _sortedDataBuf = null;
        _tempBuf?.Dispose(); _tempBuf = null;
    }

    public void Dispose()
    {
        DisposeBuffers();
    }
}
