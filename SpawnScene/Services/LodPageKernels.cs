using ILGPU;
using ILGPU.Runtime;

namespace SpawnScene.Services;

/// <summary>
/// The kernels <see cref="GpuLodPager"/> fills its pool with, on their own so the tests run them on the ILGPU CPU
/// accelerator: a page slot's parent SLOT and children's chunk from a chunk's file indices.
/// </summary>
public static class LodPageKernels
{
    public struct SlotParams
    {
        public int Slot0, PageNodes, Chunks;
    }

    public static void FillKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> dst, int offset, int value) => dst[offset + i] = value;

    /// <summary>
    /// Chunk node j into slot Slot0 + j: its parent as a SLOT (the parent's chunk is resident - loaded first - so its
    /// page is known), its children's chunk, sphere and LOD size. Chunks are found by binary search over their starts.
    /// </summary>
    public static void SlotKernel(Index1D j, ArrayView1D<int, Stride1D.Dense> parent, ArrayView1D<int, Stride1D.Dense> firstChild,
        ArrayView1D<float, Stride1D.Dense> bounds, ArrayView1D<float, Stride1D.Dense> size, ArrayView1D<int, Stride1D.Dense> starts,
        ArrayView1D<int, Stride1D.Dense> chunkPage, ArrayView1D<int, Stride1D.Dense> outParentSlot,
        ArrayView1D<int, Stride1D.Dense> outChildChunk, ArrayView1D<float, Stride1D.Dense> outBounds,
        ArrayView1D<float, Stride1D.Dense> outSize, SlotParams sp)
    {
        int slot = sp.Slot0 + j;
        int p = parent[j];
        if (p < 0) outParentSlot[slot] = -1;
        else
        {
            int pc = ChunkOf(starts, sp.Chunks, p);
            outParentSlot[slot] = chunkPage[pc] * sp.PageNodes + (p - starts[pc]);
        }
        int fc = firstChild[j];
        outChildChunk[slot] = fc < 0 ? -1 : ChunkOf(starts, sp.Chunks, fc);
        for (int k = 0; k < 4; k++) outBounds[slot * 4 + k] = bounds[j * 4 + k];
        outSize[slot] = size[j];
    }

    /// <summary>The chunk holding node <paramref name="node"/>: the last start at or below it.</summary>
    public static int ChunkOf(ArrayView1D<int, Stride1D.Dense> starts, int chunks, int node)
    {
        int lo = 0, hi = chunks - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (starts[mid] <= node) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

}
