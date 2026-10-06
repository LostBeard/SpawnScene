using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The level-of-detail tree built on the GPU (Plans/lod-streaming.md) - <see cref="LodTree"/>'s algorithm, with the
/// scene's splats never leaving the device. Per level: every frontier node gets its grid cell as a 64-bit key (or a
/// "wait" key when it is larger than the cell), two stable 32-bit sorts order the nodes by cell, and one thread per cell
/// merges the cell's nodes into a parent (float moments centred on the first child, Jacobi eigen to scale + rotation)
/// or passes a lone node up. The sort is injected: <see cref="GpuRadixSort"/> on WebGPU, a host sort in the CPU tests.
/// </summary>
public sealed class GpuLodTree : IDisposable
{
    const int F = SplatFormat.Floats;
    const uint WaitKey = 0xFFFFFFFFu;
    const int AxisBits = 21;
    const int AxisBias = 1 << 20;

    /// <summary>Sort <c>count</c> u32 keys ascending, stably, carrying the u32 values.</summary>
    public delegate void SortPairs(MemoryBuffer1D<uint, Stride1D.Dense> keys, MemoryBuffer1D<uint, Stride1D.Dense> values, int count);

    /// <summary>Every node: leaves first (the input rows), then parents. Capacity 2 x leaves.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Nodes = null!;
    public MemoryBuffer1D<int, Stride1D.Dense> Parent = null!;
    /// <summary>Bounding sphere of each subtree: centre xyz + radius.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Bounds = null!;
    public MemoryBuffer1D<float, Stride1D.Dense> LodSize = null!;
    /// <summary>Children of node p: ChildList[FirstChild[p] .. + ChildCount[p]].</summary>
    public MemoryBuffer1D<int, Stride1D.Dense> FirstChild = null!, ChildCount = null!, ChildList = null!;
    public int LeafCount, NodeCount, Levels;

    /// <summary>A breadth-first layout's depth ranges (GpuLodLayout): depth d is nodes DepthStarts[d] .. DepthStarts[d+1]-1.</summary>
    public int[]? DepthStarts;

    // Counters: [0] next node id (absolute), [1] next frontier length, [2] child-list fill.
    MemoryBuffer1D<int, Stride1D.Dense>? _counters;

    public void Dispose()
    {
        Nodes?.Dispose(); Parent?.Dispose(); Bounds?.Dispose(); LodSize?.Dispose();
        FirstChild?.Dispose(); ChildCount?.Dispose(); ChildList?.Dispose(); _counters?.Dispose();
    }

    // ── scalar math (float, so it runs in WGSL) ──────────────────────────────────────────────────────────────

    static float Area(float sx, float sy, float sz)
    {
        float a = XMath.Max(sx, XMath.Max(sy, sz));
        float c = XMath.Min(sx, XMath.Min(sy, sz));
        return 3.14159265f * a * (sx + sy + sz - a - c);
    }

    static float SizeOf(ArrayView1D<float, Stride1D.Dense> rows, long o) =>
        2f * XMath.Max(rows[o + 6], XMath.Max(rows[o + 7], rows[o + 8]));

    /// <summary>One Jacobi rotation (float): zero a_pq, rotate the coupled entries and V's columns p, q.</summary>
    static void Jacobi(ref float app, ref float aqq, ref float apq, ref float apr, ref float aqr,
        ref float v0p, ref float v0q, ref float v1p, ref float v1q, ref float v2p, ref float v2q)
    {
        if (XMath.Abs(apq) < 1e-30f) return;
        float theta = (aqq - app) / (2f * apq);
        float t = (theta >= 0f ? 1f : -1f) / (XMath.Abs(theta) + XMath.Sqrt(theta * theta + 1f));
        float cs = 1f / XMath.Sqrt(t * t + 1f), sn = t * cs;
        float nApp = app - t * apq, nAqq = aqq + t * apq;
        float nApr = cs * apr - sn * aqr, nAqr = sn * apr + cs * aqr;
        app = nApp; aqq = nAqq; apq = 0f; apr = nApr; aqr = nAqr;
        float t0 = v0p, t1 = v1p, t2 = v2p;
        v0p = cs * t0 - sn * v0q; v0q = sn * t0 + cs * v0q;
        v1p = cs * t1 - sn * v1q; v1q = sn * t1 + cs * v1q;
        v2p = cs * t2 - sn * v2q; v2q = sn * t2 + cs * v2q;
    }

    /// <summary>Covariance -> 1-sigma scales and rotation quaternion, written into row <paramref name="o"/>.</summary>
    public static void WriteScaleQuat(float a00, float a01, float a02, float a11, float a12, float a22,
        ArrayView1D<float, Stride1D.Dense> rows, long o)
    {
        float v00 = 1f, v01 = 0f, v02 = 0f, v10 = 0f, v11 = 1f, v12 = 0f, v20 = 0f, v21 = 0f, v22 = 1f;
        for (int sweep = 0; sweep < 8; sweep++)
        {
            Jacobi(ref a00, ref a11, ref a01, ref a02, ref a12, ref v00, ref v01, ref v10, ref v11, ref v20, ref v21);
            Jacobi(ref a00, ref a22, ref a02, ref a01, ref a12, ref v00, ref v02, ref v10, ref v12, ref v20, ref v22);
            Jacobi(ref a11, ref a22, ref a12, ref a01, ref a02, ref v01, ref v02, ref v11, ref v12, ref v21, ref v22);
        }
        float det = v00 * (v11 * v22 - v12 * v21) - v01 * (v10 * v22 - v12 * v20) + v02 * (v10 * v21 - v11 * v20);
        if (det < 0f) { v02 = -v02; v12 = -v12; v22 = -v22; }
        rows[o + 6] = XMath.Sqrt(XMath.Max(a00, 1e-24f));
        rows[o + 7] = XMath.Sqrt(XMath.Max(a11, 1e-24f));
        rows[o + 8] = XMath.Sqrt(XMath.Max(a22, 1e-24f));
        // Shepperd: matrix (columns = local axes) -> quaternion x y z w
        float tr = v00 + v11 + v22, qx, qy, qz, qw;
        if (tr > 0f)
        {
            float s = XMath.Sqrt(tr + 1f) * 2f;
            qw = 0.25f * s; qx = (v21 - v12) / s; qy = (v02 - v20) / s; qz = (v10 - v01) / s;
        }
        else if (v00 > v11 && v00 > v22)
        {
            float s = XMath.Sqrt(1f + v00 - v11 - v22) * 2f;
            qw = (v21 - v12) / s; qx = 0.25f * s; qy = (v01 + v10) / s; qz = (v02 + v20) / s;
        }
        else if (v11 > v22)
        {
            float s = XMath.Sqrt(1f + v11 - v00 - v22) * 2f;
            qw = (v02 - v20) / s; qx = (v01 + v10) / s; qy = 0.25f * s; qz = (v12 + v21) / s;
        }
        else
        {
            float s = XMath.Sqrt(1f + v22 - v00 - v11) * 2f;
            qw = (v10 - v01) / s; qx = (v02 + v20) / s; qy = (v12 + v21) / s; qz = 0.25f * s;
        }
        float qn = XMath.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        rows[o + 10] = qx / qn; rows[o + 11] = qy / qn; rows[o + 12] = qz / qn; rows[o + 13] = qw / qn;
    }

    // ── kernels ───────────────────────────────────────────────────────────────────────────────────────────────

    static void InitKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> nodes, ArrayView1D<int, Stride1D.Dense> parent,
        ArrayView1D<float, Stride1D.Dense> bounds, ArrayView1D<float, Stride1D.Dense> lodSize,
        ArrayView1D<int, Stride1D.Dense> childCount, ArrayView1D<int, Stride1D.Dense> frontier, int n)
    {
        if (i >= n) return;
        long o = (long)i.X * F;
        float s = SizeOf(nodes, o);
        parent[i] = -1; childCount[i] = 0; lodSize[i] = s; frontier[i] = i;
        bounds[i * 4] = nodes[o]; bounds[i * 4 + 1] = nodes[o + 1]; bounds[i * 4 + 2] = nodes[o + 2]; bounds[i * 4 + 3] = 1.5f * s;
    }

    static uint Axis(float v, float step, float origin)
    {
        long c = (long)XMath.Floor((v - origin) / step) + AxisBias;
        c = c < 0 ? 0 : c > (1 << AxisBits) - 1 ? (1 << AxisBits) - 1 : c;
        return (uint)c;
    }

    static void KeyKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> frontier, ArrayView1D<float, Stride1D.Dense> nodes,
        ArrayView1D<float, Stride1D.Dense> lodSize, ArrayView1D<uint, Stride1D.Dense> keyLo, ArrayView1D<uint, Stride1D.Dense> keyHi,
        ArrayView1D<uint, Stride1D.Dense> sortLo, ArrayView1D<uint, Stride1D.Dense> perm, int m, float step, float origin, int lastLevel)
    {
        if (i >= m) return;
        int node = frontier[i];
        perm[i] = (uint)i.X;
        if (lastLevel == 0 && lodSize[node] > step) { keyLo[i] = WaitKey; keyHi[i] = WaitKey; sortLo[i] = WaitKey; return; }
        long o = (long)node * F;
        ulong k = Axis(nodes[o], step, origin) | ((ulong)Axis(nodes[o + 1], step, origin) << AxisBits)
            | ((ulong)Axis(nodes[o + 2], step, origin) << (2 * AxisBits));
        uint lo = (uint)(k & 0xFFFFFFFFul), hi = (uint)(k >> 32);
        keyLo[i] = lo; keyHi[i] = hi; sortLo[i] = lo;
    }

    static void GatherKernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> perm, ArrayView1D<uint, Stride1D.Dense> src,
        ArrayView1D<uint, Stride1D.Dense> dst, int m)
    {
        if (i >= m) return;
        dst[i] = src[(int)perm[i]];
    }

    static bool SameKey(ArrayView1D<uint, Stride1D.Dense> perm, ArrayView1D<uint, Stride1D.Dense> keyLo,
        ArrayView1D<uint, Stride1D.Dense> keyHi, int a, int b)
    {
        int pa = (int)perm[a], pb = (int)perm[b];
        return keyLo[pa] == keyLo[pb] && keyHi[pa] == keyHi[pb];
    }

    /// <summary>One thread per sorted slot; the first slot of each cell's run merges the run (or passes it up).</summary>
    static void MergeKernel(Index1D i, ArrayView1D<uint, Stride1D.Dense> perm, ArrayView1D<uint, Stride1D.Dense> keyLo,
        ArrayView1D<uint, Stride1D.Dense> keyHi, ArrayView1D<int, Stride1D.Dense> frontier,
        ArrayView1D<float, Stride1D.Dense> nodes, ArrayView1D<int, Stride1D.Dense> parent,
        ArrayView1D<float, Stride1D.Dense> bounds, ArrayView1D<float, Stride1D.Dense> lodSize,
        ArrayView1D<int, Stride1D.Dense> firstChild, ArrayView1D<int, Stride1D.Dense> childCount,
        ArrayView1D<int, Stride1D.Dense> childList, ArrayView1D<int, Stride1D.Dense> nextFrontier,
        ArrayView1D<int, Stride1D.Dense> counters, int m)
    {
        if (i >= m) return;
        int self = frontier[(int)perm[i]];
        if (keyLo[(int)perm[i]] == WaitKey && keyHi[(int)perm[i]] == WaitKey)
        {
            nextFrontier[Atomic.Add(ref counters[1], 1)] = self;   // too large for this cell: waits
            return;
        }
        if (i > 0 && SameKey(perm, keyLo, keyHi, i, i - 1)) return;   // not the run's first slot
        int end = i + 1;
        while (end < m && SameKey(perm, keyLo, keyHi, end, i)) end++;
        int count = end - i;
        if (count == 1) { nextFrontier[Atomic.Add(ref counters[1], 1)] = self; return; }

        int p = Atomic.Add(ref counters[0], 1);
        int list = Atomic.Add(ref counters[2], count);
        firstChild[p] = list; childCount[p] = count;
        // Moments centred on the first child, in float.
        long o0 = (long)self * F;
        float cx = nodes[o0], cy = nodes[o0 + 1], cz = nodes[o0 + 2];
        float W = 0f, mx = 0f, my = 0f, mz = 0f, cr = 0f, cg = 0f, cb = 0f, aa = 0f, maxChild = 0f;
        float s00 = 0f, s01 = 0f, s02 = 0f, s11 = 0f, s12 = 0f, s22 = 0f;
        for (int k = i; k < end; k++)
        {
            int c = frontier[(int)perm[k]];
            childList[list + (k - i)] = c;
            parent[c] = p;
            maxChild = XMath.Max(maxChild, lodSize[c]);
            long o = (long)c * F;
            float sx = nodes[o + 6], sy = nodes[o + 7], sz = nodes[o + 8], al = nodes[o + 9];
            float area = Area(sx, sy, sz);
            float w = XMath.Max(1e-20f, al * area);
            // Covariance R S S^T R^T of the child.
            float qx = nodes[o + 10], qy = nodes[o + 11], qz = nodes[o + 12], qw = nodes[o + 13];
            float qn = XMath.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            qn = qn > 1e-20f ? 1f / qn : 0f; qx *= qn; qy *= qn; qz *= qn; qw = qn > 0f ? qw * qn : 1f;
            float xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
            float r00 = 1f - 2f * (yy + zz), r01 = 2f * (xy - wz), r02 = 2f * (xz + wy);
            float r10 = 2f * (xy + wz), r11 = 1f - 2f * (xx + zz), r12 = 2f * (yz - wx);
            float r20 = 2f * (xz - wy), r21 = 2f * (yz + wx), r22 = 1f - 2f * (xx + yy);
            float m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz, m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
            float m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
            float dx = nodes[o] - cx, dy = nodes[o + 1] - cy, dz = nodes[o + 2] - cz;
            W += w; mx += w * dx; my += w * dy; mz += w * dz;
            cr += w * nodes[o + 3]; cg += w * nodes[o + 4]; cb += w * nodes[o + 5];
            s00 += w * (m00 * m00 + m01 * m01 + m02 * m02 + dx * dx);
            s01 += w * (m00 * m10 + m01 * m11 + m02 * m12 + dx * dy);
            s02 += w * (m00 * m20 + m01 * m21 + m02 * m22 + dx * dz);
            s11 += w * (m10 * m10 + m11 * m11 + m12 * m12 + dy * dy);
            s12 += w * (m10 * m20 + m11 * m21 + m12 * m22 + dy * dz);
            s22 += w * (m20 * m20 + m21 * m21 + m22 * m22 + dz * dz);
            aa += al * area;
        }
        float inv = 1f / W;
        mx *= inv; my *= inv; mz *= inv;
        long po = (long)p * F;
        nodes[po] = cx + mx; nodes[po + 1] = cy + my; nodes[po + 2] = cz + mz;
        nodes[po + 3] = cr * inv; nodes[po + 4] = cg * inv; nodes[po + 5] = cb * inv;
        WriteScaleQuat(s00 * inv - mx * mx, s01 * inv - mx * my, s02 * inv - mx * mz,
            s11 * inv - my * my, s12 * inv - my * mz, s22 * inv - mz * mz, nodes, po);
        float areaP = Area(nodes[po + 6], nodes[po + 7], nodes[po + 8]);
        float alphaP = aa / XMath.Max(areaP, 1e-20f);
        nodes[po + 9] = alphaP > 1f ? 1f : alphaP;
        float sizeP = SizeOf(nodes, po);
        lodSize[p] = XMath.Max(sizeP, maxChild);
        // Sphere around the children's spheres, centred on the parent's mean.
        float px = nodes[po], py = nodes[po + 1], pz = nodes[po + 2], rad = 0f;
        for (int k = i; k < end; k++)
        {
            int c = frontier[(int)perm[k]];
            float bx = bounds[c * 4] - px, by = bounds[c * 4 + 1] - py, bz = bounds[c * 4 + 2] - pz;
            rad = XMath.Max(rad, XMath.Sqrt(bx * bx + by * by + bz * bz) + bounds[c * 4 + 3]);
        }
        bounds[p * 4] = px; bounds[p * 4 + 1] = py; bounds[p * 4 + 2] = pz; bounds[p * 4 + 3] = rad;
        parent[p] = -1;
        nextFrontier[Atomic.Add(ref counters[1], 1)] = p;
    }

    // ── host loop ─────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Build the tree over the first <paramref name="n"/> rows of <paramref name="leaves"/> (copied, not modified).
    /// <paramref name="baseStep"/> is the first level's cell (the CPU builder uses the median splat size).
    /// CPU transfers: two ints a level (frontier length, node count).
    /// </summary>
    public static async Task<GpuLodTree> BuildAsync(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> leaves, int n,
        float baseStep, SortPairs sort, int maxLevels = 64)
    {
        var t = new GpuLodTree { LeafCount = n };
        long cap = 2L * Math.Max(1, n);
        t.Nodes = a.Allocate1D<float>(cap * F);
        t.Nodes.View.SubView(0, (long)n * F).CopyFrom(leaves.View.SubView(0, (long)n * F));
        t.Parent = a.Allocate1D<int>(cap);
        t.Bounds = a.Allocate1D<float>(cap * 4);
        t.LodSize = a.Allocate1D<float>(cap);
        t.FirstChild = a.Allocate1D<int>(cap);
        t.ChildCount = a.Allocate1D<int>(cap);
        t.ChildList = a.Allocate1D<int>(cap);
        t.ChildCount.MemSetToZero();
        t.FirstChild.MemSetToZero();
        t._counters = a.Allocate1D<int>(3);

        using var frontierA = a.Allocate1D<int>(Math.Max(1, n));
        using var frontierB = a.Allocate1D<int>(Math.Max(1, n));
        using var keyLo = a.Allocate1D<uint>(Math.Max(1, n));
        using var keyHi = a.Allocate1D<uint>(Math.Max(1, n));
        using var sortKey = a.Allocate1D<uint>(Math.Max(1, n));
        using var perm = a.Allocate1D<uint>(Math.Max(1, n));

        var init = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(InitKernel);
        var key = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
            ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, int, float, float, int>(KeyKernel);
        var gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
            ArrayView1D<uint, Stride1D.Dense>, int>(GatherKernel);
        var merge = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>,
            ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(MergeKernel);

        init(n, t.Nodes.View, t.Parent.View, t.Bounds.View, t.LodSize.View, t.ChildCount.View, frontierA.View, n);
        // counters: next node id = n, frontier fill = 0, child-list fill = 0
        t._counters.CopyFromCPU(new[] { n, 0, 0 });

        var cur = frontierA; var next = frontierB;
        int m = n;
        float step = baseStep;
        int level = 0;
        for (; level < maxLevels && m > 1; level++, step *= LodTree.LevelGrowth)
        {
            int last = level == maxLevels - 1 ? 1 : 0;
            key(m, cur.View, t.Nodes.View, t.LodSize.View, keyLo.View, keyHi.View, sortKey.View, perm.View, m, step,
                LodTree.LevelOrigin(level, step), last);
            sort(sortKey, perm, m);                                   // by the low half
            gather(m, perm.View, keyHi.View, sortKey.View, m);
            sort(sortKey, perm, m);                                   // then by the high half (stable)
            await ResetFrontierCountAsync(t._counters);
            merge(m, perm.View, keyLo.View, keyHi.View, cur.View, t.Nodes.View, t.Parent.View, t.Bounds.View, t.LodSize.View,
                t.FirstChild.View, t.ChildCount.View, t.ChildList.View, next.View, t._counters.View, m);
            await a.SynchronizeAsync();
            // CPU transfer: the next frontier's length and the node count (two ints a level).
            var c = await t._counters.CopyToHostAsync<int>(0, 3);
            m = c[1];
            t.NodeCount = c[0];
            (cur, next) = (next, cur);
        }
        t.Levels = level;
        if (t.NodeCount == 0) t.NodeCount = n;
        return t;
    }

    static Task ResetFrontierCountAsync(MemoryBuffer1D<int, Stride1D.Dense> counters)
    {
        counters.View.SubView(1, 1).MemSetToZero();
        return Task.CompletedTask;
    }
}
