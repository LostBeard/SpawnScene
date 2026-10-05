using System.Numerics;
using System.Text.Json;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Chunk layouts measured on a REAL trained scene (Explicit: needs the TruckFull 30K sample, ~2 min of CPU): for views
/// around the scene's home view, how many chunks hold the nodes the cut needs (drawn in a view cone, plus ancestors)
/// against the fewest chunks that many nodes could fit in. The synthetic corridor said little about real trees.
/// <para>
/// 2026-10-05, TruckFull 30K (1.9M leaves, 2.4M nodes, 37 chunks of 64K), tau 1.5 px, a 100 deg cone: from the home
/// view the cut needs 1.14M nodes (ideal 18 chunks) and both layouts touch all 37; turned 180 deg, 310K nodes (ideal
/// 5): breadth-first 24 chunks, band x space 31. A one-object capture seen from its photos needs half its tree -
/// streaming pays on large multi-room scenes, not here - and the band x space order is no better: breadth-first stays.
/// Chunk size, turned 180 deg: 64K chunks load 29 of 37 (closed), 16K 87 of 147, 8K 143 of 294 - smaller is more
/// selective; LodChunkFile.DefaultChunkNodes is 16K.
/// </para>
/// </summary>
[Explicit]
public class LodLayoutRealSceneTests
{
    const int F = SplatFormat.Floats;
    const string Sample = @"D:\users\tj\Projects\SpawnScene\SpawnScene\_pub_tuvok_est\wwwroot\samples\truck_ours30k.spawnscene";

    static (float[] Rows, int N, float[] Home) LoadV1(string path)
    {
        using var fs = File.OpenRead(path);
        var head = new byte[12];
        fs.ReadExactly(head);
        int len = BitConverter.ToInt32(head, 8);
        var json = new byte[len];
        fs.ReadExactly(json);
        var h = JsonSerializer.Deserialize<SceneFile.Header>(json)!;
        var bytes = new byte[(long)h.SplatCount * h.FloatsPerSplat * 4];
        fs.ReadExactly(bytes);
        var rows = new float[h.SplatCount * F];
        Buffer.BlockCopy(bytes, 0, rows, 0, bytes.Length);
        return (rows, h.SplatCount, h.HomeView!);
    }

    /// <summary>
    /// Detail-band x space order: sibling runs sorted by their parent's LOD-size band (coarse first), then depth, then a
    /// Morton code of the parent's position - so a chunk is one detail level of one region. Parent-first holds: a
    /// node's own run has a band at least its children's run, and when equal a smaller depth.
    /// </summary>
    static int[] BandSpaceOrder(LodTree t)
    {
        int n = t.NodeCount;
        var depth = new int[n];
        var bfs = LodLayout.BreadthFirst(t);
        foreach (int i in bfs) depth[i] = t.Parent[i] < 0 ? 0 : depth[t.Parent[i]] + 1;
        float lo = float.MaxValue, hi = float.MinValue;
        var xs = Enumerable.Range(0, t.LeafCount).Select(i => t.Rows[i * F]).OrderBy(v => v).ToArray();
        var ys = Enumerable.Range(0, t.LeafCount).Select(i => t.Rows[i * F + 1]).OrderBy(v => v).ToArray();
        var zs = Enumerable.Range(0, t.LeafCount).Select(i => t.Rows[i * F + 2]).OrderBy(v => v).ToArray();
        int q1 = t.LeafCount / 100, q99 = t.LeafCount - 1 - q1;
        var mn = new Vector3(xs[q1], ys[q1], zs[q1]); var mx = new Vector3(xs[q99], ys[q99], zs[q99]);
        uint Morton(int node)
        {
            uint Q(float v, float a, float b) => (uint)Math.Clamp((v - a) / Math.Max(b - a, 1e-6f) * 1023f, 0f, 1023f);
            uint x = Q(t.Rows[node * F], mn.X, mx.X), y = Q(t.Rows[node * F + 1], mn.Y, mx.Y), z = Q(t.Rows[node * F + 2], mn.Z, mx.Z);
            uint m = 0;
            for (int b = 0; b < 10; b++) m |= ((x >> b) & 1) << (3 * b) | ((y >> b) & 1) << (3 * b + 1) | ((z >> b) & 1) << (3 * b + 2);
            return m;
        }
        int Band(float s) => s <= 0 ? int.MinValue : (int)MathF.Floor(MathF.Log(s) / MathF.Log(1.5f));
        var runs = new List<(int Band, int Depth, uint Morton, int Parent)>();
        for (int p = 0; p < n; p++)
            if (t.ChildCount[p] > 0) runs.Add((Band(t.LodSize[p]), depth[p] + 1, Morton(p), p));
        runs.Sort((a, b) => a.Band != b.Band ? b.Band.CompareTo(a.Band) : a.Depth != b.Depth ? a.Depth.CompareTo(b.Depth) : a.Morton.CompareTo(b.Morton));
        var order = new List<int>(n);
        order.AddRange(Enumerable.Range(0, n).Where(i => t.Parent[i] < 0));
        foreach (var r in runs) for (int k = 0; k < t.ChildCount[r.Parent]; k++) order.Add(t.ChildIndex[t.FirstChild[r.Parent] + k]);
        return order.ToArray();
    }

    [Test]
    public void Layouts_OnTruck30K()
    {
        var (rows, n, home) = LoadV1(Sample);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var t = LodTree.Build(rows, n);
        TestContext.Out.WriteLine($"{n:N0} leaves -> {t.NodeCount:N0} nodes in {sw.Elapsed.TotalSeconds:F0}s");
        var pos = new Vector3(home[0], home[1], home[2]);
        var fwd = Vector3.Normalize(new Vector3(home[3], home[4], home[5]));
        var side = Vector3.Normalize(Vector3.Cross(fwd, Vector3.UnitY));
        var views = new (string Name, Vector3 Pos, Vector3 Fwd)[]
        {
            ("home", pos, fwd),
            ("back 1", pos - fwd * 1f, fwd),
            ("forward 0.5", pos + fwd * 0.5f, fwd),
            ("turned 90", pos, side),
            ("turned 180", pos, -fwd),
        };
        const float focal = 950f, tau = 1.5f, cosHalf = 0.64f;   // ~100 deg cone, 1600 px wide at 80 deg
        var bfs = LodLayout.BreadthFirst(t);
        foreach (var (name, order, chunkNodes) in new[] { ("breadth-first", bfs, 65536), ("band x space", BandSpaceOrder(t), 65536),
            ("bf 16K", bfs, 16384), ("bf 8K", bfs, 8192) })
        {
            var l = LodLayout.Reorder(t, order);
            var starts = LodLayout.ChunkStarts(l, chunkNodes);
            int chunks = starts.Length - 1;
            foreach (var v in views)
            {
                var on = new bool[l.NodeCount];
                int neededNodes = 0;
                var needed = new HashSet<int>();
                for (int i = 0; i < l.NodeCount; i++)
                {
                    if (!LodLayout.InCut(l, i, v.Pos, focal, tau)) continue;
                    var c = new Vector3(l.Bounds[i * 4], l.Bounds[i * 4 + 1], l.Bounds[i * 4 + 2]);
                    var d = c - v.Pos;
                    float dist = d.Length();
                    if (dist > l.Bounds[i * 4 + 3] && Vector3.Dot(d / dist, v.Fwd) < cosHalf) continue;   // outside the view cone
                    for (int a = i; a >= 0 && !on[a]; a = l.Parent[a]) { on[a] = true; neededNodes++; needed.Add(LodLayout.ChunkOf(starts, a)); }
                }
                var closed = new HashSet<int>(needed);
                var todo = new Stack<int>(closed);
                while (todo.Count > 0) foreach (int pc in LodLayout.ParentChunks(l, starts, todo.Pop())) if (closed.Add(pc)) todo.Push(pc);
                TestContext.Out.WriteLine($"{name,-14} {v.Name,-12}: {neededNodes,9:N0} nodes needed (ideal {(neededNodes + chunkNodes - 1) / chunkNodes} chunks), " +
                    $"in {needed.Count} chunks, {closed.Count} closed, of {chunks}");
            }
        }
    }
}
