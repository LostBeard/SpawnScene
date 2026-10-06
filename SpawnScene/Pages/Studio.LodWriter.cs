using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// Writes a .spawnscene v3 one BLOCK at a time (Plans/lod-streaming.md phase D; the layout LodLayout.Forest tests):
    /// each block becomes its own LOD tree, laid out breadth-first, its node indices moved to where it sits in the file
    /// and its roots hung off a block node; when every block is in, one root over the block nodes is chunk 0. Only one
    /// block is ever on the GPU, and each gzipped chunk goes straight into a Blob (the browser may keep it on disk) - so a
    /// scene too large for one GPU can still be written, then streamed. Export streaming is the one-block case.
    /// </summary>
    sealed class LodStreamWriter
    {
        const int F = SplatFormat.Floats;
        const int W = SphericalHarmonics.PartFloatsPerSplat;

        readonly Studio _s;
        readonly int _blocks;
        readonly bool _withSh;
        readonly List<LodChunkFile.Chunk> _chunks = new();
        readonly List<Blob> _blobs = new();
        long _offset;
        int _next, _leaves, _maxChunk;
        bool _finished;
        // Each block node: its row, sphere, LOD size, SH rows (Parts x W) and its block's first file node.
        readonly List<(float[] Row, float[] Bounds, float Size, float[][]? Sh, int First, int Roots)> _blockNodes = new();

        public LodStreamWriter(Studio s, int blocks, bool withSh)
        {
            _s = s; _blocks = blocks; _withSh = withSh;
            _next = 1 + blocks;   // the root and the block nodes come first
        }

        public int Nodes => _next;
        public int Leaves => _leaves;
        public int ChunkCount => _finished ? _chunks.Count : 1 + _chunks.Count;
        public long RawBytes { get; private set; }

        /// <summary>
        /// Add the next block: <paramref name="n"/> packed rows (and, with SH, its leaf SH parts). The buffers stay the
        /// caller's. CPU transfers: the file's own bytes, plus the block's parent / child counts and its roots' rows.
        /// </summary>
        public async Task AddBlockAsync(MemoryBuffer1D<float, Stride1D.Dense> packed, int n, MemoryBuffer1D<float, Stride1D.Dense>[]? leafSh)
        {
            if (_blockNodes.Count >= _blocks) throw new InvalidOperationException("more blocks than the writer was made for");
            var a = _s._gpuService.WebGPUAccelerator;
            int k = _blockNodes.Count, at = _next;
            var owned = new List<IDisposable>();
            try
            {
                float baseStep = await LodBaseStepAsync(a, packed, n);
                GpuLodTree laid;
                MemoryBuffer1D<int, Stride1D.Dense>? order = null;
                using (var tree = await GpuLodTree.BuildAsync(a, packed, n, baseStep, _s.LodSortPairs()))
                    laid = await GpuLodLayout.BuildAsync(a, tree, o => order = o);
                owned.Add(laid); owned.Add(order!);
                int nodes = laid.NodeCount;

                // SH for every node: the leaves' own, each merge its children's weighted mean.
                MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
                if (_withSh && leafSh is { Length: SphericalHarmonics.Parts })
                {
                    sh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
                    for (int p = 0; p < sh.Length; p++)
                    {
                        sh[p] = GpuLodLayout.GatherLeafRows(a, leafSh[p], order!, n, nodes, W);
                        owned.Add(sh[p]);
                        GpuLodLayout.MergeSh(a, laid, sh[p], W);
                    }
                }
                await a.SynchronizeAsync();

                // Chunk boundaries and needs, from the block's topology (written into the chunks anyway).
                var topo = new LodTree
                {
                    NodeCount = nodes, LeafCount = n,
                    Parent = (await laid.Parent.CopyToHostAsync<int>(0, nodes)).ToArray(),
                    ChildCount = (await laid.ChildCount.CopyToHostAsync<int>(0, nodes)).ToArray(),
                };
                int roots = 0;
                while (roots < nodes && topo.Parent[roots] < 0) roots++;
                topo.FirstChild = new int[nodes];
                for (int i = 0, next = roots; i < nodes; i++) { topo.FirstChild[i] = topo.ChildCount[i] > 0 ? next : -1; next += topo.ChildCount[i]; }
                var starts = LodLayout.ChunkStarts(topo, LodChunkFile.DefaultChunkNodes);
                var (parentF, firstF) = GpuLodLayout.OffsetIndices(a, laid, at, 1 + k);
                owned.Add(parentF); owned.Add(firstF);
                int chunkBase = 1 + _chunks.Count;

                for (int ci = 0; ci + 1 < starts.Length; ci++)
                {
                    int first = starts[ci], count = starts[ci + 1] - first;
                    var needs = LodLayout.ParentChunks(topo, starts, ci).Select(c => chunkBase + c).ToList();
                    if (first < roots) needs.Insert(0, 0);   // its roots hang off the top
                    await WriteChunkAsync(a, laid.Nodes, sh, first, count, at + first, parentF, firstF, laid.Bounds, laid.LodSize, needs.ToArray());
                }

                // The block node: the merge of the block's roots, around them, their SH weighted like their colour.
                var rootRows = (await laid.Nodes.CopyToHostAsync<float>(0, (long)roots * F)).ToArray();
                var rootBounds = (await laid.Bounds.CopyToHostAsync<float>(0, (long)roots * 4)).ToArray();
                var rootSizes = (await laid.LodSize.CopyToHostAsync<float>(0, roots)).ToArray();
                var row = new float[F];
                LodMerge.Merge(rootRows, roots, row);
                var bounds = new float[4];
                LodLayout.Enclose(row, rootBounds, rootSizes, bounds, out float size);
                float[][]? blockSh = null;
                if (sh != null)
                {
                    blockSh = new float[sh.Length][];
                    for (int p = 0; p < sh.Length; p++)
                        blockSh[p] = WeightedMean((await sh[p].CopyToHostAsync<float>(0, (long)roots * W)).ToArray(), rootRows, roots);
                }
                _blockNodes.Add((row, bounds, size, blockSh, at, roots));
                _next += nodes;
                _leaves += n;
            }
            finally { foreach (var d in owned) d.Dispose(); }
        }

        /// <summary>The weighted mean (LodMerge's opacity x area weights of <paramref name="rows"/>) of <paramref name="count"/> W-float rows.</summary>
        static float[] WeightedMean(float[] values, float[] rows, int count)
        {
            var mean = new float[W];
            float wsum = 0f;
            for (int c = 0; c < count; c++)
            {
                float w = LodMerge.WeightOf(rows, c * F);
                wsum += w;
                for (int f = 0; f < W; f++) mean[f] += w * values[c * W + f];
            }
            if (wsum > 0f) for (int f = 0; f < W; f++) mean[f] /= wsum;
            return mean;
        }

        /// <summary>
        /// One chunk: rows [first, first+count) of <paramref name="nodes"/> on a codec frame of their own, then the cut's
        /// raw data (file indices), gzipped into a Blob at the current data offset.
        /// </summary>
        async Task WriteChunkAsync(WebGPUAccelerator a, MemoryBuffer1D<float, Stride1D.Dense> nodes,
            MemoryBuffer1D<float, Stride1D.Dense>[]? sh, int first, int count, int fileFirst,
            MemoryBuffer1D<int, Stride1D.Dense> parent, MemoryBuffer1D<int, Stride1D.Dense> firstChild,
            MemoryBuffer1D<float, Stride1D.Dense> bounds, MemoryBuffer1D<float, Stride1D.Dense> lodSize, int[] needs)
        {
            var part = new List<IDisposable>();
            try
            {
                var rows = a.Allocate1D<float>((long)count * F); part.Add(rows);
                rows.View.CopyFrom(nodes.View.SubView((long)first * F, (long)count * F));
                MemoryBuffer1D<float, Stride1D.Dense>[]? shRows = null;
                if (sh != null)
                {
                    shRows = new MemoryBuffer1D<float, Stride1D.Dense>[sh.Length];
                    for (int p = 0; p < sh.Length; p++)
                    {
                        shRows[p] = a.Allocate1D<float>((long)count * W); part.Add(shRows[p]);
                        shRows[p].View.CopyFrom(sh[p].View.SubView((long)first * W, (long)count * W));
                    }
                }
                var frame = await ChunkFrameAsync(a, rows, count);
                var (geo, app, shq) = SceneCodec.Encode(a, rows, shRows, frame);
                part.Add(geo); part.Add(app); part.Add(shq);
                await a.SynchronizeAsync();
                // CPU transfer: file I/O - the chunk's streams, then gzipped by the browser.
                var raw = new List<Uint8Array>
                {
                    await geo.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.GeoWords * 4),
                    await app.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.AppWords * 4),
                };
                if (shRows != null) raw.Add(await shq.CopyToHostUint8ArrayAsync(0, (long)count * SceneCodec.ShWords * 4));
                raw.Add(await parent.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                raw.Add(await firstChild.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                raw.Add(await bounds.CopyToHostUint8ArrayAsync((long)first * 16, (long)count * 16));
                raw.Add(await lodSize.CopyToHostUint8ArrayAsync((long)first * 4, (long)count * 4));
                await AddChunkAsync(raw, fileFirst, count, frame, needs);
            }
            finally { foreach (var d in part) d.Dispose(); }
        }

        async Task AddChunkAsync(List<Uint8Array> raw, int fileFirst, int count, SceneCodec.Frame frame, int[] needs)
        {
            try
            {
                RawBytes += raw.Sum(r => (long)r.ByteLength);
                using var blobIn = new Blob(raw, new BlobOptions { Type = "application/octet-stream" });
                using var z = await GzipAsync(blobIn, decompress: false);
                using var zu = new Uint8Array(z);
                _blobs.Add(new Blob(new[] { zu }, new BlobOptions { Type = "application/octet-stream" }));
                _chunks.Add(new LodChunkFile.Chunk(fileFirst, count, _offset, z.ByteLength, OuterBox(frame), frame.InnerArray(), needs));
                _offset += z.ByteLength;
                _maxChunk = Math.Max(_maxChunk, count);
            }
            finally { foreach (var r in raw) r.Dispose(); }
        }

        static float[] OuterBox(SceneCodec.Frame f) => new[]
        {
            f.MinX - f.TailLoX, f.MinY - f.TailLoY, f.MinZ - f.TailLoZ,
            f.MinX + f.SizeX + f.TailHiX, f.MinY + f.SizeY + f.TailHiY, f.MinZ + f.SizeZ + f.TailHiZ,
        };

        /// <summary>
        /// The top (the root and the block nodes) as chunk 0, then the file: header, the blocks' chunks, the top's chunk
        /// last in the bytes (chunk 0 by node order, wherever its bytes sit).
        /// </summary>
        public async Task<Blob> FinishAsync(string name, int shDegree, bool coloursAreShDc, int trainedIterations, float[]? homeView)
        {
            int b = _blockNodes.Count;
            if (b != _blocks) throw new InvalidOperationException($"{b} of {_blocks} blocks were added");
            var a = _s._gpuService.WebGPUAccelerator;
            int top = 1 + b;
            var rows = new float[top * F];
            var bounds = new float[top * 4];
            var sizes = new float[top];
            var parents = new int[top];
            var firsts = new int[top];
            for (int k = 0; k < b; k++)
            {
                var bn = _blockNodes[k];
                System.Array.Copy(bn.Row, 0, rows, (1 + k) * F, F);
                System.Array.Copy(bn.Bounds, 0, bounds, (1 + k) * 4, 4);
                sizes[1 + k] = bn.Size;
                parents[1 + k] = 0;
                firsts[1 + k] = bn.First;
            }
            LodMerge.Merge(rows.AsSpan(F, b * F), b, rows.AsSpan(0, F));
            LodLayout.Enclose(rows.AsSpan(0, F), bounds.AsSpan(4, b * 4), sizes.AsSpan(1, b), bounds.AsSpan(0, 4), out sizes[0]);
            parents[0] = -1;
            firsts[0] = 1;

            var owned = new List<IDisposable>();
            try
            {
                using var topRows = a.Allocate1D(rows);
                MemoryBuffer1D<float, Stride1D.Dense>[]? topSh = null;
                if (_withSh && _blockNodes.All(x => x.Sh != null))
                {
                    topSh = new MemoryBuffer1D<float, Stride1D.Dense>[SphericalHarmonics.Parts];
                    for (int p = 0; p < topSh.Length; p++)
                    {
                        var blockShP = new float[b * W];
                        for (int k = 0; k < b; k++) System.Array.Copy(_blockNodes[k].Sh![p], 0, blockShP, k * W, W);
                        var all = new float[top * W];
                        System.Array.Copy(WeightedMean(blockShP, rows[F..], b), 0, all, 0, W);
                        System.Array.Copy(blockShP, 0, all, W, b * W);
                        topSh[p] = a.Allocate1D(all); owned.Add(topSh[p]);
                    }
                }
                var frame = await ChunkFrameAsync(a, topRows, top);
                var (geo, app, shq) = SceneCodec.Encode(a, topRows, topSh, frame);
                owned.Add(geo); owned.Add(app); owned.Add(shq);
                await a.SynchronizeAsync();
                var raw = new List<Uint8Array>
                {
                    await geo.CopyToHostUint8ArrayAsync(0, (long)top * SceneCodec.GeoWords * 4),
                    await app.CopyToHostUint8ArrayAsync(0, (long)top * SceneCodec.AppWords * 4),
                };
                if (topSh != null) raw.Add(await shq.CopyToHostUint8ArrayAsync(0, (long)top * SceneCodec.ShWords * 4));
                raw.Add(Bytes(parents)); raw.Add(Bytes(firsts)); raw.Add(Bytes(bounds)); raw.Add(Bytes(sizes));
                int blockChunks = _chunks.Count;
                await AddChunkAsync(raw, 0, top, frame, System.Array.Empty<int>());
                // The top was appended last; it is chunk 0.
                var topChunk = _chunks[blockChunks];
                _chunks.RemoveAt(blockChunks);
                _chunks.Insert(0, topChunk);
                _finished = true;
                var topBlob = _blobs[^1];
                _blobs.RemoveAt(_blobs.Count - 1);
                _blobs.Add(topBlob);   // bytes stay last; its Offset says so

                var header = new LodChunkFile.Header3(name, _next, _leaves, 1, coloursAreShDc, topSh != null ? shDegree : 0,
                    trainedIterations, DateTime.UtcNow, homeView, Math.Max(_maxChunk, LodChunkFile.DefaultChunkNodes), _chunks.ToArray());
                using var prefix = new Uint8Array(LodChunkFile.Prefix3(header));
                using var prefixBlob = new Blob(new[] { prefix }, new BlobOptions { Type = "application/octet-stream" });
                var parts = new List<Blob> { prefixBlob };
                parts.AddRange(_blobs);
                var file = new Blob(parts, new BlobOptions { Type = "application/octet-stream" });
                foreach (var bl in _blobs) bl.Dispose();
                _blobs.Clear();
                return file;
            }
            finally { foreach (var d in owned) d.Dispose(); }
        }

        static Uint8Array Bytes<T>(T[] values) where T : unmanaged
        {
            var bytes = new byte[values.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>()];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return new Uint8Array(bytes);
        }
    }
}
