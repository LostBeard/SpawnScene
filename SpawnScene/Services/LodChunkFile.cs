using System.Text;
using System.Text.Json;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// The .spawnscene v3 file: a scene as its LOD tree, laid out breadth-first (<see cref="LodLayout"/>) and cut into
/// chunks of <see cref="Header3.ChunkNodes"/> nodes, each gzipped on its own so a viewer can fetch any chunk by its byte
/// range (Plans/lod-streaming.md, phase B). Layout: "SPSCENE3", the header's byte length (int32 LE), the header (UTF-8
/// JSON, <see cref="Header3"/>), then the chunks back to back. Chunk 0 is the top of the tree - on its own a coarse
/// copy of the whole scene. Chunks hold at most <see cref="Header3.ChunkNodes"/> nodes and never split a sibling run
/// (LodLayout.ChunkStarts), so their sizes vary.
/// <para>
/// A chunk of n nodes, before gzip: the <see cref="SceneCodec"/> streams over the chunk's own frame (geometry n x 4
/// words, appearance n x 3, SH n x 12 when the file has SH bands; internal nodes' SH are zero), then the cut's data,
/// raw so the cut stays exact: parent (n int32, -1 for a root), first child (n int32, -1 for a leaf; the children are
/// a run that never crosses a chunk), bounding sphere (n x 4 float32), LOD size (n float32, 0 for a leaf -
/// <see cref="LodLayout"/>). Node indices are the file's (breadth-first) ones.
/// </para>
/// </summary>
public static class LodChunkFile
{
    public const string Magic3 = "SPSCENE3";

    /// <summary>Nodes a chunk (Spark's page size).</summary>
    public const int DefaultChunkNodes = 65536;

    /// <param name="First">The chunk's first node (breadth-first index).</param>
    /// <param name="Offset">Byte offset of the gzipped chunk from the first byte after the header (SceneFile.DataOffset).</param>
    /// <param name="Bounds">min x,y,z then max x,y,z of the chunk's node positions (the codec frame's outer box).</param>
    /// <param name="Inner">The frame's linear box (SceneCodec.QuantPosP), min then max.</param>
    /// <param name="Needs">The chunks holding its nodes' parents (LodLayout.ParentChunks): load them first.</param>
    public sealed record Chunk(int First, int Count, long Offset, long Bytes, float[] Bounds, float[] Inner, int[] Needs);

    /// <param name="LeafCount">The scene's splats (the leaves); the rest of the nodes are merges.</param>
    public sealed record Header3(
        string Name, int NodeCount, int LeafCount, int RootCount, bool ColoursAreShDc, int ShDegree, int TrainedIterations,
        DateTime SavedAt, float[]? HomeView, int ChunkNodes, Chunk[] Chunks, string Compression = "gzip");

    /// <summary>Where each part of a raw chunk of <paramref name="n"/> nodes starts (bytes).</summary>
    public readonly record struct Layout(long Geo, long App, long Sh, long Parent, long FirstChild, long Bounds, long LodSize, long End)
    {
        public static Layout Of(int n, bool withSh)
        {
            long geo = 0, app = geo + (long)n * SceneCodec.GeoWords * 4, sh = app + (long)n * SceneCodec.AppWords * 4;
            long parent = sh + (withSh ? (long)n * SceneCodec.ShWords * 4 : 0);
            long firstChild = parent + (long)n * 4, bounds = firstChild + (long)n * 4, lodSize = bounds + (long)n * 16;
            return new Layout(geo, app, sh, parent, firstChild, bounds, lodSize, lodSize + (long)n * 4);
        }
    }

    public static byte[] Prefix3(Header3 header)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(header);
        var bytes = new byte[Magic3.Length + 4 + json.Length];
        Encoding.ASCII.GetBytes(Magic3).CopyTo(bytes, 0);
        BitConverter.GetBytes(json.Length).CopyTo(bytes, Magic3.Length);
        json.CopyTo(bytes, Magic3.Length + 4);
        return bytes;
    }

    public static Header3 ParseHeader3(ReadOnlySpan<byte> json)
        => JsonSerializer.Deserialize<Header3>(json) ?? throw new InvalidDataException("empty .spawnscene v3 header");

    /// <summary>The codec frame of a chunk from its header entry.</summary>
    public static SceneCodec.Frame FrameOf(Chunk c)
    {
        var b = c.Bounds; var r = c.Inner;
        return SceneCodec.Frame.From(new SplatBounds.Aabb(b[0], b[1], b[2], b[3], b[4], b[5]),
            new SplatBounds.Aabb(r[0], r[1], r[2], r[3], r[4], r[5]), c.Count);
    }
}
