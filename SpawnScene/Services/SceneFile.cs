using System.Text;
using System.Text.Json;

namespace SpawnScene.Services;

/// <summary>
/// The .spawnscene file: one saved scene, complete, to carry between browsers (OPFS is per browser - a scene trained on
/// the desktop is not on the headset). Layout: "SPSCENE1" (8 bytes), the header's byte length (int32, little-endian),
/// the header (UTF-8 JSON, <see cref="Header"/>), then the packed splats (SplatCount x FloatsPerSplat floats) and, when
/// ShParts &gt; 0, each SH part (SplatCount x PartFloatsPerSplat floats). The bytes stay in the browser: only the header
/// passes through .NET.
/// </summary>
public static class SceneFile
{
    public const string Magic = "SPSCENE1";
    public const string Extension = ".spawnscene";

    /// <param name="HomeView">Where the viewer starts: position, forward, up (9 floats) - the view it was exported
    /// from. Optional.</param>
    public sealed record Header(
        string Name, int SplatCount, int FloatsPerSplat, bool ColoursAreShDc, int ShDegree, int ShParts,
        int TrainedIterations, DateTime SavedAt, float[]? HomeView = null);

    /// <summary>Magic + length + header JSON: the bytes that precede the splat data.</summary>
    public static byte[] Prefix(Header header)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(header);
        var bytes = new byte[Magic.Length + 4 + json.Length];
        Encoding.ASCII.GetBytes(Magic).CopyTo(bytes, 0);
        BitConverter.GetBytes(json.Length).CopyTo(bytes, Magic.Length);
        json.CopyTo(bytes, Magic.Length + 4);
        return bytes;
    }

    /// <summary>The header's byte length from the first 12 bytes; throws if they are not a .spawnscene file.</summary>
    public static int HeaderLength(ReadOnlySpan<byte> first12)
    {
        if (first12.Length < Magic.Length + 4 || Encoding.ASCII.GetString(first12[..Magic.Length]) != Magic)
            throw new InvalidDataException("not a .spawnscene file");
        int len = BitConverter.ToInt32(first12.Slice(Magic.Length, 4));
        if (len <= 0 || len > 1 << 20) throw new InvalidDataException($"bad .spawnscene header length {len}");
        return len;
    }

    public static Header ParseHeader(ReadOnlySpan<byte> json)
        => JsonSerializer.Deserialize<Header>(json) ?? throw new InvalidDataException("empty .spawnscene header");

    /// <summary>Where the packed splats start (after magic, length and header).</summary>
    public static long DataOffset(int headerLength) => Magic.Length + 4 + headerLength;
}
