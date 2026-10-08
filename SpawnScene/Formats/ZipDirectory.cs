namespace SpawnScene.Formats;

/// <summary>
/// The central directory of a zip (PKWARE APPNOTE 4.3.12/4.3.16), for SOG bundles: names, methods (0 stored, 8 deflate),
/// sizes and where each entry's local header sits. Pure parsing of the few KB at the end of the file; the entries' data is
/// read by the caller as Blob slices (stored) or through the browser's DecompressionStream("deflate-raw"). No zip64.
/// </summary>
public static class ZipDirectory
{
    public sealed record Entry(string Name, int Method, long CompressedSize, long Size, long LocalHeaderOffset);

    public static bool IsZip(ReadOnlySpan<byte> head) => head.Length >= 4 && head[0] == 0x50 && head[1] == 0x4b && head[2] == 0x03 && head[3] == 0x04;

    /// <summary>Find the end-of-central-directory record in the file's last bytes (<paramref name="tail"/>, which end
    /// at <paramref name="fileSize"/>): (central directory offset, size, entry count).</summary>
    public static (long Offset, long Size, int Count) FindEnd(ReadOnlySpan<byte> tail, long fileSize)
    {
        for (int i = tail.Length - 22; i >= 0; i--)
        {
            if (BitConverter.ToUInt32(tail[i..]) != 0x06054b50) continue;
            int count = BitConverter.ToUInt16(tail[(i + 10)..]);
            long size = BitConverter.ToUInt32(tail[(i + 12)..]);
            long offset = BitConverter.ToUInt32(tail[(i + 16)..]);
            if (offset == 0xFFFFFFFF || count == 0xFFFF) throw new FormatException("zip64 archives are not read");
            if (offset + size > fileSize) throw new FormatException("zip central directory lies past the end of the file");
            return (offset, size, count);
        }
        throw new FormatException("not a zip file (no end-of-central-directory record)");
    }

    /// <summary>Parse the central directory bytes.</summary>
    public static List<Entry> Parse(ReadOnlySpan<byte> cd, int count)
    {
        var list = new List<Entry>(count);
        int p = 0;
        for (int k = 0; k < count; k++)
        {
            if (p + 46 > cd.Length || BitConverter.ToUInt32(cd[p..]) != 0x02014b50) throw new FormatException("bad zip central directory");
            int method = BitConverter.ToUInt16(cd[(p + 10)..]);
            long comp = BitConverter.ToUInt32(cd[(p + 20)..]);
            long size = BitConverter.ToUInt32(cd[(p + 24)..]);
            int nameLen = BitConverter.ToUInt16(cd[(p + 28)..]);
            int extraLen = BitConverter.ToUInt16(cd[(p + 30)..]);
            int commentLen = BitConverter.ToUInt16(cd[(p + 32)..]);
            long local = BitConverter.ToUInt32(cd[(p + 42)..]);
            string name = System.Text.Encoding.UTF8.GetString(cd.Slice(p + 46, nameLen));
            list.Add(new Entry(name, method, comp, size, local));
            p += 46 + nameLen + extraLen + commentLen;
        }
        return list;
    }

    /// <summary>Where an entry's data starts, from its local header's 30 fixed bytes (name and extra lengths there may
    /// differ from the central directory's).</summary>
    public static long DataOffset(ReadOnlySpan<byte> localHeader30, Entry e)
    {
        if (BitConverter.ToUInt32(localHeader30) != 0x04034b50) throw new FormatException($"bad zip local header for {e.Name}");
        return e.LocalHeaderOffset + 30 + BitConverter.ToUInt16(localHeader30[26..]) + BitConverter.ToUInt16(localHeader30[28..]);
    }
}
