using System.Buffers.Binary;

namespace SpawnScene.Models;

/// <summary>
/// Minimal EXIF parser — extracts focal length and exposure (time, f-number, ISO) tags from JPEG byte arrays.
/// No allocations beyond the result record. Works in Blazor WASM (pure C#, no native deps).
/// </summary>
public static class ExifReader
{
    /// <summary>
    /// Result of EXIF focal length extraction.
    /// FocalLength35mm is preferred (directly usable for pixel conversion).
    /// FocalLengthMm is the raw lens focal length (needs sensor size for pixel conversion).
    /// </summary>
    public record ExifFocalLength(float? FocalLength35mm, float? FocalLengthMm);

    // EXIF tag IDs
    private const ushort TagExifIfdPointer = 0x8769;
    private const ushort TagFocalLength = 0x920A;       // RATIONAL (mm)
    private const ushort TagFocalLength35mm = 0xA405;    // SHORT (mm, 35mm equivalent)

    // EXIF data types
    private const ushort TypeShort = 3;    // uint16
    private const ushort TypeLong = 4;     // uint32
    private const ushort TypeRational = 5; // two uint32 (numerator/denominator)

    /// <summary>
    /// Extract focal length from JPEG EXIF data.
    /// Returns null if the image has no EXIF or no focal length tags.
    /// </summary>
    public static ExifFocalLength? ExtractFocalLength(ReadOnlySpan<byte> jpegBytes)
    {
        int start = TiffStart(jpegBytes, out int length);
        return start < 0 ? null : ParseTiff(jpegBytes.Slice(start, length));
    }

    /// <summary>A photo's exposure settings (null fields: tag absent).</summary>
    public record ExifExposure(float? ExposureTime, float? FNumber, float? Iso)
    {
        /// <summary>The exposure in stops, ISO-normalised: log2(N^2 / t) - log2(ISO / 100). Higher = less light let in
        /// (a brighter scene). Null unless all three are known.</summary>
        public float? Stops => ExposureTime is > 0f && FNumber is > 0f && Iso is > 0f
            ? MathF.Log2(FNumber.Value * FNumber.Value / ExposureTime.Value) - MathF.Log2(Iso.Value / 100f)
            : null;
    }

    private const ushort TagExposureTime = 0x829A;   // RATIONAL (s)
    private const ushort TagFNumber = 0x829D;        // RATIONAL
    private const ushort TagIso = 0x8827;            // SHORT (ISOSpeedRatings / PhotographicSensitivity)

    /// <summary>
    /// Exposure time, f-number and ISO from JPEG EXIF, or null when the photo has no EXIF IFD. Used to tell a capture whose
    /// exposure varied (auto exposure: per-photo gains help, Bathroom spans 6.1 stops) from a fixed one (gains only add
    /// freedom that costs held-out quality - Mip-NeRF 360 / Tanks and Temples, parity 2026-10-09).
    /// </summary>
    public static ExifExposure? ExtractExposure(ReadOnlySpan<byte> jpegBytes)
    {
        int start = TiffStart(jpegBytes, out int length);
        if (start < 0) return null;
        var tiff = jpegBytes.Slice(start, length);
        if (tiff.Length < 8) return null;
        bool le = tiff[0] == 'I' && tiff[1] == 'I';
        if (!le && !(tiff[0] == 'M' && tiff[1] == 'M')) return null;
        if (ReadUInt16(tiff, 2, le) != 0x002A) return null;
        uint ifd0 = ReadUInt32(tiff, 4, le);
        if (ifd0 >= tiff.Length) return null;
        uint exifIfd = FindIfdTag(tiff, (int)ifd0, le, TagExifIfdPointer);
        if (exifIfd == 0 || exifIfd >= tiff.Length) return null;
        float? t = null, n = null, iso = null;
        int count = GetIfdEntryCount(tiff, (int)exifIfd, le);
        for (int i = 0; i < count; i++)
        {
            int ofs = (int)exifIfd + 2 + i * 12;
            if (ofs + 12 > tiff.Length) break;
            ushort tag = ReadUInt16(tiff, ofs, le), type = ReadUInt16(tiff, ofs + 2, le);
            if ((tag == TagExposureTime || tag == TagFNumber) && type == TypeRational)
            {
                uint at = ReadUInt32(tiff, ofs + 8, le);
                if (at + 8 > tiff.Length) continue;
                uint num = ReadUInt32(tiff, (int)at, le), den = ReadUInt32(tiff, (int)at + 4, le);
                if (den == 0) continue;
                if (tag == TagExposureTime) t = (float)num / den; else n = (float)num / den;
            }
            else if (tag == TagIso && (type == TypeShort || type == TypeLong))
            {
                uint v = ReadUInt32FromValue(tiff.Slice(ofs + 8, 4), type, le);
                if (v > 0) iso = v;
            }
        }
        return new ExifExposure(t, n, iso);
    }

    /// <summary>Where the TIFF block of the JPEG's EXIF APP1 segment starts, or -1.</summary>
    private static int TiffStart(ReadOnlySpan<byte> jpegBytes, out int length)
    {
        length = 0;
        // JPEG must start with SOI marker
        if (jpegBytes.Length < 12 || jpegBytes[0] != 0xFF || jpegBytes[1] != 0xD8)
            return -1;

        // Scan for APP1 marker (0xFF 0xE1) containing EXIF
        int pos = 2;
        while (pos + 4 < jpegBytes.Length)
        {
            if (jpegBytes[pos] != 0xFF)
                return -1; // not a valid marker

            byte marker = jpegBytes[pos + 1];

            // Skip padding bytes
            if (marker == 0xFF) { pos++; continue; }

            // SOS marker — end of metadata
            if (marker == 0xDA) return -1;

            int segmentLength = BinaryPrimitives.ReadUInt16BigEndian(jpegBytes.Slice(pos + 2, 2));

            if (marker == 0xE1) // APP1
            {
                // Check for "Exif\0\0" header
                if (pos + 10 < jpegBytes.Length &&
                    jpegBytes[pos + 4] == 'E' && jpegBytes[pos + 5] == 'x' &&
                    jpegBytes[pos + 6] == 'i' && jpegBytes[pos + 7] == 'f' &&
                    jpegBytes[pos + 8] == 0 && jpegBytes[pos + 9] == 0)
                {
                    length = Math.Min(segmentLength - 8, jpegBytes.Length - (pos + 10));
                    return length > 0 ? pos + 10 : -1;
                }
            }

            pos += 2 + segmentLength;
        }

        return -1;
    }

    private static ExifFocalLength? ParseTiff(ReadOnlySpan<byte> tiff)
    {
        if (tiff.Length < 8) return null;

        // Byte order: "II" = little-endian, "MM" = big-endian
        bool littleEndian = tiff[0] == 'I' && tiff[1] == 'I';
        if (!littleEndian && !(tiff[0] == 'M' && tiff[1] == 'M'))
            return null;

        // Verify TIFF magic (0x002A)
        ushort magic = ReadUInt16(tiff, 2, littleEndian);
        if (magic != 0x002A) return null;

        // Offset to first IFD
        uint ifd0Offset = ReadUInt32(tiff, 4, littleEndian);
        if (ifd0Offset >= tiff.Length) return null;

        // Scan IFD0 for ExifIFD pointer
        uint exifIfdOffset = FindIfdTag(tiff, (int)ifd0Offset, littleEndian, TagExifIfdPointer);
        if (exifIfdOffset == 0 || exifIfdOffset >= tiff.Length)
            return null;

        // Scan ExifIFD for focal length tags
        float? focalLength35mm = null;
        float? focalLengthMm = null;

        int entryCount = GetIfdEntryCount(tiff, (int)exifIfdOffset, littleEndian);
        int entryStart = (int)exifIfdOffset + 2;

        for (int i = 0; i < entryCount; i++)
        {
            int ofs = entryStart + i * 12;
            if (ofs + 12 > tiff.Length) break;

            ushort tag = ReadUInt16(tiff, ofs, littleEndian);
            ushort type = ReadUInt16(tiff, ofs + 2, littleEndian);

            if (tag == TagFocalLength35mm && type == TypeShort)
            {
                ushort val = ReadUInt16(tiff, ofs + 8, littleEndian);
                if (val > 0) focalLength35mm = val;
            }
            else if (tag == TagFocalLength && type == TypeRational)
            {
                uint rationalOffset = ReadUInt32(tiff, ofs + 8, littleEndian);
                if (rationalOffset + 8 <= tiff.Length)
                {
                    uint num = ReadUInt32(tiff, (int)rationalOffset, littleEndian);
                    uint den = ReadUInt32(tiff, (int)rationalOffset + 4, littleEndian);
                    if (den > 0) focalLengthMm = (float)num / den;
                }
            }
        }

        if (focalLength35mm == null && focalLengthMm == null)
            return null;

        return new ExifFocalLength(focalLength35mm, focalLengthMm);
    }

    private static int GetIfdEntryCount(ReadOnlySpan<byte> tiff, int ifdOffset, bool littleEndian)
    {
        if (ifdOffset + 2 > tiff.Length) return 0;
        return ReadUInt16(tiff, ifdOffset, littleEndian);
    }

    /// <summary>
    /// Find a LONG/SHORT tag value in an IFD. Returns 0 if not found.
    /// </summary>
    private static uint FindIfdTag(ReadOnlySpan<byte> tiff, int ifdOffset, bool littleEndian, ushort targetTag)
    {
        int entryCount = GetIfdEntryCount(tiff, ifdOffset, littleEndian);
        int entryStart = ifdOffset + 2;

        for (int i = 0; i < entryCount; i++)
        {
            int ofs = entryStart + i * 12;
            if (ofs + 12 > tiff.Length) break;

            ushort tag = ReadUInt16(tiff, ofs, littleEndian);
            if (tag == targetTag)
            {
                ushort type = ReadUInt16(tiff, ofs + 2, littleEndian);
                return ReadUInt32FromValue(tiff.Slice(ofs + 8, 4), type, littleEndian);
            }
        }

        return 0;
    }

    private static ushort ReadUInt16(ReadOnlySpan<byte> data, int offset, bool littleEndian)
    {
        var slice = data.Slice(offset, 2);
        return littleEndian
            ? BinaryPrimitives.ReadUInt16LittleEndian(slice)
            : BinaryPrimitives.ReadUInt16BigEndian(slice);
    }

    private static uint ReadUInt32(ReadOnlySpan<byte> data, int offset, bool littleEndian)
    {
        var slice = data.Slice(offset, 4);
        return littleEndian
            ? BinaryPrimitives.ReadUInt32LittleEndian(slice)
            : BinaryPrimitives.ReadUInt32BigEndian(slice);
    }

    private static uint ReadUInt32FromValue(ReadOnlySpan<byte> valueBytes, ushort type, bool littleEndian)
    {
        if (type == TypeShort)
            return ReadUInt16(valueBytes, 0, littleEndian);
        return ReadUInt32(valueBytes, 0, littleEndian);
    }

    /// <summary>
    /// Convert 35mm-equivalent focal length to pixel focal length.
    /// Formula: f_pixels = f_35mm * max(width, height) / 36
    /// Where 36mm is the width of a 35mm film frame.
    /// </summary>
    public static float FocalLength35mmToPixels(float focalLength35mm, int width, int height)
        => focalLength35mm * MathF.Max(width, height) / 36f;
}
