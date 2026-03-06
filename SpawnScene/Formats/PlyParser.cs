using SpawnScene.Models;
using System.Text;

namespace SpawnScene.Formats;

/// <summary>
/// Parses .ply files containing 3D Gaussian Splat data.
/// Supports both binary (little-endian) and ASCII formats.
/// 
/// Expected properties per vertex (Gaussian):
///   x, y, z           — position
///   scale_0..2         — log-scale
///   rot_0..3           — quaternion (w, x, y, z)
///   opacity            — logit opacity
///   f_dc_0..2          — SH DC coefficients (base RGB)
///   f_rest_0..N        — higher-order SH (optional, ignored for SH degree 0)
/// </summary>
public static class PlyParser
{
    private enum PlyFormat { Ascii, BinaryLittleEndian, BinaryBigEndian }

    private class PlyProperty
    {
        public string Name { get; set; } = "";
        public string Type { get; set; } = "float";
        public int ByteSize => Type switch
        {
            "float" or "float32" => 4,
            "double" or "float64" => 8,
            "uchar" or "uint8" => 1,
            "char" or "int8" => 1,
            "ushort" or "uint16" => 2,
            "short" or "int16" => 2,
            "uint" or "uint32" => 4,
            "int" or "int32" => 4,
            _ => 4
        };
    }

    /// <summary>
    /// Parse a .ply file from raw bytes into a GaussianScene.
    /// </summary>
    public static GaussianScene Parse(byte[] data)
    {
        int headerEnd = FindHeaderEnd(data);
        if (headerEnd < 0)
            throw new FormatException("Invalid PLY file: could not find end_header");

        string headerText = Encoding.ASCII.GetString(data, 0, headerEnd);
        var (format, vertexCount, properties) = ParseHeader(headerText);

        int dataStart = headerEnd + Encoding.ASCII.GetByteCount("end_header\n");
        // Try \r\n ending
        if (dataStart <= data.Length && dataStart > headerEnd + 11)
        {
            // Already correct
        }
        else
        {
            dataStart = headerEnd + Encoding.ASCII.GetByteCount("end_header\r\n");
        }

        Gaussian3D[] gaussians;

        if (format == PlyFormat.BinaryLittleEndian)
        {
            gaussians = ParseBinaryLE(data, dataStart, vertexCount, properties);
        }
        else if (format == PlyFormat.Ascii)
        {
            gaussians = ParseAscii(data, dataStart, vertexCount, properties);
        }
        else
        {
            throw new NotSupportedException("Binary big-endian PLY not supported");
        }

        var scene = new GaussianScene
        {
            Gaussians = gaussians,
            ShDegree = 0,
        };
        scene.ComputeBounds();
        return scene;
    }

    private static int FindHeaderEnd(byte[] data)
    {
        var needle = Encoding.ASCII.GetBytes("end_header");
        for (int i = 0; i < data.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (data[i + j] != needle[j]) { match = false; break; }
            }
            if (match) return i;
        }
        return -1;
    }

    private static (PlyFormat format, int vertexCount, List<PlyProperty> properties) ParseHeader(string header)
    {
        var format = PlyFormat.BinaryLittleEndian;
        int vertexCount = 0;
        var properties = new List<PlyProperty>();
        bool inVertexElement = false;

        foreach (var rawLine in header.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.StartsWith("format"))
            {
                if (line.Contains("ascii")) format = PlyFormat.Ascii;
                else if (line.Contains("binary_big_endian")) format = PlyFormat.BinaryBigEndian;
                else format = PlyFormat.BinaryLittleEndian;
            }
            else if (line.StartsWith("element vertex"))
            {
                vertexCount = int.Parse(line.Split(' ')[2]);
                inVertexElement = true;
            }
            else if (line.StartsWith("element") && !line.StartsWith("element vertex"))
            {
                inVertexElement = false;
            }
            else if (line.StartsWith("property") && inVertexElement)
            {
                var parts = line.Split(' ');
                if (parts.Length >= 3 && parts[0] == "property" && parts[1] != "list")
                {
                    properties.Add(new PlyProperty { Type = parts[1], Name = parts[2] });
                }
            }
        }

        return (format, vertexCount, properties);
    }

    private static Gaussian3D[] ParseBinaryLE(byte[] data, int offset, int count, List<PlyProperty> properties)
    {
        var gaussians = new Gaussian3D[count];
        int stride = properties.Sum(p => p.ByteSize);

        // Build lookup for property indices
        var propIndex = new Dictionary<string, int>();
        for (int i = 0; i < properties.Count; i++)
            propIndex[properties[i].Name] = i;

        // Precompute byte offsets for each property
        var byteOffsets = new int[properties.Count];
        int running = 0;
        for (int i = 0; i < properties.Count; i++)
        {
            byteOffsets[i] = running;
            running += properties[i].ByteSize;
        }

        for (int i = 0; i < count; i++)
        {
            int vertexStart = offset + i * stride;

            gaussians[i] = new Gaussian3D
            {
                X = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "x"),
                Y = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "y"),
                Z = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "z"),
                ScaleX = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "scale_0"),
                ScaleY = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "scale_1"),
                ScaleZ = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "scale_2"),
                RotW = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "rot_0"),
                RotX = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "rot_1"),
                RotY = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "rot_2"),
                RotZ = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "rot_3"),
                OpacityLogit = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "opacity"),
                SH_DC_R = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "f_dc_0"),
                SH_DC_G = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "f_dc_1"),
                SH_DC_B = ReadFloat(data, vertexStart, propIndex, byteOffsets, properties, "f_dc_2"),
            };
        }

        return gaussians;
    }

    private static float ReadFloat(byte[] data, int vertexStart,
        Dictionary<string, int> propIndex, int[] byteOffsets,
        List<PlyProperty> properties, string name)
    {
        if (!propIndex.TryGetValue(name, out int idx)) return 0f;
        int pos = vertexStart + byteOffsets[idx];
        var prop = properties[idx];

        return prop.Type switch
        {
            "float" or "float32" => BitConverter.ToSingle(data, pos),
            "double" or "float64" => (float)BitConverter.ToDouble(data, pos),
            "uchar" or "uint8" => data[pos] / 255.0f,
            _ => BitConverter.ToSingle(data, pos),
        };
    }

    private static Gaussian3D[] ParseAscii(byte[] data, int offset, int count, List<PlyProperty> properties)
    {
        var text = Encoding.ASCII.GetString(data, offset, data.Length - offset);
        var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var gaussians = new Gaussian3D[Math.Min(count, lines.Length)];

        var propIndex = new Dictionary<string, int>();
        for (int i = 0; i < properties.Count; i++)
            propIndex[properties[i].Name] = i;

        for (int i = 0; i < gaussians.Length; i++)
        {
            var values = lines[i].Trim().Split(' ');

            gaussians[i] = new Gaussian3D
            {
                X = GetVal(values, propIndex, "x"),
                Y = GetVal(values, propIndex, "y"),
                Z = GetVal(values, propIndex, "z"),
                ScaleX = GetVal(values, propIndex, "scale_0"),
                ScaleY = GetVal(values, propIndex, "scale_1"),
                ScaleZ = GetVal(values, propIndex, "scale_2"),
                RotW = GetVal(values, propIndex, "rot_0"),
                RotX = GetVal(values, propIndex, "rot_1"),
                RotY = GetVal(values, propIndex, "rot_2"),
                RotZ = GetVal(values, propIndex, "rot_3"),
                OpacityLogit = GetVal(values, propIndex, "opacity"),
                SH_DC_R = GetVal(values, propIndex, "f_dc_0"),
                SH_DC_G = GetVal(values, propIndex, "f_dc_1"),
                SH_DC_B = GetVal(values, propIndex, "f_dc_2"),
            };
        }

        return gaussians;
    }

    private static float GetVal(string[] values, Dictionary<string, int> propIndex, string name)
    {
        if (!propIndex.TryGetValue(name, out int idx) || idx >= values.Length) return 0f;
        return float.TryParse(values[idx], System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float val) ? val : 0f;
    }
}
