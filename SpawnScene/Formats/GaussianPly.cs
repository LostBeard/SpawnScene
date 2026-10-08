using System.Text;

namespace SpawnScene.Formats;

/// <summary>
/// The header of a 3D Gaussian Splatting .ply (graphdeco-inria's layout, written by the reference trainer, gsplat,
/// nerfstudio, Postshot, Polycam and most tools that export "3DGS PLY"): binary little-endian vertices with x y z,
/// f_dc_0..2 (SH degree 0), f_rest_* (the higher bands, CHANNEL-major: all of red's, then green's, then blue's), opacity
/// (a logit), scale_0..2 (log) and rot_0..3 (quaternion w x y z, not normalised). Other properties (normals, colours)
/// are skipped by their size. The vertex data itself is converted on the GPU (Services/GaussianPlyImport).
/// </summary>
public static class GaussianPly
{
    /// <summary>Where each property sits inside one vertex (byte offsets), and how big a vertex is.</summary>
    public sealed record Layout(
        long Count, int HeaderBytes, int StrideBytes,
        int X, int Y, int Z, int Dc0, int Dc1, int Dc2, int Opacity,
        int Scale0, int Scale1, int Scale2, int Rot0, int Rot1, int Rot2, int Rot3,
        int RestFirst, int RestPerChannel)
    {
        /// <summary>SH degree the file carries (0..3), from the f_rest count.</summary>
        public int ShDegree => RestPerChannel switch { 0 => 0, 3 => 1, 8 => 2, _ => 3 };
        public long DataBytes => Count * StrideBytes;
    }

    /// <summary>True when <paramref name="head"/> starts like a PLY file.</summary>
    public static bool IsPly(ReadOnlySpan<byte> head) =>
        head.Length >= 4 && head[0] == (byte)'p' && head[1] == (byte)'l' && head[2] == (byte)'y' && (head[3] == (byte)'\n' || head[3] == (byte)'\r');

    static int SizeOf(string type) => type switch
    {
        "char" or "uchar" or "int8" or "uint8" => 1,
        "short" or "ushort" or "int16" or "uint16" => 2,
        "int" or "uint" or "float" or "int32" or "uint32" or "float32" => 4,
        "double" or "float64" => 8,
        _ => throw new FormatException($"PLY property type '{type}' is not supported"),
    };

    /// <summary>
    /// Parse the header from the first bytes of the file (it must contain <c>end_header</c>; 64 KB is plenty).
    /// Throws <see cref="FormatException"/> with a reason a person can act on when the file is not a 3DGS PLY.
    /// </summary>
    public static Layout Parse(ReadOnlySpan<byte> head)
    {
        if (!IsPly(head)) throw new FormatException("not a PLY file");
        var text = Encoding.ASCII.GetString(head);
        int end = text.IndexOf("end_header", StringComparison.Ordinal);
        if (end < 0) throw new FormatException("PLY header has no end_header in its first bytes");
        int headerBytes = text.IndexOf('\n', end) + 1;
        if (headerBytes <= 0) throw new FormatException("PLY header is cut off");
        var lines = text[..end].Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        string? format = null;
        long count = -1;
        bool inVertex = false, vertexSeen = false;
        int stride = 0;
        var offsets = new Dictionary<string, int>(StringComparer.Ordinal);
        var types = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var line in lines)
        {
            var t = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (t.Length == 0) continue;
            switch (t[0])
            {
                case "format": format = t.Length > 1 ? t[1] : null; break;
                case "element":
                    if (t.Length < 3) throw new FormatException($"bad PLY line '{line}'");
                    if (vertexSeen) { inVertex = false; break; }   // elements after the vertices are not read
                    if (t[1] != "vertex")
                        throw new FormatException($"PLY element '{t[1]}' comes before the vertices (a compressed or packed PLY?) - not supported yet");
                    count = long.Parse(t[2], System.Globalization.CultureInfo.InvariantCulture);
                    inVertex = vertexSeen = true;
                    break;
                case "property":
                    if (!inVertex) break;
                    if (t.Length >= 2 && t[1] == "list") throw new FormatException("PLY vertex has a list property - not a 3DGS PLY");
                    if (t.Length < 3) throw new FormatException($"bad PLY line '{line}'");
                    offsets[t[2]] = stride;
                    types[t[2]] = t[1];
                    stride += SizeOf(t[1]);
                    break;
            }
        }
        if (format != "binary_little_endian")
            throw new FormatException($"PLY format '{format}' - only binary_little_endian is read");
        if (count < 0) throw new FormatException("PLY has no vertex element");

        int Need(string name)
        {
            if (!offsets.TryGetValue(name, out int o))
                throw new FormatException($"PLY has no '{name}' - not a Gaussian splat file (a plain point cloud or mesh?)");
            if (types[name] is not ("float" or "float32")) throw new FormatException($"PLY '{name}' is {types[name]}, expected float");
            return o;
        }
        int rest = 0;
        while (offsets.ContainsKey($"f_rest_{rest}")) rest++;
        if (rest is not (0 or 9 or 24 or 45)) throw new FormatException($"PLY has {rest} f_rest values - expected 0, 9, 24 or 45");
        int restFirst = rest > 0 ? Need("f_rest_0") : 0;
        // The f_rest values must be consecutive floats (they are, in every writer) - the kernel reads them by index.
        for (int k = 1; k < rest; k++)
            if (Need($"f_rest_{k}") != restFirst + 4 * k) throw new FormatException("PLY f_rest values are not consecutive");

        return new Layout(count, headerBytes, stride,
            Need("x"), Need("y"), Need("z"), Need("f_dc_0"), Need("f_dc_1"), Need("f_dc_2"), Need("opacity"),
            Need("scale_0"), Need("scale_1"), Need("scale_2"), Need("rot_0"), Need("rot_1"), Need("rot_2"), Need("rot_3"),
            restFirst, rest / 3);
    }
}
