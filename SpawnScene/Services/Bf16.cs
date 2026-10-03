namespace SpawnScene.Services;

/// <summary>
/// bfloat16: the top 16 bits of an f32 (same sign and 8-bit exponent, 7-bit mantissa), two to a 32-bit word, low half
/// first. The trainer keeps the SH Adam moments in it (SplatTrainerShaders.AdamShRest): f32's exponent range, so a tiny
/// second moment does not underflow the way IEEE half would, at half the memory. The GPU stores them with stochastic
/// rounding; this host side (gate and densify restores) rounds to nearest even.
/// </summary>
public static class Bf16
{
    /// <summary>Words per row of <paramref name="floatsPerRow"/> values (two per word, the last half-empty when odd).</summary>
    public static int WordsPerRow(int floatsPerRow) => (floatsPerRow + 1) / 2;

    /// <summary>Round to nearest, ties to even. NaN stays NaN; infinities stay infinite.</summary>
    public static ushort FromFloat(float x)
    {
        uint b = BitConverter.SingleToUInt32Bits(x);
        if ((b & 0x7f800000u) == 0x7f800000u)
            return (ushort)((b >> 16) | ((b & 0xffffu) != 0 ? 0x40u : 0u)); // keep NaN a NaN after truncation
        uint lsb = (b >> 16) & 1u;
        return (ushort)((b + 0x7fffu + lsb) >> 16);
    }

    public static float ToFloat(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    /// <summary>The value bf16 storage keeps of <paramref name="x"/>.</summary>
    public static float Round(float x) => ToFloat(FromFloat(x));

    /// <summary>Pack rows of <paramref name="floatsPerRow"/> floats into <see cref="WordsPerRow"/> words each.</summary>
    public static uint[] Pack(ReadOnlySpan<float> values, int floatsPerRow)
    {
        int rows = values.Length / floatsPerRow, words = WordsPerRow(floatsPerRow);
        var packed = new uint[rows * words];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < floatsPerRow; j++)
                packed[r * words + j / 2] |= (uint)FromFloat(values[r * floatsPerRow + j]) << (16 * (j & 1));
        return packed;
    }

    /// <summary>The inverse of <see cref="Pack"/>.</summary>
    public static float[] Unpack(ReadOnlySpan<uint> packed, int floatsPerRow)
    {
        int words = WordsPerRow(floatsPerRow), rows = packed.Length / words;
        var values = new float[rows * floatsPerRow];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < floatsPerRow; j++)
                values[r * floatsPerRow + j] = ToFloat((ushort)(packed[r * words + j / 2] >> (16 * (j & 1))));
        return values;
    }
}
