using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Real SH RGB evaluation for 3DGS (Kerbl / graphdeco-inria), degrees 0-3.
///
/// Coefficient layout matches the reference and cvlab-epfl/gaussian-splatting-web:
/// DC is three floats (stored in <see cref="SplatFormat.OffColor"/>), rest is
/// <see cref="RestFloatsPerSplat"/> = 15 vec3 bands (45 floats), index (k-1)*3+c for band k in 1..15.
/// </summary>
public static class SphericalHarmonics
{
    public const float C0 = 0.28209479177387814f;
    public const float C1 = 0.4886025119029199f;

    public static readonly float[] C2 =
    [
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f,
    ];

    public static readonly float[] C3 =
    [
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f,
    ];

    public const int MaxDegree = 3;
    public const int RestBands = 15;
    public const int RestFloatsPerSplat = RestBands * 3;

    public static Vector3 RgbToDc(Vector3 rgb) => (rgb - new Vector3(0.5f)) / C0;

    public static Vector3 DcToRgb(Vector3 dc) => dc * C0 + new Vector3(0.5f);

    /// <summary>View-dependent linear RGB, clamped at zero like the reference forward.</summary>
    public static Vector3 EvalRgb(int degree, ReadOnlySpan<float> dc, ReadOnlySpan<float> rest, Vector3 dir)
    {
        degree = Math.Clamp(degree, 0, MaxDegree);
        dir = Vector3.Normalize(dir);

        float x = dir.X, y = dir.Y, z = dir.Z;
        var result = C0 * new Vector3(dc[0], dc[1], dc[2]);

        if (degree >= 1 && rest.Length >= 9)
        {
            var sh1 = new Vector3(rest[0], rest[1], rest[2]);
            var sh2 = new Vector3(rest[3], rest[4], rest[5]);
            var sh3 = new Vector3(rest[6], rest[7], rest[8]);
            result += C1 * (-y * sh1 + z * sh2 - x * sh3);
        }

        if (degree >= 2 && rest.Length >= 24)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, xz = x * z, yz = y * z;
            var sh4 = new Vector3(rest[9], rest[10], rest[11]);
            var sh5 = new Vector3(rest[12], rest[13], rest[14]);
            var sh6 = new Vector3(rest[15], rest[16], rest[17]);
            var sh7 = new Vector3(rest[18], rest[19], rest[20]);
            var sh8 = new Vector3(rest[21], rest[22], rest[23]);
            result += C2[0] * xy * sh4
                    + C2[1] * yz * sh5
                    + C2[2] * (2f * zz - xx - yy) * sh6
                    + C2[3] * xz * sh7
                    + C2[4] * (xx - yy) * sh8;
        }

        if (degree >= 3 && rest.Length >= RestFloatsPerSplat)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, xz = x * z, yz = y * z;
            var sh9 = new Vector3(rest[24], rest[25], rest[26]);
            var sh10 = new Vector3(rest[27], rest[28], rest[29]);
            var sh11 = new Vector3(rest[30], rest[31], rest[32]);
            var sh12 = new Vector3(rest[33], rest[34], rest[35]);
            var sh13 = new Vector3(rest[36], rest[37], rest[38]);
            var sh14 = new Vector3(rest[39], rest[40], rest[41]);
            var sh15 = new Vector3(rest[42], rest[43], rest[44]);
            result += C3[0] * y * (3f * xx - yy) * sh9
                    + C3[1] * xy * z * sh10
                    + C3[2] * y * (4f * zz - xx - yy) * sh11
                    + C3[3] * z * (2f * zz - 3f * xx - 3f * yy) * sh12
                    + C3[4] * x * (4f * zz - xx - yy) * sh13
                    + C3[5] * z * (xx - yy) * sh14
                    + C3[6] * x * (xx - 3f * yy) * sh15;
        }

        result += new Vector3(0.5f);
        return Vector3.Max(result, Vector3.Zero);
    }

    /// <summary>Progressive SH degree: +1 every 1000 iterations, capped at 3 (Kerbl default).</summary>
    public static int DegreeForIteration(int iteration) =>
        Math.Min(MaxDegree, iteration / 1000);
}
