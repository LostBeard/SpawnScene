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

    /// <summary>
    /// The ONE WGSL evaluation of view-dependent colour, shared by the trainer (SplatTrainerShaders) and the
    /// viewer's pack pass (GpuGaussianRenderer) so the two renderers of one buffer cannot drift. Requires a
    /// storage binding named <c>sh_rest</c> (45 floats per splat, <see cref="RestFloatsPerSplat"/> layout).
    /// Returns display RGB: C0*dc + bands + 0.5, clamped at 0 - the same as <see cref="EvalRgb"/>.
    /// </summary>
    public const string WgslViewRgb = @"
fn sh_view_rgb(base : u32, dir : vec3<f32>, dc : vec3<f32>, deg : u32) -> vec3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    var result = SH_C0 * dc;

    if (deg >= 1u) {
        let sh1 = vec3<f32>(sh_rest[base + 0u], sh_rest[base + 1u], sh_rest[base + 2u]);
        let sh2 = vec3<f32>(sh_rest[base + 3u], sh_rest[base + 4u], sh_rest[base + 5u]);
        let sh3 = vec3<f32>(sh_rest[base + 6u], sh_rest[base + 7u], sh_rest[base + 8u]);
        result = result + SH_C1 * (-y * sh1 + z * sh2 - x * sh3);
    }
    if (deg >= 2u) {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let sh4 = vec3<f32>(sh_rest[base + 9u], sh_rest[base + 10u], sh_rest[base + 11u]);
        let sh5 = vec3<f32>(sh_rest[base + 12u], sh_rest[base + 13u], sh_rest[base + 14u]);
        let sh6 = vec3<f32>(sh_rest[base + 15u], sh_rest[base + 16u], sh_rest[base + 17u]);
        let sh7 = vec3<f32>(sh_rest[base + 18u], sh_rest[base + 19u], sh_rest[base + 20u]);
        let sh8 = vec3<f32>(sh_rest[base + 21u], sh_rest[base + 22u], sh_rest[base + 23u]);
        result = result
            + 1.0925484305920792 * xy * sh4
            + (-1.0925484305920792) * yz * sh5
            + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh6
            + (-1.0925484305920792) * xz * sh7
            + 0.5462742152960396 * (xx - yy) * sh8;
    }
    if (deg >= 3u) {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let sh9 = vec3<f32>(sh_rest[base + 24u], sh_rest[base + 25u], sh_rest[base + 26u]);
        let sh10 = vec3<f32>(sh_rest[base + 27u], sh_rest[base + 28u], sh_rest[base + 29u]);
        let sh11 = vec3<f32>(sh_rest[base + 30u], sh_rest[base + 31u], sh_rest[base + 32u]);
        let sh12 = vec3<f32>(sh_rest[base + 33u], sh_rest[base + 34u], sh_rest[base + 35u]);
        let sh13 = vec3<f32>(sh_rest[base + 36u], sh_rest[base + 37u], sh_rest[base + 38u]);
        let sh14 = vec3<f32>(sh_rest[base + 39u], sh_rest[base + 40u], sh_rest[base + 41u]);
        let sh15 = vec3<f32>(sh_rest[base + 42u], sh_rest[base + 43u], sh_rest[base + 44u]);
        result = result
            + (-0.5900435899266435) * y * (3.0 * xx - yy) * sh9
            + 2.890611442640554 * xy * z * sh10
            + (-0.4570457994644658) * y * (4.0 * zz - xx - yy) * sh11
            + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh12
            + (-0.4570457994644658) * x * (4.0 * zz - xx - yy) * sh13
            + 1.445305721320277 * z * (xx - yy) * sh14
            + (-0.5900435899266435) * x * (xx - 3.0 * yy) * sh15;
    }
    return max(result + vec3<f32>(0.5), vec3<f32>(0.0));
}
";

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
