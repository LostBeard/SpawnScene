using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// CPU-oracle world↔camera geometry for Middlebury / OpenCV GT poses.
/// The GPU <c>UnprojectWorldSpaceKernel</c> MUST match <see cref="UnprojectPixel"/> bit-for-bit
/// (same cam axes, same pixel unproject). Gate: SpawnScene.Tests TempleRingWorldSpaceTests.
/// </summary>
public static class WorldSpaceGeometry
{
    /// <summary>
    /// OpenCV/Middlebury camera axes in world: row0=right, row1=down, row2=forward.
    /// <see cref="CameraParams.Up"/> is stored Y-up (= -down).
    /// </summary>
    public static void GetOpenCvAxes(CameraParams cam, out Vector3 right, out Vector3 down, out Vector3 forward)
    {
        forward = Vector3.Normalize(cam.Forward);
        var up = Vector3.Normalize(cam.Up); // Y-up
        right = Vector3.Normalize(Vector3.Cross(forward, up));
        down = -up; // OpenCV Y-down
        // Re-orthogonalize right against (down, forward) if needed
        right = Vector3.Normalize(Vector3.Cross(down, forward));
    }

    /// <summary>
    /// World → camera (OpenCV): Xc = [right|down|forward]^T * (Xw - C) … i.e. dot with each axis.
    /// </summary>
    public static Vector3 WorldToCamera(CameraParams cam, Vector3 world)
    {
        GetOpenCvAxes(cam, out var right, out var down, out var forward);
        var d = world - cam.Position;
        return new Vector3(Vector3.Dot(right, d), Vector3.Dot(down, d), Vector3.Dot(forward, d));
    }

    /// <summary>
    /// Camera → world: Xw = C + right*x + down*y + forward*z.
    /// </summary>
    public static Vector3 CameraToWorld(CameraParams cam, Vector3 camPt)
    {
        GetOpenCvAxes(cam, out var right, out var down, out var forward);
        return cam.Position + right * camPt.X + down * camPt.Y + forward * camPt.Z;
    }

    /// <summary>
    /// Project world point to pixel (OpenCV pinhole). Returns false if behind camera.
    /// </summary>
    public static bool Project(CameraParams cam, Vector3 world, out float u, out float v, out float zCam)
    {
        var xc = WorldToCamera(cam, world);
        zCam = xc.Z;
        if (zCam <= 1e-6f) { u = v = 0; return false; }
        u = cam.FocalX * xc.X / zCam + cam.CenterX;
        v = cam.FocalY * xc.Y / zCam + cam.CenterY;
        return true;
    }

    /// <summary>
    /// Unproject pixel + depth (camera Z) → world. Matches GPU UnprojectWorldSpaceKernel.
    /// </summary>
    public static Vector3 UnprojectPixel(CameraParams cam, float u, float v, float depthCamZ)
    {
        float x = (u - cam.CenterX) * depthCamZ / cam.FocalX;
        float y = (v - cam.CenterY) * depthCamZ / cam.FocalY;
        return CameraToWorld(cam, new Vector3(x, y, depthCamZ));
    }

    /// <summary>
    /// Parse Middlebury templeR_par.txt / dinoSR_par.txt.
    /// Format per line: filename K[0..8] R[0..8] t[0..2] (row-major).
    /// </summary>
    public static List<(string filename, CameraParams camera)> ParseMiddleburyParams(
        string parFileContent, int imageWidth, int imageHeight)
    {
        var results = new List<(string, CameraParams)>();
        var lines = parFileContent.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 22) continue;

            string filename = parts[0];
            float fx = float.Parse(parts[1]);
            float fy = float.Parse(parts[5]);
            float cx = float.Parse(parts[3]);
            float cy = float.Parse(parts[6]);

            var R = new double[3, 3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R[r, c] = double.Parse(parts[10 + r * 3 + c]);

            var t = new double[]
            {
                double.Parse(parts[19]),
                double.Parse(parts[20]),
                double.Parse(parts[21])
            };

            // C = -R^T * t ; Forward = R[2,:] ; Up (Y-up) = -R[1,:]
            var cam = new CameraParams
            {
                Width = imageWidth,
                Height = imageHeight,
                FocalX = fx,
                FocalY = fy,
                CenterX = cx,
                CenterY = cy,
                Forward = new Vector3((float)R[2, 0], (float)R[2, 1], (float)R[2, 2]),
                Up = new Vector3(-(float)R[1, 0], -(float)R[1, 1], -(float)R[1, 2]),
                Position = new Vector3(
                    -((float)(R[0, 0] * t[0] + R[1, 0] * t[1] + R[2, 0] * t[2])),
                    -((float)(R[0, 1] * t[0] + R[1, 1] * t[1] + R[2, 1] * t[2])),
                    -((float)(R[0, 2] * t[0] + R[1, 2] * t[1] + R[2, 2] * t[2]))),
            };
            results.Add((filename, cam));
        }

        return results;
    }

    /// <summary>
    /// Raw Middlebury R,t (no CameraParams) — ground-truth oracle for bisect.
    /// Xc = R*Xw + t ; C = -R^T*t ; Xw = R^T*(Xc - t) wait: Xw = R^T*Xc + C with C=-R^T*t.
    /// </summary>
    public static Vector3 UnprojectRaw(float[,] R, float[] t, float fx, float fy, float cx, float cy,
        float u, float v, float z)
    {
        float x = (u - cx) * z / fx;
        float y = (v - cy) * z / fy;
        // Xw = R^T * Xc + C, C = -R^T * t  ⇒ Xw = R^T * (Xc - t)
        float cx_ = x - t[0], cy_ = y - t[1], cz_ = z - t[2];
        return new Vector3(
            R[0, 0] * cx_ + R[1, 0] * cy_ + R[2, 0] * cz_,
            R[0, 1] * cx_ + R[1, 1] * cy_ + R[2, 1] * cz_,
            R[0, 2] * cx_ + R[1, 2] * cy_ + R[2, 2] * cz_);
    }

    public static Vector3 CameraCenterFromRt(float[,] R, float[] t) => new(
        -(R[0, 0] * t[0] + R[1, 0] * t[1] + R[2, 0] * t[2]),
        -(R[0, 1] * t[0] + R[1, 1] * t[1] + R[2, 1] * t[2]),
        -(R[0, 2] * t[0] + R[1, 2] * t[1] + R[2, 2] * t[2]));

    /// <summary>
    /// CPU-oracle twin of the GPU consistency fuse.
    /// Keep only when the splat projects into the ref view AND depths agree.
    /// Out-of-frustum / behind / missing ref depth → drop (avoids relative-MDE floaters).
    /// Gate: <c>TempleRingWorldSpaceTests.ConsistencyFuse_*</c>.
    /// </summary>
    public static bool ShouldKeepSplatVsRef(
        float zCam, float refDepthRaw, float splatConf, float refConf,
        float depthScale, float relThresh, bool inBounds, bool hasConf)
    {
        if (!inBounds || zCam <= 1e-6f) return false;
        float refZ = refDepthRaw * depthScale;
        if (!(refZ > 1e-4f)) return false;
        float denom = MathF.Max(refZ, zCam);
        float rel = MathF.Abs(zCam - refZ) / denom;
        return rel <= relThresh;
    }

    /// <summary>
    /// Farthest-point sampling on camera positions — avoids near-duplicate views
    /// (CDP: index-spaced picks still selected templeR0001≈templeR0031).
    /// </summary>
    public static List<int> PickFarthestCameras(IReadOnlyList<CameraParams> cams, int count)
    {
        if (cams.Count == 0) return new List<int>();
        count = Math.Min(count, cams.Count);
        var picked = new List<int> { 0 };
        while (picked.Count < count)
        {
            int best = -1;
            float bestMin = -1f;
            for (int i = 0; i < cams.Count; i++)
            {
                if (picked.Contains(i)) continue;
                float minD = float.MaxValue;
                foreach (int p in picked)
                    minD = MathF.Min(minD, Vector3.Distance(cams[i].Position, cams[p].Position));
                if (minD > bestMin) { bestMin = minD; best = i; }
            }
            if (best < 0) break;
            picked.Add(best);
        }
        picked.Sort();
        return picked;
    }

    /// <summary>
    /// Umeyama similarity: p' = scale * R * p + t mapping source → target (e.g. DAv3 cams → GT cams).
    /// Returns false if degenerate (&lt;2 points or near-zero variance).
    /// </summary>
    public static bool TryUmeyamaSimilarity(
        IReadOnlyList<Vector3> source, IReadOnlyList<Vector3> target,
        out float scale, out Matrix4x4 rotation, out Vector3 translation, out float rms)
    {
        scale = 1f;
        rotation = Matrix4x4.Identity;
        translation = Vector3.Zero;
        rms = float.MaxValue;
        int n = Math.Min(source.Count, target.Count);
        if (n < 2) return false;

        Vector3 cs = Vector3.Zero, ct = Vector3.Zero;
        for (int i = 0; i < n; i++) { cs += source[i]; ct += target[i]; }
        cs /= n; ct /= n;

        // 3×3 cross-covariance H = Σ (s' * t'^T); also variance of source.
        float h00 = 0, h01 = 0, h02 = 0, h10 = 0, h11 = 0, h12 = 0, h20 = 0, h21 = 0, h22 = 0;
        float varS = 0;
        for (int i = 0; i < n; i++)
        {
            var a = source[i] - cs;
            var b = target[i] - ct;
            varS += a.LengthSquared();
            h00 += a.X * b.X; h01 += a.X * b.Y; h02 += a.X * b.Z;
            h10 += a.Y * b.X; h11 += a.Y * b.Y; h12 += a.Y * b.Z;
            h20 += a.Z * b.X; h21 += a.Z * b.Y; h22 += a.Z * b.Z;
        }
        if (varS < 1e-12f) return false;

        // SVD via Matrix4x4 trick is awkward; use closed-form 3×3 via System.Numerics isn't available.
        // Build H as rows and use a compact Jacobi-free path: Quaternion from covariance (Horn).
        // Horn's method: N matrix from H.
        float[] N = new float[16];
        float tr = h00 + h11 + h22;
        N[0] = tr;
        N[1] = h12 - h21; N[2] = h20 - h02; N[3] = h01 - h10;
        N[4] = h12 - h21;
        N[5] = h00 - h11 - h22; N[6] = h01 + h10; N[7] = h02 + h20;
        N[8] = h20 - h02;
        N[9] = h01 + h10; N[10] = -h00 + h11 - h22; N[11] = h12 + h21;
        N[12] = h01 - h10;
        N[13] = h02 + h20; N[14] = h12 + h21; N[15] = -h00 - h11 + h22;

        // Power iteration for dominant eigenvector of symmetric N (quaternion).
        float qw = 1, qx = 0, qy = 0, qz = 0;
        for (int it = 0; it < 64; it++)
        {
            float rw = N[0] * qw + N[1] * qx + N[2] * qy + N[3] * qz;
            float rx = N[4] * qw + N[5] * qx + N[6] * qy + N[7] * qz;
            float ry = N[8] * qw + N[9] * qx + N[10] * qy + N[11] * qz;
            float rz = N[12] * qw + N[13] * qx + N[14] * qy + N[15] * qz;
            float len = MathF.Sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
            if (len < 1e-20f) return false;
            qw = rw / len; qx = rx / len; qy = ry / len; qz = rz / len;
        }

        // Quaternion → rotation matrix (row-major action on column vectors).
        float xx = qx * qx, yy = qy * qy, zz = qz * qz;
        float xy = qx * qy, xz = qx * qz, yz = qy * qz;
        float wx = qw * qx, wy = qw * qy, wz = qw * qz;
        float r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
        float r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
        float r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);

        // scale = Σ (t' · R s') / Σ ||s'||²
        float num = 0;
        for (int i = 0; i < n; i++)
        {
            var a = source[i] - cs;
            var ra = new Vector3(
                r00 * a.X + r01 * a.Y + r02 * a.Z,
                r10 * a.X + r11 * a.Y + r12 * a.Z,
                r20 * a.X + r21 * a.Y + r22 * a.Z);
            num += Vector3.Dot(target[i] - ct, ra);
        }
        scale = num / varS;
        if (!(scale > 1e-6f) || scale > 1e6f) return false;

        // System.Numerics Vector3.Transform uses row-vector v*M; pack R^T into rows.
        rotation = new Matrix4x4(
            r00, r10, r20, 0,
            r01, r11, r21, 0,
            r02, r12, r22, 0,
            0, 0, 0, 1);

        var rCs = Vector3.Transform(cs, rotation);
        translation = ct - scale * rCs;

        float err2 = 0;
        for (int i = 0; i < n; i++)
        {
            var p = scale * Vector3.Transform(source[i], rotation) + translation;
            err2 += (p - target[i]).LengthSquared();
        }
        rms = MathF.Sqrt(err2 / n);
        return true;
    }

    public static Vector3 ApplySimilarity(Vector3 p, float scale, Matrix4x4 rotation, Vector3 translation)
        => scale * Vector3.Transform(p, rotation) + translation;

    /// <summary>
    /// Forward vector for a yaw/pitch FPS camera. Yaw turns about world +Y, pitch is the
    /// elevation, and yaw = 0 looks down world -Z.
    /// </summary>
    public static Vector3 ForwardFromYawPitch(float yaw, float pitch)
    {
        float cp = MathF.Cos(pitch);
        return new Vector3(cp * MathF.Sin(yaw), MathF.Sin(pitch), -cp * MathF.Cos(yaw));
    }

    /// <summary>
    /// Exact inverse of <see cref="ForwardFromYawPitch"/>. Note the NEGATED z in the atan2:
    /// forward.z is -cos(yaw)*cos(pitch), so <c>Atan2(x, z)</c> gives a yaw 180 degrees out and
    /// makes the view snap the first time the user looks around after an exact pose is set.
    /// Gate: <c>CameraPoseTests.YawPitch_RoundTripsThroughForward</c>.
    /// </summary>
    public static void YawPitchFromForward(Vector3 forward, out float yaw, out float pitch)
    {
        var f = Vector3.Normalize(forward);
        pitch = MathF.Asin(Math.Clamp(f.Y, -1f, 1f));
        yaw = MathF.Atan2(f.X, -f.Z);
    }

    /// <summary>
    /// Recover the world-space camera basis and position from a .NET row-vector view matrix
    /// (<c>camPos = Vector3.Transform(world, view)</c>, right-handed, eye looking down -Z).
    ///
    /// The splat renderer needs this for covariance projection, and it is the only way to get it
    /// on the XR paths, which are handed a view matrix and never a <see cref="CameraParams"/>.
    /// <paramref name="forward"/> comes back as the direction the camera LOOKS (so it is the
    /// negated third column), matching <see cref="GetOpenCvAxes"/>.
    /// Gate: <c>SplatCovarianceTests.ViewMatrixToCameraBasis_*</c>.
    /// </summary>
    public static void ViewMatrixToCameraBasis(
        Matrix4x4 view, out Vector3 right, out Vector3 up, out Vector3 forward, out Vector3 position)
    {
        // Columns of the upper-left 3x3 are the world-space camera axes.
        right = new Vector3(view.M11, view.M21, view.M31);
        up = new Vector3(view.M12, view.M22, view.M32);
        forward = -new Vector3(view.M13, view.M23, view.M33);

        // world = (camPos - t) * R^T, and camPos = 0 at the eye, so eye = -t * R^T.
        float tx = view.M41, ty = view.M42, tz = view.M43;
        position = new Vector3(
            -(tx * view.M11 + ty * view.M12 + tz * view.M13),
            -(tx * view.M21 + ty * view.M22 + tz * view.M23),
            -(tx * view.M31 + ty * view.M32 + tz * view.M33));
    }
}
