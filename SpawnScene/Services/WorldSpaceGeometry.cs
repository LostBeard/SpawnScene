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
    /// Why a splat was dropped by the consistency screen, so a caller can tell "the reference
    /// disagreed" from "the reference could not see it". The two want opposite treatment and the
    /// screen used to give them the same one.
    /// </summary>
    public enum FuseOutcome
    {
        Kept,
        BehindReference,
        OutsideReferenceView,
        ReferenceHasNoDepth,
        DepthsDisagree,
    }

    /// <summary>
    /// CPU-oracle twin of the GPU consistency fuse.
    ///
    /// <paramref name="keepOutsideView"/> decides what happens to a splat the reference camera
    /// cannot see, and it is a POLICY, not a detail:
    ///
    /// - Dropping it is right for an object on a turntable. Every view sees the temple, so
    ///   out-of-frustum means the far side, which a camera that cannot see it has no business
    ///   asserting - it ghosts under relative monocular depth. That is why this was written.
    /// - Dropping it is wrong for a ROOM, which is what this project is for. Views point at
    ///   different walls, so out-of-frustum is most of the scene, and it is exactly the new
    ///   coverage that makes a capture a room rather than an object. MEASURED on Bathroom with
    ///   34 views posed: the screen kept 3% of the non-reference splats and the reconstruction
    ///   was, in effect, the ten views that skip the screen.
    ///
    /// A splat the reference cannot see is UNVERIFIED, not WRONG. Depth disagreement is a
    /// separate question and is still rejected either way.
    /// Gate: <c>TempleRingWorldSpaceTests.ConsistencyFuse_*</c>.
    /// </summary>
    public static FuseOutcome ClassifySplatVsRef(
        float zCam, float refDepthRaw, float depthScale, float relThresh,
        bool inBounds, bool keepOutsideView)
    {
        if (zCam <= 1e-6f) return FuseOutcome.BehindReference;
        if (!inBounds)
            return keepOutsideView ? FuseOutcome.Kept : FuseOutcome.OutsideReferenceView;

        float refZ = refDepthRaw * depthScale;
        if (!(refZ > 1e-4f)) return FuseOutcome.ReferenceHasNoDepth;

        float denom = MathF.Max(refZ, zCam);
        float rel = MathF.Abs(zCam - refZ) / denom;
        return rel <= relThresh ? FuseOutcome.Kept : FuseOutcome.DepthsDisagree;
    }

    /// <summary>
    /// Keep/drop decision. Preserved for callers that only want the boolean; the default keeps
    /// the original object-centric policy so nothing changes without asking.
    /// </summary>
    public static bool ShouldKeepSplatVsRef(
        float zCam, float refDepthRaw, float splatConf, float refConf,
        float depthScale, float relThresh, bool inBounds, bool hasConf,
        bool keepOutsideView = false)
        => ClassifySplatVsRef(zCam, refDepthRaw, depthScale, relThresh, inBounds, keepOutsideView)
           == FuseOutcome.Kept;

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

        if (!TryDominantEigenvector(N, out float qw, out float qx, out float qy, out float qz))
            return false;

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

    /// <summary>
    /// Which way is up in a reconstruction that has no gravity in it.
    ///
    /// DAv3 and SfM both recover geometry up to an ARBITRARY rotation - their world +Y is
    /// whatever the solver happened to land on, and for a handheld capture there is nothing to
    /// tie it to the room. Everything downstream assumes +Y is up: the viewer's camera
    /// controller rebuilds its up vector as <c>Vector3.UnitY</c> on every frame, the yaw/pitch
    /// model is defined about world +Y, and VR needs a real horizon. So an unaligned
    /// reconstruction renders correctly only from the exact pose it was seated at, and tips over
    /// the moment anyone moves - MEASURED on Bathroom, where the room appeared rotated about 90
    /// degrees with the floor up the side of the screen.
    ///
    /// The estimate is the mean of the cameras' own up vectors. A person walking through a room
    /// holds the phone roughly upright, so across enough frames the average points at gravity;
    /// it needs no calibration file, no EXIF and no floor detection. It fails honestly when the
    /// cameras disagree - a capture that rolled all the way round has no consistent up, and
    /// <paramref name="confidence"/> (the mean vector's length before normalising, 1 for
    /// perfect agreement) says so rather than returning a confident average of nothing.
    /// </summary>
    public static bool TryEstimateSceneUp(
        IEnumerable<CameraParams> cameras, out Vector3 up, out float confidence)
    {
        up = Vector3.UnitY;
        confidence = 0f;

        var sum = Vector3.Zero;
        int n = 0;
        foreach (var cam in cameras)
        {
            var u = cam.Up;
            if (!(u.LengthSquared() > 1e-12f)) continue;
            sum += Vector3.Normalize(u);
            n++;
        }
        if (n == 0) return false;

        var mean = sum / n;
        confidence = mean.Length();
        if (!(confidence > 1e-3f)) return false;   // the ups cancel: no consistent up exists

        up = Vector3.Normalize(mean);
        return true;
    }

    /// <summary>
    /// Shortest-arc rotation taking <paramref name="from"/> onto world +Y, packed for the
    /// row-vector convention so <c>Vector3.Transform(v, M)</c> applies it - the same packing
    /// <see cref="TryUmeyamaSimilarity"/> produces, so the two compose.
    /// </summary>
    public static Matrix4x4 RotationBringingUpToY(Vector3 from)
    {
        var a = Vector3.Normalize(from);
        var b = Vector3.UnitY;

        float dot = Math.Clamp(Vector3.Dot(a, b), -1f, 1f);
        if (dot > 0.999999f) return Matrix4x4.Identity;

        if (dot < -0.999999f)
        {
            // Exactly upside down: the shortest arc is undefined, so any perpendicular axis will
            // do. Picking one deterministically matters - a run that reproduces is worth more
            // than a marginally prettier choice of axis.
            var axis180 = Vector3.Cross(a, Vector3.UnitX);
            if (axis180.LengthSquared() < 1e-8f) axis180 = Vector3.Cross(a, Vector3.UnitZ);
            return Matrix4x4.CreateFromQuaternion(
                Quaternion.CreateFromAxisAngle(Vector3.Normalize(axis180), MathF.PI));
        }

        var axis = Vector3.Normalize(Vector3.Cross(a, b));
        return Matrix4x4.CreateFromQuaternion(
            Quaternion.CreateFromAxisAngle(axis, MathF.Acos(dot)));
    }

    public static Vector3 ApplySimilarity(Vector3 p, float scale, Matrix4x4 rotation, Vector3 translation)
        => scale * Vector3.Transform(p, rotation) + translation;

    /// <summary>
    /// Eigenvector of the LARGEST eigenvalue of the symmetric 4x4 <paramref name="n16"/> (row
    /// major), which for Horn's N matrix is the optimal rotation as a quaternion.
    ///
    /// This is a cyclic Jacobi eigendecomposition rather than a power iteration, and it got here
    /// by two MEASURED failures of the iteration it replaces:
    ///
    /// 1. Power iteration converges to the largest eigenvalue by MAGNITUDE, and Horn's N is
    ///    traceless - its four eigenvalues sum to zero, so a negative one always exists.
    ///    Whenever |lambda_min| &gt; lambda_max it converged to the WORST rotation and returned
    ///    it as success: at 3 anchors with a general rotation, scale 1.94 against a true 2.5 and
    ///    a residual of 0.43, reported true. A caller that trusted the bool placed geometry by it.
    /// 2. Shifting the spectrum to fix that is a trap. N + cI with a Gershgorin c is positive
    ///    semidefinite, so the right eigenvector wins - but the convergence RATIO is
    ///    (lambda_1 + c)/(lambda_2 + c), which the shift drives toward 1. The fix for
    ///    correctness destroyed the convergence rate, and a 200-rotation sweep found the
    ///    survivors at residual 0.056.
    ///
    /// Jacobi needs no shift, converges quadratically, and a 4x4 is small enough that the whole
    /// decomposition is cheaper than the iteration was. Eigenvalues come out on the diagonal and
    /// eigenvectors as the columns of the accumulated rotation.
    ///
    /// Gate: <c>TempleRingWorldSpaceTests.UmeyamaSimilarity_IsExactAcrossRandomRotations</c> -
    /// one rotation is a sample, and the original bug was invisible to the sample in the test.
    /// </summary>
    private static bool TryDominantEigenvector(
        float[] n16, out float qw, out float qx, out float qy, out float qz)
    {
        qw = 1; qx = 0; qy = 0; qz = 0;

        // Work in double: the eigenvector feeds a rotation matrix that geometry is placed by.
        var a = new double[4, 4];
        double magnitude = 0;
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                a[r, c] = n16[r * 4 + c];
                magnitude += Math.Abs(a[r, c]);
            }
        if (magnitude < 1e-20) return false;      // N is all zeros: no rotation is determined.

        var v = new double[4, 4];
        for (int i = 0; i < 4; i++) v[i, i] = 1.0;

        for (int sweep = 0; sweep < 64; sweep++)
        {
            double off = 0;
            for (int p = 0; p < 4; p++)
                for (int q = p + 1; q < 4; q++) off += a[p, q] * a[p, q];
            if (off < 1e-28) break;

            for (int p = 0; p < 3; p++)
                for (int q = p + 1; q < 4; q++)
                {
                    double apq = a[p, q];
                    if (Math.Abs(apq) < 1e-300) continue;

                    // Rotation that zeroes a[p,q]; the numerically stable branch for t.
                    double theta = (a[q, q] - a[p, p]) / (2.0 * apq);
                    double t = theta >= 0
                        ? 1.0 / (theta + Math.Sqrt(theta * theta + 1.0))
                        : -1.0 / (-theta + Math.Sqrt(theta * theta + 1.0));
                    double cs = 1.0 / Math.Sqrt(t * t + 1.0);
                    double sn = t * cs;

                    for (int k = 0; k < 4; k++)
                    {
                        double akp = a[k, p], akq = a[k, q];
                        a[k, p] = cs * akp - sn * akq;
                        a[k, q] = sn * akp + cs * akq;
                    }
                    for (int k = 0; k < 4; k++)
                    {
                        double apk = a[p, k], aqk = a[q, k];
                        a[p, k] = cs * apk - sn * aqk;
                        a[q, k] = sn * apk + cs * aqk;
                    }
                    for (int k = 0; k < 4; k++)
                    {
                        double vkp = v[k, p], vkq = v[k, q];
                        v[k, p] = cs * vkp - sn * vkq;
                        v[k, q] = sn * vkp + cs * vkq;
                    }
                }
        }

        int best = 0;
        for (int i = 1; i < 4; i++) if (a[i, i] > a[best, best]) best = i;

        double q0 = v[0, best], q1 = v[1, best], q2 = v[2, best], q3 = v[3, best];
        double len = Math.Sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        if (!(len > 1e-12)) return false;

        qw = (float)(q0 / len); qx = (float)(q1 / len);
        qy = (float)(q2 / len); qz = (float)(q3 / len);
        return true;
    }

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
