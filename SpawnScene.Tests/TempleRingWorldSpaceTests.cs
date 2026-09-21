using System.Numerics;
using NUnit.Framework;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// CPU geometry gate for TempleRing / Middlebury world-space unproject.
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter TempleRing</c>
/// No browser, no GPU, no TJ click-testing.
/// </summary>
public class TempleRingWorldSpaceTests
{
    static string ParPath => FindParFile();

    static string FindParFile()
    {
        // TestDirectory is <testproj>/bin/Release/net10.0, so four levels up is the repo root
        // now that the test project lives INSIDE the repo. The deeper candidate is retained for
        // the old out-of-repo layout, and the absolute path as a last resort.
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "SpawnScene", "wwwroot", "datasets", "TempleRing", "templeR_par.txt")),
            Path.GetFullPath(Path.Combine(TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "SpawnScene", "SpawnScene", "wwwroot", "datasets", "TempleRing", "templeR_par.txt")),
            @"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\TempleRing\templeR_par.txt",
        };
        foreach (var c in candidates)
            if (File.Exists(c)) return c;
        Assert.Fail("templeR_par.txt not found. Looked:\n" + string.Join("\n", candidates));
        return "";
    }

    [Test]
    public void RawMiddlebury_ProjectUnproject_RoundTripsKnownPoint()
    {
        var par = File.ReadAllText(ParPath);
        var lines = par.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        // Use 4 spaced cameras: lines 1,13,25,37 (1-based file lines after count)
        int[] lineIdx = { 1, 13, 25, 37 };
        // TempleRing expected scene center (from PLANS / SfM notes)
        var P = new Vector3(0.028f, 0.042f, -0.054f);

        foreach (int li in lineIdx)
        {
            if (li >= lines.Length) Assert.Fail($"par file too short for line {li}");
            var parts = lines[li].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            float fx = float.Parse(parts[1]), fy = float.Parse(parts[5]);
            float cx = float.Parse(parts[3]), cy = float.Parse(parts[6]);
            var R = new float[3, 3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R[r, c] = float.Parse(parts[10 + r * 3 + c]);
            var t = new[] { float.Parse(parts[19]), float.Parse(parts[20]), float.Parse(parts[21]) };

            // Project with OpenCV: Xc = R*P + t
            float xc = R[0, 0] * P.X + R[0, 1] * P.Y + R[0, 2] * P.Z + t[0];
            float yc = R[1, 0] * P.X + R[1, 1] * P.Y + R[1, 2] * P.Z + t[1];
            float zc = R[2, 0] * P.X + R[2, 1] * P.Y + R[2, 2] * P.Z + t[2];
            Assert.That(zc, Is.GreaterThan(0.01f), $"{parts[0]}: point behind camera z={zc}");

            float u = fx * xc / zc + cx;
            float v = fy * yc / zc + cy;
            var P2 = WorldSpaceGeometry.UnprojectRaw(R, t, fx, fy, cx, cy, u, v, zc);
            float err = Vector3.Distance(P, P2);
            Assert.That(err, Is.LessThan(1e-4f),
                $"{parts[0]}: raw round-trip err={err:E} P={P} P2={P2}");
        }
    }

    [Test]
    public void CameraParams_ProjectUnproject_MatchesRawOracle()
    {
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        Assert.That(cams.Count, Is.GreaterThanOrEqualTo(4));

        var P = new Vector3(0.028f, 0.042f, -0.054f);
        int[] idx = { 0, 12, 24, 36 };
        foreach (int i in idx)
        {
            var (name, cam) = cams[i];
            Assert.That(WorldSpaceGeometry.Project(cam, P, out float u, out float v, out float z), Is.True,
                $"{name}: project failed");
            var P2 = WorldSpaceGeometry.UnprojectPixel(cam, u, v, z);
            float err = Vector3.Distance(P, P2);
            TestContext.Out.WriteLine($"{name}: round-trip err={err:E3} uv=({u:F1},{v:F1}) z={z:F4}");
            Assert.That(err, Is.LessThan(1e-3f),
                $"{name}: CameraParams round-trip err={err:E} (axes/unproject mismatch vs Middlebury)");
        }
    }

    [Test]
    public void SpacedViews_PrincipalRayAtTempleCenter_MeetNearTemple()
    {
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        int n = cams.Count;
        var selected = new[] { cams[0], cams[n / 3], cams[2 * n / 3], cams[n - 1] };
        // Known TempleRing scene center (Middlebury bbox mid), NOT the camera-ring centroid.
        var temple = new Vector3(0.028f, 0.042f, -0.054f);

        var hits = new List<Vector3>();
        foreach (var (name, cam) in selected)
        {
            Assert.That(WorldSpaceGeometry.Project(cam, temple, out float u, out float v, out float z), Is.True);
            var hit = WorldSpaceGeometry.UnprojectPixel(cam, u, v, z);
            hits.Add(hit);
            float err = Vector3.Distance(hit, temple);
            TestContext.Out.WriteLine($"{name}: uv=({u:F1},{v:F1}) z={z:F4} err={err:E3}");
            Assert.That(err, Is.LessThan(1e-3f), $"{name}: unproject(project(temple)) drifted");
        }

        float maxPair = 0;
        for (int a = 0; a < hits.Count; a++)
            for (int b = a + 1; b < hits.Count; b++)
                maxPair = MathF.Max(maxPair, Vector3.Distance(hits[a], hits[b]));
        TestContext.Out.WriteLine($"Max pairwise separation={maxPair:E3}");
        Assert.That(maxPair, Is.LessThan(1e-3f), "Views must agree on temple point (one object)");
    }

    [Test]
    public void PickFarthestCameras_MinSeparation_BeatsIndexSpacing()
    {
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        // Simulate 16 on-disk: every 3rd
        var available = Enumerable.Range(0, cams.Count).Where(i => i % 3 == 0).Take(16)
            .Select(i => cams[i].camera).ToList();
        var idxSpaced = new[] { 0, 5, 10, 15 }.Where(i => i < available.Count).ToList();
        var farthest = WorldSpaceGeometry.PickFarthestCameras(available, 4);

        float MinPair(IReadOnlyList<int> idx)
        {
            float m = float.MaxValue;
            for (int a = 0; a < idx.Count; a++)
                for (int b = a + 1; b < idx.Count; b++)
                    m = MathF.Min(m, Vector3.Distance(available[idx[a]].Position, available[idx[b]].Position));
            return m;
        }

        float sepIdx = MinPair(idxSpaced);
        float sepFar = MinPair(farthest);
        TestContext.Out.WriteLine($"index-spaced minPair={sepIdx:F4}; farthest minPair={sepFar:F4} picks=[{string.Join(",", farthest)}]");
        Assert.That(sepFar, Is.GreaterThanOrEqualTo(sepIdx - 1e-4f));
        Assert.That(sepFar, Is.GreaterThan(0.15f), "farthest picks must be well separated on the ring");
    }

    [Test]
    public void LookAtAnchoredScales_FuseAtTemple_UnlikeDav3IdentityJunk()
    {
        // Simulates correct GT path: each view's depth scaled so lookAt unprojects to temple.
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        int n = cams.Count;
        var selected = new[] { cams[0], cams[n / 3], cams[2 * n / 3], cams[n - 1] };
        var temple = new Vector3(0.028f, 0.042f, -0.054f);

        var hits = new List<Vector3>();
        foreach (var (name, cam) in selected)
        {
            Assert.That(WorldSpaceGeometry.Project(cam, temple, out float u, out float v, out float z), Is.True);
            // Fake monocular raw depth = 2.0; scale = z/raw → unproject lands on temple
            float raw = 2.0f;
            float scale = z / raw;
            var hit = WorldSpaceGeometry.UnprojectPixel(cam, u, v, raw * scale);
            hits.Add(hit);
            Assert.That(Vector3.Distance(hit, temple), Is.LessThan(1e-3f), name);
        }
        float sep = 0;
        for (int a = 0; a < 4; a++)
            for (int b = a + 1; b < 4; b++)
                sep = MathF.Max(sep, Vector3.Distance(hits[a], hits[b]));
        Assert.That(sep, Is.LessThan(1e-3f));
    }


    [Test]
    public void SyntheticPlane_FourViews_UnprojectToSameWorldPlane()
    {
        // Perfect depths: a fronto-parallel plane at the GT scene center for each camera.
        // If unproject is correct, samples from all views land on one plane (not 4 sheets).
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        int n = cams.Count;
        var selected = new[] { cams[0], cams[n / 3], cams[2 * n / 3], cams[n - 1] };
        var planePoint = new Vector3(0.028f, 0.042f, -0.054f);

        var worldPts = new List<Vector3>();
        foreach (var (name, cam) in selected)
        {
            // Depth of plane along each of 5 pixels (center + 4 offsets)
            foreach (var (du, dv) in new[] { (0f, 0f), (-50f, 0f), (50f, 0f), (0f, -40f), (0f, 40f) })
            {
                float u = cam.CenterX + du, v = cam.CenterY + dv;
                // Ray direction in camera frame
                var dir = Vector3.Normalize(new Vector3(
                    (u - cam.CenterX) / cam.FocalX,
                    (v - cam.CenterY) / cam.FocalY,
                    1f));
                // Intersect ray with plane through planePoint with normal = average forward... 
                // Simpler: use exact depth from Project of a point on the plane near the ray.
                // Place a point on the plane: for true geometry use Project of planePoint for center only.
                if (!WorldSpaceGeometry.Project(cam, planePoint, out _, out _, out float zPlane))
                    Assert.Fail($"{name}: plane behind camera");
                // For offset pixels, invent depth by intersecting plane: n·(C + d*dir_w - planePoint)=0
                WorldSpaceGeometry.GetOpenCvAxes(cam, out var right, out var down, out var forward);
                var dirW = Vector3.Normalize(right * dir.X + down * dir.Y + forward * dir.Z);
                // Plane normal ≈ toward cameras from plane (use -normalize(centroid-ish)); use forward of cam0
                var normal = Vector3.Normalize(selected[0].Item2.Position - planePoint);
                float denom = Vector3.Dot(normal, dirW);
                if (MathF.Abs(denom) < 1e-5f) continue;
                float t = Vector3.Dot(normal, planePoint - cam.Position) / denom;
                if (t <= 0) continue;
                var hit = cam.Position + dirW * t;
                // Also via UnprojectPixel with camera-Z depth
                float zCam = Vector3.Dot(hit - cam.Position, forward);
                var hit2 = WorldSpaceGeometry.UnprojectPixel(cam, u, v, zCam);
                float err = Vector3.Distance(hit, hit2);
                Assert.That(err, Is.LessThan(1e-3f), $"{name}: plane unproject err={err}");
                worldPts.Add(hit2);
            }
        }

        // All points should be near the plane (plane distance)
        var nrm = Vector3.Normalize(selected[0].Item2.Position - planePoint);
        float maxPlaneDist = worldPts.Max(p => MathF.Abs(Vector3.Dot(nrm, p - planePoint)));
        TestContext.Out.WriteLine($"Synthetic plane: {worldPts.Count} pts, max plane dist={maxPlaneDist:F5}");
        Assert.That(maxPlaneDist, Is.LessThan(0.01f));

        // Bounding box should be compact (one surface), not 4 separated clusters
        float minX = worldPts.Min(p => p.X), maxX = worldPts.Max(p => p.X);
        float minY = worldPts.Min(p => p.Y), maxY = worldPts.Max(p => p.Y);
        float minZ = worldPts.Min(p => p.Z), maxZ = worldPts.Max(p => p.Z);
        float diag = MathF.Sqrt((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY) + (maxZ - minZ) * (maxZ - minZ));
        TestContext.Out.WriteLine($"Synthetic bbox diag={diag:F4}");
        Assert.That(diag, Is.LessThan(0.25f), "Points span too much — views not fusing to one surface");
    }

    [Test]
    public void UmeyamaSimilarity_RecoversKnownSimilarity()
    {
        var src = new[]
        {
            new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0),
            new Vector3(0, 0, 1),
        };
        float trueScale = 2.5f;
        // 90° about Y: (x,y,z) → (z, y, -x)
        var trueR = Matrix4x4.CreateRotationY(MathF.PI / 2);
        var trueT = new Vector3(3, -1, 2);
        var dst = src.Select(p => trueScale * Vector3.Transform(p, trueR) + trueT).ToList();

        Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(src, dst,
            out float s, out var R, out var t, out float rms), Is.True);
        TestContext.Out.WriteLine($"Umeyama s={s:F4} rms={rms:E3} t={t}");
        Assert.That(s, Is.EqualTo(trueScale).Within(1e-3f));
        Assert.That(rms, Is.LessThan(1e-3f));
        Assert.That(Vector3.Distance(t, trueT), Is.LessThan(1e-3f));

        for (int i = 0; i < src.Length; i++)
        {
            var p = WorldSpaceGeometry.ApplySimilarity(src[i], s, R, t);
            Assert.That(Vector3.Distance(p, dst[i]), Is.LessThan(1e-3f));
        }
    }

    /// <summary>
    /// Regression: the fit used to come back TRUE with a wrong transform.
    ///
    /// <see cref="UmeyamaSimilarity_RecoversKnownSimilarity"/> passed throughout, because four
    /// points and a 90-degree turn about Y happen to land on the good side of the bug. Three
    /// non-collinear points determine a similarity exactly, and with a general rotation the old
    /// power iteration converged to the most NEGATIVE eigenvalue of Horn's traceless N - the
    /// worst rotation rather than the best - returning scale 1.94 against a true 2.5 with a
    /// residual of 0.43 and no error. A caller that trusted the bool placed geometry by it.
    /// </summary>
    [Test]
    public void UmeyamaSimilarity_ThreePointsLargeRotation_IsExactNotMerelySuccessful()
    {
        float trueScale = 2.5f;
        var trueR = Matrix4x4.CreateFromYawPitchRoll(0.7f, -0.35f, 1.1f);
        var trueT = new Vector3(-4f, 2f, 9f);

        var src = new[] { new Vector3(0, 0, 0), new Vector3(1, 0.4f, 0), new Vector3(0, 1, 0.6f) };
        var dst = src.Select(p => trueScale * Vector3.Transform(p, trueR) + trueT).ToList();

        Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(
            src, dst, out float s, out var R, out var t, out float rms), Is.True);
        Assert.That(rms, Is.LessThan(1e-4f), "an exactly determined fit must be exact, not close");
        Assert.That(s, Is.EqualTo(trueScale).Within(1e-3f));
        for (int i = 0; i < src.Length; i++)
            Assert.That(Vector3.Distance(WorldSpaceGeometry.ApplySimilarity(src[i], s, R, t), dst[i]),
                Is.LessThan(1e-3f));
    }

    /// <summary>
    /// The same solver over many random rotations. One rotation is a sample, and the old bug was
    /// invisible to the sample that happened to be in the test.
    /// </summary>
    [Test]
    public void UmeyamaSimilarity_IsExactAcrossRandomRotations()
    {
        var rng = new Random(20260921);
        float F() => (float)(rng.NextDouble() * 2.0 - 1.0);

        float worst = 0f;
        for (int trial = 0; trial < 200; trial++)
        {
            float trueScale = 0.2f + (float)rng.NextDouble() * 4f;
            var trueR = Matrix4x4.CreateFromYawPitchRoll(F() * 3.1f, F() * 1.5f, F() * 3.1f);
            var trueT = new Vector3(F() * 10f, F() * 10f, F() * 10f);

            var src = new List<Vector3>();
            for (int i = 0; i < 3 + trial % 4; i++) src.Add(new Vector3(F(), F(), F()));
            var dst = src.Select(p => trueScale * Vector3.Transform(p, trueR) + trueT).ToList();

            Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(
                src, dst, out float s, out _, out _, out float rms), Is.True, $"trial {trial}");
            float spread = src.Select(p => p.Length()).Max();
            Assert.That(rms, Is.LessThan(1e-3f * Math.Max(1f, trueScale * spread)), $"trial {trial}");
            Assert.That(s, Is.EqualTo(trueScale).Within(1e-2f * trueScale), $"trial {trial}");
            worst = MathF.Max(worst, rms);
        }
        TestContext.Out.WriteLine($"worst residual over 200 random similarities: {worst:E3}");
    }

    [Test]
    public void UmeyamaSimilarity_TempleRingCameras_SelfAligns()
    {
        var cams = WorldSpaceGeometry.ParseMiddleburyParams(File.ReadAllText(ParPath), 640, 480);
        var idx = WorldSpaceGeometry.PickFarthestCameras(cams.Select(c => c.camera).ToList(), 4);
        var pts = idx.Select(i => cams[i].camera.Position).ToList();
        // Identity map
        Assert.That(WorldSpaceGeometry.TryUmeyamaSimilarity(pts, pts, out float s, out _, out var t, out float rms), Is.True);
        Assert.That(s, Is.EqualTo(1f).Within(1e-3f));
        Assert.That(rms, Is.LessThan(1e-4f));
        Assert.That(t.Length(), Is.LessThan(1e-3f));
    }

    [Test]
    public void ConsistencyFuse_AgreeingDepth_IsKept()
    {
        const float scale = 2f, thresh = 0.06f;
        float refRaw = 1.0f;
        float zCam = refRaw * scale; // exact agree
        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam, refRaw, splatConf: 0.5f, refConf: 0.5f,
            scale, thresh, inBounds: true, hasConf: true), Is.True);

        // Within threshold (5%)
        zCam = refRaw * scale * 1.04f;
        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam, refRaw, 0.5f, 0.5f, scale, thresh, true, true), Is.True);
    }

    [Test]
    public void ConsistencyFuse_DisagreeingDepth_IsDropped()
    {
        const float scale = 2f, thresh = 0.06f;
        float refRaw = 1.0f;
        float zCam = refRaw * scale * 1.5f; // 50% off → disagree

        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam, refRaw, 0.9f, 0.1f, scale, thresh, true, hasConf: false), Is.False);
        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam, refRaw, splatConf: 0.9f, refConf: 0.2f, scale, thresh, true, hasConf: true), Is.False);
    }

    [Test]
    public void ConsistencyFuse_NovelOutOfBounds_IsDropped()
    {
        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam: 1f, refDepthRaw: 1f, splatConf: 0.9f, refConf: 0.9f,
            depthScale: 1f, relThresh: 0.06f, inBounds: false, hasConf: true), Is.False);

        Assert.That(WorldSpaceGeometry.ShouldKeepSplatVsRef(
            zCam: -0.1f, refDepthRaw: 1f, splatConf: 0.9f, refConf: 0.9f,
            depthScale: 1f, relThresh: 0.06f, inBounds: true, hasConf: true), Is.False);
    }
}
