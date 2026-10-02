using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Points triangulated from many cheap matches with the cameras HELD FIXED (COLMAP's point_triangulator, in spirit):
/// the learned front end's 1,024 keypoints per image are what makes the poses robust, but they leave a thin cloud to
/// start the splats from. MEASURED 2026-10-01, TruckFull: learned-front-end BA cloud 8,405 points vs 41,734 from
/// FAST/BRIEF, held-out PSNR 18.20 vs 19.78 dB with poses as good or better (BA 0.08% vs 0.07%). With known poses a
/// wrong match is cheap to reject - it fails the epipolar check against the final poses, then the reprojection check.
/// </summary>
public static class DenseTriangulation
{
    public sealed record Result(List<Vector3> Points, List<int> Colors, List<List<(int Camera, int Feature)>> Tracks,
        int Matches, int EpipolarConsistent, int TracksBuilt, string Summary);

    /// <summary>
    /// <paramref name="pairMatches"/>: per camera pair, matches between <paramref name="features"/>(camera) lists (pixel
    /// coordinates of the cameras' images). A match is kept when its Sampson distance under the pair's fundamental matrix
    /// from the FIXED cameras is within <paramref name="sampsonPx"/>; tracks are chained from the kept matches (a track with
    /// two features in one camera is dropped); each track is triangulated, observations reprojecting further than
    /// <paramref name="reprojPx"/> are removed (re-triangulated while 2+ remain), and the point is kept when it is in front
    /// of every remaining camera and its widest ray pair spans at least <paramref name="minAngleDeg"/>.
    /// </summary>
    public static Result Triangulate(IReadOnlyList<CameraParams> cams, Func<int, IReadOnlyList<ImageFeature>> features,
        IEnumerable<(int A, int B, IReadOnlyList<FeatureMatch> Matches)> pairMatches,
        double sampsonPx = 2.0, double reprojPx = 2.0, double minAngleDeg = 1.5)
    {
        int total = 0, consistent = 0;
        var kept = new List<(int, int, int, int)>();
        foreach (var (a, b, matches) in pairMatches)
        {
            if (matches.Count == 0) continue;
            var f = FundamentalFromCameras(cams[a], cams[b]);
            var fa = features(a); var fb = features(b);
            foreach (var m in matches)
            {
                total++;
                var pa = fa[m.IndexA]; var pb = fb[m.IndexB];
                if (SampsonPx(f, pa.X, pa.Y, pb.X, pb.Y) > sampsonPx) continue;
                consistent++;
                kept.Add((a, m.IndexA, b, m.IndexB));
            }
        }
        var tracks = BundleAdjuster.BuildTracks(kept);
        var points = new List<Vector3>(); var colors = new List<int>(); var outTracks = new List<List<(int, int)>>();
        double cosMin = Math.Cos(minAngleDeg * Math.PI / 180);
        var obs = new List<(int Camera, float U, float V)>();
        var keys = new List<(int Camera, int Feature)>();
        foreach (var track in tracks)
        {
            if (track.Select(t => t.Image).Distinct().Count() != track.Count) continue;   // two features in one camera
            obs.Clear(); keys.Clear();
            foreach (var (cam, feat) in track) { var p = features(cam)[feat]; obs.Add((cam, p.X, p.Y)); keys.Add((cam, feat)); }
            Vector3 x = default;
            bool ok = false;
            while (obs.Count >= 2)
            {
                if (!BundleAdjuster.Triangulate(cams, obs, out x)) break;
                int worst = -1; double worstErr = reprojPx;
                for (int i = 0; i < obs.Count; i++)
                {
                    if (!WorldSpaceGeometry.Project(cams[obs[i].Camera], x, out var u, out var v, out var zc) || zc <= 0) { worst = i; worstErr = double.MaxValue; break; }
                    double e = Math.Sqrt((u - obs[i].U) * (u - obs[i].U) + (v - obs[i].V) * (v - obs[i].V));
                    if (e > worstErr) { worstErr = e; worst = i; }
                }
                if (worst < 0) { ok = true; break; }
                obs.RemoveAt(worst); keys.RemoveAt(worst);
            }
            if (!ok) continue;
            // Parallax: the widest angle between two rays to the point.
            double bestCos = 1;
            for (int i = 0; i < obs.Count; i++)
                for (int j = i + 1; j < obs.Count; j++)
                {
                    var ri = Vector3.Normalize(x - cams[obs[i].Camera].Position);
                    var rj = Vector3.Normalize(x - cams[obs[j].Camera].Position);
                    bestCos = Math.Min(bestCos, Vector3.Dot(ri, rj));
                }
            if (bestCos > cosMin) continue;
            points.Add(x);
            colors.Add(features(keys[0].Camera)[keys[0].Feature].PackedColor);
            outTracks.Add(new List<(int, int)>(keys));
        }
        return new Result(points, colors, outTracks, total, consistent, tracks.Count,
            $"{total} dense matches, {consistent} consistent with the poses (<= {sampsonPx} px), {tracks.Count} tracks -> " +
            $"{points.Count} points (<= {reprojPx} px in every view, >= {minAngleDeg} deg parallax)");
    }

    /// <summary>F with x_b^T F x_a = 0 in PIXELS for two calibrated cameras (row-major).</summary>
    public static double[] FundamentalFromCameras(CameraParams a, CameraParams b)
    {
        var ra = GlobalSfmInit.RotationOf(a); var rb = GlobalSfmInit.RotationOf(b);
        var r = GlobalSfmInit.Mul(rb, GlobalSfmInit.Transpose(ra));
        var d = a.Position - b.Position;
        var t = new double[] { rb[0] * d.X + rb[1] * d.Y + rb[2] * d.Z, rb[3] * d.X + rb[4] * d.Y + rb[5] * d.Z, rb[6] * d.X + rb[7] * d.Y + rb[8] * d.Z };
        var tx = new[] { 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0 };
        var e = GlobalSfmInit.Mul(tx, r);
        var kaInv = new[] { 1 / (double)a.FocalX, 0, -a.CenterX / (double)a.FocalX, 0, 1 / (double)a.FocalY, -a.CenterY / (double)a.FocalY, 0, 0, 1 };
        var kbInv = new[] { 1 / (double)b.FocalX, 0, -b.CenterX / (double)b.FocalX, 0, 1 / (double)b.FocalY, -b.CenterY / (double)b.FocalY, 0, 0, 1 };
        return GlobalSfmInit.Mul(GlobalSfmInit.Mul(GlobalSfmInit.Transpose(kbInv), e), kaInv);
    }

    static double SampsonPx(double[] f, double xa, double ya, double xb, double yb)
    {
        double fx0 = f[0] * xa + f[1] * ya + f[2], fx1 = f[3] * xa + f[4] * ya + f[5], fx2 = f[6] * xa + f[7] * ya + f[8];
        double ftx0 = f[0] * xb + f[3] * yb + f[6], ftx1 = f[1] * xb + f[4] * yb + f[7];
        double num = xb * fx0 + yb * fx1 + fx2;
        double den = fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1;
        return den > 1e-300 ? Math.Abs(num) / Math.Sqrt(den) : double.MaxValue;
    }
}
