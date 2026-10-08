using System.Globalization;
using System.Numerics;
using System.Text;
using SpawnScene.Models;

namespace SpawnScene.Pages;

public partial class Studio
{
    /// <summary>
    /// &amp;exportcolmap=1 (harness): after Generate, SpawnScene's own structure from motion as COLMAP text (cameras.txt,
    /// images.txt, points3D.txt) on window.__colmapExport, announced by "[Dataset] COLMAP-EXPORT n"; the dataset harness
    /// writes the three files. A reference trainer (gsplat) can then run on exactly our cameras: the parity matrix's
    /// "trainer vs capture" test (TJ 2026-10-08, Hamamni Baths).
    /// </summary>
    public static bool ExportColmapOption { get; set; }

    /// <summary>&amp;dumpheld=1 (harness): the trainer's render of every held-out view after Generate (sample autotest).</summary>
    public static bool DumpHeldOption { get; set; }

    async Task ExportColmapAsync()
    {
        if (_multiViewService.LastSfm is not { } sfm) { Console.WriteLine("[Dataset] COLMAP-EXPORT none (no SfM result)"); return; }
        var ci = CultureInfo.InvariantCulture;
        string F(float v) => v.ToString("R", ci);
        var cams = new StringBuilder("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] - SpawnScene SfM, intrinsics at the photo's own size\n");
        var imgs = new StringBuilder("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world to camera; x right, y down, z forward)\n");
        int placed = 0;
        for (int i = 0; i < sfm.Cameras.Length; i++)
        {
            var c = sfm.Cameras[i];
            if (c == null) continue;
            placed++;
            // The cameras are at the decoded size; the photo on disk is the source size.
            var (sw, sh) = sfm.SourceSizes[i];
            float sx = sw / (float)Math.Max(1, c.Width), sy = sh / (float)Math.Max(1, c.Height);
            cams.Append($"{i + 1} PINHOLE {sw} {sh} {F(c.FocalX * sx)} {F(c.FocalY * sy)} {F(c.CenterX * sx)} {F(c.CenterY * sy)}\n");
            var fwd = Vector3.Normalize(c.Forward);
            var right = c.Right;
            var up = Vector3.Normalize(Vector3.Cross(right, fwd));
            // Rows of R (world -> camera): right, -up (y down), forward.
            var r0 = right; var r1 = -up; var r2 = fwd;
            var q = QuatFromRows(r0, r1, r2);
            var t = new Vector3(-Vector3.Dot(r0, c.Position), -Vector3.Dot(r1, c.Position), -Vector3.Dot(r2, c.Position));
            imgs.Append($"{i + 1} {F(q.W)} {F(q.X)} {F(q.Y)} {F(q.Z)} {F(t.X)} {F(t.Y)} {F(t.Z)} {i + 1} {sfm.Names[i]}\n\n");
        }
        var pts = new StringBuilder("# POINT3D_ID X Y Z R G B ERROR TRACK[] (tracks not exported)\n");
        for (int k = 0; k < sfm.Cloud.Count; k++)
        {
            var p = sfm.Cloud.Positions[k];
            var col = k < sfm.Cloud.Colors.Length ? sfm.Cloud.Colors[k] : new Vector3(0.5f);
            int R(float v) => Math.Clamp((int)MathF.Round(v * 255f), 0, 255);
            pts.Append($"{k + 1} {F(p.X)} {F(p.Y)} {F(p.Z)} {R(col.X)} {R(col.Y)} {R(col.Z)} 0\n");
        }
        _js.Set("__colmapExport", new Dictionary<string, string>
        {
            ["cameras.txt"] = cams.ToString(), ["images.txt"] = imgs.ToString(), ["points3D.txt"] = pts.ToString(),
        });
        Console.WriteLine($"[Dataset] COLMAP-EXPORT {placed} cameras, {sfm.Cloud.Count} points");
        await Task.Delay(1500);   // the harness polls every 500 ms
    }

    /// <summary>The unit quaternion of the rotation whose ROWS are r0, r1, r2.</summary>
    static Quaternion QuatFromRows(Vector3 r0, Vector3 r1, Vector3 r2)
    {
        float m00 = r0.X, m01 = r0.Y, m02 = r0.Z, m10 = r1.X, m11 = r1.Y, m12 = r1.Z, m20 = r2.X, m21 = r2.Y, m22 = r2.Z;
        float tr = m00 + m11 + m22, qw, qx, qy, qz;
        if (tr > 0f)
        {
            float s = MathF.Sqrt(tr + 1f) * 2f;
            qw = 0.25f * s; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s;
        }
        else if (m00 > m11 && m00 > m22)
        {
            float s = MathF.Sqrt(1f + m00 - m11 - m22) * 2f;
            qw = (m21 - m12) / s; qx = 0.25f * s; qy = (m01 + m10) / s; qz = (m02 + m20) / s;
        }
        else if (m11 > m22)
        {
            float s = MathF.Sqrt(1f + m11 - m00 - m22) * 2f;
            qw = (m02 - m20) / s; qx = (m01 + m10) / s; qy = 0.25f * s; qz = (m12 + m21) / s;
        }
        else
        {
            float s = MathF.Sqrt(1f + m22 - m00 - m11) * 2f;
            qw = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = 0.25f * s;
        }
        return Quaternion.Normalize(new Quaternion(qx, qy, qz, qw));
    }
}
