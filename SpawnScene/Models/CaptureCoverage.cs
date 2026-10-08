using System.Numerics;

namespace SpawnScene.Models;

/// <summary>
/// Which horizontal directions a capture's photos face (PLANS 4, capture feedback). A room walked around with the camera
/// pointed at three walls has a fourth wall the scene never saw - from the middle of the room it is a hole, and the fix
/// is a few more photos facing it. The scene is turned upright after SfM (world up = +Y), so each photo's forward
/// direction, flattened to the horizontal plane, falls in one of 8 sectors of 45 degrees measured clockwise (seen from
/// above) from the FIRST photo's heading - the one direction a user can name ("behind my first photo"). Studio.Wander
/// logs the same per heading in the harness; this puts it on the scene card.
/// </summary>
public static class CaptureCoverage
{
    static readonly string[] Names = { "ahead", "ahead-right", "right", "back-right", "back", "back-left", "left", "ahead-left" };

    /// <summary>Photos per sector (8), or null when fewer than 2 cameras or the first looks straight up / down.</summary>
    public static int[]? FacingCounts(IReadOnlyList<CameraParams> cams)
    {
        if (cams.Count < 2) return null;
        var up = Vector3.UnitY;
        static Vector3 Flat(Vector3 f, Vector3 up)
        {
            var v = Vector3.Normalize(f);
            v -= up * Vector3.Dot(v, up);
            return v.LengthSquared() < 1e-4f ? Vector3.Zero : Vector3.Normalize(v);
        }
        var f0 = Flat(cams[0].Forward, up);
        if (f0 == Vector3.Zero) return null;
        var right0 = Vector3.Cross(f0, up);   // clockwise from above when +Y is up
        var counts = new int[8];
        foreach (var c in cams)
        {
            var f = Flat(c.Forward, up);
            if (f == Vector3.Zero) continue;   // straight up or down: no heading
            float angle = MathF.Atan2(Vector3.Dot(f, right0), Vector3.Dot(f, f0));   // 0 = ahead, +90 = right
            int sector = (int)MathF.Floor(((angle * 180f / MathF.PI) + 360f + 22.5f) % 360f / 45f) % 8;
            counts[sector]++;
        }
        return counts;
    }

    /// <summary>
    /// The card's line: the directions no photo faces, relative to the first photo, or that every direction is covered.
    /// Null when coverage says nothing useful (a capture of one object from one side has 1-3 sectors and that is its
    /// intent only if all photos face roughly one way - then the gaps are not news).
    /// </summary>
    public static string? Describe(int[] counts)
    {
        if (counts.Length != 8 || counts.Sum() < 2) return null;
        var empty = Enumerable.Range(0, 8).Where(k => counts[k] == 0).ToList();
        if (empty.Count == 0) return "Photos face every direction";
        // Everything within 90 degrees of one heading: a front-facing capture, not a 360 one - say nothing.
        if (empty.Count >= 5) return null;
        return "No photos face " + string.Join(", ", empty.Select(k => Names[k])) + " of the first photo";
    }
}
