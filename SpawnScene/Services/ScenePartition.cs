using System.Numerics;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Splits a scene into blocks that each train inside the device's splat budget (VastGaussian, Lin et al. CVPR 2024):
/// a grid on the ground plane with equal numbers of cameras per cell, each block trained on its own cameras plus every
/// camera that sees enough of it, then merged by keeping each block's splats only inside its own cell. One training
/// run holds about 900 bytes a splat on the GPU (GpuMemoryBudget.BytesPerSplat), so under Chrome's 8 GB GPU-process
/// limit a single run tops out at a few million splats; a building or a street needs more than that.
/// <para>
/// Pure host code over camera poses and point positions (no GPU), so the plan is unit tested.
/// </para>
/// </summary>
public static class ScenePartition
{
    /// <summary>One block: the cell it owns, the box it trains in, and the views that supervise it.</summary>
    /// <param name="Index">Row-major index in the grid.</param>
    /// <param name="CoreMin">Cell minimum in plane coordinates (u, v); float.NegativeInfinity on an outer edge.</param>
    /// <param name="CoreMax">Cell maximum in plane coordinates; float.PositiveInfinity on an outer edge.</param>
    /// <param name="TrainMin">Training box minimum in plane coordinates: the core grown by the overlap.</param>
    /// <param name="TrainMax">Training box maximum in plane coordinates.</param>
    /// <param name="Views">Indices of the views that train this block, ascending.</param>
    /// <param name="OwnViews">How many of <paramref name="Views"/> stand inside the core cell.</param>
    /// <param name="Points">Points inside the training box (seed geometry for the block).</param>
    public sealed record Block(int Index, Vector2 CoreMin, Vector2 CoreMax, Vector2 TrainMin, Vector2 TrainMax,
        int[] Views, int OwnViews, int Points)
    {
        /// <summary>True when plane point <paramref name="p"/> is in the cell this block keeps (half-open, so
        /// every point belongs to exactly one block).</summary>
        public bool Owns(Vector2 p) => p.X >= CoreMin.X && p.X < CoreMax.X && p.Y >= CoreMin.Y && p.Y < CoreMax.Y;

        /// <summary>True when <paramref name="p"/> is in the training box.</summary>
        public bool Trains(Vector2 p) => p.X >= TrainMin.X && p.X <= TrainMax.X && p.Y >= TrainMin.Y && p.Y <= TrainMax.Y;
    }

    /// <summary>The plan: the ground plane's frame and the blocks.</summary>
    /// <param name="Origin">A point on the plane (the cameras' mean).</param>
    /// <param name="AxisU">First in-plane axis (unit).</param>
    /// <param name="AxisV">Second in-plane axis (unit).</param>
    /// <param name="Up">The plane normal (unit): the cameras' mean up.</param>
    public sealed record Plan(Vector3 Origin, Vector3 AxisU, Vector3 AxisV, Vector3 Up, Block[] Blocks)
    {
        /// <summary>World point to plane coordinates.</summary>
        public Vector2 ToPlane(Vector3 p)
        {
            var d = p - Origin;
            return new Vector2(Vector3.Dot(d, AxisU), Vector3.Dot(d, AxisV));
        }

        /// <summary>The block that keeps world point <paramref name="p"/>.</summary>
        public int OwnerOf(Vector3 p)
        {
            var q = ToPlane(p);
            foreach (var b in Blocks) if (b.Owns(q)) return b.Index;
            return -1;   // unreachable: the outer cells are unbounded
        }
    }

    /// <summary>Settings for <see cref="Make"/>.</summary>
    /// <param name="Columns">Cells along the plane's long axis.</param>
    /// <param name="Rows">Cells along the other axis.</param>
    /// <param name="Overlap">Training box margin, as a fraction of the cell's size on each side (VastGaussian: 0.2).</param>
    /// <param name="Visibility">A view outside a cell also trains it when the cell's points cover at least this
    /// fraction of its image (VastGaussian's visibility-based selection: 0.25).</param>
    public sealed record Options(int Columns = 2, int Rows = 2, float Overlap = 0.2f, float Visibility = 0.25f);

    /// <summary>
    /// Plan the blocks for <paramref name="cameras"/> over the scene's <paramref name="points"/> (the SfM points or the
    /// seed splats' centres; a subsample is fine).
    /// </summary>
    public static Plan Make(IReadOnlyList<CameraParams> cameras, ReadOnlySpan<Vector3> points, Options o)
    {
        if (cameras.Count == 0) throw new ArgumentException("no cameras to partition", nameof(cameras));
        if (o.Columns < 1 || o.Rows < 1) throw new ArgumentOutOfRangeException(nameof(o), "need at least one column and row");

        // Ground plane: normal = the cameras' mean up (photos are taken upright, mostly), axes = the principal
        // directions of the camera positions in that plane, so a street splits along the street.
        var origin = Vector3.Zero;
        var upSum = Vector3.Zero;
        foreach (var c in cameras) { origin += c.Position; upSum += c.Up; }
        origin /= cameras.Count;
        var up = upSum.LengthSquared() > 1e-12f ? Vector3.Normalize(upSum) : Vector3.UnitY;
        var (axisU, axisV) = PlaneAxes(cameras, origin, up);
        var plan = new Plan(origin, axisU, axisV, up, Array.Empty<Block>());

        var camPlane = new Vector2[cameras.Count];
        for (int i = 0; i < cameras.Count; i++) camPlane[i] = plan.ToPlane(cameras[i].Position);

        // Cells: split the cameras into columns of equal count along u, then each column into rows along v. Boundaries
        // sit halfway between neighbouring cameras; the outermost edges are open, so every point has an owner.
        var order = Enumerable.Range(0, cameras.Count).OrderBy(i => camPlane[i].X).ToArray();
        var colEdges = Edges(order.Select(i => camPlane[i].X).ToArray(), o.Columns);
        var cells = new List<(Vector2 Min, Vector2 Max)>();
        for (int c = 0; c < o.Columns; c++)
        {
            var inCol = order.Where(i => camPlane[i].X >= colEdges[c] && camPlane[i].X < colEdges[c + 1])
                .Select(i => camPlane[i].Y).OrderBy(v => v).ToArray();
            var rowEdges = inCol.Length > 0 ? Edges(inCol, o.Rows) : Enumerable.Range(0, o.Rows + 1)
                .Select(r => r == 0 ? float.NegativeInfinity : r == o.Rows ? float.PositiveInfinity : 0f).ToArray();
            for (int r = 0; r < o.Rows; r++)
                cells.Add((new Vector2(colEdges[c], rowEdges[r]), new Vector2(colEdges[c + 1], rowEdges[r + 1])));
        }

        // The extent of the cameras and points, to give the open outer cells a finite size for the overlap. The points'
        // 2nd-98th percentiles, not their min/max: SfM floaters far out made every margin span the whole scene, so each
        // block's training box held all of it (TruckFull 2x2: 43,404 of 43,405 seed points per block, 2026-10-05).
        var ptPlane = new Vector2[points.Length];
        for (int i = 0; i < points.Length; i++) ptPlane[i] = plan.ToPlane(points[i]);
        var lo = new Vector2(float.PositiveInfinity);
        var hi = new Vector2(float.NegativeInfinity);
        if (points.Length > 0)
        {
            var us = ptPlane.Select(p => p.X).OrderBy(x => x).ToArray();
            var vs = ptPlane.Select(p => p.Y).OrderBy(y => y).ToArray();
            int qLo = (int)(0.02 * (us.Length - 1)), qHi = (int)Math.Ceiling(0.98 * (us.Length - 1));
            lo = new Vector2(us[qLo], vs[qLo]);
            hi = new Vector2(us[qHi], vs[qHi]);
        }
        foreach (var p in camPlane) { lo = Vector2.Min(lo, p); hi = Vector2.Max(hi, p); }

        var blocks = new Block[cells.Count];
        for (int b = 0; b < cells.Count; b++)
        {
            var (cMin, cMax) = cells[b];
            // Finite stand-ins for the open edges, then the margin on that finite size.
            var fMin = new Vector2(float.IsNegativeInfinity(cMin.X) ? lo.X : cMin.X, float.IsNegativeInfinity(cMin.Y) ? lo.Y : cMin.Y);
            var fMax = new Vector2(float.IsPositiveInfinity(cMax.X) ? hi.X : cMax.X, float.IsPositiveInfinity(cMax.Y) ? hi.Y : cMax.Y);
            var margin = Vector2.Max(fMax - fMin, Vector2.Zero) * o.Overlap;
            var tMin = new Vector2(float.IsNegativeInfinity(cMin.X) ? float.NegativeInfinity : cMin.X - margin.X,
                                   float.IsNegativeInfinity(cMin.Y) ? float.NegativeInfinity : cMin.Y - margin.Y);
            var tMax = new Vector2(float.IsPositiveInfinity(cMax.X) ? float.PositiveInfinity : cMax.X + margin.X,
                                   float.IsPositiveInfinity(cMax.Y) ? float.PositiveInfinity : cMax.Y + margin.Y);
            var block = new Block(b, cMin, cMax, tMin, tMax, Array.Empty<int>(), 0, 0);

            // The cell's own points (what a view must see to be useful), and the points in its training box.
            var core = new List<Vector3>();
            int trainPts = 0;
            for (int i = 0; i < points.Length; i++)
            {
                if (block.Owns(ptPlane[i])) core.Add(points[i]);
                if (block.Trains(ptPlane[i])) trainPts++;
            }

            var views = new List<int>();
            int own = 0;
            for (int i = 0; i < cameras.Count; i++)
            {
                if (block.Owns(camPlane[i])) { views.Add(i); own++; continue; }
                if (block.Trains(camPlane[i]) || CoverageOf(cameras[i], core) >= o.Visibility) views.Add(i);
            }
            blocks[b] = block with { Views = views.ToArray(), OwnViews = own, Points = trainPts };
        }
        return plan with { Blocks = blocks };
    }

    /// <summary>
    /// Fraction of <paramref name="cam"/>'s image covered by the bounding rectangle of <paramref name="points"/> that
    /// project in front of it, clipped to the image: the share of the photo spent on this cell. A rectangle, not a
    /// hull: generous on a diagonal cell, and selection should err toward including a view.
    /// </summary>
    public static float CoverageOf(CameraParams cam, IReadOnlyList<Vector3> points)
    {
        if (points.Count == 0 || cam.Width <= 0 || cam.Height <= 0) return 0f;
        var right = cam.Right;
        var up = Vector3.Normalize(Vector3.Cross(right, cam.Forward));
        var fwd = Vector3.Normalize(cam.Forward);
        float x0 = float.PositiveInfinity, y0 = float.PositiveInfinity, x1 = float.NegativeInfinity, y1 = float.NegativeInfinity;
        int seen = 0;
        foreach (var p in points)
        {
            var d = p - cam.Position;
            float z = Vector3.Dot(d, fwd);
            if (z <= cam.Near) continue;
            float u = cam.CenterX + cam.FocalX * Vector3.Dot(d, right) / z;
            float v = cam.CenterY - cam.FocalY * Vector3.Dot(d, up) / z;
            if (u < 0f || u > cam.Width || v < 0f || v > cam.Height) continue;
            seen++;
            x0 = MathF.Min(x0, u); x1 = MathF.Max(x1, u); y0 = MathF.Min(y0, v); y1 = MathF.Max(y1, v);
        }
        if (seen < 2) return 0f;
        return (x1 - x0) * (y1 - y0) / ((float)cam.Width * cam.Height);
    }

    /// <summary>Two in-plane axes: the camera positions' principal direction (projected onto the plane) and its normal in the plane.</summary>
    static (Vector3 U, Vector3 V) PlaneAxes(IReadOnlyList<CameraParams> cameras, Vector3 origin, Vector3 up)
    {
        // Any basis of the plane, then the 2x2 covariance of the positions in it; its major eigenvector is U.
        var a = MathF.Abs(up.X) < 0.9f ? Vector3.UnitX : Vector3.UnitZ;
        var e1 = Vector3.Normalize(a - Vector3.Dot(a, up) * up);
        var e2 = Vector3.Cross(up, e1);
        double sxx = 0, sxy = 0, syy = 0;
        foreach (var c in cameras)
        {
            var d = c.Position - origin;
            double x = Vector3.Dot(d, e1), y = Vector3.Dot(d, e2);
            sxx += x * x; sxy += x * y; syy += y * y;
        }
        double angle = 0.5 * Math.Atan2(2 * sxy, sxx - syy);   // major axis of [[sxx, sxy], [sxy, syy]]
        var u = Vector3.Normalize((float)Math.Cos(angle) * e1 + (float)Math.Sin(angle) * e2);
        var v = Vector3.Cross(up, u);
        return (u, v);
    }

    /// <summary>
    /// <paramref name="parts"/> + 1 edges splitting the sorted values into equal-count groups; inner edges halfway
    /// between the neighbours on either side of a split, outer edges infinite.
    /// </summary>
    static float[] Edges(float[] sorted, int parts)
    {
        var e = new float[parts + 1];
        e[0] = float.NegativeInfinity;
        e[parts] = float.PositiveInfinity;
        for (int k = 1; k < parts; k++)
        {
            int at = (int)Math.Round((double)k * sorted.Length / parts);
            at = Math.Clamp(at, 1, Math.Max(1, sorted.Length - 1));
            e[k] = sorted.Length >= 2 ? 0.5f * (sorted[at - 1] + sorted[at]) : sorted[0];
        }
        // Duplicate positions can leave an edge below its predecessor; keep them ordered so cells never invert.
        for (int k = 1; k < parts; k++) e[k] = MathF.Max(e[k], e[k - 1]);
        return e;
    }
}
