using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Plane-induced homography between two views by RANSAC over normalized 4-point DLT. Used to tell a pair whose inliers
/// lie on one plane: its essential-matrix pose is then ambiguous, and a symmetric planar texture (DrJohnson's patterned
/// rug, 2026-10-03) can be matched at a rotated correspondence that is still geometrically consistent - image 41 was
/// placed ~95 deg wrong with every one of its verified pairs agreeing.
/// </summary>
public static class Homography
{
    /// <summary>
    /// The inliers (transfer error under <paramref name="thresholdPx"/>) of the best homography mapping
    /// <paramref name="a"/> to <paramref name="b"/>, or 0 when fewer than 4 points or no hypothesis is non-degenerate.
    /// </summary>
    public static int InlierCount(IReadOnlyList<Vector2> a, IReadOnlyList<Vector2> b, float thresholdPx = 4f,
        int iterations = 300, int seed = 1)
    {
        int n = Math.Min(a.Count, b.Count);
        if (n < 4) return 0;
        var rng = new Random(seed);
        float t2 = thresholdPx * thresholdPx;
        int best = 0;
        var idx = new int[4];
        for (int it = 0; it < iterations; it++)
        {
            for (int q = 0; q < 4; q++)
            {
                int r;
                do r = rng.Next(n); while (Array.IndexOf(idx, r, 0, q) >= 0);
                idx[q] = r;
            }
            if (!Solve4(a, b, idx, out var h)) continue;
            int inl = 0;
            for (int i = 0; i < n; i++) if (TransferSq(h, a[i], b[i]) <= t2) inl++;
            if (inl > best) best = inl;
        }
        return best;
    }

    /// <summary>Squared distance from H*a to b in pixels (infinite when a maps to infinity).</summary>
    public static float TransferSq(double[] h, Vector2 a, Vector2 b)
    {
        double w = h[6] * a.X + h[7] * a.Y + h[8];
        if (Math.Abs(w) < 1e-12) return float.PositiveInfinity;
        double x = (h[0] * a.X + h[1] * a.Y + h[2]) / w - b.X;
        double y = (h[3] * a.X + h[4] * a.Y + h[5]) / w - b.Y;
        return (float)(x * x + y * y);
    }

    /// <summary>The exact homography through four correspondences (h33 = 1), false when degenerate.</summary>
    public static bool Solve4(IReadOnlyList<Vector2> a, IReadOnlyList<Vector2> b, int[] idx, out double[] h)
    {
        // 8x8 linear system for h11..h32 with h33 = 1.
        var m = new double[8, 9];
        for (int q = 0; q < 4; q++)
        {
            double x = a[idx[q]].X, y = a[idx[q]].Y, u = b[idx[q]].X, v = b[idx[q]].Y;
            int r0 = 2 * q, r1 = r0 + 1;
            m[r0, 0] = x; m[r0, 1] = y; m[r0, 2] = 1; m[r0, 6] = -u * x; m[r0, 7] = -u * y; m[r0, 8] = u;
            m[r1, 3] = x; m[r1, 4] = y; m[r1, 5] = 1; m[r1, 6] = -v * x; m[r1, 7] = -v * y; m[r1, 8] = v;
        }
        h = new double[9];
        // Gaussian elimination with partial pivoting.
        for (int c = 0; c < 8; c++)
        {
            int piv = c;
            for (int r = c + 1; r < 8; r++) if (Math.Abs(m[r, c]) > Math.Abs(m[piv, c])) piv = r;
            if (Math.Abs(m[piv, c]) < 1e-9) return false;
            if (piv != c) for (int k = 0; k < 9; k++) (m[c, k], m[piv, k]) = (m[piv, k], m[c, k]);
            for (int r = 0; r < 8; r++)
            {
                if (r == c) continue;
                double f = m[r, c] / m[c, c];
                if (f == 0) continue;
                for (int k = c; k < 9; k++) m[r, k] -= f * m[c, k];
            }
        }
        for (int c = 0; c < 8; c++) h[c] = m[c, 8] / m[c, c];
        h[8] = 1;
        return h.All(double.IsFinite);
    }
}
