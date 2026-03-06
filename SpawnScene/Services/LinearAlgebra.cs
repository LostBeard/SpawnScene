using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Linear algebra utilities for Structure from Motion computations.
/// Provides SVD, matrix operations, and geometric transforms needed
/// for essential matrix estimation and point triangulation.
/// </summary>
public static class LinearAlgebra
{
    /// <summary>
    /// Solve a homogeneous linear system Ax = 0 using SVD via eigendecomposition of A^T*A.
    /// Returns the right singular vector corresponding to the smallest singular value.
    /// </summary>
    public static double[] SolveHomogeneous(double[,] A)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);

        // Compute A^T * A (n x n symmetric)
        var ATA = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < m; k++)
                    sum += A[k, i] * A[k, j];
                ATA[i, j] = sum;
                ATA[j, i] = sum;
            }
        }

        // Find eigenvector of smallest eigenvalue via Jacobi iteration
        return JacobiSmallestEigenvector(ATA, n);
    }

    /// <summary>
    /// Jacobi eigenvalue iteration on a symmetric NxN matrix.
    /// Returns the eigenvector corresponding to the smallest eigenvalue.
    /// </summary>
    private static double[] JacobiSmallestEigenvector(double[,] S, int n)
    {
        // Work on a copy
        var A = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i, j] = S[i, j];

        // Eigenvectors (start as identity)
        var V = new double[n, n];
        for (int i = 0; i < n; i++) V[i, i] = 1.0;

        // Jacobi iteration — sweep through all off-diagonal pairs
        for (int sweep = 0; sweep < 100; sweep++)
        {
            // Check convergence: largest off-diagonal element
            double offDiag = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    offDiag = Math.Max(offDiag, Math.Abs(A[i, j]));
            if (offDiag < 1e-14) break;

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(A[p, q]) < 1e-15) continue;

                    // Compute rotation angle
                    double tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q]);
                    double t;
                    if (tau >= 0)
                        t = 1.0 / (tau + Math.Sqrt(1.0 + tau * tau));
                    else
                        t = -1.0 / (-tau + Math.Sqrt(1.0 + tau * tau));

                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    // Apply rotation to A (symmetric, so only update needed elements)
                    double app = A[p, p];
                    double aqq = A[q, q];
                    double apq = A[p, q];

                    A[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    A[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    A[p, q] = 0;
                    A[q, p] = 0;

                    for (int r = 0; r < n; r++)
                    {
                        if (r == p || r == q) continue;
                        double arp = A[r, p];
                        double arq = A[r, q];
                        A[r, p] = c * arp - s * arq;
                        A[p, r] = A[r, p];
                        A[r, q] = s * arp + c * arq;
                        A[q, r] = A[r, q];
                    }

                    // Accumulate eigenvectors
                    for (int i = 0; i < n; i++)
                    {
                        double vip = V[i, p];
                        double viq = V[i, q];
                        V[i, p] = c * vip - s * viq;
                        V[i, q] = s * vip + c * viq;
                    }
                }
            }
        }

        // Find index of smallest eigenvalue
        int minIdx = 0;
        double minEig = A[0, 0];
        for (int i = 1; i < n; i++)
        {
            if (A[i, i] < minEig)
            {
                minEig = A[i, i];
                minIdx = i;
            }
        }

        // Return corresponding eigenvector
        var result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = V[i, minIdx];

        return result;
    }

    /// <summary>
    /// Compute SVD of a 3x3 matrix using Jacobi iteration on A^T*A.
    /// Returns U, S (diagonal), Vt such that A = U * diag(S) * Vt.
    /// Handles the degenerate case (σ3≈0) needed for Essential matrix decomposition.
    /// Both U and V are guaranteed to be proper rotations (det = +1).
    /// </summary>
    public static (double[,] U, double[] S, double[,] Vt) SVD3x3(double[,] A)
    {
        // Compute A^T * A
        var ATA = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                    sum += A[k, i] * A[k, j];
                ATA[i, j] = sum;
            }

        // Eigendecompose A^T*A using Jacobi to get V and eigenvalues
        var V = new double[3, 3];
        for (int i = 0; i < 3; i++) V[i, i] = 1.0;
        var D = new double[3, 3];
        Array.Copy(ATA, D, 9);

        // Jacobi sweeps on symmetric 3x3
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double offDiag = Math.Abs(D[0, 1]) + Math.Abs(D[0, 2]) + Math.Abs(D[1, 2]);
            if (offDiag < 1e-14) break;

            for (int p = 0; p < 2; p++)
            {
                for (int q = p + 1; q < 3; q++)
                {
                    if (Math.Abs(D[p, q]) < 1e-15) continue;

                    double tau = (D[q, q] - D[p, p]) / (2.0 * D[p, q]);
                    double t;
                    if (tau >= 0)
                        t = 1.0 / (tau + Math.Sqrt(1.0 + tau * tau));
                    else
                        t = -1.0 / (-tau + Math.Sqrt(1.0 + tau * tau));

                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    double dpp = D[p, p], dqq = D[q, q], dpq = D[p, q];
                    D[p, p] = c * c * dpp - 2 * s * c * dpq + s * s * dqq;
                    D[q, q] = s * s * dpp + 2 * s * c * dpq + c * c * dqq;
                    D[p, q] = 0;
                    D[q, p] = 0;

                    for (int r = 0; r < 3; r++)
                    {
                        if (r == p || r == q) continue;
                        double drp = D[r, p], drq = D[r, q];
                        D[r, p] = c * drp - s * drq;
                        D[p, r] = D[r, p];
                        D[r, q] = s * drp + c * drq;
                        D[q, r] = D[r, q];
                    }

                    for (int i = 0; i < 3; i++)
                    {
                        double vp = V[i, p], vq = V[i, q];
                        V[i, p] = c * vp - s * vq;
                        V[i, q] = s * vp + c * vq;
                    }
                }
            }
        }

        // Eigenvalues are on diagonal of D, singular values = sqrt(eigenvalues)
        var eigenvals = new[] { D[0, 0], D[1, 1], D[2, 2] };
        var singVals = new double[3];
        for (int i = 0; i < 3; i++)
            singVals[i] = Math.Sqrt(Math.Max(0, eigenvals[i]));

        // Sort by descending singular value
        var indices = new[] { 0, 1, 2 };
        Array.Sort(indices, (a, b) => singVals[b].CompareTo(singVals[a]));

        var sortedS = new double[3];
        var sortedV = new double[3, 3];
        for (int j = 0; j < 3; j++)
        {
            sortedS[j] = singVals[indices[j]];
            for (int i = 0; i < 3; i++)
                sortedV[i, j] = V[i, indices[j]];
        }

        // Ensure V has det = +1 BEFORE computing U
        if (Det3x3(sortedV) < 0)
        {
            for (int i = 0; i < 3; i++) sortedV[i, 2] = -sortedV[i, 2];
        }

        // U = A * V * S^{-1} for non-zero singular values
        var U = new double[3, 3];
        int validCols = 0;
        for (int j = 0; j < 3; j++)
        {
            if (sortedS[j] < 1e-6) continue;
            validCols++;

            // Column j of U = (A * v_j) / s_j
            for (int i = 0; i < 3; i++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                    sum += A[i, k] * sortedV[k, j];
                U[i, j] = sum / sortedS[j];
            }
        }

        // Fill degenerate column(s) of U
        if (validCols == 2)
        {
            // Two valid columns (0,1), fill column 2 via cross product: u2 = u0 × u1
            double ux = U[1, 0] * U[2, 1] - U[2, 0] * U[1, 1];
            double uy = U[2, 0] * U[0, 1] - U[0, 0] * U[2, 1];
            double uz = U[0, 0] * U[1, 1] - U[1, 0] * U[0, 1];
            U[0, 2] = ux;
            U[1, 2] = uy;
            U[2, 2] = uz;
        }
        else if (validCols == 1)
        {
            // Only column 0 valid; fill columns 1 and 2
            double ax = Math.Abs(U[0, 0]), ay = Math.Abs(U[1, 0]), az = Math.Abs(U[2, 0]);
            double px, py, pz;
            if (ax <= ay && ax <= az) { px = 1; py = 0; pz = 0; }
            else if (ay <= az) { px = 0; py = 1; pz = 0; }
            else { px = 0; py = 0; pz = 1; }

            double dot = px * U[0, 0] + py * U[1, 0] + pz * U[2, 0];
            double rx = px - dot * U[0, 0], ry = py - dot * U[1, 0], rz = pz - dot * U[2, 0];
            double rn = Math.Sqrt(rx * rx + ry * ry + rz * rz);
            U[0, 1] = rx / rn; U[1, 1] = ry / rn; U[2, 1] = rz / rn;

            U[0, 2] = U[1, 0] * U[2, 1] - U[2, 0] * U[1, 1];
            U[1, 2] = U[2, 0] * U[0, 1] - U[0, 0] * U[2, 1];
            U[2, 2] = U[0, 0] * U[1, 1] - U[1, 0] * U[0, 1];
        }
        else if (validCols == 0)
        {
            // Completely zero matrix, return identity
            U[0, 0] = 1; U[1, 1] = 1; U[2, 2] = 1;
        }

        // Ensure det(U) = +1. If not, flip last column and negate S[2].
        double detU = Det3x3(U);
        if (detU < 0)
        {
            for (int i = 0; i < 3; i++) U[i, 2] = -U[i, 2];
            sortedS[2] = -sortedS[2];
        }

        // V^T
        var Vt = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Vt[i, j] = sortedV[j, i];

        return (U, sortedS, Vt);
    }

    public static double Det3x3(double[,] M)
    {
        return M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
             - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
             + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]);
    }

    /// <summary>
    /// Triangulate a 3D point from two camera views using DLT (Direct Linear Transform).
    /// </summary>
    public static Vector3? TriangulatePoint(
        Vector2 pt1, Vector2 pt2,
        double[,] P1, double[,] P2)
    {
        // Build 4x4 system: each point gives 2 equations
        var A = new double[4, 4];
        A[0, 0] = pt1.X * P1[2, 0] - P1[0, 0];
        A[0, 1] = pt1.X * P1[2, 1] - P1[0, 1];
        A[0, 2] = pt1.X * P1[2, 2] - P1[0, 2];
        A[0, 3] = pt1.X * P1[2, 3] - P1[0, 3];

        A[1, 0] = pt1.Y * P1[2, 0] - P1[1, 0];
        A[1, 1] = pt1.Y * P1[2, 1] - P1[1, 1];
        A[1, 2] = pt1.Y * P1[2, 2] - P1[1, 2];
        A[1, 3] = pt1.Y * P1[2, 3] - P1[1, 3];

        A[2, 0] = pt2.X * P2[2, 0] - P2[0, 0];
        A[2, 1] = pt2.X * P2[2, 1] - P2[0, 1];
        A[2, 2] = pt2.X * P2[2, 2] - P2[0, 2];
        A[2, 3] = pt2.X * P2[2, 3] - P2[0, 3];

        A[3, 0] = pt2.Y * P2[2, 0] - P2[1, 0];
        A[3, 1] = pt2.Y * P2[2, 1] - P2[1, 1];
        A[3, 2] = pt2.Y * P2[2, 2] - P2[1, 2];
        A[3, 3] = pt2.Y * P2[2, 3] - P2[1, 3];

        var X = SolveHomogeneous(A);
        if (Math.Abs(X[3]) < 1e-10) return null;

        return new Vector3(
            (float)(X[0] / X[3]),
            (float)(X[1] / X[3]),
            (float)(X[2] / X[3])
        );
    }

    // ══════════════════════════════════════════════════════════
    //  EPnP (Efficient Perspective-n-Point)
    //  Lepetit, Moreno-Noguer, Fua (IJCV 2009)
    // ══════════════════════════════════════════════════════════

    /// <summary>
    /// Solve PnP using the EPnP algorithm.
    /// Input: normalized 2D points (already (pixel - principal) / focal) and world 3D points.
    /// Returns (R[3,3], t[3]) or null on failure.
    /// </summary>
    public static (double[,] R, double[] t)? SolveEPnP(
        ReadOnlySpan<float> pts2D, ReadOnlySpan<float> pts3D, int n)
    {
        if (n < 4) return null;

        // ═══ STEP 1: Choose 4 control points ═══
        // Control point 0 = centroid of 3D points
        // Control points 1-3 = centroid + principal axes (scaled by sqrt(eigenvalue/n))
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < n; i++)
        {
            cx += pts3D[i * 3]; cy += pts3D[i * 3 + 1]; cz += pts3D[i * 3 + 2];
        }
        cx /= n; cy /= n; cz /= n;

        // Covariance matrix of 3D points
        var cov = new double[3, 3];
        for (int i = 0; i < n; i++)
        {
            double dx = pts3D[i * 3] - cx;
            double dy = pts3D[i * 3 + 1] - cy;
            double dz = pts3D[i * 3 + 2] - cz;
            cov[0, 0] += dx * dx; cov[0, 1] += dx * dy; cov[0, 2] += dx * dz;
            cov[1, 1] += dy * dy; cov[1, 2] += dy * dz;
            cov[2, 2] += dz * dz;
        }
        cov[1, 0] = cov[0, 1]; cov[2, 0] = cov[0, 2]; cov[2, 1] = cov[1, 2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cov[i, j] /= n;

        // Eigendecompose covariance → principal axes
        var (eigVecs, eigVals) = SymEig3x3(cov);

        // Control points in world frame
        var cw = new double[4, 3];
        cw[0, 0] = cx; cw[0, 1] = cy; cw[0, 2] = cz;
        for (int k = 0; k < 3; k++)
        {
            double scale = Math.Sqrt(Math.Max(eigVals[k], 1e-10));
            cw[k + 1, 0] = cx + scale * eigVecs[0, k];
            cw[k + 1, 1] = cy + scale * eigVecs[1, k];
            cw[k + 1, 2] = cz + scale * eigVecs[2, k];
        }

        // ═══ STEP 2: Compute barycentric coordinates ═══
        // Each world point pi = sum(alpha_ij * cw_j), with sum(alpha_ij) = 1
        var alphas = new double[n, 4];
        {
            // Solve: [cw1-cw0, cw2-cw0, cw3-cw0]^T * [a1,a2,a3]^T = (pi-cw0)^T
            // Then a0 = 1 - a1 - a2 - a3
            var C = new double[3, 3];
            for (int j = 0; j < 3; j++)
            {
                C[0, j] = cw[j + 1, 0] - cw[0, 0];
                C[1, j] = cw[j + 1, 1] - cw[0, 1];
                C[2, j] = cw[j + 1, 2] - cw[0, 2];
            }

            // Invert C (3x3)
            double det = C[0, 0] * (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1])
                       - C[0, 1] * (C[1, 0] * C[2, 2] - C[1, 2] * C[2, 0])
                       + C[0, 2] * (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0]);
            if (Math.Abs(det) < 1e-15) return null;

            double invDet = 1.0 / det;
            var Cinv = new double[3, 3];
            Cinv[0, 0] = (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1]) * invDet;
            Cinv[0, 1] = (C[0, 2] * C[2, 1] - C[0, 1] * C[2, 2]) * invDet;
            Cinv[0, 2] = (C[0, 1] * C[1, 2] - C[0, 2] * C[1, 1]) * invDet;
            Cinv[1, 0] = (C[1, 2] * C[2, 0] - C[1, 0] * C[2, 2]) * invDet;
            Cinv[1, 1] = (C[0, 0] * C[2, 2] - C[0, 2] * C[2, 0]) * invDet;
            Cinv[1, 2] = (C[0, 2] * C[1, 0] - C[0, 0] * C[1, 2]) * invDet;
            Cinv[2, 0] = (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0]) * invDet;
            Cinv[2, 1] = (C[0, 1] * C[2, 0] - C[0, 0] * C[2, 1]) * invDet;
            Cinv[2, 2] = (C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]) * invDet;

            for (int i = 0; i < n; i++)
            {
                double dx = pts3D[i * 3] - cw[0, 0];
                double dy = pts3D[i * 3 + 1] - cw[0, 1];
                double dz = pts3D[i * 3 + 2] - cw[0, 2];
                alphas[i, 1] = Cinv[0, 0] * dx + Cinv[0, 1] * dy + Cinv[0, 2] * dz;
                alphas[i, 2] = Cinv[1, 0] * dx + Cinv[1, 1] * dy + Cinv[1, 2] * dz;
                alphas[i, 3] = Cinv[2, 0] * dx + Cinv[2, 1] * dy + Cinv[2, 2] * dz;
                alphas[i, 0] = 1.0 - alphas[i, 1] - alphas[i, 2] - alphas[i, 3];
            }
        }

        // ═══ STEP 3: Build M matrix (2n × 12) ═══
        // Each 2D point gives 2 equations relating the 12 unknowns
        // (x,y,z coords of 4 control points in camera frame)
        var M = new double[2 * n, 12];
        for (int i = 0; i < n; i++)
        {
            double u = pts2D[i * 2];
            double vp = pts2D[i * 2 + 1];

            for (int j = 0; j < 4; j++)
            {
                double a = alphas[i, j];
                // Row 2i: alpha_j * fu * cx_j + alpha_j * u0 * cz_j - alpha_j * u * cz_j
                // Since pts2D are already normalized (u = (px-cx)/fx), fu=1, u0=0:
                // Row 2i:   a * cx_j - u * a * cz_j = 0  →  a*cx_j + 0*cy_j + (-u*a)*cz_j
                M[2 * i, j * 3] = a;          // coefficient of cx_j
                M[2 * i, j * 3 + 1] = 0;          // coefficient of cy_j
                M[2 * i, j * 3 + 2] = -u * a;     // coefficient of cz_j

                // Row 2i+1: a * cy_j - vp * a * cz_j = 0  →  0*cx_j + a*cy_j + (-vp*a)*cz_j
                M[2 * i + 1, j * 3] = 0;
                M[2 * i + 1, j * 3 + 1] = a;
                M[2 * i + 1, j * 3 + 2] = -vp * a;
            }
        }

        // ═══ STEP 4: Compute MᵀM and find its eigenvectors ═══
        var MtM = new double[12, 12];
        for (int i = 0; i < 12; i++)
            for (int j = i; j < 12; j++)
            {
                double sum = 0;
                for (int k = 0; k < 2 * n; k++)
                    sum += M[k, i] * M[k, j];
                MtM[i, j] = sum;
                MtM[j, i] = sum;
            }

        // Get the 4 eigenvectors with smallest eigenvalues
        var (allEigVecs, allEigVals) = JacobiAllEigenvectors(MtM, 12);

        // Sort by eigenvalue ascending
        var sortIdx = Enumerable.Range(0, 12).OrderBy(i => allEigVals[i]).ToArray();

        // The null space vectors (v1..v4 correspond to smallest eigenvalues)
        var v = new double[4, 12];
        for (int k = 0; k < 4; k++)
            for (int j = 0; j < 12; j++)
                v[k, j] = allEigVecs[j, sortIdx[k]];

        // ═══ STEP 5: Try N=1 solution (most common, works for non-planar) ═══
        // Control points in camera frame: cc_j = beta * v1
        double bestErr = double.MaxValue;
        double[,]? bestR = null;
        double[]? bestT = null;

        // --- N=1: cc = beta * v1 ---
        {
            // Recover beta from distance constraint:
            // ||cc_i - cc_j||² = ||cw_i - cw_j||² for all pairs
            // With N=1: cc_j = beta * v1_j, so ||beta*(v1_i - v1_j)||² = ||cw_i - cw_j||²
            // beta² * Σ(dv²) = Σ(dw²) → beta = sqrt(Σdw² / Σdv²)

            double sumDV2 = 0, sumDW2 = 0;
            for (int i = 0; i < 4; i++)
                for (int j = i + 1; j < 4; j++)
                {
                    for (int d = 0; d < 3; d++)
                    {
                        double dv = v[0, i * 3 + d] - v[0, j * 3 + d];
                        double dw = cw[i, d] - cw[j, d];
                        sumDV2 += dv * dv;
                        sumDW2 += dw * dw;
                    }
                }

            if (sumDV2 > 1e-15)
            {
                double beta = Math.Sqrt(sumDW2 / sumDV2);

                // Try both signs of beta
                foreach (double b in new[] { beta, -beta })
                {
                    var cc = new double[4, 3];
                    for (int j = 0; j < 4; j++)
                        for (int d = 0; d < 3; d++)
                            cc[j, d] = b * v[0, j * 3 + d];

                    // Check: control points should have positive Z (in front of camera)
                    double avgZ = 0;
                    for (int j = 0; j < 4; j++) avgZ += cc[j, 2];
                    if (avgZ < 0) continue; // wrong sign

                    var result = RecoverRtFromControlPoints(cw, cc, pts2D, pts3D, n, alphas);
                    if (result.HasValue && result.Value.err < bestErr)
                    {
                        bestErr = result.Value.err;
                        bestR = result.Value.R;
                        bestT = result.Value.t;
                    }
                }
            }
        }

        // --- N=2: cc = beta1*v1 + beta2*v2 ---
        {
            // Distance constraints: for each pair (i,j):
            // sum_d (beta1*(v1_id - v1_jd) + beta2*(v2_id - v2_jd))² = ||cw_i - cw_j||²
            // Expand: beta1² * a + 2*beta1*beta2 * b + beta2² * c = rhs
            // We have C(4,2)=6 distance constraints for 3 unknowns (beta1², beta1*beta2, beta2²)
            // Solve via least squares

            var L = new double[6, 3]; // 6 constraints × 3 unknowns
            var rhs = new double[6];
            int row = 0;
            for (int i = 0; i < 4; i++)
                for (int j = i + 1; j < 4; j++)
                {
                    double a = 0, b = 0, c = 0, dist2 = 0;
                    for (int d = 0; d < 3; d++)
                    {
                        double dv1 = v[0, i * 3 + d] - v[0, j * 3 + d];
                        double dv2 = v[1, i * 3 + d] - v[1, j * 3 + d];
                        double dw = cw[i, d] - cw[j, d];
                        a += dv1 * dv1;
                        b += dv1 * dv2;
                        c += dv2 * dv2;
                        dist2 += dw * dw;
                    }
                    L[row, 0] = a;
                    L[row, 1] = 2 * b;
                    L[row, 2] = c;
                    rhs[row] = dist2;
                    row++;
                }

            // Solve L * [b1², b1*b2, b2²]ᵀ = rhs via normal equations
            var betas2 = SolveLeastSquares(L, rhs, 6, 3);
            if (betas2 != null)
            {
                // b1² = betas2[0], b1*b2 = betas2[1], b2² = betas2[2]
                double b1sq = Math.Max(betas2[0], 0);
                double b2sq = Math.Max(betas2[2], 0);
                double b1 = Math.Sqrt(b1sq);
                double b2 = (b1 > 1e-10) ? betas2[1] / b1 : Math.Sqrt(b2sq);

                foreach (var (s1, s2) in new[] { (b1, b2), (b1, -b2), (-b1, b2), (-b1, -b2) })
                {
                    var cc = new double[4, 3];
                    for (int j = 0; j < 4; j++)
                        for (int d = 0; d < 3; d++)
                            cc[j, d] = s1 * v[0, j * 3 + d] + s2 * v[1, j * 3 + d];

                    double avgZ = 0;
                    for (int j = 0; j < 4; j++) avgZ += cc[j, 2];
                    if (avgZ < 0) continue;

                    var result = RecoverRtFromControlPoints(cw, cc, pts2D, pts3D, n, alphas);
                    if (result.HasValue && result.Value.err < bestErr)
                    {
                        bestErr = result.Value.err;
                        bestR = result.Value.R;
                        bestT = result.Value.t;
                    }
                }
            }
        }

        if (bestR == null) return null;

        // ═══ STEP 6: Gauss-Newton refinement ═══
        var refined = GaussNewtonRefine(bestR, bestT!, pts2D, pts3D, n);
        if (refined.HasValue)
        {
            bestR = refined.Value.R;
            bestT = refined.Value.t;
        }

        return (bestR, bestT!);
    }

    /// <summary>
    /// Recover R, t from world control points (cw) and camera-frame control points (cc)
    /// using Procrustes alignment (SVD-based rigid body fitting).
    /// </summary>
    private static (double[,] R, double[] t, double err)?
        RecoverRtFromControlPoints(
            double[,] cw, double[,] cc,
            ReadOnlySpan<float> pts2D, ReadOnlySpan<float> pts3D, int n,
            double[,] alphas)
    {
        // Compute centroids
        double cwx = 0, cwy = 0, cwz = 0;
        double ccx = 0, ccy = 0, ccz = 0;
        for (int j = 0; j < 4; j++)
        {
            cwx += cw[j, 0]; cwy += cw[j, 1]; cwz += cw[j, 2];
            ccx += cc[j, 0]; ccy += cc[j, 1]; ccz += cc[j, 2];
        }
        cwx /= 4; cwy /= 4; cwz /= 4;
        ccx /= 4; ccy /= 4; ccz /= 4;

        // Cross-covariance H = sum( (cc_j - cc_mean) * (cw_j - cw_mean)^T )
        var H = new double[3, 3];
        for (int j = 0; j < 4; j++)
        {
            double ax = cc[j, 0] - ccx, ay = cc[j, 1] - ccy, az = cc[j, 2] - ccz;
            double bx = cw[j, 0] - cwx, by = cw[j, 1] - cwy, bz = cw[j, 2] - cwz;
            H[0, 0] += ax * bx; H[0, 1] += ax * by; H[0, 2] += ax * bz;
            H[1, 0] += ay * bx; H[1, 1] += ay * by; H[1, 2] += ay * bz;
            H[2, 0] += az * bx; H[2, 1] += az * by; H[2, 2] += az * bz;
        }

        // SVD of H: H = U * diag(S) * Vt
        var (U, _, Vt) = SVD3x3(H);

        // R = V * U^T (note: V = Vt^T)
        var R = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double sum = 0;
                for (int k = 0; k < 3; k++)
                    sum += Vt[k, i] * U[j, k]; // V[i,k] * U^T[k,j]
                R[i, j] = sum;
            }

        // Ensure det(R) = +1 (proper rotation)
        if (Det3x3(R) < 0)
        {
            // Flip last column of V (= last row of Vt)
            for (int i = 0; i < 3; i++)
                Vt[2, i] = -Vt[2, i];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < 3; k++)
                        sum += Vt[k, i] * U[j, k];
                    R[i, j] = sum;
                }
        }

        // t = cc_centroid - R * cw_centroid
        var t = new double[3];
        t[0] = ccx - (R[0, 0] * cwx + R[0, 1] * cwy + R[0, 2] * cwz);
        t[1] = ccy - (R[1, 0] * cwx + R[1, 1] * cwy + R[1, 2] * cwz);
        t[2] = ccz - (R[2, 0] * cwx + R[2, 1] * cwy + R[2, 2] * cwz);

        // Compute reprojection error
        double err = ComputeReprojError(R, t, pts2D, pts3D, n);

        return (R, t, err);
    }

    /// <summary>
    /// Compute mean reprojection error for a given R, t.
    /// </summary>
    private static double ComputeReprojError(
        double[,] R, double[] t,
        ReadOnlySpan<float> pts2D, ReadOnlySpan<float> pts3D, int n)
    {
        double totalErr = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            double X = pts3D[i * 3], Y = pts3D[i * 3 + 1], Z = pts3D[i * 3 + 2];
            double px = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0];
            double py = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1];
            double pz = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2];
            if (pz < 1e-6) continue;
            double u = px / pz, v = py / pz;
            double du = u - pts2D[i * 2], dv = v - pts2D[i * 2 + 1];
            totalErr += du * du + dv * dv;
            valid++;
        }
        return valid > 0 ? totalErr / valid : double.MaxValue;
    }

    /// <summary>
    /// Gauss-Newton refinement of R, t to minimize reprojection error.
    /// R is parameterized as angle-axis (3) + t (3).
    /// </summary>
    private static (double[,] R, double[] t)? GaussNewtonRefine(
        double[,] R0, double[] t0,
        ReadOnlySpan<float> pts2D, ReadOnlySpan<float> pts3D, int n)
    {
        // Convert R to angle-axis
        var params6 = new double[6];
        RotationToAngleAxis(R0, params6);
        params6[3] = t0[0]; params6[4] = t0[1]; params6[5] = t0[2];

        for (int iter = 0; iter < 10; iter++)
        {
            // Current R, t
            var R = AngleAxisToRotation(params6);
            double tx = params6[3], ty = params6[4], tz = params6[5];

            // Build J^T*J and J^T*r
            var JtJ = new double[6, 6];
            var Jtr = new double[6];
            double totalErr = 0;

            for (int i = 0; i < n; i++)
            {
                double X = pts3D[i * 3], Y = pts3D[i * 3 + 1], Z = pts3D[i * 3 + 2];
                double px = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + tx;
                double py = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + ty;
                double pz = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + tz;
                if (pz < 1e-6) continue;

                double invZ = 1.0 / pz;
                double u = px * invZ, v2 = py * invZ;
                double ru = u - pts2D[i * 2];
                double rv = v2 - pts2D[i * 2 + 1];
                totalErr += ru * ru + rv * rv;

                // Jacobian via finite differences (robust for small implementation)
                double eps = 1e-6;
                var J = new double[2, 6];
                for (int d = 0; d < 6; d++)
                {
                    var pp = (double[])params6.Clone();
                    pp[d] += eps;
                    var Rp = AngleAxisToRotation(pp);
                    double ppx = Rp[0, 0] * X + Rp[0, 1] * Y + Rp[0, 2] * Z + pp[3];
                    double ppy = Rp[1, 0] * X + Rp[1, 1] * Y + Rp[1, 2] * Z + pp[4];
                    double ppz = Rp[2, 0] * X + Rp[2, 1] * Y + Rp[2, 2] * Z + pp[5];
                    if (ppz < 1e-6) ppz = 1e-6;
                    J[0, d] = (ppx / ppz - u) / eps;
                    J[1, d] = (ppy / ppz - v2) / eps;
                }

                // Accumulate JᵀJ and Jᵀr
                for (int a = 0; a < 6; a++)
                {
                    for (int b = 0; b < 6; b++)
                        JtJ[a, b] += J[0, a] * J[0, b] + J[1, a] * J[1, b];
                    Jtr[a] += J[0, a] * ru + J[1, a] * rv;
                }
            }

            // Add damping (Levenberg-Marquardt)
            for (int d = 0; d < 6; d++)
                JtJ[d, d] *= 1.001;

            // Solve JᵀJ * delta = -Jᵀr
            var negJtr = new double[6];
            for (int d = 0; d < 6; d++) negJtr[d] = -Jtr[d];
            var delta = SolveLinearSystem6x6(JtJ, negJtr);
            if (delta == null) break;

            // Update
            for (int d = 0; d < 6; d++) params6[d] += delta[d];

            // Check convergence
            double norm = 0;
            for (int d = 0; d < 6; d++) norm += delta[d] * delta[d];
            if (norm < 1e-14) break;
        }

        var finalR = AngleAxisToRotation(params6);
        var finalT = new double[] { params6[3], params6[4], params6[5] };
        return (finalR, finalT);
    }

    // ── Helpers ──

    /// <summary>
    /// Eigendecomposition of a 3×3 symmetric matrix using Jacobi iteration.
    /// Returns (eigenvectors[3,3], eigenvalues[3]) sorted by eigenvalue descending.
    /// </summary>
    private static (double[,] vecs, double[] vals) SymEig3x3(double[,] S)
    {
        var A = new double[3, 3];
        var V = new double[3, 3];
        for (int i = 0; i < 3; i++) { V[i, i] = 1.0; for (int j = 0; j < 3; j++) A[i, j] = S[i, j]; }

        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = Math.Abs(A[0, 1]) + Math.Abs(A[0, 2]) + Math.Abs(A[1, 2]);
            if (off < 1e-14) break;

            for (int p = 0; p < 2; p++)
                for (int q = p + 1; q < 3; q++)
                {
                    if (Math.Abs(A[p, q]) < 1e-15) continue;
                    double tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q]);
                    double t = (tau >= 0) ? 1.0 / (tau + Math.Sqrt(1 + tau * tau))
                                           : -1.0 / (-tau + Math.Sqrt(1 + tau * tau));
                    double c = 1.0 / Math.Sqrt(1 + t * t), s = t * c;

                    double app = A[p, p], aqq = A[q, q], apq = A[p, q];
                    A[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    A[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    A[p, q] = 0; A[q, p] = 0;
                    for (int r = 0; r < 3; r++)
                    {
                        if (r == p || r == q) continue;
                        double arp = A[r, p], arq = A[r, q];
                        A[r, p] = c * arp - s * arq; A[p, r] = A[r, p];
                        A[r, q] = s * arp + c * arq; A[q, r] = A[r, q];
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        double vp = V[i, p], vq = V[i, q];
                        V[i, p] = c * vp - s * vq;
                        V[i, q] = s * vp + c * vq;
                    }
                }
        }

        // Sort descending
        var vals = new double[] { A[0, 0], A[1, 1], A[2, 2] };
        var idx = new[] { 0, 1, 2 };
        Array.Sort(idx, (a, b) => vals[b].CompareTo(vals[a]));

        var sortedVals = new double[3];
        var sortedVecs = new double[3, 3];
        for (int j = 0; j < 3; j++)
        {
            sortedVals[j] = vals[idx[j]];
            for (int i = 0; i < 3; i++) sortedVecs[i, j] = V[i, idx[j]];
        }
        return (sortedVecs, sortedVals);
    }

    /// <summary>
    /// Full Jacobi eigendecomposition of a symmetric NxN matrix.
    /// Returns (eigenvectors[n,n], eigenvalues[n]).
    /// </summary>
    private static (double[,] vecs, double[] vals) JacobiAllEigenvectors(double[,] S, int n)
    {
        var A = new double[n, n];
        var V = new double[n, n];
        for (int i = 0; i < n; i++) { V[i, i] = 1.0; for (int j = 0; j < n; j++) A[i, j] = S[i, j]; }

        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    off = Math.Max(off, Math.Abs(A[i, j]));
            if (off < 1e-14) break;

            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(A[p, q]) < 1e-15) continue;
                    double tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q]);
                    double t = (tau >= 0) ? 1.0 / (tau + Math.Sqrt(1 + tau * tau))
                                           : -1.0 / (-tau + Math.Sqrt(1 + tau * tau));
                    double c = 1.0 / Math.Sqrt(1 + t * t), s = t * c;

                    double app = A[p, p], aqq = A[q, q], apq = A[p, q];
                    A[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    A[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    A[p, q] = 0; A[q, p] = 0;
                    for (int r = 0; r < n; r++)
                    {
                        if (r == p || r == q) continue;
                        double arp = A[r, p], arq = A[r, q];
                        A[r, p] = c * arp - s * arq; A[p, r] = A[r, p];
                        A[r, q] = s * arp + c * arq; A[q, r] = A[r, q];
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double vp = V[i, p], vq = V[i, q];
                        V[i, p] = c * vp - s * vq;
                        V[i, q] = s * vp + c * vq;
                    }
                }
        }

        var vals = new double[n];
        for (int i = 0; i < n; i++) vals[i] = A[i, i];
        return (V, vals);
    }

    /// <summary>
    /// Solve least squares Ax = b via normal equations (AᵀA)x = Aᵀb.
    /// </summary>
    private static double[]? SolveLeastSquares(double[,] A, double[] b, int m, int n)
    {
        var AtA = new double[n, n];
        var Atb = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < m; k++) sum += A[k, i] * A[k, j];
                AtA[i, j] = sum;
            }
            double s = 0;
            for (int k = 0; k < m; k++) s += A[k, i] * b[k];
            Atb[i] = s;
        }
        return SolveLinearSystemGeneral(AtA, Atb, n);
    }

    /// <summary>
    /// Solve a general NxN linear system via Gaussian elimination with partial pivoting.
    /// </summary>
    private static double[]? SolveLinearSystemGeneral(double[,] A, double[] b, int n)
    {
        var M = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) M[i, j] = A[i, j];
            M[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(M[row, col]) > Math.Abs(M[maxRow, col])) maxRow = row;
            if (Math.Abs(M[maxRow, col]) < 1e-14) return null;
            for (int j = 0; j <= n; j++)
                (M[col, j], M[maxRow, j]) = (M[maxRow, j], M[col, j]);
            for (int row = col + 1; row < n; row++)
            {
                double f = M[row, col] / M[col, col];
                for (int j = col; j <= n; j++) M[row, j] -= f * M[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = M[i, n];
            for (int j = i + 1; j < n; j++) sum -= M[i, j] * x[j];
            x[i] = sum / M[i, i];
        }
        return x;
    }

    /// <summary>
    /// Solve a 6x6 linear system via Gaussian elimination.
    /// </summary>
    private static double[]? SolveLinearSystem6x6(double[,] A, double[] b)
        => SolveLinearSystemGeneral(A, b, 6);

    /// <summary>
    /// Convert rotation matrix to angle-axis (Rodrigues).
    /// </summary>
    private static void RotationToAngleAxis(double[,] R, double[] aa)
    {
        double trace = R[0, 0] + R[1, 1] + R[2, 2];
        double cos = (trace - 1.0) / 2.0;
        cos = Math.Max(-1.0, Math.Min(1.0, cos));
        double angle = Math.Acos(cos);

        if (angle < 1e-10)
        {
            aa[0] = aa[1] = aa[2] = 0;
            return;
        }

        double s = 2.0 * Math.Sin(angle);
        if (Math.Abs(s) < 1e-10)
        {
            // angle ≈ π: use eigendecomposition
            aa[0] = aa[1] = aa[2] = 0;
            return;
        }

        aa[0] = (R[2, 1] - R[1, 2]) / s * angle;
        aa[1] = (R[0, 2] - R[2, 0]) / s * angle;
        aa[2] = (R[1, 0] - R[0, 1]) / s * angle;
    }

    /// <summary>
    /// Convert angle-axis to rotation matrix (Rodrigues formula).
    /// </summary>
    private static double[,] AngleAxisToRotation(double[] p)
    {
        double ax = p[0], ay = p[1], az = p[2];
        double theta2 = ax * ax + ay * ay + az * az;
        var R = new double[3, 3];

        if (theta2 < 1e-10)
        {
            R[0, 0] = 1; R[1, 1] = 1; R[2, 2] = 1;
            R[0, 1] = -az; R[0, 2] = ay;
            R[1, 0] = az; R[1, 2] = -ax;
            R[2, 0] = -ay; R[2, 1] = ax;
            return R;
        }

        double theta = Math.Sqrt(theta2);
        double c = Math.Cos(theta), s = Math.Sin(theta);
        double ic = 1.0 - c;
        double nx = ax / theta, ny = ay / theta, nz = az / theta;

        R[0, 0] = c + nx * nx * ic;
        R[0, 1] = nx * ny * ic - nz * s;
        R[0, 2] = nx * nz * ic + ny * s;
        R[1, 0] = ny * nx * ic + nz * s;
        R[1, 1] = c + ny * ny * ic;
        R[1, 2] = ny * nz * ic - nx * s;
        R[2, 0] = nz * nx * ic - ny * s;
        R[2, 1] = nz * ny * ic + nx * s;
        R[2, 2] = c + nz * nz * ic;
        return R;
    }
}
