using System.Numerics;

namespace SpawnScene.Services;

/// <summary>When the viewer's frustum cull + depth sort must be recomputed (see GpuSplatSorter).</summary>
public static class SortInvalidation
{
    /// <summary>
    /// True when the cull/sort must be redone for this frame: the camera moved, OR the view-projection
    /// the last sort used differs from this frame's (field of view, intrinsics, viewport).
    /// </summary>
    public static bool NeedsResort(float velocity, Matrix4x4 lastSortedMvp, Matrix4x4 mvp)
    {
        if (velocity > 1e-8f) return true;
        if (float.IsNaN(lastSortedMvp.M11)) return true;
        const float eps = 1e-6f;
        return MathF.Abs(mvp.M11 - lastSortedMvp.M11) > eps || MathF.Abs(mvp.M12 - lastSortedMvp.M12) > eps
            || MathF.Abs(mvp.M13 - lastSortedMvp.M13) > eps || MathF.Abs(mvp.M14 - lastSortedMvp.M14) > eps
            || MathF.Abs(mvp.M21 - lastSortedMvp.M21) > eps || MathF.Abs(mvp.M22 - lastSortedMvp.M22) > eps
            || MathF.Abs(mvp.M23 - lastSortedMvp.M23) > eps || MathF.Abs(mvp.M24 - lastSortedMvp.M24) > eps
            || MathF.Abs(mvp.M31 - lastSortedMvp.M31) > eps || MathF.Abs(mvp.M32 - lastSortedMvp.M32) > eps
            || MathF.Abs(mvp.M33 - lastSortedMvp.M33) > eps || MathF.Abs(mvp.M34 - lastSortedMvp.M34) > eps
            || MathF.Abs(mvp.M41 - lastSortedMvp.M41) > eps || MathF.Abs(mvp.M42 - lastSortedMvp.M42) > eps
            || MathF.Abs(mvp.M43 - lastSortedMvp.M43) > eps || MathF.Abs(mvp.M44 - lastSortedMvp.M44) > eps;
    }
}
