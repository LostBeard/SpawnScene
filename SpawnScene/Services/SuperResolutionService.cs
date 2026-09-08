using ILGPU;
using ILGPU.Runtime;

namespace SpawnScene.Services;

/// <summary>
/// PARKED SCAFFOLD (native super-resolution). ORT path removed 2026-07-01 as part of the
/// zero-ORT migration — SpawnScene is now 100% native ML (SpawnDev.ILGPU + SpawnDev.ILGPU.ML).
///
/// The old implementation ran an ONNX SR model (sr_x2/sr_x4.onnx) through ONNX Runtime Web to
/// upscale the source image before depth estimation, for higher-resolution depth maps and more
/// detailed Gaussian splats. That was the last ORT consumer in the project and was never wired to
/// a live call site, so it is retired rather than carried as dead ORT weight.
///
/// The two GPU kernels below (packed-RGBA int ↔ NCHW float32 [0,1]) are the reusable, backend-agnostic
/// pieces and are kept so a FUTURE NATIVE SR pass is a drop-in: implement upscaling via a
/// SpawnDev.ILGPU.ML pipeline (e.g. a Real-ESRGAN / realesr-animevideov3 ONNX imported through the
/// native engine, like DepthEstimationPipeline), reusing these kernels for the pre/post format
/// conversion. Keep the data GPU-resident end to end (no CPU readback) per the GPU-First Pipeline Rule.
///
/// The <c>ProjectSettings.UseSuperResolution</c> flag is likewise parked for that future path.
/// </summary>
public static class SuperResolutionService
{
    // ─────────────────────────────────────────────────────────────
    //  GPU Kernels (parked — reusable for a future native SR pass)
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: packed RGBA int → NCHW float32 [0,1].
    /// No resize (SR models process at full resolution).
    /// Channel order: R=ch0, G=ch1, B=ch2.
    /// </summary>
    public static void RgbaToNchwKernel(
        Index1D idx,
        ArrayView1D<int, Stride1D.Dense> srcRgba,
        ArrayView1D<float, Stride1D.Dense> dstNchw,
        int w, int h, int offset)
    {
        int absIdx = idx + offset;
        int pixCount = w * h;
        int c = absIdx / pixCount;
        int pix = absIdx % pixCount;
        int packed = srcRgba[pix];
        dstNchw[absIdx] = ((packed >> (c * 8)) & 0xFF) / 255f;
    }

    /// <summary>
    /// GPU kernel: NCHW float32 [0,1] → packed RGBA int.
    /// Each invocation writes one pixel (R+G+B channels, A=255).
    /// </summary>
    public static void NchwToRgbaKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> srcNchw,
        ArrayView1D<int, Stride1D.Dense> dstRgba,
        int w, int h, int offset)
    {
        int pixIdx = idx + offset;
        int pixCount = w * h;
        float r = srcNchw[0 * pixCount + pixIdx];
        float g = srcNchw[1 * pixCount + pixIdx];
        float b = srcNchw[2 * pixCount + pixIdx];

        int rv = r <= 0f ? 0 : r >= 1f ? 255 : (int)(r * 255f + 0.5f);
        int gv = g <= 0f ? 0 : g >= 1f ? 255 : (int)(g * 255f + 0.5f);
        int bv = b <= 0f ? 0 : b >= 1f ? 255 : (int)(b * 255f + 0.5f);

        dstRgba[pixIdx] = rv | (gv << 8) | (bv << 16) | (255 << 24);
    }
}
