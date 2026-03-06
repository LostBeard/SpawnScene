using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Rendering;

namespace SpawnScene.Services;

/// <summary>
/// GPU depth map colorizer using the Turbo colormap.
/// Outputs a MemoryBuffer2D packed-RGBA buffer and blits it to an HTML canvas
/// via ICanvasRenderer (WebGPUCanvasRenderer = zero CPU readback).
/// </summary>
public class GpuDepthColorizer : IAsyncDisposable
{
    private readonly GpuService _gpu;

    private Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,  // depth [h*w]
        ArrayView2D<uint, Stride2D.DenseX>,  // output [h, w] packed RGBA
        float, float,                        // minDepth, maxDepth
        int, int,                            // depthW, depthH
        int, int>?                           // outW, outH
        _colorizeKernel;

    private ICanvasRenderer? _renderer;
    private MemoryBuffer2D<uint, Stride2D.DenseX>? _colorBuf;
    private int _colorBufW, _colorBufH;

    public GpuDepthColorizer(GpuService gpu) => _gpu = gpu;

    // ─────────────────────────────────────────────────────────────
    //  GPU Kernel
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: depth map → Turbo colormap RGBA pixel.
    /// Maps each output pixel to the corresponding depth value, applies
    /// the Turbo colormap (close = warm, far = cool), writes packed RGBA uint32.
    /// Pixel format: little-endian RGBA — R in bits 0–7, A in bits 24–31.
    /// </summary>
    private static void ColorizeKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView2D<uint, Stride2D.DenseX> output,
        float minDepth, float maxDepth,
        int depthW, int depthH,
        int outW, int outH)
    {
        int px = idx.X;
        int py = idx.Y;

        // Nearest-neighbor sample from depth to output resolution
        int dx = px * depthW / outW;
        int dy = py * depthH / outH;
        dx = dx < 0 ? 0 : (dx >= depthW ? depthW - 1 : dx);
        dy = dy < 0 ? 0 : (dy >= depthH ? depthH - 1 : dy);

        float v = depth[dy * depthW + dx];
        float range = maxDepth - minDepth;
        float t = (range > 1e-6f) ? (v - minDepth) / range : 0f;

        // Invert: close = warm end (t→0), far = cool end (t→1)
        t = 1f - t;
        t = t < 0f ? 0f : (t > 1f ? 1f : t);

        // Turbo colormap polynomial approximation (Google Research)
        float r = 0.13572138f + t * (4.61539260f + t * (-42.66032258f + t * (132.13108234f + t * (-152.94239396f + t * 59.28637943f))));
        float g = 0.09140261f + t * (2.19418839f + t * (4.84296658f + t * (-14.18503333f + t * (4.27729857f + t * 2.82956604f))));
        float b = 0.10667330f + t * (12.64194608f + t * (-60.58204836f + t * (110.36276771f + t * (-89.90310912f + t * 27.34824973f))));

        r = r < 0f ? 0f : (r > 1f ? 1f : r);
        g = g < 0f ? 0f : (g > 1f ? 1f : g);
        b = b < 0f ? 0f : (b > 1f ? 1f : b);

        uint rb = (uint)(r * 255f);
        uint gb = (uint)(g * 255f);
        uint bb = (uint)(b * 255f);

        // Little-endian RGBA: R in byte 0, A in byte 3
        output[idx] = 0xFF000000u | (bb << 16) | (gb << 8) | rb;
    }

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Colorize a GPU-resident depth map and present it to an HTML canvas.
    /// Zero CPU readback — depth stays on GPU throughout, canvas receives the
    /// result via WebGPUCanvasRenderer's fullscreen-triangle blit (no drawImage copy).
    /// </summary>
    public async Task ColorizePresentAsync(DepthResult depth, ElementReference canvasRef, int outW, int outH)
    {
        if (depth.RawDepthGpu == null) return;

        if (!_gpu.IsInitialized) await _gpu.InitializeAsync();
        var accelerator = _gpu.WebGPUAccelerator;

        _colorizeKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<uint, Stride2D.DenseX>,
            float, float,
            int, int,
            int, int>(ColorizeKernel);

        // Reallocate output buffer only when resolution changes
        if (_colorBuf == null || _colorBufW != outW || _colorBufH != outH)
        {
            _colorBuf?.Dispose();
            _colorBuf = accelerator.Allocate2DDenseX<uint>(new Index2D(outW, outH));
            _colorBufW = outW;
            _colorBufH = outH;
        }

        // Run colorize kernel — entirely GPU-resident
        _colorizeKernel(
            _colorBuf.IntExtent,
            depth.RawDepthGpu.View,
            _colorBuf.View,
            depth.MinDepth, depth.MaxDepth,
            depth.Width, depth.Height,
            outW, outH);

        await _gpu.SynchronizeAsync();

        // Create renderer once (or reattach when canvas ref changes)
        _renderer ??= _gpu.CreateCanvasRenderer();
        using var canvas = new HTMLCanvasElement(canvasRef);
        _renderer.AttachCanvas(canvas);

        // Present: WebGPUCanvasRenderer blits the GPU buffer to the canvas
        // via a fullscreen-triangle render pass — no CPU readback, no drawImage copy.
        await _renderer.PresentAsync(_colorBuf);
    }

    public async ValueTask DisposeAsync()
    {
        _renderer?.Dispose();
        _renderer = null;
        _colorBuf?.Dispose();
        _colorBuf = null;
        GC.SuppressFinalize(this);
    }
}
