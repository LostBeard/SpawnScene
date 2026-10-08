using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Generative fill for what no photo saw (TJ 2026-10-08: "Generative fill (opt-in)", <c>&amp;fillunseen=1</c>). After a
/// multi-view scene is trained, turning around at the capture position shows holes where nothing was photographed
/// (Bathroom pan-2: 0 of 33 photos face it; Hamamni pan-4/5: 0 of 57). From the cameras' centre this renders views all
/// around (the trainer's rasteriser: colour, transmittance T and inverse depth), masks the pixels the scene does not
/// cover, has big-LaMa / MI-GAN paint them (HiddenLayerInpaint), and adds the painted pixels as thin camera-facing splats
/// at a depth spread from the covered pixels around them (push-pull). Each view's fill is pasted before the next view
/// renders, so overlapping views do not paint the same gap twice. Invented content - opt-in, measured off-path.
/// Everything stays on the GPU except two counters per view.
/// </summary>
public static class UnseenFill
{
    public const int S = HiddenLayerInpaint.Size;

    public struct Params
    {
        public int W, H;              // the render
        public float Known;           // coverage (1 - T) at or above which a pixel is the scene's, not a hole
        public float Fx, Fy, Cx, Cy;  // the render's intrinsics
        public float Px, Py, Pz;      // camera position
        public float Rx, Ry, Rz, Ux, Uy, Uz, Fwx, Fwy, Fwz;   // right, up (image y runs DOWN = -up), forward
        public int Stride;            // emit one splat every Stride grid cells
        public int Capacity;
        public float Opacity;
    }

    /// <summary>The render resampled to the S x S paint grid: un-premultiplied colour 0..255, mask (255 = the scene's, 0 =
    /// paint), camera depth where covered (0 = unknown); masked cells counted in counters[0].</summary>
    static void PrepKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> colour, ArrayView1D<float, Stride1D.Dense> trans,
        ArrayView1D<float, Stride1D.Dense> invDepth, ArrayView1D<float, Stride1D.Dense> image,
        ArrayView1D<float, Stride1D.Dense> mask, ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<int, Stride1D.Dense> counters,
        Params p)
    {
        const int Px = S * S;
        if (i >= Px) return;
        int x = i % S, y = i / S;
        int sx = XMath.Min(p.W - 1, (int)((x + 0.5f) * p.W / S));
        int sy = XMath.Min(p.H - 1, (int)((y + 0.5f) * p.H / S));
        int o = sy * p.W + sx;
        float cov = 1f - trans[o];
        float inv = cov > 0.05f ? 1f / cov : 0f;
        for (int c = 0; c < 3; c++)
            image[c * Px + i] = XMath.Clamp(colour[o * 3 + c] * inv, 0f, 1f) * 255f;
        bool known = cov >= p.Known;
        mask[i] = known ? 255f : 0f;
        float id = invDepth[o];
        depth[i] = cov > 0.5f && id > 1e-12f ? cov / id : 0f;
        if (!known) Atomic.Add(ref counters[0], 1);
    }

    /// <summary>Push: a 2x coarser level's mean of the known depths below it (weight = how many were known).</summary>
    static void DownKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> fineZ, ArrayView1D<float, Stride1D.Dense> fineW,
        ArrayView1D<float, Stride1D.Dense> coarseZ, ArrayView1D<float, Stride1D.Dense> coarseW, int fineSize)
    {
        int cs = fineSize / 2;
        if (i >= cs * cs) return;
        int x = i % cs, y = i / cs;
        float zw = 0f, w = 0f;
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
            {
                int f = (2 * y + dy) * fineSize + 2 * x + dx;
                zw += fineZ[f] * fineW[f];
                w += fineW[f];
            }
        coarseW[i] = w;
        coarseZ[i] = w > 0f ? zw / w : 0f;
    }

    /// <summary>
    /// Pull: a fine cell nothing below knew takes the coarse level's depth, BILINEARLY at its own centre. Taking its
    /// parent's value (nearest) left the filled depth in flat steps the size of the coarse cells: the fill splats sat on
    /// stepped planes that split into hard-edged blocks with gaps from any other viewpoint (x0 Bathroom ceiling).
    /// </summary>
    static void UpKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> fineZ, ArrayView1D<float, Stride1D.Dense> fineW,
        ArrayView1D<float, Stride1D.Dense> coarseZ, int fineSize)
    {
        if (i >= fineSize * fineSize) return;
        if (fineW[i] > 0f) return;
        int x = i % fineSize, y = i / fineSize;
        int cs = fineSize / 2;
        float cx = XMath.Clamp((x + 0.5f) * 0.5f - 0.5f, 0f, cs - 1f);
        float cy = XMath.Clamp((y + 0.5f) * 0.5f - 0.5f, 0f, cs - 1f);
        int x0 = (int)cx, y0 = (int)cy;
        int x1 = XMath.Min(cs - 1, x0 + 1), y1 = XMath.Min(cs - 1, y0 + 1);
        float fx = cx - x0, fy = cy - y0;
        float top = coarseZ[y0 * cs + x0] * (1f - fx) + coarseZ[y0 * cs + x1] * fx;
        float bottom = coarseZ[y1 * cs + x0] * (1f - fx) + coarseZ[y1 * cs + x1] * fx;
        fineZ[i] = top * (1f - fy) + bottom * fy;
        fineW[i] = 1e-6f;
    }

    static void WeightKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> z, ArrayView1D<float, Stride1D.Dense> w)
    {
        if (i >= z.Length) return;
        w[i] = z[i] > 0f ? 1f : 0f;
    }

    /// <summary>One thin splat per painted grid cell (every Stride-th), facing the camera at the filled depth, sized to its
    /// cell's footprint; rows counted in counters[1].</summary>
    static void EmitKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> painted, ArrayView1D<float, Stride1D.Dense> mask,
        ArrayView1D<float, Stride1D.Dense> depth, ArrayView1D<float, Stride1D.Dense> rows,
        ArrayView1D<int, Stride1D.Dense> counters, Params p)
    {
        const int Px = S * S;
        if (i >= Px) return;
        int x = i % S, y = i / S;
        if (x % p.Stride != 0 || y % p.Stride != 0) return;
        if (mask[i] > 127f) return;
        float z = depth[i];
        if (!(z > 0f)) return;
        // The cell's centre in render pixels, then its ray (image y down = -up).
        float u = (x + 0.5f * p.Stride) * p.W / S, v = (y + 0.5f * p.Stride) * p.H / S;
        float a = (u - p.Cx) / p.Fx, b = (v - p.Cy) / p.Fy;
        float dx = p.Fwx + a * p.Rx - b * p.Ux;
        float dy = p.Fwy + a * p.Ry - b * p.Uy;
        float dz = p.Fwz + a * p.Rz - b * p.Uz;
        int k = Atomic.Add(ref counters[1], 1);
        if (k >= p.Capacity) return;
        int o = k * SplatFormat.Floats;
        rows[o + SplatFormat.OffPos] = p.Px + dx * z;
        rows[o + SplatFormat.OffPos + 1] = p.Py + dy * z;
        rows[o + SplatFormat.OffPos + 2] = p.Pz + dz * z;
        for (int c = 0; c < 3; c++) rows[o + SplatFormat.OffColor + c] = painted[c * Px + i] / 255f;
        // In-plane 1 sigma = 0.6 of the cell's footprint (neighbours overlap and blend into a surface); 0.2 of that across.
        float ray = XMath.Sqrt(1f + a * a + b * b);
        float sigma = 0.6f * z * ray * p.Stride * p.W / S / p.Fx;
        rows[o + SplatFormat.OffScale] = sigma;
        rows[o + SplatFormat.OffScale + 1] = sigma;
        rows[o + SplatFormat.OffScale + 2] = 0.2f * sigma;
        rows[o + SplatFormat.OffOpacity] = p.Opacity;
        // Local axes -> world: x = right, y = up, z = -forward (right-handed; the disk's normal along the view).
        float m00 = p.Rx, m10 = p.Ry, m20 = p.Rz;
        float m01 = p.Ux, m11 = p.Uy, m21 = p.Uz;
        float m02 = -p.Fwx, m12 = -p.Fwy, m22 = -p.Fwz;
        float qw, qx, qy, qz;
        float tr = m00 + m11 + m22;
        if (tr > 0f)
        {
            float s = XMath.Sqrt(tr + 1f) * 2f;
            qw = 0.25f * s; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s;
        }
        else if (m00 > m11 && m00 > m22)
        {
            float s = XMath.Sqrt(1f + m00 - m11 - m22) * 2f;
            qw = (m21 - m12) / s; qx = 0.25f * s; qy = (m01 + m10) / s; qz = (m02 + m20) / s;
        }
        else if (m11 > m22)
        {
            float s = XMath.Sqrt(1f + m11 - m00 - m22) * 2f;
            qw = (m02 - m20) / s; qx = (m01 + m10) / s; qy = 0.25f * s; qz = (m12 + m21) / s;
        }
        else
        {
            float s = XMath.Sqrt(1f + m22 - m00 - m11) * 2f;
            qw = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = 0.25f * s;
        }
        rows[o + SplatFormat.OffQuat] = qx;
        rows[o + SplatFormat.OffQuat + 1] = qy;
        rows[o + SplatFormat.OffQuat + 2] = qz;
        rows[o + SplatFormat.OffQuat + 3] = qw;
    }

    static Accelerator? _for;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, Params>? _prep;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _down;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _up;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _weight;
    static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>? _emit;

    static void Load(Accelerator a)
    {
        if (ReferenceEquals(_for, a) && _prep != null) return;
        _for = a;
        _prep = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>(PrepKernel);
        _down = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(DownKernel);
        _up = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(UpKernel);
        _weight = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(WeightKernel);
        _emit = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, Params>(EmitKernel);
    }

    /// <summary>Fill every unknown depth from the known ones around it: a push-pull pyramid over the S x S grid.</summary>
    static void PushPull(Accelerator a, MemoryBuffer1D<float, Stride1D.Dense> depth, List<MemoryBuffer1D<float, Stride1D.Dense>> scratch)
    {
        var zs = new List<MemoryBuffer1D<float, Stride1D.Dense>> { depth };
        var ws = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        var w0 = a.Allocate1D<float>((long)S * S); scratch.Add(w0); ws.Add(w0);
        _weight!((int)depth.Length, depth.View, w0.View);
        for (int size = S; size > 1; size /= 2)
        {
            int cs = size / 2;
            var cz = a.Allocate1D<float>((long)cs * cs); scratch.Add(cz);
            var cw = a.Allocate1D<float>((long)cs * cs); scratch.Add(cw);
            _down!(cs * cs, zs[^1].View, ws[^1].View, cz.View, cw.View, size);
            zs.Add(cz); ws.Add(cw);
        }
        for (int l = zs.Count - 2; l >= 0; l--)
        {
            int size = S >> l;
            _up!(size * size, zs[l].View, ws[l].View, zs[l + 1].View, size);
        }
    }

    /// <summary>
    /// Fill the scene on <paramref name="renderer"/> around <paramref name="centre"/>: 8 headings 45 degrees apart
    /// (60 degree horizontal field) plus straight up and down, rendered by <paramref name="trainer"/> at its size.
    /// Returns the splats added.
    /// </summary>
    public static async Task<int> FillAsync(Accelerator a, SplatTrainerGpu trainer, GpuGaussianRenderer renderer,
        HiddenLayerInpaint inpaint, Vector3 centre, Vector3 sceneUp, int stride = 2, float known = 0.7f, float minMasked = 0.01f)
    {
        Load(a);
        var (w, h) = trainer.Size;
        if (w <= 0 || h <= 0) return 0;
        trainer.EnableDepthRender();
        int degree = trainer.ActiveShDegree;
        // DC colour only: the trainer's SH rows cover the trained splats, not the ones pasted here.
        trainer.ActiveShDegree = 0;
        float fx = 0.5f * w / MathF.Tan(30f * MathF.PI / 180f);
        var up = Vector3.Normalize(sceneUp);
        var side = Vector3.Normalize(MathF.Abs(up.X) < 0.9f ? Vector3.Cross(up, Vector3.UnitX) : Vector3.Cross(up, Vector3.UnitZ));
        var side2 = Vector3.Cross(up, side);
        var views = new List<(string Name, Vector3 Fwd, Vector3 Up)>();
        for (int k = 0; k < 8; k++)
        {
            float t = k * MathF.PI / 4f;
            views.Add(($"h{k * 45}", MathF.Cos(t) * side + MathF.Sin(t) * side2, up));
        }
        views.Add(("up", up, side));
        views.Add(("down", -up, side));
        int added = 0;
        try
        {
            foreach (var (name, fwdRaw, upRaw) in views)
            {
                var packed = renderer.PackedSplatBuffer;
                int n = renderer.SplatCount;
                if (packed == null || n <= 0) break;
                var cam = new CameraParams
                {
                    Width = w, Height = h, FocalX = fx, FocalY = fx, CenterX = w / 2f, CenterY = h / 2f,
                    Position = centre, Forward = Vector3.Normalize(fwdRaw), Up = upRaw,
                };
                var box = await SplatBounds.ComputeAsync(a, packed, n);
                if (box == null) break;
                var (near, far) = SplatBounds.DepthRangeFor(box.Value, cam);
                await trainer.RenderForwardAsync(packed, n, cam, near, far, readback: false, depth: true);
                // No splat in view: the raster never ran (T stays cleared to 0 = "covered"), and there is no depth to
                // place a fill at.
                if (trainer.LastKeyCount == 0) { Console.WriteLine($"[Fill] {name}: nothing of the scene in view - skipped"); continue; }
                var right = cam.Right;
                var upO = Vector3.Normalize(Vector3.Cross(right, cam.Forward));
                var p = new Params
                {
                    W = w, H = h, Known = known, Fx = fx, Fy = fx, Cx = w / 2f, Cy = h / 2f,
                    Px = centre.X, Py = centre.Y, Pz = centre.Z,
                    Rx = right.X, Ry = right.Y, Rz = right.Z, Ux = upO.X, Uy = upO.Y, Uz = upO.Z,
                    Fwx = cam.Forward.X, Fwy = cam.Forward.Y, Fwz = cam.Forward.Z,
                    Stride = stride, Opacity = 0.95f,
                };
                var scratch = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
                try
                {
                    var image = a.Allocate1D<float>(3L * S * S); scratch.Add(image);
                    var mask = a.Allocate1D<float>((long)S * S); scratch.Add(mask);
                    var depth = a.Allocate1D<float>((long)S * S); scratch.Add(depth);
                    using var counters = a.Allocate1D<int>(2);
                    counters.MemSetToZero();
                    _prep!(S * S, trainer.RenderedColour!.View, trainer.RenderedTransmittance!.View,
                        trainer.RenderedInverseDepth!.View.SubView(0, (long)w * h), image.View, mask.View, depth.View,
                        counters.View, p);
                    // CPU transfer: one counter - how much of this view is a hole.
                    int masked = (await counters.CopyToHostAsync<int>(0, 1))[0];
                    float share = masked / (float)(S * S);
                    if (share < minMasked)
                    {
                        Console.WriteLine($"[Fill] {name}: {share:P1} uncovered - nothing to paint");
                        continue;
                    }
                    PushPull(a, depth, scratch);
                    using var painted = await inpaint.RunAsync(a, image.View, mask.View);
                    if (painted == null) { Console.WriteLine($"[Fill] {name}: the inpainting model is unavailable - stopping"); break; }
                    int capacity = (S / stride) * (S / stride);
                    p.Capacity = capacity;
                    // Owned by the clipboard below (it disposes it), not by scratch.
                    var rows = a.Allocate1D<float>((long)capacity * SplatFormat.Floats);
                    _emit!(S * S, painted.View, mask.View, depth.View, rows.View, counters.View, p);
                    // CPU transfer: one counter - the rows emitted.
                    int count = Math.Min(capacity, (await counters.CopyToHostAsync<int>(1, 1))[0]);
                    if (count <= 0) { rows.Dispose(); continue; }
                    using var clip = await SplatClipboard.FromSceneAsync(a, rows, count, null, 0, shDc: false);
                    await clip.MatchColoursAsync(a, renderer.ColoursAreShDc);
                    await clip.PasteAsync(a, renderer, Vector3.Zero);
                    added += count;
                    Console.WriteLine($"[Fill] {name}: {share:P1} uncovered, painted, {count:N0} splats added");
                }
                finally
                {
                    foreach (var b in scratch) b.Dispose();
                }
            }
        }
        finally
        {
            trainer.ActiveShDegree = degree;
        }
        return added;
    }
}
