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
        public float CellStretch;     // longest a cell's disk may grow, in footprints (a parameter: kernels cannot read a mutable static)
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
        // The cell's point and its +x / +y neighbours' (each at its filled depth), relative to the camera: their surface
        // disk (SplatCovariance.SurfaceDiskFromNeighbors, the single-photo layer's) lies ALONG the filled surface and
        // spans to the next cells. x1 made camera-facing disks: from any other viewpoint a ceiling of them was tilted
        // against the surface and striped with gaps.
        int xq = XMath.Min(S - 1, x + p.Stride), yr = XMath.Min(S - 1, y + p.Stride);
        float zq = depth[y * S + xq], zr = depth[yr * S + x];
        if (!(zq > 0f)) zq = z;
        if (!(zr > 0f)) zr = z;
        Ray(p, x, y, out float ax, out float ay, out float az);
        Ray(p, xq, y, out float bx, out float by, out float bz);
        Ray(p, x, yr, out float cx, out float cy, out float cz);
        var disk = SplatCovariance.SurfaceDiskFromNeighbors(ax * z, ay * z, az * z, bx * zq, by * zq, bz * zq, cx * zr, cy * zr, cz * zr);
        int k = Atomic.Add(ref counters[1], 1);
        if (k >= p.Capacity) return;
        int o = k * SplatFormat.Floats;
        rows[o + SplatFormat.OffPos] = p.Px + ax * z;
        rows[o + SplatFormat.OffPos + 1] = p.Py + ay * z;
        rows[o + SplatFormat.OffPos + 2] = p.Pz + az * z;
        for (int c = 0; c < 3; c++) rows[o + SplatFormat.OffColor + c] = painted[c * Px + i] / 255f;
        // A cell's footprint head-on; a receding surface may stretch to CellStretch of it.
        float aa = XMath.Sqrt(ax * ax + ay * ay + az * az);
        float footprint = z * aa * p.Stride * p.W / S / p.Fx;
        float cap = p.CellStretch * footprint;
        float su = XMath.Min(disk.Su, cap), sv = XMath.Min(disk.Sv, cap);
        if (!(su > 1e-6f)) su = footprint;
        if (!(sv > 1e-6f)) sv = footprint;
        rows[o + SplatFormat.OffScale] = su;
        rows[o + SplatFormat.OffScale + 1] = sv;
        rows[o + SplatFormat.OffScale + 2] = XMath.Min(su, sv) * 0.15f;
        rows[o + SplatFormat.OffOpacity] = p.Opacity;
        rows[o + SplatFormat.OffQuat] = disk.Q.X;
        rows[o + SplatFormat.OffQuat + 1] = disk.Q.Y;
        rows[o + SplatFormat.OffQuat + 2] = disk.Q.Z;
        rows[o + SplatFormat.OffQuat + 3] = disk.Q.W;
    }

    /// <summary>
    /// Diagnostic (&amp;filldump=1): one view's [render with the mask in magenta | painted | filled depth | mask] as RGBA,
    /// 4S x S. CPU transfer: a debugging dump, never on the fill's own path.
    /// </summary>
    static async Task<byte[]> DumpStripAsync(MemoryBuffer1D<float, Stride1D.Dense> image, MemoryBuffer1D<float, Stride1D.Dense> mask,
        MemoryBuffer1D<float, Stride1D.Dense> painted, MemoryBuffer1D<float, Stride1D.Dense> depth)
    {
        const int Px = S * S;
        var img = await image.CopyToHostAsync<float>(0, 3L * Px);
        var msk = await mask.CopyToHostAsync<float>(0, Px);
        var pnt = await painted.CopyToHostAsync<float>(0, 3L * Px);
        var dep = await depth.CopyToHostAsync<float>(0, Px);
        float zMax = 0f;
        foreach (var z in dep) if (z > zMax && float.IsFinite(z)) zMax = z;
        var rgba = new byte[4 * S * S * 4];
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
            {
                int i = y * S + x;
                bool known = msk[i] > 127f;
                for (int panel = 0; panel < 4; panel++)
                {
                    int o = ((y * 4 * S) + panel * S + x) * 4;
                    float r, g, b;
                    if (panel == 0) { r = known ? img[i] : 255f; g = known ? img[Px + i] : 0f; b = known ? img[2 * Px + i] : 255f; }
                    else if (panel == 1) { r = pnt[i]; g = pnt[Px + i]; b = pnt[2 * Px + i]; }
                    else if (panel == 2) { float v = zMax > 0f ? 255f * dep[i] / zMax : 0f; r = g = b = v; }
                    else { r = g = b = known ? 255f : 0f; }
                    rgba[o] = (byte)Math.Clamp(r, 0f, 255f); rgba[o + 1] = (byte)Math.Clamp(g, 0f, 255f);
                    rgba[o + 2] = (byte)Math.Clamp(b, 0f, 255f); rgba[o + 3] = 255;
                }
            }
        return rgba;
    }

    /// <summary>The (unnormalised, forward-component 1) world-space ray of grid cell (x, y)'s centre: depth z along it is
    /// the camera-space depth z.</summary>
    static void Ray(Params p, int x, int y, out float dx, out float dy, out float dz)
    {
        float u = (x + 0.5f) * p.W / S, v = (y + 0.5f) * p.H / S;
        float a = (u - p.Cx) / p.Fx, b = (v - p.Cy) / p.Fy;
        dx = p.Fwx + a * p.Rx - b * p.Ux;
        dy = p.Fwy + a * p.Ry - b * p.Uy;
        dz = p.Fwz + a * p.Rz - b * p.Uz;
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
    /// Fill the scene on <paramref name="renderer"/> around <paramref name="centre"/>: 8 headings 45 degrees apart at
    /// pitches 0 / +40 / -40 (60 degree horizontal field) plus straight up and down, rendered by <paramref name="trainer"/>
    /// at its size.
    /// Returns the splats added.
    /// </summary>
    public static async Task<int> FillAsync(Accelerator a, SplatTrainerGpu trainer, GpuGaussianRenderer renderer,
        HiddenLayerInpaint inpaint, Vector3 centre, Vector3 sceneUp, int stride = 2, float known = 0.7f, float minMasked = 0.01f,
        Func<string, byte[], int, int, Task>? dump = null)
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
        // 8 headings at three pitches (0, +40, -40 degrees) plus straight up and down. Level views alone (60 degrees
        // across, ~47 up and down at 4:3) left the band from ~23 to ~60 degrees above the horizon to no view at all:
        // x2's ceiling kept a magenta stripe there.
        var views = new List<(string Name, Vector3 Fwd, Vector3 Up)>();
        foreach (int pitch in new[] { 0, 40, -40 })
            for (int k = 0; k < 8; k++)
            {
                float t = k * MathF.PI / 4f, pr = pitch * MathF.PI / 180f;
                var heading = MathF.Cos(t) * side + MathF.Sin(t) * side2;
                views.Add(($"h{k * 45}p{pitch}", MathF.Cos(pr) * heading + MathF.Sin(pr) * up, up));
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
                    // 3 footprints, not the single-photo layer's 8: seen from the centre the far ceiling is grazing, and
                    // 8 turned its cells into long sawtooth sheets (x2).
                    Stride = stride, Opacity = 0.95f, CellStretch = 3f,
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
                    if (dump != null) await dump($"fill-{name}", await DumpStripAsync(image, mask, painted, depth), 4 * S, S);
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
