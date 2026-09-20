namespace SpawnScene.Services;

/// <summary>
/// WGSL for the differentiable tile rasteriser used by the optimiser.
///
/// These kernels MUST reproduce <see cref="SplatRasterizer"/>, which is finite-difference
/// verified on CPU (SpawnScene.Tests/SplatRasterizerTests.cs). Where a constant appears in both,
/// it is named identically so a drift is visible on inspection.
///
/// This is a SEPARATE path from <see cref="GpuGaussianRenderer"/>'s display renderer, and has to
/// be: hardware alpha blending cannot report per-pixel final transmittance or which splat
/// contributed last, and the backward pass cannot run without both.
///
/// Design constraints that shaped this (see Research/README.md):
///  - WebGPU has NO f32 atomics, so gradient accumulation cannot scatter floats from pixels into
///    splats. The backward kernel is therefore GAUSSIAN-MAJOR: one workgroup per splat gathering
///    from the pixels it touched, needing no cross-thread atomics at all.
///  - 256 invocations per workgroup is the default ceiling, which is exactly a 16x16 tile, and
///    workgroup storage is 16 KB. Neither has headroom.
///  - A single long dispatch can trip TDR and lose the device, so the host submits a few
///    iterations per frame rather than a whole training run.
/// </summary>
internal static class SplatTrainerShaders
{
    /// <summary>Tile edge in pixels. 16x16 = 256 threads = the default workgroup ceiling.</summary>
    public const int TileSize = 16;

    /// <summary>Matches <see cref="SplatRasterizer.MinAlpha"/>.</summary>
    public const float MinAlpha = 1f / 255f;

    /// <summary>Matches <see cref="SplatRasterizer.MaxAlpha"/>.</summary>
    public const float MaxAlpha = 0.99f;

    /// <summary>Matches <see cref="SplatRasterizer.MinTransmittance"/>.</summary>
    public const float MinTransmittance = 1e-4f;

    /// <summary>
    /// Shared declarations: uniforms, the packed splat layout, and the projection.
    /// Concatenated ahead of each kernel so projection exists in exactly one place.
    /// </summary>
    public const string Common = @"
struct TrainUniforms {
    cam_right  : vec4<f32>,   // world-space camera basis; xyz used
    cam_up     : vec4<f32>,
    cam_fwd    : vec4<f32>,
    cam_pos    : vec4<f32>,
    focal      : vec2<f32>,   // fx, fy in pixels
    principal  : vec2<f32>,   // cx, cy in pixels
    viewport   : vec2<f32>,   // width, height in pixels
    tiles      : vec2<u32>,   // tile counts in x and y
    // Depth range of the scene from this view. The sort key quantises depth into 18 bits, so
    // it must be normalised against the ACTUAL range: a fixed scale (depth * 1024) spent only
    // ~400 of 262143 levels on a scene 0.4-0.8 deep, collapsing many splats onto the same key.
    // Ties then composite in whatever order the atomic allocator happened to hand out.
    depth_near : f32,
    depth_far  : f32,
    splat_count: u32,
    _pad0      : u32,
};

@group(0) @binding(0) var<uniform> u : TrainUniforms;

// Packed splat source: SplatFormat.Floats (14) per splat.
//   0..2 pos   3..5 colour   6..8 scale   9 opacity   10..13 quat(x,y,z,w)
@group(0) @binding(1) var<storage, read> splats : array<f32>;

const FLOATS_PER_SPLAT : u32 = 14u;
const TILE : u32 = 16u;
const MIN_ALPHA : f32 = 0.00392156862;   // 1/255, matches SplatRasterizer.MinAlpha
const MAX_ALPHA : f32 = 0.99;            // matches SplatRasterizer.MaxAlpha
const MIN_T : f32 = 1e-4;                // matches SplatRasterizer.MinTransmittance
const SIGMA_CUTOFF : f32 = 3.0;
const EWA_FILTER_PX2 : f32 = 0.3;

struct Projected {
    valid  : bool,
    centre : vec2<f32>,   // pixels
    conic  : vec3<f32>,   // inverse 2D covariance (a, b, c)
    depth  : f32,
    colour : vec3<f32>,
    opacity: f32,
    extent : vec2<f32>,   // half-extent of the 3-sigma footprint, pixels
};

// Project one splat to screen space. Mirrors SplatCovariance.Cov3DFromScaleQuat ->
// RotateToCamera -> ProjectCov2D, which are unit-tested against analytic answers.
fn project(i : u32) -> Projected {
    var p : Projected;
    p.valid = false;

    let o = i * FLOATS_PER_SPLAT;
    let pos = vec3<f32>(splats[o + 0u], splats[o + 1u], splats[o + 2u]);
    p.colour = vec3<f32>(splats[o + 3u], splats[o + 4u], splats[o + 5u]);
    let scale = vec3<f32>(splats[o + 6u], splats[o + 7u], splats[o + 8u]);
    p.opacity = splats[o + 9u];
    let q = normalize(vec4<f32>(splats[o + 10u], splats[o + 11u], splats[o + 12u], splats[o + 13u]));

    // Camera space: x right, y UP, z forward and positive in front.
    let rel = pos - u.cam_pos.xyz;
    let cx = dot(u.cam_right.xyz, rel);
    let cy = dot(u.cam_up.xyz, rel);
    let cz = dot(u.cam_fwd.xyz, rel);
    if (cz <= 1e-6) { return p; }
    p.depth = cz;

    // Sigma_world = R S S^T R^T, columns of R pre-scaled.
    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
    let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;
    let s = max(scale, vec3<f32>(1e-9, 1e-9, 1e-9));
    let m0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)) * s.x;
    let m1 = vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)) * s.y;
    let m2 = vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)) * s.z;
    let M = mat3x3<f32>(m0, m1, m2);
    let sigma_world = M * transpose(M);

    // Into camera space: A rows are (right, up, fwd).
    let A = transpose(mat3x3<f32>(u.cam_right.xyz, u.cam_up.xyz, u.cam_fwd.xyz));
    let sc = A * sigma_world * transpose(A);

    // Perspective Jacobian of (fx*x/z, fy*y/z).
    let invz = 1.0 / cz;
    let invz2 = invz * invz;
    let j00 = u.focal.x * invz;
    let j02 = -u.focal.x * cx * invz2;
    let j11 = u.focal.y * invz;
    let j12 = -u.focal.y * cy * invz2;

    let s00 = sc[0][0]; let s01 = sc[1][0]; let s02 = sc[2][0];
    let s11 = sc[1][1]; let s12 = sc[2][1]; let s22 = sc[2][2];

    let a0 = j00 * s00 + j02 * s02;
    let a1 = j00 * s01 + j02 * s12;
    let a2 = j00 * s02 + j02 * s22;
    let b1 = j11 * s11 + j12 * s12;
    let b2 = j11 * s12 + j12 * s22;

    let cov_a = a0 * j00 + a2 * j02 + EWA_FILTER_PX2;
    let cov_b = a1 * j11 + a2 * j12;
    let cov_c = b1 * j11 + b2 * j12 + EWA_FILTER_PX2;

    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 1e-20) { return p; }
    let inv_det = 1.0 / det;

    // Conic is the INVERSE covariance; the rasteriser consumes it directly.
    p.conic = vec3<f32>(cov_c * inv_det, -cov_b * inv_det, cov_a * inv_det);

    // Screen position. y is measured UP in camera space but DOWN in pixels.
    p.centre = vec2<f32>(
        u.focal.x * cx * invz + u.principal.x,
        u.principal.y - u.focal.y * cy * invz);

    // 3-sigma half-extent from the covariance eigenvalues, used for tile overlap.
    let mid = 0.5 * (cov_a + cov_c);
    let disc = sqrt(max(mid * mid - det, 0.0));
    let l1 = mid + disc;
    let r = SIGMA_CUTOFF * sqrt(max(l1, 1e-20));
    p.extent = vec2<f32>(r, r);

    p.valid = p.opacity > 0.0;
    return p;
}

// Gaussian weight at a pixel centre. Mirrors SplatRasterizer.Weight, including the
// power > 0 rejection - omitting it changes which splats contribute AND the gradient.
fn splat_weight(conic : vec3<f32>, centre : vec2<f32>, pixel : vec2<f32>) -> f32 {
    let d = pixel - centre;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.0) { return 0.0; }
    return exp(power);
}
";

    /// <summary>
    /// Pass 1: emit one (tile, depth) key per overlapped tile.
    ///
    /// Slots are reserved with a single global <c>atomicAdd</c> per splat rather than a
    /// count pass plus a prefix sum. The WebGPU caution about atomics concerns ORDERING
    /// guarantees, and allocation needs none - the keys are sorted immediately afterwards, so
    /// only uniqueness matters, and the previous value returned by atomicAdd gives exactly that.
    /// This removes a kernel and a multi-level scan.
    ///
    /// The caller sizes <c>keys</c>/<c>values</c> for the worst case and checks the final
    /// counter; overflow is reported rather than silently truncating the scene.
    /// </summary>
    public const string EmitKeys = Common + @"
@group(0) @binding(2) var<storage, read_write> keys    : array<u32>;
@group(0) @binding(3) var<storage, read_write> values  : array<u32>;
@group(0) @binding(4) var<storage, read_write> counter : atomic<u32>;
@group(0) @binding(5) var<uniform>             caps    : vec4<u32>;   // x = key capacity

// Depth occupies the low bits; the tile id sits above it. 18 bits of depth leaves 14 for the
// tile, i.e. up to 16384 tiles - 2048x2048 pixels at 16px tiles.
const DEPTH_BITS : u32 = 18u;
const DEPTH_MAX : f32 = 262143.0;

@compute @workgroup_size(64)
fn emit_keys(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= u.splat_count) { return; }

    let p = project(i);
    if (!p.valid) { return; }

    let lo = vec2<i32>(floor((p.centre - p.extent) / f32(TILE)));
    let hi = vec2<i32>(floor((p.centre + p.extent) / f32(TILE)));
    let x0 = max(lo.x, 0);
    let y0 = max(lo.y, 0);
    let x1 = min(hi.x, i32(u.tiles.x) - 1);
    let y1 = min(hi.y, i32(u.tiles.y) - 1);
    if (x1 < x0 || y1 < y0) { return; }

    let n = u32((x1 - x0 + 1) * (y1 - y0 + 1));
    let base = atomicAdd(&counter, n);
    if (base + n > caps.x) { return; }   // overflow; host sees counter > capacity and reports

    // Normalised into the full 18-bit range so distinct depths get distinct keys.
    let span = max(u.depth_far - u.depth_near, 1e-6);
    let dn = clamp((p.depth - u.depth_near) / span, 0.0, 1.0);
    let dq = u32(dn * DEPTH_MAX);

    var slot = base;
    for (var ty = y0; ty <= y1; ty = ty + 1) {
        for (var tx = x0; tx <= x1; tx = tx + 1) {
            let tile = u32(ty) * u.tiles.x + u32(tx);
            keys[slot] = (tile << DEPTH_BITS) | dq;
            values[slot] = i;
            slot = slot + 1u;
        }
    }
}
";

    /// <summary>
    /// Pass 3: find where each tile's run begins and ends in the sorted key array, by looking
    /// for boundaries between adjacent keys.
    /// </summary>
    public const string TileRanges = @"
@group(0) @binding(0) var<storage, read>       keys   : array<u32>;
@group(0) @binding(1) var<storage, read_write> ranges : array<vec2<u32>>;  // start, end per tile
@group(0) @binding(2) var<uniform>             counts : vec4<u32>;         // x = key count

const DEPTH_BITS : u32 = 18u;

@compute @workgroup_size(64)
fn tile_ranges(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= counts.x) { return; }

    let tile = keys[i] >> DEPTH_BITS;
    if (i == 0u) {
        ranges[tile].x = 0u;
    } else {
        let prev = keys[i - 1u] >> DEPTH_BITS;
        if (prev != tile) {
            ranges[prev].y = i;
            ranges[tile].x = i;
        }
    }
    if (i == counts.x - 1u) {
        ranges[tile].y = counts.x;
    }
}
";

    /// <summary>
    /// Pass 4: forward rasterise. One workgroup per tile, one thread per pixel.
    ///
    /// Writes the colour AND the two things the backward pass needs and hardware blending
    /// cannot give: the final transmittance, and how far into the tile's list the forward got
    /// before saturating. Mirrors SplatRasterizer.Render, including the early-out.
    /// </summary>
    public const string RasterForward = Common + @"
@group(0) @binding(2) var<storage, read> ranges : array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> values : array<u32>;
@group(0) @binding(4) var<storage, read_write> out_colour : array<f32>;   // 3 per pixel
@group(0) @binding(5) var<storage, read_write> out_final_t : array<f32>;  // 1 per pixel
@group(0) @binding(6) var<storage, read_write> out_end : array<u32>;      // 1 per pixel

// Staging for one batch of splats, shared by the whole tile. 256 entries x 32 bytes = 8 KB,
// inside the 16 KB workgroup budget with room to spare.
var<workgroup> sh_centre : array<vec2<f32>, 256>;
var<workgroup> sh_conic  : array<vec3<f32>, 256>;
var<workgroup> sh_colour : array<vec3<f32>, 256>;
var<workgroup> sh_opacity: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn raster_forward(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let tile = wg.y * u.tiles.x + wg.x;
    let px = wg.x * TILE + (li % TILE);
    let py = wg.y * TILE + (li / TILE);
    let inside = px < u32(u.viewport.x) && py < u32(u.viewport.y);
    let pixel = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    let range = ranges[tile];
    var t = 1.0;
    var acc = vec3<f32>(0.0);
    var consumed = 0u;
    var done = !inside;

    var base = range.x;
    loop {
        if (base >= range.y) { break; }
        let batch = min(256u, range.y - base);

        // Cooperative load: every thread stages one splat for the whole tile.
        if (li < batch) {
            let p = project(values[base + li]);
            sh_centre[li] = p.centre;
            sh_conic[li] = p.conic;
            sh_colour[li] = p.colour;
            sh_opacity[li] = select(0.0, p.opacity, p.valid);
        }
        workgroupBarrier();

        if (!done) {
            for (var k = 0u; k < batch; k = k + 1u) {
                if (t < MIN_T) { done = true; break; }
                let g = splat_weight(sh_conic[k], sh_centre[k], pixel);
                consumed = consumed + 1u;
                if (g <= 0.0) { continue; }
                let alpha = min(MAX_ALPHA, sh_opacity[k] * g);
                if (alpha < MIN_ALPHA) { continue; }
                acc = acc + sh_colour[k] * alpha * t;
                t = t * (1.0 - alpha);
            }
        }
        workgroupBarrier();
        base = base + batch;
        // NOTE: deliberately no per-thread break here. `done` is per-pixel, so breaking on it
        // would leave some invocations exiting the loop while others reach workgroupBarrier(),
        // and a barrier in non-uniform control flow is undefined behaviour in WGSL - a hang or
        // silently wrong results depending on the driver. The loop bound (`base`, `range.y`) is
        // uniform across the workgroup; saturated pixels just skip the inner work via `done`.
    }

    if (inside) {
        let o = py * u32(u.viewport.x) + px;
        out_colour[o * 3u + 0u] = acc.x;
        out_colour[o * 3u + 1u] = acc.y;
        out_colour[o * 3u + 2u] = acc.z;
        out_final_t[o] = t;
        // Exclusive bound into this tile's list, matching SplatRasterizer.Forward.OrderEnd.
        out_end[o] = range.x + consumed;
    }
}
";

    /// <summary>
    /// Pass 5: backward for COLOUR and OPACITY, one workgroup per tile.
    ///
    /// The whole shape of this kernel is dictated by WebGPU having no f32 atomics, so a
    /// pixel-major scatter (each pixel atomically adding into every splat it touched) is not
    /// merely slow - it is unavailable. Instead:
    ///
    ///  - All 256 threads walk the tile's splat list BACK TO FRONT in LOCKSTEP. A thread whose
    ///    pixel never reached splat k simply contributes zero for it. Lockstep is what makes the
    ///    workgroup reduction below legal, and keeps every barrier in uniform control flow.
    ///  - For each splat the workgroup reduces its 256 per-pixel contributions to one value and
    ///    a single thread writes it to grad_per_key[k]. There is exactly one slot per
    ///    (tile, splat) pair, which is precisely what the sorted key array already enumerates,
    ///    so nothing is shared between workgroups and no atomics are needed here.
    ///  - A separate, much lighter scatter pass folds per-key gradients into per-splat totals.
    ///
    /// Transmittance is recovered the same way as the CPU oracle: start from the pixel's final
    /// T and divide out each splat's alpha walking backwards.
    /// </summary>
    public const string RasterBackward = Common + @"
@group(0) @binding(2) var<storage, read> ranges  : array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> values  : array<u32>;
@group(0) @binding(4) var<storage, read> final_t : array<f32>;   // 1 per pixel, from forward
@group(0) @binding(5) var<storage, read> end_idx : array<u32>;   // 1 per pixel, from forward
@group(0) @binding(6) var<storage, read> dL_dpix : array<f32>;   // 3 per pixel
// 4 per KEY: dL/dR, dL/dG, dL/dB, dL/dopacity. One slot per (tile, splat) pair.
@group(0) @binding(7) var<storage, read_write> grad_per_key : array<f32>;

var<workgroup> red : array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn raster_backward(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let tile = wg.y * u.tiles.x + wg.x;
    let px = wg.x * TILE + (li % TILE);
    let py = wg.y * TILE + (li / TILE);
    let inside = px < u32(u.viewport.x) && py < u32(u.viewport.y);
    let pixel = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    let range = ranges[tile];

    var t = 1.0;
    var my_end = range.x;
    var dL = vec3<f32>(0.0);
    if (inside) {
        let o = py * u32(u.viewport.x) + px;
        t = final_t[o];
        my_end = end_idx[o];
        dL = vec3<f32>(dL_dpix[o * 3u + 0u], dL_dpix[o * 3u + 1u], dL_dpix[o * 3u + 2u]);
    }

    // Colour accumulated by everything BEHIND the splat currently being processed.
    var rec = vec3<f32>(0.0);

    // Back to front, in lockstep. range.x/range.y are uniform, so every barrier below is
    // reached by every invocation.
    var k = range.y;
    loop {
        if (k <= range.x) { break; }
        k = k - 1u;

        var contrib = vec4<f32>(0.0);

        // A thread only participates for splats its own pixel actually reached.
        if (inside && k < my_end) {
            let p = project(values[k]);
            if (p.valid) {
                let g = splat_weight(p.conic, p.centre, pixel);
                if (g > 0.0) {
                    let raw_alpha = p.opacity * g;
                    let alpha = min(MAX_ALPHA, raw_alpha);
                    if (alpha >= MIN_ALPHA) {
                        // Undo this splat to recover the transmittance it rendered against.
                        t = t / (1.0 - alpha);
                        let w = alpha * t;

                        contrib = vec4<f32>(w * dL.x, w * dL.y, w * dL.z, 0.0);

                        let dL_dalpha =
                            (p.colour.x - rec.x) * t * dL.x +
                            (p.colour.y - rec.y) * t * dL.y +
                            (p.colour.z - rec.z) * t * dL.z;

                        // A CLAMPED alpha is constant in opacity, so its derivative is zero.
                        // Dropping this guard produces a phantom gradient that drives opacity
                        // up without bound.
                        if (raw_alpha < MAX_ALPHA) {
                            contrib.w = g * dL_dalpha;
                        }

                        rec = alpha * p.colour + (1.0 - alpha) * rec;
                    }
                }
            }
        }

        // Reduce this splat's contribution across the tile's 256 pixels.
        red[li] = contrib;
        workgroupBarrier();
        var stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (li < stride) { red[li] = red[li] + red[li + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if (li == 0u) {
            grad_per_key[k * 4u + 0u] = red[0].x;
            grad_per_key[k * 4u + 1u] = red[0].y;
            grad_per_key[k * 4u + 2u] = red[0].z;
            grad_per_key[k * 4u + 3u] = red[0].w;
        }
        workgroupBarrier();
    }
}
";

    /// <summary>
    /// Pass 6: fold per-(tile, splat) gradients into per-splat totals.
    ///
    /// This is the only place atomics are needed, and the traffic is one add per KEY rather
    /// than one per (pixel, splat) - smaller by roughly the pixel count of a tile. Floats go
    /// through fixed point because WebGPU only offers integer atomics.
    /// </summary>
    public const string ScatterGradients = @"
@group(0) @binding(0) var<storage, read>       grad_per_key : array<f32>;
@group(0) @binding(1) var<storage, read>       values       : array<u32>;
@group(0) @binding(2) var<storage, read_write> grad_fixed   : array<atomic<i32>>;
@group(0) @binding(3) var<uniform>             counts       : vec4<u32>;   // x = key count

// Gradients here are sums over a tile's pixels of quantities around 1e-4..1e-1. 2^20 keeps
// ~6 decimal digits while leaving headroom before a 32-bit overflow.
const FIXED_SCALE : f32 = 1048576.0;

@compute @workgroup_size(64)
fn scatter_gradients(@builtin(global_invocation_id) gid : vec3<u32>) {
    let k = gid.x;
    if (k >= counts.x) { return; }

    let splat = values[k];
    for (var c = 0u; c < 4u; c = c + 1u) {
        let v = grad_per_key[k * 4u + c];
        if (v != 0.0) {
            atomicAdd(&grad_fixed[splat * 4u + c], i32(round(v * FIXED_SCALE)));
        }
    }
}
";

    /// <summary>
    /// Pass 7: L1 loss and its gradient w.r.t. the rendered image, in one pass.
    ///
    /// The forward's colour buffer is compared against the target image and dL/d(pixel) is
    /// written for the backward kernel. The loss itself is accumulated in fixed point because
    /// WebGPU has no float atomics; it is only a scalar for reporting, so precision there is
    /// not load-bearing.
    /// </summary>
    public const string LossL1 = @"
@group(0) @binding(0) var<storage, read>       rendered : array<f32>;   // 3 per pixel
// 'target' is a RESERVED KEYWORD in WGSL; the binding has to be named something else.
@group(0) @binding(1) var<storage, read>       ref_image : array<f32>;  // 3 per pixel
@group(0) @binding(2) var<storage, read_write> dL_dpix  : array<f32>;   // 3 per pixel
@group(0) @binding(3) var<storage, read_write> loss_fixed : atomic<i32>;
@group(0) @binding(4) var<uniform>             dims     : vec4<u32>;    // x = pixel count

const LOSS_SCALE : f32 = 1048576.0;

@compute @workgroup_size(64)
fn loss_l1(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    if (p >= dims.x) { return; }

    let n = f32(dims.x * 3u);
    var total = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let i = p * 3u + c;
        let d = rendered[i] - ref_image[i];
        total = total + abs(d);
        // d|x|/dx = sign(x), averaged over every channel of every pixel.
        dL_dpix[i] = sign(d) / n;
    }
    atomicAdd(&loss_fixed, i32(round(total / n * LOSS_SCALE)));
}
";

    /// <summary>
    /// Pass 8: Adam on colour and opacity.
    ///
    /// Parameterisation matches the reference implementation, because the published learning
    /// rates mean nothing otherwise: opacity is stored as a LOGIT and read through a sigmoid,
    /// so the incoming gradient is chained by d(sigmoid)/d(logit) = a(1-a). Bias correction is
    /// included; without it the first steps are tiny and read as a wrong learning rate.
    ///
    /// Gradients arrive as fixed-point integers from the scatter pass and are dequantised here.
    /// </summary>
    public const string AdamStep = @"
@group(0) @binding(0) var<storage, read_write> splats     : array<f32>;       // 14 per splat
@group(0) @binding(1) var<storage, read>       grad_fixed : array<i32>;       // 4 per splat
@group(0) @binding(2) var<storage, read_write> opacity_logit : array<f32>;    // 1 per splat
@group(0) @binding(3) var<storage, read_write> adam_m     : array<f32>;       // 4 per splat
@group(0) @binding(4) var<storage, read_write> adam_v     : array<f32>;       // 4 per splat
@group(0) @binding(5) var<uniform>             cfg        : vec4<f32>;        // x=colourLr y=opacityLr z=step w=splatCount

const FLOATS_PER_SPLAT : u32 = 14u;
const FIXED_SCALE : f32 = 1048576.0;
const BETA1 : f32 = 0.9;
const BETA2 : f32 = 0.999;
const EPS : f32 = 1e-15;

fn adam(value : f32, grad : f32, lr : f32, step : f32, m : ptr<function, f32>, v : ptr<function, f32>) -> f32 {
    *m = BETA1 * (*m) + (1.0 - BETA1) * grad;
    *v = BETA2 * (*v) + (1.0 - BETA2) * grad * grad;
    let m_hat = *m / (1.0 - pow(BETA1, step));
    let v_hat = *v / (1.0 - pow(BETA2, step));
    return value - lr * m_hat / (sqrt(v_hat) + EPS);
}

@compute @workgroup_size(64)
fn adam_step(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= u32(cfg.w)) { return; }

    let o = i * FLOATS_PER_SPLAT;
    let step = cfg.z;

    // Colour (SH degree 0 / DC term).
    for (var c = 0u; c < 3u; c = c + 1u) {
        let g = f32(grad_fixed[i * 4u + c]) / FIXED_SCALE;
        var m = adam_m[i * 4u + c];
        var v = adam_v[i * 4u + c];
        let updated = adam(splats[o + 3u + c], g, cfg.x, step, &m, &v);
        adam_m[i * 4u + c] = m;
        adam_v[i * 4u + c] = v;
        splats[o + 3u + c] = clamp(updated, 0.0, 1.0);
    }

    // Opacity, optimised in logit space.
    let a = splats[o + 9u];
    let g_op = f32(grad_fixed[i * 4u + 3u]) / FIXED_SCALE;
    let g_logit = g_op * a * (1.0 - a);
    var m3 = adam_m[i * 4u + 3u];
    var v3 = adam_v[i * 4u + 3u];
    let new_logit = adam(opacity_logit[i], g_logit, cfg.y, step, &m3, &v3);
    adam_m[i * 4u + 3u] = m3;
    adam_v[i * 4u + 3u] = v3;
    opacity_logit[i] = new_logit;
    splats[o + 9u] = 1.0 / (1.0 + exp(-new_logit));
}
";

    /// <summary>Seed the logit buffer from the splats' current opacity, once before training.</summary>
    public const string InitLogits = @"
@group(0) @binding(0) var<storage, read>       splats        : array<f32>;
@group(0) @binding(1) var<storage, read_write> opacity_logit : array<f32>;
@group(0) @binding(2) var<uniform>             cfg           : vec4<u32>;   // x = splat count

const FLOATS_PER_SPLAT : u32 = 14u;

@compute @workgroup_size(64)
fn init_logits(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.x) { return; }
    let a = clamp(splats[i * FLOATS_PER_SPLAT + 9u], 1e-6, 1.0 - 1e-6);
    opacity_logit[i] = log(a / (1.0 - a));
}
";
}
