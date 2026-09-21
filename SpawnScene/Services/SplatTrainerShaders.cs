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
    /// <summary>
    /// Camera and viewport, shared by every pass. Split out of <see cref="Common"/> because the
    /// optimiser kernels need the same uniforms with a WRITABLE splat buffer, and a binding
    /// cannot be declared twice.
    /// </summary>
    public const string UniformsBlock = @"
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

const FLOATS_PER_SPLAT : u32 = 14u;
// Gradient slots accumulated per splat: 3 colour, 1 opacity, 2 screen centre, 3 conic.
// Must match SplatTileRasterizer.GradsPerKey.
const GRADS_PER_SPLAT : u32 = 9u;
// Adam moment slots per splat: 3 colour, 1 opacity logit, 3 position, 3 log-scale, 4 quaternion.
const ADAM_SLOTS : u32 = 14u;
const ADAM_POS : u32 = 4u;
const ADAM_SCALE : u32 = 7u;
const ADAM_QUAT : u32 = 10u;
// Gradients are summed across tiles through integer atomics; WebGPU has no float ones, so
// every gradient crosses this boundary as a scaled i32 and the scale sets BOTH the range and
// the precision. One scale cannot serve all nine slots, because their bounds differ by orders
// of magnitude:
//
//   colour      |dL/dc| <= 1/3          - sum of |dL/d(pixel)| over the image is exactly 1/3
//   opacity     same order
//   screen xy   ~ 85*sigma/(W*H)        - about 0.03 for a 100px splat
//   conic       ~ 127*sigma^4/(W*H)     - about 4e4 for the same splat: grows with AREA
//
// So the conic keeps the coarse scale (it is the only one that can get large) and everything
// else gets 64x finer. That matters: dL/d(pixel) is 1/(3*W*H) ~ 1e-6, and at the coarse scale a
// small splat's POSITION gradient was 1.1 quanta - half of it was rounding, and only 1.4% of
// splats got a non-zero one at all.
// The scales are UNIFORMS now, not constants - see SplatTrainerGpu.GradientScales.
//
// These two values were consts, 2^20 and 2^26, and the comment above them claimed they left
// about 100x headroom over the largest value measured on a real scene. That was true of the only
// scenes that existed when it was written. MEASURED on drjohnson, a 14.5-unit room against
// Bathroom's 3.8:
//
//   centre |grad| mean 1.84E-08  ->  ONE quantum of 32. Position is rounded away entirely.
//   conic max      1.98E+03      ->  2.07e9 of 2.147e9, 96% of the i32 WRAP point.
//
// Wrong in opposite directions at the same time, on the same scene, and the atomic wraps
// rather than clamping so the conic ones are wrong rather than coarse. A scale chosen for one
// scene size cannot serve another: the conic gradient grows with a splat's screen AREA.
//
// fixed_scale_for() is gone with them. It was dead in every shader that included this block.
const EWA_FILTER_PX2 : f32 = 0.3;
";

    public const string Common = UniformsBlock + @"
// Packed splat source: SplatFormat.Floats (14) per splat.
//   0..2 pos   3..5 colour   6..8 scale   9 opacity   10..13 quat(x,y,z,w)
@group(0) @binding(1) var<storage, read> splats : array<f32>;

const TILE : u32 = 16u;
const MIN_ALPHA : f32 = 0.00392156862;   // 1/255, matches SplatRasterizer.MinAlpha
const MAX_ALPHA : f32 = 0.99;            // matches SplatRasterizer.MaxAlpha
const MIN_T : f32 = 1e-4;                // matches SplatRasterizer.MinTransmittance
const SIGMA_CUTOFF : f32 = 3.0;

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

// 2D workgroup grid - see the note on tile_ranges.
@compute @workgroup_size(64)
fn emit_keys(
    @builtin(workgroup_id) wg_ : vec3<u32>,
    @builtin(num_workgroups) nwg_ : vec3<u32>,
    @builtin(local_invocation_index) li_ : u32
) {
    let gid = vec3<u32>((wg_.y * nwg_.x + wg_.x) * 64u + li_, 0u, 0u);
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

// The index comes from a 2D workgroup grid, not from global_invocation_id.x.
// maxComputeWorkgroupsPerDimension is 65535, so anything dispatched one-dimensionally caps out
// at 4.19M threads at 64 per group - and a 580k-splat room emits 7.5M keys. Going over does
// not clamp; the dispatch is rejected and the error surfaces later, from some other pass.
@compute @workgroup_size(64)
fn tile_ranges(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(num_workgroups) nwg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let i = (wg.y * nwg.x + wg.x) * 64u + li;
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
// Nine gradients per KEY, split across THREE bindings of three.
//
// Not cosmetic: maxStorageBufferBindingSize is guaranteed to be only 128 MiB, and it applies
// per BINDING rather than per dispatch. As one buffer at 36 bytes a key the ceiling was about
// 3.7M keys, and a 580k-splat room wants 13 keys each - so it overflowed and lost gradients.
// Three bindings of 12 bytes each raise the same ceiling to about 11M.
@group(0) @binding(7) var<storage, read_write> grad_a : array<f32>;   // dR, dG, dB
@group(0) @binding(8) var<storage, read_write> grad_b : array<f32>;   // dOpacity, dCentre.x, dCentre.y
@group(0) @binding(9) var<storage, read_write> grad_c : array<f32>;   // dConic a, b, c

// Three tile-wide reductions, 12 KB of workgroup storage against a 16 KB guaranteed minimum.
// One pass rather than three sequential ones: the space is affordable and tripling the barrier
// count in the innermost loop is not.
var<workgroup> redA : array<vec4<f32>, 256>;   // dR, dG, dB, dOpacity
var<workgroup> redB : array<vec4<f32>, 256>;   // dCentreX, dCentreY, dConicA, dConicB
var<workgroup> redC : array<f32, 256>;         // dConicC

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
        var geom = vec4<f32>(0.0);
        var geom_cc = 0.0;

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

                            // Geometry, at the 2D level. The chain on to position, scale and
                            // rotation is linear in these and depends only on the splat and the
                            // view, so it runs ONCE PER SPLAT after the scatter - not per pixel.
                            // Mirrors SplatRasterizer.Backward, which is finite-difference
                            // verified, and SplatTileRasterizer.Backward, which the GPU gate
                            // compares against.
                            let d = pixel - p.centre;
                            let dL_dpower = p.opacity * dL_dalpha * g;
                            geom = vec4<f32>(
                                dL_dpower * (p.conic.x * d.x + p.conic.y * d.y),
                                dL_dpower * (p.conic.z * d.y + p.conic.y * d.x),
                                dL_dpower * (-0.5 * d.x * d.x),
                                dL_dpower * (-d.x * d.y));
                            geom_cc = dL_dpower * (-0.5 * d.y * d.y);
                        }

                        rec = alpha * p.colour + (1.0 - alpha) * rec;
                    }
                }
            }
        }

        // Reduce this contribution across the tile's 256 pixels.
        redA[li] = contrib;
        redB[li] = geom;
        redC[li] = geom_cc;
        workgroupBarrier();
        var stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (li < stride) {
                redA[li] = redA[li] + redA[li + stride];
                redB[li] = redB[li] + redB[li + stride];
                redC[li] = redC[li] + redC[li + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if (li == 0u) {
            let b3 = k * 3u;
            grad_a[b3 + 0u] = redA[0].x;
            grad_a[b3 + 1u] = redA[0].y;
            grad_a[b3 + 2u] = redA[0].z;
            grad_b[b3 + 0u] = redA[0].w;
            grad_b[b3 + 1u] = redB[0].x;
            grad_b[b3 + 2u] = redB[0].y;
            grad_c[b3 + 0u] = redB[0].z;
            grad_c[b3 + 1u] = redB[0].w;
            grad_c[b3 + 2u] = redC[0];
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
@group(0) @binding(0) var<storage, read>       grad_a     : array<f32>;   // dR, dG, dB
@group(0) @binding(1) var<storage, read>       grad_b     : array<f32>;   // dOpacity, dCentre.xy
@group(0) @binding(2) var<storage, read>       grad_c     : array<f32>;   // dConic a, b, c
@group(0) @binding(3) var<storage, read>       values     : array<u32>;
@group(0) @binding(4) var<storage, read_write> grad_fixed : array<atomic<i32>>;
@group(0) @binding(5) var<uniform>             counts     : vec4<u32>;   // x = key count
@group(0) @binding(6) var<uniform>             grad_scale : vec4<f32>;  // x=colour/opacity 0..3, y=centre 4..5, z=conic 6..8

// Gradients here are sums over a tile's pixels of quantities around 1e-4..1e-1. 2^20 keeps
// ~6 decimal digits while leaving headroom before a 32-bit overflow.
const GRADS_PER_SPLAT : u32 = 9u;

// 2D workgroup grid - see the note on tile_ranges.
@compute @workgroup_size(64)
fn scatter_gradients(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(num_workgroups) nwg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let k = (wg.y * nwg.x + wg.x) * 64u + li;
    if (k >= counts.x) { return; }

    let splat = values[k];
    let b3 = k * 3u;
    let out = splat * GRADS_PER_SPLAT;

    // Slots 0..5 are bounded small and get 64x the precision; only the conic grows with a
    // splat's pixel area, so only the conic needs the coarse range.
    for (var c = 0u; c < 3u; c = c + 1u) {
        let va = grad_a[b3 + c];
        if (va != 0.0) { atomicAdd(&grad_fixed[out + c], i32(round(va * grad_scale.x))); }

        let vb = grad_b[b3 + c];
        // Slot 3 is opacity and shares the colour scale; slots 4 and 5 are the screen centre and
        // do not - the centre is orders of magnitude smaller, which is the whole reason these
        // are separate now.
        let sb = select(grad_scale.y, grad_scale.x, c == 0u);
        if (vb != 0.0) { atomicAdd(&grad_fixed[out + 3u + c], i32(round(vb * sb))); }

        let vc = grad_c[b3 + c];
        if (vc != 0.0) { atomicAdd(&grad_fixed[out + 6u + c], i32(round(vc * grad_scale.z))); }
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
@group(0) @binding(1) var<storage, read>       grad_fixed : array<i32>;       // 9 per splat
@group(0) @binding(2) var<storage, read_write> opacity_logit : array<f32>;    // 1 per splat
@group(0) @binding(3) var<storage, read_write> adam_m     : array<f32>;       // 14 per splat
@group(0) @binding(4) var<storage, read_write> adam_v     : array<f32>;       // 14 per splat
@group(0) @binding(5) var<uniform>             cfg        : vec4<f32>;        // x=colourLr y=opacityLr z=step w=splatCount
@group(0) @binding(6) var<uniform>             flags      : vec4<f32>;        // x=skip zero-gradient splats
@group(0) @binding(7) var<uniform>             grad_scale : vec4<f32>;        // x=colour/opacity 0..3

const FLOATS_PER_SPLAT : u32 = 14u;
const GRADS_PER_SPLAT : u32 = 9u;
const ADAM_SLOTS : u32 = 14u;
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

    // A splat this view never touched has no gradient. Stepping it anyway lets stale momentum
    // drag a parameter that nothing is currently constraining - adam_geometry has always
    // refused to do that for position, and this is the same argument for colour and opacity.
    //
    // It matters more at batch size 1 than the original 'harmless for colour' allowed: a splat
    // visible in one view of 26 takes about 25 zero-gradient steps per cycle, and opacity is
    // optimised in LOGIT space, so stale momentum walks splats in and out of visibility.
    //
    // Behind a flag because its effect on the held-out curve is a measurement, not an
    // assumption. flags.x != 0 enables it.
    if (flags.x != 0.0) {
        let gz0 = grad_fixed[i * GRADS_PER_SPLAT];
        let gz1 = grad_fixed[i * GRADS_PER_SPLAT + 1u];
        let gz2 = grad_fixed[i * GRADS_PER_SPLAT + 2u];
        let gz3 = grad_fixed[i * GRADS_PER_SPLAT + 3u];
        if (gz0 == 0 && gz1 == 0 && gz2 == 0 && gz3 == 0) { return; }
    }

    // Colour (SH degree 0 / DC term).
    for (var c = 0u; c < 3u; c = c + 1u) {
        let g = f32(grad_fixed[i * GRADS_PER_SPLAT + c]) / grad_scale.x;
        var m = adam_m[i * ADAM_SLOTS + c];
        var v = adam_v[i * ADAM_SLOTS + c];
        let updated = adam(splats[o + 3u + c], g, cfg.x, step, &m, &v);
        adam_m[i * ADAM_SLOTS + c] = m;
        adam_v[i * ADAM_SLOTS + c] = v;
        splats[o + 3u + c] = clamp(updated, 0.0, 1.0);
    }

    // Opacity, optimised in logit space.
    let a = splats[o + 9u];
    let g_op = f32(grad_fixed[i * GRADS_PER_SPLAT + 3u]) / grad_scale.x;
    let g_logit = g_op * a * (1.0 - a);
    var m3 = adam_m[i * ADAM_SLOTS + 3u];
    var v3 = adam_v[i * ADAM_SLOTS + 3u];
    let new_logit = adam(opacity_logit[i], g_logit, cfg.y, step, &m3, &v3);
    adam_m[i * ADAM_SLOTS + 3u] = m3;
    adam_v[i * ADAM_SLOTS + 3u] = v3;
    opacity_logit[i] = new_logit;
    splats[o + 9u] = 1.0 / (1.0 + exp(-new_logit));
}
";

    /// <summary>
    /// Sum of squared error between the render and one target, for PSNR.
    ///
    /// The evaluation used to read BOTH the rendered image and the target back to the host and
    /// difference them in a loop: about 650 MB of GPU-to-CPU traffic on a 35-view capture, to
    /// produce one number per view. This reduces on the GPU and returns one partial sum per
    /// workgroup - a few KB - which the host adds up.
    ///
    /// Partial sums in f32, not a fixed-point atomic: a per-pixel squared error is around 1e-4
    /// and there are a million of them, so any single shared scale either overflows on the
    /// total or quantises every term to zero. Summing per workgroup first keeps both ends of
    /// that range representable.
    /// </summary>
    public const string EvalSse = @"
@group(0) @binding(0) var<storage, read>       rendered  : array<f32>;   // 3 per pixel
@group(0) @binding(1) var<storage, read>       ref_image : array<f32>;   // the whole target stack
@group(0) @binding(2) var<storage, read_write> partials  : array<f32>;   // 1 per workgroup
@group(0) @binding(3) var<uniform>             dims      : vec4<u32>;    // x=pixels, y=target float offset

var<workgroup> acc : array<f32, 256>;

@compute @workgroup_size(256)
fn eval_sse(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let p = gid.x;
    var e = 0.0;
    if (p < dims.x) {
        for (var c = 0u; c < 3u; c = c + 1u) {
            let i = p * 3u + c;
            let d = rendered[i] - ref_image[dims.y + i];
            e = e + d * d;
        }
    }
    acc[li] = e;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) { acc[li] = acc[li] + acc[li + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (li == 0u) { partials[wg.x] = acc[0]; }
}
";

    /// <summary>
    /// Gradient health over EVERY splat, reduced to a few KB.
    ///
    /// This replaces a readback. The probe used to copy the whole accumulator to the host -
    /// 725k splats is 6.5M ints, 26 MB - to report what FRACTION of splats have a gradient, and
    /// when that was cut to a 65,536-splat prefix it started reading zeros: the merged splat
    /// buffer is VIEW-MAJOR, so a prefix is the top ~9% of view 0's depth map in raster order,
    /// which can legitimately be all zero while the buffer is full. A prefix of a view-major
    /// buffer is not a sample. A reduction needs the same amount of code, covers every splat,
    /// and moves 6 KB instead of 26 MB.
    ///
    /// Fixed 256 workgroups with a grid-stride loop, so the readback size does not depend on the
    /// scene. Values stay in QUANTA - the raw fixed-point integers - because the fixed-point
    /// scales live in the host and this shader should not hold a second copy of them.
    ///
    /// ⚠ Slots 0-3 are SUMS and slots 4-5 are MAXES. Two reduction operators share one array,
    /// which is a foot-gun; the read side says so too.
    /// </summary>
    public const string GradStats = @"
@group(0) @binding(0) var<storage, read>       grad_fixed : array<i32>;   // 9 per splat
@group(0) @binding(1) var<storage, read_write> partials   : array<f32>;   // 7 per workgroup
@group(0) @binding(2) var<uniform>             dims       : vec4<u32>;    // x = splat count

const SLOTS : u32 = 7u;
const GRADS_PER_SPLAT : u32 = 9u;
const THREADS : u32 = 65536u;   // 256 workgroups x 256

var<workgroup> acc : array<f32, 1792>;   // 256 threads x 7 slots

@compute @workgroup_size(256)
fn grad_stats(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    var colourLive = 0.0;
    var centreLive = 0.0;
    var conicLive = 0.0;
    var sumCentre = 0.0;
    var maxCentre = 0.0;
    var maxConic = 0.0;
    var maxColour = 0.0;

    for (var i = gid.x; i < dims.x; i = i + THREADS) {
        let b = i * GRADS_PER_SPLAT;

        // Slots 0..3 are colour and opacity. Their magnitude matters as much as their presence:
        // they share a scale with the screen centre, and fitting that scale to the CENTRE alone
        // put colour past the i32 wrap - measured, by doing exactly that.
        let col = f32(max(max(abs(grad_fixed[b]), abs(grad_fixed[b + 1u])),
                      max(abs(grad_fixed[b + 2u]), abs(grad_fixed[b + 3u]))));
        if (col > 0.0) { colourLive = colourLive + 1.0; }
        maxColour = max(maxColour, col);

        let cen = f32(max(abs(grad_fixed[b + 4u]), abs(grad_fixed[b + 5u])));
        if (cen > 0.0) { centreLive = centreLive + 1.0; sumCentre = sumCentre + cen; }
        maxCentre = max(maxCentre, cen);

        let con = f32(max(abs(grad_fixed[b + 6u]),
                      max(abs(grad_fixed[b + 7u]), abs(grad_fixed[b + 8u]))));
        if (con > 0.0) { conicLive = conicLive + 1.0; }
        maxConic = max(maxConic, con);
    }

    let o = li * SLOTS;
    acc[o] = colourLive; acc[o + 1u] = centreLive; acc[o + 2u] = conicLive;
    acc[o + 3u] = sumCentre; acc[o + 4u] = maxCentre; acc[o + 5u] = maxConic;
    acc[o + 6u] = maxColour;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) {
            let a = li * SLOTS;
            let b2 = (li + stride) * SLOTS;
            acc[a] = acc[a] + acc[b2];
            acc[a + 1u] = acc[a + 1u] + acc[b2 + 1u];
            acc[a + 2u] = acc[a + 2u] + acc[b2 + 2u];
            acc[a + 3u] = acc[a + 3u] + acc[b2 + 3u];
            acc[a + 4u] = max(acc[a + 4u], acc[b2 + 4u]);
            acc[a + 5u] = max(acc[a + 5u], acc[b2 + 5u]);
            acc[a + 6u] = max(acc[a + 6u], acc[b2 + 6u]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (li == 0u) {
        let d = wg.x * SLOTS;
        partials[d] = acc[0]; partials[d + 1u] = acc[1]; partials[d + 2u] = acc[2];
        partials[d + 3u] = acc[3]; partials[d + 4u] = acc[4]; partials[d + 5u] = acc[5];
        partials[d + 6u] = acc[6];
    }
}
";

    /// <summary>
    /// Accumulate the per-splat screen-position gradient that adaptive density control needs.
    ///
    /// The signal Kerbl et al. densify on: a Gaussian the loss keeps trying to drag across the
    /// image is being asked to explain more than one thing, and the answer is to clone or split
    /// it rather than keep moving it. Averaged over the iterations the splat was VISIBLE, not
    /// over all iterations - otherwise a splat seen in one view of thirty looks thirty times
    /// less urgent than an identical one seen in all of them, purely because of camera placement.
    ///
    /// Converted to NDC here, where the frame size is known, because the published threshold of
    /// 2e-4 is in NDC and a pixel-space gradient is about 320x smaller at 640 wide.
    /// </summary>
    public const string DensifyAccum = @"
@group(0) @binding(0) var<storage, read>       grad_fixed : array<i32>;   // 9 per splat
@group(0) @binding(1) var<storage, read_write> accum      : array<f32>;   // 2 per splat
@group(0) @binding(2) var<uniform>             dims       : vec4<u32>;    // x=count y=w z=h
@group(0) @binding(3) var<uniform>             grad_scale : vec4<f32>;    // y = centre scale

const GRADS_PER_SPLAT : u32 = 9u;

@compute @workgroup_size(256)
fn densify_accum(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }

    let b = i * GRADS_PER_SPLAT;

    var live = false;
    for (var c = 0u; c < GRADS_PER_SPLAT; c = c + 1u) {
        if (grad_fixed[b + c] != 0) { live = true; }
    }
    if (!live) { return; }

    // Slots 4 and 5 are the screen centre, in PIXELS, stored at the centre scale.
    let gx = f32(grad_fixed[b + 4u]) / grad_scale.y;
    let gy = f32(grad_fixed[b + 5u]) / grad_scale.y;

    let nx = gx * f32(dims.y) * 0.5;
    let ny = gy * f32(dims.z) * 0.5;

    accum[i * 2u] = accum[i * 2u] + sqrt(nx * nx + ny * ny);
    accum[i * 2u + 1u] = accum[i * 2u + 1u] + 1.0;
}
";

    /// <summary>
    /// Count, per splat, how many DISTINCT views have ever given it a gradient.
    ///
    /// This is the question underneath every optimiser experiment on this project. The
    /// initialisation unprojects a monocular depth map PER VIEW, so 44 views give 44 private
    /// depth shells stacked in one world. If a splat is only ever constrained by the single
    /// view it came from, then training is not a reconstruction at all - it is 44 independent
    /// per-view fits sharing a buffer, every one of which can lower its own view's loss while
    /// explaining nothing about a view nobody trained on. Supervised PSNR would rise, held-out
    /// PSNR would fall, and no learning rate, momentum guard or loss term could change that,
    /// because the parameterisation itself is degenerate.
    ///
    /// One thread per splat, run after each step of a full cycle. A gradient of exactly zero
    /// means the splat did not affect that view's render; anything else means it did.
    /// </summary>
    public const string AccumulateSupport = @"
@group(0) @binding(0) var<storage, read>       grad_fixed : array<i32>;   // 9 per splat
@group(0) @binding(1) var<storage, read_write> support    : array<u32>;   // 1 per splat
@group(0) @binding(2) var<uniform>             dims       : vec4<u32>;    // x = splat count

const GRADS_PER_SPLAT : u32 = 9u;

@compute @workgroup_size(256)
fn accumulate_support(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }

    let b = i * GRADS_PER_SPLAT;
    var live = false;
    for (var c = 0u; c < GRADS_PER_SPLAT; c = c + 1u) {
        if (grad_fixed[b + c] != 0) { live = true; }
    }

    // One increment per CALL, and the caller calls once per view, so this counts views rather
    // than steps. No atomic: one thread owns one splat.
    if (live) { support[i] = support[i] + 1u; }
}
";

    /// <summary>
    /// Histogram the per-splat view-support counts into buckets the host can print.
    ///
    /// Every slot is a SUM - unlike grad_stats, which mixes sums and maxes - so the tree
    /// reduction below has exactly one operator and needs no warning.
    /// </summary>
    public const string SupportHistogram = @"
@group(0) @binding(0) var<storage, read>       support  : array<u32>;   // 1 per splat
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;   // 6 per workgroup
@group(0) @binding(2) var<uniform>             dims     : vec4<u32>;    // x = splat count

const SLOTS : u32 = 6u;
const THREADS : u32 = 65536u;   // 256 workgroups x 256

var<workgroup> acc : array<f32, 1536>;   // 256 threads x 6 slots

@compute @workgroup_size(256)
fn support_histogram(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    // Buckets: never constrained, one view only, two, three, four or more; plus the total so
    // the host can report a mean without a second pass.
    var c0 = 0.0; var c1 = 0.0; var c2 = 0.0; var c3 = 0.0; var c4 = 0.0; var total = 0.0;

    for (var i = gid.x; i < dims.x; i = i + THREADS) {
        let v = support[i];
        total = total + f32(v);
        if (v == 0u) { c0 = c0 + 1.0; }
        else if (v == 1u) { c1 = c1 + 1.0; }
        else if (v == 2u) { c2 = c2 + 1.0; }
        else if (v == 3u) { c3 = c3 + 1.0; }
        else { c4 = c4 + 1.0; }
    }

    let o = li * SLOTS;
    acc[o] = c0; acc[o + 1u] = c1; acc[o + 2u] = c2;
    acc[o + 3u] = c3; acc[o + 4u] = c4; acc[o + 5u] = total;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) {
            let a = li * SLOTS;
            let b2 = (li + stride) * SLOTS;
            for (var k = 0u; k < SLOTS; k = k + 1u) {
                acc[a + k] = acc[a + k] + acc[b2 + k];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (li == 0u) {
        let d = wg.x * SLOTS;
        for (var k = 0u; k < SLOTS; k = k + 1u) { partials[d + k] = acc[k]; }
    }
}
";

    /// <summary>
    /// SSIM, horizontal half. One thread per (window column, source row); writes the five
    /// filtered channels a window needs: a, b, a*a, b*b, a*b.
    ///
    /// Separable because the 2-D window is an outer product of a NORMALISED 1-D Gaussian
    /// (see <c>ImageQuality.GaussianKernel1D</c>). Fusing it into one 121-tap pass would read
    /// about 1.1 GB per view at 720x540 against roughly 100 MB this way.
    ///
    /// ⭐ This shader contains no SSIM CONSTANTS. The luma coefficients, the stabilisers and all
    /// eleven Gaussian taps arrive in <c>cfg</c>, written by the host from
    /// <c>ImageQuality</c>. The one duplicated value is WINDOW, which sizes the loop, and the
    /// host asserts it against <c>ImageQuality.WindowSize</c>. A GPU kernel that keeps its own
    /// copy of a rule its CPU oracle also implements WILL drift - that already happened once in
    /// this codebase, to the consistency screen.
    /// </summary>
    public const string SsimRows = @"
struct SsimCfg {
    luma   : vec4<f32>,            // Rec.601 R, G, B, unused
    consts : vec4<f32>,            // C1, C2, unused, unused
    w      : array<vec4<f32>, 3>,  // 11 Gaussian taps, padded to 12
};

@group(0) @binding(0) var<storage, read>       rendered  : array<f32>;   // 3 per pixel
@group(0) @binding(1) var<storage, read>       ref_image : array<f32>;   // the whole target stack
@group(0) @binding(2) var<storage, read_write> rows      : array<f32>;   // 5 per (x, y)
@group(0) @binding(3) var<uniform>             dims      : vec4<u32>;    // wx, wy, srcW, target float offset
@group(0) @binding(4) var<uniform>             cfg       : SsimCfg;

const WINDOW : u32 = 11u;

// Explicit rather than cfg.w[t/4u][t%4u]: dynamic indexing INTO a vector is the part of WGSL
// most likely to be refused by a driver, and this loop is eleven iterations.
fn weight(t : u32) -> f32 {
    let v = cfg.w[t / 4u];
    let m = t % 4u;
    if (m == 0u) { return v.x; }
    if (m == 1u) { return v.y; }
    if (m == 2u) { return v.z; }
    return v.w;
}

fn luma_at(base : u32) -> f32 {
    return rendered[base] * cfg.luma.x
         + rendered[base + 1u] * cfg.luma.y
         + rendered[base + 2u] * cfg.luma.z;
}

fn luma_ref(base : u32) -> f32 {
    return ref_image[base] * cfg.luma.x
         + ref_image[base + 1u] * cfg.luma.y
         + ref_image[base + 2u] * cfg.luma.z;
}

@compute @workgroup_size(64)
fn ssim_rows(@builtin(global_invocation_id) gid : vec3<u32>) {
    let wx = dims.x;
    let srcH = dims.y + WINDOW - 1u;
    let i = gid.x;
    if (i >= wx * srcH) { return; }

    let y = i / wx;
    let x = i - y * wx;

    var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0;
    for (var t = 0u; t < WINDOW; t = t + 1u) {
        let wt = weight(t);
        let p = (y * dims.z + x + t) * 3u;
        let va = luma_at(p);
        let vb = luma_ref(dims.w + p);
        sa = sa + wt * va;
        sb = sb + wt * vb;
        saa = saa + wt * va * va;
        sbb = sbb + wt * vb * vb;
        sab = sab + wt * va * vb;
    }

    let o = i * 5u;
    rows[o] = sa; rows[o + 1u] = sb; rows[o + 2u] = saa;
    rows[o + 3u] = sbb; rows[o + 4u] = sab;
}
";

    /// <summary>
    /// SSIM, vertical half plus the formula, reduced per workgroup exactly as <c>eval_sse</c>
    /// does so the host reads a few KB rather than a per-window image.
    ///
    /// The variances here are the BIASED ones, <c>E[a^2] - E[a]^2</c>, and on a flat window that
    /// can come out very slightly negative through cancellation. The reference implementation in
    /// <c>tools/score_novel_view.py</c> does not clamp it, so this must not either: a
    /// <c>max(sa, 0)</c> would be a silent disagreement with the definition rather than a
    /// safety measure.
    /// </summary>
    public const string SsimReduce = @"
struct SsimCfg {
    luma   : vec4<f32>,
    consts : vec4<f32>,
    w      : array<vec4<f32>, 3>,
};

@group(0) @binding(0) var<storage, read>       rows     : array<f32>;   // 5 per (x, y)
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;   // 1 per workgroup
@group(0) @binding(2) var<uniform>             dims     : vec4<u32>;    // wx, wy, srcW, target float offset
@group(0) @binding(3) var<uniform>             cfg      : SsimCfg;

const WINDOW : u32 = 11u;

var<workgroup> acc : array<f32, 256>;

fn weight(t : u32) -> f32 {
    let v = cfg.w[t / 4u];
    let m = t % 4u;
    if (m == 0u) { return v.x; }
    if (m == 1u) { return v.y; }
    if (m == 2u) { return v.z; }
    return v.w;
}

@compute @workgroup_size(256)
fn ssim_reduce(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) li : u32
) {
    let wx = dims.x;
    let wy = dims.y;
    let j = gid.x;

    var v = 0.0;
    if (j < wx * wy) {
        let y = j / wx;
        let x = j - y * wx;

        var m0 = 0.0; var m1 = 0.0; var m2 = 0.0; var m3 = 0.0; var m4 = 0.0;
        for (var t = 0u; t < WINDOW; t = t + 1u) {
            let wt = weight(t);
            let o = ((y + t) * wx + x) * 5u;
            m0 = m0 + wt * rows[o];
            m1 = m1 + wt * rows[o + 1u];
            m2 = m2 + wt * rows[o + 2u];
            m3 = m3 + wt * rows[o + 3u];
            m4 = m4 + wt * rows[o + 4u];
        }

        let mu_a = m0;
        let mu_b = m1;
        let sa = m2 - mu_a * mu_a;
        let sb = m3 - mu_b * mu_b;
        let sab = m4 - mu_a * mu_b;

        let c1 = cfg.consts.x;
        let c2 = cfg.consts.y;
        let num = (2.0 * mu_a * mu_b + c1) * (2.0 * sab + c2);
        let den = (mu_a * mu_a + mu_b * mu_b + c1) * (sa + sb + c2);
        v = num / den;
    }

    acc[li] = v;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) { acc[li] = acc[li] + acc[li + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (li == 0u) { partials[wg.x] = acc[0]; }
}
";

    /// <summary>
    /// Expand one target photograph from packed RGBA bytes into the float stack.
    ///
    /// The pixels arrive from the canvas as a JS typed array and are written straight to the
    /// GPU; they never enter the managed heap. Before this, each target was read into a .NET
    /// byte[], converted by a per-pixel CPU loop into a .NET float[] and uploaded - the
    /// "CPU packing loop plus upload" anti-pattern, at a million pixels a view.
    ///
    /// Byte order is the canvas's: R in the low byte of each u32 on a little-endian machine,
    /// which is every machine that runs WebGPU.
    /// </summary>
    public const string UnpackTarget = @"
@group(0) @binding(0) var<storage, read>       src  : array<u32>;   // one packed RGBA per pixel
@group(0) @binding(1) var<storage, read_write> dst  : array<f32>;   // the whole target stack
@group(0) @binding(2) var<uniform>             dims : vec4<u32>;    // x=pixels, y=dst float offset

@compute @workgroup_size(64)
fn unpack_target(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    if (p >= dims.x) { return; }
    let v = src[p];
    let o = dims.y + p * 3u;
    dst[o + 0u] = f32(v & 255u) / 255.0;
    dst[o + 1u] = f32((v >> 8u) & 255u) / 255.0;
    dst[o + 2u] = f32((v >> 16u) & 255u) / 255.0;
}
";

    /// <summary>Seed the logit buffer from the splats' current opacity, once before training.</summary>
    public const string InitLogits = @"
@group(0) @binding(0) var<storage, read>       splats        : array<f32>;
@group(0) @binding(1) var<storage, read_write> opacity_logit : array<f32>;
@group(0) @binding(2) var<uniform>             cfg           : vec4<u32>;   // x = splat count
@group(0) @binding(3) var<storage, read_write> log_scale     : array<f32>;  // 3 per splat

const FLOATS_PER_SPLAT : u32 = 14u;
const MIN_SCALE : f32 = 1e-7;

@compute @workgroup_size(64)
fn init_logits(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.x) { return; }
    let o = i * FLOATS_PER_SPLAT;

    let a = clamp(splats[o + 9u], 1e-6, 1.0 - 1e-6);
    opacity_logit[i] = log(a / (1.0 - a));

    // Scale is optimised in LOG space, as the reference does. The published learning rate is
    // meaningless in any other parameterisation, and it keeps a scale from going negative.
    for (var c = 0u; c < 3u; c = c + 1u) {
        log_scale[i * 3u + c] = log(max(splats[o + 6u + c], MIN_SCALE));
    }
}
";

    /// <summary>
    /// Pass 9: carry the 2D gradients back to GEOMETRY, and step it.
    ///
    /// The rasteriser produces dL/d(screen centre) and dL/d(conic). Turning those into
    /// dL/d(position, scale, rotation) is a chain that depends only on the splat and the view -
    /// not on the pixel - so it runs once per splat here rather than once per pixel in the
    /// backward or once per key in the scatter. That is the whole reason the tile rasteriser
    /// only ever accumulates nine numbers per splat.
    ///
    /// This is a line-for-line transcription of <see cref="SplatGeometryGradients.Backward"/>,
    /// which is verified against central finite differences of its own forward. The GPU gate
    /// compares this kernel back against that oracle on real data.
    ///
    /// Parameterisation matches the reference implementation or the published learning rates
    /// mean nothing: scale in log space, rotation as a quaternion normalised on use.
    /// </summary>
    public const string GeometryAdam = UniformsBlock + @"
@group(0) @binding(1) var<storage, read_write> splats     : array<f32>;   // 14 per splat
@group(0) @binding(2) var<storage, read>       grad_fixed : array<i32>;   // 9 per splat
@group(0) @binding(3) var<storage, read_write> log_scale  : array<f32>;   // 3 per splat
@group(0) @binding(4) var<storage, read_write> adam_m     : array<f32>;   // 14 per splat
@group(0) @binding(5) var<storage, read_write> adam_v     : array<f32>;   // 14 per splat

struct GeomCfg {
    lr    : vec4<f32>,   // x = position, y = log-scale, z = rotation, w = Adam step number
    limit : vec4<f32>,   // x = splat count, y = max scale, z = min scale, w unused
};
@group(0) @binding(6) var<uniform> g : GeomCfg;
// 10 per splat: dL/d(pos xyz, scale xyz, quat xyzw), written before the step.
// The GPU gate compares these against SplatGeometryGradients.Backward, and density control
// will read the screen-space position gradient that feeds them.
@group(0) @binding(7) var<storage, read_write> geom_out : array<f32>;
@group(0) @binding(8) var<uniform> grad_scale : vec4<f32>;   // y=centre 4..5, z=conic 6..8

const BETA1 : f32 = 0.9;
const BETA2 : f32 = 0.999;
const EPS : f32 = 1e-15;

fn adam(value : f32, grad : f32, lr : f32, step : f32,
        m : ptr<function, f32>, v : ptr<function, f32>) -> f32 {
    *m = BETA1 * (*m) + (1.0 - BETA1) * grad;
    *v = BETA2 * (*v) + (1.0 - BETA2) * grad * grad;
    let m_hat = *m / (1.0 - pow(BETA1, step));
    let v_hat = *v / (1.0 - pow(BETA2, step));
    return value - lr * m_hat / (sqrt(v_hat) + EPS);
}

@compute @workgroup_size(64)
fn adam_geometry(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= u32(g.limit.x)) { return; }

    let gb = i * GRADS_PER_SPLAT;
    let up_cx = f32(grad_fixed[gb + 4u]) / grad_scale.y;
    let up_cy = f32(grad_fixed[gb + 5u]) / grad_scale.y;
    let up_ca = f32(grad_fixed[gb + 6u]) / grad_scale.z;
    let up_cb = f32(grad_fixed[gb + 7u]) / grad_scale.z;
    let up_cc = f32(grad_fixed[gb + 8u]) / grad_scale.z;

    // A splat this view never touched has no gradient. Taking a step anyway would let stale
    // momentum drag geometry that nothing is currently constraining - harmless for colour,
    // but for position it sends invisible splats travelling.
    if (up_cx == 0.0 && up_cy == 0.0 && up_ca == 0.0 && up_cb == 0.0 && up_cc == 0.0) { return; }

    let o = i * FLOATS_PER_SPLAT;
    let pos = vec3<f32>(splats[o + 0u], splats[o + 1u], splats[o + 2u]);
    let s_raw = vec3<f32>(splats[o + 6u], splats[o + 7u], splats[o + 8u]);
    let q_raw = vec4<f32>(splats[o + 10u], splats[o + 11u], splats[o + 12u], splats[o + 13u]);

    // ---- forward again, keeping the intermediates ----
    let rel = pos - u.cam_pos.xyz;
    let tx = dot(u.cam_right.xyz, rel);
    let ty = dot(u.cam_up.xyz, rel);
    let tz = dot(u.cam_fwd.xyz, rel);
    if (tz <= 1e-6) { return; }

    let qlen = length(q_raw);
    if (qlen < 1e-20) { return; }
    let q = q_raw / qlen;

    let sc3 = max(s_raw, vec3<f32>(1e-9, 1e-9, 1e-9));
    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
    let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;

    let r00 = 1.0 - 2.0 * (yy + zz); let r01 = 2.0 * (xy - wz); let r02 = 2.0 * (xz + wy);
    let r10 = 2.0 * (xy + wz); let r11 = 1.0 - 2.0 * (xx + zz); let r12 = 2.0 * (yz - wx);
    let r20 = 2.0 * (xz - wy); let r21 = 2.0 * (yz + wx); let r22 = 1.0 - 2.0 * (xx + yy);

    let m00 = r00 * sc3.x; let m01 = r01 * sc3.y; let m02 = r02 * sc3.z;
    let m10 = r10 * sc3.x; let m11 = r11 * sc3.y; let m12 = r12 * sc3.z;
    let m20 = r20 * sc3.x; let m21 = r21 * sc3.y; let m22 = r22 * sc3.z;

    let M = mat3x3<f32>(
        vec3<f32>(m00, m10, m20),
        vec3<f32>(m01, m11, m21),
        vec3<f32>(m02, m12, m22));
    let sigma_world = M * transpose(M);
    let A = transpose(mat3x3<f32>(u.cam_right.xyz, u.cam_up.xyz, u.cam_fwd.xyz));
    let sc = A * sigma_world * transpose(A);

    let S00 = sc[0][0]; let S01 = sc[1][0]; let S02 = sc[2][0];
    let S11 = sc[1][1]; let S12 = sc[2][1]; let S22 = sc[2][2];

    let invz = 1.0 / tz;
    let invz2 = invz * invz;
    let j00 = u.focal.x * invz;
    let j02 = -u.focal.x * tx * invz2;
    let j11 = u.focal.y * invz;
    let j12 = -u.focal.y * ty * invz2;

    let cov_a = j00 * j00 * S00 + 2.0 * j00 * j02 * S02 + j02 * j02 * S22 + EWA_FILTER_PX2;
    let cov_b = j00 * j11 * S01 + j00 * j12 * S02 + j02 * j11 * S12 + j02 * j12 * S22;
    let cov_c = j11 * j11 * S11 + 2.0 * j11 * j12 * S12 + j12 * j12 * S22 + EWA_FILTER_PX2;

    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 1e-20) { return; }
    let iD = 1.0 / det;
    let iD2 = iD * iD;

    // ---- 1. conic = Sigma_2D^-1 ----
    let gA = up_ca * (-cov_c * cov_c * iD2)
           + up_cb * (cov_b * cov_c * iD2)
           + up_cc * (iD - cov_a * cov_c * iD2);
    let gB = up_ca * (2.0 * cov_b * cov_c * iD2)
           + up_cb * (-iD - 2.0 * cov_b * cov_b * iD2)
           + up_cc * (2.0 * cov_a * cov_b * iD2);
    let gC = up_ca * (iD - cov_a * cov_c * iD2)
           + up_cb * (cov_a * cov_b * iD2)
           + up_cc * (-cov_a * cov_a * iD2);

    // ---- 2. Sigma_2D = J Sigma_cam J^T ----
    let gS00 = gA * j00 * j00;
    let gS01 = gB * j00 * j11;
    let gS02 = gA * 2.0 * j00 * j02 + gB * j00 * j12;
    let gS11 = gC * j11 * j11;
    let gS12 = gB * j02 * j11 + gC * 2.0 * j11 * j12;
    let gS22 = gA * j02 * j02 + gB * j02 * j12 + gC * j12 * j12;

    let gj00 = gA * 2.0 * (j00 * S00 + j02 * S02) + gB * (j11 * S01 + j12 * S02);
    let gj02 = gA * 2.0 * (j00 * S02 + j02 * S22) + gB * (j11 * S12 + j12 * S22);
    let gj11 = gB * (j00 * S01 + j02 * S12) + gC * 2.0 * (j11 * S11 + j12 * S12);
    let gj12 = gB * (j00 * S02 + j02 * S22) + gC * 2.0 * (j11 * S12 + j12 * S22);

    // ---- 3. Sigma_cam = A Sigma_world A^T, so dL/dSigma_world = A^T (dL/dSigma_cam) A ----
    // Unique-component gradients become a full symmetric matrix by HALVING the off-diagonals.
    // The result of A^T G A is already a matrix, so it must NOT be halved again on the way out -
    // doing so scales every scale and rotation gradient by a view-dependent 1.2 to 1.3.
    let h01 = 0.5 * gS01; let h02 = 0.5 * gS02; let h12 = 0.5 * gS12;
    let G = mat3x3<f32>(
        vec3<f32>(gS00, h01, h02),
        vec3<f32>(h01, gS11, h12),
        vec3<f32>(h02, h12, gS22));
    let W = transpose(A) * G * A;

    // ---- 4. Sigma_world = M M^T ----
    let dM = 2.0 * W * M;
    let gsx = dM[0][0] * r00 + dM[0][1] * r10 + dM[0][2] * r20;
    let gsy = dM[1][0] * r01 + dM[1][1] * r11 + dM[1][2] * r21;
    let gsz = dM[2][0] * r02 + dM[2][1] * r12 + dM[2][2] * r22;

    let gr00 = dM[0][0] * sc3.x; let gr10 = dM[0][1] * sc3.x; let gr20 = dM[0][2] * sc3.x;
    let gr01 = dM[1][0] * sc3.y; let gr11 = dM[1][1] * sc3.y; let gr21 = dM[1][2] * sc3.y;
    let gr02 = dM[2][0] * sc3.z; let gr12 = dM[2][1] * sc3.z; let gr22 = dM[2][2] * sc3.z;

    // ---- 5. R from the NORMALISED quaternion, then back through the normalisation ----
    let gnx = 2.0 * q.y * (gr01 + gr10) + 2.0 * q.z * (gr02 + gr20)
            - 4.0 * q.x * (gr11 + gr22) + 2.0 * q.w * (gr21 - gr12);
    let gny = -4.0 * q.y * (gr00 + gr22) + 2.0 * q.x * (gr01 + gr10)
            + 2.0 * q.w * (gr02 - gr20) + 2.0 * q.z * (gr12 + gr21);
    let gnz = -4.0 * q.z * (gr00 + gr11) + 2.0 * q.w * (gr10 - gr01)
            + 2.0 * q.x * (gr02 + gr20) + 2.0 * q.y * (gr12 + gr21);
    let gnw = 2.0 * q.z * (gr10 - gr01) + 2.0 * q.y * (gr02 - gr20)
            + 2.0 * q.x * (gr21 - gr12);

    let gn = vec4<f32>(gnx, gny, gnz, gnw);
    let gq = (gn - dot(gn, q) * q) / qlen;

    // ---- 6. Camera-space position: through the screen centre AND through J ----
    var gt = vec3<f32>(
        up_cx * u.focal.x * invz,
        up_cy * (-u.focal.y * invz),
        up_cx * (-u.focal.x * tx * invz2) + up_cy * (u.focal.y * ty * invz2));

    gt.x = gt.x + gj02 * (-u.focal.x * invz2);
    gt.y = gt.y + gj12 * (-u.focal.y * invz2);
    gt.z = gt.z
         + gj00 * (-u.focal.x * invz2)
         + gj02 * (2.0 * u.focal.x * tx * invz2 * invz)
         + gj11 * (-u.focal.y * invz2)
         + gj12 * (2.0 * u.focal.y * ty * invz2 * invz);

    let gpos = u.cam_right.xyz * gt.x + u.cam_up.xyz * gt.y + u.cam_fwd.xyz * gt.z;

    let gb_out = i * 10u;
    geom_out[gb_out + 0u] = gpos.x;
    geom_out[gb_out + 1u] = gpos.y;
    geom_out[gb_out + 2u] = gpos.z;
    geom_out[gb_out + 3u] = select(0.0, gsx, s_raw.x > 1e-9);
    geom_out[gb_out + 4u] = select(0.0, gsy, s_raw.y > 1e-9);
    geom_out[gb_out + 5u] = select(0.0, gsz, s_raw.z > 1e-9);
    geom_out[gb_out + 6u] = gq.x;
    geom_out[gb_out + 7u] = gq.y;
    geom_out[gb_out + 8u] = gq.z;
    geom_out[gb_out + 9u] = gq.w;

    // ---- Adam ----
    let step = g.lr.w;
    let ab = i * ADAM_SLOTS;

    for (var c = 0u; c < 3u; c = c + 1u) {
        var m = adam_m[ab + ADAM_POS + c];
        var v = adam_v[ab + ADAM_POS + c];
        let updated = adam(splats[o + c], gpos[c], g.lr.x, step, &m, &v);
        adam_m[ab + ADAM_POS + c] = m;
        adam_v[ab + ADAM_POS + c] = v;
        splats[o + c] = updated;
    }

    // Scale in log space. d/d(log s) = s * d/ds. A clamped scale contributed nothing to the
    // forward, so it gets no gradient either - see SplatGeometryGradients.
    let gs = vec3<f32>(gsx, gsy, gsz);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let live = select(0.0, 1.0, s_raw[c] > 1e-9);
        var m = adam_m[ab + ADAM_SCALE + c];
        var v = adam_v[ab + ADAM_SCALE + c];
        let updated = adam(log_scale[i * 3u + c], gs[c] * sc3[c] * live, g.lr.y, step, &m, &v);
        adam_m[ab + ADAM_SCALE + c] = m;
        adam_v[ab + ADAM_SCALE + c] = v;
        let bounded = clamp(updated, log(g.limit.z), log(g.limit.y));
        log_scale[i * 3u + c] = bounded;
        splats[o + 6u + c] = exp(bounded);
    }

    var qn = vec4<f32>(0.0);
    for (var c = 0u; c < 4u; c = c + 1u) {
        var m = adam_m[ab + ADAM_QUAT + c];
        var v = adam_v[ab + ADAM_QUAT + c];
        qn[c] = adam(q_raw[c], gq[c], g.lr.z, step, &m, &v);
        adam_m[ab + ADAM_QUAT + c] = m;
        adam_v[ab + ADAM_QUAT + c] = v;
    }
    // Renormalise. The gradient is already orthogonal to q, so the length only drifts through
    // Adam curvature - but letting it drift makes the effective rotation rate depend on how
    // long the splat has been training.
    let ql = length(qn);
    let qout = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), qn / ql, ql > 1e-12);
    splats[o + 10u] = qout.x;
    splats[o + 11u] = qout.y;
    splats[o + 12u] = qout.z;
    splats[o + 13u] = qout.w;
}
";
}
