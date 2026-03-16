namespace SpawnScene.UI;

/// <summary>
/// WGSL shaders for the WebGPU UI overlay renderer.
/// Renders batched 2D quads (solid color or font-atlas-textured) on top of the 3D scene.
/// </summary>
internal static class UIShaders
{
    /// <summary>
    /// UI quad vertex + fragment shader.
    /// Vertex: transforms screen-pixel coords (0,0 = top-left) to NDC.
    /// Fragment: either solid color (mode=0) or atlas-sampled text (mode=1).
    /// </summary>
    public const string QuadShaderSource = @"
struct Uniforms {
    viewport : vec2<f32>,
    _pad     : vec2<f32>,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var t_atlas : texture_2d<f32>;
@group(0) @binding(2) var s_atlas : sampler;

struct VertexInput {
    @location(0) pos   : vec2<f32>,  // screen pixels (0,0 = top-left)
    @location(1) uv    : vec2<f32>,  // atlas UV (or -1,-1 for solid color)
    @location(2) color : vec4<f32>,  // RGBA tint / solid color
};

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) uv    : vec2<f32>,
    @location(1) color : vec4<f32>,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    // Screen pixels → NDC: x:[0,W]→[-1,1], y:[0,H]→[1,-1] (Y flipped)
    let ndc_x = input.pos.x / u.viewport.x * 2.0 - 1.0;
    let ndc_y = 1.0 - input.pos.y / u.viewport.y * 2.0;

    var out : VertexOutput;
    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // Always sample texture (textureSample requires uniform control flow).
    // UV.x < 0 signals solid color mode — texture sample is ignored via select().
    let tex = textureSample(t_atlas, s_atlas, max(input.uv, vec2<f32>(0.0)));
    let is_solid = input.uv.x < 0.0;
    // Solid: vertex color only. Textured: texture RGB * vertex tint, texture alpha * vertex alpha.
    // Text: atlas is white glyphs on transparent → tex.rgb * tint.rgb = tint color. Works.
    // Images: vertex color is white (1,1,1,1) → tex.rgb * 1 = image color. Works.
    let rgb = select(tex.rgb * input.color.rgb, input.color.rgb, is_solid);
    let alpha = select(tex.a * input.color.a, input.color.a, is_solid);
    return vec4<f32>(rgb, alpha);
}
";
}
