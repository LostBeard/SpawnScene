using SpawnDev.BlazorJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Minimal WebGL helper that blits an OffscreenCanvas (rendered by WebGPU)
/// into the XR compositor's framebuffer. This is the only WebGL code in
/// the XR pipeline — all scene rendering stays on WebGPU.
/// Will be removed once browsers support WebGPU XR natively.
/// </summary>
public class WebGLXRBlit : IDisposable
{
    private WebGL2RenderingContext? _gl;
    private WebGLProgram? _program;
    private WebGLTexture? _texture;
    private WebGLUniformLocation? _texUniformLoc;

    private const string VertexShader = @"#version 300 es
void main() {
    // Fullscreen triangle from gl_VertexID (no buffers needed)
    vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,
                    float((gl_VertexID & 2) << 1) - 1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}";

    private const string FragmentShader = @"#version 300 es
precision mediump float;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    // gl_FragCoord → [0,1] UV via viewport dimensions
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uTex, 0));
    fragColor = texture(uTex, uv);
}";

    public void Initialize(WebGL2RenderingContext gl)
    {
        _gl = gl;

        // Compile shaders
        using var vs = gl.CreateShader(GL.VERTEX_SHADER);
        gl.ShaderSource(vs, VertexShader);
        gl.CompileShader(vs);

        using var fs = gl.CreateShader(GL.FRAGMENT_SHADER);
        gl.ShaderSource(fs, FragmentShader);
        gl.CompileShader(fs);

        // Link program
        _program = gl.CreateProgram();
        gl.AttachShader(_program, vs);
        gl.AttachShader(_program, fs);
        gl.LinkProgram(_program);

        _texUniformLoc = gl.GetUniformLocation(_program, "uTex");

        // Create texture for canvas blit
        _texture = gl.CreateTexture();
        gl.BindTexture(GL.TEXTURE_2D, _texture);
        gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.LINEAR);
        gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.LINEAR);
        gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
        gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
    }

    /// <summary>
    /// Copy the WebGPU-rendered OffscreenCanvas into the XR framebuffer at the given viewport.
    /// This is the WebGPU→WebGL bridge: texImage2D(canvas) is browser-optimized GPU→GPU copy.
    /// </summary>
    public void Blit(OffscreenCanvas source, WebGLFramebuffer xrFramebuffer, XRViewport viewport)
    {
        if (_gl == null || _program == null || _texture == null || _texUniformLoc == null) return;

        _gl.BindFramebuffer(GL.FRAMEBUFFER, xrFramebuffer);
        _gl.Viewport((int)viewport.X, (int)viewport.Y, (int)viewport.Width, (int)viewport.Height);

        // Upload canvas content to texture (browser handles GPU→GPU copy internally)
        _gl.BindTexture(GL.TEXTURE_2D, _texture);
        _gl.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA, GL.RGBA, GL.UNSIGNED_BYTE, source);

        // Draw fullscreen triangle
        _gl.UseProgram(_program);
        _gl.Uniform1i(_texUniformLoc, 0);
        _gl.Disable(GL.DEPTH_TEST);
        _gl.DrawArrays(GL.TRIANGLES, 0, 3);
    }

    public void Dispose()
    {
        if (_gl != null && _program != null)
            _gl.DeleteProgram(_program);
        if (_gl != null && _texture != null)
            _gl.DeleteTexture(_texture);
        _texUniformLoc?.Dispose();
        _texture?.Dispose();
        _program?.Dispose();
        _texUniformLoc = null;
        _texture = null;
        _program = null;
        _gl = null;
    }
}
