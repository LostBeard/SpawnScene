using SpawnDev.BlazorJS.JSObjects;
using System.Drawing;
using System.Runtime.InteropServices;

namespace SpawnScene.UI;

/// <summary>
/// Batched 2D quad renderer for the WebGPU UI overlay.
/// All UI elements emit quads via Draw* methods between Begin() and End().
/// End() flushes the batch as a single draw call on top of the 3D scene.
/// </summary>
public class UIRenderer : IDisposable
{
    private const int MaxQuads = 4096;
    private const int VerticesPerQuad = 6; // 2 triangles
    private const int FloatsPerVertex = 8; // pos(2) + uv(2) + color(4)
    private const int BytesPerVertex = FloatsPerVertex * sizeof(float);

    private GPUDevice? _device;
    private GPUQueue? _queue;
    private FontAtlas? _fontAtlas;

    // GPU resources
    private GPURenderPipeline? _pipeline;
    private GPUBuffer? _vertexBuffer;
    private GPUBuffer? _uniformBuffer;
    private GPUBindGroup? _bindGroup;
    private GPUSampler? _sampler;

    // CPU-side vertex batch
    private readonly float[] _vertices = new float[MaxQuads * VerticesPerQuad * FloatsPerVertex];
    private byte[]? _vertexBytes;
    private int _quadCount;

    // Deferred image draws (separate bind group per texture)
    private readonly List<(GPUTextureView view, float x, float y, float w, float h)> _imageBatch = new();
    private readonly Dictionary<GPUTextureView, GPUBindGroup> _imageBindGroups = new();

    // Per-frame state
    private int _viewportWidth;
    private int _viewportHeight;

    public bool IsReady => _pipeline != null;

    /// <summary>
    /// Initialize the UI render pipeline. Call after WebGPU device and FontAtlas are ready.
    /// </summary>
    public void Init(GPUDevice device, GPUQueue queue, FontAtlas fontAtlas, string canvasFormat)
    {
        _device = device;
        _queue = queue;
        _fontAtlas = fontAtlas;

        // Vertex buffer (dynamic, updated each frame)
        _vertexBuffer = device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)(MaxQuads * VerticesPerQuad * BytesPerVertex),
            Usage = GPUBufferUsage.Vertex | GPUBufferUsage.CopyDst,
        });
        _vertexBytes = new byte[_vertices.Length * sizeof(float)];

        // Uniform buffer (viewport size, 16 bytes aligned)
        _uniformBuffer = device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // Sampler for font atlas
        _sampler = device.CreateSampler(new GPUSamplerDescriptor
        {
            MinFilter = "linear",
            MagFilter = "linear",
        });

        // Shader module
        using var shader = device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = UIShaders.QuadShaderSource });

        // Render pipeline: alpha blend, no depth, renders on top of scene
        _pipeline = device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = shader,
                EntryPoint = "vs_main",
                Buffers = new[]
                {
                    new GPUVertexBufferLayout
                    {
                        ArrayStride = (ulong)BytesPerVertex,
                        StepMode = GPUVertexStepMode.Vertex,
                        Attributes = new GPUVertexAttribute[]
                        {
                            new() { ShaderLocation = 0, Offset = 0,  Format = GPUVertexFormat.Float32x2 }, // pos
                            new() { ShaderLocation = 1, Offset = 8,  Format = GPUVertexFormat.Float32x2 }, // uv
                            new() { ShaderLocation = 2, Offset = 16, Format = GPUVertexFormat.Float32x4 }, // color
                        }
                    }
                }
            },
            Fragment = new GPUFragmentState
            {
                Module = shader,
                EntryPoint = "fs_main",
                Targets = new[]
                {
                    new GPUColorTargetState
                    {
                        Format = canvasFormat,
                        Blend = new GPUBlendState
                        {
                            Color = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.SrcAlpha,
                                DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                Operation = GPUBlendOperation.Add,
                            },
                            Alpha = new GPUBlendComponent
                            {
                                SrcFactor = GPUBlendFactor.One,
                                DstFactor = GPUBlendFactor.OneMinusSrcAlpha,
                                Operation = GPUBlendOperation.Add,
                            }
                        }
                    }
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
        });

        // Bind group: uniform + atlas texture + sampler
        RebuildBindGroup();

        Console.WriteLine("[UIRenderer] Initialized");
    }

    private void RebuildBindGroup()
    {
        if (_pipeline == null || _fontAtlas?.View == null || _sampler == null || _uniformBuffer == null) return;
        _bindGroup?.Dispose();
        _bindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _pipeline.GetBindGroupLayout(0),
            Entries = new GPUBindGroupEntry[]
            {
                new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuffer } },
                new() { Binding = 1, Resource = _fontAtlas.View },
                new() { Binding = 2, Resource = _sampler },
            }
        });
    }

    /// <summary>Start a new UI frame. Clears the quad batch.</summary>
    public void Begin(int viewportWidth, int viewportHeight)
    {
        _viewportWidth = viewportWidth;
        _viewportHeight = viewportHeight;
        _quadCount = 0;
        _imageBatch.Clear();
    }

    /// <summary>Draw a solid-color rectangle.</summary>
    public void DrawRect(float x, float y, float w, float h, Color color)
    {
        if (_quadCount >= MaxQuads) return;
        float r = color.R / 255f, g = color.G / 255f, b = color.B / 255f, a = color.A / 255f;
        // UV = (-1, -1) signals solid color mode to the fragment shader
        AddQuad(x, y, x + w, y + h, -1, -1, -1, -1, r, g, b, a);
    }

    /// <summary>Draw a text string at the given position.</summary>
    public void DrawText(string text, float x, float y, FontSize size, Color color)
    {
        if (_fontAtlas == null || !_fontAtlas.IsReady) return;
        float r = color.R / 255f, g = color.G / 255f, b = color.B / 255f, a = color.A / 255f;
        float cursorX = x;

        foreach (char c in text)
        {
            if (_quadCount >= MaxQuads) break;
            var m = _fontAtlas.GetChar(c, size);
            if (m.Width > 0 && m.Height > 0 && c != ' ')
            {
                AddQuad(cursorX, y, cursorX + m.Width, y + m.Height,
                        m.U0, m.V0, m.U1, m.V1, r, g, b, a);
            }
            cursorX += m.Advance;
        }
    }

    /// <summary>Draw a textured image quad (rendered as a separate draw call per unique texture).</summary>
    public void DrawImage(GPUTextureView textureView, float x, float y, float w, float h)
    {
        _imageBatch.Add((textureView, x, y, w, h));
    }

    /// <summary>Measure text width without drawing.</summary>
    public float MeasureText(string text, FontSize size)
    {
        return _fontAtlas?.MeasureString(text, size) ?? 0;
    }

    /// <summary>Get line height for a font size.</summary>
    public float GetLineHeight(FontSize size)
    {
        return _fontAtlas?.GetLineHeight(size) ?? (int)size;
    }

    /// <summary>
    /// Flush the batch: upload vertices and append a render pass to the encoder.
    /// The render pass writes to the given target (swapchain texture) with LoadOp.Load
    /// so the 3D scene underneath is preserved.
    /// </summary>
    public void End(GPUCommandEncoder encoder, GPUTextureView target)
    {
        if (_pipeline == null) return;
        if (_quadCount == 0 && _imageBatch.Count == 0) return;

        // Upload viewport uniform
        var uniformData = new float[] { _viewportWidth, _viewportHeight, 0, 0 };
        var uniformBytes = new byte[16];
        Buffer.BlockCopy(uniformData, 0, uniformBytes, 0, 16);
        _queue!.WriteBuffer(_uniformBuffer!, 0, uniformBytes);

        // Append image quads to the vertex array AFTER the main batch.
        // All data must be written in one writeBuffer call because writeBuffer executes
        // before any GPU draw calls — writing per-image would clobber previous data.
        int imageStartQuad = _quadCount;
        foreach (var (view, ix, iy, iw, ih) in _imageBatch)
        {
            if (imageStartQuad >= MaxQuads) break;
            int offset = imageStartQuad * VerticesPerQuad * FloatsPerVertex;
            SetVertex(offset + 0 * FloatsPerVertex, ix, iy, 0, 0, 1, 1, 1, 1);
            SetVertex(offset + 1 * FloatsPerVertex, ix + iw, iy, 1, 0, 1, 1, 1, 1);
            SetVertex(offset + 2 * FloatsPerVertex, ix, iy + ih, 0, 1, 1, 1, 1, 1);
            SetVertex(offset + 3 * FloatsPerVertex, ix + iw, iy, 1, 0, 1, 1, 1, 1);
            SetVertex(offset + 4 * FloatsPerVertex, ix + iw, iy + ih, 1, 1, 1, 1, 1, 1);
            SetVertex(offset + 5 * FloatsPerVertex, ix, iy + ih, 0, 1, 1, 1, 1, 1);
            imageStartQuad++;
        }

        // Upload entire vertex buffer at once
        int totalQuads = imageStartQuad;
        int totalBytes = totalQuads * VerticesPerQuad * FloatsPerVertex * sizeof(float);
        Buffer.BlockCopy(_vertices, 0, _vertexBytes!, 0, totalBytes);
        _queue.WriteBuffer(_vertexBuffer!, 0, _vertexBytes, 0, (ulong)totalBytes);

        // Render pass: overlay on existing content (LoadOp.Load preserves scene)
        var colorAttach = new GPURenderPassColorAttachment
        {
            View = target,
            LoadOp = GPULoadOp.Load,
            StoreOp = GPUStoreOp.Store,
        };
        var passDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { colorAttach },
        };

        using var pass = encoder.BeginRenderPass(passDesc);
        pass.SetPipeline(_pipeline);
        pass.SetVertexBuffer(0, _vertexBuffer!);

        // Draw main batch (text + solid color quads)
        if (_quadCount > 0 && _bindGroup != null)
        {
            pass.SetBindGroup(0, _bindGroup);
            pass.Draw((uint)(_quadCount * VerticesPerQuad), 1, 0, 0);
        }

        // Draw images — each uses its own bind group but reads from its own offset in the vertex buffer
        int imgQuadIdx = _quadCount;
        foreach (var (view, _, _, _, _) in _imageBatch)
        {
            if (imgQuadIdx >= MaxQuads) break;
            var imgBindGroup = GetOrCreateImageBindGroup(view);
            if (imgBindGroup == null) { imgQuadIdx++; continue; }

            pass.SetBindGroup(0, imgBindGroup);
            pass.Draw(VerticesPerQuad, 1, (uint)(imgQuadIdx * VerticesPerQuad), 0);
            imgQuadIdx++;
        }

        pass.End();
    }

    /// <summary>Get or create a bind group for an image texture (cached per view).</summary>
    private GPUBindGroup? GetOrCreateImageBindGroup(GPUTextureView view)
    {
        if (_imageBindGroups.TryGetValue(view, out var existing)) return existing;
        if (_pipeline == null || _sampler == null || _uniformBuffer == null) return null;

        var bg = _device!.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _pipeline.GetBindGroupLayout(0),
            Entries = new GPUBindGroupEntry[]
            {
                new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = _uniformBuffer } },
                new() { Binding = 1, Resource = view },
                new() { Binding = 2, Resource = _sampler },
            }
        });
        _imageBindGroups[view] = bg;
        return bg;
    }

    /// <summary>Add a quad (2 triangles, 6 vertices) to the batch.</summary>
    private void AddQuad(float x0, float y0, float x1, float y1,
                         float u0, float v0, float u1, float v1,
                         float r, float g, float b, float a)
    {
        int offset = _quadCount * VerticesPerQuad * FloatsPerVertex;

        // Triangle 1: top-left, top-right, bottom-left
        SetVertex(offset + 0 * FloatsPerVertex, x0, y0, u0, v0, r, g, b, a);
        SetVertex(offset + 1 * FloatsPerVertex, x1, y0, u1, v0, r, g, b, a);
        SetVertex(offset + 2 * FloatsPerVertex, x0, y1, u0, v1, r, g, b, a);

        // Triangle 2: top-right, bottom-right, bottom-left
        SetVertex(offset + 3 * FloatsPerVertex, x1, y0, u1, v0, r, g, b, a);
        SetVertex(offset + 4 * FloatsPerVertex, x1, y1, u1, v1, r, g, b, a);
        SetVertex(offset + 5 * FloatsPerVertex, x0, y1, u0, v1, r, g, b, a);

        _quadCount++;
    }

    private void SetVertex(int offset, float x, float y, float u, float v,
                           float r, float g, float b, float a)
    {
        _vertices[offset + 0] = x;
        _vertices[offset + 1] = y;
        _vertices[offset + 2] = u;
        _vertices[offset + 3] = v;
        _vertices[offset + 4] = r;
        _vertices[offset + 5] = g;
        _vertices[offset + 6] = b;
        _vertices[offset + 7] = a;
    }

    public void Dispose()
    {
        _vertexBuffer?.Destroy();
        _vertexBuffer?.Dispose();
        _uniformBuffer?.Destroy();
        _uniformBuffer?.Dispose();
        _bindGroup?.Dispose();
        _sampler?.Dispose();
        _pipeline?.Dispose();
    }
}
