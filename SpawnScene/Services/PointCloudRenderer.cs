using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// WebGPU-based point cloud renderer. Renders 3D points directly on the GPU
/// using a render pipeline with point-sprite geometry (triangle billboards).
/// No CPU-side pixel computation — all projection happens in the vertex shader.
/// </summary>
public class PointCloudRenderer : IDisposable
{
    private readonly BlazorJSRuntime _js;

    private GPUDevice? _device;
    private GPUQueue? _queue;
    private GPUCanvasContext? _context;
    private GPURenderPipeline? _pipeline;
    private GPUShaderModule? _shaderModule;
    private string _canvasFormat = "bgra8unorm";

    private GPUTexture? _depthTexture;
    private GPUTextureView? _depthView;
    private string? _canvasId;
    private int _canvasWidth;
    private int _canvasHeight;

    // Vertex buffer: 6 vertices per point (2 triangles = quad billboard)
    private GPUBuffer? _vertexBuffer;
    private int _pointCount;

    // Camera uniform buffer (MVP matrix = 64 bytes)
    private GPUBuffer? _uniformBuffer;
    private GPUBindGroup? _uniformBindGroup;

    private bool _running;
    private bool _disposed;
    private double _lastTimestamp;
    private ActionCallback<double>? _rafCallback;

    // Orbit camera
    private float _rotX = 0.3f;
    private float _rotY = 0.5f;
    private float _zoom = 1.5f;
    private Vector3 _center;
    private float _extent = 1.0f;

    // Mouse state
    private bool _isDragging;
    private double _lastMouseX, _lastMouseY;

    // Per-vertex: center (vec3) + offset (vec2) + color (vec3) = 8 floats = 32 bytes
    private const int FloatsPerVertex = 8;
    private const int BytesPerVertex = FloatsPerVertex * 4;
    private float _pointSize = 0.02f;
    private bool _logMvpOnce = true;

    public bool IsInitialized { get; private set; }
    public int PointCount => _pointCount;

    public PointCloudRenderer(BlazorJSRuntime js)
    {
        _js = js;
    }

    /// <summary>
    /// Initialize the WebGPU render pipeline using the device from the ILGPU accelerator.
    /// </summary>
    public void Init(HTMLCanvasElement canvas, Accelerator accelerator)
    {
        if (IsInitialized) return;

        if (accelerator is not WebGPUAccelerator webGpuAccel)
            throw new InvalidOperationException("PointCloudRenderer requires a WebGPU accelerator");

        var nativeAccel = webGpuAccel.NativeAccelerator;
        _device = nativeAccel.NativeDevice
            ?? throw new InvalidOperationException("WebGPU native device is null");
        _queue = nativeAccel.Queue
            ?? throw new InvalidOperationException("WebGPU queue is null");

        _context = canvas.GetContext<GPUCanvasContext>("webgpu");

        using var navigator = _js.Get<Navigator>("navigator");
        using var gpu = navigator.Gpu;
        if (gpu is not null)
            _canvasFormat = gpu.GetPreferredCanvasFormat();

        _context.Configure(new GPUCanvasConfiguration
        {
            Device = _device,
            Format = _canvasFormat,
        });

        _canvasId = canvas.Id;
        _canvasWidth = canvas.ClientWidth;
        _canvasHeight = canvas.ClientHeight;
        if (_canvasWidth <= 0) _canvasWidth = 800;
        if (_canvasHeight <= 0) _canvasHeight = 500;
        canvas.Width = _canvasWidth;
        canvas.Height = _canvasHeight;

        _shaderModule = _device.CreateShaderModule(new GPUShaderModuleDescriptor
        {
            Code = WgslShaderSource
        });

        _pipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = _shaderModule,
                EntryPoint = "vs_main",
                Buffers = new[]
                {
                    new GPUVertexBufferLayout
                    {
                        ArrayStride = (ulong)BytesPerVertex,
                        StepMode = GPUVertexStepMode.Vertex,
                        Attributes = new GPUVertexAttribute[]
                        {
                            new() { ShaderLocation = 0, Offset = 0,      Format = GPUVertexFormat.Float32x3 }, // center
                            new() { ShaderLocation = 1, Offset = 3 * 4,  Format = GPUVertexFormat.Float32x2 }, // offset
                            new() { ShaderLocation = 2, Offset = 5 * 4,  Format = GPUVertexFormat.Float32x3 }, // color
                        }
                    }
                }
            },
            Fragment = new GPUFragmentState
            {
                Module = _shaderModule,
                EntryPoint = "fs_main",
                Targets = new[]
                {
                    new GPUColorTargetState { Format = _canvasFormat }
                }
            },
            Primitive = new GPUPrimitiveState
            {
                Topology = GPUPrimitiveTopology.TriangleList,
            },
            DepthStencil = new GPUDepthStencilState
            {
                Format = "depth24plus",
                DepthWriteEnabled = true,
                DepthCompare = "less",
            }
        });

        CreateDepthTexture();

        // Create uniform buffer for MVP matrix + pointSize (4x4 float + 1 float = 68 bytes, padded to 80)
        _uniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 80,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _uniformBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _pipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry
                {
                    Binding = 0,
                    Resource = new GPUBufferBinding { Buffer = _uniformBuffer }
                }
            }
        });

        IsInitialized = true;
        Console.WriteLine($"[PointCloudRenderer] Initialized. Canvas: {_canvasWidth}x{_canvasHeight}");
    }

    /// <summary>
    /// Upload point cloud data to the GPU vertex buffer.
    /// Each point becomes a small quad (2 triangles, 6 vertices) for visibility.
    /// Uses per-axis normalization so elongated clouds fill the view properly.
    /// </summary>
    public void UploadPoints(List<ReconstructedPoint> points)
    {
        if (_device == null || _queue == null || points.Count == 0) return;

        _pointCount = points.Count;

        // Compute bounds for centering
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue;

        foreach (var pt in points)
        {
            minX = Math.Min(minX, pt.Position.X); maxX = Math.Max(maxX, pt.Position.X);
            minY = Math.Min(minY, pt.Position.Y); maxY = Math.Max(maxY, pt.Position.Y);
            minZ = Math.Min(minZ, pt.Position.Z); maxZ = Math.Max(maxZ, pt.Position.Z);
        }

        _center = new Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);

        // Per-axis extents (avoid division by zero)
        float extX = Math.Max(maxX - minX, 0.01f);
        float extY = Math.Max(maxY - minY, 0.01f);
        float extZ = Math.Max(maxZ - minZ, 0.01f);

        // Use the median extent for the orbit camera distance calculation
        var exts = new[] { extX, extY, extZ };
        System.Array.Sort(exts);
        _extent = exts[1]; // median extent for zoom scaling

        Console.WriteLine($"[PointCloudRenderer] Extents: X={extX:F2}, Y={extY:F2}, Z={extZ:F2}, using median={_extent:F2}");

        // UNIFORM normalization — same scale all axes to show TRUE SHAPE
        float maxExtent = Math.Max(extX, Math.Max(extY, extZ));
        if (maxExtent < 0.001f) maxExtent = 1f;

        Console.WriteLine($"[Renderer] RAW ranges: X=[{minX:F4},{maxX:F4}] Y=[{minY:F4},{maxY:F4}] Z=[{minZ:F4},{maxZ:F4}]");
        Console.WriteLine($"[Renderer] Extents: X={extX:F4}, Y={extY:F4}, Z={extZ:F4}, maxExtent={maxExtent:F4}");

        // Log PCA stats for diagnostics (but DON'T rotate — just show as-is)
        {
            double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
            foreach (var p in points)
            {
                double dx = p.Position.X - _center.X, dy = p.Position.Y - _center.Y, dz = p.Position.Z - _center.Z;
                cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
                cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
            }
            int n = points.Count;
            var cov = new double[3, 3] {
                { cxx/n, cxy/n, cxz/n },
                { cxy/n, cyy/n, cyz/n },
                { cxz/n, cyz/n, czz/n }
            };
            var (_, eigVals, _) = LinearAlgebra.SVD3x3(cov);
            double ratio = eigVals[0] > 1e-15 ? eigVals[2] / eigVals[0] : 0;
            Console.WriteLine($"[Renderer] PCA: λ0={eigVals[0]:E3}, λ1={eigVals[1]:E3}, λ2={eigVals[2]:E3}, ratio={ratio * 100:F2}% {(ratio > 0.01 ? "✅ 3D" : "❌ FLAT")}");
        }

        // Build vertex data: 6 vertices per point (quad billboard)
        _pointSize = Math.Max(0.03f, 0.5f / MathF.Sqrt(_pointCount));
        var verts = new float[_pointCount * 6 * FloatsPerVertex];
        int vi = 0;

        // Billboard offsets for 2 triangles (6 vertices)
        float[] ox = { -1, 1, 1, -1, 1, -1 };
        float[] oy = { -1, -1, 1, -1, 1, 1 };

        // Simple center + uniform normalize — no PCA rotation
        foreach (var pt in points)
        {
            float px = (pt.Position.X - _center.X) / maxExtent;
            float py = (pt.Position.Y - _center.Y) / maxExtent;
            float pz = (pt.Position.Z - _center.Z) / maxExtent;

            float r = pt.Color.X;
            float g = pt.Color.Y;
            float b = pt.Color.Z;

            // 6 vertices per quad (2 triangles), each has center + offset + color
            for (int v = 0; v < 6; v++)
            {
                verts[vi++] = px;    // center.x
                verts[vi++] = py;    // center.y
                verts[vi++] = pz;    // center.z
                verts[vi++] = ox[v]; // offset.x
                verts[vi++] = oy[v]; // offset.y
                verts[vi++] = r;     // color.r
                verts[vi++] = g;     // color.g
                verts[vi++] = b;     // color.b
            }
        }

        // Destroy old buffer if exists
        _vertexBuffer?.Destroy();
        _vertexBuffer?.Dispose();

        _vertexBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)(verts.Length * 4),
            Usage = GPUBufferUsage.Vertex | GPUBufferUsage.CopyDst,
        });

        using var jsArray = new Float32Array(verts);
        _queue.WriteBuffer(_vertexBuffer, 0, jsArray);

        Console.WriteLine($"[Renderer] Uploaded {_pointCount} points, billboard={_pointSize:F4}, maxExtent={maxExtent:F4}");

        // Reset orbit camera — 30° elevated, 45° rotated, well outside the scene
        _rotX = 0.5f;     // 30° above horizontal
        _rotY = 0.6f;     // ~35° around
        _zoom = 2.5f;     // Camera at 2.5x, scene in [-0.5, 0.5] — should fill nicely
    }

    private static void WriteVertex(float[] buf, ref int idx, float cx, float cy, float cz, float ox, float oy, float r, float g, float b)
    {
        buf[idx++] = cx;
        buf[idx++] = cy;
        buf[idx++] = cz;
        buf[idx++] = ox;
        buf[idx++] = oy;
        buf[idx++] = r;
        buf[idx++] = g;
        buf[idx++] = b;
    }

    public void StartRenderLoop()
    {
        if (_running) return;
        _running = true;
        _lastTimestamp = 0;
        _rafCallback ??= new ActionCallback<double>(OnAnimationFrame);
        RequestFrame();
    }

    public void StopRenderLoop() => _running = false;

    private void RequestFrame()
    {
        if (!_running || _disposed || _rafCallback == null) return;
        using var window = _js.Get<Window>("window");
        window.RequestAnimationFrame(_rafCallback);
    }

    private void OnAnimationFrame(double timestamp)
    {
        if (!_running || _disposed) return;
        RenderFrame();
        RequestFrame();
    }

    private void RenderFrame()
    {
        if (_device == null || _context == null || _pipeline == null ||
            _vertexBuffer == null || _pointCount == 0)
            return;

        // Dynamic resize
        if (_canvasId != null)
        {
            using var doc = _js.Get<Document>("document");
            using var canvasEl = doc.GetElementById<HTMLCanvasElement>(_canvasId);
            if (canvasEl != null)
            {
                int cw = canvasEl.ClientWidth;
                int ch = canvasEl.ClientHeight;
                if (cw > 0 && ch > 0 && (cw != _canvasWidth || ch != _canvasHeight))
                {
                    _canvasWidth = cw;
                    _canvasHeight = ch;
                    canvasEl.Width = cw;
                    canvasEl.Height = ch;
                    CreateDepthTexture();
                }
            }
        }

        // Build orbit camera MVP matrix
        float aspect = (float)_canvasWidth / _canvasHeight;
        var mvp = BuildOrbitMvp(aspect);



        // Upload MVP + pointSize to uniform buffer
        var mvpFloats = new float[16];
        CopyMatrixToArray(mvp, mvpFloats);
        var uniformData = new byte[80];
        Buffer.BlockCopy(mvpFloats, 0, uniformData, 0, 64);
        var pointSizeBytes = BitConverter.GetBytes(_pointSize);
        Buffer.BlockCopy(pointSizeBytes, 0, uniformData, 64, 4);
        _queue!.WriteBuffer(_uniformBuffer!, 0, uniformData);

        // Render
        using var colorTexture = _context.GetCurrentTexture();
        using var colorView = colorTexture.CreateView();
        using var encoder = _device.CreateCommandEncoder();

        using var pass = encoder.BeginRenderPass(new GPURenderPassDescriptor
        {
            ColorAttachments = new[]
            {
                new GPURenderPassColorAttachment
                {
                    View = colorView,
                    LoadOp = GPULoadOp.Clear,
                    StoreOp = GPUStoreOp.Store,
                    ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.10, A = 1.0 },
                }
            },
            DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
            {
                View = _depthView!,
                DepthLoadOp = "clear",
                DepthStoreOp = "store",
                DepthClearValue = 1.0f,
            }
        });

        pass.SetPipeline(_pipeline);
        pass.SetBindGroup(0, _uniformBindGroup!);
        pass.SetVertexBuffer(0, _vertexBuffer);
        pass.Draw((uint)(_pointCount * 6)); // 6 vertices per point (quad)
        pass.End();

        using var commandBuffer = encoder.Finish();
        _queue!.Submit(new[] { commandBuffer });
    }

    private Matrix4x4 BuildOrbitMvp(float aspect)
    {
        // View matrix: orbit camera
        float camDist = _zoom;
        float camX = camDist * MathF.Sin(_rotY) * MathF.Cos(_rotX);
        float camY = camDist * MathF.Sin(_rotX);
        float camZ = camDist * MathF.Cos(_rotY) * MathF.Cos(_rotX);

        var eye = new Vector3(camX, camY, camZ);
        var target = Vector3.Zero;
        var up = new Vector3(0, 1, 0);

        var view = Matrix4x4.CreateLookAt(eye, target, up);
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(
            MathF.PI / 4f, aspect, 0.001f, 100f);

        return view * proj;
    }

    private static void CopyMatrixToArray(Matrix4x4 m, float[] arr)
    {
        // Column-major order for WGSL
        arr[0] = m.M11; arr[1] = m.M21; arr[2] = m.M31; arr[3] = m.M41;
        arr[4] = m.M12; arr[5] = m.M22; arr[6] = m.M32; arr[7] = m.M42;
        arr[8] = m.M13; arr[9] = m.M23; arr[10] = m.M33; arr[11] = m.M43;
        arr[12] = m.M14; arr[13] = m.M24; arr[14] = m.M34; arr[15] = m.M44;
    }

    // Mouse interaction — called from the page
    public void OnMouseDown(double x, double y)
    {
        _isDragging = true;
        _lastMouseX = x;
        _lastMouseY = y;
    }

    public void OnMouseMove(double x, double y)
    {
        if (!_isDragging) return;
        double dx = x - _lastMouseX;
        double dy = y - _lastMouseY;
        _lastMouseX = x;
        _lastMouseY = y;

        _rotY += (float)(dx * 0.01);
        _rotX += (float)(dy * 0.01);
        _rotX = Math.Clamp(_rotX, -MathF.PI / 2 + 0.01f, MathF.PI / 2 - 0.01f);
    }

    public void OnMouseUp() => _isDragging = false;

    public void OnWheel(double deltaY)
    {
        _zoom += (float)(deltaY * 0.002);
        _zoom = Math.Clamp(_zoom, 0.5f, 20f);
    }

    private void CreateDepthTexture()
    {
        _depthView?.Dispose();
        _depthTexture?.Destroy();
        _depthTexture?.Dispose();

        _depthTexture = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = "depth24plus",
            Usage = GPUTextureUsage.RenderAttachment,
        });
        _depthView = _depthTexture.CreateView();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _running = false;

        _rafCallback?.Dispose();
        _rafCallback = null;

        _vertexBuffer?.Destroy();
        _vertexBuffer?.Dispose();
        _vertexBuffer = null;

        _uniformBindGroup?.Dispose();
        _uniformBindGroup = null;
        _uniformBuffer?.Destroy();
        _uniformBuffer?.Dispose();
        _uniformBuffer = null;
        _depthView?.Dispose();
        _depthView = null;
        _depthTexture?.Destroy();
        _depthTexture?.Dispose();
        _depthTexture = null;
        _shaderModule?.Dispose();
        _shaderModule = null;
        _pipeline = null;
        _context?.Unconfigure();
        _context?.Dispose();
        _context = null;
        _canvasId = null;

        IsInitialized = false;
        _disposed = false;
    }

    #region WGSL Shader

    private const string WgslShaderSource = @"
struct Uniforms {
    mvp : mat4x4<f32>,
    pointSize : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexInput {
    @location(0) center   : vec3<f32>,
    @location(1) offset   : vec2<f32>,
    @location(2) color    : vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) point_color : vec3<f32>,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    // Transform center to clip space — use pos*mvp (row-vector convention matching .NET)
    let clip = vec4<f32>(input.center, 1.0) * uniforms.mvp;
    // Add offset in clip space (screen-aligned billboard)
    let screenOffset = input.offset * uniforms.pointSize * clip.w;
    output.clip_position = vec4<f32>(clip.x + screenOffset.x, clip.y + screenOffset.y, clip.z, clip.w);
    output.point_color = input.color;
    return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.point_color, 1.0);
}
";

    #endregion
}
