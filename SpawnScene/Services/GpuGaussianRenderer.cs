using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Native WebGPU Gaussian splat renderer with GPU-sorted splats.
/// Architecture:
///   - ILGPU = data generation + sorting (GpuSplatSorter)
///   - WebGPU = rendering only (this class)
///   - GPU-side buffer copy (no CPU round-trips)
///   - EWA anti-alias filter for distant splats
///   - CAS post-processing sharpening
/// </summary>
public class GpuGaussianRenderer : IDisposable
{
    private readonly GpuService _gpu;
    private readonly GpuSplatSorter _sorter;

    // WebGPU objects
    private GPUDevice? _device;
    private GPUQueue? _queue;
    private GPUCanvasContext? _context;
    private GPURenderPipeline? _splatPipeline;
    private GPURenderPipeline? _casPipeline;
    private string _canvasFormat = "bgra8unorm";

    // Gaussian vertex buffer: packed format (position f32x3 + color_alpha u8x4 + scale f16x4)
    private GPUBuffer? _splatBuffer;
    private int _splatCount;
    private const int PackedBytesPerSplat = 24; // 12 (pos) + 4 (color) + 8 (scale) = 24

    // Pack compute pipeline: converts Float32 sort output → packed vertex format
    private GPUComputePipeline? _packPipeline;
    private GPUBindGroup? _packBindGroup;
    private GPUBuffer? _srcDataCached; // cached ILGPU data buffer handle for bind group invalidation
    private GPUBuffer? _srcIdxCached;  // cached ILGPU index buffer handle for bind group invalidation
    private GPUBuffer? _packCountBuf;  // uniform: visible count for pack dispatch guard
    private Uint32Array? _packCountJsArray; // cached JS array for WriteBuffer (no per-frame alloc)

    // Uniform buffer: MVP matrix (64 bytes) + viewport (8 bytes) + focal (8 bytes) = 80 bytes
    private GPUBuffer? _uniformBuffer;
    private GPUBindGroup? _uniformBindGroup;
    private readonly float[] _uniformData = new float[20]; // 16 (mat4) + 2 (viewport) + 2 (focal)
    private Float32Array? _uniformJsArray; // cached — reused every frame, no per-frame alloc

    // CAS sharpening pass
    private GPUTexture? _offscreenTexture;
    private GPUTextureView? _offscreenView;
    private GPUBindGroup? _casBindGroup;
    private GPUBuffer? _casUniformBuffer;
    private GPUSampler? _casSampler;
    private float _sharpeningStrength = 0.5f;
    private readonly float[] _casData = new float[4];
    private Float32Array? _casJsArray; // cached — reused every frame, no per-frame alloc

    // Depth texture
    private GPUTexture? _depthTexture;
    private GPUTextureView? _depthView;

    private int _canvasWidth;
    private int _canvasHeight;
    private bool _disposed;

    /// <summary>Sharpening intensity (0 = off, 1 = maximum).</summary>
    public float SharpeningStrength
    {
        get => _sharpeningStrength;
        set => _sharpeningStrength = Math.Clamp(value, 0f, 1f);
    }

    public GpuGaussianRenderer(GpuService gpuService, GpuSplatSorter sorter)
    {
        _gpu = gpuService;
        _sorter = sorter;
    }

    /// <summary>Whether the GPU has a valid packed splat buffer ready to render.</summary>
    public bool HasGpuData => _splatCount > 0;

    /// <summary>
    /// Initialize the WebGPU render pipeline. Called once when canvas is attached.
    /// </summary>
    public void AttachCanvas(HTMLCanvasElement canvas)
    {
        var webGpuAccel = _gpu.WebGPUAccelerator;
        var nativeAccel = webGpuAccel.NativeAccelerator;
        _device = nativeAccel.NativeDevice
            ?? throw new InvalidOperationException("WebGPU native device is null");
        _queue = nativeAccel.Queue
            ?? throw new InvalidOperationException("WebGPU queue is null");

        _context = canvas.GetContext<GPUCanvasContext>("webgpu");

        using var navigator = BlazorJSRuntime.JS.Get<Navigator>("navigator");
        using var gpu = navigator.Gpu;
        if (gpu is not null)
            _canvasFormat = gpu.GetPreferredCanvasFormat();

        _context.Configure(new GPUCanvasConfiguration
        {
            Device = _device,
            Format = _canvasFormat,
        });

        _canvasWidth = canvas.Width;
        _canvasHeight = canvas.Height;

        // ── Pack Compute Pipeline (Float32 → Float16/Unorm8 packing) ──
        using var packShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = PackComputeSource });
        _packPipeline = _device.CreateComputePipeline(new GPUComputePipelineDescriptor
        {
            Layout = "auto",
            Compute = new GPUProgrammableStage
            {
                Module = packShader,
                EntryPoint = "pack_splats",
            }
        });

        // ── Splat Pipeline (packed vertex format) ──
        using var splatShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = SplatShaderSource });
        _splatPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = splatShader,
                EntryPoint = "vs_main",
                Buffers = new[]
                {
                    new GPUVertexBufferLayout
                    {
                        ArrayStride = (ulong)PackedBytesPerSplat,
                        StepMode = GPUVertexStepMode.Instance,
                        Attributes = new GPUVertexAttribute[]
                        {
                            new() { ShaderLocation = 0, Offset = 0,  Format = GPUVertexFormat.Float32x3 },  // position (12B)
                            new() { ShaderLocation = 1, Offset = 12, Format = GPUVertexFormat.UNorm8x4 },   // color+alpha (4B)
                            new() { ShaderLocation = 2, Offset = 16, Format = GPUVertexFormat.Float16x4 },  // scale (8B)
                        }
                    }
                }
            },
            Fragment = new GPUFragmentState
            {
                Module = splatShader,
                EntryPoint = "fs_main",
                Targets = new[]
                {
                    new GPUColorTargetState
                    {
                        Format = _canvasFormat,
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
            DepthStencil = new GPUDepthStencilState
            {
                Format = "depth24plus",
                DepthWriteEnabled = false,
                DepthCompare = "less",
            }
        });

        // ── CAS Pipeline ──
        using var casShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = CasShaderSource });
        _casPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState { Module = casShader, EntryPoint = "vs_fullscreen" },
            Fragment = new GPUFragmentState
            {
                Module = casShader,
                EntryPoint = "fs_cas",
                Targets = new[] { new GPUColorTargetState { Format = _canvasFormat } }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
        });

        // Depth texture
        CreateDepthTexture();

        // Uniform buffer (80 bytes: 16 mat4 floats + viewport + focal)
        _uniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 80,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        _uniformBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _splatPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry
                {
                    Binding = 0,
                    Resource = new GPUBufferBinding { Buffer = _uniformBuffer }
                }
            }
        });

        // Pre-allocate reusable JS typed arrays — avoids per-frame JS object allocations
        _uniformJsArray = new Float32Array(_uniformData.Length);
        _casJsArray = new Float32Array(4);
        _packCountJsArray = new Uint32Array(1);

        // Pack count uniform (4 bytes): holds visibleCount for pack shader guard
        _packCountBuf = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 4,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // CAS uniform (16 bytes aligned: sharpening strength + texel size)
        _casUniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // Sampler for CAS
        _casSampler = _device.CreateSampler(new GPUSamplerDescriptor
        {
            MinFilter = "linear",
            MagFilter = "linear",
        });

        // Offscreen texture for CAS input (must be after CAS resources are created)
        CreateOffscreenTexture();

        // If splat data was uploaded before the canvas was attached, create the vertex buffer now.
        EnsureSplatBuffer();

        Console.WriteLine($"[GpuRenderer] Pipeline created with EWA filter + CAS sharpening. Format: {_canvasFormat}");
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

    private void CreateOffscreenTexture()
    {
        _offscreenView?.Dispose();
        _offscreenTexture?.Destroy();
        _offscreenTexture?.Dispose();

        _offscreenTexture = _device!.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = _canvasFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        });
        _offscreenView = _offscreenTexture.CreateView();

        // Rebuild CAS bind group when texture changes
        if (_casPipeline != null && _casSampler != null && _casUniformBuffer != null)
        {
            _casBindGroup?.Dispose();
            _casBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _casPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _offscreenView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler },
                    new GPUBindGroupEntry
                    {
                        Binding = 2,
                        Resource = new GPUBufferBinding { Buffer = _casUniformBuffer }
                    }
                }
            });
        }
    }

    /// <summary>
    /// Creates _splatBuffer if splat data is ready and device is available.
    /// Safe to call multiple times — no-ops if buffer already exists or data not ready.
    /// </summary>
    private void EnsureSplatBuffer()
    {
        if (_device == null || _splatCount == 0) return;

        _splatBuffer?.Destroy();
        _splatBuffer?.Dispose();
        _splatBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = (ulong)_splatCount * PackedBytesPerSplat,
            Usage = GPUBufferUsage.Vertex | GPUBufferUsage.Storage,
        });

        // Invalidate pack bind group whenever the vertex buffer is recreated
        _packBindGroup?.Dispose();
        _packBindGroup = null;
        _srcDataCached = null;
        _srcIdxCached = null;
    }

    /// <summary>
    /// Upload scene data to GPU sorter and create vertex buffer.
    /// Called once when scene changes.
    /// </summary>
    public async Task UploadScene(GaussianScene scene)
    {
        await _sorter.UploadAsync(scene);

        _splatCount = _sorter.SplatCount;
        if (_splatCount == 0) return;

        // _splatBuffer requires _device — created here if canvas already attached,
        // otherwise deferred to AttachCanvas.
        EnsureSplatBuffer();

        Console.WriteLine($"[GpuRenderer] Packed vertex buffer: {_splatCount:N0} splats ({_splatCount * PackedBytesPerSplat / 1024}KB, was {_splatCount * 40 / 1024}KB)");
    }

    /// <summary>
    /// Upload a GPU-resident packed buffer directly (GPU fast path, no CPU involvement).
    /// Transfers ownership of packedBuf to the sorter — caller must NOT dispose it.
    /// Safe to call before AttachCanvas — vertex buffer is deferred until canvas is ready.
    /// </summary>
    public async Task UploadSceneFromGpuBuffer(
        MemoryBuffer1D<float, Stride1D.Dense> packedBuf, int splatCount)
    {
        await _sorter.UploadFromGpuBufferAsync(packedBuf, splatCount);

        _splatCount = _sorter.SplatCount;
        if (_splatCount == 0) return;

        // _splatBuffer requires _device — created here if canvas already attached,
        // otherwise deferred to AttachCanvas.
        EnsureSplatBuffer();

        Console.WriteLine($"[GpuRenderer] GPU fast-path upload: {_splatCount:N0} splats" +
            (_splatBuffer != null ? $", {_splatCount * PackedBytesPerSplat / 1024}KB vertex buffer" : " (vertex buffer deferred)"));
    }

    /// <summary>
    /// Resize canvas and recreate GPU textures for new dimensions.
    /// Called when the browser window is resized.
    /// </summary>
    public void ResizeCanvas(int newWidth, int newHeight)
    {
        if (_device == null || newWidth <= 0 || newHeight <= 0) return;
        if (newWidth == _canvasWidth && newHeight == _canvasHeight) return;

        _canvasWidth = newWidth;
        _canvasHeight = newHeight;

        // Recreate depth and offscreen textures for new size
        CreateDepthTexture();
        CreateOffscreenTexture();

        Console.WriteLine($"[GpuRenderer] Resized GPU textures: {newWidth}×{newHeight}");
    }

    /// <summary>
    /// Render one frame. GPU sort → GPU pack → splat render → CAS sharpen.
    /// Fully synchronous — no GPU drain. WebGPU's getCurrentTexture() provides
    /// natural frame pacing (~1 frame ahead of GPU) without any explicit await.
    /// </summary>
    private bool _renderLogged;
    public void Render(GaussianScene scene, CameraParams camera)
    {
        if (_device == null || _context == null || _splatPipeline == null ||
            _splatBuffer == null || _splatCount == 0)
        {
            if (!_renderLogged)
            {
                _renderLogged = true;
                Console.WriteLine($"[Render] Early exit: device={_device != null} ctx={_context != null} pipeline={_splatPipeline != null} buf={_splatBuffer != null} count={_splatCount}");
            }
            return;
        }
        if (!_renderLogged) { _renderLogged = true; Console.WriteLine($"[Render] First frame: {_splatCount} splats, cam={camera.Position} fwd={camera.Forward}"); }

        // Ensure camera dimensions match canvas
        if (camera.Width == 0 || camera.Height == 0)
        {
            camera.Width = _canvasWidth;
            camera.Height = _canvasHeight;
            camera.CenterX = _canvasWidth / 2f;
            camera.CenterY = _canvasHeight / 2f;
            camera.FocalX = MathF.Max(_canvasWidth, _canvasHeight) * 1.2f;
            camera.FocalY = camera.FocalX;
        }

        // ── Step 1: Build MVP (needed by both sort and render) ──
        var view = camera.ViewMatrix;
        float fovY = 2f * MathF.Atan(camera.Height / (2f * camera.FocalY));
        float aspect = (float)camera.Width / camera.Height;
        var proj = CreateWebGPUPerspective(fovY, aspect, camera.Near, camera.Far);
        var mvp = view * proj;

        // ── Step 2: GPU Frustum Cull + Sort ──
        // Returns sortRan=true when indices changed → vertex buffer must be repacked.
        // On non-sort frames the cached vertex buffer is still valid (same sort order).
        // visibleCount: deferred readback from previous frame; conservative on first frame.
        var (dataBuf, idxBuf, sortRan, visibleCount) = _sorter.Sort(camera, mvp);

        // ── Step 3: Upload MVP + viewport uniforms ──
        _uniformData[0] = mvp.M11; _uniformData[1] = mvp.M12; _uniformData[2] = mvp.M13; _uniformData[3] = mvp.M14;
        _uniformData[4] = mvp.M21; _uniformData[5] = mvp.M22; _uniformData[6] = mvp.M23; _uniformData[7] = mvp.M24;
        _uniformData[8] = mvp.M31; _uniformData[9] = mvp.M32; _uniformData[10] = mvp.M33; _uniformData[11] = mvp.M34;
        _uniformData[12] = mvp.M41; _uniformData[13] = mvp.M42; _uniformData[14] = mvp.M43; _uniformData[15] = mvp.M44;
        _uniformData[16] = camera.Width;
        _uniformData[17] = camera.Height;
        _uniformData[18] = camera.FocalX;
        _uniformData[19] = camera.FocalY;

        _uniformJsArray!.Set(_uniformData);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformJsArray);

        // ── Step 4: Pack (only when sort ran) + Render — single encoder/submit ──
        // Merging pack compute + render pass into one command buffer halves GPU submit overhead.
        using var colorTexture = _context.GetCurrentTexture();
        using var colorView = colorTexture.CreateView();
        using var encoder = _device.CreateCommandEncoder();

        // Pack: ILGPU Float32 → packed vertex buffer (only if sort produced new indices)
        if (sortRan && dataBuf != null && idxBuf != null)
            AppendPackComputePass(encoder, dataBuf, idxBuf, visibleCount);

        // Render splats → render target (offscreen for CAS, canvas otherwise)
        bool useCas = _sharpeningStrength > 0f;
        var renderTarget = useCas ? _offscreenView! : colorView;

        using var splatPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
        {
            ColorAttachments = new[]
            {
                new GPURenderPassColorAttachment
                {
                    View = renderTarget,
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

        splatPass.SetPipeline(_splatPipeline);
        splatPass.SetBindGroup(0, _uniformBindGroup!);
        splatPass.SetVertexBuffer(0, _splatBuffer);
        // Draw only visible splats (deferred count from previous frame's GPU readback).
        // Sorted splats are [0..visibleCount-1] visible + [visibleCount..N-1] sentinels (-1).
        // Drawing visibleCount skips the sentinel range entirely → no fragment shader overhead.
        splatPass.Draw(6, (uint)visibleCount, 0, 0);
        splatPass.End();

        if (useCas)
        {
            _casData[0] = _sharpeningStrength;
            _casData[1] = 1f / _canvasWidth;
            _casData[2] = 1f / _canvasHeight;
            _casData[3] = 0f;
            _casJsArray!.Set(_casData);
            _queue.WriteBuffer(_casUniformBuffer!, 0, _casJsArray);

            using var casPass = encoder.BeginRenderPass(new GPURenderPassDescriptor
            {
                ColorAttachments = new[]
                {
                    new GPURenderPassColorAttachment
                    {
                        View = colorView,
                        LoadOp = GPULoadOp.Clear,
                        StoreOp = GPUStoreOp.Store,
                        ClearValue = new GPUColorDict { R = 0.0, G = 0.0, B = 0.0, A = 1.0 },
                    }
                }
            });
            casPass.SetPipeline(_casPipeline!);
            casPass.SetBindGroup(0, _casBindGroup!);
            casPass.Draw(3, 1, 0, 0);
            casPass.End();
        }

        using var commandBuffer = encoder.Finish();
        _queue!.Submit(new[] { commandBuffer });
        // No explicit GPU sync needed. WebGPU's getCurrentTexture() (called at the
        // top of the next frame) will not return until the swapchain texture is free,
        // naturally throttling CPU to ~1 frame ahead without a full GPU drain.
    }

    /// <summary>
    /// Appends a pack compute pass to the supplied encoder.
    /// Converts ILGPU Float32 splat data → packed vertex buffer using the sorted index buffer.
    /// Only packs visibleCount splats — culled sentinels at [visibleCount..N-1] are skipped.
    /// Caller is responsible for submitting the encoder.
    /// Called only on frames where the sort ran (indices changed).
    /// </summary>
    private void AppendPackComputePass(
        GPUCommandEncoder encoder,
        MemoryBuffer1D<float, Stride1D.Dense> dataBuf,
        MemoryBuffer1D<int, Stride1D.Dense> idxBuf,
        int visibleCount)
    {
        if (_splatBuffer == null || _device == null || _packPipeline == null || _packCountBuf == null) return;

        var dataView = (IArrayView)(MemoryBuffer)dataBuf;
        if (dataView.Buffer is not WebGPUMemoryBuffer dataMem) return;
        var srcDataBuffer = dataMem.NativeBuffer?.NativeBuffer;
        if (srcDataBuffer == null) return;

        var idxView = (IArrayView)(MemoryBuffer)idxBuf;
        if (idxView.Buffer is not WebGPUMemoryBuffer idxMem) return;
        var srcIdxBuffer = idxMem.NativeBuffer?.NativeBuffer;
        if (srcIdxBuffer == null) return;

        // Write visible count to uniform buffer (before encoder submit, queue.writeBuffer runs first).
        _packCountJsArray![0] = (uint)visibleCount;
        _queue!.WriteBuffer(_packCountBuf, 0, _packCountJsArray);

        // Create or reuse pack bind group (invalidate only when GPU buffer refs change)
        if (_packBindGroup == null || _srcDataCached != srcDataBuffer || _srcIdxCached != srcIdxBuffer)
        {
            _packBindGroup?.Dispose();
            _srcDataCached = srcDataBuffer;
            _srcIdxCached = srcIdxBuffer;

            using var layout = _packPipeline.GetBindGroupLayout(0);
            _packBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = layout,
                Entries = new GPUBindGroupEntry[]
                {
                    new() { Binding = 0, Resource = new GPUBufferBinding { Buffer = srcDataBuffer } },
                    new() { Binding = 1, Resource = new GPUBufferBinding { Buffer = srcIdxBuffer } },
                    new() { Binding = 2, Resource = new GPUBufferBinding { Buffer = _splatBuffer } },
                    new() { Binding = 3, Resource = new GPUBufferBinding { Buffer = _packCountBuf } },
                }
            });
        }

        // Dispatch only over visible splats (culled sentinels at tail are skipped).
        uint workgroups = (uint)((visibleCount + 63) / 64);
        using var pass = encoder.BeginComputePass();
        pass.SetPipeline(_packPipeline);
        pass.SetBindGroup(0, _packBindGroup);
        pass.DispatchWorkgroups(workgroups, 1, 1);
        pass.End();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _splatBuffer?.Destroy();
        _splatBuffer?.Dispose();
        _uniformBuffer?.Destroy();
        _uniformBuffer?.Dispose();
        _uniformBindGroup?.Dispose();
        _depthTexture?.Destroy();
        _depthTexture?.Dispose();
        _depthView?.Dispose();
        _offscreenTexture?.Destroy();
        _offscreenTexture?.Dispose();
        _offscreenView?.Dispose();
        _casBindGroup?.Dispose();
        _casUniformBuffer?.Destroy();
        _casUniformBuffer?.Dispose();
        _casSampler?.Dispose();
        _splatPipeline?.Dispose();
        _casPipeline?.Dispose();
        _uniformJsArray?.Dispose();
        _casJsArray?.Dispose();
        _packCountBuf?.Destroy();
        _packCountBuf?.Dispose();
        _packCountJsArray?.Dispose();
    }

    /// <summary>
    /// Create a perspective projection matrix for WebGPU (clip-space Z = [0, 1]).
    /// Compatible with System.Numerics.Matrix4x4.CreateLookAt (right-handed view space).
    /// In right-handed view space, objects in front have Z < 0, so w = -z_eye.
    /// </summary>
    private static Matrix4x4 CreateWebGPUPerspective(float fovY, float aspect, float near, float far)
    {
        float f = 1.0f / MathF.Tan(fovY * 0.5f);
        float rangeInv = 1.0f / (near - far); // Note: near - far (negative)
        return new Matrix4x4(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * rangeInv, -1,  // -1 for right-handed → w = -z_eye
            0, 0, near * far * rangeInv, 0
        );
    }

    // ════════════════════════════════════════════════════════════
    //  WGSL Splat Shader — with EWA anti-aliasing filter
    // ════════════════════════════════════════════════════════════
    private const string SplatShaderSource = @"
struct Uniforms {
    mvp      : mat4x4<f32>,
    viewport : vec2<f32>,
    focal    : vec2<f32>,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) position    : vec3<f32>,
    @location(1) color_alpha : vec4<f32>,  // Unorm8x4: RGBA packed as 4 bytes
    @location(2) scale       : vec4<f32>,  // Float16x4: sx,sy,sz,pad
};

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) color   : vec3<f32>,
    @location(1) opacity : f32,
    @location(2) uv      : vec2<f32>,
};

@vertex
fn vs_main(
    input : VertexInput,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32
) -> VertexOutput {
    // Billboard quad vertices (2 triangles)
    var quad_pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0)
    );

    let uv = quad_pos[vid];

    // Project Gaussian center to clip space
    let center_clip = u.mvp * vec4<f32>(input.position, 1.0);

    var out : VertexOutput;
    // ── Frustum Culling (all 6 planes) ──
    // Discard behind camera
    if (center_clip.w <= 0.001) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = input.color_alpha.rgb;
        out.opacity = 0.0;
        out.uv = uv;
        return out;
    }

    // NDC center
    let ndc_center = center_clip.xyz / center_clip.w;

    // ── Anisotropic splat: use separate X and Y scales ──
    let scale_x = max(input.scale.x, 0.001);
    let scale_y = max(input.scale.y, 0.001);

    // Project each axis to screen pixels: pixels = world_size * focal / depth
    let screen_rx = scale_x * u.focal.x / center_clip.w;
    let screen_ry = scale_y * u.focal.y / center_clip.w;

    // EWA Anti-Alias Filter: minimum 0.8px radius per axis
    let ewa_rx = max(screen_rx, 0.8);
    let ewa_ry = max(screen_ry, 0.8);

    // Convert pixel radii to NDC
    let ndc_radius_x = ewa_rx * 2.0 / u.viewport.x;
    let ndc_radius_y = ewa_ry * 2.0 / u.viewport.y;
    let ndc_radius_max = max(ndc_radius_x, ndc_radius_y) * 3.0; // 3σ cutoff

    // Frustum cull: discard if splat (including its radius) is entirely outside NDC cube
    if (ndc_center.x + ndc_radius_max < -1.0 || ndc_center.x - ndc_radius_max > 1.0 ||
        ndc_center.y + ndc_radius_max < -1.0 || ndc_center.y - ndc_radius_max > 1.0 ||
        ndc_center.z < -0.1 || ndc_center.z > 1.1) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = input.color_alpha.rgb;
        out.opacity = 0.0;
        out.uv = uv;
        return out;
    }

    // Offset quad vertex from center (3x radius = Gaussian cutoff at 3 sigma)
    let offset_ndc = uv * vec2<f32>(ndc_radius_x, ndc_radius_y) * 3.0;
    let final_ndc = vec3<f32>(ndc_center.xy + offset_ndc, ndc_center.z);

    out.clip_pos = vec4<f32>(final_ndc * center_clip.w, center_clip.w);
    out.color = input.color_alpha.rgb;
    out.opacity = input.color_alpha.a;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // Gaussian alpha falloff: exp(-dist^2 / 2)
    let dist_sq = dot(input.uv, input.uv);

    // Discard pixels outside the Gaussian radius
    if (dist_sq > 9.0) {
        discard;
    }

    let alpha = input.opacity * exp(-dist_sq * 0.5);

    // Minimum alpha threshold
    if (alpha < 0.002) {
        discard;
    }

    return vec4<f32>(input.color, alpha);
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL CAS (Contrast Adaptive Sharpening) Post-Processing
    // ════════════════════════════════════════════════════════════
    private const string CasShaderSource = @"
struct CASUniforms {
    strength   : f32,
    texel_x    : f32,
    texel_y    : f32,
    _padding   : f32,
};

@group(0) @binding(0) var t_color : texture_2d<f32>;
@group(0) @binding(1) var s_color : sampler;
@group(0) @binding(2) var<uniform> cas : CASUniforms;

struct VSOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> VSOutput {
    // Fullscreen triangle (covers entire screen with 3 vertices)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let pos = positions[vid];
    var out : VSOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    // Flip Y for WebGPU UV convention
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_cas(input : VSOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let tx = cas.texel_x;
    let ty = cas.texel_y;

    // Sample center and 8 neighbors (cross + diagonals)
    let c  = textureSample(t_color, s_color, uv);
    let n  = textureSample(t_color, s_color, uv + vec2<f32>(0.0, -ty));
    let s  = textureSample(t_color, s_color, uv + vec2<f32>(0.0,  ty));
    let e  = textureSample(t_color, s_color, uv + vec2<f32>( tx, 0.0));
    let w  = textureSample(t_color, s_color, uv + vec2<f32>(-tx, 0.0));
    let ne = textureSample(t_color, s_color, uv + vec2<f32>( tx, -ty));
    let nw = textureSample(t_color, s_color, uv + vec2<f32>(-tx, -ty));
    let se = textureSample(t_color, s_color, uv + vec2<f32>( tx,  ty));
    let sw = textureSample(t_color, s_color, uv + vec2<f32>(-tx,  ty));

    // CAS: find min/max of full 8-neighbor pattern
    let mn_cross = min(min(n, s), min(e, w));
    let mx_cross = max(max(n, s), max(e, w));
    let mn_diag  = min(min(ne, nw), min(se, sw));
    let mx_diag  = max(max(ne, nw), max(se, sw));
    let mn = min(mn_cross, mn_diag);
    let mx = max(mx_cross, mx_diag);

    // Adaptive sharpening weight (sharper where neighbors are similar)
    let amp = clamp(min(mn, vec4<f32>(2.0) - mx) / mx, vec4<f32>(0.0), vec4<f32>(1.0));
    let sharp = amp * cas.strength;

    // Weighted average: cross neighbors 2x weight, diagonals 1x (total = 12)
    let avg = (n + s + e + w) * 0.166666 + (ne + nw + se + sw) * 0.083333;
    let result = mix(c, c + (c - avg) * sharp, vec4<f32>(cas.strength));

    return vec4<f32>(clamp(result.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL Pack Compute — Float32 sort output → packed vertex format
    //  Input:  10 floats per splat (pos3, color3, scale3, opacity1)
    //  Output: 6 u32s per splat (pos3_bitcast, color_alpha_u8x4, scale_f16x4)
    //  = 24 bytes per splat (was 40 bytes)
    // ════════════════════════════════════════════════════════════
    private const string PackComputeSource = @"
@group(0) @binding(0) var<storage, read>       src     : array<f32>;  // original packed splat data (10 floats/splat)
@group(0) @binding(1) var<storage, read>       idx     : array<i32>;  // sorted indices; -1 = culled sentinel
@group(0) @binding(2) var<storage, read_write> dst     : array<u32>;  // packed vertex output (6 u32s/splat)
@group(0) @binding(3) var<uniform>             u_count : u32;         // visible splat count (deferred readback)

@compute @workgroup_size(64)
fn pack_splats(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= u_count) { return; }

    let dstOff = i * 6u;

    // Culled splats have idx=-1 sentinel (sorted last by DescendingInt32).
    // Write a fully-transparent vertex so the fragment shader discards it cheaply.
    let origIdx = idx[i];
    if (origIdx < 0) {
        dst[dstOff + 0u] = 0u;
        dst[dstOff + 1u] = 0u;
        dst[dstOff + 2u] = 0u;
        dst[dstOff + 3u] = 0u;  // opacity = 0 → fragment discard
        dst[dstOff + 4u] = 0u;
        dst[dstOff + 5u] = 0u;
        return;
    }

    // Index lookup: maps sorted position i to original splat data — eliminates CPU reorder pass
    let srcOff = u32(origIdx) * 10u;

    // Position: 3 floats bitcast to 3 u32s (preserve full precision)
    dst[dstOff + 0u] = bitcast<u32>(src[srcOff + 0u]);  // pos.x
    dst[dstOff + 1u] = bitcast<u32>(src[srcOff + 1u]);  // pos.y
    dst[dstOff + 2u] = bitcast<u32>(src[srcOff + 2u]);  // pos.z

    // Color + opacity: pack RGBA as 4 normalized bytes (Unorm8x4)
    let color_alpha = vec4<f32>(
        src[srcOff + 3u],   // R
        src[srcOff + 4u],   // G
        src[srcOff + 5u],   // B
        src[srcOff + 9u]    // opacity
    );
    dst[dstOff + 3u] = pack4x8unorm(color_alpha);

    // Scale: pack as Float16x4 (sx, sy, sz, 0)
    dst[dstOff + 4u] = pack2x16float(vec2<f32>(src[srcOff + 6u], src[srcOff + 7u]));
    dst[dstOff + 5u] = pack2x16float(vec2<f32>(src[srcOff + 8u], 0.0));
}
";
}
