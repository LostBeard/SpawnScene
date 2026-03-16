using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>Controls whether adaptive half-resolution mode is applied.</summary>
public enum AdaptiveResMode
{
    Auto,       // velocity-gated: enters half-res above LowResEnterVelocity
    ForceFull,  // always render at full physical resolution
    ForceHalf,  // always render at half physical resolution
}

/// <summary>Controls the splat rendering technique.</summary>
public enum SplatRenderMode
{
    /// <summary>Traditional sorted alpha blending (cull → radix sort → pack → render).</summary>
    Sorted,
    /// <summary>Sort-free stochastic rasterization with temporal accumulation (StochasticSplats, ICCV 2025).</summary>
    Stochastic,
}

/// <summary>
/// Native WebGPU Gaussian splat renderer with GPU-sorted splats.
/// Architecture:
///   - ILGPU = data generation + sorting (GpuSplatSorter)
///   - WebGPU = rendering only (this class)
///   - GPU-side buffer copy (no CPU round-trips)
///   - EWA anti-alias filter for distant splats
///   - CAS post-processing sharpening
///   - Adaptive resolution: halves canvas pixel dims during fast movement, restores when still
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

    // Uniform buffer: MVP matrix (64 bytes) + viewport (8 bytes) + focal (8 bytes) + frame_index (4 bytes) + pad (12 bytes) = 96 bytes
    private GPUBuffer? _uniformBuffer;
    private GPUBindGroup? _uniformBindGroup;
    private readonly float[] _uniformData = new float[24]; // 16 (mat4) + 2 (viewport) + 2 (focal) + 1 (frame_index) + 3 (pad)
    private byte[]? _uniformByteData; // pre-allocated byte mirror of _uniformData for direct WriteBuffer

    // CAS sharpening pass
    private GPUTexture? _offscreenTexture;
    private GPUTextureView? _offscreenView;
    private GPUBindGroup? _casBindGroup;
    private GPUBuffer? _casUniformBuffer;
    private GPUSampler? _casSampler;
    private float _sharpeningStrength = 0.5f;
    private readonly float[] _casData = new float[4];
    private byte[]? _casByteData; // pre-allocated byte mirror of _casData for direct WriteBuffer

    // Stochastic rasterization — sort-free rendering with temporal accumulation
    private GPURenderPipeline? _stochasticSplatPipeline;
    private GPUTexture? _stochasticTexture;      // per-frame stochastic render target (cleared each frame)
    private GPUTextureView? _stochasticView;
    private GPUTexture? _accumTexture;            // persistent accumulation texture (NOT cleared per frame)
    private GPUTextureView? _accumView;
    private GPURenderPipeline? _accumPipeline;    // fullscreen blend pass for temporal accumulation
    private GPUBindGroup? _accumBindGroup;        // samples _stochasticTexture
    private GPUBuffer? _accumUniformBuffer;       // accumulation weight uniform
    private readonly float[] _accumData = new float[4]; // weight + padding
    private byte[]? _accumByteData;
    private int _accumFrameCount;                 // velocity-adaptive: capped by movement speed
    private int _globalFrameCount;                // monotonically increasing, never resets (hash seed)

    // Velocity-adaptive dilation: subtle splat fattening to bridge sub-pixel spatial gaps
    private const float DilationScale = 5f;       // sqrt(velocity) * DilationScale
    private const float MaxDilationFactor = 0.05f; // max additional scale (0.05 = up to 5% larger)

    // Multi-SPP: render multiple stochastic passes per frame for faster convergence
    /// <summary>Max samples per pixel per frame (1-4). Higher = faster convergence, lower FPS. Default 2 is optimal for 60fps.</summary>
    public int StochasticSPP { get; set; } = 2;

    // Cached render pass descriptors for stochastic mode
    private GPURenderPassColorAttachment? _stochasticColorAttach;
    private GPURenderPassDescriptor? _stochasticPassDesc;
    private GPURenderPassColorAttachment? _accumColorAttach;
    private GPURenderPassDescriptor? _accumPassDesc;

    // Depth texture
    private GPUTexture? _depthTexture;
    private GPUTextureView? _depthView;

    private int _canvasWidth;
    private int _canvasHeight;

    // Adaptive resolution — physical dims track the true canvas pixel size.
    // _canvasWidth/_canvasHeight may be half of physical during fast movement.
    private ElementReference _canvasRef;
    private int _physicalWidth;
    private int _physicalHeight;
    private bool _lowResActive;
    // Thresholds calibrated for per-frame DistanceSquared (see GpuSplatSorter for scale reference).
    // Fast movement (aggressive mouse or Shift+WASD) pushes _smoothedVelocity above 0.0001.
    private const float LowResEnterVelocity = 0.0002f; // enter half-res when velocity exceeds this
    private const float LowResExitVelocity  = 0.00005f; // exit half-res once velocity drops below this (hysteresis)

    private bool _disposed;

    // Reused 1-element array for Submit (avoids per-frame allocation)
    private static readonly GPUCommandBuffer[] _submitArray = new GPUCommandBuffer[1];

    // Cached render pass descriptors — rebuilt on resize, reused every frame.
    // GPURenderPassColorAttachment.View is { get; set; } so we update it per frame for swapchain targets.
    // GPURenderPassDepthStencilAttachment.View is { get; init; } so we recreate on resize.
    private GPURenderPassColorAttachment? _splatColorAttachCas;    // View = _offscreenView (stable)
    private GPURenderPassColorAttachment? _splatColorAttachDirect; // View updated per frame
    private GPURenderPassColorAttachment? _casColorAttach;          // View updated per frame
    private GPURenderPassDescriptor? _splatPassDescCas;            // fully stable (CAS path)
    private GPURenderPassDescriptor? _splatPassDescDirect;          // color View updated per frame
    private GPURenderPassDescriptor? _casPassDesc;                  // color View updated per frame

    /// <summary>Sharpening intensity (0 = off, 1 = maximum).</summary>
    public float SharpeningStrength
    {
        get => _sharpeningStrength;
        set => _sharpeningStrength = Math.Clamp(value, 0f, 1f);
    }

    /// <summary>Controls adaptive resolution behavior.</summary>
    public AdaptiveResMode AdaptiveResMode { get; set; } = AdaptiveResMode.Auto;

    /// <summary>Controls whether to use sorted alpha blending or stochastic rasterization.</summary>
    public SplatRenderMode RenderMode { get; set; } = SplatRenderMode.Stochastic;

    /// <summary>Sort precision passthrough: true = 4-pass 16-bit (faster), false = 8-pass 32-bit.</summary>
    public bool Use16BitSort
    {
        get => _sorter.Use16BitSort;
        set => _sorter.Use16BitSort = value;
    }

    /// <summary>Diagnostic: skip radix sort entirely (render unsorted).</summary>
    public bool SkipSort
    {
        get => _sorter.SkipSort;
        set => _sorter.SkipSort = value;
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
    /// <paramref name="canvasRef"/> is stored for adaptive-resolution canvas pixel resizing.
    /// </summary>
    public void AttachCanvas(HTMLCanvasElement canvas, ElementReference canvasRef)
    {
        _canvasRef = canvasRef;

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

        _physicalWidth = canvas.Width;
        _physicalHeight = canvas.Height;
        _canvasWidth = _physicalWidth;
        _canvasHeight = _physicalHeight;

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

        // ── Stochastic Splat Pipeline (sort-free, depth-tested, opaque writes) ──
        var splatVertexBuffers = new[]
        {
            new GPUVertexBufferLayout
            {
                ArrayStride = (ulong)PackedBytesPerSplat,
                StepMode = GPUVertexStepMode.Instance,
                Attributes = new GPUVertexAttribute[]
                {
                    new() { ShaderLocation = 0, Offset = 0,  Format = GPUVertexFormat.Float32x3 },
                    new() { ShaderLocation = 1, Offset = 12, Format = GPUVertexFormat.UNorm8x4 },
                    new() { ShaderLocation = 2, Offset = 16, Format = GPUVertexFormat.Float16x4 },
                }
            }
        };
        using var stochasticShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = StochasticSplatShaderSource });
        _stochasticSplatPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState
            {
                Module = stochasticShader,
                EntryPoint = "vs_main",
                Buffers = splatVertexBuffers,
            },
            Fragment = new GPUFragmentState
            {
                Module = stochasticShader,
                EntryPoint = "fs_main",
                Targets = new[]
                {
                    new GPUColorTargetState { Format = _canvasFormat } // No blend — opaque writes
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
            DepthStencil = new GPUDepthStencilState
            {
                Format = "depth24plus",
                DepthWriteEnabled = true, // Stochastic NEEDS depth writes (hardware selects closest surviving sample)
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

        // ── Accumulation Pipeline (fullscreen EMA blend for temporal convergence) ──
        using var accumShader = _device.CreateShaderModule(new GPUShaderModuleDescriptor { Code = AccumulateShaderSource });
        _accumPipeline = _device.CreateRenderPipeline(new GPURenderPipelineDescriptor
        {
            Layout = "auto",
            Vertex = new GPUVertexState { Module = accumShader, EntryPoint = "vs_fullscreen" },
            Fragment = new GPUFragmentState
            {
                Module = accumShader,
                EntryPoint = "fs_accum",
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
                                DstFactor = GPUBlendFactor.Zero,
                                Operation = GPUBlendOperation.Add,
                            }
                        }
                    }
                }
            },
            Primitive = new GPUPrimitiveState { Topology = GPUPrimitiveTopology.TriangleList },
        });

        // Depth texture
        CreateDepthTexture();

        // Uniform buffer (96 bytes: mat4(64) + viewport(8) + focal(8) + frame_index(4) + pad(12))
        _uniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 96,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });

        // Both sorted and stochastic pipelines share the same uniform layout
        _uniformBindGroup = _device.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _stochasticSplatPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry
                {
                    Binding = 0,
                    Resource = new GPUBufferBinding { Buffer = _uniformBuffer }
                }
            }
        });

        // Pre-allocate reusable byte buffers for direct WriteBuffer — avoids HeapView/PrimeHeap on every frame
        _uniformByteData = new byte[_uniformData.Length * sizeof(float)];
        _casByteData = new byte[_casData.Length * sizeof(float)];
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

        // Accumulation uniform (16 bytes aligned: weight + padding)
        _accumUniformBuffer = _device.CreateBuffer(new GPUBufferDescriptor
        {
            Size = 16,
            Usage = GPUBufferUsage.Uniform | GPUBufferUsage.CopyDst,
        });
        _accumByteData = new byte[_accumData.Length * sizeof(float)];

        // Offscreen texture for CAS input (must be after CAS resources are created)
        CreateOffscreenTexture();

        // Stochastic textures (render target + accumulation)
        CreateStochasticTextures();

        // If splat data was uploaded before the canvas was attached, create the vertex buffer now.
        EnsureSplatBuffer();
        // Pack vertex buffer if upload already happened (deferred pack-at-upload)
        PackAtUpload();

        Console.WriteLine($"[GpuRenderer] Pipeline created: sorted + stochastic + CAS. Format: {_canvasFormat}");
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
        RebuildCachedDescriptors();
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
        RebuildCachedDescriptors();

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

    private void CreateStochasticTextures()
    {
        _stochasticView?.Dispose();
        _stochasticTexture?.Destroy();
        _stochasticTexture?.Dispose();
        _accumView?.Dispose();
        _accumTexture?.Destroy();
        _accumTexture?.Dispose();

        var desc = new GPUTextureDescriptor
        {
            Size = new[] { _canvasWidth, _canvasHeight },
            Format = _canvasFormat,
            Usage = GPUTextureUsage.RenderAttachment | GPUTextureUsage.TextureBinding,
        };

        _stochasticTexture = _device!.CreateTexture(desc);
        _stochasticView = _stochasticTexture.CreateView();
        _accumTexture = _device.CreateTexture(desc);
        _accumView = _accumTexture.CreateView();

        _accumFrameCount = 0;

        RebuildStochasticDescriptors();
    }

    /// <summary>Rebuild stochastic render pass descriptors and bind groups after texture recreation.</summary>
    private void RebuildStochasticDescriptors()
    {
        if (_stochasticView == null || _depthView == null || _accumView == null) return;

        // Stochastic splat pass → _stochasticTexture (cleared each frame, with depth)
        _stochasticColorAttach = new GPURenderPassColorAttachment
        {
            View = _stochasticView,
            LoadOp = GPULoadOp.Clear,
            StoreOp = GPUStoreOp.Store,
            ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.10, A = 1.0 },
        };
        _stochasticPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _stochasticColorAttach },
            DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
            {
                View = _depthView,
                DepthLoadOp = "clear",
                DepthStoreOp = "store",
                DepthClearValue = 1.0f,
            },
        };

        // Accumulation pass → _accumTexture (LoadOp toggled per frame: clear on reset, load normally)
        _accumColorAttach = new GPURenderPassColorAttachment
        {
            View = _accumView,
            LoadOp = GPULoadOp.Clear, // toggled per frame
            StoreOp = GPUStoreOp.Store,
            ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.10, A = 1.0 },
        };
        _accumPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _accumColorAttach },
        };

        // Accumulation bind group: samples _stochasticTexture
        if (_accumPipeline != null && _casSampler != null && _accumUniformBuffer != null)
        {
            _accumBindGroup?.Dispose();
            _accumBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
            {
                Layout = _accumPipeline.GetBindGroupLayout(0),
                Entries = new[]
                {
                    new GPUBindGroupEntry { Binding = 0, Resource = _stochasticView },
                    new GPUBindGroupEntry { Binding = 1, Resource = _casSampler }, // reuse sampler
                    new GPUBindGroupEntry
                    {
                        Binding = 2,
                        Resource = new GPUBufferBinding { Buffer = _accumUniformBuffer }
                    }
                }
            });
        }

        // CAS reads _accumTexture in stochastic mode (rebuilt here since _accumView changed)
        RebuildCasBindGroupForAccum();
    }

    /// <summary>Rebuild CAS bind group to sample _accumTexture (for stochastic display path).</summary>
    private void RebuildCasBindGroupForAccum()
    {
        if (_casPipeline == null || _casSampler == null || _casUniformBuffer == null || _accumView == null) return;

        // In stochastic mode, CAS reads from accumulation texture instead of offscreen.
        // We rebuild _casBindGroup to point to _accumView. When switching back to sorted mode,
        // CreateOffscreenTexture() will rebuild it to point to _offscreenView again.
        _casBindGroup?.Dispose();
        _casBindGroup = _device!.CreateBindGroup(new GPUBindGroupDescriptor
        {
            Layout = _casPipeline.GetBindGroupLayout(0),
            Entries = new[]
            {
                new GPUBindGroupEntry { Binding = 0, Resource = _accumView },
                new GPUBindGroupEntry { Binding = 1, Resource = _casSampler },
                new GPUBindGroupEntry
                {
                    Binding = 2,
                    Resource = new GPUBufferBinding { Buffer = _casUniformBuffer }
                }
            }
        });
    }

    /// <summary>
    /// Rebuilds the cached render pass descriptor objects.
    /// Called after depth or offscreen texture recreation (canvas resize).
    /// On cache hit frames, these objects are reused directly — only the swapchain View field
    /// is updated per frame in Render() for the direct and CAS pass targets.
    /// </summary>
    private void RebuildCachedDescriptors()
    {
        // Direct splat pass (no CAS) — depth stencil is stable; color View updated per frame
        if (_depthView != null)
        {
            _splatColorAttachDirect = new GPURenderPassColorAttachment
            {
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.10, A = 1.0 },
            };
            _splatPassDescDirect = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { _splatColorAttachDirect },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = _depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
        }

        // CAS splat pass — renders to offscreen texture (fully stable, no per-frame update needed)
        if (_offscreenView != null && _depthView != null)
        {
            _splatColorAttachCas = new GPURenderPassColorAttachment
            {
                View = _offscreenView,
                LoadOp = GPULoadOp.Clear,
                StoreOp = GPUStoreOp.Store,
                ClearValue = new GPUColorDict { R = 0.04, G = 0.04, B = 0.10, A = 1.0 },
            };
            _splatPassDescCas = new GPURenderPassDescriptor
            {
                ColorAttachments = new[] { _splatColorAttachCas },
                DepthStencilAttachment = new GPURenderPassDepthStencilAttachment
                {
                    View = _depthView,
                    DepthLoadOp = "clear",
                    DepthStoreOp = "store",
                    DepthClearValue = 1.0f,
                },
            };
        }

        // CAS post-process pass — renders to swapchain; color View updated per frame
        _casColorAttach = new GPURenderPassColorAttachment
        {
            LoadOp = GPULoadOp.Clear,
            StoreOp = GPUStoreOp.Store,
            ClearValue = new GPUColorDict { R = 0.0, G = 0.0, B = 0.0, A = 1.0 },
        };
        _casPassDesc = new GPURenderPassDescriptor
        {
            ColorAttachments = new[] { _casColorAttach },
        };
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

        EnsureSplatBuffer();

        // Fill identity indices and pack vertex buffer at upload time (for stochastic mode)
        await _sorter.FillIdentityIndicesAsync();
        PackAtUpload();

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

        EnsureSplatBuffer();

        // Fill identity indices and pack vertex buffer at upload time (for stochastic mode)
        await _sorter.FillIdentityIndicesAsync();
        PackAtUpload();

        Console.WriteLine($"[GpuRenderer] GPU fast-path upload: {_splatCount:N0} splats" +
            (_splatBuffer != null ? $", {_splatCount * PackedBytesPerSplat / 1024}KB vertex buffer" : " (vertex buffer deferred)"));
    }

    /// <summary>
    /// One-time pack at upload: converts ILGPU Float32 data → packed vertex buffer using identity indices.
    /// Used by stochastic mode so the vertex buffer is ready without per-frame sort+pack.
    /// Also provides initial data for sorted mode's first frame before sort completes.
    /// </summary>
    private void PackAtUpload()
    {
        var dataBuf = _sorter.PackedDataBuf;
        var idxBuf = _sorter.IndicesBuf;
        if (_device == null || _splatBuffer == null || _packPipeline == null || dataBuf == null || idxBuf == null) return;

        using var encoder = _device.CreateCommandEncoder();
        AppendPackComputePass(encoder, dataBuf, idxBuf, _splatCount);
        using var cmdBuf = encoder.Finish();
        _submitArray[0] = cmdBuf;
        _queue!.Submit(_submitArray);

        Console.WriteLine($"[GpuRenderer] Pack-at-upload complete: {_splatCount:N0} splats packed");
    }

    /// <summary>
    /// Resize canvas and recreate GPU textures for new dimensions.
    /// Called when the browser window is resized.
    /// Preserves the current adaptive resolution mode (low-res active = render at half the new physical size).
    /// </summary>
    public void ResizeCanvas(int newWidth, int newHeight)
    {
        if (_device == null || newWidth <= 0 || newHeight <= 0) return;

        _physicalWidth = newWidth;
        _physicalHeight = newHeight;

        int renderW = _lowResActive ? Math.Max(1, newWidth / 2) : newWidth;
        int renderH = _lowResActive ? Math.Max(1, newHeight / 2) : newHeight;

        if (renderW == _canvasWidth && renderH == _canvasHeight) return;

        _canvasWidth = renderW;
        _canvasHeight = renderH;

        CreateDepthTexture();
        CreateOffscreenTexture();
        CreateStochasticTextures();

        Console.WriteLine($"[GpuRenderer] Resized GPU textures: {renderW}×{renderH} (physical: {newWidth}×{newHeight})");
    }

    /// <summary>
    /// Render one frame. Fully synchronous — no GPU drain.
    /// Sorted mode: GPU sort → GPU pack → splat render → CAS sharpen.
    /// Stochastic mode: stochastic render → temporal accumulate → CAS display.
    /// </summary>
    private bool _renderLogged;
    public void Render(GaussianScene scene, CameraParams camera)
    {
        if (_device == null || _context == null || _splatBuffer == null || _splatCount == 0)
        {
            if (!_renderLogged)
            {
                _renderLogged = true;
                Console.WriteLine($"[Render] Early exit: device={_device != null} ctx={_context != null} buf={_splatBuffer != null} count={_splatCount}");
            }
            return;
        }
        if (!_renderLogged) { _renderLogged = true; Console.WriteLine($"[Render] First frame ({RenderMode}): {_splatCount} splats, cam={camera.Position} fwd={camera.Forward}"); }

        // ── Adaptive Resolution: enter/exit half-res based on mode and camera velocity ──
        {
            float velocity = _sorter.SmoothedVelocity;
            bool wantLowRes = AdaptiveResMode switch
            {
                AdaptiveResMode.ForceFull => false,
                AdaptiveResMode.ForceHalf => true,
                _                         => _lowResActive ? velocity > LowResExitVelocity : velocity > LowResEnterVelocity,
            };
            if (wantLowRes != _lowResActive && _physicalWidth > 0)
            {
                _lowResActive = wantLowRes;
                int rw = _lowResActive ? Math.Max(1, _physicalWidth / 2) : _physicalWidth;
                int rh = _lowResActive ? Math.Max(1, _physicalHeight / 2) : _physicalHeight;

                using var canvasEl = new HTMLCanvasElement(_canvasRef);
                canvasEl.Width = rw;
                canvasEl.Height = rh;

                _canvasWidth = rw;
                _canvasHeight = rh;
                CreateDepthTexture();
                CreateOffscreenTexture();
                CreateStochasticTextures();

                Console.WriteLine($"[GpuRenderer] Adaptive res: {(wantLowRes ? "LOW" : "FULL")} {rw}×{rh} (velocity={velocity:F4})");
            }
        }

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

        // ── Build MVP (needed by both modes) ──
        var view = camera.ViewMatrix;
        float fovY = 2f * MathF.Atan(camera.Height / (2f * camera.FocalY));
        float aspect = (float)camera.Width / camera.Height;
        var proj = CreateWebGPUPerspective(fovY, aspect, camera.Near, camera.Far);
        var mvp = view * proj;

        // ── Upload MVP + viewport uniforms ──
        _uniformData[0] = mvp.M11; _uniformData[1] = mvp.M12; _uniformData[2] = mvp.M13; _uniformData[3] = mvp.M14;
        _uniformData[4] = mvp.M21; _uniformData[5] = mvp.M22; _uniformData[6] = mvp.M23; _uniformData[7] = mvp.M24;
        _uniformData[8] = mvp.M31; _uniformData[9] = mvp.M32; _uniformData[10] = mvp.M33; _uniformData[11] = mvp.M34;
        _uniformData[12] = mvp.M41; _uniformData[13] = mvp.M42; _uniformData[14] = mvp.M43; _uniformData[15] = mvp.M44;
        float focalScaleX = camera.Width > 0 ? (float)_canvasWidth / camera.Width : 1f;
        float focalScaleY = camera.Height > 0 ? (float)_canvasHeight / camera.Height : 1f;
        _uniformData[16] = _canvasWidth;
        _uniformData[17] = _canvasHeight;
        _uniformData[18] = camera.FocalX * focalScaleX;
        _uniformData[19] = camera.FocalY * focalScaleY;

        if (RenderMode == SplatRenderMode.Stochastic)
        {
            RenderStochastic(camera, mvp);
        }
        else
        {
            RenderSorted(camera, mvp);
        }
    }

    /// <summary>Sorted alpha-blend rendering: cull → sort → pack → render → optional CAS.</summary>
    private void RenderSorted(CameraParams camera, Matrix4x4 mvp)
    {
        var (dataBuf, idxBuf, sortRan, visibleCount) = _sorter.Sort(camera, mvp);

        // Upload uniforms (frame_index/dilation/min_alpha not used in sorted mode)
        _uniformData[20] = 0f;
        _uniformData[21] = 1f; // no dilation
        _uniformData[22] = 0f; // no alpha floor
        Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
        _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

        using var colorTexture = _context!.GetCurrentTexture();
        using var colorView = colorTexture.CreateView();
        using var encoder = _device!.CreateCommandEncoder();

        if (sortRan && dataBuf != null && idxBuf != null)
            AppendPackComputePass(encoder, dataBuf, idxBuf, visibleCount);

        bool useCas = _sharpeningStrength > 0f && !_lowResActive;
        GPURenderPassDescriptor splatPassDesc;
        if (useCas)
        {
            splatPassDesc = _splatPassDescCas!;
        }
        else
        {
            _splatColorAttachDirect!.View = colorView;
            splatPassDesc = _splatPassDescDirect!;
        }

        using var splatPass = encoder.BeginRenderPass(splatPassDesc);
        splatPass.SetPipeline(_splatPipeline!);
        splatPass.SetBindGroup(0, _uniformBindGroup!);
        splatPass.SetVertexBuffer(0, _splatBuffer!);
        splatPass.Draw(6, (uint)visibleCount, 0, 0);
        splatPass.End();

        if (useCas)
        {
            _casData[0] = _sharpeningStrength;
            _casData[1] = 1f / _canvasWidth;
            _casData[2] = 1f / _canvasHeight;
            _casData[3] = 0f;
            Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
            _queue.WriteBuffer(_casUniformBuffer!, 0, _casByteData);

            _casColorAttach!.View = colorView;
            using var casPass = encoder.BeginRenderPass(_casPassDesc!);
            casPass.SetPipeline(_casPipeline!);
            casPass.SetBindGroup(0, _casBindGroup!);
            casPass.Draw(3, 1, 0, 0);
            casPass.End();
        }

        using var commandBuffer = encoder.Finish();
        _submitArray[0] = commandBuffer;
        _queue.Submit(_submitArray);
    }

    /// <summary>
    /// Stochastic rasterization with velocity-adaptive dilation, multi-SPP, and temporal accumulation.
    /// Per frame: SPP × (stochastic render + accumulate) + 1 display pass.
    /// Each sub-sample uses a unique hash seed. Accumulation weight = 1/totalSamples (running average).
    /// </summary>
    private void RenderStochastic(CameraParams camera, Matrix4x4 mvp)
    {
        // Velocity tracking (no sort needed)
        _sorter.UpdateVelocity(camera.Position, camera.Forward);
        float velocity = _sorter.SmoothedVelocity;
        bool moving = velocity > 1e-7f;

        // ── Velocity-adaptive parameters ──

        // Dilation: very subtle splat fattening to bridge sub-pixel spatial gaps (max +5%)
        _uniformData[21] = 1f + MathF.Min(MathF.Sqrt(velocity) * DilationScale, MaxDilationFactor);

        // Min alpha floor: boost survival of low-alpha edge fragments during movement.
        // At splat edges, Gaussian alpha drops to 0.05-0.1 → 90%+ discard rate → holes.
        // Floor of 0.15 ensures at least 15% survival at edges, filling gaps between splats.
        // During convergence: floor=0 restores exact Monte Carlo sampling for correct result.
        _uniformData[22] = moving ? 0.15f : 0f;

        // When moving: RESET accumulation each frame to prevent ghosting entirely.
        // Each frame's SPP sub-samples are still properly averaged (weight = 1/1, 1/2, 1/3...),
        // but no inter-frame blending occurs since frameCount resets to 0.
        // When still: accumulation grows across frames for progressive convergence.
        if (moving)
            _accumFrameCount = 0;

        // Multi-SPP: more samples per frame to fill stochastic holes.
        // Moving: SPP=2 (two independent samples, much fewer holes than 1).
        // Just stopped: brief SPP burst for fast initial convergence.
        // Converged: SPP=1 (image is clean, maximize FPS).
        // SPP=2@60fps = 120 samples/sec > SPP=4@25fps = 100 samples/sec, so keep SPP low.
        int spp;
        if (moving)
            spp = StochasticSPP; // default 2 during movement
        else if (_accumFrameCount < 60) // first ~1 second after stopping
            spp = StochasticSPP + 1;    // convergence burst (e.g., 3)
        else
            spp = 1;                    // converged, save GPU

        // ── Multi-SPP loop: each sub-sample gets stochastic render + accumulate ──
        // Each queue.submit() includes preceding writeBuffer operations, so uniform updates
        // between sub-samples are correctly sequenced by the GPU.
        _accumColorAttach!.LoadOp = GPULoadOp.Load;

        for (int s = 0; s < spp; s++)
        {
            _accumFrameCount = Math.Min(_accumFrameCount + 1, 1024);
            float accumWeight = 1f / _accumFrameCount;

            // Upload uniforms: unique seed per sub-sample (global counter, never repeats)
            _globalFrameCount++;
            _uniformData[20] = BitConverter.Int32BitsToSingle(_globalFrameCount);
            Buffer.BlockCopy(_uniformData, 0, _uniformByteData!, 0, _uniformByteData!.Length);
            _queue!.WriteBuffer(_uniformBuffer!, 0, _uniformByteData);

            // Upload accumulation weight for this sub-sample
            _accumData[0] = accumWeight;
            Buffer.BlockCopy(_accumData, 0, _accumByteData!, 0, _accumByteData!.Length);
            _queue.WriteBuffer(_accumUniformBuffer!, 0, _accumByteData);

            using var encoder = _device!.CreateCommandEncoder();

            // Stochastic splat render → _stochasticTexture (cleared each sub-sample)
            {
                using var pass = encoder.BeginRenderPass(_stochasticPassDesc!);
                pass.SetPipeline(_stochasticSplatPipeline!);
                pass.SetBindGroup(0, _uniformBindGroup!);
                pass.SetVertexBuffer(0, _splatBuffer!);
                pass.Draw(6, (uint)_splatCount, 0, 0);
                pass.End();
            }

            // Accumulate blend → _accumTexture (load previous, blend with weight)
            {
                using var pass = encoder.BeginRenderPass(_accumPassDesc!);
                pass.SetPipeline(_accumPipeline!);
                pass.SetBindGroup(0, _accumBindGroup!);
                pass.Draw(3, 1, 0, 0);
                pass.End();
            }

            using var cmdBuf = encoder.Finish();
            _submitArray[0] = cmdBuf;
            _queue.Submit(_submitArray);
        }

        // ── Display pass (once per frame): CAS reads _accumTexture → canvas ──
        {
            float displayStrength = _lowResActive ? 0f : _sharpeningStrength;
            _casData[0] = displayStrength;
            _casData[1] = 1f / _canvasWidth;
            _casData[2] = 1f / _canvasHeight;
            _casData[3] = 0f;
            Buffer.BlockCopy(_casData, 0, _casByteData!, 0, _casByteData!.Length);
            _queue!.WriteBuffer(_casUniformBuffer!, 0, _casByteData);

            using var colorTexture = _context!.GetCurrentTexture();
            using var colorView = colorTexture.CreateView();
            using var displayEncoder = _device!.CreateCommandEncoder();

            _casColorAttach!.View = colorView;
            using var pass = displayEncoder.BeginRenderPass(_casPassDesc!);
            pass.SetPipeline(_casPipeline!);
            pass.SetBindGroup(0, _casBindGroup!);
            pass.Draw(3, 1, 0, 0);
            pass.End();

            using var displayCmdBuf = displayEncoder.Finish();
            _submitArray[0] = displayCmdBuf;
            _queue.Submit(_submitArray);
        }
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

        // 2D dispatch to stay within WebGPU's maxComputeWorkgroupsPerDimension (65535).
        // For scenes ≤ 4.2M splats: wgY=1 (identical to old 1D path).
        // For larger scenes (e.g. 5K full-res = 14.7M splats): wgY=4.
        // Out-of-bounds threads hit the i >= u_count guard in the shader and return early.
        const uint maxWG = 65535u;
        uint totalWG = (uint)((visibleCount + 63) / 64);
        uint wgX = Math.Min(totalWG, maxWG);
        uint wgY = (totalWG + maxWG - 1) / maxWG;
        using var pass = encoder.BeginComputePass();
        pass.SetPipeline(_packPipeline);
        pass.SetBindGroup(0, _packBindGroup);
        pass.DispatchWorkgroups(wgX, wgY, 1);
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
        _packCountBuf?.Destroy();
        _packCountBuf?.Dispose();
        _packCountJsArray?.Dispose();

        // Stochastic rasterization resources
        _stochasticSplatPipeline?.Dispose();
        _stochasticTexture?.Destroy();
        _stochasticTexture?.Dispose();
        _stochasticView?.Dispose();
        _accumTexture?.Destroy();
        _accumTexture?.Dispose();
        _accumView?.Dispose();
        _accumPipeline?.Dispose();
        _accumBindGroup?.Dispose();
        _accumUniformBuffer?.Destroy();
        _accumUniformBuffer?.Dispose();
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
    mvp         : mat4x4<f32>,
    viewport    : vec2<f32>,
    focal       : vec2<f32>,
    frame_index : u32,
    dilation    : f32,
    min_alpha   : f32,
    _pad3       : u32,
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
    let scale_x = max(input.scale.x * u.dilation, 0.001);
    let scale_y = max(input.scale.y * u.dilation, 0.001);

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
    //  WGSL Stochastic Splat Shader — sort-free rendering via stochastic transparency
    //  Same vertex shader as sorted mode. Fragment shader does stochastic discard + depth test.
    //  Per-pixel: random u ∈ [0,1), discard if u >= alpha. Depth test selects closest survivor.
    //  Over multiple frames, temporal accumulation converges to correct alpha-blended result.
    // ════════════════════════════════════════════════════════════
    private const string StochasticSplatShaderSource = @"
struct Uniforms {
    mvp         : mat4x4<f32>,
    viewport    : vec2<f32>,
    focal       : vec2<f32>,
    frame_index : u32,
    dilation    : f32,
    min_alpha   : f32,
    _pad3       : u32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) position    : vec3<f32>,
    @location(1) color_alpha : vec4<f32>,
    @location(2) scale       : vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) color   : vec3<f32>,
    @location(1) opacity : f32,
    @location(2) uv      : vec2<f32>,
};

// lowbias32 hash — fast, good avalanche properties
fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    x *= 0x45d9f3bu;
    x ^= x >> 16u;
    return x;
}

@vertex
fn vs_main(
    input : VertexInput,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32
) -> VertexOutput {
    var quad_pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0)
    );

    let uv = quad_pos[vid];
    let center_clip = u.mvp * vec4<f32>(input.position, 1.0);

    var out : VertexOutput;
    if (center_clip.w <= 0.001) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = input.color_alpha.rgb;
        out.opacity = 0.0;
        out.uv = uv;
        return out;
    }

    let ndc_center = center_clip.xyz / center_clip.w;

    let scale_x = max(input.scale.x * u.dilation, 0.001);
    let scale_y = max(input.scale.y * u.dilation, 0.001);

    let screen_rx = scale_x * u.focal.x / center_clip.w;
    let screen_ry = scale_y * u.focal.y / center_clip.w;

    let ewa_rx = max(screen_rx, 0.8);
    let ewa_ry = max(screen_ry, 0.8);

    let ndc_radius_x = ewa_rx * 2.0 / u.viewport.x;
    let ndc_radius_y = ewa_ry * 2.0 / u.viewport.y;
    let ndc_radius_max = max(ndc_radius_x, ndc_radius_y) * 3.0;

    if (ndc_center.x + ndc_radius_max < -1.0 || ndc_center.x - ndc_radius_max > 1.0 ||
        ndc_center.y + ndc_radius_max < -1.0 || ndc_center.y - ndc_radius_max > 1.0 ||
        ndc_center.z < -0.1 || ndc_center.z > 1.1) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = input.color_alpha.rgb;
        out.opacity = 0.0;
        out.uv = uv;
        return out;
    }

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
    let dist_sq = dot(input.uv, input.uv);
    if (dist_sq > 9.0) { discard; }

    let alpha = input.opacity * exp(-dist_sq * 0.5);
    if (alpha < 0.002) { discard; }

    // Stochastic transparency: discard with probability (1 - effective_alpha).
    // min_alpha floor: during movement, boost survival of low-alpha edge fragments
    // to fill holes. During convergence, min_alpha=0 restores exact Monte Carlo sampling.
    let effective_alpha = max(alpha, u.min_alpha);
    let pixel = vec2<u32>(input.clip_pos.xy);
    let seed = pixel.x + pixel.y * 65537u + u.frame_index * 2654435761u;
    let u_rand = f32(hash_u32(seed)) / 4294967295.0;
    if (u_rand >= effective_alpha) { discard; }

    return vec4<f32>(input.color, 1.0);  // Opaque write — no alpha blending
}
";

    // ════════════════════════════════════════════════════════════
    //  WGSL Temporal Accumulation — fullscreen EMA blend
    //  Reads per-frame stochastic render, blends into persistent accumulation texture.
    //  Uses SrcAlpha/OneMinusSrcAlpha blend: output alpha = 1/frameCount = running average weight.
    // ════════════════════════════════════════════════════════════
    private const string AccumulateShaderSource = @"
struct AccumUniforms {
    weight   : f32,
    _pad1    : f32,
    _pad2    : f32,
    _pad3    : f32,
};

@group(0) @binding(0) var t_current : texture_2d<f32>;
@group(0) @binding(1) var s_current : sampler;
@group(0) @binding(2) var<uniform> accum : AccumUniforms;

struct VSOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> VSOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let pos = positions[vid];
    var out : VSOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_accum(input : VSOutput) -> @location(0) vec4<f32> {
    let current = textureSample(t_current, s_current, input.uv);
    return vec4<f32>(current.rgb, accum.weight);
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
fn pack_splats(@builtin(global_invocation_id) gid : vec3<u32>,
               @builtin(num_workgroups) nwg : vec3<u32>) {
    // 2D dispatch: recompute linear index from row (gid.y) and column (gid.x).
    // nwg.x = wgX (workgroups in X), so nwg.x * 64 = total threads per row.
    let i = gid.y * nwg.x * 64u + gid.x;
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
