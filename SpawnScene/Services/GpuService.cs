using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.Rendering;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnScene.Services;

/// <summary>
/// Manages the ILGPU WebGPU accelerator lifecycle.
/// WebGPU is required — no fallbacks. All compute and data processing runs on
/// the same GPUDevice, enabling zero-copy integration with ONNX Runtime Web.
/// </summary>
public class GpuService : IAsyncDisposable
{
    private Context? _context;
    private bool _initialized;

    public GpuService(GpuShareService gpuShare)
    {
        gpuShare.OnDeviceCreated += OnDeviceCreated;
    }

    private async Task OnDeviceCreated(GPUAdapterHook adapterHook, GPUDeviceReturnOverride args)
    {
        if (args.Device != null && !_initialized)
            await InitializeFromExternalDeviceAsync(args.Device);
    }

    /// <summary>The active WebGPU accelerator.</summary>
    public WebGPUAccelerator WebGPUAccelerator { get; private set; } = default!;

    /// <summary>The active accelerator typed as the base class (for backward compat).</summary>
    public Accelerator Accelerator => WebGPUAccelerator;

    /// <summary>Whether the GPU has been initialized.</summary>
    public bool IsInitialized => _initialized;

    /// <summary>Name of the active device.</summary>
    public string DeviceName => _initialized ? WebGPUAccelerator.Name : "Not initialized";

    /// <summary>Type of the active accelerator.</summary>
    public AcceleratorType AcceleratorType => _initialized
        ? WebGPUAccelerator.AcceleratorType
        : AcceleratorType.CPU;

    /// <summary>
    /// The native WebGPU GPUDevice.
    /// Pass to ort.env.webgpu.device before creating ONNX sessions so that ORT and
    /// ILGPU share one GPUDevice, enabling zero-copy buffer sharing via
    /// ExternalWebGPUMemoryBuffer and TensorFromGpuBuffer.
    /// </summary>
    public GPUDevice NativeDevice =>
        WebGPUAccelerator?.NativeAccelerator?.NativeDevice
        ?? throw new InvalidOperationException("GPU not initialized. Call InitializeAsync first.");

    /// <summary>
    /// Initialize ILGPU by adopting an externally-created GPUDevice (e.g. from ORT).
    /// Both libraries share the same device — zero-copy buffer exchange is possible.
    /// </summary>
    public async Task InitializeFromExternalDeviceAsync(GPUDevice externalDevice)
    {
        if (_initialized) return;

        var builder = Context.Create()
            .EnableAlgorithms()
            .EnableWebGPUAlgorithms();

        await builder.WebGPU();
        _context = builder.ToContext();

        WebGPUAccelerator = WebGPUAccelerator.CreateFromExternalDevice(_context, externalDevice);
        _initialized = true;

        Console.WriteLine($"[GpuService] ILGPU adopted external device: {DeviceName}");
    }

    /// <summary>
    /// Initialize the WebGPU accelerator (standalone — no device sharing).
    /// Throws NotSupportedException if WebGPU is unavailable (Chrome 113+, Edge 113+, Safari 18+).
    /// </summary>
    public async Task InitializeAsync()
    {
        if (_initialized) return;

        var builder = Context.Create()
            .EnableAlgorithms()
            .EnableWebGPUAlgorithms();

        await builder.WebGPU();
        _context = builder.ToContext();

        var devices = _context.GetDevices<WebGPUILGPUDevice>();
        if (devices.Count == 0)
            throw new NotSupportedException(
                "WebGPU is required. Please use Chrome 113+, Edge 113+, or Safari 18+.");

        WebGPUAccelerator = (WebGPUAccelerator)await devices[0].CreateAcceleratorAsync(_context, null);
        _initialized = true;

        Console.WriteLine($"[GpuService] WebGPU initialized: {DeviceName}");
    }

    /// <summary>
    /// Create an ICanvasRenderer for zero-copy GPU→canvas blitting.
    /// Returns WebGPUCanvasRenderer (no CPU readback, fullscreen-triangle render pass).
    /// Must call InitializeAsync first.
    /// </summary>
    public ICanvasRenderer CreateCanvasRenderer() =>
        CanvasRendererFactory.Create(WebGPUAccelerator);

    /// <summary>Allocate a 1D GPU buffer initialized from host data.</summary>
    public MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(T[] data) where T : unmanaged =>
        WebGPUAccelerator.Allocate1D(data);

    /// <summary>Allocate an empty 1D GPU buffer.</summary>
    public MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(int length) where T : unmanaged =>
        WebGPUAccelerator.Allocate1D<T>(length);

    /// <summary>Synchronize the accelerator (wait for all pending GPU work to complete).</summary>
    public async Task SynchronizeAsync()
    {
        if (_initialized)
            await WebGPUAccelerator.SynchronizeAsync();
    }

    public async ValueTask DisposeAsync()
    {
        WebGPUAccelerator?.Dispose();
        _context?.Dispose();
        _initialized = false;
        GC.SuppressFinalize(this);
    }
}
