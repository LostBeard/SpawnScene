using ILGPU;
using ILGPU.Runtime;
using SpawnDev;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.Rendering;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnScene.Services;

/// <summary>
/// Manages the ILGPU WebGPU accelerator lifecycle.
/// WebGPU is required — no fallbacks. ILGPU + SpawnDev.ILGPU.ML share this single accelerator.
/// (Pre-migration: <see cref="GpuShareService"/> adopted ORT's GPUDevice for zero-copy tensors.
/// ORT was retired 2026-07-01; adopting via that hook during our own <c>requestAdapter</c> caused
/// a double-init that leaked the first accelerator.)
/// </summary>
public class GpuService : IBackgroundService, IAsyncDisposable
{
    private Context? _context;
    private bool _initialized;
    SpawnJSRuntime _js;
    public GpuService(SpawnJSRuntime js)
    {
        _js = js;
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
    /// The native WebGPU GPUDevice backing the ILGPU accelerator.
    /// </summary>
    public GPUDevice NativeDevice =>
        WebGPUAccelerator?.NativeAccelerator?.NativeDevice
        ?? throw new InvalidOperationException("GPU not initialized. Call InitializeAsync first.");

    /// <summary>
    /// Initialize the WebGPU accelerator.
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

        // Name the cause of a device loss. Without these two, a loss surfaces only as Dawn's
        // wire message "A valid external Instance reference no longer exists" thrown from
        // whichever interop call happened to be awaiting - which says the device is gone and
        // nothing about why. device.lost carries Dawn's reason and message; the uncaptured
        // error that precedes an out-of-memory loss names the allocation that failed.
        // MEASURED: DrJohnson 912k and Bathroom 757k both died on the first ApplySplatPlan with
        // exactly that message and no further information, twice, until this was added.
        WebGPUAccelerator.DeviceLost += (reason, message) =>
            Console.WriteLine($"[GpuService] DEVICE LOST: reason={reason} message={message}");
        NativeDevice.OnUncapturedError += OnUncapturedGpuError;

        Console.WriteLine($"[GpuService] WebGPU initialized: {DeviceName}");
    }

    /// <summary>
    /// Instance-method listener so it can be detached in <see cref="DisposeAsync"/>: SpawnJS
    /// ActionEvent subscriptions live in a static table keyed by delegate and pin the target.
    /// </summary>
    private void OnUncapturedGpuError(GPUUncapturedErrorEvent e)
    {
        try
        {
            Console.WriteLine($"[GpuService] GPU ERROR ({e.Error?.GetType().Name}): {e.Error?.Message}");
        }
        catch
        {
            // Never let a diagnostic take the runtime down on top of the error it reports.
        }
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
        if (_initialized)
        {
            try { NativeDevice.OnUncapturedError -= OnUncapturedGpuError; } catch { }
        }
        WebGPUAccelerator?.Dispose();
        _context?.Dispose();
        _initialized = false;
        GC.SuppressFinalize(this);
    }
}
