using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// Device settings: properties of THIS machine, not of a project, so they live in localStorage (a per-viewer convenience;
// a missing or blocked store just means Auto).
public partial class Studio
{
    const string GpuMemoryKey = "spawnscene.gpuMemoryGB";
    int _gpuMemoryGB = -1; // -1 = not read yet

    /// <summary>The training GPU memory budget in GB, 0 = Auto (<see cref="GpuMemoryBudget"/>).</summary>
    int GpuMemoryGB
    {
        get
        {
            if (_gpuMemoryGB >= 0) return _gpuMemoryGB;
            _gpuMemoryGB = 0;
            try
            {
                using var store = _js.Get<Storage>("localStorage");
                if (int.TryParse(store?.GetItem(GpuMemoryKey), out int gb) && GpuMemoryBudget.ChoicesGB.Contains(gb))
                    _gpuMemoryGB = gb;
            }
            catch { }
            return _gpuMemoryGB;
        }
        set
        {
            _gpuMemoryGB = value;
            try
            {
                using var store = _js.Get<Storage>("localStorage");
                store?.SetItem(GpuMemoryKey, value.ToString());
            }
            catch { }
        }
    }

    /// <summary>The device's storage-binding limit (the target stack's ceiling), read once.</summary>
    long DeviceBindingLimitBytes =>
        _deviceBindingLimit ??= GpuMemoryBudget.ReadMaxStorageBindingBytes(
            _gpuService.IsInitialized ? _gpuService.WebGPUAccelerator.NativeAccelerator.NativeDevice : null);
    long? _deviceBindingLimit;
}
