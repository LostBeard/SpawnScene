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

    const string TrainMarksKey = "spawnscene.trainSecondsPerMegapixelAtMark";
    Dictionary<int, double>? _trainMarks;

    /// <summary>
    /// This device's measured training times (<see cref="TrainingTimeEstimate"/>): iteration mark -> elapsed seconds per
    /// training megapixel, empty until a run reaches its first mark. Stored as "1000:18.9;3000:71.2" (no JSON
    /// serializer to keep trim safe).
    /// </summary>
    Dictionary<int, double> TrainMarks
    {
        get
        {
            if (_trainMarks != null) return _trainMarks;
            _trainMarks = new();
            try
            {
                using var store = _js.Get<Storage>("localStorage");
                foreach (var pair in (store?.GetItem(TrainMarksKey) ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries))
                {
                    var kv = pair.Split(':');
                    if (kv.Length == 2 && int.TryParse(kv[0], out int mark) &&
                        double.TryParse(kv[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
                            out double s) && s > 0 && double.IsFinite(s))
                        _trainMarks[mark] = s;
                }
            }
            catch { }
            return _trainMarks;
        }
        set
        {
            _trainMarks = value;
            try
            {
                using var store = _js.Get<Storage>("localStorage");
                store?.SetItem(TrainMarksKey, string.Join(";", value.OrderBy(kv => kv.Key)
                    .Select(kv => $"{kv.Key}:{kv.Value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)}")));
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
