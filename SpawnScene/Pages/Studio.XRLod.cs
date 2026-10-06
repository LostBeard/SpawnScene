namespace SpawnScene.Pages;

/// <summary>
/// LOD in a headset (Plans/lod-streaming.md phase C): two eyes at headset resolution, measured at the eye's pixel
/// density (GpuSplatSorter.LodFocalOverride), refine an LOD tree to ~1.1M nodes on TruckFull - more than a standalone
/// Quest draws at its frame rate. Without a budget of its own (&amp;lodbudget), an XR session on an LOD scene gets one
/// for the device - the Quest's own browser less than a PC-tethered headset - and the desktop's comes back on exit.
/// </summary>
public partial class Studio
{
    /// <summary>Splats a frame through an LOD cut in the Quest's own browser.</summary>
    public const int XrLodBudgetStandalone = 600_000;

    /// <summary>... and on a PC (a tethered headset, the emulator).</summary>
    public const int XrLodBudgetTethered = 1_500_000;

    int _lodBudgetBeforeXr = -1;

    void ApplyXrLodBudget()
    {
        if (!_gpuRenderer.LodActive || _gpuRenderer.LodBudget > 0) return;
        bool standalone;
        try
        {
            using var navigator = _js.Get<SpawnDev.SpawnJS.JSObjects.Navigator>("navigator");
            standalone = navigator.UserAgent.Contains("OculusBrowser", StringComparison.OrdinalIgnoreCase);
        }
        catch { standalone = true; }   // unknown: the safe side
        _lodBudgetBeforeXr = _gpuRenderer.LodBudget;
        _gpuRenderer.LodBudget = standalone ? XrLodBudgetStandalone : XrLodBudgetTethered;
        Console.WriteLine($"[XR] LOD budget {_gpuRenderer.LodBudget:N0} splats a frame ({(standalone ? "standalone headset" : "tethered")})");
    }

    void RestoreLodBudgetAfterXr()
    {
        if (_lodBudgetBeforeXr < 0) return;
        _gpuRenderer.LodBudget = _lodBudgetBeforeXr;
        _lodBudgetBeforeXr = -1;
    }
}
