namespace SpawnScene.Services;

/// <summary>Controls the splat rendering technique. Its own file so ProjectSettings (which stores it) can be compiled
/// without the browser renderer, e.g. into SpawnScene.Tests.</summary>
public enum SplatRenderMode
{
    /// <summary>Traditional sorted alpha blending (cull → radix sort → pack → render).</summary>
    Sorted,
    /// <summary>Sort-free stochastic rasterization with temporal accumulation (StochasticSplats, ICCV 2025).</summary>
    Stochastic,
}
