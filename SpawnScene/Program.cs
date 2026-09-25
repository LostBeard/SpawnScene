using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.SpawnJS;
using SpawnDev.GameUI;
using SpawnScene;
using SpawnScene.Services;

// Reduce GC pauses during render loop (helps avoid ~500ms stalls when moving camera)
// below disabled becuase it is not supprot on browser platforms
// GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
// SpawnDev.SpawnJS runtime
builder.Services.AddSpawnJSRuntime(out var JS);
builder.Services.AddGameUI(UITheme.Dark);

SpawnJSRuntime.EnableIDisposableWatcher = false;

// HttpClient for fetching static assets
builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// App services (Scoped == Singleton in Blazor WASM, but avoids DI conflicts with HttpClient)
builder.Services.AddScoped<GpuService>();
builder.Services.AddScoped<SceneManager>();
builder.Services.AddScoped<RenderService>();
builder.Services.AddScoped<DepthEstimationService>();
// SuperResolutionService retired 2026-07-01 (zero-ORT migration) — now a parked static kernel
// holder for a future native SR pass, no longer a DI service. See SuperResolutionService.cs.
// GpuShareService retired with ORT — it patched navigator.gpu.requestAdapter for device sharing
// and double-initialized ILGPU when no second consumer existed.
builder.Services.AddScoped<DepthToGaussianKernel>();
builder.Services.AddScoped<GpuDepthColorizer>();
builder.Services.AddScoped<GpuSplatSorter>();
builder.Services.AddScoped<GpuGaussianRenderer>();
builder.Services.AddScoped<ProjectService>();
builder.Services.AddScoped<XRService>();
builder.Services.AddScoped<GpuFeatureMatcher>();
builder.Services.AddScoped<VideoFrameExtractor>();
builder.Services.AddScoped<ImageImportService>();
builder.Services.AddScoped<SfmReconstructor>();
builder.Services.AddScoped<MultiViewGenerationService>();

// SpawnDev.ILGPU.ML model acquisition: hub HTTP + OPFS cache, streamed JS-side to the GPU.
// HubModelSource implements IModelSource for DepthEstimationPipeline.CreateFromHubAsync.
// (WebTorrent HubModelStream is an optional swap via SpawnDev.ILGPU.ML.WebTorrent.)
builder.Services.AddScoped<IModelSource, HubModelSource>(sp => new HubModelSource(sp.GetRequiredService<SpawnJSRuntime>(), sp.GetRequiredService<HttpClient>()));
// SpawnDev.ILGPU.ML kernels are created on-demand (GPU must be initialized first)

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().SpawnJSRunAsync();
