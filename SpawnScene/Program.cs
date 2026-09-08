using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.SpawnJS;
using SpawnScene;
using SpawnScene.Services;

// Reduce GC pauses during render loop (helps avoid ~500ms stalls when moving camera)
// below disabled becuase it is not supprot on browser platforms
// GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
// SpawnDev.SpawnJS runtime
builder.Services.AddSpawnJSRuntime(out var JS);

SpawnJSRuntime.EnableIDisposableWatcher = false;

// HttpClient for fetching static assets
builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// App services (Scoped == Singleton in Blazor WASM, but avoids DI conflicts with HttpClient)
builder.Services.AddSingleton<GpuShareService>();
builder.Services.AddScoped<GpuService>();
builder.Services.AddScoped<SceneManager>();
builder.Services.AddScoped<RenderService>();
builder.Services.AddScoped<DepthEstimationService>();
// SuperResolutionService retired 2026-07-01 (zero-ORT migration) — now a parked static kernel
// holder for a future native SR pass, no longer a DI service. See SuperResolutionService.cs.
builder.Services.AddScoped<DepthToGaussianKernel>();
builder.Services.AddScoped<GpuDepthColorizer>();
builder.Services.AddScoped<GpuSplatSorter>();
builder.Services.AddScoped<GpuGaussianRenderer>();
builder.Services.AddScoped<ProjectService>();
builder.Services.AddScoped<XRService>();
builder.Services.AddScoped<GpuFeatureMatcher>();
builder.Services.AddScoped<ImageImportService>();
builder.Services.AddScoped<SfmReconstructor>();
builder.Services.AddScoped<MultiViewGenerationService>();

// SpawnDev.ILGPU.ML model acquisition: WebTorrent delivery + OPFS cache, streamed JS-side to the GPU.
// HubModelStream feeds DepthEstimationPipeline.CreateFromHubAsync (download + cache + zero-copy load).
builder.Services.AddScoped<SpawnDev.WebTorrent.WebTorrentClient>();
builder.Services.AddScoped(sp => new SpawnDev.ILGPU.ML.Hub.HubModelStream(
    sp.GetRequiredService<SpawnDev.WebTorrent.WebTorrentClient>(),
    sp.GetRequiredService<HttpClient>()));
// SpawnDev.ILGPU.ML kernels are created on-demand (GPU must be initialized first)

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().SpawnJSRunAsync();




// NOTE (Data, pipeline-migration fork): commented out — `class ByteBuffer : Uint8Array, byte[]`
// is invalid C# (you cannot inherit `byte[]`) and was the sole build blocker. It is unreferenced
// anywhere in the project. Left commented rather than deleted so Captain can confirm intent.
// public class ByteBuffer : Uint8Array, byte[]
// {
//
// }
