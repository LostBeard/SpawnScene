using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnScene;
using SpawnScene.Services;

// Reduce GC pauses during render loop (helps avoid ~500ms stalls when moving camera)
// below disabled becuase it is not supprot on browser platforms
// GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
// SpawnDev.BlazorJS runtime
builder.Services.AddBlazorJSRuntime(out var JS);

IDisposableTracker.UndisposedHandleVerboseMode = false;

// HttpClient for fetching static assets
builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// App services (Scoped == Singleton in Blazor WASM, but avoids DI conflicts with HttpClient)
builder.Services.AddSingleton<GpuShareService>();
builder.Services.AddScoped<GpuService>();
builder.Services.AddScoped<SceneManager>();
builder.Services.AddScoped<RenderService>();
builder.Services.AddScoped<DepthEstimationService>();
builder.Services.AddScoped<SuperResolutionService>();
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
// SpawnDev.ILGPU.ML kernels are created on-demand (GPU must be initialized first)

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
