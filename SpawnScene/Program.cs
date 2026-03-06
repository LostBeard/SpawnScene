using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnScene;
using SpawnScene.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

// HttpClient for fetching static assets
builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// SpawnDev.BlazorJS runtime
builder.Services.AddBlazorJSRuntime();

// App services (Scoped == Singleton in Blazor WASM, but avoids DI conflicts with HttpClient)
builder.Services.AddSingleton<GpuShareService>();
builder.Services.AddScoped<GpuService>();
builder.Services.AddScoped<SceneManager>();
builder.Services.AddScoped<RenderService>();
builder.Services.AddScoped<DepthEstimationService>();
builder.Services.AddScoped<DepthToGaussianKernel>();
builder.Services.AddScoped<GpuDepthColorizer>();
builder.Services.AddScoped<GpuSplatSorter>();
builder.Services.AddScoped<GpuGaussianRenderer>();

await builder.Build().BlazorJSRunAsync();
