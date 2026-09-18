#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// Smoke gate: boot SpawnScene Studio on system Chrome (hardware WebGPU) with a dedicated profile.
// Usage:
//   dotnet run tools/smoke-studio.cs -- [url]
// Default url: https://localhost:5001/studio
// Env:
//   SPAWNSCENE_PROFILE  - Chrome user-data-dir (default: %TEMP%\spawnscene-chrome-profile)
using Microsoft.Playwright;

var url = args.FirstOrDefault(a => !a.StartsWith("--")) ?? "https://localhost:5001/studio";
var profileDir = Environment.GetEnvironmentVariable("SPAWNSCENE_PROFILE")
    ?? Path.Combine(Path.GetTempPath(), "spawnscene-chrome-profile");
Directory.CreateDirectory(profileDir);

var consoleLines = new List<string>();
var pageErrors = new List<string>();
bool sawGpuInit = false;
bool sawBlazorError = false;
bool sawFatal = false;

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome", // installed Chrome = real GPU; Playwright Chromium is software-only
    IgnoreHTTPSErrors = true,
    Args = new[]
    {
        "--enable-unsafe-webgpu",
        "--disable-blink-features=AutomationControlled",
    },
});

var page = ctx.Pages.Count > 0 ? ctx.Pages[0] : await ctx.NewPageAsync();
page.Console += (_, msg) =>
{
    var line = $"[{msg.Type}] {msg.Text}";
    consoleLines.Add(line);
    Console.WriteLine($"[console] {line}");
    if (msg.Text.Contains("[GpuService]", StringComparison.Ordinal))
        sawGpuInit = true;
    if (msg.Text.Contains("blazor-error-ui", StringComparison.OrdinalIgnoreCase)
        || msg.Text.Contains("An unhandled error has occurred", StringComparison.OrdinalIgnoreCase))
        sawBlazorError = true;
    if (msg.Type == "error"
        && (msg.Text.Contains("WebGPU", StringComparison.OrdinalIgnoreCase)
            || msg.Text.Contains("Unhandled", StringComparison.OrdinalIgnoreCase)
            || msg.Text.Contains("Failed to", StringComparison.OrdinalIgnoreCase)))
        sawFatal = true;
};
page.PageError += (_, err) =>
{
    pageErrors.Add(err);
    Console.WriteLine($"[pageerror] {err}");
    sawFatal = true;
};

Console.WriteLine($"[gate] profile={profileDir}");
Console.WriteLine($"[gate] goto {url}");

try
{
    await page.GotoAsync(url, new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60000 });
}
catch (Exception ex)
{
    Console.WriteLine($"[gate] FAIL navigate: {ex.Message}");
    return 2;
}

// Secure-context WebGPU probe (localhost qualifies).
var adapterInfo = await page.EvaluateAsync<string>(@"async () => {
  if (!navigator.gpu) return JSON.stringify({ err: 'no navigator.gpu' });
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!a) return JSON.stringify({ err: 'no adapter' });
  const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
  return JSON.stringify({
    vendor: i.vendor, architecture: i.architecture, device: i.device,
    description: i.description, isFallback: !!a.isFallbackAdapter
  });
}");
Console.WriteLine($"[gate] adapter: {adapterInfo}");
if (adapterInfo.Contains("\"err\"", StringComparison.Ordinal)
    || adapterInfo.Contains("\"isFallback\":true", StringComparison.Ordinal))
{
    Console.WriteLine("[gate] FAIL: no hardware WebGPU adapter");
    await ctx.CloseAsync();
    return 3;
}

// Wait for Blazor + Studio canvas (WASM boot). Loading screen text goes away once #app mounts Studio.
try
{
    await page.WaitForSelectorAsync("canvas", new() { Timeout = 120000 });
    Console.WriteLine("[gate] canvas present");
}
catch (Exception ex)
{
    Console.WriteLine($"[gate] FAIL no canvas: {ex.Message}");
    DumpTail(consoleLines, pageErrors);
    await ctx.CloseAsync();
    return 4;
}

// Give GpuService / Studio OnAfterRenderAsync time to finish.
var deadline = DateTime.UtcNow.AddSeconds(60);
while (DateTime.UtcNow < deadline && !sawGpuInit)
    await page.WaitForTimeoutAsync(500);

var errorUiVisible = await page.EvaluateAsync<bool>(
    "() => { const e = document.getElementById('blazor-error-ui'); return !!e && getComputedStyle(e).display !== 'none'; }");
var loadingStillVisible = await page.EvaluateAsync<bool>(
    "() => !!document.querySelector('.loading-screen')");

Console.WriteLine($"[gate] sawGpuInit={sawGpuInit} errorUi={errorUiVisible} loading={loadingStillVisible} pageErrors={pageErrors.Count}");

var shot = Path.Combine(Path.GetTempPath(), "spawnscene-studio-smoke.png");
await page.ScreenshotAsync(new() { Path = shot, FullPage = true });
Console.WriteLine($"[gate] screenshot: {shot}");

bool pass = sawGpuInit && !errorUiVisible && !sawBlazorError && !sawFatal && pageErrors.Count == 0 && !loadingStillVisible;
if (!pass)
{
    Console.WriteLine("[gate] FAIL");
    DumpTail(consoleLines, pageErrors);
    await ctx.CloseAsync();
    return 1;
}

Console.WriteLine("[gate] PASS - Studio canvas up, WebGPU initialized, no fatal console errors");
await ctx.CloseAsync();
return 0;

static void DumpTail(List<string> consoleLines, List<string> pageErrors)
{
    Console.WriteLine("--- console tail ---");
    foreach (var l in consoleLines.TakeLast(40))
        Console.WriteLine(l);
    if (pageErrors.Count > 0)
    {
        Console.WriteLine("--- page errors ---");
        foreach (var e in pageErrors)
            Console.WriteLine(e);
    }
}
