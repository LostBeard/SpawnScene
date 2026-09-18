#:package Microsoft.Playwright@1.49.0
#:property JsonSerializerIsReflectionEnabledByDefault=true
// End-to-end generate gate via /studio?autotest=generate-room (system Chrome + hardware WebGPU).
// Usage: dotnet run tools/smoke-generate.cs -- [baseUrl]
// Default base: https://localhost:7199
using Microsoft.Playwright;

var baseUrl = (args.FirstOrDefault(a => !a.StartsWith("--")) ?? "https://localhost:7199").TrimEnd('/');
var url = $"{baseUrl}/studio?autotest=generate-room";
var profileDir = Environment.GetEnvironmentVariable("SPAWNSCENE_PROFILE")
    ?? Path.Combine(Path.GetTempPath(), "spawnscene-chrome-profile");
Directory.CreateDirectory(profileDir);

var consoleLines = new List<string>();
bool pass = false;
bool fail = false;

using var pw = await Playwright.CreateAsync();
await using var ctx = await pw.Chromium.LaunchPersistentContextAsync(profileDir, new()
{
    Headless = false,
    Channel = "chrome",
    IgnoreHTTPSErrors = true,
    ViewportSize = new() { Width = 1280, Height = 720 },
    DeviceScaleFactor = 1,
    Args = new[] { "--enable-unsafe-webgpu" },
});

var page = ctx.Pages.Count > 0 ? ctx.Pages[0] : await ctx.NewPageAsync();
page.Console += (_, msg) =>
{
    var line = $"[{msg.Type}] {msg.Text}";
    consoleLines.Add(line);
    Console.WriteLine($"[console] {line}");
    if (msg.Text.Contains("[Autotest] PASS", StringComparison.Ordinal)) pass = true;
    if (msg.Text.Contains("[Autotest] FAIL", StringComparison.Ordinal)) fail = true;
};
page.PageError += (_, err) =>
{
    consoleLines.Add($"[pageerror] {err}");
    Console.WriteLine($"[pageerror] {err}");
    fail = true;
};

Console.WriteLine($"[gate] profile={profileDir}");
Console.WriteLine($"[gate] goto {url}");
await page.GotoAsync(url, new() { WaitUntil = WaitUntilState.DOMContentLoaded, Timeout = 60000 });
await page.WaitForSelectorAsync("canvas", new() { Timeout = 120000 });

Console.WriteLine("[gate] waiting for Autotest PASS/FAIL (model download may take minutes)...");
var deadline = DateTime.UtcNow.AddMinutes(12);
while (DateTime.UtcNow < deadline && !pass && !fail)
    await page.WaitForTimeoutAsync(1000);

var shot = Path.Combine(Path.GetTempPath(), "spawnscene-generate.png");
await page.ScreenshotAsync(new() { Path = shot });
Console.WriteLine($"[gate] screenshot: {shot}");

if (pass)
{
    Console.WriteLine("[gate] PASS");
    return 0;
}

Console.WriteLine("[gate] FAIL");
Console.WriteLine("--- console tail ---");
foreach (var l in consoleLines.TakeLast(80))
    Console.WriteLine(l);
return 1;
