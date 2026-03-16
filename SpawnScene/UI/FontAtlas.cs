using SpawnDev.BlazorJS.JSObjects;

namespace SpawnScene.UI;

/// <summary>Font size presets for the UI.</summary>
public enum FontSize
{
    Caption = 12,
    Body = 16,
    Heading = 24,
    Title = 32,
}

/// <summary>Metrics for a single character in the font atlas.</summary>
public struct CharMetrics
{
    /// <summary>UV coordinates in the atlas texture (normalized 0-1).</summary>
    public float U0, V0, U1, V1;
    /// <summary>Character advance width in pixels.</summary>
    public float Advance;
    /// <summary>Character glyph width and height in pixels.</summary>
    public float Width, Height;
    /// <summary>Vertical offset from baseline to top of glyph.</summary>
    public float BearingY;
}

/// <summary>
/// Generates a bitmap font atlas at runtime using OffscreenCanvas,
/// then uploads it to a WebGPU texture for UI text rendering.
/// </summary>
public class FontAtlas : IDisposable
{
    private const int AtlasSize = 1024;
    private const string FontFamily = "Inter, system-ui, -apple-system, sans-serif";
    private const int FirstChar = 32;  // space
    private const int LastChar = 126;  // tilde

    private readonly Dictionary<FontSize, Dictionary<char, CharMetrics>> _metrics = new();

    public GPUTexture? Texture { get; private set; }
    public GPUTextureView? View { get; private set; }
    public bool IsReady => View != null;

    /// <summary>
    /// Generate the font atlas and upload to GPU.
    /// Must be called after WebGPU device is available.
    /// </summary>
    public void Init(GPUDevice device, GPUQueue queue)
    {
        using var canvas = new OffscreenCanvas(AtlasSize, AtlasSize);
        using var ctx = canvas.Get2DContext();

        // Clear to transparent
        ctx.ClearRect(0, 0, AtlasSize, AtlasSize);

        // White text on transparent background (tinted by vertex color at render time)
        ctx.FillStyle = "white";
        ctx.TextBaseline = "top";

        int cursorX = 1;
        int cursorY = 1;
        int rowHeight = 0;

        // Generate glyphs for each font size
        foreach (var size in new[] { FontSize.Caption, FontSize.Body, FontSize.Heading, FontSize.Title })
        {
            int px = (int)size;
            ctx.Font = $"{px}px {FontFamily}";
            var charMap = new Dictionary<char, CharMetrics>();

            // Measure ascent for this size (use 'M' as reference)
            using var mMetrics = ctx.MeasureText("M");
            float ascent = (float)mMetrics.FontBoundingBoxAscent;
            float descent = (float)mMetrics.FontBoundingBoxDescent;
            float lineHeight = ascent + descent;
            int glyphHeight = (int)Math.Ceiling(lineHeight) + 2; // padding

            for (int c = FirstChar; c <= LastChar; c++)
            {
                char ch = (char)c;
                string s = ch.ToString();

                using var tm = ctx.MeasureText(s);
                float advance = (float)tm.Width;
                int glyphWidth = (int)Math.Ceiling(advance) + 2; // padding

                // Wrap to next row if needed
                if (cursorX + glyphWidth + 1 >= AtlasSize)
                {
                    cursorX = 1;
                    cursorY += rowHeight + 1;
                    rowHeight = 0;
                }

                if (cursorY + glyphHeight + 1 >= AtlasSize)
                {
                    Console.WriteLine($"[FontAtlas] WARNING: Atlas full at size {px}px, char '{ch}'");
                    break;
                }

                // Render glyph
                ctx.FillText(s, cursorX, cursorY + 1); // +1 for top padding

                charMap[ch] = new CharMetrics
                {
                    U0 = (float)cursorX / AtlasSize,
                    V0 = (float)cursorY / AtlasSize,
                    U1 = (float)(cursorX + glyphWidth) / AtlasSize,
                    V1 = (float)(cursorY + glyphHeight) / AtlasSize,
                    Advance = advance,
                    Width = glyphWidth,
                    Height = glyphHeight,
                    BearingY = ascent + 1, // offset from top of glyph box to baseline
                };

                cursorX += glyphWidth + 1;
                rowHeight = Math.Max(rowHeight, glyphHeight);
            }

            _metrics[size] = charMap;
        }

        // Read pixel data from canvas
        using var imageData = ctx.GetImageData(0, 0, AtlasSize, AtlasSize);
        using var dataArray = imageData.Data;
        var pixelBytes = dataArray.ReadBytes();

        // Upload to WebGPU texture
        Texture = device.CreateTexture(new GPUTextureDescriptor
        {
            Size = new[] { AtlasSize, AtlasSize },
            Format = "rgba8unorm",
            Usage = GPUTextureUsage.TextureBinding | GPUTextureUsage.CopyDst,
        });
        View = Texture.CreateView();

        queue.WriteTexture(
            new GPUTexelCopyTextureInfo { Texture = Texture },
            pixelBytes,
            new GPUTexelCopyBufferLayout
            {
                Offset = 0,
                BytesPerRow = (uint)(AtlasSize * 4),
                RowsPerImage = (uint)AtlasSize,
            },
            new uint[] { AtlasSize, AtlasSize }
        );

        Console.WriteLine($"[FontAtlas] Generated {AtlasSize}×{AtlasSize} atlas with {_metrics.Count} sizes, {_metrics.Values.Sum(m => m.Count)} glyphs");
    }

    /// <summary>Get metrics for a character at the given font size.</summary>
    public CharMetrics GetChar(char c, FontSize size)
    {
        if (_metrics.TryGetValue(size, out var map) && map.TryGetValue(c, out var m))
            return m;
        // Fallback: return space metrics
        if (_metrics.TryGetValue(size, out var fallback) && fallback.TryGetValue(' ', out var sp))
            return sp;
        return default;
    }

    /// <summary>Measure the width of a string in pixels at the given font size.</summary>
    public float MeasureString(string text, FontSize size)
    {
        float width = 0;
        foreach (char c in text)
            width += GetChar(c, size).Advance;
        return width;
    }

    /// <summary>Get the line height in pixels for the given font size.</summary>
    public float GetLineHeight(FontSize size)
    {
        var m = GetChar('M', size);
        return m.Height;
    }

    public void Dispose()
    {
        View?.Dispose();
        Texture?.Destroy();
        Texture?.Dispose();
    }
}
