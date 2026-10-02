using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using System.Drawing;

namespace SpawnScene.Pages;

// ?autotest=textlab - the UI's text, drawn the same way from both font atlases, for the thin-stroke question
// (2026-10-02: hyphens and "=" bars at Caption size rendered faint or as dots). Speaks the dataset harness's markers:
// AUTOTEST=textlab node tools/_cdp_dataset.js Text 0 -> free-textlab_sdf, free-textlab_bitmap.
public partial class Studio
{
    async Task RunTextLabAsync()
    {
        const string sample = "autotest Bathroom 1002-095123  well-covered  251-photo  (Photo = their size)  3,000,000  a·b  x×y  -=-=-";
        var sizes = new[] { FontSize.Caption, FontSize.Body, FontSize.Heading };
        foreach (var mode in new[] { TextRenderMode.Sdf, TextRenderMode.Bitmap, TextRenderMode.Auto })
        {
            _gameUI.Renderer.TextMode = mode;
            _uiRoot.ClearChildren();
            var panel = _uiRoot.AddChild(new UIPanel { X = 20, Y = 20, Width = _canvasWidth - 40, Height = _canvasHeight - 40 });
            panel.AddChild(new UILabel { X = 16, Y = 12, Text = $"Text mode: {mode}", FontSize = FontSize.Heading, Color = UITheme.Current.TextPrimary });
            float y = 56;
            foreach (var size in sizes)
            {
                // The same string on four consecutive pixel rows' worth of offsets: a stroke that depends on the vertical
                // phase shows up as a difference between these lines.
                for (int k = 0; k < 4; k++)
                {
                    panel.AddChild(new UILabel
                    {
                        X = 16, Y = y + k * 0.25f, Text = $"{size} +{k * 0.25f:F2}: {sample}",
                        FontSize = size, Color = UITheme.Current.TextSecondary,
                    });
                    y += (int)size * 1.6f + 4;
                }
                y += 10;
            }
            await Task.Delay(1500);
            Console.WriteLine($"[Dataset] READY-FOR-CAPTURE free-textlab_{mode.ToString().ToLowerInvariant()}");
            await Task.Delay(2500);
        }
        _gameUI.Renderer.TextMode = TextRenderMode.Auto;
        Console.WriteLine("[Dataset] DONE");
    }
}
