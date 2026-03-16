using System.Drawing;

namespace SpawnScene.UI.Elements;

/// <summary>
/// Text label rendered via the font atlas.
/// Auto-sizes Width/Height to fit the text if not explicitly set.
/// </summary>
public class UILabel : UIElement
{
    private string _text = "";
    private FontSize _fontSize = FontSize.Body;
    private bool _dirty = true;

    public string Text
    {
        get => _text;
        set { if (_text != value) { _text = value; _dirty = true; } }
    }

    public FontSize FontSize
    {
        get => _fontSize;
        set { if (_fontSize != value) { _fontSize = value; _dirty = true; } }
    }

    public Color Color { get; set; } = Color.White;

    public override void Draw(UIRenderer renderer)
    {
        if (!Visible || string.IsNullOrEmpty(Text)) return;

        // Auto-size on first draw or when text changes
        if (_dirty)
        {
            Width = renderer.MeasureText(Text, FontSize);
            Height = renderer.GetLineHeight(FontSize);
            _dirty = false;
        }

        var bounds = ScreenBounds;
        renderer.DrawText(Text, bounds.X, bounds.Y, FontSize, Color);
        base.Draw(renderer);
    }
}
