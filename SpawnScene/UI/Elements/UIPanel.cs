using System.Drawing;

namespace SpawnScene.UI.Elements;

/// <summary>
/// Container panel with a solid background color.
/// Children are positioned relative to this panel's top-left corner.
/// </summary>
public class UIPanel : UIElement
{
    public Color BackgroundColor { get; set; } = Color.FromArgb(200, 20, 20, 30);
    public Color BorderColor { get; set; } = Color.FromArgb(60, 255, 255, 255);
    public float BorderWidth { get; set; } = 0;

    public float Padding { get; set; } = 8;

    public override void Draw(UIRenderer renderer)
    {
        if (!Visible) return;

        var bounds = ScreenBounds;

        // Border (drawn as a slightly larger rect behind the background)
        if (BorderWidth > 0)
        {
            renderer.DrawRect(bounds.X - BorderWidth, bounds.Y - BorderWidth,
                              bounds.Width + BorderWidth * 2, bounds.Height + BorderWidth * 2,
                              BorderColor);
        }

        // Background
        renderer.DrawRect(bounds.X, bounds.Y, bounds.Width, bounds.Height, BackgroundColor);

        // Children
        base.Draw(renderer);
    }
}
