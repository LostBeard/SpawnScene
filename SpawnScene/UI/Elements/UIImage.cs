using SpawnDev.BlazorJS.JSObjects;
using System.Drawing;

namespace SpawnScene.UI.Elements;

/// <summary>
/// Displays a GPU texture as a rectangular image.
/// Set TextureView after uploading image data to a GPUTexture.
/// </summary>
public class UIImage : UIElement
{
    /// <summary>The GPU texture view to display. Null = draws a placeholder rect.</summary>
    public GPUTextureView? TextureView { get; set; }

    /// <summary>Placeholder color when no texture is set.</summary>
    public Color PlaceholderColor { get; set; } = Color.FromArgb(255, 40, 40, 55);

    public override void Draw(UIRenderer renderer)
    {
        if (!Visible) return;

        var bounds = ScreenBounds;

        if (TextureView != null)
        {
            renderer.DrawImage(TextureView, bounds.X, bounds.Y, bounds.Width, bounds.Height);
        }
        else
        {
            renderer.DrawRect(bounds.X, bounds.Y, bounds.Width, bounds.Height, PlaceholderColor);
        }

        base.Draw(renderer);
    }
}
