using System.Drawing;
using System.Numerics;

namespace SpawnScene.UI.Elements;

/// <summary>
/// Clickable button: background rectangle + centered text label.
/// Supports hover and pressed visual states.
/// </summary>
public class UIButton : UIElement
{
    public string Text { get; set; } = "";
    public FontSize FontSize { get; set; } = FontSize.Body;
    public Action? OnClick { get; set; }

    // Colors
    public Color NormalColor { get; set; } = Color.FromArgb(255, 108, 92, 231);   // purple
    public Color HoverColor { get; set; } = Color.FromArgb(255, 129, 116, 236);   // lighter purple
    public Color PressedColor { get; set; } = Color.FromArgb(255, 86, 72, 200);   // darker purple
    public Color DisabledColor { get; set; } = Color.FromArgb(255, 60, 60, 70);
    public Color TextColor { get; set; } = Color.White;

    // State
    public bool IsHovered { get; private set; }
    public bool IsPressed { get; private set; }

    public float PaddingX { get; set; } = 16;
    public float PaddingY { get; set; } = 8;

    public override void Update(InputManager input, float dt)
    {
        if (!Visible || !Enabled)
        {
            IsHovered = false;
            IsPressed = false;
            return;
        }

        var bounds = ScreenBounds;
        var mp = input.MousePosition;
        IsHovered = mp.X >= bounds.X && mp.X < bounds.X + bounds.Width &&
                    mp.Y >= bounds.Y && mp.Y < bounds.Y + bounds.Height;

        if (IsHovered)
        {
            IsPressed = input.IsMouseDown(0);
            if (input.WasMouseReleased(0))
                OnClick?.Invoke();
        }
        else
        {
            IsPressed = false;
        }

        base.Update(input, dt);
    }

    public override void Draw(UIRenderer renderer)
    {
        if (!Visible) return;

        var bounds = ScreenBounds;
        Color bgColor = !Enabled ? DisabledColor :
                         IsPressed ? PressedColor :
                         IsHovered ? HoverColor :
                         NormalColor;

        // Background
        renderer.DrawRect(bounds.X, bounds.Y, bounds.Width, bounds.Height, bgColor);

        // Centered text
        if (!string.IsNullOrEmpty(Text))
        {
            float textW = renderer.MeasureText(Text, FontSize);
            float textH = renderer.GetLineHeight(FontSize);
            float textX = bounds.X + (bounds.Width - textW) / 2;
            float textY = bounds.Y + (bounds.Height - textH) / 2;
            renderer.DrawText(Text, textX, textY, FontSize, Enabled ? TextColor : Color.Gray);
        }

        base.Draw(renderer);
    }

    /// <summary>Auto-size the button to fit its text + padding.</summary>
    public void AutoSize(UIRenderer renderer)
    {
        float textW = renderer.MeasureText(Text, FontSize);
        float textH = renderer.GetLineHeight(FontSize);
        Width = textW + PaddingX * 2;
        Height = textH + PaddingY * 2;
    }
}
