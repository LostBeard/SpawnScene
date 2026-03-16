using System.Drawing;
using System.Numerics;

namespace SpawnScene.UI.Elements;

/// <summary>
/// Horizontal drag slider for float values.
/// Renders a track bar with a draggable thumb and value label.
/// </summary>
public class UISlider : UIElement
{
    public float MinValue { get; set; } = 0f;
    public float MaxValue { get; set; } = 1f;
    public float Value { get; set; } = 0.5f;
    public string Label { get; set; } = "";
    public string Format { get; set; } = "F2";
    public Action<float>? OnChanged { get; set; }

    // Colors
    public Color TrackColor { get; set; } = Color.FromArgb(255, 50, 50, 65);
    public Color FillColor { get; set; } = Color.FromArgb(255, 108, 92, 231);
    public Color ThumbColor { get; set; } = Color.White;
    public Color LabelColor { get; set; } = Color.FromArgb(255, 200, 200, 220);

    private const float TrackHeight = 6f;
    private const float ThumbRadius = 8f;
    private bool _dragging;

    public override void Update(InputManager input, float dt)
    {
        if (!Visible || !Enabled) return;

        var bounds = ScreenBounds;
        var mp = input.MousePosition;
        bool inBounds = mp.X >= bounds.X && mp.X < bounds.X + bounds.Width &&
                        mp.Y >= bounds.Y - 4 && mp.Y < bounds.Y + bounds.Height + 4;

        if (inBounds && input.WasMousePressed(0))
            _dragging = true;

        if (_dragging)
        {
            if (input.IsMouseDown(0))
            {
                // Map mouse X to value
                float trackX = bounds.X;
                float trackW = bounds.Width;
                float t = Math.Clamp((mp.X - trackX) / trackW, 0f, 1f);
                float newValue = MinValue + t * (MaxValue - MinValue);
                if (MathF.Abs(newValue - Value) > 0.001f)
                {
                    Value = newValue;
                    OnChanged?.Invoke(Value);
                }
            }
            else
            {
                _dragging = false;
            }
        }

        base.Update(input, dt);
    }

    public override void Draw(UIRenderer renderer)
    {
        if (!Visible) return;

        var bounds = ScreenBounds;
        float trackY = bounds.Y + bounds.Height / 2f - TrackHeight / 2f;
        float t = (MaxValue > MinValue) ? (Value - MinValue) / (MaxValue - MinValue) : 0f;

        // Label + value
        if (!string.IsNullOrEmpty(Label))
        {
            string text = $"{Label}: {Value.ToString(Format)}";
            renderer.DrawText(text, bounds.X, bounds.Y, FontSize.Caption, LabelColor);
        }

        float labelOffset = string.IsNullOrEmpty(Label) ? 0 : 18;
        float sliderY = bounds.Y + labelOffset + (bounds.Height - labelOffset) / 2f - TrackHeight / 2f;

        // Track background
        renderer.DrawRect(bounds.X, sliderY, bounds.Width, TrackHeight, TrackColor);

        // Filled portion
        float fillW = bounds.Width * t;
        if (fillW > 1)
            renderer.DrawRect(bounds.X, sliderY, fillW, TrackHeight, FillColor);

        // Thumb
        float thumbX = bounds.X + fillW - ThumbRadius;
        float thumbY = sliderY + TrackHeight / 2f - ThumbRadius;
        renderer.DrawRect(thumbX, thumbY, ThumbRadius * 2, ThumbRadius * 2,
                          _dragging ? FillColor : ThumbColor);

        base.Draw(renderer);
    }
}
