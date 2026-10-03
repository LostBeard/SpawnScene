using System.Drawing;
using System.Numerics;
using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnDev.GameUI.Input;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// The in-headset menu: a GameUI <see cref="UIWorldPanel"/> floating in the room (VR and AR alike - the whole UI is
/// canvas-drawn so it can live here too). The X/A button opens it in front of the viewer, world-locked where it opened;
/// the right controller's ray points at it (a cursor dot shows where) and the trigger clicks. Buttons are ordinary
/// GameUI children: they receive a pointer at the ray's hit, in panel pixels, through a GameInput of their own.
///
/// Placement and hit-testing use the exact transform the panel is drawn with (<see cref="UIWorldPanel.DrawWorld"/>:
/// panel pixel (x, y) -> local ((x / W) - 0.5, 0.5 - (y / H), 0), scaled to metres by W and H times WorldScale, then the
/// panel's world transform). GameUI's own RayIntersectsPanel ignores WorldScale, so it is not used.
/// </summary>
public sealed class XRMenu
{
    public const float PanelWidth = XRMenuGeometry.PanelWidth, PanelHeight = XRMenuGeometry.PanelHeight;
    public const float WorldScale = XRMenuGeometry.WorldScale;
    const float CursorSize = 14;

    readonly GameInput _input = new();
    readonly UIPanel _cursor;
    bool _triggerWasDown;
    Matrix4x4 _model = Matrix4x4.Identity;

    /// <summary>The panel; add buttons and labels to it in panel pixels.</summary>
    public UIWorldPanel Panel { get; }
    public bool IsOpen { get; private set; }
    /// <summary>Where the ray meets the panel this frame, in panel pixels (null: not on the panel or closed).</summary>
    public Vector2? Cursor { get; private set; }

    public XRMenu()
    {
        Panel = new UIWorldPanel
        {
            PanelWidth = PanelWidth, PanelHeight = PanelHeight, WorldScale = WorldScale,
            BackgroundColor = Color.FromArgb(225, 14, 18, 26),
            BorderColor = Color.FromArgb(255, 70, 110, 150), BorderWidth = 2, CornerRadius = 14,
        };
        _cursor = new UIPanel
        {
            Width = CursorSize, Height = CursorSize, CornerRadius = CursorSize / 2, BorderWidth = 0,
            BackgroundColor = Color.FromArgb(240, 120, 220, 255), Visible = false,
        };
    }

    /// <summary>Open in front of the head (or close if open).</summary>
    public void Toggle(Vector3 headPosition, Quaternion headOrientation)
    {
        IsOpen = !IsOpen;
        if (IsOpen) _model = XRMenuGeometry.PlaceInFront(headPosition, headOrientation);
        Cursor = null;
        _cursor.Visible = false;
    }

    public void Close() { IsOpen = false; Cursor = null; _cursor.Visible = false; }

    /// <summary>Where the panel is (its world transform in the room).</summary>
    public Matrix4x4 Model => _model;

    /// <summary>Open at a given place - another menu's (a sub-page replacing it where it stood).</summary>
    public void OpenAt(Matrix4x4 model)
    {
        _model = model;
        IsOpen = true;
        Cursor = null;
        _cursor.Visible = false;
    }

    /// <summary>
    /// One frame of pointing. Returns true while the ray is on the open panel, so the caller does not also treat the
    /// trigger as a scene action (AR placement).
    /// </summary>
    public bool Step(Vector3? rayOrigin, Vector3? rayDirection, bool triggerDown, float dt)
    {
        Vector2? px = null;
        if (IsOpen && rayOrigin is { } o && rayDirection is { } d)
            px = XRMenuGeometry.RayToPanelPixel(_model, o, d);
        Cursor = px;

        _input.Poll();   // no providers: clears last frame's pointers
        if (px is { } p)
        {
            _input.AddPointer(new Pointer
            {
                Type = PointerType.Controller,
                ScreenPosition = p,
                IsPressed = triggerDown,
                WasPressed = triggerDown && !_triggerWasDown,
                WasReleased = !triggerDown && _triggerWasDown,
            });
        }
        _triggerWasDown = triggerDown;
        if (IsOpen) Panel.Update(_input, dt);
        return px != null;
    }

    /// <summary>Draw the panel (and the cursor) into an eye view; <paramref name="roomViewProjection"/> is the eye's
    /// room-space view times its projection (metres, the space the panel was placed in).</summary>
    public void Draw(UIRenderer renderer, Matrix4x4 roomViewProjection, GPUCommandEncoder encoder,
        GPUTextureView colorTarget, GPUTextureView depthTarget)
    {
        if (!IsOpen) return;
        // The cursor is the last child so it draws on top.
        if (_cursor.Parent != Panel) Panel.AddChild(_cursor);
        else if (Panel.Children[^1] != _cursor) { Panel.RemoveChild(_cursor); Panel.AddChild(_cursor); }
        _cursor.Visible = Cursor != null;
        if (Cursor is { } c) { _cursor.X = c.X - CursorSize / 2; _cursor.Y = c.Y - CursorSize / 2; }
        Panel.WorldTransform = _model;
        Panel.DrawWorld(renderer, roomViewProjection, encoder, colorTarget, depthTarget);
    }
}
