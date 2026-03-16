using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using System.Numerics;

namespace SpawnScene.UI;

/// <summary>
/// Unified polling-based input manager for mouse, keyboard, and gamepad.
/// Designed for game-loop style polling (compatible with gamepad and future VR controllers).
/// Call Attach() once, Poll() each frame, Detach() on dispose.
/// </summary>
public class InputManager : IDisposable
{
    // Mouse state — event-based press/release tracking (DOM events fire between RAF frames)
    private Vector2 _mousePos;
    private readonly bool[] _mouseDown = new bool[3];          // current held state
    private readonly bool[] _pendingPressed = new bool[3];     // accumulated between polls
    private readonly bool[] _pendingReleased = new bool[3];    // accumulated between polls
    private readonly bool[] _framePressed = new bool[3];       // snapshot for this frame
    private readonly bool[] _frameReleased = new bool[3];      // snapshot for this frame
    private float _scrollDelta;
    private float _scrollAccum; // accumulated between polls

    // Keyboard state
    private readonly HashSet<string> _keysDown = new();
    private readonly HashSet<string> _keysPrev = new();
    private readonly HashSet<string> _keysDownSnapshot = new();
    private readonly HashSet<string> _pendingKeyPressed = new();
    private readonly HashSet<string> _pendingKeyReleased = new();
    private readonly HashSet<string> _frameKeyPressed = new();
    private readonly HashSet<string> _frameKeyReleased = new();
    private string? _textInput;
    private string _textInputAccum = "";

    // Gamepad
    private Gamepad?[] _gamepads = System.Array.Empty<Gamepad?>();

    // DOM event callbacks (prevent GC)
    private ActionCallback<MouseEvent>? _onMouseMove;
    private ActionCallback<MouseEvent>? _onMouseDown;
    private ActionCallback<MouseEvent>? _onMouseUp;
    private ActionCallback<WheelEvent>? _onWheel;
    private ActionCallback<KeyboardEvent>? _onKeyDown;
    private ActionCallback<KeyboardEvent>? _onKeyUp;
    private HTMLCanvasElement? _canvas;   // owned by InputManager — disposed in Detach
    private Window? _window;              // owned — disposed in Detach
    private bool _attached;

    // Public mouse API
    public Vector2 MousePosition => _mousePos;
    public bool IsMouseDown(int button = 0) => button < 3 && _mouseDown[button];
    public bool WasMousePressed(int button = 0) => button < 3 && _framePressed[button];
    public bool WasMouseReleased(int button = 0) => button < 3 && _frameReleased[button];
    public float ScrollDelta => _scrollDelta;

    // Public keyboard API
    public bool IsKeyDown(string key) => _keysDownSnapshot.Contains(key);
    public bool WasKeyPressed(string key) => _frameKeyPressed.Contains(key);
    public IReadOnlyCollection<string> FrameKeysPressed => _frameKeyPressed;
    public IReadOnlyCollection<string> FrameKeysReleased => _frameKeyReleased;
    public string? TextInput => _textInput;

    // Public gamepad API
    public bool HasGamepad => _gamepads.Any(g => g != null && g.Connected);
    public Vector2 LeftStick => GetGamepadAxes(0, 1);
    public Vector2 RightStick => GetGamepadAxes(2, 3);
    public bool IsGamepadButtonDown(int button) => GetGamepadButton(button);
    public bool WasGamepadButtonPressed(int button) => GetGamepadButton(button); // simplified for now

    /// <summary>Attach DOM event listeners to the canvas element.</summary>
    public void Attach(ElementReference canvasRef)
    {
        if (_attached) return;
        _attached = true;
        // Create our own HTMLCanvasElement wrapper — we own its lifetime
        _canvas = new HTMLCanvasElement(canvasRef);

        _onMouseMove = new ActionCallback<MouseEvent>(OnMouseMove);
        _onMouseDown = new ActionCallback<MouseEvent>(OnMouseDown);
        _onMouseUp = new ActionCallback<MouseEvent>(OnMouseUp);
        _onWheel = new ActionCallback<WheelEvent>(OnWheel);
        _onKeyDown = new ActionCallback<KeyboardEvent>(OnKeyDown);
        _onKeyUp = new ActionCallback<KeyboardEvent>(OnKeyUp);

        _canvas.AddEventListener("mousemove", _onMouseMove);
        _canvas.AddEventListener("mousedown", _onMouseDown);
        _canvas.AddEventListener("mouseup", _onMouseUp);
        _canvas.AddEventListener("wheel", _onWheel);

        // Keyboard events on window (canvas may not have focus) — we own this too
        _window = BlazorJSRuntime.JS.Get<Window>("window");
        _window.AddEventListener("keydown", _onKeyDown);
        _window.AddEventListener("keyup", _onKeyUp);
    }

    /// <summary>
    /// Snapshot current input state for this frame.
    /// Call at the start of each RAF tick, before UI update.
    /// </summary>
    public void Poll()
    {
        // Mouse: snapshot pending press/release events, then clear pending
        System.Array.Copy(_pendingPressed, _framePressed, 3);
        System.Array.Copy(_pendingReleased, _frameReleased, 3);
        System.Array.Clear(_pendingPressed);
        System.Array.Clear(_pendingReleased);

        // Scroll: consume accumulated delta
        _scrollDelta = _scrollAccum;
        _scrollAccum = 0;

        // Keyboard: snapshot pending key presses and releases
        _frameKeyPressed.Clear();
        foreach (var k in _pendingKeyPressed) _frameKeyPressed.Add(k);
        _pendingKeyPressed.Clear();
        _frameKeyReleased.Clear();
        foreach (var k in _pendingKeyReleased) _frameKeyReleased.Add(k);
        _pendingKeyReleased.Clear();

        _keysDownSnapshot.Clear();
        foreach (var k in _keysDown) _keysDownSnapshot.Add(k);

        // Text input: consume accumulated chars
        _textInput = _textInputAccum.Length > 0 ? _textInputAccum : null;
        _textInputAccum = "";

        // Gamepad: poll connected gamepads
        try
        {
            using var navigator = BlazorJSRuntime.JS.Get<Navigator>("navigator");
            _gamepads = navigator.GetGamepads();
        }
        catch
        {
            _gamepads = System.Array.Empty<Gamepad?>();
        }
    }

    // DOM event handlers (buffer state between polls)
    private void OnMouseMove(MouseEvent e)
    {
        _mousePos = new Vector2((float)e.OffsetX, (float)e.OffsetY);
    }

    private void OnMouseDown(MouseEvent e)
    {
        int btn = (int)e.Button;
        if (btn < 3) { _mouseDown[btn] = true; _pendingPressed[btn] = true; }
    }

    private void OnMouseUp(MouseEvent e)
    {
        int btn = (int)e.Button;
        if (btn < 3) { _mouseDown[btn] = false; _pendingReleased[btn] = true; }
    }

    private void OnWheel(WheelEvent e)
    {
        _scrollAccum += (float)e.DeltaY;
    }

    private void OnKeyDown(KeyboardEvent e)
    {
        _keysDown.Add(e.Key);
        _pendingKeyPressed.Add(e.Key);
        // Accumulate printable characters for text input
        if (e.Key.Length == 1 && !e.CtrlKey && !e.AltKey && !e.MetaKey)
            _textInputAccum += e.Key;
    }

    private void OnKeyUp(KeyboardEvent e)
    {
        _keysDown.Remove(e.Key);
        _pendingKeyReleased.Add(e.Key);
    }

    // Gamepad helpers
    private Vector2 GetGamepadAxes(int xAxis, int yAxis)
    {
        var gp = _gamepads.FirstOrDefault(g => g != null && g.Connected);
        if (gp == null) return Vector2.Zero;
        var axes = gp.Axes;
        float x = xAxis < axes.Length ? (float)axes[xAxis] : 0;
        float y = yAxis < axes.Length ? (float)axes[yAxis] : 0;
        // Dead zone
        if (MathF.Abs(x) < 0.15f) x = 0;
        if (MathF.Abs(y) < 0.15f) y = 0;
        return new Vector2(x, y);
    }

    private bool GetGamepadButton(int button)
    {
        var gp = _gamepads.FirstOrDefault(g => g != null && g.Connected);
        if (gp == null) return false;
        var buttons = gp.Buttons;
        return button < buttons.Length && buttons[button].Pressed;
    }

    public void Detach()
    {
        if (!_attached) return;
        _attached = false;

        try
        {
            _canvas?.RemoveEventListener("mousemove", _onMouseMove);
            _canvas?.RemoveEventListener("mousedown", _onMouseDown);
            _canvas?.RemoveEventListener("mouseup", _onMouseUp);
            _canvas?.RemoveEventListener("wheel", _onWheel);
        }
        catch { }

        try
        {
            _window?.RemoveEventListener("keydown", _onKeyDown);
            _window?.RemoveEventListener("keyup", _onKeyUp);
        }
        catch { }

        _canvas?.Dispose();
        _canvas = null;
        _window?.Dispose();
        _window = null;
    }

    public void Dispose()
    {
        Detach();
        _onMouseMove?.Dispose();
        _onMouseDown?.Dispose();
        _onMouseUp?.Dispose();
        _onWheel?.Dispose();
        _onKeyDown?.Dispose();
        _onKeyUp?.Dispose();
    }
}
