using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// FPS-style camera controller with WASD movement and mouse look.
/// Left-click drag or right-click drag = look (yaw/pitch).
/// Shift+Left or Middle drag = pan.
/// Scroll wheel = zoom/speed.
/// WASD = move, Q/E = down/up, Shift = fast.
/// </summary>
public class CameraController : IDisposable
{
    private readonly SceneManager _sceneManager;

    // FPS camera state
    private float _yaw;   // Horizontal rotation (radians)
    private float _pitch; // Vertical rotation (radians, clamped ±89°)
    private Vector3 _position = new(0, 0, 3);
    private float _moveSpeed = 0.2f;

    // Drag state
    private bool _isDragging;
    private bool _isPanning;
    private double _lastMouseX, _lastMouseY;

    // Key state
    private readonly HashSet<string> _heldKeys = new(StringComparer.OrdinalIgnoreCase);

    // Sensitivity
    private const float LookSensitivity = 0.001f;
    private const float PanSensitivity = 0.001f;
    private const float ZoomSensitivity = 0.02f;
    private const float MinPitch = -MathF.PI / 2f + 0.01f;
    private const float MaxPitch = MathF.PI / 2f - 0.01f;
    private const float FastMultiplier = 3.0f;

    /// <summary>Whether any movement keys are currently held.</summary>
    public bool IsMoving => _heldKeys.Count > 0;

    public CameraController(SceneManager sceneManager)
    {
        _sceneManager = sceneManager;
    }

    /// <summary>
    /// Reset camera to frame the current scene, looking at its center.
    /// </summary>
    public void FitToScene()
    {
        var scene = _sceneManager.ActiveScene;
        if (scene == null || scene.Count == 0) return;

        var camera = _sceneManager.Camera;

        // 2. FOV from camera intrinsics
        float fovY = 2f * MathF.Atan(camera.Height / (2f * camera.FocalY));

        // 3. Position based on scene type
        if (scene.SourceName == "depth-splat")
        {
            // Depth scenes place all splats at Z > 0 (posZ = depth).
            // Yaw = π gives Forward = (0,0,+1), looking in +Z direction toward the splats.
            _position = Vector3.Zero;
            _moveSpeed = 0.1f;
            _yaw = MathF.PI;
            _pitch = 0f;
        }
        else if (scene.Gaussians != null && scene.Gaussians.Length > 0)
        {
            // 1. Compute AABB → center + bounding sphere radius from CPU data
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            for (int i = 0; i < scene.Gaussians.Length; i++)
            {
                var pos = scene.Gaussians[i].Position;
                if (pos.X < -999990f) continue;
                min = Vector3.Min(min, pos);
                max = Vector3.Max(max, pos);
            }
            var center = (min + max) * 0.5f;
            float radius = Vector3.Distance(min, max) * 0.5f;
            if (radius < 0.001f) radius = 1.0f;

            float D = radius / MathF.Sin(fovY * 0.5f);
            D = MathF.Max(D, 0.5f);
            _position = center - Vector3.UnitZ * D;
            _moveSpeed = MathF.Max(radius * 0.05f, 0.1f);

            var dir = Vector3.Normalize(center - _position);
            _yaw = MathF.Atan2(dir.X, -dir.Z);
            _pitch = MathF.Asin(Math.Clamp(dir.Y, -1f, 1f));
        }
        else
        {
            // GPU-only scene, unknown source — safe default
            _position = new Vector3(0f, 0f, -3f);
            _moveSpeed = 0.1f;
            _yaw = 0f;
            _pitch = 0f;
        }

        UpdateCamera();
    }

    // --- Derived axes from yaw/pitch ---
    private Vector3 Forward
    {
        get
        {
            float cp = MathF.Cos(_pitch);
            return new Vector3(
                cp * MathF.Sin(_yaw),
                MathF.Sin(_pitch),
                -cp * MathF.Cos(_yaw)
            );
        }
    }

    private Vector3 Right => Vector3.Normalize(Vector3.Cross(Forward, Vector3.UnitY));

    private Vector3 Up => Vector3.Normalize(Vector3.Cross(Right, Forward));

    // --- Input handlers ---

    public void OnMouseDown(int button, double clientX, double clientY, bool shiftKey)
    {
        _isDragging = true;
        _isPanning = button == 1 || (button == 0 && shiftKey); // Middle or Shift+Left
        _lastMouseX = clientX;
        _lastMouseY = clientY;
    }

    public void OnMouseUp()
    {
        _isDragging = false;
        _isPanning = false;
    }

    public void OnMouseMove(double clientX, double clientY)
    {
        if (!_isDragging) return;

        double dx = clientX - _lastMouseX;
        double dy = clientY - _lastMouseY;
        _lastMouseX = clientX;
        _lastMouseY = clientY;

        if (_isPanning)
        {
            // Pan: move position in camera's XY plane
            float panScale = _moveSpeed * PanSensitivity;
            _position -= Right * (float)dx * panScale;
            _position += Up * (float)dy * panScale;
        }
        else
        {
            // Look: yaw/pitch
            _yaw += (float)dx * LookSensitivity;
            _pitch -= (float)dy * LookSensitivity; // Mouse up = look up (positive pitch)
            _pitch = Math.Clamp(_pitch, MinPitch, MaxPitch);
        }

        UpdateCamera();
    }

    public void OnWheel(double deltaY)
    {
        // Normalize: browser sends ±100+ per notch, we want ±1
        float normalized = Math.Clamp((float)deltaY / 100f, -2f, 2f);
        float movement = normalized * _moveSpeed * 0.01f;
        Console.WriteLine($"[Wheel] deltaY={deltaY:F1} normalized={normalized:F3} moveSpeed={_moveSpeed:F3} movement={movement:F6} pos={_position}");
        _position -= Forward * movement;
        UpdateCamera();
    }

    public void OnKeyDown(string key)
    {
        _heldKeys.Add(key.ToLowerInvariant());
    }

    public void OnKeyUp(string key)
    {
        _heldKeys.Remove(key.ToLowerInvariant());
    }

    /// <summary>
    /// Tick the camera movement based on held keys. Call each frame.
    /// </summary>
    /// <param name="dt">Delta time in seconds.</param>
    /// <returns>True if camera moved.</returns>
    public bool Tick(float dt)
    {
        if (_heldKeys.Count == 0) return false;

        float speed = _moveSpeed * dt;
        if (_heldKeys.Contains("shift")) speed *= FastMultiplier;

        var move = Vector3.Zero;

        if (_heldKeys.Contains("w")) move += Forward;
        if (_heldKeys.Contains("s")) move -= Forward;
        if (_heldKeys.Contains("a")) move -= Right;
        if (_heldKeys.Contains("d")) move += Right;
        if (_heldKeys.Contains("e")) move += Vector3.UnitY;
        if (_heldKeys.Contains("q")) move -= Vector3.UnitY;

        if (move.LengthSquared() < 0.001f) return false;

        _position += Vector3.Normalize(move) * speed;
        UpdateCamera();
        return true;
    }

    private void UpdateCamera()
    {
        var camera = _sceneManager.Camera;
        camera.Position = _position;
        camera.Forward = Forward;
        camera.Up = Vector3.UnitY;
        _sceneManager.Camera = camera;
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}
