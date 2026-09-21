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
    private float _moveSpeed = 1.0f;

    // Drag state
    private bool _isDragging;
    private bool _isPanning;
    private double _lastMouseX, _lastMouseY;

    // Key state
    private readonly HashSet<string> _heldKeys = new(StringComparer.OrdinalIgnoreCase);

    // Sensitivity
    private const float LookSensitivity = 0.002f;
    private const float PanSensitivity = 0.001f;
    private const float ZoomSensitivity = 0.02f;
    private const float MinPitch = -MathF.PI / 2f + 0.01f;
    private const float MaxPitch = MathF.PI / 2f - 0.01f;
    private const float FastMultiplier = 4.0f;

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
            _moveSpeed = 1.0f;
            _yaw = MathF.PI;
            _pitch = 0f;
        }
        else if (scene.SourceName == "multi-view" && scene.TrainingCameras.Count > 0)
        {
            // Multi-view: sit at first camera, look at the camera-ring / scene center (not raw Forward —
            // Forward can disagree slightly with the intended look-at after depth scaling).
            var camCenter = Vector3.Zero;
            foreach (var cam in scene.TrainingCameras)
                camCenter += cam.Position;
            camCenter /= scene.TrainingCameras.Count;

            var firstCam = scene.TrainingCameras[0];
            _position = firstCam.Position;

            float maxDist = 0;
            foreach (var cam in scene.TrainingCameras)
                maxDist = MathF.Max(maxDist, Vector3.Distance(cam.Position, camCenter));
            _moveSpeed = MathF.Max(maxDist * 0.5f, 0.05f);

            // Aim where the cameras' rays CONVERGE, which is what they were looking at.
            //
            // This used to try a literal TempleRing bounding-box midpoint first and fall back to
            // the camera CENTROID. Both are wrong. The hardcoded point is a coin toss on a
            // meaningless coordinate for any other scene - on drjohnson it won and produced
            // 1,129,128 splats and a black frame. The centroid is where the cameras are STANDING,
            // so on a row of cameras the viewer looks along the row: measured at dot 0.56 with
            // the direction to the subject, on a rig built to check exactly that.
            Vector3 dir = firstCam.Forward;
            if (WorldSpaceGeometry.TryConvergencePoint(scene.TrainingCameras, out var lookAt))
            {
                var toLookAt = lookAt - _position;
                if (toLookAt.LengthSquared() > 1e-6f) dir = toLookAt;
            }
            else
            {
                // Parallel rays - no convergence point exists, so keep the camera's own heading
                // rather than inventing one.
                var toCenter = camCenter - _position;
                if (toCenter.LengthSquared() > 1e-6f) dir = toCenter;
            }
            dir = Vector3.Normalize(dir);
            _yaw = MathF.Atan2(dir.X, -dir.Z);
            _pitch = MathF.Asin(Math.Clamp(dir.Y, -1f, 1f));
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
            _moveSpeed = MathF.Max(radius * 0.5f, 1.0f);

            var dir = Vector3.Normalize(center - _position);
            _yaw = MathF.Atan2(dir.X, -dir.Z);
            _pitch = MathF.Asin(Math.Clamp(dir.Y, -1f, 1f));
        }
        else
        {
            // GPU-only scene, unknown source — safe default
            _position = new Vector3(0f, 0f, -3f);
            _moveSpeed = 1.0f;
            _yaw = 0f;
            _pitch = 0f;
        }

        UpdateCamera();
    }

    // --- Derived axes from yaw/pitch ---
    private Vector3 Forward => WorldSpaceGeometry.ForwardFromYawPitch(_yaw, _pitch);

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

    /// <summary>
    /// Handle mouse movement. When isPointerLocked=true, dx/dy are raw movement deltas.
    /// When false, dx/dy are absolute client coordinates (legacy drag mode).
    /// </summary>
    public void OnMouseMove(double dx, double dy, bool isPointerLocked = false)
    {
        if (isPointerLocked)
        {
            // Pointer lock: dx/dy are movement deltas — always apply look
            _yaw += (float)dx * LookSensitivity;
            _pitch -= (float)dy * LookSensitivity;
            _pitch = Math.Clamp(_pitch, MinPitch, MaxPitch);
            UpdateCamera();
            return;
        }

        // Legacy drag mode: dx/dy are absolute client coordinates
        if (!_isDragging) return;

        double deltaX = dx - _lastMouseX;
        double deltaY = dy - _lastMouseY;
        _lastMouseX = dx;
        _lastMouseY = dy;

        if (_isPanning)
        {
            // Pan: move position in camera's XY plane
            float panScale = _moveSpeed * PanSensitivity;
            _position -= Right * (float)deltaX * panScale;
            _position += Up * (float)deltaY * panScale;
        }
        else
        {
            // Look: yaw/pitch
            _yaw += (float)deltaX * LookSensitivity;
            _pitch -= (float)deltaY * LookSensitivity;
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

    /// <summary>
    /// Park the camera at an EXACT pose, orientation included.
    ///
    /// The normal input path stores yaw/pitch and forces <c>Up = world +Y</c>, which cannot
    /// represent roll. A dataset ground-truth pose generally IS rolled, so this writes
    /// <paramref name="forward"/>/<paramref name="up"/> straight onto the camera and then syncs
    /// yaw/pitch so that if the user takes over afterwards, movement continues from here rather
    /// than snapping. Roll is lost the moment they look around - that is inherent to a yaw/pitch
    /// controller, not a bug here.
    ///
    /// Used by the novel-view fidelity gate, and the same primitive saved viewpoints /
    /// hotspots will need (NOTES.md SuperSplat parity).
    /// </summary>
    public void SetPose(Vector3 position, Vector3 forward, Vector3 up)
    {
        var f = Vector3.Normalize(forward);
        var u = Vector3.Normalize(up);

        // Re-orthogonalize up against forward (Gram-Schmidt) so the view matrix is well formed
        // even if the dataset's stored up is only approximately perpendicular.
        var right = Vector3.Cross(f, u);
        if (right.LengthSquared() < 1e-12f)
            right = Vector3.Cross(f, MathF.Abs(f.Y) > 0.9f ? Vector3.UnitX : Vector3.UnitY);
        right = Vector3.Normalize(right);
        u = Vector3.Normalize(Vector3.Cross(right, f));

        _position = position;

        // Keep the yaw/pitch model consistent with where we just pointed, so if the user takes
        // over with WASD the view continues from here instead of snapping.
        WorldSpaceGeometry.YawPitchFromForward(f, out float yaw, out float pitch);
        _yaw = yaw;
        _pitch = Math.Clamp(pitch, MinPitch, MaxPitch);

        // Apply the pose this controller can actually HOLD, not the one it was handed.
        //
        // It used to assign the requested basis including its roll, and then the first input
        // event called UpdateCamera and replaced the up with WorldUp - so the view jumped the
        // instant anyone touched it. Measured on a rolled capture pose: the up flipped to dot
        // -0.97, a room on its side. Even an upright pose tilted, to 0.983, because the
        // orthogonalised up is not world up when the forward is pitched.
        //
        // A yaw/pitch camera cannot represent roll (ForwardFromYawPitch is defined about
        // WorldUp), so promising to keep it is a lie with a one-frame delay. Say so instead.
        float roll = Vector3.Dot(u, WorldUp);
        if (roll < 0.98f)
            Console.WriteLine(
                $"[Camera] the pose has roll this controller cannot hold (up . worldUp = {roll:F3}). " +
                "Using world up. If the scene looks tilted, it needs gravity alignment - see " +
                "MultiViewGenerationService.AlignToGravityAsync.");

        var camera = _sceneManager.Camera;
        camera.Position = position;
        camera.Forward = Forward;     // from the yaw/pitch just derived, so it is reproducible
        camera.Up = WorldUp;
        _sceneManager.Camera = camera;
    }

    /// <summary>
    /// Which way is up for this controller. +Y, because the yaw/pitch model is defined about it
    /// (<see cref="WorldSpaceGeometry.ForwardFromYawPitch"/>) - this is not a free parameter yet,
    /// it is a named assumption.
    ///
    /// It used to be a bare Vector3.UnitY literal inside UpdateCamera, and that cost a session.
    /// SetPose carefully computes a capture camera's true up INCLUDING its roll, and then the
    /// first mouse move or WASD step called UpdateCamera and replaced it with world +Y. A
    /// reconstruction has no gravity in it - DAv3 and COLMAP both recover geometry up to an
    /// arbitrary rotation - so Bathroom came out with its up at essentially -Y and the room
    /// rendered on its side and tumbled when the camera moved, while every number stayed good,
    /// because the TRAINER renders from the real camera basis and only the viewer forces this.
    ///
    /// The fix is upstream: MultiViewGenerationService.AlignToGravityAsync stands the
    /// reconstruction up so +Y is true. This name exists so the assumption is visible at the
    /// point it is made, and so the day this controller needs to support a scene that cannot be
    /// aligned, there is one place to change rather than a literal to go hunting for.
    /// </summary>
    public Vector3 WorldUp { get; } = Vector3.UnitY;

    private void UpdateCamera()
    {
        var camera = _sceneManager.Camera;
        camera.Position = _position;
        camera.Forward = Forward;
        camera.Up = WorldUp;
        _sceneManager.Camera = camera;
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}
