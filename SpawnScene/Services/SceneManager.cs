using SpawnScene.Models;

namespace SpawnScene.Services;

/// <summary>
/// Manages loaded Gaussian Splat scenes and the active camera.
/// Acts as the central state store for the application.
/// </summary>
public class SceneManager
{
    private GaussianScene? _activeScene;
    private CameraParams _camera;

    /// <summary>Fired when the active scene changes.</summary>
    public event Action? OnSceneChanged;

    /// <summary>Fired when the camera changes.</summary>
    public event Action? OnCameraChanged;

    /// <summary>The currently loaded scene.</summary>
    public GaussianScene? ActiveScene
    {
        get => _activeScene;
        set
        {
            _activeScene = value;
            if (value != null)
            {
                // Position camera to frame the scene
                FitCameraToScene(value);
            }
            OnSceneChanged?.Invoke();
        }
    }

    /// <summary>The active camera.</summary>
    public CameraParams Camera
    {
        get => _camera;
        set
        {
            _camera = value;
            OnCameraChanged?.Invoke();
        }
    }

    /// <summary>Whether a scene is currently loaded.</summary>
    public bool HasScene => _activeScene != null;

    public SceneManager()
    {
        _camera = CameraParams.CreateDefault(800, 600);
    }

    /// <summary>
    /// Position the camera to see the entire scene.
    /// </summary>
    public void FitCameraToScene(GaussianScene scene)
    {
        if (scene.Count == 0) return;

        // 2. Set focal length for current viewport
        if (_camera.Width > 0)
        {
            _camera.FocalX = MathF.Max(_camera.Width, _camera.Height) * 1.2f;
            _camera.FocalY = _camera.FocalX;
        }

        // 3. Calculate FOV from focal length
        float fovY = 2f * MathF.Atan(_camera.Height / (2f * _camera.FocalY));

        // 4. Position camera based on scene type
        if (scene.SourceName == "depth-splat")
        {
            // Depth-unprojected scenes: camera at capture origin (0,0,0) looking +Z
            // Splats are already in world space from the capture viewpoint — no AABB needed.
            _camera.Position = System.Numerics.Vector3.Zero;
            _camera.Forward = System.Numerics.Vector3.UnitZ;
        }
        else if (scene.Gaussians != null && scene.Gaussians.Length > 0)
        {
            // 1. Compute scene AABB → center + bounding sphere radius (CPU data only)
            var min = new System.Numerics.Vector3(float.MaxValue);
            var max = new System.Numerics.Vector3(float.MinValue);
            for (int i = 0; i < scene.Gaussians.Length; i++)
            {
                var pos = scene.Gaussians[i].Position;
                if (pos.X < -999990f) continue;
                min = System.Numerics.Vector3.Min(min, pos);
                max = System.Numerics.Vector3.Max(max, pos);
            }
            var center = (min + max) * 0.5f;
            float radius = System.Numerics.Vector3.Distance(min, max) * 0.5f;
            if (radius < 0.001f) radius = 1.0f;

            float D = radius / MathF.Sin(fovY * 0.5f);
            D = MathF.Max(D, 0.5f);
            _camera.Position = center - System.Numerics.Vector3.UnitZ * D;
            _camera.Forward = System.Numerics.Vector3.UnitZ;
        }
        else
        {
            // GPU-only scene with no CPU Gaussians and unknown source — use a safe default.
            _camera.Position = new System.Numerics.Vector3(0f, 0f, -3f);
            _camera.Forward = System.Numerics.Vector3.UnitZ;
        }

        OnCameraChanged?.Invoke();
    }

    /// <summary>
    /// Update the camera viewport size.
    /// </summary>
    public void ResizeViewport(int width, int height)
    {
        _camera.Width = width;
        _camera.Height = height;
        _camera.CenterX = width / 2.0f;
        _camera.CenterY = height / 2.0f;
        OnCameraChanged?.Invoke();
    }
}
