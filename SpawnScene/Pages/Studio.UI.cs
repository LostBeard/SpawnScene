using System.Drawing;
using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// WebGPU UI building via SpawnDev.GameUI
public partial class Studio
{
    private static Color AccentSelected => Color.FromArgb(255, 36, 120, 140);
    private static Color AccentDanger => Color.FromArgb(255, 150, 50, 50);
    private static Color AccentDangerHover => Color.FromArgb(255, 180, 60, 60);
    private static Color AccentMuted => Color.FromArgb(255, 50, 70, 90);

    private void BuildViewerHudUI()
    {
        _uiRoot.ClearChildren();
        _statusLabel = null;
        _hudTrainLabel = null;

        if (_showDepthMap && _depthMapView != null)
        {
            float aspect = (float)_depthMapW / Math.Max(1, _depthMapH);
            float canvasAspect = (float)_canvasWidth / Math.Max(1, _canvasHeight);
            float dispW, dispH;
            if (aspect > canvasAspect) { dispW = _canvasWidth; dispH = _canvasWidth / aspect; }
            else { dispH = _canvasHeight; dispW = _canvasHeight * aspect; }
            float dispX = (_canvasWidth - dispW) * 0.5f;
            float dispY = (_canvasHeight - dispH) * 0.5f;
            _uiRoot.AddChild(new UIImage
            {
                X = dispX, Y = dispY, Width = dispW, Height = dispH,
                TextureView = _depthMapView,
            });
        }

        // A translucent bar under the top controls: they sat straight on the scene and vanished over bright areas.
        _uiRoot.AddChild(new UIPanel
        {
            X = 0, Y = 0, Width = _canvasWidth, Height = 56,
            BackgroundColor = Color.FromArgb(150, 10, 13, 18), BorderWidth = 0, CornerRadius = 0, Padding = 0,
        });

        // The stats panel is as wide as its help line, measured with the real font (it clipped "ESC release").
        // Phones and tablets (a coarse primary pointer) navigate by touch (TouchNavigator), not mouse-look + WASD.
        string hudHelp = TouchIsPrimaryInput()
            ? "Drag to look  ·  pinch to move  ·  two fingers to pan"
            : "Click the scene to look  ·  WASD to move  ·  Esc to release";
        float hudW = _gameUI.Renderer.MeasureText(hudHelp, FontSize.Caption);
        hudW = hudW > 0 ? hudW + 28 : 360;
        var hud = _uiRoot.AddChild(new UIPanel
        {
            X = 12, Y = _canvasHeight - 88,
            Width = hudW, Height = 76,
            BackgroundColor = Color.FromArgb(180, 12, 16, 22),
        });

        _hudSplatLabel = hud.AddChild(new UILabel
        {
            X = 12, Y = 10,
            Text = "",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextPrimary,
        });
        _hudFpsLabel = hud.AddChild(new UILabel
        {
            X = 12, Y = 30,
            Text = "",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextPrimary,
        });
        hud.AddChild(new UILabel
        {
            X = 12, Y = 50,
            Text = hudHelp,
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextMuted,
        });

        if (_trainingActive)
        {
            // Training runs in this view: the scene on screen improves as it goes (live repack).
            var trainPanel = _uiRoot.AddChild(new UIPanel
            {
                X = 12, Y = _canvasHeight - 88 - 52,
                Width = Math.Min(620, _canvasWidth - 24), Height = 44,
                BackgroundColor = Color.FromArgb(200, 12, 16, 22),
            });
            _hudTrainLabel = trainPanel.AddChild(new UILabel
            {
                X = 12, Y = 14,
                Text = _trainHudText,
                FontSize = FontSize.Caption,
                Color = UITheme.Current.TextPrimary,
            });
            trainPanel.AddChild(new UIButton
            {
                X = Math.Min(620, _canvasWidth - 24) - 132, Y = 6,
                Width = 124, Height = 32,
                Text = "Stop training",
                FontSize = FontSize.Caption,
                OnClick = () =>
                {
                    // Keeps what has been learned: training finishes its current iteration, then evaluates,
                    // hands the viewer its SH bands and the scene is saved.
                    _trainStopRequested = true;
                    _trainHudText = "Stopping - finishing up and saving…";
                    if (_hudTrainLabel != null) _hudTrainLabel.Text = _trainHudText;
                },
            });
        }

        // Back names where it goes; the project is the context the scene belongs to.
        string backTo = _activeProject?.Name ?? "Projects";
        if (backTo.Length > 28) backTo = backTo[..25] + "...";
        float backW = _gameUI.Renderer.MeasureText("< " + backTo, FontSize.Caption);
        _uiRoot.AddChild(new UIButton
        {
            X = 12, Y = 12,
            Width = backW > 0 ? Math.Max(100, backW + 32) : 220, Height = 32,
            Text = "< " + backTo,
            FontSize = FontSize.Caption,
            OnClick = async () =>
            {
                if (_trainingActive)
                {
                    // Leaving mid-training would pull the scene out from under the trainer. Stop first; the
                    // scene is saved when it finishes, then Back works normally.
                    _trainStopRequested = true;
                    _trainHudText = "Stopping - finishing up and saving… (press Back again when done)";
                    if (_hudTrainLabel != null) _hudTrainLabel.Text = _trainHudText;
                    return;
                }
                _showSettings = false;
                if (_activeProject != null)
                {
                    _state = StudioState.ProjectDetail;
                    _projectTab = "Scenes"; // back from a scene: show the scenes (a new one after Generate)
                    _projects = await _projectService.ListProjectsAsync();
                    _activeProject = _projects?.FirstOrDefault(p => p.Id == _activeProject.Id) ?? _activeProject;
                    BuildProjectDetailUI();
                }
                else
                {
                    _state = StudioState.ProjectBrowser;
                    _projects = await _projectService.ListProjectsAsync();
                    BuildProjectBrowserUI();
                }
                ReleasePointerLock();
            },
        });

        float btnRight = _canvasWidth - 12;

        btnRight -= 110;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 12,
            Width = 100, Height = 32,
            Text = "Settings",
            FontSize = FontSize.Caption,
            OnClick = () =>
            {
                _showSettings = !_showSettings;
                BuildSettingsPanel();
            },
        });

        if (_depthMapView != null)
        {
            btnRight -= 80;
            _uiRoot.AddChild(new UIButton
            {
                X = btnRight, Y = 12,
                Width = 72, Height = 32,
                Text = _showDepthMap ? "Scene" : "Depth",
                FontSize = FontSize.Caption,
                NormalColor = _showDepthMap ? AccentSelected : UITheme.Current.ButtonNormal,
                OnClick = () => { _showDepthMap = !_showDepthMap; BuildViewerHudUI(); },
            });
        }

        btnRight -= 76;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 12,
            Width = 68, Height = 32,
            Text = "VR",
            FontSize = FontSize.Caption,
            OnClick = () => _ = EnterXRAsync("immersive-vr"),
        });

        btnRight -= 70;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 12,
            Width = 64, Height = 32,
            Text = "AR",
            FontSize = FontSize.Caption,
            OnClick = () => _ = EnterXRAsync("immersive-ar"),
        });

        // Send this page to a Meta Quest without typing the URL in the headset: Meta's open_url page pushes the link to
        // the signed-in account's headset (https://developers.meta.com/vr/documentation/web/web-launch/). Hidden inside
        // the Quest Browser (already there) and on a loopback origin, which the headset cannot reach.
        if (CanSendToHeadset())
        {
            const string sendText = "Send to Headset";
            float sendW = _gameUI.Renderer.MeasureText(sendText, FontSize.Caption);
            sendW = sendW > 0 ? sendW + 28 : 132;
            btnRight -= sendW + 8;
            _uiRoot.AddChild(new UIButton
            {
                X = btnRight, Y = 12,
                Width = sendW, Height = 32,
                Text = sendText,
                FontSize = FontSize.Caption,
                OnClick = SendToHeadset,
            });
        }

        if (_showSettings)
            BuildSettingsPanel();
    }

    private void BuildSettingsPanel()
    {
        if (_settingsPanel != null)
        {
            _uiRoot.RemoveChild(_settingsPanel);
            _settingsPanel = null;
        }
        if (!_showSettings) return;

        _settingsPanel = _uiRoot.AddChild(new UIPanel
        {
            X = _canvasWidth - 292, Y = 52,
            Width = 280, Height = 270,
        });

        _settingsPanel.AddChild(new UILabel
        {
            X = 14, Y = 12,
            Text = "Render Settings",
            FontSize = FontSize.Body,
            Color = UITheme.Current.TextPrimary,
        });

        _settingsPanel.AddChild(new UISlider
        {
            X = 14, Y = 44,
            Width = 250, Height = 40,
            Label = "Sharpening",
            MinValue = 0f, MaxValue = 1f,
            Value = _renderService.SharpeningStrength,
            OnChanged = v => _renderService.SharpeningStrength = v,
        });

        _settingsPanel.AddChild(new UILabel
        {
            X = 14, Y = 92,
            Text = "Render Mode",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });

        bool isStochastic = _gpuRenderer.RenderMode == SplatRenderMode.Stochastic;
        _settingsPanel.AddChild(new UIButton
        {
            X = 14, Y = 112,
            Width = 120, Height = 30,
            Text = "Stochastic",
            FontSize = FontSize.Caption,
            NormalColor = isStochastic ? AccentSelected : UITheme.Current.ButtonNormal,
            OnClick = () =>
            {
                _gpuRenderer.RenderMode = SplatRenderMode.Stochastic;
                BuildSettingsPanel();
            },
        });
        _settingsPanel.AddChild(new UIButton
        {
            X = 142, Y = 112,
            Width = 120, Height = 30,
            Text = "Sorted",
            FontSize = FontSize.Caption,
            NormalColor = !isStochastic ? AccentSelected : UITheme.Current.ButtonNormal,
            OnClick = () =>
            {
                _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
                BuildSettingsPanel();
            },
        });

        _settingsPanel.AddChild(new UILabel
        {
            X = 14, Y = 152,
            Text = "Resolution",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });

        var resMode = _gpuRenderer.AdaptiveResMode;
        AddToggleChip(_settingsPanel, 14, 172, 80, "Auto", resMode == AdaptiveResMode.Auto,
            () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.Auto; BuildSettingsPanel(); });
        AddToggleChip(_settingsPanel, 100, 172, 80, "Full", resMode == AdaptiveResMode.ForceFull,
            () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull; BuildSettingsPanel(); });
        AddToggleChip(_settingsPanel, 186, 172, 80, "Half", resMode == AdaptiveResMode.ForceHalf,
            () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceHalf; BuildSettingsPanel(); });

        _settingsPanel.AddChild(new UILabel
        {
            X = 14, Y = 210,
            Text = "XR Sharpening",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
        AddToggleChip(_settingsPanel, 14, 230, 80, _xrCasEnabled ? "On" : "Off", _xrCasEnabled,
            () => { _xrCasEnabled = !_xrCasEnabled; BuildSettingsPanel(); });
    }

    private static void AddToggleChip(UIPanel parent, float x, float y, float w, string text, bool on, Action click)
    {
        parent.AddChild(new UIButton
        {
            X = x, Y = y,
            Width = w, Height = 26,
            Text = text,
            FontSize = FontSize.Caption,
            NormalColor = on ? AccentSelected : UITheme.Current.ButtonNormal,
            OnClick = click,
        });
    }

    private void UpdateViewerHud()
    {
        if (_hudSplatLabel != null)
            _hudSplatLabel.Text = $"{_sceneManager.ActiveScene?.Count.ToString("N0") ?? "0"} splats";
        if (_hudFpsLabel != null)
            _hudFpsLabel.Text = $"{_renderService.Fps:F0} FPS";
    }

    private void BuildTestingUI()
    {
        _uiRoot.ClearChildren();

        float margin = 28;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;

        var shell = _uiRoot.AddChild(new UIPanel
        {
            X = margin, Y = margin,
            Width = panelW, Height = panelH,
        });

        shell.AddChild(new UIButton
        {
            X = 16, Y = 14,
            Width = 90, Height = 32,
            Text = "< Back",
            FontSize = FontSize.Caption,
            OnClick = () =>
            {
                _state = StudioState.ProjectBrowser;
                BuildProjectBrowserUI();
            },
        });

        shell.AddChild(new UILabel
        {
            X = 120, Y = 18,
            Text = "Testing",
            FontSize = FontSize.Heading,
            Color = UITheme.Current.TextPrimary,
        });
        shell.AddChild(new UILabel
        {
            X = 120, Y = 48,
            Text = "Sample datasets — pose, generate, and train without query strings.",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });

        float bodyTop = 78;
        float statusH = 36;
        var scroll = shell.AddChild(new UIScrollView
        {
            X = 0, Y = bodyTop,
            Width = panelW, Height = panelH - bodyTop - statusH,
            Padding = 0,
            BackgroundColor = Color.Transparent,
            BorderWidth = 0,
        });

        float y = 8;

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Dataset",
            FontSize = FontSize.Body,
            Color = UITheme.Current.TextPrimary,
        });
        y += 28;

        foreach (var name in new[] { "Bathroom", "DrJohnson", "TempleRing" })
        {
            bool sel = string.Equals(_uiDataset, name, StringComparison.OrdinalIgnoreCase);
            var n = name;
            scroll.AddChild(new UIButton
            {
                X = 20 + (name switch { "Bathroom" => 0, "DrJohnson" => 118, _ => 236 }),
                Y = y,
                Width = 110, Height = 32,
                Text = name,
                FontSize = FontSize.Caption,
                NormalColor = sel ? AccentSelected : AccentMuted,
                Enabled = !_pipelineBusy,
                OnClick = () =>
                {
                    _uiDataset = n;
                    // Bathroom ships no poses.par - GT toggle would silently no-op then recover
                    // poses and look like "COLMAP failed" when it never ran.
                    if (string.Equals(n, "Bathroom", StringComparison.OrdinalIgnoreCase))
                    {
                        _uiUseGtPoses = false;
                        _uiInitFromCloud = false;
                    }
                    BuildTestingUI();
                },
            });
        }
        y += 44;

        bool datasetHasGt = !string.Equals(_uiDataset, "Bathroom", StringComparison.OrdinalIgnoreCase);
        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Poses",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
        y += 22;
        scroll.AddChild(new UIButton
        {
            X = 20, Y = y,
            Width = 110, Height = 28,
            Text = "DAv3",
            FontSize = FontSize.Caption,
            NormalColor = !_uiUseGtPoses ? AccentSelected : UITheme.Current.ButtonNormal,
            Enabled = !_pipelineBusy,
            OnClick = () => { _uiUseGtPoses = false; _uiInitFromCloud = false; BuildTestingUI(); },
        });
        scroll.AddChild(new UIButton
        {
            X = 138, Y = y,
            Width = 140, Height = 28,
            Text = "COLMAP GT",
            FontSize = FontSize.Caption,
            NormalColor = _uiUseGtPoses ? AccentSelected : UITheme.Current.ButtonNormal,
            Enabled = !_pipelineBusy && datasetHasGt,
            OnClick = () => { _uiUseGtPoses = true; BuildTestingUI(); },
        });
        y += 36;

        if (_uiUseGtPoses)
        {
            scroll.AddChild(new UIButton
            {
                X = 20, Y = y,
                Width = 160, Height = 28,
                Text = _uiInitFromCloud ? "Init: SfM points" : "Init: depth",
                FontSize = FontSize.Caption,
                NormalColor = AccentMuted,
                Enabled = !_pipelineBusy,
                OnClick = () => { _uiInitFromCloud = !_uiInitFromCloud; BuildTestingUI(); },
            });
            y += 36;
        }

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Width = panelW - 60,
            Text = datasetHasGt
                ? (_uiUseGtPoses
                    ? "COLMAP cameras (oracle). If this looks right and DAv3 looks wrong, poses are the cliff."
                    : "DAv3 joint poses. MEASURED weak on room walk-throughs vs COLMAP (~60deg+ forward error).")
                : "Bathroom has no ground-truth poses. Alignment is entirely DAv3 - same room-pose cliff as DrJohnson.",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextMuted,
        });
        y += 40;

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = $"Train iters: {_uiTrainIters}",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
        y += 24;
        foreach (var (label, iters, ox) in new[] { ("0", 0, 0), ("200", 200, 70), ("500", 500, 140), ("1600", 1600, 210) })
        {
            bool sel = _uiTrainIters == iters;
            var iv = iters;
            scroll.AddChild(new UIButton
            {
                X = 20 + ox, Y = y,
                Width = 64, Height = 28,
                Text = label,
                FontSize = FontSize.Caption,
                NormalColor = sel ? AccentSelected : UITheme.Current.ButtonNormal,
                Enabled = !_pipelineBusy,
                OnClick = () => { _uiTrainIters = iv; BuildTestingUI(); },
            });
        }
        y += 40;

        scroll.AddChild(new UIButton
        {
            X = 20, Y = y,
            Width = 140, Height = 32,
            Text = _uiTrainGeom ? "Geometry: On" : "Geometry: Off",
            FontSize = FontSize.Caption,
            NormalColor = _uiTrainGeom ? AccentSelected : UITheme.Current.ButtonNormal,
            Enabled = !_pipelineBusy,
            OnClick = () => { _uiTrainGeom = !_uiTrainGeom; BuildTestingUI(); },
        });

        scroll.AddChild(new UIButton
        {
            X = 172, Y = y,
            Width = 220, Height = 40,
            Text = _pipelineBusy ? "Running…" : $"Run {_uiDataset}",
            Enabled = !_pipelineBusy,
            OnClick = () => _ = OnRunDatasetFromUiAsync(),
        });
        y += 56;

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Runs the same pipeline as ?autotest=dataset. Results open in the viewer.",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextMuted,
        });

        _statusLabel = shell.AddChild(new UILabel
        {
            X = 20, Y = panelH - statusH + 8,
            Width = panelW - 40,
            Text = string.IsNullOrEmpty(_statusMessage) ? "Ready" : _statusMessage,
            FontSize = FontSize.Caption,
            Color = Color.FromArgb(255, 80, 210, 220),
        });
    }

    /// <summary>
    /// One project-setting row: a caption and a button per choice, the current value highlighted. Saves the project
    /// on click. Returns the next row's y.
    /// </summary>
    private void SetUiStatus(string message)
    {
        _statusMessage = message;
        if (_statusLabel != null)
            _statusLabel.Text = message;
        else if (_state == StudioState.Testing)
            BuildTestingUI();
        else if (_state == StudioState.ProjectDetail)
            BuildProjectDetailUI();
    }

    bool? _touchPrimary;
    /// <summary>True when the device's primary pointer is coarse (a finger): phones and tablets, not touch laptops.</summary>
    bool TouchIsPrimaryInput()
    {
        if (_touchPrimary is { } known) return known;
        try
        {
            using var window = _js.Get<Window>("window");
            using var query = window.MatchMedia("(pointer: coarse)");
            _touchPrimary = query.Matches;
        }
        catch { _touchPrimary = false; }
        return _touchPrimary.Value;
    }

    bool CanSendToHeadset()
    {
        var uri = new Uri(_nav.Uri);
        if (uri.IsLoopback) return false;
        try
        {
            using var navigator = _js.Get<Navigator>("navigator");
            return !navigator.UserAgent.Contains("OculusBrowser", StringComparison.OrdinalIgnoreCase);
        }
        catch { return true; }
    }

    void SendToHeadset()
    {
        var url = $"https://www.oculus.com/open_url/?url={Uri.EscapeDataString(_nav.Uri)}";
        Console.WriteLine($"[Studio] Send to headset: {_nav.Uri}");
        using var window = _js.Get<Window>("window");
        using var opened = window.Open(url, "_blank");
        opened?.Focus();
    }
}
