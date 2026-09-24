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

    private void BuildProjectBrowserUI()
    {
        _uiRoot.ClearChildren();
        _statusLabel = null;

        float margin = 32;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;

        var shell = _uiRoot.AddChild(new UIPanel
        {
            X = margin, Y = margin,
            Width = panelW, Height = panelH,
        });

        // Header
        shell.AddChild(new UILabel
        {
            X = 28, Y = 22,
            Text = "SpawnScene",
            FontSize = FontSize.Title,
            Color = UITheme.Current.TextPrimary,
        });
        shell.AddChild(new UILabel
        {
            X = 28, Y = 60,
            Text = "Studio - Gaussian splat projects",
            FontSize = FontSize.Body,
            Color = UITheme.Current.TextSecondary,
        });

        shell.AddChild(new UIButton
        {
            X = panelW - 110, Y = 20,
            Width = 90, Height = 32,
            Text = "Home",
            FontSize = FontSize.Caption,
            OnClick = () => _nav.NavigateTo(""),
        });

        shell.AddChild(new UIButton
        {
            X = panelW - 220, Y = 20,
            Width = 100, Height = 32,
            Text = "Testing",
            FontSize = FontSize.Caption,
            NormalColor = AccentMuted,
            OnClick = () =>
            {
                _state = StudioState.Testing;
                BuildTestingUI();
            },
        });

        shell.AddChild(new UIButton
        {
            X = 28, Y = 100,
            Width = 180, Height = 38,
            Text = "+ New Project",
            OnClick = OnNewProjectClicked,
        });

        float listTop = 156;
        float cardW = Math.Min(panelW - 56, 560);
        float cardH = 110;
        float thumbW = 160;
        float thumbH = cardH - 10;

        var list = shell.AddChild(new UIScrollView
        {
            X = 0, Y = listTop,
            Width = panelW, Height = panelH - listTop,
            Padding = 0,
            BackgroundColor = Color.Transparent,
            BorderWidth = 0,
        });

        if (_projects == null || _projects.Count == 0)
        {
            list.AddChild(new UILabel
            {
                X = 28, Y = 8,
                Text = "No projects yet. Create one to get started, or open Testing for sample datasets.",
                FontSize = FontSize.Caption,
                Color = UITheme.Current.TextMuted,
            });
            return;
        }

        float cardY = 8;
        foreach (var project in _projects)
        {
            var card = list.AddChild(new UIPanel
            {
                X = 28, Y = cardY,
                Width = cardW, Height = cardH,
                BackgroundColor = Color.FromArgb(200, 24, 28, 36),
            });

            GPUTextureView? projThumbView = null;
            var latestScene = project.Scenes.LastOrDefault();
            if (latestScene != null)
            {
                string thumbKey = $"scene:{latestScene.Id}";
                if (_thumbnailCache.TryGetValue(thumbKey, out var cached))
                    projThumbView = cached.view;
                else
                    LoadSceneThumbnailAsync(project.Id, latestScene.Id);
            }

            card.AddChild(new UIImage
            {
                X = 5, Y = 5,
                Width = thumbW, Height = thumbH,
                TextureView = projThumbView,
                PlaceholderColor = Color.FromArgb(255, 28, 32, 40),
            });

            if (latestScene == null)
            {
                card.AddChild(new UILabel
                {
                    X = 5 + thumbW / 2 - 30, Y = 5 + thumbH / 2 - 8,
                    Text = "No scenes",
                    FontSize = FontSize.Caption,
                    Color = UITheme.Current.TextMuted,
                });
            }

            float textX = thumbW + 15;
            card.AddChild(new UILabel
            {
                X = textX, Y = 10,
                Text = project.Name,
                FontSize = FontSize.Heading,
                Color = UITheme.Current.TextPrimary,
            });

            long sizeBytes = _projectService.GetProjectSize(project);
            string sizeStr = sizeBytes < 1024 * 1024
                ? $"{sizeBytes / 1024.0:F0} KB"
                : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
            card.AddChild(new UILabel
            {
                X = textX, Y = 42,
                Text = $"{project.Sources.Count} source(s) · {project.Scenes.Count} scene(s) · {sizeStr}",
                FontSize = FontSize.Caption,
                Color = UITheme.Current.TextSecondary,
            });

            var p = project;
            card.AddChild(new UIButton
            {
                X = textX, Y = 68,
                Width = 90, Height = 30,
                Text = "Open",
                FontSize = FontSize.Caption,
                OnClick = () => OnOpenProject(p),
            });
            card.AddChild(new UIButton
            {
                X = textX + 100, Y = 68,
                Width = 90, Height = 30,
                Text = "Delete",
                FontSize = FontSize.Caption,
                NormalColor = AccentDanger,
                HoverColor = AccentDangerHover,
                PressedColor = Color.FromArgb(255, 120, 40, 40),
                OnClick = () => _ = OnDeleteProject(p),
            });

            cardY += cardH + 12;
        }
    }

    private void BuildViewerHudUI()
    {
        _uiRoot.ClearChildren();
        _statusLabel = null;

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

        var hud = _uiRoot.AddChild(new UIPanel
        {
            X = 12, Y = _canvasHeight - 88,
            Width = 280, Height = 76,
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
            Text = "Click scene to look · WASD move · ESC release",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextMuted,
        });

        _uiRoot.AddChild(new UIButton
        {
            X = 12, Y = 12,
            Width = 100, Height = 32,
            Text = "< Back",
            FontSize = FontSize.Caption,
            OnClick = async () =>
            {
                _showSettings = false;
                if (_activeProject != null)
                {
                    _state = StudioState.ProjectDetail;
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
            NormalColor = Color.FromArgb(255, 40, 100, 180),
            HoverColor = Color.FromArgb(255, 50, 120, 210),
            OnClick = () => _ = EnterXRAsync("immersive-vr"),
        });

        btnRight -= 70;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 12,
            Width = 64, Height = 32,
            Text = "AR",
            FontSize = FontSize.Caption,
            NormalColor = Color.FromArgb(255, 40, 150, 100),
            HoverColor = Color.FromArgb(255, 50, 180, 120),
            OnClick = () => _ = EnterXRAsync("immersive-ar"),
        });

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

    private void BuildProjectDetailUI()
    {
        _uiRoot.ClearChildren();
        if (_activeProject == null) return;

        float margin = 28;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;

        var shell = _uiRoot.AddChild(new UIPanel
        {
            X = margin, Y = margin,
            Width = panelW, Height = panelH,
        });

        // Fixed header
        shell.AddChild(new UIButton
        {
            X = 16, Y = 14,
            Width = 90, Height = 32,
            Text = "< Back",
            FontSize = FontSize.Caption,
            OnClick = async () =>
            {
                _projects = await _projectService.ListProjectsAsync();
                _state = StudioState.ProjectBrowser;
                _activeProject = null;
                BuildProjectBrowserUI();
            },
        });

        shell.AddChild(new UILabel
        {
            X = 120, Y = 18,
            Text = _activeProject.Name,
            FontSize = FontSize.Heading,
            Color = UITheme.Current.TextPrimary,
        });

        long sizeBytes = _projectService.GetProjectSize(_activeProject);
        string sizeStr = sizeBytes < 1024 * 1024
            ? $"{sizeBytes / 1024.0:F0} KB" : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
        shell.AddChild(new UILabel
        {
            X = 120, Y = 48,
            Text = $"{_activeProject.Sources.Count} source(s) · {_activeProject.Scenes.Count} scene(s) · {sizeStr}",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });

        // Scrollable body under header; leave room for status bar
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
        float contentW = panelW - 40;

        // ── Source Images ──
        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Source Images",
            FontSize = FontSize.Body,
            Color = UITheme.Current.TextPrimary,
        });
        scroll.AddChild(new UIButton
        {
            X = 180, Y = y - 4,
            Width = 140, Height = 30,
            Text = "+ Add Images",
            FontSize = FontSize.Caption,
            Enabled = !_pipelineBusy,
            OnClick = OnAddImagesClicked,
        });
        y += 34;

        if (_activeProject.Sources.Count == 0)
        {
            scroll.AddChild(new UILabel
            {
                X = 20, Y = y,
                Text = "Or load a single-image sample:",
                FontSize = FontSize.Caption,
                Color = UITheme.Current.TextMuted,
            });
            y += 22;

            var samples = new[] {
                ("Room", "samples/room.png"),
                ("Garden", "samples/garden.png"),
                ("Living Room HD", "samples/living_room_hd.png"),
                ("Garden HD", "samples/garden_hd.png"),
            };

            float btnX = 20;
            foreach (var (name, path) in samples)
            {
                float btnW = Math.Max(80, name.Length * 8 + 20);
                var samplePath = path;
                scroll.AddChild(new UIButton
                {
                    X = btnX, Y = y,
                    Width = btnW, Height = 26,
                    Text = name,
                    FontSize = FontSize.Caption,
                    NormalColor = AccentMuted,
                    Enabled = !_pipelineBusy,
                    OnClick = () => _ = LoadSampleImage(name, samplePath),
                });
                btnX += btnW + 6;
                if (btnX > contentW - 40) { btnX = 20; y += 32; }
            }
            y += 36;

            scroll.AddChild(new UIButton
            {
                X = 20, Y = y,
                Width = 180, Height = 28,
                Text = "TempleRing (GT cameras)",
                FontSize = FontSize.Caption,
                NormalColor = Color.FromArgb(255, 90, 50, 70),
                Enabled = !_pipelineBusy,
                OnClick = () => _ = GenerateFromTempleRingAsync(),
            });
            y += 40;
        }
        else
        {
            foreach (var src in _activeProject.Sources)
            {
                float cardH = 100;
                string srcKey = $"source:{src.FileName}";
                var thumbView = _thumbnailCache.TryGetValue(srcKey, out var cached) ? cached.view : null;
                scroll.AddChild(new UIImage
                {
                    X = 20, Y = y,
                    Width = 140, Height = cardH - 6,
                    TextureView = thumbView,
                });
                if (thumbView == null)
                    LoadThumbnailAsync(_activeProject.Id, src.FileName);

                scroll.AddChild(new UILabel
                {
                    X = 176, Y = y + 6,
                    Text = src.FileName,
                    FontSize = FontSize.Body,
                    Color = UITheme.Current.TextPrimary,
                });
                scroll.AddChild(new UILabel
                {
                    X = 176, Y = y + 30,
                    Text = $"{src.Width}x{src.Height} · {src.SizeBytes / 1024}KB",
                    FontSize = FontSize.Caption,
                    Color = UITheme.Current.TextSecondary,
                });

                var srcRef = src;
                scroll.AddChild(new UIButton
                {
                    X = 176, Y = y + 56,
                    Width = 80, Height = 26,
                    Text = "Remove",
                    FontSize = FontSize.Caption,
                    NormalColor = AccentDanger,
                    HoverColor = AccentDangerHover,
                    Enabled = !_pipelineBusy,
                    OnClick = () => _ = OnRemoveSource(srcRef),
                });

                y += cardH + 8;
            }
        }

        // ── Generate Scene ──
        y += 8;
        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Scene Generation",
            FontSize = FontSize.Body,
            Color = UITheme.Current.TextPrimary,
        });
        y += 28;

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Quality",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
        var presets = new[] { ("Fast", 4, 0f), ("Standard", 2, 0.3f), ("High", 1, 0.3f) };
        float presetX = 90;
        foreach (var (presetName, sub, edge) in presets)
        {
            bool active = _activeProject.Settings.QualityPreset == presetName;
            var pn = presetName; var ps = sub; var pe = edge;
            scroll.AddChild(new UIButton
            {
                X = presetX, Y = y - 3,
                Width = 90, Height = 26,
                Text = presetName,
                FontSize = FontSize.Caption,
                NormalColor = active ? AccentSelected : UITheme.Current.ButtonNormal,
                Enabled = !_pipelineBusy,
                OnClick = () =>
                {
                    _activeProject.Settings.QualityPreset = pn;
                    _activeProject.Settings.Subsample = ps;
                    _activeProject.Settings.EdgeSharpness = pe;
                    _ = _projectService.UpdateProjectAsync(_activeProject);
                    BuildProjectDetailUI();
                },
            });
            presetX += 96;
        }
        y += 34;

        scroll.AddChild(new UILabel
        {
            X = 20, Y = y,
            Text = "Depth Model",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
        float modelX = 120;
        foreach (var model in DepthEstimationService.AvailableModels)
        {
            bool active = _activeProject.Settings.DepthModel == model.Id;
            var modelId = model.Id;
            float btnW = Math.Max(90, model.Name.Length * 7 + 16);
            scroll.AddChild(new UIButton
            {
                X = modelX, Y = y - 3,
                Width = btnW, Height = 26,
                Text = model.Name,
                FontSize = FontSize.Caption,
                NormalColor = active ? AccentSelected : UITheme.Current.ButtonNormal,
                Enabled = !_pipelineBusy,
                OnClick = () =>
                {
                    _activeProject.Settings.DepthModel = modelId;
                    _ = _projectService.UpdateProjectAsync(_activeProject);
                    BuildProjectDetailUI();
                },
            });
            modelX += btnW + 6;
        }
        y += 36;

        bool canGenerate = _activeProject.Sources.Count > 0 && !_pipelineBusy;
        scroll.AddChild(new UIButton
        {
            X = 20, Y = y,
            Width = 200, Height = 40,
            Text = "Generate Scene",
            Enabled = canGenerate,
            OnClick = canGenerate ? OnGenerateSceneClicked : null,
        });
        y += 56;

        if (_activeProject.Scenes.Count > 0)
        {
            scroll.AddChild(new UILabel
            {
                X = 20, Y = y,
                Text = "Generated Scenes",
                FontSize = FontSize.Body,
                Color = UITheme.Current.TextPrimary,
            });
            y += 28;

            foreach (var scene in _activeProject.Scenes)
            {
                string sceneSizeStr = scene.SizeBytes < 1024 * 1024
                    ? $"{scene.SizeBytes / 1024.0:F0} KB" : $"{scene.SizeBytes / (1024.0 * 1024.0):F1} MB";

                float sceneCardH = 108;
                var sceneCard = scroll.AddChild(new UIPanel
                {
                    X = 20, Y = y,
                    Width = Math.Min(contentW, 540), Height = sceneCardH,
                    BackgroundColor = Color.FromArgb(180, 24, 28, 36),
                });

                string sceneThumbKey = $"scene:{scene.Id}";
                var sceneThumbView = _thumbnailCache.TryGetValue(sceneThumbKey, out var sceneCached) ? sceneCached.view : null;
                sceneCard.AddChild(new UIImage
                {
                    X = 5, Y = 5,
                    Width = 160, Height = sceneCardH - 10,
                    TextureView = sceneThumbView,
                });
                if (sceneThumbView == null)
                    LoadSceneThumbnailAsync(_activeProject.Id, scene.Id);

                sceneCard.AddChild(new UILabel
                {
                    X = 178, Y = 12,
                    Text = $"{scene.SplatCount:N0} splats · {scene.QualityPreset} · {sceneSizeStr}",
                    FontSize = FontSize.Body,
                    Color = UITheme.Current.TextPrimary,
                });
                sceneCard.AddChild(new UILabel
                {
                    X = 178, Y = 38,
                    Text = $"Created {scene.CreatedAt:g}",
                    FontSize = FontSize.Caption,
                    Color = UITheme.Current.TextMuted,
                });

                var sceneRef = scene;
                sceneCard.AddChild(new UIButton
                {
                    X = 178, Y = 64,
                    Width = 70, Height = 28,
                    Text = "View",
                    FontSize = FontSize.Caption,
                    Enabled = !_pipelineBusy,
                    OnClick = () => OnViewScene(sceneRef),
                });
                sceneCard.AddChild(new UIButton
                {
                    X = 258, Y = 64,
                    Width = 70, Height = 28,
                    Text = "Delete",
                    FontSize = FontSize.Caption,
                    NormalColor = AccentDanger,
                    HoverColor = AccentDangerHover,
                    Enabled = !_pipelineBusy,
                    OnClick = () => _ = OnDeleteScene(sceneRef),
                });

                y += sceneCardH + 10;
            }
        }

        // Status bar (fixed)
        _statusLabel = shell.AddChild(new UILabel
        {
            X = 20, Y = panelH - statusH + 8,
            Width = panelW - 40,
            Text = string.IsNullOrEmpty(_statusMessage) ? "Ready" : _statusMessage,
            FontSize = FontSize.Caption,
            Color = Color.FromArgb(255, 80, 210, 220),
        });
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
}
