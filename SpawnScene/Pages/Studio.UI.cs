using SpawnDev.BlazorJS.JSObjects;
using SpawnScene.Models;
using SpawnScene.Services;
using SpawnScene.UI;
using SpawnScene.UI.Elements;

namespace SpawnScene.Pages;

// All WebGPU UI building methods
public partial class Studio
{
    private void BuildProjectBrowserUI()
    {
        _uiRoot.ClearChildren();

        float margin = 40;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;

        var mainPanel = _uiRoot.AddChild(new UIPanel
        {
            X = margin, Y = margin,
            Width = panelW, Height = panelH,
            BackgroundColor = System.Drawing.Color.FromArgb(220, 15, 15, 25),
        });

        // Home button (top-right)
        mainPanel.AddChild(new UIButton
        {
            X = panelW - 110, Y = 15,
            Width = 95, Height = 30,
            Text = "Home",
            FontSize = FontSize.Caption,
            NormalColor = System.Drawing.Color.FromArgb(255, 50, 50, 65),
            HoverColor = System.Drawing.Color.FromArgb(255, 70, 70, 85),
            PressedColor = System.Drawing.Color.FromArgb(255, 40, 40, 55),
            OnClick = () => _nav.NavigateTo(""),
        });

        // Header row
        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = 20,
            Text = "SpawnScene Studio",
            FontSize = FontSize.Title,
            Color = System.Drawing.Color.White,
        });

        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = 60,
            Text = "Gaussian Splat Projects",
            FontSize = FontSize.Body,
            Color = System.Drawing.Color.FromArgb(255, 180, 180, 200),
        });

        // New Project button
        mainPanel.AddChild(new UIButton
        {
            X = 30, Y = 100,
            Width = 200, Height = 40,
            Text = "+ New Project",
            OnClick = OnNewProjectClicked,
        });

        // Project list
        float cardY = 160;
        float cardW = Math.Min(panelW - 60, 550);
        float cardH = 110;
        float thumbW = 160;
        float thumbH = cardH - 10;

        if (_projects == null || _projects.Count == 0)
        {
            mainPanel.AddChild(new UILabel
            {
                X = 30, Y = cardY,
                Text = "No projects yet. Create one to get started.",
                FontSize = FontSize.Caption,
                Color = System.Drawing.Color.Gray,
            });
        }
        else
        {
            foreach (var project in _projects)
            {
                var card = mainPanel.AddChild(new UIPanel
                {
                    X = 30, Y = cardY,
                    Width = cardW, Height = cardH,
                    BackgroundColor = System.Drawing.Color.FromArgb(180, 30, 30, 45),
                    BorderWidth = 1,
                    BorderColor = System.Drawing.Color.FromArgb(40, 255, 255, 255),
                });

                // Project thumbnail: use latest scene thumbnail, or placeholder
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
                    PlaceholderColor = System.Drawing.Color.FromArgb(255, 30, 30, 45),
                });

                // Placeholder label when no scene exists
                if (latestScene == null)
                {
                    card.AddChild(new UILabel
                    {
                        X = 5 + thumbW / 2 - 30, Y = 5 + thumbH / 2 - 8,
                        Text = "No scenes",
                        FontSize = FontSize.Caption,
                        Color = System.Drawing.Color.FromArgb(255, 80, 80, 100),
                    });
                }

                float textX = thumbW + 15;

                card.AddChild(new UILabel
                {
                    X = textX, Y = 10,
                    Text = project.Name,
                    FontSize = FontSize.Heading,
                    Color = System.Drawing.Color.White,
                });

                long sizeBytes = _projectService.GetProjectSize(project);
                string sizeStr = sizeBytes < 1024 * 1024
                    ? $"{sizeBytes / 1024.0:F0} KB"
                    : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
                string info = $"{project.Sources.Count} source(s) · {project.Scenes.Count} scene(s) · {sizeStr}";

                card.AddChild(new UILabel
                {
                    X = textX, Y = 40,
                    Text = info,
                    FontSize = FontSize.Caption,
                    Color = System.Drawing.Color.FromArgb(255, 150, 150, 170),
                });

                // Open button
                var p = project;
                card.AddChild(new UIButton
                {
                    X = textX, Y = 65,
                    Width = 90, Height = 32,
                    Text = "Open",
                    FontSize = FontSize.Caption,
                    OnClick = () => OnOpenProject(p),
                });

                // Delete button
                card.AddChild(new UIButton
                {
                    X = textX + 100, Y = 65,
                    Width = 90, Height = 32,
                    Text = "Delete",
                    FontSize = FontSize.Caption,
                    NormalColor = System.Drawing.Color.FromArgb(255, 150, 50, 50),
                    HoverColor = System.Drawing.Color.FromArgb(255, 180, 60, 60),
                    PressedColor = System.Drawing.Color.FromArgb(255, 120, 40, 40),
                    OnClick = () => _ = OnDeleteProject(p),
                });

                cardY += cardH + 10;
            }
        }
    }

    private void BuildViewerHudUI()
    {
        _uiRoot.ClearChildren();

        // HUD panel (bottom-left)
        var hud = _uiRoot.AddChild(new UIPanel
        {
            X = 10, Y = _canvasHeight - 90,
            Width = 260, Height = 80,
            BackgroundColor = System.Drawing.Color.FromArgb(160, 10, 10, 20),
        });

        _hudSplatLabel = hud.AddChild(new UILabel
        {
            X = 10, Y = 8,
            Text = "",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.White,
        });

        _hudFpsLabel = hud.AddChild(new UILabel
        {
            X = 10, Y = 28,
            Text = "",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.White,
        });

        hud.AddChild(new UILabel
        {
            X = 10, Y = 48,
            Text = "Click to look · WASD move · ESC release",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.Gray,
        });

        // Back button (top-left)
        _uiRoot.AddChild(new UIButton
        {
            X = 10, Y = 10,
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

        // Top-right button row
        float btnRight = _canvasWidth - 10;

        // Settings
        btnRight -= 110;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 10,
            Width = 110, Height = 32,
            Text = "Settings",
            FontSize = FontSize.Caption,
            NormalColor = System.Drawing.Color.FromArgb(200, 50, 50, 65),
            HoverColor = System.Drawing.Color.FromArgb(220, 70, 70, 85),
            PressedColor = System.Drawing.Color.FromArgb(200, 40, 40, 55),
            OnClick = () =>
            {
                _showSettings = !_showSettings;
                BuildSettingsPanel();
            },
        });

        // Enter VR button
        btnRight -= 80;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 10,
            Width = 75, Height = 32,
            Text = "VR",
            FontSize = FontSize.Caption,
            NormalColor = System.Drawing.Color.FromArgb(255, 40, 100, 180),
            HoverColor = System.Drawing.Color.FromArgb(255, 50, 120, 210),
            PressedColor = System.Drawing.Color.FromArgb(255, 30, 80, 150),
            OnClick = () => _ = EnterXRAsync("immersive-vr"),
        });

        // Enter AR button
        btnRight -= 75;
        _uiRoot.AddChild(new UIButton
        {
            X = btnRight, Y = 10,
            Width = 70, Height = 32,
            Text = "AR",
            FontSize = FontSize.Caption,
            NormalColor = System.Drawing.Color.FromArgb(255, 40, 150, 100),
            HoverColor = System.Drawing.Color.FromArgb(255, 50, 180, 120),
            PressedColor = System.Drawing.Color.FromArgb(255, 30, 120, 80),
            OnClick = () => _ = EnterXRAsync("immersive-ar"),
        });

        // Build settings panel if visible
        if (_showSettings)
            BuildSettingsPanel();
    }

    private void BuildSettingsPanel()
    {
        // Remove old settings panel if it exists
        if (_settingsPanel != null)
        {
            _uiRoot.RemoveChild(_settingsPanel);
            _settingsPanel = null;
        }

        if (!_showSettings) return;

        _settingsPanel = _uiRoot.AddChild(new UIPanel
        {
            X = _canvasWidth - 280, Y = 50,
            Width = 270, Height = 260,
            BackgroundColor = System.Drawing.Color.FromArgb(230, 20, 20, 30),
            BorderWidth = 1,
            BorderColor = System.Drawing.Color.FromArgb(40, 255, 255, 255),
        });

        _settingsPanel.AddChild(new UILabel
        {
            X = 12, Y = 8,
            Text = "Render Settings",
            FontSize = FontSize.Body,
            Color = System.Drawing.Color.White,
        });

        // Sharpening slider
        _settingsPanel.AddChild(new UISlider
        {
            X = 12, Y = 38,
            Width = 245, Height = 40,
            Label = "Sharpening",
            MinValue = 0f, MaxValue = 1f,
            Value = _renderService.SharpeningStrength,
            OnChanged = v => _renderService.SharpeningStrength = v,
        });

        // Render mode toggle
        _settingsPanel.AddChild(new UILabel
        {
            X = 12, Y = 88,
            Text = "Render Mode",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 200, 200, 220),
        });

        bool isStochastic = _gpuRenderer.RenderMode == SplatRenderMode.Stochastic;
        _settingsPanel.AddChild(new UIButton
        {
            X = 12, Y = 108,
            Width = 120, Height = 30,
            Text = "Stochastic",
            FontSize = FontSize.Caption,
            NormalColor = isStochastic
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () =>
            {
                _gpuRenderer.RenderMode = SplatRenderMode.Stochastic;
                BuildSettingsPanel();
            },
        });

        _settingsPanel.AddChild(new UIButton
        {
            X = 140, Y = 108,
            Width = 120, Height = 30,
            Text = "Sorted",
            FontSize = FontSize.Caption,
            NormalColor = !isStochastic
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () =>
            {
                _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
                BuildSettingsPanel();
            },
        });

        // Resolution mode
        _settingsPanel.AddChild(new UILabel
        {
            X = 12, Y = 148,
            Text = "Resolution",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 200, 200, 220),
        });

        var resMode = _gpuRenderer.AdaptiveResMode;
        _settingsPanel.AddChild(new UIButton
        {
            X = 12, Y = 168,
            Width = 80, Height = 26,
            Text = "Auto",
            FontSize = FontSize.Caption,
            NormalColor = resMode == AdaptiveResMode.Auto
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.Auto; BuildSettingsPanel(); },
        });
        _settingsPanel.AddChild(new UIButton
        {
            X = 98, Y = 168,
            Width = 80, Height = 26,
            Text = "Full",
            FontSize = FontSize.Caption,
            NormalColor = resMode == AdaptiveResMode.ForceFull
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull; BuildSettingsPanel(); },
        });
        _settingsPanel.AddChild(new UIButton
        {
            X = 184, Y = 168,
            Width = 80, Height = 26,
            Text = "Half",
            FontSize = FontSize.Caption,
            NormalColor = resMode == AdaptiveResMode.ForceHalf
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () => { _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceHalf; BuildSettingsPanel(); },
        });

        // XR sharpening toggle
        _settingsPanel.AddChild(new UILabel
        {
            X = 12, Y = 204,
            Text = "XR Sharpening",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 200, 200, 220),
        });
        _settingsPanel.AddChild(new UIButton
        {
            X = 12, Y = 224,
            Width = 80, Height = 26,
            Text = _xrCasEnabled ? "On" : "Off",
            FontSize = FontSize.Caption,
            NormalColor = _xrCasEnabled
                ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                : System.Drawing.Color.FromArgb(255, 50, 50, 65),
            OnClick = () => { _xrCasEnabled = !_xrCasEnabled; BuildSettingsPanel(); },
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

        float margin = 40;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;

        var mainPanel = _uiRoot.AddChild(new UIPanel
        {
            X = margin, Y = margin,
            Width = panelW, Height = panelH,
            BackgroundColor = System.Drawing.Color.FromArgb(220, 15, 15, 25),
        });

        // Back button
        mainPanel.AddChild(new UIButton
        {
            X = 15, Y = 15,
            Width = 90, Height = 32,
            Text = "< Back",
            FontSize = FontSize.Caption,
            NormalColor = System.Drawing.Color.FromArgb(255, 60, 60, 75),
            HoverColor = System.Drawing.Color.FromArgb(255, 80, 80, 95),
            PressedColor = System.Drawing.Color.FromArgb(255, 45, 45, 60),
            OnClick = async () =>
            {
                _projects = await _projectService.ListProjectsAsync();
                _state = StudioState.ProjectBrowser;
                _activeProject = null;
                BuildProjectBrowserUI();
            },
        });

        // Project name
        mainPanel.AddChild(new UILabel
        {
            X = 120, Y = 18,
            Text = _activeProject.Name,
            FontSize = FontSize.Heading,
            Color = System.Drawing.Color.White,
        });

        // Project info
        long sizeBytes = _projectService.GetProjectSize(_activeProject);
        string sizeStr = sizeBytes < 1024 * 1024
            ? $"{sizeBytes / 1024.0:F0} KB" : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = 55,
            Text = $"{_activeProject.Sources.Count} source image(s) · {_activeProject.Scenes.Count} scene(s) · {sizeStr}",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 150, 150, 170),
        });

        // ── Source Images Section ──
        float sectionY = 90;

        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = sectionY,
            Text = "Source Images",
            FontSize = FontSize.Body,
            Color = System.Drawing.Color.White,
        });

        mainPanel.AddChild(new UIButton
        {
            X = 200, Y = sectionY - 4,
            Width = 160, Height = 30,
            Text = "+ Add Images",
            FontSize = FontSize.Caption,
            OnClick = OnAddImagesClicked,
        });

        sectionY += 35;

        // Sample image buttons (only show if no sources yet)
        if (_activeProject.Sources.Count == 0)
        {
            mainPanel.AddChild(new UILabel
            {
                X = 30, Y = sectionY,
                Text = "Or load a sample:",
                FontSize = FontSize.Caption,
                Color = System.Drawing.Color.Gray,
            });
            sectionY += 22;

            var samples = new[] {
                ("Room", "samples/room.png"),
                ("Garden", "samples/garden.png"),
                ("Living Room HD", "samples/living_room_hd.png"),
                ("Garden HD", "samples/garden_hd.png"),
                ("Living Room 5K", "samples/living-room-hd-2.jpg"),
            };

            float btnX = 30;
            foreach (var (name, path) in samples)
            {
                float btnW = Math.Max(80, name.Length * 8 + 20);
                var samplePath = path;
                mainPanel.AddChild(new UIButton
                {
                    X = btnX, Y = sectionY,
                    Width = btnW, Height = 26,
                    Text = name,
                    FontSize = FontSize.Caption,
                    NormalColor = System.Drawing.Color.FromArgb(255, 50, 70, 90),
                    HoverColor = System.Drawing.Color.FromArgb(255, 60, 85, 110),
                    PressedColor = System.Drawing.Color.FromArgb(255, 40, 55, 75),
                    OnClick = () => _ = LoadSampleImage(name, samplePath),
                });
                btnX += btnW + 6;
                if (btnX > panelW - 100) { btnX = 30; sectionY += 32; }
            }
            sectionY += 35;

            // Multi-view test datasets
            mainPanel.AddChild(new UILabel
            {
                X = 30, Y = sectionY,
                Text = "Multi-view test (ground truth cameras):",
                FontSize = FontSize.Caption,
                Color = System.Drawing.Color.Gray,
            });
            sectionY += 22;

            mainPanel.AddChild(new UIButton
            {
                X = 30, Y = sectionY,
                Width = 140, Height = 26,
                Text = "TempleRing (4 views)",
                FontSize = FontSize.Caption,
                NormalColor = System.Drawing.Color.FromArgb(255, 90, 50, 70),
                HoverColor = System.Drawing.Color.FromArgb(255, 110, 60, 85),
                PressedColor = System.Drawing.Color.FromArgb(255, 70, 40, 55),
                OnClick = () => _ = GenerateFromTempleRingAsync(),
            });
            sectionY += 35;
        }
        else
        {
            foreach (var src in _activeProject.Sources)
            {
                float cardH = 120;

                // Thumbnail
                string srcKey = $"source:{src.FileName}";
                var thumbView = _thumbnailCache.TryGetValue(srcKey, out var cached) ? cached.view : null;
                mainPanel.AddChild(new UIImage
                {
                    X = 30, Y = sectionY,
                    Width = 160, Height = cardH - 6,
                    TextureView = thumbView,
                });

                if (thumbView == null)
                    LoadThumbnailAsync(_activeProject.Id, src.FileName);

                // Info text
                mainPanel.AddChild(new UILabel
                {
                    X = 200, Y = sectionY + 8,
                    Text = src.FileName,
                    FontSize = FontSize.Body,
                    Color = System.Drawing.Color.White,
                });
                string sizeInfo = $"{src.Width}x{src.Height} · {src.SizeBytes / 1024}KB";
                mainPanel.AddChild(new UILabel
                {
                    X = 200, Y = sectionY + 30,
                    Text = sizeInfo,
                    FontSize = FontSize.Caption,
                    Color = System.Drawing.Color.FromArgb(255, 150, 150, 170),
                });

                // Remove button
                var srcRef = src;
                mainPanel.AddChild(new UIButton
                {
                    X = 200, Y = sectionY + 52,
                    Width = 80, Height = 26,
                    Text = "Remove",
                    FontSize = FontSize.Caption,
                    NormalColor = System.Drawing.Color.FromArgb(255, 120, 50, 50),
                    HoverColor = System.Drawing.Color.FromArgb(255, 160, 60, 60),
                    PressedColor = System.Drawing.Color.FromArgb(255, 90, 40, 40),
                    OnClick = () => _ = OnRemoveSource(srcRef),
                });

                sectionY += cardH + 4;
            }
        }

        // ── Generate Scene Section ──
        sectionY += 15;
        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = sectionY,
            Text = "Scene Generation",
            FontSize = FontSize.Body,
            Color = System.Drawing.Color.White,
        });
        sectionY += 28;

        // Quality presets
        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = sectionY,
            Text = "Quality:",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 180, 180, 200),
        });

        var presets = new[] { ("Fast", 4, 0f), ("Standard", 2, 0.3f), ("High", 1, 0.3f) };
        float presetX = 100;
        foreach (var (presetName, sub, edge) in presets)
        {
            bool active = _activeProject.Settings.QualityPreset == presetName;
            var pn = presetName; var ps = sub; var pe = edge;
            mainPanel.AddChild(new UIButton
            {
                X = presetX, Y = sectionY - 3,
                Width = 90, Height = 26,
                Text = presetName,
                FontSize = FontSize.Caption,
                NormalColor = active
                    ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                    : System.Drawing.Color.FromArgb(255, 50, 50, 65),
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
        sectionY += 35;

        // Depth model selector
        mainPanel.AddChild(new UILabel
        {
            X = 30, Y = sectionY,
            Text = "Depth Model:",
            FontSize = FontSize.Caption,
            Color = System.Drawing.Color.FromArgb(255, 180, 180, 200),
        });

        float modelX = 140;
        foreach (var model in DepthEstimationService.AvailableModels)
        {
            bool active = _activeProject.Settings.DepthModel == model.Id;
            var modelId = model.Id;
            float btnW = Math.Max(90, model.Name.Length * 7 + 16);
            mainPanel.AddChild(new UIButton
            {
                X = modelX, Y = sectionY - 3,
                Width = btnW, Height = 26,
                Text = model.Name,
                FontSize = FontSize.Caption,
                NormalColor = active
                    ? System.Drawing.Color.FromArgb(255, 108, 92, 231)
                    : System.Drawing.Color.FromArgb(255, 50, 50, 65),
                OnClick = () =>
                {
                    _activeProject.Settings.DepthModel = modelId;
                    _ = _projectService.UpdateProjectAsync(_activeProject);
                    BuildProjectDetailUI();
                },
            });
            modelX += btnW + 6;
        }
        sectionY += 35;

        bool canGenerate = _activeProject.Sources.Count > 0;
        mainPanel.AddChild(new UIButton
        {
            X = 30, Y = sectionY,
            Width = 200, Height = 40,
            Text = "Generate Scene",
            Enabled = canGenerate,
            OnClick = canGenerate ? OnGenerateSceneClicked : null,
        });
        sectionY += 55;

        // ── Generated Scenes Section ──
        if (_activeProject.Scenes.Count > 0)
        {
            mainPanel.AddChild(new UILabel
            {
                X = 30, Y = sectionY,
                Text = "Generated Scenes",
                FontSize = FontSize.Body,
                Color = System.Drawing.Color.White,
            });
            sectionY += 30;

            foreach (var scene in _activeProject.Scenes)
            {
                string sceneSizeStr = scene.SizeBytes < 1024 * 1024
                    ? $"{scene.SizeBytes / 1024.0:F0} KB" : $"{scene.SizeBytes / (1024.0 * 1024.0):F1} MB";

                float sceneCardH = 120;
                var sceneCard = mainPanel.AddChild(new UIPanel
                {
                    X = 30, Y = sectionY,
                    Width = Math.Min(panelW - 60, 550), Height = sceneCardH,
                    BackgroundColor = System.Drawing.Color.FromArgb(180, 30, 30, 45),
                });

                // Scene thumbnail
                string sceneThumbKey = $"scene:{scene.Id}";
                var sceneThumbView = _thumbnailCache.TryGetValue(sceneThumbKey, out var sceneCached) ? sceneCached.view : null;
                sceneCard.AddChild(new UIImage
                {
                    X = 5, Y = 5,
                    Width = 180, Height = sceneCardH - 10,
                    TextureView = sceneThumbView,
                });
                if (sceneThumbView == null)
                    LoadSceneThumbnailAsync(_activeProject.Id, scene.Id);

                sceneCard.AddChild(new UILabel
                {
                    X = 195, Y = 10,
                    Text = $"{scene.SplatCount:N0} splats · {scene.QualityPreset} · {sceneSizeStr}",
                    FontSize = FontSize.Body,
                    Color = System.Drawing.Color.FromArgb(255, 200, 200, 220),
                });

                sceneCard.AddChild(new UILabel
                {
                    X = 195, Y = 35,
                    Text = $"Created {scene.CreatedAt:g}",
                    FontSize = FontSize.Caption,
                    Color = System.Drawing.Color.Gray,
                });

                var sceneRef = scene;
                sceneCard.AddChild(new UIButton
                {
                    X = 195, Y = 60,
                    Width = 70, Height = 30,
                    Text = "View",
                    FontSize = FontSize.Caption,
                    OnClick = () => OnViewScene(sceneRef),
                });

                sceneCard.AddChild(new UIButton
                {
                    X = 280, Y = 60,
                    Width = 70, Height = 30,
                    Text = "Delete",
                    FontSize = FontSize.Caption,
                    NormalColor = System.Drawing.Color.FromArgb(255, 150, 50, 50),
                    HoverColor = System.Drawing.Color.FromArgb(255, 180, 60, 60),
                    PressedColor = System.Drawing.Color.FromArgb(255, 120, 40, 40),
                    OnClick = () => _ = OnDeleteScene(sceneRef),
                });

                sectionY += sceneCardH + 8;
            }
        }

        // Status message
        if (!string.IsNullOrEmpty(_statusMessage))
        {
            mainPanel.AddChild(new UILabel
            {
                X = 30, Y = panelH - 40,
                Text = _statusMessage,
                FontSize = FontSize.Caption,
                Color = System.Drawing.Color.FromArgb(255, 0, 206, 201),
            });
        }
    }
}
