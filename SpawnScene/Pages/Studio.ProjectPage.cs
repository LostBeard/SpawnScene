using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnScene.Models;
using SpawnScene.Services;
using System.Drawing;

namespace SpawnScene.Pages;

// The project page (TJ 2026-10-02: "the ui for the projects settings should not be under the project images ... the ui
// needs a lot of work to look professional"). Header; a main column with the photo grid and the generated scenes; a
// settings sidebar with Generate pinned at its foot. Below ProjectPageTwoColumnMinWidth the sidebar's sections follow the
// scenes in the one scrolling column instead.
public partial class Studio
{
    const float ProjectPageHeaderH = 76, ProjectPageStatusH = 34, ProjectPageGutter = 20;
    const float ProjectPageTwoColumnMinWidth = 860, ProjectSidebarMinW = 340, ProjectSidebarMaxW = 420;
    const float ProjectGenerateFooterH = 76;

    private static Color SidebarBackground => Color.FromArgb(255, 21, 26, 33);
    private static Color SectionRule => Color.FromArgb(90, 140, 150, 160);
    private static Color PrimaryAction => Color.FromArgb(255, 30, 132, 150);
    private static Color PrimaryActionHover => Color.FromArgb(255, 40, 156, 176);

    /// <summary>Photo tiles of the page on screen, by thumbnail key, so a thumbnail that finishes loading updates its
    /// tile instead of rebuilding the page (35 photos used to mean 35 rebuilds).</summary>
    private readonly Dictionary<string, UIImage> _thumbTiles = new();

    /// <summary>Photos selected on the project page (by file name): click tiles to select, then Remove.</summary>
    private readonly HashSet<string> _selectedSources = new();

    private void BuildProjectDetailUI()
    {
        _uiRoot.ClearChildren();
        _thumbTiles.Clear();
        if (_activeProject == null) return;

        float margin = 28;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;
        var shell = _uiRoot.AddChild(new UIPanel { X = margin, Y = margin, Width = panelW, Height = panelH, Padding = 0 });

        BuildProjectHeader(shell, panelW);

        float bodyTop = ProjectPageHeaderH, bodyH = panelH - bodyTop - ProjectPageStatusH;
        bool twoColumns = panelW >= ProjectPageTwoColumnMinWidth;
        float sideW = twoColumns ? Math.Clamp(panelW * 0.32f, ProjectSidebarMinW, ProjectSidebarMaxW) : 0;
        float mainW = panelW - sideW;

        shell.AddChild(new UIPanel { X = 0, Y = bodyTop - 1, Width = panelW, Height = 1, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });

        // One column pins Generate at the foot of the page (as the sidebar does), so the scroll area stops above it.
        float mainH = twoColumns ? bodyH : bodyH - ProjectGenerateFooterH;
        var main = shell.AddChild(new UIScrollView
        {
            X = 0, Y = bodyTop, Width = mainW, Height = mainH,
            Padding = 0, BackgroundColor = Color.Transparent, BorderWidth = 0,
        });
        _projectDetailScroll = main;

        // Nothing to act on may sit below the photo grid (TJ 2026-10-02: with many photos, controls under the images meant
        // scrolling to the bottom every time). Both layouts put the generated scenes - what you come back for - first;
        // one column follows them with the settings, then the photos, and pins Generate at the foot.
        float y = 18;
        float scenesTop = y;
        y = BuildScenesSection(main, y, mainW);
        if (y > scenesTop) y += 20;
        if (!twoColumns)
        {
            y = BuildSettingsSections(main, y, mainW) + 12;
            BuildGenerateFooter(shell, bodyTop + mainH, mainW);
        }
        y = BuildPhotosSection(main, y, mainW);

        if (twoColumns)
        {
            var side = shell.AddChild(new UIPanel
            {
                X = mainW, Y = bodyTop, Width = sideW, Height = bodyH,
                BackgroundColor = SidebarBackground, BorderWidth = 0, CornerRadius = 0, Padding = 0,
            });
            side.AddChild(new UIPanel { X = 0, Y = 0, Width = 1, Height = bodyH, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });
            var sideScroll = side.AddChild(new UIScrollView
            {
                X = 0, Y = 0, Width = sideW, Height = bodyH - ProjectGenerateFooterH,
                Padding = 0, BackgroundColor = Color.Transparent, BorderWidth = 0,
            });
            BuildSettingsSections(sideScroll, 18, sideW);
            BuildGenerateFooter(side, bodyH - ProjectGenerateFooterH, sideW);
        }

        shell.AddChild(new UIPanel { X = 0, Y = panelH - ProjectPageStatusH, Width = panelW, Height = 1, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });
        _statusLabel = shell.AddChild(new UILabel
        {
            X = ProjectPageGutter, Y = panelH - ProjectPageStatusH + 9,
            Width = panelW - ProjectPageGutter * 2,
            Text = string.IsNullOrEmpty(_statusMessage) ? "Ready" : _statusMessage,
            FontSize = FontSize.Caption,
            Color = Color.FromArgb(255, 80, 210, 220),
        });
    }

    private void BuildProjectHeader(UIElement shell, float panelW)
    {
        shell.AddChild(new UIButton
        {
            X = ProjectPageGutter, Y = 20, Width = 104, Height = 34,
            Text = "< Projects",
            FontSize = FontSize.Caption,
            Enabled = !_pipelineBusy,
            OnClick = async () =>
            {
                _projects = await _projectService.ListProjectsAsync();
                _state = StudioState.ProjectBrowser;
                _activeProject = null;
                _selectedSources.Clear();
                BuildProjectBrowserUI();
            },
        });
        shell.AddChild(new UILabel
        {
            X = ProjectPageGutter + 124, Y = 14,
            Text = _activeProject!.Name,
            FontSize = FontSize.Heading,
            Color = UITheme.Current.TextPrimary,
        });
        long sizeBytes = _projectService.GetProjectSize(_activeProject);
        string sizeStr = sizeBytes < 1024 * 1024 ? $"{sizeBytes / 1024.0:F0} KB" : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
        int photos = _activeProject.Sources.Count, scenes = _activeProject.Scenes.Count;
        shell.AddChild(new UILabel
        {
            X = ProjectPageGutter + 124, Y = 46,
            Text = $"{photos} photo{(photos == 1 ? "" : "s")}  ·  {scenes} scene{(scenes == 1 ? "" : "s")}  ·  {sizeStr}",
            FontSize = FontSize.Caption,
            Color = UITheme.Current.TextSecondary,
        });
    }

    /// <summary>A section heading with a rule under it; returns the y below it.</summary>
    private float AddSectionHeading(UIElement parent, float x, float y, float width, string title, string? detail = null)
    {
        parent.AddChild(new UILabel { X = x, Y = y, Text = title, FontSize = FontSize.Body, Color = UITheme.Current.TextPrimary });
        if (detail != null)
            parent.AddChild(new UILabel
            {
                // Right after the title (a count, a scope): pushed to the far edge it read as unrelated to the title.
                X = x + title.Length * 9.5f + 10, Y = y + 3,
                Text = detail, FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
            });
        parent.AddChild(new UIPanel { X = x, Y = y + 28, Width = width, Height = 1, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });
        return y + 40;
    }

    private float BuildPhotosSection(UIElement parent, float y, float width)
    {
        float x = ProjectPageGutter, w = width - ProjectPageGutter * 2;
        var sources = _activeProject!.Sources;
        _selectedSources.RemoveWhere(n => !sources.Any(src => src.FileName == n));
        float hx = x + w - 132;
        parent.AddChild(new UIButton
        {
            X = hx, Y = y - 4, Width = 132, Height = 30,
            Text = "+ Add photos",
            FontSize = FontSize.Caption,
            Enabled = !_pipelineBusy,
            OnClick = OnAddImagesClicked,
        });
        if (_selectedSources.Count > 0)
        {
            hx -= 128;
            parent.AddChild(new UIButton
            {
                X = hx, Y = y - 4, Width = 120, Height = 30,
                Text = $"Remove {_selectedSources.Count}",
                FontSize = FontSize.Caption,
                NormalColor = AccentDanger, HoverColor = AccentDangerHover,
                Enabled = !_pipelineBusy,
                OnClick = () => _ = RemoveSelectedSourcesAsync(),
            });
            hx -= 100;
            parent.AddChild(new UIButton
            {
                X = hx, Y = y - 4, Width = 92, Height = 30,
                Text = "Deselect", FontSize = FontSize.Caption,
                Enabled = !_pipelineBusy,
                OnClick = () => { _selectedSources.Clear(); BuildProjectDetailUI(); },
            });
        }
        y = AddSectionHeading(parent, x, y, w, "Photos", sources.Count > 0 ? $"{sources.Count}" : null);

        if (sources.Count >= 2)
        {
            // MEASURED 2026-10-02: on a small room (Bathroom, 24 training photos) neither resolution nor iterations moved
            // the views between the photos; coverage does. Say so where the photos are added.
            parent.AddChild(new UILabel
            {
                X = x, Y = y - 6,
                Text = "Best results: many overlapping photos from many positions - coverage matters more than any setting.",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
            });
            y += 18;
        }

        if (sources.Count == 0)
        {
            parent.AddChild(new UITextBlock
            {
                X = x, Y = y, Width = w, Height = 44,
                Text = "Add two or more photos of a scene, taken from different positions, to reconstruct it in 3D. " +
                       "One photo makes a depth-based scene.",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
            });
            y += 52;
            parent.AddChild(new UILabel { X = x, Y = y, Text = "Or try a sample:", FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted });
            y += 24;
            var samples = new[]
            {
                ("Room", "samples/room.png"), ("Garden", "samples/garden.png"),
                ("Living Room HD", "samples/living_room_hd.png"), ("Garden HD", "samples/garden_hd.png"),
            };
            float bx = x;
            foreach (var (name, path) in samples)
            {
                float bw = Math.Max(80, name.Length * 8 + 20);
                if (bx + bw > x + w) { bx = x; y += 34; }
                var samplePath = path; var sampleName = name;
                parent.AddChild(new UIButton
                {
                    X = bx, Y = y, Width = bw, Height = 28, Text = name, FontSize = FontSize.Caption,
                    NormalColor = AccentMuted, Enabled = !_pipelineBusy,
                    OnClick = () => _ = LoadSampleImage(sampleName, samplePath),
                });
                bx += bw + 8;
            }
            y += 40;
            parent.AddChild(new UIButton
            {
                X = x, Y = y, Width = 200, Height = 28, Text = "TempleRing (GT cameras)", FontSize = FontSize.Caption,
                NormalColor = AccentMuted, Enabled = !_pipelineBusy,
                OnClick = () => _ = GenerateFromTempleRingAsync(),
            });
            return y + 40;
        }

        // Square tiles (the thumbnails are centre-cropped squares), as many columns as fit at ~148 px. A click selects.
        const float gap = 10, captionH = 0;
        int cols = Math.Max(2, (int)((w + gap) / (148 + gap)));
        float tile = (w - gap * (cols - 1)) / cols;
        for (int i = 0; i < sources.Count; i++)
        {
            var src = sources[i];
            float tx = x + (i % cols) * (tile + gap), ty = y + (i / cols) * (tile + captionH + gap);
            string key = SourceThumbKey(_activeProject.Id, src.FileName);
            bool selected = _selectedSources.Contains(src.FileName);
            if (selected)
                parent.AddChild(new UIPanel
                {
                    X = tx - 3, Y = ty - 3, Width = tile + 6, Height = tile + 6,
                    BackgroundColor = Color.FromArgb(255, 80, 210, 230), BorderWidth = 0, CornerRadius = 3,
                });
            var view = _thumbnailCache.TryGetValue(key, out var cached) ? cached.view : null;
            var image = parent.AddChild(new UIImage { X = tx, Y = ty, Width = tile, Height = tile, TextureView = view });
            _thumbTiles[key] = image;
            if (view == null) LoadThumbnailAsync(_activeProject.Id, src.FileName);

            // A transparent button over the tile: the hover highlight, and a click toggles the selection.
            string fileName = src.FileName;
            parent.AddChild(new UIButton
            {
                X = tx, Y = ty, Width = tile, Height = tile, Text = "",
                NormalColor = selected ? Color.FromArgb(50, 80, 210, 230) : Color.Transparent,
                HoverColor = Color.FromArgb(45, 255, 255, 255),
                PressedColor = Color.FromArgb(70, 255, 255, 255),
                Enabled = !_pipelineBusy,
                OnClick = () =>
                {
                    if (!_selectedSources.Remove(fileName)) _selectedSources.Add(fileName);
                    BuildProjectDetailUI();
                },
            });
        }
        int rows = (sources.Count + cols - 1) / cols;
        return y + rows * (tile + captionH + gap);
    }

    private async Task RemoveSelectedSourcesAsync()
    {
        if (_activeProject == null) return;
        var picked = _activeProject.Sources.Where(src => _selectedSources.Contains(src.FileName)).ToList();
        _selectedSources.Clear();
        foreach (var src in picked) await OnRemoveSource(src);
        BuildProjectDetailUI();
    }

    private float BuildScenesSection(UIElement parent, float y, float width)
    {
        var scenes = _activeProject!.Scenes;
        if (scenes.Count == 0) return y;
        float x = ProjectPageGutter, w = width - ProjectPageGutter * 2;
        y = AddSectionHeading(parent, x, y, w, "Scenes", $"{scenes.Count}");

        const float cardH = 112, gap = 12;
        int cols = Math.Max(1, (int)((w + gap) / (420 + gap)));
        float cardW = (w - gap * (cols - 1)) / cols;
        for (int i = 0; i < scenes.Count; i++)
        {
            var scene = scenes[i];
            float cx = x + (i % cols) * (cardW + gap), cy = y + (i / cols) * (cardH + gap);
            var card = parent.AddChild(new UIPanel
            {
                X = cx, Y = cy, Width = cardW, Height = cardH,
                BackgroundColor = Color.FromArgb(255, 26, 31, 39), BorderWidth = 0, Padding = 0,
            });
            string thumbKey = $"scene:{scene.Id}";
            var thumb = _thumbnailCache.TryGetValue(thumbKey, out var cached) ? cached.view : null;
            float thumbW = Math.Min(150, cardW * 0.4f);
            card.AddChild(new UIImage { X = 6, Y = 6, Width = thumbW, Height = cardH - 12, TextureView = thumb });
            if (thumb == null) LoadSceneThumbnailAsync(_activeProject.Id, scene.Id);

            float lx = thumbW + 18;
            string size = scene.SizeBytes < 1024 * 1024 ? $"{scene.SizeBytes / 1024.0:F0} KB" : $"{scene.SizeBytes / (1024.0 * 1024.0):F1} MB";
            card.AddChild(new UILabel { X = lx, Y = 12, Text = $"{scene.SplatCount:N0} splats", FontSize = FontSize.Body, Color = UITheme.Current.TextPrimary });
            card.AddChild(new UILabel
            {
                X = lx, Y = 38,
                Text = (scene.TrainedIterations > 0 ? $"Trained {scene.TrainedIterations:N0} iterations" : "Untrained") + $"  ·  {size}",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
            });
            card.AddChild(new UILabel { X = lx, Y = 58, Text = $"{scene.CreatedAt:g}", FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted });
            var sceneRef = scene;
            card.AddChild(new UIButton
            {
                X = lx, Y = cardH - 36, Width = 72, Height = 28, Text = "Open", FontSize = FontSize.Caption,
                NormalColor = PrimaryAction, HoverColor = PrimaryActionHover,
                Enabled = !_pipelineBusy, OnClick = () => OnViewScene(sceneRef),
            });
            card.AddChild(new UIButton
            {
                X = lx + 80, Y = cardH - 36, Width = 72, Height = 28, Text = "Delete", FontSize = FontSize.Caption,
                NormalColor = UITheme.Current.ButtonNormal, HoverColor = AccentDangerHover,
                Enabled = !_pipelineBusy, OnClick = () => _ = OnDeleteScene(sceneRef),
            });
        }
        int rows = (scenes.Count + cols - 1) / cols;
        return y + rows * (cardH + gap);
    }

    private float BuildSettingsSections(UIElement parent, float y, float width)
    {
        float x = ProjectPageGutter, w = width - ProjectPageGutter * 2;
        var s = _activeProject!.Settings;

        // -- Quality preset: one choice sets the reconstruction rows below; changing a row makes it "Custom" --
        string preset = ReconstructionPresets.Match(s);
        var presetNames = ReconstructionPresets.All.Select(p => p.Name).ToList();
        y = AddSectionHeading(parent, x, y, w, "Quality", "2+ photos");
        y = AddChoiceRow(parent, x, y, w, "Preset",
            presetNames.Select((n, i) => (n, i)).ToArray(),
            presetNames.IndexOf(preset),
            i => ReconstructionPresets.Apply(s, presetNames[i]),
            preset == "Custom" ? "Custom: the settings below were changed by hand. Pick a preset to reset them."
                : ReconstructionPresets.All.First(p => p.Name == preset).Hint);

        y = AddSectionHeading(parent, x, y + 8, w, "Reconstruction", preset);
        y = AddChoiceRow(parent, x, y, w, "Training resolution",
            new (string, int)[] { ("720", 720), ("1024", 1024), ("1600", 1600), ("Photo", ReconstructionPresets.PhotoSize) },
            s.TrainMaxDimension, v => { s.TrainMaxDimension = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            "Longest side the photos are trained at, never above the photos' own size (Photo = their size). Needs more GPU memory and time.");
        y = AddChoiceRow(parent, x, y, w, "Training iterations",
            new (string, int)[] { ("Off", 0), ("3K", 3000), ("7K", 7000), ("15K", 15000), ("30K", 30000) },
            s.TrainIterations, v => { s.TrainIterations = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            "More iterations refine detail on well-covered scenes. A 251-photo scene takes about 70 minutes at 30K.");
        y = AddChoiceRow(parent, x, y, w, "Max splats",
            new (string, int)[] { ("500K", 500_000), ("1M", 1_000_000), ("3M", 3_000_000) },
            s.TrainMaxSplats, v => { s.TrainMaxSplats = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            "Upper bound on scene size while training grows it. Lower it on a GPU with less memory.");

        // -- Device: this machine's GPU memory budget (localStorage, all projects) --
        var (budgetTargets, budgetSplats) = GpuMemoryBudget.Derive(GpuMemoryGB, DeviceBindingLimitBytes, s.TrainMaxSplats);
        y = AddSectionHeading(parent, x, y + 8, w, "Device", "this computer");
        y = AddChoiceRow(parent, x, y, w, "GPU memory",
            GpuMemoryBudget.ChoicesGB.Select(gb => (gb == 0 ? $"Auto" : $"{gb} GB", gb)).ToArray(),
            GpuMemoryGB, v => GpuMemoryGB = v,
            $"Training uses up to {budgetTargets >> 20} MB for photos and {budgetSplats:N0} splats" +
            (budgetSplats < s.TrainMaxSplats ? $" (the preset allows {s.TrainMaxSplats:N0})" : "") +
            $". Auto assumes {GpuMemoryBudget.AutoGB} GB; set yours if it differs.");

        y = AddSectionHeading(parent, x, y + 8, w, "Single photo", "depth");
        var presets = new[] { ("Fast", 4, 0f), ("Standard", 2, 0.3f), ("High", 1, 0.3f) };
        y = AddChoiceRow(parent, x, y, w, "Quality",
            presets.Select((p, i) => (p.Item1, i)).ToArray(),
            Array.FindIndex(presets, p => p.Item1 == s.QualityPreset),
            i => { s.QualityPreset = presets[i].Item1; s.Subsample = presets[i].Item2; s.EdgeSharpness = presets[i].Item3; },
            "Splat density from the depth map: High makes one splat per pixel.");
        var models = DepthEstimationService.AvailableModels.ToList();
        y = AddChoiceRow(parent, x, y, w, "Depth model",
            models.Select((m, i) => (m.Name, i)).ToArray(),
            models.FindIndex(m => m.Id == s.DepthModel),
            i => s.DepthModel = models[i].Id, null);
        return y;
    }

    /// <summary>
    /// A labelled row of mutually exclusive choices (a segmented control): the caption above, the choices sharing the full
    /// width, an optional muted hint under them. Wraps onto further lines when the labels do not fit. Returns the y below.
    /// </summary>
    private float AddChoiceRow(UIElement parent, float x, float y, float width, string caption,
        (string Label, int Value)[] choices, int current, Action<int> set, string? hint)
    {
        parent.AddChild(new UILabel { X = x, Y = y, Text = caption, FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary });
        y += 22;
        const float gap = 4, h = 30;
        float minW = choices.Max(c => c.Label.Length * 8f + 24);
        int perRow = Math.Max(1, Math.Min(choices.Length, (int)((width + gap) / (minW + gap))));
        float bw = (width - gap * (perRow - 1)) / perRow;
        for (int i = 0; i < choices.Length; i++)
        {
            var (label, value) = choices[i];
            var v = value;
            bool active = current == value;
            parent.AddChild(new UIButton
            {
                X = x + (i % perRow) * (bw + gap), Y = y + (i / perRow) * (h + gap),
                Width = bw, Height = h, Text = label, FontSize = FontSize.Caption,
                NormalColor = active ? AccentSelected : UITheme.Current.ButtonNormal,
                Enabled = !_pipelineBusy,
                OnClick = () =>
                {
                    if (_activeProject == null) return;
                    set(v);
                    _ = _projectService.UpdateProjectAsync(_activeProject);
                    BuildProjectDetailUI();
                },
            });
        }
        y += ((choices.Length + perRow - 1) / perRow) * (h + gap) + 2;
        if (hint != null)
        {
            // Height from the real wrap (GameUI MeasureHeight, the same font and word breaks Draw uses). A character-count
            // estimate either ran a three-line hint into the next heading or left gaps; it stays only as the fallback for a
            // font atlas that is not ready yet.
            var block = parent.AddChild(new UITextBlock
            {
                X = x, Y = y, Width = width,
                Text = hint, FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
            });
            float hintH = block.MeasureHeight(_gameUI.Renderer);
            if (hintH <= 0) hintH = Math.Max(1, (int)Math.Ceiling(hint.Length * 5.6f / Math.Max(1f, width))) * 18;
            block.Height = hintH;
            y += hintH + 4;
        }
        return y + 10;
    }

    private void BuildGenerateFooter(UIElement parent, float y, float width)
    {
        parent.AddChild(new UIPanel { X = 0, Y = y, Width = width, Height = 1, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });
        bool canGenerate = _activeProject!.Sources.Count > 0 && !_pipelineBusy;
        parent.AddChild(new UIButton
        {
            X = ProjectPageGutter, Y = y + 16, Width = width - ProjectPageGutter * 2, Height = 44,
            Text = _pipelineBusy ? "Working..." : "Generate scene",
            NormalColor = PrimaryAction, HoverColor = PrimaryActionHover,
            Enabled = canGenerate,
            OnClick = canGenerate ? OnGenerateSceneClicked : null,
        });
    }
}
