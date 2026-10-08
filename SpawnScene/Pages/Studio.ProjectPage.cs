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

    /// <summary>The project page's open tab, and the project it belongs to (another project starts on its default).</summary>
    private string? _projectTab;
    private string? _projectTabProjectId;

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

        // One column pins Generate at the foot of the page (as the sidebar does), so the main area stops above it.
        float mainH = twoColumns ? bodyH : bodyH - ProjectGenerateFooterH;

        // Tabs, not a stack (TJ 2026-10-02): with the sections stacked, whatever came after a long photo grid needed a
        // scroll to the bottom. Scenes | Photos, plus Settings when there is no sidebar; each tab scrolls on its own.
        var tabNames = new List<string> { "Scenes", "Photos" };
        if (!twoColumns) tabNames.Add("Settings");
        int sceneCount = _activeProject!.Scenes.Count, photoCount = _activeProject.Sources.Count;
        string Label(string n) => n switch
        {
            "Scenes" => sceneCount > 0 ? $"Scenes  {sceneCount}" : "Scenes",
            "Photos" => photoCount > 0 ? $"Photos  {photoCount}" : "Photos",
            _ => n,
        };
        // The tab survives rebuilds (selecting a photo, a thumbnail arriving); a new project opens on Scenes when it has
        // any, else Photos.
        if (_projectTabProjectId != _activeProject.Id || !tabNames.Contains(_projectTab ?? ""))
        {
            _projectTab = sceneCount > 0 ? "Scenes" : "Photos";
            _projectTabProjectId = _activeProject.Id;
        }
        const float tabH = 44;
        float tabsW = mainW - ProjectPageGutter * 2;
        var tabs = shell.AddChild(new UITabPanel
        {
            X = ProjectPageGutter, Y = bodyTop + 8, Width = tabsW, Height = mainH - 8,
            Padding = 0, TabHeight = tabH, TabWidth = 150, TabFontSize = FontSize.Body,
            BackgroundColor = Color.Transparent, BorderWidth = 0, CornerRadius = 0,
            TabColor = Color.Transparent, HoverTabColor = Color.FromArgb(40, 255, 255, 255),
            ActiveTabColor = Color.FromArgb(255, 26, 31, 39),
        });
        float contentH = mainH - 8 - tabH;
        foreach (var name in tabNames)
        {
            var scroll = new UIScrollView
            {
                Width = tabsW, Height = contentH,
                Padding = 0, BackgroundColor = Color.Transparent, BorderWidth = 0,
            };
            // Leave the scrollbar its own strip on the right.
            float contentW = tabsW - 12;
            switch (name)
            {
                case "Scenes":
                    if (sceneCount > 0) BuildScenesSection(scroll, 18, contentW, 0);
                    else
                        scroll.AddChild(new UITextBlock
                        {
                            X = 0, Y = 22, Width = Math.Min(contentW, 560), Height = 40,
                            Text = photoCount >= 2 ? "No scenes yet. Generate scene builds one from the photos."
                                : "No scenes yet. Add photos (two or more of the same place, or one for a depth scene), then Generate scene.",
                            FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
                        });
                    break;
                case "Photos":
                    BuildPhotosSection(scroll, 18, contentW, 0);
                    break;
                case "Settings":
                    BuildSettingsSections(scroll, 18, contentW, 0);
                    break;
            }
            tabs.AddTab(Label(name), scroll);
            if (name == (twoColumns ? "Photos" : "Settings")) _projectDetailScroll = scroll;
        }
        tabs.ActiveIndex = tabNames.IndexOf(_projectTab!);
        tabs.OnTabChanged = (i, _) => _projectTab = tabNames[i];
        if (!twoColumns) BuildGenerateFooter(shell, bodyTop + mainH, mainW);

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

    private float BuildPhotosSection(UIElement parent, float y, float width, float gutter = ProjectPageGutter)
    {
        float x = gutter, w = width - gutter * 2;
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
        // No "Photos" heading: the tab already says it (with the count); this row is the Add / Remove actions.
        y += 40;

        if (!string.IsNullOrEmpty(_activeProject?.Credit))
        {
            // Wraps: the credit carries its source link, which the CC BY-SA sets ask for.
            parent.AddChild(new UITextBlock
            {
                X = x, Y = y - 6, Width = w, Height = 34, Text = "Photos: " + _activeProject!.Credit,
                FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
            });
            y += 36;
        }

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
            // Samples: samples/catalog.json, photos hosted off-site (SampleCatalog). TJ 2026-10-07: the old 640 px PNGs
            // were not worth demoing - high-resolution single photos and real multi-photo sets instead.
            if (_sampleCatalog == null) { _ = EnsureSampleCatalogAsync(); return y; }
            if (_sampleCatalog.Samples.Count == 0) return y;
            var catalog = _sampleCatalog;
            foreach (var (heading, sets) in new[] { ("Or try a photo set:", true), ("Or a single photo:", false) })
            {
                var group = catalog.Samples.Where(sm => sm.IsSet == sets).ToList();
                if (group.Count == 0) continue;
                parent.AddChild(new UILabel { X = x, Y = y, Text = heading, FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted });
                y += 24;
                float bx = x;
                foreach (var sample in group)
                {
                    string label = sets
                        ? $"{sample.Name} ({sample.Images.Count} photos, {sample.Bytes / (1024 * 1024)} MB)"
                        : $"{sample.Name} ({Math.Max(1, sample.Bytes / (1024 * 1024))} MB)";
                    float bw = Math.Max(80, label.Length * 7 + 20);
                    if (bx + bw > x + w) { bx = x; y += 34; }
                    var picked = sample;
                    parent.AddChild(new UIButton
                    {
                        X = bx, Y = y, Width = bw, Height = 28, Text = label, FontSize = FontSize.Caption,
                        NormalColor = AccentMuted, Enabled = !_pipelineBusy,
                        OnClick = () => _ = LoadSampleAsync(catalog, picked),
                    });
                    bx += bw + 8;
                }
                y += 40;
            }
            parent.AddChild(new UILabel
            {
                X = x, Y = y - 6, Text = "Sample photos are openly licensed; each keeps its credit with the project.",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
            });
            return y + 24;
        }

        // Square tiles (the thumbnails are centre-cropped squares), as many columns as fit at ~148 px. A click selects.
        const float gap = 10, captionH = 0;
        int cols = Math.Max(2, (int)((w + gap) / (148 + gap)));
        float tile = (w - gap * (cols - 1)) / cols;
        HashSet<string>? notPlaced = null;
        for (int i = 0; i < sources.Count; i++)
        {
            var src = sources[i];
            float tx = x + (i % cols) * (tile + gap), ty = y + (i / cols) * (tile + captionH + gap);
            notPlaced ??= (_activeProject.Scenes.LastOrDefault(sc => sc.PhotosTotal >= 2)?.PhotosNotPlaced ?? Array.Empty<string>()).ToHashSet();
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
            // The latest generate could not place this photo (no camera found for it): say so on the tile.
            if (notPlaced.Contains(src.FileName))
            {
                parent.AddChild(new UIPanel
                {
                    X = tx, Y = ty + tile - 22, Width = tile, Height = 22,
                    BackgroundColor = Color.FromArgb(200, 120, 60, 20), BorderWidth = 0, CornerRadius = 0,
                });
                parent.AddChild(new UILabel { X = tx + 6, Y = ty + tile - 19, Text = "not placed", FontSize = FontSize.Caption, Color = Color.White });
            }

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

    private float BuildScenesSection(UIElement parent, float y, float width, float gutter = ProjectPageGutter)
    {
        var scenes = _activeProject!.Scenes;
        if (scenes.Count == 0) return y;
        float x = gutter, w = width - gutter * 2;
        // No "Scenes" heading: the tab already says it.

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
                Text = (scene.EditedFrom != null ? "Edited  ·  " : "")
                    + (scene.TrainedIterations > 0 ? $"Trained {scene.TrainedIterations:N0} iterations" : "Untrained") + $"  ·  {size}",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
            });
            card.AddChild(new UILabel { X = lx, Y = 58, Text = $"{scene.CreatedAt:g}", FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted });
            if (scene.PhotosTotal >= 2)
            {
                // Which photos made it in - a user who knows which ones were left out can retake them (PLANS: capture feedback).
                var missing = scene.PhotosNotPlaced ?? Array.Empty<string>();
                string text = missing.Length == 0
                    ? $"All {scene.PhotosTotal} photos placed"
                    : $"{scene.PhotosPlaced} of {scene.PhotosTotal} photos placed (see Photos)";
                card.AddChild(new UILabel
                {
                    X = lx, Y = 76, Text = text, FontSize = FontSize.Caption,
                    Color = missing.Length == 0 ? UITheme.Current.TextMuted : Color.FromArgb(255, 230, 180, 90),
                });
            }
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

    private float BuildSettingsSections(UIElement parent, float y, float width, float gutter = ProjectPageGutter)
    {
        float x = gutter, w = width - gutter * 2;
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

        // What these photos would train at under this device's budget, and for how long (TrainingTimeEstimate: the trainer's
        // own sizing rule, and the speed this device measured on its last run).
        var (budgetTargets, _) = GpuMemoryBudget.Derive(GpuMemoryGB, DeviceBindingLimitBytes, s.TrainMaxSplats);
        var photos = _activeProject.Sources.Where(p => p.Width > 0 && p.Height > 0).ToList();
        string sizeNote = "", timeNote;
        TrainingTimeEstimate.Duration? trainTime = null;
        if (photos.Count >= 2)
        {
            var (tw, th, shrunk) = TrainingTimeEstimate.TrainingSize(photos[0].Width, photos[0].Height,
                photos.Min(p => Math.Max(p.Width, p.Height)), s.TrainMaxDimension, photos.Count, budgetTargets);
            sizeNote = $" These {photos.Count} photos train at {tw}x{th}" + (shrunk ? ", smaller to fit the GPU memory budget." : ".");
            if (s.TrainIterations > 0)
                trainTime = TrainingTimeEstimate.Estimate(TrainMarks, s.TrainIterations, TrainingTimeEstimate.Megapixels(tw, th));
        }
        if (s.TrainIterations == 0)
            timeNote = "Off skips training: the scene stays the matched point cloud.";
        else if (trainTime is TrainingTimeEstimate.Duration time)
            timeNote = $"Training takes {TrainingTimeEstimate.Describe(time)} on this computer, from its earlier runs " +
                "(scenes that grow more splats take longer). More iterations refine detail on well-covered scenes.";
        else
            timeNote = "More iterations refine detail on well-covered scenes. A time estimate appears here after this computer's first training run.";

        y = AddSectionHeading(parent, x, y + 8, w, "Reconstruction", preset);
        y = AddChoiceRow(parent, x, y, w, "Training resolution",
            new (string, int)[] { ("720", 720), ("1024", 1024), ("1600", 1600), ("Photo", ReconstructionPresets.PhotoSize) },
            s.TrainMaxDimension, v => { s.TrainMaxDimension = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            "Longest side the photos are trained at, never above the photos' own size (Photo = their size). Needs more GPU memory and time." +
            sizeNote);
        y = AddChoiceRow(parent, x, y, w, "Keypoints",
            new (string, int)[] { ("1024", ReconstructionPresets.StandardKeypoints), ("3072", ReconstructionPresets.HighKeypoints) },
            s.LearnedKeypoints, v => { s.LearnedKeypoints = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            "Learned keypoints a photo for matching. 3072 places more cameras when photos are far apart (DrJohnson: 30 vs 27 " +
            "cameras, +3.3 dB) for about 6x the matching time (251 photos: 20 vs 3.5 minutes).");
        y = AddChoiceRow(parent, x, y, w, "Training iterations",
            new (string, int)[] { ("Off", 0), ("3K", 3000), ("7K", 7000), ("15K", 15000), ("30K", 30000) },
            s.TrainIterations, v => { s.TrainIterations = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            timeNote);
        var (_, deviceSplats) = GpuMemoryBudget.Derive(GpuMemoryGB, DeviceBindingLimitBytes, ReconstructionPresets.DeviceMaxSplats);
        y = AddChoiceRow(parent, x, y, w, "Max splats",
            new (string, int)[] { ("500K", 500_000), ("1M", 1_000_000), ("3M", 3_000_000), ("6M", 6_000_000),
                ("10M", 10_000_000), ("Device", ReconstructionPresets.DeviceMaxSplats) },
            s.TrainMaxSplats, v => { s.TrainMaxSplats = v; s.ReconstructionPreset = ReconstructionPresets.Match(s); },
            $"Upper bound on scene size while training grows it; a scene stops growing once its photos are covered. " +
            $"Device = as many as this computer's GPU memory setting fits: {deviceSplats:N0}" +
            (s.TrainMaxSplats != ReconstructionPresets.DeviceMaxSplats && s.TrainMaxSplats > deviceSplats
                ? $", so it caps this {s.TrainMaxSplats:N0}." : "."));

        // -- Device: this machine's GPU memory budget (localStorage, all projects) --
        y = AddSectionHeading(parent, x, y + 8, w, "Device", "this computer");
        y = AddChoiceRow(parent, x, y, w, "GPU memory",
            GpuMemoryBudget.ChoicesGB.Select(gb => (gb == 0 ? $"Auto" : $"{gb} GB", gb)).ToArray(),
            GpuMemoryGB, v => GpuMemoryGB = v,
            $"Training uses up to {budgetTargets >> 20} MB for photos and {deviceSplats:N0} splats " +
            $"(about {GpuMemoryBudget.BytesPerSplat:N0} bytes each while training). Auto assumes {GpuMemoryBudget.AutoGB} GB; " +
            "set your GPU's memory to train larger scenes. Setting more than the GPU has can lose the device mid-run. " +
            "Chrome and Edge on Windows also cap all GPU use at 8 GB on PCs with 16 GB of RAM or less (16 GB with 32 GB RAM).");

        y = AddSectionHeading(parent, x, y + 8, w, "Single photo", "depth");
        var presets = new[] { ("Fast", 4, 0f), ("Standard", 2, 0.3f), ("High", 1, 0.3f) };
        y = AddChoiceRow(parent, x, y, w, "Quality",
            presets.Select((p, i) => (p.Item1, i)).ToArray(),
            Array.FindIndex(presets, p => p.Item1 == s.QualityPreset),
            i => { s.QualityPreset = presets[i].Item1; s.Subsample = presets[i].Item2; s.EdgeSharpness = presets[i].Item3; },
            "Splat density from the depth map: High makes one splat per pixel.");
        y = AddChoiceRow(parent, x, y, w, "Super-resolution",
            new (string, int)[] { ("Off", (int)SuperResolutionMode.Off), ("Auto", (int)SuperResolutionMode.Auto), ("x3", (int)SuperResolutionMode.On) },
            (int)s.SuperResolution, v => s.SuperResolution = (SuperResolutionMode)v,
            $"Triples a photo's resolution before it becomes splats (ESPCN, on the GPU): finer colour detail and splats, " +
            $"more splats (about 9x). Auto does it for photos under {SuperResAutoBelowPx} px; x3 for any photo up to " +
            $"{SuperResMaxOutputPx / 3} px (larger ones already have the detail).");
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
