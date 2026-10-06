using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Models;
using System.Drawing;

namespace SpawnScene.Pages;

// The project browser (2026-10-02, after the project page redesign): a header with the actions, then a gallery of project
// cards - a 16:10 thumbnail (the latest scene, else the first photo), the name and the counts. The whole card opens the
// project; Delete asks before it deletes (it used to delete on one click).
public partial class Studio
{
    /// <summary>The project whose Delete was pressed once and now shows "Confirm"; any other click clears it.</summary>
    private string? _pendingDeleteProjectId;

    /// <summary>Thumbnail cache key of a project's source photo. Per project: phone photos share names (IMG_0001.jpg)
    /// across projects, and a key of the file name alone showed one project's photo in another.</summary>
    private static string SourceThumbKey(string projectId, string fileName, int w = 256, int h = 256) =>
        w == h ? $"source:{projectId}:{fileName}" : $"source:{projectId}:{fileName}:{w}x{h}";

    private void BuildProjectBrowserUI()
    {
        _uiRoot.ClearChildren();
        _thumbTiles.Clear();
        _statusLabel = null;

        float margin = 28;
        float panelW = _canvasWidth - margin * 2;
        float panelH = _canvasHeight - margin * 2;
        var shell = _uiRoot.AddChild(new UIPanel { X = margin, Y = margin, Width = panelW, Height = panelH, Padding = 0 });

        // -- Header: product name, then the actions on the right (primary first from the right) --
        shell.AddChild(new UILabel { X = ProjectPageGutter, Y = 16, Text = "SpawnScene", FontSize = FontSize.Title, Color = UITheme.Current.TextPrimary });
        shell.AddChild(new UILabel
        {
            X = ProjectPageGutter, Y = 54, Text = "Gaussian splat scenes from your photos, on your GPU",
            FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
        });
        float bx = panelW - ProjectPageGutter - 150;
        shell.AddChild(new UIButton
        {
            X = bx, Y = 22, Width = 150, Height = 38, Text = "+ New project",
            NormalColor = PrimaryAction, HoverColor = PrimaryActionHover,
            OnClick = OnNewProjectClicked,
        });
        bx -= 138;
        shell.AddChild(new UIButton
        {
            X = bx, Y = 25, Width = 130, Height = 32, Text = "Open scene file", FontSize = FontSize.Caption,
            OnClick = OnOpenSceneFileClicked,
        });
        bx -= 108;
        shell.AddChild(new UIButton
        {
            X = bx, Y = 25, Width = 100, Height = 32, Text = "Testing", FontSize = FontSize.Caption,
            OnClick = () => { _state = StudioState.Testing; BuildTestingUI(); },
        });
        bx -= 88;
        shell.AddChild(new UIButton
        {
            X = bx, Y = 25, Width = 80, Height = 32, Text = "Home", FontSize = FontSize.Caption,
            OnClick = () => _nav.NavigateTo(""),
        });
        shell.AddChild(new UIPanel { X = 0, Y = 86, Width = panelW, Height = 1, BackgroundColor = SectionRule, BorderWidth = 0, CornerRadius = 0 });

        var list = shell.AddChild(new UIScrollView
        {
            X = 0, Y = 87, Width = panelW, Height = panelH - 87,
            Padding = 0, BackgroundColor = Color.Transparent, BorderWidth = 0,
        });
        float x = ProjectPageGutter, w = panelW - ProjectPageGutter * 2, y = 20;

        if (_projects == null || _projects.Count == 0)
        {
            list.AddChild(new UILabel { X = x, Y = y, Text = "No projects yet", FontSize = FontSize.Heading, Color = UITheme.Current.TextPrimary });
            list.AddChild(new UITextBlock
            {
                X = x, Y = y + 36, Width = Math.Min(w, 560), Height = 40,
                Text = "A project holds the photos of one scene. Create one, add two or more photos taken from different " +
                       "positions, and Generate builds the scene. Testing has sample datasets.",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
            });
            list.AddChild(new UIButton
            {
                X = x, Y = y + 92, Width = 170, Height = 38, Text = "+ New project",
                NormalColor = PrimaryAction, HoverColor = PrimaryActionHover, OnClick = OnNewProjectClicked,
            });
            return;
        }

        y = AddSectionHeading(list, x, y, w, "Projects", $"{_projects.Count}");

        // Cards as many as fit at ~300 px; the thumbnail is 16:10 over the card's width - the scene thumbnails' own aspect
        // (320x200), and photos get a 16:10 cover crop, so nothing is stretched.
        const float gap = 16, infoH = 74;
        int cols = Math.Max(1, (int)((w + gap) / (300 + gap)));
        float cardW = (w - gap * (cols - 1)) / cols;
        float thumbH = cardW * 10f / 16f;
        float cardH = thumbH + infoH;
        for (int i = 0; i < _projects.Count; i++)
        {
            var project = _projects[i];
            float cx = x + (i % cols) * (cardW + gap), cy = y + (i / cols) * (cardH + gap);
            var card = list.AddChild(new UIPanel
            {
                X = cx, Y = cy, Width = cardW, Height = cardH,
                BackgroundColor = Color.FromArgb(255, 26, 31, 39), BorderWidth = 0, Padding = 0,
            });

            // Thumbnail: the latest scene, else the first photo, else a placeholder that says why it is empty.
            string? key = null;
            var latestScene = project.Scenes.LastOrDefault();
            var firstSource = project.Sources.FirstOrDefault();
            if (latestScene != null) key = $"scene:{latestScene.Id}";
            else if (firstSource != null) key = SourceThumbKey(project.Id, firstSource.FileName, 320, 200);
            var view = key != null && _thumbnailCache.TryGetValue(key, out var cached) ? cached.view : null;
            var image = card.AddChild(new UIImage
            {
                X = 0, Y = 0, Width = cardW, Height = thumbH, TextureView = view,
                PlaceholderColor = Color.FromArgb(255, 32, 38, 47),
            });
            if (key != null) _thumbTiles[key] = image;
            if (view == null)
            {
                if (latestScene != null) LoadSceneThumbnailAsync(project.Id, latestScene.Id);
                else if (firstSource != null) LoadThumbnailAsync(project.Id, firstSource.FileName, 320, 200);
                else
                    card.AddChild(new UILabel
                    {
                        X = 0, Y = thumbH / 2 - 9, Width = cardW, Align = TextAlign.Center,
                        Text = "No photos yet", FontSize = FontSize.Caption, Color = UITheme.Current.TextMuted,
                    });
            }

            // The whole card opens the project (a transparent button over it; Delete sits on top and wins).
            var p = project;
            card.AddChild(new UIButton
            {
                X = 0, Y = 0, Width = cardW, Height = cardH, Text = "",
                NormalColor = Color.Transparent, HoverColor = Color.FromArgb(28, 255, 255, 255),
                PressedColor = Color.FromArgb(50, 255, 255, 255),
                OnClick = () => { _pendingDeleteProjectId = null; OnOpenProject(p); },
            });

            int maxChars = Math.Max(8, (int)((cardW - 24) / 9.5f));
            string name = project.Name.Length <= maxChars ? project.Name : project.Name[..(maxChars - 3)] + "...";
            card.AddChild(new UILabel { X = 12, Y = thumbH + 10, Text = name, FontSize = FontSize.Body, Color = UITheme.Current.TextPrimary });
            long sizeBytes = _projectService.GetProjectSize(project);
            string sizeStr = sizeBytes < 1024 * 1024 ? $"{sizeBytes / 1024.0:F0} KB" : $"{sizeBytes / (1024.0 * 1024.0):F1} MB";
            int photos = project.Sources.Count, scenes = project.Scenes.Count;
            card.AddChild(new UILabel
            {
                X = 12, Y = thumbH + 38,
                Text = $"{photos} photo{(photos == 1 ? "" : "s")}  ·  {scenes} scene{(scenes == 1 ? "" : "s")}  ·  {sizeStr}",
                FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
            });

            bool confirming = _pendingDeleteProjectId == project.Id;
            card.AddChild(new UIButton
            {
                X = cardW - (confirming ? 132 : 76), Y = thumbH + 30, Width = confirming ? 120 : 64, Height = 28,
                Text = confirming ? "Confirm delete" : "Delete", FontSize = FontSize.Caption,
                NormalColor = confirming ? AccentDanger : Color.FromArgb(255, 34, 40, 50), HoverColor = AccentDangerHover,
                OnClick = () =>
                {
                    if (_pendingDeleteProjectId == p.Id) { _pendingDeleteProjectId = null; _ = OnDeleteProject(p); }
                    else { _pendingDeleteProjectId = p.Id; BuildProjectBrowserUI(); }
                },
            });
        }
    }
}
