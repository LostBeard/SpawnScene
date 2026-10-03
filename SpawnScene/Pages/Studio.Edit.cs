using System.Drawing;
using System.Numerics;
using SpawnDev.GameUI;
using SpawnDev.GameUI.Elements;
using SpawnScene.Services;

namespace SpawnScene.Pages;

// Scene editing in the viewer: select splats, delete them or keep only them, undo (SplatEditor does the GPU work).
// Desktop/touch first; the same edits are what the in-headset tools drive.
public partial class Studio
{
    readonly SplatEditor _splatEditor = new();
    bool _editOpen, _selectMode, _editBusy, _dragging;
    Vector2? _dragStart, _dragEnd;
    SplatEditor.Volume? _selection;
    int _selectedCount;
    UIPanel? _selectRect;
    UILabel? _editStatus;
    UIButton? _selectButton;

    /// <summary>The Edit toolbar (left edge, under the top bar), built with the viewer HUD when Edit is open.</summary>
    void BuildEditToolbar()
    {
        _selectRect = null;
        _editStatus = null;
        _selectButton = null;
        if (!_editOpen) return;
        const float x = 12, w = 132, h = 34, gap = 8;
        float y = 68;
        var panel = _uiRoot.AddChild(new UIPanel
        {
            X = x, Y = y, Width = w + 24, Height = 6 * (h + gap) + 54,
            BackgroundColor = Color.FromArgb(200, 12, 16, 22),
        });
        float by = 12;
        UIButton Add(string text, Action click, bool accent = false)
        {
            var b = panel.AddChild(new UIButton
            {
                X = 12, Y = by, Width = w, Height = h, Text = text, FontSize = FontSize.Caption, OnClick = click,
            });
            if (accent) b.NormalColor = AccentSelected;
            by += h + gap;
            return b;
        }
        _selectButton = Add(_selectMode ? "Selecting..." : "Select", () =>
        {
            _selectMode = !_selectMode;
            ReleasePointerLock();
            BuildViewerHudUI();
        }, _selectMode);
        Add("Delete", () => _ = ApplyEditAsync(SplatEditor.Mode.DeleteInside));
        Add("Keep only", () => _ = ApplyEditAsync(SplatEditor.Mode.KeepInside));
        Add("Undo", () => _ = UndoEditAsync());
        Add("Clear selection", () => { _selection = null; _selectedCount = 0; _dragStart = _dragEnd = null; BuildViewerHudUI(); });
        Add("Save as new scene", () => _ = SaveEditedSceneAsync());
        _editStatus = panel.AddChild(new UILabel
        {
            X = 12, Y = by + 2, Text = EditStatusText(), FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
        });
        if (_dragStart is { } a && _dragEnd is { } b2) ShowSelectRect(a, b2);
    }

    string? _editNote;   // a one-off result ("Saved", "No project") shown until the next edit

    /// <summary>
    /// Save the scene on screen, edits included, to the active project as a NEW scene (the original stays). Deleted
    /// splats are saved at opacity 0 (they load invisible); trained colour (SH) bands come along.
    /// </summary>
    async Task SaveEditedSceneAsync()
    {
        if (_editBusy) return;
        if (_activeProject == null) { _editNote = "Open a project to save into"; RefreshEditStatus(); return; }
        _editBusy = true; _editNote = null; RefreshEditStatus();
        try
        {
            await SaveViewedSceneToProjectAsync(_viewedProjectScene?.TrainedIterations ?? 0, editedFrom: _viewedProjectScene?.Id);
            _editNote = "Saved as a new scene";
            // Its card thumbnail, from this view once it has settled (as a generated scene's is).
            if (_viewedProjectScene != null)
            {
                _pendingThumbnailProjectId = _activeProject.Id;
                _pendingThumbnailSceneId = _viewedProjectScene.Id;
                _thumbnailDelayFrames = 30;
            }
            Console.WriteLine($"[Edit] saved as a new scene in '{_activeProject.Name}'");
        }
        catch (Exception ex) { _editNote = "Save failed"; Console.WriteLine($"[Edit] save failed: {ex.Message}"); }
        finally { _editBusy = false; }
        RefreshEditStatus();
    }

    string EditStatusText() => _editBusy ? "Working..." : _editNote != null ? _editNote
        : _selection != null ? $"{_selectedCount:N0} selected"
        : _selectMode ? "Drag over the scene" : "Nothing selected";

    void ShowSelectRect(Vector2 a, Vector2 b)
    {
        float x0 = MathF.Min(a.X, b.X), y0 = MathF.Min(a.Y, b.Y);
        if (_selectRect == null)
        {
            _selectRect = _uiRoot.AddChild(new UIPanel
            {
                BackgroundColor = Color.FromArgb(50, 120, 220, 255),
                BorderColor = Color.FromArgb(230, 120, 220, 255), BorderWidth = 2, CornerRadius = 2,
            });
        }
        _selectRect.X = x0; _selectRect.Y = y0;
        _selectRect.Width = MathF.Abs(b.X - a.X); _selectRect.Height = MathF.Abs(b.Y - a.Y);
    }

    /// <summary>
    /// Select mode's pointer handling (mouse or a finger): a drag over the scene draws the rectangle; on release the
    /// splats under it, near to far, are the selection. Returns true when it consumed the pointer (no pointer lock and
    /// no touch navigation).
    /// </summary>
    bool HandleSelectPointer(SpawnDev.GameUI.Input.Pointer? p)
    {
        if (!_selectMode || p == null || p.ScreenPosition is not { } pos) return false;
        if (p.WasPressed)
        {
            if (_uiRoot.HitTest(pos) != null) return false;   // a click on the toolbar, not a selection
            _dragStart = _dragEnd = pos;
            _dragging = true;
            _editNote = null;
            _selection = null;
            ShowSelectRect(pos, pos);
            return true;
        }
        if (!_dragging || _dragStart is not { } start) return false;
        if (p.IsPressed)
        {
            _dragEnd = pos;
            ShowSelectRect(start, pos);
            return true;
        }
        if (p.WasReleased || !p.IsPressed)
        {
            _dragEnd = pos;
            _dragging = false;   // the rectangle stays drawn (_dragStart/_dragEnd) until the selection is used or cleared
            FinishSelection(start, pos);
            return true;
        }
        return false;
    }

    void FinishSelection(Vector2 a, Vector2 b)
    {
        if (MathF.Abs(a.X - b.X) < 3 || MathF.Abs(a.Y - b.Y) < 3)
        {
            _dragStart = _dragEnd = null;
            BuildViewerHudUI();
            return;
        }
        // Screen pixels -> NDC (y up). The camera's own projection gives the same NDC as the renderer's
        // canvas-scaled one (focal and principal point scale with the canvas per axis).
        float Nx(float px) => px / Math.Max(1, _canvasWidth) * 2f - 1f;
        float Ny(float py) => 1f - py / Math.Max(1, _canvasHeight) * 2f;
        var cam = _sceneManager.Camera;
        _selection = SplatEditor.Volume.ScreenRect(cam.ViewMatrix * cam.ProjectionMatrix, Nx(a.X), Nx(b.X), Ny(a.Y), Ny(b.Y));
        _ = CountSelectionAsync();
    }

    async Task CountSelectionAsync()
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null || _selection is not { } v) return;
        _editBusy = true; RefreshEditStatus();
        try { _selectedCount = await _splatEditor.CountAsync(_gpuService.WebGPUAccelerator, packed, _gpuRenderer.SplatCount, v); }
        catch (Exception ex) { Console.WriteLine($"[Edit] count failed: {ex.Message}"); }
        finally { _editBusy = false; }
        Console.WriteLine($"[Edit] selected {_selectedCount:N0} splats");
        RefreshEditStatus();
    }

    async Task ApplyEditAsync(SplatEditor.Mode mode)
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null || _selection is not { } v || _editBusy) return;
        _editBusy = true; _editNote = null; RefreshEditStatus();
        try
        {
            await _splatEditor.ApplyAsync(_gpuService.WebGPUAccelerator, packed, _gpuRenderer.SplatCount, v, mode);
            _gpuRenderer.SplatsEdited(_sceneManager.Camera.Position);
            Console.WriteLine($"[Edit] {(mode == SplatEditor.Mode.DeleteInside ? "deleted" : "kept only")} {_selectedCount:N0} splats (undo depth {_splatEditor.UndoDepth})");
        }
        finally { _editBusy = false; }
        _selection = null; _selectedCount = 0; _dragStart = _dragEnd = null;
        BuildViewerHudUI();
    }

    async Task UndoEditAsync()
    {
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null || _editBusy) return;
        _editBusy = true; RefreshEditStatus();
        bool undone;
        try
        {
            undone = await _splatEditor.UndoAsync(_gpuService.WebGPUAccelerator, packed, _gpuRenderer.SplatCount);
            if (undone) _gpuRenderer.SplatsEdited(_sceneManager.Camera.Position);
        }
        finally { _editBusy = false; }
        Console.WriteLine(undone ? $"[Edit] undone (undo depth {_splatEditor.UndoDepth})" : "[Edit] nothing to undo");
        RefreshEditStatus();
    }

    void RefreshEditStatus()
    {
        if (_editStatus != null) _editStatus.Text = EditStatusText();
    }
}
