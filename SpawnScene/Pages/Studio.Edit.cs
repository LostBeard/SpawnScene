using System.Drawing;
using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
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
    SplatEditor.Volume? _selectionValue;
    /// <summary>The edit selection; setting it also tints it on screen (GpuGaussianRenderer.SetSelectionHighlight).</summary>
    SplatEditor.Volume? _selection
    {
        get => _selectionValue;
        set { _selectionValue = value; _gpuRenderer.SetSelectionHighlight(value); }
    }
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
            X = x, Y = y, Width = w + 24, Height = 12 * (h + gap) + 54,
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
        Add("Copy", () => _ = CopySelectionAsync(cut: false));
        Add("Cut", () => _ = CopySelectionAsync(cut: true));
        Add("Paste", () => _ = PasteClipboardAsync());
        Add(_insertListOpen ? "Insert scene  <" : "Insert scene  >", () => _ = ToggleInsertListAsync());
        Add("Undo", () => _ = UndoEditAsync());
        Add("Clear selection", () => { _selection = null; _selectedCount = 0; _dragStart = _dragEnd = null; BuildViewerHudUI(); });
        Add("Save as new scene", () => _ = SaveEditedSceneAsync());
        Add("Export file", () => _ = ExportSceneFileAsync());
        // The same scene as an LOD tree in chunks (.spawnscene v3): opens at once and streams, however large.
        Add("Export streaming", () => _ = ExportLodSceneFileAsync());
        _editStatus = panel.AddChild(new UILabel
        {
            X = 12, Y = by + 2, Text = EditStatusText(), FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary,
        });
        // Move the selection, relative to the camera: a step is a tenth of the scene's move speed.
        float my = by + 26, mw = (w - 4) / 2f;
        panel.AddChild(new UILabel { X = 12, Y = my, Text = "Move selection", FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary });
        my += 22;
        (string Label, Func<(Vector3 R, Vector3 U, Vector3 F), Vector3> Dir)[] moves =
        {
            ("Left", c => -c.R), ("Right", c => c.R),
            ("Up", c => c.U), ("Down", c => -c.U),
            ("Nearer", c => -c.F), ("Farther", c => c.F),
        };
        for (int m = 0; m < moves.Length; m++)
        {
            var dir = moves[m].Dir;
            panel.AddChild(new UIButton
            {
                X = 12 + (m % 2) * (mw + 4), Y = my + (m / 2) * (h + 4), Width = mw, Height = h,
                Text = moves[m].Label, FontSize = FontSize.Caption,
                OnClick = () => _ = MoveSelectionAsync(dir(CameraAxes())),
            });
        }
        panel.Height = my + 3 * (h + 4) + 8;
        if (_dragStart is { } a && _dragEnd is { } b2) ShowSelectRect(a, b2);
        if (_insertListOpen) BuildInsertList(x + w + 24 + 8, y);
    }

    void SelectRows(int from, int to)
    {
        _selection = SplatEditor.Volume.Rows(from, to);
        _selectedCount = to - from;
        _dragStart = _dragEnd = null;
    }

    /// <summary>The desktop camera's right, world up, and horizontal forward (for moving things).</summary>
    (Vector3 R, Vector3 U, Vector3 F) CameraAxes()
    {
        var cam = _sceneManager.Camera;
        var f = new Vector3(cam.Forward.X, 0, cam.Forward.Z);
        f = f.LengthSquared() < 1e-8f ? cam.Forward : Vector3.Normalize(f);
        var r = Vector3.Normalize(Vector3.Cross(cam.Forward, cam.Up));
        return (r, Vector3.UnitY, f);
    }

    /// <summary>Move the selection one step along <paramref name="direction"/> (scene units, unit length); undoable.</summary>
    async Task MoveSelectionAsync(Vector3 direction, float? step = null)
    {
        if (_editBusy) return;
        var packed = _gpuRenderer.PackedSplatBuffer;
        if (packed == null) return;
        if (_selection is not { } v) { _editNote = "Select something first"; RefreshEditStatus(); return; }
        var offset = direction * (step ?? (_cameraController?.MoveSpeed ?? 1f) * 0.05f);
        _editBusy = true; _editNote = null; RefreshEditStatus();
        try
        {
            await _splatEditor.MoveAsync(_gpuService.WebGPUAccelerator, packed, _gpuRenderer.SplatCount, v, offset);
            _gpuRenderer.SplatsEdited(_sceneManager.Camera.Position);
            _selection = v.MovedBy(offset);
            _dragStart = _dragEnd = null;   // a screen rectangle no longer outlines a region that moved
            Console.WriteLine($"[Edit] moved {_selectedCount:N0} splats by {offset}");
        }
        finally { _editBusy = false; }
        BuildViewerHudUI();
    }

    // ── Insert scene: combine another saved scene into this one ─────────────────────────────────────────────
    bool _insertListOpen;
    List<(Models.Project Project, Models.ProjectScene Scene)> _insertCandidates = new();

    async Task ToggleInsertListAsync()
    {
        _insertListOpen = !_insertListOpen;
        if (_insertListOpen)
        {
            _insertCandidates = new();
            foreach (var p in await _projectService.ListProjectsAsync())
                foreach (var s in p.Scenes)
                    if (s.Id != _viewedProjectScene?.Id) _insertCandidates.Add((p, s));
            _insertCandidates = _insertCandidates.OrderByDescending(c => c.Scene.CreatedAt).Take(10).ToList();
        }
        BuildViewerHudUI();
    }

    void BuildInsertList(float x, float y)
    {
        const float w = 300, h = 34, gap = 6;
        var panel = _uiRoot.AddChild(new UIPanel
        {
            X = x, Y = y, Width = w + 24, Height = Math.Max(1, _insertCandidates.Count) * (h + gap) + 44,
            BackgroundColor = Color.FromArgb(220, 12, 16, 22),
        });
        panel.AddChild(new UILabel { X = 12, Y = 10, Text = "Insert beside this scene:", FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary });
        float by = 34;
        if (_insertCandidates.Count == 0)
            panel.AddChild(new UILabel { X = 12, Y = by, Text = "No other saved scenes", FontSize = FontSize.Caption, Color = UITheme.Current.TextSecondary });
        foreach (var (p, s) in _insertCandidates)
        {
            var (proj, scene) = (p, s);
            string label = $"{proj.Name}: {scene.SplatCount:N0} splats" + (scene.TrainedIterations > 0 ? ", trained" : "") + (scene.EditedFrom != null ? ", edited" : "");
            panel.AddChild(new UIButton
            {
                X = 12, Y = by, Width = w, Height = h, Text = label, FontSize = FontSize.Caption,
                OnClick = () => _ = InsertSceneAsync(proj, scene),
            });
            by += h + gap;
        }
    }

    /// <summary>
    /// Combine: stream another saved scene (and its SH bands) onto the GPU and paste it beside this one - right of it
    /// as seen from the viewer, by both scenes' half widths. Colours are matched to this scene's representation. Undo
    /// removes the inserted scene.
    /// </summary>
    async Task InsertSceneAsync(Models.Project project, Models.ProjectScene scene, System.Numerics.Vector3? sceneRight = null)
    {
        if (_editBusy) return;
        var packedNow = _gpuRenderer.PackedSplatBuffer;
        if (packedNow == null) return;
        _editBusy = true; _editNote = null; _insertListOpen = false; RefreshEditStatus();
        var a = _gpuService.WebGPUAccelerator;
        try
        {
            using var stream = await _projectService.OpenSceneStreamAsync(project.Id, scene.Id);
            if (stream == null) { _editNote = "Could not read that scene"; return; }
            var packed = await _gpuRenderer.LoadPackedFromStreamAsync(stream, scene.SplatCount, scene.EffectiveFloatsPerSplat);
            MemoryBuffer1D<float, Stride1D.Dense>[]? sh = null;
            if (scene.ShDegree > 0 && scene.ShParts == SphericalHarmonics.Parts && _gpuRenderer.ShDegree > 0)
            {
                var bytes = await _projectService.ReadSceneShRestPartsAsync(project.Id, scene.Id, scene.ShParts);
                if (bytes != null)
                {
                    sh = bytes.Select(b => _gpuRenderer.IlgpuFromArrayBuffer(a, b)).ToArray();
                    foreach (var b in bytes) b.Dispose();
                }
            }
            using var clip = await SplatClipboard.FromSceneAsync(a, packed, scene.SplatCount, sh, scene.ShDegree, scene.ColoursAreShDc);
            await clip.MatchColoursAsync(a, _gpuRenderer.ColoursAreShDc);

            // Beside this scene: centre to centre along the viewer's right, by both half widths + 5%.
            var here = await SplatBounds.ComputeRobustAsync(a, packedNow, _gpuRenderer.SplatCount) ?? clip.Bounds;
            var right = sceneRight ?? Vector3.Normalize(Vector3.Cross(_sceneManager.Camera.Forward, _sceneManager.Camera.Up));
            float Half(SplatBounds.Aabb b) => 0.5f * (MathF.Abs(right.X) * (b.MaxX - b.MinX) + MathF.Abs(right.Y) * (b.MaxY - b.MinY) + MathF.Abs(right.Z) * (b.MaxZ - b.MinZ));
            var hereC = new Vector3(here.CentreX, here.CentreY, here.CentreZ);
            var thereC = new Vector3(clip.Bounds.CentreX, clip.Bounds.CentreY, clip.Bounds.CentreZ);
            var offset = hereC - thereC + right * (Half(here) + Half(clip.Bounds)) * 1.05f;

            int before = _gpuRenderer.SplatCount;
            int after = await clip.PasteAsync(a, _gpuRenderer, offset);
            if (_gpuRenderer.PackedSplatBuffer is { } grown)
                await _splatEditor.ResetUndoForPasteAsync(a, grown, after, before);
            if (_sceneManager.ActiveScene != null) _sceneManager.ActiveScene.GpuSplatCount = after;
            SelectRows(before, after);
            _editNote = $"Inserted {scene.SplatCount:N0} splats (selected - Move them)";
            Console.WriteLine($"[Edit] inserted '{project.Name}' scene {scene.Id}: {scene.SplatCount:N0} splats ({before:N0} -> {after:N0})" +
                (sh != null ? $", SH degree {scene.ShDegree}" : "") + $", offset {offset}");
        }
        catch (Exception ex) { _editNote = "Insert failed"; Console.WriteLine($"[Edit] insert failed: {ex.Message}"); }
        finally { _editBusy = false; }
        BuildViewerHudUI();
    }

    string? _editNote;   // a one-off result ("Saved", "No project") shown until the next edit
    SplatClipboard? _clipboard;

    /// <summary>Copy the selection to the clipboard (with its SH colour rows); Cut also deletes it (undoable).</summary>
    async Task CopySelectionAsync(bool cut)
    {
        if (_editBusy) return;
        if (_selection is not { } v) { _editNote = "Select something first"; RefreshEditStatus(); return; }
        _editBusy = true; _editNote = null; RefreshEditStatus();
        try
        {
            var copy = await SplatClipboard.CopyAsync(_gpuService.WebGPUAccelerator, _gpuRenderer, _splatEditor, v);
            if (copy == null) { _editNote = "Nothing visible in the selection"; return; }
            _clipboard?.Dispose();
            _clipboard = copy;
            _editNote = $"{(cut ? "Cut" : "Copied")} {copy.Count:N0} splats";
            Console.WriteLine($"[Edit] {(cut ? "cut" : "copied")} {copy.Count:N0} splats" + (copy.Sh != null ? $" with SH degree {copy.ShDegree}" : ""));
        }
        catch (Exception ex) { _editNote = "Copy failed"; Console.WriteLine($"[Edit] copy failed: {ex.Message}"); }
        finally { _editBusy = false; RefreshEditStatus(); }
        if (cut && _clipboard != null)
        {
            var note = _editNote;
            await ApplyEditAsync(SplatEditor.Mode.DeleteInside);
            _editNote = note; RefreshEditStatus();
        }
    }

    /// <summary>
    /// Paste the clipboard into the scene beside where it was copied from: moved to the right (as seen from the camera)
    /// by its own width plus a little gap. Undo removes the pasted copy.
    /// </summary>
    async Task PasteClipboardAsync(Vector3? sceneRight = null)
    {
        if (_editBusy) return;
        if (_clipboard is not { } clip) { _editNote = "Nothing copied yet"; RefreshEditStatus(); return; }
        _editBusy = true; _editNote = null; RefreshEditStatus();
        try
        {
            // To the viewer's right: the desktop camera's, or the headset's (sceneRight) in XR.
            var right = sceneRight ?? Vector3.Normalize(Vector3.Cross(_sceneManager.Camera.Forward, _sceneManager.Camera.Up));
            var b = clip.Bounds;
            float width = MathF.Abs(right.X) * (b.MaxX - b.MinX) + MathF.Abs(right.Y) * (b.MaxY - b.MinY) + MathF.Abs(right.Z) * (b.MaxZ - b.MinZ);
            // Beside it, but never out of sight: a screen rectangle selects near to far, so a deep selection is wide
            // along the view's right too (a Truck paste landed 2.4 units off-screen); cap at the scene's move speed
            // (about a second's walk) - Move places it from there.
            float cap = _cameraController?.MoveSpeed ?? 1f;
            var offset = right * MathF.Min(width * 1.1f, cap);
            var a = _gpuService.WebGPUAccelerator;
            int before = _gpuRenderer.SplatCount;
            int after = await clip.PasteAsync(a, _gpuRenderer, offset);
            if (_gpuRenderer.PackedSplatBuffer is { } packed)
                await _splatEditor.ResetUndoForPasteAsync(a, packed, after, before);
            if (_sceneManager.ActiveScene != null) _sceneManager.ActiveScene.GpuSplatCount = after;
            SelectRows(before, after);
            _editNote = $"Pasted {clip.Count:N0} splats (selected - Move them)";
            Console.WriteLine($"[Edit] pasted {clip.Count:N0} splats ({before:N0} -> {after:N0}), offset {offset}");
        }
        catch (Exception ex) { _editNote = "Paste failed"; Console.WriteLine($"[Edit] paste failed: {ex.Message}"); }
        finally { _editBusy = false; }
        BuildViewerHudUI();
    }

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
            await SaveViewedSceneToProjectAsync(_viewedProjectScene?.TrainedIterations ?? 0, editedFrom: _viewedProjectScene?.Id,
                dropDeleted: true);
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
        _editBusy = true; _editNote = null; RefreshEditStatus();
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
