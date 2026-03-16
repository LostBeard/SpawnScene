using System.Drawing;
using System.Numerics;

namespace SpawnScene.UI;

/// <summary>
/// Base class for all WebGPU-rendered UI elements.
/// Retained-mode tree: elements have position, size, children, and support hit testing.
/// Coordinates are relative to parent (absolute for root).
/// </summary>
public class UIElement
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
    public bool Visible { get; set; } = true;
    public bool Enabled { get; set; } = true;

    public UIElement? Parent { get; private set; }
    public List<UIElement> Children { get; } = new();

    /// <summary>Absolute screen-space bounds (computed from parent chain).</summary>
    public RectangleF ScreenBounds
    {
        get
        {
            float ax = X, ay = Y;
            var p = Parent;
            while (p != null)
            {
                ax += p.X;
                ay += p.Y;
                p = p.Parent;
            }
            return new RectangleF(ax, ay, Width, Height);
        }
    }

    /// <summary>Add a child element.</summary>
    public T AddChild<T>(T child) where T : UIElement
    {
        child.Parent = this;
        Children.Add(child);
        return child;
    }

    /// <summary>Remove a child element.</summary>
    public void RemoveChild(UIElement child)
    {
        child.Parent = null;
        Children.Remove(child);
    }

    /// <summary>Remove all children.</summary>
    public void ClearChildren()
    {
        foreach (var child in Children)
            child.Parent = null;
        Children.Clear();
    }

    /// <summary>
    /// Hit test: find the deepest visible+enabled element at the given screen position.
    /// Returns null if no element is hit.
    /// </summary>
    public UIElement? HitTest(Vector2 screenPos)
    {
        if (!Visible || !Enabled) return null;

        // Check children in reverse order (front-to-back, last child is on top)
        for (int i = Children.Count - 1; i >= 0; i--)
        {
            var hit = Children[i].HitTest(screenPos);
            if (hit != null) return hit;
        }

        // Check self
        var bounds = ScreenBounds;
        if (screenPos.X >= bounds.X && screenPos.X < bounds.X + bounds.Width &&
            screenPos.Y >= bounds.Y && screenPos.Y < bounds.Y + bounds.Height)
        {
            return this;
        }

        return null;
    }

    /// <summary>Update element state from input. Override in subclasses for interactive behavior.</summary>
    public virtual void Update(InputManager input, float dt)
    {
        if (!Visible || !Enabled) return;
        // Snapshot children to avoid InvalidOperationException if OnClick modifies the tree
        var snapshot = Children.ToArray();
        foreach (var child in snapshot)
            child.Update(input, dt);
    }

    /// <summary>Draw this element and its children. Override in subclasses for custom rendering.</summary>
    public virtual void Draw(UIRenderer renderer)
    {
        if (!Visible) return;
        // Snapshot for same reason as Update
        var snapshot = Children.ToArray();
        foreach (var child in snapshot)
            child.Draw(renderer);
    }
}
