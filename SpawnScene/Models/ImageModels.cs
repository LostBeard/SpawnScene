namespace SpawnScene.Models;

/// <summary>
/// A detected feature (keypoint) in an image.
/// Contains position, scale, orientation, and a binary descriptor.
/// </summary>
public class ImageFeature
{
    /// <summary>X position in pixels.</summary>
    public float X { get; set; }

    /// <summary>Y position in pixels.</summary>
    public float Y { get; set; }

    /// <summary>Corner response score (higher = stronger).</summary>
    public float Score { get; set; }

    /// <summary>Feature scale (octave level).</summary>
    public int Octave { get; set; }

    /// <summary>
    /// Binary descriptor (256-bit = 32 bytes).
    /// Used for Hamming distance matching.
    /// </summary>
    public byte[] Descriptor { get; set; } = new byte[32];
}

/// <summary>
/// A match between two features in different images.
/// </summary>
public class FeatureMatch
{
    /// <summary>Index into image A's feature list.</summary>
    public int IndexA { get; set; }

    /// <summary>Index into image B's feature list.</summary>
    public int IndexB { get; set; }

    /// <summary>Hamming distance between descriptors (lower = better).</summary>
    public int Distance { get; set; }
}

/// <summary>
/// An imported image with metadata and detected features.
/// </summary>
public class ImportedImage
{
    /// <summary>Original filename.</summary>
    public string FileName { get; set; } = "";

    /// <summary>Image dimensions.</summary>
    public int Width { get; set; }
    public int Height { get; set; }

    /// <summary>Grayscale pixel data (for feature detection).</summary>
    public byte[] GrayPixels { get; set; } = [];

    /// <summary>RGBA pixel data (for display).</summary>
    public byte[] RgbaPixels { get; set; } = [];

    /// <summary>Resolution used for feature detection (may be downsampled).</summary>
    public int FeatureWidth { get; set; }
    public int FeatureHeight { get; set; }

    /// <summary>Object URL for displaying the image in the browser.</summary>
    public string? ObjectUrl { get; set; }

    /// <summary>Detected features (keypoints + descriptors).</summary>
    public List<ImageFeature> Features { get; set; } = [];

    /// <summary>Whether features have been detected.</summary>
    public bool HasFeatures => Features.Count > 0;

    /// <summary>Estimated camera parameters (from SfM).</summary>
    public CameraParams? EstimatedCamera { get; set; }
}

/// <summary>
/// An image pair with matched features.
/// </summary>
public class ImagePair
{
    public int ImageIndexA { get; set; }
    public int ImageIndexB { get; set; }
    public List<FeatureMatch> Matches { get; set; } = [];
    public int InlierCount { get; set; }
}
