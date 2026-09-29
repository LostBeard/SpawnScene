using ILGPU;
using ILGPU.Runtime;

namespace SpawnScene.Models;

/// <summary>
/// GPU-resident image: packed RGBA int buffer + dimensions.
/// Used to keep SR output on GPU through depth estimation and Gaussian generation.
/// Caller must Dispose() to release the GPU buffer (unless ownership is transferred).
/// </summary>
public class GpuImage : IDisposable
{
    /// <summary>Packed RGBA as int[W*H] on GPU. Owned by this instance.</summary>
    public MemoryBuffer1D<int, Stride1D.Dense> PackedRgba { get; set; } = default!;

    public int Width { get; set; }
    public int Height { get; set; }
    public string FileName { get; set; } = "";

    public void Dispose()
    {
        PackedRgba?.Dispose();
        PackedRgba = null!;
        GC.SuppressFinalize(this);
    }
}

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

    /// <summary>
    /// The photo's colour at this feature (packed RGBA, R in the low byte; 0 = not sampled), taken at detection from
    /// pixel (round(X), round(Y)). The SfM cloud and the BA sparse-cloud init colour their points from this, so a
    /// GPU-resident import never has to bring the image back to the host to colour a point.
    /// </summary>
    public int PackedColor { get; set; }

    /// <summary><see cref="PackedColor"/> as RGB in [0,1], or null when it was never sampled.</summary>
    public System.Numerics.Vector3? ColorRgb => PackedColor == 0 ? null : new System.Numerics.Vector3(
        (PackedColor & 0xFF) / 255f, ((PackedColor >> 8) & 0xFF) / 255f, ((PackedColor >> 16) & 0xFF) / 255f);
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

    /// <summary>
    /// Where this image can be fetched again, relative to the app base - e.g.
    /// "datasets/Bathroom/IMG_20260223_133436884.jpg". The optimiser re-reads the photograph as
    /// a training target long after the import buffers are freed, and a bare filename is not
    /// enough to find it. Empty for images that came from a file picker rather than a URL.
    /// </summary>
    public string SourceUrl { get; set; } = "";

    /// <summary>Image dimensions.</summary>
    public int Width { get; set; }
    public int Height { get; set; }

    /// <summary>Grayscale pixel data (for feature detection).</summary>
    public byte[] GrayPixels { get; set; } = [];

    /// <summary>RGBA pixel data in MANAGED memory - legacy / CPU paths only. Empty for a GPU-resident import
    /// (<see cref="GpuRgba"/>), which is the only kind the project flow creates.</summary>
    public byte[] RgbaPixels { get; set; } = [];

    /// <summary>
    /// The photo on the device: packed RGBA (R low byte), <see cref="Width"/> x <see cref="Height"/> ints, straight
    /// from <c>MediaInterop.DecodeToDeviceAsync</c> - the pixels never entered the .NET heap. Depth, unprojection and
    /// feature detection read it in place. Owned by the image: call <see cref="DisposeGpu"/> when the run is over.
    /// </summary>
    public MemoryBuffer1D<int, Stride1D.Dense>? GpuRgba { get; set; }

    /// <summary>Release <see cref="GpuRgba"/> (only after every dispatch that reads it has completed).</summary>
    public void DisposeGpu()
    {
        GpuRgba?.Dispose();
        GpuRgba = null;
    }

    /// <summary>
    /// The photo's ENCODED source (JPEG/PNG bytes as a JS Blob - an OPFS File, a fetched response): a few hundred KB in
    /// the browser, never in .NET. When set, <see cref="GpuRgba"/> is a disposable, re-creatable decode of it
    /// (GpuImageOps.EnsureOnDeviceAsync), made where the pixels are used and released after, so neither the managed
    /// heap nor the GPU holds every photo at once (251 x 2.3 MB resident on TruckFull was deliberately avoided,
    /// 2026-09-27). Owned by the image: <see cref="DisposeSource"/>.
    /// </summary>
    public SpawnDev.SpawnJS.JSObjects.Blob? Source { get; set; }

    /// <summary>Longest edge <see cref="Source"/> is decoded at (so every re-decode is the same size).</summary>
    public int DecodeMaxEdge { get; set; }

    /// <summary>Release the encoded source and any device copy. Idempotent.</summary>
    public void DisposeSource()
    {
        DisposeGpu();
        Source?.Dispose();
        Source = null;
    }

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
