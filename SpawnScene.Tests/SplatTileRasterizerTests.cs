using NUnit.Framework;
using SpawnScene.Services;

namespace SpawnScene.Tests;

/// <summary>
/// Does the TILED decomposition compute the same thing as the simple oracle?
///
/// The simple rasteriser is finite-difference verified, so it is the reference. What the tiled
/// version adds is work decomposition - binning, the packed sort key, per-tile ranges, a
/// lockstep backward walk, per-key reduction and scatter - and every one of those can be subtly
/// wrong while still producing a picture that looks fine.
///
/// The WGSL kernels implement exactly this algorithm, so agreement here means the shader port
/// is a transcription rather than a new derivation.
///
/// Run: <c>dotnet test SpawnScene.Tests -c Release --filter SplatTileRasterizer</c>
/// </summary>
public class SplatTileRasterizerTests
{
    const int W = 61, H = 43;   // deliberately NOT multiples of 16: exercises partial edge tiles

    static List<SplatRasterizer.Splat2D> Scene(int n, int seed)
    {
        var rng = new Random(seed);
        var list = new List<SplatRasterizer.Splat2D>();
        for (int i = 0; i < n; i++)
        {
            float sx = 1.5f + (float)rng.NextDouble() * 2.5f;
            float sy = 1.5f + (float)rng.NextDouble() * 2.5f;
            float rot = (float)(rng.NextDouble() * Math.PI);
            float c = MathF.Cos(rot), s = MathF.Sin(rot);
            float a = c * c * sx * sx + s * s * sy * sy;
            float b = c * s * (sx * sx - sy * sy);
            float d = s * s * sx * sx + c * c * sy * sy;
            float det = a * d - b * b;

            list.Add(new SplatRasterizer.Splat2D
            {
                Px = (float)rng.NextDouble() * W,
                Py = (float)rng.NextDouble() * H,
                ConicA = d / det, ConicB = -b / det, ConicC = a / det,
                R = (float)rng.NextDouble(), G = (float)rng.NextDouble(), B = (float)rng.NextDouble(),
                Opacity = 0.1f + (float)rng.NextDouble() * 0.8f,
                Depth = (float)rng.NextDouble(),
            });
        }
        return list;
    }

    /// <summary>The oracle must see splats in the same global depth order the tiles impose.</summary>
    static int[] DepthOrder(List<SplatRasterizer.Splat2D> s) => SplatRasterizer.DepthOrder(s);

    [Test]
    public void Binning_CoversEveryTileASplatTouchesAndNoOthers()
    {
        var splats = Scene(20, 4);
        var b = SplatTileRasterizer.Bin(splats, W, H);

        Assert.That(b.TilesX, Is.EqualTo((W + 15) / 16));
        Assert.That(b.TilesY, Is.EqualTo((H + 15) / 16));

        // Keys must be sorted, and each tile's range must contain exactly its own keys.
        for (int k = 1; k < b.Keys.Length; k++)
            Assert.That(b.Keys[k], Is.GreaterThanOrEqualTo(b.Keys[k - 1]), $"keys unsorted at {k}");

        for (int tile = 0; tile < b.TilesX * b.TilesY; tile++)
        {
            var (start, end) = b.Ranges[tile];
            for (int k = start; k < end; k++)
                Assert.That((int)(b.Keys[k] >> SplatTileRasterizer.DepthBits), Is.EqualTo(tile),
                    $"key {k} in the wrong tile range");
        }

        // Within a tile, depth must be non-decreasing (front to back).
        for (int tile = 0; tile < b.TilesX * b.TilesY; tile++)
        {
            var (start, end) = b.Ranges[tile];
            for (int k = start + 1; k < end; k++)
                Assert.That(splats[b.Values[k]].Depth,
                    Is.GreaterThanOrEqualTo(splats[b.Values[k - 1]].Depth - 1e-4f),
                    $"tile {tile} not front-to-back at {k}");
        }
    }

    [Test]
    public void TiledForward_MatchesTheSimpleRasteriser()
    {
        var splats = Scene(25, 9);
        var b = SplatTileRasterizer.Bin(splats, W, H);
        var (colour, finalT, _) = SplatTileRasterizer.Forward(splats, b);

        var reference = SplatRasterizer.Render(splats, W, H, DepthOrder(splats));

        for (int p = 0; p < W * H; p++)
        {
            for (int c = 0; c < 3; c++)
                Assert.That(colour[p * 3 + c], Is.EqualTo(reference.Colour[p * 3 + c]).Within(1e-4f),
                    $"pixel {p} channel {c}");
            Assert.That(finalT[p], Is.EqualTo(reference.FinalT[p]).Within(1e-4f), $"transmittance {p}");
        }
    }

    [Test]
    public void TiledBackward_MatchesTheSimpleRasteriser()
    {
        // This is the one that matters. It exercises the lockstep walk, the per-key reduction
        // and the scatter all at once, against gradients already proven by finite differences.
        var splats = Scene(25, 13);
        var b = SplatTileRasterizer.Bin(splats, W, H);
        var (colour, finalT, endIdx) = SplatTileRasterizer.Forward(splats, b);

        var rng = new Random(77);
        var target = new float[W * H * 3];
        for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
        var dPix = SplatRasterizer.L1Gradient(colour, target);

        var tiled = SplatTileRasterizer.Backward(splats, b, finalT, endIdx, dPix);

        var order = DepthOrder(splats);
        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var reference = new SplatRasterizer.Grad2D[splats.Count];
        SplatRasterizer.Backward(splats, fwd, dPix, reference);

        AssertGradsMatch(tiled, reference);
    }

    /// <summary>
    /// Compare every component, not just colour and opacity. The screen-position and conic
    /// gradients are what the geometry chain consumes, and they travel the same lockstep walk -
    /// checking only the two that were there first would leave the other five unverified.
    /// </summary>
    static void AssertGradsMatch(SplatRasterizer.Grad2D[] tiled, SplatRasterizer.Grad2D[] reference)
    {
        Assert.That(tiled.Length, Is.EqualTo(reference.Length));
        for (int i = 0; i < reference.Length; i++)
        {
            var t = tiled[i];
            var r = reference[i];
            // Position and conic gradients are far larger than the colour ones (a conic term
            // carries a factor of dx*dy in pixels squared), so the bound scales with the
            // reference rather than being one absolute number for all nine.
            void Same(float got, float want, string what)
            {
                float tol = MathF.Max(2e-5f, 1e-4f * MathF.Abs(want));
                Assert.That(got, Is.EqualTo(want).Within(tol), $"{what} gradient, splat {i}");
            }
            Same(t.R, r.R, "colour R");
            Same(t.G, r.G, "colour G");
            Same(t.B, r.B, "colour B");
            Same(t.Opacity, r.Opacity, "opacity");
            Same(t.Px, r.Px, "screen x");
            Same(t.Py, r.Py, "screen y");
            Same(t.ConicA, r.ConicA, "conic a");
            Same(t.ConicB, r.ConicB, "conic b");
            Same(t.ConicC, r.ConicC, "conic c");
        }
    }

    [Test]
    public void TiledBackward_MatchesWhenPixelsSaturateEarly()
    {
        // The previous test never saturated transmittance, so every pixel consumed its tile's
        // whole list and the lockstep early-out bound was a no-op - removing it still passed.
        // A red-check that cannot fail is not a check, so this fixture stacks many opaque
        // splats to force the forward to stop early, which is the case where a thread must
        // contribute ZERO for splats it never reached and must NOT divide out their alpha.
        var splats = new List<SplatRasterizer.Splat2D>();
        var rng = new Random(101);
        for (int i = 0; i < 60; i++)
        {
            splats.Add(new SplatRasterizer.Splat2D
            {
                Px = 24f + (float)rng.NextDouble() * 6f,
                Py = 20f + (float)rng.NextDouble() * 6f,
                ConicA = 0.04f, ConicB = 0f, ConicC = 0.04f,   // broad, heavily overlapping
                R = (float)rng.NextDouble(), G = (float)rng.NextDouble(), B = (float)rng.NextDouble(),
                Opacity = 0.9f,
                Depth = i * 0.01f,
            });
        }

        var b = SplatTileRasterizer.Bin(splats, W, H);
        var (colour, finalT, endIdx) = SplatTileRasterizer.Forward(splats, b);

        // The fixture must actually saturate somewhere, or this proves nothing.
        bool saturated = false;
        for (int tile = 0; tile < b.TilesX * b.TilesY; tile++)
        {
            var (start, end) = b.Ranges[tile];
            if (end <= start) continue;
            for (int ly = 0; ly < 16 && !saturated; ly++)
                for (int lx = 0; lx < 16 && !saturated; lx++)
                {
                    int px = (tile % b.TilesX) * 16 + lx, py = (tile / b.TilesX) * 16 + ly;
                    if (px >= W || py >= H) continue;
                    if (endIdx[py * W + px] < end) saturated = true;
                }
        }
        Assert.That(saturated, Is.True, "fixture must saturate at least one pixel early");

        var target = new float[W * H * 3];
        for (int i = 0; i < target.Length; i++) target[i] = (float)rng.NextDouble();
        var dPix = SplatRasterizer.L1Gradient(colour, target);

        var tiled = SplatTileRasterizer.Backward(splats, b, finalT, endIdx, dPix);

        var order = DepthOrder(splats);
        var fwd = SplatRasterizer.Render(splats, W, H, order);
        var reference = new SplatRasterizer.Grad2D[splats.Count];
        SplatRasterizer.Backward(splats, fwd, dPix, reference);

        AssertGradsMatch(tiled, reference);
    }

    [Test]
    public void DepthKeysUseTheFullPrecisionOfTheirBitField()
    {
        // The sort key packs depth into 18 bits (262143 levels). Quantising with a FIXED scale
        // (the original depth * 1024) spends only a few hundred of those on a scene whose depth
        // spans 0.4 to 0.8 - so distinct splats collapse onto the same key and their
        // compositing order is decided by allocation order instead of by depth. Normalising
        // against the actual range is what keeps depth ordering meaningful.
        var splats = new List<SplatRasterizer.Splat2D>();
        for (int i = 0; i < 200; i++)
        {
            splats.Add(new SplatRasterizer.Splat2D
            {
                // All in one tile, so every key shares a tile id and only depth distinguishes them.
                Px = 8f, Py = 8f,
                ConicA = 0.5f, ConicB = 0f, ConicC = 0.5f,
                R = 0.5f, G = 0.5f, B = 0.5f, Opacity = 0.05f,
                // A realistic span. TempleRing's object is ~0.26 units across at ~0.5 depth with
                // hundreds of thousands of splats, so neighbouring depths differ by far less
                // than 1/1024. Spreading 200 splats over 0.05 gives 2.5e-4 steps, which a fixed
                // depth*1024 scale cannot resolve at all - they all collapse together.
                Depth = 0.40f + i * (0.05f / 200f),
            });
        }

        var b = SplatTileRasterizer.Bin(splats, W, H);
        int distinct = b.Keys.Distinct().Count();

        Assert.That(distinct, Is.EqualTo(splats.Count),
            $"every distinct depth should get its own key, got {distinct} of {splats.Count}");
    }

    [Test]
    public void SplatsOffScreenOrBehindTheCameraAreBinnedOut()
    {
        var splats = new List<SplatRasterizer.Splat2D>
        {
            // Well off screen.
            new() { Px = -500, Py = -500, ConicA = 0.5f, ConicB = 0, ConicC = 0.5f,
                    R = 1, G = 1, B = 1, Opacity = 0.9f, Depth = 0.1f },
            // Zero opacity.
            new() { Px = 20, Py = 20, ConicA = 0.5f, ConicB = 0, ConicC = 0.5f,
                    R = 1, G = 1, B = 1, Opacity = 0f, Depth = 0.2f },
            // Legitimate.
            new() { Px = 20, Py = 20, ConicA = 0.5f, ConicB = 0, ConicC = 0.5f,
                    R = 1, G = 1, B = 1, Opacity = 0.9f, Depth = 0.3f },
        };
        var b = SplatTileRasterizer.Bin(splats, W, H);

        Assert.That(b.Values, Does.Not.Contain(0), "off-screen splat should not be binned");
        Assert.That(b.Values, Does.Not.Contain(1), "zero-opacity splat should not be binned");
        Assert.That(b.Values, Does.Contain(2), "visible splat must be binned");
    }

    [Test]
    public void EdgeTilesAreHandled()
    {
        // W and H are not multiples of the tile size, so the right and bottom tiles are partial.
        // A kernel that forgets the bounds check writes outside the image or drops those pixels.
        var splats = Scene(15, 31);
        var b = SplatTileRasterizer.Bin(splats, W, H);
        var (colour, _, _) = SplatTileRasterizer.Forward(splats, b);
        var reference = SplatRasterizer.Render(splats, W, H, DepthOrder(splats));

        // Check the last column and last row specifically.
        for (int y = 0; y < H; y++)
        {
            int p = y * W + (W - 1);
            for (int c = 0; c < 3; c++)
                Assert.That(colour[p * 3 + c], Is.EqualTo(reference.Colour[p * 3 + c]).Within(1e-4f),
                    $"right edge pixel row {y} channel {c}");
        }
        for (int x = 0; x < W; x++)
        {
            int p = (H - 1) * W + x;
            for (int c = 0; c < 3; c++)
                Assert.That(colour[p * 3 + c], Is.EqualTo(reference.Colour[p * 3 + c]).Within(1e-4f),
                    $"bottom edge pixel col {x} channel {c}");
        }
    }
}
