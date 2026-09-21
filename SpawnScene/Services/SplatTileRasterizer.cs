namespace SpawnScene.Services;

/// <summary>
/// CPU model of the TILED algorithm the WGSL kernels implement.
///
/// <see cref="SplatRasterizer"/> already proves the maths (finite-difference verified). This
/// proves the GPU's work decomposition on top of it: tile binning, the packed (tile, depth)
/// sort key, per-tile ranges, the lockstep back-to-front backward walk, per-key gradient
/// reduction, and the scatter into per-splat totals.
///
/// Those are the parts most likely to be subtly wrong, and every one of them is invisible in a
/// rendered image until it is badly wrong. Checking them here against the non-tiled oracle costs
/// milliseconds; checking them on the GPU costs a publish, a browser run, and a guess.
///
/// Gate: <c>SpawnScene.Tests/SplatTileRasterizerTests.cs</c>.
/// </summary>
public static class SplatTileRasterizer
{
    public const int TileSize = 16;

    /// <summary>Bits of the sort key given to quantised depth; the rest hold the tile id.</summary>
    public const int DepthBits = 18;
    const uint DepthMax = (1u << DepthBits) - 1u;

    /// <summary>Screen-space footprint of a splat, in tiles.</summary>
    public readonly record struct TileSpan(int X0, int Y0, int X1, int Y1)
    {
        public int Count => X1 < X0 || Y1 < Y0 ? 0 : (X1 - X0 + 1) * (Y1 - Y0 + 1);
    }

    /// <summary>
    /// Tiles a splat's 3-sigma footprint overlaps, clamped to the screen.
    /// Mirrors the bounds arithmetic in the count_tiles / emit_keys kernels.
    /// </summary>
    public static TileSpan Span(in SplatRasterizer.Splat2D s, float extent, int tilesX, int tilesY)
    {
        int x0 = (int)MathF.Floor((s.Px - extent) / TileSize);
        int y0 = (int)MathF.Floor((s.Py - extent) / TileSize);
        int x1 = (int)MathF.Floor((s.Px + extent) / TileSize);
        int y1 = (int)MathF.Floor((s.Py + extent) / TileSize);
        return new TileSpan(
            Math.Max(x0, 0), Math.Max(y0, 0),
            Math.Min(x1, tilesX - 1), Math.Min(y1, tilesY - 1));
    }

    /// <summary>3-sigma half-extent from the conic (inverse covariance).</summary>
    public static float Extent(in SplatRasterizer.Splat2D s)
    {
        // Invert the conic back to a covariance to get its major eigenvalue.
        float det = s.ConicA * s.ConicC - s.ConicB * s.ConicB;
        if (!(det > 1e-20f)) return 0f;
        float a = s.ConicC / det, c = s.ConicA / det;
        float mid = 0.5f * (a + c);
        float b = -s.ConicB / det;
        float disc = MathF.Sqrt(MathF.Max(mid * mid - (a * c - b * b), 0f));
        return 3f * MathF.Sqrt(MathF.Max(mid + disc, 1e-20f));
    }

    /// <summary>Everything the tiled forward produced, per pixel and per key.</summary>
    public sealed class Binned
    {
        public required int Width { get; init; }
        public required int Height { get; init; }
        public required int TilesX { get; init; }
        public required int TilesY { get; init; }
        /// <summary>Sorted (tile, depth) keys.</summary>
        public required uint[] Keys { get; init; }
        /// <summary>Splat index per key, in the same order.</summary>
        public required int[] Values { get; init; }
        /// <summary>[start, end) into Keys for each tile.</summary>
        public required (int start, int end)[] Ranges { get; init; }
    }

    /// <summary>
    /// Bin, key and sort. One ascending sort groups by tile AND orders front-to-back inside
    /// each tile, because the tile id occupies the high bits of the key.
    /// </summary>
    public static Binned Bin(IReadOnlyList<SplatRasterizer.Splat2D> splats, int width, int height)
    {
        // Normalise depth into the full key range. A fixed scale wastes most of the 18 bits on
        // a scene with a narrow depth span and collapses distinct splats onto the same key,
        // where their compositing order then depends on allocation order rather than depth.
        float near = float.MaxValue, far = float.MinValue;
        foreach (var sp in splats)
        {
            if (!(sp.Opacity > 0f)) continue;
            near = MathF.Min(near, sp.Depth);
            far = MathF.Max(far, sp.Depth);
        }
        float depthSpan = MathF.Max(far - near, 1e-6f);

        int tilesX = (width + TileSize - 1) / TileSize;
        int tilesY = (height + TileSize - 1) / TileSize;

        var keys = new List<uint>();
        var values = new List<int>();

        for (int i = 0; i < splats.Count; i++)
        {
            var s = splats[i];
            if (!(s.Opacity > 0f)) continue;
            float ext = Extent(in s);
            if (ext <= 0f) continue;

            var span = Span(in s, ext, tilesX, tilesY);
            if (span.Count == 0) continue;

            uint dq = (uint)(Math.Clamp((s.Depth - near) / depthSpan, 0f, 1f) * DepthMax);
            for (int ty = span.Y0; ty <= span.Y1; ty++)
            {
                for (int tx = span.X0; tx <= span.X1; tx++)
                {
                    uint tile = (uint)(ty * tilesX + tx);
                    keys.Add((tile << DepthBits) | dq);
                    values.Add(i);
                }
            }
        }

        // Stable sort by key so equal depths keep a deterministic order, matching what a stable
        // radix sort gives on the GPU.
        var idx = Enumerable.Range(0, keys.Count).ToArray();
        var keyArr = keys.ToArray();
        var valArr = values.ToArray();
        Array.Sort(idx, (a, b) => keyArr[a] != keyArr[b] ? keyArr[a].CompareTo(keyArr[b]) : a.CompareTo(b));

        var sortedKeys = new uint[idx.Length];
        var sortedVals = new int[idx.Length];
        for (int k = 0; k < idx.Length; k++) { sortedKeys[k] = keyArr[idx[k]]; sortedVals[k] = valArr[idx[k]]; }

        var ranges = new (int, int)[tilesX * tilesY];
        for (int k = 0; k < sortedKeys.Length; k++)
        {
            int tile = (int)(sortedKeys[k] >> DepthBits);
            if (k == 0 || (int)(sortedKeys[k - 1] >> DepthBits) != tile) ranges[tile].Item1 = k;
            if (k == sortedKeys.Length - 1 || (int)(sortedKeys[k + 1] >> DepthBits) != tile) ranges[tile].Item2 = k + 1;
        }

        return new Binned
        {
            Width = width, Height = height, TilesX = tilesX, TilesY = tilesY,
            Keys = sortedKeys, Values = sortedVals, Ranges = ranges,
        };
    }

    /// <summary>Tiled forward. Per-pixel colour, final transmittance, and exclusive key end.</summary>
    public static (float[] colour, float[] finalT, int[] endIdx) Forward(
        IReadOnlyList<SplatRasterizer.Splat2D> splats, Binned b)
    {
        var colour = new float[b.Width * b.Height * 3];
        var finalT = new float[b.Width * b.Height];
        var endIdx = new int[b.Width * b.Height];

        for (int ty = 0; ty < b.TilesY; ty++)
        {
            for (int tx = 0; tx < b.TilesX; tx++)
            {
                var (start, end) = b.Ranges[ty * b.TilesX + tx];

                for (int ly = 0; ly < TileSize; ly++)
                {
                    int py = ty * TileSize + ly;
                    if (py >= b.Height) continue;
                    for (int lx = 0; lx < TileSize; lx++)
                    {
                        int px = tx * TileSize + lx;
                        if (px >= b.Width) continue;

                        int p = py * b.Width + px;
                        float cx = px + 0.5f, cy = py + 0.5f;
                        float t = 1f;
                        float accR = 0, accG = 0, accB = 0;
                        int consumed = 0;

                        for (int k = start; k < end; k++)
                        {
                            if (t < SplatRasterizer.MinTransmittance) break;
                            var s = splats[b.Values[k]];
                            consumed++;

                            float g = SplatRasterizer.Weight(in s, cx, cy);
                            if (g <= 0f) continue;
                            float alpha = MathF.Min(SplatRasterizer.MaxAlpha, s.Opacity * g);
                            if (alpha < SplatRasterizer.MinAlpha) continue;

                            accR += s.R * alpha * t;
                            accG += s.G * alpha * t;
                            accB += s.B * alpha * t;
                            t *= (1f - alpha);
                        }

                        colour[p * 3 + 0] = accR;
                        colour[p * 3 + 1] = accG;
                        colour[p * 3 + 2] = accB;
                        finalT[p] = t;
                        endIdx[p] = start + consumed;
                    }
                }
            }
        }

        return (colour, finalT, endIdx);
    }

    /// <summary>
    /// Tiled backward, modelling the GPU's lockstep walk exactly: every pixel in a tile visits
    /// every splat in that tile's list back-to-front, contributing zero where it did not reach.
    /// Gradients are reduced per (tile, splat) into grad_per_key, then scattered per splat.
    /// </summary>
    /// <summary>Gradient slots carried per (tile, splat) key. Must match GRADS_PER_KEY in WGSL.</summary>
    public const int GradsPerKey = 9;

    public static SplatRasterizer.Grad2D[] Backward(
        IReadOnlyList<SplatRasterizer.Splat2D> splats, Binned b,
        float[] finalT, int[] endIdx, float[] dLdPixel)
    {
        var gradPerKey = new float[b.Keys.Length * GradsPerKey];

        for (int ty = 0; ty < b.TilesY; ty++)
        {
            for (int tx = 0; tx < b.TilesX; tx++)
            {
                var (start, end) = b.Ranges[ty * b.TilesX + tx];
                if (end <= start) continue;

                int threads = TileSize * TileSize;
                var tState = new float[threads];
                var recR = new float[threads];
                var recG = new float[threads];
                var recB = new float[threads];
                var myEnd = new int[threads];
                var dR = new float[threads];
                var dG = new float[threads];
                var dB = new float[threads];
                var live = new bool[threads];

                for (int li = 0; li < threads; li++)
                {
                    int px = tx * TileSize + (li % TileSize);
                    int py = ty * TileSize + (li / TileSize);
                    if (px >= b.Width || py >= b.Height) { live[li] = false; continue; }
                    live[li] = true;
                    int p = py * b.Width + px;
                    tState[li] = finalT[p];
                    myEnd[li] = endIdx[p];
                    dR[li] = dLdPixel[p * 3 + 0];
                    dG[li] = dLdPixel[p * 3 + 1];
                    dB[li] = dLdPixel[p * 3 + 2];
                }

                // Lockstep, back to front.
                for (int k = end - 1; k >= start; k--)
                {
                    var s = splats[b.Values[k]];
                    float sumR = 0, sumG = 0, sumB = 0, sumO = 0;
                    float sumPx = 0, sumPy = 0, sumCa = 0, sumCb = 0, sumCc = 0;

                    for (int li = 0; li < threads; li++)
                    {
                        if (!live[li] || k >= myEnd[li]) continue;

                        int px = tx * TileSize + (li % TileSize);
                        int py = ty * TileSize + (li / TileSize);
                        float cx = px + 0.5f, cy = py + 0.5f;

                        float g = SplatRasterizer.Weight(in s, cx, cy);
                        if (g <= 0f) continue;
                        float rawAlpha = s.Opacity * g;
                        float alpha = MathF.Min(SplatRasterizer.MaxAlpha, rawAlpha);
                        if (alpha < SplatRasterizer.MinAlpha) continue;

                        tState[li] /= (1f - alpha);
                        float t = tState[li];
                        float w = alpha * t;

                        sumR += w * dR[li];
                        sumG += w * dG[li];
                        sumB += w * dB[li];

                        float dLdAlpha =
                            (s.R - recR[li]) * t * dR[li] +
                            (s.G - recG[li]) * t * dG[li] +
                            (s.B - recB[li]) * t * dB[li];

                        if (rawAlpha < SplatRasterizer.MaxAlpha)
                        {
                            sumO += g * dLdAlpha;

                            // Geometry, at the 2D level. The chain on to position, scale and
                            // rotation is linear and depends only on the splat and the view, so
                            // it is applied ONCE PER SPLAT after the scatter rather than per
                            // pixel or per key - see SplatGeometryGradients.
                            float dx = cx - s.Px;
                            float dy = cy - s.Py;
                            float dLdPower = s.Opacity * dLdAlpha * g;
                            sumPx += dLdPower * (s.ConicA * dx + s.ConicB * dy);
                            sumPy += dLdPower * (s.ConicC * dy + s.ConicB * dx);
                            sumCa += dLdPower * (-0.5f * dx * dx);
                            sumCb += dLdPower * (-dx * dy);
                            sumCc += dLdPower * (-0.5f * dy * dy);
                        }

                        recR[li] = alpha * s.R + (1f - alpha) * recR[li];
                        recG[li] = alpha * s.G + (1f - alpha) * recG[li];
                        recB[li] = alpha * s.B + (1f - alpha) * recB[li];
                    }

                    int gk = k * GradsPerKey;
                    gradPerKey[gk + 0] = sumR;
                    gradPerKey[gk + 1] = sumG;
                    gradPerKey[gk + 2] = sumB;
                    gradPerKey[gk + 3] = sumO;
                    gradPerKey[gk + 4] = sumPx;
                    gradPerKey[gk + 5] = sumPy;
                    gradPerKey[gk + 6] = sumCa;
                    gradPerKey[gk + 7] = sumCb;
                    gradPerKey[gk + 8] = sumCc;
                }
            }
        }

        // Scatter: fold per-key gradients into per-splat totals.
        var grads = new SplatRasterizer.Grad2D[splats.Count];
        for (int k = 0; k < b.Keys.Length; k++)
        {
            int i = b.Values[k];
            int gk = k * GradsPerKey;
            grads[i].R += gradPerKey[gk + 0];
            grads[i].G += gradPerKey[gk + 1];
            grads[i].B += gradPerKey[gk + 2];
            grads[i].Opacity += gradPerKey[gk + 3];
            grads[i].Px += gradPerKey[gk + 4];
            grads[i].Py += gradPerKey[gk + 5];
            grads[i].ConicA += gradPerKey[gk + 6];
            grads[i].ConicB += gradPerKey[gk + 7];
            grads[i].ConicC += gradPerKey[gk + 8];
        }
        return grads;
    }
}
