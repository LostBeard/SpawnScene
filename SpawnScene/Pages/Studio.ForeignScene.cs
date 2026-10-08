using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Formats;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// "Open scene file" for scenes other tools made, converted on the GPU into a new project as a .spawnscene v1/v2 import
/// is: a 3DGS .ply (the reference trainer, gsplat, nerfstudio, Postshot, Polycam...; GaussianPlyImport), turned y-up by
/// default (a 3DGS .ply is in its SfM frame, y down), or Niantic's .spz (SpzImport; in practice also y down);
/// <c>&amp;sceneup=keep</c> leaves either as it is.
/// </summary>
public partial class Studio
{
    /// <summary>Another tool's scene is turned y-up unless <c>&amp;sceneup=keep</c> (or the older <c>&amp;plyup=keep</c>).</summary>
    static bool TurnImportYUp(Dictionary<string, string> query) =>
        !((query.TryGetValue("sceneup", out var s) && s == "keep") || (query.TryGetValue("plyup", out var p) && p == "keep"));

    /// <summary>A .ply or .spz: imported (or refused with a status) and true; false when it is neither.</summary>
    async Task<bool> ImportForeignBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        if (await ImportPlyBlobAsync(file, name, query)) return true;
        if (await ImportSpzBlobAsync(file, name, query)) return true;
        if (await ImportSogBlobAsync(file, name, query)) return true;
        return await ImportSplatBlobAsync(file, name, query);
    }

    static async Task<byte[]> SliceBytesAsync(Blob file, long start, long end)
    {
        using var slice = file.Slice(start, end);
        using var ab = await slice.ArrayBuffer();
        using var u = new Uint8Array(ab);
        return u.ReadBytes();
    }

    /// <summary>Import PlayCanvas's SOG bundle (a zip of WebP textures and meta.json; Formats/SogMeta, Services/SogImport).
    /// False when the file is not a zip with a meta.json.</summary>
    async Task<bool> ImportSogBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        if (file.Size < 22 || !ZipDirectory.IsZip(await SliceBytesAsync(file, 0, 4))) return false;
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        List<ZipDirectory.Entry> entries;
        try
        {
            long tailStart = Math.Max(0, file.Size - 65557);
            var tail = await SliceBytesAsync(file, tailStart, file.Size);
            var (cdOffset, cdSize, cdCount) = ZipDirectory.FindEnd(tail, file.Size);
            entries = ZipDirectory.Parse(await SliceBytesAsync(file, cdOffset, cdOffset + cdSize), cdCount);
        }
        catch (FormatException) { return false; }
        var byName = entries.GroupBy(e => Path.GetFileName(e.Name)).ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);
        if (!byName.ContainsKey("meta.json")) return false;
        async Task<Blob> Open(string entryName)
        {
            if (!byName.TryGetValue(entryName, out var e)) throw new FormatException($"the SOG has no {entryName}");
            long data = ZipDirectory.DataOffset(await SliceBytesAsync(file, e.LocalHeaderOffset, e.LocalHeaderOffset + 30), e);
            var raw = file.Slice(data, data + e.CompressedSize);
            if (e.Method == 0) return raw;
            if (e.Method != 8) { raw.Dispose(); throw new FormatException($"zip method {e.Method} for {entryName} is not read"); }
            using (raw)
            {
                // CPU transfer: none - inflated by the browser, JS-side.
                using var src = raw.Stream();
                using var inflate = new DecompressionStream("deflate-raw");
                using var piped = src.PipeThrough(inflate);
                using var resp = new Response(piped, (ResponseOptions?)null);
                return await resp.Blob();
            }
        }
        SogMeta meta;
        try
        {
            using var metaBlob = await Open("meta.json");
            meta = SogMeta.Parse(await metaBlob.Text());
        }
        catch (Exception ex) when (ex is FormatException or System.Text.Json.JsonException or KeyNotFoundException)
        {
            _statusMessage = $"Cannot open {name}: {ex.Message}";
            Console.WriteLine($"[Import] {name}: {ex.Message}");
            return true;
        }
        await ImportSogAsync(meta, Open, name, query, t0);
        return true;
    }

    /// <summary>An unbundled SOG: <c>?import=.../meta.json</c>, its textures fetched beside it (PlayCanvas serves them so).</summary>
    async Task ImportSogUrlAsync(string metaUrl, string metaText, Dictionary<string, string> query)
    {
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        SogMeta meta;
        try { meta = SogMeta.Parse(metaText); }
        catch (Exception ex) when (ex is FormatException or System.Text.Json.JsonException or KeyNotFoundException)
        {
            _statusMessage = $"Cannot open {metaUrl}: {ex.Message}";
            Console.WriteLine($"[Import] {metaUrl}: {ex.Message}");
            return;
        }
        var baseUri = new Uri(new Uri(_nav.Uri), metaUrl);
        async Task<Blob> Open(string file)
        {
            using var window = _js.Get<Window>("window");
            using var response = await window.Fetch(new Uri(baseUri, file).ToString());
            if (!response.Ok) throw new FormatException($"{file}: HTTP {response.Status}");
            return await response.Blob();
        }
        string name = baseUri.Segments.Length >= 2 ? baseUri.Segments[^2].TrimEnd('/') : "scene";
        await ImportSogAsync(meta, Open, name, query, t0);
    }

    /// <summary>Decode a SOG (SogImport) and open it as a new project.</summary>
    async Task ImportSogAsync(SogMeta meta, Func<string, Task<Blob>> Open, string name, Dictionary<string, string> query,
        System.Diagnostics.Stopwatch t0)
    {
        bool flip = TurnImportYUp(query);
        using var window = _js.Get<Window>("window");
        var (packed, sh, shDegree) = await SogImport.ConvertAsync(_gpuService.WebGPUAccelerator, window, meta, Open, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the decoded rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)meta.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, (long)meta.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = meta.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true, ShDegree = shDegree, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: SOG v{meta.Version}, {meta.Count:N0} splats, SH degree {shDegree}, decoded in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, shU8, query, "sog");
    }

    /// <summary>Import antimatter15's .splat - no magic, so by name and a whole number of 32-byte splats.</summary>
    async Task<bool> ImportSplatBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        if (!name.EndsWith(".splat", StringComparison.OrdinalIgnoreCase) || file.Size == 0 || file.Size % SplatFileImport.BytesPerSplat != 0)
            return false;
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        bool flip = TurnImportYUp(query);
        // CPU transfer: file I/O. The bytes stay JS-side; the GPU converts them.
        using var whole = await file.ArrayBuffer();
        int n = (int)(whole.ByteLength / SplatFileImport.BytesPerSplat);
        var packed = await SplatFileImport.ConvertAsync(_gpuService.WebGPUAccelerator, whole, flip);
        Uint8Array packedU8;
        try { packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)n * SplatFormat.Floats * sizeof(float)); }
        finally { packed.Dispose(); }
        var scene = new ProjectScene { SplatCount = n, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true, ImportedFrom = name };
        Console.WriteLine($"[Import] {name}: .splat, {n:N0} splats, converted in {t0.Elapsed.TotalSeconds:F1}s" + (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, new List<Uint8Array>(), query, "splat");
        return true;
    }

    /// <summary>Import a .spz (gzip around the NGSP stream); false when the file is not gzip or not SPZ inside.</summary>
    async Task<bool> ImportSpzBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        byte[] magic;
        using (var slice = file.Slice(0, Math.Min(2L, file.Size)))
        using (var mb = await slice.ArrayBuffer())
        using (var u = new Uint8Array(mb))
            magic = u.ReadBytes();
        if (magic.Length < 2) return false;
        if (magic[0] == (byte)'N' && magic[1] == (byte)'G')
        {
            // A raw NGSP header: SPZ v4 (per-attribute zstd streams after a 32-byte header and a table of contents). The
            // browser's DecompressionStream has no zstd (Chrome 151: gzip / deflate only), so it needs a zstd decoder.
            using var hs = file.Slice(0, Math.Min(8L, file.Size));
            using var hb = await hs.ArrayBuffer();
            using var hu = new Uint8Array(hb);
            var h8 = hu.ReadBytes();
            if (h8.Length == 8 && BitConverter.ToUInt32(h8) == SpzImport.Magic)
            {
                int v = (int)BitConverter.ToUInt32(h8, 4);
                _statusMessage = $"Cannot open {name}: SPZ version {v} (zstd-compressed) is not read yet - versions 2 and 3 are";
                Console.WriteLine($"[Import] {name}: SPZ v{v}, zstd streams - not read yet");
                return true;
            }
            return false;
        }
        if (magic[0] != 0x1f || magic[1] != 0x8b) return false;
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        // CPU transfer: none - gunzipped by the browser, JS-side.
        using var raw = await GzipAsync(file, decompress: true);
        byte[] head;
        using (var hu = new Uint8Array(raw, 0, Math.Min(16L, raw.ByteLength)))
            head = hu.ReadBytes();
        if (head.Length < 4 || BitConverter.ToUInt32(head) != SpzImport.Magic) return false;
        SpzImport.Header h;
        try { h = SpzImport.ParseHeader(head); }
        catch (FormatException ex)
        {
            _statusMessage = $"Cannot open {name}: {ex.Message}";
            Console.WriteLine($"[Import] {name}: {ex.Message}");
            return true;
        }
        bool flip = TurnImportYUp(query);
        var (packed, sh) = await SpzImport.ConvertAsync(_gpuService.WebGPUAccelerator, raw, h, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the decoded rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)h.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, (long)h.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = h.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true,
            ShDegree = sh != null ? h.KeptShDegree : 0, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: SPZ v{h.Version}, {h.Count:N0} splats, SH degree {h.ShDegree}" +
            (h.ShDegree > 3 ? " (band 4 dropped)" : "") + (h.Antialiased ? ", antialiased" : "") + $", decoded in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, shU8, query, "spz");
        return true;
    }

    /// <summary>Import a picked or fetched PLY; false (and a status) when it is not one we can read.</summary>
    async Task<bool> ImportPlyBlobAsync(Blob file, string name, Dictionary<string, string> query)
    {
        byte[] head;
        using (var slice = file.Slice(0, Math.Min(65536L, file.Size)))
        using (var hb = await slice.ArrayBuffer())
        using (var u = new Uint8Array(hb))
            head = u.ReadBytes();
        if (!GaussianPly.IsPly(head)) return false;
        if (GaussianPly.ParseCompressed(head) is { } compressed) return await ImportCompressedPlyAsync(file, name, compressed, query);
        GaussianPly.Layout L;
        try { L = GaussianPly.Parse(head); }
        catch (FormatException ex)
        {
            _statusMessage = $"Cannot open {name}: {ex.Message}";
            Console.WriteLine($"[Import] {name}: {ex.Message}");
            return true;
        }
        bool flip = TurnImportYUp(query);
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        // CPU transfer: file I/O. The bytes stay JS-side; the GPU converts them a chunk at a time.
        using var whole = await file.ArrayBuffer();
        var a = _gpuService.WebGPUAccelerator;
        var (packed, sh) = await GaussianPlyImport.ConvertAsync(a, whole, L, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the converted rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, L.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, L.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = (int)L.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true,
            ShDegree = sh != null ? L.ShDegree : 0, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: 3DGS PLY, {L.Count:N0} splats, SH degree {L.ShDegree}, converted in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name), scene, packedU8, shU8, query, "ply");
        return true;
    }

    /// <summary>PlayCanvas's compressed PLY (SuperSplat's export), decoded on the GPU (GaussianPlyImport.ConvertCompressedAsync).</summary>
    async Task<bool> ImportCompressedPlyAsync(Blob file, string name, GaussianPly.CompressedLayout L, Dictionary<string, string> query)
    {
        bool flip = TurnImportYUp(query);
        var t0 = System.Diagnostics.Stopwatch.StartNew();
        // CPU transfer: file I/O. The bytes stay JS-side; the GPU decodes them.
        using var whole = await file.ArrayBuffer();
        var (packed, sh) = await GaussianPlyImport.ConvertCompressedAsync(_gpuService.WebGPUAccelerator, whole, L, flip);
        Uint8Array packedU8;
        var shU8 = new List<Uint8Array>();
        try
        {
            // CPU transfer: file I/O - the decoded rows go to the project store as any saved scene does.
            packedU8 = await packed.CopyToHostUint8ArrayAsync(0, (long)L.Count * SplatFormat.Floats * sizeof(float));
            if (sh != null)
                foreach (var part in sh)
                    shU8.Add(await part.CopyToHostUint8ArrayAsync(0, (long)L.Count * SphericalHarmonics.PartFloatsPerSplat * sizeof(float)));
        }
        finally
        {
            packed.Dispose();
            if (sh != null) foreach (var part in sh) part.Dispose();
        }
        var scene = new ProjectScene
        {
            SplatCount = L.Count, FloatsPerSplat = SplatFormat.Floats, ColoursAreShDc = true,
            ShDegree = sh != null ? L.ShDegree : 0, ImportedFrom = name,
        };
        Console.WriteLine($"[Import] {name}: compressed PLY, {L.Count:N0} splats, SH degree {L.ShDegree}, decoded in {t0.Elapsed.TotalSeconds:F1}s" +
            (flip ? " (turned y-up)" : ""));
        await SaveAndOpenImportedSceneAsync(Path.GetFileNameWithoutExtension(name).Replace(".compressed", ""), scene, packedU8, shU8, query, "compressed ply");
        return true;
    }

    /// <summary>Store an imported scene's rows (and SH parts) in a new project and open it. Disposes the arrays.</summary>
    async Task SaveAndOpenImportedSceneAsync(string sceneName, ProjectScene scene, Uint8Array packedU8, List<Uint8Array> sh,
        Dictionary<string, string> query, string kind)
    {
        var project = await _projectService.CreateProjectAsync(sceneName);
        await _projectService.SaveSceneAsync(project.Id, scene, packedU8);
        if (sh.Count > 0) await _projectService.SaveSceneShRestAsync(project.Id, scene, sh.ToArray());
        packedU8.Dispose();
        foreach (var s in sh) s.Dispose();
        Console.WriteLine($"[Import] '{sceneName}' ({kind}): {scene.SplatCount:N0} splats" + (scene.ShDegree > 0 ? $", SH degree {scene.ShDegree}" : "") + " - opening");

        _projects = await _projectService.ListProjectsAsync();
        var opened = _projects.First(p => p.Id == project.Id);
        OnOpenProject(opened);
        ApplyImportViewerOptions(query);
        await LoadProjectSceneAsync(opened.Scenes.First(s => s.Id == scene.Id));
        await FinishImportAsync(query);
    }
}
