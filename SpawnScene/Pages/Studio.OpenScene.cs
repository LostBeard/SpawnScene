using Microsoft.AspNetCore.Components.Forms;
using SpawnDev.SpawnJS.JSObjects;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// "Open scene file": a .spawnscene from the user's disk. The browser's own File is read by slices (never copied
/// through .NET): a v3 LOD tree streams straight from it - its header now, each chunk when the view needs it - so a
/// file far larger than memory opens at once; v1/v2 are read whole and imported into a new project (Studio.SceneFile).
/// </summary>
public partial class Studio
{
    InputFile? _sceneInput;

    /// <summary>Slots a v3 file opened from disk streams through by default: all of a typical scene, bounded memory for a huge one.</summary>
    const int DefaultLodPoolNodes = 4_000_000;

    void OnOpenSceneFileClicked()
    {
        using var el = _sceneInput!.Element!.Value.As<HTMLElement>();
        el.Click();
    }

    async void OnSceneFileSelected(InputFileChangeEventArgs e)
    {
        try
        {
            using var input = _sceneInput!.Element!.Value.As<HTMLInputElement>();
            using var files = input.Files;
            var file = files?.FirstOrDefault();
            if (file == null) return;
            Console.WriteLine($"[OpenScene] {file.Name}: {file.Size / (1024 * 1024)} MB");
            await OpenSceneBlobAsync(file, file.Name);
        }
        catch (Exception ex) { Console.WriteLine($"[OpenScene] FAIL: {ex.Message}"); }
    }

    /// <summary>Open a .spawnscene held in a Blob (a picked File): v3 streamed by slices, v1/v2 imported.</summary>
    async Task OpenSceneBlobAsync(Blob file, string name)
    {
        byte[] first;
        using (var head = file.Slice(0, 12))
        using (var hb = await head.ArrayBuffer())
        using (var u = new Uint8Array(hb))
            first = u.ReadBytes();
        int version = SceneFile.Version(first);
        var noOptions = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (version == 0)
        {
            // Another tool's scene: a 3DGS .ply or an .spz (Studio.ForeignScene).
            if (!await ImportForeignBlobAsync(file, name, noOptions))
            {
                _statusMessage = $"{name} is not a scene SpawnScene can open (.spawnscene, a 3DGS .ply or .spz)";
                Console.WriteLine($"[OpenScene] {name} is not a .spawnscene, a 3DGS .ply or an .spz");
            }
            file.Dispose();
            return;
        }
        if (version != 3)
        {
            using var whole = await file.ArrayBuffer();
            file.Dispose();
            await ImportSceneBytesAsync(whole, noOptions);
            return;
        }
        int headerLen = SceneFile.HeaderLength(first);
        LodChunkFile.Header3 h;
        using (var hs = file.Slice(12, 12 + headerLen))
        using (var hb = await hs.ArrayBuffer())
        using (var u = new Uint8Array(hb))
            h = LodChunkFile.ParseHeader3(u.ReadBytes());
        long dataStart = SceneFile.DataOffset(headerLen);
        // The File stays open while the scene streams from it (replaces whatever the last open held).
        _lodFileBlob?.Dispose();
        _lodFileBlob = file;
        async Task<ArrayBuffer> ChunkBytes(LodChunkFile.Chunk c)
        {
            using var slice = file.Slice(dataStart + c.Offset, dataStart + c.Offset + c.Bytes);
            return await GzipAsync(slice, decompress: true);
        }
        await OpenLodStreamAsync(h, ChunkBytes, "file", DefaultLodPoolNodes);
    }
}
