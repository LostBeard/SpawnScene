using System.Collections.Concurrent;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Frames from a video, chosen for reconstruction: <c>count</c> evenly spaced slots across the clip, and in each
/// slot the SHARPEST of a few candidate frames (variance of the Laplacian on luma), so a handheld pan's motion-
/// blurred frames are skipped rather than trained on. Decoding is the browser's (&lt;video&gt; seek + canvas), so
/// any codec the browser plays works; each chosen frame is JPEG-encoded and handed over as bytes, after which a
/// video is just a set of photographs to the rest of the pipeline.
/// </summary>
public sealed class VideoFrameExtractor
{
    readonly SpawnJSRuntime _js;
    bool _installed;

    public VideoFrameExtractor(SpawnJSRuntime js) => _js = js;

    /// <summary>One chosen frame: a file-like name, JPEG bytes, size, its time in the clip, and its sharpness score.</summary>
    public sealed record Frame(string Name, byte[] Jpeg, int Width, int Height, double TimeSeconds, double Sharpness);

    /// <summary>
    /// An object URL for a file the user picked (by name, from any file input on the page), so a video is decoded
    /// straight from the browser's File - never copied into .NET memory. Empty when no such file is picked.
    /// Revoke it with <see cref="RevokeAsync"/>.
    /// </summary>
    public async Task<string> UrlForPickedFileAsync(string fileName)
    {
        await InstallAsync();
        return await _js.CallAsync<string, string>("__svfPickedUrl", fileName);
    }

    public async Task RevokeAsync(string url)
    {
        await InstallAsync();
        await _js.CallVoidAsync("__svfRevoke", url);
    }

    /// <summary>
    /// Extract <paramref name="count"/> frames from <paramref name="url"/> (an http(s) or blob: URL the page can
    /// load). <paramref name="candidatesPerSlot"/> frames are scored per slot; <paramref name="maxDimension"/> caps
    /// the longer side of the output.
    /// </summary>
    public async Task<List<Frame>> ExtractAsync(string url, int count, int candidatesPerSlot = 3,
        int maxDimension = 1600, Action<int, int>? progress = null)
    {
        await InstallAsync();
        int n = await _js.CallAsync<string, int, int, int, int>("__svfExtract", url, Math.Max(1, count),
            Math.Max(1, candidatesPerSlot), Math.Max(64, maxDimension));
        var frames = new List<Frame>(n);
        for (int i = 0; i < n; i++)
        {
            using var bytes = await _js.CallAsync<int, Uint8Array>("__svfBytes", i);
            double t = await _js.CallAsync<int, double>("__svfTime", i);
            double s = await _js.CallAsync<int, double>("__svfSharpness", i);
            int fw = await _js.CallAsync<int, int>("__svfWidth", i);
            int fh = await _js.CallAsync<int, int>("__svfHeight", i);
            frames.Add(new Frame($"frame_{i + 1:D4}.jpg", bytes.ReadBytes(), fw, fh, t, s));
            progress?.Invoke(i + 1, n);
        }
        await _js.CallVoidAsync("__svfClear");
        return frames;
    }

    async Task InstallAsync()
    {
        if (_installed) return;
        await _js.CallVoidAsync("eval", Script);
        _installed = true;
    }

    const string Script = @"
window.__spawnVideoFrames = (function () {
  let frames = [];
  function sharpness(img, w, h) {
    // Variance of the 4-neighbour Laplacian on luma: high for crisp detail, low for motion blur.
    const d = img.data, y = new Float32Array(w * h);
    for (let i = 0, p = 0; i < w * h; i++, p += 4) y[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    let sum = 0, sum2 = 0, n = 0;
    for (let r = 1; r < h - 1; r++) for (let c = 1; c < w - 1; c++) {
      const i = r * w + c;
      const l = y[i - 1] + y[i + 1] + y[i - w] + y[i + w] - 4 * y[i];
      sum += l; sum2 += l * l; n++;
    }
    const m = sum / Math.max(1, n);
    return sum2 / Math.max(1, n) - m * m;
  }
  function seek(v, t) {
    // Draw straight after 'seeked': MEASURED in this Chrome, the element already shows the sought frame then.
    // (Waiting for requestVideoFrameCallback cost ~0.75 s a seek - it never fires for a frame already presented.)
    return new Promise(res => {
      v.addEventListener('seeked', () => res(), { once: true });
      v.currentTime = t;
    });
  }
  function signature(ctx, w, h) {
    const d = ctx.getImageData(0, 0, w, h).data;
    let a = 0;
    for (let i = 0; i < d.length; i += 16) a = (a * 31 + d[i]) >>> 0;
    return a;
  }
  async function extract(url, count, candidates, maxDim) {
    frames = [];
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.preload = 'auto'; v.crossOrigin = 'anonymous';
    // In the document, not detached: Chrome does not present seeked frames for a detached, paused element -
    // every drawImage returned the first frame (MEASURED: 126 identical frames from truck.mp4).
    v.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:-1';
    document.body.appendChild(v);
    // Seek within a blob, not the network: a server without HTTP Range support makes the element unseekable and
    // every seek snaps back to 0 (MEASURED: currentTime stayed 0.000 after seeking to 6.05 s on the dev server).
    // A picked file is already a blob: URL; anything else is fetched once into one.
    let blobUrl = null;
    if (!url.startsWith('blob:')) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('video fetch failed: ' + resp.status + ' ' + url);
      blobUrl = URL.createObjectURL(await resp.blob());
    }
    v.src = blobUrl || url;
    await new Promise((res, rej) => {
      v.addEventListener('loadeddata', res, { once: true });
      v.addEventListener('error', () => rej(new Error('video failed to load: ' + url)), { once: true });
    });
    const dur = v.duration, W = v.videoWidth, H = v.videoHeight;
    if (!(dur > 0) || !(W > 0)) throw new Error('video has no duration or size: ' + url);
    const s = Math.min(1, maxDim / Math.max(W, H));
    const w = Math.max(2, Math.round(W * s)), h = Math.max(2, Math.round(H * s));
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    // Sharpness is judged at <= 480 px on the long side: cheap, and blur shows at any scale.
    const ss = Math.min(1, 480 / Math.max(w, h));
    const sw = Math.max(8, Math.round(w * ss)), sh = Math.max(8, Math.round(h * ss));
    const small = new OffscreenCanvas(sw, sh);
    const sctx = small.getContext('2d', { willReadFrequently: true });
    const eps = 1e-3;
    for (let k = 0; k < count; k++) {
      const t0 = dur * k / count, t1 = dur * (k + 1) / count;
      let bestT = (t0 + t1) / 2, bestS = -1;
      for (let j = 0; j < candidates; j++) {
        const t = Math.min(dur - eps, t0 + (t1 - t0) * (j + 0.5) / candidates);
        await seek(v, t);
        if (t > 0.5 && v.currentTime < t * 0.5)
          throw new Error('video is not seekable here (asked for ' + t.toFixed(2) + ' s, got ' + v.currentTime.toFixed(2) + ' s)');
        sctx.drawImage(v, 0, 0, sw, sh);
        const sc = sharpness(sctx.getImageData(0, 0, sw, sh), sw, sh);
        if (sc > bestS) { bestS = sc; bestT = t; }
      }
      await seek(v, bestT);
      ctx.drawImage(v, 0, 0, w, h);
      sctx.drawImage(v, 0, 0, sw, sh);
      const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.95 });
      const sig = signature(sctx, sw, sh);
      frames.push({ t: bestT, s: bestS, w: w, h: h, sig: sig, bytes: new Uint8Array(await blob.arrayBuffer()) });
    }
    v.removeAttribute('src'); v.load(); v.remove();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    // Identical consecutive frames mean the decoder was not actually seeking - fail loudly rather than hand the
    // pipeline one picture 126 times (every pair matches completely and SfM runs out of memory).
    let same = 0;
    for (let i = 1; i < frames.length; i++) if (frames[i].sig === frames[i - 1].sig) same++;
    if (frames.length > 2 && same > frames.length / 2)
      throw new Error('video frame extraction returned the same picture ' + (same + 1) + ' times - seeking did not update the frame');
    return frames.length;
  }
  window.__svfExtract = extract;
  window.__svfBytes = i => frames[i].bytes;
  window.__svfTime = i => frames[i].t;
  window.__svfSharpness = i => frames[i].s;
  window.__svfClear = () => { frames = []; };
  window.__svfWidth = i => frames[i].w;
  window.__svfHeight = i => frames[i].h;
  window.__svfPickedUrl = name => {
    for (const inp of document.querySelectorAll('input[type=file]'))
      for (const f of (inp.files || [])) if (f.name === name) return URL.createObjectURL(f);
    return '';
  };
  window.__svfRevoke = url => { try { URL.revokeObjectURL(url); } catch (e) { } };
  return true;
})();
";
}

/// <summary>
/// Video frames by URL-like key (<c>video-frame:name</c>), so a frame can be fetched again as a training target
/// long after import, like a dataset photo is by URL. A <c>blob:</c> URL would do the same job, but HttpClient
/// does not accept that scheme.
/// </summary>
public static class VideoFrameStore
{
    public const string Prefix = "video-frame:";
    static readonly ConcurrentDictionary<string, byte[]> Frames = new();

    public static string Put(string name, byte[] jpeg)
    {
        string key = Prefix + name;
        Frames[key] = jpeg;
        return key;
    }

    public static bool TryGet(string key, out byte[] jpeg) => Frames.TryGetValue(key, out jpeg!);

    public static void Clear() => Frames.Clear();
}
