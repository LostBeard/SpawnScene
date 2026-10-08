"""Package a downloaded Commons set (commons_fetch.py output) as a SpawnScene sample: long side <= MAX px, JPEG q88,
the ORIGINAL photo's camera EXIF written back (Commons thumbnails drop it; SpawnScene reads FocalLength /
FocalLengthIn35mmFilm), files 001.jpg.., plus a catalog entry and a harness manifest.
usage: python package_sample.py <download dir> <out dir> <folder name> <display name> <max px> [title prefix filter]"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

UA = {"User-Agent": "SpawnSceneResearch/1.0 (https://spawnscene.com; lostbeard)"}
src, out, folder, display, maxpx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
prefix = sys.argv[6] if len(sys.argv) > 6 else ""
os.makedirs(out, exist_ok=True)


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("rate limited")


credits = json.load(open(os.path.join(src, "credits.json"), encoding="utf-8"))["files"]
credits = [c for c in credits if c["title"].startswith("File:" + prefix)]

# Original EXIF per photo (focal can change shot to shot on a zoom).
meta = {}
for b in range(0, len(credits), 40):
    q = {"action": "query", "format": "json", "titles": "|".join(c["title"] for c in credits[b:b + 40]),
         "prop": "imageinfo", "iiprop": "metadata"}
    for p in get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q))["query"]["pages"].values():
        md = (p.get("imageinfo") or [{}])[0].get("metadata") or []
        meta[p["title"]] = {m["name"]: m["value"] for m in md}
    time.sleep(1)


def rational(v):
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return (int(a), int(b))
    return (int(round(float(v) * 100)), 100)


images, total = [], 0
for i, c in enumerate(credits):
    im = Image.open(os.path.join(src, c["file"])).convert("RGB")
    s = maxpx / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    ex = Image.Exif()
    m = meta.get(c["title"], {})
    if m.get("Make"): ex[0x010F] = m["Make"]
    if m.get("Model"): ex[0x0110] = m["Model"]
    exif_ifd = {}
    if m.get("FocalLength"): exif_ifd[0x920A] = rational(m["FocalLength"])
    if m.get("FocalLengthIn35mmFilm"): exif_ifd[0xA405] = int(m["FocalLengthIn35mmFilm"])
    if exif_ifd:
        ex.get_ifd(0x8769).update(exif_ifd)
    name = f"{i + 1:03d}.jpg"
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True, exif=ex.tobytes())
    data = buf.getvalue()
    open(os.path.join(out, name), "wb").write(data)
    images.append(name)
    total += len(data)

c0 = credits[0]
import re
artist = re.sub(r"<[^>]+>", "", c0.get("artist") or "").strip()
entry = {"name": display, "kind": "set" if len(images) > 1 else "photo", "folder": folder, "images": images,
         "bytes": total, "credit": artist, "license": c0.get("license") or "", "licenseUrl": c0.get("licenseUrl") or "",
         "source": c0["page"] if len(images) == 1 else "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(
             json.load(open(os.path.join(src, "credits.json"), encoding="utf-8"))["category"].replace(" ", "_"))}
json.dump(entry, open(os.path.join(out, "_entry.json"), "w", encoding="utf-8"), indent=1)
json.dump({"name": folder, "source": os.path.abspath(os.path.dirname(out)), "imageDir": os.path.basename(out),
           "images": images}, open(os.path.join(out, "_manifest.json"), "w", encoding="utf-8"), indent=1)
lic = {(c.get("license"), re.sub(r"<[^>]+>", "", c.get("artist") or "").strip()) for c in credits}
print(folder, len(images), "photos", round(total / 1e6, 1), "MB; licenses/artists:", lic)
print("exif sample:", meta.get(c0["title"], {}).get("Model"), meta.get(c0["title"], {}).get("FocalLength"),
      meta.get(c0["title"], {}).get("FocalLengthIn35mmFilm"))
