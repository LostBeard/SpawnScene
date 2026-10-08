"""Download a Commons category's photos at a given width (Commons thumbnails) plus a credits.json.
usage: python commons_fetch.py <Category:...> <out dir> <width>"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "SpawnSceneResearch/1.0 (https://spawnscene.com; lostbeard)"}
cat, out, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)


def get(url, binary=False):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read() if binary else json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("rate limited: " + url)


api = "https://commons.wikimedia.org/w/api.php?"
if cat.startswith("@"):   # @titles.txt: named files, kept in the file's order
    titles = [t.strip() for t in open(cat[1:], encoding="utf-8") if t.strip()]
else:
    params = {"action": "query", "format": "json", "list": "categorymembers", "cmtitle": cat, "cmtype": "file", "cmlimit": "500"}
    d = get(api + urllib.parse.urlencode(params))
    titles = sorted(m["title"] for m in d["query"]["categorymembers"])
credits = []
for b in range(0, len(titles), 40):
    q = {"action": "query", "format": "json", "titles": "|".join(titles[b:b + 40]), "prop": "imageinfo",
         "iiprop": "url|size|extmetadata", "iiurlwidth": str(width),
         "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|DateTimeOriginal"}
    pages = get(api + urllib.parse.urlencode(q))["query"]["pages"].values()
    for p in pages:
        ii = p["imageinfo"][0]
        m = ii.get("extmetadata", {})
        credits.append({"title": p["title"], "page": ii["descriptionurl"], "original": ii["url"],
                        "width": ii["width"], "height": ii["height"], "thumb": ii["thumburl"],
                        "license": m.get("LicenseShortName", {}).get("value"),
                        "licenseUrl": m.get("LicenseUrl", {}).get("value"),
                        "artist": m.get("Artist", {}).get("value")})
    time.sleep(1)
order = {t.replace("_", " "): i for i, t in enumerate(titles)}
credits.sort(key=lambda c: order.get(c["title"], 1 << 30))
for i, c in enumerate(credits):
    name = f"{i + 1:03d}.jpg"
    c["file"] = name
    path = os.path.join(out, name)
    if not os.path.exists(path):
        data = get(c["thumb"], binary=True)
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(1.0)
    print(name, c["title"][:80].encode("ascii", "replace").decode(), flush=True)
with open(os.path.join(out, "credits.json"), "w", encoding="utf-8") as f:
    json.dump({"category": cat, "width": width, "files": credits}, f, indent=1)
print("done", len(credits))
