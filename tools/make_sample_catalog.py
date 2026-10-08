"""Write SpawnScene's wwwroot/samples/catalog.json from the packaged samples' _entry.json files.
usage: python make_catalog.py <hf dir> <catalog.json path>"""
import json
import os
import sys

hf, out = sys.argv[1], sys.argv[2]
# Through the hub's /src proxy: shipped code never requests huggingface.co directly (TJ's standing rule - the hub
# caches, answers CORS and keeps us out of HF's rate limiter). /hf only parses model repos (org/repo).
BASE = ("https://hub.spawndev.com:44365/src?url="
        "https://huggingface.co/datasets/LostBeard/spawnscene-samples/resolve/main/")
# Display order and cleaned credits (the Commons artist field is HTML with user handles).
ORDER = [
    ("hamamni-baths", "Nassima Chahboun"),
    ("pinecone", "NELAC, University of São Paulo"),
    # korno-rock (CC0, Zbytovsky) stays on HF but out of the list: s3 2026-10-07 - 106 MB, 25 min to train, a 594 MB
    # scene of a plain rock face; not a demo.
    ("kitchen", "NeONBRAND (Unsplash)"),
    ("living-room", "Jarosław Ceborski (Unsplash)"),
    ("castle-room", "Daderot"),
    ("garden-path", "Daderot"),
    ("tivoli-garden", "Daderot"),
]
samples = []
for folder, credit in ORDER:
    p = os.path.join(hf, folder, "_entry.json")
    if not os.path.exists(p):
        continue
    e = json.load(open(p, encoding="utf-8"))
    e["credit"] = credit
    e["source"] = e["source"].replace("%3A", ":")
    samples.append(e)
json.dump({"base": BASE, "samples": samples}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(out, len(samples), "samples")
