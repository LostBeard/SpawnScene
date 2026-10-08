"""Side-by-side sheets from tools/_cdp_viewer_bench.js screenshots: one image per pose, the viewers in a 2x2 grid.

    python tools/viewer-bench/compose_sheets.py <shotDir> <outDir> [viewer ...]

Every frame is cropped to the same band (below SpawnScene's top bar, above its status box) so no viewer's own UI
covers the comparison, then halved. Each tile is labelled with the viewer only: frame rates belong in a table with every round, since
some viewers' rates swing 2-3x between runs at an unchanged camera (Docs/benchmarks.md).
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

CROP = (0, 60, 1600, 805)  # x0, y0, x1, y1 of a 1600x900 capture


def main():
    shots, out = sys.argv[1], sys.argv[2]
    viewers = sys.argv[3:] or ["spawnscene", "spark", "playcanvas", "gs3d"]
    names = {"spawnscene": "SpawnScene", "spawnscene-all": "SpawnScene (every splat)", "spark": "Spark 2.3.1",
             "playcanvas": "PlayCanvas 2.23.1", "gs3d": "GaussianSplats3D 0.4.7"}
    fps = {}
    for v in viewers:
        with open(os.path.join(shots, f"{v}.json")) as f:
            fps[v] = {r["pose"]: r["fps"] for r in json.load(f)["results"]}
    os.makedirs(out, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tw, th = (CROP[2] - CROP[0]) // 2, (CROP[3] - CROP[1]) // 2
    poses = sorted(fps[viewers[0]])
    for k in poses:
        sheet = Image.new("RGB", (tw * 2, th * 2), (0, 0, 0))
        for i, v in enumerate(viewers):
            im = Image.open(os.path.join(shots, f"{v}_pose{k}.png")).convert("RGB").crop(CROP)
            im = im.resize((tw, th), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            label = names.get(v, v)
            box = d.textbbox((10, 8), label, font=font)
            d.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
            d.text((10, 8), label, fill=(255, 255, 255), font=font)
            sheet.paste(im, ((i % 2) * tw, (i // 2) * th))
        path = os.path.join(out, f"train-pose{k}.jpg")
        sheet.save(path, quality=88)
        print(path)


if __name__ == "__main__":
    main()
