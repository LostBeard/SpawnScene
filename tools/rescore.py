"""One scorer for every tool (parity, 2026-10-08): PSNR, SSIM and LPIPS (AlexNet and VGG) of held-out renders against the
photos, so SpawnScene and gsplat (and Brush) are compared by the SAME code - each tool's own scorer differs (SSIM window,
LPIPS backbone). Run with the gsplat env's python (torch + torchmetrics):

  gsplat:     python rescore.py gsplat <renders dir>                (val_stepN_XXXX.png = photo | render side by side)
  SpawnScene: python rescore.py spawnscene <shots dir> <scene tag prefix> <images dir>
              (<prefix>__held-<photo>-trainer.png, scored against <images dir>/<photo>.*)

Prints one line per image and the mean.
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

dev = "cuda" if torch.cuda.is_available() else "cpu"
psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
lp_alex = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
lp_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)


def t(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().to(dev) / 255.0


def score(gt, im):
    g, r = t(gt), t(im)
    with torch.no_grad():
        return (psnr(r, g).item(), ssim(r, g).item(), lp_alex(r, g).item(), lp_vgg(r, g).item())


def main():
    mode = sys.argv[1]
    pairs = []
    if mode == "gsplat":
        for f in sorted(glob.glob(os.path.join(sys.argv[2], "val_step*_*.png"))):
            a = np.asarray(Image.open(f).convert("RGB"))
            w = a.shape[1] // 2
            pairs.append((os.path.basename(f), a[:, :w], a[:, w:]))
    else:
        shots, prefix, images = sys.argv[2], sys.argv[3], sys.argv[4]
        names = {os.path.splitext(n)[0]: n for n in os.listdir(images)}
        for f in sorted(glob.glob(os.path.join(shots, f"{prefix}__held-*-trainer.png"))):
            stem = os.path.basename(f)[len(prefix) + len("__held-"):-len("-trainer.png")]
            if stem not in names:
                print("no photo for", stem); continue
            gt = Image.open(os.path.join(images, names[stem])).convert("RGB")
            im = Image.open(f).convert("RGB")
            if im.size != gt.size:
                print(f"{stem}: render {im.size} vs photo {gt.size} - resized the photo (check the resolutions!)")
                gt = gt.resize(im.size, Image.BICUBIC)
            pairs.append((stem, np.asarray(gt), np.asarray(im)))
    rows = []
    for name, gt, im in pairs:
        r = score(gt, im)
        rows.append(r)
        print(f"{name}: PSNR {r[0]:.2f} SSIM {r[1]:.4f} LPIPS-alex {r[2]:.3f} LPIPS-vgg {r[3]:.3f}")
    if not rows:
        sys.exit("nothing to score: no renders matched (for SpawnScene: was the run made with &dumpheld=1 on a build that has it?)")
    m = np.mean(rows, 0)
    print(f"MEAN over {len(rows)}: PSNR {m[0]:.2f} SSIM {m[1]:.4f} LPIPS-alex {m[2]:.3f} LPIPS-vgg {m[3]:.3f}")


if __name__ == "__main__":
    main()
