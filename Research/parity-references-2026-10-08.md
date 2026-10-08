# Parity references: published numbers and how to run the other trainers (2026-10-08)

Research only: nothing here was run on the GPU. Every number has a source and was read from it on 2026-10-08.
"Derived" marks arithmetic or reasoning of mine, not a published figure. Companion tracker: [parity-matrix.md](parity-matrix.md).

## 0. Gotchas first: they move numbers more than the gaps we are measuring

1. **Each tool scores differently.** 3DGS `metrics.py`: own SSIM plus `lpipsPyTorch` VGG. gsplat `simple_trainer.py`: torchmetrics
   PSNR/SSIM, **LPIPS AlexNet by default** (`lpips_net="alex"`). gsplat's docs say this "differs from the original paper".
   Splatfacto: torchmetrics PSNR, `pytorch_msssim` SSIM, torchmetrics LPIPS with its default net (alex). Brush: its own SSIM
   (11-tap Gaussian, sigma 1.5, C1=0.01², C2=0.03², zero padding), renders rounded to 8 bit, and **no LPIPS**. nvs-bench:
   torchmetrics everywhere, LPIPS **VGG**. AlexNet and VGG LPIPS cannot be compared with each other.
2. **The same "images_4" is not the same pixels.** 3DGS `full_eval.py` trains on the JPEGs that ship in the dataset, with
   `-i images_4` for outdoor and `-i images_2` for indoor scenes. gsplat (`examples/datasets/colmap.py`, main and 1.5.3) checks
   the `images_N` folder: if it holds `.jpg` files, gsplat **ignores them**. It re-downsamples full-res `images/` with PIL
   **BICUBIC** into `images_N_png/`, which it writes into the data dir. Brush resolves each COLMAP name to the lexicographically
   first matching path (`find_image_by_name(...).min()`), so on a stock Mip-NeRF 360 folder it picks **`images/` (full-res)**
   and caps the long edge at `--max-resolution` (default 1920) with Lanczos3. That is bicycle at about 1920 px, not 1237.
   Splatfacto uses `images_N` if it exists; otherwise it asks to make it with ffmpeg. Its auto factor keeps the long side
   at 1600 px or less.
3. **Split ordering.** 3DGS (`dataset_readers.py`), gsplat (`colmap.py`), Brush (`colmap.rs`) and nvs-bench (`evaluate.py`)
   all sort by **file name** and test on `idx % 8 == 0`. nerfstudio 1.1.5's COLMAP parser orders by **COLMAP image_id**
   (`ordered_im_id = sorted(im_id_to_image.keys())`), then uses the same `% 8`. Its test set matches the others only if the
   ids follow name order, so dump its eval file list and check.
4. **What "7K" means differs by tool.**
   - 3DGS: `position_lr_max_steps=30_000` is separate from `--iterations`, and `--save/--test_iterations` default to
     `[7000, 30000]`. So the paper's "Ours-7K" is the step-7000 checkpoint of the 30K run. For this code, a 7K-only run uses
     the same schedule.
   - gsplat: `--max_steps 7000` changes **only** the means LR decay (`ExponentialLR(gamma=0.01**(1/max_steps))`, plus the
     pose and bilagrid LRs). DefaultStrategy steps are absolute: refine 500..15000 every 100, opacity reset every 3000, SH
     degree up every 1000. So a 7K run still densifies to the end and resets opacity at 3000 and 6000. `--steps_scaler 0.2333`
     scales all of these instead (`Config.adjust_steps`). gsplat's own published 7K numbers are step 7000 of the 30K run:
     `benchmarks/basic.sh` evaluates the 7k and 30k checkpoints of one run.
   - Splatfacto: the means scheduler has a fixed `max_steps=30000` in `method_configs.py`. Splatfacto also **trains at 1/4
     resolution until step 3000 and 1/2 until 6000** (`num_downscales=2`, `resolution_schedule=3000`).
   - Brush: the mean (and in v0.3.0 the scale) LR decays over `--total-steps`. Growth stops at 15000; on main it is clamped to
     the total. So `--total-steps 7000` is a real 7K schedule.
5. **tandt_db intrinsics vs pixels (read locally from `C:\Users\TJ\Downloads\tandt_db`).** `cameras.bin` says truck is
   1957x1091 and train is 1959x1090, but the JPEGs are 979x546 and 980x545. DrJohnson (1332x876) and Playroom (1264x832)
   match. 3DGS and Brush use FoV plus a normalised principal point, so they are unaffected. gsplat rescales K. nerfstudio
   **asserts** that image size equals camera size (`full_images_datamanager.py:207`). Derived workaround, not run: put the
   images under `images_2` and pass `--downscale-factor 2 --downscale-rounding-mode ceil`, since ceil(1957/2)=979 and
   ceil(1091/2)=546. The same floor-vs-ceil trap applies to Mip-NeRF 360 `images_4` if the full-res width is odd.
6. **The 3DGS repo is not the paper.** README: "the code base has been cleaned up and includes bugfixes, hence the metrics you
   get from evaluating them will differ from those in the paper." gsplat's re-run of Inria gives 28.95 dB on 7 scenes; the
   paper's 7-scene mean is 28.69 (derived from Table 5). That 0.26 dB is noise from code version alone.
7. **7 scenes vs 9 scenes.** gsplat tables average 7 Mip-NeRF 360 scenes (no flowers or treehill). The paper's 27.21 dB
   average is over 9. Compare per scene, or use the 7-scene means derived in section 1.
8. **Backgrounds.** Brush trains with a noisy background (`background_noise_strength` 0.1 on main) and evaluates on black.
   gsplat `random_bkgd=False` uses black. Splatfacto defaults to a `random` background. These are opaque scenes, so this is
   probably minor, but write it down.

## 1. 3DGS paper (Kerbl et al. 2023), per scene

Source: arXiv 2308.04079, https://arxiv.org/html/2308.04079. Appendix Tables 4/5/6 give Mip-NeRF 360 SSIM, PSNR and LPIPS;
Tables 7/8/9 give the same for Tanks&Temples and Deep Blending. Protocol: test = "every 8th photo"; A6000. Resolution comes
from `full_eval.py`: Mip-NeRF 360 outdoor uses `-i images_4`, indoor `-i images_2`. T&T and DB use the shipped `images/`
(no `-r`; 3DGS auto-rescales only images wider than 1600 px, and none of these are). LPIPS is VGG (`lpipsPyTorch`). The paper
gives **no per-scene Gaussian counts** (text: "1-5 million Gaussians") and **no per-scene times**.

| Scene | res used | 7K PSNR | 7K SSIM | 7K LPIPS | 30K PSNR | 30K SSIM | 30K LPIPS |
|---|---|---|---|---|---|---|---|
| bicycle | images_4 | 23.604 | 0.675 | 0.318 | 25.246 | 0.771 | 0.205 |
| garden | images_4 | 26.245 | 0.836 | 0.153 | 27.410 | 0.868 | 0.103 |
| stump | images_4 | 25.709 | 0.728 | 0.287 | 26.550 | 0.775 | 0.210 |
| room | images_2 | 28.139 | 0.884 | 0.272 | 30.632 | 0.914 | 0.220 |
| counter | images_2 | 26.705 | 0.873 | 0.254 | 28.700 | 0.905 | 0.204 |
| kitchen | images_2 | 28.546 | 0.900 | 0.161 | 30.317 | 0.922 | 0.129 |
| bonsai | images_2 | 28.850 | 0.910 | 0.244 | 31.980 | 0.938 | 0.205 |
| truck | 979x546 | 23.506 | 0.840 | 0.209 | 25.187 | 0.879 | 0.148 |
| train | 980x545 | 18.892 | 0.694 | 0.350 | 21.097 | 0.802 | 0.218 |
| drjohnson | 1332x876 | 26.306 | 0.853 | 0.343 | 28.766 | 0.899 | 0.244 |
| playroom | 1264x832 | 29.245 | 0.896 | 0.291 | 30.044 | 0.906 | 0.241 |

Table 1 averages (train time and model size, A6000):

| Dataset | 7K | 30K |
|---|---|---|
| Mip-NeRF 360 (9 scenes) | 25.60 / 0.770 / 0.279, 6m25s, 523 MB | 27.21 / 0.815 / 0.214, 41m33s, 734 MB |
| Tanks&Temples | 21.20 / 0.767 / 0.280, 6m55s, 270 MB | 23.14 / 0.841 / 0.183, 26m54s, 411 MB |
| Deep Blending | 27.78 / 0.875 / 0.317, 4m35s, 386 MB | 29.41 / 0.903 / 0.243, 36m2s, 676 MB |

Check (derived): the per-scene rows above reproduce every Table 1 average to rounding. The 7-scene means (no flowers or
treehill) are 7K **26.83 / 0.829 / 0.241** and 30K **28.69 / 0.870 / 0.182**.

## 2. gsplat's published numbers

**a) gsplat docs** (https://docs.gsplat.studio/main/tests/eval.html, source `docs/source/tests/eval.rst`). 3DGS rows: gsplat
commit 6acdce4 against Inria "benchmark" branch 36546ce, TITAN RTX. Mip-NeRF 360, 7 scenes, averages only:

| | PSNR | SSIM | LPIPS (alex) | Mem | Time |
|---|---|---|---|---|---|
| inria-7k | 27.23 | 0.829 | 0.204 | 7.7 GB | 6m05s |
| gsplat-7k | 27.21 | 0.831 | 0.202 | 4.3 GB | 5m35s |
| inria-30k | 28.95 | 0.870 | 0.138 | 9.0 GB | 37m13s |
| gsplat-30k (1 GPU) | 28.95 | 0.870 | 0.135 | 5.7 GB | 35m49s |

Command: `cd examples; bash benchmarks/basic.sh`. That is `simple_trainer.py default --data_factor 4` for
garden/bicycle/stump and `2` for bonsai/counter/kitchen/room, `test_every` 8 by default.

**b) gsplat paper** (Ye et al., arXiv 2409.06765, https://arxiv.org/html/2409.06765). A100, PyTorch 2.1.2, CUDA 11.8,
LPIPS AlexNet. Table 1 (7-scene averages): gsplat-7k 27.23 / 0.83 / 0.20 at 3.36 min, gsplat-30k 29.00 / 0.87 / 0.14 at
19.39 min. Table 2: default 3.24M Gaussians (absgrad 2.47M, mcmc 1.00M, antialiased 3.38M). Per-scene 30K values come from
Tables 3/4/5 (PSNR/SSIM/LPIPS) and Table 6 (memory). **No per-scene Gaussian counts.**

| 30K | bicycle | garden | stump | room | counter | kitchen | bonsai |
|---|---|---|---|---|---|---|---|
| default PSNR | 25.29 | 27.39 | 26.51 | 31.23 | 29.01 | 31.37 | 32.21 |
| default SSIM | 0.77 | 0.87 | 0.77 | 0.92 | 0.91 | 0.93 | 0.94 |
| default LPIPS (alex) | 0.17 | 0.08 | 0.16 | 0.17 | 0.15 | 0.10 | 0.13 |
| absgrad PSNR | 25.44 | 27.47 | 26.71 | 31.43 | 29.07 | 31.65 | 31.98 |
| mcmc 3M PSNR | 25.58 | 27.65 | 26.93 | 32.40 | 29.65 | 32.21 | 33.13 |
| default mem GB | 10.47 | 9.89 | 8.20 | 2.84 | 2.36 | 3.16 | 2.41 |

**c) AMD ROCm port docs** (https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html).
This is the only per-scene **7K and Gaussian-count** table found. It is gsplat 1.5.3b2 on an **MI300X**, not upstream on
NVIDIA, run with `bash benchmarks/basic.sh`. Values are PSNR / SSIM / LPIPS (net not stated, presumably alex as default), then
Gaussian count and train time in seconds:

| Scene | 7K | 7K GS, s | 30K | 30K GS, s |
|---|---|---|---|---|
| bicycle | 23.69 / 0.67 / 0.30 | 4.46M, 147 | 24.93 / 0.76 / 0.16 | 7.78M, 1103 |
| garden | 26.59 / 0.83 / 0.11 | 4.83M, 169 | 27.63 / 0.87 / 0.07 | 6.69M, 1067 |
| stump | 25.90 / 0.73 / 0.23 | 4.11M, 142 | 26.78 / 0.77 / 0.14 | 5.73M, 870 |
| room | 29.97 / 0.90 / 0.19 | 1.48M, 156 | 31.75 / 0.92 / 0.14 | 2.29M, 758 |
| counter | 27.57 / 0.89 / 0.18 | 1.64M, 169 | 29.19 / 0.91 / 0.13 | 1.30M, 771 |
| kitchen | 29.42 / 0.91 / 0.11 | 1.89M, 175 | 31.54 / 0.93 / 0.08 | 2.12M, 815 |
| bonsai | 30.15 / 0.93 / 0.13 | 1.55M, 161 | 32.26 / 0.94 / 0.11 | 1.85M, 719 |

Our local gsplat `bicycle7k` (`--max_steps 7000`) gave 21.29 / 0.552 (parity-matrix.md). AMD's step-7000-of-30K number is
23.69 / 0.67. Per gotcha 4, the only schedule difference between those two is the means LR decay. The gap is still
unexplained.

**d) gsplat per-scene Gaussian counts, 30K default.** Source: Brush PR #121 table (https://github.com/ArthurBrussee/brush/pull/121);
protocol not stated. bicycle 6.26M, garden 5.84M, stump 4.81M, room 1.59M, counter 1.21M, kitchen 1.79M, bonsai 1.25M.

## 3. One scorer for all: nvs-bench (third party, 30K only)

https://github.com/nvs-bench/nvs-bench (`website/public/results/<method>/<dataset>/<scene>/result.json`). Setup: L40S. Images
are pre-selected (`images_*` copied to `images/`: 4x outdoor, 2x indoor). Test = sorted names, `idx % 8 == 0`. Every method's
test renders are scored by the same `evaluate/evaluate.py`: torchmetrics PSNR (data_range 255, mean per image), torchmetrics
SSIM, LPIPS **VGG**. Commands per method (`nvs-bench/eval.sh` in the N-Demir forks):
- 3DGS: `train.py --eval --iterations 30000`.
- gsplat: **`simple_trainer.py mcmc --data_factor 1 --strategy.cap-max 1000000 --max_steps 30000`** (MCMC with a 1M cap,
  not default).
- Brush: `brush_app <data> --eval-split-every 8 --eval-save-to-disk --total-steps 30000`, at about v0.3.0 (PR #271 head
  7dbd36c, 2025-09-19).

Values are PSNR / SSIM / LPIPS(vgg), then train time in seconds:

| Scene | 3DGS | gsplat (MCMC 1M) | Brush |
|---|---|---|---|
| bicycle | 25.249 / 0.7633 / 0.2371, 1800 | 25.251 / 0.7645 / 0.2544, 553 | 25.721 / 0.7907 / 0.2026, 890 |
| garden | 27.517 / 0.8656 / 0.1223, 1997 | 26.989 / 0.8483 / 0.1590, 583 | 27.818 / 0.8713 / 0.1165, 1006 |
| stump | 26.682 / 0.7693 / 0.2499, 1739 | 26.643 / 0.7782 / 0.2575, 561 | 27.241 / 0.8033 / 0.2103, 684 |
| room | 31.675 / 0.9201 / 0.2841, 1587 | 32.168 / 0.9262 / 0.2580, 870 | 32.224 / 0.9264 / 0.2704, 1072 |
| counter | 29.083 / 0.9085 / 0.2564, 1570 | 29.393 / 0.9158 / 0.2371, 1085 | 29.254 / 0.9136 / 0.2443, 1296 |
| kitchen | 31.371 / 0.9269 / 0.1539, 1908 | 31.478 / 0.9294 / 0.1541, 1116 | 31.830 / 0.9298 / 0.1510, 1485 |
| bonsai | 32.349 / 0.9416 / 0.2523, 1456 | 32.588 / 0.9462 / 0.2355, 845 | 32.297 / 0.9446 / 0.2420, 1704 |
| truck | 25.486 / 0.8823 / 0.1711, 1171 | 26.078 / 0.8907 / 0.1491, 678 | 26.339 / 0.8954 / 0.1390, 507 |
| train | 22.046 / 0.8167 / 0.2359, 1087 | 22.564 / 0.8314 / 0.2216, 500 | 22.723 / 0.8464 / 0.1891, 824 |
| drjohnson | 29.524 / 0.9038 / 0.3063, 2348 | 29.668 / 0.9080 / 0.3119, 590 | 29.449 / 0.9111 / 0.3106, 727 |
| playroom | 30.220 / 0.9079 / 0.3028, 1285 | 29.998 / 0.9098 / 0.3064, 558 | 30.246 / 0.9114 / 0.3134, 466 |

Other Brush numbers: Brush PR #121 (v0.2 to "MCMC-like", 2025-03) gives a 7-scene 30K table with brush-MCMC-like mean
29.72 / 0.886 / 0.196 and 1.757M splats (bicycle 25.67 with 2.98M). It does not state resolution, iterations or LPIPS net.

## 4. Splatfacto's published numbers

The only per-scene table found is gsplat docs at commit a45e203 (PR #134, 2024-02-22),
https://github.com/nerfstudio-project/gsplat/blob/a45e203ad0935c30c6c1050f621e01187ecdd41e/docs/source/tests/eval.rst.
nerfstudio 1d070f5 against Inria 2eee0e2, RTX 4090. **"same resolution (2x downscale)" for every scene**, outdoor included,
so these do **not** match the images_4 protocol. No room. LPIPS net not stated. Values are PSNR / SSIM / LPIPS:

| Scene | splatfacto 7K | splatfacto 30K | splatfacto-big 30K | inria 30K (same run) |
|---|---|---|---|---|
| bicycle | 22.99 / 0.65 / 0.31 | 24.99 / 0.75 / 0.18 | 25.70 / 0.78 / 0.15 | 25.61 / 0.78 / 0.21 |
| garden | 25.76 / 0.85 / 0.15 | 27.31 / 0.85 / 0.09 | 27.83 / 0.88 / 0.07 | 27.60 / 0.87 / 0.11 |
| stump | 24.59 / 0.68 / 0.28 | 25.64 / 0.73 / 0.18 | 26.70 / 0.77 / 0.15 | 25.89 / 0.77 / 0.22 |
| counter | 26.92 / 0.88 / 0.21 | 28.72 / 0.90 / 0.17 | 28.95 / 0.91 / 0.15 | 28.96 / 0.91 / 0.20 |
| kitchen | 28.48 / 0.90 / 0.14 | 31.18 / 0.92 / 0.10 | 31.60 / 0.93 / 0.09 | 31.30 / 0.92 / 0.13 |
| bonsai | 29.45 / 0.92 / 0.16 | 32.14 / 0.94 / 0.13 | 32.23 / 0.94 / 0.13 | 31.89 / 0.94 / 0.21 |

Times (min:s, 4090): splatfacto 30K bicycle 18:03 and bonsai 10:13; 7K bicycle 2:36. NerfBaselines
(https://nerfbaselines.github.io/mipnerf360) has 3DGS (27.43) and gsplat (27.41) on 9 scenes, but **no splatfacto**.

## 5. Brush: running it on this box

- **Versions.** The latest release is **v0.3.0** (2025-09-14). Its Windows asset is `brush-app-x86_64-pc-windows-msvc.zip`
  (159 MB, contains `brush_app.exe`). `main` is at 1388f74c (2026-10-03), with workspace version 1.0.0, unreleased. It has
  **renamed flags** (`--total-steps` became `--total-train-iters`) and adds a headless `brush-cli` binary. Rust 1.88+ is
  required; this box has cargo/rustc 1.96.0. Backend is wgpu (DX12/Vulkan), so no CUDA is involved.
- **Dataset folder** (gotcha 2): make a clean dir with `sparse/0/` plus `images/` holding the benchmark-resolution JPEGs. A
  junction to `images_4` or `images_2` works. Brush uses FoV plus a normalised center, so full-res `cameras.bin` is fine.
- v0.3.0: `brush_app.exe D:\bench\bicycle --total-steps 30000 --eval-split-every 8 --eval-every 30000 --eval-save-to-disk
  --export-every 30000 --export-path D:\bench\out\brush_bicycle`. With a source given, `--with-viewer` defaults to false
  (headless). It keeps the console when run from a terminal (`bin.rs` only calls `FreeConsole` with the viewer on).
- main: `cargo build --release -p brush-cli`, then `target\release\brush-cli.exe <dir> --total-train-iters 30000
  --eval-split-every 8 ...`. Other flags: `--max-resolution` (default 1920, long edge), `--max-splats` (default 10M),
  `--seed` (default 42), `--sh-degree` (default 3).
- **Output.** It logs `Eval iter N: PSNR x, ssim y`: per-image PSNR averaged over the eval views (render on black, 8-bit
  round trip). Eval also runs on the last step. `--eval-every` defaults to 1000, so raise it for clean timing. It prints the
  splat count (v0.3 notes). The PLY lands at `<export-path>/export_{iter}.ply`; the main default is
  `./{dataset}_exports/`. Eval renders go under the export path (nvs-bench moved `<out>/eval_30000`). **No LPIPS**: score
  the saved renders externally.

## 6. nerfstudio splatfacto on this box (RTX 4070, driver 596.21, nvcc 13.2, Python 3.10)

- **Feasible, not tested.** nerfstudio **1.1.5** (PyPI latest, 2024-11-11; repo last commit 2025-07-29) pins
  **`gsplat==1.4.0`**, which conflicts with our gsplat 1.5.3 env, so use a **separate venv**. gsplat v1.4.0 ships
  `gsplat-1.4.0+pt24cu124-cp310-cp310-win_amd64.whl`, so no JIT build is needed. A JIT build would need an nvcc matching
  cu124, and the box has 13.2. Windows wheel check (PyPI, cp310): open3d 0.20.0 yes; pymeshlab 2023.12.post1 yes (pinned
  `<2023.12.post2` on win32); xatlas yes; nerfacc 0.5.2 is pure Python. **fpsample is unpinned and 1.0.x is sdist-only**:
  pin `fpsample==0.3.3`, which has a cp310 win wheel. nerfstudio's docs still recommend torch 2.1.2+cu118 plus vcvars64; with
  the prebuilt wheel, vcvars should not be needed. Not verified.
- Install (not run):
  `uv venv ns-env --python 3.10` ;
  `uv pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124` ;
  `uv pip install https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt24cu124-cp310-cp310-win_amd64.whl` ;
  `uv pip install fpsample==0.3.3 nerfstudio==1.1.5`. Then check `from gsplat import csrc`.
- Train. The default dataparser reads `transforms.json`, so add the `colmap` subcommand **last**. `colmap_path` defaults to
  `colmap/sparse/0`:
  `ns-train splatfacto --data D:\bench\bicycle --max-num-iterations 30000 --vis tensorboard
  --viewer.quit-on-train-completion True colmap --colmap-path sparse/0 --images-path images --downscale-factor 4
  --downscale-rounding-mode ceil --eval-mode interval --eval-interval 8`
- Metrics, renders and PLY: `ns-eval --load-config outputs/.../config.yml --output-path eval.json --render-output-path
  renders\` ; `ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir exports\splat`.

## 7. Recommended protocol (identical inputs, one scorer)

1. **One canonical folder per scene**: `sparse/0` plus `images/` holding the exact benchmark JPEGs. Use shipped `images_4`
   for bicycle/garden/stump and `images_2` for room/counter/kitchen/bonsai, which is what 3DGS used. Use the shipped tandt_db
   images as they are. Give every tool this folder at factor 1. gsplat then gets `--data_factor 1` and never re-downsamples
   to PNG. Brush gets no full-res folder to pick. nerfstudio needs the `images_N` plus `ceil` variant (gotcha 5).
2. **Same test photos.** Sorted name, `idx % 8 == 0`. Before reading any number, dump each tool's test file list and diff it
   against SpawnScene's `&llffhold=8` list (nerfstudio orders by image_id).
3. **Iterations.** Report 30K and 7K. A "7K" cell is a run whose schedule ends at 7K: gsplat `--max_steps 7000
   --eval_steps 7000`, Brush `--total-steps 7000`, 3DGS `--iterations 7000` (same schedule as its 30K run). Splatfacto has no
   7K schedule: `--max-num-iterations 7000` still decays toward 30K and spends steps 0-6000 at reduced resolution, so label
   it. Where a tool also reports step 7000 of a 30K run (gsplat docs/AMD, paper "7K"), keep it in a separate, labelled column.
4. **One scorer.** Save 8-bit PNG test renders from every tool, SpawnScene included. Rescore all of them with one script:
   nvs-bench's `evaluate.py` recipe (torchmetrics PSNR on 0-255 averaged per image, torchmetrics SSIM, LPIPS **VGG**).
   Publish each tool's self-reported number next to it. gsplat: set `--lpips_net vgg` if its own number is shown.
5. **Same background** (black) for eval renders. Default SH degree 3 everywhere. Record seed.
6. **Record per run**: tool version or commit, exact command, final Gaussian count, train-only wall time (eval off until the
   end), peak VRAM, GPU and driver. One GPU job at a time on the 12 GB 4070. A concurrent run thrashed gsplat
   (memory `ref-gsplat-reference-trainer-windows`).
7. **Losses are reported as losses.** A row where SpawnScene is behind keeps its number and gets a reason or "unexplained".
   It is never dropped.
