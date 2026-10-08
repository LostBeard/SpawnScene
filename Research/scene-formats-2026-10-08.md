# Opening other tools' splat scenes (2026-10-08)

Goal (PLANS 8, TJ's "best generator and viewer on the web"): a viewer people can drop any published splat into. Before
tonight the Studio opened only `.spawnscene`; the legacy Viewer page had a PLY/SPLAT reader with bugs. Now "Open scene
file" and `?import=<url>` take every format below, each decoded **on the GPU** (a 30K PLY is 0.5-0.8 GB - it never enters
the wasm heap) into a new project, then opened and seated on the scene's dense core.

| Format | Who writes it | Decoder (spec source) | Verified on |
|---|---|---|---|
| 3DGS `.ply` | reference trainer, gsplat, nerfstudio, Postshot, Polycam | GaussianPly + GaussianPlyImport (graphdeco layout) | Inria's Train 7K (741,883 splats, SH 3) |
| compressed `.ply` | SuperSplat export (PlayCanvas) | GaussianPly.ParseCompressed + ConvertCompressedAsync (playcanvas/engine ply.js, gsplat-compressed-data.js) | PlayCanvas biker (152,746), guitar (90,854) |
| `.sog` (bundle or `meta.json` URL) | splat-transform, SuperSplat | SogMeta + SogImport (gsplat-sog-data.js, sog.js) | PlayCanvas skull (246,821, SH 3); unbundled pixel-identical |
| `.spz` v2/v3 | Niantic, Scaniverse | SpzImport (nianticlabs/spz load-spz.cc, MIT) | Spark's butterfly (177,132, SH 3), penguin (128,136) |
| `.splat` | antimatter15 convert.py and many web viewers | SplatFileImport (convert.py) | antimatter15's train.splat (1,026,508) |

Every decoder has a CPU-accelerator test against a port of the format's own reference decoder (or its packer, for SPZ),
on random or quantised data: GaussianPlyImportTests, CompressedPlyImportTests, SogImportTests, SpzImportTests,
SplatFileImportTests (19 tests). Two cross-checks between independent decoders: Inria's Train PLY vs antimatter15's
train.splat (same scene, same framing); PlayCanvas's biker compressed PLY vs SPZ (the SPZ turned out to be v4, below).

## Up is y-down everywhere in practice

A 3DGS PLY lives in its SfM (COLMAP) frame: y DOWN. Measured on Inria's Train: the least-variance axis of the dense core
is -Y (0.97), and sky-coloured splats sit 2.3:1 on the -Y side. The SPZ spec calls its frame RUB (y up), but the files
in circulation carry the PLY's frame - Spark's examples rotate every SPZ 180 degrees about X, and its butterfly drew
upside down here without it. SuperSplat's compressed PLY and SOG, and antimatter15's .splat, are converted from PLYs
without turning. So every import turns the scene 180 degrees about X by default: positions (y, z negated), rotations
(q' = (1,0,0,0) * q) and SH (that turn is diagonal in the real SH basis: coefficients 1, 2, 4, 7, 9, 11, 12, 14 flip sign
- checked in the tests by the physics: the colour seen from F d equals the original from d). `&sceneup=keep` leaves it.

## Bugs found on the way

- **Aliased stand-in buffers.** With no SH bands, each importer bound ONE 1-float stand-in to three read_write slots.
  WebGPU refuses it ("Storage buffer aliasing detected"); the CPU accelerator does not check, so every unit test passed
  and the first real SH-less file (biker) failed. Three distinct stand-ins now.
- **Legacy SplatParser** read .splat rotations as signed bytes (they are q * 128 + 128) and called the linear scale
  log-space. The new decoder follows convert.py.
- **Seat.** The generic seat framed the 1-99% box; a 360 capture's background shell put the camera among the trees
  (Train). Imports frame the 20-80% box.

## Not read yet

- **SPZ v4** (32-byte NGSP header, a table of contents, per-attribute zstd streams). Already in PlayCanvas's examples
  (biker.spz). Chrome 151's DecompressionStream has gzip / deflate / deflate-raw only, no zstd - it needs a decoder:
  ZstdSharp (MIT, pure C#, NuGet) or a small JS one (fzstd, MIT). A dependency, so TJ's call. Refused with that reason.
- `.ksplat` (mkkellogg GaussianSplats3D), `.splatv` (4D), glTF KHR_gaussian_splatting (draft).

## Test files (research only, not redistributed)

`_models_scratch/`: `ply/train7k.ply` (Voxel51/gaussian_splatting, Apache-2.0 per its card; via the hub /src proxy),
`splat/train.splat` (cakewalk/splat-data, via the hub /hf), `spz/{butterfly,penguin}.spz` (sparkjs.dev assets),
`pc/{biker,guitar}.compressed.ply, skull.sog, biker.spz` (playcanvas/engine examples, MIT repo).
