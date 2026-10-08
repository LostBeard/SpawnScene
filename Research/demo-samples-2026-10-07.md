# Demo samples: sources, licenses, hosting (2026-10-07)

TJ, 2026-10-07: remove the old "Or try a sample" buttons (640 px PNGs of unknown origin, plus a TempleRing dev button);
offer high-resolution single photos and good multi-photo sets for demoing.

## What may be redistributed

The license has to allow us to host the photos and anyone to download them from spawnscene.com, commercially (SpawnDev
is a business). Checked 2026-10-07:

| Source | License | Usable? |
|---|---|---|
| Mip-NeRF 360 (bicycle, garden, room, counter, kitchen, bonsai, stump...) | **none stated** (jonbarron.info/mipnerf360) | No: no license = all rights reserved. Research use only, never rehost. |
| Tanks and Temples (Truck, Train...) | non-commercial research; redistribution needs written permission | No |
| Deep Blending (DrJohnson, Playroom) | none stated | No |
| COLMAP sample sets (South Building, Gerrard/Person/Graham Hall) | none stated on colmap.github.io/datasets.html | No |
| AliceVision dataset_monstree (Meshroom's sample) | GitHub repo has no license file | No |
| ETH3D | CC BY-NC-SA 4.0 | No (non-commercial) |
| CO3D, MVImgNet | non-commercial | No |
| SceneSplat-49K (HF) | CC BY-SA 4.0, but it is trained splats, not photos | Not a photo source |
| **Wikimedia Commons, Category:Photogrammetry of Korno climbing rock** | **CC0** (96 of 97 files; 1 CC BY-SA) | Yes. 8000x4500, Zby, 2026-08. A rock face: technically good, visually plain. |
| **Commons, Category:Set of 95 pictures for photogrammetry exercise** (a pine cone, "Pinha") | **CC BY-SA 4.0** | Yes, with credit + share-alike. 95 x 6000x4000, N.ELAC (2019). An object in the round. |
| **Commons, Category:Hamamni Persian Baths photogrammetry** | **CC BY-SA 4.0** | Yes, with credit. 59 x 1848x4000 portrait, Nassima Chahboun (2022). An INTERIOR: octagonal pool, tiled floor, domed ceiling. |
| Commons CC0 / public-domain single photos (search `haswbstatement:P275=Q6938433`) | CC0 / PD | Yes. Plenty at 5000-8000 px: living rooms, museum interiors, gardens. |
| TJ's own captures (Bathroom: 35 phone photos, 3120x4160) | TJ's | Only with TJ's explicit OK - photos of his home. |

Other Commons photogrammetry categories are unusable as sets: Fazenda do Pinhal (48 mixed screenshots and point-cloud
renders), Kazakova street (5 TIFFs).

**Credit rules:** CC BY-SA needs the author, the license and a link, and adaptations (our resized copies) stay CC BY-SA.
The catalog entry carries `credit`, `license`, `licenseUrl`, `source`; loading a sample stores the credit on the project
(`Project.Credit`), shown on its Photos tab. A scene made from a CC BY-SA set is an adaptation: if it is ever published
by us (a showcase), credit it and keep it CC BY-SA.

## Hosting

| Option | Notes |
|---|---|
| GitHub Pages (the app deploy) | Every sample would ride in every deploy; repo size limits. Only the catalog JSON lives here. |
| **Hugging Face dataset repo (LostBeard)** | Free public hosting, `resolve/main/...` URLs send `Access-Control-Allow-Origin` for any origin (checked with Origin: spawnscene.com), served from a CDN. Chosen. |
| hub.spawndev.com | Ours, but a deploy needs TJ, and it is one server. Fallback. |
| AWS S3 (CLI configured, buckets exist) | Costs per GB served; CORS to configure. Fallback. |

Layout: `samples/catalog.json` in the app (`SampleCatalog`: `base` + entries); photos stored at
`huggingface.co/datasets/LostBeard/spawnscene-samples/resolve/main/<folder>/<NNN>.jpg` and FETCHED through the hub:
`https://hub.spawndev.com:44365/src?url=<that URL>`. Standing rule (memory ref-never-request-huggingface): shipped code
never requests huggingface.co directly. The hub's `/hf/{org}/{repo}` route parses model repos only (a dataset path
becomes `datasets/LostBeard` as the repo); `/src` proxies any URL - MEASURED 2026-10-07: 200, CORS `*`, an uncached
photo in 0.9-1.1 s.

## Sizes

The trainer works at <= 1024 px on the long side by default (ProjectSettings.TrainMaxDimension), learned features at
1024: a set downscaled to 2048 px on the long side loses nothing today and leaves headroom for a higher training
resolution. JPEG at 2048 px is ~0.5-1 MB a photo, so a 60-photo set is 30-60 MB: say the size on the button.
Single photos are the opposite case: depth-based scenes use the full photo, so keep them at 4000+ px.

## Status

- [x] Old buttons removed; catalog UI + loader (Studio.ProjectPage / Studio.Projects.LoadSampleAsync).
- [ ] Each set reconstructed through the user path and judged off the photo path (wander + pan views) before it
      goes in the catalog - a demo that looks bad is worse than none.
- [ ] Upload to the HF dataset repo.
- [ ] Ask TJ: may the Bathroom set be a sample? Would he capture 2-3 demo sets (an object on a table, a room, an
      outdoor scene) with his phone? Our own captures are the only fully clean multi-photo sources at quality.
