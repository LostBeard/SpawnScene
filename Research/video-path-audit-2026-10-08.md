# Video input path audit (2026-10-08, read-only)

## RESULT 2026-10-10: the user path is tested and works

The three changes in section 4 are in (`&videoframes`, the project autotest's video branch with `[Dataset] PICK-FILE`,
`tools/_cdp_dataset.js` answering with CDP `DOM.setFileInputFiles`). Run `_runs/tuvok-vid.sh` (build f16, the five
2026-10-09/10 defaults): TruckVideo picked into the real `<input type=file>` -> `[Studio] truck.mp4: 126 frames saved as
sources in 11.2s` -> `exposure auto: 0 of 126 photos carry EXIF exposure (unknown: keep the gains) -> per-photo gains ON`
-> own SfM 126 of 126 posed, shared focal 582.9 px (COLMAP 581.9 at this size) -> 7K, 16 held out -> save -> reopen ->
captures -> `[Dataset] DONE`, no FAIL.

| Truck, 7K, user path, llffhold 8 | held out PSNR / SSIM |
|---|---|
| video (126 frames of truck.mp4) | **22.41 / 0.826** |
| the same 126 photos | 22.93 / 0.828 |

The video plumbing (H.264 + JPEG q0.95 frames) costs 0.52 dB PSNR and nothing measurable in SSIM. Still a slideshow: a real
phone clip (blur, rolling shutter, stabilisation, near-duplicates) is the next test.


Scope: how a video becomes training input, what tests exist, risks, and an end-to-end test plan. Nothing was built,
run or changed. Paths are relative to `SpawnScene/SpawnScene/` unless they start with `tools/`, `_runs/` or `Research/`.

**Short answer:** the README's claim (`README.md:60`, "A video is split into frames first") is TRUE in code, and the
video code has been run, but only through the **dataset harness** (`autotest=dataset`, Tuvok 2026-09-24,
`_runs/tuvok-truckvideo-sfm.*` and `_runs/tuvok-truckvideo-7k.*`). The **user path** (pick a video file into a project,
press Generate) has never been tested. `PLANS.md:50,56` already lists that gap. The project autotest skips video
manifests (`Pages/Studio.ProjectAutotest.cs:35`).

## 1. How a video becomes training input today

### 1a. User path (project page)
1. The hidden picker accepts video: `<InputFile ... OnChange="OnFileSelected" accept="image/*,video/*" multiple>`
   (`Pages/Studio.razor:34`).
2. `OnFileSelected` sends each file whose `ContentType` starts with `video/` to `AddVideoSourcesAsync`
   (`Pages/Studio.Projects.cs:1214-1218`). Anything else goes down the image path, which reads at most 50 MB into a
   `byte[]` (`:1223-1226`).
3. `AddVideoSourcesAsync` (`Pages/Studio.Projects.cs:1268-1296`):
   - `UrlForPickedFileAsync(fileName)` looks through every `input[type=file]` on the page for a File with that name and
     returns a `blob:` URL (`Services/VideoFrameExtractor.cs:29-33`, JS `:174-178`). The video itself never enters .NET.
   - `ExtractAsync(url, VideoFrameCount, progress)` uses the defaults **3 candidates per slot, maxDimension 1600**
     (`Services/VideoFrameExtractor.cs:46-47`). `VideoFrameCount` is a static set to **120** (`Pages/Studio.Projects.cs:1262`).
   - Each frame is saved as an ordinary project source named `{stem}_frame_0001.jpg`, ... through
     `ProjectService.AddSourceAsync` (`Pages/Studio.Projects.cs:1287-1289`, `Services/ProjectService.cs:94-114`).
     From here on, a frame is treated exactly like a photo.
4. Frame extraction in JS (`Services/VideoFrameExtractor.cs:104-166`):
   - A `<video>` element is attached to the document, because a detached element returned the first frame every time
     (measured, `:108-111`). An http URL is first fetched into a blob, because a server without Range support could
     not seek (`:112-120`).
   - Output size: the long side is capped at maxDim and the image is never upscaled (`:128-129`).
   - Sampling: `count` equal time slots over the duration, `t0 = dur*k/count` (`:138-139`). Inside each slot,
     `candidates` evenly spaced times are tried (`:141-142`), each scored by Laplacian variance on luma at <= 480 px
     (`:77-89`, `:132-136`, `:146-148`). The sharpest time is sought again and drawn at full output size (`:150-151`).
   - Encoding: `convertToBlob` writes JPEG at quality 0.95 (`:153`). **No EXIF.**
   - Seek check: throws if `currentTime < t/2` for t > 0.5 s (`:144-145`). Duplicate guard: throws only when **more
     than half** of the consecutive frame pairs have identical pixel signatures (`:159-164`).
   - Bytes go to .NET with `bytes.ReadBytes()`, one frame at a time (`:55-60`). The JS-side array is cleared only after
     every frame has been copied (`:63`).
5. Generate: `GenerateMultiViewScene` (`Pages/Studio.Projects.cs:651-838`) loops over `_activeProject.Sources`
   (`:676`):
   - It reads the first 256 KB of each file for EXIF (`:686-694`). A canvas JPEG has none, so `CreateFromExif` falls
     back to `focal = 1.2 * max(w,h)` (`Models/CameraParams.cs:256-259`).
   - Each photo is decoded to the GPU at `ImageImportService.MaxImportDimension` (default 1024, see the comment at
     `:671`). Only the size is kept (`:700-706`).
   - Then `_multiViewService.GenerateAsync(images, ...)` (`:746`), which goes to `GenerateCoreAsync`
     (`Services/MultiViewGenerationService.cs:1209`) and, with DAv3 loaded, to `GenerateWithDav3MultiViewAsync`
     (`:1228-1232`).
   - Focal for video frames: DAv3 gives a focal per view, and bundle adjustment shares one focal when every image has
     the same size (`oneCamera`, `Services/MultiViewGenerationService.cs:487-489`, reported at `:870-873`). Mixing
     video frames with photos of a different size switches to per-view focals.
   - Pairs: all pairs up to 60 images. Above 60, retrieval keeps each image's top 30 partners
     (`Services/ImageImportService.cs:668-698`, learned features are the default, `:71`). Time order is not used as a
     prior.
6. Training views: `RecordTrainingViews(scene, images, fromProjectStore: true, holdOut: LlffHold > 0)`
   (`Pages/Studio.Projects.cs:779`). Hold-out is by image index `i % LlffHold == 0` (`:407`), and frames are in
   insertion order, so that is time order. Training reloads each target from OPFS by file name
   (`Pages/Studio.Training.cs:1855-1866`). `TrainProjectSceneAsync` turns off the mid-run held-out curve
   (`HeldOutEveryCycles = 0`, `Pages/Studio.Projects.cs:887`). The final `[Train] trainer HELD OUT PSNR` line is still
   printed (`Pages/Studio.Training.cs:1003-1012`, inside `TrainOnTrainingViewsAsync`, which starts at `:308`).

### 1b. Dataset path (harness only)
- `TryLoadManifestAsync` accepts a manifest that has a `Video` field (`Services/ImageImportService.cs:1113`). The
  fields are `Video`, `VideoFrames` (0 means 120), `VideoCandidates` (`:1068-1075`).
- `ExtractAsync("datasets/{name}/{video}", VideoFrames, VideoCandidates, 1600)` (`:849-853`). The frames are put in the
  static `VideoFrameStore` under the key `video-frame:frame_NNNN.jpg` (`:847`, `:854-858`;
  `Services/VideoFrameExtractor.cs:190-205`), and wrapped back into a JS Blob for decoding (`:941-943`).
- Training fetches targets from `VideoFrameStore` (`Pages/Studio.Training.cs:1867-1875`). The harness photo dump reads
  it too (`Pages/Studio.TrainerRenderDump.cs:17-20`, `Pages/Studio.Dataset.cs:402`). `VideoFrameStore.Clear()` has no
  caller.

## 2. Existing harness and test support, and what TruckVideo is

- **Dataset harness:** `node tools/_cdp_dataset.js TruckVideo N` (the default `AUTOTEST=dataset`, URL built at
  `tools/_cdp_dataset.js:144`). Two runs exist:
  - `_runs/tuvok-truckvideo-sfm.cmd` (train=0)
  - `_runs/tuvok-truckvideo-7k.cmd` (7K, poses=dav3, every-4th held out)

  Results of the 7K run (`_runs/tuvok-truckvideo-7k.log`, 2026-09-25, a build more than 2 weeks old):

  | Measure | Result |
  |---|---|
  | Frame extraction | 126 frames in 10.3 s. The sfm run's identical extraction took 266.8 s. |
  | Views posed | 126/126 |
  | Pose vs ground truth | 0.1% of spread, forward error 0.1 deg |
  | Focal | BA shared 582.3 px. COLMAP scaled to 979 px wide is 1163.25 x 979/1957 = 581.9. |
  | Held-out PSNR | 10.82 -> 20.64 dB, SSIM 0.764 (curve mean 18.65 dB) |
  | Splats | 1,197,831 |

- **Project (user-path) autotest:** `Pages/Studio.ProjectAutotest.cs:23-231`, started by `autotest=project`
  (`Pages/Studio.razor.cs:420-435`). It only takes photo manifests (`:35`) and stores bytes with `AddSourceAsync`
  directly (`:58-65`), so it never goes through `OnFileSelected` or `AddVideoSourcesAsync`. None of the 25
  `AUTOTEST=project` run scripts in `_runs/` use video.
- **Other tools:**
  - `tools/_cdp_video_probe.js` only measures how Chrome hands out seeked frames. It does not run the pipeline.
  - `tools/make_truck_video.py` built the clip.
  - `tools/_cdp_edit.js:98-105` already shows how to "pick" a file with CDP `DOM.setFileInputFiles`
    (`SPAWNSCENE_EDIT_PICK`), which fires the input's change event the way a real pick does.
- **`&videoframes=N` is documented but never parsed.** The doc comment is at `Pages/Studio.Projects.cs:1261`, but
  nothing in `Pages/Studio.razor.cs` sets `VideoFrameCount`. A grep for `videoframes` finds only that comment. The UI
  has no frame-count setting either.
- **TruckVideo contents** (`_pub_tuvok_est/wwwroot/datasets/TruckVideo`, identical to `SpawnScene/wwwroot/datasets/TruckVideo`):

  | File | Size | Notes |
  |---|---|---|
  | `manifest.json` | 363 B | `video: truck.mp4`, `videoFrames: 126`, `videoCandidates: 3`, `width 978`, `height 546`, `poses: poses.par`, `images: []` |
  | `truck.mp4` | 13,532,665 B | H.264; mp4 header: 979x546, 12.6 s. One Truck photo per frame at 10 fps, single GOP (`tools/make_truck_video.py:1-9`). |
  | `poses.par` | 26,032 B | 126 COLMAP poses renamed `frame_0001.jpg`... with full-resolution intrinsics (fx 1163.25, cx 978.5). Used for pose-vs-GT only, unless `gtposes=1`. |

  - The script's filter `scale=trunc(iw/2)*2:trunc(ih/2)*2` (`tools/make_truck_video.py:28`) would give 1956x1090
    from Truck's 1957x1091 (`datasets/Truck/manifest.json`). The file is about half that size, so it was not made by
    the script as committed.
  - The manifest says 978 wide while the header and the extracted frames say 979.
- **Caveat:** TruckVideo is a **slideshow of still photos**. It has no motion blur, no rolling shutter, no
  stabilisation and no near-duplicate frames, and with 126 slots each slot holds exactly one source picture, so the
  "sharpest of 3" choice never picks between different images. It tests the plumbing (decode, seek, naming, storage,
  target reload), not how robust the pipeline is to real video.

## 3. Risks seen by reading

1. **User path untested.** No run has exercised the `ContentType` gate, blob-URL lookup, OPFS naming or OPFS target
   reload for a video.
2. **Missing or unusual MIME type.** A file whose browser type is empty or not `video/*` (some `.mkv` or `.mov` on
   Windows) goes to the image path. It is read up to the 50 MB cap (`Pages/Studio.Projects.cs:1223`) and any error
   aborts the rest of the batch (`:1253-1258`). A codec Chrome cannot decode (for example some HEVC `.mov`) throws
   "video failed to load" (`Services/VideoFrameExtractor.cs:124`). The user sees that message but not the reason.
3. **Fixed 120 frames whatever the clip length** (`Pages/Studio.Projects.cs:1262`).
   - A 10 s clip gives 12 frames a second: heavy redundancy and all-pairs work.
   - A 5-minute walk gives one frame every 2.5 s, which can break overlap.
   - Sampling ignores motion, there is no URL knob (see section 2), and there is no UI setting.
4. **Near-duplicates.** Standing still produces frames that are similar but not identical. The guard only catches
   exact consecutive repeats, and only when they are the majority (`Services/VideoFrameExtractor.cs:159-164`). Pairs
   with almost no baseline reach SfM unfiltered.
5. **Focal.**
   - Canvas JPEGs carry no EXIF, so the initial focal is the 1.2 x max-side guess (`Models/CameraParams.cs:256-259`),
     and DAv3 plus BA must recover it. That worked on TruckVideo (582.3 vs 581.9).
   - Container metadata (QuickTime focal or 35 mm-equivalent tags) is never read.
   - Phone electronic stabilisation and digital zoom crop the image from frame to frame, which breaks the one shared
     focal and principal point (`Services/MultiViewGenerationService.cs:487-489`).
6. **Rolling shutter.** Nothing in the code models it (grep for "rolling shutter" finds nothing). Fast pans from phone
   CMOS sensors skew straight lines. The Laplacian sharpness score does not catch skew.
7. **Rotation and HDR not verified.** I have not checked how Chrome applies a portrait video's rotation metadata, or
   how it tone-maps HDR (HLG/Dolby Vision) video, when drawing to a 2D canvas (`Services/VideoFrameExtractor.cs:146,151`).
8. **Memory and copies.**
   - During extraction every JPEG is held twice: in the JS `frames` array and in the .NET `List<Frame>`
     (`Services/VideoFrameExtractor.cs:55-63`). That was 25 MB for 126 frames at 979x546 (7K log).
   - Every frame crosses into .NET with `ReadBytes()` and back out to OPFS as a `byte[]`
     (`Pages/Studio.Projects.cs:1289`), against the "bulk bytes stay in JS" rule.
   - The dataset path keeps every frame in a static `ConcurrentDictionary` that is never cleared
     (`Services/VideoFrameExtractor.cs:193`).
   - Features for 126 images used 373 MB of live heap (7K log).
9. **Seek cost depends on the GOP.** One GOP makes every seek decode from frame 0: identical extraction took 10.3 s in
   one run and 266.8 s in the other. That is 4 x `count` seeks per clip (`Services/VideoFrameExtractor.cs:141-150`).
10. **Name collisions.** Two videos with the same stem, or a re-add, write the same OPFS file names, and `Sources` gains
    duplicate entries (`Services/ProjectService.cs:103-111` does not de-duplicate).
11. **Held-out leakage.** With video, the neighbours of a held-out frame are 0.1 s away, so `llffhold` scores read
    higher than on photo sets. Compare video against video, or against the identical photos (Truck), not against
    other datasets.

## 4. End-to-end test plan (user path, scored)

**Goal:** TruckVideo goes through `OnFileSelected`, then `AddVideoSourcesAsync`, then OPFS, then the real
`GenerateMultiViewScene`, then train, then save and reopen, and is scored on held-out frames.

### Smallest change needed (none of it exists today)

1. **`Pages/Studio.razor.cs`, global options block near `:223`:** parse
   `if (query.TryGetValue("videoframes", out var vf) && int.TryParse(vf, out var vfn)) VideoFrameCount = Math.Max(2, vfn);`
   This makes the existing doc comment true.
2. **`Pages/Studio.ProjectAutotest.cs`, `RunProjectAutotestAsync`:**
   - Add a branch before `:35` for `!string.IsNullOrEmpty(manifest?.Video)`:
     - Create the project with no photos.
     - Set `_activeProject` and `StudioState.ProjectDetail` (`OnFileSelected` returns when `_activeProject` is null,
       `Pages/Studio.Projects.cs:1204`).
     - If `&videoframes` was not given, use `manifest.VideoFrames`.
     - Log `[Dataset] PICK-FILE input[type=file][accept^="image/*"]|datasets/{name}/{manifest.Video}`.
     - Wait, with a deadline, until `OnFileSelected` has refreshed `_activeProject` with
       `Sources.Count >= VideoFrameCount` (`:1249-1250`). A `TaskCompletionSource` set at the end of `OnFileSelected`
       is cleaner than polling.
   - Then fall through to the existing Generate / reopen / capture code (`:92-225`).
   - Log `[Dataset] FAIL` if no frames arrive.
3. **`tools/_cdp_dataset.js`, console handler (`:101-127`):** on `[Dataset] PICK-FILE <selector>|<path>`, run
   `DOM.getDocument`, then `DOM.querySelector`, then `DOM.setFileInputFiles` with
   `path.resolve(__dirname, '../SpawnScene/wwwroot', <path>)`. Copy this from `tools/_cdp_edit.js:101-105`. That is
   the browser's own file-pick event: Chrome sets `type=video/mp4` from the extension, so the `ContentType` gate is
   exercised too.

### Run (Windows cmd, one run at a time, published release build, after checking the board and peers)
```
set AUTOTEST=project
set "EXTRA=&videoframes=126&llffhold=8&trainres=1024"
set VIEWPORT=979x546
set MINUTES=150
set RUN_TAG=video-user-7k
node tools/_cdp_dataset.js TruckVideo 7000 > "_runs\video-user-7k.log" 2>&1
```
The URL this produces is
`studio?autotest=project&name=TruckVideo&train=7000&videoframes=126&llffhold=8&trainres=1024`.

### Pass/fail lines to read
- `[Studio] truck.mp4: 126 frames saved as sources` (`Pages/Studio.Projects.cs:1290`).
- `[Studio] supervision from ...: N of 126 views posed, 16 held out` (`Pages/Studio.Projects.cs:437`). With
  `i % 8 == 0` over 126 images, 16 are held out.
- `[Train] trainer HELD OUT PSNR a -> b dB, SSIM` (`Pages/Studio.Training.cs:1011-1012`).
- The autotest's own checks: the saved scene got 7000 iterations, `shDc` and SH degree are right, and the reloaded
  capture matches the live capture (`Pages/Studio.ProjectAutotest.cs:126-140,142-147`).

### Controls (same build, same knobs)
- **Same images as photos:** `AUTOTEST=project`, `COUNT=126`, `node tools/_cdp_dataset.js Truck 7000` with
  `&llffhold=8`. Truck photos are 1957 px wide and both runs train at `trainres=1024`. The held-out gap measures only
  the cost of the video plumbing (H.264 at CRF 18 plus JPEG at q0.95).
- **Dataset path on the video:** `AUTOTEST=dataset`, TruckVideo, `&llffhold=8`. This repeats the 09-25 numbers on
  today's build and adds pose-vs-GT, which the project path does not report.

### Then a real clip
TruckVideo cannot exercise blur, rolling shutter, stabilisation or near-duplicate frames. Add one short phone video
(about 20-30 s walk-around, `video/mp4`, H.264) as a second manifest-`Video` dataset, and run the same command.
