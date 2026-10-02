# Offline SfM harness - DrJohnson (2026-10-01)

Python replays of SpawnScene's pair filtering, relative-pose estimation and positioning on DrJohnson, scored against
COLMAP's published reconstruction. See `Research/sfm-drjohnson-positioning-2026-10-01.md` for the results.

- `dj_cache_k1024.py` - runs the app's learned front end (Kornia RaCo-ALIKED + LightGlue+ k1024, `_scratch/kornia`) through
  onnxruntime on all 946 pairs and writes `dj_k1024.npz` next to it (~7 min CPU).
- `dj_analyze.py [F|FLM|E]` - pair filters (F verify, relative pose, loop filter, rotation averaging, rotation filter)
  labelled with COLMAP truth.
- `dj_relpose.py`, `dj_eproj.py` - relative-pose estimators on true pairs.
- `dj_ta.py` - translation-averaging experiment.
- `dj_colmap.py`, `dj_colmap_sift.py` - pycolmap 4.2.1 (`pip install --user pycolmap`) incremental and global mapping on
  our matches / on COLMAP's own SIFT.
- `dj_matchers.py`, `dj_rotavg.py` - COLMAP model readers and rotation averaging (from the 09-30 study).

Paths: the dataset at `C:\Users\TJ\Downloads\tandt_db\db\drjohnson`. For the C# side, the cache is exported to
`_scratch/djmatch/k1024.bin` (format in `SpawnScene.Tests/DrJohnsonGlobalInitTests.cs`).
