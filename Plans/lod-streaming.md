# LOD streaming for massive scenes (plan, 2026-10-05)

Step 3 of the massive-scenes plan (1: .spawnscene v2 compression - done; 2: partitioned training - done, opt-in).
Goal: view scenes far larger than one GPU's memory (multi-room, 10M-100M+ splats) at a steady frame rate, in the
browser and on Quest, by rendering a view-dependent subset of a level-of-detail tree and streaming only the chunks
that subset needs.

## What the reference viewers do

- **Spark 2.0** (World Labs, [LoD deep dive](https://sparkjs.dev/docs/new-spark-renderer/),
  [blog](https://www.worldlabs.ai/blog/spark-2.0)): an LoD splat TREE built by merging splats on a voxel grid whose
  step grows by r^level (r = 1.5 default; "Tiny-LoD", in-browser) or by Bhattacharyya-distance pair merging
  (offline, ~30-40% larger tree). Parent = weighted merge of its children, weight = opacity x area. Per view, a
  priority-queue traversal picks the cut under a splat BUDGET (500K-2.5M by device), O(N log N) in the RENDERED
  count, independent of the tree size. File `.RAD`: JSON header with chunk offsets, 64K-splat chunks, columnar,
  gzipped per property, chunk 0 = the 64K LARGEST splats (renders a coarse scene at once). GPU: a fixed pool of 16M
  splats in 64K pages, LRU-evicted.
- **PlayCanvas** ([splat-transform](https://github.com/playcanvas/splat-transform)): "Streamed SOG" - a spatial
  octree of chunks, each at several LODs, a manifest + a folder of SOG files.

## SpawnScene design

GPU-first (the project rule): the cut runs on the GPU in parallel, not as a CPU priority queue.

1. **LOD tree build (after training, in the browser).** Tiny-LoD style: level l uses grid step s0 * 1.5^l. Level 0 =
   the trained splats. For each next level, splats are keyed by their grid cell (GPU radix sort on the cell key) and
   each cell's splats merge into one parent (segmented reduce): weights w = opacity x projected area; mean = sum(w p) /
   sum(w); covariance = sum(w (Sigma + (p - mean)(p - mean)^T)) / sum(w) (moment matching) -> scale + rotation by a
   3x3 eigen-decomposition; colour/SH = weighted mean; opacity from the coverage (Spark's D > 1 extension, or a
   clamped sum). Stop at one root (or a handful). Store parent index per node, children contiguous (BFS order).
2. **GPU cut.** Each node i, in parallel: render i iff (pixelSize(i) <= tau or i is a leaf) and pixelSize(parent(i)) >
   tau (and parent resident). pixelSize = projected 2 x max scale / depth x focal. tau starts at ~1 px and is adjusted
   frame to frame to hold the splat budget (a GPU count + one scalar readback, or a histogram of pixel sizes to pick
   tau exactly). Output: an index list fed to the existing sort + render (both modes).
3. **Chunked file (.spawnscene v3 or a folder).** Nodes in BFS order, cut into 64K-node chunks, each SceneCodec-encoded
   and gzipped; header = chunk offsets + per-chunk bounds + level range. Chunk 0 = the top of the tree.
4. **Paging + streaming.** A fixed GPU pool (budget from GpuMemoryBudget) of 64K-node pages; a page table maps chunk
   -> page. The cut reports which non-resident chunks it wanted (a GPU flag per chunk, read back as a bit set); the
   loader fetches them (OPFS for a local project, HTTP Range for a hosted scene) nearest-first, LRU-evicts. A node in a
   non-resident chunk renders its nearest resident ancestor (BFS order makes ancestors load first).
5. **Quest/VR.** The same cut per eye pair; the budget from the device (Quest 3 lower).

## Phases and gates

- **A. Tree + cut, in memory** (no streaming): build the tree for a trained scene, render through the GPU cut.
  Gates: (a) tau -> 0 renders exactly the leaves = today's image (bit-for-bit vs the normal path); (b) a CPU oracle for
  the merge (moments preserved: total weight, mean, covariance) and for the cut (every leaf covered exactly once on
  any root-to-leaf path); (c) quality vs budget curve on DrJohnson/bicycle at real views (PSNR of the cut render vs
  the full render at 500K / 1M / 2M budgets); (d) FPS at a fixed budget independent of scene size.
- **B. Chunked format + paging + streaming:** OPFS first, then HTTP Range. Gates: coarse frame within ~1 s of open;
  memory stays at the pool size for a scene 4x larger than the pool; no holes (ancestor fallback) while streaming.
- **C. VR + device budgets.**

## Status (2026-10-05)

Phase A is in: LodMerge/LodTree (CPU oracles + tests), GpuLodTree (GPU build: Truck 997,615 leaves -> 1,343,516 nodes,
36 levels, 1.9 s), the GPU cut in GpuSplatSorter (exact, monotone metric), a drawn counter (one atomic int, read
back without blocking) steering tau to `&lodbudget=N`, and the cut packed and drawn INDIRECT (sentinels sort last,
so the first drawn-count indices are the cut; a one-thread WGSL pass writes the dispatch + draw arguments).

- (a) tau -> 0: 74.3 dB vs the normal path (Truck parked view).
- (c) quality vs budget (sorted, vs the full render): Truck 400K 43.0 / 200K 26.5 / 100K 23.9 / 50K 21.9 dB;
  Bicycle 30K (3.28M) 2M 35.8 / 1M 30.1 / 500K 25.0 dB (a 2M budget then capped at the 1.30M in view).
- (d) uncapped FPS (SPAWNSCENE_CHROME_UNCAPPED=1, `&fpslog=1`), still camera: Truck full 365 -> 500K-budget 405;
  Bicycle full 222 -> 500K-budget 403. Same budget, same FPS, whatever the scene's size.

Open in A: the radix sort still runs over every node (sorting only the cut needs a compacted count on the GPU);
stochastic mode has no cut yet; internal nodes carry no SH (view-independent colour).

### Phase B status (2026-10-05, evening)

In code, CPU-tested; browser checks pending where noted:
- **Layout** (LodLayout / GpuLodLayout): breadth-first, leaves told apart by LOD size 0; GPU layout == CPU oracle
  bit for bit (3K, 20K leaves).
- **Chunks** (LodLayout.ChunkStarts): at most 16K nodes, never splitting a sibling run (split runs made every chunk
  need the next: a streamed view loaded all of them). 16K measured on TruckFull: a view turned away from the truck
  loads 59% of the file vs 78% at 64K (LodLayoutRealSceneTests). Treelet and detail-band x Morton orders were tried
  and were not better than breadth-first.
- **Paged cut** (LodLayout.InCutPaged, GpuSplatSorter.CullAndDistanceLodPagedKernel): a node whose children's chunk
  is missing draws in their place and (when on screen) asks for it. Exact when the resident set is closed under
  ParentChunks (a chunk loads after its nodes' parents' chunks); random closed sets draw every leaf path once, and
  streaming from chunk 0 settles on exactly the full cut, loading exactly the closure of what the cut needs.
- **File** (.spawnscene v3, LodChunkFile): per chunk a SceneCodec frame + raw parent / first child / sphere / LOD
  size, gzipped alone; header lists each chunk's byte range and Needs. Export + full open checked in the browser:
  TruckFull 30K 1.9M splats -> 2.4M nodes, 141 MB, all leaves vs the flat scene 52.5 dB at photo 1's camera
  (re-checked after the 16K / first-child changes: 52.56 dB).
- **Pager** (GpuLodPager, &lodpool=N): fixed pool of pages, chunk 0 pinned; loads the cut's wants biggest stand-in
  first, evicts a page nothing resident needs whose loss shows a stand-in under 2/3 of the incoming one's (never
  before two cut readbacks since its load); chunks from memory or by HTTP Range (header first, chunks on demand;
  tools/_spa_server.js serves Range). LodPagerSimTests run its kernels and rules on the CPU accelerator.
- **Browser (20:15):** v3 export 147 chunks / 142 MB in 19 s; full open, all leaves vs flat 52.56 dB; streamed at
  photo 1, tau 1.5 vs the full open: pool 3M 83.0 dB (138 chunks in 2.5 s), pool 600K 20.5 dB with the near truck
  refined first, 300K 18.1 dB; HTTP Range: 34 KB header + 855 KB chunk 0, on screen in 0.9 s, 72.8 dB settled.
- **Done since (22:00-22:45):** UI - "Open scene file" on the project browser (a v3 streams straight from the picked
  File by slices, 4M-slot pool; v1/v2 imported) and "Export streaming" in the Edit toolbar; no Edit while an LOD tree
  is shown; LOD scenes always draw sorted. The cut is written compacted and only ~1.25x the last drawn count is
  sorted (TruckFull 2.4M pool: ~960K keys, not 2.4M; 59 dB vs before). XR: the cut measures at the eye's pixel
  density (emulated Quest: 1.13M drawn, 144/147 chunks, vs ~795K through the wide head camera), and an XR session on
  an LOD scene without &lodbudget gets a device budget (600K Quest browser, 1.5M tethered; emulator check pending).
- **Multi-room end to end (10-06 02:55):** DrJohnsonFull (263 photos) 30K at a 6 GB budget -> 2,838,151 splats (66 min)
  -> Export streaming: 3,035,408 nodes, 186 chunks, 180 MB in 26.5 s -> streamed over HTTP Range: 44 KB header +
  1 MB chunk 0, on screen in 0.9 s; the home view (one room) needs 85 of 186 chunks (the one-object Truck needed 138
  of 147); a 1M-slot pool (a third of the tree) holds 62 chunks and draws the room at 36.5 dB vs the unconstrained.
  Merged nodes carry SH now (a6e6c31).
- **Next:** a hosted massive scene (GitHub Pages serves Range); Quest on-device check of the budget; scenes larger
  than one GPU (train per block, export per block, one tree).
