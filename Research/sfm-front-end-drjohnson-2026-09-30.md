# SfM front end vs DrJohnson - research and measurements (2026-09-30, Tuvok)

Question: SpawnScene's global SfM (GLOMAP-style: rotation averaging + global positioning + GPU BA) is COLMAP-grade on
TruckFull (BA 0.08% of COLMAP, held-out 21.63 dB, SpawnScene `2a57c21`) but fails on DrJohnson (~100% off). Is global
SfM the wrong approach for unordered captures, or is something upstream failing?

Answer so far: **global SfM is not the problem. The front end (pair geometry) is, and SpawnScene's 44-view DrJohnson
subset is about half as dense as the captures COLMAP / GLOMAP are evaluated on.** Do not switch to incremental SfM on
this evidence (Tuvok proposed it once today and withdrew it after reading the GLOMAP paper).

## What GLOMAP actually is (Pan et al. 2024, "Global Structure-from-Motion Revisited", arXiv 2407.20219)

GLOMAP matches COLMAP's accuracy on unordered collections (ETH3D, LaMAR, IMC 2023, MIP360) - but it runs on COLMAP's
front end:

| Stage | GLOMAP / COLMAP | SpawnScene (2a57c21) |
|---|---|---|
| Features | RootSIFT (COLMAP, ~8k/image) | FAST-9 + BRIEF-256, 2000 (ORB-style pyramid opt-in) |
| Matching | vocab-tree / exhaustive, ratio test | exhaustive Hamming, ratio 0.75 |
| Two-view geometry | H / F / E model selection; E when calibrated | F-RANSAC only, >= 15 inliers at 2 px |
| Match filtering | drop matches near the epipoles or with small triangulation angle | none |
| Intrinsics | view-graph calibration (Sweeney et al.): focal refined from the F matrices | DAv3 focal (DrJohnson 812 vs COLMAP 1035, -22%) |
| Relative pose | E decomposition + cheirality | E = K^T F K decomposition + cheirality |
| Rotation averaging | Chatterjee et al. robust (Huber) | chordal IRLS, annealed delta |
| View-graph filtering | edges whose relative rotation disagrees with the averaged ones are removed | IRLS weights only |
| Positioning | sum rho(|v_ik - d_ik (X_k - c_i)|), random start in [-1,1], d = 1 start, Huber, LM (Ceres) | same formulation (GlobalPositioningRobust / GpuGlobalPositioner) |
| BA | iterative BA + retriangulation + filtering until < 0.1% tracks filtered | GPU BA + PnP re-registration |

## Measurements (DrJohnson, 44 views; ground truth = the dataset's COLMAP model)

Tooling: `tools/gt_matching_export.py` -> `_scratch/djgt` (views + true pairs with F and relative R);
`SpawnScene.Tests/DrJohnsonMatchingTests.cs`. 119 of 946 pairs truly overlap (>= 30 shared COLMAP points).
Measurement validated on TruckFull adjacent frames: pair rotation error median 0.23 deg.

1. **Matching recall** (pairs with 15+ correct matches under the true F): FAST+BRIEF 7 / 119; ORB-style opt-in
   (2000 + 2000, oriented) 47 at ratio 0.75, 76 at 0.9; OpenCV SIFT-8000 79.
2. **False pairs are separable**: our F-RANSAC (15 inliers) verifies 52 true + 10 false; at 25 inliers 38 + 2.
   SIFT + 5-point E: false edges median 19 inliers, true edges median 41.
3. **Pair rotation error on TRUE pairs** (median / p75):
   - ours (ORB mode + F-RANSAC): 12.8 deg at COLMAP's focal, 14.7 at DAv3's
   - OpenCV ORB 5000: F->E 24.7 / 62, 5-pt E 17.6 / 83
   - OpenCV SIFT 4000: F->E 16.3 / 43, **5-pt E 7.2 / 29 at the true focal**, 14.8 / 47 at 812
4. **Rotation averaging** (validated: ground-truth relative rotations -> 0.00 deg) on SIFT + 5-pt E edges, true
   focal: all 210 edges 34 deg median absolute error; only the 90 TRUE edges 9.6 / 86 (p75); GLOMAP-style edge filtering
   (5-15 deg) did not rescue it - with this many bad edges the averaged solution the filter measures against is wrong.
5. **Not planar degeneracy**: true pairs split by homography/E inlier ratio: >= 0.8 (planar-dominated) 3.7 deg median,
   < 0.5 (non-planar) 7.8 / 94 - the large errors are wrong/flipped E solutions on thin evidence, not planes.
6. **The subset is sparse**: best-overlap neighbour shares a median 279 COLMAP points (p10 54) at 36 deg in the 44-view
   subset, vs 397 (p10 116) at 23 deg for every 3rd photo (88 views) and 544 (p10 165) at 17 deg for the full 263-view
   COLMAP set.

## Conclusions

- The 44-view DrJohnson subset is a wide-baseline problem beyond what classical features + two-view geometry resolve:
  even SIFT + 5-point E at the true focal leaves p75 pair errors of ~29 deg, and averaging cannot fix that.
- The GLOMAP front-end gaps are real and each matters (features, E with calibration, focal calibration, inlier
  thresholds), but closing them is not shown to be sufficient for THIS subset.
- The cheapest decisive experiment is density: re-run the same measurements with 88 views (every 3rd) - if pair poses
  and averaged rotations become good, the subset (not the method) is the limit.
- For captures as sparse as the 44-view subset, the literature answer is learned wide-baseline matching (SuperPoint +
  LightGlue, LoFTR-class), which SpawnDev.ILGPU.ML could run on the GPU.

## Proposed order

1. Density experiment (88 views) with the Python/C# workbench - no GPU needed.
2. Front end to GLOMAP parity, measured at each step on TruckFull (must not regress) and DrJohnson: stricter inlier
   minimum (25-30), 5-point E with known K + H model selection, view-graph focal calibration, view-graph filtering.
3. If the 44-view target stands: a learned matcher on ILGPU.ML.

## Follow-up: density and loop consistency (same day)

Same front end (SIFT 4000 + 5-pt E with K, >= 30 cheirality inliers, NO ground truth in the pairing):

| View set | Focal | Edges | Pair error median / p75 | Averaged, all cams: abs rot median / p75 |
|---|---|---|---|---|
| 44 subset | 1035.5 (COLMAP) | 82 | 6.2 / 92 | 33.1 / 131 |
| 88 (every 3rd) | 1035.5 | 387 | 7.9 / 112 | **2.8** / 110 |
| 44 subset | 812 (DAv3) | 82 | 18.0 / 93 | 91.8 / 129 |
| 88 | 812 | 365 | 17.4 / 111 | 11.8 / 79 |

Pair-error histogram (88, COLMAP focal): 174 edges < 5 deg, 39 in 5-15, but 121 (31%) > 90 deg spread over 90-180 -
not the E twisted-pair ambiguity (only 20 near 180): **repeated structure** (one wall matched to the opposite wall;
geometrically consistent, so no two-view check can reject it).

**Loop (triplet) consistency** (Zach et al. 2010; GLOMAP's view-graph filtering is the same principle): keep an edge only
if some triangle through it composes to within 5 deg of the identity.

| View set, focal | Filter | Edges | Kept pair error median / p75 | Cams connected | Abs rot median / p75 |
|---|---|---|---|---|---|
| 88, 1035.5 | >= 1 consistent triangle | 160 | **1.39 / 3.00** | 50 / 88 | 4.3 / 6.1 |
| 88, 1035.5 | >= 2 | 102 | 1.29 / 2.48 | 17 / 88 | **1.07 / 1.72** |
| 44, 1035.5 | >= 1 | 32 | 1.38 / 3.70 | 11 / 44 | 4.4 / 6.7 |
| 88, 812 | >= 1 | 97 | 5.24 / 8.17 | 34 / 88 | 20.3 / 123 |

Reading: the loop filter turns a garbage view graph into an accurate CORE, but not all cameras are in it. That is the
shape of a hybrid: global rotations + positioning on the loop-consistent core, every other camera registered by PnP
against the core's points (SpawnScene's re-registration, which since `2a57c21` also places views the cascade missed).
Focal calibration is REQUIRED: at DAv3's 812 the kept pair error is 3-4x worse.

## Revised order

1. Loop-consistency filter in GlobalSfmInit before rotation averaging; cameras outside the consistent core go to
   re-registration. Measure: TruckFull must hold 0.08% / 21.63 dB; DrJohnson core size + accuracy.
2. Focal calibration from the view graph (GLOMAP / Sweeney et al.) - DAv3's focal is 22% off on DrJohnson.
3. 5-point E with known K + H model selection in pair verification.
4. SIFT-class or learned descriptors (ORB-mode pair precision is far below SIFT's on DrJohnson).
5. Density: the 44-view subset leaves an 11-camera core even with SIFT; 88 views give 50. The product may need a
   minimum-coverage capture guide as much as a better solver.
