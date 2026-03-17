# SpawnScene Development Plans

## Current State (2026-03-16)

### What Works
- **Single-image pipeline**: Photo → Depth (DistillAnyDepth Small) → Unproject → Gaussian Splats → Stochastic Renderer @ 60 FPS
- **Multi-view 2D offset**: Feature matching → pixel offset → extended panoramic coverage (near-parallel views)
- **Multi-view world-space (GT)**: Ground truth R/t/K → per-view depth → R^T rotation → world-space fusion (validated with TempleRing)
- **Per-view depth scale alignment**: Project scene center into each view, sample MDE depth, compute scale to match actual camera-to-center distance
- **Studio UI**: WebGPU-rendered UI, project management, OPFS persistence
- **XR/VR**: WebGL fallback path (session starts, rendering not yet verified on headset)

### What Doesn't Work Yet
- **SfM for real photos**: Camera poses from essential matrix + PnP are too noisy for near-parallel views (bathroom photos). Works conceptually but produces garbled fusion.
- **Seam blending**: Hard cut between reference and extension views creates visible depth discontinuity at boundaries
- **Depth scale alignment from SfM**: Only 54 sparse points → noisy scale ratios. Per-view center projection works better with GT.

---

## Priority 1: Single-Image Quality (Quick Wins from BLUNT)

Source: https://github.com/SonnyC56/blunt — Python tool doing similar single-image-to-splat conversion.

### 1.1 EXIF Focal Length
- Read focal length from image EXIF data (SpawnDev.BlazorJS can access this)
- BLUNT uses `max(w,h) * 0.7` as fallback; we use `1.2` — test which is better
- **Files**: `DepthToGaussianKernel.cs` (RunUnprojectAsync), `MultiViewGenerationService.cs`

### 1.2 Flying Pixel Removal
- At depth discontinuities, pixels get unprojected to wrong positions ("floaters")
- BLUNT detects these via depth gradient magnitude and removes them
- We already compute depth gradient for edge sharpness — extend to cull extreme gradients
- **Files**: `DepthToGaussianKernel.cs` (UnprojectAndPackKernel)

### 1.3 Near-Camera Culling
- Remove closest ~5-8% of splats by depth (often noise/artifacts)
- Simple threshold after min/max normalization
- **Files**: `DepthToGaussianKernel.cs`

### 1.4 Edge-Aware Opacity
- Splats at depth edges get reduced opacity (currently all 0.9)
- Use the existing depth gradient to modulate opacity: high gradient → lower alpha
- **Files**: `DepthToGaussianKernel.cs` (line that sets `outPacked[outOff + 9] = 0.9f`)

### 1.5 Median Filtering on Depth
- Smooth depth map before unprojection to reduce noise
- ILGPU kernel: 3x3 or 5x5 median filter on the depth buffer
- **Files**: New kernel in `DepthToGaussianKernel.cs` or `DepthEstimationService.cs`

### 1.6 Depth Anything V3 Upgrade
- ONNX models available: `onnx-community/depth-anything-v3-small` (105MB, Apache 2.0)
- Also base and large variants available
- DAv3 provides **metric depth** (meters from camera) — eliminates depth scale alignment problem entirely
- If metric, multi-view fusion becomes: `pos_world = R^T * (metric_pos_cam) + C_world` — no scale factor needed
- Need to verify: does the ONNX model output metric depth directly, or does it need post-processing?
- Also may output confidence maps / semantic segments (useful for adaptive splat scale)
- Normal-based flying pixel check: compute surface normals from depth, cull pixels with normals nearly perpendicular to camera ray (grazing angle)
- **Files**: `DepthEstimationService.cs` (add DAv3 model option), `DepthToGaussianKernel.cs`

### 1.7 Kernel Struct Refactor
- Replace `float[] p` parameter array with a proper struct type
- SpawnDev.ILGPU supports struct kernel params (bypasses parameter count limit)
- Eliminates all `p[0]`, `p[10]`, `p[14]` magic indices
- **Files**: `DepthToGaussianKernel.cs` — define `SplatKernelParams` struct

---

## Priority 2: Multi-View Fusion Improvements

### 2.1 Better SfM for Wide-Baseline Views
- TempleRing GT validates the world-space kernel math is correct
- SfM needs: better RANSAC thresholds, cheirality check, more features
- Test with TempleRing (SfM vs GT) to measure pose error
- **Files**: `SfmReconstructor.cs`, `GpuSfmKernels.cs`

### 2.2 Multi-Point Depth Scale Alignment
- Current: project scene center into view, sample one MDE depth → one scale ratio
- Better: project multiple SfM sparse points → multiple ratios → robust median/RANSAC
- Or: least-squares fit `D_sfm = s * D_mde + b` per view using multiple correspondences
- **Files**: `MultiViewGenerationService.cs`

### 2.3 Seam Blending for 2D Offset Mode
- Current: hard cut between reference and extension → visible seam
- Better: overlap band (100-200px) where both views contribute with opacity gradient
- Reference fades from 1.0→0.0, extension fades from 0.0→1.0 across the band
- **Files**: `DepthToGaussianKernel.cs` (exclusion zone → blend zone)

### 2.4 Homography Instead of Translation
- 2D offset (median dx, dy) assumes pure translation between views
- For rotating camera (panning), correct transformation is a homography
- Compute homography from 4+ matched feature pairs using DLT + RANSAC
- Near objects shift more than far objects (parallax) — homography handles this for planar scenes
- **Files**: New `HomographyEstimator.cs` or extend `MultiViewGenerationService.cs`

---

## Priority 3: SOG Format + Streaming LOD

### 3.1 Morton Code Spatial Organization
- Assign each splat to a spatial chunk using Z-order curves
- ILGPU kernel: quantize position → interleave bits → Morton code
- Radix sort by Morton code (reuse GpuSplatSorter)
- Build chunk table: (mortonCode, startIndex, count)[]
- **Files**: New `SpatialOrganizer.cs`

### 3.2 Frustum Culling by Chunk
- Before rendering, test chunk AABBs against camera frustum
- Build active chunk list per frame
- **Files**: `GpuGaussianRenderer.cs`

### 3.3 Distance-Based LOD
- Budget ~14-20M active splats for 60 FPS
- Near chunks: full density, Medium: 1/4, Far: 1/16
- **Files**: `GpuGaussianRenderer.cs`

### 3.4 SOG File Format
- Header + chunk index + quantized splat data
- Save/load via OPFS
- **Files**: New `SogFormat.cs`, extend `ProjectService.cs`

---

## Priority 4: Video Input

### 4.1 Keyframe Extraction
- HTMLVideoElement → seek at intervals → OffscreenCanvas → RGBA
- Score frames by sharpness (Laplacian variance) and diversity (feature distance)
- **Files**: New `VideoFrameExtractor.cs`

### 4.2 Studio Integration
- Accept video files in file picker
- Extract keyframes → add as project sources → run multi-view pipeline
- **Files**: `Studio.Projects.cs`, `Studio.UI.cs`

---

## Future: WebRTC Multi-Device Scanning
- Multiple phones stream camera feeds to PC via WebRTC
- PC runs incremental SfM + scene generation in real-time
- VR headset on PC views the scene being built
- See NOTES.md for full architecture notes

---

## Test Datasets (wwwroot/datasets/)
| Dataset | Images | Ground Truth | Notes |
|---------|--------|-------------|-------|
| TempleRing | 16 (every 3rd) | R/t/K in templeR_par.txt | Middlebury benchmark, ring around temple |
| DinoSparseRing | 16 | Possibly (check) | Similar ring capture |
| Skull | 75 | None | Real photos, good coverage |
| Bathroom | 16+ | None | Phone photos, near-parallel, challenging |
| SouthBuilding | 22 | None | Outdoor, wide baseline |
| SmallPlastic | 14 | None | Small object |

---

## Reference Projects
- **SuperSplat** (PlayCanvas): Editor/viewer for pre-made splats, SOG format, walk mode, annotations
- **BLUNT**: Python single-image-to-splat tool, good quality improvements (flying pixel removal, edge-aware opacity, EXIF focal length)
- **DepthSplat** (CVPR 2025): Multi-view depth + transformer → high-quality splats
- **Splatt3r**: Pose-free stereo pairs → splats at 4 FPS
- **DUSt3R/MASt3R**: Foundation models for dense 3D from 2+ images
