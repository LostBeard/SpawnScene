using System.Numerics;
using SpawnScene.Models;
using SpawnScene.Services;

namespace SpawnScene.Pages;

/// <summary>
/// Run the whole pipeline on an UNPOSED capture, which is what the product is actually for.
///
/// TempleRing ships a calibration file. Nothing else does - not Bathroom, not Skull, not
/// SouthBuilding, not SmallPlastic, not DinoSparseRing - so until now the optimiser could only
/// ever touch the one dataset that is least like the target. Bathroom is the honest test: 35
/// handheld phone frames of a room, 13 megapixels each, 34 portrait and one landscape, no EXIF
/// orientation, no poses.
///
/// This drives: load dataset -> pose cascade (SfM, then DAv3 extrinsics) -> depth init ->
/// photometric optimisation -> report. It reports rather than asserts, because the question it
/// exists to answer is "are the recovered poses good enough to optimise against", and a
/// pass/fail threshold would just be a guess dressed up as a gate.
///
/// Entry: <c>/studio?autotest=dataset&amp;name=Bathroom&amp;train=1600&amp;geom=1</c>
/// </summary>
public partial class Studio
{
    private async Task RunDatasetAutotestAsync(
        string datasetName, int trainIters, bool optimiseGeometry, int maxTrainDimension,
        string posePreference = "auto", int depthPatchesPerSide = DepthEstimationService.SafeMultiViewPatches)
    {
        Console.WriteLine(
            $"[Dataset] starting name={datasetName} train={trainIters} geom={optimiseGeometry} " +
            $"maxDim={maxTrainDimension} poses={posePreference} patches={depthPatchesPerSide}");
        try
        {
            if (!_gpuService.IsInitialized) await _gpuService.InitializeAsync();

            // -- 1. Load the photographs --
            var t0 = DateTime.UtcNow;
            await _importService.LoadSampleDatasetAsync(datasetName);
            var images = _importService.Images.ToList();
            if (images.Count < 2)
            {
                Console.WriteLine($"[Dataset] FAIL: {datasetName} gave {images.Count} image(s)");
                return;
            }
            Console.WriteLine(
                $"[Dataset] {images.Count} images in {(DateTime.UtcNow - t0).TotalSeconds:F1}s, " +
                $"first {images[0].Width}x{images[0].Height} ({images[0].SourceUrl})");

            // MEASURED WORSE - left off by default.
            //
            // An aspect-matched input looked like a free win: a square spends part of its patch
            // budget on letterbox padding, and 32x43 patches costs the same as 37x37. On
            // Bathroom it cost 4 dB (held out 8.57 -> 4.34) and changed which views survived
            // the consistency screen.
            //
            // The likely reason is that this ONNX export's position embeddings are tuned for
            // the 37x37 grid it was exported at. DA3 itself interpolates them for other sizes;
            // an export pinned to one grid need not. So the patch budget is only worth raising
            // in SQUARE steps until that is confirmed, and the letterbox - which measured
            // BETTER - is what handles aspect.
            // Square, always: the grid ASPECT is the variable that measured 4 dB worse, while
            // scaling a square grid is what recovered the detail in the living-room maps.
            DepthEstimationService.SetSquareInput(depthPatchesPerSide);

            // -- 2. Poses + depth init, through the ordinary cascade --
            void OnStatus() => Console.WriteLine($"[Dataset] {_multiViewService.Status}");
            _multiViewService.OnStatusChanged += OnStatus;
            (ILGPU.Runtime.MemoryBuffer1D<float, ILGPU.Stride1D.Dense> buf, int count)? result;
            try
            {
                t0 = DateTime.UtcNow;
                _multiViewService.PosePreference = posePreference;
                result = await _multiViewService.GenerateAsync(images, subsample: 2, edgeSharpness: 0.3f);
            }
            finally
            {
                _multiViewService.OnStatusChanged -= OnStatus;
            }

            if (result == null)
            {
                Console.WriteLine($"[Dataset] FAIL: generation returned nothing - {_multiViewService.Status}");
                return;
            }
            var (packedBuf, splatCount) = result.Value;
            Console.WriteLine(
                $"[Dataset] {splatCount:N0} splats in {(DateTime.UtcNow - t0).TotalSeconds:F1}s, " +
                $"pose source = {_multiViewService.LastPoseSource}");

            await _gpuRenderer.UploadSceneFromGpuBuffer(packedBuf, splatCount);

            var scene = new GaussianScene
            {
                GpuSplatCount = splatCount,
                SourceName = "multi-view",
            };
            // The dataset loader fetched these over HTTP, so they are re-fetchable by URL and
            // do not need the project store.
            RecordTrainingViews(scene, images, fromProjectStore: false);

            _renderService.SetActiveSceneGpuLoaded(scene);
            _sceneManager.ActiveScene = scene;
            _gpuRenderer.AdaptiveResMode = AdaptiveResMode.ForceFull;
            _gpuRenderer.RenderMode = SplatRenderMode.Sorted;
            _state = StudioState.SceneViewer;

            // Stand the viewer where one of the photographs was actually taken.
            //
            // Two reasons not to use FitToScene here. It leaves the camera at the origin looking
            // down -Z when there is nothing to fit to, and an SfM reconstruction can be anywhere
            // - Bathroom lands at z 0.49 to 1.44, entirely behind that - so the viewer renders
            // half a million splats at 58 fps and shows an empty frame. And its multi-view
            // branch aims at a HARDCODED TempleRing point (0.028, 0.042, -0.054), which has
            // nothing to do with a bathroom.
            //
            // A capture pose needs no heuristic: it is a viewpoint that definitely saw the
            // subject, because a photograph was taken from it.
            if (scene.TrainingCameras.Count > 0 && _cameraController != null)
            {
                var seat = scene.TrainingCameras[0];
                _cameraController.SetPose(seat.Position, seat.Forward, seat.Up);
                Console.WriteLine(
                    $"[Dataset] viewer seated at capture pose 0: " +
                    $"pos=({seat.Position.X:F3},{seat.Position.Y:F3},{seat.Position.Z:F3}) " +
                    $"fwd=({seat.Forward.X:F3},{seat.Forward.Y:F3},{seat.Forward.Z:F3})");
            }
            else
            {
                _cameraController?.FitToScene();
            }

            if (scene.TrainingViews.Count == 0)
            {
                // Still capture a frame. This branch has a full initialisation on screen - the
                // run that hit it had 242,440 splats - and a finding with no picture beside it is
                // how three runs got scored and committed while the viewer was blank. The
                // question about any reconstruction is what it LOOKS like, especially this one.
                _hideUiOverlay = true;
                await Task.Delay(1500);
                Console.WriteLine("[Dataset] READY-FOR-CAPTURE");
                await Task.Delay(2500);
                Console.WriteLine(
                    "[Dataset] DONE (no optimisation): the cascade produced no usable poses. " +
                    "That is the finding, not a failure of this test.");
                return;
            }

            // -- 3. Optimise --
            if (trainIters > 0)
                await TrainOnTrainingViewsAsync(
                    trainIters, optimiseGeometry: optimiseGeometry,
                    maxTrainDimension: maxTrainDimension);

            // Announce before DONE so the harness can capture a frame of the finished scene.
            _hideUiOverlay = true;
            await Task.Delay(1500);
            Console.WriteLine("[Dataset] READY-FOR-CAPTURE");
            await Task.Delay(2500);

            // Then LOOK AROUND. A capture-pose render is close to a re-projection of the photo
            // it was taken from, so it flatters any reconstruction; the question a room has to
            // answer is what it looks like from somewhere nobody stood. TJ found the scene
            // rotated and tumbling this way while every number said it was fine.
            await CaptureFreeViewsAsync(scene);

            Console.WriteLine("[Dataset] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] FAIL: {ex}");
        }
    }

    /// <summary>
    /// Render from poses nobody captured from: step away from the seat and look back.
    ///
    /// Deliberately modest offsets. The point is not a flythrough, it is to catch a
    /// reconstruction that only holds together from the exact pose it was seated at - which is
    /// what an unaligned world frame, a broken depth sort or a set of per-view shells all look
    /// like, and what a capture-pose screenshot cannot show.
    /// </summary>
    private async Task CaptureFreeViewsAsync(GaussianScene scene)
    {
        if (_cameraController == null || scene.TrainingCameras.Count == 0) return;

        var seat = scene.TrainingCameras[0];
        var fwd = Vector3.Normalize(seat.Forward);
        var up = Vector3.Normalize(seat.Up);
        var right = Vector3.Normalize(Vector3.Cross(fwd, up));

        // Scale the steps to the scene so this means the same thing on any capture.
        float span = 0.25f;
        if (scene.TrainingCameras.Count > 1)
        {
            var centroid = Vector3.Zero;
            foreach (var c in scene.TrainingCameras) centroid += c.Position;
            centroid /= scene.TrainingCameras.Count;
            float spread = 0f;
            foreach (var c in scene.TrainingCameras)
                spread = MathF.Max(spread, Vector3.Distance(c.Position, centroid));
            if (spread > 1e-3f) span = spread * 0.35f;
        }

        var moves = new (string Name, Vector3 Offset, float Yaw)[]
        {
            ("left",  -right * span, 0f),
            ("right",  right * span, 0f),
            ("back",  -fwd * span,   0f),
            ("up",     up * span * 0.5f, 0f),
            ("turned", Vector3.Zero, 0.35f),
        };

        foreach (var (name, offset, yaw) in moves)
        {
            var look = fwd;
            if (yaw != 0f)
            {
                var q = Quaternion.CreateFromAxisAngle(up, yaw);
                look = Vector3.Normalize(Vector3.Transform(fwd, q));
            }
            _cameraController.SetPose(seat.Position + offset, look, up);
            await Task.Delay(1200);
            Console.WriteLine($"[Dataset] READY-FOR-CAPTURE free-{name}");
            await Task.Delay(1800);
        }
    }
}
