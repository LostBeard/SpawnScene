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
        string posePreference = "auto")
    {
        Console.WriteLine(
            $"[Dataset] starting name={datasetName} train={trainIters} geom={optimiseGeometry} " +
            $"maxDim={maxTrainDimension} poses={posePreference}");
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
            Console.WriteLine("[Dataset] DONE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Dataset] FAIL: {ex}");
        }
    }
}
