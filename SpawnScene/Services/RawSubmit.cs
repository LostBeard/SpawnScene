using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnScene.Services;

/// <summary>
/// Every raw <c>queue.submit</c> in SpawnScene goes through here: ILGPU's pending work is submitted FIRST.
/// </summary>
/// <remarks>
/// SpawnDev.ILGPU's WebGPU stream batches its work - kernels, clears, buffer-to-buffer copies and, since 8eaf762a,
/// small CopyFromCPU uploads ("ordered batched uploads") - and submits the batch at the next flush. A raw submit goes
/// to the queue at once, so it overtakes anything ILGPU still holds. MEASURED 2026-10-02 in the trainer gate: a 13 KB
/// CopyFromCPU of the splats, then the raw init_logits dispatch, read a buffer the upload had not reached yet; every
/// opacity logit came from 0 and the next step zeroed all 240 opacities (the "densify frustum denominator" stage,
/// failing since ILGPU's upload batching). A flush with nothing pending returns at once, so this costs nothing when
/// the order is already right.
/// </remarks>
public static class RawSubmit
{
    /// <summary>Flush <paramref name="accelerator"/>'s pending ILGPU work, then submit <paramref name="commands"/>.</summary>
    public static void Submit(WebGPUAccelerator? accelerator, GPUQueue queue, GPUCommandBuffer[] commands)
    {
        accelerator?.FlushPendingCommands();
        queue.Submit(commands);
    }
}
