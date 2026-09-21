# Datasets: what we have, and what each one can actually prove

A dataset is only useful for the question it can answer. The trap this project kept falling into
is measuring on a capture that cannot separate two error sources, then drawing a conclusion about
one of them.

## The shape of the problem

| | has images | has poses | is a ROOM | what it can prove |
|---|---|---|---|---|
| TempleRing | yes | **yes** (Middlebury) | no, object on a turntable | pose accuracy, regression fixture |
| Bathroom | yes | no | yes | nothing on its own - see below |
| Deep Blending drjohnson / playroom | yes | **yes** (COLMAP) | **yes** | optimiser alone, poses on a room |
| Tanks and Temples train / truck | yes | **yes** (COLMAP) | no, outdoor | generalisation beyond indoor |
| reference 3DGS `.ply` outputs | no | no | - | **the renderer**, and a calibrated ceiling |
| InteriorGS | **no** | no | yes, synthetic | renderer only; starts where we finish |

**Bathroom is the honest target and the worst measuring instrument.** It has no ground-truth poses,
so every number it produces mixes *our poses are wrong* with *our optimiser is wrong*. Held-out
12.41 dB on Bathroom is not attributable to anything. It is still the right thing to get working,
because it is what the product is for; it is just not where a conclusion should come from.

**TempleRing is a fine regression fixture and a poor proxy for the goal.** Object on a turntable,
cameras outside looking in, 61% black frame (which flatters PSNR), 16 views. It is also the only
posed data this project had for a long time, which is why "DAv3 poses are accurate" was known for
an object and unknown for a room.

## Deep Blending + Tanks and Temples (`tandt_db`)

The standard 3DGS evaluation bundle from the INRIA release. Local at
`C:\Users\TJ\Downloads\tandt_db`.

| scene | images | resolution | kind |
|---|---|---|---|
| `db/drjohnson` | 263 | 1332x876 | indoor room |
| `db/playroom` | 225 | 1264x832 | indoor room |
| `tandt/train` | 301 | 980x545 | outdoor |
| `tandt/truck` | 251 | 979x546 | outdoor |

Each ships `sparse/0/{cameras,images,points3D}.bin` - a COLMAP reconstruction with a single
undistorted PINHOLE camera, so the intrinsics map straight onto `CameraParams` with no undistortion
step.

**This is what breaks the Bathroom confound.** Two separate measurements become possible:

1. Feed the COLMAP poses (`?gtposes=1`) and the optimiser is measured **on its own**, on a room.
2. Run our pose pipeline and fit its cameras to COLMAP, and our **poses** are measured on a room -
   the question TempleRing cannot answer.

Also note the view count. The literature needs 100-300 views for 27-29 dB; these sit in that band,
where Bathroom's 34 views cap the honest expectation nearer 20.

### Converting one

    python tools/colmap_to_dataset.py C:/Users/TJ/Downloads/tandt_db/db/drjohnson DrJohnson --every 3

Writes `SpawnScene/wwwroot/datasets/<Name>/{manifest.json,poses.par}`. The poses are written in the
Middlebury `par` format that `WorldSpaceGeometry.ParseMiddleburyParams` already reads, because
COLMAP stores the same quantities in the same convention (world-to-camera R and t, OpenCV axes) -
one pose parser to be wrong in rather than two.

The converter **proves its own conversion** rather than trusting a quaternion ordering: COLMAP
ships the 2D observations that tie its points to its poses, so it reprojects them and refuses above
2 px. drjohnson comes out at **0.586 px mean over 7,896 observations**. A transposed rotation or a
camera-to-world mix-up would produce a plausible-looking file that puts every camera somewhere
wrong, and this catches that.

Images are **not copied** - drjohnson alone is 168 MB and anything under `wwwroot` is recopied on
every publish. The manifest records the source directory and `tools/_spa_server.js` mounts it.

⚠ **COLMAP's world orientation is arbitrary**, exactly like DAv3's - drjohnson's cameras come back
pointing along +/-Y. Posed does not mean upright. Both generation paths gravity-align when the
cameras agree strongly enough on an up direction (see below).

## Reference 3DGS outputs (Voxel51/gaussian_splatting on HuggingFace)

Pre-trained `.ply` for **exactly these four scenes** at 7k and 30k iterations, 3.49 GB for the set,
Apache 2.0. No images, no poses - useless for training, which is not the point.

Two things nothing else here provides:

1. **A known-good splat scene to put through our display renderer.** That renderer is the one
   unvalidated link that has already produced a wrong conclusion (see `validation-strategy.md`).
   If a reference scene renders correctly in it, the renderer is sound; if it tips over, the bug is
   ours and provably not the generator's.
2. **A calibrated ceiling on identical data.** We hold the same images and poses locally, so
   rendering their 30k result from our held-out poses and scoring it with OUR PSNR and SSIM turns
   "the paper reports ~28.8 dB" into "the reference implementation scores X under our scorer, on
   our split, at our resolution". That removes every difference of metric definition, test split
   and resolution in one step - and those differences are usually larger than the gap being argued
   about.

## InteriorGS (manycore-research)

1,000 synthetic indoor scenes as pre-trained Gaussian splats with semantic annotations, object
bounding boxes and floorplans. **Ships no raw images** - it is derived from renders of handcrafted
3D environments. It starts where our pipeline finishes, so it cannot test reconstruction at all.

Useful for the same thing as the reference outputs (exercising the renderer, at much greater
variety), and for anything semantic later. Not a reconstruction benchmark.

## HuggingFace access

The standing rule that nothing requests `huggingface.co` is about **the software we write** -
clients and tests go through `hub.spawndev.com`, which exists for caching, CORS and rate limiting.
It is not a ban on reading a dataset page while deciding whether it is worth using. They throttle
quickly, so fetch sparingly.

## What is missing

- **A posed capture with the camera actually walking through a room** at our target view count,
  with gravity. Deep Blending is the closest available.
- **Video**, as opposed to stills. The stated goal is rooms from video and streams, and every
  dataset here is a photo set.
- A capture where the **depth model's failure modes** can be isolated - textureless walls, mirrors
  and glass are exactly what a bathroom has and what monocular depth handles worst.
