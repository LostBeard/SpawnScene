# Porting Brush's growth scheme - plan (2026-10-10)

Why: after the five parity defaults (mean PSNR 27.26 vs Brush 26.95 / gsplat 26.73 over 11 scenes at 7K), Brush still leads
Train by 1.0 dB and Bonsai by 0.54. Every single Brush setting tried in isolation is ruled out (Research/parity-matrix.md:
position lr rate and end, near plane, opacity lr 0.01 / 0.05, no opacity reset, SH from step 0, initial size; scale-lr schedule
and revised opacity WON and are defaults). What is left is the refine algorithm itself.

## Brush v0.3.0 refine (crates/brush-train/src/train.rs:332-540, stats.rs, config.rs)

Every `refine_every` = 200 iterations:

1. **Prune**: opacity < MIN_OPACITY, any log-scale < -15, or a centre more than 10 x the 90th-percentile bounds size from the
   bounds centre.
2. **Replace** (always, even after growth stops): `pruned_count` indices drawn by multinomial sampling of the survivors
   weighted by opacity - relocation, the splat count never drops.
3. **Grow** (until `growth_stop_iter` 15000): candidates = `refine_weight_norm / visible_count > 4e-5`, where
   `refine_weight_norm` is the MAX over views of the per-view gradient norm (`max_pair`, not a sum); `grow_count =
   round(candidates x 0.1) - pruned_count`, capped at max_splats; drawn by multinomial sampling weighted by
   `refine_weight_norm`.
4. **One operation for every selected index** (relocated or grown): sample = rot x N(0, 0.5) x exp(log_scale); the existing
   splat moves by -sample and a copy is appended at +sample; BOTH get log_scale - ln(sqrt 2) and opacity 1 - sqrt(1 - a).
   New rows get zero Adam moments; SH copied.
5. Also every step: noise on the means of low-opacity splats (`mean_noise_weight` 40), tiny opacity (1e-9) and scale (1e-8)
   losses for 90% of the run.

## Ours (GpuDensify + SplatDensityControl)

Every 100 iterations from 500 to the growth stop: AbsGS signal (SUM of per-pixel |dL/dmean2D| over views / visible count) vs
8e-4; deterministic - EVERY candidate grows (DensifyFrac 1); small ones CLONE (an exact copy, now revised opacity), large ones
SPLIT into two children (parent removed, scale / 1.6, revised opacity); prune faint / bloated; opacity reset at 3000; floater
carve every 1000.

## Port, as an opt-in mode (`&growth=brush`)

- **Selection**: weighted sampling without replacement. Brush reads the weights to the CPU (`into_data_async`) and samples
  there. For us: either the same (CPU transfer of n floats every refine - a measured exception, documented) or on the GPU:
  exponential-race keys (u^(1/w), Efraimidis-Spirakis) + the existing radix sort and take the top k - no scan needed (our
  WebGPU exclusive scan is wrong past ~70K, ref-ilgpu-webgpu-scan-wrong-at-scale). Prefer the GPU race.
- **Signal**: keep AbsGS per-pixel abs, but accumulate the MAX over views as Brush does (a second accumulator, or a mode).
- **Operation**: one symmetric split for every selected index (GpuDensify CompactKernel: add[i] = 1 means "relocate/grow":
  parent row offset -sample, appended row +sample, both scale / sqrt 2, both revised opacity). Host oracle mirrors it for
  GpuDensifyTests.
- **Relocation of pruned**: sample survivors weighted by opacity for the pruned count (same race).
- **Schedule**: refine every 200, growth stop at the run's end (7K) or 15000; keep the opacity reset? Brush has none - test
  both; keep the floater carve off in this mode at first (Brush has none) and A/B it.
- **Mean noise**: separate later A/B (McmcNoiseLr exists for &mcmc=1 - check it matches Brush's low-opacity-weighted noise).

## Test plan

Train, Bonsai, Kitchen (the losses) + Bicycle, Room (wins to keep) at 7K vs the five defaults; then all 11 + Bathroom. TrainerGate
unaffected (no shader change); GpuDensifyTests extended for the symmetric split (GPU vs host oracle).
