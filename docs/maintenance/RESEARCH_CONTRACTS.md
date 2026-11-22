# Research and training contracts

Read [the method contract](../predictive_world.md) for the supported route and
[architecture](ARCHITECTURE.md) for module ownership.

## Inputs and objectives

- Deployment inputs remain causal. Recorded future observations supervise training.
- Preserve masks, units, timestamps, horizons, protocol adapters, valid measurement
  signals and dataset split/exclusion provenance.
- Final task adaptation uses the current-state input contract. Existing pretraining
  source bundles retain their recorded history/input ABI and explicit migrations.
- Qwen weights, feature geometry, image preprocessing and cache hashes are immutable
  identities. A runtime/processor change needs an explicit verified transition.
- Changes to objectives, sampling, schedules or effective batch are method changes.
  Record a versioned contract and new output; do not silently relabel older runs.
- Seed zero is a valid explicit seed. Preserve the seed actually used by the process.

## Numerical safety

- Preserve `qk_norm_v1`, FP32 history attention, bounded state injection and
  zero-initialized prediction heads.
- Keep strict architecture/output-semantics checks even when tensor shapes match.
- Execute synchronized finite-loss checks, backward, read-only FP64 gradient-norm
  checks, clipping and optimizer/scheduler updates in that order.
- Do not use numerical replacement, sample skipping or exception swallowing to
  hide invalid input, non-finite losses or bad gradients.
- Keep collective order consistent. Input-error synchronization cannot wrap
  forward/backward code that itself contains collectives.
- Sampled numerical diagnostics are read-only and do not certify learned stability.

## Immutable artifacts and execution

- Loader/process-group timeouts default to 600 seconds. Validate finite positive
  values and group timeout >= loader timeout; equality remains supported.
- Publish configs/checkpoints atomically and reject conflicting existing content.
  Resume requires the recorded complete checkpoint and compatible optimizer/data
  cursor identities.
- Keep source/config/data/teacher hashes with results. Relocation maps change
  active paths only; they do not rewrite historical provenance.
- Imports must not start jobs, open hardware or initialize GPU resources.

The retired endpoint and goal-intent execution branches are unsupported. Their
original source and experiment artifacts remain available in the local cleanup
backup; old result records retain their original interpretation.
