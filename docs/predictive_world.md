# Zeta-0 method contract

The supported route is `predictive_world`, with a shared Zeta core and two host
backends. New joint recipes use FULL+ERT dual response, body supervision and
explicit v6/raw controls. [Backend selection](backends.md) describes native Psi0
and GR00T differences. Historical v1–v5 checkpoints keep their recorded objective
identity; the v5 adaptation profile below remains available for those artifacts.
Experiment evidence is [external](artifacts.md#historical-records).

## Current state, body supervision and dual response

The causal inputs are current RGB, the measured current state, instructions and
execution history and any committed action prefix. Real G1/BrainCo data uses
state41/motion76. Body supervision uses measurements marked valid in the
source schema.

The future representation has three groups ending at +0.2/+0.4/+0.6 seconds,
with eight 256-dimensional tokens per group. E encodes recorded future visual
features and measured body trajectories; an EMA encoder supplies prediction
labels. D reads out absolute scene/body targets and temporal differences.
The trainable teacher T also shapes E through demonstration action loss.
Frozen Qwen features and their exact preprocessing/cache identity are shared.

H builds causal conditioning. The same F predicts future latents in two
training modes: `prior` uses causal inputs and any committed prefix; `world`
additionally conditions on demonstrated future actions. World action coverage
masks prediction eligibility without dropping policy rows. Deployment uses prior.
G consumes causal predicted latents plus current vision-language context/state.
T and G are distinct flow-matching action models.

New runs explicitly set `training.latent_prediction_space=raw`: F outputs raw
future latents and minimizes masked MSE against the stop-gradient EMA target.
Readout, response transfer and G consume this same raw prediction. No latent
dataset statistics are fitted or updated; `normalizer_calibration_updates=0`.
Target EMA, network normalization, representation regularization and the shared
state/action/body normalization remain. Forecast warmup/ramp can be set separately;
the new raw preparation defaults both to zero and retains learning-rate warmup.

Omitting this field preserves the historical whitened objective and checkpoint
identity. A whitened joint checkpoint needs the explicit source-bound
`init_latent_prediction_migration` and persisted F output affine. If the old F
returns `a*f(h)+b`, the migrated F returns `sigma*a*f(h)+mu+sigma*b` using the
source statistics once. No new statistics are estimated, and the raw prediction,
readouts and actions are preserved at initialization. Legacy normalizer buffers
remain in bundles as provenance/compatibility data and are unused by raw training
or inference. Raw and whitened loss contracts cannot be silently interchanged.

Let `v_ref = stopgrad(T(X, z_EMA))`. With identical noisy actions, flow time,
prefix, validity masks and observation context, dual response is:

```text
0.5 * MSE(T_fixed(X, z_pred), v_ref)
+ 0.5 * MSE(G_fixed(X, z_pred), v_ref)
```

The outer response coefficient is 1, body coefficient 1, temporal coefficient
0.25 and forecast-readout coefficient 0.25. Response applies on prior updates.
The readers' parameters and buffers are detached only within response transfer;
latent input gradients train the causal prediction path. Ordinary teacher/student
demonstration losses continue to update T/G. G's ordinary action gradient also
trains its prediction path. Equal response coefficients do not imply equal
numerical loss or gradient magnitudes.

## Training composition and data

- `world_model` learns E/EMA/D/T and H/F. Only the legacy whitened route fits a
  latent normalizer; forecast warmup/ramp are separate configuration fields.
- `joint` trains the representation, forecasting and executing-policy objectives.
  It starts from a complete world bundle or a verified pretrained joint bundle.
  Exact resume restores its optimizer/scheduler/RNG/data cursor separately.
- `policy` and deploy-only construction retain their explicit checkpoint contracts.
- The explicit `joint_from_scratch` control retains the frozen VLM and initializes
  all task modules and EMA afresh; all pretrained task initialization paths are
  forbidden. New scratch preparation uses raw latents without calibration.

Future visual, body and action timestamps/masks come from source measurements.
No missing measurement is synthesized. Fixed nominal measurement query slots and
matching valid endpoint pairs define temporal-difference supervision. Real
collector snapshots use independently timestamped camera/body/hand receipts and
recorded held-command labels, described in [the real-data recipe](real_data_final_method.md).
Keep frame rate, control frequency and receive-time synchronization distinct.

The index, capability/measurement/action/state registries and feature manifests
are immutable. Frozen visual-feature tensors may be reused only after their
content, ordering and provenance bindings pass verification. A state projection
or input-contract transition creates fresh metadata and an explicit source-bound
migration; it does not modify original experiment artifacts.

## Checkpoint and numerical invariants

Architecture, temporal ABI, preprocessing identity, registries, normalization,
loss identity, resolved configuration and source fingerprints are checked at
owning boundaries. Preserve state-dict names and tensor shapes when refactoring.
Explicit current-state, observation, objective and processor migrations remain
because they are used by the selected pretrained source and historical evidence.

Attention retains `qk_norm_v1`, including FP32 state-history attention, bounded
state gates and zero-initialized prediction heads. Finite-loss/gradient checks
precede clipping and optimizer updates. No silent sample skipping or numerical
replacement is permitted. Distributed collectives retain consistent ordering.

The retired standalone endpoint/Psi0 policies and the unselected goal-intent v3
model branch are absent from this source distribution. Neutral legacy goal fields
remain parsable so supported resolved-config identities do not change. Historical
v3 loss metadata can be inspected, but enabling v3/goal supervision or migration
fails explicitly; reproduce such experiments from their original source snapshot.

## Entry points and validation

- Training: `scripts/train/predictive_world/train.py`.
- New raw joint configurations: `scripts/train/predictive_world/prepare_raw_latent_training.py`.
- Real preparation: `prepare_real_sonic_task.py` in that directory.
- Frozen features: `scripts/data/cache_predictive_world_features.py`.
- Deployment composition: `src/psi/models/predictive_world/policy.py` and `src/psi/deploy/`.
- Export: `scripts/export_predictive_world_policy.py` and `scripts/export_real_sonic_research_policy.py`.

[Configuration templates](../configs/predictive_world/README.md) distinguish the
base pretraining recipes from final task adaptation. See [architecture](maintenance/ARCHITECTURE.md)
for ownership and [development](../CONTRIBUTING.md) for validation tiers.
CPU regressions establish software contracts, not training quality or robot success.

Prepare a fresh 20k raw configuration from an existing target joint recipe:

```bash
PYTHONPATH=src python scripts/train/predictive_world/prepare_raw_latent_training.py \
  --reference-config /path/to/target-joint.json --initialization scratch \
  --output-config /path/to/new-raw.json --output-dir /path/to/new-run
```

`--initialization pretrained` uses the reference's `init_joint` and
`init_source_config`, composes the source output coordinates and preserves any
independent supported transitions. Preparation writes a new configuration only;
it retains the reference data, effective batch and GPU topology, and does not
generate or rewrite a dataset.

Compare common success/action/body-state metrics under a fixed protocol. Raw
latent MSE remains an internal diagnostic with target RMS/variance; different
learned encoders or objective terms do not make total losses comparable.

Checkpoint readers accept the pre-rename `oraclevla.predictive_world.parallel_process` format at schema revision 1. Writers continue to emit `zeta0.predictive_world.parallel_process`; all metadata, phase, and weight integrity checks remain required.
