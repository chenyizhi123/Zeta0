# Collection, training and real deployment

Use the [framework CLI](framework.md) to inspect inputs and launch workflows.
Each workflow uses its own environment and model/data requirements.

| Workflow | Entry | Environment | Inputs → outputs |
| --- | --- | --- | --- |
| Collection | `real/g1/record_episode.sh` | Python 3.12, LeRobot and G1 control stack | Camera/body/hand streams and task labels → LeRobot v3 dataset |
| Training | `scripts/train/predictive_world/train.py` | Python 3.10, Psi0-derived or native GR00T N1.7 | Prepared data, encoder, feature cache and config → checkpoint |
| Deployment | `real/g1/deploy_*.sh` | SONIC control and policy CUDA environments | Matching model, runtime, first frame and calibration → robot actions and logs |

Start with [installation](installation.md) to choose your environment.

## Collection to training

For native `sonic_g1` LeRobot v3 recordings, use `zeta data inspect --full`
followed by `zeta workflow prepare`. See [data inspection](framework.md#inspect-captured-data)
for commands. The documented G1 fine-tuning recipe uses the Psi0-derived host.
For native GR00T N1.7 training, follow the separate
[input requirements](gr00t_n1d7_preparation.md#data-and-training-entry). Historical
v2.1/cache recipes use their own loaders, and other LeRobot exports require an
adapter matching their schema.

Check state/action columns, units, timestamps, camera offsets, hand masks and
outcome labels before training. Keep measured states separate from action labels.
Define how failed or unknown outcomes and timing gaps are handled in your task's
data protocol, then fix the train/validation split. See
[data preparation](reproducibility.md) and [real-data adaptation](real_data_final_method.md).

## Training to real deployment

Use the [export and package inspection commands](framework.md#plan-and-execute-a-workflow)
to produce a research inference package from the completed joint checkpoint and
its original resolved config. The [real SONIC research integration](../real/README.md)
documents the matching service and robot client.

Choose a model export, encoder, processor and runtime with matching observation,
action and timing formats. The unified G1 runtime profile (`runtime_profiles/unified`)
loads directly from `src/psi` and supports both `zeta_native` (v5/v6) and `gr00t_n1d7`
exports (including 50Hz/40-step grids). The five historical v5 task runtimes are
preserved on branch `archive/legacy-v5-runtimes` and indexed in
[`LEGACY_ARCHIVE.json`](../real/g1/runtime_profiles/LEGACY_ARCHIVE.json).
Task weights must be supplied separately. See
[runtime compatibility](../real/g1/README.md#runtime-compatibility) and
[required assets](artifacts.md).

Before launch, check calibration, controller and hand services, stale-data guards,
action limits and first-frame acknowledgements through the
[G1 procedure](../real/g1/README.md#deployment).

## Validation status

CI covers CPU regressions, native GR00T components and G1 source checks. A bounded
[CUDA preflight](maintenance/CUDA_ACCEPTANCE_20261116.md) has also been completed.
Full-model training, live collection and physical deployment require validation
with the selected assets and hardware. See the [G1 review](maintenance/G1_INTEGRATION_REVIEW.md).

## Reproduction record

Save the source commit, environment, resolved config, seed, data revision/split,
encoder/cache identity, checkpoint, controller build and evaluation outcomes with
each run. Follow [reproducibility](reproducibility.md) for preparation and
[release checks](maintenance/RELEASE_CHECKLIST.md) for sharing results.
