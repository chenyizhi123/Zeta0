# Reproducing Zeta-0

The supported route is `predictive_world`, shared by Psi0-derived and native
GR00T N1.7 hosts. Start with [the backend workflow](backends.md). New joint
recipes use v6/raw; the historical v5 task-adaptation recipe remains supported
under its original identity. Datasets, weights, feature caches and runtimes are
external inputs; see [artifact requirements](artifacts.md).

## Environment

Use Python 3.10 and uv 0.8.20. The package distribution is `zeta0-humanoid`;
Python imports use `psi`. Environment overrides use `ZETA0_`, and serialized
formats use the `zeta0` namespace. For existing environments, start with
[environment inspection](framework.md#start-with-your-existing-environment).

For bounded CPU checks:

```bash
uv venv --python 3.10 .venv-check
uv pip install --python .venv-check/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-check/bin/python --group cpu-test
make check PYTHON=.venv-check/bin/python RUFF=.venv-check/bin/ruff
```

For training, data preparation, HTTP serving and full CPU regressions:

```bash
UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --python 3.10 --group psi --group cpu-test --group serve --group robot --group data-bootstrap
uv pip check --python .venv-training/bin/python
uv lock --check
```

Prepare CUDA and video libraries through [installation](installation.md).
Raw-data converters may also require an MCAP, Pinocchio or Open3D environment.

For offline HIW conversion, prepare the pinned MCAP/lz4/zstandard wheels first
and set `HIW_WHEELHOUSE`. Pretraining registry rebuild tools take `SOURCE_SPEC`,
`RUNTIME_SPEC` and, where applicable, `ACTION_PROTOCOLS_SPEC` input manifests.

## Inputs and configuration

Prepare the measurement/capability/action/state registries, immutable temporal
index, frozen-Qwen feature manifest/shards and matching local Qwen model.
`prepare_predictive_world_release.py` converts the project's audited legacy
pretraining release into these inputs. Task-specific preparers handle real
collector datasets. Acquire the model assets from the
[asset catalog](public_assets.md).

```bash
export PYTHONPATH="$PWD/src"
python scripts/data/prepare_predictive_world_release.py --help
python scripts/train/predictive_world/prepare_sonic_lerobot_task.py --help
python scripts/train/predictive_world/prepare_real_sonic_task.py --help
```

The [configuration guide](../configs/predictive_world/README.md) distinguishes
base pretraining templates and the final task-adaptation profile. Replace required
paths and use migrations generated for the same checkpoint/config pair.

For a resolved configuration, preview before model/GPU construction:

```bash
python scripts/train/predictive_world/train.py --config /path/to/resolved.json --dry-run
```

After the config preview, use these entrypoints to extract features and train:

```bash
python scripts/data/cache_predictive_world_features.py --help
python scripts/train/predictive_world/train.py --help
```

The trainer supports `world_model`, `joint`, distributed execution, preflight and
checkpoint resume. Use a new run directory and save the config, data and source
versions with each run.
The [real-data recipe](real_data_final_method.md) specifies the final four-GPU,
20k-update route. Earlier experiment notes are in the
[external experiment records](artifacts.md#historical-records).

## Export and deployment

Use the [RTC contracts](predictive_world_realtime_control.md) to check observation
timing and action execution. Record the exact checkpoint and resolved training
config used to export each inference package.

```bash
python scripts/export_predictive_world_policy.py --help
python scripts/export_real_sonic_research_policy.py --help
```

Before deployment, check the model package and choose a compatible client and
controller through the [G1 guide](../real/g1/README.md#deployment).

## Optional external sources

[`third_party/sources.toml`](../third_party/sources.toml) is the single source of
truth for the optional Isaac-GR00T and GR00T-WholeBodyControl checkouts.
Use the source fetcher for the required integration:

```bash
python scripts/fetch_sources.py Isaac-GR00T --apply
python scripts/check_source_pins.py Isaac-GR00T
```

For an installed wheel, set `ZETA0_GR00T_SOURCE` to the pinned checkout root.
Source checkouts use `third_party/Isaac-GR00T` by default. Model, scene and
controller assets are listed in the [asset catalog](public_assets.md).
