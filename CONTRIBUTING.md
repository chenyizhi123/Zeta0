# Development

Use the `predictive_world` route for Zeta-0. Reusable computation belongs in
`src/psi`; launchers configure resources and call the single training loop at
`scripts/train/predictive_world/train.py`.

Read [AGENTS.md](AGENTS.md), [architecture](docs/maintenance/ARCHITECTURE.md)
and [research contracts](docs/maintenance/RESEARCH_CONTRACTS.md) for cross-module
changes. Preserve the semantics and identities of supported experiments.

## Environment

Python 3.10 is required. Follow [reproducibility](docs/reproducibility.md#environment)
for isolated CPU or training environments. Update `pyproject.toml` and `uv.lock`
together. Do not replace packages in environments used by existing jobs.

## CPU checks

Use Python 3.10 and uv 0.8.20:

```bash
uv venv --python 3.10 .venv-check
uv pip install --python .venv-check/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-check/bin/python --group cpu-test
make check PYTHON=.venv-check/bin/python RUFF=.venv-check/bin/ruff
```

On macOS ARM, omit `--index-url` when installing Torch. Native GR00T checks
also need the [backend dependencies and source](docs/backends.md#validation).

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/psi/framework/` | CLI environment checks, data/package inspection and workflow dispatch |
| `src/psi/models/predictive_world/` | Predictor, model hosts, objectives and trainer components |
| `src/psi/data/` | Dataset adapters, temporal sampling and feature caches |
| `src/psi/config/` | Training configuration and compatibility checks |
| `src/psi/deploy/` | Model services and robot clients |
| `scripts/train/` | Data preparation and training entrypoints |
| `configs/` | Training templates |
| `real/g1/` | Collection, teleoperation and task deployment tools |
| `third_party/` | Pins for optional integration checkouts |
| `tests/` | Unit and integration tests |

## Test tiers

| Command | Scope |
| --- | --- |
| `make check-release` | Source-export boundaries and credential patterns |
| `make test-gr00t` | Native N1.7 CPU regressions; requires pinned upstream source and dependencies |
| `make check-docs` | Local links and headings in maintained source documentation |
| `make test-dev` | Export, package, documentation and development tools |
| `make test-cpu` | Small predictive-world model, objective, checkpoint and runtime tests |
| `make test-mainline` | All predictive-world regressions with training/robot dependencies |
| `make lint` | Correctness lint in maintained source, scripts and tests |
| `make audit` | Read-only source syntax and credential-pattern inventory |

Override executables with `PYTHON=/path/to/python` and `RUFF=/path/to/ruff`.
CPU checks disable CUDA and constrain CPU threads. GPU training and robot execution require their matching inputs and runtimes. CPU checks do not
establish learned policy quality.

## Changes

- Trace producers, schemas and consumers before editing.
- Separate causal deployment inputs from future supervision.
- Preserve masks, units, timestamps, checkpoint keys and preprocessing identity.
- Version objective, data-sampling and schedule changes as method changes.
- Check numerical equivalence and checkpoint compatibility for refactors.
- Run relevant checks and report failures, skips and untested integrations.
- Preserve licenses, attribution and immutable experiment records.

Use a fresh output/config for a new experiment. Maintenance does not start training,
publish code or operate a robot.


## G1 collection and deployment integration

Run `make test-g1` for archive import, runtime hashes, isolated profile selection,
configuration preservation, body provenance and episode admission boundaries.
It needs pytest and no robot/model dependencies. CI runs it on Python 3.10 and
3.12. `make check` includes this tier. Hardware source is kept outside the library
wheel. Keep `real/g1/SOURCE_MANIFEST.json` integrated hashes current when changing
imported source, preserve incoming hashes, and never rewrite the 46-file frozen
runtime identity maps to accommodate a changed model. Update the documented
migration instead. See [G1 setup](real/g1/README.md) for separate environments.
