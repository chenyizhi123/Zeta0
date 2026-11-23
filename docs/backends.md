# Layered VLA hosts (`Ψ₀`, `GR00T N1.7`, and reserved `π₀.₅`)

Zeta-0 turns layered / dual-system VLA policies into World Action Models by
embedding a shared System 1.5 foresight core between System 2 (VLM) and System 1
(flow-matching action expert). Host selection is unified across training, export,
and G1 deployment through the Layered VLA Host Registry in
[`hosts/`](../src/psi/models/predictive_world/hosts/__init__.py) (also exposed as
`zeta0.hosts`) and invoked from [`factory.py`](../src/psi/models/predictive_world/factory.py)
and [`real_sonic_runtime.py`](../src/psi/deploy/real_sonic_runtime.py).
Run `zeta hosts` to inspect supported and planned host contracts.

## Install

Use Python 3.10 and uv 0.8.20. Work in a Git clone with an initial commit: training
provenance requires Git. Source archives can be initialized and committed locally.
Run from the repository root. To reuse an environment, start with
[environment inspection](framework.md#start-with-your-existing-environment).
For fresh installs:

```bash
# Psi0-derived Zeta, serving and full CPU regressions
UV_PROJECT_ENVIRONMENT=.venv-psi0 uv sync --locked --python 3.10 --group psi --group cpu-test --group serve --group robot
uv pip check --python .venv-psi0/bin/python

# Native GR00T Zeta
UV_PROJECT_ENVIRONMENT=.venv-gr00t uv sync --locked --python 3.10 --group psi --group gr00t --group cpu-test --group serve --group robot
uv pip check --python .venv-gr00t/bin/python
```

The native backend also needs the source revision in
[`third_party/sources.toml`](../third_party/sources.toml):

```bash
python scripts/fetch_sources.py Isaac-GR00T --apply
python scripts/check_source_pins.py Isaac-GR00T
export PYTHONPATH="$PWD/src"
```

For G1 controller dependencies, see the [asset catalog](public_assets.md).
Install Zeta's dependency groups above rather than the upstream GR00T trainer.

## Configure and prepare

| Setting | Psi0-derived Zeta | GR00T N1.7 Zeta |
| --- | --- | --- |
| Template | [Psi0 joint](../configs/psi0/joint.template.json) | [N1.7 joint](../configs/gr00t_n1d7/joint.template.json) |
| `host_architecture` | `zeta_native` | `gr00t_n1d7` |
| Initialization | Frozen Psi0 VLM; random Zeta task modules | N1.7 backbone/native head; new Zeta modules and embodiment projections |
| Training flag | `joint_from_scratch=true` | `joint_from_gr00t=true` |
| Action horizon / solver steps | 50 / 10 | 40 / 4 |
| Latent prediction / objective | Raw / v6 | Raw / v6 |

Replace `REQUIRED_` fields with local model paths, prepared data/index, registries,
feature cache and a new output directory. Check GPU memory with a preflight run
at your intended physical batch size.

These templates describe joint training initialization. For the documented
Psi0-derived G1 fine-tuning path, start with
[collection and data preparation](workflows.md#collection-to-training) and the
[real-data recipe](real_data_final_method.md). For native N1.7, follow its
[backbone and data preparation guide](gr00t_n1d7_preparation.md).

Build each backend's feature cache with its own backbone and processor. Checkpoints
also bind architecture, time grid, registries and initialization; switching the
backend flag alone cannot convert weights or caches.

New joint recipes use v6/raw. Legacy omission of
`latent_prediction_space` still means `whitened`; omission of `host_architecture`
retains `zeta_native`. Use the documented migrations for legacy artifacts.
The original Psi0 **baseline** is a separate model.

## Train and preflight

Use the Python from the selected environment. Both backends share one entry:

```bash
python scripts/train/predictive_world/train.py --config /path/to/resolved.json --dry-run
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train/predictive_world/train.py --config /path/to/preflight.json \
  --num-workers 12 --preflight-only --preflight-updates 3 --verify-preflight-checkpoint
```

The second command is a real bounded GPU job. Match ranks/workers to the declared
recipe (the inherited Psi0 study uses 16 workers/rank; N1.7 uses 12). Use a separate
preflight config/output. Validate inputs, memory, gradients and checkpoint recovery
before launching the same trainer without preflight flags. Resume uses the
original resolved config and `--resume`.

For inference export and G1 runtime compatibility, follow the
[deployment handoff](workflows.md#training-to-real-deployment).

## Validation

```bash
make check
make test-mainline
make test-gr00t
make check-release
uv build --wheel
python scripts/check_distribution.py dist/*.whl
```

`make check` runs the lightweight CPU tier. `make test-mainline` needs the
training, serving and robot dependency groups. `make test-gr00t` also requires
the pinned GR00T source and dependencies; it checks the native head, solver,
gradients, optimizer and distributed recovery. See [test tiers](../CONTRIBUTING.md#test-tiers)
and [validation status](workflows.md#validation-status).
