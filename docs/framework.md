# Operate the framework

Use `zeta` to check environments, inspect data and model packages, run workflows
and summarize session logs.

## Start with your existing environment

To inspect an environment from the checkout, use Python 3.10 or later:

```bash
PYTHONPATH=src python -m psi.cli --help
PYTHONPATH=src python -m psi.cli --json doctor --profile train --python /path/to/existing/python
```

For the `zeta` command, install the checkout into your prepared environment:

```bash
uv pip install --no-deps --python /path/to/python -e .
```

Alternatively, replace `zeta` below with
`PYTHONPATH=src /path/to/python -m psi.cli`.

`doctor` reads Python versions and installed package metadata. It checks the
current interpreter, active environment and known environments in the checkout.
Use `--python` to select or compare environments, or `--profile capture` for
the Python 3.12 capture setup.

```bash
zeta --json doctor --profile train --python /path/to/train/python \
  --asset-path-map /path/to/assets.local.json
```

An asset path map maps original absolute paths to local paths, either absolute
or relative to the map file. After choosing an environment, run `uv pip check`
and the relevant [test tier](../CONTRIBUTING.md#test-tiers) to check runtime compatibility.

## Inspect captured data

```bash
zeta --json data inspect /path/to/recording
PYTHONPATH=src /path/to/train/python -m psi.cli --json data inspect \
  /path/to/recording --full --task my_task --validation-fraction 0.1 \
  --output /path/to/new-audit.json
```

The first command reads `meta/info.json`. Add `--full` to check native `sonic_g1`
LeRobot v3 recordings: state41/action76 columns, 30 Hz timestamps, numeric values,
episode boundaries, task labels and camera offsets. The report includes outcome
labels and train/validation splits.

Use the same task and validation fraction for preparation; both tools default
to split seed 292285. Review failed or unknown outcomes under your task's data
protocol. Video decoding is checked when materializing training data.

## Plan and execute a workflow

Workflow commands preview the command and working directory. Add `--execute` to
run it with the selected interpreter. Put wrapper options before the workflow
name and script options after `--`.

```bash
# Plan only:
zeta --root /path/to/Zeta0 workflow --python /path/to/train/python demo -- --device cpu
# Run the predictor, checkpoint reload and resume example:
zeta --root /path/to/Zeta0 workflow --python /path/to/train/python --execute demo -- --device cpu
```

| Workflow | Existing entrypoint | Effect when executed |
| --- | --- | --- |
| `demo` | `scripts/examples/quickstart.py` | Synthetic optimizer/checkpoint example |
| `prepare` | `scripts/train/predictive_world/prepare_sonic_lerobot_task.py` | Native v3 audit and index/registry/config preparation |
| `train` | `scripts/train/predictive_world/train.py` | Training; target `--dry-run` / `--preflight-only` retain their meanings |
| `export` | `scripts/export_real_sonic_research_policy.py` | New real-SONIC inference package |
| `record` | `real/g1/scripts/record_lerobot.py` | Capture in the separate capture environment |
| `deploy` | `real/g1/scripts/deploy_online_model.py` | Existing hardware deployment procedure |

Preview preparation, training and export:

```bash
zeta workflow --python /path/to/train/python prepare -- \
  --task my_task --dataset /path/to/recording --output /path/to/new-preparation \
  --init-joint /path/to/joint-checkpoint --source-config /path/to/source-config.json \
  --qwen-model /path/to/matching-qwen --current-state-only --fixed-action-grid \
  --validation-fraction 0.1
zeta workflow --python /path/to/train/python train -- --config /path/to/resolved.json --dry-run
zeta workflow --python /path/to/train/python export -- \
  --checkpoint /path/to/completed-joint --config /path/to/original-config.json --output /path/to/new-package
```

For distributed training, use `torch.distributed.run` as shown in
[backends](backends.md). Preparation options are documented in
[real-data preparation](real_data_final_method.md).

For deployment, the wrapper's `--execute` launches the script; the script's own
`--execute` enables hardware control. Use the target's `--dry-run` to check
prerequisites first. See [G1 deployment](../real/g1/README.md#deployment) for setup.

## Inspect a model package

```bash
zeta --json bundle inspect /path/to/inference-package
PYTHONPATH=src /path/to/train/python -m psi.cli --json bundle inspect \
  /path/to/inference-package --full --output /path/to/new-package-audit.json
```

Metadata inspection shows the task, encoder, processor, action/state formats,
time grid and RTC capacity. Add `--full` to verify checksums and source contracts.
The report distinguishes legacy weight-only packages from complete packages.

## Add a task without editing Python

Copy [custom_task.example.json](../real/g1/config/custom_task.example.json) to a local
configuration. Set the task name, instruction, model paths and matching runtime
profile. Relative paths resolve against the configuration directory. The task
instruction and action grid must match the model manifest and runtime.

```bash
zeta workflow --python /path/to/control/python deploy -- \
  --task my_task --model-paths /path/to/models.local.json \
  --trajectory-dir /path/to/matching-first-frame --dry-run
```

Prepare calibration and a matching first-frame reference through the
[G1 procedure](../real/g1/README.md). Check
[runtime compatibility](../real/g1/README.md#runtime-compatibility) before selecting a profile.

## Diagnose a recorded session offline

```bash
zeta --json logs summarize /path/to/session.jsonl --output /path/to/new-summary.json
```

The report summarizes G1 `RuntimeAudit` events: errors, reported dropped events,
inference time, anchor-to-accept delay, queue depth and delay steps. Timing metrics
include p50, p95 and maximum values. These are diagnostics for the recorded events;
task outcomes require separate evaluation.

The default limit is 100,000 events; adjust `--max-events` for longer logs.
`--output` requires a new file. Invalid JSON is reported with its line number.

Exit codes: 0 for successful inspection, 1 for failed doctor checks, and 2 for
invalid inputs or missing dependencies. Workflow execution returns the script's
exit code.
