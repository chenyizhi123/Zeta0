# Zeta on GR00T N1.7

The `gr00t_n1d7` backend implements Zeta inside the pinned native N1.7 architecture.
It uses the existing [training loop](../scripts/train/predictive_world/train.py),
with independent teacher/executor heads, H/F/E/EMA/D, raw latent prediction and
FULL+ERT dual response. Formal training still requires fresh features, a validated
training environment and a GPU preflight. Results belong in the canonical
experiment record referenced by the repository README.

## Model and initialization

The actual base config selects 16 Qwen text layers, 2048-dimensional VL context,
32 DiT layers, four VL self-attention layers, a 40-step horizon, padded state/action
width 132 and four Euler steps. Configuration and weight payload digests bind the
exported backbone, feature provenance and checkpoint initialization.

- The frozen Qwen weights come from the N1.7 checkpoint itself. Cosmos supplies
  processor/config files; no separate Cosmos weight download is required.
- Training and deployment use the same processor loader. If a Cosmos snapshot
  stores its chat template only in `tokenizer_config.json`, the loader assigns
  that exact string to the processor. It preserves existing processor templates
  and rejects missing or ambiguous templates without rewriting model assets.
- Native DiT, VL layer normalization, VL self-attention and position embeddings
  load strictly from the base. The selected state and action schemas use a new
  native embodiment with its own state/action encoders and action decoder.
- H/F/E/D and future projections initialize separately. T is initialized once;
  G starts as an independent copy, with separate parameter storage and gradients.
  `joint_from_gr00t=true` records this separately from random task initialization.
- Three groups at 0.2/0.4/0.6 seconds, each containing eight latent tokens, project
  to native context width after each head's VL preprocessing. The 24 logical
  tokens have two identical masked views (48 physical tokens): one for image
  attention and one for text attention. Each alternating native cross-attention
  block sees exactly one view. Upstream attention code remains unmodified.
- External flow time is Zeta's noise fraction sigma. The head receives t=1-sigma
  and returns the negative native velocity. Ordinary action sampling uses the
  native Beta time distribution. RTC uses clean committed prefix values plus a
  common suffix time, clamps the prefix at every Euler step and rejects arbitrary
  per-action time schedules.
- Ordinary T/G losses train their heads. Response readers run deterministically
  with detached parameters/buffers; gradients through predicted latents remain
  enabled. Shared response noise, time, masks and prefix retain Zeta's contract.
- `runtime.gr00t_activation_checkpointing=true` trades recomputation for memory
  while retaining the physical batch. Ordinary native reader calls preserve RNG
  state; response recomputation reinstates detached parameters and evaluation
  mode around the complete call. Weight keys, masks and objectives stay the same.
  Record the runtime setting explicitly in fresh configs for the full run and
  all interventions; it does not establish GPU fit without a new preflight.
- `runtime.ddp_gradient_as_bucket_view=true` lets parameter gradients share DDP
  reduction storage. This removes a duplicate gradient buffer without changing
  the physical batch, optimizer or loss. The trainer retains its unused-parameter
  handling for alternating forecast branches. Record this setting in every run;
  check multiple updates because DDP establishes the views during execution.

Zeta's causal observation/image processing is retained: `aspect_v1`, current
multiview inputs and post-merger content-area 4x4 visual targets. This is not the
upstream GR00T augmentation pipeline. State/action statistics remain Zeta's audited
training-set affine coordinates. Native padded dimensions are masked explicitly.
The old `expert_*` configuration fields describe the legacy head; N1.7 head sizes
come from the SHA-bound base config.

## Environment and backbone preparation

Use a fresh Python 3.10 environment, leaving other experiments' environments intact:

```bash
UV_PROJECT_ENVIRONMENT=.venv-gr00t uv sync --locked --python 3.10 --group psi --group gr00t --group cpu-test
uv pip check --python .venv-gr00t/bin/python
export PYTHONPATH="$PWD/src"
export NO_ALBUMENTATIONS_UPDATE=1
```

The `gr00t` dependency group supplies the native component dependencies. The
upstream trainer's separate Python/CUDA environment is not used. Clone the
revision in [sources.toml](../third_party/sources.toml) into
`third_party/Isaac-GR00T`; construction checks its revision and clean working tree.
Use `GIT_LFS_SKIP_SMUDGE=1` for source-only checks. An installed wheel can use an
external checkout selected by `ZETA0_GR00T_SOURCE`.

Acquire the base weights and processor separately; see [external inputs](artifacts.md).
Export once to a new directory; the exporter refuses overwrites and publishes only
complete outputs:

```bash
python scripts/train/gr00t_n1d7/export_backbone.py \
  --base models/gr00t_n17_3b \
  --cosmos-processor /path/to/Cosmos-Reason2-2B/snapshot \
  --output models/gr00t_n17_backbone
python scripts/train/gr00t_n1d7/audit_architecture.py \
  --base models/gr00t_n17_3b --backbone models/gr00t_n17_backbone
```

The architecture audit constructs meta-device modules and checks actual weight
keys/shapes and payload identities. It does not run full-model forward/backward.

## Data and training entry

[prepare_training.py](../scripts/train/gr00t_n1d7/prepare_training.py) accepts an
audited Zeta `joint_data.json` release with a 50 Hz SONIC action grid, at least
40 action steps, current-state observations and one shared action/state schema. It checks source identity,
index bytes, temporal ABI and registry hashes; retains only active schemas;
shortens a native action window of at least 40 steps to 40; and preserves future
queries, measured-body targets, episode splits and equal task sampling. It writes
new registry bindings, an immutable index, a resolved config and fresh feature
provenance. It never links the old VLM feature cache.

```bash
python scripts/train/gr00t_n1d7/prepare_training.py \
  --source-data /path/to/audited/joint_data.json \
  --base models/gr00t_n17_3b --backbone models/gr00t_n17_backbone \
  --output /path/to/new/preparation --run-output /path/to/new/training
python scripts/data/cache_predictive_world_features.py \
  --index /path/to/new/preparation/predictive_world_index.jsonl \
  --provenance /path/to/new/preparation/feature_provenance.json \
  --model models/gr00t_n17_backbone --output /path/to/new/preparation/feature_cache
```

Raw G1 LeRobot v3 recordings need data preparation before training. The
collection adapter documents the real-robot path. N1.7 preparation in this
guide consumes the audited joint release format above, with an immutable index,
registries and compatible action timing. Use the matching input contract for
each backend. [Collection to training](workflows.md#collection-to-training)

The [template](../configs/gr00t_n1d7/joint.template.json) specifies 40k updates,
two ranks, physical batch 256/rank, accumulation 1, global batch 512, raw latents,
no latent calibration/prediction warmup, LR warmup 1000, FULL+ERT v6 and RTC delay
up to 6. These are explicit initial settings; their full-model GPU fit is untested.
Do not silently lower physical batch or relabel a small test as this configuration.
The example uses 12 data-loader workers per rank. Save every 5k updates, retaining
the latest two complete intermediate checkpoints plus the final 40k bundle.

Once fresh features and the intended GPUs are available, use a separate output
for the existing bounded preflight, including checkpoint recovery:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train/predictive_world/train.py --config /path/to/preflight.json \
  --num-workers 12 --preflight-only --preflight-updates 3 --verify-preflight-checkpoint
```

A formal run uses the same entry point with its fresh training config and without
preflight flags. Resume uses `--resume /path/to/checkpoint` and the original config.
CPU tests cover actual small native modules, gradient ownership, flow/solver
agreement, masked futures, RTC, optimizer updates and exact next-step recovery.
They do not establish full-model memory, distributed behavior, convergence or
closed-loop quality. The former full-FP32 CPU attempt exceeded an 8 GiB container;
use adequate GPU resources for full-model validation.
