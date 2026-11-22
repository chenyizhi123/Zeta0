# Shared CPU decoding for Zeta0-Humanoid caches

The seven-task launchers use one node-local CPU decoder service, shared by all
cache clients through a private Unix socket. The training allocation supplied
for this change is 160 CPU cores, 1800 GB RAM and eight GPUs. The default pool is
128 single-threaded decoder processes, leaving CPU capacity for GPU drivers,
image preprocessing and training data readers. It is **128 total**, not 128 per
task. This is a configured budget, not a measured optimal throughput setting.

Requests are scheduled round-robin between clients with pending work. A client
waiting for its GPU reserves no decoder processes. When a task finishes caching
or disconnects, the remaining clients can use all available processes without
restarting their caches. CPU work already executing for a disconnected client
finishes and is discarded; its queued work is removed. There is no fixed
per-task worker allocation. The service exits after 60 seconds without clients
or executing requests and is automatically started again when needed.

The repository launcher exposes these controls as `--cache-decode-workers-total`,
`--cache-decode-pool` and `--cache-decode-prefetch-batches`:

```bash
PYTHONPATH=src python scripts/train/predictive_world/launch_seven_tasks.py --cache-decode-workers-total 128 --dry-run
```

`ZETA0_DECODE_POOL` overrides the socket path (default:
`/tmp/zeta0-decode-<uid>/pool.sock`). Its containing directory must be private
to the user. Jobs that should share CPU capacity must use the same socket and
worker budget; a mismatched budget or decoder code/environment is rejected.
Separate allocations/checkouts can specify separate private socket directories.
Changing the pool budget applies on the next service startup, after current
clients have finished; it does not resize a running service.

The repository launcher `launch_seven_tasks.py` exposes the same controls as
`--cache-decode-workers-total`, `--cache-decode-pool` and
`--cache-decode-prefetch-batches`. For a direct cache invocation, use
`--decode-workers 128 --decode-pool /tmp/zeta0-decode-<uid>/pool.sock`.
Without `--decode-pool`, workers belong to that single cache process; a worker
count of zero selects synchronous decoding.

Each work item decodes up to eight distinct frames from the same or adjacent
videos. Exact uint8 images cross process boundaries, then use the original
float32 conversion and frozen-teacher preprocessing. The Qwen batch remains 32.
`--decode-prefetch-batches` bounds each client's outstanding
batches; zero selects twice the pool size. Already-ready batches are consumed
without waiting to fill this limit. Duplicate requests are coalesced even while
decoding is in flight. A client retains at most the configured prefetch batches,
the current Qwen batch, and the existing bounded recent-feature cache.

Progress logs now include cumulative decode worker time, consumer wait time,
preprocessing/encoding time (including completed GPU work and transfers), model
load time, feature assignment time and commit/flush time. Worker time is summed
across processes and can exceed wall time; it must not be added to wall time.
`session_committed_groups` counts newly committed groups, excluding previously
completed slots during resume. Per-session counters restart when a cache process
restarts. The final shard SHA-256 pass is outside these pipeline timings.

The cache array/provenance/index format and progress journal are unchanged;
unfinished caches resume in place. Feature storage is still indexed by sample
and future group. Unique-frame on-disk storage is not part of this change.

Training readers now retain eight video decoder handles by default instead of
two, including the TorchCodec backend, so three cameras no longer evict each
other on every sample. Qwen observation preprocessing runs once. Tests compare
the real local Qwen processor's single/three-view pixels, grids and token tensors
against the previous two-pass path with exact equality.

Validation uses real CPU AV1 decoding, local and shared multiprocessing, multiple
clients, client departure, error propagation, cache de-duplication, an interrupted
cache resumed from its journal, and real cache manifest/hash verification. The
cache pipeline test substitutes the teacher and row index only. Full frozen-Qwen
GPU throughput and a 128-process training-node run require measurement on that
allocation; the development container has no accessible NVIDIA driver.

## Training input and throughput monitoring

Zeta0-Humanoid training and its memory preflight now default to 12 DataLoader workers
per task and prefetch factor 1. Seven joint-training tasks therefore use 84 data
workers, separately from the shared cache decoder pool. These are not reserved
CPU cores: cache construction and training can overlap across tasks. Observe
CPU contention during that mixed phase before raising either budget further.

The repository launcher and training entry point accept `--num-workers` and
`--prefetch-factor`. For example:

```bash
PYTHONPATH=src python scripts/train/predictive_world/launch_seven_tasks.py --num-workers 16 --prefetch-factor 1 --dry-run
```

Worker count zero still works without multiprocessing-only options. Launch logs
record actual loader settings and free `/dev/shm` space; system RAM and shared
memory capacity are separate limits. Pinning and persistent workers remain
enabled on the existing supported paths. Tests compare occurrence ordering and
resume cursors with 0, 2 and 12 real DataLoader workers.

`train_metrics.jsonl` now includes `perf/step_seconds`,
`perf/input_wait_seconds`, `perf/input_wait_fraction` and
`perf/rank_samples_per_second`. These measure the logged update on rank zero,
including its accumulation microbatches; input wait measures `next(loader)` only,
excluding host-to-device transfer. CUDA is synchronized at existing log points
to include completed GPU work. These are sampled update timings, not whole-run
averages or distributed worst-rank throughput. Logging, validation and checkpoint
time are excluded from the step measurement.

The `hardware` record uses background `nvidia-smi` sampling every 10 seconds,
with a bounded 3-second subprocess timeout, GPU identity and sample age. Reported
utilization/memory are device-wide snapshots, not per-process or per-step values.
Unsupported/unavailable telemetry is explicitly marked; it does not become a
false zero-utilization measurement or fail training. Complete checkpoint duration
(weights, trainer state, completion publication, retention and final barrier) is
written separately to `performance_events.jsonl` and stdout.

## Real-robot motion76 RTC 0–13 recipe

`scripts/train/predictive_world/finetune-place-doll-motion76-rtc13-20k.sh`
reuses the existing real-robot launcher with `rtc_training_max_delay=13` and
a separate output directory. On complete 30-step action windows, the prefix
length is sampled uniformly from 0 through 13, inclusive (up to 260 ms at
50 Hz). Short windows retain at least one valid suffix step. This experiment
tests whether broader delay coverage improves RTC robustness relative to the
existing 0–6 recipe, using the same joint-15000 initialization, data, optimizer,
four GPUs, per-GPU batch 256, and 20,000-update budget. Save all four checkpoints
at 5,000-update intervals; assess delay robustness with closed-loop evaluation.

Allocate at least 12 CPU cores per GPU on the execution platform. Training and
cache construction run sequentially, each with 12 worker processes per rank
(48 workers across four GPUs). DataLoader workers persist, CUDA inputs are
pinned, and each worker prefetches one batch by default. Cache decoding
automatically queues twice the decoder worker count. OMP, MKL, OpenBLAS and
NumExpr default to one thread per process to avoid nested thread contention.
Override `NUM_WORKERS`, `PREFETCH_FACTOR`, `CACHE_DECODE_WORKERS` and
`CACHE_DECODE_PREFETCH_BATCHES` for the actual CPU and shared-memory allocation;
worker counts are per GPU rank. For example, with at least 64 allocated CPU
cores, use `NUM_WORKERS=14 CACHE_DECODE_WORKERS=14` to leave capacity for four
GPU ranks and system work. Measure input wait and cache throughput before
increasing prefetch: high CPU utilization alone is not evidence of faster
training.

```bash
bash scripts/train/predictive_world/finetune-place-doll-motion76-rtc13-20k.sh --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/predictive_world/finetune-place-doll-motion76-rtc13-20k.sh
```

The existing real-robot deployment client currently accepts at most six prefix
steps. Exercising 7–13-step prefixes at deployment requires a coordinated client
and checkpoint-contract update; this recipe only broadens training coverage.
