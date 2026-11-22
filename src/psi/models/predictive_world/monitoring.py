"""Read-only, reproducible joint validation and explicit logging cadence."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.distributed as dist

from psi.data.predictive_world.cache import canonical_sha256
from psi.models.predictive_world.engine import move_to_device


def training_log_due(update: int, every: int, both_modes: bool) -> bool:
    # Log adjacent updates, not an average that mixes two different objectives.
    return update == 1 or update % every == 0 or (
        both_modes and (update == 2 or (update + 1) % every == 0)
    )


@contextmanager
def isolated_evaluation(model, *, seed: int, cuda_devices=None):
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = (list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []) if cuda_devices is None else cuda_devices
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        # Preserve deliberately frozen children's eval flags, too.
        for module, training in modes:
            module.training = training


def evaluate_joint(model, *, rows, collator, device, seed: int, update: int):
    """Evaluate a bounded held-out sample; aggregate exact metric denominators.

    ``rows`` yields lists of raw reader rows. Recreate it from occurrence zero
    for each call. Forward uses the bare model, not DDP (source groups can differ
    between ranks). No training iterator or persistent worker RNG is consumed.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    sums = defaultdict(lambda: [0.0, 0.0])
    sample_ids = []
    with isolated_evaluation(model, seed=seed + rank):
        for raw_rows in rows:
            groups = defaultdict(list)
            for row in raw_rows:
                if row["split"] != "validation":
                    raise ValueError("joint validation requires the held-out validation split")
                groups[(row["source_id"], row["action_protocol_id"])].append(row)
                sample_ids.append(row["sample_id"])
            for (source, protocol), group in sorted(groups.items()):
                batch = move_to_device(collator(group), device)
                # Identical draws across modes and checkpoints for this group.
                group_seed = seed + int(canonical_sha256([r["sample_id"] for r in group])[:8], 16)
                for mode in ("prior", "world"):
                    with isolated_evaluation(model, seed=group_seed):
                        prepared = dict(batch, forecast_mode=mode, _optimizer_update=update)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "cuda"):
                            result = model(prepared)
                        result.validate()
                        for name, value in result.metrics.items():
                            weight = (result.valid.double().sum() if result.metric_weights is None
                                      else result.metric_weights[name])
                            if (value.numel() != 1 or weight.numel() != 1
                                    or not torch.isfinite(value) or not torch.isfinite(weight)
                                    or weight < 0):
                                raise FloatingPointError(f"invalid validation metric {name}")
                            for scope in ((source, protocol), ("__all__", "__all__")):
                                entry = sums[(*scope, mode, name)]
                                entry[0] += float(value.double() * weight.double())
                                entry[1] += float(weight)
    local = (dict(sums), sample_ids)
    gathered = [local]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    combined = defaultdict(lambda: [0.0, 0.0])
    ids = []
    for metrics, samples in gathered:
        ids.extend(samples)
        for key, (numerator, denominator) in metrics.items():
            combined[key][0] += numerator
            combined[key][1] += denominator
    records = {}
    for (source, protocol, mode, name), (numerator, denominator) in sorted(combined.items()):
        record = records.setdefault((source, protocol, mode), {
            "source_id": source, "action_protocol_id": protocol, "mode": mode,
            "metrics": {}, "metric_weights": {},
        })
        # Unmeasured diagnostics are absent, never misleading zeros.
        if denominator > 0:
            record["metrics"][name] = numerator / denominator
            record["metric_weights"][name] = denominator
    return {
        "update": update, "split": "validation", "seed": seed,
        "sample_count": len(ids), "unique_sample_count": len(set(ids)),
        "sample_identity": canonical_sha256(ids),
        "aggregation": "metric-specific weighted mean over sampled occurrences",
        "groups": list(records.values()),
    }
