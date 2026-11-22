#!/usr/bin/env python3
"""Fail-fast rank-local filesystem, topology, BF16 and NCCL preflight."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.assets import resolve_asset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--collective-mib", type=int, default=256)
    parser.add_argument("--require-gpu-name", default="H100")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=75.0)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--effective-global-batch", type=int)
    args = parser.parse_args()
    if args.collective_mib <= 0:
        raise ValueError("collective-mib must be positive")
    config, payload = load_predictive_world_config(
        args.config,
        allow_existing_output_override=args.allow_existing_output,
        training_overrides={
            "micro_batch": args.micro_batch,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": args.effective_global_batch,
        },
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=config.runtime.process_group_timeout_s)
    )
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != config.runtime.world_size:
            raise ValueError(
                f"launched world size {world} != configured {config.runtime.world_size}"
            )
        paths = [
            Path(args.config), resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map),
            Path(config.data.measurement_registry),
            Path(config.data.canonical_future_capabilities),
            Path(config.data.action_protocol_registry),
            Path(config.data.state_schema_registry), Path(config.data.index_path),
            Path(config.data.feature_cache),
        ]
        if config.training.world_model:
            paths.append(Path(config.training.world_model))
        missing = [str(path.resolve()) for path in paths if not path.exists()]
        identity = {
            "rank": rank,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(local_rank),
            "bf16": torch.cuda.is_bf16_supported(),
            "memory_gib": torch.cuda.get_device_properties(local_rank).total_memory / 2**30,
            "config_sha256": file_sha256(Path(args.config)),
            "resolved_config_sha256": canonical_sha256(payload),
            "missing": missing,
        }
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        if len({item["config_sha256"] for item in identities}) != 1:
            raise RuntimeError("config bytes differ across ranks")
        if len({item["resolved_config_sha256"] for item in identities}) != 1:
            raise RuntimeError("resolved config overrides differ across ranks")
        if any(item["missing"] for item in identities):
            raise FileNotFoundError(f"rank-local predictive-world inputs are missing: {identities}")
        required_name = args.require_gpu_name.upper()
        if any(
            not item["bf16"]
            or (required_name and required_name not in item["gpu"].upper())
            or item["memory_gib"] < args.min_gpu_memory_gib
            for item in identities
        ):
            raise RuntimeError(f"GPU allocation violates the configured contract: {identities}")
        device = torch.device("cuda", local_rank)
        values = torch.ones(args.collective_mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)
        for _ in range(2):
            dist.all_reduce(values)
            values.fill_(1)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        dist.all_reduce(values)
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if not torch.isfinite(values).all() or float(values[0]) != float(world):
            raise RuntimeError("NCCL all-reduce produced an invalid result")
        if rank == 0:
            print(json.dumps({
                "event": "predictive_world_distributed_preflight_passed",
                "world_size": world,
                "hosts": sorted({item["host"] for item in identities}),
                "gpus": sorted({item["gpu"] for item in identities}),
                "config_sha256": identities[0]["config_sha256"],
                "all_reduce_seconds_max_rank": float(elapsed),
            }, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
