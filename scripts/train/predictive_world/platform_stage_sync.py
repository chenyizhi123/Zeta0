#!/usr/bin/env python3
"""Synchronize one launcher stage across platform worker nodes."""

from __future__ import annotations

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--local-status", required=True, type=int)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(hours=24),
    )
    status = torch.tensor(max(0, args.local_status), dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    aggregate = int(status.item())
    dist.barrier()
    dist.destroy_process_group()
    if args.rank == 0:
        outcome = "passed" if aggregate == 0 else f"failed({aggregate})"
        print(f"platform_stage={args.stage} status={outcome}", flush=True)
    raise SystemExit(aggregate)


if __name__ == "__main__":
    main()
