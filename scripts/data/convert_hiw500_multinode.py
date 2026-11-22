#!/usr/bin/env python3
"""Distributed CPU conversion with one shared inventory and byte-balanced shards.

GPU load is managed separately by the launcher; it is not conversion throughput.
Workers use spawn so they never fork an initialized Gloo/Torch process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from convert_hiw500_mcap import convert_episode, digest_file, json_atomic, publish


class LiveProgress:
    """Heartbeat runs during inventory, conversion, peer waits and publication."""
    def __init__(self, rank, nodes, store, interval=10):
        self.rank, self.nodes, self.store, self.interval = rank, nodes, store, interval
        self.started = time.monotonic()
        self.state = {"stage": "starting"}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.run_dir = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            value = dict(self.state)
        value.update(event="heartbeat", rank=self.rank,
                     timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                     runtime_seconds=round(time.monotonic() - self.started, 1))
        value["target_exceeded"] = value["runtime_seconds"] > value.get("target_seconds", 3600)
        if self.rank == 0 and self.run_dir is not None:
            peers = []
            for rank in range(self.nodes):
                try:
                    peers.append(json.loads((self.run_dir / f"progress-node-{rank}.json").read_text()))
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            value["global"] = {
                "reporting_nodes": len(peers),
                "completed": sum(p["completed"] for p in peers),
                "failed": sum(p["failed"] for p in peers),
                "selected": sum(p["selected"] for p in peers),
                "remaining_seconds_estimate": max(
                    (p.get("estimated_remaining_seconds_by_input_bytes") or 0 for p in peers), default=0
                ) if len(peers) == self.nodes and all(
                    p.get("estimated_remaining_seconds_by_input_bytes") is not None for p in peers
                ) else None,
                "eta_note": "last committed estimates; excludes final publication, approximate",
            }
        return value

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.store.set(f"heartbeat/{self.rank}", str(time.time()))
                print(json.dumps(self.snapshot()), flush=True)
            except Exception as error:
                self.error = error
                print(json.dumps({"rank": self.rank, "heartbeat_error": str(error)}), flush=True)
                return
            self.stop.wait(self.interval)

    def start(self):
        self.store.set(f"heartbeat/{self.rank}", str(time.time()))
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def wait_for_stage(store, stage, rank, nodes, progress, stale_seconds=300):
    """No healthy-job wall-clock deadline; only missing/dead peers time out."""
    store.set(f"ready/{stage}/{rank}", "1")
    keys = [f"ready/{stage}/{peer}" for peer in range(nodes)]
    started = time.monotonic()
    while not store.check(keys):
        if progress.error is not None:
            raise RuntimeError(f"heartbeat connection failed: {progress.error}")
        for peer in range(nodes):
            heartbeat = f"heartbeat/{peer}"
            if store.check([heartbeat]):
                if time.time() - float(store.get(heartbeat).decode()) > stale_seconds:
                    raise RuntimeError(f"node {peer} heartbeat lost while waiting for {stage}")
            elif time.monotonic() - started > stale_seconds:
                raise RuntimeError(f"node {peer} never sent a heartbeat")
        time.sleep(1)


def balance_shards(entries, count):
    if count < 1:
        raise ValueError("node count must be positive")
    shards, loads = [[] for _ in range(count)], [0] * count
    for entry in sorted(entries, key=lambda value: (-value["size"], value["path"])):
        rank = min(range(count), key=lambda index: (loads[index], index))
        shards[rank].append(entry)
        loads[rank] += entry["size"]
    return shards, loads


def check_coverage(expected, records, failures):
    paths = [record["source"]["relative_path"] for record in records]
    paths.extend(failure["source"] for failure in failures)
    if len(paths) != len(set(paths)) or set(paths) != set(expected):
        raise ValueError("distributed conversion coverage has duplicates or missing episodes")


def worker_init():
    import cv2
    import pyarrow
    cv2.setNumThreads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


def run_shard(args, run_dir, entries, reporter=None):
    started = time.monotonic()
    completed = failed = frames = processed_bytes = 0
    total_bytes = sum(entry["size"] for entry in entries)
    initial = {"rank": args.rank, "completed": 0, "failed": 0, "selected": len(entries),
               "frames": 0, "processed_input_bytes": 0,
               "committed_frames_per_second": 0, "processed_input_mib_per_second": 0,
               "remaining_episodes": len(entries), "target_seconds": args.target_seconds,
               "estimated_remaining_seconds_by_input_bytes": 0 if not entries else None}
    json_atomic(run_dir / f"progress-node-{args.rank}.json", initial)
    if reporter:
        reporter.update(stage="converting", **initial)
    records_path = run_dir / f"node-{args.rank}.jsonl"
    jobs = [(str(args.raw_root), entry["path"], str(args.output_root), 18, 1, 70, 50, args.codec)
            for entry in entries]
    print(json.dumps({"rank": args.rank, "episodes": len(jobs), "workers": args.workers,
                      "codec": args.codec, "input_bytes": total_bytes}), flush=True)
    with records_path.open("w") as log:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(convert_episode, job): entry for job, entry in zip(jobs, entries)}
            for future in as_completed(pending):
                entry = pending[future]
                try:
                    record = future.result()
                    log.write(json.dumps({"record": record}) + "\n")
                    completed += 1
                    frames += record["length"]
                except Exception as error:
                    failure = {"source": entry["path"], "error": f"{type(error).__name__}: {error}"}
                    log.write(json.dumps({"failure": failure}) + "\n")
                    failed += 1
                    print(json.dumps({"rank": args.rank, "failure": failure}), flush=True)
                log.flush()
                processed_bytes += entry["size"]
                elapsed = time.monotonic() - started
                remaining_bytes = total_bytes - processed_bytes
                estimate = elapsed * remaining_bytes / processed_bytes if processed_bytes else None
                progress = {"rank": args.rank, "completed": completed, "failed": failed,
                            "selected": len(entries), "frames": frames, "elapsed_seconds": elapsed,
                            "processed_input_bytes": processed_bytes,
                            "estimated_remaining_seconds_by_input_bytes": estimate,
                            "eta_note": "startup/reused/failed episodes can bias this estimate",
                            "committed_frames_per_second": frames / elapsed,
                            "processed_input_mib_per_second": processed_bytes / elapsed / 2**20,
                            "remaining_episodes": len(entries) - completed - failed,
                            "target_exceeded": elapsed > args.target_seconds,
                            "target_seconds": args.target_seconds}
                json_atomic(run_dir / f"progress-node-{args.rank}.json", progress)
                if reporter:
                    reporter.update(**progress)
                print(json.dumps(progress), flush=True)
        log.flush()
        os.fsync(log.fileno())
    return {"rank": args.rank, "completed": completed, "failed": failed,
            "records_path": str(records_path), "records_sha256": digest_file(records_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--codec", choices=("mjpeg", "h264"), default="mjpeg")
    parser.add_argument("--communication-timeout-seconds", "--timeout-seconds",
                        dest="timeout_seconds", type=int, default=300)
    parser.add_argument("--join-timeout-seconds", type=int, default=300)
    parser.add_argument("--progress-every-seconds", type=float, default=10)
    parser.add_argument("--target-seconds", type=int, default=3600)
    parser.add_argument("--limit", type=int, help="smoke only: cap globally, before sharding")
    args = parser.parse_args()
    if not 0 <= args.rank < args.nodes or args.workers < 1 or args.progress_every_seconds <= 0:
        parser.error("invalid rank/nodes/workers")
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.raw_root == args.output_root or args.raw_root in args.output_root.parents or args.output_root in args.raw_root.parents:
        parser.error("raw/output directories must be separate and non-nested")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    import torch.distributed as dist
    store = dist.TCPStore(args.master_addr, args.master_port, args.nodes, args.rank == 0,
                          timeout=timedelta(seconds=args.join_timeout_seconds))
    dist.init_process_group("gloo", store=store,
                            rank=args.rank, world_size=args.nodes,
                            timeout=timedelta(seconds=args.timeout_seconds))
    lock = None
    reporter = LiveProgress(args.rank, args.nodes, store, args.progress_every_seconds)
    reporter.start()
    started = time.monotonic()
    try:
        contract = {"raw": str(args.raw_root), "output": str(args.output_root), "codec": args.codec,
                    "nodes": args.nodes, "limit": args.limit,
                    "converter_sha256": digest_file(Path(__file__).with_name("convert_hiw500_mcap.py")),
                    "coordinator_sha256": digest_file(__file__)}
        contracts = [None] * args.nodes
        dist.all_gather_object(contracts, contract)
        if any(value != contract for value in contracts):
            raise ValueError("nodes disagree on paths, converter code, codec or inventory limit")
        # Rank zero alone owns inventory/publication; old status files are never
        # barriers. A fresh run id is broadcast over the current live group.
        envelope = [None]
        reporter.update(stage="inventory" if args.rank == 0 else "waiting_for_inventory")
        if args.rank == 0:
            try:
                args.output_root.mkdir(parents=True, exist_ok=True)
                lock = (args.output_root / ".conversion.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_dir = args.output_root / "runs" / uuid.uuid4().hex
                run_dir.mkdir(parents=True)
                entries, incomplete = [], 0
                for directory, _, files in os.walk(args.raw_root):
                    for name in files:
                        if name.endswith(".mcap"):
                            path = Path(directory) / name
                            entries.append({"path": str(path.relative_to(args.raw_root)),
                                            "size": path.stat().st_size})
                        elif name.endswith(".incomplete"):
                            incomplete += 1
                entries.sort(key=lambda value: value["path"])
                inventory = {"complete_mcap_files": len(entries), "incomplete_files": incomplete,
                             "scope": "local_snapshot"}
                if args.limit:
                    entries = entries[:args.limit]
                    inventory["scope"] = "smoke_limit"
                if not entries:
                    raise ValueError("no completed MCAP files found")
                inventory["selected"] = len(entries)
                shards, loads = balance_shards(entries, args.nodes)
                json_atomic(run_dir / "plan.json", {"contract": contract, "inventory": inventory,
                                                    "shards": shards, "input_bytes_per_node": loads})
                json_atomic(args.output_root / "active_run.json", {"run_dir": str(run_dir),
                                                                   "inventory": inventory})
                envelope[0] = {"run_dir": str(run_dir), "plan_sha256": digest_file(run_dir / "plan.json")}
            except Exception as error:
                envelope[0] = {"error": f"{type(error).__name__}: {error}"}
        wait_for_stage(store, "inventory", args.rank, args.nodes, reporter, args.timeout_seconds)
        dist.broadcast_object_list(envelope, src=0)
        if "error" in envelope[0]:
            raise RuntimeError(envelope[0]["error"])
        run_dir = Path(envelope[0]["run_dir"])
        reporter.run_dir = run_dir
        if digest_file(run_dir / "plan.json") != envelope[0]["plan_sha256"]:
            raise ValueError("shared plan is missing or corrupted")
        plan = json.loads((run_dir / "plan.json").read_text())
        try:
            local = run_shard(args, run_dir, plan["shards"][args.rank], reporter)
        except Exception as error:
            local = {"rank": args.rank, "fatal": f"{type(error).__name__}: {error}"}
        results = [None] * args.nodes
        reporter.update(stage="waiting_for_nodes")
        wait_for_stage(store, "converted", args.rank, args.nodes, reporter, args.timeout_seconds)
        dist.all_gather_object(results, local)
        final = [None]
        reporter.update(stage="publishing" if args.rank == 0 else "waiting_for_publication")
        if args.rank == 0:
            try:
                if any("fatal" in result for result in results):
                    raise RuntimeError(f"node conversion failed: {results}")
                records, failures = [], []
                for result in results:
                    path = Path(result["records_path"])
                    if digest_file(path) != result["records_sha256"]:
                        raise ValueError("node result digest mismatch")
                    for line in path.read_text().splitlines():
                        entry = json.loads(line)
                        if "record" in entry:
                            records.append(entry["record"])
                        else:
                            failures.append(entry["failure"])
                expected = [e["path"] for shard in plan["shards"] for e in shard]
                check_coverage(expected, records, failures)
                release = publish(args.output_root, records, plan["inventory"], failures)
                import subprocess
                subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_hiw500_pretrain_source.py")),
                                "--release", str(release), "--output-dir", str(release / "pretrain_source")],
                               check=True)
                final[0] = {"root": str(release), "revision": release.name,
                            "completed": len(records), "failed": len(failures),
                            "elapsed_seconds": time.monotonic() - started,
                            "target_seconds": args.target_seconds,
                            "target_met": (time.monotonic() - started <= args.target_seconds
                                           and not failures and plan["inventory"]["scope"] == "local_snapshot"),
                            "status": "partial_review_required" if failures else "complete"}
                json_atomic(args.output_root / "latest_release.json", final[0])
            except Exception as error:
                final[0] = {"fatal": f"{type(error).__name__}: {error}"}
            json_atomic(run_dir / "result.json", final[0])
        wait_for_stage(store, "published", args.rank, args.nodes, reporter, args.timeout_seconds)
        dist.broadcast_object_list(final, src=0)
        reporter.update(stage="finished")
        print(json.dumps(final[0]), flush=True)
        if "fatal" in final[0] or final[0].get("failed", 0):
            raise SystemExit(1)
    finally:
        reporter.close()
        if lock is not None:
            lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
