#!/usr/bin/env python3
"""Expand frozen LeRobot episode splits into physical-time anchor rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from psi.data.predictor_pretrain.lerobot_source import build_lerobot_anchor_rows
from psi.data.predictor_pretrain.registry import (
    PT_SPLITS,
    canonical_json_bytes,
    iter_manifest_rows,
    manifest_sha256,
    normalize_manifest_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--anchor-hz", type=float, default=2.0)
    parser.add_argument("--future-seconds", type=float, default=0.6)
    parser.add_argument("--protocol-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--candidate-phases", type=int, default=1)
    parser.add_argument("--visual-history-seconds", type=float, default=0.0)
    parser.add_argument("--action-steps", type=int, default=30)
    parser.add_argument(
        "--dataset-freeze",
        help="Write the final dataset freeze while materializing already-split episodes. "
             "This avoids a second multi-gigabyte registry rewrite.",
    )
    parser.add_argument(
        "--episode-freeze",
        help="Required with --dataset-freeze; proves the episode manifest and split contract.",
    )
    parser.add_argument("--progress-every-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if bool(args.dataset_freeze) != bool(args.episode_freeze):
        raise ValueError("--dataset-freeze and --episode-freeze must be provided together")
    if args.progress_every_seconds <= 0:
        raise ValueError("--progress-every-seconds must be positive")
    allowed = set(args.source_id or [])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    count = 0
    episode_count = 0
    started = last_progress = time.monotonic()
    digest = hashlib.sha256(b"[")
    first = True
    episode_freeze = None
    if args.episode_freeze:
        episode_freeze = json.loads(Path(args.episode_freeze).read_text())
        if episode_freeze.get("manifest_sha256") != manifest_sha256(args.episode_manifest):
            raise ValueError("episode manifest does not match its freeze")
        if episode_freeze.get("input_protocol") != args.protocol_version:
            raise ValueError("episode freeze protocol differs from requested anchor protocol")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for episode in iter_manifest_rows(args.episode_manifest):
                if allowed and episode["source_id"] not in allowed:
                    continue
                if not Path(episode["data_path"]).is_file():
                    raise FileNotFoundError(
                        "refusing a partial anchor manifest; missing data file: "
                        f"{episode['data_path']}"
                    )
                if episode.get("split") not in PT_SPLITS:
                    raise ValueError("fast materialization requires a frozen episode split")
                # Bound peak memory to one episode. The next registry stage
                # performs a disk-backed canonical sort and split assignment.
                anchors = build_lerobot_anchor_rows(
                    [episode], anchor_hz=args.anchor_hz,
                    future_seconds=args.future_seconds,
                    protocol_version=args.protocol_version,
                    candidate_phases=args.candidate_phases,
                    visual_history_seconds=args.visual_history_seconds,
                    action_steps=args.action_steps,
                )
                for row in anchors:
                    row = normalize_manifest_row(row)
                    if not first:
                        digest.update(b",")
                    digest.update(canonical_json_bytes(row))
                    first = False
                    handle.write(json.dumps(
                        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ) + "\n")
                count += len(anchors)
                episode_count += 1
                now = time.monotonic()
                if now - last_progress >= args.progress_every_seconds:
                    elapsed = now - started
                    print(json.dumps({
                        "event": "anchor_materialization_progress",
                        "episodes": episode_count,
                        "anchors": count,
                        "anchors_per_second": round(count / max(elapsed, 1e-9), 1),
                        "elapsed_seconds": round(elapsed, 1),
                    }), flush=True)
                    last_progress = now
        digest.update(b"]")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if args.dataset_freeze:
        assert episode_freeze is not None
        freeze = {
            "format_version": f"predictor_pretrain_dataset_freeze_{args.protocol_version}",
            "input_protocol": args.protocol_version,
            "split_seed": int(episode_freeze["split_seed"]),
            "train_fraction": float(episode_freeze["train_fraction"]),
            "val_fraction": float(episode_freeze["val_fraction"]),
            "inputs": [str(Path(args.episode_manifest).resolve())],
            "episode_manifest_sha256": str(episode_freeze["manifest_sha256"]),
            "num_rows": count,
            "manifest_sha256": digest.hexdigest(),
            "materialization": {
                "anchor_hz": args.anchor_hz,
                "future_seconds": args.future_seconds,
                "candidate_phases": args.candidate_phases,
                "visual_history_seconds": args.visual_history_seconds,
                "action_steps": args.action_steps,
            },
        }
        freeze_path = Path(args.dataset_freeze)
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_tmp = freeze_path.with_name(f".{freeze_path.name}.tmp")
        freeze_tmp.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
        freeze_tmp.replace(freeze_path)
    print(f"wrote {count} anchors")


if __name__ == "__main__":
    main()
