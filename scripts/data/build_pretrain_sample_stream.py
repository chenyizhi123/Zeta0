#!/usr/bin/env python3
"""Materialize the global sample stream shared by P80/P333/P931."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from psi.data.predictor_pretrain.registry import canonical_json_bytes, iter_manifest_rows
from psi.data.predictor_pretrain.runtime import AnchorIndex
from psi.data.predictor_pretrain.sample_stream import (
    COMPACT_V2_ALGORITHM,
    iter_sample_occurrences,
    sample_stream_spec_sha256,
)


def _weights(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        source, separator, weight = value.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError("weights must have SOURCE=INTEGER form")
        if source in result:
            raise argparse.ArgumentTypeError(f"duplicate source weight: {source}")
        result[source] = int(weight)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-manifest", required=True)
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument(
        "--anchor-index",
        help="For V2/V3, read frozen capacities/protocols from the verified index "
             "instead of reparsing the anchor JSONL.",
    )
    parser.add_argument("--output", required=True, help="output JSONL occurrence stream")
    parser.add_argument("--source-weight", action="append", required=True)
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--state-dropout", type=float, default=0.2)
    parser.add_argument("--sampling-version", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args()
    if args.updates <= 0 or args.global_batch_size <= 0:
        raise ValueError("updates and global batch size must be positive")
    if not 0.0 <= args.state_dropout <= 1.0:
        raise ValueError("state dropout must be in [0,1]")

    freeze = json.loads(Path(args.dataset_freeze).read_text(encoding="utf-8"))
    frozen_sha = str(freeze.get("manifest_sha256", ""))
    if not frozen_sha:
        raise ValueError("dataset freeze lacks its canonical manifest hash")
    # V2/V3 need only source capacities, not millions of resident row objects.
    # New indexes persist those values from their already-verified build scan.
    rows = []
    train_capacities = Counter()
    protocols = set()
    use_index_summary = False
    if args.anchor_index and args.sampling_version in {"v2", "v3"}:
        index = AnchorIndex(args.anchor_index, expected_manifest_sha256=frozen_sha)
        try:
            summary = index.frozen_summary()
            protocols = {
                str(value).removeprefix("predictor_pretrain_anchor_")
                for value in summary["protocols"]
            }
            train_capacities.update({
                source: int(counts.get("PT_train", 0))
                for source, counts in summary["split_counts"].items()
            })
            use_index_summary = True
        finally:
            index.connection.close()
    if use_index_summary:
        manifest_sha = frozen_sha
        print(json.dumps({
            "event": "sample_stream_manifest_reused_from_index",
            "sources": len(train_capacities),
        }), flush=True)
    else:
        manifest_digest = hashlib.sha256(b"[")
        row_count = 0
        for row in iter_manifest_rows(args.pretrain_manifest):
            if row_count:
                manifest_digest.update(b",")
            manifest_digest.update(canonical_json_bytes(row))
            row_count += 1
            protocols.add(str(row.get("format_version", "predictor_pretrain_anchor_v1")).removeprefix("predictor_pretrain_anchor_"))
            if row.get("split") == "PT_train":
                train_capacities[str(row["source_id"])] += 1
            if args.sampling_version == "v1":
                rows.append(row)
        manifest_digest.update(b"]")
        manifest_sha = manifest_digest.hexdigest()
        if manifest_sha != frozen_sha:
            raise SystemExit("pretraining manifest does not match dataset_freeze.json")
    weights = _weights(args.source_weight)
    if not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("source weights must be positive")
    if set(train_capacities) != set(weights):
        raise ValueError(
            "source weights and PT_train sources differ: "
            f"{sorted(weights)} vs {sorted(train_capacities)}"
        )
    if protocols != {args.sampling_version}:
        raise ValueError(
            f"sampling version {args.sampling_version} requires matching anchor identities; got {protocols}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    digest = hashlib.sha256()
    count = args.updates * args.global_batch_size
    source_capacities = {source: train_capacities[source] for source in weights}
    if any(value <= 0 for value in source_capacities.values()):
        raise ValueError("every weighted source must contain PT_train anchors")
    if args.sampling_version in {"v2", "v3"}:
        # V2 is a random-access algorithmic stream.  Materializing 491.5M
        # redundant JSON records used 110 GiB and forced every loader worker
        # to rescan the prefix on resume.  This immutable descriptor plus the
        # source-position SQLite index defines exactly the same information.
        descriptor = {
            "format_version": "predictor_pretrain_compact_sample_stream_v2",
            "sampling_algorithm": COMPACT_V2_ALGORITHM,
            "dataset_manifest_sha256": manifest_sha,
            "source_weights": dict(sorted(weights.items())),
            "source_capacities": dict(sorted(source_capacities.items())),
            "total_updates": args.updates,
            "global_batch_size": args.global_batch_size,
            "seed": args.seed,
            "state_dropout_probability": args.state_dropout,
            "num_occurrences": count,
        }
        payload = (json.dumps(descriptor, sort_keys=True) + "\n").encode()
        temporary.write_bytes(payload)
        digest.update(payload)
        stream_encoding = "algorithmic_v2"
    else:
        with temporary.open("wb") as handle:
            for occurrence in iter_sample_occurrences(
                rows,
                source_weights=weights,
                total_updates=args.updates,
                global_batch_size=args.global_batch_size,
                seed=args.seed,
                state_dropout_probability=args.state_dropout,
                sampling_version=args.sampling_version,
            ):
                payload = (json.dumps(occurrence, sort_keys=True) + "\n").encode()
                handle.write(payload)
                digest.update(payload)
        stream_encoding = "materialized_jsonl_v1"
    temporary.replace(output)
    descriptor = {
        "format_version": f"predictor_pretrain_sample_stream_{args.sampling_version}",
        "dataset_manifest_sha256": manifest_sha,
        "stream_spec_sha256": sample_stream_spec_sha256(
            dataset_manifest_sha256=manifest_sha,
            source_weights=weights,
            total_updates=args.updates,
            global_batch_size=args.global_batch_size,
            seed=args.seed,
            state_dropout_probability=args.state_dropout,
            sampling_version=args.sampling_version,
        ),
        "stream_file": output.name,
        "stream_file_sha256": digest.hexdigest(),
        "num_occurrences": count,
        "source_weights": dict(sorted(weights.items())),
        "total_updates": args.updates,
        "global_batch_size": args.global_batch_size,
        "seed": args.seed,
        "state_dropout_probability": args.state_dropout,
        "stream_encoding": stream_encoding,
        "source_capacities": dict(sorted(source_capacities.items())),
    }
    if args.sampling_version in {"v2", "v3"}:
        descriptor["sampling_algorithm"] = COMPACT_V2_ALGORITHM
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_temporary = meta_path.with_name(f".{meta_path.name}.tmp")
    meta_temporary.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    meta_temporary.replace(meta_path)


if __name__ == "__main__":
    main()
