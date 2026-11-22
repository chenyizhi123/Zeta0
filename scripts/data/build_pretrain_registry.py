#!/usr/bin/env python3
"""Normalize source episode records and freeze group-level PT splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from psi.data.predictor_pretrain.registry import (
    canonical_json_bytes,
    iter_manifest_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="source JSON/JSONL/CSV/parquet records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if not 0 < args.train_fraction < 1 or not 0 <= args.val_fraction < 1:
        raise ValueError("split fractions are out of range")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_name(f".{output.name}.split.sqlite.tmp")
    temporary = output.with_name(f".{output.name}.tmp")
    database.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    protocols: set[str] = set()
    groups_by_source: dict[str, set[str]] = {}
    owner_by_group: dict[str, str] = {}
    count = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE rows(sample_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
            "group_id TEXT NOT NULL, episode_id TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        pending = []
        for input_path in args.inputs:
            for row in iter_manifest_rows(input_path):
                source = str(row.get("source_id", ""))
                group = str(row.get("group_id", ""))
                sample = str(row.get("sample_id", ""))
                if not source or not group or not sample:
                    raise ValueError("every pretraining row needs source_id, group_id and sample_id")
                previous = owner_by_group.setdefault(group, source)
                if previous != source:
                    raise ValueError(f"group_id {group!r} spans multiple sources")
                groups_by_source.setdefault(source, set()).add(group)
                format_version = str(row.get("format_version", ""))
                protocols.add(
                    "v3" if format_version.endswith("_v3") else
                    "v2" if format_version.endswith("_v2") else "v1"
                )
                pending.append((
                    sample, source, group, str(row.get("episode_id", "")),
                    json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                ))
                count += 1
                if len(pending) == 2_000:
                    try:
                        connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", pending)
                    except sqlite3.IntegrityError as error:
                        raise ValueError("duplicate pretraining sample_id") from error
                    pending.clear()
        if pending:
            try:
                connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", pending)
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate pretraining sample_id") from error
        connection.commit()
        if not count:
            raise ValueError("cannot freeze an empty pretraining manifest")
        if len(protocols) != 1:
            raise ValueError("cannot freeze a mixture of V1/V2/V3 input protocols")
        protocol = next(iter(protocols))

        assignment: dict[str, str] = {}
        for source, source_groups in sorted(groups_by_source.items()):
            groups = sorted(
                source_groups,
                key=lambda group: hashlib.sha256(
                    f"{args.seed}|{source}|{group}".encode()
                ).digest(),
            )
            train_end = round(len(groups) * args.train_fraction)
            val_end = train_end + round(len(groups) * args.val_fraction)
            assignment.update({
                group: (
                    "PT_train" if index < train_end else
                    "PT_val" if index < val_end else "PT_test"
                )
                for index, group in enumerate(groups)
            })

        digest = hashlib.sha256(b"[")
        first = True
        with temporary.open("w", encoding="utf-8") as handle:
            cursor = connection.execute(
                "SELECT group_id,payload FROM rows "
                "ORDER BY source_id,group_id,episode_id,sample_id"
            )
            for group, payload in cursor:
                row = json.loads(payload)
                row["split"] = assignment[group]
                if not first:
                    digest.update(b",")
                digest.update(canonical_json_bytes(row))
                first = False
                handle.write(json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ) + "\n")
        digest.update(b"]")
        temporary.replace(output)
    finally:
        connection.close()
        database.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

    freeze = {
        "format_version": f"predictor_pretrain_dataset_freeze_{protocol}",
        "input_protocol": protocol,
        "split_seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "inputs": [str(Path(value).resolve()) for value in args.inputs],
        "num_rows": count,
        "manifest_sha256": digest.hexdigest(),
    }
    freeze_path = Path(args.dataset_freeze)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
