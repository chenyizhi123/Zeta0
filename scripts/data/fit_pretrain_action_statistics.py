#!/usr/bin/env python3
"""Fit native action-protocol statistics from frozen PT_train V3 chunks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from psi.data.predictor_pretrain.registry import iter_manifest_rows, manifest_sha256
from psi.data.predictor_pretrain.runtime import AnchorIndex, LeRobotAnchorReader, LeRobotRuntimeSource


def fit_action_moments(
    rows: Iterable[Mapping[str, Any]], sources: Mapping[str, LeRobotRuntimeSource],
    *, parquet_batch_size: int = 65536,
) -> dict[str, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = LeRobotAnchorReader(sources)
    moments: dict[str, dict[str, Any]] = {}
    current_path: str | None = None
    current_groups: dict[tuple[str, tuple[str, ...]], Counter[int]] = {}

    def flush(path: str | None, groups: dict[tuple[str, tuple[str, ...]], Counter[int]]) -> None:
        if path is None or not groups:
            return
        columns = sorted({column for _, names in groups for column in names})
        with pq.ParquetFile(path) as parquet:
            prepared = []
            for (protocol, names), counts in groups.items():
                indices = np.array(sorted(counts), dtype=np.int64)
                if int(indices[-1]) >= parquet.metadata.num_rows:
                    raise ValueError(f"action row index out of bounds in {path}")
                weights = np.fromiter((counts[int(index)] for index in indices), dtype=np.int64)
                prepared.append((protocol, names, indices, weights))
            start = 0
            for batch in parquet.iter_batches(batch_size=parquet_batch_size, columns=columns):
                end = start + len(batch)
                for protocol, names, indices, weights in prepared:
                    left, right = np.searchsorted(indices, [start, end])
                    if left == right:
                        continue
                    selected = batch.take(pa.array(indices[left:right] - start))
                    pieces = [
                        np.asarray(selected.column(column).to_pylist(), dtype=np.float32)
                        .reshape(len(selected), -1)
                        for column in names
                    ]
                    values = np.concatenate(pieces, axis=1).astype(np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"non-finite PT_train action in {path}, protocol {protocol!r}"
                        )
                    current = moments.setdefault(protocol, {
                        "count": 0, "sum": np.zeros(values.shape[1]),
                        "sumsq": np.zeros(values.shape[1]),
                    })
                    if np.asarray(current["sum"]).shape != (values.shape[1],):
                        raise ValueError(f"action dimension changed inside protocol {protocol!r}")
                    batch_weights = weights[left:right, None]
                    current["count"] += int(batch_weights.sum())
                    current["sum"] += (values * batch_weights).sum(0)
                    current["sumsq"] += (np.square(values) * batch_weights).sum(0)
                start = end
                start = end

    for row in rows:
        if row.get("split") != "PT_train":
            continue
        source = sources[str(row["source_id"])]
        protocol = reader._action_protocol(row, source)
        if protocol is None:
            continue
        if row.get("format_version") != "predictor_pretrain_anchor_v3":
            raise ValueError("action statistics require V3 anchors")
        indices = np.asarray(row["action_row_indices"], dtype=np.int64)
        valid = np.asarray(row["action_time_mask"], dtype=bool)
        selected = indices[valid]
        if not len(selected) or (selected < 0).any():
            continue
        path = str(row["data_path"])
        if current_path is not None and path != current_path:
            flush(current_path, current_groups)
            current_groups = {}
        current_path = path
        key = (protocol, tuple(source.action_columns))
        counts = current_groups.setdefault(key, Counter())
        counts.update(selected.tolist())
    flush(current_path, current_groups)
    return moments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--anchor-index")
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument("--source-configs", required=True)
    parser.add_argument("--action-protocols", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trust-frozen-index", action="store_true",
        help="Use the matching dataset-freeze/index hash instead of reparsing the "
             "multi-gigabyte JSONL manifest. Requires --anchor-index.",
    )
    args = parser.parse_args()
    if args.trust_frozen_index and not args.anchor_index:
        raise ValueError("--trust-frozen-index requires --anchor-index")
    frozen_sha = str(json.loads(Path(args.dataset_freeze).read_text()).get("manifest_sha256", ""))
    manifest_sha = frozen_sha if args.trust_frozen_index else manifest_sha256(args.anchor_manifest)
    if not frozen_sha or frozen_sha != manifest_sha:
        raise ValueError("action statistics input does not match dataset freeze")
    source_values = json.loads(Path(args.source_configs).read_text())
    sources = {key: LeRobotRuntimeSource(source_id=key, **value) for key, value in source_values.items()}
    declarations = json.loads(Path(args.action_protocols).read_text())
    index = None
    try:
        if args.anchor_index:
            index = AnchorIndex(args.anchor_index, expected_manifest_sha256=manifest_sha)
            rows = index.iter_split("PT_train")
        else:
            rows = iter_manifest_rows(args.anchor_manifest)
        moments = fit_action_moments(rows, sources)
    finally:
        if index is not None:
            index.connection.close()
    declared = declarations["protocols"]
    expected_observed = {
        name for name, value in declared.items() if value.get("pretrain_available", True)
    }
    if set(moments) != expected_observed:
        raise ValueError(f"observed/declared training action protocols differ: {set(moments)} vs {expected_observed}")
    result = {
        "format_version": "predictor_pretrain_action_statistics_v1",
        "dataset_manifest_sha256": manifest_sha,
        "horizon_seconds": declarations["horizon_seconds"],
        "action_steps": declarations["action_steps"],
        "protocols": {},
    }
    for protocol, value in sorted(moments.items()):
        count = int(value["count"])
        mean = np.asarray(value["sum"]) / count
        variance = np.maximum(np.asarray(value["sumsq"]) / count - np.square(mean), 0)
        std = np.sqrt(variance)
        constant = std < 1e-6
        std[constant] = 1.0
        if len(mean) != int(declared[protocol]["dim"]):
            raise ValueError(f"observed dimension disagrees with {protocol} declaration")
        result["protocols"][protocol] = {
            **declared[protocol], "count": count, "mean": mean.tolist(), "std": std.tolist(),
            "constant_dimensions": np.flatnonzero(constant).tolist(),
        }
    for protocol, declaration in sorted(declared.items()):
        if declaration.get("pretrain_available", True):
            continue
        if declaration.get("normalization") != "identity":
            raise ValueError(f"unobserved protocol {protocol} must declare identity normalization")
        dim = int(declaration["dim"])
        result["protocols"][protocol] = {
            **declaration, "count": 0, "mean": [0.0] * dim, "std": [1.0] * dim,
            "constant_dimensions": [],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
