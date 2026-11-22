#!/usr/bin/env python3
"""Fit schema statistics strictly from PT_train anchor histories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from psi.data.predictor_pretrain.registry import iter_manifest_rows, manifest_sha256
from psi.data.predictor_pretrain.runtime import AnchorIndex, LeRobotAnchorReader, LeRobotRuntimeSource


def fit_weighted_moments(
    rows: Iterable[Mapping[str, Any]],
    sources: Mapping[str, LeRobotRuntimeSource],
    *,
    parquet_batch_size: int = 65536,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Read each referenced parquet once, preserving every valid history use.

    A state row reused by multiple anchors (or repeated within one history)
    contributes its full occurrence count, not one vote per unique state.
    Both paths retain the legacy float32 state conversion before float64
    population moments. Reassociated sums can differ only by roundoff.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    if parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size must be positive")
    reader = LeRobotAnchorReader(sources)
    moments: dict[str, dict[str, np.ndarray | int]] = {}
    current_path: str | None = None
    current_groups: dict[tuple[str, tuple[str, ...]], Counter[int]] = {}
    parquet_files = unique_references = 0

    def flush(path: str | None, groups: dict[tuple[str, tuple[str, ...]], Counter[int]]) -> None:
        nonlocal parquet_files, unique_references
        if path is None or not groups:
            return
        parquet_files += 1
        unique_references += sum(len(counts) for counts in groups.values())
        columns = sorted({column for _, names in groups for column in names})
        with pq.ParquetFile(path) as parquet:
            prepared = []
            for (schema, names), counts in groups.items():
                indices = np.array(sorted(counts), dtype=np.int64)
                if int(indices[-1]) >= parquet.metadata.num_rows:
                    raise ValueError(f"state row index out of bounds in {path}")
                weights = np.fromiter((counts[int(index)] for index in indices), dtype=np.int64)
                prepared.append((schema, names, indices, weights))
            start = 0
            for batch in parquet.iter_batches(batch_size=parquet_batch_size, columns=columns):
                end = start + len(batch)
                for schema, names, indices, weights in prepared:
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
                        raise ValueError(f"non-finite PT_train state in {path}, schema {schema!r}")
                    current = moments.setdefault(schema, {
                        "count": 0, "sum": np.zeros(values.shape[1], np.float64),
                        "sumsq": np.zeros(values.shape[1], np.float64),
                    })
                    if np.asarray(current["sum"]).shape != (values.shape[1],):
                        raise ValueError(f"state dimension changed inside schema {schema!r}")
                    batch_weights = weights[left:right, None]
                    current["count"] = int(current["count"]) + int(batch_weights.sum())
                    current["sum"] = np.asarray(current["sum"]) + (values * batch_weights).sum(0)
                    current["sumsq"] = np.asarray(current["sumsq"]) + (
                        np.square(values) * batch_weights
                    ).sum(0)
                start = end

    for row in rows:
        if row.get("split") != "PT_train":
            continue
        source = sources[str(row["source_id"])]
        schema = reader._schema(row, source)
        if schema is None:
            continue
        indices = np.asarray(row["state_row_indices"], dtype=np.int64)
        valid = np.asarray(row["state_history_mask"], dtype=bool)
        if indices.ndim != 1 or valid.shape != indices.shape:
            raise ValueError(f"invalid state history indices/mask in {row['sample_id']}")
        selected = indices[valid]
        if not len(selected):
            continue
        if (selected < 0).any():
            raise ValueError(f"negative state row index in {row['sample_id']}")
        path = str(row["data_path"])
        if current_path is not None and path != current_path:
            flush(current_path, current_groups)
            current_groups = {}
        current_path = path
        key = (schema, tuple(source.state_columns))
        counts = current_groups.setdefault(key, Counter())
        counts.update(selected.tolist())
    flush(current_path, current_groups)
    print(
        f"state statistics: {parquet_files} contiguous parquet groups, "
        f"{unique_references} unique schema/column/row references",
        flush=True,
    )
    return moments


def statistics_from_moments(moments: Mapping[str, Mapping[str, Any]], manifest_sha: str) -> dict:
    result = {"format_version": "predictor_pretrain_state_statistics_v1",
              "dataset_manifest_sha256": manifest_sha, "schemas": {}}
    for schema, value in sorted(moments.items()):
        count = int(value["count"])
        if count < 2:
            raise ValueError(f"schema {schema!r} has fewer than two valid state rows")
        mean = np.asarray(value["sum"]) / count
        variance = np.maximum(np.asarray(value["sumsq"]) / count - np.square(mean), 0)
        std = np.sqrt(variance)
        clamped = std < 1e-6
        std[clamped] = 1.0
        result["schemas"][schema] = {
            "dim": len(mean), "count": count, "mean": mean.tolist(), "std": std.tolist(),
            "constant_dimensions": np.flatnonzero(clamped).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--anchor-index", help="Optional frozen SQLite index; streams PT_train only")
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument("--source-configs", required=True)
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
        raise ValueError("state statistics input does not match dataset freeze")
    source_values = json.loads(Path(args.source_configs).read_text())
    sources = {key: LeRobotRuntimeSource(source_id=key, **value) for key, value in source_values.items()}
    index = None
    try:
        if args.anchor_index:
            index = AnchorIndex(args.anchor_index, expected_manifest_sha256=manifest_sha)
            rows = index.iter_split("PT_train")
        else:
            rows = iter_manifest_rows(args.anchor_manifest)
        moments = fit_weighted_moments(rows, sources)
    finally:
        if index is not None:
            index.connection.close()
    result = statistics_from_moments(moments, manifest_sha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
