"""Bounded-memory anchor-index reads for endpoint-cache construction."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SPLITS = ("PT_test", "PT_train", "PT_val")


@dataclass(frozen=True)
class AnchorCacheShard:
    rows: list[dict]
    source_num_samples: int
    start: int
    stop: int
    shard_index: int
    num_shards: int
    selection_order: str
    index_format_version: str
    manifest_sha256: str


@dataclass(frozen=True)
class AnchorCacheIdentity:
    sample_ids: list[str]
    target_timestamps: list[float]
    source_num_samples: int
    selection_order: str
    index_format_version: str
    manifest_sha256: str


def _open_index(
    path: str | Path, *, expected_manifest_sha256: str
) -> tuple[sqlite3.Connection, dict[str, str], set[str]]:
    resolved = Path(path).resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("manifest_sha256") != str(expected_manifest_sha256):
            raise ValueError("anchor index was built from a different frozen manifest")
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(anchors)")
        }
        required = {"sample_id", "source_id", "split", "payload"}
        if not required.issubset(columns):
            raise ValueError("anchor index schema is missing required cache columns")
        return connection, metadata, columns
    except BaseException:
        connection.close()
        raise


def _decode_rows(records: Iterable[tuple[str]]) -> list[dict]:
    rows = []
    for (payload,) in records:
        row = json.loads(payload)
        if not isinstance(row, dict):
            raise ValueError("anchor index contains a non-object payload")
        rows.append(row)
    return rows


def load_source_anchor_shard(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    source_id: str,
    shard_index: int = 0,
    num_shards: int = 1,
) -> AnchorCacheShard:
    """Read one stable, contiguous all-split source shard from SQLite.

    Index v4 has a source-local ``cache_position`` and performs a bounded
    range read.  Older indexes remain usable through an explicitly stable
    split/sample-id pagination fallback; the one-click rebuild upgrades them
    before a 32-way cache launch.
    """

    shard_index, num_shards = int(shard_index), int(num_shards)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("cache shard index/count are invalid")
    source_id = str(source_id)
    connection, metadata, columns = _open_index(
        path, expected_manifest_sha256=expected_manifest_sha256
    )
    try:
        split_counts = {
            split: int(
                connection.execute(
                    "SELECT COUNT(*) FROM anchors WHERE split=? AND source_id=?",
                    (split, source_id),
                ).fetchone()[0]
            )
            for split in _SPLITS
        }
        total = sum(split_counts.values())
        if total <= 0:
            raise ValueError(f"anchor index has no rows for source {source_id!r}")
        start = total * shard_index // num_shards
        stop = total * (shard_index + 1) // num_shards
        if "cache_position" in columns:
            records = connection.execute(
                "SELECT payload FROM anchors WHERE source_id=? "
                "AND cache_position>=? AND cache_position<? ORDER BY cache_position",
                (source_id, start, stop),
            )
            rows = _decode_rows(records)
            order = "source_sample_id_cache_position_v1"
        else:
            # V3 compatibility.  Keep memory bounded to this shard even
            # though SQLite must paginate the older split/source index.
            rows = []
            split_base = 0
            for split in _SPLITS:
                count = split_counts[split]
                local_start = max(0, start - split_base)
                local_stop = min(count, stop - split_base)
                if local_start < local_stop:
                    rows.extend(
                        _decode_rows(
                            connection.execute(
                                "SELECT payload FROM anchors WHERE split=? AND source_id=? "
                                "ORDER BY sample_id LIMIT ? OFFSET ?",
                                (split, source_id, local_stop - local_start, local_start),
                            )
                        )
                    )
                split_base += count
            order = "legacy_split_then_sample_id_v1"
        if len(rows) != stop - start:
            raise ValueError(
                f"anchor index returned {len(rows)} rows for cache shard "
                f"[{start},{stop}), expected {stop - start}"
            )
        if any(
            str(row.get("source_id")) != source_id
            or row.get("split") not in _SPLITS
            for row in rows
        ):
            raise ValueError("anchor index cache shard contains a foreign source/split")
        if len({str(row.get("sample_id")) for row in rows}) != len(rows):
            raise ValueError("anchor index cache shard contains duplicate sample ids")
        return AnchorCacheShard(
            rows=rows,
            source_num_samples=total,
            start=start,
            stop=stop,
            shard_index=shard_index,
            num_shards=num_shards,
            selection_order=order,
            index_format_version=str(metadata.get("format_version", "")),
            manifest_sha256=str(metadata["manifest_sha256"]),
        )
    finally:
        connection.close()


def load_anchor_rows_by_sample_ids(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    sample_ids: Iterable[str],
    source_id: str,
) -> list[dict]:
    """Fetch only requested legacy rows through the sample-id primary index."""

    ordered_ids = list(dict.fromkeys(str(value) for value in sample_ids if str(value)))
    if not ordered_ids:
        return []
    connection, _metadata, _columns = _open_index(
        path, expected_manifest_sha256=expected_manifest_sha256
    )
    found: dict[str, dict] = {}
    try:
        for offset in range(0, len(ordered_ids), 500):
            chunk = ordered_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            for sample_id, payload in connection.execute(
                f"SELECT sample_id,payload FROM anchors WHERE sample_id IN ({placeholders})",
                chunk,
            ):
                row = json.loads(payload)
                if str(row.get("source_id")) != str(source_id):
                    raise ValueError("legacy anchor lookup returned a foreign source")
                found[str(sample_id)] = row
    finally:
        connection.close()
    return [found[value] for value in ordered_ids if value in found]


def load_source_cache_identity(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    source_id: str,
) -> AnchorCacheIdentity:
    """Read only IDs/timestamps for a complete source composite release."""

    source_id = str(source_id)
    connection, metadata, columns = _open_index(
        path, expected_manifest_sha256=expected_manifest_sha256
    )
    try:
        if "cache_position" in columns:
            records = connection.execute(
                "SELECT sample_id,json_extract(payload,'$.target_timestamp') "
                "FROM anchors WHERE source_id=? ORDER BY cache_position",
                (source_id,),
            )
            order = "source_sample_id_cache_position_v1"
        else:
            records = (
                record
                for split in _SPLITS
                for record in connection.execute(
                    "SELECT sample_id,json_extract(payload,'$.target_timestamp') "
                    "FROM anchors WHERE split=? AND source_id=? ORDER BY sample_id",
                    (split, source_id),
                )
            )
            order = "legacy_split_then_sample_id_v1"
        pairs = [(str(sample_id), float(timestamp)) for sample_id, timestamp in records]
        if not pairs:
            raise ValueError(f"anchor index has no rows for source {source_id!r}")
        sample_ids = [value[0] for value in pairs]
        timestamps = [value[1] for value in pairs]
        if len(sample_ids) != len(set(sample_ids)) or not all(map(math.isfinite, timestamps)):
            raise ValueError("anchor index cache identity is duplicate or non-finite")
        return AnchorCacheIdentity(
            sample_ids=sample_ids,
            target_timestamps=timestamps,
            source_num_samples=len(sample_ids),
            selection_order=order,
            index_format_version=str(metadata.get("format_version", "")),
            manifest_sha256=str(metadata["manifest_sha256"]),
        )
    finally:
        connection.close()


__all__ = [
    "AnchorCacheIdentity",
    "AnchorCacheShard",
    "load_anchor_rows_by_sample_ids",
    "load_source_anchor_shard",
    "load_source_cache_identity",
]
