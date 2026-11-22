"""Source-bound current-state views of immutable predictive-world datasets.

Only the three observed-state history arrays change. Future body/image/action
labels, sample ordering, splits and feature rows remain exactly as recorded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT,
    canonical_sha256,
    file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.index import (
    predictive_world_index_binding,
    publish_predictive_world_index,
    verify_predictive_world_index,
)

CURRENT_STATE_PROJECTION_FORMAT = "zeta0.predictive_world.current_state_index_projection.v1"
STATE_FIELDS = ("state_indices", "state_valid", "state_actual_times")


def project_current_state_row(
    row: Mapping[str, Any], *, source_state_history_count: int
) -> dict[str, Any]:
    """Retain the latest original state without changing any supervision field."""
    if type(source_state_history_count) is not int or source_state_history_count <= 1:
        raise ValueError("current-state projection requires a source with state history")
    for field in STATE_FIELDS:
        if not isinstance(row.get(field), list) or len(row[field]) != source_state_history_count:
            raise ValueError(f"current-state projection source {field} length disagrees")
    index, valid, actual_time = (row[field][-1] for field in STATE_FIELDS)
    if type(index) is not int or index < 0 or type(valid) is not bool:
        raise ValueError("current-state projection source index/mask is invalid")
    if isinstance(actual_time, bool) or not isinstance(actual_time, (int, float)):
        raise ValueError("current-state projection source time is invalid")
    if (valid and not math.isfinite(actual_time)) or actual_time > 0:
        raise ValueError("current-state projection source state is nonfinite or noncausal")
    return {**row, **{field: row[field][-1:] for field in STATE_FIELDS}}


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"immutable current-state artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rebind_packed_features(
    source_path: Path, source: Mapping[str, Any], output: Path,
    target_binding: Mapping[str, Any], projection: Mapping[str, Any],
) -> Path:
    # Rows and future frames were proved unchanged by the projection writer.
    # Reuse the immutable feature objects, but publish a separately bound manifest.
    output.mkdir(parents=True, exist_ok=True)
    for shard in source["shards"]:
        relative = Path(shard["file"])
        original = (source_path.parent / relative).resolve()
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != original:
                raise ValueError("current-state feature shard link points to a different object")
        elif destination.exists():
            raise ValueError("current-state feature shard destination already exists")
        else:
            destination.symlink_to(original)
    payload = copy.deepcopy(dict(source))
    payload["index_binding"] = dict(target_binding)
    payload["index_projection"] = dict(projection)
    payload["manifest_identity"] = canonical_sha256({
        "cache_identity": payload["cache_identity"],
        "index_binding": dict(target_binding),
        "rows": int(payload["rows"]),
        "groups": int(payload["groups"]),
        "shards": payload["shards"],
    })
    path = output / "feature_manifest.json"
    _immutable_json(path, payload)
    verify_predictive_world_feature_manifest(path)
    return path


def publish_current_state_projection(
    source_index: str | Path,
    output_index: str | Path,
    *,
    source_feature_manifest: str | Path | None = None,
    output_feature_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a state[-1:] index and, optionally, a zero-copy feature view.

    The source index and lookup are content-verified. Feature reuse accepts only
    an already validated packed manifest bound to that exact source index; no
    GPU extraction, tensor copying, sample filtering or new temporal sampling
    occurs. Existing publications are immutable and safely repeatable.
    """
    source_index = Path(source_index).expanduser().resolve()
    output_index = Path(output_index).expanduser().resolve()
    if source_index == output_index:
        raise ValueError("current-state projection must publish a new index")
    if (source_feature_manifest is None) != (output_feature_cache is None):
        raise ValueError("current-state feature reuse requires both source and destination")
    source_meta = verify_predictive_world_index(source_index)
    source_binding = predictive_world_index_binding(source_index, verify_content_hashes=False)
    source_temporal = source_meta["temporal_abi"]
    count = source_temporal["state_history_count"]
    if type(count) is not int or count <= 1 or source_temporal["state_offsets_s"][-1] != 0:
        raise ValueError("current-state projection requires a source with state history ending at zero")
    temporal = copy.deepcopy(source_temporal)
    temporal.update(state_history_count=1, state_offsets_s=[0.0])
    projection = {
        "format": CURRENT_STATE_PROJECTION_FORMAT,
        "source_index_binding": source_binding,
        "source_state_history_count": count,
        "target_state_history_count": 1,
        "retained_position": "last_original_state",
        "unchanged_fields": "all_except_state_indices_state_valid_state_actual_times",
        "sample_order": "unchanged",
        "projection_code_sha256": file_sha256(Path(__file__)),
    }
    manifest_path = output_cache = manifest = None
    if source_feature_manifest is not None:
        manifest_path = Path(source_feature_manifest).expanduser().resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "feature_manifest.json"
        output_cache = Path(output_feature_cache).expanduser().resolve()
        if output_cache == manifest_path.parent or manifest_path.parent in output_cache.parents:
            raise ValueError("current-state feature reuse must publish outside the source cache")
        manifest = verify_predictive_world_feature_manifest(manifest_path)
        if manifest.get("format") != PACKED_FEATURE_FORMAT:
            raise ValueError("current-state feature reuse requires a packed manifest")
        if manifest["index_binding"] != source_binding:
            raise ValueError("current-state source features are bound to a different index")
        if (int(manifest["rows"]) != source_meta["rows"]
                or int(manifest["groups"]) != len(temporal["future_group_ends_s"])):
            raise ValueError("current-state source feature dimensions disagree with index")
        projection.update(
            source_feature_manifest_sha256=file_sha256(manifest_path),
            source_feature_manifest_identity=manifest["manifest_identity"],
        )
    provenance = {**source_meta["provenance"], "current_state_projection": projection}
    split_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    unchanged_digest = hashlib.sha256()

    def rows():
        digest = hashlib.sha256()
        total = 0
        with source_index.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("feature_row_index") != total:
                    raise ValueError("current-state source feature row ordering is not contiguous")
                projected = project_current_state_row(row, source_state_history_count=count)
                unchanged = {key: value for key, value in row.items() if key not in STATE_FIELDS}
                unchanged_digest.update((json.dumps(
                    unchanged, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ) + "\n").encode("utf-8"))
                split_counts[str(row["split"])] += 1
                eligibility_counts.update(key for key, value in row["eligibility"].items() if value)
                total += 1
                yield projected
        if total != source_meta["rows"] or digest.hexdigest() != source_binding["index_sha256"]:
            raise ValueError("current-state source changed during projection")

    publish_predictive_world_index(
        output_index, rows=rows(), provenance=provenance, temporal_abi=temporal, presorted=True,
    )
    target_binding = predictive_world_index_binding(output_index, verify_content_hashes=False)
    report = {
        **projection,
        "source_index": str(source_index),
        "target_index": str(output_index),
        "target_index_binding": target_binding,
        "rows": source_meta["rows"],
        "split_counts": dict(sorted(split_counts.items())),
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "unchanged_nonstate_rows_sha256": unchanged_digest.hexdigest(),
    }
    if manifest_path is not None and manifest is not None and output_cache is not None:
        target_manifest = _rebind_packed_features(
            manifest_path, manifest, output_cache, target_binding, projection,
        )
        report.update(
            source_feature_manifest=str(manifest_path),
            target_feature_manifest=str(target_manifest),
            target_feature_manifest_sha256=file_sha256(target_manifest),
            feature_reuse="unchanged_future_frames_and_row_order_original_shard_symlinks",
            feature_bytes_copied=0,
        )
    _immutable_json(output_index.with_suffix(output_index.suffix + ".current_state.json"), report)
    return report
