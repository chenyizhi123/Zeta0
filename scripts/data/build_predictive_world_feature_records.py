#!/usr/bin/env python3
"""Derive immutable future-frame cache records from the predictive-world index."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from psi.data.predictive_world.cache import (
    FEATURE_RECORDS_FORMAT,
    feature_records_metadata_path,
)
from psi.data.predictive_world.index import (
    predictive_world_index_binding,
    verify_predictive_world_index,
)
from psi.data.predictive_world.measurement_schema import (
    SHARED_FRAME_INDEX_ALIGNMENT,
    SINGLE_CAMERA_ALIGNMENT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify_predictive_world_index(args.index)
    index_binding = predictive_world_index_binding(args.index)
    records = []
    for line in Path(args.index).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        materialization = row.get("materialization", {})
        videos = materialization.get("video_paths")
        if not isinstance(videos, dict) or not videos:
            raise ValueError("every index row requires camera video_paths")
        capability = row.get("source_capability")
        if not isinstance(capability, dict):
            raise ValueError("every index row requires an immutable source capability snapshot")
        visual_keys = capability.get("visual_keys")
        if not isinstance(visual_keys, list) or set(map(str, visual_keys)) != set(map(str, videos)):
            raise ValueError("feature materialization visual keys disagree with source capability")
        mode = str(capability.get("visual_alignment_mode", ""))
        if len(videos) == 1:
            if mode != SINGLE_CAMERA_ALIGNMENT:
                raise ValueError("single-camera feature records require single_camera alignment")
        elif (
            mode != SHARED_FRAME_INDEX_ALIGNMENT
            or not capability.get("visual_synchronization_audit_revision")
            or "REQUIRED_" in str(capability.get("visual_synchronization_audit_revision"))
        ):
            raise ValueError(
                "shared multi-camera frame indices require an explicit synchronization audit"
            )
        indices = row["future_visual_indices"]
        valid = row["future_visual_valid"]
        times = row["future_visual_actual_times"]
        if not (len(indices) == len(valid) == len(times)):
            raise ValueError("future visual index arrays disagree")
        if any(bool(value) for value in valid) and not capability.get("supports_future_visual"):
            raise ValueError("feature records grant unsupported future visual supervision")
        for group, (index, present, actual_time) in enumerate(zip(indices, valid, times)):
            if not present:
                continue
            if float(actual_time) <= 0:
                raise ValueError("future visual supervision must have positive physical time")
            future_visual_keys = capability.get("future_visual_keys", visual_keys)
            if (
                not isinstance(future_visual_keys, list)
                or not future_visual_keys
                or not set(map(str, future_visual_keys)).issubset(set(map(str, videos)))
            ):
                raise ValueError("future visual keys are absent from materialized cameras")
            for camera_id in sorted(map(str, future_visual_keys)):
                video_path = videos[camera_id]
                resolved = str(Path(video_path).expanduser().resolve())
                records.append({
                    "sample_id": str(row["sample_id"]), "group": group,
                    "camera_id": str(camera_id), "actual_time": float(actual_time),
                    "frame_id": f"{resolved}:{int(index)}", "video_path": resolved,
                    "frame_index": int(index),
                })
    if not records:
        raise ValueError("predictive-world index produced no valid future feature records")
    encoded = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    output = Path(args.output).expanduser().resolve()
    metadata_path = feature_records_metadata_path(output)
    metadata = {
        "format": FEATURE_RECORDS_FORMAT,
        "records": len(records),
        "records_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "index_binding": index_binding,
    }
    if output.exists() or metadata_path.exists():
        if not output.is_file() or not metadata_path.is_file():
            raise ValueError("partial feature records publication exists")
        if (
            output.read_text(encoding="utf-8") != encoded
            or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata
        ):
            raise ValueError("immutable feature records already exist with different content")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
        metadata_temporary = metadata_path.with_name(
            f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
        )
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_temporary.replace(metadata_path)
    print(json.dumps({
        "event": "predictive_world_feature_records_published",
        "records": len(records), "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
